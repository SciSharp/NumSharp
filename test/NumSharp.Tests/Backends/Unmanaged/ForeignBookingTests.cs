using System;
using AwesomeAssertions;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Tests.Backends.Unmanaged
{
    /// <summary>
    ///     <see cref="SizeBucketedBufferPool.BookForeign"/> / <see cref="SizeBucketedBufferPool.ReturnForeign"/>: native
    ///     memory NumSharp keeps alive or hands out without allocating it (an ONNX Runtime tensor from <c>ToOrtValue</c>,
    ///     a numpy or ONNX Runtime buffer under a NumSharp view) is booked with the pool's accounting, and the GC is paced
    ///     once per pacing window of GROWTH - never for bookings given back as fast as they are made.
    /// </summary>
    /// <remarks>
    ///     The pool's counters are process-wide, and the unit suite runs in parallel, so this class runs alone
    ///     (<c>[DoNotParallelize]</c>: after the parallel tests, one test at a time). Every test gives back what it books.
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class ForeignBookingTests : TestClass
    {
        private const long MiB = 1024 * 1024;

        /// <summary>A booking raises the outstanding foreign bytes by its count, and giving it back lowers them by the same count.</summary>
        [TestMethod]
        public void BookAndReturn_MoveTheOutstandingBalanceBothWays()
        {
            long before = SizeBucketedBufferPool.OutstandingForeignBytes;
            SizeBucketedBufferPool.BookForeign(5000);
            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(before + 5000);
            SizeBucketedBufferPool.ReturnForeign(5000);
            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(before);
        }

        /// <summary>Zero and negative counts book and give back nothing.</summary>
        [TestMethod]
        public void ZeroAndNegativeCounts_AreIgnored()
        {
            long before = SizeBucketedBufferPool.OutstandingForeignBytes;
            long paced = SizeBucketedBufferPool.PacedCollections;
            SizeBucketedBufferPool.BookForeign(0);
            SizeBucketedBufferPool.BookForeign(-7);
            SizeBucketedBufferPool.ReturnForeign(0);
            SizeBucketedBufferPool.ReturnForeign(-7);
            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(before);
            SizeBucketedBufferPool.PacedCollections.Should().Be(paced);
        }

        /// <summary>
        ///     Bookings given back as fast as they are made never grow the balance, so they never pace a collection - however
        ///     many bytes pass through. This is code that disposes what it books.
        /// </summary>
        [TestMethod]
        public void BookingsGivenBackAtOnce_NeverPace()
        {
            long paced = SizeBucketedBufferPool.PacedCollections;
            for (int i = 0; i < 200; i++)
            {
                SizeBucketedBufferPool.BookForeign(MiB);
                SizeBucketedBufferPool.ReturnForeign(MiB);
            }
            SizeBucketedBufferPool.PacedCollections.Should().Be(paced, "200 MiB passed through, but the balance never grew");
        }

        /// <summary>
        ///     Bookings kept (dropped holders) grow the balance, and each pacing window of growth paces exactly one
        ///     collection: 48 MiB booked in 1 MiB steps paces three times with the default 16 MiB window.
        /// </summary>
        [TestMethod]
        public void GrowingBookings_PaceOneCollectionPerWindow()
        {
            if (!SizeBucketedBufferPool.GcPacingEnabled)
                Assert.Inconclusive("GC pacing is switched off (NUMSHARP_POOL_GC_PACING=0)");
            ResetWindow();
            long window = SizeBucketedBufferPool.PaceThresholdBytes;
            long paced = SizeBucketedBufferPool.PacedCollections;
            long steps = 3 * window / MiB;
            for (long i = 0; i < steps; i++)
                SizeBucketedBufferPool.BookForeign(MiB);
            try
            {
                SizeBucketedBufferPool.PacedCollections.Should().Be(paced + 3, "three windows of growth, one paced collection each");
            }
            finally
            {
                SizeBucketedBufferPool.ReturnForeign(steps * MiB);
            }
        }

        /// <summary>
        ///     When give-backs since the last pace exceed the bookings since then (a collection's finalizers returned a
        ///     backlog), the window restarts from the next booking instead of first paying the backlog off: one window of new
        ///     growth paces again.
        /// </summary>
        [TestMethod]
        public void GiveBacksBelowZero_RestartTheWindowFromTheNextBooking()
        {
            if (!SizeBucketedBufferPool.GcPacingEnabled)
                Assert.Inconclusive("GC pacing is switched off (NUMSHARP_POOL_GC_PACING=0)");
            ResetWindow();
            long window = SizeBucketedBufferPool.PaceThresholdBytes;
            long steps = window / MiB;

            // A backlog booked earlier and given back now, below the window: the balance goes far below zero.
            for (long i = 0; i < steps - 1; i++)
                SizeBucketedBufferPool.BookForeign(MiB);
            SizeBucketedBufferPool.ReturnForeign((steps - 1) * MiB);
            SizeBucketedBufferPool.ReturnForeign(0);
            SizeBucketedBufferPool.BookForeign(10 * window);   // pace once here, then give it all back: the balance ends far below zero
            SizeBucketedBufferPool.ReturnForeign(10 * window);

            long paced = SizeBucketedBufferPool.PacedCollections;
            for (long i = 0; i < steps - 1; i++)
                SizeBucketedBufferPool.BookForeign(MiB);
            SizeBucketedBufferPool.PacedCollections.Should().Be(paced, "one MiB short of a window since the restart");
            SizeBucketedBufferPool.BookForeign(MiB);
            try
            {
                SizeBucketedBufferPool.PacedCollections.Should().Be(paced + 1,
                    "a full window of growth since the restart paces, although the balance had been 10 windows below zero");
            }
            finally
            {
                SizeBucketedBufferPool.ReturnForeign(steps * MiB);
            }
        }

        /// <summary>
        ///     A booking is not a hand-out: it leaves the pool's own pacing window (the bytes handed out since the last paced
        ///     collection) untouched, so a loop that books and gives back cannot make the next ordinary pool miss pace a
        ///     collection. The first cut counted bookings there, and the pythonnet lease test caught a paced collection in a
        ///     loop that disposed every view.
        /// </summary>
        [TestMethod]
        public void Bookings_DoNotCountTowardThePoolsOwnPacingWindow()
        {
            var window = typeof(SizeBucketedBufferPool).GetField("_takenBytesSincePace",
                System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic);
            if (window is null)
                Assert.Inconclusive("the pool's pacing-window field was renamed; update this test");
            long before = (long)window.GetValue(null);
            for (int i = 0; i < 64; i++)
            {
                SizeBucketedBufferPool.BookForeign(MiB);
                SizeBucketedBufferPool.ReturnForeign(MiB);
            }
            ((long)window.GetValue(null)).Should().Be(before, "64 MiB of bookings were not handed out by the pool");
        }

        /// <summary>Starts the growth window from zero: one pacing booking, then everything given back and the floor applied by the next booking.</summary>
        private static void ResetWindow()
        {
            long window = SizeBucketedBufferPool.PaceThresholdBytes;
            SizeBucketedBufferPool.BookForeign(window);     // paces (and zeroes the window) if anything was pending, or reaches it now
            SizeBucketedBufferPool.ReturnForeign(window);   // the window is now at or below zero; the next booking floors it
        }
    }
}
