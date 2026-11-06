using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Tests.Collections;

namespace NumSharp.Tests.Backends.Unmanaged
{
    /// <summary>
    ///     <see cref="SizeBucketedBufferPool.BookForeign"/> / <see cref="SizeBucketedBufferPool.ReturnForeign"/> under
    ///     concurrency: sixteen threads booking and giving back, give-backs on other threads and on the finalizer thread, and
    ///     a thread collecting all the while. The foreign balance is a plain interlocked sum, so it must come out EXACT; only
    ///     the pacing window is a heuristic, which may move which call paces, never the books.
    /// </summary>
    /// <remarks>
    ///     <para>The pool's counters are process-wide and the unit suite runs in parallel, so the class runs alone
    ///     (<c>[DoNotParallelize]</c>): one full drain when the class starts, a young one per test (tests.md: a full
    ///     collection marks the whole run's heap). No test allocates a NumSharp array, and nothing else in this suite books
    ///     foreign memory, so the foreign balance moves only with these bookings. The pool's total outstanding bytes are NOT
    ///     asserted exactly: earlier tests' arrays can still return buffers whenever a collection finalizes them.</para>
    ///     <para>Written for the P8 verification (2026-11-06): the owner asked to prove the booking is good for
    ///     multithreading, memory safety and GC load.</para>
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class ForeignBookingConcurrencyTests : TestClass
    {
        /// <summary>Booking threads per test.</summary>
        private const int Threads = 16;

        /// <summary>The pool's private registered pressure (whole chunks told to <see cref="GC.AddMemoryPressure"/>).</summary>
        private static readonly FieldInfo RegisteredField =
            typeof(SizeBucketedBufferPool).GetField("_registeredPressure", BindingFlags.Static | BindingFlags.NonPublic);

        /// <summary>Clears the backlog earlier tests left - their pending finalizers - once, before any test reads a baseline.</summary>
        /// <param name="context">Unused.</param>
        [ClassInitialize]
        public static void DrainBacklog(TestContext context) => GcQuiescence.CollectFull();

        /// <summary>
        ///     Sixteen threads book random sizes and give each booking back three ways - on the same thread, on another
        ///     thread (through a shared queue), or from the finalizer of a dropped holder - while another thread runs young
        ///     collections nonstop, so finalizer give-backs land in the middle of the booking loops. Afterwards the foreign
        ///     balance is back to the byte, and the registered pressure is still whole, non-negative chunks.
        /// </summary>
        [TestMethod]
        public void BookingsAndGiveBacks_OnEveryThread_BalanceExactly()
        {
            if (RegisteredField is null)
                Assert.Inconclusive("the pool's private _registeredPressure was renamed; update this test");
            GcQuiescence.CollectYoung();
            long foreignBefore = SizeBucketedBufferPool.OutstandingForeignBytes;
            int iterations = StressBudget.Pick(full: 20_000, ci: 5_000);

            var crossThread = new ConcurrentQueue<long>();
            using var stop = new CancellationTokenSource();
            Task collector = Task.Run(() =>
            {
                // Young collections find the dropped holders; their finalizers give the bookings back on the finalizer
                // thread while the booking threads keep going.
                while (!stop.IsCancellationRequested)
                {
                    GC.Collect(0);
                    Thread.Sleep(1);
                }
            });
            Parallel.For(0, Threads, new ParallelOptions { MaxDegreeOfParallelism = Threads }, t =>
            {
                var rng = new System.Random(1000 + t);   // qualified: this test project has a Random namespace
                for (int i = 0; i < iterations; i++)
                {
                    long bytes = rng.Next(1, 16 * 1024);
                    SizeBucketedBufferPool.BookForeign(bytes);
                    switch (i % 3)
                    {
                        case 0:
                            SizeBucketedBufferPool.ReturnForeign(bytes);
                            break;
                        case 1:
                            crossThread.Enqueue(bytes);   // given back by whichever thread dequeues it
                            if (crossThread.TryDequeue(out long other))
                                SizeBucketedBufferPool.ReturnForeign(other);
                            break;
                        default:
                            Drop(bytes);                  // given back by the finalizer thread
                            break;
                    }
                }
            });
            stop.Cancel();
            collector.Wait();
            while (crossThread.TryDequeue(out long rest))
                SizeBucketedBufferPool.ReturnForeign(rest);
            DrainDroppedHolders(foreignBefore);

            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(foreignBefore, "every booking was given back exactly once");
            long registered = (long)RegisteredField.GetValue(null);
            (registered % SizeBucketedBufferPool.PressureChunkBytes).Should().Be(0, "pressure is registered in whole chunks only");
            registered.Should().BeGreaterThanOrEqualTo(0, "a chunk is removed only while one is registered");
        }

        /// <summary>
        ///     Sixteen threads that give every booking back at once never hold more than 16 x 64 KiB together - far below the
        ///     16 MiB pacing window - so not one collection is paced, however many bytes pass through.
        /// </summary>
        [TestMethod]
        public void DisposingLoops_OnSixteenThreads_PaceNothing()
        {
            if (!SizeBucketedBufferPool.GcPacingEnabled)
                Assert.Inconclusive("GC pacing is switched off (NUMSHARP_POOL_GC_PACING=0)");
            GcQuiescence.CollectYoung();
            ResetWindow();
            long foreignBefore = SizeBucketedBufferPool.OutstandingForeignBytes;
            long paced = SizeBucketedBufferPool.PacedCollections;
            int iterations = StressBudget.Pick(full: 20_000, ci: 5_000);
            Parallel.For(0, Threads, new ParallelOptions { MaxDegreeOfParallelism = Threads }, _ =>
            {
                for (int i = 0; i < iterations; i++)
                {
                    SizeBucketedBufferPool.BookForeign(64 * 1024);
                    SizeBucketedBufferPool.ReturnForeign(64 * 1024);
                }
            });
            SizeBucketedBufferPool.PacedCollections.Should().Be(paced, "the bookings never grew by a pacing window");
            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(foreignBefore);
        }

        /// <summary>
        ///     Sixteen threads each keep 3 MiB booked (48 MiB in all, below one pressure chunk) while another thread collects:
        ///     the growth paces collections requested from many threads at once without deadlock (requests collapse into the
        ///     one in flight), then everything is given back and the balance is exact.
        /// </summary>
        [TestMethod]
        public void GrowingBookings_OnSixteenThreads_PaceWithoutDeadlock_AndBalance()
        {
            if (!SizeBucketedBufferPool.GcPacingEnabled)
                Assert.Inconclusive("GC pacing is switched off (NUMSHARP_POOL_GC_PACING=0)");
            GcQuiescence.CollectYoung();
            ResetWindow();
            long foreignBefore = SizeBucketedBufferPool.OutstandingForeignBytes;
            long paced = SizeBucketedBufferPool.PacedCollections;
            const long Step = 1024 * 1024;
            const int StepsPerThread = 3;
            using var stop = new CancellationTokenSource();
            Task collector = Task.Run(() =>
            {
                while (!stop.IsCancellationRequested)
                {
                    GC.Collect(0);
                    Thread.Sleep(1);
                }
            });
            Task growth = Task.Run(() => Parallel.For(0, Threads, new ParallelOptions { MaxDegreeOfParallelism = Threads }, _ =>
            {
                for (int i = 0; i < StepsPerThread; i++)
                    SizeBucketedBufferPool.BookForeign(Step);
            }));
            bool finished = growth.Wait(TimeSpan.FromSeconds(60));
            stop.Cancel();
            collector.Wait();
            try
            {
                finished.Should().BeTrue("concurrent paced collections must not deadlock");
                long pacedNow = SizeBucketedBufferPool.PacedCollections - paced;
                pacedNow.Should().BeGreaterThanOrEqualTo(1, "48 MiB of growth crossed the 16 MiB window");
                pacedNow.Should().BeLessThanOrEqualTo(Threads * StepsPerThread, "only a booking requests a pace, once at most");
            }
            finally
            {
                // Give back what the growth booked, whatever the assertions found, so later tests start balanced.
                for (int i = 0; i < Threads * StepsPerThread; i++)
                    SizeBucketedBufferPool.ReturnForeign(Step);
            }
            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(foreignBefore, "every booking was given back");
        }

        /// <summary>
        ///     Runs young collections until the dropped holders' finalizers gave back every booking, with one full collection
        ///     as the fallback for a holder that a collection promoted while its constructor still held it.
        /// </summary>
        /// <param name="foreignBefore">The foreign balance the test started from.</param>
        private static void DrainDroppedHolders(long foreignBefore)
        {
            for (int i = 0; i < 4 && SizeBucketedBufferPool.OutstandingForeignBytes != foreignBefore; i++)
                GcQuiescence.CollectYoung();
            if (SizeBucketedBufferPool.OutstandingForeignBytes != foreignBefore)
                GcQuiescence.CollectFull();
        }

        /// <summary>Starts the growth window from zero: one window booked (it paces, or reaches the window and paces) and given back.</summary>
        private static void ResetWindow()
        {
            long window = SizeBucketedBufferPool.PaceThresholdBytes;
            SizeBucketedBufferPool.BookForeign(window);
            SizeBucketedBufferPool.ReturnForeign(window);
        }

        /// <summary>Hands a booking to a holder that nothing references, in a frame of its own, so only its finalizer gives the booking back.</summary>
        /// <param name="bytes">The bytes already booked.</param>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void Drop(long bytes) => _ = new ForgottenBooking(bytes);

        /// <summary>A dropped holder of one booking: its finalizer gives the booking back, the way a forgotten view's lease does.</summary>
        private sealed class ForgottenBooking
        {
            /// <summary>The booked bytes this holder gives back.</summary>
            private readonly long _bytes;

            /// <summary>Takes over one booking.</summary>
            /// <param name="bytes">The bytes already booked with <see cref="SizeBucketedBufferPool.BookForeign"/>.</param>
            public ForgottenBooking(long bytes) => _bytes = bytes;

            /// <summary>Gives the booking back on the finalizer thread.</summary>
            ~ForgottenBooking() => SizeBucketedBufferPool.ReturnForeign(_bytes);
        }
    }
}
