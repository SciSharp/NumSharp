using System;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>
    ///     A zero-copy view of a <see cref="Tensor{T}"/> books the bytes of the elements its pin keeps alive with NumSharp's
    ///     memory accounting (<see cref="SizeBucketedBufferPool.BookForeign"/>) until the last view releases the pin - P8 Q3
    ///     of the bridge QA, the owner's decision of 2026-11-06 ("Book with the pool").
    /// </summary>
    /// <remarks>
    ///     <para>Why: the backing array is managed memory the GC counts, but a FORGOTTEN view's pin keeps it alive past the
    ///     collection that found the view dead; with nothing reported, 5,000 forgotten views of fresh 602 KB tensors held
    ///     +2.6 GB. With one <see cref="GC.AddMemoryPressure"/> per lease (before), code that disposed every view paid
    ///     12-41 gen-2 collections per 5,000 views. Booked: a disposing loop pays nothing, a forgetting one is paced.</para>
    ///     <para>The suite runs without parallelism (<c>[assembly: DoNotParallelize]</c>), so the process-wide booking and
    ///     pacing counters move only with each test's own views.</para>
    /// </remarks>
    [TestClass]
    public class TensorLeaseBookingTests : TensorsTestBase
    {
        /// <summary>One MiB of float32: a tensor whose views book a measurable share of a pacing window.</summary>
        private const int MiBFloats = 1 << 18;

        /// <summary>
        ///     A view books the 4,000 bytes its pin keeps alive when it opens; a derived slice shares the one lease and its
        ///     one booking; the booking is given back when the LAST view is disposed, and not before.
        /// </summary>
        [TestMethod]
        public void View_BooksThePinnedElements_UntilTheLastViewIsGone()
        {
            Tensor<float> tensor = Tensor.Create(new float[1000], new nint[] { 1000 });
            long before = SizeBucketedBufferPool.OutstandingForeignBytes;

            NDArray view = tensor.AsNDArray();
            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(before + 4000, "the lease books the elements its pin keeps alive");
            NDArray slice = view["100:"];
            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(before + 4000, "a derived slice shares the one lease and its one booking");

            view.Dispose();
            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(before + 4000, "the slice still holds the pin");
            slice.Dispose();
            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(before, "the last view released the pin and gave the booking back");
        }

        /// <summary>
        ///     Views opened and disposed in a loop give every booking back at Dispose, so the bookings never grow and pace
        ///     no collection however many bytes pass through.
        /// </summary>
        /// <remarks>
        ///     The gen-2 collections one <see cref="GC.AddMemoryPressure"/> per lease caused in such a loop are measured by
        ///     the P8 probes (12-41 per 5,000 views), not asserted here: the runtime skips a pressure collection within five
        ///     durations of the last full one, and the test base settles with a full collection just before, so a gen-2
        ///     count passed under the old per-lease pressure too (a mutant run, 2026-11-06).
        /// </remarks>
        [TestMethod]
        public void ViewsDisposedInALoop_PaceNoCollection()
        {
            Tensor<float> tensor = Tensor.Create(new float[MiBFloats], new nint[] { MiBFloats });
            long before = SizeBucketedBufferPool.OutstandingForeignBytes;
            long paced = SizeBucketedBufferPool.PacedCollections;
            for (int i = 0; i < 64; i++)
            {
                using NDArray view = tensor.AsNDArray();
            }
            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(before, "every lease gave its booking back at its view's Dispose");
            SizeBucketedBufferPool.PacedCollections.Should().Be(paced,
                "64 MiB of views were opened and disposed, but the outstanding bookings never grew by a pacing window");
        }

        /// <summary>
        ///     Views dropped undisposed keep their bookings, which grow and pace collections; the collections' finalizers
        ///     release the pins and give every booking back.
        /// </summary>
        [TestMethod]
        public void ForgottenViews_PaceCollections_AndTheirFinalizersGiveTheBookingsBack()
        {
            if (!SizeBucketedBufferPool.GcPacingEnabled)
                Assert.Inconclusive("GC pacing is switched off (NUMSHARP_POOL_GC_PACING=0)");
            Tensor<float> tensor = Tensor.Create(new float[MiBFloats], new nint[] { MiBFloats });
            GC.Collect();
            GC.WaitForPendingFinalizers();
            long before = SizeBucketedBufferPool.OutstandingForeignBytes;
            long paced = SizeBucketedBufferPool.PacedCollections;

            DropViews(tensor, 64);
            SizeBucketedBufferPool.PacedCollections.Should().BeGreaterThan(paced,
                "64 MiB of forgotten views grew the bookings past the 16 MiB pacing window");

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            GC.WaitForPendingFinalizers();
            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(before,
                "every forgotten view's finalizer released its pin and gave its booking back");
            GC.KeepAlive(tensor);
        }

        /// <summary>Takes <paramref name="count"/> views of <paramref name="tensor"/> and drops them, in a frame of its own so nothing keeps them reachable.</summary>
        /// <param name="tensor">The tensor to view.</param>
        /// <param name="count">How many views to take and drop.</param>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void DropViews(Tensor<float> tensor, int count)
        {
            for (int i = 0; i < count; i++)
                tensor.AsNDArray();
        }
    }
}
