using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     A zero-copy view of Python memory books the buffer's bytes with NumSharp's memory accounting for as long as its
    ///     lease lives (P8 of the 0.80.0-beta1 bridge QA, the owner's decision of 2026-11-06), instead of calling
    ///     <c>GC.AddMemoryPressure</c> per lease: the GC counted every addition toward a full collection even after its
    ///     removal, so 5,000 views of one 602 KB numpy array, each disposed at once, ran 8 / 15 gen-2 collections; booked,
    ///     none.
    /// </summary>
    [TestClass]
    public class ImportLeaseBookingTests : InteropTestBase
    {
        /// <summary>
        ///     The view's lease books the 4,096 bytes it keeps alive when it opens, and gives them back once the last view is
        ///     gone and the lease's Python side is released (a queued drain under the GIL).
        /// </summary>
        [TestMethod]
        public void ImportView_BooksTheBuffer_UntilTheLeaseIsReleased()
        {
            PyExec("booked_view_source = bytearray(range(256)) * 16");   // 4,096 bytes
            long before = SizeBucketedBufferPool.OutstandingForeignBytes;

            NDArray view = ViewOf("booked_view_source");
            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(before + 4096, "the lease books the Python buffer it keeps alive");
            NDArray slice = view["1024:"];
            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(before + 4096, "a derived slice shares the one lease and its one booking");

            view.Dispose();
            Pump();
            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(before + 4096, "the slice still holds the lease");
            slice.Dispose();
            WaitFor(() => SizeBucketedBufferPool.OutstandingForeignBytes == before)
                .Should().BeTrue("the last view released the lease, and the release gave the booking back");
        }

        /// <summary>
        ///     Views opened and disposed in a loop never grow the bookings, so they never pace a collection: the booking is
        ///     given back with each lease (at the drain the next conversion runs), however many bytes pass through.
        /// </summary>
        [TestMethod]
        public void ImportViews_DisposedInALoop_PaceNoCollection()
        {
            PyExec("booked_loop_source = bytearray(1 << 20)");   // 1 MiB
            long before = SizeBucketedBufferPool.OutstandingForeignBytes;
            long paced = SizeBucketedBufferPool.PacedCollections;
            for (int i = 0; i < 64; i++)
            {
                using NDArray view = ViewOf("booked_loop_source");
            }
            Pump();
            WaitFor(() => SizeBucketedBufferPool.OutstandingForeignBytes == before).Should().BeTrue("every lease was drained and gave its booking back");
            SizeBucketedBufferPool.PacedCollections.Should().Be(paced,
                "64 MiB of views were opened and disposed, but the outstanding bookings never grew by a pacing window");
        }
    }
}
