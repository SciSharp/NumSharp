using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    /// Guards the <c>using</c>-scoped intermediates introduced into <see cref="np.concatenate"/>
    /// (the <c>dstSlice</c> view inside the general loop, plus the ravel'd workArrays
    /// when <c>axis=null</c>).
    /// </summary>
    [TestClass]
    public class np_concatenate_using_test : TestClass
    {
        // --------------------------- correctness ---------------------------

        /// <summary>
        /// Exercises the general path (NDIter.Copy). A transposed source is
        /// non-contiguous, which forces both fast paths (TryDirectMemcpyConcat,
        /// TryDirectCastConcat) to bail and the dstSlice loop to fire.
        /// </summary>
        [TestMethod]
        public void Concatenate_GeneralPath_TransposedSource_ProducesCorrectValues()
        {
            var a = np.arange(12).reshape(3, 4).astype(NPTypeCode.Int32);
            var b = np.arange(12, 24).reshape(3, 4).astype(NPTypeCode.Int32);

            // Transpose forces non-contig — fast paths must reject these.
            var aT = a.T;
            var bT = b.T;
            aT.Shape.IsContiguous.Should().BeFalse();
            bT.Shape.IsContiguous.Should().BeFalse();

            var c = np.concatenate(new[] { aT, bT }, axis: 1);

            c.shape.Should().ContainInOrder(4L, 6L);
            // Column 0 of aT == row 0 of a == [0, 4, 8].
            ((int)c[0, 0]).Should().Be(0);
            ((int)c[1, 0]).Should().Be(1);
            ((int)c[2, 0]).Should().Be(2);
            ((int)c[3, 0]).Should().Be(3);
            ((int)c[0, 3]).Should().Be(12);
            ((int)c[3, 5]).Should().Be(23);
        }

        /// <summary>
        /// axis=null path: ravel each input and concatenate. The fresh wrappers
        /// allocated in <c>disposableWorkArrays</c> are released in the finally;
        /// the caller's original arrays must remain untouched.
        /// </summary>
        [TestMethod]
        public void Concatenate_AxisNull_RavelsAndConcatenates_PreservingInputs()
        {
            var a = np.arange(6).reshape(2, 3).astype(NPTypeCode.Int32);
            var b = np.arange(6, 14).reshape(2, 4).astype(NPTypeCode.Int32);

            var c = np.concatenate(new[] { a, b }, axis: (int?)null);

            c.shape.Should().ContainInOrder(14L);
            for (int i = 0; i < 14; i++)
                ((int)c[i]).Should().Be(i);

            // Caller's arrays must still be alive and readable after the
            // concatenate returns (disposing ravel wrappers must not dispose
            // the inputs they alias).
            a.IsDisposed.Should().BeFalse();
            b.IsDisposed.Should().BeFalse();
            a.Storage.InternalArray.IsReleased.Should().BeFalse();
            b.Storage.InternalArray.IsReleased.Should().BeFalse();
            ((int)a[1, 2]).Should().Be(5);
            ((int)b[1, 3]).Should().Be(13);
        }

        /// <summary>
        /// Non-contig sources with cross-dtype: forces the general path AND
        /// makes dstSlice + NDIter.Copy do the casting work. Verifies the
        /// using on dstSlice doesn't cut the slice off mid-copy.
        /// </summary>
        [TestMethod]
        public void Concatenate_GeneralPath_CrossDtype_TransposedSource_CorrectValues()
        {
            var a = np.arange(6).reshape(2, 3).astype(NPTypeCode.Int32);
            var b = np.arange(6, 12).reshape(2, 3).astype(NPTypeCode.Double);

            var aT = a.T;
            var bT = b.T;

            var c = np.concatenate(new[] { aT, bT }, axis: 1);

            c.dtype.Should().Be(typeof(double));
            c.shape.Should().ContainInOrder(3L, 4L);
            // aT[0,0] == a[0,0] == 0; bT[0,0] == b[0,0] == 6.
            ((double)c[0, 0]).Should().Be(0.0);
            ((double)c[0, 2]).Should().Be(6.0);
            ((double)c[2, 3]).Should().Be(11.0);
        }

        // --------------------------- leak guard ---------------------------

        /// <summary>
        /// Repeated axis=null concatenates should not leak working-set growth.
        /// Without <c>using</c> on the ravel'd wrappers, each iteration left
        /// two NDArray wrappers on the finalizer queue, each holding an ARC
        /// ref to the input buffer.
        /// </summary>
        [TestMethod]
        public void Concatenate_AxisNull_TightLoop_DoesNotLeakWorkingSet()
        {
            // Warm-up pass — bring JIT, kernels, and one-shot allocations
            // into steady-state before we measure.
            for (int i = 0; i < 20; i++)
            {
                using var a = new NDArray(NPTypeCode.Double, new Shape(50_000), fillZeros: true);
                using var b = new NDArray(NPTypeCode.Double, new Shape(50_000), fillZeros: true);
                using var c = np.concatenate(new[] { a, b }, axis: (int?)null);
            }
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var p = Process.GetCurrentProcess();
            p.Refresh();
            long start = p.WorkingSet64;

            for (int i = 0; i < 500; i++)
            {
                using var a = new NDArray(NPTypeCode.Double, new Shape(50_000), fillZeros: true);
                using var b = new NDArray(NPTypeCode.Double, new Shape(50_000), fillZeros: true);
                using var c = np.concatenate(new[] { a, b }, axis: (int?)null);
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            p.Refresh();
            long deltaMB = (p.WorkingSet64 - start) / (1024 * 1024);

            // 500 iterations × (2 inputs + 1 output ≈ 800K + 800K + 1.6M ≈ 3.2 MiB
            // per iteration) — without ARC release, the finalizer queue would hold
            // 500-iter * 2-ravels = 1000 NDArray wrappers in flight plus all their
            // backing buffers. Steady-state with `using` keeps the delta near zero.
            // 20 MiB headroom covers natural GC pacing variation.
            // macOS WorkingSet64 reclaim is noisier than Windows/Linux; allow more headroom there
            // (the leak guarded against is platform-independent and stays tight on the other CI OSes).
            long limitMB = RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? 96 : 20;
            deltaMB.Should().BeLessThan(limitMB);
        }

        /// <summary>
        /// General path with transposed (non-contig) sources: tight-loop allocation
        /// behaviour. The dstSlice wrapper allocated per source-per-iteration must
        /// not accumulate.
        /// </summary>
        [TestMethod]
        public void Concatenate_GeneralPath_TightLoop_DoesNotLeakWorkingSet()
        {
            // Warm-up
            for (int i = 0; i < 20; i++)
            {
                using var a = np.arange(2500).reshape(50, 50).astype(NPTypeCode.Int32);
                using var b = np.arange(2500).reshape(50, 50).astype(NPTypeCode.Int32);
                using var c = np.concatenate(new[] { a.T, b.T }, axis: 1);
            }
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var p = Process.GetCurrentProcess();
            p.Refresh();
            long start = p.WorkingSet64;

            for (int i = 0; i < 500; i++)
            {
                using var a = np.arange(2500).reshape(50, 50).astype(NPTypeCode.Int32);
                using var b = np.arange(2500).reshape(50, 50).astype(NPTypeCode.Int32);
                using var c = np.concatenate(new[] { a.T, b.T }, axis: 1);
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            p.Refresh();
            long deltaMB = (p.WorkingSet64 - start) / (1024 * 1024);

            // macOS WorkingSet64 reclaim is noisier than Windows/Linux; allow more headroom there
            // (the leak guarded against is platform-independent and stays tight on the other CI OSes).
            long limitMB = RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? 96 : 20;
            deltaMB.Should().BeLessThan(limitMB);
        }
    }
}
