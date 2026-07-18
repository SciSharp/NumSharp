using System;
using System.Diagnostics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// Guards the `using` on `fcopy = this.copy('F')` inside the F-order
    /// branch of NDArray.flatten. The returned NDArray shares storage with
    /// fcopy and bumps the refcount via InitializeArc — disposing fcopy
    /// only drops fcopy's wrapper ref.
    /// </summary>
    [TestClass]
    public class NDArray_flatten_UsingTests : TestClass
    {
        // --------------------------- correctness ---------------------------

        [TestMethod]
        public void Flatten_CDefault_StillCorrect()
        {
            // np.flatten() default == 'C'. Doesn't hit the modified branch.
            var a = np.arange(6).reshape(2, 3).astype(NPTypeCode.Int32);
            var f = a.flatten();

            f.shape.Should().ContainInOrder(6L);
            for (int i = 0; i < 6; i++)
                ((int)f[i]).Should().Be(i);
        }

        [TestMethod]
        public void Flatten_EmptyViewWithNonZeroOffset_ReturnsEmpty()
        {
            // Regression: an empty slice keeps its parent's offset while its own backing buffer
            // Count collapses to 0. UnmanagedStorage.CloneData used to Slice(offset, 0) on that
            // zero-length buffer (start=offset > Count=0) and throw ArgumentOutOfRangeException.
            // NumPy: arr[1:,1:,1:1].flatten() -> shape (0,).
            var v = np.arange(24).reshape(4, 3, 2)["1:, 1:, 1:1"];   // (3,2,0), offset != 0
            v.size.Should().Be(0);

            var f = v.flatten();
            f.shape.Should().ContainInOrder(0L);
            f.size.Should().Be(0);

            // 1-D empty offset view (offset via a mid-array empty slice)
            np.arange(10)["5:5"].flatten().size.Should().Be(0);

            // The sibling materializers agree.
            v.ravel().size.Should().Be(0);
            v.copy().size.Should().Be(0);
            v.astype(NPTypeCode.Double).size.Should().Be(0);
        }

        [TestMethod]
        public void Flatten_FOrder_StillCorrect()
        {
            // np.flatten('F') reads column-major.
            // For 2x3 [[0,1,2],[3,4,5]] → F-flat = [0,3,1,4,2,5].
            var a = np.arange(6).reshape(2, 3).astype(NPTypeCode.Int32);
            var f = a.flatten('F');

            f.shape.Should().ContainInOrder(6L);
            ((int)f[0]).Should().Be(0);
            ((int)f[1]).Should().Be(3);
            ((int)f[2]).Should().Be(1);
            ((int)f[3]).Should().Be(4);
            ((int)f[4]).Should().Be(2);
            ((int)f[5]).Should().Be(5);
        }

        /// <summary>
        /// flatten('F') always materializes a fresh copy — disposing the
        /// source must not invalidate the returned flat array.
        /// </summary>
        [TestMethod]
        public void Flatten_FOrder_ResultSurvivesSourceDispose()
        {
            var source = np.arange(20).reshape(4, 5).astype(NPTypeCode.Int32);
            var flat = source.flatten('F');

            source.Dispose();

            // flat must remain valid: it shares storage with the using-bound
            // fcopy inside flatten, which holds its own ARC ref.
            flat.IsDisposed.Should().BeFalse();
            flat.Storage.InternalArray.IsReleased.Should().BeFalse();
            ((int)flat[0]).Should().Be(0);
            ((int)flat[19]).Should().Be(19);
        }

        // --------------------------- leak guard ---------------------------

        [TestMethod]
        public void Flatten_FOrder_TightLoop_DoesNotLeakWorkingSet()
        {
            using var a = np.arange(200 * 100).reshape(200, 100).astype(NPTypeCode.Double);

            for (int i = 0; i < 20; i++)
            {
                using var f = a.flatten('F');
            }
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var p = Process.GetCurrentProcess();
            p.Refresh();
            long start = p.WorkingSet64;

            for (int i = 0; i < 500; i++)
            {
                using var f = a.flatten('F');
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            p.Refresh();
            long deltaMB = (p.WorkingSet64 - start) / (1024 * 1024);

            // Each call allocates a fresh 200×100 Double F-copy (~160 KiB).
            // Without using on fcopy, each iteration left an NDArray wrapper
            // on the finalizer queue (the buffer itself is kept alive by
            // the returned flat anyway, so this is wrapper churn).
            deltaMB.Should().BeLessThan(30);
        }
    }
}
