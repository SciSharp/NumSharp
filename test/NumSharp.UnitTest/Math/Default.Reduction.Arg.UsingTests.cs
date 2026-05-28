using System;
using System.Diagnostics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.MathSuite
{
    /// <summary>
    /// Guards the `using` on per-iter `slice = arr[slices]` inside
    /// ArgReductionAxisFallback. The fallback rarely fires because IL kernels
    /// cover all common (dtype, op, contig/strided) combinations, but the
    /// refactor must still be correctness-safe for the cases it does cover.
    /// </summary>
    [TestClass]
    public class Default_Reduction_Arg_UsingTests : TestClass
    {
        // --------------------------- correctness ---------------------------

        [TestMethod]
        public void ArgMax_Axis_2D_StillCorrect()
        {
            // np.argmax(arange(12).reshape(3,4), axis=1) -> [3, 3, 3]
            var a = np.arange(12).reshape(3, 4).astype(NPTypeCode.Int32);
            var r = np.argmax(a, axis: 1);
            ((long)r[0]).Should().Be(3L);
            ((long)r[1]).Should().Be(3L);
            ((long)r[2]).Should().Be(3L);
        }

        [TestMethod]
        public void ArgMin_Axis_2D_StillCorrect()
        {
            // np.argmin(arange(12).reshape(3,4), axis=1) -> [0, 0, 0]
            var a = np.arange(12).reshape(3, 4).astype(NPTypeCode.Int32);
            var r = np.argmin(a, axis: 1);
            ((long)r[0]).Should().Be(0L);
            ((long)r[1]).Should().Be(0L);
            ((long)r[2]).Should().Be(0L);
        }

        [TestMethod]
        public void ArgMax_Axis_Transposed_StillCorrect()
        {
            // Transpose forces a non-contig source — the strided IL kernel
            // path. Numerical answers still match the contiguous version.
            var a = np.arange(12).reshape(3, 4).astype(NPTypeCode.Int32);
            var aT = a.T;  // (4, 3), non-contig
            var r = np.argmax(aT, axis: 1);  // along the original-rows axis

            r.size.Should().Be(4L);
            // aT[0] = [0, 4, 8]   → argmax = 2
            // aT[1] = [1, 5, 9]   → argmax = 2
            // aT[2] = [2, 6, 10]  → argmax = 2
            // aT[3] = [3, 7, 11]  → argmax = 2
            ((long)r[0]).Should().Be(2L);
            ((long)r[3]).Should().Be(2L);
        }

        // --------------------------- leak guard ---------------------------

        /// <summary>
        /// Tight loop of axis argmax — even if the IL kernel path is taken,
        /// the broader argmax flow can leak intermediates. The using on
        /// the fallback's `slice` is a belt-and-suspenders guard.
        /// </summary>
        [TestMethod]
        public void ArgMax_Axis_TightLoop_DoesNotLeakWorkingSet()
        {
            using var a = np.arange(2000 * 64).reshape(2000, 64).astype(NPTypeCode.Int32);

            for (int i = 0; i < 20; i++)
                _ = np.argmax(a, axis: 1);
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var p = Process.GetCurrentProcess();
            p.Refresh();
            long start = p.WorkingSet64;

            for (int i = 0; i < 500; i++)
            {
                using var r = np.argmax(a, axis: 1);
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            p.Refresh();
            long deltaMB = (p.WorkingSet64 - start) / (1024 * 1024);

            deltaMB.Should().BeLessThan(30);
        }
    }
}
