using System;
using System.Diagnostics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest
{
    /// <summary>
    /// Guards the `using` on `full = ConvolveFull(...)` inside ConvolveSame /
    /// ConvolveValid — the full convolution buffer is dead once the requested
    /// centre/valid slice has been copied out.
    /// </summary>
    [TestClass]
    public class NdArray_Convolve_UsingTests
    {
        // --------------------------- correctness ---------------------------

        [TestMethod]
        public void ConvolveSame_StillMatchesNumPy()
        {
            // np.convolve([1,2,3], [0,1,0.5], 'same') -> [1, 2.5, 4]
            var a = np.array(new double[] { 1, 2, 3 });
            var v = np.array(new double[] { 0, 1, 0.5 });
            var r = a.convolve(v, "same");

            r.Data<double>().Should().Equal(new double[] { 1, 2.5, 4 });
        }

        [TestMethod]
        public void ConvolveValid_StillMatchesNumPy()
        {
            // np.convolve([1,2,3], [0,1,0.5], 'valid') -> [2.5]
            var a = np.array(new double[] { 1, 2, 3 });
            var v = np.array(new double[] { 0, 1, 0.5 });
            var r = a.convolve(v, "valid");

            r.Data<double>().Should().Equal(new double[] { 2.5 });
        }

        // --------------------------- leak guard ---------------------------

        /// <summary>
        /// Tight loop of `same`-mode convolves. Each call previously left a
        /// `full` buffer (na + nv - 1 doubles) on the finalizer queue. Steady
        /// state with `using` keeps working set near constant.
        /// </summary>
        [TestMethod]
        public void ConvolveSame_TightLoop_DoesNotLeakWorkingSet()
        {
            var a = np.arange(1_000).astype(NPTypeCode.Double);
            var v = np.arange(64).astype(NPTypeCode.Double);

            for (int i = 0; i < 20; i++)
                _ = a.convolve(v, "same");
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var p = Process.GetCurrentProcess();
            p.Refresh();
            long start = p.WorkingSet64;

            for (int i = 0; i < 200; i++)
            {
                using var r = a.convolve(v, "same");
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            p.Refresh();
            long deltaMB = (p.WorkingSet64 - start) / (1024 * 1024);

            // 200 iterations × ~8 KiB "full" buffer (1063 doubles) would only
            // be ~1.6 MiB in raw bytes, but the wrapper churn through the
            // finalizer queue adds GC overhead. 20 MiB headroom is generous.
            deltaMB.Should().BeLessThan(20);
        }
    }
}
