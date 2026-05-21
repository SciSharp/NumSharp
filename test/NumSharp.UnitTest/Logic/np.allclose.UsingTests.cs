using System;
using System.Diagnostics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Logic
{
    /// <summary>
    /// Guards the <c>using</c> on the <c>np.isclose</c> intermediate inside
    /// <c>DefaultEngine.AllClose</c>. The intermediate is a bool array the
    /// shape of broadcast(a, b); without atomic release each call in a tight
    /// loop left a finalizer-queue entry per evaluation.
    /// </summary>
    [TestClass]
    public class np_allclose_using_test : TestClass
    {
        // --------------------------- correctness ---------------------------

        [TestMethod]
        public void AllClose_TrueCase_AfterRefactor()
        {
            // NumPy: np.allclose([1e10, 1e-8], [1.00001e10, 1e-9]) -> True
            np.allclose(new[] { 1e10, 1e-8 }, new[] { 1.00001e10, 1e-9 })
                .Should().BeTrue();
        }

        [TestMethod]
        public void AllClose_FalseCase_AfterRefactor()
        {
            // NumPy: np.allclose([1e10, 1e-7], [1.00001e10, 1e-8]) -> False
            np.allclose(new[] { 1e10, 1e-7 }, new[] { 1.00001e10, 1e-8 })
                .Should().BeFalse();
        }

        [TestMethod]
        public void AllClose_EqualNan_AfterRefactor()
        {
            // equal_nan=True: NaN==NaN by special-case branch in IsClose.
            np.allclose(new[] { 1.0, np.nan }, new[] { 1.0, np.nan }, equal_nan: true)
                .Should().BeTrue();
            np.allclose(new[] { 1.0, np.nan }, new[] { 1.0, np.nan })
                .Should().BeFalse();
        }

        // --------------------------- leak guard ---------------------------

        /// <summary>
        /// Tight loop of allcloses on 50K-element arrays. Each call previously
        /// allocated and dropped a 50K-bool array; the using on the np.isclose
        /// intermediate should drive working-set delta to near zero.
        /// </summary>
        [TestMethod]
        public void AllClose_TightLoop_DoesNotLeakWorkingSet()
        {
            // Warm-up
            for (int i = 0; i < 20; i++)
            {
                using var a = np.zeros(new Shape(50_000), NPTypeCode.Double);
                using var b = np.zeros(new Shape(50_000), NPTypeCode.Double);
                _ = np.allclose(a, b);
            }
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var p = Process.GetCurrentProcess();
            p.Refresh();
            long start = p.WorkingSet64;

            for (int i = 0; i < 1000; i++)
            {
                using var a = np.zeros(new Shape(50_000), NPTypeCode.Double);
                using var b = np.zeros(new Shape(50_000), NPTypeCode.Double);
                _ = np.allclose(a, b);
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            p.Refresh();
            long deltaMB = (p.WorkingSet64 - start) / (1024 * 1024);

            // Without `using` on the closeness array, 1000 calls would queue
            // up 1000 * 50K-bool wrappers. 20 MiB headroom covers natural
            // GC pacing variation.
            deltaMB.Should().BeLessThan(20);
        }
    }
}
