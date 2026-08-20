using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OpenBLAS;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     <b>Live byte-for-byte parity of <c>np.correlate</c> / <c>np.convolve</c> against real NumPy.</b>
    ///     NumPy's <c>_pyarray_correlate</c> reduces every ramp position — and the middle whenever
    ///     <c>small_correlate</c> declines (a real kernel longer than 11, or any complex kernel) — with
    ///     the cblas <c>?dot</c> (<c>?dotu</c> for complex) of its per-dtype <c>dotfunc</c>. NumSharp.Core
    ///     computes that reduction with its own SIMD kernel, which reorders the sum, so a LONG float
    ///     kernel drifts by a bounded ULP from NumPy. With <c>NumSharp.Interop.OpenBLAS</c> installed the
    ///     sliding kernels route those exact positions through the SAME scipy-openblas
    ///     <c>sdot</c>/<c>ddot</c>/<c>zdotu</c> NumPy calls (the <see cref="ISlidingDotBackend"/> seam),
    ///     so the answer becomes byte-identical — the same lever, and the same three parity conditions
    ///     (build, thread count, DYNAMIC_ARCH kernel), as the product family in
    ///     <see cref="ProductsLiveParityTests"/>.
    /// </summary>
    /// <remarks>
    ///     Each result is computed once by NumSharp's backend-routed kernel and once by the embedded
    ///     CPython's numpy over the SAME zero-copy exported operands, and asserted byte-identical. The
    ///     kernels are kept LONG (n2 = 64 ≫ 11) with mixed-magnitude values, which is exactly the regime
    ///     the managed reduction diverges on — so a passing byte compare is non-vacuous
    ///     (<see cref="Convolve_Float64_Long_Managed_Diverges_NonVacuity"/> pins that the managed path
    ///     really does differ, i.e. the backend routing is what earns the parity). The small-kernel
    ///     boundary is pinned too: a length-5 real kernel STAYS on the managed path (NumPy's
    ///     <c>small_correlate</c>) and is already byte-exact without the backend.
    /// </remarks>
    [TestClass]
    public class SlidingDotLiveParityTests : InteropTestBase
    {
        /// <summary>Inconclusive when no CBLAS is loaded (the real f32/f64 sliding dot needs one).</summary>
        private static void RequireBackend()
        {
            if (!OpenBlasEngine.Enabled)
                Assert.Inconclusive("OpenBLAS backend not enabled on this host.");
        }

        /// <summary>Inconclusive when the loaded library lacks the complex128 products.</summary>
        private static void RequireComplex()
        {
            RequireBackend();
            if (!OpenBlasEngine.ComplexProductsAvailable)
                Assert.Inconclusive("OpenBLAS complex128 products not available on this host.");
        }

        /// <summary>
        ///     Computes the sliding op in NumSharp (passed as <paramref name="ns"/>) and again in numpy
        ///     over the SAME exported operands, asserting byte-identical. <paramref name="npExpr"/> is a
        ///     numpy expression over the bound names <c>a</c> and <c>b</c>.
        /// </summary>
        private void AssertSliding(NDArray ns, string npExpr, NDArray a, NDArray b, string because)
        {
            using (ns)
            using (Gil())
            {
                using PyObject va = a.ToNumpy();
                using PyObject vb = b.ToNumpy();
                using PyObject np_ = Python.np.with(npExpr, ("a", va), ("b", vb));
                ByteContract.AssertSameBytes(ns, np_, because);
            }
        }

        // ---- operand builders: LONG (n2 = 64), mixed-magnitude so the sum order is observable --------

        private const int N1 = 300;
        private const int N2Long = 64;   // > 11 => the cblas regime
        private const int N2Small = 5;   // <= 11 => small_correlate regime (managed, already byte-exact)

        private static double Sig(int i) => Math.Sin(i * 0.7) * 1000.0 + Math.Cos(i * i * 0.001) * 1e6 + i;
        private static double Ker(int j) => Math.Cos(j * 0.3) * 1e5 - j * 0.01 + Math.Sin(j * 1.3) * 1e3;

        private static NDArray RSignal()
        {
            var d = new double[N1];
            for (int i = 0; i < N1; i++) d[i] = Sig(i);
            return np.array(d);
        }

        private static NDArray RKernel(int n)
        {
            var d = new double[n];
            for (int j = 0; j < n; j++) d[j] = Ker(j);
            return np.array(d);
        }

        private static NDArray CSignal()
        {
            var d = new Complex[N1];
            for (int i = 0; i < N1; i++) d[i] = new Complex(Sig(i), Math.Cos(i * 0.9) * 1e4);
            return np.array(d);
        }

        private static NDArray CKernel(int n)
        {
            var d = new Complex[n];
            for (int j = 0; j < n; j++) d[j] = new Complex(Ker(j), Math.Sin(j * 1.1) * 1e3);
            return np.array(d);
        }

        // =====================================  backend  =========================================

        [TestMethod]
        public void Backend_IsOpenBlasEnabled()
        {
            Assert.IsTrue(OpenBlasEngine.Enabled, "the interop suite runs with OpenBLAS as the default backend");
        }

        // =====================================  np.convolve  =====================================

        [TestMethod]
        public void Convolve_Float64_Long_Full_ByteExact()
        {
            RequireBackend();
            AssertSliding(RSignal().convolve(RKernel(N2Long), "full"), "np.convolve(a,b,'full')",
                RSignal(), RKernel(N2Long), "convolve float64 long full — ramps + middle via ddot");
        }

        [TestMethod]
        public void Convolve_Float64_Long_Same_ByteExact()
        {
            RequireBackend();
            AssertSliding(RSignal().convolve(RKernel(N2Long), "same"), "np.convolve(a,b,'same')",
                RSignal(), RKernel(N2Long), "convolve float64 long same — asymmetric ramps + middle via ddot");
        }

        [TestMethod]
        public void Convolve_Float64_Long_Valid_ByteExact()
        {
            RequireBackend();
            // 'valid' has no ramps: this pins the MIDDLE-only path (every fully-overlapping dot via ddot).
            AssertSliding(RSignal().convolve(RKernel(N2Long), "valid"), "np.convolve(a,b,'valid')",
                RSignal(), RKernel(N2Long), "convolve float64 long valid — middle only via ddot");
        }

        [TestMethod]
        public void Convolve_Float32_Long_Full_ByteExact()
        {
            RequireBackend();
            var a = RSignal().astype(np.float32);
            var v = RKernel(N2Long).astype(np.float32);
            AssertSliding(a.convolve(v, "full"), "np.convolve(a,b,'full')",
                RSignal().astype(np.float32), RKernel(N2Long).astype(np.float32),
                "convolve float32 long full — double-accumulated sdot");
        }

        [TestMethod]
        public void Convolve_Complex128_Long_Full_ByteExact()
        {
            RequireComplex();
            // small_correlate never applies to complex, so EVERY position (any kernel length) routes
            // through zdotu — this is the complex byte-parity the managed double-accumulate could not give.
            AssertSliding(CSignal().convolve(CKernel(N2Long), "full"), "np.convolve(a,b,'full')",
                CSignal(), CKernel(N2Long), "convolve complex128 long full — zdotu (unconjugated)");
        }

        // =====================================  np.correlate  ====================================

        [TestMethod]
        public void Correlate_Float64_Long_Valid_ByteExact()
        {
            RequireBackend();
            AssertSliding(RSignal().correlate(RKernel(N2Long), "valid"), "np.correlate(a,b,'valid')",
                RSignal(), RKernel(N2Long), "correlate float64 long valid — middle only via ddot");
        }

        [TestMethod]
        public void Correlate_Float64_Long_Full_ByteExact()
        {
            RequireBackend();
            AssertSliding(RSignal().correlate(RKernel(N2Long), "full"), "np.correlate(a,b,'full')",
                RSignal(), RKernel(N2Long), "correlate float64 long full — growing/shrinking ramps via ddot");
        }

        [TestMethod]
        public void Correlate_Complex128_Long_Valid_ByteExact()
        {
            RequireComplex();
            // correlate conjugates the kernel first, then the engine reads it forward — the zdotu (NOT
            // zdotc) still matches, because the conjugation happened on the operand, not in the dot.
            AssertSliding(CSignal().correlate(CKernel(N2Long), "valid"), "np.correlate(a,b,'valid')",
                CSignal(), CKernel(N2Long), "correlate complex128 long valid — conj-kernel then zdotu");
        }

        // =====================================  boundary  ========================================

        [TestMethod]
        public void Convolve_SmallRealKernel_StaysManaged_ByteExact()
        {
            RequireBackend();
            // A length-5 real kernel is NumPy's small_correlate regime: the managed outer-vec middle and
            // the <= 11 ramps are already byte-identical to NumPy, so the engine keeps it managed — and it
            // must STILL be byte-exact. This pins the boundary the backend gate sits on (n2 > 11).
            AssertSliding(RSignal().convolve(RKernel(N2Small), "full"), "np.convolve(a,b,'full')",
                RSignal(), RKernel(N2Small), "convolve float64 SMALL full — stays managed, still byte-exact");
        }

        // =====================================  non-vacuity  =====================================

        [TestMethod]
        public void Convolve_Float64_Long_Managed_Diverges_NonVacuity()
        {
            RequireBackend();
            // Guard the gate against vacuity: on a LONG kernel the managed reduction reorders the sum vs
            // cblas ddot, so with the backend DISABLED the result must DIFFER from numpy. If it matched,
            // every byte-exact assertion above would be proving nothing (the backend would be a no-op).
            var a = RSignal();
            var v = RKernel(N2Long);

            NDArray managed;
            OpenBlasEngine.Disable();
            try { managed = a.convolve(v, "full"); }
            finally { OpenBlasEngine.TryEnable(threads: 1); } // restore the suite's backend state

            using (managed)
            using (Gil())
            {
                using PyObject va = a.ToNumpy();
                using PyObject vb = v.ToNumpy();
                using PyObject np_ = Python.np.with("np.convolve(a,b,'full')", ("a", va), ("b", vb));
                ByteContract.NsBytes(managed).Should().NotEqual(np_.bytes_c(),
                    "the managed sliding-dot reorders the long-kernel sum vs cblas, so it must diverge — " +
                    "which is what the backend routing exists to fix");
            }

            Assert.IsTrue(OpenBlasEngine.Enabled, "backend must be restored for the rest of the suite");
        }
    }
}
