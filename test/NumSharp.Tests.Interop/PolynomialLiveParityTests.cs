using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OpenBLAS;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     Live-numpy byte-parity for the BLAS-routed members of the polynomial family —
    ///     <c>np.roots</c> and <c>np.poly</c> (of a matrix) through <c>eigvals</c> (geev), and
    ///     <c>np.polyfit</c> through <c>lstsq</c> (gelsd). Each case computes the op TWICE over the same
    ///     exported bytes — once in NumSharp on the bundled scipy-openblas backend, once in the LIVE
    ///     numpy in this process — and asserts they agree BYTE-FOR-BYTE.
    /// </summary>
    /// <remarks>
    ///     <c>roots</c>'s companion-matrix eigenvalues and <c>polyfit</c>'s scaled-Vandermonde least
    ///     squares both bottom out in the SAME single-threaded scipy-openblas routine numpy pins, so the
    ///     result bytes are a deterministic function of the input. The managed pre/post-processing
    ///     (<c>vander</c>, the column scaling, the convolve fold) is small and exact for these operands.
    ///     Offline value gates (no Python, no backend contract) live in
    ///     <c>test/NumSharp.Tests/Polynomial/PolynomialTests.cs</c>.
    /// </remarks>
    [TestClass]
    public class PolynomialLiveParityTests : InteropTestBase
    {
        private static void RequireLapack()
        {
            if (!OpenBlasEngine.LapackAvailable)
                Assert.Inconclusive("OpenBLAS LAPACK backend not available on this host.");
        }

        /// <summary>Assert NumSharp's result is byte-identical to numpy computing <paramref name="npExpr"/> over the SAME export.</summary>
        private void SameBytes(NDArray nsResult, string npExpr, params (string name, NDArray operand)[] ops)
        {
            using (Gil())
            {
                var exports = new PyObject[ops.Length];
                try
                {
                    var vars = new (string, PyObject)[ops.Length];
                    for (int i = 0; i < ops.Length; i++)
                    {
                        exports[i] = ops[i].operand.ToNumpy();
                        vars[i] = (ops[i].name, exports[i]);
                    }

                    using PyObject npResult = Python.np.with(npExpr, vars);
                    ByteContract.AssertSameBytes(nsResult, npResult, npExpr);
                }
                finally
                {
                    foreach (var e in exports) e?.Dispose();
                }
            }
        }

        // ================================  roots  ==================================================

        [TestMethod]
        public void Roots_ByteExact_RealAndComplex()
        {
            RequireLapack();
            // real roots
            SameBytes(np.roots(np.array(new double[] { 1, -6, 11, -6 })), "np.roots(p)",
                ("p", np.array(new double[] { 1, -6, 11, -6 })));
            // complex-conjugate roots
            SameBytes(np.roots(np.array(new double[] { 3.2, 2, 1 })), "np.roots(p)",
                ("p", np.array(new double[] { 3.2, 2, 1 })));
            // trailing zeros -> roots at 0 appended
            SameBytes(np.roots(np.array(new double[] { 1, -1, 0, 0 })), "np.roots(p)",
                ("p", np.array(new double[] { 1, -1, 0, 0 })));
        }

        [TestMethod]
        public void Roots_ByteExact_IntAndFloat32Inputs()
        {
            RequireLapack();
            SameBytes(np.roots(np.array(new long[] { 1, 0, -1 })), "np.roots(p)",
                ("p", np.array(new long[] { 1, 0, -1 })));
            SameBytes(np.roots(np.array(new float[] { 1, 0, -1 })), "np.roots(p)",
                ("p", np.array(new float[] { 1, 0, -1 })));
        }

        // ================================  poly (matrix)  ==========================================

        [TestMethod]
        public void Poly_OfMatrix_ByteExact()
        {
            RequireLapack();
            var m = np.array(new double[,] { { 0, 1.0 / 3 }, { -0.5, 0 } });
            SameBytes(np.poly(m), "np.poly(m)", ("m", m));

            var m2 = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            SameBytes(np.poly(m2), "np.poly(m)", ("m", m2));
        }

        // ================================  polyfit  ================================================

        private static NDArray Xs() => np.array(new double[] { 0, 1, 2, 3, 4, 5 });
        private static NDArray Ys() => np.array(new double[] { 0, 0.8, 0.9, 0.1, -0.8, -1.0 });

        [TestMethod]
        public void Polyfit_Coeffs_ByteExact()
        {
            RequireLapack();
            SameBytes(np.polyfit(Xs(), Ys(), 2), "np.polyfit(x, y, 2)", ("x", Xs()), ("y", Ys()));
            SameBytes(np.polyfit(Xs(), Ys(), 3), "np.polyfit(x, y, 3)", ("x", Xs()), ("y", Ys()));
        }

        [TestMethod]
        public void Polyfit_Full_ByteExact()
        {
            RequireLapack();
            var (c, resids, _, s, _) = np.polyfit(Xs(), Ys(), 3, full: true);
            SameBytes(c, "np.polyfit(x, y, 3, full=True)[0]", ("x", Xs()), ("y", Ys()));
            SameBytes(resids, "np.polyfit(x, y, 3, full=True)[1]", ("x", Xs()), ("y", Ys()));
            SameBytes(s, "np.polyfit(x, y, 3, full=True)[3]", ("x", Xs()), ("y", Ys()));
        }

        [TestMethod]
        public void Polyfit_Weighted_ByteExact()
        {
            RequireLapack();
            var w = np.array(new double[] { 1, 1, 1, 1, 2, 2 });
            SameBytes(np.polyfit(Xs(), Ys(), 2, w: w), "np.polyfit(x, y, 2, w=w)",
                ("x", Xs()), ("y", Ys()), ("w", w));
        }

        [TestMethod]
        public void Polyfit_Covariance_ByteExact()
        {
            RequireLapack();
            var (c, cov) = np.polyfit(Xs(), Ys(), 2, cov: true);
            SameBytes(c, "np.polyfit(x, y, 2, cov=True)[0]", ("x", Xs()), ("y", Ys()));
            SameBytes(cov, "np.polyfit(x, y, 2, cov=True)[1]", ("x", Xs()), ("y", Ys()));
        }

        [TestMethod]
        public void Polyfit_UnscaledCovariance_ByteExact()
        {
            RequireLapack();
            var (_, cov) = np.polyfit(Xs(), Ys(), 2, cov: "unscaled");
            SameBytes(cov, "np.polyfit(x, y, 2, cov='unscaled')[1]", ("x", Xs()), ("y", Ys()));
        }

        [TestMethod]
        public void Polyfit_Deg0_NegativeZero_ByteExact()
        {
            RequireLapack();
            // The degree-0 fit is the (weighted) mean's negation artifact -0.0 — a signed zero that
            // must survive byte-for-byte (bit pattern 0x8000000000000000).
            SameBytes(np.polyfit(Xs(), Ys(), 0), "np.polyfit(x, y, 0)", ("x", Xs()), ("y", Ys()));
        }
    }
}
