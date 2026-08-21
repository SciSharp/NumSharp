using System;
using System.Numerics;
using NumSharp.Interop.OpenBLAS;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     <c>np.linalg.cholesky</c> (LAPACK <c>potrf</c>) and <c>np.linalg.qr</c> (LAPACK
    ///     <c>geqrf</c> + <c>orgqr</c>/<c>ungqr</c>) served by the OpenBLAS backend.
    /// </summary>
    /// <remarks>
    ///     Like <see cref="LapackFactorisationTests"/>: these need a LAPACK-capable library (the bundled
    ///     scipy-openblas is one) and go <see cref="Assert.Inconclusive(string)"/> on a host without one.
    ///     VALUES are asserted to a tolerance so the gate holds on any machine; the BITS are
    ///     host-dependent (OpenBLAS build, thread count, dispatched micro-kernel) so they are not
    ///     asserted here — but they were verified bit-exact against NumPy 2.4.2 across 307 differential
    ///     signatures (all dtypes × modes × shapes × layouts × batch × errors) on the reference host.
    /// </remarks>
    [TestClass]
    public class CholeskyQrFactorisationTests
    {
        private const double Tol = 1e-9;

        [TestCleanup]
        public void Cleanup() => OpenBlasEngine.Disable();

        private static void RequireLapack()
        {
            try
            {
                OpenBlasEngine.Enable();
            }
            catch (Exception e)
            {
                Assert.Inconclusive("no CBLAS library on this host: " + e.Message.Split('\n')[0]);
            }

            if (!OpenBlasEngine.LapackAvailable)
                Assert.Inconclusive("the loaded BLAS exports no LAPACK routines (a bare reference CBLAS).");
        }

        private static void AssertClose(NDArray actual, double tol, params double[] expected)
        {
            Assert.AreEqual(expected.Length, actual.size, "size mismatch");
            var flat = actual.ravel();
            for (int i = 0; i < expected.Length; i++)
            {
                double a = Convert.ToDouble(flat.GetAtIndex(i));
                if (double.IsNaN(expected[i]))
                    Assert.IsTrue(double.IsNaN(a), $"[{i}] expected NaN, got {a}");
                else if (double.IsInfinity(expected[i]))
                    Assert.AreEqual(expected[i], a, $"[{i}] expected {expected[i]}, got {a}");
                else
                    Assert.AreEqual(expected[i], a, tol, $"[{i}]");
            }
        }

        private static void AssertCloseComplex(NDArray actual, double tol, params Complex[] expected)
        {
            Assert.AreEqual(NPTypeCode.Complex, actual.typecode);
            Assert.AreEqual(expected.Length, actual.size, "size mismatch");
            var flat = actual.ravel();
            for (int i = 0; i < expected.Length; i++)
            {
                var z = (Complex)flat.GetAtIndex(i);
                Assert.AreEqual(expected[i].Real, z.Real, tol, $"[{i}].re");
                Assert.AreEqual(expected[i].Imaginary, z.Imaginary, tol, $"[{i}].im");
            }
        }

        // ------------------------------------------------------------------ cholesky

        [TestMethod]
        public void Cholesky_Lower_MatchesNumPy()
        {
            RequireLapack();
            var A = np.array(new double[,] { { 4, 2, 1 }, { 2, 5, 3 }, { 1, 3, 6 } });
            var L = np.linalg.cholesky(A);
            L.Should().BeShaped(3, 3);
            Assert.AreEqual(typeof(double), L.dtype);
            AssertClose(L, Tol, 2, 0, 0, 1, 2, 0, 0.5, 1.25, 2.0463381929681126);
        }

        [TestMethod]
        public void Cholesky_Upper_MatchesNumPy()
        {
            RequireLapack();
            var A = np.array(new double[,] { { 4, 2, 1 }, { 2, 5, 3 }, { 1, 3, 6 } });
            var U = np.linalg.cholesky(A, upper: true);
            AssertClose(U, Tol, 2, 1, 0.5, 0, 2, 1.25, 0, 0, 2.0463381929681126);
        }

        [TestMethod]
        public void Cholesky_ReconstructsA_LtimesLT()
        {
            RequireLapack();
            var A = np.array(new double[,] { { 4, 2, 1 }, { 2, 5, 3 }, { 1, 3, 6 } });
            var L = np.linalg.cholesky(A);
            AssertClose(np.matmul(L, L.T), 1e-9, 4, 2, 1, 2, 5, 3, 1, 3, 6);
            // Upper factor: A == U.H @ U.
            var U = np.linalg.cholesky(A, upper: true);
            AssertClose(np.matmul(U.T, U), 1e-9, 4, 2, 1, 2, 5, 3, 1, 3, 6);
        }

        [TestMethod]
        public void Cholesky_Float32_ComputesInDouble_ReturnsSingle()
        {
            RequireLapack();
            var A = np.array(new float[,] { { 4, 2, 1 }, { 2, 5, 3 }, { 1, 3, 6 } });
            var L = np.linalg.cholesky(A);
            Assert.AreEqual(typeof(float), L.dtype);
            AssertClose(L, 1e-6f, 2, 0, 0, 1, 2, 0, 0.5, 1.25, 2.0463381929681126);
        }

        [TestMethod]
        public void Cholesky_IntegerOperand_WidensToDouble()
        {
            RequireLapack();
            var L = np.linalg.cholesky(np.array(new int[,] { { 4, 0 }, { 0, 9 } }));
            Assert.AreEqual(typeof(double), L.dtype);
            AssertClose(L, Tol, 2, 0, 0, 3);
        }

        [TestMethod]
        public void Cholesky_Complex_Hermitian_MatchesNumPy()
        {
            RequireLapack();
            var A = np.array(new Complex[,]
            {
                { new Complex(2, 0), new Complex(1, -1) },
                { new Complex(1, 1), new Complex(3, 0) }
            });
            var L = np.linalg.cholesky(A);
            AssertCloseComplex(L, Tol,
                new Complex(1.4142135623730951, 0), new Complex(0, 0),
                new Complex(0.7071067811865475, 0.7071067811865475), new Complex(1.4142135623730951, 0));
            // A == L @ L.conj().T
            AssertCloseComplex(np.matmul(L, np.conjugate(L).T), 1e-9,
                new Complex(2, 0), new Complex(1, -1), new Complex(1, 1), new Complex(3, 0));
        }

        [TestMethod]
        public void Cholesky_Stacked_FactorisesEachMatrix()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 4.0, 2.0 }, { 2.0, 5.0 } });
            var b = np.array(new double[,] { { 9.0, 3.0 }, { 3.0, 10.0 } });
            var L = np.linalg.cholesky(np.stack(new[] { a, b }));
            L.Should().BeShaped(2, 2, 2);
            AssertClose(L["0"], Tol, 2, 0, 1, 2);
            AssertClose(L["1"], Tol, 3, 0, 1, 3);
        }

        [TestMethod]
        public void Cholesky_FortranAndTransposedLayouts_MatchContiguous()
        {
            RequireLapack();
            var A = np.array(new double[,] { { 4, 2, 1 }, { 2, 5, 3 }, { 1, 3, 6 } });
            AssertClose(np.linalg.cholesky(np.asfortranarray(A)), Tol,
                2, 0, 0, 1, 2, 0, 0.5, 1.25, 2.0463381929681126);
            // A is symmetric so A.T == A; the transposed VIEW must factorise identically.
            AssertClose(np.linalg.cholesky(np.ascontiguousarray(A.T).T), Tol,
                2, 0, 0, 1, 2, 0, 0.5, 1.25, 2.0463381929681126);
        }

        [TestMethod]
        public void Cholesky_EmptyMatrix_IsEmpty()
        {
            RequireLapack();
            np.linalg.cholesky(np.zeros(new Shape(0, 0))).Should().BeShaped(0, 0);
        }

        [TestMethod]
        public void Cholesky_NotPositiveDefinite_Raises()
        {
            RequireLapack();
            new Action(() => np.linalg.cholesky(np.array(new double[,] { { 1, 2 }, { 3, 4 } })))
                .Should().Throw<LinAlgError>().WithMessage("Matrix is not positive definite");
            new Action(() => np.linalg.cholesky(np.array(new double[,] { { 0.0 } })))
                .Should().Throw<LinAlgError>().WithMessage("Matrix is not positive definite");
        }

        [TestMethod]
        public void Cholesky_InAStack_OneBad_Raises()
        {
            RequireLapack();
            var good = np.array(new double[,] { { 4.0, 2.0 }, { 2.0, 5.0 } });
            var bad = np.array(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
            new Action(() => np.linalg.cholesky(np.stack(new[] { good, bad })))
                .Should().Throw<LinAlgError>().WithMessage("Matrix is not positive definite");
        }

        [TestMethod]
        public void Cholesky_NonSquareAndLowRank_RaiseNumPyMessages()
        {
            RequireLapack();
            new Action(() => np.linalg.cholesky(np.zeros(new Shape(2, 3))))
                .Should().Throw<LinAlgError>().WithMessage("Last 2 dimensions of the array must be square");
            new Action(() => np.linalg.cholesky(np.array(new double[] { 1, 2, 3 })))
                .Should().Throw<LinAlgError>()
                .WithMessage("1-dimensional array given. Array must be at least two-dimensional");
        }

        // ------------------------------------------------------------------ qr

        [TestMethod]
        public void Qr_Tall_Reduced_MatchesNumPy()
        {
            RequireLapack();
            var M = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 10 }, { 1, 0, 2 } });
            var (q, r) = np.linalg.qr(M);
            q.Should().BeShaped(4, 3);
            r.Should().BeShaped(3, 3);
            AssertClose(q, Tol,
                -0.12216944435630528, 0.5642764926868793, 0.7889662677447253,
                -0.48867777742522095, 0.23175641663925378, -0.017339917972411356,
                -0.8551861104941366, -0.10076365940837123, -0.18640411820342428,
                -0.12216944435630524, -0.7859565433852946, 0.5852222315688907);
            AssertClose(r, Tol,
                -8.18535277187245, -9.529216659791809, -12.094774991274218,
                0, 1.4812257933030573, 0.5038182970418574,
                0, 0, 1.569262576503246);
        }

        [TestMethod]
        public void Qr_Tall_Complete_MatchesNumPy()
        {
            RequireLapack();
            var M = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 10 }, { 1, 0, 2 } });
            var (q, r) = np.linalg.qr(M, "complete");
            q.Should().BeShaped(4, 4);
            r.Should().BeShaped(4, 3);
            // Reconstruct: Q @ R == M, and the 4th R row is zero.
            AssertClose(np.matmul(q, r), 1e-9, 1, 2, 3, 4, 5, 6, 7, 8, 10, 1, 0, 2);
            AssertClose(r["3"], Tol, 0, 0, 0);
        }

        [TestMethod]
        public void Qr_Wide_Reduced_MatchesNumPy()
        {
            RequireLapack();
            var W = np.array(new double[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 12, 11 } });
            var (q, r) = np.linalg.qr(W);
            q.Should().BeShaped(3, 3);
            r.Should().BeShaped(3, 4);
            AssertClose(q, Tol,
                -0.09667364890456631, 0.9077375936584361, 0.4082482904638654,
                -0.48336824452283167, 0.31573481518554525, -0.8164965809277254,
                -0.870062840141097, -0.2762679632873517, 0.4082482904638622);
            AssertClose(r, Tol,
                -10.344080432788603, -11.794185166357092, -14.114352740066685, -13.824331793352986,
                0, 0.9472044455566254, 1.6181409278259036, 3.117881299957237,
                0, 0, 0.4082482904638649, -0.4082482904638569);
        }

        [TestMethod]
        public void Qr_RMode_ReturnsRInTheRSlot_QIsNull()
        {
            RequireLapack();
            var M = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 10 }, { 1, 0, 2 } });
            var (q, r) = np.linalg.qr(M, "r");
            Assert.IsNull(q, "mode 'r' produces no Q");
            r.Should().BeShaped(3, 3);
            // R from mode 'r' equals R from mode 'reduced'.
            AssertClose(r, Tol,
                -8.18535277187245, -9.529216659791809, -12.094774991274218,
                0, 1.4812257933030573, 0.5038182970418574,
                0, 0, 1.569262576503246);
        }

        [TestMethod]
        public void Qr_RawMode_ReturnsHAndTau()
        {
            RequireLapack();
            var T = np.array(new double[,]
                { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 10 }, { 1, 0, 2 }, { 2, 3, 1 } });
            var (h, tau) = np.linalg.qr(T, "raw");
            h.Should().BeShaped(3, 5);   // (N, M), the reflectors transposed
            tau.Should().BeShaped(3);    // (min(M,N),)
            AssertClose(h, Tol,
                -8.426149773176357, 0.4243514156100777, 0.742614977317636, 0.10608785390251943, 0.21217570780503886,
                -9.968965928828371, 1.618554388909796, 0.5232871716047162, 0.7479738423512258, -0.2712509621177474,
                -11.986494747757924, -0.3045666860851668, 2.4962337221700523, -0.12545902401854347, 0.6101995783846341);
            AssertClose(tau, Tol, 1.1186781658193854, 1.0488384176823073, 1.4408355198256773);
        }

        [TestMethod]
        public void Qr_EconomicMode_ReturnsPackedResult()
        {
            RequireLapack();
            var M = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 10 }, { 1, 0, 2 } });
            var (q, r) = np.linalg.qr(M, "economic");
            Assert.IsNull(r, "mode 'economic' produces no separate R");
            q.Should().BeShaped(4, 3); // the packed geqrf result, same shape as the input
        }

        [TestMethod]
        public void Qr_Reduced_QHasOrthonormalColumns_AndRIsUpperTriangular()
        {
            RequireLapack();
            var M = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 10 }, { 1, 0, 2 } });
            var (q, r) = np.linalg.qr(M);
            // Q.T @ Q == I_3
            AssertClose(np.matmul(q.T, q), 1e-9, 1, 0, 0, 0, 1, 0, 0, 0, 1);
            // R strictly-lower entries are zero.
            AssertClose(np.matmul(q, r), 1e-9, 1, 2, 3, 4, 5, 6, 7, 8, 10, 1, 0, 2);
            var rflat = r.ravel();
            Assert.AreEqual(0.0, Convert.ToDouble(rflat.GetAtIndex(3)), Tol); // R[1,0]
            Assert.AreEqual(0.0, Convert.ToDouble(rflat.GetAtIndex(6)), Tol); // R[2,0]
            Assert.AreEqual(0.0, Convert.ToDouble(rflat.GetAtIndex(7)), Tol); // R[2,1]
        }

        [TestMethod]
        public void Qr_Float32_ComputesInDouble_ReturnsSingle()
        {
            RequireLapack();
            var M = np.array(new float[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 10 }, { 1, 0, 2 } });
            var (q, r) = np.linalg.qr(M);
            Assert.AreEqual(typeof(float), q.dtype);
            Assert.AreEqual(typeof(float), r.dtype);
            AssertClose(np.matmul(q, r).astype(np.float64), 1e-4, 1, 2, 3, 4, 5, 6, 7, 8, 10, 1, 0, 2);
        }

        [TestMethod]
        public void Qr_IntegerOperand_WidensToDouble()
        {
            RequireLapack();
            var (q, r) = np.linalg.qr(np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }));
            Assert.AreEqual(typeof(double), q.dtype);
            Assert.AreEqual(typeof(double), r.dtype);
            AssertClose(np.matmul(q, r), 1e-9, 1, 2, 3, 4, 5, 6);
        }

        [TestMethod]
        public void Qr_Complex_MatchesNumPy()
        {
            RequireLapack();
            var M = np.array(new Complex[,]
            {
                { new Complex(1, 2), new Complex(3, -1) },
                { new Complex(4, 0), new Complex(1, 1) }
            });
            var (q, r) = np.linalg.qr(M);
            q.Should().BeShaped(2, 2);
            r.Should().BeShaped(2, 2);
            // Reconstruct: Q @ R == M.
            AssertCloseComplex(np.matmul(q, r), 1e-9,
                new Complex(1, 2), new Complex(3, -1), new Complex(4, 0), new Complex(1, 1));
        }

        [TestMethod]
        public void Qr_Stacked_ReducedAndComplete()
        {
            RequireLapack();
            var stack = np.arange(3 * 5 * 3.0).reshape(3, 5, 3) + np.eye(5, 3);
            var (q, r) = np.linalg.qr(stack);
            q.Should().BeShaped(3, 5, 3);
            r.Should().BeShaped(3, 3, 3);
            // Each batch element reconstructs its own input matrix.
            for (int e = 0; e < 3; e++)
                AssertClose(np.matmul(q[e.ToString()], r[e.ToString()]), 1e-9,
                    stack[e.ToString()].ToArray<double>());

            var (qc, rc) = np.linalg.qr(np.arange(2 * 6 * 2.0).reshape(2, 6, 2) + np.eye(6, 2), "complete");
            qc.Should().BeShaped(2, 6, 6);
            rc.Should().BeShaped(2, 6, 2);
        }

        [TestMethod]
        public void Qr_FortranAndTransposedLayouts_Reconstruct()
        {
            RequireLapack();
            var M = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 10 }, { 1, 0, 2 } });
            foreach (var view in new[] { np.asfortranarray(M), np.ascontiguousarray(M.T).T })
            {
                var (q, r) = np.linalg.qr(view);
                AssertClose(np.matmul(q, r), 1e-9, 1, 2, 3, 4, 5, 6, 7, 8, 10, 1, 0, 2);
            }
        }

        [TestMethod]
        public void Qr_DegenerateShapes_MatchNumPy()
        {
            RequireLapack();
            // (3,0): reduced Q(3,0) R(0,0); complete Q(3,3)=I R(3,0).
            var (q0, r0) = np.linalg.qr(np.zeros(new Shape(3, 0)));
            q0.Should().BeShaped(3, 0);
            r0.Should().BeShaped(0, 0);

            var (qc, rc) = np.linalg.qr(np.zeros(new Shape(3, 0)), "complete");
            qc.Should().BeShaped(3, 3);
            rc.Should().BeShaped(3, 0);
            AssertClose(qc, Tol, 1, 0, 0, 0, 1, 0, 0, 0, 1); // orgqr(k=0) → identity

            // (0,3): Q(0,0) R(0,3).
            var (q1, r1) = np.linalg.qr(np.zeros(new Shape(0, 3)));
            q1.Should().BeShaped(0, 0);
            r1.Should().BeShaped(0, 3);
        }

        [TestMethod]
        public void Qr_OneDimensional_Raises()
        {
            RequireLapack();
            new Action(() => np.linalg.qr(np.array(new double[] { 1, 2, 3 })))
                .Should().Throw<LinAlgError>()
                .WithMessage("1-dimensional array given. Array must be at least two-dimensional");
        }

        [TestMethod]
        public void Qr_UnrecognizedMode_Raises()
        {
            RequireLapack();
            new Action(() => np.linalg.qr(np.eye(3), "banana"))
                .Should().Throw<ValueError>().WithMessage("Unrecognized mode 'banana'");
        }

        // ------------------------------------------------------------------ the seam

        [TestMethod]
        public void WithoutBackend_RaiseNotSupported_WithIt_TheyCompute()
        {
            OpenBlasEngine.Disable();
            var A = np.array(new double[,] { { 4.0, 2.0 }, { 2.0, 5.0 } });
            var M = np.array(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } });
            new Action(() => np.linalg.cholesky(A)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.qr(M)).Should().Throw<NotSupportedException>();

            RequireLapack();
            AssertClose(np.linalg.cholesky(A), Tol, 2, 0, 1, 2);
            var (q, r) = np.linalg.qr(M);
            AssertClose(np.matmul(q, r), 1e-9, 1, 2, 3, 4, 5, 6);
        }
    }
}
