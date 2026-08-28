using System;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Interop.OpenBLAS;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.LinearAlgebra
{
    /// <summary>
    ///     The pure-managed LU family — <c>det</c>, <c>slogdet</c>, <c>solve</c>, <c>inv</c> and the
    ///     functions that compose on them — served by <see cref="ManagedLu"/> with NO backend
    ///     installed. This is the fallback a plain <c>NumSharp.Core</c> gets, so these tests pin that
    ///     it COMPUTES (rather than raising <see cref="OpenBlasMissingBackendException"/>) and that the
    ///     numbers, shapes, dtypes and error taxonomy match NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     VALUES are asserted to a tolerance, not bit-for-bit: NumPy runs this family through LAPACK's
    ///     BLOCKED <c>getrf</c>, whose Schur-complement accumulation NumSharp's unblocked LU cannot
    ///     reproduce (the same reason the matrix products need the bundled binary for byte-parity).
    ///     The results are allclose — bit-exact for tiny matrices, a few ULP apart as they grow — and
    ///     the one boundary case where the two factorisations genuinely disagree (an exactly-singular
    ///     matrix that cancels to a true zero pivot in blocked LAPACK but not in the unblocked kernel)
    ///     is pinned as a documented divergence in the last region.
    ///     <para>
    ///     The reference values were produced by running NumPy 2.4.2 (see the probes in the
    ///     implementing change). <c>[TestInitialize]</c> forces the backend off so the managed path is
    ///     the one under test even if a sibling suite left a backend enabled.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class ManagedLuTests
    {
        private const double Tol = 1e-9;

        [TestInitialize]
        public void ForceManaged()
        {
            OpenBlasEngine.Disable();
            np.arange(4.0).reshape(2, 2).TensorEngine.Blas.Should().BeNull(
                "these tests pin the managed LU fallback, which only runs when no backend is installed");
        }

        // ---------------------------------------------------------------------------- det

        [TestMethod]
        public void Det_TwoByTwo_MatchesNumPy()
            => AssertClose(np.linalg.det(np.array(new[,] { { 1.0, 2 }, { 3, 4 } })), Tol, -2.0000000000000004);

        [TestMethod]
        public void Det_OneByOne_GoesThroughTheLogExpFold_NotTheProductOfPivots()
        {
            // det([[5.]]) is 4.999999999999999, NOT 5.0 — because NumPy (and this kernel) compute
            // det = sign·exp(Σ log|Uᵢᵢ|), not the product of the pivots. This is the discriminator that
            // the fold is right; assert it EXACTLY.
            double v = Convert.ToDouble(np.linalg.det(np.array(new[,] { { 5.0 } })).ravel().GetAtIndex(0));
            Assert.AreEqual(4.999999999999999, v);
        }

        [TestMethod]
        public void Det_EmptyMatrix_IsTheEmptyProductOne()
            => AssertClose(np.linalg.det(np.zeros(new Shape(0, 0))), Tol, 1.0);

        [TestMethod]
        public void Det_Stack_IsPerElement()
            => AssertClose(np.linalg.det(np.arange(8.0).reshape(2, 2, 2)), Tol, -2.0, -2.0);

        [TestMethod]
        public void Det_IntegerWidensToFloat64()
        {
            var d = np.linalg.det(np.array(new[,] { { 1, 2 }, { 3, 4 } }));
            Assert.AreEqual(typeof(double), d.dtype);
            AssertClose(d, Tol, -2.0);
        }

        [TestMethod]
        public void Det_Float32_ComputesInDouble_ReturnsSingle()
        {
            var d = np.linalg.det(np.array(new[,] { { 1, 2 }, { 3, 4 } }).astype(np.float32));
            Assert.AreEqual(typeof(float), d.dtype);
            AssertClose(d, 1e-5, -2.0);
        }

        [TestMethod]
        public void Det_Complex_MatchesNumPy()
            => AssertCloseComplex(
                np.linalg.det(np.array(new Complex[,] { { new(1, 2), new(3, -1) }, { new(0, 1), new(4, 0) } })),
                Tol, new Complex(3, 5));

        [TestMethod]
        public void Det_NonContiguous_TransposeAndStrided_ReadThroughStrides()
        {
            // A transpose and a stepped slice are read through their own strides (linearised), so both
            // give the same answer as the contiguous matrix would.
            AssertClose(np.linalg.det(np.array(new[,] { { 1.0, 2 }, { 3, 4 } }).T), Tol, -2.0);
            AssertClose(np.linalg.det(np.arange(16.0).reshape(4, 4)["::2, ::2"]), Tol, -15.999999999999998);
        }

        [TestMethod]
        public void Det_NonFinite_PropagatesRatherThanFalselyReportingSingular()
        {
            // A NaN on the diagonal is not a zero pivot: det is NaN, det of an infinite matrix is inf.
            AssertClose(np.linalg.det(np.array(new[,] { { double.NaN, 1 }, { 2, 3 } })), Tol, double.NaN);
            AssertClose(np.linalg.det(np.array(new[,] { { double.PositiveInfinity, 0 }, { 0, 1 } })), Tol,
                double.PositiveInfinity);
        }

        [TestMethod]
        public void Det_ExactlySingular_IsZero_WithoutRaising()
            => AssertClose(np.linalg.det(np.array(new[,] { { 1.0, 2 }, { 2, 4 } })), Tol, 0.0);

        [TestMethod]
        public void Det_ZeroBatch_IsEmpty_And_EmptyMatricesInAStackAreOne()
        {
            np.linalg.det(np.zeros(new Shape(0, 2, 2))).Should().BeShaped(0);
            AssertClose(np.linalg.det(np.zeros(new Shape(2, 0, 0))), Tol, 1.0, 1.0); // each 0×0 → empty product
        }

        // ------------------------------------------------------------------------ slogdet

        [TestMethod]
        public void Slogdet_TwoByTwo_MatchesNumPy()
        {
            var (sign, log) = np.linalg.slogdet(np.array(new[,] { { 1.0, 2 }, { 3, 4 } }));
            AssertClose(sign, Tol, -1.0);
            AssertClose(log, Tol, 0.6931471805599455);
        }

        [TestMethod]
        public void Slogdet_Singular_IsZeroAndNegInf_WithoutRaising()
        {
            var (sign, log) = np.linalg.slogdet(np.array(new[,] { { 1.0, 2 }, { 2, 4 } }));
            AssertClose(sign, Tol, 0.0);
            AssertClose(log, Tol, double.NegativeInfinity);
        }

        [TestMethod]
        public void Slogdet_Empty_IsOneAndZero()
        {
            var (sign, log) = np.linalg.slogdet(np.zeros(new Shape(0, 0)));
            AssertClose(sign, Tol, 1.0);
            AssertClose(log, Tol, 0.0);
        }

        [TestMethod]
        public void Slogdet_Complex_SignIsUnitModulus_LogabsdetIsReal()
        {
            var (sign, log) =
                np.linalg.slogdet(np.array(new Complex[,] { { new(1, 2), new(3, -1) }, { new(0, 1), new(4, 0) } }));
            Assert.AreEqual(NPTypeCode.Complex, sign.typecode);
            Assert.AreEqual(typeof(double), log.dtype);
            AssertCloseComplex(sign, Tol, new Complex(0.5144957554275266, 0.8574929257125442));
            AssertClose(log, Tol, 1.7631802623080806);
        }

        // -------------------------------------------------------------------------- solve

        [TestMethod]
        public void Solve_VectorRhs_MatchesNumPy()
            => AssertClose(np.linalg.solve(np.array(new[,] { { 3.0, 2 }, { 1, 2 } }), np.array(new[] { 2.0, 0 })),
                Tol, 1.0, -0.5);

        [TestMethod]
        public void Solve_MatrixRhs_MatchesNumPy()
            => AssertClose(
                np.linalg.solve(np.array(new[,] { { 3.0, 2 }, { 1, 2 } }), np.array(new[,] { { 2.0, 1 }, { 0, 3 } })),
                Tol, 1.0, -1.0, -0.5, 2.0);

        [TestMethod]
        public void Solve_Batched_BroadcastsTheVectorAcrossTheStack()
        {
            var a = np.array(new[,] { { 3.0, 2 }, { 1, 2 } });
            var stack = np.stack(new[] { a, a * 2.0 });
            var r = np.linalg.solve(stack, np.array(new[] { 2.0, 0 }));
            r.Should().BeShaped(2, 2);
            AssertClose(r, Tol, 1.0, -0.5, 0.5, -0.25);
        }

        [TestMethod]
        public void Solve_Float32_ComputesInDouble_ReturnsSingle()
        {
            var r = np.linalg.solve(np.array(new float[,] { { 3, 2 }, { 1, 2 } }), np.array(new float[] { 2, 0 }));
            Assert.AreEqual(typeof(float), r.dtype);
            AssertClose(r, 1e-6, 1.0, -0.5);
        }

        [TestMethod]
        public void Solve_Complex_MatchesNumPy()
        {
            // np.linalg.solve([[2,1],[1,2]], [1j, 0]) = [0.6667j, -0.3333j]
            var r = np.linalg.solve(np.array(new Complex[,] { { 2, 1 }, { 1, 2 } }),
                np.array(new Complex[] { new(0, 1), 0 }));
            AssertCloseComplex(r, Tol, new Complex(0, 2.0 / 3), new Complex(0, -1.0 / 3));
        }

        [TestMethod]
        public void Solve_ReconstructsTheRightHandSide()
        {
            var a = np.array(new[,] { { 4.0, 3, 2 }, { 1, 5, 1 }, { 2, 1, 6 } });
            var b = np.array(new[] { 1.0, 2, 3 });
            var recon = np.matmul(a, np.linalg.solve(a, b));
            AssertClose(recon, 1e-9, 1.0, 2.0, 3.0);
        }

        [TestMethod]
        public void Solve_ExactlySingular_RaisesLinAlgError()
        {
            var s = np.array(new[,] { { 1.0, 2 }, { 2, 4 } });
            new Action(() => np.linalg.solve(s, np.array(new[] { 1.0, 2 })))
                .Should().Throw<LinAlgError>().WithMessage("Singular matrix");
        }

        // ---------------------------------------------------------------------------- inv

        [TestMethod]
        public void Inv_TwoByTwo_MatchesNumPy()
            => AssertClose(np.linalg.inv(np.array(new[,] { { 4.0, 7 }, { 2, 6 } })), Tol, 0.6, -0.7, -0.2, 0.4);

        [TestMethod]
        public void Inv_ReconstructsTheIdentity()
        {
            var m = np.array(new[,] { { 4.0, 3, 2 }, { 1, 5, 1 }, { 2, 1, 6 } });
            AssertClose(np.matmul(m, np.linalg.inv(m)), 1e-9, 1, 0, 0, 0, 1, 0, 0, 0, 1);
        }

        [TestMethod]
        public void Inv_Batch_IsPerElement()
        {
            var r = np.linalg.inv(np.stack(new[] { np.array(new[,] { { 4.0, 7 }, { 2, 6 } }), np.eye(2) * 2.0 }));
            r.Should().BeShaped(2, 2, 2);
            AssertClose(r, Tol, 0.6, -0.7, -0.2, 0.4, 0.5, 0, 0, 0.5);
        }

        [TestMethod]
        public void Inv_Complex_MatchesNumPy()
            => AssertCloseComplex(np.linalg.inv(np.array(new Complex[,] { { new(1, 1), 2 }, { 3, 4 } })), Tol,
                new Complex(-0.4, -0.8), new Complex(0.2, 0.4), new Complex(0.3, 0.6), new Complex(0.1, -0.3));

        [TestMethod]
        public void Inv_IntegerWidensToFloat64_Float32StaysSingle()
        {
            Assert.AreEqual(typeof(double), np.linalg.inv(np.array(new[,] { { 4, 7 }, { 2, 6 } })).dtype);
            Assert.AreEqual(typeof(float), np.linalg.inv(np.array(new float[,] { { 4, 7 }, { 2, 6 } })).dtype);
        }

        [TestMethod]
        public void Inv_ExactlySingular_RaisesLinAlgError()
            => new Action(() => np.linalg.inv(np.array(new[,] { { 1.0, 2 }, { 2, 4 } })))
                .Should().Throw<LinAlgError>().WithMessage("Singular matrix");

        [TestMethod]
        public void Inv_Empty_KeepsTheShape()
        {
            np.linalg.inv(np.zeros(new Shape(0, 0))).Should().BeShaped(0, 0);
            np.linalg.inv(np.zeros(new Shape(0, 2, 2))).Should().BeShaped(0, 2, 2);
            np.linalg.inv(np.zeros(new Shape(2, 0, 0))).Should().BeShaped(2, 0, 0);
        }

        [TestMethod]
        public void Inv_NonContiguous_MatchesTheContiguousInverse()
        {
            var m = np.array(new[,] { { 4.0, 7 }, { 2, 6 } });
            AssertClose(np.linalg.inv(m.T), Tol, 0.6, -0.2, -0.7, 0.4); // inv(mᵀ) = inv(m)ᵀ
        }

        // ------------------------------------------------------ blocked path (large matrices)

        [TestMethod]
        public void Blocked_LargeMatrix_FactorsCorrectly()
        {
            // n > 256 routes the double factorisation through the BLOCKED path, whose Schur update
            // rides SimdMatMul. A diagonal-dominant matrix keeps it well-conditioned; reconstruction
            // proves the whole blocked pipeline (panel factor + pivot application + TRSM + GEMM) end
            // to end, and slogdet exercises the blocked factor's diagonal fold.
            const int n = 300;
            var a = np.zeros(new Shape(n, n), NPTypeCode.Double);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    a.SetData((double)(((long)i * 131 + (long)j * 67) % 997), i, j);
            for (int i = 0; i < n; i++)
                a.SetData(a.GetDouble(i, i) + 100000.0, i, i);

            var recon = np.matmul(a, np.linalg.inv(a));
            double maxI = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    maxI = System.Math.Max(maxI, System.Math.Abs(recon.GetDouble(i, j) - (i == j ? 1.0 : 0.0)));
            Assert.IsTrue(maxI < 1e-9, $"A @ inv(A) should be I; max error = {maxI:E3}");

            var b = np.arange((double)n);
            var ax = np.matmul(a, np.linalg.solve(a, b));
            double maxB = 0;
            for (int i = 0; i < n; i++)
                maxB = System.Math.Max(maxB, System.Math.Abs(ax.GetDouble(i) - b.GetDouble(i)));
            Assert.IsTrue(maxB < 1e-7, $"A @ solve(A, b) should be b; max residual = {maxB:E3}");

            var (sign, log) = np.linalg.slogdet(a);
            AssertClose(sign, Tol, 1.0);
            Assert.IsTrue(double.IsFinite(Convert.ToDouble(log.ravel().GetAtIndex(0))), "log|det| should be finite");
        }

        // ---------------------------------------------------------------- composing functions

        [TestMethod]
        public void MatrixPower_NegativeExponent_ComputesThroughTheManagedInverse()
        {
            // matrix_power(a, -1) = inv(a); inv([[0,1],[2,3]]) = [[-1.5,0.5],[1,0]].
            AssertClose(np.linalg.matrix_power(np.arange(4.0).reshape(2, 2), -1), Tol, -1.5, 0.5, 1, 0);
            Assert.AreEqual(typeof(double),
                np.linalg.matrix_power(np.array(new[,] { { 2, 1, 0 }, { 0, 2, 1 }, { 1, 0, 2 } }), -1).dtype);
        }

        [TestMethod]
        public void Tensorinv_ComposesOverTheManagedInverse()
            => np.linalg.tensorinv(np.eye(24).reshape(24, 8, 3), 1).Should().BeShaped(8, 3, 24);

        [TestMethod]
        public void Tensorsolve_ComposesOverTheManagedSolve()
        {
            var a = np.eye(24).reshape(4, 6, 8, 3);
            var b = np.ones(new Shape(4, 6));
            np.linalg.tensorsolve(a, b).Should().BeShaped(8, 3);
        }

        // ------------------------------------------------------ documented divergences

        [TestMethod]
        [Misaligned]
        public void ExactlySingularButNotZeroPivot_DivergesFromNumPysBlockedLapack()
        {
            // The textbook singular [[1,2,3],[4,5,6],[7,8,9]] is where blocked and unblocked LU part
            // ways: NumPy's LAPACK cancels the last pivot to an EXACT zero, so it reports the matrix
            // singular — det 0, and inv/solve raise LinAlgError. NumSharp's unblocked kernel lands on
            // ~6.66e-16 instead (allclose to zero, but nonzero), so det is ~0 all the same, while
            // inv/solve do NOT raise — they return the large-but-finite inverse of a matrix that, to
            // this factorisation, is merely near-singular. This is inherent to unblocked-vs-blocked LU
            // and is the accepted cost of a managed factorisation with no LAPACK to reproduce.
            var a = np.arange(1.0, 10).reshape(3, 3);

            AssertClose(np.linalg.det(a), 1e-12, 0.0); // ~6.66e-16, allclose to NumPy's exact 0

            new Action(() => np.linalg.inv(a)).Should().NotThrow(
                "the unblocked kernel sees a tiny-but-nonzero pivot where NumPy sees an exact zero");
            new Action(() => np.linalg.solve(a, np.ones(new Shape(3)))).Should().NotThrow();
        }

        // ------------------------------------------------------------------------- helpers

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
    }
}
