using System;
using System.Numerics;
using NumSharp.Interop.OpenBLAS;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     The SVD-based <c>np.linalg</c> surface served by the OpenBLAS backend — <c>svd</c>,
    ///     <c>svdvals</c>, <c>pinv</c>, <c>matrix_rank</c>, <c>cond</c>, the spectral/nuclear matrix
    ///     norms (<c>ord</c> 2, -2, 'nuc'), and <c>lstsq</c> — plus the seam that turns them on.
    /// </summary>
    /// <remarks>
    ///     These need a LAPACK-capable library (the bundled scipy-openblas is one); they go
    ///     <see cref="Assert.Inconclusive(string)"/> on a host without one, exactly as the LU
    ///     factorisation and matmul-parity tests do. VALUES are asserted to a tolerance so the gate
    ///     holds on any machine — the BITS depend on the OpenBLAS build, thread count and dispatched
    ///     micro-kernel (SVD/lstsq drive many internal BLAS calls). U and Vh carry a per-column sign
    ///     freedom, so the singular VALUES, the RECONSTRUCTION and the derived quantities (pinv, rank,
    ///     cond, norm, lstsq) are what is asserted rather than U/Vh entry-by-entry. Verified against
    ///     NumPy 2.4.2 on the reference host.
    /// </remarks>
    [TestClass]
    [DoNotParallelize] // binds the process-wide OpenBLAS backend (TensorEngine.Blas)
    public class LapackSvdTests
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
                var z = (Complex) flat.GetAtIndex(i);
                Assert.AreEqual(expected[i].Real, z.Real, tol, $"[{i}].re");
                Assert.AreEqual(expected[i].Imaginary, z.Imaginary, tol, $"[{i}].im");
            }
        }

        /// <summary>Reduced reconstruction <c>(U * S) @ Vh</c> — must equal the original matrix.</summary>
        private static NDArray ReconstructReduced(NDArray u, NDArray s, NDArray vh)
            => np.matmul(np.multiply(u, s), vh);

        #region svd / svdvals

        [TestMethod]
        public void Svdvals_2x3_MatchesNumPy()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            AssertClose(np.linalg.svdvals(a), Tol, 9.508032000695724, 0.7728696356734843);
        }

        [TestMethod]
        public void Svd_Reduced_2x3_ShapesAndReconstructs()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var (u, s, vh) = np.linalg.svd(a, full_matrices: false);
            u.Should().BeShaped(2, 2);
            s.Should().BeShaped(2);
            vh.Should().BeShaped(2, 3);
            AssertClose(s, Tol, 9.508032000695724, 0.7728696356734843);
            AssertClose(ReconstructReduced(u, s, vh), 1e-9, 1, 2, 3, 4, 5, 6);
        }

        [TestMethod]
        public void Svd_Full_2x3_Shapes()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var (u, s, vh) = np.linalg.svd(a, full_matrices: true);
            u.Should().BeShaped(2, 2);
            s.Should().BeShaped(2);
            vh.Should().BeShaped(3, 3);
            AssertClose(s, Tol, 9.508032000695724, 0.7728696356734843);
        }

        [TestMethod]
        public void Svd_Reduced_3x2_ShapesAndReconstructs()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 1, 4 }, { 2, 5 }, { 3, 6 } }); // 3x2
            var (u, s, vh) = np.linalg.svd(a, full_matrices: false);
            u.Should().BeShaped(3, 2);
            s.Should().BeShaped(2);
            vh.Should().BeShaped(2, 2);
            AssertClose(ReconstructReduced(u, s, vh), 1e-9, 1, 4, 2, 5, 3, 6);
        }

        [TestMethod]
        public void Svd_Full_3x2_Shapes()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 1, 4 }, { 2, 5 }, { 3, 6 } });
            var (u, s, vh) = np.linalg.svd(a, full_matrices: true);
            u.Should().BeShaped(3, 3);
            s.Should().BeShaped(2);
            vh.Should().BeShaped(2, 2);
        }

        [TestMethod]
        public void Svd_NoComputeUv_ReturnsOnlyS()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var (u, s, vh) = np.linalg.svd(a, compute_uv: false);
            Assert.IsNull(u);
            Assert.IsNull(vh);
            s.Should().BeShaped(2);
            AssertClose(s, Tol, 9.508032000695724, 0.7728696356734843);
        }

        [TestMethod]
        public void Svd_Float32_ReturnsSingle_AndReconstructs()
        {
            RequireLapack();
            var a = np.array(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var (u, s, vh) = np.linalg.svd(a, full_matrices: false);
            Assert.AreEqual(typeof(float), u.dtype);
            Assert.AreEqual(typeof(float), s.dtype); // real_t = single
            Assert.AreEqual(typeof(float), vh.dtype);
            AssertClose(s, 1e-4, 9.508032000695724, 0.7728696356734843);
            AssertClose(ReconstructReduced(u, s, vh), 1e-4, 1, 2, 3, 4, 5, 6);
        }

        [TestMethod]
        public void Svd_Complex_SReal_AndReconstructs()
        {
            RequireLapack();
            var z = np.array(new Complex[,]
            {
                { new Complex(1, 1), new Complex(2, 0) },
                { new Complex(3, 0), new Complex(4, -1) }
            });
            var (u, s, vh) = np.linalg.svd(z, full_matrices: false);
            Assert.AreEqual(NPTypeCode.Complex, u.typecode);
            Assert.AreEqual(typeof(double), s.dtype); // singular values are always real
            Assert.AreEqual(NPTypeCode.Complex, vh.typecode);
            AssertCloseComplex(ReconstructReduced(u, s, vh), 1e-9,
                new Complex(1, 1), new Complex(2, 0), new Complex(3, 0), new Complex(4, -1));
        }

        [TestMethod]
        public void Svd_IntegerOperand_WidensToDouble()
        {
            RequireLapack();
            var a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var s = np.linalg.svdvals(a);
            Assert.AreEqual(typeof(double), s.dtype);
            AssertClose(s, Tol, 9.508032000695724, 0.7728696356734843);
        }

        [TestMethod]
        public void Svd_Stacked_ShapesAndPerMatrixReconstruct()
        {
            RequireLapack();
            var st = np.arange(2 * 3 * 4).astype(np.float64).reshape(2, 3, 4);
            var (u, s, vh) = np.linalg.svd(st, full_matrices: true);
            u.Should().BeShaped(2, 3, 3);
            s.Should().BeShaped(2, 3);
            vh.Should().BeShaped(2, 4, 4);

            var (ur, sr, vr) = np.linalg.svd(st, full_matrices: false);
            ur.Should().BeShaped(2, 3, 3); // M=3 <= N=4, so reduced U is (M, K) = (3, 3)
            sr.Should().BeShaped(2, 3);
            vr.Should().BeShaped(2, 3, 4);
            var recon = np.matmul(np.multiply(ur, np.expand_dims(sr, -2)), vr);
            AssertClose(recon["0"], 1e-9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
            AssertClose(recon["1"], 1e-9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23);
        }

        [TestMethod]
        public void Svd_Transposed_ReadsThroughStrides()
        {
            // A transposed (non-contiguous) operand must decompose correctly, not read raw memory.
            RequireLapack();
            var a = np.array(new double[,] { { 1, 4 }, { 2, 5 }, { 3, 6 } }).T; // (2,3) view
            AssertClose(np.linalg.svdvals(a), Tol, 9.508032000695724, 0.7728696356734843);
        }

        #endregion

        #region pinv

        [TestMethod]
        public void Pinv_Overdetermined_MatchesNumPy()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 0, 1 }, { 1, 1 }, { 2, 1 }, { 3, 1 } });
            var b = np.linalg.pinv(a);
            b.Should().BeShaped(2, 4);
            AssertClose(b, 1e-9, -0.3, -0.1, 0.1, 0.3, 0.7, 0.4, 0.1, -0.2);
        }

        [TestMethod]
        public void Pinv_Complex_MatchesNumPy()
        {
            RequireLapack();
            var z = np.array(new Complex[,]
            {
                { new Complex(1, 1), new Complex(2, 0) },
                { new Complex(3, 0), new Complex(4, -1) }
            });
            AssertCloseComplex(np.linalg.pinv(z), 1e-9,
                new Complex(-0.7, -1.1), new Complex(0.2, 0.6),
                new Complex(0.3, 0.9), new Complex(0.2, -0.4));
        }

        [TestMethod]
        public void Pinv_RoundTrips_A_Pinv_A_IsA()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 0.0, 1 }, { 1, 1 }, { 2, 1 }, { 3, 1 } });
            var recon = np.matmul(np.matmul(a, np.linalg.pinv(a)), a);
            AssertClose(recon, 1e-9, 0, 1, 1, 1, 2, 1, 3, 1);
        }

        [TestMethod]
        public void Pinv_Float32_ReturnsSingle()
        {
            RequireLapack();
            var a = np.array(new float[,] { { 0, 1 }, { 1, 1 }, { 2, 1 }, { 3, 1 } });
            var b = np.linalg.pinv(a);
            Assert.AreEqual(typeof(float), b.dtype);
            AssertClose(b, 1e-5, -0.3, -0.1, 0.1, 0.3, 0.7, 0.4, 0.1, -0.2);
        }

        #endregion

        #region matrix_rank

        [TestMethod]
        public void MatrixRank_FullRank_And_Deficient()
        {
            RequireLapack();
            Assert.AreEqual(4L, Convert.ToInt64(np.linalg.matrix_rank(np.eye(4)).GetAtIndex(0)));

            var i = np.eye(4);
            i[-1, -1] = (NDArray) 0.0;
            Assert.AreEqual(3L, Convert.ToInt64(np.linalg.matrix_rank(i).GetAtIndex(0)));
        }

        [TestMethod]
        public void MatrixRank_Stacked_OnePerMatrix()
        {
            RequireLapack();
            var i = np.eye(3);
            i[-1, -1] = (NDArray) 0.0;
            var r = np.linalg.matrix_rank(np.stack(new[] { np.eye(3), i }));
            r.Should().BeShaped(2);
            AssertClose(r, Tol, 3, 2);
        }

        [TestMethod]
        public void MatrixRank_ExplicitTol()
        {
            RequireLapack();
            var m = np.array(new double[,] { { 1.0, 2 }, { 2, 4.0001 } });
            Assert.AreEqual(1L, Convert.ToInt64(np.linalg.matrix_rank(m, tol: 0.01).GetAtIndex(0)));
        }

        [TestMethod]
        public void MatrixRank_Vector_IsPredicateNotCount()
        {
            // Below rank 2 NumPy short-circuits: matrix_rank([1,2,3]) is 1 (not 3), zeros are 0.
            RequireLapack();
            Assert.AreEqual(1L, Convert.ToInt64(np.linalg.matrix_rank(np.array(new double[] { 1, 2, 3 })).GetAtIndex(0)));
            Assert.AreEqual(0L, Convert.ToInt64(np.linalg.matrix_rank(np.zeros(new Shape(3))).GetAtIndex(0)));
        }

        #endregion

        #region cond

        [TestMethod]
        public void Cond_SingularValueOrders_MatchNumPy()
        {
            RequireLapack();
            var c = np.array(new double[,] { { 1.0, 2 }, { 3, 4 } });
            Assert.AreEqual(14.933034373659268, Convert.ToDouble(np.linalg.cond(c).GetAtIndex(0)), 1e-9);
            Assert.AreEqual(14.933034373659268, Convert.ToDouble(np.linalg.cond(c, 2).GetAtIndex(0)), 1e-9);
            Assert.AreEqual(0.06696562634074717, Convert.ToDouble(np.linalg.cond(c, -2).GetAtIndex(0)), 1e-9);
        }

        [TestMethod]
        public void Cond_CompositionOrders_MatchNumPy()
        {
            RequireLapack();
            var c = np.array(new double[,] { { 1.0, 2 }, { 3, 4 } });
            Assert.AreEqual(21.0, Convert.ToDouble(np.linalg.cond(c, 1).GetAtIndex(0)), 1e-9);
            Assert.AreEqual(6.0, Convert.ToDouble(np.linalg.cond(c, -1).GetAtIndex(0)), 1e-9);
            Assert.AreEqual(21.0, Convert.ToDouble(np.linalg.cond(c, double.PositiveInfinity).GetAtIndex(0)), 1e-9);
            Assert.AreEqual(6.0, Convert.ToDouble(np.linalg.cond(c, double.NegativeInfinity).GetAtIndex(0)), 1e-9);
            Assert.AreEqual(15.0, Convert.ToDouble(np.linalg.cond(c, "fro").GetAtIndex(0)), 1e-9);
        }

        [TestMethod]
        public void Cond_AllZeroMatrix_IsInfinite()
        {
            RequireLapack();
            Assert.AreEqual(double.PositiveInfinity, Convert.ToDouble(np.linalg.cond(np.zeros(new Shape(2, 2))).GetAtIndex(0)));
        }

        [TestMethod]
        public void Cond_EmptyMatrix_Raises()
        {
            RequireLapack();
            new Action(() => np.linalg.cond(np.zeros(new Shape(0, 0))))
                .Should().Throw<LinAlgError>().WithMessage("cond is not defined on empty arrays");
        }

        #endregion

        #region norm — the singular-value orders

        [TestMethod]
        public void Norm_SpectralAndNuclear_MatchNumPy()
        {
            RequireLapack();
            var c = np.array(new double[,] { { 1.0, 2 }, { 3, 4 } });
            Assert.AreEqual(5.464985704219043, Convert.ToDouble(np.linalg.norm(c, 2).GetAtIndex(0)), 1e-9);
            Assert.AreEqual(0.36596619062625746, Convert.ToDouble(np.linalg.norm(c, -2).GetAtIndex(0)), 1e-9);
            Assert.AreEqual(5.8309518948453, Convert.ToDouble(np.linalg.norm(c, "nuc").GetAtIndex(0)), 1e-9);
        }

        [TestMethod]
        public void Norm_Stacked_OverAxisTuple()
        {
            RequireLapack();
            var m = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            AssertClose(np.linalg.norm(m, 2, new[] { 1, 2 }), 1e-9, 22.409298163270435, 61.785896875070776);
            AssertClose(np.linalg.norm(m, "nuc", new[] { 1, 2 }), 1e-9, 24.36463849928471, 62.49508468032368);
        }

        [TestMethod]
        public void Norm_Nuclear_Keepdims()
        {
            RequireLapack();
            var m = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            np.linalg.norm(m, "nuc", new[] { 1, 2 }, keepdims: true).Should().BeShaped(2, 1, 1);
        }

        #endregion

        #region lstsq

        [TestMethod]
        public void Lstsq_Overdetermined_1D_MatchesNumPy()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 0, 1 }, { 1, 1 }, { 2, 1 }, { 3, 1 } });
            var y = np.array(new double[] { -1, 0.2, 0.9, 2.1 });
            var (x, res, rank, s) = np.linalg.lstsq(a, y);
            x.Should().BeShaped(2);
            AssertClose(x, 1e-9, 1.0, -0.95);
            res.Should().BeShaped(1);
            AssertClose(res, 1e-9, 0.049999999999999864);
            Assert.AreEqual(typeof(int), rank.dtype);
            Assert.AreEqual(2L, Convert.ToInt64(rank.GetAtIndex(0)));
            AssertClose(s, 1e-9, 4.10003044816824, 1.0907567666961073);
        }

        [TestMethod]
        public void Lstsq_Overdetermined_2D_MatchesNumPy()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 0, 1 }, { 1, 1 }, { 2, 1 }, { 3, 1 } });
            var y = np.array(new double[,] { { -1, -2 }, { 0.2, 0.4 }, { 0.9, 1.8 }, { 2.1, 4.2 } });
            var (x, res, rank, _) = np.linalg.lstsq(a, y);
            x.Should().BeShaped(2, 2);
            AssertClose(x, 1e-9, 1.0, 2.0, -0.95, -1.9);
            res.Should().BeShaped(2);
            AssertClose(res, 1e-9, 0.049999999999999864, 0.19999999999999946);
            Assert.AreEqual(2L, Convert.ToInt64(rank.GetAtIndex(0)));
        }

        [TestMethod]
        public void Lstsq_Underdetermined_ResidsEmpty()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var b = np.array(new double[] { 1, 2 });
            var (x, res, rank, s) = np.linalg.lstsq(a, b);
            x.Should().BeShaped(3);
            AssertClose(x, 1e-9, -0.05555555555555583, 0.11111111111111112, 0.277777777777778);
            res.Should().BeShaped(0); // discarded: m <= n
            Assert.AreEqual(2L, Convert.ToInt64(rank.GetAtIndex(0)));
            AssertClose(s, 1e-9, 9.508032000695724, 0.7728696356734843);
        }

        [TestMethod]
        public void Lstsq_RankDeficient_ResidsEmpty()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 1.0, 2 }, { 2, 4 }, { 3, 6 } });
            var b = np.array(new double[] { 1, 2, 3 });
            var (_, res, rank, _) = np.linalg.lstsq(a, b);
            res.Should().BeShaped(0); // discarded: rank (1) != n (2)
            Assert.AreEqual(1L, Convert.ToInt64(rank.GetAtIndex(0)));
        }

        [TestMethod]
        public void Lstsq_Float32_ReturnsSingle()
        {
            RequireLapack();
            var a = np.array(new float[,] { { 0, 1 }, { 1, 1 }, { 2, 1 }, { 3, 1 } });
            var y = np.array(new float[] { -1, 0.2f, 0.9f, 2.1f });
            var (x, res, rank, s) = np.linalg.lstsq(a, y);
            Assert.AreEqual(typeof(float), x.dtype);
            Assert.AreEqual(typeof(float), res.dtype);
            Assert.AreEqual(typeof(float), s.dtype);
            Assert.AreEqual(typeof(int), rank.dtype);
            AssertClose(x, 1e-4, 1.0, -0.95);
        }

        #endregion

        #region degenerate / empty operands

        [TestMethod]
        public void Svd_EmptyMatrix_FullMatrices_FillsIdentity()
        {
            // LAPACK leaves the factors uninitialised for a zero-sized K; NumPy (and this backend)
            // substitute an identity for whichever of U/Vh is non-empty.
            RequireLapack();
            var (u1, s1, vh1) = np.linalg.svd(np.zeros(new Shape(0, 3)), full_matrices: true);
            u1.Should().BeShaped(0, 0);
            s1.Should().BeShaped(0);
            vh1.Should().BeShaped(3, 3);
            AssertClose(vh1, Tol, 1, 0, 0, 0, 1, 0, 0, 0, 1);

            var (u2, s2, vh2) = np.linalg.svd(np.zeros(new Shape(3, 0)), full_matrices: true);
            u2.Should().BeShaped(3, 3);
            AssertClose(u2, Tol, 1, 0, 0, 0, 1, 0, 0, 0, 1);
            vh2.Should().BeShaped(0, 0);
        }

        [TestMethod]
        public void Svd_EmptyMatrix_Reduced_Shapes()
        {
            RequireLapack();
            var (u, s, vh) = np.linalg.svd(np.zeros(new Shape(0, 3)), full_matrices: false);
            u.Should().BeShaped(0, 0);
            s.Should().BeShaped(0);
            vh.Should().BeShaped(0, 3);
            np.linalg.svdvals(np.zeros(new Shape(0, 3))).Should().BeShaped(0);
        }

        [TestMethod]
        public void Pinv_Wide_MatchesNumPy()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }); // 2x3, M<N
            var b = np.linalg.pinv(a);
            b.Should().BeShaped(3, 2);
            AssertClose(b, 1e-9, -0.9444444444444444, 0.4444444444444443, -0.11111111111111108,
                0.11111111111111109, 0.7222222222222221, -0.22222222222222207);
        }

        [TestMethod]
        public void Lstsq_NrhsZero_TrimsTheAddedColumn()
        {
            // LAPACK can't take nrhs == 0, so NumPy pads b then trims x/resids — rank and s come from
            // the real coefficient matrix.
            RequireLapack();
            var (x, res, rank, s) = np.linalg.lstsq(np.ones(new Shape(3, 2)), np.zeros(new Shape(3, 0)));
            x.Should().BeShaped(2, 0);
            res.Should().BeShaped(0);
            s.Should().BeShaped(2);
            Assert.AreEqual(1L, Convert.ToInt64(rank.GetAtIndex(0)));
        }

        [TestMethod]
        public void Lstsq_MZero_ZeroSolution()
        {
            RequireLapack();
            var (x, res, rank, s) = np.linalg.lstsq(np.zeros(new Shape(0, 2)), np.zeros(new Shape(0)));
            x.Should().BeShaped(2);
            AssertClose(x, Tol, 0, 0);
            res.Should().BeShaped(0);
            s.Should().BeShaped(0);
            Assert.AreEqual(0L, Convert.ToInt64(rank.GetAtIndex(0)));
        }

        [TestMethod]
        public void Cond_Stacked_OnePerMatrix()
        {
            RequireLapack();
            var c = np.stack(new[]
            {
                np.array(new double[,] { { 1.0, 2 }, { 3, 4 } }),
                np.array(new double[,] { { 2.0, 0 }, { 0, 1 } })
            });
            AssertClose(np.linalg.cond(c), 1e-9, 14.933034373659268, 2.0);
            AssertClose(np.linalg.cond(c, -2), 1e-9, 0.06696562634074717, 0.5);
        }

        [TestMethod]
        public void MatrixRank_ExplicitRtol()
        {
            RequireLapack();
            var m = np.array(new double[,] { { 1.0, 2 }, { 2, 4.0001 } });
            Assert.AreEqual(1L, Convert.ToInt64(np.linalg.matrix_rank(m, rtol: 0.01).GetAtIndex(0)));
        }

        #endregion

        #region hermitian=True

        /// <summary>
        ///     NumPy's <c>svd(a, hermitian=True)</c> is eigh's decomposition of the matrix whose lower triangle <c>a</c>
        ///     holds: S the absolute eigenvalues descending, U the eigenvectors in that order, Vh the conjugate transpose of U
        ///     with each row multiplied by its eigenvalue's sign. NumSharp ran the ordinary SVD of the whole matrix, so a
        ///     non-symmetric <c>a</c> gave the singular values of another matrix. Expected values: NumPy 2.4.2.
        /// </summary>
        [TestMethod]
        public void Svd_Hermitian_IsEighOfTheLowerTriangle()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 2.0, 7.0, 1.0 }, { -1.0, 3.0, 5.0 }, { 4.0, 0.5, -2.0 } });
            // The matrix NumPy decomposes: a's lower triangle mirrored (eigenvalues -4.59..., 2.90..., 4.69...).
            var mirrored = np.array(new double[,] { { 2.0, -1.0, 4.0 }, { -1.0, 3.0, 0.5 }, { 4.0, 0.5, -2.0 } });

            var (u, s, vh) = np.linalg.svd(a, hermitian: true);
            AssertClose(s, 1e-12, 4.689342475693309, 4.591878741146129, 2.9025362654528197);
            np.allclose(ReconstructReduced(u, s, vh), mirrored).Should().BeTrue();
            // Vh = U.T with the row of the negative eigenvalue (the second singular value) negated.
            np.allclose(vh, np.multiply(np.swapaxes(u, -1, -2), np.array(new[,] { { 1.0 }, { -1.0 }, { 1.0 } })))
                .Should().BeTrue();
            // A real operand's Vh is a transposed VIEW of that product (ndarray.conjugate returns a real array itself).
            vh.flags.f_contiguous.Should().BeTrue();
            vh.flags.c_contiguous.Should().BeFalse();
            vh.flags.owndata.Should().BeFalse();
            u.flags.c_contiguous.Should().BeTrue();

            // full_matrices is ignored on this route (a Hermitian matrix is square).
            var (_, sReduced, _) = np.linalg.svd(a, full_matrices: false, hermitian: true);
            AssertClose(sReduced, 1e-12, 4.689342475693309, 4.591878741146129, 2.9025362654528197);

            // compute_uv=False: sort(abs(eigvalsh(a)))[..., ::-1], a reversed view of the sorted copy.
            var (_, values, _) = np.linalg.svd(a, compute_uv: false, hermitian: true);
            AssertClose(values, 1e-12, 4.689342475693309, 4.591878741146129, 2.9025362654528197);
            values.strides.Should().Equal(-8L);
            values.flags.owndata.Should().BeFalse();

            // hermitian=False decomposes all of a.
            var (_, full, _) = np.linalg.svd(a, compute_uv: false);
            AssertClose(full, 1e-12, 8.386121997487018, 5.9292969929916435, 1.9407202302662188);
        }

        /// <summary>
        ///     A zero eigenvalue's sign is 0, so NumPy's Vh has a zero row there (Vh is not unitary then). A complex operand's
        ///     Vh is a new array in the order of the transposed view the conjugate ufunc reads (F for one matrix, the swapped
        ///     axes for a stack); a float32 operand gives float32 factors. Expected values: NumPy 2.4.2.
        /// </summary>
        [TestMethod]
        public void Svd_Hermitian_ZeroEigenvalue_Complex_Float32()
        {
            RequireLapack();
            var z = np.array(new double[,] { { 1.0, 2.0 }, { 2.0, 4.0 } });
            var (_, sz, vz) = np.linalg.svd(z, hermitian: true);
            AssertClose(sz, 1e-12, 5.0, 0.0);
            vz.GetDouble(1, 0).Should().Be(0.0);
            vz.GetDouble(1, 1).Should().Be(0.0);

            var c = np.array(new[,] { { new Complex(2, 0), new Complex(1, -2) }, { new Complex(1, 2), new Complex(-3, 0) } });
            var (uc, sc, vc) = np.linalg.svd(c, hermitian: true);
            uc.dtype.Should().Be(np.complex128);
            sc.dtype.Should().Be(np.float64);
            AssertClose(sc, 1e-12, 3.8541019662496847, 2.8541019662496847);
            vc.flags.owndata.Should().BeTrue();
            vc.flags.f_contiguous.Should().BeTrue();

            var stack = np.stack(new[] { c, np.conjugate(c) });
            var (_, _, vStack) = np.linalg.svd(stack, hermitian: true);
            // NumPy's (2, 2, 2) complex Vh: byte strides (64, 16, 32), the stack's matrices each in F order.
            vStack.strides.Should().Equal(64L, 16L, 32L);
            vStack.flags.owndata.Should().BeTrue();

            var f = np.array(new float[,] { { 2.0f, 1.0f }, { 1.0f, -3.0f } });
            var (uf, sf, vf) = np.linalg.svd(f, hermitian: true);
            uf.dtype.Should().Be(np.float32);
            sf.dtype.Should().Be(np.float32);
            vf.dtype.Should().Be(np.float32);
            AssertClose(sf, 1e-6, 3.192582368850708, 2.192582368850708);
        }

        /// <summary>
        ///     NumPy's <c>pinv(a, hermitian=True)</c> and <c>matrix_rank(A, hermitian=True)</c> take their singular values
        ///     from the hermitian SVD, so they read the lower triangle only: <c>[[0, 5], [0, 0]]</c> is the zero matrix
        ///     there (rank 0, a zero pseudo-inverse) where the ordinary route gives rank 1. Expected values: NumPy 2.4.2.
        /// </summary>
        [TestMethod]
        public void Pinv_MatrixRank_Hermitian_ReadTheLowerTriangle()
        {
            RequireLapack();
            var m = np.array(new double[,] { { 0.0, 5.0 }, { 0.0, 0.0 } });
            Assert.AreEqual(0L, Convert.ToInt64(np.linalg.matrix_rank(m, hermitian: true).GetAtIndex(0)));
            Assert.AreEqual(1L, Convert.ToInt64(np.linalg.matrix_rank(m).GetAtIndex(0)));
            AssertClose(np.linalg.pinv(m, hermitian: true), 0.0, 0, 0, 0, 0);
            AssertClose(np.linalg.pinv(m), 1e-15, 0, 0, 0.2, 0);

            var a = np.array(new double[,] { { 2.0, 7.0, 1.0 }, { -1.0, 3.0, 5.0 }, { 4.0, 0.5, -2.0 } });
            AssertClose(np.linalg.pinv(a, hermitian: true), 1e-12, 0.1, 0, 0.2, 0, 0.32, 0.08, 0.2, 0.08, -0.08);
        }

        #endregion

        #region the seam

        [TestMethod]
        public void WithoutTheBackend_SvdFamily_RaisesNotSupported()
        {
            OpenBlasEngine.Disable();
            var a = np.array(new double[,] { { 1.0, 2, 3 }, { 4, 5, 6 } });
            new Action(() => np.linalg.svd(a)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.svdvals(a)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.pinv(a)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.matrix_rank(a)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.cond(a, 2)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.norm(a, "nuc")).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.lstsq(a, np.zeros(new Shape(2))))
                .Should().Throw<NotSupportedException>();
        }

        #endregion
    }
}
