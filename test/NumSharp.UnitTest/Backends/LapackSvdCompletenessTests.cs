using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.UnitTest.Backends
{
    /// <summary>
    ///     Completeness/parity audit for the SVD-based <c>np.linalg</c> surface (<c>svd</c>,
    ///     <c>svdvals</c>, <c>pinv</c>, <c>matrix_rank</c>, <c>cond</c>, the singular-value matrix
    ///     norms, <c>lstsq</c>) — the DOD variation matrix: every dtype, every memory layout, the
    ///     NaN/inf/singular edges, the error taxonomy, and metamorphic invariants over random data.
    /// </summary>
    /// <remarks>
    ///     Like <see cref="LapackSvdTests"/> and <see cref="LapackFactorisationTests"/> these need a
    ///     LAPACK-capable library and go <see cref="Assert.Inconclusive(string)"/> without one. The
    ///     value-anchored cases were computed with NumPy 2.4.2; the metamorphic cases assert
    ///     mathematical invariants (reconstruction, the Moore–Penrose conditions, the least-squares
    ///     normal equations, storage-independence) that hold on ANY input and so need no oracle.
    /// </remarks>
    [TestClass]
    public class LapackSvdCompletenessTests
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

        /// <summary>Element-wise closeness of two arrays (real or complex) — the metamorphic comparator.</summary>
        private static void AssertAllClose(NDArray actual, NDArray expected, double tol)
        {
            Assert.AreEqual(expected.size, actual.size, "size mismatch");
            var fa = actual.ravel();
            var fe = expected.ravel();
            bool complex = actual.typecode == NPTypeCode.Complex || expected.typecode == NPTypeCode.Complex;
            for (int i = 0; i < expected.size; i++)
            {
                if (complex)
                {
                    var za = (Complex) fa.GetAtIndex(i);
                    var ze = (Complex) fe.GetAtIndex(i);
                    Assert.AreEqual(ze.Real, za.Real, tol, $"[{i}].re");
                    Assert.AreEqual(ze.Imaginary, za.Imaginary, tol, $"[{i}].im");
                }
                else
                {
                    Assert.AreEqual(Convert.ToDouble(fe.GetAtIndex(i)), Convert.ToDouble(fa.GetAtIndex(i)), tol, $"[{i}]");
                }
            }
        }

        private static void AssertDescendingNonNegative(NDArray s)
        {
            var f = s.ravel();
            double prev = double.PositiveInfinity;
            for (int i = 0; i < s.size; i++)
            {
                double v = Convert.ToDouble(f.GetAtIndex(i));
                Assert.IsTrue(v >= -1e-12, $"singular value [{i}] = {v} is negative");
                Assert.IsTrue(v <= prev + 1e-9, $"singular values not descending at [{i}]: {v} > {prev}");
                prev = v;
            }
        }

        #region dtype coverage — all 15

        [TestMethod]
        public void Svdvals_EveryWideningDtype_ComputesInFloat64()
        {
            // bool + every integer width widen to float64 (NumPy's _commonType), and a 0/1 matrix is
            // representable in ALL of them, so one anchor covers the sweep.
            RequireLapack();
            var baseArr = np.array(new double[,] { { 1, 1 }, { 0, 1 } });
            foreach (var tc in new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64
            })
            {
                var s = np.linalg.svdvals(baseArr.astype(tc));
                Assert.AreEqual(typeof(double), s.dtype, $"{tc} must widen to float64");
                AssertClose(s, Tol, 1.618033988749895, 0.6180339887498948);
            }
        }

        [TestMethod]
        public void Svdvals_FloatDtypes_KeepTheirWidth_ComplexIsRealDouble()
        {
            RequireLapack();
            var baseArr = np.array(new double[,] { { 1, 1 }, { 0, 1 } });

            var ss = np.linalg.svdvals(baseArr.astype(NPTypeCode.Single));
            Assert.AreEqual(typeof(float), ss.dtype);
            AssertClose(ss, 1e-5, 1.618033988749895, 0.6180339887498948);

            var sd = np.linalg.svdvals(baseArr.astype(NPTypeCode.Double));
            Assert.AreEqual(typeof(double), sd.dtype);
            AssertClose(sd, Tol, 1.618033988749895, 0.6180339887498948);

            var sc = np.linalg.svdvals(baseArr.astype(NPTypeCode.Complex));
            Assert.AreEqual(typeof(double), sc.dtype); // singular values are always real
            AssertClose(sc, Tol, 1.618033988749895, 0.6180339887498948);
        }

        [TestMethod]
        public void Linalg_UnsupportedDtypes_RaiseTypeError()
        {
            // Half/Decimal/Char have no NumPy dtype; _commonType rejects them ("unsupported in linalg").
            RequireLapack();
            var baseArr = np.array(new double[,] { { 1, 1 }, { 0, 1 } });
            new Action(() => np.linalg.svdvals(baseArr.astype(NPTypeCode.Half)))
                .Should().Throw<TypeError>().WithMessage("array type float16 is unsupported in linalg");
            new Action(() => np.linalg.svd(baseArr.astype(NPTypeCode.Decimal)))
                .Should().Throw<TypeError>().WithMessage("array type decimal is unsupported in linalg");
            new Action(() => np.linalg.pinv(baseArr.astype(NPTypeCode.Char)))
                .Should().Throw<TypeError>().WithMessage("array type char is unsupported in linalg");
        }

        #endregion

        #region memory layouts — svdvals is storage-independent

        [TestMethod]
        public void Svdvals_SameAcross_Contiguous_Fortran_Strided_Transposed_NegativeStride()
        {
            RequireLapack();
            var expected = new[] { 17.412505166808597, 0.8751613501104364, 0.19686652111742997 };
            var c = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 10 } });
            AssertClose(np.linalg.svdvals(c), Tol, expected);                              // C-contiguous
            AssertClose(np.linalg.svdvals(np.asfortranarray(c)), Tol, expected);           // F-contiguous
            AssertClose(np.linalg.svdvals(c.T), Tol, expected);                            // transposed (same S)
            AssertClose(np.linalg.svdvals(c["::-1"]), Tol, expected);                      // negative-stride rows

            // Strided view: the SAME logical matrix embedded in a padded buffer, read every 2nd column.
            var padded = np.array(new double[,]
            {
                { 1, 99, 2, 99, 3, 99 }, { 4, 99, 5, 99, 6, 99 }, { 7, 99, 8, 99, 10, 99 }
            })[":, ::2"];
            AssertClose(np.linalg.svdvals(padded), Tol, expected);
        }

        [TestMethod]
        public void Svd_BroadcastBatch_ReadsThroughZeroStride()
        {
            // A broadcast leading axis (stride 0) must read the SAME matrix for every batch element.
            RequireLapack();
            var c = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 10 } });
            var batch = np.broadcast_to(c, new Shape(2, 3, 3));
            var s = np.linalg.svdvals(batch);
            s.Should().BeShaped(2, 3);
            AssertClose(s["0"], Tol, 17.412505166808597, 0.8751613501104364, 0.19686652111742997);
            AssertClose(s["1"], Tol, 17.412505166808597, 0.8751613501104364, 0.19686652111742997);
        }

        #endregion

        #region metamorphic invariants over random data (no oracle needed)

        [TestMethod]
        public void Svd_ReducedReconstruction_Random_AllShapes()
        {
            RequireLapack();
            np.random.seed(12345);
            foreach (var (m, n) in new[] { (5, 3), (3, 5), (4, 4), (6, 2), (2, 6) })
            {
                var a = np.random.randn(m, n);
                var (u, s, vh) = np.linalg.svd(a, full_matrices: false);
                u.Should().BeShaped(m, Math.Min(m, n));
                vh.Should().BeShaped(Math.Min(m, n), n);
                AssertAllClose(np.matmul(np.multiply(u, s), vh), a, 1e-9); // (U*S) @ Vh == a
                AssertDescendingNonNegative(s);
            }
        }

        [TestMethod]
        public void Svd_ComplexReconstruction_Random()
        {
            RequireLapack();
            np.random.seed(24680);
            var re = np.random.randn(4, 3);
            var im = np.random.randn(4, 3);
            var a = np.add(re, np.multiply(im, NDArray.Scalar(Complex.ImaginaryOne)));
            var (u, s, vh) = np.linalg.svd(a, full_matrices: false);
            Assert.AreEqual(NPTypeCode.Complex, u.typecode);
            Assert.AreEqual(typeof(double), s.dtype);
            AssertAllClose(np.matmul(np.multiply(u, s), vh), a, 1e-9);
        }

        [TestMethod]
        public void Svd_FullMatrices_ProducesUnitaryFactors_Random()
        {
            // full_matrices fills U/Vh out to square with the null-space basis; verify those extra
            // columns/rows are genuinely orthonormal (the delinearization of the full factors is right).
            RequireLapack();
            np.random.seed(2024);
            foreach (var (m, n) in new[] { (5, 3), (3, 5) })
            {
                var a = np.random.randn(m, n);
                var (u, s, vh) = np.linalg.svd(a, full_matrices: true);
                u.Should().BeShaped(m, m);
                vh.Should().BeShaped(n, n);
                AssertAllClose(np.matmul(u.T, u), np.eye(m), 1e-9);   // Uᵀ U = Iₘ
                AssertAllClose(np.matmul(vh, vh.T), np.eye(n), 1e-9); // Vh Vhᵀ = Iₙ
            }
        }

        [TestMethod]
        public void Pinv_Stacked_SatisfiesMoorePenrose_Random()
        {
            RequireLapack();
            np.random.seed(4040);
            var a = np.random.randn(3, 4, 2); // a stack of three 4x2 matrices
            var p = np.linalg.pinv(a);
            p.Should().BeShaped(3, 2, 4);
            AssertAllClose(np.matmul(np.matmul(a, p), a), a, 1e-8);
        }

        [TestMethod]
        public void Pinv_SatisfiesMoorePenrose_Random()
        {
            RequireLapack();
            np.random.seed(999);
            foreach (var (m, n) in new[] { (5, 3), (3, 5), (4, 4) })
            {
                var a = np.random.randn(m, n);
                var p = np.linalg.pinv(a);
                p.Should().BeShaped(n, m);
                AssertAllClose(np.matmul(np.matmul(a, p), a), a, 1e-8); // A P A == A
                AssertAllClose(np.matmul(np.matmul(p, a), p), p, 1e-8); // P A P == P
            }
        }

        [TestMethod]
        public void Lstsq_SatisfiesNormalEquations_Random()
        {
            RequireLapack();
            np.random.seed(7);
            var a = np.random.randn(6, 3); // over-determined, full rank
            var b = np.random.randn(6);
            var (x, res, rank, s) = np.linalg.lstsq(a, b);
            x.Should().BeShaped(3);
            Assert.AreEqual(3L, Convert.ToInt64(rank.GetAtIndex(0)));

            // A^T (A x - b) ≈ 0 (the least-squares optimality condition)
            var resid = np.subtract(np.matmul(a, x), b);
            var normalEq = np.matmul(a.T, resid);
            for (int i = 0; i < 3; i++)
                Assert.AreEqual(0.0, Convert.ToDouble(normalEq.GetAtIndex(i)), 1e-9, $"normalEq[{i}]");

            // residual sum equals ||A x - b||^2
            AssertClose(res, 1e-8, Convert.ToDouble(np.sum(np.multiply(resid, resid)).GetAtIndex(0)));
        }

        [TestMethod]
        public void SvdValues_MatchNorm_And_MatrixRank_Random()
        {
            // Cross-check three functions against each other on random data: nuclear norm == sum(S),
            // spectral norm == max(S), rank == count of S above the tolerance.
            RequireLapack();
            np.random.seed(321);
            var a = np.random.randn(5, 4);
            var s = np.linalg.svdvals(a);
            Assert.AreEqual(Convert.ToDouble(np.sum(s).GetAtIndex(0)),
                Convert.ToDouble(np.linalg.norm(a, "nuc").GetAtIndex(0)), 1e-9);
            Assert.AreEqual(Convert.ToDouble(np.amax(s).GetAtIndex(0)),
                Convert.ToDouble(np.linalg.norm(a, 2).GetAtIndex(0)), 1e-9);
            Assert.AreEqual(4L, Convert.ToInt64(np.linalg.matrix_rank(a).GetAtIndex(0)));
        }

        #endregion

        #region NaN / inf / singular

        [TestMethod]
        public void Svd_NaN_DoesNotConverge_Inf_ProducesNaNValues()
        {
            RequireLapack();
            var nan = np.array(new double[,] { { double.NaN, 0 }, { 0, 1 } });
            new Action(() => np.linalg.svd(nan)).Should().Throw<LinAlgError>().WithMessage("SVD did not converge");
            new Action(() => np.linalg.svdvals(nan)).Should().Throw<LinAlgError>().WithMessage("SVD did not converge");
            new Action(() => np.linalg.matrix_rank(nan)).Should().Throw<LinAlgError>().WithMessage("SVD did not converge");

            // An infinite matrix CONVERGES (info == 0) but the singular values are NaN — no raise.
            var s = np.linalg.svdvals(np.array(new double[,] { { double.PositiveInfinity, 0 }, { 0, 1 } }));
            AssertClose(s, Tol, double.NaN, double.NaN);
        }

        [TestMethod]
        public void Cond_Singular_SvdOrdersStayFinite_CompositionOrdersAreInf()
        {
            RequireLapack();
            var s = np.array(new double[,] { { 1.0, 2 }, { 2, 4 } }); // singular

            // None/2/-2 go through the singular-value ratio: the smallest singular value is ~1e-16
            // (numerically nonzero), so the ratio is huge but FINITE — and it never raises.
            Assert.IsTrue(Convert.ToDouble(np.linalg.cond(s).GetAtIndex(0)) > 1e14);
            Assert.IsTrue(Convert.ToDouble(np.linalg.cond(s, 2).GetAtIndex(0)) > 1e14);

            // 1/-1/±inf/'fro' compose norm*norm(inv); NumPy's raw inv nan-fills → inf. The 2-D raise is
            // caught and routed to the same inf.
            foreach (var p in new object[] { 1, -1, double.PositiveInfinity, double.NegativeInfinity, "fro" })
                Assert.AreEqual(double.PositiveInfinity,
                    Convert.ToDouble(np.linalg.cond(s, p).GetAtIndex(0)), $"cond(singular, {p})");
        }

        [TestMethod, Misaligned]
        public void Cond_StackedWithSingularElement_Composition_Raises_WhereNumPyReturnsInf()
        {
            // Documented divergence: NumPy nan-fills the singular element of the STACKED inverse and
            // returns inf for that element (finite for the rest). NumSharp's np.linalg.inv raises on the
            // first singular element in a stack, so a composition-order cond over a stack containing a
            // singular matrix raises rather than returning per-element inf. The single 2-D case IS
            // handled (see the test above); closing the stacked case needs a non-raising inv.
            RequireLapack();
            var good = np.array(new double[,] { { 1.0, 2 }, { 3, 4 } });
            var singular = np.array(new double[,] { { 1.0, 2 }, { 2, 4 } });
            var stack = np.stack(new[] { good, singular });
            new Action(() => np.linalg.cond(stack, 1)).Should().Throw<LinAlgError>();
        }

        #endregion

        #region error taxonomy — verbatim NumPy messages

        [TestMethod]
        public void SubTwoDimensional_Operands_RaiseAtLeastTwoDimensional()
        {
            RequireLapack();
            var v = np.arange(3.0);
            new Action(() => np.linalg.svd(v)).Should().Throw<LinAlgError>()
                .WithMessage("1-dimensional array given. Array must be at least two-dimensional");
            new Action(() => np.linalg.svdvals(v)).Should().Throw<LinAlgError>()
                .WithMessage("1-dimensional array given. Array must be at least two-dimensional");
            new Action(() => np.linalg.pinv(v)).Should().Throw<LinAlgError>()
                .WithMessage("1-dimensional array given. Array must be at least two-dimensional");
        }

        [TestMethod]
        public void MutuallyExclusiveTolerances_Raise()
        {
            RequireLapack();
            new Action(() => np.linalg.pinv(np.eye(2), rcond: 1e-9, rtol: 1e-9))
                .Should().Throw<ValueError>().WithMessage("`rtol` and `rcond` can't be both set.");
            new Action(() => np.linalg.matrix_rank(np.eye(2), tol: 1e-9, rtol: 1e-9))
                .Should().Throw<ValueError>().WithMessage("`tol` and `rtol` can't be both set.");
        }

        [TestMethod]
        public void Lstsq_DimensionErrors_MatchNumPy()
        {
            RequireLapack();
            new Action(() => np.linalg.lstsq(np.ones(new Shape(2, 3)), np.ones(new Shape(4))))
                .Should().Throw<LinAlgError>().WithMessage("Incompatible dimensions");
            new Action(() => np.linalg.lstsq(np.ones(new Shape(2, 3)), np.ones(new Shape(2, 2, 2))))
                .Should().Throw<LinAlgError>().WithMessage("3-dimensional array given. Array must be two-dimensional");
        }

        #endregion

        #region value-anchored differential (NumPy 2.4.2)

        [TestMethod]
        public void Svdvals_3x3_AndDerived_MatchNumPy()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 10 } });
            AssertClose(np.linalg.svdvals(a), Tol, 17.412505166808597, 0.8751613501104364, 0.19686652111742997);
            Assert.AreEqual(3L, Convert.ToInt64(np.linalg.matrix_rank(a).GetAtIndex(0)));
        }

        [TestMethod]
        public void Pinv_Anchors_MatchNumPy()
        {
            RequireLapack();
            AssertClose(np.linalg.pinv(np.array(new double[,] { { 2, 0 }, { 0, 1 } })), Tol, 0.5, 0, 0, 1);
            AssertClose(np.linalg.pinv(np.array(new double[,] { { 1, 1 }, { 0, 1 } })), 1e-9, 1, -1, 0, 1);
        }

        [TestMethod]
        public void Lstsq_Complex_MatchesNumPy()
        {
            RequireLapack();
            var a = np.array(new Complex[,]
            {
                { new Complex(1, 0), new Complex(1, 0) },
                { new Complex(1, 0), new Complex(0, 1) },
                { new Complex(1, 0), new Complex(2, 0) },
                { new Complex(0, 0), new Complex(1, 0) }
            });
            var b = np.array(new Complex[] { new Complex(1, 1), new Complex(2, 0), new Complex(3, 0), new Complex(0, 1) });
            var (x, res, rank, s) = np.linalg.lstsq(a, b);
            Assert.AreEqual(NPTypeCode.Complex, x.typecode);
            Assert.AreEqual(typeof(double), s.dtype);
            Assert.AreEqual(2L, Convert.ToInt64(rank.GetAtIndex(0)));
            AssertClose(s, 1e-9, 2.9566293962507273, 1.1217587143526266);
            var fx = x.ravel();
            Assert.AreEqual(1.9090909090909092, ((Complex) fx.GetAtIndex(0)).Real, 1e-9);
            Assert.AreEqual(0.18181818181818155, ((Complex) fx.GetAtIndex(1)).Real, 1e-9);
            Assert.AreEqual(0.27272727272727265, ((Complex) fx.GetAtIndex(1)).Imaginary, 1e-9);
        }

        [TestMethod]
        public void MatrixNorm_SingularValueOrders_EqualTheNormOverload()
        {
            // The Array-API matrix_norm routes 2/'nuc' to the same singular-value path as norm(x, ·).
            RequireLapack();
            np.random.seed(55);
            var a = np.random.randn(4, 4);
            Assert.AreEqual(Convert.ToDouble(np.linalg.norm(a, 2).GetAtIndex(0)),
                Convert.ToDouble(np.linalg.matrix_norm(a, ord: 2).GetAtIndex(0)), 1e-9);
            Assert.AreEqual(Convert.ToDouble(np.linalg.norm(a, "nuc").GetAtIndex(0)),
                Convert.ToDouble(np.linalg.matrix_norm(a, ord: "nuc").GetAtIndex(0)), 1e-9);
        }

        #endregion
    }
}
