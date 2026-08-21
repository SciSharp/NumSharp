using System;
using System.Numerics;
using NumSharp.Interop.OpenBLAS;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     The eigen surface served by the OpenBLAS backend — <c>eigh</c>/<c>eigvalsh</c> (LAPACK
    ///     <c>syevd</c>/<c>heevd</c>) and <c>eig</c>/<c>eigvals</c> (LAPACK <c>geev</c>) — plus the seam
    ///     that turns them on.
    /// </summary>
    /// <remarks>
    ///     These need a LAPACK-capable library (the bundled scipy-openblas is one); they go
    ///     <see cref="Assert.Inconclusive(string)"/> on a host without one, exactly as the SVD/QR/LU and
    ///     matmul-parity tests do. VALUES are asserted to a tolerance so the gate holds on any machine —
    ///     the BITS depend on the OpenBLAS build, thread count and dispatched micro-kernel. Eigenvectors
    ///     carry a per-column sign/phase freedom and (for <c>eig</c>) the eigenvalue ORDER is
    ///     LAPACK-defined, so the primary check is the eigen equation <c>A·v = λ·v</c> (order- and
    ///     sign-invariant); eigenvalues are additionally compared directly (<c>eigh</c> ascending,
    ///     <c>eig</c> sorted). Byte-exactness against NumPy 2.4.2 is proven live in
    ///     <c>NumSharp.Tests.Interop</c>.
    /// </remarks>
    [TestClass]
    public class LapackEigTests
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

        // ---- assertion helpers -------------------------------------------------------------------

        private static void AssertClose(NDArray actual, double tol, params double[] expected)
        {
            Assert.AreEqual(expected.Length, actual.size, "size mismatch");
            var flat = actual.ravel();
            for (int i = 0; i < expected.Length; i++)
            {
                double a = Convert.ToDouble(flat.GetAtIndex(i));
                if (double.IsNaN(expected[i]))
                    Assert.IsTrue(double.IsNaN(a), $"[{i}] expected NaN, got {a}");
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

        /// <summary>
        ///     The eigen equation, order- and sign-invariant: <c>A · V == V · diag(w)</c> per stacked
        ///     matrix, i.e. column <c>j</c> of <c>A·V</c> equals <c>w[j]·V[:,j]</c>. Promotes <c>A</c> to
        ///     the eigenvector dtype so a real operand with complex eigenpairs still reconstructs.
        /// </summary>
        private static void AssertEigenEquation(NDArray a, NDArray w, NDArray v, double tol)
        {
            var ac = a.astype(v.typecode);
            var lhs = np.matmul(ac, v);                          // A · V
            var rhs = np.multiply(v, np.expand_dims(w, -2));     // V · diag(w): scale column j by w[j]
            AssertElementsClose(lhs, rhs, tol);
        }

        private static void AssertElementsClose(NDArray x, NDArray y, double tol)
        {
            Assert.AreEqual(x.size, y.size, "reconstruction size mismatch");
            var fx = x.ravel();
            var fy = y.ravel();
            for (long i = 0; i < x.size; i++)
            {
                var ox = fx.GetAtIndex(i);
                var oy = fy.GetAtIndex(i);
                if (ox is Complex zx)
                {
                    var zy = (Complex) oy;
                    Assert.AreEqual(zx.Real, zy.Real, tol, $"[{i}].re");
                    Assert.AreEqual(zx.Imaginary, zy.Imaginary, tol, $"[{i}].im");
                }
                else
                {
                    Assert.AreEqual(Convert.ToDouble(ox), Convert.ToDouble(oy), tol, $"[{i}]");
                }
            }
        }

        // =========================================================================================
        //  eigh / eigvalsh  (syevd / heevd)
        // =========================================================================================

        [TestMethod]
        public void Eigvalsh_Symmetric2x2_AscendingValues()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 2, 1 }, { 1, 2 } });
            AssertClose(np.linalg.eigvalsh(a), Tol, 1, 3);
        }

        [TestMethod]
        public void Eigvalsh_Symmetric3x3_MatchesNumPy()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 4, 1, 2 }, { 1, 5, 3 }, { 2, 3, 6 } });
            AssertClose(np.linalg.eigvalsh(a), Tol, 2.194397167415062, 3.386770156605786, 9.418832675979153);
        }

        [TestMethod]
        public void Eigh_Symmetric_ValuesAscending_AndVectorsSolveTheEquation()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 4, 1, 2 }, { 1, 5, 3 }, { 2, 3, 6 } });
            var (w, v) = np.linalg.eigh(a);

            Assert.AreEqual(typeof(double), w.dtype);
            Assert.AreEqual(typeof(double), v.dtype);
            AssertClose(w, Tol, 2.194397167415062, 3.386770156605786, 9.418832675979153);
            AssertEigenEquation(a, w, v, Tol);
        }

        [TestMethod]
        public void Eigh_ComplexHermitian_RealAscendingValues_ComplexVectors()
        {
            RequireLapack();
            var a = np.array(new Complex[,]
            {
                { new(1, 0), new(0, -2) },
                { new(0, 2), new(5, 0) }
            });
            var (w, v) = np.linalg.eigh(a);

            Assert.AreEqual(typeof(double), w.dtype);        // W is the real basetype
            Assert.AreEqual(typeof(Complex), v.dtype);
            AssertClose(w, Tol, 0.17157287525380982, 5.82842712474619);
            AssertEigenEquation(a, w, v, Tol);
        }

        [TestMethod]
        public void Eigvalsh_UpperAndLowerTriangle_ReadDifferentData()
        {
            RequireLapack();
            // The two triangles disagree, so 'L' (reads [[2],[1,2]]) and 'U' (reads [[2,99],[2]]) differ.
            var a = np.array(new double[,] { { 2, 99 }, { 1, 2 } });
            AssertClose(np.linalg.eigvalsh(a, 'L'), Tol, 1, 3);
            AssertClose(np.linalg.eigvalsh(a, 'U'), Tol, -97, 101);
            // Case-insensitive.
            AssertClose(np.linalg.eigvalsh(a, 'l'), Tol, 1, 3);
        }

        [TestMethod]
        public void Eigh_Float32_StaysSingle_ComputedInDouble()
        {
            RequireLapack();
            var a = np.array(new float[,] { { 2, 1 }, { 1, 2 } });
            var (w, v) = np.linalg.eigh(a);
            Assert.AreEqual(typeof(float), w.dtype);
            Assert.AreEqual(typeof(float), v.dtype);
            AssertClose(w, 1e-6, 1, 3);
            AssertEigenEquation(a, w, v, 1e-5);
        }

        [TestMethod]
        public void Eigh_IntegerAndBool_WidenToFloat64()
        {
            RequireLapack();
            AssertClose(np.linalg.eigvalsh(np.array(new int[,] { { 2, 1 }, { 1, 2 } })), Tol, 1, 3);
            Assert.AreEqual(typeof(double), np.linalg.eigh(np.array(new int[,] { { 2, 1 }, { 1, 2 } })).eigenvalues.dtype);
            AssertClose(np.linalg.eigvalsh(np.array(new bool[,] { { true, false }, { false, true } })), Tol, 1, 1);
        }

        [TestMethod]
        public void Eigh_Batched_OneDecompositionPerMatrix()
        {
            RequireLapack();
            var a = np.array(new double[,,]
            {
                { { 2, 0 }, { 0, 3 } },
                { { 5, 0 }, { 0, 1 } }
            });
            var (w, v) = np.linalg.eigh(a);
            w.Should().BeShaped(2, 2);
            v.Should().BeShaped(2, 2, 2);
            AssertClose(w, Tol, 2, 3, 1, 5); // each row ascending
            AssertEigenEquation(a, w, v, Tol);
        }

        [TestMethod]
        public void Eigh_Degenerate_1x1_And_0x0_And_EmptyBatch()
        {
            RequireLapack();
            AssertClose(np.linalg.eigvalsh(np.array(new double[,] { { 7 } })), Tol, 7);

            var (w0, v0) = np.linalg.eigh(np.zeros(new Shape(0, 0)));
            w0.Should().BeShaped(0);
            v0.Should().BeShaped(0, 0);

            np.linalg.eigvalsh(np.zeros(new Shape(0, 3, 3))).Should().BeShaped(0, 3);
        }

        [TestMethod]
        public void Eigh_DoesNotCheckFinite_InfReturnsNaN()
        {
            RequireLapack();
            // eigh/eigvalsh, unlike eig/eigvals, run no _assert_finite — an inf operand yields NaN, no raise.
            var a = np.array(new double[,] { { double.PositiveInfinity, 0 }, { 0, 1 } });
            AssertClose(np.linalg.eigvalsh(a), Tol, double.NaN, double.NaN);
        }

        // =========================================================================================
        //  eig / eigvals  (geev)
        // =========================================================================================

        [TestMethod]
        public void Eig_RealMatrixWithRealEigenvalues_CollapsesToReal()
        {
            RequireLapack();
            var a = np.diag(np.array(new double[] { 1, 2, 3 }));
            var (w, v) = np.linalg.eig(a);

            // all imaginary parts zero -> real dtype (NumPy's real-output optimization)
            Assert.AreEqual(typeof(double), w.dtype);
            Assert.AreEqual(typeof(double), v.dtype);
            AssertClose(w, Tol, 1, 2, 3);
            AssertEigenEquation(a, w, v, Tol);
        }

        [TestMethod]
        public void Eig_RealMatrixWithComplexEigenvalues_IsComplex()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 1, -1 }, { 1, 1 } });
            var (w, v) = np.linalg.eig(a);

            Assert.AreEqual(typeof(Complex), w.dtype);
            Assert.AreEqual(typeof(Complex), v.dtype);
            AssertCloseComplex(w, Tol, new Complex(1, 1), new Complex(1, -1));
            AssertEigenEquation(a, w, v, Tol);
        }

        [TestMethod]
        public void Eigvals_MatchesEig_WithoutVectors()
        {
            RequireLapack();
            AssertClose(np.linalg.eigvals(np.diag(np.array(new double[] { 1, 2, 3 }))), Tol, 1, 2, 3);
            AssertCloseComplex(np.linalg.eigvals(np.array(new double[,] { { 1, -1 }, { 1, 1 } })), Tol,
                new Complex(1, 1), new Complex(1, -1));
        }

        [TestMethod]
        public void Eig_Float32_RealEigsStaySingle_ComplexEigsBecomeComplex128()
        {
            RequireLapack();
            // float32 real-eigenvalue result stays single...
            var wReal = np.linalg.eig(np.diag(np.array(new float[] { 1, 2, 3 }))).eigenvalues;
            Assert.AreEqual(typeof(float), wReal.dtype);
            AssertClose(wReal, 1e-6, 1, 2, 3);

            // ...but a complex result is complex128 — NumSharp has no complex64 (documented divergence,
            // values identical to NumPy's complex64).
            var wCplx = np.linalg.eig(np.array(new float[,] { { 1, -1 }, { 1, 1 } })).eigenvalues;
            Assert.AreEqual(typeof(Complex), wCplx.dtype);
            AssertCloseComplex(wCplx, 1e-6, new Complex(1, 1), new Complex(1, -1));
        }

        [TestMethod]
        public void Eig_IntegerWidensToFloat64()
        {
            RequireLapack();
            var w = np.linalg.eig(np.diag(np.array(new int[] { 1, 2, 3 }))).eigenvalues;
            Assert.AreEqual(typeof(double), w.dtype);
            AssertClose(w, Tol, 1, 2, 3);
        }

        [TestMethod]
        public void Eig_ComplexInput_StaysComplex_NoRealCollapse()
        {
            RequireLapack();
            var a = np.array(new Complex[,]
            {
                { new(1, 0), new(0, 1) },
                { new(0, -1), new(1, 0) }
            });
            var (w, v) = np.linalg.eig(a);
            Assert.AreEqual(typeof(Complex), w.dtype);
            AssertCloseComplex(w, Tol, new Complex(2, 0), new Complex(0, 0));
            AssertEigenEquation(a, w, v, Tol);
        }

        [TestMethod]
        public void Eig_BatchedWithOneComplexElement_MakesWholeResultComplex()
        {
            RequireLapack();
            // `all(w.imag == 0)` is a GLOBAL reduction, so one complex-eigenvalue matrix in the stack
            // makes the entire result complex.
            var a = np.array(new double[,,]
            {
                { { 1, -1 }, { 1, 1 } },  // complex eigenpair
                { { 2, 0 }, { 0, 3 } }    // real eigenvalues
            });
            var (w, v) = np.linalg.eig(a);
            Assert.AreEqual(typeof(Complex), w.dtype);
            w.Should().BeShaped(2, 2);
            AssertCloseComplex(w, Tol,
                new Complex(1, 1), new Complex(1, -1), new Complex(2, 0), new Complex(3, 0));
            AssertEigenEquation(a, w, v, Tol);
        }

        [TestMethod]
        public void Eig_UpperTriangular_EigenvaluesAreTheDiagonal()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 1, 2, 3 }, { 0, 4, 5 }, { 0, 0, 6 } });
            var w = np.linalg.eigvals(a);
            Assert.AreEqual(typeof(double), w.dtype);
            AssertClose(w, Tol, 1, 4, 6);
        }

        [TestMethod]
        public void Eig_Degenerate_1x1_And_0x0()
        {
            RequireLapack();
            AssertClose(np.linalg.eig(np.array(new double[,] { { 7 } })).eigenvalues, Tol, 7);
            np.linalg.eig(np.zeros(new Shape(0, 0))).eigenvalues.Should().BeShaped(0);
        }

        // =========================================================================================
        //  Error taxonomy
        // =========================================================================================

        [TestMethod]
        public void Eig_And_Eigvals_RejectNonFinite()
        {
            RequireLapack();
            var inf = np.array(new double[,] { { double.PositiveInfinity, 0 }, { 0, 1 } });
            var nan = np.array(new double[,] { { double.NaN, 0 }, { 0, 1 } });
            new Action(() => np.linalg.eig(inf)).Should().Throw<LinAlgError>()
                .WithMessage("Array must not contain infs or NaNs");
            new Action(() => np.linalg.eigvals(nan)).Should().Throw<LinAlgError>()
                .WithMessage("Array must not contain infs or NaNs");
        }

        [TestMethod]
        public void Eigh_UploValidatedBeforeArray()
        {
            RequireLapack();
            new Action(() => np.linalg.eigh(np.array(new double[,] { { 2, 1 }, { 1, 2 } }), 'X'))
                .Should().Throw<ValueError>().WithMessage("UPLO argument must be 'L' or 'U'");
        }

        [TestMethod]
        public void Eigen_RejectNonSquareAndTooFewDimensions()
        {
            RequireLapack();
            new Action(() => np.linalg.eig(np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } })))
                .Should().Throw<LinAlgError>().WithMessage("Last 2 dimensions of the array must be square");
            new Action(() => np.linalg.eigh(np.array(new double[] { 1, 2, 3 })))
                .Should().Throw<LinAlgError>()
                .WithMessage("1-dimensional array given. Array must be at least two-dimensional");
        }
    }
}
