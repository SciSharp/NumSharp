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
    ///     <c>eig</c> sorted). Byte-exactness against NumPy 2.5.4 is proven live in
    ///     <c>NumSharp.Tests.Interop</c>. Since NumPy 2.5 (gh-30411) <c>eig</c> / <c>eigvals</c> always return complex
    ///     results: a real operand with real eigenvalues no longer collapses to a real dtype.
    /// </remarks>
    [TestClass]
    [DoNotParallelize] // binds the process-wide OpenBLAS backend (TensorEngine.Blas)
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
        public void Eig_RealMatrixWithRealEigenvalues_IsComplex()
        {
            RequireLapack();
            var a = np.diag(np.array(new double[] { 1, 2, 3 }));
            var (w, v) = np.linalg.eig(a);

            // NumPy 2.5 dropped the real-output collapse (2.4 returned float64 when every imaginary part was zero).
            Assert.AreEqual(typeof(Complex), w.dtype);
            Assert.AreEqual(typeof(Complex), v.dtype);
            AssertCloseComplex(w, Tol, new Complex(1, 0), new Complex(2, 0), new Complex(3, 0));
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
            AssertCloseComplex(np.linalg.eigvals(np.diag(np.array(new double[] { 1, 2, 3 }))), Tol, new Complex(1, 0), new Complex(2, 0), new Complex(3, 0));
            AssertCloseComplex(np.linalg.eigvals(np.array(new double[,] { { 1, -1 }, { 1, 1 } })), Tol,
                new Complex(1, 1), new Complex(1, -1));
        }

        /// <summary>
        ///     An eigenvalue result must keep its buffer alive past later calls. The real-collapsed result NumPy 2.4 returned
        ///     (<c>w.real</c>, a lane VIEW of the complex buffer the eig scope disposes) once went back to the pool the moment
        ///     eigvals returned, and the next same-size allocation overwrote it; NumPy 2.5's result is the complex array
        ///     itself, which owns its data (probed: <c>flags.owndata</c> True).
        /// </summary>
        [TestMethod]
        public void Eigvals_Result_OwnsItsBuffer_SurvivesLaterCallsAndAllocations()
        {
            RequireLapack();
            var r1 = np.linalg.eigvals(np.array(new double[,] { { 2, 0 }, { 0, 3 } }));
            var (w1, _) = np.linalg.eig(np.array(new double[,] { { 4, 0 }, { 0, 5 } }));
            var r2 = np.linalg.eigvals(np.array(new double[,] { { 7, 0 }, { 0, 9 } }));
            using (var thief = np.full(new Shape(2), new Complex(-111, -222)))
            {
                AssertCloseComplex(r1, 0, new Complex(2, 0), new Complex(3, 0));
                AssertCloseComplex(w1, 0, new Complex(4, 0), new Complex(5, 0));
                AssertCloseComplex(r2, 0, new Complex(7, 0), new Complex(9, 0));
            }
            r1.Storage.InternalArray.IsReleased.Should().BeFalse();
            r1.flags.owndata.Should().BeTrue("NumPy 2.5's eigvals returns the complex result itself");
        }

        [TestMethod]
        public void Eig_Float32_IsComplex128WithSingleValues()
        {
            RequireLapack();
            // NumPy 2.5's float32 result is complex64 whatever the eigenvalues (2.4 kept a real result float32); NumSharp has
            // no complex64, so it is complex128 holding the same float32-rounded values (documented divergence, #569).
            var wReal = np.linalg.eig(np.diag(np.array(new float[] { 1, 2, 3 }))).eigenvalues;
            Assert.AreEqual(typeof(Complex), wReal.dtype);
            AssertCloseComplex(wReal, 1e-6, new Complex(1, 0), new Complex(2, 0), new Complex(3, 0));

            var wCplx = np.linalg.eig(np.array(new float[,] { { 1, -1 }, { 1, 1 } })).eigenvalues;
            Assert.AreEqual(typeof(Complex), wCplx.dtype);
            AssertCloseComplex(wCplx, 1e-6, new Complex(1, 1), new Complex(1, -1));
        }

        /// <summary>
        ///     A float32 operand's complex result is NumPy's complex64: geev runs in double and <c>astype(complex64)</c> rounds
        ///     every component to float32. NumSharp keeps the complex128 dtype (#569) but must carry exactly those values —
        ///     numpy.polynomial's <c>{p}roots</c> of a float32 series sort them, so a component left at double precision
        ///     would differ from NumPy's in its low bits. NumPy 2.4.2: <c>np.linalg.eigvals(np.array([[0, -2], [1, 0]],
        ///     np.float32))</c> is <c>[1.4142135j, -1.4142135j]</c> (complex64), widened <c>0x3ff6a09e60000000</c>.
        /// </summary>
        [TestMethod]
        public void Eig_Float32ComplexResult_CarriesNumPysComplex64Values()
        {
            RequireLapack();
            static bool IsSingle(double d) => (double)(float)d == d || double.IsNaN(d);

            var a = np.array(new float[,] { { 0, -2 }, { 1, 0 } });
            var w = np.linalg.eigvals(a);
            Assert.AreEqual(NPTypeCode.Complex, w.typecode);
            double root2 = BitConverter.Int64BitsToDouble(0x3ff6a09e60000000);   // (double)(float)Math.Sqrt(2)
            var z0 = (Complex)w.GetAtIndex(0);
            var z1 = (Complex)w.GetAtIndex(1);
            Assert.AreEqual(0.0, z0.Real);
            Assert.AreEqual(root2, z0.Imaginary, 0.0);
            Assert.AreEqual(0.0, z1.Real);
            Assert.AreEqual(-root2, z1.Imaginary, 0.0);

            // eig rounds the eigenvectors the same way (NumPy's `v.astype(complex64)`), and a non-trivial spectrum keeps
            // every component single-representable.
            var (we, ve) = np.linalg.eig(np.array(new float[,] { { 1, -3, 0.5f }, { 2, 0.25f, -1 }, { 0, 1, 1.5f } }));
            foreach (var arr in new[] { we, ve })
                foreach (Complex z in arr.ravel().ToArray<Complex>())
                    Assert.IsTrue(IsSingle(z.Real) && IsSingle(z.Imaginary), $"component of {z} is not a float32 value");

            // Control: a float64 operand's complex eigenvalues keep double precision.
            var w64 = np.linalg.eigvals(np.array(new double[,] { { 0, -2 }, { 1, 0 } }));
            Assert.IsFalse(IsSingle(((Complex)w64.GetAtIndex(0)).Imaginary));
        }

        [TestMethod]
        public void Eig_IntegerWidensToComplex128()
        {
            RequireLapack();
            var w = np.linalg.eig(np.diag(np.array(new int[] { 1, 2, 3 }))).eigenvalues;
            Assert.AreEqual(typeof(Complex), w.dtype);
            AssertCloseComplex(w, Tol, new Complex(1, 0), new Complex(2, 0), new Complex(3, 0));
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
            Assert.AreEqual(typeof(Complex), w.dtype);
            AssertCloseComplex(w, Tol, new Complex(1, 0), new Complex(4, 0), new Complex(6, 0));
        }

        [TestMethod]
        public void Eig_Degenerate_1x1_And_0x0()
        {
            RequireLapack();
            AssertCloseComplex(np.linalg.eig(np.array(new double[,] { { 7 } })).eigenvalues, Tol, new Complex(7, 0));
            var empty = np.linalg.eig(np.zeros(new Shape(0, 0)));
            empty.eigenvalues.Should().BeShaped(0);
            Assert.AreEqual(typeof(Complex), empty.eigenvalues.dtype);
            Assert.AreEqual(typeof(Complex), empty.eigenvectors.dtype);
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
