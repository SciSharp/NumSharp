using System;
using System.Numerics;
using NumSharp.Interop.OpenBLAS;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     The LU-based <c>np.linalg</c> factorisations served by the OpenBLAS backend —
    ///     <c>solve</c>, <c>inv</c>, <c>det</c>, <c>slogdet</c>, and the functions that compose on them
    ///     (<c>matrix_power</c> with a negative exponent, <c>tensorinv</c>, <c>tensorsolve</c>).
    /// </summary>
    /// <remarks>
    ///     These need a LAPACK-capable library (a full OpenBLAS; the bundled scipy-openblas is one).
    ///     They go <see cref="Assert.Inconclusive(string)"/> on a host without one, exactly as the
    ///     matmul-parity tests do. VALUES are asserted to a tolerance so the gate holds on any machine;
    ///     the BITS are host-dependent (they depend on the OpenBLAS build, thread count and dispatched
    ///     micro-kernel, the same three levers matmul parity documents) and are not asserted here.
    ///     Correctness — that the numbers, shapes, dtypes and error taxonomy match NumPy 2.4.2 — is
    ///     the claim, and it was verified bit-exact against NumPy on the reference host.
    /// </remarks>
    [TestClass]
    public class LapackFactorisationTests
    {
        private const double Tol = 1e-9;

        [TestCleanup]
        public void Cleanup() => OpenBlasEngine.Disable();

        /// <summary>Enable the backend, or skip loudly on a host with no LAPACK-capable BLAS.</summary>
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
                Assert.Inconclusive("the loaded BLAS exports no LAPACK LU routines (a bare reference CBLAS).");
        }

        private static void AssertClose(NDArray actual, double tol, params double[] expected)
        {
            Assert.AreEqual(expected.Length, actual.size, "size mismatch");
            var flat = actual.ravel();
            for (int i = 0; i < expected.Length; i++)
            {
                // GetAtIndex returns the boxed native element (GetDouble reinterprets a non-double
                // buffer as a half-Count double view and asserts out of bounds — a stock accessor,
                // not the result being wrong: the float32 values themselves are correct).
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

        #region inv

        [TestMethod]
        public void Inv_TwoByTwo_MatchesNumPy()
        {
            RequireLapack();
            var r = np.linalg.inv(np.array(new double[,] { { 1, 2 }, { 3, 4 } }));
            r.Should().BeShaped(2, 2);
            Assert.AreEqual(typeof(double), r.dtype);
            AssertClose(r, Tol, -2.0, 1.0, 1.5, -0.5);
        }

        [TestMethod]
        public void Inv_ThreeByThree_MatchesNumPy()
        {
            RequireLapack();
            var m = np.array(new double[,]
            {
                { 0.5488135039273248, 0.7151893663724195, 0.6027633760716439 },
                { 0.5448831829968969, 0.4236547993389047, 0.6458941130666561 },
                { 0.4375872112626925, 0.8917730007820798, 0.9636627605010293 }
            });
            AssertClose(np.linalg.inv(m), Tol,
                1.9896085216746318, 1.7991376599275117, -2.450354698257213,
                2.8759088731373965, -3.1447116479211425, 0.3088821227074869,
                -3.564820880378896, 2.0931485518096618, 1.8645435054747006);
        }

        [TestMethod]
        public void Inv_Float32_ComputesInDouble_AndReturnsSingle()
        {
            // NumPy's linalg is "lite": a float32 operand is factorised in float64 and cast back, so
            // the result dtype is float32 but every value is the double answer rounded once.
            RequireLapack();
            var r = np.linalg.inv(np.array(new float[,] { { 1, 2 }, { 3, 4 } }));
            Assert.AreEqual(typeof(float), r.dtype);
            AssertClose(r, 1e-6, -2.0, 1.0, 1.5, -0.5);
        }

        [TestMethod]
        public void Inv_IntegerOperand_WidensToDouble()
        {
            RequireLapack();
            var r = np.linalg.inv(np.array(new int[,] { { 2, 0 }, { 0, 4 } }));
            Assert.AreEqual(typeof(double), r.dtype);
            AssertClose(r, Tol, 0.5, 0.0, 0.0, 0.25);
        }

        [TestMethod]
        public void Inv_Complex_MatchesNumPy()
        {
            RequireLapack();
            var z = np.array(new Complex[,]
            {
                { new Complex(1, 1), new Complex(2, 0) },
                { new Complex(3, 0), new Complex(4, -1) }
            });
            AssertCloseComplex(np.linalg.inv(z), Tol,
                new Complex(-0.7, -1.1), new Complex(0.2, 0.6),
                new Complex(0.3, 0.9), new Complex(0.2, -0.4));
        }

        [TestMethod]
        public void Inv_Stacked_InvertsEachMatrix()
        {
            RequireLapack();
            var c = np.array(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
            var d = np.array(new double[,] { { 0.5488135039273248, 0.7151893663724195 },
                                             { 0.5448831829968969, 0.4236547993389047 } });
            var r = np.linalg.inv(np.stack(new[] { c, d }));
            r.Should().BeShaped(2, 2, 2);
            AssertClose(r["0"], Tol, -2.0, 1.0, 1.5, -0.5);
            AssertClose(r["1"], Tol, -2.695224826023495, 4.549921630919536,
                3.466460628765684, -3.4914647088857707);
        }

        [TestMethod]
        public void Inv_RoundTrips_AInvA_IsIdentity()
        {
            RequireLapack();
            var m = np.array(new double[,]
            {
                { 0.5488135039273248, 0.7151893663724195, 0.6027633760716439 },
                { 0.5448831829968969, 0.4236547993389047, 0.6458941130666561 },
                { 0.4375872112626925, 0.8917730007820798, 0.9636627605010293 }
            });
            AssertClose(np.matmul(m, np.linalg.inv(m)), 1e-9, 1, 0, 0, 0, 1, 0, 0, 0, 1);
        }

        #endregion

        #region solve

        [TestMethod]
        public void Solve_Vector_MatchesNumPy()
        {
            RequireLapack();
            var r = np.linalg.solve(np.array(new double[,] { { 1, 2 }, { 3, 5 } }), np.array(new double[] { 1, 2 }));
            r.Should().BeShaped(2);
            AssertClose(r, Tol, -1.0, 1.0);
        }

        [TestMethod]
        public void Solve_MatrixRhs_MatchesNumPy()
        {
            RequireLapack();
            var r = np.linalg.solve(np.array(new double[,] { { 1, 2 }, { 3, 5 } }),
                np.array(new double[,] { { 1, 0 }, { 0, 1 } }));
            r.Should().BeShaped(2, 2);
            // == inv([[1,2],[3,5]])
            AssertClose(r, Tol, -5.0, 2.0, 3.0, -1.0);
        }

        [TestMethod]
        public void Solve_BroadcastsBatchDimensions()
        {
            // a is a single 2x2; b is a stack of four 2x3 — a broadcasts across the batch.
            RequireLapack();
            var r = np.linalg.solve(np.array(new double[,] { { 1, 2 }, { 3, 5 } }),
                np.arange(24.0).reshape(4, 2, 3));
            r.Should().BeShaped(4, 2, 3);
            AssertClose(r["0"], Tol, 6.0, 3.0, 0.0, -3.0, -1.0, 1.0);
        }

        [TestMethod]
        public void Solve_Float32_ReturnsSingle()
        {
            RequireLapack();
            var r = np.linalg.solve(np.array(new float[,] { { 1, 2 }, { 3, 5 } }), np.array(new float[] { 1, 2 }));
            Assert.AreEqual(typeof(float), r.dtype);
            AssertClose(r, 1e-6, -1.0, 1.0);
        }

        [TestMethod]
        public void Solve_ProducesTheInverseWhenRhsIsIdentity()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 4.0, 3.0 }, { 6.0, 3.0 } });
            var x = np.linalg.solve(a, np.eye(2));
            AssertClose(np.matmul(a, x), 1e-9, 1, 0, 0, 1);
        }

        #endregion

        #region det / slogdet

        [TestMethod]
        public void Det_MatchesNumPy_AcrossDtypes()
        {
            RequireLapack();
            var c = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            var det = np.linalg.det(c);
            det.Should().BeShaped();
            AssertClose(det, Tol, -2.0);

            AssertClose(np.linalg.det(np.array(new int[,] { { 1, 2 }, { 3, 4 } })), Tol, -2.0);
            Assert.AreEqual(typeof(float), np.linalg.det(c.astype(np.float32)).dtype);
            AssertClose(np.linalg.det(c.astype(np.float32)), 1e-5, -2.0);
        }

        [TestMethod]
        public void Det_Stacked_OnePerMatrix()
        {
            RequireLapack();
            var c = np.array(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
            var d = np.array(new double[,] { { 0.5488135039273248, 0.7151893663724195 },
                                             { 0.5448831829968969, 0.4236547993389047 } });
            var r = np.linalg.det(np.stack(new[] { c, d }));
            r.Should().BeShaped(2);
            AssertClose(r, Tol, -2.0000000000000004, -0.15718718351372576);
        }

        [TestMethod]
        public void Det_Complex_MatchesNumPy()
        {
            RequireLapack();
            var z = np.array(new Complex[,]
            {
                { new Complex(1, 1), new Complex(2, 0) },
                { new Complex(3, 0), new Complex(4, -1) }
            });
            AssertCloseComplex(np.linalg.det(z), Tol, new Complex(-1.0, 3.0));
        }

        [TestMethod]
        public void Slogdet_MatchesNumPy()
        {
            RequireLapack();
            var (sign, logabsdet) = np.linalg.slogdet(np.array(new double[,] { { 1, 2 }, { 3, 4 } }));
            AssertClose(sign, Tol, -1.0);
            AssertClose(logabsdet, Tol, 0.6931471805599453);
        }

        [TestMethod]
        public void Slogdet_Complex_SignIsUnitModulus()
        {
            RequireLapack();
            var z = np.array(new Complex[,]
            {
                { new Complex(1, 1), new Complex(2, 0) },
                { new Complex(3, 0), new Complex(4, -1) }
            });
            var (sign, logabsdet) = np.linalg.slogdet(z);
            AssertCloseComplex(sign, Tol, new Complex(-0.31622776601683805, 0.9486832980505138));
            Assert.AreEqual(typeof(double), logabsdet.dtype); // real_t is float64 for a complex operand
            AssertClose(logabsdet, Tol, 1.151292546497023);
        }

        #endregion

        #region composed: matrix_power(<0), tensorinv, tensorsolve

        [TestMethod]
        public void MatrixPower_NegativeExponent_InvertsThenExponentiates()
        {
            RequireLapack();
            var c = np.array(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
            AssertClose(np.linalg.matrix_power(c, -1), Tol, -2.0, 1.0, 1.5, -0.5);
            AssertClose(np.linalg.matrix_power(c, -2), Tol,
                5.499999999999997, -2.499999999999999, -3.7499999999999987, 1.7499999999999996);
        }

        [TestMethod]
        public void Tensorinv_ComposesOnInv()
        {
            RequireLapack();
            var a = np.eye(4 * 6).reshape(4, 6, 8, 3);
            var ainv = np.linalg.tensorinv(a, 2);
            ainv.Should().BeShaped(8, 3, 4, 6);
            // a is the identity-tensor, so its tensordot inverse reshapes an identity too.
            Assert.AreEqual(1.0, ainv.reshape(24, 24).GetDouble(0), Tol);
        }

        [TestMethod]
        public void Tensorsolve_ComposesOnSolve()
        {
            RequireLapack();
            var a = np.eye(2 * 3 * 4).reshape(2 * 3, 4, 2, 3, 4);
            var b = np.arange(2 * 3 * 4).reshape(2 * 3, 4).astype(np.float64);
            var x = np.linalg.tensorsolve(a, b);
            x.Should().BeShaped(2, 3, 4);
            // a is identity → x is b reshaped: 0,1,2,...,23.
            AssertClose(x.reshape(24), Tol,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23);
        }

        #endregion

        #region edge cases: singular, empty, 1x1, non-finite

        [TestMethod]
        public void Singular_Inv_And_Solve_RaiseLinAlgError()
        {
            RequireLapack();
            var s = np.array(new double[,] { { 1, 2 }, { 2, 4 } });
            new Action(() => np.linalg.inv(s)).Should().Throw<LinAlgError>().WithMessage("Singular matrix");
            new Action(() => np.linalg.solve(s, np.array(new double[] { 1, 2 })))
                .Should().Throw<LinAlgError>().WithMessage("Singular matrix");
        }

        [TestMethod]
        public void Singular_Det_Is_Zero_And_Slogdet_Is_NegInf_WithoutRaising()
        {
            RequireLapack();
            var s = np.array(new double[,] { { 1, 2 }, { 2, 4 } });
            AssertClose(np.linalg.det(s), Tol, 0.0);
            var (sign, logabsdet) = np.linalg.slogdet(s);
            AssertClose(sign, Tol, 0.0);
            AssertClose(logabsdet, Tol, double.NegativeInfinity);
        }

        [TestMethod]
        public void Singular_InAStack_Inv_Raises_But_Det_ReportsPerElement()
        {
            RequireLapack();
            var good = np.array(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
            var singular = np.array(new double[,] { { 1.0, 2.0 }, { 2.0, 4.0 } });
            var stack = np.stack(new[] { good, singular });

            new Action(() => np.linalg.inv(stack)).Should().Throw<LinAlgError>();
            AssertClose(np.linalg.det(stack), Tol, -2.0000000000000004, 0.0);
            var (sign, logabsdet) = np.linalg.slogdet(stack);
            AssertClose(sign, Tol, -1.0, 0.0);
            AssertClose(logabsdet, Tol, 0.6931471805599453, double.NegativeInfinity);
        }

        [TestMethod]
        public void Empty_ZeroByZero_Matches_NumPy()
        {
            RequireLapack();
            np.linalg.inv(np.zeros(new Shape(0, 0))).Should().BeShaped(0, 0);
            AssertClose(np.linalg.det(np.zeros(new Shape(0, 0))), Tol, 1.0);   // empty product
            var (sign, logabsdet) = np.linalg.slogdet(np.zeros(new Shape(0, 0)));
            AssertClose(sign, Tol, 1.0);
            AssertClose(logabsdet, Tol, 0.0);
            np.linalg.solve(np.zeros(new Shape(0, 0)), np.zeros(new Shape(0))).Should().BeShaped(0);
        }

        [TestMethod]
        public void Empty_BatchAndZeroSizedMatrices()
        {
            RequireLapack();
            np.linalg.det(np.zeros(new Shape(0, 3, 3))).Should().BeShaped(0);   // zero batch
            AssertClose(np.linalg.det(np.zeros(new Shape(2, 0, 0))), Tol, 1.0, 1.0); // two 0x0 → [1,1]
        }

        [TestMethod]
        public void OneByOne_Works()
        {
            RequireLapack();
            AssertClose(np.linalg.inv(np.array(new double[,] { { 4.0 } })), Tol, 0.25);
            AssertClose(np.linalg.det(np.array(new double[,] { { 4.0 } })), Tol, 4.0);
        }

        [TestMethod]
        public void NonFinite_DoesNotFalselyReportSingular()
        {
            // A NaN on the diagonal does not make LAPACK report a zero pivot: inv produces NaNs and
            // does NOT raise, and det of an infinite matrix is infinite. Both match NumPy.
            RequireLapack();
            AssertClose(np.linalg.inv(np.array(new double[,] { { double.NaN, 0 }, { 0, 1 } })), Tol,
                double.NaN, double.NaN, 0.0, 1.0);
            AssertClose(np.linalg.det(np.array(new double[,] { { double.PositiveInfinity, 0 }, { 0, 1 } })), Tol,
                double.PositiveInfinity);
        }

        #endregion

        #region completeness pins (validation sweep)

        [TestMethod]
        public void AllFifteenDtypes_WidenComputeOrReject_LikeNumPy()
        {
            // bool + every integer widen to float64 and compute; float32/float64/complex compute at
            // their result width; the three dtypes with no NumPy counterpart (Half=float16,
            // Decimal, Char) raise the verbatim "unsupported in linalg" TypeError — float16 sharing
            // NumPy's own message.
            RequireLapack();

            foreach (var tc in new[]
            {
                NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64
            })
            {
                var m = np.array(new double[,] { { 2, 1 }, { 1, 2 } }).astype(tc);
                var inv = np.linalg.inv(m);
                Assert.AreEqual(typeof(double), inv.dtype, $"{tc} should widen to double");
                AssertClose(inv, Tol, 2.0 / 3, -1.0 / 3, -1.0 / 3, 2.0 / 3);
            }

            // bool widens too, but [[2,1],[1,2]] casts to all-True (singular), so use an identity.
            var boolInv = np.linalg.inv(np.array(new double[,] { { 1, 0 }, { 0, 1 } }).astype(NPTypeCode.Boolean));
            Assert.AreEqual(typeof(double), boolInv.dtype);
            AssertClose(boolInv, Tol, 1.0, 0.0, 0.0, 1.0);

            Assert.AreEqual(typeof(float), np.linalg.inv(np.array(new float[,] { { 2, 1 }, { 1, 2 } })).dtype);
            Assert.AreEqual(typeof(double), np.linalg.inv(np.array(new double[,] { { 2, 1 }, { 1, 2 } })).dtype);
            Assert.AreEqual(NPTypeCode.Complex,
                np.linalg.inv(np.array(new System.Numerics.Complex[,] { { 2, 1 }, { 1, 2 } })).typecode);

            foreach (var (tc, name) in new[]
            {
                (NPTypeCode.Half, "float16"), (NPTypeCode.Decimal, "decimal"), (NPTypeCode.Char, "char")
            })
            {
                var m = np.array(new double[,] { { 2, 1 }, { 1, 2 } }).astype(tc);
                new Action(() => np.linalg.inv(m)).Should().Throw<TypeError>()
                    .WithMessage($"array type {name} is unsupported in linalg");
            }
        }

        [TestMethod]
        public void MatrixPower_DtypeRule_IntStaysIntForNonNegative_FloatForNegative()
        {
            RequireLapack();
            var i = np.array(new int[,] { { 2, 1, 0 }, { 0, 2, 1 }, { 1, 0, 2 } });
            Assert.AreEqual(typeof(int), np.linalg.matrix_power(i, 0).dtype);   // identity, still int
            Assert.AreEqual(typeof(int), np.linalg.matrix_power(i, 3).dtype);
            Assert.AreEqual(typeof(double), np.linalg.matrix_power(i, -1).dtype); // inverse floats it
            Assert.AreEqual(typeof(double), np.linalg.matrix_power(i, -2).dtype);
        }

        [TestMethod]
        public void Tensorinv_IndOneAndThree_ReshapeTheRightWay()
        {
            RequireLapack();
            np.linalg.tensorinv(np.eye(24).reshape(24, 8, 3), 1).Should().BeShaped(8, 3, 24);
            np.linalg.tensorinv(np.eye(24).reshape(2, 3, 4, 24), 3).Should().BeShaped(24, 2, 3, 4);
        }

        [TestMethod]
        public void NonContiguousLayouts_MatchTheContiguousInverse()
        {
            // The operand is read through its own strides, so a reversed / transposed / F-order /
            // offset view inverts to the same values as its contiguous copy.
            RequireLapack();
            var b = np.array(new double[,] { { 4.0, 1.0, 2.0 }, { 0.0, 3.0, 1.0 }, { 1.0, 0.0, 5.0 } });
            var expected = np.linalg.inv(b.copy());

            foreach (var view in new[] { b["::-1"]["::-1"], b.T.T, np.asfortranarray(b) })
                AssertClose(np.linalg.inv(view.copy()), Tol,
                    ToDoubles(np.linalg.inv(view)));  // view path == contiguous path

            AssertClose(np.linalg.inv(np.asfortranarray(b)), Tol, ToDoubles(expected));
        }

        private static double[] ToDoubles(NDArray a)
        {
            a = a.ravel();
            var r = new double[a.size];
            for (long i = 0; i < a.size; i++)
                r[i] = Convert.ToDouble(a.GetAtIndex(i));
            return r;
        }

        #endregion

        #region the seam: the backend is what enables these

        [TestMethod]
        public void TheLuFamily_ComputesManagedWithoutTheBackend_AndThroughLapackWithIt()
        {
            // The LU family — inv/det/solve/slogdet — has a managed fallback (ManagedLu), so unlike the
            // Cholesky/QR/SVD/eigen factorisations it computes WITH OR WITHOUT a backend: without one it
            // takes the managed path, installing the backend routes it through LAPACK, and both agree
            // with NumPy. (The seam LinAlgEngineSeamTests documents, exercised from the other side.)
            OpenBlasEngine.Disable();
            var c = np.array(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
            c.TensorEngine.Blas.Should().BeNull("no backend is installed at this point");
            AssertClose(np.linalg.inv(c), Tol, -2.0, 1.0, 1.5, -0.5);        // managed
            AssertClose(np.linalg.det(c), Tol, -2.0);
            AssertClose(np.linalg.solve(c, np.arange(2.0)), Tol, 1.0, -0.5);

            RequireLapack();
            AssertClose(np.linalg.inv(c), Tol, -2.0, 1.0, 1.5, -0.5);        // through LAPACK
            AssertClose(np.linalg.det(c), Tol, -2.0);
            AssertClose(np.linalg.solve(c, np.arange(2.0)), Tol, 1.0, -0.5);
        }

        [TestMethod]
        public void EnablingTheBackend_DoesNotChangeUnrelatedOperations()
        {
            RequireLapack();
            var a = np.arange(6).astype(NPTypeCode.Double).reshape(2, 3);
            Assert.AreEqual(15.0, (double)np.sum(a));
            (a + a).Should().BeShaped(2, 3);
        }

        #endregion
    }
}
