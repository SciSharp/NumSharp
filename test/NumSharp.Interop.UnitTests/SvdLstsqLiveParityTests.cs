using System;
using System.Collections.Generic;
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
    ///     Live-numpy byte-parity for the OpenBLAS SVD/least-squares surface
    ///     (<c>np.linalg.svd/svdvals/pinv/matrix_rank/cond/norm{2,-2,'nuc'}/lstsq</c>): every case
    ///     computes the op TWICE over the very same exported bytes — once in NumSharp through the
    ///     bundled scipy-openblas backend, once in the LIVE numpy in this process — and asserts they
    ///     agree BYTE-FOR-BYTE.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Both stacks call the SAME single-threaded scipy-openblas <c>gesdd</c>/<c>gelsd</c> (numpy
    ///     pins it, this package bundles the byte-identical copy), so the factor bytes are a
    ///     deterministic function of the input — U/S/Vh are byte-identical even though they carry a
    ///     per-column sign freedom, because the same LAPACK routine resolves that sign the same way on
    ///     both sides. numpy's linalg computes every factorisation in double/cdouble and casts the
    ///     result back (`_commonType`), so a float32 operand is factorised with the double routine and
    ///     rounded once — NumSharp reproduces that exactly, so a float32 result is byte-identical too,
    ///     and bool/integer operands widen to float64 on both sides.
    ///     </para>
    ///     <para>
    ///     Every input is handed to numpy through the interop's zero-copy export, so numpy computes
    ///     over NumSharp's actual buffer (a stride/dtype either side got wrong surfaces as a byte
    ///     divergence). The base's per-test leak gate proves each export is released. The sibling
    ///     <see cref="LapackDifferentialTests"/> covers the LU family the same way; offline value gates
    ///     (tolerance, no Python) live in <c>test/NumSharp.UnitTest/Backends/LapackSvd*Tests.cs</c>.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class SvdLstsqLiveParityTests : InteropTestBase
    {
        private static void RequireLapack()
        {
            if (!OpenBlasEngine.LapackAvailable)
                Assert.Inconclusive("OpenBLAS LAPACK backend not available on this host.");
        }

        /// <summary>
        ///     Assert NumSharp's <paramref name="nsResult"/> is byte-identical to numpy computing
        ///     <paramref name="npExpr"/> over the SAME exported operands.
        /// </summary>
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

        /// <summary>Byte-compare all three SVD outputs (U, S, Vh) against numpy over the same export.</summary>
        private void SvdSameBytes(NDArray a, bool full)
        {
            string fm = full ? "True" : "False";
            var (u, s, vh) = np.linalg.svd(a, full_matrices: full);
            SameBytes(u, $"np.linalg.svd(a, full_matrices={fm})[0]", ("a", a));
            SameBytes(s, $"np.linalg.svd(a, full_matrices={fm})[1]", ("a", a));
            SameBytes(vh, $"np.linalg.svd(a, full_matrices={fm})[2]", ("a", a));
        }

        // ---- reference operands (integer-valued so every dtype holds them) --------------------------

        private static NDArray W34() => np.array(new double[,]
        {
            { 4.0, 1.0, 2.0, 0.0 }, { 0.0, 3.0, 1.0, 2.0 }, { 1.0, 0.0, 5.0, 1.0 }
        });

        private static NDArray T43() => np.array(new double[,]
        {
            { 4.0, 1.0, 2.0 }, { 0.0, 3.0, 1.0 }, { 1.0, 0.0, 5.0 }, { 2.0, 1.0, 0.0 }
        });

        private static NDArray S33() => np.array(new double[,]
        {
            { 4.0, 1.0, 2.0 }, { 0.0, 3.0, 1.0 }, { 1.0, 0.0, 5.0 }
        });

        private static NDArray Cplx33() => np.array(new Complex[,]
        {
            { new Complex(4, 1), new Complex(1, 0), new Complex(2, -1) },
            { new Complex(0, 2), new Complex(3, 0), new Complex(1, 1) },
            { new Complex(1, 0), new Complex(0, -1), new Complex(5, 0) }
        });

        /// <summary>The five layouts a rectangular sweep walks, all sharing the same logical values.</summary>
        private static IEnumerable<NDArray> RectLayouts(NDArray baseC)
        {
            yield return baseC;                        // C-contiguous
            yield return np.asfortranarray(baseC);     // F-contiguous
            yield return baseC.T.copy().T;             // transposed view of a copy
            yield return baseC["::-1, ::-1"];          // negative strides, both axes
            long m = baseC.shape[0], n = baseC.shape[1];
            var big = np.zeros(new Shape(2 * m, 2 * n), baseC.typecode);
            big["::2, ::2"] = baseC;
            yield return big["::2, ::2"];              // strided
        }

        // ================================  svd  ====================================================

        [TestMethod]
        public void Svd_ByteExact_EveryDtype()
        {
            RequireLapack();
            foreach (var tc in new[]
            {
                NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64,
                NPTypeCode.Single, NPTypeCode.Double
            })
                SvdSameBytes(W34().astype(tc), full: false);

            // bool 0/1 matrix
            var boolm = np.array(new double[,] { { 1, 0, 1, 0 }, { 0, 1, 1, 1 }, { 1, 1, 0, 1 } })
                .astype(NPTypeCode.Boolean);
            SvdSameBytes(boolm, full: false);

            // complex
            SvdSameBytes(Cplx33(), full: false);
        }

        [TestMethod]
        public void Svd_ByteExact_EveryLayout()
        {
            RequireLapack();
            foreach (var a in RectLayouts(W34()))
                SvdSameBytes(a, full: false);
            // complex transposed/reversed views too
            SvdSameBytes(Cplx33().T, full: false);
            SvdSameBytes(Cplx33()["::-1, ::-1"], full: false);
        }

        [TestMethod]
        public void Svd_ByteExact_ShapesAndModes()
        {
            RequireLapack();
            foreach (var a in new[] { W34(), T43(), S33() })
            {
                SvdSameBytes(a, full: false); // reduced
                SvdSameBytes(a, full: true);  // full
                // compute_uv=False returns just S (not a tuple)
                SameBytes(np.linalg.svd(a, compute_uv: false).S,
                    "np.linalg.svd(a, compute_uv=False)", ("a", a));
            }
        }

        [TestMethod]
        public void Svd_ByteExact_Stacked_AndHigherRank()
        {
            RequireLapack();
            var stack = np.stack(new[] { W34(), W34() + 1.0, W34() * 2.0 });   // (3,3,4)
            SvdSameBytes(stack, full: false);
            SvdSameBytes(stack, full: true);

            var b4 = np.stack(new[] { stack, stack + 0.5 });                    // (2,3,3,4)
            SvdSameBytes(b4, full: false);
        }

        [TestMethod]
        public void Svd_ByteExact_Edge_OneByOne_And_Empty()
        {
            RequireLapack();
            SvdSameBytes(np.array(new double[,] { { 7.0 } }), full: true);

            // empty: (0,3) and (3,0), full_matrices fills identity for the non-empty factor
            SvdSameBytes(np.zeros(new Shape(0, 3)), full: true);
            SvdSameBytes(np.zeros(new Shape(3, 0)), full: true);
            SameBytes(np.linalg.svdvals(np.zeros(new Shape(0, 3))), "np.linalg.svdvals(a)",
                ("a", np.zeros(new Shape(0, 3))));
        }

        // ================================  svdvals  ================================================

        [TestMethod]
        public void Svdvals_ByteExact_Dtypes_Layouts_Stacked_Complex()
        {
            RequireLapack();
            foreach (var tc in new[] { NPTypeCode.Int32, NPTypeCode.Int64, NPTypeCode.Single, NPTypeCode.Double })
                SameBytes(np.linalg.svdvals(W34().astype(tc)), "np.linalg.svdvals(a)", ("a", W34().astype(tc)));
            foreach (var a in RectLayouts(T43()))
                SameBytes(np.linalg.svdvals(a), "np.linalg.svdvals(a)", ("a", a));
            var stack = np.stack(new[] { W34(), W34() + 1.0 });
            SameBytes(np.linalg.svdvals(stack), "np.linalg.svdvals(a)", ("a", stack));
            SameBytes(np.linalg.svdvals(Cplx33()), "np.linalg.svdvals(a)", ("a", Cplx33()));
        }

        // ================================  pinv  ===================================================

        [TestMethod]
        public void Pinv_ByteExact_Shapes_Dtypes_Layouts()
        {
            RequireLapack();
            foreach (var a in new[] { W34(), T43(), S33() })
                SameBytes(np.linalg.pinv(a), "np.linalg.pinv(a)", ("a", a));
            foreach (var tc in new[] { NPTypeCode.Int32, NPTypeCode.Single, NPTypeCode.Double })
                SameBytes(np.linalg.pinv(W34().astype(tc)), "np.linalg.pinv(a)", ("a", W34().astype(tc)));
            foreach (var a in RectLayouts(W34()))
                SameBytes(np.linalg.pinv(a), "np.linalg.pinv(a)", ("a", a));
        }

        [TestMethod]
        public void Pinv_ByteExact_Rcond_And_Stacked()
        {
            RequireLapack();
            SameBytes(np.linalg.pinv(W34(), rcond: 1e-10), "np.linalg.pinv(a, rcond=1e-10)", ("a", W34()));
            var stack = np.stack(new[] { W34(), W34() + 1.0 });
            SameBytes(np.linalg.pinv(stack), "np.linalg.pinv(a)", ("a", stack));
        }

        [TestMethod]
        public void Pinv_Complex_ValuesClose_ManagedMatmulNotByteExact()
        {
            // Complex pinv ends in a COMPLEX reconstruction matmul, and OpenBlasBackend serves
            // float32/float64 only — so NumSharp's complex matmul runs its own managed GEMM while numpy
            // calls zgemm. The two agree to ~1 ULP but NOT byte-for-byte (unlike the REAL pinv above,
            // whose matmul IS routed through OpenBLAS). The SVD factors feeding it are themselves
            // byte-exact — see Svd_ByteExact_EveryDtype's complex case — so this pins the one step that
            // isn't, with numpy's own allclose over NumSharp's exported result.
            RequireLapack();
            using (Gil())
            {
                using PyObject nsPinv = np.linalg.pinv(Cplx33()).ToNumpy();
                using PyObject va = Cplx33().ToNumpy();
                Python.np.truthy("np.allclose(p, np.linalg.pinv(a))", ("p", nsPinv), ("a", va))
                    .Should().BeTrue("complex pinv agrees to tolerance (managed complex matmul, not zgemm)");
            }
        }

        // ================================  matrix_rank  ============================================

        [TestMethod]
        public void MatrixRank_ByteExact_Default_Tol_Rtol_Stacked()
        {
            RequireLapack();
            var a = np.array(new double[,] { { 1.0, 2, 3 }, { 4, 5, 6 }, { 7, 8, 10 } });
            SameBytes(np.linalg.matrix_rank(a), "np.linalg.matrix_rank(a)", ("a", a));

            var deficient = np.array(new double[,] { { 1.0, 2 }, { 2, 4.0 } }); // rank 1
            SameBytes(np.linalg.matrix_rank(deficient), "np.linalg.matrix_rank(a)", ("a", deficient));
            SameBytes(np.linalg.matrix_rank(deficient, tol: 0.5), "np.linalg.matrix_rank(a, tol=0.5)", ("a", deficient));
            SameBytes(np.linalg.matrix_rank(deficient, rtol: 0.1), "np.linalg.matrix_rank(a, rtol=0.1)", ("a", deficient));

            // stacked → one rank per matrix (int64/intp array on both sides)
            var stack3 = np.stack(new[]
            {
                a, np.eye(3), np.array(new double[,] { { 1.0, 2, 3 }, { 2, 4, 6 }, { 0, 0, 1 } })
            });
            SameBytes(np.linalg.matrix_rank(stack3), "np.linalg.matrix_rank(a)", ("a", stack3));
        }

        // ================================  cond  ===================================================

        [TestMethod]
        public void Cond_ByteExact_EveryOrder()
        {
            RequireLapack();
            var a = S33();
            SameBytes(np.linalg.cond(a), "np.linalg.cond(a)", ("a", a));          // None (2-norm via SVD)
            SameBytes(np.linalg.cond(a, 2), "np.linalg.cond(a, 2)", ("a", a));
            SameBytes(np.linalg.cond(a, -2), "np.linalg.cond(a, -2)", ("a", a));
            SameBytes(np.linalg.cond(a, 1), "np.linalg.cond(a, 1)", ("a", a));
            SameBytes(np.linalg.cond(a, -1), "np.linalg.cond(a, -1)", ("a", a));
            SameBytes(np.linalg.cond(a, double.PositiveInfinity), "np.linalg.cond(a, np.inf)", ("a", a));
            SameBytes(np.linalg.cond(a, double.NegativeInfinity), "np.linalg.cond(a, -np.inf)", ("a", a));
            SameBytes(np.linalg.cond(a, "fro"), "np.linalg.cond(a, 'fro')", ("a", a));
        }

        [TestMethod]
        public void Cond_ByteExact_Complex_And_Stacked()
        {
            RequireLapack();
            SameBytes(np.linalg.cond(Cplx33()), "np.linalg.cond(a)", ("a", Cplx33()));
            SameBytes(np.linalg.cond(Cplx33(), 1), "np.linalg.cond(a, 1)", ("a", Cplx33()));
            var stack = np.stack(new[] { S33(), S33() + np.eye(3) });
            SameBytes(np.linalg.cond(stack), "np.linalg.cond(a)", ("a", stack));
            SameBytes(np.linalg.cond(stack, "fro"), "np.linalg.cond(a, 'fro')", ("a", stack));
        }

        // ================================  norm — the singular-value orders  =======================

        [TestMethod]
        public void Norm_ByteExact_SingularValueOrders()
        {
            RequireLapack();
            var a = S33();
            SameBytes(np.linalg.norm(a, 2), "np.linalg.norm(a, 2)", ("a", a));
            SameBytes(np.linalg.norm(a, -2), "np.linalg.norm(a, -2)", ("a", a));
            SameBytes(np.linalg.norm(a, "nuc"), "np.linalg.norm(a, 'nuc')", ("a", a));

            // non-square + layouts
            foreach (var m in RectLayouts(W34()))
            {
                SameBytes(np.linalg.norm(m, 2), "np.linalg.norm(a, 2)", ("a", m));
                SameBytes(np.linalg.norm(m, "nuc"), "np.linalg.norm(a, 'nuc')", ("a", m));
            }

            // stacked over an axis tuple, and keepdims
            var stack = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            SameBytes(np.linalg.norm(stack, 2, new[] { 1, 2 }), "np.linalg.norm(a, 2, axis=(1, 2))", ("a", stack));
            SameBytes(np.linalg.norm(stack, "nuc", new[] { 1, 2 }), "np.linalg.norm(a, 'nuc', axis=(1, 2))", ("a", stack));
            SameBytes(np.linalg.norm(stack, "nuc", new[] { 1, 2 }, keepdims: true),
                "np.linalg.norm(a, 'nuc', axis=(1, 2), keepdims=True)", ("a", stack));
        }

        // ================================  lstsq  ==================================================

        [TestMethod]
        public void Lstsq_ByteExact_ShapesAndB()
        {
            RequireLapack();
            var over = np.array(new double[,] { { 0.0, 1 }, { 1, 1 }, { 2, 1 }, { 3, 1 } }); // 4x2 overdetermined
            var y1 = np.array(new double[] { -1.0, 0.2, 0.9, 2.1 });                          // 1-D b
            var y2 = np.array(new double[,] { { -1.0, -2 }, { 0.2, 0.4 }, { 0.9, 1.8 }, { 2.1, 4.2 } }); // 2-D b
            LstsqSameBytes(over, y1);
            LstsqSameBytes(over, y2);

            var under = np.array(new double[,] { { 1.0, 2, 3 }, { 4, 5, 6 } });               // 2x3 underdetermined
            LstsqSameBytes(under, np.array(new double[] { 1.0, 2 }));

            var square = np.array(new double[,] { { 1.0, 2 }, { 3, 5 } });
            LstsqSameBytes(square, np.array(new double[] { 1.0, 2 }));

            // rank-deficient overdetermined
            var rankdef = np.array(new double[,] { { 1.0, 2 }, { 2, 4 }, { 3, 6 } });
            LstsqSameBytes(rankdef, np.array(new double[] { 1.0, 2, 3 }));

            // dtypes: int widens to float64, single stays single
            foreach (var tc in new[] { NPTypeCode.Int32, NPTypeCode.Single, NPTypeCode.Double })
                LstsqSameBytes(over.astype(tc), y1.astype(tc));
        }

        [TestMethod]
        public void Lstsq_ByteExact_Complex()
        {
            RequireLapack();
            var a = np.array(new Complex[,]
            {
                { new Complex(1, 0), new Complex(1, 0) }, { new Complex(1, 0), new Complex(0, 1) },
                { new Complex(1, 0), new Complex(2, 0) }, { new Complex(0, 0), new Complex(1, 0) }
            });
            var b = np.array(new Complex[] { new Complex(1, 1), new Complex(2, 0), new Complex(3, 0), new Complex(0, 1) });
            LstsqSameBytes(a, b);
        }

        private void LstsqSameBytes(NDArray a, NDArray b)
        {
            var (x, res, rank, s) = np.linalg.lstsq(a, b);
            SameBytes(x, "np.linalg.lstsq(a, b, rcond=None)[0]", ("a", a), ("b", b));
            SameBytes(res, "np.linalg.lstsq(a, b, rcond=None)[1]", ("a", a), ("b", b));
            SameBytes(rank, "np.linalg.lstsq(a, b, rcond=None)[2]", ("a", a), ("b", b));
            SameBytes(s, "np.linalg.lstsq(a, b, rcond=None)[3]", ("a", a), ("b", b));
        }

        // ================================  error parity  ===========================================

        [TestMethod]
        public void ErrorParity_BothStacksRaise()
        {
            RequireLapack();

            // sub-2-D operand
            var v = np.arange(3.0);
            new Action(() => np.linalg.svd(v)).Should().Throw<LinAlgError>()
                .WithMessage("1-dimensional array given. Array must be at least two-dimensional");
            RaisesInNumpy("np.linalg.svd(a)", ("a", v), "at least two-dimensional");

            // NaN does not converge on either side
            var nan = np.array(new double[,] { { double.NaN, 0 }, { 0, 1 } });
            new Action(() => np.linalg.svd(nan)).Should().Throw<LinAlgError>().WithMessage("SVD did not converge");
            RaisesInNumpy("np.linalg.svd(a)", ("a", nan), "converge");

            // lstsq incompatible dimensions
            var a = np.ones(new Shape(2, 3));
            var badb = np.ones(new Shape(4));
            new Action(() => np.linalg.lstsq(a, badb)).Should().Throw<LinAlgError>().WithMessage("Incompatible dimensions");
            RaisesInNumpy("np.linalg.lstsq(a, b, rcond=None)", ("a", a), ("b", badb), because: "Incompatible dimensions");
        }

        private void RaisesInNumpy(string npExpr, (string, NDArray) op, string because)
            => RaisesInNumpy(npExpr, because, op);

        private void RaisesInNumpy(string npExpr, (string, NDArray) op1, (string, NDArray) op2, string because)
            => RaisesInNumpy(npExpr, because, op1, op2);

        private void RaisesInNumpy(string npExpr, string because, params (string name, NDArray operand)[] ops)
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

                    var pyVars = vars;
                    new Action(() => { using var _ = Python.np.with(npExpr, pyVars); })
                        .Should().Throw<PythonException>().Which.Message.Should().Contain(because);
                }
                finally
                {
                    foreach (var e in exports) e?.Dispose();
                }
            }
        }
    }
}
