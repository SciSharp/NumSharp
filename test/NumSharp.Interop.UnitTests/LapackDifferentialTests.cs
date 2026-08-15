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
    ///     The strongest possible parity gate for the OpenBLAS LU factorisations
    ///     (<c>np.linalg.solve/inv/det/slogdet</c> and the functions composed on them —
    ///     <c>matrix_power</c> with a negative exponent, <c>tensorinv</c>, <c>tensorsolve</c>):
    ///     every case computes the op TWICE over the very same bytes — once in NumSharp through the
    ///     bundled scipy-openblas backend, once in the LIVE numpy loaded in this process — and asserts
    ///     they agree.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     numpy's <c>np.linalg</c> and NumSharp's backend call the SAME scipy-openblas build (numpy
    ///     pins it, this package bundles the byte-identical copy), so for a single-threaded factorisation
    ///     the result is a deterministic function of the input and the two agree BYTE-FOR-BYTE. That is
    ///     the <see cref="SameBytes"/> assertion, used for the small/curated matrices that exercise every
    ///     code path (dtype, layout, batch, edge case). Larger matrices, where numpy may dispatch a
    ///     multi-threaded blocked LU while this backend is pinned to one thread, use
    ///     <see cref="AllClose"/> instead — the sum reordering is within tolerance, not a defect.
    ///     </para>
    ///     <para>
    ///     Every input is handed to numpy through the interop's zero-copy export, so numpy computes over
    ///     NumSharp's actual buffer — a stride or dtype either side got wrong shows up as a value
    ///     divergence here. The base class's per-test leak gate additionally proves each export is
    ///     released. Self-skips (Inconclusive) when numpy or the OpenBLAS LAPACK is unavailable.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class LapackDifferentialTests : InteropTestBase
    {
        private static void RequireLapack()
        {
            if (!OpenBlasEngine.LapackAvailable)
                Assert.Inconclusive("OpenBLAS LAPACK backend not available on this host.");
        }

        // ---- differential helpers -----------------------------------------------------------------

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

        /// <summary>The invertible reference matrix (integer-valued so every dtype can hold it).</summary>
        private static NDArray Ref3() => np.array(new double[,] { { 4.0, 1.0, 2.0 }, { 0.0, 3.0, 1.0 }, { 1.0, 0.0, 5.0 } });

        /// <summary>The six square layouts every layout-sweep walks, all sharing the same logical values.</summary>
        private static IEnumerable<(NDArray a, string label)> SquareLayouts(NDArray baseC)
        {
            yield return (baseC, "C");
            yield return (np.asfortranarray(baseC), "F");
            yield return (baseC.T.copy().T, "T-of-T");           // exercises a transposed view
            yield return (baseC["::-1, ::-1"], "reversed");        // negative strides both axes
            // strided: embed in a 2x-spaced grid and take every other row/col back out (square, strided)
            long n = baseC.shape[0];
            var big = np.zeros(new Shape(2 * n, 2 * n), baseC.typecode);
            big["::2, ::2"] = baseC;
            yield return (big["::2, ::2"], "strided");
            // offset slice of a larger contiguous matrix
            var padded = np.zeros(new Shape(n + 2, n + 2), baseC.typecode);
            padded[$"1:{1 + n}, 1:{1 + n}"] = baseC;
            yield return (padded[$"1:{1 + n}, 1:{1 + n}"], "offset");
        }

        // ==============================  backend is the project default  ===========================

        [TestMethod]
        public void Backend_IsOpenBlasDefault_WithLapack()
        {
            Assert.IsTrue(OpenBlasEngine.Enabled, "OpenBLAS is the default backend in this interop project");
            RequireLapack();
            StringAssert.Contains(OpenBlasEngine.Info, "symbols");
        }

        // ==============================  inv  ======================================================

        [TestMethod]
        public void Inv_ByteExact_EveryDtype()
        {
            RequireLapack();
            // bool + every integer widen to float64; float32/float64/complex compute at their width.
            var dtypes = new[]
            {
                NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64,
                NPTypeCode.Single, NPTypeCode.Double
            };
            foreach (var tc in dtypes)
            {
                var a = Ref3().astype(tc);
                SameBytes(np.linalg.inv(a), "np.linalg.inv(a)", ("a", a));
            }

            // bool: [[4,1,2],..] casts to all-True (singular), so use an identity.
            var boolId = np.eye(3).astype(NPTypeCode.Boolean);
            SameBytes(np.linalg.inv(boolId), "np.linalg.inv(a)", ("a", boolId));

            // complex
            var z = np.array(new Complex[,]
            {
                { new Complex(4, 1), new Complex(1, 0), new Complex(2, -1) },
                { new Complex(0, 2), new Complex(3, 0), new Complex(1, 1) },
                { new Complex(1, 0), new Complex(0, -1), new Complex(5, 0) }
            });
            SameBytes(np.linalg.inv(z), "np.linalg.inv(a)", ("a", z));
        }

        [TestMethod]
        public void Inv_ByteExact_EveryLayout()
        {
            RequireLapack();
            foreach (var (a, label) in SquareLayouts(Ref3()))
                SameBytes(np.linalg.inv(a), "np.linalg.inv(a)", ("a", a));

            // complex layouts too (transposed + reversed)
            var z = np.array(new Complex[,] { { new Complex(2, 1), 1 }, { 3, new Complex(4, -1) } });
            SameBytes(np.linalg.inv(z.T), "np.linalg.inv(a)", ("a", z.T));
            SameBytes(np.linalg.inv(z["::-1, ::-1"]), "np.linalg.inv(a)", ("a", z["::-1, ::-1"]));
        }

        [TestMethod]
        public void Inv_ByteExact_Stacked_AndHigherRank()
        {
            RequireLapack();
            var stack = np.stack(new[] { Ref3(), Ref3() + np.eye(3), Ref3() * 2.0 - np.eye(3) });
            SameBytes(np.linalg.inv(stack), "np.linalg.inv(a)", ("a", stack));

            // 4-D batch
            var b4 = np.stack(new[] { stack, stack + np.eye(3) });
            SameBytes(np.linalg.inv(b4), "np.linalg.inv(a)", ("a", b4));
        }

        [TestMethod]
        public void Inv_ThenMatmul_IsByteExact_EndToEnd()
        {
            // inv → matmul, computed in both stacks over the same export: both call the same BLAS
            // single-threaded, so `a @ inv(a)` is byte-identical (it is NOT exactly I — floating error —
            // but the two stacks round it the same way, which is the parity claim).
            RequireLapack();
            var a = Ref3();
            SameBytes(np.matmul(a, np.linalg.inv(a)), "a @ np.linalg.inv(a)", ("a", a));
        }

        // ==============================  solve  ====================================================

        [TestMethod]
        public void Solve_ByteExact_VectorRhs_Dtypes()
        {
            RequireLapack();
            var b = np.array(new double[] { 1.0, -2.0, 3.0 });
            foreach (var tc in new[] { NPTypeCode.Int32, NPTypeCode.Int64, NPTypeCode.Single, NPTypeCode.Double })
            {
                var a = Ref3().astype(tc);
                var bb = b.astype(tc);
                SameBytes(np.linalg.solve(a, bb), "np.linalg.solve(a, b)", ("a", a), ("b", bb));
            }
        }

        [TestMethod]
        public void Solve_ByteExact_MatrixRhs_AndLayouts()
        {
            RequireLapack();
            var a = Ref3();
            var bmat = np.array(new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 }, { 2.0, -1.0 } }); // (3,2)
            SameBytes(np.linalg.solve(a, bmat), "np.linalg.solve(a, b)", ("a", a), ("b", bmat));

            // non-contiguous a and b
            SameBytes(np.linalg.solve(a.T.copy().T, bmat["::-1"]),
                "np.linalg.solve(a, b)", ("a", a.T.copy().T), ("b", bmat["::-1"]));
        }

        [TestMethod]
        public void Solve_ByteExact_BatchBroadcast()
        {
            RequireLapack();
            var sa = np.stack(new[] { Ref3(), Ref3() + np.eye(3) });   // (2,3,3)
            // vector rhs broadcasts across the batch
            var v = np.array(new double[] { 1.0, 2.0, 3.0 });
            SameBytes(np.linalg.solve(sa, v), "np.linalg.solve(a, b)", ("a", sa), ("b", v));

            // matrix rhs, a broadcasts across b's batch
            var sb = np.stack(new[]
            {
                np.array(new double[,] { { 1.0 }, { 0.0 }, { 0.0 } }),
                np.array(new double[,] { { 0.0 }, { 1.0 }, { 2.0 } })
            });   // (2,3,1)
            SameBytes(np.linalg.solve(np.expand_dims(Ref3(), 0), sb),
                "np.linalg.solve(a, b)", ("a", np.expand_dims(Ref3(), 0)), ("b", sb));
        }

        [TestMethod]
        public void Solve_Complex_ByteExact()
        {
            RequireLapack();
            var z = np.array(new Complex[,] { { new Complex(2, 1), 1 }, { 3, new Complex(4, -1) } });
            var b = np.array(new Complex[] { new Complex(1, 1), new Complex(0, -2) });
            SameBytes(np.linalg.solve(z, b), "np.linalg.solve(a, b)", ("a", z), ("b", b));
        }

        // ==============================  det / slogdet  ============================================

        [TestMethod]
        public void Det_ByteExact_DtypesAndLayouts()
        {
            RequireLapack();
            foreach (var tc in new[] { NPTypeCode.Int32, NPTypeCode.Int64, NPTypeCode.Single, NPTypeCode.Double })
            {
                var a = Ref3().astype(tc);
                SameBytes(np.linalg.det(a), "np.linalg.det(a)", ("a", a));
            }

            foreach (var (a, label) in SquareLayouts(Ref3()))
                SameBytes(np.linalg.det(a), "np.linalg.det(a)", ("a", a));

            // complex + stacked
            var z = np.array(new Complex[,] { { new Complex(1, 1), 2 }, { 3, new Complex(4, -1) } });
            SameBytes(np.linalg.det(z), "np.linalg.det(a)", ("a", z));
            var stack = np.stack(new[] { Ref3(), Ref3() + np.eye(3) });
            SameBytes(np.linalg.det(stack), "np.linalg.det(a)", ("a", stack));
        }

        [TestMethod]
        public void Slogdet_ByteExact_SignAndLogabsdet()
        {
            RequireLapack();
            foreach (var (a, label) in SquareLayouts(Ref3()))
            {
                var (sign, logabsdet) = np.linalg.slogdet(a);
                SameBytes(sign, "np.linalg.slogdet(a)[0]", ("a", a));
                SameBytes(logabsdet, "np.linalg.slogdet(a)[1]", ("a", a));
            }

            // complex sign is unit-modulus; logabsdet is real (float64)
            var z = np.array(new Complex[,] { { new Complex(1, 1), 2 }, { 3, new Complex(4, -1) } });
            var (zs, zl) = np.linalg.slogdet(z);
            SameBytes(zs, "np.linalg.slogdet(a)[0]", ("a", z));
            SameBytes(zl, "np.linalg.slogdet(a)[1]", ("a", z));

            // stacked, with one singular member → per-element (0, -inf) on both sides
            var stack = np.stack(new[] { Ref3(), np.array(new double[,] { { 1.0, 2.0, 3.0 }, { 2.0, 4.0, 6.0 }, { 0.0, 0.0, 1.0 } }) });
            var (ss, sl) = np.linalg.slogdet(stack);
            SameBytes(ss, "np.linalg.slogdet(a)[0]", ("a", stack));
            SameBytes(sl, "np.linalg.slogdet(a)[1]", ("a", stack));
        }

        [TestMethod]
        public void Det_Singular_IsZero_OnBothSides()
        {
            RequireLapack();
            var s = np.array(new double[,] { { 1.0, 2.0 }, { 2.0, 4.0 } });
            SameBytes(np.linalg.det(s), "np.linalg.det(a)", ("a", s));   // both 0.0
        }

        // ==============================  matrix_power / tensorinv / tensorsolve  ===================

        [TestMethod]
        public void MatrixPower_ByteExact_Range()
        {
            RequireLapack();
            var a = Ref3();
            foreach (int n in new[] { 0, 1, 2, 3, 5, 8, -1, -2, -3 })
                SameBytes(np.linalg.matrix_power(a, n), $"np.linalg.matrix_power(a, {n})", ("a", a));

            // int operand: n>=0 stays int (byte-exact int64), n<0 floats
            var ai = np.array(new int[,] { { 2, 1, 0 }, { 0, 2, 1 }, { 1, 0, 2 } });
            SameBytes(np.linalg.matrix_power(ai, 3), "np.linalg.matrix_power(a, 3)", ("a", ai));
            SameBytes(np.linalg.matrix_power(ai, -1), "np.linalg.matrix_power(a, -1)", ("a", ai));
        }

        [TestMethod]
        public void Tensorinv_ByteExact_IndVariations()
        {
            RequireLapack();
            var t2 = tensor(24).reshape(4, 6, 8, 3);
            SameBytes(np.linalg.tensorinv(t2, 2), "np.linalg.tensorinv(a, 2)", ("a", t2));
            var t1 = tensor(24).reshape(24, 8, 3);
            SameBytes(np.linalg.tensorinv(t1, 1), "np.linalg.tensorinv(a, 1)", ("a", t1));
            var t3 = tensor(24).reshape(2, 3, 4, 24);
            SameBytes(np.linalg.tensorinv(t3, 3), "np.linalg.tensorinv(a, 3)", ("a", t3));
        }

        [TestMethod]
        public void Tensorsolve_ByteExact_AxesVariations()
        {
            RequireLapack();
            var a = tensor(24).reshape(6, 4, 2, 3, 4);
            var b = (np.arange(24).astype(np.float64) % np.array(5.0)).reshape(6, 4);
            SameBytes(np.linalg.tensorsolve(a, b), "np.linalg.tensorsolve(a, b)", ("a", a), ("b", b));

            var a3 = tensor(6).reshape(2, 3, 2, 3);
            var b3 = (np.arange(6).astype(np.float64) % np.array(5.0)).reshape(2, 3);
            SameBytes(np.linalg.tensorsolve(a3, b3, new[] { 0, 1 }),
                "np.linalg.tensorsolve(a, b, axes=(0, 1))", ("a", a3), ("b", b3));
        }

        private static NDArray tensor(int sz)
            => (np.arange(sz * sz).astype(np.float64) % np.array(7.0)).reshape(sz, sz) + np.eye(sz) * np.array(30.0);

        // ==============================  edge cases / error parity  =================================

        [TestMethod]
        public void Singular_Inv_And_Solve_RaiseOnBothSides()
        {
            RequireLapack();
            var s = np.array(new double[,] { { 1.0, 2.0 }, { 2.0, 4.0 } });

            // NumSharp raises LinAlgError("Singular matrix")
            new Action(() => np.linalg.inv(s)).Should().Throw<LinAlgError>().WithMessage("Singular matrix");
            new Action(() => np.linalg.solve(s, np.array(new double[] { 1.0, 2.0 })))
                .Should().Throw<LinAlgError>().WithMessage("Singular matrix");

            // numpy raises its own LinAlgError over the same export
            using (Gil())
            {
                using PyObject va = s.ToNumpy();
                new Action(() => { using var _ = Python.np.with("np.linalg.inv(a)", ("a", va)); })
                    .Should().Throw<PythonException>().Which.Message.Should().Contain("Singular");
            }
        }

        [TestMethod]
        public void Empty_And_OneByOne_MatchNumpy()
        {
            RequireLapack();
            SameBytes(np.linalg.inv(np.zeros(new Shape(0, 0))), "np.linalg.inv(a)", ("a", np.zeros(new Shape(0, 0))));
            SameBytes(np.linalg.det(np.zeros(new Shape(0, 0))), "np.linalg.det(a)", ("a", np.zeros(new Shape(0, 0))));

            var one = np.array(new double[,] { { 4.0 } });
            SameBytes(np.linalg.inv(one), "np.linalg.inv(a)", ("a", one));
            SameBytes(np.linalg.det(one), "np.linalg.det(a)", ("a", one));

            // zero-sized batch and a stack of 0x0
            SameBytes(np.linalg.det(np.zeros(new Shape(0, 3, 3))), "np.linalg.det(a)", ("a", np.zeros(new Shape(0, 3, 3))));
            SameBytes(np.linalg.det(np.zeros(new Shape(2, 0, 0))), "np.linalg.det(a)", ("a", np.zeros(new Shape(2, 0, 0))));
        }
    }
}
