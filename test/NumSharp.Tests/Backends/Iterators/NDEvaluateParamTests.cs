using System;
using System.Numerics;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Backends.Iterators
{
    /// <summary>
    /// np.evaluate's hoisted parameters (ndexpr-evaluate.md 3.3, NDExpr.Params.cs): a 0-d input is
    /// copied into the kernel's aux block once per call and read from a local — one kernel for every
    /// value (unlike a literal, which bakes one kernel per value) and no per-element cost (unlike a
    /// stride-0 iterator operand). These tests pin the semantics that make that substitution invisible:
    /// same values, same dtypes, same NEP50 promotion, every dtype, both kernel paths, reductions,
    /// aliasing, and the guards around it.
    /// </summary>
    [TestClass]
    public class NDEvaluateParamTests
    {
        private static NDArray Vec(params double[] v) => np.array(v);

        private static void AssertSameArray(NDArray expected, NDArray actual, string what = null)
        {
            Assert.AreEqual(expected.typecode, actual.typecode, $"dtype {what}");
            CollectionAssert.AreEqual(expected.shape, actual.shape, $"shape {what}");
            Assert.IsTrue(np.array_equal(expected, actual), $"values differ {what}: expected {expected} but got {actual}");
        }

        [TestMethod]
        public void ZeroDInput_IsHoistedAsAParameter_NotStreamed()
        {
            var a = Vec(1, 2, 3, 4, 5, 6, 7, 8);
            var b = Vec(8, 7, 6, 5, 4, 3, 2, 1);
            var k = NDArray.Scalar(2.5);
            var root = (NDExpr)a * b + k;

            var program = root.GetProgram(null, out var inputs);
            CollectionAssert.AreEqual(new[] { false, false, true }, program.IsParam, "k is the parameter");
            Assert.AreEqual(1, program.ParamCount);
            Assert.AreEqual(3, inputs.Length, "the input list still carries every distinct array");
            CollectionAssert.AreEqual(new[] { a, b }, program.IteratorOperands(inputs), "the iterator streams only a and b");

            AssertSameArray(a * b + k, np.evaluate(root));
            AssertSameArray(a * b + 2.5, np.evaluate(root), "a strong 0-d float64 promotes like the weak literal here");
        }

        [TestMethod]
        public void OneKernel_ForEveryValue_AndTheStructuralCacheKeysOnTheMask()
        {
            var a = Vec(1, 2, 3, 4, 5, 6, 7, 8);
            var b = Vec(8, 7, 6, 5, 4, 3, 2, 1);

            AssertSameArray(a * b + 1.5, np.evaluate((NDExpr)a * b + NDArray.Scalar(1.5)));
            int kernels = GeneratedDelegates.InnerLoopCount;
            var first = ((NDExpr)a * b + NDArray.Scalar(1.5)).GetProgram(null);
            foreach (var k in new[] { 2.5, -3.0, 1e6, 0.0, double.PositiveInfinity })
            {
                var ks = NDArray.Scalar(k);
                var root = (NDExpr)a * b + ks;                              // a NEW root and a NEW 0-d array per value
                Assert.AreSame(first, root.GetProgram(null), $"k={k}: one program serves every value");
                AssertSameArray(a * b + ks, np.evaluate(root));
                Assert.AreEqual(kernels, GeneratedDelegates.InnerLoopCount, $"k={k} must not compile a new kernel");
            }

            // The same tree over a 1-D `k` is another program (the kernel streams three operands).
            var k1 = Vec(1, 1, 1, 1, 1, 1, 1, 1);
            var streamed = ((NDExpr)a * b + k1).GetProgram(null);
            Assert.AreNotSame(first, streamed);
            Assert.IsNull(streamed.IsParam);
            AssertSameArray(a * b + k1, np.evaluate((NDExpr)a * b + k1));
        }

        [TestMethod]
        public void EveryDtype_VectorAndScalarPaths_MatchTheUnfusedChain()
        {
            var codes = new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
                NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
            };
            foreach (var tc in codes)
            {
                NDArray a = tc == NPTypeCode.Boolean
                    ? np.array(new[] { true, false, true, true, false, false, true, false, true, true, false, true, true, false, true, false, true, false })
                    : np.arange(1, 19).astype(tc);
                NDArray k = tc == NPTypeCode.Boolean ? NDArray.Scalar(true) : NDArray.Scalar(3, tc);
                Assert.AreEqual(0, k.ndim);
                Assert.AreEqual(tc, k.typecode);

                var expected = a + k;                                        // the engine's own 0-d operand path
                var root = (NDExpr)a + k;
                var program = root.GetProgram(null);
                CollectionAssert.AreEqual(new[] { false, true }, program.IsParam, tc.ToString());

                AssertSameArray(expected, np.evaluate(root), $"{tc} vector/plan path");

                NDExpr.ForceScalar = true;
                try { AssertSameArray(expected, np.evaluate((NDExpr)a + k), $"{tc} scalar path"); }
                finally { NDExpr.ForceScalar = false; }
            }
        }

        [TestMethod]
        public void BoolParameter_IsALaneMaskInWMode_AndABoolInByteMode()
        {
            var a = Vec(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17);
            var b = Vec(-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17);
            var yes = NDArray.Scalar(true);
            var no = NDArray.Scalar(false);

            // W-mode (float64 lanes): the bool parameter is a constant lane mask feeding the select.
            AssertSameArray(a, np.evaluate(NDExpr.Where(NDExpr.Arr(yes), NDExpr.Arr(a), NDExpr.Arr(b))));
            AssertSameArray(b, np.evaluate(NDExpr.Where(NDExpr.Arr(no), NDExpr.Arr(a), NDExpr.Arr(b))));
            // …and meeting arithmetic it is exact 1/0 (NumPy: True * a == a).
            AssertSameArray(a * 1.0, np.evaluate((NDExpr)a * yes));
            AssertSameArray(a * 0.0, np.evaluate((NDExpr)a * no));

            // Byte mode (every input bool): the parameter is a 0/1 byte vector.
            var m = np.array(new[] { true, false, true, true, false, false, true, false, true, true, false, true, true, false, true, false, true });
            AssertSameArray(m & yes, np.evaluate((NDExpr)m & yes));
            AssertSameArray(m & no, np.evaluate((NDExpr)m & no));
            AssertSameArray(m | no, np.evaluate((NDExpr)m | no));
        }

        [TestMethod]
        public void Reductions_FlatAndAxis_ReadTheParameter()
        {
            var a = Vec(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            var a2 = a.reshape(3, 4);
            var k = NDArray.Scalar(0.5);

            Assert.AreEqual(np.sum(a * k).GetDouble(0), np.evaluate(NDExpr.Sum((NDExpr)a * k)).GetDouble(0));
            Assert.AreEqual(np.max(a * k).GetDouble(0), np.evaluate(NDExpr.Max((NDExpr)a * k)).GetDouble(0));
            Assert.AreEqual(np.mean(a * k).GetDouble(0), np.evaluate(NDExpr.Mean((NDExpr)a * k)).GetDouble(0));
            AssertSameArray(np.sum(a2 * k, 0), np.evaluate(NDExpr.Sum((NDExpr)a2 * k, axis: 0)));
            AssertSameArray(np.sum(a2 * k, 1), np.evaluate(NDExpr.Sum((NDExpr)a2 * k, axis: 1)));
            AssertSameArray(np.max(a2 * k, 1, keepdims: true), np.evaluate(NDExpr.Max((NDExpr)a2 * k, axis: 1, keepdims: true)));

            var program = NDExpr.Sum((NDExpr)a2 * k, axis: 0).GetProgram(null);
            Assert.AreEqual(1, program.ParamCount);
            Assert.IsNotNull(program.Reduce);
        }

        [TestMethod]
        public void MixedDtype_ParameterPromotesAsTheStrongScalarItIs()
        {
            var ai = np.array(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            var kf = NDArray.Scalar(2.0);
            var kl = NDArray.Scalar(2L);

            var rf = np.evaluate((NDExpr)ai * kf);
            Assert.AreEqual(NPTypeCode.Double, rf.typecode, "int32 * 0-d float64 → float64 (np.float64(k) is strong)");
            AssertSameArray(ai * kf, rf);

            // NumPy: int32 array * np.int64(2) → int64 (a 0-d array is a strong scalar). The unfused
            // NumSharp operator still treats the 0-d int64 as weak (int32 result) — a known engine
            // inconsistency — so the expectation is spelled with an explicit int64 array.
            var rl = np.evaluate((NDExpr)ai * kl);
            Assert.AreEqual(NPTypeCode.Int64, rl.typecode, "int32 * 0-d int64 → int64");
            AssertSameArray(ai.astype(NPTypeCode.Int64) * 2L, rl);

            // A mixed-dtype tree takes the scalar kernel path; the parameter is still hoisted there.
            var program = ((NDExpr)ai * kf).GetProgram(null);
            Assert.AreEqual(1, program.ParamCount);
        }

        [TestMethod]
        public void ParameterAliasingTheOutput_ReadsItsOriginalValue()
        {
            var x = Vec(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
            var k = x["3"];                                                  // a 0-d VIEW into x
            Assert.AreEqual(0, k.ndim);
            var expected = x * 3.0;

            var r = np.evaluate((NDExpr)x * k, @out: x);
            Assert.AreSame(x, r);
            // The parameter was read before the pass overwrote x[3] — the value NumPy's COPY_IF_OVERLAP gives.
            AssertSameArray(expected, x);
        }

        [TestMethod]
        public void AllZeroDInputs_StayOnTheIterator()
        {
            var s1 = NDArray.Scalar(2.0);
            var s2 = NDArray.Scalar(3.0);
            var root = (NDExpr)s1 * s2 + 1;

            var program = root.GetProgram(null);
            Assert.IsNull(program.IsParam, "a kernel needs at least one streamed operand");
            var r = np.evaluate(root);
            Assert.AreEqual(0, r.ndim);
            Assert.AreEqual(7.0, r.GetDouble(0));

            Assert.AreEqual(6.0, np.evaluate(NDExpr.Sum((NDExpr)s1 * s2)).GetDouble(0));
        }

        [TestMethod]
        public void ParameterResizedInPlace_IsReResolved_AndACompiledHandleRefuses()
        {
            var a = Vec(1, 2, 3, 4, 5, 6, 7, 8);
            var k = NDArray.Scalar(2.0);
            var root = (NDExpr)a * k;
            var handle = root.Compile();

            AssertSameArray(a * 2.0, np.evaluate(root));
            AssertSameArray(a * 2.0, handle.Evaluate());
            var hoisted = root.GetProgram(null);
            Assert.AreEqual(1, hoisted.ParamCount);

            // The 0-d array becomes a (8,) array in place: the hoisted read would be wrong, so the root
            // re-resolves to a streaming program and the pinned handle refuses.
            k.resize(8L);
            Assert.AreEqual(1, k.ndim);
            var again = root.GetProgram(null);
            Assert.AreNotSame(hoisted, again);
            Assert.IsNull(again.IsParam);
            AssertSameArray(a * k, np.evaluate(root));
            var ex = Assert.ThrowsException<InvalidOperationException>(() => handle.Evaluate());
            StringAssert.Contains(ex.Message, "no longer 0-d");
        }

        [TestMethod]
        public void PositionalForms_HoistOnlyWhenTheCallResolvesShapes()
        {
            var a = Vec(1, 2, 3, 4, 5, 6, 7, 8);
            var b = Vec(8, 7, 6, 5, 4, 3, 2, 1);
            var expr = NDExpr.Input(0) * NDExpr.Input(1) + NDExpr.Input(2);

            // np.evaluate(expr, operands) resolves per call → the 0-d operand is a parameter.
            var k = NDArray.Scalar(1.5);
            AssertSameArray(a * b + 1.5, np.evaluate(expr, new[] { a, b, k }));
            var perCall = expr.GetProgram(new[] { a, b, k });
            Assert.AreEqual(1, perCall.ParamCount);

            // A dtype-pinned handle knows no shapes: every operand streams, a 0-d one as a stride-0 operand.
            var handle = expr.Compile(NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double);
            AssertSameArray(a * b + 1.5, handle.Evaluate(new[] { a, b, k }));
            AssertSameArray(a * b + a, handle.Evaluate(new[] { a, b, a }));
        }

        [TestMethod]
        public void ComplexAndDecimalParameters_UseTheFull16ByteSlot()
        {
            var z = np.array(new[] { new Complex(1, 2), new Complex(-3, 0.5), new Complex(0, -1) });
            var kz = NDArray.Scalar(new Complex(2, -1));
            AssertSameArray(z * kz, np.evaluate((NDExpr)z * kz));
            AssertSameArray(z + kz, np.evaluate((NDExpr)z + kz));

            var d = np.arange(1, 9).astype(NPTypeCode.Decimal);
            var kd = NDArray.Scalar(1.25m);
            AssertSameArray(d * kd, np.evaluate((NDExpr)d * kd));
            AssertSameArray(d / kd, np.evaluate((NDExpr)d / kd));

            // Two parameters: each in its own slot, in input order.
            var a = Vec(1, 2, 3, 4, 5, 6, 7, 8);
            var p = NDArray.Scalar(10.0);
            var q = NDArray.Scalar(0.25);
            var program = ((NDExpr)a * p + q).GetProgram(null);
            Assert.AreEqual(2, program.ParamCount);
            AssertSameArray(a * p + q, np.evaluate((NDExpr)a * p + q));
            AssertSameArray(a * q + p, np.evaluate((NDExpr)a * q + p));
        }
    }
}
