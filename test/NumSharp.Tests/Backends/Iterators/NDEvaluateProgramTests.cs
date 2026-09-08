using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Backends.Iterators
{
    /// <summary>
    /// np.evaluate's compiled program (ndexpr-evaluate.md Phase 3): the per-root cache behind
    /// <c>np.evaluate</c>, the explicit <see cref="CompiledExpression"/> handle, the single-allocation
    /// N-ary broadcast of the inputs (with NumPy's every-operand error text) and the 0-d "parameter"
    /// operand form that serves a constant sweep with ONE kernel.
    /// </summary>
    [TestClass]
    public class NDEvaluateProgramTests
    {
        private static NDArray Vec(params double[] v) => np.array(v);

        private static void AssertSameArray(NDArray expected, NDArray actual)
        {
            Assert.AreEqual(expected.typecode, actual.typecode, "dtype");
            CollectionAssert.AreEqual(expected.shape, actual.shape, "shape");
            Assert.IsTrue(np.array_equal(expected, actual), $"values differ: expected {expected} but got {actual}");
        }

        // =====================================================================
        // The per-root program cache behind np.evaluate
        // =====================================================================

        [TestMethod]
        public void Program_Embedded_IsCachedPerRootInstance_AndSharesTheKernel()
        {
            var a = Vec(1, 2, 3, 4);
            var b = Vec(5, 6, 7, 8);
            var c = Vec(9, 10, 11, 12);
            var root = (NDExpr)a * b + c;

            var p1 = root.GetProgram(null);
            var p2 = root.GetProgram(null);
            Assert.AreSame(p1, p2, "the second call must reuse the root's program");
            Assert.IsNotNull(p1.EmbeddedOperands);
            Assert.AreEqual(3, p1.EmbeddedOperands.Length);
            Assert.AreEqual(NPTypeCode.Double, p1.ResultType);

            // A NEW tree instance over the same arrays gets its own program (the cache is per root),
            // but the kernel underneath is the same cached delegate (same structure + dtype signature).
            var other = (NDExpr)a * b + c;
            var p3 = other.GetProgram(null);
            Assert.AreNotSame(p1, p3);
            Assert.AreSame(p1.Kernel, p3.Kernel);
        }

        [TestMethod]
        public void Program_Reused_ReadsTheOperandsCurrentContents()
        {
            var a = Vec(1, 2, 3, 4);
            var b = Vec(10, 10, 10, 10);
            var root = (NDExpr)a * b;

            AssertSameArray(Vec(10, 20, 30, 40), np.evaluate(root));

            // The program binds the array INSTANCES, never a snapshot of their data.
            a.SetAtIndex(100.0, 0);
            AssertSameArray(Vec(1000, 20, 30, 40), np.evaluate(root));
        }

        [TestMethod]
        public void Program_Positional_RecompilesOnDtypeSignatureChange_AndReusesOtherwise()
        {
            var root = NDExpr.Input(0) * NDExpr.Input(1);
            var af = Vec(1, 2, 3);
            var bf = Vec(2, 2, 2);
            var ai = np.array(new int[] { 1, 2, 3 });
            var bi = np.array(new int[] { 2, 2, 2 });

            var rf = np.evaluate(root, new[] { af, bf });
            var pf = root.GetProgram(new[] { af, bf });
            Assert.AreEqual(NPTypeCode.Double, rf.typecode);
            AssertSameArray(af * bf, rf);

            // int32 operands: a different signature → a fresh program with the int32 loop.
            var ri = np.evaluate(root, new[] { ai, bi });
            var pi = root.GetProgram(new[] { ai, bi });
            Assert.AreEqual(NPTypeCode.Int32, ri.typecode);
            AssertSameArray(ai * bi, ri);
            Assert.AreNotSame(pf, pi);
            Assert.IsNull(pi.EmbeddedOperands);

            // Back to float64: recompiled (the slot holds one program), still correct.
            var rf2 = np.evaluate(root, new[] { af, bf });
            AssertSameArray(af * bf, rf2);
            Assert.AreSame(root.GetProgram(new[] { af, bf }), root.GetProgram(new[] { af, bf }));
        }

        [TestMethod]
        public void Program_NotReused_AcrossTheForceScalarHook()
        {
            var a = Vec(1, 2, 3, 4, 5, 6, 7, 8, 9);
            var root = NDExpr.Greater((NDExpr)a * 2.0, 5.0);

            var vector = root.GetProgram(null);
            Assert.IsFalse(vector.ForcedScalar);
            var rv = np.evaluate(root);

            NDExprProgram scalar;
            NDArray rs;
            NDExpr.ForceScalar = true;
            try
            {
                scalar = root.GetProgram(null);
                rs = np.evaluate(root);
            }
            finally { NDExpr.ForceScalar = false; }

            Assert.AreNotSame(vector, scalar, "a program compiled under the scalar hook must not serve the vector mode");
            Assert.IsTrue(scalar.ForcedScalar);
            AssertSameArray(rv, rs);

            var again = root.GetProgram(null);
            Assert.AreNotSame(scalar, again);
            Assert.IsFalse(again.ForcedScalar);
        }

        // =====================================================================
        // CompiledExpression — the explicit handle
        // =====================================================================

        [TestMethod]
        public void Compile_Embedded_HandleDescribesAndEvaluatesTheTree()
        {
            var a = Vec(1, 2, 3, 4);
            var b = Vec(5, 6, 7, 8);
            var c = Vec(9, 10, 11, 12);

            var h = ((NDExpr)a * b + c).Compile();
            Assert.IsFalse(h.IsPositional);
            Assert.IsFalse(h.IsReduction);
            Assert.AreEqual(3, h.OperandCount);
            Assert.AreEqual(NPTypeCode.Double, h.ResultType);
            CollectionAssert.AreEqual(new[] { NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double }, h.InputTypes);
            Assert.IsNotNull(h.Expression);

            AssertSameArray(a * b + c, h.Evaluate());

            var o = np.empty_like(a);
            var r = h.Evaluate(o);
            Assert.AreSame(o, r, "out= returns the same instance");
            AssertSameArray(a * b + c, o);

            // The handle reads the arrays' current contents, like np.evaluate on the same root.
            a.SetAtIndex(0.0, 0);
            AssertSameArray(a * b + c, h.Evaluate());
        }

        [TestMethod]
        public void Compile_Embedded_Reduction()
        {
            var a = Vec(1, 2, 3, 4, 5, 6, 7, 8);
            var b = Vec(2, 2, 2, 2, 2, 2, 2, 2);

            var h = NDExpr.Sum((NDExpr)a * b).Compile();
            Assert.IsTrue(h.IsReduction);
            Assert.AreEqual(NPTypeCode.Double, h.ResultType);

            var r = h.Evaluate();
            Assert.AreEqual(0, r.ndim);
            Assert.AreEqual(72.0, r.GetDouble(0));
        }

        [TestMethod]
        public void Compile_Positional_PinsTheDtypeSignature()
        {
            var a = Vec(1, 2, 3, 4);
            var b = Vec(5, 6, 7, 8);
            var c = Vec(9, 10, 11, 12);
            var ai = np.array(new int[] { 1, 2, 3, 4 });

            var h = (NDExpr.Input(0) * NDExpr.Input(1) + NDExpr.Input(2))
                .Compile(NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double);
            Assert.IsTrue(h.IsPositional);
            Assert.AreEqual(3, h.OperandCount);
            Assert.AreEqual(NPTypeCode.Double, h.ResultType);

            AssertSameArray(a * b + c, h.Evaluate(new[] { a, b, c }));

            var ex = Assert.ThrowsException<TypeError>(() => h.Evaluate(new[] { ai, b, c }));
            StringAssert.Contains(ex.Message,
                "expects operand dtypes (float64, float64, float64) but was called with (int32, float64, float64)");
        }

        [TestMethod]
        public void Compile_Positional_MixedSignature_TypesLikeNumPy()
        {
            var ai = np.array(new int[] { 1, 2, 3 });
            var bf = Vec(0.5, 0.5, 0.5);

            var h = (NDExpr.Input(0) * NDExpr.Input(1)).Compile(NPTypeCode.Int32, NPTypeCode.Double);
            Assert.AreEqual(NPTypeCode.Double, h.ResultType, "int32 * float64 → float64 (NEP50)");
            CollectionAssert.AreEqual(new[] { NPTypeCode.Int32, NPTypeCode.Double }, h.InputTypes);
            AssertSameArray(ai * bf, h.Evaluate(new[] { ai, bf }));
        }

        [TestMethod]
        public void Compile_BindingFormMismatches_AreRejected()
        {
            var a = Vec(1, 2, 3);

            // Compile() needs array leaves; a positional tree names the typed overload.
            var positional = NDExpr.Input(0) * 2.0;
            var ex1 = Assert.ThrowsException<InvalidOperationException>(() => positional.Compile());
            StringAssert.Contains(ex1.Message, "Compile(params NPTypeCode[] inputTypes)");

            // Compile(types) is the positional form; an embedded tree mixes binding styles.
            var embedded = (NDExpr)a * 2.0;
            var ex2 = Assert.ThrowsException<ArgumentException>(() => embedded.Compile(NPTypeCode.Double));
            StringAssert.Contains(ex2.Message, "mixes embedded array leaves with a positional operand list");

            // Signature validation.
            Assert.ThrowsException<ArgumentException>(() => positional.Compile(Array.Empty<NPTypeCode>()));
            Assert.ThrowsException<ArgumentException>(() => positional.Compile(NPTypeCode.Empty));

            // The wrong Evaluate overload for the handle's form is an error, not a guess.
            var hp = positional.Compile(NPTypeCode.Double);
            Assert.ThrowsException<InvalidOperationException>(() => hp.Evaluate());
            var he = embedded.Compile();
            Assert.ThrowsException<InvalidOperationException>(() => he.Evaluate(new[] { a }));
        }

        // =====================================================================
        // The N-ary input broadcast
        // =====================================================================

        [TestMethod]
        public void Broadcast_Mismatch_ListsEveryOperandShape_LikeNumPy()
        {
            var a = np.arange(6).astype(NPTypeCode.Double).reshape(2, 3);
            var b = np.arange(6).astype(NPTypeCode.Double).reshape(2, 3);
            var c = Vec(1, 2, 3, 4);

            // NumPy: ValueError: operands could not be broadcast together with shapes (2,3) (2,3) (4,)
            // (trailing space included). The old pairwise fold could only name the running shape
            // and the offending operand.
            var ex = Assert.ThrowsException<IncorrectShapeException>(() => np.evaluate((NDExpr)a * b + c));
            Assert.AreEqual("operands could not be broadcast together with shapes (2,3) (2,3) (4,) ", ex.Message);

            var exr = Assert.ThrowsException<IncorrectShapeException>(() => np.evaluate(NDExpr.Sum((NDExpr)a * b + c)));
            Assert.AreEqual("operands could not be broadcast together with shapes (2,3) (2,3) (4,) ", exr.Message);

            var exa = Assert.ThrowsException<IncorrectShapeException>(() => np.evaluate(NDExpr.Sum((NDExpr)a * b + c, axis: 0)));
            Assert.AreEqual("operands could not be broadcast together with shapes (2,3) (2,3) (4,) ", exa.Message);
        }

        [TestMethod]
        public void Broadcast_ColumnTimesRow_MatchesTheUnfusedChain()
        {
            var col = Vec(1, 2, 3).reshape(3, 1);
            var row = Vec(10, 20, 30, 40).reshape(1, 4);
            var k = NDArray.Scalar(0.5);

            var fused = np.evaluate((NDExpr)col * row + k);
            var unfused = col * row + k;
            CollectionAssert.AreEqual(new[] { 3L, 4L }, new[] { (long)fused.shape[0], (long)fused.shape[1] });
            AssertSameArray(unfused, fused);

            // The same through a reduction (flat count) and an axis reduction.
            Assert.AreEqual(np.sum(unfused).GetDouble(0), np.evaluate(NDExpr.Sum((NDExpr)col * row + k)).GetDouble(0));
            AssertSameArray(np.sum(unfused, 1), np.evaluate(NDExpr.Sum((NDExpr)col * row + k, axis: 1)));
        }

        [TestMethod]
        public void Broadcast_ZeroLengthAxisAgainstOne_IsEmpty()
        {
            var empty = np.zeros(new Shape(0), NPTypeCode.Double);
            var one = Vec(7);
            var r = np.evaluate((NDExpr)empty * one);
            Assert.AreEqual(1, r.ndim);
            Assert.AreEqual(0, r.size);
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
        }

        [TestMethod]
        public void Broadcast_Out_IsNeverStretched()
        {
            var a = Vec(1, 2, 3, 4);
            var b = Vec(1, 1, 1, 1);
            var tooSmall = np.zeros(new Shape(1), NPTypeCode.Double);

            var ex = Assert.ThrowsException<ArgumentException>(() => np.evaluate((NDExpr)a + b, @out: tooSmall));
            StringAssert.Contains(ex.Message,
                "non-broadcastable output operand with shape (1,) doesn't match the broadcast shape (4,)");
        }

        // =====================================================================
        // Literal-as-parameter: a 0-d operand instead of a baked constant
        // =====================================================================

        [TestMethod]
        public void ParameterForm_ZeroDOperand_ServesEveryConstantWithOneKernel()
        {
            var a = Vec(1, 2, 3, 4, 5, 6, 7, 8);
            var b = Vec(8, 7, 6, 5, 4, 3, 2, 1);
            var expr = NDExpr.Input(0) * NDExpr.Input(1) + NDExpr.Input(2);

            // First call may JIT the kernel; every later constant must reuse it.
            AssertSameArray(a * b + 1.5, np.evaluate(expr, new[] { a, b, NDArray.Scalar(1.5) }));
            int kernels = GeneratedDelegates.InnerLoopCount;
            foreach (var k in new[] { 2.5, -3.0, 1e6, 0.0 })
            {
                AssertSameArray(a * b + k, np.evaluate(expr, new[] { a, b, NDArray.Scalar(k) }));
                Assert.AreEqual(kernels, GeneratedDelegates.InnerLoopCount, $"k={k} must not compile a new kernel");
            }

            // A 0-d operand is a STRONG NumPy scalar (np.float64(k)), so it takes part in promotion,
            // unlike a weak literal: int32 * float64-scalar → float64. Both spellings are legal;
            // the literal is baked into the kernel (one kernel per distinct value), the operand is not.
            var ai = np.array(new int[] { 1, 2, 3 });
            var strong = np.evaluate(NDExpr.Input(0) * NDExpr.Input(1), new[] { ai, NDArray.Scalar(2.0) });
            Assert.AreEqual(NPTypeCode.Double, strong.typecode);
            var weak = np.evaluate((NDExpr)ai * 2.0);
            Assert.AreEqual(NPTypeCode.Double, weak.typecode, "a float literal promotes int32 to float64 too (NEP50)");
            var weakInt = np.evaluate((NDExpr)ai * 2);
            Assert.AreEqual(NPTypeCode.Int32, weakInt.typecode, "an int literal is weak and keeps int32");
            var strongInt = np.evaluate(NDExpr.Input(0) * NDExpr.Input(1), new[] { ai, NDArray.Scalar(2L) });
            Assert.AreEqual(NPTypeCode.Int64, strongInt.typecode, "a 0-d int64 operand is strong: int32 * int64 → int64");
        }
    }
}
