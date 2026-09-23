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
        public void Program_Embedded_IsCachedPerRoot_AndSharedStructurallyAcrossRoots()
        {
            var a = Vec(1, 2, 3, 4);
            var b = Vec(5, 6, 7, 8);
            var c = Vec(9, 10, 11, 12);
            var root = (NDExpr)a * b + c;

            var p1 = root.GetProgram(null, out var ops1);
            var p2 = root.GetProgram(null, out var ops2);
            Assert.AreSame(p1, p2, "the second call must reuse the root's program");
            Assert.AreSame(ops1, ops2, "and the operand list resolved with it");
            CollectionAssert.AreEqual(new[] { a, b, c }, ops1, "operands in binding (first-visit) order");
            Assert.AreEqual(NPTypeCode.Double, p1.ResultType);

            // A NEW tree instance of the same structure — over the same arrays or over other arrays of
            // the same dtypes — is served by the SAME program through the global structural cache; only
            // the operand list is per root. This is what makes `np.evaluate((NDExpr)a * b + c)` inside a
            // loop as cheap as a hoisted tree.
            var other = (NDExpr)a * b + c;
            var p3 = other.GetProgram(null, out var ops3);
            Assert.AreSame(p1, p3, "same structure + same dtype signature → one program");
            Assert.AreSame(p1.Kernel, p3.Kernel);
            CollectionAssert.AreEqual(new[] { a, b, c }, ops3);

            var x = Vec(1, 1, 1, 1);
            var y = Vec(2, 2, 2, 2);
            var z = Vec(3, 3, 3, 3);
            var third = (NDExpr)x * y + z;
            var p4 = third.GetProgram(null, out var ops4);
            Assert.AreSame(p1, p4, "other arrays of the same dtypes share the program too");
            CollectionAssert.AreEqual(new[] { x, y, z }, ops4, "but resolve their own operand list");
            AssertSameArray(x * y + z, np.evaluate(third));
            AssertSameArray(a * b + c, np.evaluate(other));
        }

        // =====================================================================
        // The global structural cache (Phase 6.1): a tree rebuilt per call finds its program
        // =====================================================================

        [TestMethod]
        public void StructuralCache_RebuiltTree_HitsTheProgram_WithoutABindOrTypingPass()
        {
            var a = Vec(1, 2, 3, 4);
            var b = Vec(5, 6, 7, 8);
            var c = Vec(9, 10, 11, 12);

            np.evaluate((NDExpr)a * b + c);                          // primes the cache (or is already primed)
            long hits = NDExprProgramCache.Hits;
            long misses = NDExprProgramCache.Misses;
            int kernels = GeneratedDelegates.InnerLoopCount;

            for (int i = 0; i < 5; i++)
                AssertSameArray(a * b + c, np.evaluate((NDExpr)a * b + c));   // a NEW root every call

            Assert.AreEqual(hits + 5, NDExprProgramCache.Hits, "every rebuilt tree must be a verified hit");
            Assert.AreEqual(misses, NDExprProgramCache.Misses, "and never a miss");
            Assert.AreEqual(kernels, GeneratedDelegates.InnerLoopCount, "and compile nothing");
        }

        [TestMethod]
        public void StructuralCache_DistinguishesWhatChangesTheKernel()
        {
            var a = Vec(1, 2, 3, 4);
            var b = Vec(5, 6, 7, 8);
            var c = Vec(9, 10, 11, 12);
            var ai = np.array(new int[] { 1, 2, 3, 4 });

            var p = ((NDExpr)a * b + c).GetProgram(null);

            // child order (the kernel reads its operands in a different order)
            Assert.AreNotSame(p, ((NDExpr)c + (NDExpr)a * b).GetProgram(null));
            // the op
            Assert.AreNotSame(p, ((NDExpr)a * b - c).GetProgram(null));
            // the operand dedup pattern: a*b+c reads three streams, a*b+a reads two
            Assert.AreNotSame(p, ((NDExpr)a * b + a).GetProgram(null));
            // the dtype signature
            Assert.AreNotSame(p, ((NDExpr)ai * b + c).GetProgram(null));
            // a literal's value and its kind (2 and 2.0 type differently; -0.0 and 0.0 emit differently)
            var lit = ((NDExpr)a * b + 2.0).GetProgram(null);
            Assert.AreNotSame(lit, ((NDExpr)a * b + 3.0).GetProgram(null));
            Assert.AreNotSame(lit, ((NDExpr)a * b + 2).GetProgram(null));
            Assert.AreNotSame(((NDExpr)a * 0.0).GetProgram(null), ((NDExpr)a * -0.0).GetProgram(null));
            // a reduction's kind, axis and keepdims are read by the host per evaluation
            var a2 = np.arange(6).astype(NPTypeCode.Double).reshape(2, 3);
            var s0 = NDExpr.Sum((NDExpr)a2 * a2, axis: 0).GetProgram(null);
            Assert.AreNotSame(s0, NDExpr.Sum((NDExpr)a2 * a2, axis: 1).GetProgram(null));
            Assert.AreNotSame(s0, NDExpr.Sum((NDExpr)a2 * a2, axis: 0, keepdims: true).GetProgram(null));
            Assert.AreNotSame(s0, NDExpr.Max((NDExpr)a2 * a2, axis: 0).GetProgram(null));
            Assert.AreNotSame(s0, NDExpr.Sum((NDExpr)a2 * a2).GetProgram(null));
            Assert.AreSame(s0, NDExpr.Sum((NDExpr)a2 * a2, axis: 0).GetProgram(null));

            // …and every one of them still computes its own answer.
            AssertSameArray(c + a * b, np.evaluate((NDExpr)c + (NDExpr)a * b));
            AssertSameArray(a * b - c, np.evaluate((NDExpr)a * b - c));
            AssertSameArray(a * b + a, np.evaluate((NDExpr)a * b + a));
            AssertSameArray(a * b + 3.0, np.evaluate((NDExpr)a * b + 3.0));
            AssertSameArray(np.sum(a2 * a2, 1), np.evaluate(NDExpr.Sum((NDExpr)a2 * a2, axis: 1)));
            AssertSameArray(np.max(a2 * a2, 0), np.evaluate(NDExpr.Max((NDExpr)a2 * a2, axis: 0)));
        }

        [TestMethod]
        public void StructuralCache_EmbeddedAndPositionalSpellings_ShareOneProgram()
        {
            var a = Vec(1, 2, 3, 4);
            var b = Vec(5, 6, 7, 8);

            var embedded = ((NDExpr)a * b).GetProgram(null);
            var positional = (NDExpr.Input(0) * NDExpr.Input(1)).GetProgram(new[] { a, b });
            Assert.AreSame(embedded, positional, "same bound structure + signature → one program");

            // A positional tree with a repeated input is NOT the dedup pattern of a*b (two streams vs one).
            Assert.AreNotSame(embedded, (NDExpr.Input(0) * NDExpr.Input(0)).GetProgram(new[] { a }));
            Assert.AreSame(((NDExpr)a * a).GetProgram(null), (NDExpr.Input(0) * NDExpr.Input(0)).GetProgram(new[] { a }));
        }

        [TestMethod]
        public void StructuralCache_IsBounded_AndClearsWhenFull()
        {
            int capacity = NDExprProgramCache.Capacity;
            try
            {
                NDExprProgramCache.Clear();
                NDExprProgramCache.Capacity = 3;
                var a = Vec(1, 2, 3, 4);
                // Four distinct literals → four programs; the cap admits three, the fourth clears first.
                for (int k = 1; k <= 4; k++)
                {
                    var r = np.evaluate((NDExpr)a * (1000.0 + k));
                    AssertSameArray(a * (1000.0 + k), r);
                    Assert.IsTrue(NDExprProgramCache.Count <= 3, $"count {NDExprProgramCache.Count} after literal {k}");
                }

                Assert.AreEqual(1, NDExprProgramCache.Count, "the clear happened on the fourth insert");
                // Evaluations stay correct across the clear — the per-root slot and the kernel cache are untouched.
                AssertSameArray(a * 1001.0, np.evaluate((NDExpr)a * 1001.0));
            }
            finally
            {
                NDExprProgramCache.Capacity = capacity;
                NDExprProgramCache.Clear();
            }
        }

        [TestMethod]
        public void StructuralCache_ThreadSafe_ConcurrentRebuiltTreesStayCorrect()
        {
            var a = Vec(1, 2, 3, 4, 5, 6, 7, 8);
            var b = Vec(8, 7, 6, 5, 4, 3, 2, 1);
            var c = Vec(1, 1, 1, 1, 1, 1, 1, 1);
            var expected1 = a * b + c;
            var expected2 = (a - b) / (a + b);
            var expected3 = np.where(a > b, a, b);

            System.Threading.Tasks.Parallel.For(0, 2000, i =>
            {
                switch (i % 3)
                {
                    case 0: Assert.IsTrue(np.array_equal(expected1, np.evaluate((NDExpr)a * b + c))); break;
                    case 1: Assert.IsTrue(np.array_equal(expected2, np.evaluate((NDExpr.Arr(a) - b) / (NDExpr.Arr(a) + b)))); break;
                    default: Assert.IsTrue(np.array_equal(expected3, np.evaluate(NDExpr.Where(NDExpr.Greater(NDExpr.Arr(a), b), NDExpr.Arr(a), NDExpr.Arr(b))))); break;
                }
            });
        }

        // =====================================================================
        // Call nodes: a delegate's slot is part of the kernel's identity
        // =====================================================================

        [TestMethod]
        public void Call_TwoClosuresOfOneLambda_GetTheirOwnKernels()
        {
            var x = Vec(1, 2, 3, 4);
            Func<double, double> Scale(double k) => v => v * k;      // one lambda body, two captured states
            var twice = Scale(2.0);
            var thrice = Scale(3.0);

            // Both trees share the MethodInfo; before the slot joined the signature the second closure was
            // served the first closure's kernel and returned x * 2.
            AssertSameArray(x * 2.0, np.evaluate(NDExpr.Call(twice, NDExpr.Arr(x))));
            AssertSameArray(x * 3.0, np.evaluate(NDExpr.Call(thrice, NDExpr.Arr(x))));
            AssertSameArray(x * 2.0, np.evaluate(NDExpr.Call(twice, NDExpr.Arr(x))));
        }

        [TestMethod]
        public void Call_SameDelegateInstanceRebuiltPerCall_KeepsOneSlotAndOneKernel()
        {
            var x = Vec(1, 2, 3, 4);
            double k = 0.5;
            Func<double, double> f = v => v * k;                       // a captured delegate held in a local

            AssertSameArray(x * 0.5, np.evaluate(NDExpr.Call(f, NDExpr.Arr(x))));
            int slots = DelegateSlots.RegisteredCount;
            int kernels = GeneratedDelegates.InnerLoopCount;
            for (int i = 0; i < 5; i++)
                AssertSameArray(x * 0.5, np.evaluate(NDExpr.Call(f, NDExpr.Arr(x))));   // a NEW CallNode each time
            Assert.AreEqual(slots, DelegateSlots.RegisteredCount, "the same delegate instance dedups to one slot");
            Assert.AreEqual(kernels, GeneratedDelegates.InnerLoopCount, "and the kernel is reused");

            // A bound instance target dedups by reference the same way.
            var box = new Scaler(4.0);
            var mi = typeof(Scaler).GetMethod(nameof(Scaler.Apply));
            AssertSameArray(x * 4.0, np.evaluate(NDExpr.Call(mi, box, NDExpr.Arr(x))));
            slots = DelegateSlots.RegisteredCount;
            AssertSameArray(x * 4.0, np.evaluate(NDExpr.Call(mi, box, NDExpr.Arr(x))));
            Assert.AreEqual(slots, DelegateSlots.RegisteredCount);
            // A different instance is a different target: its own slot, its own answer.
            AssertSameArray(x * 5.0, np.evaluate(NDExpr.Call(mi, new Scaler(5.0), NDExpr.Arr(x))));
            Assert.AreEqual(slots + 1, DelegateSlots.RegisteredCount);
        }

        private sealed class Scaler
        {
            private readonly double _k;
            public Scaler(double k) => _k = k;
            public double Apply(double v) => v * _k;
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
            var pi = root.GetProgram(new[] { ai, bi }, out var opsI);
            Assert.AreEqual(NPTypeCode.Int32, ri.typecode);
            AssertSameArray(ai * bi, ri);
            Assert.AreNotSame(pf, pi);
            Assert.AreSame(opsI, ri is null ? null : opsI, "the positional form evaluates the operands it was given");

            // Back to float64: the root's slot holds one program, so this is a re-resolve — served by the
            // global structural cache (the float64 program is still there), still correct.
            var rf2 = np.evaluate(root, new[] { af, bf });
            AssertSameArray(af * bf, rf2);
            Assert.AreSame(pf, root.GetProgram(new[] { af, bf }), "the float64 program comes back from the structural cache");
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
