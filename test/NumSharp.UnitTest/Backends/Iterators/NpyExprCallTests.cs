using System;
using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// Covers the NpyExpr.Call factory family:
    ///   • Typed Func<...> overloads (arity 0–4) — allow method groups without cast
    ///   • Catch-all Delegate overload — for pre-constructed delegates
    ///   • MethodInfo for static methods
    ///   • MethodInfo + target for instance methods
    ///   • Type conversion: method param dtype vs tree output dtype
    ///   • Captured lambdas (closure state preserved across calls)
    ///   • Composition with other DSL nodes
    ///   • Cache key structure + reuse
    ///   • Validation errors
    /// </summary>
    [TestClass]
    public unsafe class NpyExprCallTests
    {
        // =====================================================================
        // Helpers
        // =====================================================================

        private static NpyIterRef Iter(NDArray input, NDArray output)
            => NpyIterRef.MultiNew(2, new[] { input, output },
                NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });

        private static NpyIterRef Iter3(NDArray a, NDArray b, NDArray c)
            => NpyIterRef.MultiNew(3, new[] { a, b, c },
                NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY,
                        NpyIterPerOpFlags.READONLY,
                        NpyIterPerOpFlags.WRITEONLY });

        // =====================================================================
        // Typed Func overloads — method group without cast
        // =====================================================================

        [TestMethod]
        public void Call_MethodGroup_UnaryMathSqrt_NoCast()
        {
            var input = np.array(new double[] { 1, 4, 9, 16, 25 });
            var output = np.empty_like(input);
            using var it = Iter(input, output);

            // No cast, no generic type args — method group inference from Func<double,double>
            var expr = NpyExpr.Call(Math.Sqrt, NpyExpr.Input(0));
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "call_mg_sqrt_v1");

            for (int i = 0; i < 5; i++)
                Assert.AreEqual(Math.Sqrt(input.GetDouble(i)), output.GetDouble(i), 1e-9);
        }

        [TestMethod]
        public void Call_MethodGroup_BinaryMathPow_NoCast()
        {
            var a = np.array(new double[] { 2, 3, 4 });
            var b = np.array(new double[] { 3, 2, 0.5 });
            var c = np.empty_like(a);
            using var it = Iter3(a, b, c);

            var expr = NpyExpr.Call(Math.Pow, NpyExpr.Input(0), NpyExpr.Input(1));
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "call_mg_pow_v1");

            Assert.AreEqual(8.0, c.GetDouble(0), 1e-9);
            Assert.AreEqual(9.0, c.GetDouble(1), 1e-9);
            Assert.AreEqual(2.0, c.GetDouble(2), 1e-9);
        }

        [TestMethod]
        public void Call_FuncExplicit_BinaryPow_WithGenericArgs()
        {
            var a = np.array(new double[] { 2, 3, 4 });
            var b = np.array(new double[] { 3, 2, 0.5 });
            var c = np.empty_like(a);
            using var it = Iter3(a, b, c);

            var expr = NpyExpr.Call<double, double, double>(Math.Pow, NpyExpr.Input(0), NpyExpr.Input(1));
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "call_func_pow_v1");

            Assert.AreEqual(8.0, c.GetDouble(0), 1e-9);
            Assert.AreEqual(9.0, c.GetDouble(1), 1e-9);
            Assert.AreEqual(2.0, c.GetDouble(2), 1e-9);
        }

        [TestMethod]
        public void Call_MathAbs_DoubleOverload_CastDisambig()
        {
            var a = np.array(new double[] { -5, -1, 0, 3.5 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);

            // Math.Abs has multiple overloads → method group ambiguous → user must cast
            var expr = NpyExpr.Call((Func<double, double>)Math.Abs, NpyExpr.Input(0));
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "call_abs_v1");

            Assert.AreEqual(5.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(1.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(2), 1e-9);
            Assert.AreEqual(3.5, r.GetDouble(3), 1e-9);
        }

        // =====================================================================
        // Captured lambdas — delegate slot lookup path
        // =====================================================================

        [TestMethod]
        public void Call_CapturedLambda_AppliesClosureState()
        {
            double scale = 3.5;
            double bias = 7.0;
            Func<double, double> affine = x => x * scale + bias;

            var a = np.array(new double[] { 1, 2, 3, 4 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);

            var expr = NpyExpr.Call(affine, NpyExpr.Input(0));
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "call_affine_v1");

            for (int i = 0; i < 4; i++)
                Assert.AreEqual((i + 1) * 3.5 + 7.0, r.GetDouble(i), 1e-9);
        }

        [TestMethod]
        public void Call_CapturedLambda_BinaryComposition()
        {
            Func<double, double, double> weighted = (x, w) => x * w + 0.5;

            var a = np.array(new double[] { 1, 2, 3 });
            var b = np.array(new double[] { 10, 20, 30 });
            var c = np.empty_like(a);
            using var it = Iter3(a, b, c);

            var expr = NpyExpr.Call(weighted, NpyExpr.Input(0), NpyExpr.Input(1));
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "call_weighted_v1");

            Assert.AreEqual(10.5, c.GetDouble(0), 1e-9);
            Assert.AreEqual(40.5, c.GetDouble(1), 1e-9);
            Assert.AreEqual(90.5, c.GetDouble(2), 1e-9);
        }

        // =====================================================================
        // MethodInfo (static)
        // =====================================================================

        [TestMethod]
        public void Call_MethodInfo_StaticMath_NoTarget()
        {
            var a = np.array(new double[] { 0.5, 1.0, 2.0 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);

            var mi = typeof(Math).GetMethod("Tanh", new[] { typeof(double) })!;
            var expr = NpyExpr.Call(mi, NpyExpr.Input(0));
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "call_mi_tanh_v1");

            Assert.AreEqual(Math.Tanh(0.5), r.GetDouble(0), 1e-9);
            Assert.AreEqual(Math.Tanh(1.0), r.GetDouble(1), 1e-9);
            Assert.AreEqual(Math.Tanh(2.0), r.GetDouble(2), 1e-9);
        }

        [TestMethod]
        public void Call_MethodInfo_UserStaticMethod()
        {
            var a = np.array(new double[] { 1, 2, 3 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);

            var mi = typeof(StaticHelpers).GetMethod(nameof(StaticHelpers.DoubleIt))!;
            var expr = NpyExpr.Call(mi, NpyExpr.Input(0));
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "call_mi_double_v1");

            Assert.AreEqual(2.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(4.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(6.0, r.GetDouble(2), 1e-9);
        }

        // =====================================================================
        // MethodInfo + target (instance)
        // =====================================================================

        [TestMethod]
        public void Call_MethodInfo_InstanceMethod_PreservesTargetState()
        {
            var obj = new Multiplier { Factor = 7.0 };
            var a = np.array(new double[] { 1, 2, 3 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);

            var mi = typeof(Multiplier).GetMethod(nameof(Multiplier.Apply))!;
            var expr = NpyExpr.Call(mi, obj, NpyExpr.Input(0));
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "call_inst_apply_v1");

            Assert.AreEqual(7.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(14.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(21.0, r.GetDouble(2), 1e-9);
        }

        [TestMethod]
        public void Call_MethodInfo_InstanceMethod_Binary()
        {
            var obj = new BinaryCalc { Offset = 100.0 };
            var a = np.array(new double[] { 1, 2, 3 });
            var b = np.array(new double[] { 10, 20, 30 });
            var c = np.empty_like(a);
            using var it = Iter3(a, b, c);

            var mi = typeof(BinaryCalc).GetMethod(nameof(BinaryCalc.Combine))!;
            var expr = NpyExpr.Call(mi, obj, NpyExpr.Input(0), NpyExpr.Input(1));
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "call_inst_combine_v1");

            // Combine(a, b) = a + b + Offset(100)
            Assert.AreEqual(111.0, c.GetDouble(0), 1e-9);
            Assert.AreEqual(122.0, c.GetDouble(1), 1e-9);
            Assert.AreEqual(133.0, c.GetDouble(2), 1e-9);
        }

        [TestMethod]
        public void Call_MethodInfo_InstanceMethod_MutatesTargetAcrossCalls()
        {
            // Target object has state; each call reads fresh state.
            var counter = new Counter();
            var a = np.array(new double[] { 1, 1, 1 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);

            var mi = typeof(Counter).GetMethod(nameof(Counter.IncrementAndAdd))!;
            var expr = NpyExpr.Call(mi, counter, NpyExpr.Input(0));
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "call_counter_v1");

            // Counter.IncrementAndAdd returns ++count + x for each element.
            // With input [1,1,1] and starting count 0:
            //   element 0: count=1, result=1+1=2
            //   element 1: count=2, result=2+1=3
            //   element 2: count=3, result=3+1=4
            Assert.AreEqual(2.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(3.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(4.0, r.GetDouble(2), 1e-9);
            Assert.AreEqual(3, counter.Count);
        }

        // =====================================================================
        // Zero-arg delegate (Func<TR>)
        // =====================================================================

        [TestMethod]
        public void Call_ZeroArg_ConstProvider()
        {
            int hitCount = 0;
            Func<double> provider = () => { hitCount++; return 42.0; };

            var a = np.array(new double[] { 1, 2, 3 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);

            // out = provider() + input (provider is called per element)
            var expr = NpyExpr.Call(provider) + NpyExpr.Input(0);
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "call_provider_v1");

            Assert.AreEqual(43.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(44.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(45.0, r.GetDouble(2), 1e-9);
            Assert.AreEqual(3, hitCount, "Provider should fire once per element");
        }

        // =====================================================================
        // Type conversion: Int32 input → double method param
        // =====================================================================

        [TestMethod]
        public void Call_Int32Input_DoubleMethod_AutoConverts()
        {
            var a = np.array(new int[] { 0, 1, 4, 9, 16, 25 });
            var r = np.empty(new Shape(6), np.float64);
            using var it = Iter(a, r);

            // Input is Int32, output is Double. The DSL converts Int32→Double at Input,
            // Double→Double for the method's param (no-op), Double→Double for the return.
            var expr = NpyExpr.Call(Math.Sqrt, NpyExpr.Input(0));
            it.ExecuteExpression(expr, new[] { NPTypeCode.Int32 }, NPTypeCode.Double,
                cacheKey: "call_i32_d_sqrt_v1");

            Assert.AreEqual(0.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(1.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(2.0, r.GetDouble(2), 1e-9);
            Assert.AreEqual(3.0, r.GetDouble(3), 1e-9);
            Assert.AreEqual(4.0, r.GetDouble(4), 1e-9);
            Assert.AreEqual(5.0, r.GetDouble(5), 1e-9);
        }

        [TestMethod]
        public void Call_DoubleTreeOutput_IntegerReturningMethod_AutoConvertsResult()
        {
            // Method returns int; tree output is double. Return value widens int→double.
            Func<double, int> floorInt = x => (int)Math.Floor(x);
            var a = np.array(new double[] { 1.7, 2.5, 3.9 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);

            var expr = NpyExpr.Call(floorInt, NpyExpr.Input(0));
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "call_flint_v1");

            Assert.AreEqual(1.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(2.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(3.0, r.GetDouble(2), 1e-9);
        }

        [TestMethod]
        public void Call_FloatTreeOutput_DoubleMethod_NarrowsReturn()
        {
            var a = np.array(new float[] { 1f, 4f, 9f, 16f });
            var r = np.empty_like(a);
            using var it = Iter(a, r);

            // Math.Sqrt is Double → Double; tree runs in float.
            // Args: float → double before call; Return: double → float.
            var expr = NpyExpr.Call(Math.Sqrt, NpyExpr.Input(0));
            it.ExecuteExpression(expr, new[] { NPTypeCode.Single }, NPTypeCode.Single,
                cacheKey: "call_f32_sqrt_v1");

            Assert.AreEqual(1f, r.GetSingle(0), 1e-6f);
            Assert.AreEqual(2f, r.GetSingle(1), 1e-6f);
            Assert.AreEqual(3f, r.GetSingle(2), 1e-6f);
            Assert.AreEqual(4f, r.GetSingle(3), 1e-6f);
        }

        // =====================================================================
        // Composition with other DSL nodes
        // =====================================================================

        [TestMethod]
        public void Call_ComposedWithOperators()
        {
            // (Math.Sqrt(x) + 1) * 2
            var a = np.array(new double[] { 1, 4, 9, 16 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);

            var expr = (NpyExpr.Call(Math.Sqrt, NpyExpr.Input(0)) + NpyExpr.Const(1.0)) * NpyExpr.Const(2.0);
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "call_composed_v1");

            Assert.AreEqual(4.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(6.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(8.0, r.GetDouble(2), 1e-9);
            Assert.AreEqual(10.0, r.GetDouble(3), 1e-9);
        }

        [TestMethod]
        public void Call_UsedInsideWhere()
        {
            // Use Call to pick different transforms per branch.
            var a = np.array(new double[] { -2, -1, 0, 1, 2 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);

            var x = NpyExpr.Input(0);
            var expr = NpyExpr.Where(
                NpyExpr.Greater(x, NpyExpr.Const(0.0)),
                NpyExpr.Call(Math.Sqrt, x),        // positive → sqrt
                NpyExpr.Call(Math.Exp, x));        // non-positive → exp

            it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "call_where_v1");

            // expected: exp(-2), exp(-1), exp(0), sqrt(1), sqrt(2)
            Assert.AreEqual(Math.Exp(-2), r.GetDouble(0), 1e-9);
            Assert.AreEqual(Math.Exp(-1), r.GetDouble(1), 1e-9);
            Assert.AreEqual(Math.Exp(0), r.GetDouble(2), 1e-9);
            Assert.AreEqual(Math.Sqrt(1), r.GetDouble(3), 1e-9);
            Assert.AreEqual(Math.Sqrt(2), r.GetDouble(4), 1e-9);
        }

        // =====================================================================
        // Cache behavior
        // =====================================================================

        [TestMethod]
        public void Call_SameStaticMethodReusesKernel()
        {
            ILKernelGenerator.ClearInnerLoopCache();

            var a = np.arange(10).astype(np.float64);
            var r = np.empty_like(a);

            using (var it = Iter(a, r))
                it.ExecuteExpression(NpyExpr.Call(Math.Sqrt, NpyExpr.Input(0)),
                    new[] { NPTypeCode.Double }, NPTypeCode.Double,
                    cacheKey: "call_reuse_v1");
            int after1 = ILKernelGenerator.InnerLoopCachedCount;

            using (var it = Iter(a, r))
                it.ExecuteExpression(NpyExpr.Call(Math.Sqrt, NpyExpr.Input(0)),
                    new[] { NPTypeCode.Double }, NPTypeCode.Double,
                    cacheKey: "call_reuse_v1");
            int after2 = ILKernelGenerator.InnerLoopCachedCount;

            Assert.AreEqual(after1, after2, "Same cache key → same kernel");
        }

        [TestMethod]
        public void Call_DifferentMethodsProduceDistinctKernels()
        {
            ILKernelGenerator.ClearInnerLoopCache();

            var a = np.arange(10).astype(np.float64);
            var r = np.empty_like(a);

            using (var it = Iter(a, r))
                it.ExecuteExpression(NpyExpr.Call(Math.Sqrt, NpyExpr.Input(0)),
                    new[] { NPTypeCode.Double }, NPTypeCode.Double);
            int afterSqrt = ILKernelGenerator.InnerLoopCachedCount;

            using (var it = Iter(a, r))
                it.ExecuteExpression(NpyExpr.Call(Math.Cbrt, NpyExpr.Input(0)),
                    new[] { NPTypeCode.Double }, NPTypeCode.Double);
            int afterCbrt = ILKernelGenerator.InnerLoopCachedCount;

            Assert.AreEqual(afterSqrt + 1, afterCbrt,
                "Different MethodInfos must produce distinct cache entries");
        }

        [TestMethod]
        public void Call_AutoDerivedCacheKey_Works()
        {
            // No explicit cacheKey — the auto-derived one must execute correctly.
            var a = np.array(new double[] { 1, 4, 9 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);

            var expr = NpyExpr.Call(Math.Sqrt, NpyExpr.Input(0));
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double);

            Assert.AreEqual(1.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(2.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(3.0, r.GetDouble(2), 1e-9);
        }

        // =====================================================================
        // Validation / errors
        // =====================================================================

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void Call_NullDelegate_Throws()
        {
            Func<double, double> f = null!;
            _ = NpyExpr.Call(f, NpyExpr.Input(0));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void Call_NullMethodInfo_Throws()
        {
            MethodInfo mi = null!;
            _ = NpyExpr.Call(mi, NpyExpr.Input(0));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void Call_NullArg_Throws()
        {
            _ = NpyExpr.Call(Math.Sqrt, (NpyExpr)null!);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Call_InstanceMethodWithoutTarget_Throws()
        {
            var mi = typeof(Multiplier).GetMethod(nameof(Multiplier.Apply))!;
            // Passing null as target but method is instance — should throw
            _ = NpyExpr.Call(mi, target: null!, NpyExpr.Input(0));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Call_StaticMethodWithTarget_Throws()
        {
            var mi = typeof(Math).GetMethod("Sqrt", new[] { typeof(double) })!;
            // Passing a target to a static method — should throw
            _ = NpyExpr.Call(mi, target: new object(), NpyExpr.Input(0));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Call_TargetTypeMismatch_Throws()
        {
            var mi = typeof(Multiplier).GetMethod(nameof(Multiplier.Apply))!;
            // Target is wrong type
            _ = NpyExpr.Call(mi, new Counter(), NpyExpr.Input(0));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Call_ArgCountMismatch_Throws()
        {
            // Math.Pow needs 2 args; we pass 1
            _ = NpyExpr.Call(Math.Pow, NpyExpr.Input(0));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Call_VoidReturningMethod_Throws()
        {
            var mi = typeof(StaticHelpers).GetMethod(nameof(StaticHelpers.VoidMethod))!;
            _ = NpyExpr.Call(mi, NpyExpr.Input(0));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Call_UnsupportedParamType_Throws()
        {
            // Method takes a string — not in the 12-type set
            var mi = typeof(StaticHelpers).GetMethod(nameof(StaticHelpers.StringLength))!;
            _ = NpyExpr.Call(mi, NpyExpr.Input(0));
        }

        // =====================================================================
        // Strided input
        // =====================================================================

        [TestMethod]
        public void Call_StridedInput_WorksViaScalarFallback()
        {
            var src = np.arange(20).astype(np.float64);
            var strided = src["::2"];  // 10 elements, non-contig stride
            var r = np.empty(new Shape(10), np.float64);

            using var it = Iter(strided, r);
            var expr = NpyExpr.Call(Math.Sqrt, NpyExpr.Input(0));
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "call_strided_v1");

            for (int i = 0; i < 10; i++)
                Assert.AreEqual(Math.Sqrt(2.0 * i), r.GetDouble(i), 1e-9);
        }

        // =====================================================================
        // Stress: varying sizes
        // =====================================================================

        [DataTestMethod]
        [DataRow(2)]
        [DataRow(7)]
        [DataRow(32)]
        [DataRow(65)]
        [DataRow(1024)]
        public void Call_AcrossSizes(int size)
        {
            var xs = new double[size];
            for (int i = 0; i < size; i++) xs[i] = i * 0.01;
            var a = np.array(xs);
            var r = np.empty_like(a);

            Func<double, double> f = x => Math.Sin(x) * Math.Cos(x);

            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.Call(f, NpyExpr.Input(0)),
                new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "call_stress_v1");

            for (int i = 0; i < size; i++)
                Assert.AreEqual(Math.Sin(xs[i]) * Math.Cos(xs[i]), r.GetDouble(i), 1e-9);
        }

        // =====================================================================
        // Higher-arity delegates
        // =====================================================================

        [TestMethod]
        public void Call_ThreeArgFunc_Blends()
        {
            Func<double, double, double, double> blend = (a, b, t) => a * (1 - t) + b * t;

            var a = np.array(new double[] { 0, 0, 0 });
            var b = np.array(new double[] { 10, 10, 10 });
            var t = np.array(new double[] { 0.0, 0.5, 1.0 });
            var r = np.empty_like(a);

            using var it = NpyIterRef.MultiNew(4, new[] { a, b, t, r },
                NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY,
                        NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });

            var expr = NpyExpr.Call(blend, NpyExpr.Input(0), NpyExpr.Input(1), NpyExpr.Input(2));
            it.ExecuteExpression(expr,
                new[] { NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double },
                NPTypeCode.Double, cacheKey: "call_blend_v1");

            Assert.AreEqual(0.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(5.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(10.0, r.GetDouble(2), 1e-9);
        }

        [TestMethod]
        public void Call_FourArgFunc_LerpWithClamp()
        {
            Func<double, double, double, double, double> quad = (a, b, c, d) => a * b + c * d;

            var ar = np.array(new double[] { 1, 2 });
            var br = np.array(new double[] { 3, 4 });
            var cr = np.array(new double[] { 5, 6 });
            var dr = np.array(new double[] { 7, 8 });
            var r = np.empty_like(ar);

            using var it = NpyIterRef.MultiNew(5, new[] { ar, br, cr, dr, r },
                NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY,
                        NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY,
                        NpyIterPerOpFlags.WRITEONLY });

            var expr = NpyExpr.Call(quad,
                NpyExpr.Input(0), NpyExpr.Input(1), NpyExpr.Input(2), NpyExpr.Input(3));
            it.ExecuteExpression(expr,
                new[] { NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double },
                NPTypeCode.Double, cacheKey: "call_4arg_v1");

            Assert.AreEqual(1 * 3 + 5 * 7, r.GetDouble(0), 1e-9);
            Assert.AreEqual(2 * 4 + 6 * 8, r.GetDouble(1), 1e-9);
        }

        // =====================================================================
        // Float32 (MathF) overload
        // =====================================================================

        [TestMethod]
        public void Call_MathF_Float32_NoTypeConversion()
        {
            var a = np.array(new float[] { 1f, 4f, 9f, 16f });
            var r = np.empty_like(a);
            using var it = Iter(a, r);

            var expr = NpyExpr.Call(MathF.Sqrt, NpyExpr.Input(0));
            it.ExecuteExpression(expr, new[] { NPTypeCode.Single }, NPTypeCode.Single,
                cacheKey: "call_mathf_sqrt_v1");

            Assert.AreEqual(1f, r.GetSingle(0), 1e-6f);
            Assert.AreEqual(2f, r.GetSingle(1), 1e-6f);
            Assert.AreEqual(3f, r.GetSingle(2), 1e-6f);
            Assert.AreEqual(4f, r.GetSingle(3), 1e-6f);
        }
    }

    // =========================================================================
    // Helper types for MethodInfo-based tests
    // =========================================================================

    internal static class StaticHelpers
    {
        public static double DoubleIt(double x) => x * 2;
        public static void VoidMethod(double x) { /* no return */ }
        public static int StringLength(string s) => s.Length;  // unsupported param type
    }

    internal sealed class Multiplier
    {
        public double Factor { get; set; } = 2.0;
        public double Apply(double x) => x * Factor;
    }

    internal sealed class BinaryCalc
    {
        public double Offset { get; set; }
        public double Combine(double a, double b) => a + b + Offset;
    }

    internal sealed class Counter
    {
        public int Count;
        public double IncrementAndAdd(double x) => ++Count + x;
    }
}
