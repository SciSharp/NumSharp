using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// np.evaluate — fused expression evaluation (roadmap Wave 6.1).
    ///
    /// Every dtype/value expectation below is pinned to NumPy 2.4.2 output
    /// (probe session in the Wave 6.1 notes): per-node result_type semantics
    /// (NEP50 + weak python-scalar literals), the special ufunc resolvers
    /// (true_divide → f64 for ints, arctan2 → tier float, power bool→int8),
    /// unary float-promotion tiers, reduction dtypes, and exact error texts.
    /// </summary>
    [TestClass]
    public class NDEvaluateTests
    {
        // =====================================================================
        // Per-node typing — THE semantic pin
        // =====================================================================

        [TestMethod]
        public void PerNode_Int32MultiplyWraps_BeforeFloatPromotion()
        {
            // NumPy: (np.int32(100000)*np.int32(100000)) + 0.5 → the multiply
            // runs in the int32 loop and WRAPS (1410065408), THEN promotes:
            // array([1.41006541e+09]) float64. Computing the whole tree at
            // the output dtype (the legacy DSL contract) would give 1e10.
            var a = np.array(new int[] { 100000 });
            var b = np.array(new int[] { 100000 });
            var c = np.array(new double[] { 0.5 });

            var r = np.evaluate((NDExpr)a * b + c);

            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.AreEqual(1410065408.5, r.GetDouble(0), 1e-6);
        }

        [TestMethod]
        public void Dtypes_SpecialUfuncResolvers()
        {
            var i1 = np.ones(new Shape(2), np.int8);
            var i2 = np.ones(new Shape(2), np.int16);
            var i4 = np.ones(new Shape(2), np.int32);
            var f4 = np.ones(new Shape(2), np.float32);

            // true_divide: ints/bool → float64 regardless of width
            Assert.AreEqual(NPTypeCode.Double, np.evaluate(NDExpr.Divide(NDExpr.Arr(i1), i1)).typecode);
            Assert.AreEqual(NPTypeCode.Double, np.evaluate(NDExpr.Divide(NDExpr.Arr(i4), i4)).typecode);

            // arctan2: tier float (i1→f16, i4→f64) — unlike divide's flat f64
            Assert.AreEqual(NPTypeCode.Half, np.evaluate(NDExpr.ATan2(NDExpr.Arr(i1), i1)).typecode);
            Assert.AreEqual(NPTypeCode.Double, np.evaluate(NDExpr.ATan2(NDExpr.Arr(i4), i4)).typecode);

            // power: plain promotion for ints; NEP50 i4+f4 crossing → f64
            Assert.AreEqual(NPTypeCode.Int32, np.evaluate(NDExpr.Power(NDExpr.Arr(i4), i4)).typecode);
            Assert.AreEqual(NPTypeCode.Double, np.evaluate(NDExpr.Power(NDExpr.Arr(f4), i4)).typecode);

            // unary float tiers: bool/i8→f16, i16→f32, i32+→f64
            Assert.AreEqual(NPTypeCode.Half, np.evaluate(NDExpr.Sqrt(NDExpr.Arr(i1))).typecode);
            Assert.AreEqual(NPTypeCode.Single, np.evaluate(NDExpr.Sqrt(NDExpr.Arr(i2))).typecode);
            Assert.AreEqual(NPTypeCode.Double, np.evaluate(NDExpr.Sqrt(NDExpr.Arr(i4))).typecode);

            // minimum follows plain NEP50 promotion: i4+f4 → f64
            Assert.AreEqual(NPTypeCode.Double, np.evaluate(NDExpr.Min(NDExpr.Arr(i4), NDExpr.Arr(f4))).typecode);
        }

        [TestMethod]
        public void Dtypes_BoolUfuncQuirks()
        {
            var b = np.array(new bool[] { true, true });

            // add(bool,bool) is logical OR; multiply is logical AND
            var sum = np.evaluate(NDExpr.Add(NDExpr.Arr(b), NDExpr.Arr(b)));
            Assert.AreEqual(NPTypeCode.Boolean, sum.typecode);
            Assert.IsTrue(sum.GetBoolean(0)); // True+True → True, not byte 2

            // power/remainder/floor_divide(bool,bool) → int8
            Assert.AreEqual(NPTypeCode.SByte, np.evaluate(NDExpr.Power(NDExpr.Arr(b), NDExpr.Arr(b))).typecode);
            var mod = np.evaluate(NDExpr.Mod(NDExpr.Arr(b), NDExpr.Arr(b)));
            Assert.AreEqual(NPTypeCode.SByte, mod.typecode);
            Assert.AreEqual((sbyte)0, mod.GetSByte(0)); // remainder(True,True) = 0
            var fdiv = np.evaluate(NDExpr.FloorDivide(NDExpr.Arr(b), NDExpr.Arr(b)));
            Assert.AreEqual(NPTypeCode.SByte, fdiv.typecode);
            Assert.AreEqual((sbyte)1, fdiv.GetSByte(0)); // floor_divide(True,True) = 1

            // square(bool) → int8; abs/invert preserve bool
            Assert.AreEqual(NPTypeCode.SByte, np.evaluate(NDExpr.Square(NDExpr.Arr(b))).typecode);
            Assert.AreEqual(NPTypeCode.Boolean, np.evaluate(NDExpr.Abs(NDExpr.Arr(b))).typecode);
            var inv = np.evaluate(NDExpr.BitwiseNot(NDExpr.Arr(b)));
            Assert.AreEqual(NPTypeCode.Boolean, inv.typecode);
            Assert.IsFalse(inv.GetBoolean(0)); // ~True → False (not byte 254)
        }

        [TestMethod]
        public void Dtypes_BoolErrors_MatchNumPyTexts()
        {
            var b = np.array(new bool[] { true });

            var negEx = Assert.ThrowsException<NotSupportedException>(
                () => np.evaluate(NDExpr.Negate(NDExpr.Arr(b))));
            StringAssert.Contains(negEx.Message, "numpy boolean negative");

            var subEx = Assert.ThrowsException<NotSupportedException>(
                () => np.evaluate(NDExpr.Subtract(NDExpr.Arr(b), NDExpr.Arr(b))));
            StringAssert.Contains(subEx.Message, "numpy boolean subtract");
        }

        [TestMethod]
        public void Dtypes_InvertAndBitwise_FloatInputsRaise()
        {
            var f4 = np.ones(new Shape(2), np.float32);

            var invEx = Assert.ThrowsException<NotSupportedException>(
                () => np.evaluate(NDExpr.BitwiseNot(NDExpr.Arr(f4))));
            StringAssert.Contains(invEx.Message, "ufunc 'invert' not supported for the input types");

            var andEx = Assert.ThrowsException<NotSupportedException>(
                () => np.evaluate(NDExpr.BitwiseAnd(NDExpr.Arr(f4), NDExpr.Arr(f4))));
            StringAssert.Contains(andEx.Message, "ufunc 'bitwise_and' not supported for the input types");
        }

        // =====================================================================
        // NEP50 weak scalars
        // =====================================================================

        [TestMethod]
        public void WeakScalars_AdoptArrayDtype()
        {
            var i4 = np.arange(4).astype(np.int32);
            var f4 = np.array(new float[] { 1.5f });
            var f2 = np.ones(new Shape(2), np.float16);
            var u1 = np.ones(new Shape(3), np.uint8);
            var b1 = np.array(new bool[] { true });

            Assert.AreEqual(NPTypeCode.Int32, np.evaluate((NDExpr)i4 + 2).typecode);       // i4+2 → i4
            Assert.AreEqual(NPTypeCode.Double, np.evaluate((NDExpr)i4 + 2.5).typecode);    // i4+2.5 → f8
            Assert.AreEqual(NPTypeCode.Double, np.evaluate((NDExpr)i4 / 2).typecode);      // i4/2 → f8
            Assert.AreEqual(NPTypeCode.Int32, np.evaluate(NDExpr.Power(NDExpr.Arr(i4), NDExpr.Const(2))).typecode);
            Assert.AreEqual(NPTypeCode.Single, np.evaluate((NDExpr)f4 + 2).typecode);      // f4+2 → f4
            Assert.AreEqual(NPTypeCode.Single, np.evaluate((NDExpr)f4 + 2.5).typecode);    // f4+2.5 → f4
            Assert.AreEqual(NPTypeCode.Half, np.evaluate((NDExpr)f2 + 2.5).typecode);      // f2+2.5 → f2
            Assert.AreEqual(NPTypeCode.Byte, np.evaluate((NDExpr)u1 + 2).typecode);        // u1+2 → u1
            Assert.AreEqual(NPTypeCode.Int64, np.evaluate((NDExpr)b1 + 2).typecode);       // bool+2 → i64
            Assert.AreEqual(NPTypeCode.Double, np.evaluate((NDExpr)b1 + 2.5).typecode);    // bool+2.5 → f8
        }

        [TestMethod]
        public void WeakScalars_OutOfBoundsLiteral_RaisesOverflow()
        {
            var u1 = np.ones(new Shape(3), np.uint8);
            var ex = Assert.ThrowsException<OverflowException>(() => np.evaluate((NDExpr)u1 + 300));
            Assert.AreEqual("Python integer 300 out of bounds for uint8", ex.Message);
        }

        [TestMethod]
        public void Power_NegativeIntegerLiteralExponent_RaisesLikeNumPy()
        {
            var i4 = np.ones(new Shape(2), np.int32);
            var ex = Assert.ThrowsException<ArgumentException>(
                () => np.evaluate(NDExpr.Power(NDExpr.Arr(i4), NDExpr.Const(-2))));
            StringAssert.Contains(ex.Message, "Integers to negative integer powers are not allowed.");
        }

        // =====================================================================
        // Values: fusion correctness
        // =====================================================================

        [TestMethod]
        public void Fused_NormalizedDifference_MatchesUnfused()
        {
            var a = np.array(new double[] { 4, 9, 16, 25 });
            var b = np.array(new double[] { 2, 3, 4, 5 });

            // (a-b)/(a+b) — a and b each appear twice; binding dedups to 3 operands.
            var fused = np.evaluate((NDExpr.Arr(a) - b) / (NDExpr.Arr(a) + b));
            var unfused = (a - b) / (a + b);

            for (int i = 0; i < 4; i++)
                Assert.AreEqual(unfused.GetDouble(i), fused.GetDouble(i), 1e-15, $"[{i}]");
        }

        [TestMethod]
        public void Fused_ComparisonProducesBool_AndComposesIntoArithmetic()
        {
            var a = np.array(new double[] { 1, 5, 3 });

            var mask = np.evaluate(NDExpr.Greater(NDExpr.Arr(a), NDExpr.Const(2.0)));
            Assert.AreEqual(NPTypeCode.Boolean, mask.typecode);
            Assert.IsFalse(mask.GetBoolean(0));
            Assert.IsTrue(mask.GetBoolean(1));

            // a * (a > 2): bool*f8 → f8 (NumPy result_type)
            var relu = np.evaluate((NDExpr)a * NDExpr.Greater(NDExpr.Arr(a), 2.0));
            Assert.AreEqual(NPTypeCode.Double, relu.typecode);
            Assert.AreEqual(0.0, relu.GetDouble(0));
            Assert.AreEqual(5.0, relu.GetDouble(1));
            Assert.AreEqual(3.0, relu.GetDouble(2));
        }

        [TestMethod]
        public void Fused_WhereNode_PromotesBranches_TestsCondAtOwnDtype()
        {
            // where(b, i4, f4) → result_type(i4,f4) = f64 (NumPy probed)
            var cond = np.array(new bool[] { true, false });
            var i4 = np.array(new int[] { 1, 2 });
            var f4 = np.array(new float[] { 10f, 20f });

            var r = np.evaluate(NDExpr.Where(NDExpr.Arr(cond), NDExpr.Arr(i4), NDExpr.Arr(f4)));
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.AreEqual(1.0, r.GetDouble(0));
            Assert.AreEqual(20.0, r.GetDouble(1));

            // nonzero test happens at the condition's own dtype (int cond works)
            var icond = np.array(new int[] { 0, 7 });
            var r2 = np.evaluate(NDExpr.Where(NDExpr.Arr(icond), NDExpr.Arr(f4), NDExpr.Const(0.0f)));
            Assert.AreEqual(0.0f, r2.GetSingle(0), 0f);
            Assert.AreEqual(20.0f, r2.GetSingle(1), 0f);
        }

        [TestMethod]
        public void Fused_BroadcastInputs()
        {
            var a = np.arange(6).reshape(2, 3).astype(np.float64);
            var row = np.array(new double[] { 10, 20, 30 });

            var r = np.evaluate((NDExpr)a + row);

            Assert.AreEqual(2L, (long)r.shape[0]);
            Assert.AreEqual(3L, (long)r.shape[1]);
            Assert.AreEqual(35.0, r.GetDouble(1, 2));
        }

        [TestMethod]
        public void Fused_StridedAndTransposedInputs()
        {
            var big = np.arange(40).astype(np.float64);
            var even = big["::2"];   // strided view, 20 elements
            var odd = big["1::2"];

            var r = np.evaluate((NDExpr)even + odd);
            for (int i = 0; i < 20; i++)
                Assert.AreEqual(4.0 * i + 1.0, r.GetDouble(i), 0.0, $"[{i}]");

            var m = np.arange(6).reshape(2, 3).astype(np.float64);
            var t = m.T; // (3,2) transposed view
            var rt = np.evaluate((NDExpr)t * 2.0);
            Assert.AreEqual(NPTypeCode.Double, rt.typecode);
            Assert.AreEqual(8.0, rt.GetDouble(1, 1)); // m[1,1]=4 → t[1,1]=4 → 8
        }

        [TestMethod]
        public void Fused_PositionalOverload_AndBindingValidation()
        {
            var a = np.array(new double[] { 1, 2 });
            var b = np.array(new double[] { 10, 20 });

            var r = np.evaluate(NDExpr.Input(0) * NDExpr.Input(1), new[] { a, b });
            Assert.AreEqual(40.0, r.GetDouble(1));

            // constant-only tree → no arrays to iterate
            Assert.ThrowsException<ArgumentException>(() => np.evaluate(NDExpr.Const(2) + NDExpr.Const(3)));

            // mixing embedded leaves with a positional list
            Assert.ThrowsException<ArgumentException>(
                () => np.evaluate(NDExpr.Arr(a) * NDExpr.Input(0), new[] { b }));
        }

        // =====================================================================
        // out=
        // =====================================================================

        [TestMethod]
        public void Out_ReferenceIdentity_Cast_AndAliasing()
        {
            var a = np.array(new double[] { 1, 2, 3 });
            var b = np.array(new double[] { 10, 20, 30 });

            var outArr = np.zeros(new Shape(3), np.float64);
            var r = np.evaluate((NDExpr)a + b, @out: outArr);
            Assert.IsTrue(ReferenceEquals(r, outArr));
            Assert.AreEqual(33.0, outArr.GetDouble(2));

            // same_kind cast f8 → f4 through the buffered flush
            var outF4 = np.zeros(new Shape(3), np.float32);
            np.evaluate((NDExpr)a + b, @out: outF4);
            Assert.AreEqual(22f, outF4.GetSingle(1));

            // out aliasing an input is overlap-safe (COPY_IF_OVERLAP)
            var x = np.array(new double[] { 1, 2, 3 });
            np.evaluate((NDExpr)x * 2 + 1, @out: x);
            Assert.AreEqual(3.0, x.GetDouble(0));
            Assert.AreEqual(5.0, x.GetDouble(1));
            Assert.AreEqual(7.0, x.GetDouble(2));
        }

        [TestMethod]
        public void Out_InvalidCast_RaisesSameKindText()
        {
            var a = np.array(new double[] { 1.5, 2.5 });
            var outI4 = np.zeros(new Shape(2), np.int32);

            var ex = Assert.ThrowsException<ArgumentException>(
                () => np.evaluate((NDExpr)a + 1.0, @out: outI4));
            StringAssert.Contains(ex.Message, "Cannot cast ufunc 'evaluate' output");
            StringAssert.Contains(ex.Message, "same_kind");
        }

        // =====================================================================
        // Reductions
        // =====================================================================

        [TestMethod]
        public void Reduce_SumOfProduct_OnePass()
        {
            var a = np.array(new double[] { 1, 2, 3, 4 });
            var b = np.array(new double[] { 10, 20, 30, 40 });

            var s = np.evaluate(NDExpr.Sum((NDExpr)a * b));
            Assert.AreEqual(0, s.ndim);
            Assert.AreEqual(300.0, s.GetDouble(0), 1e-12);
        }

        [TestMethod]
        public void Reduce_DtypeRules()
        {
            var i4 = np.arange(5).astype(np.int32);
            var u1 = np.ones(new Shape(4), np.uint8);
            var f4 = np.array(new float[] { 1f, 2f, 3f });
            var f2 = np.ones(new Shape(3), np.float16);

            Assert.AreEqual(NPTypeCode.Int64, np.evaluate(NDExpr.Sum(NDExpr.Arr(i4))).typecode);
            Assert.AreEqual(10L, np.evaluate(NDExpr.Sum(NDExpr.Arr(i4))).GetInt64(0));
            Assert.AreEqual(NPTypeCode.UInt64, np.evaluate(NDExpr.Sum(NDExpr.Arr(u1))).typecode);
            Assert.AreEqual(NPTypeCode.Single, np.evaluate(NDExpr.Sum(NDExpr.Arr(f4))).typecode);
            Assert.AreEqual(NPTypeCode.Half, np.evaluate(NDExpr.Sum(NDExpr.Arr(f2))).typecode);
            Assert.AreEqual(NPTypeCode.Int64, np.evaluate(NDExpr.Prod(NDExpr.Arr(i4))).typecode);
            Assert.AreEqual(NPTypeCode.Int32, np.evaluate(NDExpr.Min(NDExpr.Arr(i4))).typecode);
            Assert.AreEqual(NPTypeCode.Double, np.evaluate(NDExpr.Mean(NDExpr.Arr(i4))).typecode);
            Assert.AreEqual(2.0, np.evaluate(NDExpr.Mean(NDExpr.Arr(i4))).GetDouble(0), 1e-12);
            Assert.AreEqual(NPTypeCode.Single, np.evaluate(NDExpr.Mean(NDExpr.Arr(f4))).typecode);
            Assert.AreEqual(2f, np.evaluate(NDExpr.Mean(NDExpr.Arr(f4))).GetSingle(0), 1e-6f);

            // int8 sum promotes before accumulating: 127+127 = 254, no wrap
            var i1 = np.array(new sbyte[] { 127, 127 });
            Assert.AreEqual(254L, np.evaluate(NDExpr.Sum(NDExpr.Arr(i1))).GetInt64(0));
        }

        // =====================================================================
        // Axis-aware fused reductions (Phase 5a) — one pass, no a*b temp.
        // Expected values from NumPy 2.4.2: np.<op>((a*b), axis=k).
        // =====================================================================

        [TestMethod]
        public void Reduce_Axis_SumOfProduct_BothAxes()
        {
            // a = arange(12).reshape(3,4); evaluate(Sum(a*a, axis)) == np.sum(a*a, axis)
            var a = np.arange(12).astype(np.float64).reshape(3, 4);
            var s0 = np.evaluate(NDExpr.Sum((NDExpr)a * a, 0));
            Assert.AreEqual(1, s0.ndim);
            Assert.AreEqual(4, (int)s0.size);
            double[] exp0 = { 80, 107, 140, 179 };       // NumPy
            for (long i = 0; i < 4; i++) Assert.AreEqual(exp0[i], s0.GetDouble(i), 1e-9);

            var s1 = np.evaluate(NDExpr.Sum((NDExpr)a * a, 1));
            double[] exp1 = { 14, 126, 366 };            // NumPy
            for (long i = 0; i < 3; i++) Assert.AreEqual(exp1[i], s1.GetDouble(i), 1e-9);
        }

        [TestMethod]
        public void Reduce_Axis_Mean_And_Keepdims()
        {
            var a = np.arange(12).astype(np.float64).reshape(3, 4);
            var m = np.evaluate(NDExpr.Mean((NDExpr)a * a, 1));
            double[] expMean1 = { 3.5, 31.5, 91.5 };     // NumPy np.mean(a*a, axis=1)
            for (long i = 0; i < 3; i++) Assert.AreEqual(expMean1[i], m.GetDouble(i), 1e-9);

            var mk = np.evaluate(NDExpr.Mean((NDExpr)a * a, 1, keepdims: true));
            Assert.AreEqual(2, mk.ndim);
            Assert.AreEqual(3, (int)mk.shape[0]);
            Assert.AreEqual(1, (int)mk.shape[1]);
        }

        [TestMethod]
        public void Reduce_Axis_MatchesUnfused_AcrossLayouts()
        {
            // Fused must equal the unfused np.<op>(a*b, axis) on C / transpose / F layouts.
            var aBase = np.arange(35).astype(np.float64).reshape(7, 5) * 0.3 - 5.0;
            var bBase = np.arange(35).astype(np.float64).reshape(7, 5) * -0.2 + 1.0;
            var layouts = new (NDArray a, NDArray b)[]
            {
                (aBase, bBase),
                (aBase.T, bBase.T),
                (aBase.copy(order: 'F'), bBase.copy(order: 'F')),
            };
            foreach (var (a, b) in layouts)
                for (int axis = 0; axis < 2; axis++)
                {
                    var fused = np.evaluate(NDExpr.Sum((NDExpr)a * b, axis));
                    var unfused = np.sum(a * b, axis: axis);
                    Assert.AreEqual(unfused.size, fused.size);
                    for (long i = 0; i < unfused.size; i++)
                        Assert.AreEqual(unfused.GetDouble(i), fused.GetDouble(i), 1e-9, $"axis {axis} [{i}]");
                }
        }

        [TestMethod]
        public void Reduce_Axis_DtypeRules()
        {
            // sum int32 axis → int64; mean int32 axis → float64 (NumPy).
            var i4 = np.arange(6).astype(np.int32).reshape(2, 3);
            var s = np.evaluate(NDExpr.Sum((NDExpr)i4 * i4, 0));
            Assert.AreEqual(NPTypeCode.Int64, s.typecode);
            // (a*a) = [[0,1,4],[9,16,25]]; sum axis0 = [9,17,29]
            Assert.AreEqual(9L, s.GetInt64(0));
            Assert.AreEqual(17L, s.GetInt64(1));
            Assert.AreEqual(29L, s.GetInt64(2));

            var m = np.evaluate(NDExpr.Mean((NDExpr)i4 * i4, 1));
            Assert.AreEqual(NPTypeCode.Double, m.typecode);
            // rows of a*a: [0,1,4]→5/3, [9,16,25]→50/3
            Assert.AreEqual(5.0 / 3.0, m.GetDouble(0), 1e-9);
            Assert.AreEqual(50.0 / 3.0, m.GetDouble(1), 1e-9);
        }

        [TestMethod]
        public void Reduce_MultiChunk_Strided_AndNaN()
        {
            // 20005 elements forces multiple EXTERNAL_LOOP chunks
            var big = np.arange(20005).astype(np.float64);
            var s = np.evaluate(NDExpr.Sum(NDExpr.Arr(big)));
            Assert.AreEqual(20004.0 * 20005.0 / 2.0, s.GetDouble(0), 1e-6);

            var strided = big["::2"];
            double expect = 0;
            for (int i = 0; i < 20005; i += 2) expect += i;
            Assert.AreEqual(expect, np.evaluate(NDExpr.Sum(NDExpr.Arr(strided))).GetDouble(0), 1e-6);

            // NaN propagates through min/max like np.min/np.max
            var nan = np.array(new double[] { double.NaN, 1.0, -5.0 });
            Assert.IsTrue(double.IsNaN(np.evaluate(NDExpr.Min(NDExpr.Arr(nan))).GetDouble(0)));
            Assert.IsTrue(double.IsNaN(np.evaluate(NDExpr.Max(NDExpr.Arr(nan))).GetDouble(0)));

            // f16 accumulation overflows to inf exactly like np.sum(f2)
            var ones = np.ones(new Shape(70000), np.float16);
            var f2sum = np.evaluate(NDExpr.Sum(NDExpr.Arr(ones)));
            Assert.AreEqual(NPTypeCode.Half, f2sum.typecode);
            Assert.IsTrue(Half.IsPositiveInfinity(f2sum.GetHalf(0)));
        }

        [TestMethod]
        public void Reduce_EmptyInputs_MatchNumPyIdentities()
        {
            var empty = np.array(new double[0]);

            Assert.AreEqual(0.0, np.evaluate(NDExpr.Sum(NDExpr.Arr(empty))).GetDouble(0));
            Assert.AreEqual(1.0, np.evaluate(NDExpr.Prod(NDExpr.Arr(empty))).GetDouble(0));

            var ex = Assert.ThrowsException<ArgumentException>(
                () => np.evaluate(NDExpr.Min(NDExpr.Arr(empty))));
            StringAssert.Contains(ex.Message, "zero-size array to reduction operation minimum which has no identity");

            // mean([]) → NaN at the result dtype (np.mean of empty f4 → float32 nan)
            var emptyF4 = np.array(new float[0]);
            var m = np.evaluate(NDExpr.Mean(NDExpr.Arr(emptyF4)));
            Assert.AreEqual(NPTypeCode.Single, m.typecode);
            Assert.IsTrue(float.IsNaN(m.GetSingle(0)));
        }

        [TestMethod]
        public void Reduce_MustBeRoot()
        {
            var a = np.array(new double[] { 1, 2 });

            Assert.ThrowsException<NotSupportedException>(
                () => np.evaluate(NDExpr.Sum(NDExpr.Sum((NDExpr)a * 2))));
            Assert.ThrowsException<NotSupportedException>(
                () => np.evaluate(NDExpr.Sum((NDExpr)a) + 1.0));
        }

        // =====================================================================
        // Tier-3C contract
        // =====================================================================

        [TestMethod]
        public unsafe void ExecuteExpression_LegacyOutputDtypeContract_Unchanged()
        {
            var input = np.array(new double[] { 4.0, 9.0 });
            var output = np.empty_like(input);
            using var iter = NDIterRef.MultiNew(2, new[] { input, output },
                NDIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY });

            iter.ExecuteExpression(NDExpr.Sqrt(NDExpr.Input(0)), new[] { NPTypeCode.Double }, NPTypeCode.Double);

            Assert.AreEqual(2.0, output.GetDouble(0));
            Assert.AreEqual(3.0, output.GetDouble(1));
        }

        [TestMethod]
        public unsafe void ExecuteExpression_WithoutExternalLoop_ThrowsFootgunGuard()
        {
            // A strided operand defeats coalescing/ONEITERATION, so without
            // EXTERNAL_LOOP this iterator would advance per element — the
            // measured ~40× foot-gun the guard exists for.
            var input = np.arange(40).astype(np.float64)["::2"];
            var output = np.zeros(new Shape(20), np.float64);
            using var iter = NDIterRef.MultiNew(2, new[] { input, output },
                NDIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY });

            try
            {
                iter.ExecuteExpression(NDExpr.Sqrt(NDExpr.Input(0)),
                    new[] { NPTypeCode.Double }, NPTypeCode.Double);
                Assert.Fail("expected the EXTERNAL_LOOP foot-gun guard to throw");
            }
            catch (InvalidOperationException ex)
            {
                StringAssert.Contains(ex.Message, "EXTERNAL_LOOP");
            }
        }
    }
}
