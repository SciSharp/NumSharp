using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Backends.Iterators
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

        // =====================================================================
        // P2 M4 — presence / count / NaN-aware reductions
        // (any / all / count_nonzero / nansum / nanprod), probed against NumPy 2.4.2
        // =====================================================================

        [TestMethod]
        public void M4_AnyAllCountNonzero_ValuesAndDtypes()
        {
            var a = np.array(new double[] { 0, 1, 2, 0, 3 });

            var any = np.evaluate(NDExpr.Any((NDExpr)a));
            Assert.AreEqual(NPTypeCode.Boolean, any.typecode);   // np.any → bool
            Assert.IsTrue(any.GetBoolean(0));                    // a nonzero present

            var all = np.evaluate(NDExpr.All((NDExpr)a));
            Assert.AreEqual(NPTypeCode.Boolean, all.typecode);
            Assert.IsFalse(all.GetBoolean(0));                   // zeros present

            var cnz = np.evaluate(NDExpr.CountNonzero((NDExpr)a));
            Assert.AreEqual(NPTypeCode.Int64, cnz.typecode);     // np.count_nonzero → intp (int64)
            Assert.AreEqual(3L, cnz.GetInt64(0));
        }

        [TestMethod]
        public void M4_Any_NaNIsTruthy()
        {
            // np.any([nan]) == True, np.all([nan]) == True (NaN is truthy); count_nonzero counts it.
            var f = np.array(new double[] { double.NaN, 0.0 });
            Assert.IsTrue(np.evaluate(NDExpr.Any((NDExpr)f)).GetBoolean(0));
            Assert.IsFalse(np.evaluate(NDExpr.All((NDExpr)f)).GetBoolean(0)); // the 0.0 is falsy
            Assert.AreEqual(1L, np.evaluate(NDExpr.CountNonzero((NDExpr)f)).GetInt64(0));

            var allNaN = np.array(new double[] { double.NaN, double.NaN });
            Assert.IsTrue(np.evaluate(NDExpr.All((NDExpr)allNaN)).GetBoolean(0)); // both truthy
        }

        [TestMethod]
        public void M4_NanSumProd_SkipNaN_PreserveDtype()
        {
            // nansum treats NaN as 0, nanprod treats it as 1 — bit-exact via the M1 Sum/Prod divert.
            var f = np.array(new double[] { 1.0, double.NaN, 2.0, 0.0 });
            var ns = np.evaluate(NDExpr.NanSum((NDExpr)f));
            Assert.AreEqual(NPTypeCode.Double, ns.typecode);
            Assert.AreEqual(3.0, ns.GetDouble(0), 0.0);          // 1 + 0(nan) + 2 + 0
            Assert.AreEqual(0.0, np.evaluate(NDExpr.NanProd((NDExpr)f)).GetDouble(0), 0.0); // 1*1(nan)*2*0

            // A fused child: nansum(f*f) skips the NaN in the product.
            Assert.AreEqual(5.0, np.evaluate(NDExpr.NanSum((NDExpr)f * f)).GetDouble(0), 0.0); // 1 + 4

            // Integer child carries no NaN → nansum is exactly sum, and stays int64 (NEP50).
            var i = np.array(new int[] { 0, 5, 0, 7 });
            var nsi = np.evaluate(NDExpr.NanSum((NDExpr)i));
            Assert.AreEqual(NPTypeCode.Int64, nsi.typecode);
            Assert.AreEqual(12L, nsi.GetInt64(0));
        }

        [TestMethod]
        public void M4_EmptyInput_Identities()
        {
            // The identity is what separates any/all from max/min: max/min RAISE on a zero-size input,
            // but any([]) == False, all([]) == True, count_nonzero([]) == 0, nansum([]) == 0,
            // nanprod([]) == 1 (probed 2.4.2). This is the case the fuzz corpus does not reach.
            var e = np.array(new double[] { });
            Assert.IsFalse(np.evaluate(NDExpr.Any((NDExpr)e)).GetBoolean(0));
            Assert.IsTrue(np.evaluate(NDExpr.All((NDExpr)e)).GetBoolean(0));
            Assert.AreEqual(0L, np.evaluate(NDExpr.CountNonzero((NDExpr)e)).GetInt64(0));
            Assert.AreEqual(0.0, np.evaluate(NDExpr.NanSum((NDExpr)e)).GetDouble(0), 0.0);
            Assert.AreEqual(1.0, np.evaluate(NDExpr.NanProd((NDExpr)e)).GetDouble(0), 0.0);
        }

        [TestMethod]
        public void M4_Axis_Forms()
        {
            var m = np.array(new double[] { 0, 1, 2, 0, 0, 0 }).reshape(3, 2); // [[0,1],[2,0],[0,0]]

            var any1 = np.evaluate(NDExpr.Any((NDExpr)m, 1));   // per row: [T, T, F]
            Assert.AreEqual(NPTypeCode.Boolean, any1.typecode);
            Assert.IsTrue(any1.GetBoolean(0));
            Assert.IsTrue(any1.GetBoolean(1));
            Assert.IsFalse(any1.GetBoolean(2));

            var cnz0 = np.evaluate(NDExpr.CountNonzero((NDExpr)m, 0)); // per col: [1, 1]
            Assert.AreEqual(NPTypeCode.Int64, cnz0.typecode);
            Assert.AreEqual(1L, cnz0.GetInt64(0));
            Assert.AreEqual(1L, cnz0.GetInt64(1));

            // nansum along each axis, NaN-skipped: g = [[1,nan],[2,3]]
            var g = np.array(new double[] { 1, double.NaN, 2, 3 }).reshape(2, 2);
            var ns0 = np.evaluate(NDExpr.NanSum((NDExpr)g, 0)); // cols: [1+2, 0+3] = [3, 3]
            Assert.AreEqual(3.0, ns0.GetDouble(0), 0.0);
            Assert.AreEqual(3.0, ns0.GetDouble(1), 0.0);
            var ns1 = np.evaluate(NDExpr.NanSum((NDExpr)g, 1)); // rows: [1+0, 2+3] = [1, 5]
            Assert.AreEqual(1.0, ns1.GetDouble(0), 0.0);
            Assert.AreEqual(5.0, ns1.GetDouble(1), 0.0);
        }

        // =====================================================================
        // P2 M4-tail — the ORDER-INDEPENDENT range / NaN-aware min-max family
        // (ptp / nanmin / nanmax), host-delegated to np.ptp/np.nanmin/np.nanmax
        // over the materialized child. Probed against NumPy 2.4.2.
        // =====================================================================

        /// <summary>
        /// <c>Ptp</c> is <c>max - min</c> at the CHILD dtype, so an integer result WRAPS exactly as
        /// NumPy's does — the case the fuzz corpus covers only at 32/64-bit widths. Pins the two narrow
        /// widths where the wrap is visible in a single byte, plus dtype preservation.
        /// </summary>
        [TestMethod]
        public void M4Tail_Ptp_WrapsAtChildDtype()
        {
            // 127 - (-128) = 255, which wraps int8 to -1 (np.ptp(int8[-128,127]) == -1).
            var i8 = np.array(new sbyte[] { -128, 0, 127 });
            var p8 = np.evaluate(NDExpr.Ptp((NDExpr)i8));
            Assert.AreEqual(NPTypeCode.SByte, p8.typecode);      // dtype preserved (not widened)
            // 255 wrapped into int8 → -1. Read the raw byte (GetInt32 would reinterpret 4 bytes off a
            // 1-byte cell) and reinterpret it as the sbyte it is.
            Assert.AreEqual((sbyte)(-1), (sbyte)p8.GetByte(0));

            // uint8 range is exact: 255 - 0 = 255.
            var u8 = np.array(new byte[] { 0, 128, 255 });
            var pu = np.evaluate(NDExpr.Ptp((NDExpr)u8));
            Assert.AreEqual(NPTypeCode.Byte, pu.typecode);
            Assert.AreEqual((byte)255, pu.GetByte(0));

            // float64: an ordinary finite range, dtype preserved.
            var f = np.array(new double[] { 3, -7, 1.5, 5, 0 });
            var pf = np.evaluate(NDExpr.Ptp((NDExpr)f));
            Assert.AreEqual(NPTypeCode.Double, pf.typecode);
            Assert.AreEqual(12.0, pf.GetDouble(0), 0.0);         // 5 - (-7)
        }

        /// <summary>
        /// A NaN anywhere in a <c>Ptp</c> child propagates to the result (both <c>amax</c> and
        /// <c>amin</c> propagate NaN, and <c>nan - nan == nan</c>), matching NumPy — the opposite of the
        /// NaN-SKIPPING nanmin/nanmax below.
        /// </summary>
        [TestMethod]
        public void M4Tail_Ptp_PropagatesNaN()
        {
            var g = np.array(new double[] { 1.0, double.NaN, 3.0, 2.0 });
            Assert.IsTrue(double.IsNaN(np.evaluate(NDExpr.Ptp((NDExpr)g)).GetDouble(0)));
        }

        /// <summary>
        /// <c>NanMin</c>/<c>NanMax</c> IGNORE NaN on a float child (so the extreme comes from the finite
        /// values) but preserve the child dtype, and an INTEGER child — which carries no NaN — folds as a
        /// plain minimum/maximum. This is the type-awareness the ±inf-sentinel tree rewrite could not give
        /// for free; it comes from delegating to the engine's np.nanmin/np.nanmax.
        /// </summary>
        [TestMethod]
        public void M4Tail_NanMinMax_SkipNaN_PreserveDtype()
        {
            var g = np.array(new double[] { 1.0, double.NaN, 3.0, 2.0, double.NaN, 5.0 });
            var mn = np.evaluate(NDExpr.NanMin((NDExpr)g));
            Assert.AreEqual(NPTypeCode.Double, mn.typecode);
            Assert.AreEqual(1.0, mn.GetDouble(0), 0.0);          // NaNs skipped → min of {1,3,2,5}
            Assert.AreEqual(5.0, np.evaluate(NDExpr.NanMax((NDExpr)g)).GetDouble(0), 0.0);

            // A fused child: nanmin over (g - 1) skips the NaN slots the subtract carries through.
            Assert.AreEqual(0.0, np.evaluate(NDExpr.NanMin((NDExpr)g - 1.0)).GetDouble(0), 0.0);

            // Integer child (no NaN possible) → plain min/max, int dtype preserved.
            var i = np.array(new int[] { 7, 2, 9, 4 });
            var mni = np.evaluate(NDExpr.NanMin((NDExpr)i));
            Assert.AreEqual(NPTypeCode.Int32, mni.typecode);
            Assert.AreEqual(2, mni.GetInt32(0));
            Assert.AreEqual(9, np.evaluate(NDExpr.NanMax((NDExpr)i)).GetInt32(0));
        }

        /// <summary>
        /// An ALL-NaN reduction yields NaN for nanmin/nanmax (NumPy's all-NaN-slice contract, which it
        /// warns on but still returns NaN), and ptp of an all-NaN input is likewise NaN — verified here
        /// because the fuzz corpus's C5 pools always keep a non-NaN value in every slice.
        /// </summary>
        [TestMethod]
        public void M4Tail_AllNaN_YieldsNaN()
        {
            var allNaN = np.array(new[] { double.NaN, double.NaN, double.NaN });
            Assert.IsTrue(double.IsNaN(np.evaluate(NDExpr.NanMin((NDExpr)allNaN)).GetDouble(0)));
            Assert.IsTrue(double.IsNaN(np.evaluate(NDExpr.NanMax((NDExpr)allNaN)).GetDouble(0)));
            Assert.IsTrue(double.IsNaN(np.evaluate(NDExpr.Ptp((NDExpr)allNaN)).GetDouble(0)));
        }

        /// <summary>
        /// Axis + keepdims forms reduce along one axis and (optionally) keep it as size 1, matching
        /// np.ptp/np.nanmin/np.nanmax along that axis, with NaNs skipped by nanmin/nanmax and propagated
        /// by ptp per row.
        /// </summary>
        [TestMethod]
        public void M4Tail_Axis_Forms()
        {
            // g = [[1, nan], [2, 3]]
            var g = np.array(new double[] { 1, double.NaN, 2, 3 }).reshape(2, 2);

            var mn0 = np.evaluate(NDExpr.NanMin((NDExpr)g, 0));   // cols: [min(1,2), min(nan,3)] = [1, 3]
            Assert.AreEqual(1.0, mn0.GetDouble(0), 0.0);
            Assert.AreEqual(3.0, mn0.GetDouble(1), 0.0);

            var mx1 = np.evaluate(NDExpr.NanMax((NDExpr)g, 1, keepdims: true)); // rows: [[1],[3]]
            Assert.AreEqual(2, mx1.ndim);
            Assert.AreEqual(1L, mx1.shape[1]);
            Assert.AreEqual(1.0, mx1.GetDouble(0), 0.0);         // row 0 has only the finite 1
            Assert.AreEqual(3.0, mx1.GetDouble(1), 0.0);

            // ptp along axis 1 of an int matrix: rows [0..3] and [4..7] → range 3 each, int dtype.
            var m = np.arange(8).reshape(2, 4);
            var pm = np.evaluate(NDExpr.Ptp((NDExpr)m, 1));
            Assert.AreEqual(3L, pm.GetInt64(0));
            Assert.AreEqual(3L, pm.GetInt64(1));
        }

        /// <summary>
        /// <c>out=</c> follows evaluate's ufunc contract: the (same_kind-cast) result is written into the
        /// caller's array and that SAME instance is returned — a 0-d out for a flat reduction, the reduced
        /// shape for an axis one. A flat reduction into a non-0-d out is rejected with NumPy's
        /// wrong-dimensions message.
        /// </summary>
        [TestMethod]
        public void M4Tail_Out()
        {
            var f = np.array(new double[] { 3, -7, 1.5, 5, 0 });

            var o = np.zeros(new Shape());                       // 0-d
            var r = np.evaluate(NDExpr.Ptp((NDExpr)f), @out: o);
            Assert.IsTrue(ReferenceEquals(r, o));                // returns the out instance
            Assert.AreEqual(12.0, o.GetDouble(0), 0.0);

            var oa = np.zeros(2);
            var g = np.array(new double[] { 1, double.NaN, 2, 3 }).reshape(2, 2);
            np.evaluate(NDExpr.NanMin((NDExpr)g, 1), @out: oa);  // rows: [1, 2]
            Assert.AreEqual(1.0, oa.GetDouble(0), 0.0);
            Assert.AreEqual(2.0, oa.GetDouble(1), 0.0);

            // A flat reduction demands a 0-d out.
            Assert.ThrowsException<ArgumentException>(
                () => np.evaluate(NDExpr.Ptp((NDExpr)f), @out: np.zeros(3)));
        }

        /// <summary>
        /// Empty-input behaviour: <c>Ptp</c> RAISES (it composes <c>amax - amin</c>, and a zero-size
        /// max/min has no identity — NumPy raises here too), while <c>NanMin</c>/<c>NanMax</c> return NaN
        /// because they delegate to the engine's <c>np.nanmin</c>/<c>np.nanmax</c>, whose empty behaviour
        /// (a 0-d NaN) is a PRE-EXISTING NumSharp quirk that diverges from NumPy's raise — evaluate
        /// faithfully mirrors the engine reduction rather than adding parity np.nanmin itself lacks.
        /// </summary>
        [TestMethod]
        public void M4Tail_EmptyInput()
        {
            var e = np.array(new double[] { });

            Assert.ThrowsException<ArgumentException>(   // zero-size max/min has no identity, like np.ptp([])
                () => np.evaluate(NDExpr.Ptp((NDExpr)e)));

            // np.nanmin([]) / np.nanmax([]) return a 0-d NaN in NumSharp (a documented engine quirk vs
            // NumPy's ValueError); the delegating evaluate path returns exactly what the engine does.
            Assert.IsTrue(double.IsNaN(np.evaluate(NDExpr.NanMin((NDExpr)e)).GetDouble(0)));
            Assert.IsTrue(double.IsNaN(np.evaluate(NDExpr.NanMax((NDExpr)e)).GetDouble(0)));
        }

        // ===================================================================
        // Plan P2 M4c — the int64 INDEX kinds ArgMax / ArgMin (host-delegated)
        // ===================================================================

        /// <summary>
        /// <c>ArgMax</c>/<c>ArgMin</c> return the FIRST-occurrence index of the extreme (NumPy's tie rule),
        /// and the result is always <b>int64</b> — an index, not a value — regardless of the child dtype.
        /// The 0-d result of a flat reduce is pinned too. This is the property that sets the index kinds
        /// apart from the M4-tail value kinds, which preserve the child dtype.
        /// </summary>
        [TestMethod]
        public void M4c_ArgMaxMin_ValuesAndInt64Result()
        {
            // max 9 first appears at index 1 (repeats at 3); min 1 first at index 5 (repeats at 6).
            var a = np.array(new double[] { 3, 9, 4, 9, 5, 1, 1 });
            var amax = np.evaluate(NDExpr.ArgMax((NDExpr)a * 1.0));   // a fused (non-identity) child
            var amin = np.evaluate(NDExpr.ArgMin((NDExpr)a * 1.0));
            Assert.AreEqual(NPTypeCode.Int64, amax.typecode);        // int64 result, not the child's float64
            Assert.AreEqual(0, amax.ndim);                            // flat reduce → 0-d scalar
            Assert.AreEqual(1L, amax.GetInt64(0));                    // FIRST maximum wins
            Assert.AreEqual(5L, amin.GetInt64(0));                    // FIRST minimum wins

            // Child dtype is irrelevant to the result dtype: an integer child still yields an int64 index.
            var i = np.array(new int[] { 7, 2, 9, 4 });
            var imax = np.evaluate(NDExpr.ArgMax((NDExpr)i));
            Assert.AreEqual(NPTypeCode.Int64, imax.typecode);
            Assert.AreEqual(2L, imax.GetInt64(0));
        }

        /// <summary>
        /// A NaN is treated as the largest value by BOTH argmax and argmin (NumPy's first-NaN-wins rule),
        /// so the index of the FIRST NaN is returned. Bit-exact here because the delegating path reduces
        /// the fresh C-contiguous materialized child, exactly the buffer <c>np.argmax(child)</c> reduces.
        /// </summary>
        [TestMethod]
        public void M4c_ArgMaxMin_FirstNaNWins()
        {
            var g = np.array(new double[] { 3, double.NaN, 1, double.NaN, 5 });
            Assert.AreEqual(np.argmax(g), np.evaluate(NDExpr.ArgMax((NDExpr)g + 0.0)).GetInt64(0));
            Assert.AreEqual(np.argmin(g), np.evaluate(NDExpr.ArgMin((NDExpr)g + 0.0)).GetInt64(0));
            Assert.AreEqual(1L, np.evaluate(NDExpr.ArgMax((NDExpr)g + 0.0)).GetInt64(0));   // first NaN at index 1
        }

        /// <summary>
        /// The axis + keepdims forms reduce along one axis to an int64 index array, matching
        /// <c>np.argmax</c>/<c>np.argmin</c> along that axis (keepdims leaves the reduced axis as size 1).
        /// </summary>
        [TestMethod]
        public void M4c_ArgMaxMin_Axis_Forms()
        {
            var m = np.arange(12).reshape(3, 4).astype(np.float64);
            var mc = m * 1.0;

            var a0 = np.evaluate(NDExpr.ArgMax((NDExpr)m * 1.0, 0));  // per column, max is the last row → 2
            Assert.AreEqual(NPTypeCode.Int64, a0.typecode);
            Assert.IsTrue(a0.array_equal(np.argmax(mc, 0)));

            var a1 = np.evaluate(NDExpr.ArgMax((NDExpr)m * 1.0, 1, keepdims: true)); // per row, max is col 3
            Assert.AreEqual(2, a1.ndim);
            Assert.AreEqual(1L, a1.shape[1]);
            Assert.IsTrue(a1.array_equal(np.argmax(mc, 1, keepdims: true)));

            var i1 = np.evaluate(NDExpr.ArgMin((NDExpr)m * 1.0, 1));  // per row, min is col 0
            Assert.IsTrue(i1.array_equal(np.argmin(mc, 1)));
        }

        /// <summary>
        /// <c>out=</c> follows evaluate's contract: the int64 index is written (same_kind-cast) into the
        /// caller's array and that SAME instance returns — a 0-d out for a flat reduce, the reduced shape
        /// for an axis one — and a flat reduce into a non-0-d out is rejected.
        /// </summary>
        [TestMethod]
        public void M4c_ArgMaxMin_Out()
        {
            var a = np.array(new double[] { 3, 9, 4, 9, 5 });

            var o = NDArray.Scalar(0L);                              // 0-d int64
            var r = np.evaluate(NDExpr.ArgMax((NDExpr)a * 1.0), @out: o);
            Assert.IsTrue(ReferenceEquals(r, o));                    // returns the out instance
            Assert.AreEqual(1L, o.GetInt64(0));

            var m = np.arange(12).reshape(3, 4).astype(np.float64);
            var oa = np.zeros(new Shape(4), np.int64);               // axis-0 result shape
            np.evaluate(NDExpr.ArgMax((NDExpr)m * 1.0, 0), @out: oa);
            Assert.IsTrue(oa.array_equal(np.argmax(m * 1.0, 0)));

            // A flat reduction demands a 0-d out (NumPy's wrong-dimensions message).
            Assert.ThrowsException<ArgumentException>(
                () => np.evaluate(NDExpr.ArgMax((NDExpr)a * 1.0), @out: np.zeros(new Shape(3), np.int64)));
        }

        /// <summary>
        /// A zero-size input has no identity for an index reduction, so <c>ArgMax</c>/<c>ArgMin</c> RAISE —
        /// exactly as <c>np.argmax([])</c> does — rather than returning a sentinel index. (Unlike
        /// nanmin/nanmax, whose empty behaviour is the engine's 0-d NaN.)
        /// </summary>
        [TestMethod]
        public void M4c_ArgMaxMin_EmptyRaises()
        {
            var e = np.array(new double[] { });
            Assert.ThrowsException<ArgumentException>(() => np.evaluate(NDExpr.ArgMax((NDExpr)e)));
            Assert.ThrowsException<ArgumentException>(() => np.evaluate(NDExpr.ArgMin((NDExpr)e)));
        }

        // ===================================================================
        // Plan P2 M4c summation kinds — NanMean / Var / Std (host-computed over
        // the materialized child with the M1/M2-exact pairwise sum).
        // ===================================================================

        /// <summary>Var/Std values, the ddof divisor, and the dtype tiers (int→float64, float32 preserved).</summary>
        [TestMethod]
        public void M4cSum_Var_Std_Ddof_And_DtypeTiers()
        {
            var a = np.array(new double[] { 1, 2, 3, 4, 5 });
            Assert.AreEqual(2.0, np.evaluate(NDExpr.Var(a)).GetDouble(0), 1e-12);          // population variance
            Assert.AreEqual(2.5, np.evaluate(NDExpr.Var(a, ddof: 1)).GetDouble(0), 1e-12); // sample variance
            Assert.AreEqual(System.Math.Sqrt(2.0), np.evaluate(NDExpr.Std(a)).GetDouble(0), 1e-12);

            var i = np.array(new int[] { 1, 2, 3, 4, 5 });                                  // int → float64
            var vi = np.evaluate(NDExpr.Var(i));
            Assert.AreEqual(NPTypeCode.Double, vi.typecode);
            Assert.AreEqual(2.0, vi.GetDouble(0), 1e-12);

            var f = np.array(new float[] { 1, 2, 3, 4, 5 });                                // float32 preserved
            var vf = np.evaluate(NDExpr.Var(f));
            Assert.AreEqual(NPTypeCode.Single, vf.typecode);
            Assert.AreEqual(2.0f, vf.GetSingle(0), 1e-5f);
        }

        /// <summary>NanMean skips NaN, an all-NaN input yields NaN, and an integer child (no NaN) is the plain mean (float64).</summary>
        [TestMethod]
        public void M4cSum_NanMean_SkipsNaN_And_IntPassthrough()
        {
            var a = np.array(new double[] { 1.0, double.NaN, 3.0, double.NaN, 5.0 });
            Assert.AreEqual(3.0, np.evaluate(NDExpr.NanMean(a)).GetDouble(0), 1e-12);       // (1+3+5)/3

            var allnan = np.array(new double[] { double.NaN, double.NaN });
            Assert.IsTrue(double.IsNaN(np.evaluate(NDExpr.NanMean(allnan)).GetDouble(0)));  // all-NaN slice → NaN

            var i = np.array(new int[] { 2, 4, 6 });
            var r = np.evaluate(NDExpr.NanMean(i));
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.AreEqual(4.0, r.GetDouble(0), 1e-12);
        }

        /// <summary>A complex128 child yields a REAL float64 var/std (the variance of complex data), while its nanmean stays complex.</summary>
        [TestMethod]
        public void M4cSum_ComplexVar_IsRealFloat64_NanMean_IsComplex()
        {
            var z = np.array(new System.Numerics.Complex[] { new(1, 2), new(3, -1), new(-2, 0.5), new(4, 4) });

            var v = np.evaluate(NDExpr.Var(z));
            Assert.AreEqual(NPTypeCode.Double, v.typecode);                                 // REAL variance
            Assert.AreEqual(8.671875, v.GetDouble(0), 1e-12);                               // Σ|z-mean|² / 4

            var s = np.evaluate(NDExpr.Std(z));
            Assert.AreEqual(NPTypeCode.Double, s.typecode);
            Assert.AreEqual(System.Math.Sqrt(8.671875), s.GetDouble(0), 1e-12);

            var nm = np.evaluate(NDExpr.NanMean(z));                                        // no NaN → == mean, complex
            Assert.AreEqual(NPTypeCode.Complex, nm.typecode);
            Assert.AreEqual(1.5, np.real(nm).GetDouble(0), 1e-12);                          // (1+3-2+4)/4
            Assert.AreEqual(1.375, np.imag(nm).GetDouble(0), 1e-12);                        // (2-1+0.5+4)/4
        }

        /// <summary>An empty input yields NaN for all three (the divisor is zero) — the edge the byte corpus can't pin cleanly.</summary>
        [TestMethod]
        public void M4cSum_EmptyInput_YieldsNaN()
        {
            var e = np.array(new double[] { });
            Assert.IsTrue(double.IsNaN(np.evaluate(NDExpr.Var(e)).GetDouble(0)));
            Assert.IsTrue(double.IsNaN(np.evaluate(NDExpr.Std(e)).GetDouble(0)));
            Assert.IsTrue(double.IsNaN(np.evaluate(NDExpr.NanMean(e)).GetDouble(0)));
        }

        /// <summary><c>ddof ≥ N</c> divides by a clamped-zero divisor → <c>+inf</c> (positive sum of squares), matching NumPy.</summary>
        [TestMethod]
        public void M4cSum_DdofClampsToInf()
        {
            var a = np.array(new double[] { 1, 2, 3 });
            Assert.IsTrue(double.IsPositiveInfinity(np.evaluate(NDExpr.Var(a, ddof: 3)).GetDouble(0))); // N-ddof = 0
            Assert.IsTrue(double.IsPositiveInfinity(np.evaluate(NDExpr.Var(a, ddof: 5)).GetDouble(0))); // clamped to 0
        }

        /// <summary>Axis var/std/nanmean and keepdims, over a 2-D child.</summary>
        [TestMethod]
        public void M4cSum_Axis_Forms()
        {
            var m = np.arange(12).reshape(3, 4).astype(np.float64);

            var v0 = np.evaluate(NDExpr.Var(m, axis: 0));                                   // (4,)
            Assert.AreEqual(1, v0.ndim);
            Assert.AreEqual(4, v0.shape[0]);
            for (int j = 0; j < 4; j++) Assert.AreEqual(32.0 / 3.0, v0.GetDouble(j), 1e-9); // var([j,j+4,j+8])

            var v1k = np.evaluate(NDExpr.Var(m, axis: 1, keepdims: true));                  // (3,1)
            Assert.AreEqual(2, v1k.ndim);
            Assert.AreEqual(3, v1k.shape[0]);
            Assert.AreEqual(1, v1k.shape[1]);

            var nm0 = np.evaluate(NDExpr.NanMean(m, axis: 0));                              // mean of col j = j+4
            for (int j = 0; j < 4; j++) Assert.AreEqual(4.0 + j, nm0.GetDouble(j), 1e-12);
        }

        /// <summary>
        /// float16 and decimal are rejected — a bit-exact float16 variance needs a float16 pairwise sum
        /// kernel that dtype lacks (the "Half not diverted" gap M1/M2 share), and decimal has no NumPy
        /// analog / pairwise kernel here. A directed <see cref="NotSupportedException"/> rather than a
        /// silent divergence (use np.var / np.std / np.nanmean directly for those).
        /// </summary>
        [TestMethod]
        public void M4cSum_HalfAndDecimal_NotSupported()
        {
            var h = np.array(new double[] { 1, 2, 3 }).astype(np.float16);
            Assert.ThrowsException<NotSupportedException>(() => np.evaluate(NDExpr.Var(h)));
            Assert.ThrowsException<NotSupportedException>(() => np.evaluate(NDExpr.NanMean(h)));

            var d = np.array(new double[] { 1, 2, 3 }).astype(np.@decimal);
            Assert.ThrowsException<NotSupportedException>(() => np.evaluate(NDExpr.Std(d)));
        }

        // ===================================================================
        // Plan P2 M4c-average — weighted np.average over TWO trees (host-computed
        // with the M1/M2-exact pairwise sum; the FIRST two-operand reduction).
        // ===================================================================

        /// <summary>The weighted-average value, and the dtype tiers (int/bool→float64, float32 preserved).</summary>
        [TestMethod]
        public void M4cAvg_Value_And_DtypeTiers()
        {
            // Σ(v·w)/Σ(w) = (4+6+6+4)/(4+3+2+1) = 20/10 = 2.0
            var v = np.array(new double[] { 1, 2, 3, 4 });
            var w = np.array(new double[] { 4, 3, 2, 1 });
            Assert.AreEqual(2.0, np.evaluate(NDExpr.Average((NDExpr)v, w)).GetDouble(0), 1e-12);

            // int values / int weights → float64 (never a wrapping int64 product)
            var vi = np.array(new int[] { 1, 2, 3, 4 });
            var wi = np.array(new int[] { 4, 3, 2, 1 });
            var ri = np.evaluate(NDExpr.Average((NDExpr)vi, wi));
            Assert.AreEqual(NPTypeCode.Double, ri.typecode);
            Assert.AreEqual(2.0, ri.GetDouble(0), 1e-12);

            // float32 values / float32 weights → float32, bit-exact with NumPy (91/21)
            var vf = np.array(new float[] { 1, 2, 3, 4, 5, 6 });
            var wf = np.array(new float[] { 1, 2, 3, 4, 5, 6 });
            var rf = np.evaluate(NDExpr.Average((NDExpr)vf, wf));
            Assert.AreEqual(NPTypeCode.Single, rf.typecode);
            Assert.AreEqual(4.3333335f, rf.GetSingle(0), 0f);   // 0x408aaaab — bit-exact
        }

        /// <summary>An unweighted-equivalent average (unit weights) equals the plain mean.</summary>
        [TestMethod]
        public void M4cAvg_UnitWeights_EqualsMean()
        {
            var v = np.array(new double[] { 1, 2, 3, 4, 5, 6 });
            var ones = np.ones(6);
            Assert.AreEqual(np.mean(v).GetDouble(0),
                np.evaluate(NDExpr.Average((NDExpr)v, (NDExpr)ones)).GetDouble(0), 0.0);
        }

        /// <summary>A complex128 values tree yields a complex weighted average (real weights promote to complex128).</summary>
        [TestMethod]
        public void M4cAvg_Complex()
        {
            var z = np.array(new System.Numerics.Complex[] { new(1, 2), new(3, -1), new(-2, 0.5), new(4, 4) });
            var w = np.array(new double[] { 1, 2, 3, 4 });
            var r = np.evaluate(NDExpr.Average((NDExpr)z, (NDExpr)w));
            Assert.AreEqual(NPTypeCode.Complex, r.typecode);
            // Σ(z·w)/Σ(w) with Σw=10: re=1.7, im=1.75 (probed against np.average 2.4.2)
            Assert.AreEqual(1.7, np.real(r).GetDouble(0), 1e-12);
            Assert.AreEqual(1.75, np.imag(r).GetDouble(0), 1e-12);
        }

        /// <summary>Axis and keepdims forms over a 2-D values/weights pair, matching np.average.</summary>
        [TestMethod]
        public void M4cAvg_Axis_Forms()
        {
            var m = np.arange(1, 13).reshape(3, 4).astype(np.float64);
            var wm = ((np.arange(1, 13).reshape(3, 4)) % 5 + 1).astype(np.float64);

            var a0 = np.evaluate(NDExpr.Average((NDExpr)m, wm, 0));                 // (4,)
            Assert.AreEqual(1, a0.ndim);
            Assert.AreEqual(4, a0.shape[0]);
            var e0 = new[] { 6.5, 4.666666666666667, 6.111111111111111, 7.333333333333333 };
            for (int j = 0; j < 4; j++) Assert.AreEqual(e0[j], a0.GetDouble(j), 1e-9);

            var a1 = np.evaluate(NDExpr.Average((NDExpr)m, wm, 1));                 // (3,)
            var e1 = new[] { 2.857142857142857, 7.0, 10.272727272727273 };
            for (int j = 0; j < 3; j++) Assert.AreEqual(e1[j], a1.GetDouble(j), 1e-9);

            // negative axis == the last axis
            var an = np.evaluate(NDExpr.Average((NDExpr)m, wm, -1));
            for (int j = 0; j < 3; j++) Assert.AreEqual(e1[j], an.GetDouble(j), 1e-9);

            var a1k = np.evaluate(NDExpr.Average((NDExpr)m, wm, 1, keepdims: true)); // (3,1)
            Assert.AreEqual(2, a1k.ndim);
            Assert.AreEqual(3, a1k.shape[0]);
            Assert.AreEqual(1, a1k.shape[1]);
        }

        /// <summary>A fused values / weights tree (not a bare array) drives the materialize-then-average path.</summary>
        [TestMethod]
        public void M4cAvg_FusedChildren()
        {
            var a = np.array(new double[] { 1, 2, 3, 4 });
            var b = np.array(new double[] { 1, 1, 2, 2 });
            // Average(a*a, b+1) = Σ(a²·(b+1)) / Σ(b+1)
            double num = 1 * 2 + 4 * 2 + 9 * 3 + 16 * 3;   // 2 + 8 + 27 + 48 = 85
            double den = 2 + 2 + 3 + 3;                     // 10
            Assert.AreEqual(num / den,
                np.evaluate(NDExpr.Average((NDExpr)a * a, (NDExpr)b + 1.0)).GetDouble(0), 1e-12);
        }

        /// <summary>Weights that sum to zero (an empty input included) raise, with NumPy's verbatim message.</summary>
        [TestMethod]
        public void M4cAvg_ZeroWeights_Raises()
        {
            var v = np.array(new double[] { 1, 2, 3 });
            var w = np.array(new double[] { 1, -1, 0 });   // Σw == 0
            var ex = Assert.ThrowsException<DivideByZeroException>(
                () => np.evaluate(NDExpr.Average((NDExpr)v, w)));
            Assert.AreEqual("Weights sum to zero, can't be normalized", ex.Message);

            // an empty input sums the weights to zero too → the same raise (the byte corpus can't reach this)
            var e = np.array(new double[] { });
            Assert.ThrowsException<DivideByZeroException>(
                () => np.evaluate(NDExpr.Average((NDExpr)e, (NDExpr)e)));

            // an AXIS slab whose weights sum to zero raises as well (NumPy checks any slab)
            var m = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            var wm = np.array(new double[,] { { 1, -1 }, { 1, 1 } });   // row 0 sums to 0
            Assert.ThrowsException<DivideByZeroException>(
                () => np.evaluate(NDExpr.Average((NDExpr)m, wm, 1)));
        }

        /// <summary>A NaN in either tree propagates to NaN — it is not a zero-weight raise (NaN != 0).</summary>
        [TestMethod]
        public void M4cAvg_NaN_Propagates()
        {
            var v = np.array(new double[] { 1, 2, 3 });
            var wn = np.array(new double[] { 1, double.NaN, 1 });
            Assert.IsTrue(double.IsNaN(np.evaluate(NDExpr.Average((NDExpr)v, wn)).GetDouble(0)));

            var vn = np.array(new double[] { 1, double.NaN, 3 });
            var w = np.array(new double[] { 1, 1, 1 });
            Assert.IsTrue(double.IsNaN(np.evaluate(NDExpr.Average((NDExpr)vn, w)).GetDouble(0)));
        }

        /// <summary><c>out=</c> receives the average (a flat one requires a 0-d out).</summary>
        [TestMethod]
        public void M4cAvg_Out()
        {
            var v = np.array(new double[] { 1, 2, 3, 4 });
            var w = np.array(new double[] { 4, 3, 2, 1 });
            var dst = np.zeros(new Shape());                    // 0-d
            var ret = np.evaluate(NDExpr.Average((NDExpr)v, w), @out: dst);
            Assert.IsTrue(ReferenceEquals(ret, dst));
            Assert.AreEqual(2.0, dst.GetDouble(0), 1e-12);

            // a non-0-d out for a flat average is rejected (same message shape as the other reduce paths)
            Assert.ThrowsException<ArgumentException>(() => np.evaluate(NDExpr.Average((NDExpr)v, w), @out: np.zeros(1)));

            // axis out= takes the reduced shape
            var m = np.arange(1, 13).reshape(3, 4).astype(np.float64);
            var wm = ((np.arange(1, 13).reshape(3, 4)) % 5 + 1).astype(np.float64);
            var od = np.zeros(4);
            np.evaluate(NDExpr.Average((NDExpr)m, wm, 0), @out: od);
            Assert.AreEqual(6.5, od.GetDouble(0), 1e-9);
        }

        /// <summary>
        /// A float16 / decimal result dtype is rejected with a directed <see cref="NotSupportedException"/>
        /// (a bit-exact reduction there needs a pairwise sum kernel that dtype lacks — the gap M1/M2/
        /// M4c-summation share); a float16 values tree with WIDER float32 weights (result float32) is fine.
        /// </summary>
        [TestMethod]
        public void M4cAvg_HalfAndDecimal_NotSupported()
        {
            var h = np.array(new double[] { 1, 2, 3 }).astype(np.float16);   // f2 · f2 → f2 result
            Assert.ThrowsException<NotSupportedException>(() => np.evaluate(NDExpr.Average((NDExpr)h, (NDExpr)h)));

            var d = np.array(new double[] { 1, 2, 3 }).astype(np.@decimal);
            Assert.ThrowsException<NotSupportedException>(() => np.evaluate(NDExpr.Average((NDExpr)d, (NDExpr)d)));

            // f2 values with wider f4 weights → f4 result: served (not Half)
            var wf = np.array(new float[] { 1, 2, 3 });
            var r = np.evaluate(NDExpr.Average((NDExpr)h, (NDExpr)wf));
            Assert.AreEqual(NPTypeCode.Single, r.typecode);
        }

        // =====================================================================================
        // Plan P2 M5 — axis=None + keepdims flat reductions. NumPy's np.sum(a, keepdims=True) with
        // axis=None returns shape (1,)*a.ndim; the flat factories' new `keepdims` overloads reshape the
        // 0-d scalar result to that (the VALUE is identical to the 0-d form, only the wrapper rank
        // differs). Tuple axis is NOT offered — every NumSharp reduction is single-axis (a library-wide
        // gap). These pin the corpus-unreachable edges (out=, 0-d child, rank-per-input) and one value
        // per host path (fold / delegating / stat / average).
        // =====================================================================================

        /// <summary>Flat keepdims across a host-path representative of each kind: shape (1,1) and the same VALUE as the 0-d form.</summary>
        [TestMethod]
        public void M5_FlatKeepdims_ShapeAndValue_AcrossKinds()
        {
            var a = (np.arange(12).reshape(3, 4).astype(np.float64) - 5);   // has negatives, no NaN

            // (kind flat-keepdims tree, kind flat-0d tree) — the keepdims value must equal the 0-d value.
            (NDExpr kd, NDExpr flat)[] pairs =
            {
                (NDExpr.Sum((NDExpr)a, true),  NDExpr.Sum((NDExpr)a)),      // fold path
                (NDExpr.Prod((NDExpr)a, true), NDExpr.Prod((NDExpr)a)),
                (NDExpr.Min((NDExpr)a, true),  NDExpr.Min((NDExpr)a)),
                (NDExpr.Max((NDExpr)a, true),  NDExpr.Max((NDExpr)a)),
                (NDExpr.Mean((NDExpr)a, true), NDExpr.Mean((NDExpr)a)),
                (NDExpr.Ptp((NDExpr)a, true),  NDExpr.Ptp((NDExpr)a)),      // delegating path
                (NDExpr.NanMin((NDExpr)a, true), NDExpr.NanMin((NDExpr)a)),
                (NDExpr.NanMax((NDExpr)a, true), NDExpr.NanMax((NDExpr)a)),
                (NDExpr.Var((NDExpr)a, true),  NDExpr.Var((NDExpr)a)),      // stat path
                (NDExpr.Std((NDExpr)a, true),  NDExpr.Std((NDExpr)a)),
                (NDExpr.NanMean((NDExpr)a, true), NDExpr.NanMean((NDExpr)a)),
            };
            foreach (var (kd, flat) in pairs)
            {
                var kr = np.evaluate(kd);
                Assert.AreEqual(2, kr.ndim);
                Assert.AreEqual(1, kr.shape[0]);
                Assert.AreEqual(1, kr.shape[1]);
                Assert.AreEqual(np.evaluate(flat).GetDouble(0), kr.GetDouble(0, 0), 1e-12);
            }
        }

        /// <summary>The flat keepdims result's rank follows the reduced expression: 1-D→(1,), 3-D→(1,1,1).</summary>
        [TestMethod]
        public void M5_FlatKeepdims_RankMatchesChild()
        {
            var a1 = np.arange(6).astype(np.float64);
            var s1 = np.evaluate(NDExpr.Sum((NDExpr)a1, true));
            Assert.AreEqual(1, s1.ndim);
            Assert.AreEqual(1, s1.shape[0]);
            Assert.AreEqual(15.0, s1.GetDouble(0), 1e-12);

            var a3 = np.arange(24).reshape(2, 3, 4).astype(np.float64);
            var s3 = np.evaluate(NDExpr.Sum((NDExpr)a3, true));
            Assert.AreEqual(3, s3.ndim);
            Assert.AreEqual(1, s3.shape[0]);
            Assert.AreEqual(1, s3.shape[1]);
            Assert.AreEqual(1, s3.shape[2]);
            Assert.AreEqual(276.0, s3.GetDouble(0, 0, 0), 1e-12);
        }

        /// <summary>The int64 INDEX kinds keep dims to (1,)*ndim int64, first-tie index unchanged.</summary>
        [TestMethod]
        public void M5_FlatKeepdims_ArgMaxMin_Int64Index()
        {
            var a = (np.arange(12).reshape(3, 4).astype(np.float64) - 5);
            var am = np.evaluate(NDExpr.ArgMax((NDExpr)a, true));
            Assert.AreEqual(2, am.ndim);
            Assert.AreEqual(1, am.shape[0]);
            Assert.AreEqual(1, am.shape[1]);
            Assert.AreEqual(NPTypeCode.Int64, am.typecode);
            Assert.AreEqual(np.evaluate(NDExpr.ArgMax((NDExpr)a)).GetInt64(0), am.GetInt64(0, 0));

            var an = np.evaluate(NDExpr.ArgMin((NDExpr)a, true));
            Assert.AreEqual(np.evaluate(NDExpr.ArgMin((NDExpr)a)).GetInt64(0), an.GetInt64(0, 0));
        }

        /// <summary>Weighted average keeps dims over the (broadcast) rank; value equals the 0-d form.</summary>
        [TestMethod]
        public void M5_FlatKeepdims_WeightedAverage()
        {
            var v = np.array(new double[] { 1, 2, 3, 4, 5, 6 }).reshape(2, 3);
            var w = np.array(new double[] { 2, 1, 3, 1, 2, 1 }).reshape(2, 3);
            var kd = np.evaluate(NDExpr.Average((NDExpr)v, (NDExpr)w, true));
            Assert.AreEqual(2, kd.ndim);
            Assert.AreEqual(1, kd.shape[0]);
            Assert.AreEqual(1, kd.shape[1]);
            Assert.AreEqual(np.evaluate(NDExpr.Average((NDExpr)v, (NDExpr)w)).GetDouble(0), kd.GetDouble(0, 0), 1e-12);
        }

        /// <summary>Var/Std keep dims with a ddof, and the ddof is part of the value.</summary>
        [TestMethod]
        public void M5_FlatKeepdims_VarStd_Ddof()
        {
            var a = (np.arange(12).reshape(3, 4).astype(np.float64) - 5);
            var v1 = np.evaluate(NDExpr.Var((NDExpr)a, true, 1));
            Assert.AreEqual(2, v1.ndim);
            Assert.AreEqual(1, v1.shape[0]);
            Assert.AreEqual(np.evaluate(NDExpr.Var((NDExpr)a, 1)).GetDouble(0), v1.GetDouble(0, 0), 1e-12);

            var s1 = np.evaluate(NDExpr.Std((NDExpr)a, true, 1));
            Assert.AreEqual(np.evaluate(NDExpr.Std((NDExpr)a, 1)).GetDouble(0), s1.GetDouble(0, 0), 1e-12);
        }

        /// <summary>A caller out= of the keepdims shape writes through; a wrong-rank out raises (each host path).</summary>
        [TestMethod]
        public void M5_FlatKeepdims_Out()
        {
            var a = (np.arange(12).reshape(3, 4).astype(np.float64) - 5);
            var outKd = np.zeros(new Shape(1, 1), np.float64);
            var r = np.evaluate(NDExpr.Sum((NDExpr)a, true), @out: outKd);
            Assert.IsTrue(ReferenceEquals(r, outKd));
            Assert.AreEqual(np.sum(a).GetDouble(), outKd.GetDouble(0, 0), 1e-12);

            // Wrong-rank out (0-d) under keepdims raises, on the fold, delegating, stat and average paths.
            Assert.ThrowsException<ArgumentException>(() => np.evaluate(NDExpr.Sum((NDExpr)a, true), @out: np.zeros(new Shape(), np.float64)));
            Assert.ThrowsException<ArgumentException>(() => np.evaluate(NDExpr.Ptp((NDExpr)a, true), @out: np.zeros(new Shape(), np.float64)));
            Assert.ThrowsException<ArgumentException>(() => np.evaluate(NDExpr.Var((NDExpr)a, true), @out: np.zeros(new Shape(), np.float64)));
            var w = np.ones(new Shape(3, 4), np.float64);
            Assert.ThrowsException<ArgumentException>(() => np.evaluate(NDExpr.Average((NDExpr)a, (NDExpr)w, true), @out: np.zeros(new Shape(), np.float64)));
        }

        /// <summary>A 0-d child stays 0-d under keepdims (np.sum(scalar, keepdims=True) is 0-d), and keepdims=false is unchanged.</summary>
        [TestMethod]
        public void M5_FlatKeepdims_ZeroDChild_And_NoKeepdimsUnchanged()
        {
            var scalar0d = np.array(7.0);
            Assert.AreEqual(0, np.evaluate(NDExpr.Sum((NDExpr)scalar0d, true)).ndim);

            // keepdims=false remains a 0-d scalar (regression guard on the pre-M5 behaviour).
            var a = np.arange(6).reshape(2, 3).astype(np.float64);
            Assert.AreEqual(0, np.evaluate(NDExpr.Sum((NDExpr)a)).ndim);
            Assert.AreEqual(0, np.evaluate(NDExpr.Ptp((NDExpr)a)).ndim);
        }
    }
}
