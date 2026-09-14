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

        /// <summary>
        /// A flat reduce of a SIZE-1 1-D input is a 0-d scalar (keepdims=false) — surfaced by the live
        /// NumPy differential: the delegating path's np.nanmin/np.nanmax return shape (1,) for a size-1
        /// 1-D input (a pre-existing engine quirk), so evaluate must normalize the flat result to 0-d
        /// (KeepdimsFlat), matching np.nanmin(np.array([5.]), keepdims=False).shape == (). keepdims=True
        /// gives (1,).
        /// </summary>
        [TestMethod]
        public void M5_FlatDelegatingReduce_SizeOne_NormalizesTo0d()
        {
            var one = np.array(new double[] { 5.0 });   // shape (1,)
            foreach (var flat in new[] { NDExpr.NanMin((NDExpr)one), NDExpr.NanMax((NDExpr)one), NDExpr.Ptp((NDExpr)one) })
                Assert.AreEqual(0, np.evaluate(flat).ndim, "flat reduce of a size-1 1-D input must be 0-d");

            Assert.AreEqual(5.0, np.evaluate(NDExpr.NanMin((NDExpr)one)).GetDouble(), 1e-12);

            // keepdims → (1,), still one element with the same value.
            var kd = np.evaluate(NDExpr.NanMin((NDExpr)one, true));
            Assert.AreEqual(1, kd.ndim);
            Assert.AreEqual(1, kd.shape[0]);
            Assert.AreEqual(5.0, kd.GetDouble(0), 1e-12);
        }

        // =====================================================================
        // Phase 4 — binary node coverage (the engine binary ufuncs that gained
        // an NDExpr node). Values probed from NumPy 2.4.2; the fused kernel is
        // the SAME per-op scalar emitter the engine's own ufuncs use.
        // =====================================================================

        /// <summary>fmax/fmin IGNORE NaN (the number wins) while maximum/minimum PROPAGATE it — the
        /// one behavioural split that makes the two families distinct nodes.</summary>
        [TestMethod]
        public void P4_FMaxFMin_IgnoreNaN_WhileMaximumMinimumPropagate()
        {
            var a = np.array(new double[] { 1, double.NaN, 3 });
            var b = np.array(new double[] { 2, 2, double.NaN });

            var fmax = np.evaluate(NDExpr.FMax(NDExpr.Arr(a), NDExpr.Arr(b)));
            Assert.AreEqual(2.0, fmax.GetDouble(0), 0);
            Assert.AreEqual(2.0, fmax.GetDouble(1), 0);   // NaN ignored -> the number
            Assert.AreEqual(3.0, fmax.GetDouble(2), 0);   // NaN ignored -> the number

            var fmin = np.evaluate(NDExpr.FMin(NDExpr.Arr(a), NDExpr.Arr(b)));
            Assert.AreEqual(1.0, fmin.GetDouble(0), 0);
            Assert.AreEqual(2.0, fmin.GetDouble(1), 0);
            Assert.AreEqual(3.0, fmin.GetDouble(2), 0);

            // Maximum/Minimum are exact aliases of Max/Min — NaN PROPAGATES.
            var mx = np.evaluate(NDExpr.Maximum(NDExpr.Arr(a), NDExpr.Arr(b)));
            Assert.AreEqual(2.0, mx.GetDouble(0), 0);
            Assert.IsTrue(double.IsNaN(mx.GetDouble(1)));
            Assert.IsTrue(double.IsNaN(mx.GetDouble(2)));
            var mn = np.evaluate(NDExpr.Minimum(NDExpr.Arr(a), NDExpr.Arr(b)));
            Assert.IsTrue(double.IsNaN(mn.GetDouble(1)));
        }

        /// <summary>fmod takes the DIVIDEND's sign (C fmod), unlike Mod (floored, divisor's sign).</summary>
        [TestMethod]
        public void P4_Fmod_TakesDividendSign()
        {
            var a = np.array(new double[] { -7, 7, -7, 7 });
            var b = np.array(new double[] { 3, 3, -3, -3 });
            var r = np.evaluate(NDExpr.Fmod(NDExpr.Arr(a), NDExpr.Arr(b)));
            Assert.AreEqual(-1.0, r.GetDouble(0), 0);   // np.fmod(-7,3)  == -1  (Mod → 2)
            Assert.AreEqual(1.0, r.GetDouble(1), 0);    // np.fmod( 7,3)  ==  1
            Assert.AreEqual(-1.0, r.GetDouble(2), 0);   // np.fmod(-7,-3) == -1
            Assert.AreEqual(1.0, r.GetDouble(3), 0);    // np.fmod( 7,-3) ==  1

            // integer stays integer; a bool pair falls to the int8 loop.
            Assert.AreEqual(NPTypeCode.Int32, np.evaluate(NDExpr.Fmod(NDExpr.Arr(np.array(new[] { 7 })), NDExpr.Arr(np.array(new[] { 3 })))).typecode);
            Assert.AreEqual(NPTypeCode.SByte, np.evaluate(NDExpr.Fmod(NDExpr.Arr(np.array(new[] { true })), NDExpr.Arr(np.array(new[] { true })))).typecode);
        }

        /// <summary>copysign is an EXACT sign-bit copy (so -0.0 flips the sign); nextafter is an exact
        /// one-ULP step. Both are float-only, so int inputs PROMOTE to the tier float.</summary>
        [TestMethod]
        public void P4_CopySign_NextAfter_ExactAndFloatTier()
        {
            var mag = np.array(new double[] { 3, 3, 3 });
            var sgn = np.array(new double[] { -1, -0.0, 1 });
            var cs = np.evaluate(NDExpr.CopySign(NDExpr.Arr(mag), NDExpr.Arr(sgn)));
            Assert.AreEqual(-3.0, cs.GetDouble(0), 0);
            Assert.AreEqual(-3.0, cs.GetDouble(1), 0);   // sign of -0.0 is copied
            Assert.AreEqual(3.0, cs.GetDouble(2), 0);

            var na = np.evaluate(NDExpr.NextAfter(NDExpr.Arr(np.array(new double[] { 1.0 })), NDExpr.Arr(np.array(new double[] { 2.0 }))));
            Assert.AreEqual(double.BitIncrement(1.0), na.GetDouble(0), 0);   // bit-exact

            // float-tier promotion (per input, like arctan2): int8×uint8→f16, int32×int32→f64, bool→f16.
            Assert.AreEqual(NPTypeCode.Half, np.evaluate(NDExpr.CopySign(NDExpr.Arr(np.array(new sbyte[] { 1 })), NDExpr.Arr(np.array(new byte[] { 2 })))).typecode);
            Assert.AreEqual(NPTypeCode.Double, np.evaluate(NDExpr.CopySign(NDExpr.Arr(np.array(new[] { 1 })), NDExpr.Arr(np.array(new[] { 2 })))).typecode);
            Assert.AreEqual(NPTypeCode.Half, np.evaluate(NDExpr.CopySign(NDExpr.Arr(np.array(new[] { true })), NDExpr.Arr(np.array(new[] { true })))).typecode);
        }

        /// <summary>logaddexp/logaddexp2/hypot/heaviside — values + float-tier dtype.</summary>
        [TestMethod]
        public void P4_LogAddExp_Hypot_Heaviside_Values()
        {
            var z = np.array(new double[] { 0.0 });
            Assert.AreEqual(System.Math.Log(2.0), np.evaluate(NDExpr.LogAddExp(NDExpr.Arr(z), NDExpr.Arr(z))).GetDouble(0), 1e-12);
            Assert.AreEqual(1.0, np.evaluate(NDExpr.LogAddExp2(NDExpr.Arr(z), NDExpr.Arr(z))).GetDouble(0), 1e-12);   // log2(2)

            var h = np.evaluate(NDExpr.Hypot(NDExpr.Arr(np.array(new double[] { 3 })), NDExpr.Arr(np.array(new double[] { 4 }))));
            Assert.AreEqual(5.0, h.GetDouble(0), 1e-12);

            // heaviside(x, h0): 0 if x<0, h0 if x==0, 1 if x>0. NOT commutative.
            var x = np.array(new double[] { -2, 0, 2 });
            var half = np.array(new double[] { 0.5, 0.5, 0.5 });
            var hv = np.evaluate(NDExpr.Heaviside(NDExpr.Arr(x), NDExpr.Arr(half)));
            Assert.AreEqual(0.0, hv.GetDouble(0), 0);
            Assert.AreEqual(0.5, hv.GetDouble(1), 0);   // the x==0 fill
            Assert.AreEqual(1.0, hv.GetDouble(2), 0);

            // all float-tier: an int pair → f64.
            Assert.AreEqual(NPTypeCode.Double, np.evaluate(NDExpr.Hypot(NDExpr.Arr(np.array(new[] { 3 })), NDExpr.Arr(np.array(new[] { 4 })))).typecode);
        }

        /// <summary>gcd/lcm — integer values (incl. NumPy's magnitude-wrap) and the integer dtype.</summary>
        [TestMethod]
        public void P4_GcdLcm_IntegerValuesAndDtype()
        {
            var a = np.array(new[] { 12, 15, 0 });
            var b = np.array(new[] { 8, 10, 0 });
            var g = np.evaluate(NDExpr.Gcd(NDExpr.Arr(a), NDExpr.Arr(b)));
            Assert.AreEqual(4, g.GetInt32(0));
            Assert.AreEqual(5, g.GetInt32(1));
            Assert.AreEqual(0, g.GetInt32(2));   // gcd(0,0) == 0

            var l = np.evaluate(NDExpr.Lcm(NDExpr.Arr(np.array(new[] { 4, 6 })), NDExpr.Arr(np.array(new[] { 6, 10 }))));
            Assert.AreEqual(12, l.GetInt32(0));
            Assert.AreEqual(30, l.GetInt32(1));

            // int8×uint8 → int16 (per-input promotion); bool+int32 is a VALID integer loop → int32.
            Assert.AreEqual(NPTypeCode.Int16, np.evaluate(NDExpr.Gcd(NDExpr.Arr(np.array(new sbyte[] { 12 })), NDExpr.Arr(np.array(new byte[] { 8 })))).typecode);
            Assert.AreEqual(NPTypeCode.Int32, np.evaluate(NDExpr.Gcd(NDExpr.Arr(np.array(new[] { true })), NDExpr.Arr(np.array(new[] { 6 })))).typecode);
        }

        /// <summary>left_shift/right_shift — values, the bool→int8 loop, and the overflow rule.</summary>
        [TestMethod]
        public void P4_Shifts_ValuesAndBoolToInt8()
        {
            var v = np.array(new[] { 1, 2, 16 });
            var c = np.array(new[] { 3, 1, 2 });
            var ls = np.evaluate(NDExpr.LeftShift(NDExpr.Arr(v), NDExpr.Arr(c)));
            Assert.AreEqual(8, ls.GetInt32(0));    // 1<<3
            Assert.AreEqual(4, ls.GetInt32(1));    // 2<<1
            var rs = np.evaluate(NDExpr.RightShift(NDExpr.Arr(v), NDExpr.Arr(c)));
            Assert.AreEqual(0, rs.GetInt32(0));    // 1>>3
            Assert.AreEqual(1, rs.GetInt32(1));    // 2>>1
            Assert.AreEqual(4, rs.GetInt32(2));    // 16>>2

            // a bool pair falls to the int8 loop: True<<True == 2.
            var bshift = np.evaluate(NDExpr.LeftShift(NDExpr.Arr(np.array(new[] { true })), NDExpr.Arr(np.array(new[] { true }))));
            Assert.AreEqual(NPTypeCode.SByte, bshift.typecode);
        }

        /// <summary>The new nodes compose as SUB-trees in the fused kernel (not only as a root).</summary>
        [TestMethod]
        public void P4_NodesComposeAsSubtrees()
        {
            var a = np.array(new double[] { 1, 5, 3 });
            var b = np.array(new double[] { 4, 2, 3 });
            // add(fmax(a,b), a) == [1+4, 5+5, 3+3] = [5, 10, 6]
            var r = np.evaluate(NDExpr.Add(NDExpr.FMax(NDExpr.Arr(a), NDExpr.Arr(b)), NDExpr.Arr(a)));
            Assert.AreEqual(5.0, r.GetDouble(0), 0);
            Assert.AreEqual(10.0, r.GetDouble(1), 0);
            Assert.AreEqual(6.0, r.GetDouble(2), 0);
        }

        /// <summary>The no-loop error taxonomy, reproduced from NumPy 2.4.2 (probed): the coercion
        /// "not supported for the input types" form for shift/fmod/copysign-family, and the
        /// signature-specific "did not contain a loop" form for gcd/lcm (incl. the bool/bool corner,
        /// which IS a no-loop even though bool+integer is fine).</summary>
        [TestMethod]
        public void P4_NoLoop_ErrorTaxonomy()
        {
            var f = np.array(new double[] { 1, 2 });
            var cx = np.array(new System.Numerics.Complex[] { new(1, 1) });

            // shift / fmod / copysign-family over a no-loop dtype → "not supported for the input types".
            StringAssert.Contains(
                Assert.ThrowsException<NotSupportedException>(() => np.evaluate(NDExpr.LeftShift(NDExpr.Arr(f), NDExpr.Arr(f)))).Message,
                "ufunc 'left_shift' not supported for the input types");
            StringAssert.Contains(
                Assert.ThrowsException<NotSupportedException>(() => np.evaluate(NDExpr.Fmod(NDExpr.Arr(cx), NDExpr.Arr(cx)))).Message,
                "ufunc 'fmod' not supported for the input types");
            StringAssert.Contains(
                Assert.ThrowsException<NotSupportedException>(() => np.evaluate(NDExpr.CopySign(NDExpr.Arr(cx), NDExpr.Arr(cx)))).Message,
                "ufunc 'copysign' not supported for the input types");

            // uint64 + signed shift → float64 → no loop (the same "not supported" form).
            StringAssert.Contains(
                Assert.ThrowsException<NotSupportedException>(() => np.evaluate(
                    NDExpr.LeftShift(NDExpr.Arr(np.array(new ulong[] { 1 })), NDExpr.Arr(np.array(new long[] { 1 }))))).Message,
                "not supported for the input types");

            // gcd/lcm no-loop uses the signature-specific "did not contain a loop" form.
            StringAssert.Contains(
                Assert.ThrowsException<NotSupportedException>(() => np.evaluate(NDExpr.Gcd(NDExpr.Arr(f), NDExpr.Arr(f)))).Message,
                "ufunc 'gcd' did not contain a loop with signature matching types");
            // bool/bool gcd is a no-loop (unlike bool+integer) — the NumPy corner.
            StringAssert.Contains(
                Assert.ThrowsException<NotSupportedException>(() => np.evaluate(
                    NDExpr.Lcm(NDExpr.Arr(np.array(new[] { true })), NDExpr.Arr(np.array(new[] { true }))))).Message,
                "ufunc 'lcm' did not contain a loop with signature matching types");
        }

        // =====================================================================
        // Phase 4.1 — unary node coverage (the engine unary ufuncs that gained
        // an NDExpr node). Values probed from NumPy 2.4.2; every fused cell is
        // BIT-EXACT (identity / sign-bit clear / popcount / pure-bit-increment
        // spacing), so no ULP excuse. These pin the corpus-unreachable edges:
        // signed zero, ±inf, the NaN sign bit, exact int→float magnitudes, and
        // the two distinct complex-rejection messages.
        // =====================================================================

        /// <summary>np.positive is the identity at every numeric dtype — but it is dtype-PRESERVING (it
        /// forces the operand's dtype onto the result), and it has NO bool loop (probed 2.4.2), so a
        /// bool child throws the "did not contain a loop" TypeError, not an all-True identity.</summary>
        [TestMethod]
        public void P41_Positive_Identity_And_BoolNoLoop()
        {
            var a = np.array(new double[] { -2.5, 0.0, 3.0 });
            var r = np.evaluate(NDExpr.Positive(NDExpr.Arr(a)));
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.AreEqual(-2.5, r.GetDouble(0), 0);
            Assert.AreEqual(3.0, r.GetDouble(2), 0);
            // dtype-preserving: positive(int32) is int32 (not the widened int64 a reduction would give).
            Assert.AreEqual(NPTypeCode.Int32, np.evaluate(NDExpr.Positive(NDExpr.Arr(np.array(new[] { 5 })))).typecode);

            StringAssert.Contains(
                Assert.ThrowsException<NotSupportedException>(() => np.evaluate(
                    NDExpr.Positive(NDExpr.Arr(np.array(new[] { true }))))).Message,
                "ufunc 'positive' did not contain a loop with signature matching types");
        }

        /// <summary>np.conjugate flips the imaginary sign for Complex, is the identity for every real
        /// float/int dtype — with ONE NumPy quirk: a bool child PROMOTES to int8 (probed 2.4.2), it
        /// does not stay bool.</summary>
        [TestMethod]
        public void P41_Conjugate_Complex_Real_BoolToInt8()
        {
            var z = np.array(new System.Numerics.Complex[] { new(1, 2), new(-3, 4) });
            var cz = np.evaluate(NDExpr.Conjugate(NDExpr.Arr(z)));
            Assert.AreEqual(NPTypeCode.Complex, cz.typecode);
            Assert.AreEqual(new System.Numerics.Complex(1, -2), cz.GetAtIndex<System.Numerics.Complex>(0));
            Assert.AreEqual(new System.Numerics.Complex(-3, -4), cz.GetAtIndex<System.Numerics.Complex>(1));

            // real identity (dtype preserved); Conj is the alias.
            var f = np.array(new double[] { -2.5, 3.0 });
            Assert.AreEqual(-2.5, np.evaluate(NDExpr.Conj(NDExpr.Arr(f))).GetDouble(0), 0);

            // conjugate(bool) → int8 (the promotion quirk).
            var cb = np.evaluate(NDExpr.Conjugate(NDExpr.Arr(np.array(new[] { true, false }))));
            Assert.AreEqual(NPTypeCode.SByte, cb.typecode);
            Assert.AreEqual((sbyte)1, cb.GetAtIndex<sbyte>(0));
            Assert.AreEqual((sbyte)0, cb.GetAtIndex<sbyte>(1));
        }

        /// <summary>np.fabs PROMOTES int/bool to the tier float (unlike Abs, which preserves int) and
        /// has NO complex loop. Because the promotion happens BEFORE the sign clear, fabs(int.MinValue)
        /// is the exact float magnitude, never the wrapped integer abs.</summary>
        [TestMethod]
        public void P41_Fabs_FloatPromote_ExactMagnitude_ComplexRejected()
        {
            // int8 → float16; the exact magnitude of int8 min (128), not the wrapped -128.
            var r8 = np.evaluate(NDExpr.Fabs(NDExpr.Arr(np.array(new sbyte[] { -128, -1, 5 }))));
            Assert.AreEqual(NPTypeCode.Half, r8.typecode);
            Assert.AreEqual(128.0f, (float)r8.GetAtIndex<Half>(0), 0);
            Assert.AreEqual(1.0f, (float)r8.GetAtIndex<Half>(1), 0);

            // int32 min → float64 magnitude 2147483648.0 (not the wrapped int abs).
            var r32 = np.evaluate(NDExpr.Fabs(NDExpr.Arr(np.array(new[] { int.MinValue }))));
            Assert.AreEqual(NPTypeCode.Double, r32.typecode);
            Assert.AreEqual(2147483648.0, r32.GetDouble(0), 0);

            // -0.0 → +0.0 (sign cleared); float64 preserved.
            var rf = np.evaluate(NDExpr.Fabs(NDExpr.Arr(np.array(new double[] { -0.0, -2.5 }))));
            Assert.AreEqual(0L, BitConverter.DoubleToInt64Bits(rf.GetDouble(0)));   // exactly +0.0
            Assert.AreEqual(2.5, rf.GetDouble(1), 0);

            StringAssert.Contains(
                Assert.ThrowsException<NotSupportedException>(() => np.evaluate(
                    NDExpr.Fabs(NDExpr.Arr(np.array(new System.Numerics.Complex[] { new(1, 1) }))))).Message,
                "ufunc 'fabs' not supported for the input types");
        }

        /// <summary>np.spacing is a FLOAT-only ufunc (int/bool promote to the tier float, complex has no
        /// loop). float32/float64 are SIGNED and the value is a pure bit increment — so a subnormal
        /// (spacing(0.0) == 5e-324) and a signed zero round-trip exactly.</summary>
        [TestMethod]
        public void P41_Spacing_SignedSubnormal_IntPromote_ComplexRejected()
        {
            var a = np.array(new double[] { 0.0, 1.0, -2.5 });
            var r = np.evaluate(NDExpr.Spacing(NDExpr.Arr(a)));
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.AreEqual(BitConverter.Int64BitsToDouble(1L), r.GetDouble(0));                 // 5e-324 (smallest subnormal)
            Assert.AreEqual(System.Math.Pow(2, -52), r.GetDouble(1), 0);                          // spacing(1.0) == 2^-52
            Assert.IsTrue(r.GetDouble(2) < 0);                                                    // signed: spacing carries x's sign

            // int → tier float (int32 → float64); the value is the float64 spacing of the promoted input.
            Assert.AreEqual(NPTypeCode.Double, np.evaluate(NDExpr.Spacing(NDExpr.Arr(np.array(new[] { 1 })))).typecode);

            StringAssert.Contains(
                Assert.ThrowsException<NotSupportedException>(() => np.evaluate(
                    NDExpr.Spacing(NDExpr.Arr(np.array(new System.Numerics.Complex[] { new(1, 1) }))))).Message,
                "ufunc 'spacing' not supported for the input types");
        }

        /// <summary>np.signbit reads the RAW IEEE sign bit — result always bool. NOT <c>x &lt; 0</c>: a
        /// signed zero and a negative NaN are True while +0.0, +inf and a positive NaN are False.
        /// Signed integers use the two's-complement MSB; unsigned are always False; complex has no
        /// loop.</summary>
        [TestMethod]
        public void P41_SignBit_RawSignBit_IncludingSignedZeroAndNaN()
        {
            double negNaN = BitConverter.Int64BitsToDouble(unchecked((long)0xFFF8000000000000UL));
            double posNaN = BitConverter.Int64BitsToDouble(0x7FF8000000000000L);
            var a = np.array(new double[] { -0.0, 0.0, negNaN, posNaN, double.PositiveInfinity, double.NegativeInfinity, -2.5 });
            var r = np.evaluate(NDExpr.SignBit(NDExpr.Arr(a)));
            Assert.AreEqual(NPTypeCode.Boolean, r.typecode);
            Assert.IsTrue(r.GetBoolean(0));    // -0.0  → True (sign bit set)
            Assert.IsFalse(r.GetBoolean(1));   // +0.0  → False
            Assert.IsTrue(r.GetBoolean(2));    // -NaN  → True
            Assert.IsFalse(r.GetBoolean(3));   // +NaN  → False
            Assert.IsFalse(r.GetBoolean(4));   // +inf  → False
            Assert.IsTrue(r.GetBoolean(5));    // -inf  → True
            Assert.IsTrue(r.GetBoolean(6));    // -2.5  → True

            // signed int MSB; unsigned always False.
            var ri = np.evaluate(NDExpr.SignBit(NDExpr.Arr(np.array(new[] { -5, 0, 7 }))));
            Assert.IsTrue(ri.GetBoolean(0));
            Assert.IsFalse(ri.GetBoolean(1));
            Assert.IsFalse(np.evaluate(NDExpr.SignBit(NDExpr.Arr(np.array(new byte[] { 200 })))).GetBoolean(0));

            StringAssert.Contains(
                Assert.ThrowsException<NotSupportedException>(() => np.evaluate(
                    NDExpr.SignBit(NDExpr.Arr(np.array(new System.Numerics.Complex[] { new(1, 1) }))))).Message,
                "ufunc 'signbit' not supported for the input types");
        }

        /// <summary>np.isposinf / np.isneginf are FUNCTIONS (bool result, integers all-False). A complex
        /// input raises the AMBIGUITY TypeError — a DIFFERENT message from the ufunc no-loop family.</summary>
        [TestMethod]
        public void P41_IsPosInf_IsNegInf_IntegerFalse_ComplexAmbiguity()
        {
            var a = np.array(new double[] { double.PositiveInfinity, double.NegativeInfinity, 3.0, double.NaN });
            var pos = np.evaluate(NDExpr.IsPosInf(NDExpr.Arr(a)));
            var neg = np.evaluate(NDExpr.IsNegInf(NDExpr.Arr(a)));
            Assert.AreEqual(NPTypeCode.Boolean, pos.typecode);
            Assert.IsTrue(pos.GetBoolean(0));  Assert.IsFalse(pos.GetBoolean(1)); Assert.IsFalse(pos.GetBoolean(3));
            Assert.IsFalse(neg.GetBoolean(0)); Assert.IsTrue(neg.GetBoolean(1));  Assert.IsFalse(neg.GetBoolean(3));

            // integer input → all-False (no infinity).
            Assert.IsFalse(np.evaluate(NDExpr.IsPosInf(NDExpr.Arr(np.array(new[] { 7 })))).GetBoolean(0));

            var cx = np.array(new System.Numerics.Complex[] { new(1, 1) });
            StringAssert.Contains(
                Assert.ThrowsException<NotSupportedException>(() => np.evaluate(NDExpr.IsPosInf(NDExpr.Arr(cx)))).Message,
                "not supported for complex128 values because it would be ambiguous");
            StringAssert.Contains(
                Assert.ThrowsException<NotSupportedException>(() => np.evaluate(NDExpr.IsNegInf(NDExpr.Arr(cx)))).Message,
                "not supported for complex128 values because it would be ambiguous");
        }

        /// <summary>np.bitwise_count counts the set bits of |x| into a uint8, INTEGER/BOOL/CHAR only. A
        /// signed negative counts the MAGNITUDE (bitwise_count(-1) == 1, not 8); float/float16/complex
        /// have no loop.</summary>
        [TestMethod]
        public void P41_BitwiseCount_Popcount_IntegerOnly()
        {
            var r = np.evaluate(NDExpr.BitwiseCount(NDExpr.Arr(np.array(new[] { -1, 0, int.MaxValue, -2147483648 }))));
            Assert.AreEqual(NPTypeCode.Byte, r.typecode);
            Assert.AreEqual((byte)1, r.GetByte(0));    // |-1| == 1 bit
            Assert.AreEqual((byte)0, r.GetByte(1));
            Assert.AreEqual((byte)31, r.GetByte(2));   // 2^31-1
            Assert.AreEqual((byte)1, r.GetByte(3));    // int32 min → magnitude wraps to itself, 1 bit

            // bool → 1/0; uint8 255 → 8.
            var rb = np.evaluate(NDExpr.BitwiseCount(NDExpr.Arr(np.array(new[] { true, false }))));
            Assert.AreEqual((byte)1, rb.GetByte(0));
            Assert.AreEqual((byte)0, rb.GetByte(1));
            Assert.AreEqual((byte)8, np.evaluate(NDExpr.BitwiseCount(NDExpr.Arr(np.array(new byte[] { 255 })))).GetByte(0));

            // float / float16 / complex have no loop.
            foreach (var op in new[] { NDExpr.BitwiseCount(NDExpr.Arr(np.array(new double[] { 1 }))),
                                       NDExpr.BitwiseCount(NDExpr.Arr(np.array(new Half[] { (Half)1 }))),
                                       NDExpr.BitwiseCount(NDExpr.Arr(np.array(new System.Numerics.Complex[] { new(1, 1) }))) })
                StringAssert.Contains(
                    Assert.ThrowsException<NotSupportedException>(() => np.evaluate(op)).Message,
                    "ufunc 'bitwise_count' not supported for the input types");
        }

        /// <summary>The unary nodes compose as SUB-trees in the fused kernel, and the two SIMD-enabled
        /// ones (Positive/Fabs) produce bit-identical results on the vector and scalar paths — the
        /// metamorphic contract that gates the vectorized emit.</summary>
        [TestMethod]
        public void P41_UnaryNodes_ComposeAndVectorizeBitExact()
        {
            // composition: mul(fabs(a), 2) promotes int→float then multiplies.
            var comp = np.evaluate(NDExpr.Multiply(NDExpr.Fabs(NDExpr.Arr(np.array(new double[] { -3, 4, -1.5 }))), (NDExpr)2.0));
            Assert.AreEqual(6.0, comp.GetDouble(0), 0);
            Assert.AreEqual(8.0, comp.GetDouble(1), 0);
            Assert.AreEqual(3.0, comp.GetDouble(2), 0);

            // vector == scalar, bit-for-bit, over an array large enough to engage SIMD (Positive/Fabs).
            var pool = new double[100];
            for (int i = 0; i < pool.Length; i++) pool[i] = (i % 7) - 3.25 + (i % 3 == 0 ? -0.0 : 0.0);
            var big = np.array(pool);
            foreach (var op in new Func<NDExpr, NDExpr>[] { NDExpr.Positive, NDExpr.Fabs })
            {
                var vec = np.evaluate(op(NDExpr.Arr(big)));
                NDArray sca;
                NDExpr.ForceScalar = true;
                try { sca = np.evaluate(op(NDExpr.Arr(big))); }
                finally { NDExpr.ForceScalar = false; }
                for (int i = 0; i < pool.Length; i++)
                    Assert.AreEqual(BitConverter.DoubleToInt64Bits(sca.GetDouble(i)),
                                    BitConverter.DoubleToInt64Bits(vec.GetDouble(i)),
                                    $"vector≠scalar at {i}");
            }
        }

        // ===================================================================
        // Phase 4.1b — dtype-CHANGING unary nodes (this milestone: Rint)
        // ===================================================================

        /// <summary>np.rint is the TRUE ufunc round-half-to-even. Its VALUE equals <see cref="NDExpr.Round"/>
        /// (both banker's rounding; Rint aliases Round's kernel), but its DTYPE is a float TIER
        /// (bool/int8/uint8→float16, int16/uint16→float32, int32+→float64; float/complex/decimal preserved)
        /// where <c>Round</c> PRESERVES the input dtype — so <c>Rint(int32)</c> is a float64 while
        /// <c>Round(int32)</c> is the int32 identity. Complex rounds both lanes.</summary>
        [TestMethod]
        public void P41b_Rint_FloatTier_HalfToEven_ComplexBothLanes()
        {
            // int32 → float64 (PROMOTES, unlike Round which keeps int32) — the defining difference.
            var ri = np.evaluate(NDExpr.Rint(NDExpr.Arr(np.array(new[] { 1, 2, 3 }))));
            Assert.AreEqual(NPTypeCode.Double, ri.typecode);
            Assert.AreEqual(2.0, ri.GetDouble(1), 0);
            // Contrast: Round on the same int32 preserves int32 (the identity).
            Assert.AreEqual(NPTypeCode.Int32, np.evaluate(NDExpr.Round(NDExpr.Arr(np.array(new[] { 1, 2, 3 })))).typecode);

            // int8 → float16, bool → float16 (the low tiers).
            Assert.AreEqual(NPTypeCode.Half, np.evaluate(NDExpr.Rint(NDExpr.Arr(np.array(new sbyte[] { 5 })))).typecode);
            Assert.AreEqual(NPTypeCode.Half, np.evaluate(NDExpr.Rint(NDExpr.Arr(np.array(new[] { 1 }).astype(NPTypeCode.Boolean)))).typecode);
            // int16 → float32.
            Assert.AreEqual(NPTypeCode.Single, np.evaluate(NDExpr.Rint(NDExpr.Arr(np.array(new short[] { 5 })))).typecode);

            // Half-to-even (banker's) on float64, preserved dtype: 0.5→0, 2.5→2, 3.5→4, -2.5→-2, -1.5→-2.
            var rf = np.evaluate(NDExpr.Rint(NDExpr.Arr(np.array(new double[] { 0.5, 2.5, 3.5, -2.5, -1.5 }))));
            Assert.AreEqual(NPTypeCode.Double, rf.typecode);
            Assert.AreEqual(0.0, rf.GetDouble(0), 0);
            Assert.AreEqual(2.0, rf.GetDouble(1), 0);
            Assert.AreEqual(4.0, rf.GetDouble(2), 0);
            Assert.AreEqual(-2.0, rf.GetDouble(3), 0);
            Assert.AreEqual(-2.0, rf.GetDouble(4), 0);

            // Complex rounds real and imaginary separately (complex128 preserved).
            var rc = np.evaluate(NDExpr.Rint(NDExpr.Arr(np.array(new System.Numerics.Complex[] { new(2.5, 3.5) }))));
            Assert.AreEqual(NPTypeCode.Complex, rc.typecode);
            var z = rc.GetAtIndex<System.Numerics.Complex>(0);
            Assert.AreEqual(2.0, z.Real, 0);
            Assert.AreEqual(4.0, z.Imaginary, 0);

            // ±inf / NaN / -0.0 pass through the rounding unchanged (float64 preserved).
            var rs = np.evaluate(NDExpr.Rint(NDExpr.Arr(np.array(new double[] { double.PositiveInfinity, -0.0 }))));
            Assert.IsTrue(double.IsPositiveInfinity(rs.GetDouble(0)));
            Assert.AreEqual(long.MinValue, BitConverter.DoubleToInt64Bits(rs.GetDouble(1)));  // -0.0 sign preserved
        }

        /// <summary>Rint vectorizes at float32/float64 (it shares Round's Vector.Round body) — the vector
        /// and scalar paths agree bit-for-bit over a pool full of .5 ties — and it composes as a sub-tree
        /// (a genuine fused rint of a non-integer product).</summary>
        [TestMethod]
        public void P41b_Rint_VectorizesBitExact_And_Composes()
        {
            // A pool of exact half-integers so banker's rounding is exercised on every element.
            var pool = new double[128];
            for (int i = 0; i < pool.Length; i++) pool[i] = (i - 64) * 0.5;
            var big = np.array(pool);
            foreach (var tc in new[] { NPTypeCode.Single, NPTypeCode.Double })
            {
                var a = big.astype(tc);
                var vec = np.evaluate(NDExpr.Rint(NDExpr.Arr(a)));
                NDArray sca;
                NDExpr.ForceScalar = true;
                try { sca = np.evaluate(NDExpr.Rint(NDExpr.Arr(a))); }
                finally { NDExpr.ForceScalar = false; }
                Assert.AreEqual(tc, vec.typecode);
                // Read through the element's OWN width (up-cast to double) — GetDouble on a float32 array
                // would index as 8-byte elements and read garbage past the buffer.
                var vd = vec.astype(NPTypeCode.Double); var sd = sca.astype(NPTypeCode.Double);
                for (int i = 0; i < pool.Length; i++)
                    Assert.AreEqual(sd.GetDouble(i), vd.GetDouble(i), 0, $"{tc} vector≠scalar at {i}");
            }

            // Sub-tree: rint(a*1.5) + 1 — the product is non-integer, so rint does real work here.
            var v = np.array(new double[] { 1, 2, 3, 4 });                 // *1.5 → 1.5, 3.0, 4.5, 6.0
            var comp = np.evaluate(NDExpr.Rint((NDExpr)v * 1.5) + (NDExpr)1.0);
            Assert.AreEqual(3.0, comp.GetDouble(0), 0);   // rint(1.5)=2 (even) +1
            Assert.AreEqual(4.0, comp.GetDouble(1), 0);   // rint(3.0)=3 +1
            Assert.AreEqual(5.0, comp.GetDouble(2), 0);   // rint(4.5)=4 (even) +1
            Assert.AreEqual(7.0, comp.GetDouble(3), 0);   // rint(6.0)=6 +1
        }

        // ---------------------------------------------------------------------
        // Phase 4.1b — the complex→real component extractors (np.real/imag/angle).
        // NOT ufuncs: complex128 → float64 (lane extract / atan2), while a REAL child
        // degenerates — real is the identity (dtype PRESERVED), imag is zeros (dtype
        // PRESERVED, value unread), angle is atan2(0,x) at NumPy's per-dtype float tier.
        // ---------------------------------------------------------------------

        /// <summary><c>Real</c>: a complex child yields its real lane as float64; every REAL child is the
        /// IDENTITY with its dtype PRESERVED (real(int32) is int32, not a float) — the value AND dtype the
        /// fused node must reproduce. Covers the two typing branches and the extraction/identity emit.</summary>
        [TestMethod]
        public void P41b_Real_ComplexLaneFloat64_RealIdentityDtypePreserved()
        {
            // Complex → float64 real lane (extraction), including a NaN part passing through.
            var rc = np.evaluate(NDExpr.Real(NDExpr.Arr(np.array(new System.Numerics.Complex[]
                { new(3, 4), new(-2.5, 7), new(double.NaN, 1) }))));
            Assert.AreEqual(NPTypeCode.Double, rc.typecode);
            Assert.AreEqual(3.0, rc.GetDouble(0), 0);
            Assert.AreEqual(-2.5, rc.GetDouble(1), 0);
            Assert.IsTrue(double.IsNaN(rc.GetDouble(2)));

            // Real int → int32 IDENTITY (dtype preserved, values unchanged incl. negatives).
            var ri = np.evaluate(NDExpr.Real(NDExpr.Arr(np.array(new[] { -5, 0, 7 }))));
            Assert.AreEqual(NPTypeCode.Int32, ri.typecode);
            Assert.AreEqual(-5, ri.GetInt32(0));
            Assert.AreEqual(0, ri.GetInt32(1));
            Assert.AreEqual(7, ri.GetInt32(2));

            // Real float32 preserves float32 (not promoted to double).
            var rf = np.evaluate(NDExpr.Real(NDExpr.Arr(np.array(new[] { 1.5f, -2.5f }))));
            Assert.AreEqual(NPTypeCode.Single, rf.typecode);
            Assert.AreEqual(1.5, rf.astype(NPTypeCode.Double).GetDouble(0), 0);
        }

        /// <summary><c>Imag</c>: a complex child yields its imaginary lane as float64; every REAL child yields
        /// ZERO with its dtype PRESERVED (imag(int32) is int32 zeros) — and the result does not depend on the
        /// child value, matching np.imag's <c>zeros_like</c>.</summary>
        [TestMethod]
        public void P41b_Imag_ComplexLaneFloat64_RealZerosDtypePreserved()
        {
            var ic = np.evaluate(NDExpr.Imag(NDExpr.Arr(np.array(new System.Numerics.Complex[]
                { new(3, 4), new(-2.5, -7), new(1, double.NaN) }))));
            Assert.AreEqual(NPTypeCode.Double, ic.typecode);
            Assert.AreEqual(4.0, ic.GetDouble(0), 0);
            Assert.AreEqual(-7.0, ic.GetDouble(1), 0);
            Assert.IsTrue(double.IsNaN(ic.GetDouble(2)));

            // Imag int → int32 ZEROS (dtype preserved), regardless of the input values.
            var ii = np.evaluate(NDExpr.Imag(NDExpr.Arr(np.array(new[] { -5, 99, 7 }))));
            Assert.AreEqual(NPTypeCode.Int32, ii.typecode);
            Assert.AreEqual(0, ii.GetInt32(0));
            Assert.AreEqual(0, ii.GetInt32(1));
            Assert.AreEqual(0, ii.GetInt32(2));

            // Imag float32 → float32 zeros.
            var iflt = np.evaluate(NDExpr.Imag(NDExpr.Arr(np.array(new[] { 1.5f, -2.5f }))));
            Assert.AreEqual(NPTypeCode.Single, iflt.typecode);
            Assert.AreEqual(0.0, iflt.astype(NPTypeCode.Double).GetDouble(0), 0);
            Assert.AreEqual(0.0, iflt.astype(NPTypeCode.Double).GetDouble(1), 0);
        }

        /// <summary><c>Angle</c> (radians): a complex child is <c>atan2(imag, real)</c> → float64; a REAL child
        /// is <c>atan2(0, x)</c> (0 for x ≥ 0, pi for x &lt; 0, incl. <c>-0.0</c> → pi) at NumPy's per-dtype
        /// float tier — int8/uint8/f16 → f16, int16/uint16/char/f32 → f32, bool/int32+/f64 → f64. Radians only:
        /// the deg-style scale is a caller composition, verified here too.</summary>
        [TestMethod]
        public void P41b_Angle_Radians_ComplexAndRealFloatTier()
        {
            double pi = System.Math.PI;
            // Complex → float64 atan2(im, re).
            var ac = np.evaluate(NDExpr.Angle(NDExpr.Arr(np.array(new System.Numerics.Complex[]
                { new(1, 0), new(0, 1), new(-1, 0), new(0, -1) }))));
            Assert.AreEqual(NPTypeCode.Double, ac.typecode);
            Assert.AreEqual(0.0, ac.GetDouble(0), 0);
            Assert.AreEqual(pi / 2, ac.GetDouble(1), 0);
            Assert.AreEqual(pi, ac.GetDouble(2), 0);
            Assert.AreEqual(-pi / 2, ac.GetDouble(3), 0);

            // Real tier: int8 → float16, int16 → float32, int32 → float64. atan2(0, x) is 0 / pi.
            Assert.AreEqual(NPTypeCode.Half, np.evaluate(NDExpr.Angle(NDExpr.Arr(np.array(new sbyte[] { -1 })))).typecode);
            Assert.AreEqual(NPTypeCode.Single, np.evaluate(NDExpr.Angle(NDExpr.Arr(np.array(new short[] { -1 })))).typecode);
            var ai = np.evaluate(NDExpr.Angle(NDExpr.Arr(np.array(new[] { -3, 0, 5 }))));
            Assert.AreEqual(NPTypeCode.Double, ai.typecode);
            Assert.AreEqual(pi, ai.GetDouble(0), 0);      // x < 0 → pi
            Assert.AreEqual(0.0, ai.GetDouble(1), 0);     // x == 0 → 0
            Assert.AreEqual(0.0, ai.GetDouble(2), 0);     // x > 0 → 0

            // A NEGATIVE ZERO takes the negative branch: atan2(0, -0.0) == pi (not 0).
            var az = np.evaluate(NDExpr.Angle(NDExpr.Arr(np.array(new double[] { 0.0, -0.0, 2.0 }))));
            Assert.AreEqual(0.0, az.GetDouble(0), 0);
            Assert.AreEqual(pi, az.GetDouble(1), 0);
            Assert.AreEqual(0.0, az.GetDouble(2), 0);

            // Deg-style composition (fused primitive is radians): angle(x) * (180/pi).
            var deg = np.evaluate(NDExpr.Angle(NDExpr.Arr(np.array(new[] { -1 }))) * (180.0 / pi));
            Assert.AreEqual(180.0, deg.GetDouble(0), 1e-12);
        }

        /// <summary>The component extractors over NumSharp's <c>Char</c> dtype (no NumPy analog): real/imag
        /// preserve char (identity / zeros), angle takes the uint16-tier float32; and they COMPOSE in the
        /// fused kernel — <c>real(z) + imag(z)</c> recombines the two float64 lanes of a complex child in one
        /// pass. Both pin cells the ALL_DTYPES oracle sweep cannot reach (char) or only reaches as a sub-tree.</summary>
        [TestMethod]
        public void P41b_Components_Char_And_ComposeInOnePass()
        {
            var cs = np.array(new[] { 'A', 'z' });                    // char = uint16-like
            Assert.AreEqual(NPTypeCode.Char, np.evaluate(NDExpr.Real(NDExpr.Arr(cs))).typecode);   // real(char)=char identity
            var ic = np.evaluate(NDExpr.Imag(NDExpr.Arr(cs)));
            Assert.AreEqual(NPTypeCode.Char, ic.typecode);                                          // imag(char)=char zeros
            Assert.AreEqual('\0', (char)ic.GetAtIndex<char>(0));
            Assert.AreEqual(NPTypeCode.Single, np.evaluate(NDExpr.Angle(NDExpr.Arr(cs))).typecode); // angle(char)→float32 (uint16 tier)

            // Fused recombination: add(real, imag) reads a complex child's two lanes and sums them in one pass.
            var z = np.array(new System.Numerics.Complex[] { new(3, 4), new(-1, 2) });
            var rec = np.evaluate(NDExpr.Real(NDExpr.Arr(z)) + NDExpr.Imag(NDExpr.Arr(z)));
            Assert.AreEqual(NPTypeCode.Double, rec.typecode);
            Assert.AreEqual(7.0, rec.GetDouble(0), 0);   // 3 + 4
            Assert.AreEqual(1.0, rec.GetDouble(1), 0);   // -1 + 2
        }

        // ---------------------------------------------------------------------
        // Phase 4.1b — the Cast node (np.ndarray.astype, casting='unsafe'). The safe-pool
        // oracle sweep (src×target, non-negative values) covers the common matrix bit-exactly;
        // these pin the edges it deliberately excludes — the two conversions EmitConvertTo
        // mis-handled at node edges (float→bool, complex→real) and the C-undefined float→int
        // cells (verified ≡ the engine's own host-pinned astype, i.e. NumPy on win-amd64).
        // ---------------------------------------------------------------------

        /// <summary><c>Cast(float, bool)</c> is the NONZERO test (0→False, everything else incl. NaN/±inf→True).
        /// EmitConvertTo's <c>to==Boolean</c> branch did a double-vs-int stack compare (only ever reached from a
        /// 0/1 Int32 at node edges); Cast is its first float/complex caller, so CastNode emits the type-correct
        /// <c>child != 0</c> via the comparison kernel. NumPy-probed: <c>[0,5,-2,nan,inf].astype(bool)</c> =
        /// [F,T,T,T,T].</summary>
        [TestMethod]
        public void P41bCast_FloatToBool_IsNonzeroTest_NaNIsTrue()
        {
            var r = np.evaluate(NDExpr.Cast(NDExpr.Arr(np.array(new double[] { 0.0, 5.0, -2.0, double.NaN, double.PositiveInfinity })), NPTypeCode.Boolean));
            Assert.AreEqual(NPTypeCode.Boolean, r.typecode);
            CollectionAssert.AreEqual(new[] { false, true, true, true, true },
                new[] { r.GetBoolean(0), r.GetBoolean(1), r.GetBoolean(2), r.GetBoolean(3), r.GetBoolean(4) });
            // -0.0 is still zero → False (the sign bit does not make it nonzero).
            Assert.IsFalse(np.evaluate(NDExpr.Cast(NDExpr.Arr(np.array(new double[] { -0.0 })), NPTypeCode.Boolean)).GetBoolean(0));
        }

        /// <summary><c>Cast(complex, real/int)</c> takes the REAL part then converts (NumPy's ComplexWarning rule):
        /// <c>[3+4j,-2.7+9j,0+1j].astype(int32)</c> = [3,-2,0] (real part truncated toward zero), <c>[3.9+4j]
        /// .astype(float32)</c> = the float32 real part. <c>Cast(complex, bool)</c> is any-part-nonzero:
        /// <c>[0+0j,0+1j,2+0j].astype(bool)</c> = [F,T,T]. EmitConvertTo mis-handled complex→int and
        /// ACCESS-VIOLATED on complex→Half; CastNode extracts the real lane first.</summary>
        [TestMethod]
        public void P41bCast_ComplexToReal_TakesRealPart_AndToBool_AnyPart()
        {
            var ci = np.evaluate(NDExpr.Cast(NDExpr.Arr(np.array(new System.Numerics.Complex[] { new(3, 4), new(-2.7, 9), new(0, 1) })), NPTypeCode.Int32));
            Assert.AreEqual(NPTypeCode.Int32, ci.typecode);
            Assert.AreEqual(3, ci.GetInt32(0));
            Assert.AreEqual(-2, ci.GetInt32(1));   // real part -2.7 truncated toward zero
            Assert.AreEqual(0, ci.GetInt32(2));

            var cf = np.evaluate(NDExpr.Cast(NDExpr.Arr(np.array(new System.Numerics.Complex[] { new(3.9, 4) })), NPTypeCode.Single));
            Assert.AreEqual(NPTypeCode.Single, cf.typecode);
            Assert.AreEqual((double)(float)3.9, cf.astype(NPTypeCode.Double).GetDouble(0), 0);   // real part as float32

            // complex → Half no longer AVs (extract real → double → Half).
            var ch = np.evaluate(NDExpr.Cast(NDExpr.Arr(np.array(new System.Numerics.Complex[] { new(2.5, 9) })), NPTypeCode.Half));
            Assert.AreEqual(NPTypeCode.Half, ch.typecode);
            Assert.AreEqual(2.5, ch.astype(NPTypeCode.Double).GetDouble(0), 0);

            var cb = np.evaluate(NDExpr.Cast(NDExpr.Arr(np.array(new System.Numerics.Complex[] { new(0, 0), new(0, 1), new(2, 0) })), NPTypeCode.Boolean));
            CollectionAssert.AreEqual(new[] { false, true, true }, new[] { cb.GetBoolean(0), cb.GetBoolean(1), cb.GetBoolean(2) });
        }

        /// <summary>The C-undefined float→integer edges (NaN/±inf/out-of-range/negative→unsigned) go through the
        /// SAME NumPy-faithful <c>Converts.*</c> table the engine's <c>astype</c> uses, so a fused <c>Cast</c> is
        /// bit-identical to <c>expr.astype(target)</c> — the host-pinned NumPy answer on win-amd64. Pinned as a
        /// metamorphic identity (fused == engine) plus the NumPy-probed wrap values.</summary>
        [TestMethod]
        public void P41bCast_FloatToInt_CUndefinedEdges_MatchEngineAstype()
        {
            var edge = np.array(new double[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity, 300.0, -300.0, -2.9, 2.9 });
            foreach (var (tc, _) in new[] { (NPTypeCode.SByte, 0), (NPTypeCode.Byte, 0), (NPTypeCode.Int32, 0), (NPTypeCode.UInt32, 0), (NPTypeCode.Int64, 0), (NPTypeCode.UInt64, 0) })
            {
                var fused = np.evaluate(NDExpr.Cast(NDExpr.Arr(edge), tc)).astype(NPTypeCode.Int64);
                var eng = edge.astype(tc).astype(NPTypeCode.Int64);   // the host-pinned answer (astype_full-gated)
                for (int i = 0; i < edge.size; i++)
                    Assert.AreEqual(eng.GetInt64(i), fused.GetInt64(i), $"{tc}[{i}] fused≠engine");
            }
            // NumPy-probed wrap pins (win-amd64): 300→int8 44, -300→int8 -44, -2.9→int32 -2, 300→uint8 44, -1→uint8 255.
            var i8 = np.evaluate(NDExpr.Cast(NDExpr.Arr(np.array(new double[] { 300.0, -300.0 })), NPTypeCode.SByte)).astype(NPTypeCode.Int64);
            Assert.AreEqual(44L, i8.GetInt64(0)); Assert.AreEqual(-44L, i8.GetInt64(1));
            var u8 = np.evaluate(NDExpr.Cast(NDExpr.Arr(np.array(new double[] { 300.0, -1.0 })), NPTypeCode.Byte)).astype(NPTypeCode.Int64);
            Assert.AreEqual(44L, u8.GetInt64(0)); Assert.AreEqual(255L, u8.GetInt64(1));
        }

        /// <summary>Cast covers the dtypes the ALL_DTYPES oracle sweep omits (char / decimal, as source AND target),
        /// wraps integer overflow (NumPy modular), is the identity to the same dtype, and COMPOSES in the fused
        /// kernel — <c>Cast(a+b, int32)</c> truncates a fused sum.</summary>
        [TestMethod]
        public void P41bCast_CharDecimal_Wrap_Identity_And_Composes()
        {
            // int32 → char (code unit) and char → int32.
            Assert.AreEqual('A', (char)np.evaluate(NDExpr.Cast(NDExpr.Arr(np.array(new[] { 65 })), NPTypeCode.Char)).GetAtIndex<char>(0));
            Assert.AreEqual(90, np.evaluate(NDExpr.Cast(NDExpr.Arr(np.array(new[] { 'Z' })), NPTypeCode.Int32)).GetInt32(0));
            // float64 → decimal → float64 (exact for 2.5).
            var dec = np.evaluate(NDExpr.Cast(NDExpr.Arr(np.array(new double[] { 2.5 })), NPTypeCode.Decimal));
            Assert.AreEqual(NPTypeCode.Decimal, dec.typecode);
            Assert.AreEqual(2.5m, dec.GetAtIndex<decimal>(0));
            // Integer overflow WRAPS (int32 300 → int8 44, -1 → uint8 255) — NumPy modular.
            Assert.AreEqual(44L, np.evaluate(NDExpr.Cast(NDExpr.Arr(np.array(new[] { 300 })), NPTypeCode.SByte)).astype(NPTypeCode.Int64).GetInt64(0));
            Assert.AreEqual(255L, np.evaluate(NDExpr.Cast(NDExpr.Arr(np.array(new[] { -1 })), NPTypeCode.Byte)).astype(NPTypeCode.Int64).GetInt64(0));
            // Identity: Cast to the same dtype is a value-preserving no-op.
            var id = np.evaluate(NDExpr.Cast(NDExpr.Arr(np.array(new[] { 1, 2, 3 })), NPTypeCode.Int32));
            Assert.AreEqual(NPTypeCode.Int32, id.typecode);
            CollectionAssert.AreEqual(new[] { 1, 2, 3 }, new[] { id.GetInt32(0), id.GetInt32(1), id.GetInt32(2) });
            // Composition: cast(a+b, int32) truncates the fused sum toward zero.
            var a = np.array(new double[] { 1.7, 2.3, 3.9 }); var b = np.array(new double[] { 0.4, 0.4, 0.4 });
            var comp = np.evaluate(NDExpr.Cast((NDExpr)a + (NDExpr)b, NPTypeCode.Int32));
            Assert.AreEqual(NPTypeCode.Int32, comp.typecode);
            CollectionAssert.AreEqual(new[] { 2, 2, 4 }, new[] { comp.GetInt32(0), comp.GetInt32(1), comp.GetInt32(2) }); // 2.1→2, 2.7→2, 4.3→4
            // The Type overload resolves the same node.
            Assert.AreEqual(NPTypeCode.Single, np.evaluate(NDExpr.Cast(NDExpr.Arr(np.array(new[] { 1 })), typeof(float))).typecode);
        }

        // ---------------------------------------------------------------------
        // Phase 4.1b — Round(x, decimals) (np.round with decimals != 0). A dtype-PRESERVING port of
        // PyArray_Round. NB: NumSharp's ENGINE np.around is buggy (negative decimals throw, complex not
        // rounded), so the fused RoundNode is the CORRECT reference; these pins are NumPy-probed values.
        // ---------------------------------------------------------------------

        /// <summary>Float rounding is <c>op2(rint(op1(x, 10^|d|)), 10^|d|)</c> AT THE INPUT'S OWN FLOAT dtype,
        /// dtype preserved: <c>round([1.2345,2.5,-2.675,3.15], 2)</c> = [1.23, 2.5, -2.68, 3.15] (banker's),
        /// negative decimals round to tens (<c>round([123.456,250], -2)</c> = [100, 200]), and float32 rounds
        /// at float32 precision (<c>round(float32 1.005, 2)</c> = 1.0, where float64 gives 1.0 too here).</summary>
        [TestMethod]
        public void P41bRound_Float_AtInputPrecision_DtypePreserved()
        {
            var r = np.evaluate(NDExpr.Round(NDExpr.Arr(np.array(new double[] { 1.2345, 2.5, -2.675, 3.15 })), 2));
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            CollectionAssert.AreEqual(new[] { 1.23, 2.5, -2.68, 3.15 }, new[] { r.GetDouble(0), r.GetDouble(1), r.GetDouble(2), r.GetDouble(3) });
            var rn = np.evaluate(NDExpr.Round(NDExpr.Arr(np.array(new double[] { 123.456, 250.0 })), -2));
            CollectionAssert.AreEqual(new[] { 100.0, 200.0 }, new[] { rn.GetDouble(0), rn.GetDouble(1) });
            // float32 preserves float32 and rounds at float32 precision.
            var rf = np.evaluate(NDExpr.Round(NDExpr.Arr(np.array(new[] { 1.005f })), 2));
            Assert.AreEqual(NPTypeCode.Single, rf.typecode);
            Assert.AreEqual(1.0, rf.astype(NPTypeCode.Double).GetDouble(0), 0);
            // float16 preserves float16.
            Assert.AreEqual(NPTypeCode.Half, np.evaluate(NDExpr.Round(NDExpr.Arr(np.array(new[] { (System.Half)1.2345f })), 2)).typecode);
        }

        /// <summary>Integer + decimals ≥ 0 is the IDENTITY (no fractional part — and NO float round-trip, so a
        /// huge int64 is untouched); integer + decimals &lt; 0 runs in float64 and CASTS BACK to the integer
        /// dtype, so an out-of-range result WRAPS. NumPy-probed: <c>round(int64 2^60, 2)</c> = 2^60,
        /// <c>round(int32 [12345,-6789], -2)</c> = [12300,-6800], <c>round(int8 127, -1)</c> = -126,
        /// <c>round(uint8 255, -1)</c> = 4.</summary>
        [TestMethod]
        public void P41bRound_Integer_IdentityForNonneg_WrapForNegativeDecimals()
        {
            var big = np.evaluate(NDExpr.Round(NDExpr.Arr(np.array(new long[] { 1L << 60 })), 2));
            Assert.AreEqual(NPTypeCode.Int64, big.typecode);
            Assert.AreEqual(1L << 60, big.GetInt64(0));                    // identity, no float64 corruption

            var i32 = np.evaluate(NDExpr.Round(NDExpr.Arr(np.array(new[] { 12345, -6789 })), -2)).astype(NPTypeCode.Int64);
            CollectionAssert.AreEqual(new[] { 12300L, -6800L }, new[] { i32.GetInt64(0), i32.GetInt64(1) });

            // negative-decimals WRAP on cast-back (int8 127 → 130 → -126; uint8 255 → 260 → 4).
            Assert.AreEqual(-126L, np.evaluate(NDExpr.Round(NDExpr.Arr(np.array(new sbyte[] { 127 })), -1)).astype(NPTypeCode.Int64).GetInt64(0));
            Assert.AreEqual(4L, np.evaluate(NDExpr.Round(NDExpr.Arr(np.array(new byte[] { 255 })), -1)).astype(NPTypeCode.Int64).GetInt64(0));
        }

        /// <summary>Complex rounds each float64 lane separately (dtype preserved): <c>round([1.25+2.55j,
        /// -3.5-4.5j], 1)</c> = [1.2+2.6j, -3.5-4.5j]. BOOL with <c>decimals != 0</c> RAISES NumPy's
        /// <c>UFuncTypeError</c> (multiply for d&gt;0, divide for d&lt;0) — the composition's float64 output
        /// cannot cast back to bool; <c>decimals == 0</c> is the plain Round node (→ float16, unchanged).</summary>
        [TestMethod]
        public void P41bRound_Complex_PerLane_And_Bool_Raises()
        {
            var rc = np.evaluate(NDExpr.Round(NDExpr.Arr(np.array(new System.Numerics.Complex[] { new(1.25, 2.55), new(-3.5, -4.5) })), 1));
            Assert.AreEqual(NPTypeCode.Complex, rc.typecode);
            var re = np.real(rc).astype(NPTypeCode.Double); var im = np.imag(rc).astype(NPTypeCode.Double);
            Assert.AreEqual(1.2, re.GetDouble(0), 0); Assert.AreEqual(2.6, im.GetDouble(0), 0);
            Assert.AreEqual(-3.5, re.GetDouble(1), 0); Assert.AreEqual(-4.5, im.GetDouble(1), 0);

            // bool + decimals == 0 → the existing Round node (float16), NOT a RoundNode.
            Assert.AreEqual(NPTypeCode.Half, np.evaluate(NDExpr.Round(NDExpr.Arr(np.array(new[] { true, false })), 0)).typecode);
            // bool + decimals != 0 → throws, op-named (multiply for +, divide for -).
            var exPos = Assert.ThrowsException<NotSupportedException>(() => np.evaluate(NDExpr.Round(NDExpr.Arr(np.array(new[] { true })), 1)));
            StringAssert.Contains(exPos.Message, "Cannot cast ufunc 'multiply' output from dtype('float64') to dtype('bool')");
            var exNeg = Assert.ThrowsException<NotSupportedException>(() => np.evaluate(NDExpr.Round(NDExpr.Arr(np.array(new[] { true })), -1)));
            StringAssert.Contains(exNeg.Message, "Cannot cast ufunc 'divide' output from dtype('float64') to dtype('bool')");
        }

        /// <summary>The <c>decimals == 0</c> factory returns the plain <see cref="UnaryOp.Round"/> node (so
        /// <c>Round(x, 0)</c> ≡ <c>Round(x)</c>); Round COMPOSES in the fused kernel — <c>round(a+b, 2)</c>
        /// rounds a fused sum; and Decimal (no NumPy analog) rounds via the double bridge.</summary>
        [TestMethod]
        public void P41bRound_ZeroDelegates_Composes_Decimal()
        {
            // Round(x, 0) is the plain rint node: integer identity, float banker's rint, same as Round(x).
            var r0 = np.evaluate(NDExpr.Round(NDExpr.Arr(np.array(new double[] { 0.5, 1.5, 2.5 })), 0));
            CollectionAssert.AreEqual(new[] { 0.0, 2.0, 2.0 }, new[] { r0.GetDouble(0), r0.GetDouble(1), r0.GetDouble(2) });   // banker's
            Assert.AreEqual(NPTypeCode.Int32, np.evaluate(NDExpr.Round(NDExpr.Arr(np.array(new[] { 5 })), 0)).typecode);      // int identity

            // Composition: round(a+b, 2) rounds the fused sum.
            var a = np.array(new double[] { 1.111, 2.226, 3.335 }); var b = np.array(new double[] { 0.004, 0.004, 0.004 });
            var comp = np.evaluate(NDExpr.Round((NDExpr)a + (NDExpr)b, 2));
            CollectionAssert.AreEqual(new[] { 1.12, 2.23, 3.34 }, new[] { comp.GetDouble(0), comp.GetDouble(1), comp.GetDouble(2) });

            // Decimal via the double bridge (round(1.25m, 1) → 1.2m).
            var dec = np.evaluate(NDExpr.Round(NDExpr.Arr(np.array(new[] { 1.25m })), 1));
            Assert.AreEqual(NPTypeCode.Decimal, dec.typecode);
            Assert.AreEqual(1.2m, dec.GetAtIndex<decimal>(0));
        }

        // =====================================================================
        // Phase 4.3 — logical nodes (LogicalNode) + N-ary lowerings (Select / Clip)
        // + comparison operators. Values probed from NumPy 2.4.2. The logical
        // nodes are a NEW node (bool result via a per-operand nonzero test, all
        // dtypes incl. complex); Select and Clip are pure LOWERINGS to the
        // already-gated WhereNode / MinMaxNode. These pin the corpus-unreachable
        // edges: complex/NaN truthiness, first-true-wins, one-sided clip, and
        // every error path.
        // =====================================================================

        /// <summary>np.logical_and/or/xor: the result is ALWAYS bool, formed by nonzero-testing each
        /// operand at its own dtype — NaN is truthy (<c>nan != 0</c>), ±0 is falsy, and a complex is
        /// truthy iff either component is nonzero. NOT the bitwise <c>&amp;</c>/<c>|</c>/<c>^</c>.</summary>
        [TestMethod]
        public void P43_Logical_NonzeroTest_AllDtypes_BoolResult()
        {
            var f = np.array(new double[] { 0.0, 1.0, double.NaN, 2.0 });
            var g = np.array(new double[] { 0.0, 0.0, 1.0, 3.0 });
            var land = np.evaluate(NDExpr.LogicalAnd((NDExpr)f, (NDExpr)g));
            var lor = np.evaluate(NDExpr.LogicalOr((NDExpr)f, (NDExpr)g));
            var lxor = np.evaluate(NDExpr.LogicalXor((NDExpr)f, (NDExpr)g));
            Assert.AreEqual(NPTypeCode.Boolean, land.typecode);
            CollectionAssert.AreEqual(new[] { false, false, true, true }, new[] { land.GetBoolean(0), land.GetBoolean(1), land.GetBoolean(2), land.GetBoolean(3) });
            CollectionAssert.AreEqual(new[] { false, true, true, true }, new[] { lor.GetBoolean(0), lor.GetBoolean(1), lor.GetBoolean(2), lor.GetBoolean(3) });
            CollectionAssert.AreEqual(new[] { false, true, false, false }, new[] { lxor.GetBoolean(0), lxor.GetBoolean(1), lxor.GetBoolean(2), lxor.GetBoolean(3) });

            // complex: a complex is truthy iff a component is nonzero (0+2j is truthy, 0+0j falsy).
            var cz = np.array(new System.Numerics.Complex[] { new(0, 0), new(1, 0), new(0, 2) });
            var cw = np.array(new System.Numerics.Complex[] { new(1, 0), new(0, 0), new(3, 0) });
            var cland = np.evaluate(NDExpr.LogicalAnd((NDExpr)cz, (NDExpr)cw));
            Assert.AreEqual(NPTypeCode.Boolean, cland.typecode);
            CollectionAssert.AreEqual(new[] { false, false, true }, new[] { cland.GetBoolean(0), cland.GetBoolean(1), cland.GetBoolean(2) });

            // mixed dtype still yields bool; a comparison sub-tree composes (logical_and of two masks).
            var mix = np.evaluate(NDExpr.LogicalAnd(NDExpr.Greater((NDExpr)f, (NDExpr)NDExpr.Const(0.0)), NDExpr.Less((NDExpr)g, (NDExpr)NDExpr.Const(2.0))));
            CollectionAssert.AreEqual(new[] { false, true, false, false }, new[] { mix.GetBoolean(0), mix.GetBoolean(1), mix.GetBoolean(2), mix.GetBoolean(3) });
        }

        /// <summary>LogicalNode vectorizes (each operand becomes a truthiness lane mask combined by a
        /// vector AND/OR/XOR); the vector body must be bit-identical to the scalar body.</summary>
        [TestMethod]
        public void P43_Logical_VectorizesBitExact()
        {
            var poolA = new double[128];
            var poolB = new double[128];
            for (int i = 0; i < poolA.Length; i++)
            {
                poolA[i] = (i % 5 == 0) ? 0.0 : (i % 7) - 2.5;
                poolB[i] = (i % 3 == 0) ? 0.0 : (i % 4) + 0.5;
            }
            var a = np.array(poolA);
            var b = np.array(poolB);
            foreach (var factory in new Func<NDExpr, NDExpr, NDExpr>[] { NDExpr.LogicalAnd, NDExpr.LogicalOr, NDExpr.LogicalXor })
            {
                var vec = np.evaluate(factory(NDExpr.Arr(a), NDExpr.Arr(b)));
                NDArray sca;
                NDExpr.ForceScalar = true;
                try { sca = np.evaluate(factory(NDExpr.Arr(a), NDExpr.Arr(b))); }
                finally { NDExpr.ForceScalar = false; }
                for (int i = 0; i < poolA.Length; i++)
                    Assert.AreEqual(sca.GetBoolean(i), vec.GetBoolean(i), $"vector≠scalar at {i}");
            }
        }

        /// <summary>np.select: the FIRST true condition wins; positions where no condition is true take
        /// the default (weak-int 0 when omitted). Lowered to a reverse Where chain, so the result dtype
        /// is result_type(*choices, default). Empty / length-mismatch raise NumPy's verbatim text.</summary>
        [TestMethod]
        public void P43_Select_FirstTrueWins_Default_And_Errors()
        {
            var i1 = np.array(new[] { 10, 20, 30 });
            var i2 = np.array(new[] { 40, 50, 60 });
            var c0 = np.array(new[] { true, true, false });
            var c1 = np.array(new[] { true, false, true });
            // Overlap at index 0 (both true) → c0 wins (10); index 2 → only c1 → 60.
            var sel = np.evaluate(NDExpr.Select(
                new NDExpr[] { NDExpr.Arr(c0), NDExpr.Arr(c1) },
                new NDExpr[] { NDExpr.Arr(i1), NDExpr.Arr(i2) },
                NDExpr.Const(-1)));
            Assert.AreEqual(NPTypeCode.Int32, sel.typecode);   // result_type(int32, int32, weak -1) = int32
            CollectionAssert.AreEqual(new[] { 10, 20, 60 }, new[] { sel.GetInt32(0), sel.GetInt32(1), sel.GetInt32(2) });

            // Default omitted ⇒ weak 0 fill; float choices ⇒ float64 result.
            var d = np.array(new double[] { 1.5, 2.5, 3.5 });
            var selDef = np.evaluate(NDExpr.Select(
                new NDExpr[] { NDExpr.Arr(np.array(new[] { false, true, false })) },
                new NDExpr[] { NDExpr.Arr(d) }));
            Assert.AreEqual(NPTypeCode.Double, selDef.typecode);
            CollectionAssert.AreEqual(new[] { 0.0, 2.5, 0.0 }, new[] { selDef.GetDouble(0), selDef.GetDouble(1), selDef.GetDouble(2) });

            // Errors — factory-time, NumPy's verbatim order (length mismatch before emptiness).
            StringAssert.Contains(
                Assert.ThrowsException<ArgumentException>(() => NDExpr.Select(
                    new NDExpr[] { NDExpr.Arr(c0) }, new NDExpr[] { NDExpr.Arr(i1), NDExpr.Arr(i2) })).Message,
                "list of cases must be same length as list of conditions");
            StringAssert.Contains(
                Assert.ThrowsException<ArgumentException>(() => NDExpr.Select(
                    Array.Empty<NDExpr>(), Array.Empty<NDExpr>())).Message,
                "select with an empty condition list is not possible");
        }

        /// <summary>np.clip: one-sided bounds lower to the NaN-propagating np.maximum / np.minimum,
        /// two-sided to Min(Max(x, lo), hi) (the general clip ufunc). Result dtype is
        /// result_type(x, lo, hi). Both bounds null raises NumPy's "One of max or min must be given".</summary>
        [TestMethod]
        public void P43_Clip_OneSided_TwoSided_And_BothNullRaises()
        {
            var x = np.array(new double[] { -5, -1, 0, 3, 8 });
            var lo = np.array(new double[] { -2, -2, -2, -2, -2 });
            var hi = np.array(new double[] { 4, 4, 4, 4, 4 });

            var both = np.evaluate(NDExpr.Clip((NDExpr)x, (NDExpr)lo, (NDExpr)hi));
            CollectionAssert.AreEqual(new[] { -2.0, -1, 0, 3, 4 }, new[] { both.GetDouble(0), both.GetDouble(1), both.GetDouble(2), both.GetDouble(3), both.GetDouble(4) });

            var onlyLo = np.evaluate(NDExpr.Clip((NDExpr)x, (NDExpr)lo, null));   // ≡ np.maximum(x, lo)
            CollectionAssert.AreEqual(new[] { -2.0, -1, 0, 3, 8 }, new[] { onlyLo.GetDouble(0), onlyLo.GetDouble(1), onlyLo.GetDouble(2), onlyLo.GetDouble(3), onlyLo.GetDouble(4) });

            var onlyHi = np.evaluate(NDExpr.Clip((NDExpr)x, null, (NDExpr)hi));   // ≡ np.minimum(x, hi)
            CollectionAssert.AreEqual(new[] { -5.0, -1, 0, 3, 4 }, new[] { onlyHi.GetDouble(0), onlyHi.GetDouble(1), onlyHi.GetDouble(2), onlyHi.GetDouble(3), onlyHi.GetDouble(4) });

            // Integer clip preserves the integer dtype (result_type(int32, int32, int32)).
            var xi = np.array(new[] { -5, -1, 0, 3, 8 });
            var iclip = np.evaluate(NDExpr.Clip((NDExpr)xi, (NDExpr)np.array(new[] { 1, 1, 1, 1, 1 }), (NDExpr)np.array(new[] { 5, 5, 5, 5, 5 })));
            Assert.AreEqual(NPTypeCode.Int32, iclip.typecode);
            CollectionAssert.AreEqual(new[] { 1, 1, 1, 3, 5 }, new[] { iclip.GetInt32(0), iclip.GetInt32(1), iclip.GetInt32(2), iclip.GetInt32(3), iclip.GetInt32(4) });

            StringAssert.Contains(
                Assert.ThrowsException<ArgumentException>(() => NDExpr.Clip((NDExpr)x, null, null)).Message,
                "One of max or min must be given");
        }

        /// <summary>The comparison operators <c>&lt; &gt; &lt;= &gt;=</c> build a Boolean ComparisonNode and
        /// RETURN an NDExpr (composable), and unary <c>+</c> is np.positive (identity). <c>==</c>/<c>!=</c>
        /// are deliberately NOT overloaded, so <c>expr == null</c> stays a reference check.</summary>
        [TestMethod]
        public void P43_ComparisonOperators_And_UnaryPlus()
        {
            var f = np.array(new double[] { 1, 2, 3 });
            var g = np.array(new double[] { 2, 2, 2 });
            var lt = np.evaluate((NDExpr)f < (NDExpr)g);
            var gt = np.evaluate((NDExpr)f > (NDExpr)g);
            var le = np.evaluate((NDExpr)f <= (NDExpr)g);
            var ge = np.evaluate((NDExpr)f >= (NDExpr)g);
            CollectionAssert.AreEqual(new[] { true, false, false }, new[] { lt.GetBoolean(0), lt.GetBoolean(1), lt.GetBoolean(2) });
            CollectionAssert.AreEqual(new[] { false, false, true }, new[] { gt.GetBoolean(0), gt.GetBoolean(1), gt.GetBoolean(2) });
            CollectionAssert.AreEqual(new[] { true, true, false }, new[] { le.GetBoolean(0), le.GetBoolean(1), le.GetBoolean(2) });
            CollectionAssert.AreEqual(new[] { false, true, true }, new[] { ge.GetBoolean(0), ge.GetBoolean(1), ge.GetBoolean(2) });

            // Scalar RHS binds through the implicit double→NDExpr conversion.
            var ltScalar = np.evaluate((NDExpr)f < 2.0);
            CollectionAssert.AreEqual(new[] { true, false, false }, new[] { ltScalar.GetBoolean(0), ltScalar.GetBoolean(1), ltScalar.GetBoolean(2) });

            // Unary + is the dtype-preserving identity (np.positive).
            var i = np.array(new[] { -3, 0, 7 });
            var pos = np.evaluate(+(NDExpr)i);
            Assert.AreEqual(NPTypeCode.Int32, pos.typecode);
            CollectionAssert.AreEqual(new[] { -3, 0, 7 }, new[] { pos.GetInt32(0), pos.GetInt32(1), pos.GetInt32(2) });

            // == / != are NOT overloaded → reference-null checks (a comparison uses Equal/NotEqual).
            NDExpr e = (NDExpr)f;
            Assert.IsFalse(e == null);
            Assert.IsTrue(e != null);
        }
    }
}
