using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Generic;

namespace NumSharp.UnitTest.MathSuite
{
    /// <summary>
    /// out=/where= plan slice 3 (docs/OUT_WHERE_NPYITER_FAMILIES_PLAN.md §4.1):
    /// the six comparisons + isnan/isfinite/isinf gain ufunc out=/where=
    /// through the new <c>ExecuteComparisonUfuncInto</c> (comparisons) and the
    /// shared unary Into-path (predicates).
    ///
    /// Return-type design (option i): the no-out forms keep their
    /// <c>NDArray&lt;bool&gt;</c> sugar; the out-taking overloads return plain
    /// <c>NDArray</c> — NumPy's np.less(a, b, out=f64) returns the f64 out
    /// itself. Every expectation pinned to a NumPy 2.4.2 probe (plan §2.1-2.5,
    /// §2.14; texts verbatim incl. trailing spaces).
    /// </summary>
    [TestClass]
    public class UfuncComparisonOutWhereTests
    {
        private static NDArray Af() => np.arange(4).astype(np.float64);          // [0,1,2,3]
        private static NDArray Bf() => np.arange(4).astype(np.float64) + 0.5;    // a < b everywhere
        private static NDArray Mask4() => np.array(new[] { true, false, true, false });

        // =====================================================================
        // out= dtype matrix (probe A1) — bool casts same_kind to EVERYTHING
        // =====================================================================

        [TestMethod]
        public void Less_Out_Bool_IdentityAndValues()
        {
            var o = np.empty(new Shape(4), np.@bool);
            var r = np.less(Af(), Bf(), o);

            Assert.IsTrue(ReferenceEquals(r, o), "out= must return the provided instance");
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(true, o.GetBoolean(i));
        }

        [TestMethod]
        public void Less_Out_AllNumericDtypes_TrueIsOne()
        {
            // NumPy: less(f64,f64,out=X) succeeds for every numeric X; True→1.
            var oI8 = np.empty(new Shape(4), np.@byte); // sbyte
            np.less(Af(), Bf(), oI8);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual((sbyte)1, oI8.GetSByte(i));

            var oU8 = np.empty(new Shape(4), np.uint8);
            np.less(Af(), Bf(), oU8);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual((byte)1, oU8.GetByte(i));

            var oI32 = np.empty(new Shape(4), np.int32);
            np.less(Af(), Bf(), oI32);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(1, oI32.GetInt32(i));

            var oU32 = np.empty(new Shape(4), np.uint32);
            np.less(Af(), Bf(), oU32);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(1u, oU32.GetUInt32(i));

            var oF32 = np.empty(new Shape(4), np.float32);
            np.less(Af(), Bf(), oF32);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(1f, oF32.GetSingle(i));

            var oF64 = np.empty(new Shape(4), np.float64);
            np.less(Af(), Bf(), oF64);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(1.0, oF64.GetDouble(i));
        }

        [TestMethod]
        public void Greater_Out_Float64_FalseToZero_TrueToOne()
        {
            // NumPy A1 last row: False→0.0, True→1.0 into a prior-filled f64 out.
            var x = np.array(new[] { 0.0, 2, 1, 3 });
            var y = np.array(new[] { 1.0, 1, 2, 2 });
            var o = np.full(new Shape(4), -7.0, np.float64);
            var r = np.greater(x, y, o);

            Assert.IsTrue(ReferenceEquals(r, o));
            var expected = new[] { 0.0, 1, 0, 1 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetDouble(i));
        }

        [TestMethod]
        public void Family_OutSmoke_AllSixOps()
        {
            // equal/not_equal/less/less_equal/greater/greater_equal with i4 out.
            var x = np.array(new[] { 0.0, 1, 2, 3 });
            var y = np.array(new[] { 0.0, 2, 2, 1 });
            var o = np.empty(new Shape(4), np.int32);

            np.equal(x, y, o);
            CollectionAssertInts(o, 1, 0, 1, 0);

            np.not_equal(x, y, o);
            CollectionAssertInts(o, 0, 1, 0, 1);

            np.less(x, y, o);
            CollectionAssertInts(o, 0, 1, 0, 0);

            np.less_equal(x, y, o);
            CollectionAssertInts(o, 1, 1, 1, 0);

            np.greater(x, y, o);
            CollectionAssertInts(o, 0, 0, 0, 1);

            np.greater_equal(x, y, o);
            CollectionAssertInts(o, 1, 0, 1, 1);
        }

        private static void CollectionAssertInts(NDArray o, params int[] expected)
        {
            for (int i = 0; i < expected.Length; i++)
                Assert.AreEqual(expected[i], o.GetInt32(i), $"index {i}");
        }

        // =====================================================================
        // comparisons compare at result_type(lhs, rhs) — precision pin (A3)
        // =====================================================================

        [TestMethod]
        public void MixedDtypes_CompareAtCommonDtype_2Pow53Pin()
        {
            // NumPy: greater(int64 [2^53+1], float64 [2^53]) -> [False];
            // equal -> [True] — both operands cast to f64, 2^53+1 rounds down.
            var i = np.array(new[] { 9007199254740993L });
            var f = np.array(new[] { 9007199254740992.0 });

            var oG = np.empty(new Shape(1), np.@bool);
            np.greater(i, f, oG);
            Assert.AreEqual(false, oG.GetBoolean(0));

            var oE = np.empty(new Shape(1), np.@bool);
            np.equal(i, f, oE);
            Assert.AreEqual(true, oE.GetBoolean(0));
        }

        [TestMethod]
        public void MixedDtypes_Int32VsFloat64_WithSByteOut()
        {
            // NumPy D2: less(i4, f8, out=i1) — loop f64, bool→i1 flush.
            var o = np.empty(new Shape(4), np.@byte);
            np.less(np.arange(4).astype(np.int32), Bf(), o);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual((sbyte)1, o.GetSByte(i));
        }

        // =====================================================================
        // shape rules (A7) — binary error form, trailing space
        // =====================================================================

        [TestMethod]
        public void Out_WrongShape_ThrowsBroadcastText()
        {
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.less(Af(), Bf(), np.empty(new Shape(5), np.@bool)));

            Assert.AreEqual(
                "operands could not be broadcast together with shapes (4,) (4,) (5,) ",
                ex.Message);
        }

        [TestMethod]
        public void Out_WouldStretch_ThrowsNonBroadcastableText()
        {
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.less(Af(), Bf(), np.empty(new Shape(1), np.@bool)));

            Assert.AreEqual(
                "non-broadcastable output operand with shape (1,) doesn't match the " +
                "broadcast shape (4,)", ex.Message);
        }

        [TestMethod]
        public void Out_Larger_InputsBroadcastUp_RowsRepeat()
        {
            // NumPy A7 row 3: less((4,),(4,),out=(2,4)) — rows repeat.
            var o = np.empty(new Shape(2, 4), np.@bool);
            np.less(Af(), Bf(), o);

            var flat = np.ravel(o);
            for (int i = 0; i < 8; i++)
                Assert.AreEqual(true, flat.GetBoolean(i));
        }

        // =====================================================================
        // aliasing + layout (A8/A10)
        // =====================================================================

        [TestMethod]
        public void Out_FullAliasInput_BoolResultIntoF64Input()
        {
            // NumPy A8: less(aa, 4.0, out=aa) -> [1,1,1,1,0,0,0,0] (aa=arange(8) f64).
            var aa = np.arange(8).astype(np.float64);
            var r = np.less(aa, (NDArray)4.0, aa);

            Assert.IsTrue(ReferenceEquals(r, aa));
            var expected = new[] { 1.0, 1, 1, 1, 0, 0, 0, 0 };
            for (int i = 0; i < 8; i++)
                Assert.AreEqual(expected[i], aa.GetDouble(i));
        }

        [TestMethod]
        public void Out_PartialOverlap_CopyIfOverlap()
        {
            // NumPy A8 row 2: less(aa2[:-1], aa2[1:], out=aa2[:-1])
            // -> [1,1,1,1,1,1,1,7] for aa2=arange(8) f64.
            var aa2 = np.arange(8).astype(np.float64);
            np.less(aa2["0:7"], aa2["1:8"], aa2["0:7"]);

            var expected = new[] { 1.0, 1, 1, 1, 1, 1, 1, 7 };
            for (int i = 0; i < 8; i++)
                Assert.AreEqual(expected[i], aa2.GetDouble(i));
        }

        [TestMethod]
        public void Out_StridedAndOffsetViews_WriteThrough()
        {
            // NumPy A10: out=big[::2] -> [T,F,T,F,...] pattern in the big buffer.
            var big = np.zeros(new Shape(8), np.float64);
            np.less(Af(), Bf(), big["::2"]);
            var expected = new[] { 1.0, 0, 1, 0, 1, 0, 1, 0 };
            for (int i = 0; i < 8; i++)
                Assert.AreEqual(expected[i], big.GetDouble(i));

            // offset slice + cast: out=buf[3:7] f64 -> [0,0,0,1,1,1,1,0,0,0].
            var buf = np.zeros(new Shape(10), np.float64);
            np.less(Af(), Bf(), buf["3:7"]);
            var expected2 = new[] { 0.0, 0, 0, 1, 1, 1, 1, 0, 0, 0 };
            for (int i = 0; i < 10; i++)
                Assert.AreEqual(expected2[i], buf.GetDouble(i));
        }

        // =====================================================================
        // 0-d / empty (A9/D7)
        // =====================================================================

        [TestMethod]
        public void ZeroD_Scalars_BoolAndF64Outs()
        {
            var oB = np.empty(new Shape(), np.@bool);
            var rB = np.less(NDArray.Scalar(1.0), NDArray.Scalar(2.0), oB);
            Assert.IsTrue(ReferenceEquals(rB, oB));
            Assert.AreEqual(true, oB.GetBoolean());

            var oF = np.empty(new Shape(), np.float64);
            var rF = np.less(NDArray.Scalar(1.0), NDArray.Scalar(2.0), oF);
            Assert.IsTrue(ReferenceEquals(rF, oF));
            Assert.AreEqual(1.0, oF.GetDouble());

            // 0-d everything incl. a 0-d False mask -> untouched prior (D7).
            var oM = np.full(new Shape(), -1.0, np.float64);
            np.less(NDArray.Scalar(1.0), NDArray.Scalar(2.0), oM, NDArray.Scalar(false));
            Assert.AreEqual(-1.0, oM.GetDouble());
        }

        [TestMethod]
        public void Empty_WithOut_ReturnsOut()
        {
            var o = np.empty(new Shape(0), np.@bool);
            var r = np.less(np.empty(new Shape(0), np.float64), np.empty(new Shape(0), np.float64), o);
            Assert.IsTrue(ReferenceEquals(r, o));
        }

        // =====================================================================
        // where= (A4-A6, D5, D12)
        // =====================================================================

        [TestMethod]
        public void Where_WithOut_MaskedOffKeepPrior()
        {
            // NumPy A4: less(a,b,out=f64 prior=-5, where=[T,F,T,F]) -> [1,-5,1,-5].
            var o = np.full(new Shape(4), -5.0, np.float64);
            np.less(Af(), Bf(), o, Mask4());

            var expected = new[] { 1.0, -5, 1, -5 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetDouble(i));

            // greater into int out prior=9 — cast + mask compose (A4 row 2).
            var oi = np.full(new Shape(4), 9, np.int32);
            np.greater(Af(), Bf(), oi, Mask4());
            var expectedI = new[] { 0, 9, 0, 9 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expectedI[i], oi.GetInt32(i));
        }

        [TestMethod]
        public void Where_WithoutOut_MaskedOnSlotsOnly_BoolDtype()
        {
            // NumPy A5: masked-off slots are uninitialized garbage — assert
            // masked-on only; dtype is bool; shape follows the inputs.
            var r = np.less(Af(), Bf(), null, Mask4());

            Assert.AreEqual(NPTypeCode.Boolean, r.typecode);
            Assert.AreEqual(true, r.GetBoolean(0));
            Assert.AreEqual(true, r.GetBoolean(2));
        }

        [TestMethod]
        public void Where_JoinsOutputShape()
        {
            // NumPy A5: less((4,),(4,),where=(2,4)-mask) -> shape (2,4).
            var mask = np.array(new[,] { { true, false, true, false }, { false, true, false, true } });
            var r = np.less(Af(), Bf(), null, mask);

            Assert.AreEqual(2, (int)r.shape[0]);
            Assert.AreEqual(4, (int)r.shape[1]);
            Assert.AreEqual(true, r.GetBoolean(0, 0));
            Assert.AreEqual(true, r.GetBoolean(1, 1));
        }

        [TestMethod]
        public void Where_RowMask_BroadcastsOverRows()
        {
            // NumPy A6: less((2,4),(4,),out=(2,4) prior-True, where=row(4,)).
            var lhs = np.zeros(new Shape(2, 4), np.float64);
            var o = np.full(new Shape(2, 4), 9.0, np.float64);
            np.less(lhs, Bf(), o, Mask4());

            var flat = np.ravel(o);
            var expected = new[] { 1.0, 9, 1, 9, 1, 9, 1, 9 };
            for (int i = 0; i < 8; i++)
                Assert.AreEqual(expected[i], flat.GetDouble(i));
        }

        [TestMethod]
        public void Where_OutCast_Composed_SmallAndMultiWindow()
        {
            // D12: bool loop → f8 out + mask, small.
            var o = np.full(new Shape(4), -5.0, np.float64);
            np.less(Af(), Bf(), o, Mask4());
            Assert.AreEqual(1.0, o.GetDouble(0));
            Assert.AreEqual(-5.0, o.GetDouble(1));

            // multi-window (20 005 elements, bool→f4 flush, masked).
            const int n = 20_005;
            var a = np.arange(n).astype(np.float64);
            var b = np.full(new Shape(n), 10_000.0, np.float64);
            var mask = np.ones(new Shape(n), np.@bool);
            var oBig = np.full(new Shape(n), -1.0f, np.float32);
            np.less(a, b, oBig, mask);

            Assert.AreEqual(1f, oBig.GetSingle(0));
            Assert.AreEqual(1f, oBig.GetSingle(9_999));
            Assert.AreEqual(0f, oBig.GetSingle(10_000));
            Assert.AreEqual(0f, oBig.GetSingle(20_004));
        }

        [TestMethod]
        public void Where_NonBool_ThrowsSafeText_AndBeatsBadOutShape()
        {
            var intMask = np.array(new long[] { 1, 0, 1, 0 });

            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.less(Af(), Bf(), null, intMask));
            Assert.AreEqual(
                "Cannot cast array data from dtype('int64') to dtype('bool') " +
                "according to the rule 'safe'", ex.Message);

            // order: bad where beats bad out shape (A12).
            var ex2 = Assert.ThrowsException<ArgumentException>(() =>
                np.less(Af(), Bf(), np.empty(new Shape(5), np.@bool), intMask));
            StringAssert.Contains(ex2.Message, "according to the rule 'safe'");
        }

        [TestMethod]
        public void WhereShape_ErrorsListEveryOperand()
        {
            // A12: less(a,b,out=(4,),where=(3,)) lists inputs + out + where.
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.less(Af(), Bf(), np.empty(new Shape(4), np.@bool),
                    np.array(new[] { true, false, true })));
            Assert.AreEqual(
                "operands could not be broadcast together with shapes (4,) (4,) (4,) (3,) ",
                ex.Message);

            // where would stretch the provided out.
            var mask24 = np.array(new[,] { { true, false, true, false }, { false, true, false, true } });
            var ex2 = Assert.ThrowsException<ArgumentException>(() =>
                np.less(Af(), Bf(), np.empty(new Shape(4), np.@bool), mask24));
            Assert.AreEqual(
                "non-broadcastable output operand with shape (4,) doesn't match the " +
                "broadcast shape (2,4)", ex2.Message);
        }

        // =====================================================================
        // multi-window bool→f4 flush without mask (S3)
        // =====================================================================

        [TestMethod]
        public void Out_CastMultiWindow_20005Elements()
        {
            const int n = 20_005;
            var a = np.arange(n).astype(np.float64);
            var b = np.full(new Shape(n), 10_000.0, np.float64);
            var o = np.empty(new Shape(n), np.float32);

            var r = np.less(a, b, o);

            Assert.IsTrue(ReferenceEquals(r, o));
            Assert.AreEqual(1f, o.GetSingle(0));
            Assert.AreEqual(1f, o.GetSingle(8_192));
            Assert.AreEqual(1f, o.GetSingle(9_999));
            Assert.AreEqual(0f, o.GetSingle(10_000));
            Assert.AreEqual(0f, o.GetSingle(16_384));
            Assert.AreEqual(0f, o.GetSingle(20_004));
        }

        // =====================================================================
        // return-type contract (S22): ONE NumPy-shaped overload per comparison.
        // The merged overload returns plain NDArray (with out= the provided
        // array is returned whatever its dtype), but a plain call still
        // produces an NDArray<bool> INSTANCE (TensorEngine contract), so the
        // typed wrapper is one zero-alloc cast away — and the C# comparison
        // operators keep the NDArray<bool> static type.
        // =====================================================================

        [TestMethod]
        public void ReturnTypes_PlainCallIsBoolInstance_OutFormReturnsProvidedOut()
        {
            NDArray plain = np.less(Af(), Bf());
            Assert.AreEqual(NPTypeCode.Boolean, plain.typecode);
            Assert.IsInstanceOfType(plain, typeof(NDArray<bool>), "plain calls must return an NDArray<bool> instance");
            NDArray<bool> typed = (NDArray<bool>)plain;     // zero-alloc cast
            Assert.IsTrue(typed.GetBoolean(0));

            NDArray<bool> viaOperator = Af() < Bf();        // operator keeps the typed static type
            Assert.IsTrue(viaOperator.GetBoolean(0));

            var o = np.empty(new Shape(4), np.@bool);
            NDArray viaOut = np.less(Af(), Bf(), o);
            Assert.IsTrue(ReferenceEquals(viaOut, o));
        }

        // =====================================================================
        // dtype= is validate-only on comparisons (bool loops only, NumPy 2.4.2)
        // =====================================================================

        [TestMethod]
        public void Dtype_BoolIsNoOp_NonBoolRaisesNoLoopWithUfuncName()
        {
            // NumPy: np.equal(a, b, dtype=bool) → normal bool result.
            var ok = np.equal(Af(), Af(), dtype: NPTypeCode.Boolean);
            Assert.AreEqual(NPTypeCode.Boolean, ok.typecode);
            Assert.IsTrue(ok.GetBoolean(0));

            // NumPy: any other dtype raises, naming THIS ufunc:
            // "No loop matching the specified signature and casting was found for ufunc equal"
            void AssertNoLoop(Action act, string ufunc)
            {
                var ex = Assert.ThrowsException<IncorrectTypeException>(act);
                Assert.AreEqual(
                    $"No loop matching the specified signature and casting was found for ufunc {ufunc}",
                    ex.Message);
            }

            AssertNoLoop(() => np.equal(Af(), Af(), dtype: NPTypeCode.Double), "equal");
            AssertNoLoop(() => np.not_equal(Af(), Af(), dtype: NPTypeCode.Int32), "not_equal");
            AssertNoLoop(() => np.less(Af(), Bf(), dtype: NPTypeCode.Double), "less");
            AssertNoLoop(() => np.less_equal(Af(), Bf(), dtype: NPTypeCode.Double), "less_equal");
            AssertNoLoop(() => np.greater(Af(), Bf(), dtype: NPTypeCode.Double), "greater");
            AssertNoLoop(() => np.greater_equal(Af(), Bf(), dtype: NPTypeCode.Double), "greater_equal");

            // Predicates follow the same rule (probed: isnan(x, dtype=f64) raises).
            var x = np.array(new double[] { 1.0, double.NaN });
            var nan = np.isnan(x, dtype: NPTypeCode.Boolean);
            Assert.IsTrue(nan.GetBoolean(1));
            AssertNoLoop(() => np.isnan(x, dtype: NPTypeCode.Double), "isnan");
            AssertNoLoop(() => np.isinf(x, dtype: NPTypeCode.Double), "isinf");
            AssertNoLoop(() => np.isfinite(x, dtype: NPTypeCode.Double), "isfinite");
        }

        [TestMethod]
        public void Predicates_SingleOverload_NumPyCallForms()
        {
            // The np.isinf family exposes ONE overload: (a, out=, where=, dtype=).
            var x = np.array(new[] { 1.0, double.PositiveInfinity, double.NaN });

            NDArray plain = np.isinf(x);
            Assert.IsInstanceOfType(plain, typeof(NDArray<bool>));
            Assert.IsTrue(plain.GetBoolean(1));
            Assert.IsFalse(plain.GetBoolean(2));

            var o = np.empty(new Shape(3), np.float64);          // non-bool out: bool loop casts up
            Assert.IsTrue(ReferenceEquals(np.isinf(x, o), o));
            Assert.AreEqual(1.0, o.GetDouble(1));

            var masked = np.isinf(x, o, np.array(new[] { false, false, true }));
            Assert.IsTrue(ReferenceEquals(masked, o));
            Assert.AreEqual(1.0, o.GetDouble(1), "mask-false slot keeps prior contents");
            Assert.AreEqual(0.0, o.GetDouble(2));
        }

        // =====================================================================
        // dtype edges: Half + bool inputs (D3/D4)
        // =====================================================================

        [TestMethod]
        public void HalfInputs_Compare_WithIntOut()
        {
            var a = np.arange(4).astype(np.float16);
            var b = (np.arange(4).astype(np.float64) + 0.5).astype(np.float16);

            var o = np.empty(new Shape(4), np.int32);
            np.less(a, b, o);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(1, o.GetInt32(i));
        }

        [TestMethod]
        public void BoolInputs_Equal_WithIntOut()
        {
            // NumPy D4: equal(bool, bool, out=i4).
            var x = np.array(new[] { true, true, false, false });
            var y = np.array(new[] { true, false, true, false });

            var o = np.empty(new Shape(4), np.int32);
            np.equal(x, y, o);
            CollectionAssertInts(o, 1, 0, 0, 1);
        }

        // =====================================================================
        // predicates: isnan / isfinite / isinf (A11, D8, E1, E2)
        // =====================================================================

        private static NDArray V() => np.array(new[] { 1.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity });

        [TestMethod]
        public void IsNan_Out_BoolU8I32F64()
        {
            var oB = np.empty(new Shape(4), np.@bool);
            var r = np.isnan(V(), oB);
            Assert.IsTrue(ReferenceEquals(r, oB));
            Assert.AreEqual(false, oB.GetBoolean(0));
            Assert.AreEqual(true, oB.GetBoolean(1));
            Assert.AreEqual(false, oB.GetBoolean(2));
            Assert.AreEqual(false, oB.GetBoolean(3));

            var oU = np.empty(new Shape(4), np.uint8);
            np.isnan(V(), oU);
            Assert.AreEqual((byte)1, oU.GetByte(1));
            Assert.AreEqual((byte)0, oU.GetByte(2));

            var oI = np.empty(new Shape(4), np.int32);
            np.isnan(V(), oI);
            CollectionAssertInts(oI, 0, 1, 0, 0);

            var oF = np.empty(new Shape(4), np.float64);
            np.isnan(V(), oF);
            Assert.AreEqual(1.0, oF.GetDouble(1));
            Assert.AreEqual(0.0, oF.GetDouble(3));
        }

        [TestMethod]
        public void IsFinite_IsInf_Out_Values()
        {
            var oFin = np.empty(new Shape(4), np.@bool);
            np.isfinite(V(), oFin);
            Assert.AreEqual(true, oFin.GetBoolean(0));
            Assert.AreEqual(false, oFin.GetBoolean(1));
            Assert.AreEqual(false, oFin.GetBoolean(2));
            Assert.AreEqual(false, oFin.GetBoolean(3));

            var oInf = np.empty(new Shape(4), np.@bool);
            np.isinf(V(), oInf);
            Assert.AreEqual(false, oInf.GetBoolean(0));
            Assert.AreEqual(false, oInf.GetBoolean(1));
            Assert.AreEqual(true, oInf.GetBoolean(2));
            Assert.AreEqual(true, oInf.GetBoolean(3));
        }

        [TestMethod]
        public void IsNan_IntInputs_AllFalse_WithOut()
        {
            // NumPy A11: isnan(i32, out=bool/f64) -> all 0 (valid call).
            var oB = np.empty(new Shape(4), np.@bool);
            np.isnan(np.arange(4).astype(np.int32), oB);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(false, oB.GetBoolean(i));

            var oF = np.empty(new Shape(4), np.float64);
            np.isnan(np.arange(4).astype(np.int32), oF);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(0.0, oF.GetDouble(i));
        }

        [TestMethod]
        public void IsNan_WhereWithPrior_AndErrors()
        {
            // NumPy A11: isnan(v, out=f64 prior=-1, where=[T,F,T,F]) -> [0,-1,0,-1].
            var o = np.full(new Shape(4), -1.0, np.float64);
            np.isnan(V(), o, Mask4());
            var expected = new[] { 0.0, -1, 0, -1 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetDouble(i));

            // unary shape text: (4,) (5,)  with trailing space (D8).
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.isnan(V(), np.empty(new Shape(5), np.@bool)));
            Assert.AreEqual(
                "operands could not be broadcast together with shapes (4,) (5,) ",
                ex.Message);

            // where must be bool — 'safe' text (A12 analog).
            var ex2 = Assert.ThrowsException<ArgumentException>(() =>
                np.isnan(V(), null, np.array(new long[] { 1, 0, 1, 0 })));
            Assert.AreEqual(
                "Cannot cast array data from dtype('int64') to dtype('bool') " +
                "according to the rule 'safe'", ex2.Message);
        }

        [TestMethod]
        public void IsNan_StridedOut_HalfAndComplexInputs()
        {
            // strided out (D8): writes through, [0,1,0,0] interleaved.
            var big = np.zeros(new Shape(8), np.int32);
            np.isnan(V(), big["::2"]);
            var expected = new[] { 0, 0, 1, 0, 0, 0, 0, 0 };
            for (int i = 0; i < 8; i++)
                Assert.AreEqual(expected[i], big.GetInt32(i), $"index {i}");

            // float16 input (E2): isnan(f2) with i32 out.
            var h = np.array(new[] { 1.0, double.NaN, 2.0, 3.0 }).astype(np.float16);
            var oH = np.empty(new Shape(4), np.int32);
            np.isnan(h, oH);
            CollectionAssertInts(oH, 0, 1, 0, 0);

            // complex input (E1): isnan(c128) -> NaN in either component.
            var c = np.array(new[]
            {
                new System.Numerics.Complex(1, 1),
                new System.Numerics.Complex(2, 0),
                new System.Numerics.Complex(double.NaN, 1),
                new System.Numerics.Complex(3, -2),
            });
            var oC = np.empty(new Shape(4), np.uint8);
            np.isnan(c, oC);
            Assert.AreEqual((byte)0, oC.GetByte(0));
            Assert.AreEqual((byte)0, oC.GetByte(1));
            Assert.AreEqual((byte)1, oC.GetByte(2));
            Assert.AreEqual((byte)0, oC.GetByte(3));
        }
    }
}
