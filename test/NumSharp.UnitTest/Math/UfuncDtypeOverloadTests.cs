using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.MathSuite
{
    /// <summary>
    /// Wave 2.1 follow-up: ONE NumPy-shaped overload per elementwise ufunc.
    ///
    /// Every ufunc from the out=/where= wave now exposes a single overload
    /// mirroring NumPy's signature — <c>f(x[, x2], out=None, *, where=True,
    /// dtype=None)</c> — with <c>out</c> in NumPy's positional slot and
    /// <c>where</c>/<c>dtype</c> reachable by name without <c>out</c>:
    ///   unary:  sqrt, exp, log, sin, cos, tan, abs, absolute, negative, square
    ///   binary: power, floor_divide (+ add/subtract/multiply/divide/
    ///           true_divide/mod, which already had the merged shape; their
    ///           dtype= needs engine plumbing and is tracked separately).
    ///
    /// All expectations pinned to NumPy 2.4.2 (probed via python_run):
    ///  • dtype= selects the LOOP — computation runs at that precision even
    ///    when out= is given: np.sqrt([2.],out=f64,dtype=f32) stores
    ///    1.41421353816986083984 (the float32 value) into the float64 out.
    ///  • power(2,-1,dtype=f64)=0.5 — the negative-integer-exponent ValueError
    ///    applies only when the resolved loop is integer**integer.
    ///  • power(10,11,dtype=f64)=1e11 exactly — the loop computes in f64,
    ///    NumPy never computes the promoted loop then casts.
    ///  • inputs must reach the dtype= loop via same_kind casts:
    ///    "Cannot cast ufunc 'floor_divide' input 0 from dtype('float64') to
    ///    dtype('int32') with casting rule 'same_kind'" (unary texts name no
    ///    input index).
    ///  • float-only ufuncs raise "No loop matching the specified signature
    ///    and casting was found for ufunc sqrt" for integer/bool dtype=.
    ///  • negative(bool,dtype=f64) = [-1., -0.] (legal — dtype selects the
    ///    loop); negative(f64,dtype=bool) raises the boolean-negative
    ///    TypeError just like plain negative(bool).
    /// </summary>
    [TestClass]
    public class UfuncDtypeOverloadTests
    {
        // =====================================================================
        // NumPy-shaped call forms — one overload covers them all
        // =====================================================================

        [TestMethod]
        public void Sqrt_NumPyCallForms_OneOverload()
        {
            var x = np.array(new double[] { 4.0, 9.0, 16.0 });

            // np.sqrt(x)
            var r = np.sqrt(x);
            Assert.AreEqual(2.0, r.GetDouble(0));

            // np.sqrt(x, out) — positional out, instance returned (NumPy: 2nd positional IS out)
            var o = np.zeros(new Shape(3), np.float64);
            Assert.IsTrue(ReferenceEquals(np.sqrt(x, o), o));
            Assert.AreEqual(3.0, o.GetDouble(1));

            // np.sqrt(x, where=mask) — keyword where without out
            var masked = np.sqrt(x, where: np.array(new[] { true, false, true }));
            Assert.AreEqual(2.0, masked.GetDouble(0));
            Assert.AreEqual(4.0, masked.GetDouble(2));

            // np.sqrt(x, dtype=np.float32) — keyword dtype
            Assert.AreEqual(NPTypeCode.Single, np.sqrt(x, dtype: NPTypeCode.Single).typecode);

            // np.sqrt(x, out, where=m, dtype=f32) — everything at once
            var o2 = np.ones(new Shape(3), np.float64) * 99.0;
            var m = np.array(new[] { true, false, true });
            var r2 = np.sqrt(x, o2, m, NPTypeCode.Single);
            Assert.IsTrue(ReferenceEquals(r2, o2));
            Assert.AreEqual(2.0, o2.GetDouble(0));
            Assert.AreEqual(99.0, o2.GetDouble(1)); // mask-false keeps prior contents
        }

        [TestMethod]
        public void UnaryFamily_WhereAndDtype_ReachableWithoutOut()
        {
            // NumPy: where= and dtype= are keyword-only and legal without out.
            var x = np.array(new double[] { 1.0 });
            var t = np.array(new[] { true });

            Assert.AreEqual(Math.E, np.exp(x, where: t).GetDouble(0), 1e-12);
            Assert.AreEqual(NPTypeCode.Single, np.log(x, dtype: NPTypeCode.Single).typecode);
            Assert.AreEqual(NPTypeCode.Single, np.sin(x, dtype: NPTypeCode.Single).typecode);
            Assert.AreEqual(NPTypeCode.Single, np.cos(x, dtype: NPTypeCode.Single).typecode);
            Assert.AreEqual(NPTypeCode.Single, np.tan(x, dtype: NPTypeCode.Single).typecode);
            Assert.AreEqual(NPTypeCode.Single, np.square(x, dtype: NPTypeCode.Single).typecode);
            Assert.AreEqual(NPTypeCode.Single, np.abs(x, dtype: NPTypeCode.Single).typecode);
            Assert.AreEqual(NPTypeCode.Single, np.absolute(x, dtype: NPTypeCode.Single).typecode);
            Assert.AreEqual(NPTypeCode.Single, np.negative(x, dtype: NPTypeCode.Single).typecode);
        }

        [TestMethod]
        public void BinaryFamily_WhereAndDtype_ReachableWithoutOut()
        {
            var a = np.array(new int[] { 10 });
            var b = np.array(new int[] { 3 });

            // NumPy: power(10,3,dtype=f32) → 1000.0 float32
            var p = np.power(a, b, dtype: NPTypeCode.Single);
            Assert.AreEqual(NPTypeCode.Single, p.typecode);
            Assert.AreEqual(1000.0f, p.GetSingle(0));

            // NumPy: floor_divide(-7,2,dtype=f64) → -4.0 float64 (float loop)
            var f = np.floor_divide(np.array(new int[] { -7 }), np.array(new int[] { 2 }), dtype: NPTypeCode.Double);
            Assert.AreEqual(NPTypeCode.Double, f.typecode);
            Assert.AreEqual(-4.0, f.GetDouble(0));

            // where= without out compiles and masks
            var w = np.power(a, b, where: np.array(new[] { true }));
            Assert.AreEqual(1000, w.GetInt32(0));
        }

        // =====================================================================
        // dtype= selects the LOOP — composition with out= (probed 2.4.2)
        // =====================================================================

        [TestMethod]
        public void Sqrt_DtypeWithOut_LoopComputesInDtype()
        {
            // NumPy: np.sqrt(np.array([2.]), out_f64, dtype=np.float32)
            //   → 1.41421353816986083984 — the float32-rounded value lands in
            //   the float64 out (dtype is honored even when out is given).
            var x = np.array(new double[] { 2.0 });
            var o = np.zeros(new Shape(1), np.float64);
            var r = np.sqrt(x, o, dtype: NPTypeCode.Single);

            Assert.IsTrue(ReferenceEquals(r, o));
            Assert.AreEqual((double)(float)Math.Sqrt(2.0), o.GetDouble(0));
            Assert.AreNotEqual(Math.Sqrt(2.0), o.GetDouble(0), "must be the f32 loop value, not the f64 loop value");
        }

        [TestMethod]
        public void Sqrt_DtypeOutWhere_AllCompose()
        {
            // NumPy: np.sqrt([2.,3.], out=[99.,99.], where=[True,False], dtype=f32)
            //   → [1.41421354, 99.]
            var x = np.array(new double[] { 2.0, 3.0 });
            var o = np.ones(new Shape(2), np.float64) * 99.0;
            var r = np.sqrt(x, o, np.array(new[] { true, false }), NPTypeCode.Single);

            Assert.IsTrue(ReferenceEquals(r, o));
            Assert.AreEqual((double)(float)Math.Sqrt(2.0), o.GetDouble(0));
            Assert.AreEqual(99.0, o.GetDouble(1));
        }

        [TestMethod]
        public void Power_DtypeWithOut_LoopComputesInDtype()
        {
            // NumPy: np.power([10],[8], out_f64, dtype=np.float32) → 100000000.0
            //   (the float32 loop value cast up into the float64 out).
            var o = np.zeros(new Shape(1), np.float64);
            var r = np.power(np.array(new int[] { 10 }), np.array(new int[] { 8 }), o, dtype: NPTypeCode.Single);

            Assert.IsTrue(ReferenceEquals(r, o));
            Assert.AreEqual((double)(float)1e8, o.GetDouble(0));
        }

        [TestMethod]
        public void Power_OutCastValidatedAgainstDtypeLoop()
        {
            // NumPy: power(i32,i32,out=i32,dtype=f32) raises against the
            // dtype-overridden loop: "Cannot cast ufunc 'power' output from
            // dtype('float32') to dtype('int32') with casting rule 'same_kind'"
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.power(np.array(new int[] { 10 }), np.array(new int[] { 8 }),
                         np.zeros(new Shape(1), np.int32), dtype: NPTypeCode.Single));
            Assert.AreEqual(
                "Cannot cast ufunc 'power' output from dtype('float32') to dtype('int32') with casting rule 'same_kind'",
                ex.Message);
        }

        [TestMethod]
        public void FloorDivide_DtypeWithOut_LoopComputesInDtype()
        {
            // NumPy: floor_divide([-7],[2], out=f32, dtype=f64) → [-4.] float32
            var o = np.zeros(new Shape(1), np.float32);
            var r = np.floor_divide(np.array(new int[] { -7 }), np.array(new int[] { 2 }), o, dtype: NPTypeCode.Double);

            Assert.IsTrue(ReferenceEquals(r, o));
            Assert.AreEqual(-4.0f, o.GetSingle(0));
        }

        // =====================================================================
        // power: the negative-exponent rule follows the RESOLVED loop
        // =====================================================================

        [TestMethod]
        public void Power_NegativeExponent_FloatDtypeSelectsFloatLoop()
        {
            // NumPy: np.power([2], -1, dtype=np.float64) → [0.5]
            var r = np.power(np.array(new int[] { 2 }), np.array(new int[] { -1 }), dtype: NPTypeCode.Double);
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.AreEqual(0.5, r.GetDouble(0));
        }

        [TestMethod]
        public void Power_NegativeExponent_IntegerLoopStillRaises()
        {
            // NumPy: ValueError both bare and with an integer dtype=.
            var two = np.array(new int[] { 2 });
            var minusOne = np.array(new int[] { -1 });

            var ex1 = Assert.ThrowsException<ArgumentException>(() => np.power(two, minusOne));
            Assert.AreEqual("Integers to negative integer powers are not allowed.", ex1.Message);

            var ex2 = Assert.ThrowsException<ArgumentException>(() =>
                np.power(two, minusOne, dtype: NPTypeCode.Int64));
            Assert.AreEqual("Integers to negative integer powers are not allowed.", ex2.Message);
        }

        [TestMethod]
        public void Power_DtypeLoop_NoIntegerWrap()
        {
            // NumPy: np.power([10], 11, dtype=np.float64) → [1.e+11] exactly.
            // 10**11 overflows int32 — proof the loop runs in f64 rather than
            // computing the promoted i32 loop and casting the wrapped value.
            var r = np.power(np.array(new int[] { 10 }), np.array(new int[] { 11 }), dtype: NPTypeCode.Double);
            Assert.AreEqual(1e11, r.GetDouble(0));
        }

        // =====================================================================
        // dtype= input casting — same_kind, NumPy texts verbatim
        // =====================================================================

        [TestMethod]
        public void FloorDivide_DtypeInputCast_RaisesIndexedNumPyText()
        {
            // NumPy: "Cannot cast ufunc 'floor_divide' input 0 from
            // dtype('float64') to dtype('int32') with casting rule 'same_kind'"
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.floor_divide(np.array(new double[] { 7.0 }), np.array(new double[] { 2.0 }),
                                dtype: NPTypeCode.Int32));
            Assert.AreEqual(
                "Cannot cast ufunc 'floor_divide' input 0 from dtype('float64') to dtype('int32') with casting rule 'same_kind'",
                ex.Message);
        }

        [TestMethod]
        public void UnaryAllDtypeLoops_InputCast_RaisesUnindexedNumPyText()
        {
            // NumPy unary texts name no input index (probed):
            //   negative(f64, dtype=i32), square(f64, dtype=i32), abs(f64, dtype=i32)
            var f = np.array(new double[] { 1.5 });

            var exNeg = Assert.ThrowsException<ArgumentException>(() => np.negative(f, dtype: NPTypeCode.Int32));
            Assert.AreEqual(
                "Cannot cast ufunc 'negative' input from dtype('float64') to dtype('int32') with casting rule 'same_kind'",
                exNeg.Message);

            var exSq = Assert.ThrowsException<ArgumentException>(() => np.square(f, dtype: NPTypeCode.Int32));
            Assert.AreEqual(
                "Cannot cast ufunc 'square' input from dtype('float64') to dtype('int32') with casting rule 'same_kind'",
                exSq.Message);

            var exAbs = Assert.ThrowsException<ArgumentException>(() => np.abs(f, dtype: NPTypeCode.Int32));
            Assert.AreEqual(
                "Cannot cast ufunc 'absolute' input from dtype('float64') to dtype('int32') with casting rule 'same_kind'",
                exAbs.Message);
        }

        [TestMethod]
        public void FloatOnlyUfuncs_IntegerDtype_NoLoopMatching_NamesTheUfunc()
        {
            // NumPy: sqrt/exp/log/sin/cos/tan have float loops only — an
            // integer dtype= raises "No loop matching the specified signature
            // and casting was found for ufunc sqrt" (each names ITS ufunc).
            var x = np.array(new double[] { 4.0 });

            void AssertNoLoop(Action act, string ufunc)
            {
                var ex = Assert.ThrowsException<IncorrectTypeException>(act);
                Assert.AreEqual(
                    $"No loop matching the specified signature and casting was found for ufunc {ufunc}",
                    ex.Message);
            }

            AssertNoLoop(() => np.sqrt(x, dtype: NPTypeCode.Int32), "sqrt");
            AssertNoLoop(() => np.exp(x, dtype: NPTypeCode.Int32), "exp");
            AssertNoLoop(() => np.log(x, dtype: NPTypeCode.Int32), "log");
            AssertNoLoop(() => np.sin(x, dtype: NPTypeCode.Int32), "sin");
            AssertNoLoop(() => np.cos(x, dtype: NPTypeCode.Int32), "cos");
            AssertNoLoop(() => np.tan(x, dtype: NPTypeCode.Int32), "tan");
        }

        // =====================================================================
        // negative: dtype= selects the loop (bool rule, widening)
        // =====================================================================

        [TestMethod]
        public void Negative_BoolWithFloatDtype_IsLegal()
        {
            // NumPy: np.negative([True, False], dtype=np.float64) → [-1., -0.]
            var r = np.negative(np.array(new[] { true, false }), dtype: NPTypeCode.Double);
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.AreEqual(-1.0, r.GetDouble(0));
            Assert.AreEqual(0.0, r.GetDouble(1));
            Assert.IsTrue(double.IsNegative(r.GetDouble(1)), "NumPy yields -0.0 for negative(False)");
        }

        [TestMethod]
        public void Negative_BoolLoop_RaisesNumPyText()
        {
            // NumPy raises the same TypeError for plain bool input AND for an
            // explicit dtype=bool request (the loop is what's rejected).
            const string text =
                "The numpy boolean negative, the `-` operator, is not supported, " +
                "use the `~` operator or the logical_not function instead.";

            var ex1 = Assert.ThrowsException<NotSupportedException>(() => np.negative(np.array(new[] { true })));
            Assert.AreEqual(text, ex1.Message);

            var ex2 = Assert.ThrowsException<NotSupportedException>(() =>
                np.negative(np.array(new double[] { 1.0 }), dtype: NPTypeCode.Boolean));
            Assert.AreEqual(text, ex2.Message);
        }

        [TestMethod]
        public void Negative_DtypeWidening_MatchesNumPy()
        {
            // NumPy: negative(i32, dtype=i64) → int64; negative(u8, dtype=f32) → [-1., -2.]
            Assert.AreEqual(NPTypeCode.Int64,
                np.negative(np.array(new int[] { 1, 2 }), dtype: NPTypeCode.Int64).typecode);

            var r = np.negative(np.array(new byte[] { 1, 2 }), dtype: NPTypeCode.Single);
            Assert.AreEqual(NPTypeCode.Single, r.typecode);
            Assert.AreEqual(-1.0f, r.GetSingle(0));
            Assert.AreEqual(-2.0f, r.GetSingle(1));
        }

        // =====================================================================
        // square / abs dtype loops
        // =====================================================================

        [TestMethod]
        public void SquareAbs_DtypeLoops_MatchNumPy()
        {
            // NumPy: square(i32:3, dtype=i64) → 9 int64; abs(i32:-3, dtype=f64) → 3.0
            var sq = np.square(np.array(new int[] { 3 }), dtype: NPTypeCode.Int64);
            Assert.AreEqual(NPTypeCode.Int64, sq.typecode);
            Assert.AreEqual(9L, sq.GetInt64(0));

            var ab = np.abs(np.array(new int[] { -3 }), dtype: NPTypeCode.Double);
            Assert.AreEqual(NPTypeCode.Double, ab.typecode);
            Assert.AreEqual(3.0, ab.GetDouble(0));
        }

        // =====================================================================
        // legacy positional-dtype overloads stay source-compatible
        // =====================================================================

        [TestMethod]
        public void LegacyPositionalDtype_StillResolves()
        {
            var x = np.array(new double[] { 4.0 });

            Assert.AreEqual(NPTypeCode.Single, np.sqrt(x, NPTypeCode.Single).typecode);
            Assert.AreEqual(NPTypeCode.Single, np.sin(x, NPTypeCode.Single).typecode);
            Assert.AreEqual(NPTypeCode.Single, np.cos(x, NPTypeCode.Single).typecode);
            Assert.AreEqual(NPTypeCode.Single, np.tan(x, NPTypeCode.Single).typecode);
            Assert.AreEqual(NPTypeCode.Double, np.log(x, NPTypeCode.Double).typecode);
            Assert.AreEqual(NPTypeCode.Double, np.exp(x, NPTypeCode.Double).typecode);
            Assert.AreEqual(NPTypeCode.Single, np.abs(x, NPTypeCode.Single).typecode);
            Assert.AreEqual(NPTypeCode.Single, np.absolute(x, NPTypeCode.Single).typecode);
            Assert.AreEqual(NPTypeCode.Single,
                np.power(x, np.array(new double[] { 2.0 }), NPTypeCode.Single).typecode);
            Assert.AreEqual(NPTypeCode.Single,
                np.floor_divide(x, np.array(new double[] { 2.0 }), NPTypeCode.Single).typecode);
        }

        [TestMethod]
        public void TypeOverloads_NowHonorDtype()
        {
            // np.sqrt(x, typeof(float)) and np.log(x, typeof(float)) silently
            // IGNORED the dtype before this wave — now they route it.
            var x = np.array(new double[] { 4.0 });
            Assert.AreEqual(NPTypeCode.Single, np.sqrt(x, typeof(float)).typecode);
            Assert.AreEqual(NPTypeCode.Single, np.log(x, typeof(float)).typecode);
        }

        // =====================================================================
        // Binary six dtype= — the loop runs IN dtype (all probed NumPy 2.4.2)
        // =====================================================================

        [TestMethod]
        public void BinarySix_Dtype_SelectsTheLoop()
        {
            var a = np.array(new int[] { 1, 2 });
            var b = np.array(new int[] { 3, 4 });

            // add(i32, i32, dtype=f64) → [4., 6.] float64
            var r = np.add(a, b, dtype: NPTypeCode.Double);
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.AreEqual(4.0, r.GetDouble(0));

            // subtract(i64, i64, dtype=i16) → 295 in the int16 loop
            var s = np.subtract(np.array(new long[] { 300 }), np.array(new long[] { 5 }), dtype: NPTypeCode.Int16);
            Assert.AreEqual(NPTypeCode.Int16, s.typecode);
            Assert.AreEqual((short)295, s.GetInt16(0));

            // divide(i32, i32, dtype=f32) → float32 loop
            var d = np.divide(a, b, dtype: NPTypeCode.Single);
            Assert.AreEqual(NPTypeCode.Single, d.typecode);
            Assert.AreEqual(1f / 3f, d.GetSingle(0));

            // mod(i32:-7, i32:2, dtype=f64) → 1.0 (float remainder loop)
            var m = np.mod(np.array(new int[] { -7 }), np.array(new int[] { 2 }), dtype: NPTypeCode.Double);
            Assert.AreEqual(NPTypeCode.Double, m.typecode);
            Assert.AreEqual(1.0, m.GetDouble(0));

            // true_divide is the divide ufunc
            Assert.AreEqual(NPTypeCode.Single, np.true_divide(a, b, dtype: NPTypeCode.Single).typecode);

            // dtype + out compose: f32 loop value lands in the f64 out
            var o = np.zeros(new Shape(1), np.float64);
            np.add(np.array(new[] { 0.1 }), np.array(new[] { 0.2 }), o, dtype: NPTypeCode.Single);
            Assert.AreEqual((double)(0.1f + 0.2f), o.GetDouble(0));
        }

        [TestMethod]
        public void Add_BoolLoopRemap_FollowsTheResolvedLoop()
        {
            // NumPy: bool 'add' is logical OR (no integer bool loop), but
            // dtype= selects the loop — add(True, True, dtype=i32) runs the
            // int32 add loop and returns 2 (probed).
            var t = np.array(new[] { true });

            Assert.IsTrue(np.add(t, t).GetBoolean(0));                                    // True + True = True
            Assert.IsTrue(np.add(t, t, dtype: NPTypeCode.Boolean).GetBoolean(0));         // explicit bool loop
            Assert.AreEqual(2, np.add(t, t, dtype: NPTypeCode.Int32).GetInt32(0));        // i32 loop: 1+1=2
            Assert.AreEqual(1, np.multiply(t, t, dtype: NPTypeCode.Int32).GetInt32(0));   // i32 loop: 1*1=1
        }

        [TestMethod]
        public void BinarySix_Dtype_ErrorTexts()
        {
            var a = np.array(new int[] { 1, 2 });
            var x = np.array(new double[] { 1.5 });

            // divide is float-only: integer/bool dtype= names no loop.
            var exDiv = Assert.ThrowsException<IncorrectTypeException>(() =>
                np.divide(a, a, dtype: NPTypeCode.Int32));
            Assert.AreEqual("No loop matching the specified signature and casting was found for ufunc divide", exDiv.Message);
            Assert.ThrowsException<IncorrectTypeException>(() => np.divide(x, x, dtype: NPTypeCode.Boolean));

            // same_kind input casts, NumPy names ('remainder' for np.mod) + index.
            var exMod = Assert.ThrowsException<ArgumentException>(() =>
                np.mod(np.array(new double[] { 7.5 }), np.array(new double[] { 2.0 }), dtype: NPTypeCode.Int32));
            Assert.AreEqual(
                "Cannot cast ufunc 'remainder' input 0 from dtype('float64') to dtype('int32') with casting rule 'same_kind'",
                exMod.Message);

            var exAdd = Assert.ThrowsException<ArgumentException>(() =>
                np.add(a, a, dtype: NPTypeCode.Boolean));
            Assert.AreEqual(
                "Cannot cast ufunc 'add' input 0 from dtype('int32') to dtype('bool') with casting rule 'same_kind'",
                exAdd.Message);
        }

        // =====================================================================
        // Bitwise dtype= — bool/int loops only (probed)
        // =====================================================================

        [TestMethod]
        public void Bitwise_Dtype_IntLoopsOnly()
        {
            var a = np.array(new int[] { 1, 2 });
            var b = np.array(new int[] { 3, 4 });

            var r = np.bitwise_and(a, b, dtype: NPTypeCode.Int64);
            Assert.AreEqual(NPTypeCode.Int64, r.typecode);
            Assert.AreEqual(1L, r.GetInt64(0));
            Assert.AreEqual(0L, r.GetInt64(1));

            // i64 → i16 narrowing is same_kind: 300 survives
            var n = np.bitwise_and(np.array(new long[] { 300 }), np.array(new long[] { 300 }), dtype: NPTypeCode.Int16);
            Assert.AreEqual((short)300, n.GetInt16(0));

            var ex = Assert.ThrowsException<IncorrectTypeException>(() =>
                np.bitwise_and(a, b, dtype: NPTypeCode.Double));
            Assert.AreEqual("No loop matching the specified signature and casting was found for ufunc bitwise_and", ex.Message);
            Assert.ThrowsException<IncorrectTypeException>(() => np.bitwise_or(a, b, dtype: NPTypeCode.Single));
            Assert.ThrowsException<IncorrectTypeException>(() => np.bitwise_xor(a, b, dtype: NPTypeCode.Double));
        }

        // =====================================================================
        // positive — full ufunc surface (was plain-copy only; all probed)
        // =====================================================================

        [TestMethod]
        public void Positive_NumPyCallForms_OneOverload()
        {
            var x = np.array(new double[] { 1.5, -2.5 });

            // plain: identity copy
            var p = np.positive(x);
            Assert.AreEqual(1.5, p.GetDouble(0));
            Assert.AreEqual(-2.5, p.GetDouble(1));
            Assert.IsFalse(ReferenceEquals(p, x));

            // out= returns the provided instance
            var o = np.zeros(new Shape(2), np.float64);
            Assert.IsTrue(ReferenceEquals(np.positive(x, o), o));
            Assert.AreEqual(-2.5, o.GetDouble(1));

            // where= masks; false slots keep prior contents
            var o9 = np.ones(new Shape(2)) * 9.0;
            np.positive(x, o9, np.array(new[] { true, false }));
            Assert.AreEqual(1.5, o9.GetDouble(0));
            Assert.AreEqual(9.0, o9.GetDouble(1));

            // dtype= widens (positive(i32, dtype=f64) ≡ identity loop at f64)
            var w = np.positive(np.array(new int[] { 1, -2 }), dtype: NPTypeCode.Double);
            Assert.AreEqual(NPTypeCode.Double, w.typecode);
            Assert.AreEqual(-2.0, w.GetDouble(1));

            // dtype + out compose
            var of = np.zeros(new Shape(2), np.float32);
            np.positive(x, of, dtype: NPTypeCode.Double);
            Assert.AreEqual(-2.5f, of.GetSingle(1));
        }

        [TestMethod]
        public void Positive_LoopErrors_NumPyTexts()
        {
            var x = np.array(new double[] { 1.5 });

            // positive has no bool loop — plain bool raises naming "-> None"
            var ex1 = Assert.ThrowsException<TypeError>(() => np.positive(np.array(new[] { true })));
            Assert.AreEqual(
                "ufunc 'positive' did not contain a loop with signature matching types <class 'numpy.dtypes.BoolDType'> -> None",
                ex1.Message);

            // dtype=bool names both sides
            var ex2 = Assert.ThrowsException<TypeError>(() => np.positive(x, dtype: NPTypeCode.Boolean));
            Assert.AreEqual(
                "ufunc 'positive' did not contain a loop with signature matching types <class 'numpy.dtypes.Float64DType'> -> <class 'numpy.dtypes.BoolDType'>",
                ex2.Message);

            // but dtype= can SELECT a loop bool input can reach: [1., -0.] → [1., 0.]
            var ok = np.positive(np.array(new[] { true, false }), dtype: NPTypeCode.Double);
            Assert.AreEqual(1.0, ok.GetDouble(0));
            Assert.AreEqual(0.0, ok.GetDouble(1));

            // non-bool dtype follows the same_kind input rule
            var ex3 = Assert.ThrowsException<ArgumentException>(() => np.positive(x, dtype: NPTypeCode.Int32));
            Assert.AreEqual(
                "Cannot cast ufunc 'positive' input from dtype('float64') to dtype('int32') with casting rule 'same_kind'",
                ex3.Message);
        }

        // =====================================================================
        // round family — NumPy's round(a, decimals=0, out=None) single shape
        // =====================================================================

        [TestMethod]
        public void RoundFamily_SingleNumPyShape()
        {
            var x = np.array(new[] { 1.234, 5.678 });

            // round(a) / round(a, decimals) / round(a, decimals, out) — NumPy positions
            Assert.AreEqual(1.0, np.around(x).GetDouble(0));
            Assert.AreEqual(1.2, np.around(x, 1).GetDouble(0), 1e-12);

            var o = np.zeros(new Shape(2));
            var r = np.around(x, 1, o);
            Assert.IsTrue(ReferenceEquals(r, o));
            Assert.AreEqual(5.7, o.GetDouble(1), 1e-12);

            // out reachable by name without decimals (decimals defaults to 0)
            var o2 = np.zeros(new Shape(2));
            Assert.IsTrue(ReferenceEquals(np.round_(x, @out: o2), o2));
            Assert.AreEqual(6.0, o2.GetDouble(1));
        }
    }
}
