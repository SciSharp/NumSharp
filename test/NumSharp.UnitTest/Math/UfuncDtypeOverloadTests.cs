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
    }
}
