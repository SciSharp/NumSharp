using System;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// out=/where=/dtype= coverage for the three ufunc families whose np.* overloads gained the
    /// NumPy-shaped parameter surface in the 2026-09-18 "deferred three" pass — the last ufuncs that
    /// lacked it:
    ///  • bit shifts (<c>left_shift</c>/<c>right_shift</c>) — route the shift kernel through
    ///    <c>ExecuteBinaryOp</c>'s out=/where= path; a float/complex/decimal dtype= has no loop, and a
    ///    uint64×signed pair promotes to float64 (no loop) → the "not supported for the input types" error;
    ///  • logical (<c>logical_and</c>/<c>or</c>/<c>xor</c>/<c>not</c>) — bool-output; each input reduced to
    ///    its truth value (nonzero; NaN is truthy) then combined through the bitwise engine's out=/where=;
    ///  • two-output <c>modf</c> — promotion (int/bool/char→float64, float16 preserved, complex rejected)
    ///    plus <c>out=(frac,integral)</c>/<c>where=</c>/<c>dtype=</c>.
    ///
    /// Every expectation is pinned to a NumPy 2.4.2 probe.
    /// </summary>
    [TestClass]
    public class UfuncShiftLogicalModfOutWhereTests
    {
        private static NDArray Mask4() => np.array(new[] { true, false, true, false });

        // =====================================================================
        // SHIFTS — out=, where=, dtype=, cast, error taxonomy
        // =====================================================================

        [TestMethod]
        public void LeftShift_Out_ReturnsSameInstance_WithValues()
        {
            // NumPy: left_shift([1,2,4,8], 2) -> [4,8,16,32].
            var a = np.array(new[] { 1, 2, 4, 8 });
            var o = np.zeros(new Shape(4), np.int32);
            var r = np.left_shift(a, (object)2, @out: o);
            Assert.IsTrue(ReferenceEquals(r, o), "out= must return the provided instance");
            var exp = new[] { 4, 8, 16, 32 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(exp[i], o.GetInt32(i));
        }

        [TestMethod]
        public void LeftShift_Where_MaskedOffKeepPrior()
        {
            // NumPy: left_shift([1,2,4,8], 1, out=prior=99, where=[T,F,T,F]) -> [2,99,8,99].
            var a = np.array(new[] { 1, 2, 4, 8 });
            var o = np.full(new Shape(4), 99, np.int32);
            np.left_shift(a, (object)1, @out: o, where: Mask4());
            var exp = new[] { 2, 99, 8, 99 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(exp[i], o.GetInt32(i));
        }

        [TestMethod]
        public void LeftShift_Dtype_SelectsLoop_And_OutCast()
        {
            var a = np.array(new[] { 1, 2, 4, 8 });
            // dtype=int64 runs the int64 loop.
            Assert.AreEqual(np.int64, np.left_shift(a, (object)1, dtype: np.int64).dtype);
            // out=int16 accepts the same_kind cast from the int32 result.
            var o16 = np.zeros(new Shape(4), np.int16);
            np.left_shift(a, (object)1, @out: o16);
            var exp = new[] { 2, 4, 8, 16 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual((short)exp[i], o16.GetInt16(i));
            // right_shift out=float64: int32 -> float64 is a safe (hence same_kind) cast.
            var of = np.zeros(new Shape(4), np.float64);
            np.right_shift(a, (object)1, @out: of);
            var expF = new[] { 0.0, 1.0, 2.0, 4.0 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expF[i], of.GetDouble(i));
        }

        [TestMethod]
        public void LeftShift_Dtype_ComputesInLoop_ThenTruncatesIntoOut()
        {
            // NumPy: left_shift(1, 40, dtype=int64, out=int32) computes 2^40 at int64, stores int32 -> 0.
            var one = np.array(new[] { 1 });
            var o = np.zeros(new Shape(1), np.int32);
            np.left_shift(one, (object)40, dtype: np.int64, @out: o);
            Assert.AreEqual(0, o.GetInt32(0));
        }

        [TestMethod]
        public void LeftShift_OutCast_SameKindFail_Throws()
        {
            var a = np.array(new[] { 1, 2, 4, 8 });
            // int32 -> uint8 is not same_kind (signed->unsigned), NumPy UFuncTypeError.
            Assert.ThrowsException<ArgumentException>(() =>
                np.left_shift(a, (object)1, @out: np.zeros(new Shape(4), np.uint8)));
            // int32 -> bool is not same_kind either.
            Assert.ThrowsException<ArgumentException>(() =>
                np.left_shift(a, (object)1, @out: np.zeros(new Shape(4), np.@bool)));
        }

        [TestMethod]
        public void LeftShift_FloatDtype_NoLoop_Throws()
        {
            var a = np.array(new[] { 1, 2, 4, 8 });
            // dtype=float64 has no shift loop -> "No loop matching the specified signature ...".
            var ex = Assert.ThrowsException<IncorrectTypeException>(() =>
                np.left_shift(a, (object)1, dtype: np.float64));
            StringAssert.Contains(ex.Message, "No loop matching");
            StringAssert.Contains(ex.Message, "left_shift");
        }

        [TestMethod]
        public void LeftShift_FloatOperand_NotSupported_Throws()
        {
            // A float value or a float count has no shift loop -> "not supported for the input types".
            var ex1 = Assert.ThrowsException<TypeError>(() =>
                np.left_shift(np.array(new[] { 1.0, 2.0 }), (object)1));
            StringAssert.Contains(ex1.Message, "not supported for the input types");
            var ex2 = Assert.ThrowsException<TypeError>(() =>
                np.left_shift(np.array(new[] { 1, 2 }), (object)1.0));
            StringAssert.Contains(ex2.Message, "not supported for the input types");
        }

        [TestMethod]
        public void LeftShift_UInt64TimesSigned_PromotesToFloat64_NotSupported()
        {
            // result_type(uint64, int64) == float64 (NEP50) -> no shift loop.
            var ex = Assert.ThrowsException<TypeError>(() =>
                np.left_shift(np.array(new ulong[] { 1 }), np.array(new long[] { 2 })));
            StringAssert.Contains(ex.Message, "not supported for the input types");
        }

        [TestMethod]
        public void LeftShift_ArrayByArray_Out()
        {
            // NumPy: left_shift([1,2,4],[1,2,3]) -> [2,8,32].
            var o = np.zeros(new Shape(3), np.int32);
            np.left_shift(np.array(new[] { 1, 2, 4 }), np.array(new[] { 1, 2, 3 }), @out: o);
            var exp = new[] { 2, 8, 32 };
            for (int i = 0; i < 3; i++)
                Assert.AreEqual(exp[i], o.GetInt32(i));
        }

        // =====================================================================
        // LOGICAL — out=, where=, dtype=, bool-output cast, truthiness
        // =====================================================================

        [TestMethod]
        public void LogicalAnd_Out_ReturnsSameInstance_WithValues()
        {
            // NumPy: logical_and([0,1,2,0],[1,1,0,0]) -> [F,T,F,F].
            var x1 = np.array(new[] { 0, 1, 2, 0 });
            var x2 = np.array(new[] { 1, 1, 0, 0 });
            var o = np.zeros(new Shape(4), np.@bool);
            var r = np.logical_and(x1, x2, @out: o);
            Assert.IsTrue(ReferenceEquals(r, o));
            var exp = new[] { false, true, false, false };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(exp[i], o.GetBoolean(i));
        }

        [TestMethod]
        public void LogicalAnd_OutNumeric_TrueToOne()
        {
            // NumPy: logical_and(...,out=int32) -> [0,1,0,0] (bool casts same_kind to int, True->1).
            var x1 = np.array(new[] { 0, 1, 2, 0 });
            var x2 = np.array(new[] { 1, 1, 0, 0 });
            var oi = np.zeros(new Shape(4), np.int32);
            np.logical_and(x1, x2, @out: oi);
            var exp = new[] { 0, 1, 0, 0 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(exp[i], oi.GetInt32(i));
            // out=float64 / out=complex128 likewise accept bool via same_kind.
            var of = np.zeros(new Shape(4), np.float64);
            np.logical_and(x1, x2, @out: of);
            Assert.AreEqual(1.0, of.GetDouble(1));
            var oc = np.zeros(new Shape(4), np.complex128);
            np.logical_and(x1, x2, @out: oc);
            Assert.AreEqual(1.0, oc.GetComplex(1).Real);
        }

        [TestMethod]
        public void LogicalOr_Where_MaskedOffKeepPrior()
        {
            // NumPy: logical_or([0,1,2,0],[1,1,0,0], out=prior=True, where=[T,F,T,F]) -> [T,T,T,T].
            // (or(0,1)=T at 0; prior T at 1; or(2,0)=T at 2; prior T at 3)
            var x1 = np.array(new[] { 0, 1, 2, 0 });
            var x2 = np.array(new[] { 1, 1, 0, 0 });
            var o = np.full(new Shape(4), true, np.@bool);
            np.logical_or(x1, x2, @out: o, where: Mask4());
            for (int i = 0; i < 4; i++)
                Assert.IsTrue(o.GetBoolean(i));
        }

        [TestMethod]
        public void Logical_Dtype_BoolOnly()
        {
            var x1 = np.array(new[] { 0, 1, 2, 0 });
            var x2 = np.array(new[] { 1, 1, 0, 0 });
            // dtype=bool is a no-op.
            Assert.AreEqual(np.@bool, np.logical_and(x1, x2, dtype: np.@bool).dtype);
            // dtype=int32 -> no loop, naming the LOGICAL ufunc.
            var ex = Assert.ThrowsException<IncorrectTypeException>(() =>
                np.logical_and(x1, x2, dtype: np.int32));
            StringAssert.Contains(ex.Message, "No loop matching");
            StringAssert.Contains(ex.Message, "logical_and");
            var ex2 = Assert.ThrowsException<IncorrectTypeException>(() =>
                np.logical_not(x1, dtype: np.int32));
            StringAssert.Contains(ex2.Message, "logical_not");
        }

        [TestMethod]
        public void LogicalNot_Out_And_Truthiness()
        {
            // NumPy: logical_not([0,1,2,0], out=int32) -> [1,0,0,1].
            var x1 = np.array(new[] { 0, 1, 2, 0 });
            var oi = np.zeros(new Shape(4), np.int32);
            np.logical_not(x1, @out: oi);
            var exp = new[] { 1, 0, 0, 1 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(exp[i], oi.GetInt32(i));
            // NaN is truthy: logical_not([0,1.5,nan,inf]) -> [T,F,F,F].
            var f = np.array(new[] { 0.0, 1.5, double.NaN, double.PositiveInfinity });
            var rf = np.logical_not(f);
            var expF = new[] { true, false, false, false };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expF[i], rf.GetBoolean(i));
        }

        [TestMethod]
        public void LogicalAnd_Float_And_Complex_Truthiness()
        {
            // logical_and([0,1,nan],[1,0,1]) -> [F,F,T] (nan is truthy so nan&&1 == T).
            var r = np.logical_and(np.array(new[] { 0.0, 1.0, double.NaN }),
                                   np.array(new[] { 1.0, 0.0, 1.0 }));
            var exp = new[] { false, false, true };
            for (int i = 0; i < 3; i++)
                Assert.AreEqual(exp[i], r.GetBoolean(i));
            // complex nonzero = (re!=0 || im!=0): [0+0j,1+0j,0+1j] && [1,1,1] -> [F,T,T].
            var rc = np.logical_and(
                np.array(new System.Numerics.Complex[] { new(0, 0), new(1, 0), new(0, 1) }),
                np.array(new[] { 1, 1, 1 }));
            var expC = new[] { false, true, true };
            for (int i = 0; i < 3; i++)
                Assert.AreEqual(expC[i], rc.GetBoolean(i));
        }

        [TestMethod]
        public void LogicalXor_Values()
        {
            // NumPy: logical_xor([0,1,2,0],[1,1,0,0]) -> [T,F,T,F].
            var r = np.logical_xor(np.array(new[] { 0, 1, 2, 0 }), np.array(new[] { 1, 1, 0, 0 }));
            var exp = new[] { true, false, true, false };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(exp[i], r.GetBoolean(i));
        }

        [TestMethod]
        public void Logical_WhereMask_MustBeBool()
        {
            var x1 = np.array(new[] { 0, 1, 2, 0 });
            var x2 = np.array(new[] { 1, 1, 0, 0 });
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.logical_and(x1, x2, @out: np.zeros(new Shape(4), np.@bool),
                    where: np.array(new[] { 1, 0, 1, 0 })));
            StringAssert.Contains(ex.Message, "bool");
        }

        [TestMethod]
        public void Logical_ReadonlyOut_Throws()
        {
            var x1 = np.array(new[] { 0, 1, 2, 0 });
            var x2 = np.array(new[] { 1, 1, 0, 0 });
            var ro = np.broadcast_to(np.array(false), new Shape(4)); // read-only broadcast view
            var ex = Assert.ThrowsException<ValueError>(() => np.logical_and(x1, x2, @out: ro));
            StringAssert.Contains(ex.Message, "read-only");
        }

        // =====================================================================
        // MODF — promotion, out=(frac,integral), where=, dtype=, error taxonomy
        // =====================================================================

        [TestMethod]
        public void Modf_Promotes_ByWidthTier()
        {
            // NumPy uses the narrowest-float tier (NOT blanket float64): probed 2.4.2.
            //   int32 -> float64, int16 -> float32, int8/uint8/bool -> float16.
            var (f, i) = np.modf(np.array(new[] { 1, 2, 3 }));   // int32
            Assert.AreEqual(np.float64, f.dtype);
            Assert.AreEqual(np.float64, i.dtype);
            for (int k = 0; k < 3; k++) Assert.AreEqual(0.0, f.GetDouble(k));
            var expI = new[] { 1.0, 2.0, 3.0 };
            for (int k = 0; k < 3; k++) Assert.AreEqual(expI[k], i.GetDouble(k));
            // int16 -> float32.
            Assert.AreEqual(np.float32, np.modf(np.array(new short[] { 1, 2 })).Fractional.dtype);
            // int8 -> float16.
            Assert.AreEqual(np.float16, np.modf(np.array(new sbyte[] { 1, 2 })).Fractional.dtype);
            // bool -> float16, True -> 1.0.
            var (fb, ib) = np.modf(np.array(new[] { true, false, true }));
            Assert.AreEqual(np.float16, ib.dtype);
            var expIb = new[] { (Half)1.0, (Half)0.0, (Half)1.0 };
            for (int k = 0; k < 3; k++) Assert.AreEqual(expIb[k], ib.GetValue<Half>(k));
        }

        [TestMethod]
        public void Modf_Float16_Preserved_BitExact()
        {
            // NumPy HALF_modf: widen->modff->narrow both. Preserves float16 dtype, bit-exact.
            var (f, i) = np.modf(np.array(new Half[] { (Half)1.5, (Half)(-2.7), (Half)3.0 }));
            Assert.AreEqual(np.float16, f.dtype);
            Assert.AreEqual(np.float16, i.dtype);
            Assert.AreEqual((Half)0.5, f.GetValue<Half>(0));
            Assert.AreEqual((Half)1.0, i.GetValue<Half>(0));
            // -2.7 as float16 is -2.69921875; frac = -0.69921875, int = -2.0.
            Assert.AreEqual((Half)(-2.0), i.GetValue<Half>(1));
        }

        [TestMethod]
        public void Modf_Complex_NotSupported()
        {
            var ex = Assert.ThrowsException<TypeError>(() =>
                np.modf(np.array(new System.Numerics.Complex[] { new(1, 2) })));
            StringAssert.Contains(ex.Message, "not supported for the input types");
        }

        [TestMethod]
        public void Modf_SpecialValues()
        {
            // NumPy: modf([1.5,-2.7,inf,-inf,nan]) frac=[0.5,-0.7,+0,-0,nan] int=[1,-2,inf,-inf,nan].
            var (f, i) = np.modf(np.array(new[] {
                1.5, -2.7, double.PositiveInfinity, double.NegativeInfinity, double.NaN }));
            Assert.AreEqual(0.5, f.GetDouble(0));
            Assert.AreEqual(0.0, f.GetDouble(2));                     // frac(inf) = +0
            Assert.IsTrue(double.IsPositive(f.GetDouble(2)));         // +0, not -0
            Assert.IsTrue(double.IsNegative(f.GetDouble(3)));         // frac(-inf) = -0
            Assert.IsTrue(double.IsNaN(f.GetDouble(4)));
            Assert.IsTrue(double.IsPositiveInfinity(i.GetDouble(2)));
            Assert.IsTrue(double.IsNegativeInfinity(i.GetDouble(3)));
        }

        [TestMethod]
        public void Modf_Out_ReturnsProvidedInstances()
        {
            // NumPy: modf(m, out=(f,i)) writes and returns (f,i).
            var m = np.array(new[] { 1.5, -2.7, 3.0 });
            var of = np.zeros(new Shape(3), np.float64);
            var oi = np.zeros(new Shape(3), np.float64);
            var (rf, ri) = np.modf(m, of, oi);
            Assert.IsTrue(ReferenceEquals(rf, of));
            Assert.IsTrue(ReferenceEquals(ri, oi));
            Assert.AreEqual(0.5, of.GetDouble(0));
            Assert.AreEqual(1.0, oi.GetDouble(0));
            Assert.AreEqual(-2.0, oi.GetDouble(1));
        }

        [TestMethod]
        public void Modf_PositionalFracOnly_And_IntegralOnly()
        {
            var m = np.array(new[] { 1.5, -2.7, 3.0 });
            // modf(m, f): frac into f, integral fresh (matches NumPy's np.modf(x, out1)).
            var of = np.zeros(new Shape(3), np.float64);
            var (rf, ri) = np.modf(m, of);
            Assert.IsTrue(ReferenceEquals(rf, of));
            Assert.AreEqual(0.5, of.GetDouble(0));
            Assert.AreEqual(1.0, ri.GetDouble(0));                    // fresh integral computed
            // modf(m, null, i): integral into i, frac fresh.
            var oi = np.zeros(new Shape(3), np.float64);
            var (rf2, ri2) = np.modf(m, null, oi);
            Assert.IsTrue(ReferenceEquals(ri2, oi));
            Assert.AreEqual(1.0, oi.GetDouble(0));
            Assert.AreEqual(0.5, rf2.GetDouble(0));
        }

        [TestMethod]
        public void Modf_Where_MasksBothOutputs()
        {
            // NumPy: modf(m, out=(f,i), where=[T,F,T], prior=9) -> f=[0.5,9,0], i=[1,9,3].
            var m = np.array(new[] { 1.5, -2.7, 3.0 });
            var of = np.full(new Shape(3), 9.0, np.float64);
            var oi = np.full(new Shape(3), 9.0, np.float64);
            np.modf(m, of, oi, where: np.array(new[] { true, false, true }));
            Assert.AreEqual(0.5, of.GetDouble(0));
            Assert.AreEqual(9.0, of.GetDouble(1));                    // masked-off keeps prior
            Assert.AreEqual(0.0, of.GetDouble(2));
            Assert.AreEqual(1.0, oi.GetDouble(0));
            Assert.AreEqual(9.0, oi.GetDouble(1));
            Assert.AreEqual(3.0, oi.GetDouble(2));
        }

        [TestMethod]
        public void Modf_WhereScalarFalse_KeepsAllPrior()
        {
            var m = np.array(new[] { 1.5, -2.7, 3.0 });
            var of = np.full(new Shape(3), 7.0, np.float64);
            var oi = np.full(new Shape(3), 7.0, np.float64);
            np.modf(m, of, oi, where: NDArray.Scalar(false));
            for (int k = 0; k < 3; k++)
            {
                Assert.AreEqual(7.0, of.GetDouble(k));
                Assert.AreEqual(7.0, oi.GetDouble(k));
            }
        }

        [TestMethod]
        public void Modf_Dtype_And_OutCast()
        {
            var m = np.array(new[] { 1.5, -2.7, 3.0 });
            // dtype=float32 -> both outputs float32.
            var (f, i) = np.modf(m, dtype: np.float32);
            Assert.AreEqual(np.float32, f.dtype);
            Assert.AreEqual(np.float32, i.dtype);
            // out=(f32,f64): each output cast independently from the float64 loop.
            var of = np.zeros(new Shape(3), np.float32);
            var oi = np.zeros(new Shape(3), np.float64);
            np.modf(m, of, oi);
            Assert.AreEqual(np.float32, of.dtype);
            Assert.AreEqual(0.5f, of.GetSingle(0));
            Assert.AreEqual(1.0, oi.GetDouble(0));
        }

        [TestMethod]
        public void Modf_BroadcastsInputUpToBiggerOut()
        {
            // NumPy: modf((3,), out=(2,3)) broadcasts the input up.
            var m = np.array(new[] { 1.5, -2.7, 3.0 });
            var of = np.zeros(new Shape(2, 3), np.float64);
            var oi = np.zeros(new Shape(2, 3), np.float64);
            np.modf(m, of, oi);
            Assert.AreEqual(0.5, of.GetDouble(0, 0));
            Assert.AreEqual(0.5, of.GetDouble(1, 0));                 // repeated row
        }

        [TestMethod]
        public void Modf_Dtype_NonFloat_NoLoop()
        {
            var m = np.array(new[] { 1.5, -2.7, 3.0 });
            var ex1 = Assert.ThrowsException<IncorrectTypeException>(() => np.modf(m, dtype: np.int32));
            StringAssert.Contains(ex1.Message, "No loop matching");
            StringAssert.Contains(ex1.Message, "modf");
            Assert.ThrowsException<IncorrectTypeException>(() => np.modf(m, dtype: np.complex128));
        }

        [TestMethod]
        public void Modf_OutCast_ReportsOperandIndex()
        {
            // NumPy numbers the OPERAND: frac = output 1, integral = output 2.
            var m = np.array(new[] { 1.5, -2.7, 3.0 });
            var ex1 = Assert.ThrowsException<ArgumentException>(() =>
                np.modf(m, np.zeros(new Shape(3), np.int32), np.zeros(new Shape(3), np.float64)));
            StringAssert.Contains(ex1.Message, "output 1");
            StringAssert.Contains(ex1.Message, "modf");
            var ex2 = Assert.ThrowsException<ArgumentException>(() =>
                np.modf(m, np.zeros(new Shape(3), np.float64), np.zeros(new Shape(3), np.int32)));
            StringAssert.Contains(ex2.Message, "output 2");
        }

        [TestMethod]
        public void Modf_OutWrongShape_BroadcastError()
        {
            // NumPy: "operands could not be broadcast together with shapes (3,) (5,) (3,) ".
            var m = np.array(new[] { 1.5, -2.7, 3.0 });
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.modf(m, np.zeros(new Shape(5), np.float64), np.zeros(new Shape(3), np.float64)));
            StringAssert.Contains(ex.Message, "could not be broadcast together with shapes");
            StringAssert.Contains(ex.Message, "(3,) (5,) (3,)");
        }

        [TestMethod]
        public void Modf_ReadonlyOut_Throws()
        {
            var m = np.array(new[] { 1.5, -2.7, 3.0 });
            var ro = np.broadcast_to(np.array(0.0), new Shape(3));
            var ex = Assert.ThrowsException<ValueError>(() =>
                np.modf(m, ro, np.zeros(new Shape(3), np.float64)));
            StringAssert.Contains(ex.Message, "read-only");
        }

        [TestMethod]
        public void Modf_WhereMask_MustBeBool()
        {
            var m = np.array(new[] { 1.5, -2.7, 3.0 });
            Assert.ThrowsException<ArgumentException>(() =>
                np.modf(m, np.zeros(new Shape(3), np.float64), np.zeros(new Shape(3), np.float64),
                    where: np.array(new[] { 1, 0, 1 })));
        }

        [TestMethod]
        public void Modf_ScalarInput_ZeroDim()
        {
            var (f, i) = np.modf(np.array(1.5));
            Assert.AreEqual(0, f.ndim);
            Assert.AreEqual(0.5, f.GetDouble());
            Assert.AreEqual(1.0, i.GetDouble());
        }
    }
}
