using System;
using System.Linq;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// Tests for np.fabs — the FLOAT-ONLY element-wise absolute value. All expected values probed
    /// against NumPy 2.4.2 (the <c>fabs</c> ufunc, <c>TD(flts, f='fabs', astype={'e':'f'})</c>).
    /// fabs is the float sibling of <see cref="np.absolute(NDArray)"/>: identical on the float loops
    /// (clear the IEEE sign bit — bit-exact incl. NaN payload and -0.0→+0.0), but it PROMOTES
    /// integer/bool to float (NEP50 tier) and has NO complex loop (unlike abs, which maps complex→magnitude).
    /// </summary>
    [TestClass]
    public class np_fabs_Test
    {
        // C#'s double.NaN is the NEGATIVE quiet NaN (0xfff8…); build both signs + a payload by bits.
        private static readonly double NegQNaN = BitConverter.Int64BitsToDouble(unchecked((long)0xfff8000000000000UL));
        private static readonly double NegPayloadNaN = BitConverter.Int64BitsToDouble(unchecked((long)0xfff8000000abcdefUL));

        [TestMethod]
        public void fabs_1D_Float64()
        {
            var arr = np.array(new[] { -1.5, 2.5, -3.5, 0.0, -0.0 });
            Assert.IsTrue(Enumerable.SequenceEqual(np.fabs(arr).Data<double>(),
                new[] { 1.5, 2.5, 3.5, 0.0, 0.0 }));
        }

        [TestMethod]
        public void fabs_1D_Float32()
        {
            var arr = np.array(new[] { -1.5f, 2.5f, -3.5f });
            Assert.IsTrue(Enumerable.SequenceEqual(np.fabs(arr).Data<float>(), new[] { 1.5f, 2.5f, 3.5f }));
        }

        // ---- Bit-exact edge values (the operation is "clear the sign bit") ----

        [TestMethod]
        public void fabs_SignedZero_BecomesPositiveZero()
        {
            // fabs(-0.0) is +0.0 — the sign bit is cleared, so the raw bits are exactly 0.
            var r = np.fabs(np.array(new[] { -0.0, 0.0 })).Data<double>();
            Assert.AreEqual(0x0L, BitConverter.DoubleToInt64Bits(r[0]));
            Assert.AreEqual(0x0L, BitConverter.DoubleToInt64Bits(r[1]));
        }

        [TestMethod]
        public void fabs_Infinities()
        {
            var r = np.fabs(np.array(new[] { double.NegativeInfinity, double.PositiveInfinity })).Data<double>();
            Assert.AreEqual(double.PositiveInfinity, r[0]);
            Assert.AreEqual(double.PositiveInfinity, r[1]);
        }

        [TestMethod]
        public void fabs_NaN_SignClearedPayloadPreserved()
        {
            // NumPy fabs clears ONLY the sign bit: -NaN(0xfff8…) → +NaN(0x7ff8…), payload kept.
            var r = np.fabs(np.array(new[] { NegQNaN, NegPayloadNaN })).Data<double>();
            Assert.AreEqual(unchecked((long)0x7ff8000000000000UL), BitConverter.DoubleToInt64Bits(r[0]));
            Assert.AreEqual(unchecked((long)0x7ff8000000abcdefUL), BitConverter.DoubleToInt64Bits(r[1]));
        }

        // ---- NEP50 float-tier return dtype (bool/i8/u8→f16, i16/u16→f32, i32+→f64) ----

        [TestMethod]
        public void fabs_ReturnDtype_FloatTier()
        {
            Assert.AreEqual(typeof(Half),   (Type)np.fabs(np.array(new[] { true, false })).dtype.type);
            Assert.AreEqual(typeof(Half),   (Type)np.fabs(np.array(new sbyte[] { -3, 4 })).dtype.type);
            Assert.AreEqual(typeof(Half),   (Type)np.fabs(np.array(new byte[] { 3, 4 })).dtype.type);
            Assert.AreEqual(typeof(float),  (Type)np.fabs(np.array(new short[] { -3, 4 })).dtype.type);
            Assert.AreEqual(typeof(float),  (Type)np.fabs(np.array(new ushort[] { 3, 4 })).dtype.type);
            Assert.AreEqual(typeof(double), (Type)np.fabs(np.array(new[] { -3, 4 })).dtype.type);          // int32
            Assert.AreEqual(typeof(double), (Type)np.fabs(np.array(new uint[] { 3, 4 })).dtype.type);
            Assert.AreEqual(typeof(double), (Type)np.fabs(np.array(new long[] { -3, 4 })).dtype.type);
            Assert.AreEqual(typeof(double), (Type)np.fabs(np.array(new ulong[] { 3, 4 })).dtype.type);
            // Floats preserved.
            Assert.AreEqual(typeof(Half),   (Type)np.fabs(np.array(new[] { (Half)(-3) })).dtype.type);
            Assert.AreEqual(typeof(float),  (Type)np.fabs(np.array(new[] { -3f })).dtype.type);
            Assert.AreEqual(typeof(double), (Type)np.fabs(np.array(new[] { -3.0 })).dtype.type);
        }

        [TestMethod]
        public void fabs_IntMinValue_NoWrap()
        {
            // The int is cast to float BEFORE the abs, so int.MinValue does not wrap (as integer abs would):
            // fabs(int32 -2147483648) = 2147483648.0 (float64), not the wrapped negative.
            Assert.AreEqual(2147483648.0, np.fabs(np.array(new[] { int.MinValue })).Data<double>()[0]);
            Assert.AreEqual(9.2233720368547758E18, np.fabs(np.array(new[] { long.MinValue })).Data<double>()[0]);
        }

        [TestMethod]
        public void fabs_Decimal_Preserved()
        {
            // Decimal has no NumPy analog; NumSharp treats it as float-family (preserved, exact abs).
            var r = np.fabs(np.array(new[] { -3.5m, 2.5m }));
            Assert.AreEqual(typeof(decimal), (Type)r.dtype.type);
            Assert.IsTrue(Enumerable.SequenceEqual(r.Data<decimal>(), new[] { 3.5m, 2.5m }));
        }

        // ---- Complex rejection (fabs has no complex loop) ----

        [TestMethod]
        public void fabs_Complex_NoDtype_Throws()
        {
            var c = np.array(new[] { new Complex(3, 4) });
            var ex = Assert.ThrowsException<TypeError>(() => np.fabs(c));
            StringAssert.Contains(ex.Message, "ufunc 'fabs' not supported for the input types");
        }

        [TestMethod]
        public void fabs_Complex_FloatDtype_ThrowsCastError()
        {
            var c = np.array(new[] { new Complex(3, 4) });
            var ex = Assert.ThrowsException<ArgumentException>(() => np.fabs(c, dtype: np.float64));
            StringAssert.Contains(ex.Message,
                "Cannot cast ufunc 'fabs' input from dtype('complex128') to dtype('float64') with casting rule 'same_kind'");
        }

        [TestMethod]
        public void fabs_ComplexDtype_Throws_NoLoop()
        {
            var ex = Assert.ThrowsException<IncorrectTypeException>(() => np.fabs(np.array(new[] { -1.5 }), dtype: np.complex128));
            StringAssert.Contains(ex.Message, "No loop matching the specified signature and casting was found for ufunc fabs");
        }

        [TestMethod]
        public void fabs_IntegerDtype_Throws_NoLoop()
        {
            Assert.ThrowsException<IncorrectTypeException>(() => np.fabs(np.array(new[] { -3 }), dtype: np.int32));
            Assert.ThrowsException<IncorrectTypeException>(() => np.fabs(np.array(new[] { -3 }), dtype: np.int64));
        }

        [TestMethod]
        public void fabs_Complex_BadWhere_WhereErrorWins()
        {
            // NumPy validation order: the where-bool check (argument parsing) precedes the complex-input loop error.
            var c = np.array(new[] { new Complex(3, 4) });
            var ex = Assert.ThrowsException<ArgumentException>(() => np.fabs(c, where: np.array(new[] { 1 })));
            StringAssert.Contains(ex.Message, "Cannot cast array data from dtype('int32') to dtype('bool')");
        }

        // ---- dtype= parameter (float loops accepted) ----

        [TestMethod]
        public void fabs_DtypeParam_SelectsFloatLoop()
        {
            Assert.AreEqual(typeof(float), (Type)np.fabs(np.array(new[] { -3, 4, -5 }), dtype: np.float32).dtype.type);
            Assert.AreEqual(typeof(Half),  (Type)np.fabs(np.array(new[] { -3 }), dtype: np.float16).dtype.type);
            // float64 input + dtype=float32 computes at float32 precision.
            var r = np.fabs(np.array(new[] { -1.5, 2.5 }), dtype: np.float32);
            Assert.AreEqual(typeof(float), (Type)r.dtype.type);
            Assert.IsTrue(Enumerable.SequenceEqual(r.Data<float>(), new[] { 1.5f, 2.5f }));
        }

        // ---- out= / where= ----

        [TestMethod]
        public void fabs_Out_ReturnsSameInstance()
        {
            var o = np.zeros(3, np.float64);
            var r = np.fabs(np.array(new[] { -3, 4, -5 }), @out: o);
            Assert.IsTrue(ReferenceEquals(r, o));
            Assert.IsTrue(Enumerable.SequenceEqual(o.Data<double>(), new[] { 3.0, 4.0, 5.0 }));
        }

        [TestMethod]
        public void fabs_Out_IncompatibleDtype_Throws_NamesFabs()
        {
            // The loop dtype is float64 (int32 input); out=int64 is not a same_kind cast. The error must
            // name "fabs" (not "absolute") even though the kernel is UnaryOp.Abs.
            var ex = Assert.ThrowsException<ArgumentException>(
                () => np.fabs(np.array(new[] { -3 }), @out: np.zeros(1, np.int64)));
            StringAssert.Contains(ex.Message,
                "Cannot cast ufunc 'fabs' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'");
        }

        [TestMethod]
        public void fabs_Where_MasksOff_KeepsPriorContents()
        {
            var o = np.zeros(3, np.float64);
            var r = np.fabs(np.array(new[] { -1.5, 2.5, -3.5 }), @out: o, where: np.array(new[] { true, false, true }));
            Assert.IsTrue(Enumerable.SequenceEqual(r.Data<double>(), new[] { 1.5, 0.0, 3.5 }));
        }

        // ---- Layouts (fabs reuses the abs kernel across every memory layout) ----

        [TestMethod]
        public void fabs_StridedAndReversed()
        {
            var v = np.arange(6).astype(np.float64) - 3;      // [-3,-2,-1,0,1,2]
            Assert.IsTrue(Enumerable.SequenceEqual(np.fabs(v["::2"]).Data<double>(), new[] { 3.0, 1.0, 1.0 }));
            Assert.IsTrue(Enumerable.SequenceEqual(np.fabs(v["::-1"]).Data<double>(), new[] { 2.0, 1.0, 0.0, 1.0, 2.0, 3.0 }));
        }

        [TestMethod]
        public void fabs_Transposed()
        {
            var m = (np.arange(6).astype(np.float64) - 3).reshape(2, 3);   // [[-3,-2,-1],[0,1,2]]
            var r = np.fabs(m.T);                                          // (3,2)
            Assert.AreEqual(3, r.shape[0]);
            Assert.AreEqual(2, r.shape[1]);
            Assert.IsTrue(Enumerable.SequenceEqual(r.flatten().Data<double>(), new[] { 3.0, 0.0, 2.0, 1.0, 1.0, 2.0 }));
        }

        [TestMethod]
        public void fabs_Scalar0D_And_Empty()
        {
            Assert.AreEqual(7.5, np.fabs(np.array(-7.5)).GetDouble(new long[0]));
            Assert.AreEqual(0, np.fabs(np.zeros(new Shape(0, 3))).size);
        }

        [TestMethod]
        public void fabs_Broadcast()
        {
            var row = np.array(new[] { -1.0, -2.0, -3.0 }).reshape(1, 3);
            var b = np.broadcast_to(row, new Shape(2, 3));
            Assert.IsTrue(Enumerable.SequenceEqual(np.fabs(b).flatten().Data<double>(),
                new[] { 1.0, 2.0, 3.0, 1.0, 2.0, 3.0 }));
        }
    }
}
