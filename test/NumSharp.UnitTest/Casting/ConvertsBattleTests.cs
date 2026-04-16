using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Utilities;

namespace NumSharp.UnitTest.Casting
{
    /// <summary>
    /// Battletests for previously broken conversion paths in Converts.cs / Converts.Native.cs.
    ///
    /// Covers 7 bugs discovered during audit:
    /// - #1: ToChar(object) no Half/Complex/bool handling
    /// - #2: ToDecimal(object) no Complex/Half handling
    /// - #3: ToChar(bool) throws
    /// - #4: ToChar(float/double/decimal) throws
    /// - #5: ToChar(Half) no NaN/Inf check
    /// - #6: ToByte/UInt16/UInt32(decimal) throws on negatives
    /// - #7: ToDecimal(float/double) NaN/Inf throws
    ///
    /// Since Char and Decimal are NOT NumPy types, behavior is derived from
    /// the closest NumPy equivalent:
    /// - char (16-bit unsigned) mirrors uint16: wrap modularly, NaN/Inf → 0
    /// - decimal: no direct NumPy equivalent, use uint64 NaN pattern (2^63) is wrong for decimal;
    ///   pick Decimal.Zero for NaN/Inf (smallest consistent choice), wrap negatives for unsigned.
    /// </summary>
    [TestClass]
    public class ConvertsBattleTests
    {
        #region Bug #3: ToChar(bool) must not throw

        [TestMethod]
        public void ToChar_Bool_False_ReturnsZero()
        {
            Converts.ToChar(false).Should().Be((char)0);
        }

        [TestMethod]
        public void ToChar_Bool_True_ReturnsOne()
        {
            Converts.ToChar(true).Should().Be((char)1);
        }

        [TestMethod]
        public void BoolArray_AsType_Char_DoesNotThrow()
        {
            var arr = np.array(new[] { true, false, true });
            var r = arr.astype(NPTypeCode.Char);
            ((ushort)r.GetAtIndex<char>(0)).Should().Be(1);
            ((ushort)r.GetAtIndex<char>(1)).Should().Be(0);
            ((ushort)r.GetAtIndex<char>(2)).Should().Be(1);
        }

        #endregion

        #region Bug #4: ToChar(float/double/decimal) must truncate+wrap with NaN/Inf → 0

        [TestMethod]
        public void ToChar_Double_Normal_Truncates()
        {
            Converts.ToChar(65.7).Should().Be((char)65); // truncate toward zero
            Converts.ToChar(65.0).Should().Be((char)65);
        }

        [TestMethod]
        public void ToChar_Double_Negative_Wraps()
        {
            // -1 wraps to 65535 (ushort MAX)
            ((ushort)Converts.ToChar(-1.0)).Should().Be(65535);
            ((ushort)Converts.ToChar(-1.5)).Should().Be(65535);
        }

        [TestMethod]
        public void ToChar_Double_Overflow_Wraps()
        {
            // 65536 wraps to 0
            ((ushort)Converts.ToChar(65536.0)).Should().Be(0);
            ((ushort)Converts.ToChar(65537.0)).Should().Be(1);
        }

        [TestMethod]
        public void ToChar_Double_NaN_ReturnsZero()
        {
            Converts.ToChar(double.NaN).Should().Be((char)0);
        }

        [TestMethod]
        public void ToChar_Double_Infinity_ReturnsZero()
        {
            Converts.ToChar(double.PositiveInfinity).Should().Be((char)0);
            Converts.ToChar(double.NegativeInfinity).Should().Be((char)0);
        }

        [TestMethod]
        public void ToChar_Double_OutsideInt32Range_ReturnsZero()
        {
            // Values outside int32 range should overflow to 0 (NumPy small-type pattern)
            Converts.ToChar(1e20).Should().Be((char)0);
            Converts.ToChar(-1e20).Should().Be((char)0);
        }

        [TestMethod]
        public void ToChar_Float_Normal_Truncates()
        {
            Converts.ToChar(65.7f).Should().Be((char)65);
            ((ushort)Converts.ToChar(-1.0f)).Should().Be(65535);
        }

        [TestMethod]
        public void ToChar_Float_NaN_ReturnsZero()
        {
            Converts.ToChar(float.NaN).Should().Be((char)0);
            Converts.ToChar(float.PositiveInfinity).Should().Be((char)0);
            Converts.ToChar(float.NegativeInfinity).Should().Be((char)0);
        }

        [TestMethod]
        public void ToChar_Decimal_Normal_Truncates()
        {
            Converts.ToChar(65.7m).Should().Be((char)65);
        }

        [TestMethod]
        public void ToChar_Decimal_Negative_Wraps()
        {
            ((ushort)Converts.ToChar(-1m)).Should().Be(65535);
            ((ushort)Converts.ToChar(-1.5m)).Should().Be(65535);
        }

        [TestMethod]
        public void DoubleArray_AsType_Char_Works()
        {
            var arr = np.array(new[] { 65.0, -1.0, 65536.0, double.NaN, double.PositiveInfinity });
            var r = arr.astype(NPTypeCode.Char);
            ((ushort)r.GetAtIndex<char>(0)).Should().Be(65);
            ((ushort)r.GetAtIndex<char>(1)).Should().Be(65535);
            ((ushort)r.GetAtIndex<char>(2)).Should().Be(0);
            ((ushort)r.GetAtIndex<char>(3)).Should().Be(0);
            ((ushort)r.GetAtIndex<char>(4)).Should().Be(0);
        }

        [TestMethod]
        public void FloatArray_AsType_Char_Works()
        {
            var arr = np.array(new[] { 65.0f, -1.0f, 65536.0f, float.NaN, float.PositiveInfinity });
            var r = arr.astype(NPTypeCode.Char);
            ((ushort)r.GetAtIndex<char>(0)).Should().Be(65);
            ((ushort)r.GetAtIndex<char>(1)).Should().Be(65535);
            ((ushort)r.GetAtIndex<char>(2)).Should().Be(0);
            ((ushort)r.GetAtIndex<char>(3)).Should().Be(0);
            ((ushort)r.GetAtIndex<char>(4)).Should().Be(0);
        }

        [TestMethod]
        public void DecimalArray_AsType_Char_Works()
        {
            var arr = np.array(new[] { 65m, -1m, 65537m });
            var r = arr.astype(NPTypeCode.Char);
            ((ushort)r.GetAtIndex<char>(0)).Should().Be(65);
            ((ushort)r.GetAtIndex<char>(1)).Should().Be(65535);
            ((ushort)r.GetAtIndex<char>(2)).Should().Be(1);
        }

        #endregion

        #region Bug #5: ToChar(Half) NaN/Inf must return 0

        [TestMethod]
        public void ToChar_Half_NaN_ReturnsZero()
        {
            Converts.ToChar(Half.NaN).Should().Be((char)0);
        }

        [TestMethod]
        public void ToChar_Half_Infinity_ReturnsZero()
        {
            Converts.ToChar(Half.PositiveInfinity).Should().Be((char)0);
            Converts.ToChar(Half.NegativeInfinity).Should().Be((char)0);
        }

        [TestMethod]
        public void ToChar_Half_Normal_Truncates()
        {
            Converts.ToChar((Half)65.0f).Should().Be((char)65);
            ((ushort)Converts.ToChar((Half)(-1.0f))).Should().Be(65535);
        }

        [TestMethod]
        public void HalfArray_AsType_Char_HandlesSpecialValues()
        {
            var arr = np.array(new[] { (Half)65.0f, Half.NaN, Half.PositiveInfinity, Half.NegativeInfinity });
            var r = arr.astype(NPTypeCode.Char);
            ((ushort)r.GetAtIndex<char>(0)).Should().Be(65);
            ((ushort)r.GetAtIndex<char>(1)).Should().Be(0);
            ((ushort)r.GetAtIndex<char>(2)).Should().Be(0);
            ((ushort)r.GetAtIndex<char>(3)).Should().Be(0);
        }

        #endregion

        #region Bug #6: ToByte/UInt16/UInt32(decimal) negatives must wrap modularly

        [TestMethod]
        public void ToByte_Decimal_Negative_Wraps()
        {
            // -1 wraps to 255 (matches float→byte behavior)
            Converts.ToByte(-1m).Should().Be((byte)255);
            Converts.ToByte(-1.5m).Should().Be((byte)255); // truncate first, then wrap
            Converts.ToByte(-128m).Should().Be((byte)128);
        }

        [TestMethod]
        public void ToByte_Decimal_Overflow_Wraps()
        {
            Converts.ToByte(256m).Should().Be((byte)0);
            Converts.ToByte(257m).Should().Be((byte)1);
        }

        [TestMethod]
        public void ToUInt16_Decimal_Negative_Wraps()
        {
            Converts.ToUInt16(-1m).Should().Be((ushort)65535);
            Converts.ToUInt16(-1.5m).Should().Be((ushort)65535);
        }

        [TestMethod]
        public void ToUInt16_Decimal_Overflow_Wraps()
        {
            Converts.ToUInt16(65536m).Should().Be((ushort)0);
        }

        [TestMethod]
        public void ToUInt32_Decimal_Negative_Wraps()
        {
            Converts.ToUInt32(-1m).Should().Be(uint.MaxValue);
            Converts.ToUInt32(-1.5m).Should().Be(uint.MaxValue);
        }

        [TestMethod]
        public void ToUInt64_Decimal_Negative_Wraps()
        {
            Converts.ToUInt64(-1m).Should().Be(ulong.MaxValue);
        }

        [TestMethod]
        public void DecimalArray_AsType_UnsignedTypes_NegativeWraps()
        {
            var arr = np.array(new[] { -1.5m, -100m, 5m });

            var resByte = arr.astype(NPTypeCode.Byte);
            resByte.GetAtIndex<byte>(0).Should().Be(255);
            resByte.GetAtIndex<byte>(1).Should().Be(156);
            resByte.GetAtIndex<byte>(2).Should().Be(5);

            var resUInt16 = arr.astype(NPTypeCode.UInt16);
            resUInt16.GetAtIndex<ushort>(0).Should().Be(65535);
            resUInt16.GetAtIndex<ushort>(1).Should().Be(65436);
            resUInt16.GetAtIndex<ushort>(2).Should().Be(5);

            var resUInt32 = arr.astype(NPTypeCode.UInt32);
            resUInt32.GetAtIndex<uint>(0).Should().Be(4294967295);
            resUInt32.GetAtIndex<uint>(1).Should().Be(4294967196);
            resUInt32.GetAtIndex<uint>(2).Should().Be(5);

            var resUInt64 = arr.astype(NPTypeCode.UInt64);
            resUInt64.GetAtIndex<ulong>(0).Should().Be(18446744073709551615);
            resUInt64.GetAtIndex<ulong>(2).Should().Be(5);
        }

        #endregion

        #region Bug #7: ToDecimal(float/double) NaN/Inf must return 0

        [TestMethod]
        public void ToDecimal_Double_NaN_ReturnsZero()
        {
            Converts.ToDecimal(double.NaN).Should().Be(0m);
        }

        [TestMethod]
        public void ToDecimal_Double_Infinity_ReturnsZero()
        {
            Converts.ToDecimal(double.PositiveInfinity).Should().Be(0m);
            Converts.ToDecimal(double.NegativeInfinity).Should().Be(0m);
        }

        [TestMethod]
        public void ToDecimal_Float_NaN_ReturnsZero()
        {
            Converts.ToDecimal(float.NaN).Should().Be(0m);
            Converts.ToDecimal(float.PositiveInfinity).Should().Be(0m);
            Converts.ToDecimal(float.NegativeInfinity).Should().Be(0m);
        }

        [TestMethod]
        public void ToDecimal_Double_Overflow_ReturnsZero()
        {
            // Values exceeding Decimal's range must also return 0 (not throw)
            Converts.ToDecimal(1e30).Should().Be(0m);
            Converts.ToDecimal(-1e30).Should().Be(0m);
        }

        [TestMethod]
        public void DoubleArray_AsType_Decimal_HandlesSpecialValues()
        {
            var arr = np.array(new[] { 1.5, double.NaN, double.PositiveInfinity, double.NegativeInfinity });
            var r = arr.astype(NPTypeCode.Decimal);
            r.GetAtIndex<decimal>(0).Should().Be(1.5m);
            r.GetAtIndex<decimal>(1).Should().Be(0m);
            r.GetAtIndex<decimal>(2).Should().Be(0m);
            r.GetAtIndex<decimal>(3).Should().Be(0m);
        }

        [TestMethod]
        public void FloatArray_AsType_Decimal_HandlesSpecialValues()
        {
            var arr = np.array(new[] { 1.5f, float.NaN, float.PositiveInfinity });
            var r = arr.astype(NPTypeCode.Decimal);
            r.GetAtIndex<decimal>(0).Should().BeApproximately(1.5m, 0.0001m);
            r.GetAtIndex<decimal>(1).Should().Be(0m);
            r.GetAtIndex<decimal>(2).Should().Be(0m);
        }

        #endregion

        #region Bug #1: ToChar(object) must handle Half/Complex/bool

        [TestMethod]
        public void ToChar_Object_Bool_Works()
        {
            Converts.ToChar((object)true).Should().Be((char)1);
            Converts.ToChar((object)false).Should().Be((char)0);
        }

        [TestMethod]
        public void ToChar_Object_Half_Works()
        {
            Converts.ToChar((object)(Half)65.0f).Should().Be((char)65);
            Converts.ToChar((object)Half.NaN).Should().Be((char)0);
            Converts.ToChar((object)Half.PositiveInfinity).Should().Be((char)0);
        }

        [TestMethod]
        public void ToChar_Object_Complex_Works()
        {
            Converts.ToChar((object)new Complex(65, 0)).Should().Be((char)65);
            Converts.ToChar((object)new Complex(65, 5)).Should().Be((char)65); // imaginary discarded
            ((ushort)Converts.ToChar((object)new Complex(-1, 0))).Should().Be(65535);
        }

        [TestMethod]
        public void ToChar_Object_Double_Works()
        {
            Converts.ToChar((object)65.5).Should().Be((char)65);
            Converts.ToChar((object)double.NaN).Should().Be((char)0);
        }

        [TestMethod]
        public void ToChar_Object_Float_Works()
        {
            Converts.ToChar((object)65.5f).Should().Be((char)65);
            Converts.ToChar((object)float.NaN).Should().Be((char)0);
        }

        [TestMethod]
        public void ToChar_Object_Decimal_Works()
        {
            Converts.ToChar((object)65.5m).Should().Be((char)65);
            ((ushort)Converts.ToChar((object)(-1m))).Should().Be(65535);
        }

        #endregion

        #region Bug #2: ToDecimal(object) must handle Complex/Half

        [TestMethod]
        public void ToDecimal_Object_Complex_Works()
        {
            // Complex: takes real part
            Converts.ToDecimal((object)new Complex(3.5, 4.5)).Should().Be(3.5m);
            Converts.ToDecimal((object)new Complex(-1.5, 0)).Should().Be(-1.5m);
        }

        [TestMethod]
        public void ToDecimal_Object_Half_Works()
        {
            Converts.ToDecimal((object)(Half)1.5f).Should().BeApproximately(1.5m, 0.01m);
        }

        [TestMethod]
        public void ToDecimal_Object_DoubleNaN_ReturnsZero()
        {
            Converts.ToDecimal((object)double.NaN).Should().Be(0m);
            Converts.ToDecimal((object)double.PositiveInfinity).Should().Be(0m);
        }

        [TestMethod]
        public void ToDecimal_Object_FloatNaN_ReturnsZero()
        {
            Converts.ToDecimal((object)float.NaN).Should().Be(0m);
        }

        #endregion

        #region Cross-path consistency: astype() with various sources → Decimal

        [TestMethod]
        public void ComplexArray_AsType_Decimal_Works()
        {
            var arr = np.array(new[] { new Complex(3.5, 4.5), new Complex(-1.5, 0) });
            var r = arr.astype(NPTypeCode.Decimal);
            r.GetAtIndex<decimal>(0).Should().Be(3.5m);
            r.GetAtIndex<decimal>(1).Should().Be(-1.5m);
        }

        [TestMethod]
        public void HalfArray_AsType_Decimal_Works()
        {
            var arr = np.array(new[] { (Half)1.5f, (Half)(-1.5f) });
            var r = arr.astype(NPTypeCode.Decimal);
            ((double)r.GetAtIndex<decimal>(0)).Should().BeApproximately(1.5, 0.01);
            ((double)r.GetAtIndex<decimal>(1)).Should().BeApproximately(-1.5, 0.01);
        }

        #endregion
    }
}
