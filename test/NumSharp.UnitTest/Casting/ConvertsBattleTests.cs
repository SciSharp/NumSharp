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

        // ============================================================
        // ROUND 2 BUGS: char source broken paths, fallback converter
        // issues, and the 3-arg ChangeType(obj, NPTypeCode, provider)
        // ============================================================

        #region Round 2 Group A: typed ToXxx(char) scalars throw via IConvertible

        [TestMethod]
        public void ToBoolean_Char_Zero_ReturnsFalse()
        {
            Converts.ToBoolean((char)0).Should().BeFalse();
        }

        [TestMethod]
        public void ToBoolean_Char_NonZero_ReturnsTrue()
        {
            Converts.ToBoolean('A').Should().BeTrue();
            Converts.ToBoolean((char)1).Should().BeTrue();
            Converts.ToBoolean(char.MaxValue).Should().BeTrue();
        }

        [TestMethod]
        public void ToSingle_Char_ReturnsNumeric()
        {
            Converts.ToSingle('A').Should().Be(65.0f);
            Converts.ToSingle((char)0).Should().Be(0.0f);
        }

        [TestMethod]
        public void ToDouble_Char_ReturnsNumeric()
        {
            Converts.ToDouble('A').Should().Be(65.0);
            Converts.ToDouble((char)0).Should().Be(0.0);
        }

        [TestMethod]
        public void ToDecimal_Char_ReturnsNumeric()
        {
            Converts.ToDecimal('A').Should().Be(65m);
            Converts.ToDecimal((char)0).Should().Be(0m);
        }

        #endregion

        #region Round 2 Group B: ToXxx(object) dispatchers must handle char

        [TestMethod]
        public void ToBoolean_Object_Char_Works()
        {
            Converts.ToBoolean((object)'A').Should().BeTrue();
            Converts.ToBoolean((object)(char)0).Should().BeFalse();
        }

        [TestMethod]
        public void ToSingle_Object_Char_Works()
        {
            Converts.ToSingle((object)'A').Should().Be(65.0f);
        }

        [TestMethod]
        public void ToDouble_Object_Char_Works()
        {
            Converts.ToDouble((object)'A').Should().Be(65.0);
        }

        [TestMethod]
        public void ToHalf_Object_Char_Works()
        {
            ((float)Converts.ToHalf((object)'A')).Should().Be(65.0f);
        }

        [TestMethod]
        public void ToComplex_Object_Char_Works()
        {
            var r = Converts.ToComplex((object)'A');
            r.Real.Should().Be(65.0);
            r.Imaginary.Should().Be(0.0);
        }

        [TestMethod]
        public void ToDecimal_Object_Char_Works()
        {
            // Round 1 fix already handled this, but lock it in
            Converts.ToDecimal((object)'A').Should().Be(65m);
        }

        #endregion

        #region Round 2 Array path: char source to all targets

        [TestMethod]
        public void CharArray_AsType_Bool_Works()
        {
            var arr = np.array(new[] { 'A', (char)0, 'Z' });
            var r = arr.astype(NPTypeCode.Boolean);
            r.GetAtIndex<bool>(0).Should().BeTrue();
            r.GetAtIndex<bool>(1).Should().BeFalse();
            r.GetAtIndex<bool>(2).Should().BeTrue();
        }

        [TestMethod]
        public void CharArray_AsType_Single_Works()
        {
            var arr = np.array(new[] { 'A', 'B' });
            var r = arr.astype(NPTypeCode.Single);
            r.GetAtIndex<float>(0).Should().Be(65.0f);
            r.GetAtIndex<float>(1).Should().Be(66.0f);
        }

        [TestMethod]
        public void CharArray_AsType_Double_Works()
        {
            var arr = np.array(new[] { 'A', 'B' });
            var r = arr.astype(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().Be(65.0);
            r.GetAtIndex<double>(1).Should().Be(66.0);
        }

        [TestMethod]
        public void CharArray_AsType_Decimal_Works()
        {
            var arr = np.array(new[] { 'A', 'B' });
            var r = arr.astype(NPTypeCode.Decimal);
            r.GetAtIndex<decimal>(0).Should().Be(65m);
            r.GetAtIndex<decimal>(1).Should().Be(66m);
        }

        [TestMethod]
        public void CharArray_AsType_Half_Works()
        {
            var arr = np.array(new[] { 'A', 'B' });
            var r = arr.astype(NPTypeCode.Half);
            ((float)r.GetAtIndex<Half>(0)).Should().Be(65.0f);
            ((float)r.GetAtIndex<Half>(1)).Should().Be(66.0f);
        }

        [TestMethod]
        public void CharArray_AsType_Complex_Works()
        {
            var arr = np.array(new[] { 'A', 'B' });
            var r = arr.astype(NPTypeCode.Complex);
            r.GetAtIndex<Complex>(0).Should().Be(new Complex(65, 0));
            r.GetAtIndex<Complex>(1).Should().Be(new Complex(66, 0));
        }

        #endregion

        #region Round 2 Group C: CreateFallbackConverter with char source

        [TestMethod]
        public void FindConverter_Char_To_Half_Works()
        {
            var f = Converts.FindConverter<char, Half>();
            ((float)f('A')).Should().Be(65.0f);
        }

        [TestMethod]
        public void FindConverter_Char_To_Complex_Works()
        {
            var f = Converts.FindConverter<char, Complex>();
            f('A').Should().Be(new Complex(65, 0));
        }

        #endregion

        #region Round 2 Group D: CreateDefaultConverter NaN safety

        [TestMethod]
        public void FindConverter_HalfNaN_To_Decimal_ReturnsZero()
        {
            // Routes through CreateDefaultConverter which must not throw on NaN
            var f = Converts.FindConverter<Half, decimal>();
            f(Half.NaN).Should().Be(0m);
        }

        [TestMethod]
        public void FindConverter_HalfInf_To_Decimal_ReturnsZero()
        {
            var f = Converts.FindConverter<Half, decimal>();
            f(Half.PositiveInfinity).Should().Be(0m);
            f(Half.NegativeInfinity).Should().Be(0m);
        }

        [TestMethod]
        public void HalfArray_NaN_AsType_Decimal_ReturnsZero()
        {
            var arr = np.array(new[] { Half.NaN, Half.PositiveInfinity, (Half)3.5f });
            var r = arr.astype(NPTypeCode.Decimal);
            r.GetAtIndex<decimal>(0).Should().Be(0m);
            r.GetAtIndex<decimal>(1).Should().Be(0m);
            ((double)r.GetAtIndex<decimal>(2)).Should().BeApproximately(3.5, 0.01);
        }

        #endregion

        #region Round 2 Group E: ChangeType(obj, NPTypeCode, IFormatProvider)

        [TestMethod]
        public void ChangeType_WithProvider_Half_Source_Works()
        {
            Converts.ChangeType((object)Half.One, NPTypeCode.Int32, null).Should().Be(1);
            Converts.ChangeType((object)(Half)(-1.5f), NPTypeCode.Int32, null).Should().Be(-1);
        }

        [TestMethod]
        public void ChangeType_WithProvider_Complex_Source_Works()
        {
            Converts.ChangeType((object)new Complex(5, 0), NPTypeCode.Int32, null).Should().Be(5);
            Converts.ChangeType((object)new Complex(3.5, 4.5), NPTypeCode.Int32, null).Should().Be(3);
        }

        [TestMethod]
        public void ChangeType_WithProvider_Half_Target_Works()
        {
            var result = Converts.ChangeType((object)5, NPTypeCode.Half, null);
            result.Should().BeOfType<Half>();
            ((float)(Half)result).Should().Be(5.0f);
        }

        [TestMethod]
        public void ChangeType_WithProvider_Complex_Target_Works()
        {
            var result = Converts.ChangeType((object)5, NPTypeCode.Complex, null);
            result.Should().BeOfType<Complex>();
            ((Complex)result).Should().Be(new Complex(5, 0));
        }

        [TestMethod]
        public void ChangeType_WithProvider_SByte_Target_Works()
        {
            var result = Converts.ChangeType((object)5, NPTypeCode.SByte, null);
            result.Should().Be((sbyte)5);
        }

        [TestMethod]
        public void ChangeType_WithProvider_NaN_To_Int_ReturnsMinValue()
        {
            // NumPy parity: NaN -> int32.MinValue
            Converts.ChangeType((object)double.NaN, NPTypeCode.Int32, null).Should().Be(int.MinValue);
        }

        [TestMethod]
        public void ChangeType_WithProvider_NaN_To_SmallInt_ReturnsZero()
        {
            Converts.ChangeType((object)double.NaN, NPTypeCode.Byte, null).Should().Be((byte)0);
            Converts.ChangeType((object)double.NaN, NPTypeCode.Int16, null).Should().Be((short)0);
        }

        #endregion

        // ============================================================
        // ROUND 3: IConvertible constraint removed on generic ChangeType;
        // Converts<T> gets ToHalf/ToComplex/From(Half)/From(Complex).
        // ============================================================

        #region Round 3: ChangeType<T>(T, NPTypeCode) now accepts Half/Complex

        [TestMethod]
        public void ChangeTypeGeneric_HalfSource_ToInt_Works()
        {
            var r = Converts.ChangeType((Half)(-1.5f), NPTypeCode.Int32);
            r.Should().Be(-1);
        }

        [TestMethod]
        public void ChangeTypeGeneric_HalfSource_NaN_ToInt_ReturnsMinValue()
        {
            var r = Converts.ChangeType(Half.NaN, NPTypeCode.Int32);
            r.Should().Be(int.MinValue);
        }

        [TestMethod]
        public void ChangeTypeGeneric_ComplexSource_ToInt_Works()
        {
            var r = Converts.ChangeType(new Complex(3.5, 4.5), NPTypeCode.Int32);
            r.Should().Be(3);
        }

        [TestMethod]
        public void ChangeTypeGeneric_ComplexSource_ToBool_PureImaginary_True()
        {
            var r = Converts.ChangeType(new Complex(0, 1), NPTypeCode.Boolean);
            r.Should().Be(true);
        }

        [TestMethod]
        public void ChangeTypeGeneric_IntSource_ToHalf_Works()
        {
            var r = Converts.ChangeType(5, NPTypeCode.Half);
            r.Should().BeOfType<Half>();
            ((float)(Half)r).Should().Be(5.0f);
        }

        [TestMethod]
        public void ChangeTypeGeneric_IntSource_ToComplex_Works()
        {
            var r = Converts.ChangeType(5, NPTypeCode.Complex);
            r.Should().Be(new Complex(5, 0));
        }

        [TestMethod]
        public void ChangeTypeGeneric_SByteSource_ToInt_Works()
        {
            var r = Converts.ChangeType((sbyte)-1, NPTypeCode.Int32);
            r.Should().Be(-1);
        }

        [TestMethod]
        public void ChangeTypeGeneric_IntSource_ToSByte_Works()
        {
            var r = Converts.ChangeType(-1, NPTypeCode.SByte);
            r.Should().Be((sbyte)-1);
        }

        #endregion

        #region Round 3: ChangeType<TIn, TOut>(TIn) now accepts Half/Complex

        [TestMethod]
        public void ChangeType2Generic_HalfToInt_Works()
        {
            Converts.ChangeType<Half, int>((Half)3.5f).Should().Be(3);
        }

        [TestMethod]
        public void ChangeType2Generic_IntToHalf_Works()
        {
            ((float)Converts.ChangeType<int, Half>(5)).Should().Be(5.0f);
        }

        [TestMethod]
        public void ChangeType2Generic_ComplexToInt_Works()
        {
            Converts.ChangeType<Complex, int>(new Complex(3.5, 4.5)).Should().Be(3);
        }

        [TestMethod]
        public void ChangeType2Generic_ComplexToHalf_Works()
        {
            ((float)Converts.ChangeType<Complex, Half>(new Complex(3.5, 4.5))).Should().Be(3.5f);
        }

        [TestMethod]
        public void ChangeType2Generic_HalfNaN_ToDecimal_ReturnsZero()
        {
            Converts.ChangeType<Half, decimal>(Half.NaN).Should().Be(0m);
        }

        [TestMethod]
        public void ChangeType2Generic_SByteToUInt64_Wraps()
        {
            Converts.ChangeType<sbyte, ulong>(-1).Should().Be(ulong.MaxValue);
        }

        #endregion

        #region Round 3: Converts<T>.ToHalf/ToComplex + From(Half)/From(Complex)

        [TestMethod]
        public void ConvertsGeneric_ToHalf_FromInt_Works()
        {
            ((float)Converts<int>.ToHalf(5)).Should().Be(5.0f);
        }

        [TestMethod]
        public void ConvertsGeneric_ToHalf_FromDouble_NaN_KeepsNaN()
        {
            // Half can represent NaN; Converts.ToHalf(double) uses (Half)d which preserves NaN
            Half.IsNaN(Converts<double>.ToHalf(double.NaN)).Should().BeTrue();
        }

        [TestMethod]
        public void ConvertsGeneric_ToComplex_FromDouble_Works()
        {
            Converts<double>.ToComplex(3.5).Should().Be(new Complex(3.5, 0));
        }

        [TestMethod]
        public void ConvertsGeneric_ToComplex_FromSByte_Works()
        {
            Converts<sbyte>.ToComplex(-1).Should().Be(new Complex(-1, 0));
        }

        [TestMethod]
        public void ConvertsGeneric_FromHalf_ToInt_Works()
        {
            Converts<int>.From((Half)3.5f).Should().Be(3);
        }

        [TestMethod]
        public void ConvertsGeneric_FromHalf_ToDouble_Works()
        {
            Converts<double>.From((Half)3.5f).Should().BeApproximately(3.5, 0.01);
        }

        [TestMethod]
        public void ConvertsGeneric_FromComplex_ToInt_DiscardsImaginary()
        {
            Converts<int>.From(new Complex(3.5, 4.5)).Should().Be(3);
        }

        [TestMethod]
        public void ConvertsGeneric_FromComplex_ToDouble_DiscardsImaginary()
        {
            Converts<double>.From(new Complex(3.5, 4.5)).Should().Be(3.5);
        }

        [TestMethod]
        public void ConvertsGeneric_FromComplex_ToBool_Any_NonZero()
        {
            Converts<bool>.From(new Complex(0, 1)).Should().BeTrue();
            Converts<bool>.From(new Complex(0, 0)).Should().BeFalse();
        }

        #endregion

        #region Round 4: String target + .NET TypeCode-based ChangeType

        // Group A: String target for Half/Complex sources.
        // Previously ChangeType<string>(Half/Complex) and ChangeType(Half/Complex, NPTypeCode.String)
        // cast to IConvertible which fails since Half/Complex don't implement it.

        [TestMethod]
        public void ChangeTypeGeneric_HalfToString_Works()
        {
            Converts.ChangeType<string>(Half.One).Should().Be("1");
        }

        [TestMethod]
        public void ChangeTypeGeneric_ComplexToString_Works()
        {
            Converts.ChangeType<string>(new Complex(3, 4)).Should().Be("<3; 4>");
        }

        [TestMethod]
        public void ChangeType_HalfToNPTypeCodeString_Works()
        {
            Converts.ChangeType((object)(Half)3.14f, NPTypeCode.String).Should().Be("3.14");
        }

        [TestMethod]
        public void ChangeType_ComplexToNPTypeCodeString_Works()
        {
            Converts.ChangeType((object)new Complex(3, 4), NPTypeCode.String).Should().Be("<3; 4>");
        }

        // Group B: FindConverter<Half/Complex, string> routes through
        // CreateFallbackConverter → CreateDefaultConverter → Converts.ChangeType(obj, NPTypeCode.String).

        [TestMethod]
        public void FindConverter_HalfToString_Works()
        {
            var conv = Converts.FindConverter<Half, string>();
            conv(Half.One).Should().Be("1");
        }

        [TestMethod]
        public void FindConverter_ComplexToString_Works()
        {
            var conv = Converts.FindConverter<Complex, string>();
            conv(new Complex(3, 4)).Should().Be("<3; 4>");
        }

        // Group C: .NET-style ChangeType(Object, TypeCode) and (Object, TypeCode, IFormatProvider).
        // These were never fixed and used raw IConvertible casts. Half/Complex throw, char→Boolean
        // throws via IConvertible.ToBoolean (char doesn't support it).

        [TestMethod]
        public void ChangeTypeTypeCode_HalfToInt32_Works()
        {
            Converts.ChangeType((object)Half.One, TypeCode.Int32).Should().Be(1);
        }

        [TestMethod]
        public void ChangeTypeTypeCode_HalfToDouble_Works()
        {
            Converts.ChangeType((object)(Half)3.5f, TypeCode.Double).Should().Be((double)3.5);
        }

        [TestMethod]
        public void ChangeTypeTypeCode_HalfToDecimal_Works()
        {
            Converts.ChangeType((object)Half.One, TypeCode.Decimal).Should().Be(1m);
        }

        [TestMethod]
        public void ChangeTypeTypeCode_ComplexToInt32_DiscardsImaginary()
        {
            Converts.ChangeType((object)new Complex(7, 3), TypeCode.Int32).Should().Be(7);
        }

        [TestMethod]
        public void ChangeTypeTypeCode_ComplexToDouble_DiscardsImaginary()
        {
            Converts.ChangeType((object)new Complex(3.5, 4.5), TypeCode.Double).Should().Be(3.5);
        }

        [TestMethod]
        public void ChangeTypeTypeCode_CharToBoolean_Works()
        {
            // 'A' (65) is truthy per NumPy rules
            Converts.ChangeType((object)'A', TypeCode.Boolean).Should().Be(true);
            // (char)0 is falsy
            Converts.ChangeType((object)(char)0, TypeCode.Boolean).Should().Be(false);
        }

        [TestMethod]
        public void ChangeTypeTypeCode_CharToSingle_Works()
        {
            Converts.ChangeType((object)'A', TypeCode.Single).Should().Be(65f);
        }

        [TestMethod]
        public void ChangeTypeTypeCode_HalfToString_UsesInvariantCulture()
        {
            // String target: use IFormattable with invariant culture
            Converts.ChangeType((object)(Half)3.14f, TypeCode.String).Should().Be("3.14");
        }

        [TestMethod]
        public void ChangeTypeTypeCode_ComplexToString_Works()
        {
            Converts.ChangeType((object)new Complex(3, 4), TypeCode.String).Should().Be("<3; 4>");
        }

        [TestMethod]
        public void ChangeTypeTypeCode3Arg_HalfToInt32_Works()
        {
            // 3-arg overload with IFormatProvider
            Converts.ChangeType((object)Half.One, TypeCode.Int32, System.Globalization.CultureInfo.InvariantCulture).Should().Be(1);
        }

        [TestMethod]
        public void ChangeTypeTypeCode3Arg_ComplexToInt32_Works()
        {
            Converts.ChangeType((object)new Complex(7, 3), TypeCode.Int32, System.Globalization.CultureInfo.InvariantCulture).Should().Be(7);
        }

        [TestMethod]
        public void ChangeTypeTypeCode3Arg_CharToBoolean_Works()
        {
            Converts.ChangeType((object)'A', TypeCode.Boolean, System.Globalization.CultureInfo.InvariantCulture).Should().Be(true);
        }

        [TestMethod]
        public void ChangeTypeTypeCode_NullToString_ReturnsNull()
        {
            // Existing contract: null + String/Empty/Object → null
            Converts.ChangeType(null, TypeCode.String).Should().BeNull();
        }

        [TestMethod]
        public void ChangeTypeTypeCode_Int32ToString_Works()
        {
            // Regression check: classic path still works
            Converts.ChangeType((object)42, TypeCode.String).Should().Be("42");
        }

        [TestMethod]
        public void ChangeTypeTypeCode_DoubleToInt32_Truncates()
        {
            // Regression check: classic numeric path still works with NumPy-parity truncation
            Converts.ChangeType((object)3.7, TypeCode.Int32).Should().Be(3);
        }

        #endregion

        #region Round 5A: ArraySlice.Allocate(*,fill) + np.searchsorted Half/Complex

        // H1: ArraySlice.Allocate(NPTypeCode, count, fill) used IConvertible cast on fill;
        // throws when fill is Half or Complex (neither implements IConvertible).
        // H2: ArraySlice.Allocate(Type, count, fill) had identical bug.

        [TestMethod]
        public void ArraySliceAllocate_NPTypeCode_Int32_FillHalf_Works()
        {
            var slice = NumSharp.Backends.Unmanaged.ArraySlice.Allocate(NPTypeCode.Int32, 3, Half.One);
            ((int[])slice.ToArray()).Should().Equal(1, 1, 1);
        }

        [TestMethod]
        public void ArraySliceAllocate_NPTypeCode_Double_FillHalf_Works()
        {
            var slice = NumSharp.Backends.Unmanaged.ArraySlice.Allocate(NPTypeCode.Double, 3, (Half)3.5f);
            var arr = (double[])slice.ToArray();
            arr[0].Should().BeApproximately(3.5, 0.001);
        }

        [TestMethod]
        public void ArraySliceAllocate_NPTypeCode_Int32_FillComplex_DiscardsImaginary()
        {
            var slice = NumSharp.Backends.Unmanaged.ArraySlice.Allocate(NPTypeCode.Int32, 3, new Complex(7, 9));
            ((int[])slice.ToArray()).Should().Equal(7, 7, 7);
        }

        [TestMethod]
        public void ArraySliceAllocate_NPTypeCode_Half_FillComplex_DiscardsImaginary()
        {
            var slice = NumSharp.Backends.Unmanaged.ArraySlice.Allocate(NPTypeCode.Half, 3, new Complex(3.5, 4));
            var arr = (Half[])slice.ToArray();
            ((float)arr[0]).Should().BeApproximately(3.5f, 0.01f);
        }

        [TestMethod]
        public void ArraySliceAllocate_NPTypeCode_Complex_FillHalf_Works()
        {
            var slice = NumSharp.Backends.Unmanaged.ArraySlice.Allocate(NPTypeCode.Complex, 3, Half.One);
            var arr = (Complex[])slice.ToArray();
            arr[0].Should().Be(new Complex(1, 0));
        }

        [TestMethod]
        public void ArraySliceAllocate_NPTypeCode_Bool_FillComplex_NonZero()
        {
            var slice = NumSharp.Backends.Unmanaged.ArraySlice.Allocate(NPTypeCode.Boolean, 2, new Complex(0, 1));
            ((bool[])slice.ToArray()).Should().Equal(true, true);
        }

        [TestMethod]
        public void ArraySliceAllocate_NPTypeCode_Char_FillHalf_Works()
        {
            var slice = NumSharp.Backends.Unmanaged.ArraySlice.Allocate(NPTypeCode.Char, 2, (Half)65);
            ((char[])slice.ToArray()).Should().Equal('A', 'A');
        }

        [TestMethod]
        public void ArraySliceAllocate_Type_Int32_FillHalf_Works()
        {
            // Type-based overload of Allocate
            var slice = NumSharp.Backends.Unmanaged.ArraySlice.Allocate(typeof(int), 3, Half.One);
            ((int[])slice.ToArray()).Should().Equal(1, 1, 1);
        }

        [TestMethod]
        public void ArraySliceAllocate_Type_Half_FillComplex_DiscardsImaginary()
        {
            var slice = NumSharp.Backends.Unmanaged.ArraySlice.Allocate(typeof(Half), 3, new Complex(3.5, 4));
            var arr = (Half[])slice.ToArray();
            ((float)arr[0]).Should().BeApproximately(3.5f, 0.01f);
        }

        [TestMethod]
        public void ArraySliceAllocate_Type_Complex_FillHalf_Works()
        {
            var slice = NumSharp.Backends.Unmanaged.ArraySlice.Allocate(typeof(Complex), 3, Half.One);
            var arr = (Complex[])slice.ToArray();
            arr[0].Should().Be(new Complex(1, 0));
        }

        // Regression: classic IConvertible source still works
        [TestMethod]
        public void ArraySliceAllocate_NPTypeCode_Int32_FillInt_Works()
        {
            var slice = NumSharp.Backends.Unmanaged.ArraySlice.Allocate(NPTypeCode.Int32, 2, 42);
            ((int[])slice.ToArray()).Should().Equal(42, 42);
        }

        // H3: np.searchsorted used Convert.ToDouble on boxed array values.
        // Throws when the source array dtype is Half or Complex.

        [TestMethod]
        public void Searchsorted_HalfArray_FindsPosition()
        {
            var arr = np.array(new[] { (Half)1, (Half)3, (Half)5, (Half)7 });
            var idx = np.searchsorted(arr, np.asarray((Half)4));
            idx.GetAtIndex<long>(0).Should().Be(2);
        }

        [TestMethod]
        public void Searchsorted_HalfArray_DoubleValue_FindsPosition()
        {
            var arr = np.array(new[] { (Half)1, (Half)3, (Half)5, (Half)7 });
            var idx = np.searchsorted(arr, np.asarray(2.5));
            idx.GetAtIndex<long>(0).Should().Be(1);
        }

        [TestMethod]
        public void Searchsorted_ComplexArray_FindsPosition()
        {
            // Complex compared by real part (NumPy semantics — emits warning in NumPy)
            var arr = np.array(new[] { new Complex(1, 0), new Complex(3, 0), new Complex(5, 0), new Complex(7, 0) });
            var idx = np.searchsorted(arr, np.asarray(new Complex(4, 0)));
            idx.GetAtIndex<long>(0).Should().Be(2);
        }

        [TestMethod]
        public void Searchsorted_HalfArray_MultipleValues_Works()
        {
            var arr = np.array(new[] { (Half)1, (Half)3, (Half)5, (Half)7 });
            var values = np.array(new[] { (Half)0, (Half)4, (Half)8 });
            var idx = np.searchsorted(arr, values);
            idx.GetAtIndex<long>(0).Should().Be(0);
            idx.GetAtIndex<long>(1).Should().Be(2);
            idx.GetAtIndex<long>(2).Should().Be(4);
        }

        // Regression: classic dtype still works
        [TestMethod]
        public void Searchsorted_DoubleArray_FindsPosition()
        {
            var arr = np.array(new[] { 1.0, 3.0, 5.0, 7.0 });
            var idx = np.searchsorted(arr, np.asarray(4.0));
            idx.GetAtIndex<long>(0).Should().Be(2);
        }

        #endregion
    }
}
