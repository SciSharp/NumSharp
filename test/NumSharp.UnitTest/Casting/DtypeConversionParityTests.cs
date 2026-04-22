using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Utilities;

namespace NumSharp.UnitTest.Casting
{
    /// <summary>
    /// Comprehensive tests for NumPy-compatible dtype conversion behavior.
    /// All expected values are verified against NumPy 2.x output.
    /// </summary>
    [TestClass]
    public class DtypeConversionParityTests
    {
        #region Float to Integer - Truncation Toward Zero

        [TestMethod]
        public void Float_ToInt_TruncatesTowardZero_Positive()
        {
            // NumPy: np.array([3.7]).astype(np.int32) -> array([3])
            Converts.ToInt32(3.7).Should().Be(3);
            Converts.ToInt32(3.9).Should().Be(3);
            Converts.ToInt32(3.1).Should().Be(3);
            Converts.ToInt32(0.9).Should().Be(0);
            Converts.ToInt32(0.1).Should().Be(0);
        }

        [TestMethod]
        public void Float_ToInt_TruncatesTowardZero_Negative()
        {
            // NumPy: np.array([-3.7]).astype(np.int32) -> array([-3])
            // Truncation toward zero, NOT floor
            Converts.ToInt32(-3.7).Should().Be(-3);
            Converts.ToInt32(-3.9).Should().Be(-3);
            Converts.ToInt32(-3.1).Should().Be(-3);
            Converts.ToInt32(-0.9).Should().Be(0);
            Converts.ToInt32(-0.1).Should().Be(0);
        }

        [TestMethod]
        public void Float_ToInt64_TruncatesTowardZero()
        {
            Converts.ToInt64(3.7).Should().Be(3L);
            Converts.ToInt64(-3.7).Should().Be(-3L);
            Converts.ToInt64(0.9).Should().Be(0L);
            Converts.ToInt64(-0.9).Should().Be(0L);
        }

        [TestMethod]
        public void Float_ToSmallInt_TruncatesTowardZero()
        {
            // int8
            Converts.ToSByte(3.7).Should().Be((sbyte)3);
            Converts.ToSByte(-3.7).Should().Be((sbyte)-3);

            // uint8
            Converts.ToByte(3.7).Should().Be((byte)3);

            // int16
            Converts.ToInt16(3.7).Should().Be((short)3);
            Converts.ToInt16(-3.7).Should().Be((short)-3);

            // uint16
            Converts.ToUInt16(3.7).Should().Be((ushort)3);
        }

        #endregion

        #region Negative Float to Unsigned - Truncate Then Wrap

        [TestMethod]
        public void NegativeFloat_ToUInt8_TruncatesThenWraps()
        {
            // NumPy: np.array([-1.0]).astype(np.uint8) -> array([255])
            // -1.0 truncates to -1, then wraps to 255
            Converts.ToByte(-1.0).Should().Be(255);

            // -3.7 truncates to -3, then wraps to 253
            Converts.ToByte(-3.7).Should().Be(253);

            // -128 truncates to -128, wraps to 128
            Converts.ToByte(-128.0).Should().Be(128);

            // -129 truncates to -129, wraps to 127
            Converts.ToByte(-129.0).Should().Be(127);
        }

        [TestMethod]
        public void NegativeFloat_ToUInt16_TruncatesThenWraps()
        {
            // NumPy: np.array([-1.0]).astype(np.uint16) -> array([65535])
            Converts.ToUInt16(-1.0).Should().Be(65535);
            Converts.ToUInt16(-3.7).Should().Be(65533);
            Converts.ToUInt16(-32768.0).Should().Be(32768);
            Converts.ToUInt16(-32769.0).Should().Be(32767);
        }

        [TestMethod]
        public void NegativeFloat_ToUInt32_TruncatesThenWraps()
        {
            // NumPy: np.array([-1.0]).astype(np.uint32) -> array([4294967295])
            Converts.ToUInt32(-1.0).Should().Be(4294967295u);
            Converts.ToUInt32(-3.7).Should().Be(4294967293u);
        }

        [TestMethod]
        public void NegativeFloat_ToUInt64_TruncatesThenWraps()
        {
            // NumPy: np.array([-1.0]).astype(np.uint64) -> array([18446744073709551615])
            Converts.ToUInt64(-1.0).Should().Be(18446744073709551615UL);
            Converts.ToUInt64(-3.7).Should().Be(18446744073709551613UL);
        }

        #endregion

        #region Positive Float Overflow - Wrapping for Small Types

        [TestMethod]
        public void PositiveFloat_ToUInt8_Wraps()
        {
            // NumPy: wraps modulo 256
            Converts.ToByte(256.0).Should().Be(0);   // 256 % 256 = 0
            Converts.ToByte(257.0).Should().Be(1);   // 257 % 256 = 1
            Converts.ToByte(1000.0).Should().Be(232); // 1000 % 256 = 232
            Converts.ToByte(512.0).Should().Be(0);   // 512 % 256 = 0
        }

        [TestMethod]
        public void PositiveFloat_ToInt8_Wraps()
        {
            // NumPy: wraps with signed interpretation
            Converts.ToSByte(128.0).Should().Be(-128);
            Converts.ToSByte(255.0).Should().Be(-1);
            Converts.ToSByte(256.0).Should().Be(0);
            Converts.ToSByte(257.0).Should().Be(1);
        }

        [TestMethod]
        public void PositiveFloat_ToUInt16_Wraps()
        {
            Converts.ToUInt16(65536.0).Should().Be(0);
            Converts.ToUInt16(65537.0).Should().Be(1);
        }

        [TestMethod]
        public void PositiveFloat_ToInt16_Wraps()
        {
            Converts.ToInt16(32768.0).Should().Be(-32768);
            Converts.ToInt16(65535.0).Should().Be(-1);
            Converts.ToInt16(65536.0).Should().Be(0);
        }

        #endregion

        #region Float Outside Int32 Range - Returns 0 for Small Types

        [TestMethod]
        public void FloatOutsideInt32Range_ToSmallTypes_ReturnsZero()
        {
            // NumPy uses int32 as intermediate for small type conversions
            // Values outside int32 range overflow to 0

            // 2147483648 = int32.MaxValue + 1
            Converts.ToSByte(2147483648.0).Should().Be(0);
            Converts.ToByte(2147483648.0).Should().Be(0);
            Converts.ToInt16(2147483648.0).Should().Be(0);
            Converts.ToUInt16(2147483648.0).Should().Be(0);

            // 4294967295 = uint32.MaxValue (outside int32 range)
            Converts.ToSByte(4294967295.0).Should().Be(0);
            Converts.ToByte(4294967295.0).Should().Be(0);
            Converts.ToInt16(4294967295.0).Should().Be(0);
            Converts.ToUInt16(4294967295.0).Should().Be(0);

            // -2147483649 = int32.MinValue - 1
            Converts.ToSByte(-2147483649.0).Should().Be(0);
            Converts.ToByte(-2147483649.0).Should().Be(0);
            Converts.ToInt16(-2147483649.0).Should().Be(0);
            Converts.ToUInt16(-2147483649.0).Should().Be(0);
        }

        [TestMethod]
        public void FloatAtInt32Boundary_StillWraps()
        {
            // 2147483647 = int32.MaxValue (within range, should wrap)
            Converts.ToSByte(2147483647.0).Should().Be(-1);
            Converts.ToByte(2147483647.0).Should().Be(255);
            Converts.ToInt16(2147483647.0).Should().Be(-1);
            Converts.ToUInt16(2147483647.0).Should().Be(65535);
        }

        #endregion

        #region NaN and Infinity to Integer

        [TestMethod]
        public void NaN_ToSmallInt_ReturnsZero()
        {
            // NumPy: NaN -> 0 for int8, uint8, int16, uint16
            Converts.ToSByte(double.NaN).Should().Be(0);
            Converts.ToByte(double.NaN).Should().Be(0);
            Converts.ToInt16(double.NaN).Should().Be(0);
            Converts.ToUInt16(double.NaN).Should().Be(0);
        }

        [TestMethod]
        public void NaN_ToInt32_ReturnsMinValue()
        {
            // NumPy: np.array([np.nan]).astype(np.int32) -> array([-2147483648])
            Converts.ToInt32(double.NaN).Should().Be(int.MinValue);
        }

        [TestMethod]
        public void NaN_ToInt64_ReturnsMinValue()
        {
            // NumPy: np.array([np.nan]).astype(np.int64) -> array([-9223372036854775808])
            Converts.ToInt64(double.NaN).Should().Be(long.MinValue);
        }

        [TestMethod]
        public void NaN_ToUInt32_ReturnsZero()
        {
            // NumPy: np.array([np.nan]).astype(np.uint32) -> array([0])
            Converts.ToUInt32(double.NaN).Should().Be(0u);
        }

        [TestMethod]
        public void NaN_ToUInt64_Returns2Power63()
        {
            // NumPy: np.array([np.nan]).astype(np.uint64) -> array([9223372036854775808])
            Converts.ToUInt64(double.NaN).Should().Be(9223372036854775808UL);
        }

        [TestMethod]
        public void PositiveInfinity_ToInt_SameBehaviorAsNaN()
        {
            Converts.ToSByte(double.PositiveInfinity).Should().Be(0);
            Converts.ToByte(double.PositiveInfinity).Should().Be(0);
            Converts.ToInt16(double.PositiveInfinity).Should().Be(0);
            Converts.ToUInt16(double.PositiveInfinity).Should().Be(0);
            Converts.ToInt32(double.PositiveInfinity).Should().Be(int.MinValue);
            Converts.ToUInt32(double.PositiveInfinity).Should().Be(0u);
            Converts.ToInt64(double.PositiveInfinity).Should().Be(long.MinValue);
            Converts.ToUInt64(double.PositiveInfinity).Should().Be(9223372036854775808UL);
        }

        [TestMethod]
        public void NegativeInfinity_ToInt_SameBehaviorAsNaN()
        {
            Converts.ToSByte(double.NegativeInfinity).Should().Be(0);
            Converts.ToByte(double.NegativeInfinity).Should().Be(0);
            Converts.ToInt16(double.NegativeInfinity).Should().Be(0);
            Converts.ToUInt16(double.NegativeInfinity).Should().Be(0);
            Converts.ToInt32(double.NegativeInfinity).Should().Be(int.MinValue);
            Converts.ToUInt32(double.NegativeInfinity).Should().Be(0u);
            Converts.ToInt64(double.NegativeInfinity).Should().Be(long.MinValue);
            Converts.ToUInt64(double.NegativeInfinity).Should().Be(9223372036854775808UL);
        }

        #endregion

        #region Half (Float16) Conversions

        [TestMethod]
        public void Half_ToInt_TruncatesTowardZero()
        {
            Converts.ToInt32((Half)3.7).Should().Be(3);
            Converts.ToInt32((Half)(-3.7)).Should().Be(-3);
            Converts.ToSByte((Half)3.7).Should().Be((sbyte)3);
            Converts.ToSByte((Half)(-3.7)).Should().Be((sbyte)-3);
        }

        [TestMethod]
        public void Half_NaN_ToInt_MatchesDoubleNaN()
        {
            Converts.ToSByte(Half.NaN).Should().Be(0);
            Converts.ToByte(Half.NaN).Should().Be(0);
            Converts.ToInt16(Half.NaN).Should().Be(0);
            Converts.ToUInt16(Half.NaN).Should().Be(0);
            Converts.ToInt32(Half.NaN).Should().Be(int.MinValue);
            Converts.ToInt64(Half.NaN).Should().Be(long.MinValue);
            Converts.ToUInt64(Half.NaN).Should().Be(9223372036854775808UL);
        }

        [TestMethod]
        public void Half_Infinity_ToInt_MatchesDoubleInfinity()
        {
            Converts.ToSByte(Half.PositiveInfinity).Should().Be(0);
            Converts.ToByte(Half.PositiveInfinity).Should().Be(0);
            Converts.ToInt32(Half.PositiveInfinity).Should().Be(int.MinValue);
            Converts.ToInt64(Half.PositiveInfinity).Should().Be(long.MinValue);
        }

        [TestMethod]
        public void Half_NegativeToUnsigned_TruncatesThenWraps()
        {
            Converts.ToByte((Half)(-1)).Should().Be(255);
            Converts.ToByte((Half)(-3.7)).Should().Be(253);
            Converts.ToUInt16((Half)(-1)).Should().Be(65535);
            Converts.ToUInt64((Half)(-1)).Should().Be(18446744073709551615UL);
        }

        #endregion

        #region Integer to Integer - Wrapping

        [TestMethod]
        public void SignedInt_ToUnsigned_Wraps()
        {
            // Bit reinterpretation
            Converts.ToByte((sbyte)-1).Should().Be(255);
            Converts.ToByte((sbyte)-128).Should().Be(128);
            Converts.ToUInt16((short)-1).Should().Be(65535);
            Converts.ToUInt16((short)-32768).Should().Be(32768);
            Converts.ToUInt32(-1).Should().Be(4294967295u);
            Converts.ToUInt64(-1L).Should().Be(18446744073709551615UL);
        }

        [TestMethod]
        public void UnsignedInt_ToSigned_Wraps()
        {
            // Bit reinterpretation
            Converts.ToSByte((byte)255).Should().Be(-1);
            Converts.ToSByte((byte)128).Should().Be(-128);
            Converts.ToInt16((ushort)65535).Should().Be(-1);
            Converts.ToInt16((ushort)32768).Should().Be(-32768);
            Converts.ToInt32(4294967295u).Should().Be(-1);
        }

        [TestMethod]
        public void WiderInt_ToNarrower_Truncates()
        {
            // Keep low bits only
            Converts.ToByte((short)256).Should().Be(0);
            Converts.ToByte((short)257).Should().Be(1);
            Converts.ToByte((short)1000).Should().Be(232);
            Converts.ToSByte((short)256).Should().Be(0);
            Converts.ToSByte((short)128).Should().Be(-128);
            Converts.ToInt16(65536).Should().Be(0);
            Converts.ToInt16(65537).Should().Be(1);
            Converts.ToUInt16(65536).Should().Be(0);
        }

        [TestMethod]
        public void LongNegative_ToUnsigned_Wraps()
        {
            Converts.ToByte(-1L).Should().Be(255);
            Converts.ToByte(-128L).Should().Be(128);
            Converts.ToUInt16(-1L).Should().Be(65535);
            Converts.ToUInt32(-1L).Should().Be(4294967295u);
        }

        #endregion

        #region Bool Conversions

        [TestMethod]
        public void Zero_ToBool_ReturnsFalse()
        {
            Converts.ToBoolean(0).Should().BeFalse();
            Converts.ToBoolean(0L).Should().BeFalse();
            Converts.ToBoolean(0.0).Should().BeFalse();
            Converts.ToBoolean(0.0f).Should().BeFalse();
            Converts.ToBoolean((Half)0).Should().BeFalse();
        }

        [TestMethod]
        public void NonZero_ToBool_ReturnsTrue()
        {
            Converts.ToBoolean(1).Should().BeTrue();
            Converts.ToBoolean(-1).Should().BeTrue();
            Converts.ToBoolean(42).Should().BeTrue();
            Converts.ToBoolean(0.5).Should().BeTrue();
            Converts.ToBoolean(-0.5).Should().BeTrue();
        }

        [TestMethod]
        public void NaN_ToBool_ReturnsTrue()
        {
            // NumPy: np.array([np.nan]).astype(bool) -> array([True])
            Converts.ToBoolean(double.NaN).Should().BeTrue();
            Converts.ToBoolean(float.NaN).Should().BeTrue();
            Converts.ToBoolean(Half.NaN).Should().BeTrue();
        }

        [TestMethod]
        public void Infinity_ToBool_ReturnsTrue()
        {
            Converts.ToBoolean(double.PositiveInfinity).Should().BeTrue();
            Converts.ToBoolean(double.NegativeInfinity).Should().BeTrue();
            Converts.ToBoolean(float.PositiveInfinity).Should().BeTrue();
            Converts.ToBoolean(Half.PositiveInfinity).Should().BeTrue();
        }

        [TestMethod]
        public void Bool_ToNumeric_ZeroOrOne()
        {
            Converts.ToByte(false).Should().Be(0);
            Converts.ToByte(true).Should().Be(1);
            Converts.ToInt32(false).Should().Be(0);
            Converts.ToInt32(true).Should().Be(1);
            Converts.ToDouble(false).Should().Be(0.0);
            Converts.ToDouble(true).Should().Be(1.0);
        }

        #endregion

        #region NDArray.astype() Integration

        [TestMethod]
        public void Astype_Float64ToInt32_Truncates()
        {
            var arr = np.array(new double[] { 3.7, -3.7, 0.9, -0.9 });
            var result = arr.astype(np.int32);

            result.GetAtIndex<int>(0).Should().Be(3);
            result.GetAtIndex<int>(1).Should().Be(-3);
            result.GetAtIndex<int>(2).Should().Be(0);
            result.GetAtIndex<int>(3).Should().Be(0);
        }

        [TestMethod]
        public void Astype_Float64ToUInt8_NegativeWraps()
        {
            var arr = np.array(new double[] { -1.0, -3.7, 256.0, 1000.0 });
            var result = arr.astype(np.uint8);

            result.GetAtIndex<byte>(0).Should().Be(255);
            result.GetAtIndex<byte>(1).Should().Be(253);
            result.GetAtIndex<byte>(2).Should().Be(0);
            result.GetAtIndex<byte>(3).Should().Be(232);
        }

        [TestMethod]
        public void Astype_Float64WithNaN_ToInt32_ReturnsMinValue()
        {
            var arr = np.array(new double[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity });
            var result = arr.astype(np.int32);

            result.GetAtIndex<int>(0).Should().Be(int.MinValue);
            result.GetAtIndex<int>(1).Should().Be(int.MinValue);
            result.GetAtIndex<int>(2).Should().Be(int.MinValue);
        }

        [TestMethod]
        public void Astype_Float64WithNaN_ToUInt64_Returns2Power63()
        {
            var arr = np.array(new double[] { double.NaN, double.PositiveInfinity });
            var result = arr.astype(np.uint64);

            result.GetAtIndex<ulong>(0).Should().Be(9223372036854775808UL);
            result.GetAtIndex<ulong>(1).Should().Be(9223372036854775808UL);
        }

        [TestMethod]
        public void Astype_Int32ToUInt32_NegativeWraps()
        {
            var arr = np.array(new int[] { -1, -128, 0, 127 });
            var result = arr.astype(np.uint32);

            result.GetAtIndex<uint>(0).Should().Be(4294967295u);
            result.GetAtIndex<uint>(1).Should().Be(4294967168u);
            result.GetAtIndex<uint>(2).Should().Be(0u);
            result.GetAtIndex<uint>(3).Should().Be(127u);
        }

        [TestMethod]
        public void Astype_Float64ToBool_ZeroIsFalseElseTrue()
        {
            var arr = np.array(new double[] { 0.0, 1.0, -1.0, 0.5, double.NaN, double.PositiveInfinity });
            var result = arr.astype(np.@bool);

            result.GetAtIndex<bool>(0).Should().BeFalse();
            result.GetAtIndex<bool>(1).Should().BeTrue();
            result.GetAtIndex<bool>(2).Should().BeTrue();
            result.GetAtIndex<bool>(3).Should().BeTrue();
            result.GetAtIndex<bool>(4).Should().BeTrue(); // NaN is True
            result.GetAtIndex<bool>(5).Should().BeTrue(); // Inf is True
        }

        [TestMethod]
        public void Astype_LargeFloatToSmallInt_OutsideInt32Range_ReturnsZero()
        {
            // Values outside int32 range return 0 for small types
            var arr = np.array(new double[] { 2147483648.0, 4294967295.0, -2147483649.0 });

            var int8Result = arr.astype(NPTypeCode.SByte);
            int8Result.GetAtIndex<sbyte>(0).Should().Be(0);
            int8Result.GetAtIndex<sbyte>(1).Should().Be(0);
            int8Result.GetAtIndex<sbyte>(2).Should().Be(0);

            var uint8Result = arr.astype(np.uint8);
            uint8Result.GetAtIndex<byte>(0).Should().Be(0);
            uint8Result.GetAtIndex<byte>(1).Should().Be(0);
            uint8Result.GetAtIndex<byte>(2).Should().Be(0);
        }

        #endregion

        #region Complex Number Conversions

        [TestMethod]
        public void Complex_ToReal_DiscardsImaginary()
        {
            var c = new Complex(3.7, 4.2);
            Converts.ToDouble(c).Should().Be(3.7);
            Converts.ToInt32(c).Should().Be(3); // Truncates real part

            var cPureImag = new Complex(0, 5.0);
            Converts.ToDouble(cPureImag).Should().Be(0.0);
            Converts.ToInt32(cPureImag).Should().Be(0);
        }

        [TestMethod]
        public void Complex_ToBool_NonZeroIsTrue()
        {
            Converts.ToBoolean(new Complex(0, 0)).Should().BeFalse();
            Converts.ToBoolean(new Complex(1, 0)).Should().BeTrue();
            Converts.ToBoolean(new Complex(0, 1)).Should().BeTrue(); // Pure imaginary is nonzero
            Converts.ToBoolean(new Complex(3, 4)).Should().BeTrue();
        }

        #endregion
    }
}
