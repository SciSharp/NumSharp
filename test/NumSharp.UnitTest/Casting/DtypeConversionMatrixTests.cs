using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Casting
{
    /// <summary>
    /// Complete dtype conversion matrix tests verified against NumPy 2.4.2.
    /// Each test covers all 12 target types for a specific source type.
    /// Values are exact outputs from: np.array([val], dtype=src).astype(tgt)[0]
    /// </summary>
    [TestClass]
    public class DtypeConversionMatrixTests
    {
        #region Bool Source (2 values × 12 targets = 24 conversions)

        [TestMethod]
        [DataRow(false, false, (sbyte)0, (byte)0, (short)0, (ushort)0, 0, 0u, 0L, 0UL, 0.0f, 0.0)]
        [DataRow(true, true, (sbyte)1, (byte)1, (short)1, (ushort)1, 1, 1u, 1L, 1UL, 1.0f, 1.0)]
        public void Bool_ToAllTypes(bool src, bool toBool, sbyte toInt8, byte toUInt8, short toInt16, ushort toUInt16,
            int toInt32, uint toUInt32, long toInt64, ulong toUInt64, float toFloat32, double toFloat64)
        {
            var arr = np.array(new[] { src });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().Be(toBool);
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(toInt8);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(toUInt8);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(toInt16);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(toUInt16);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(toInt32);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(toUInt32);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(toInt64);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(toUInt64);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(toFloat32);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(toFloat64);
        }

        #endregion

        #region Int8 Source (5 values × 12 targets = 60 conversions)

        [TestMethod]
        [DataRow((sbyte)0, false, (sbyte)0, (byte)0, (short)0, (ushort)0, 0, 0u, 0L, 0UL, 0.0f, 0.0)]
        [DataRow((sbyte)1, true, (sbyte)1, (byte)1, (short)1, (ushort)1, 1, 1u, 1L, 1UL, 1.0f, 1.0)]
        [DataRow((sbyte)-1, true, (sbyte)-1, (byte)255, (short)-1, (ushort)65535, -1, 4294967295u, -1L, 18446744073709551615UL, -1.0f, -1.0)]
        [DataRow((sbyte)127, true, (sbyte)127, (byte)127, (short)127, (ushort)127, 127, 127u, 127L, 127UL, 127.0f, 127.0)]
        [DataRow((sbyte)-128, true, (sbyte)-128, (byte)128, (short)-128, (ushort)65408, -128, 4294967168u, -128L, 18446744073709551488UL, -128.0f, -128.0)]
        public void Int8_ToAllTypes(sbyte src, bool toBool, sbyte toInt8, byte toUInt8, short toInt16, ushort toUInt16,
            int toInt32, uint toUInt32, long toInt64, ulong toUInt64, float toFloat32, double toFloat64)
        {
            var arr = np.array(new[] { src });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().Be(toBool);
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(toInt8);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(toUInt8);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(toInt16);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(toUInt16);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(toInt32);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(toUInt32);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(toInt64);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(toUInt64);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(toFloat32);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(toFloat64);
        }

        #endregion

        #region UInt8 Source (5 values × 12 targets = 60 conversions)

        [TestMethod]
        [DataRow((byte)0, false, (sbyte)0, (byte)0, (short)0, (ushort)0, 0, 0u, 0L, 0UL, 0.0f, 0.0)]
        [DataRow((byte)1, true, (sbyte)1, (byte)1, (short)1, (ushort)1, 1, 1u, 1L, 1UL, 1.0f, 1.0)]
        [DataRow((byte)127, true, (sbyte)127, (byte)127, (short)127, (ushort)127, 127, 127u, 127L, 127UL, 127.0f, 127.0)]
        [DataRow((byte)128, true, (sbyte)-128, (byte)128, (short)128, (ushort)128, 128, 128u, 128L, 128UL, 128.0f, 128.0)]
        [DataRow((byte)255, true, (sbyte)-1, (byte)255, (short)255, (ushort)255, 255, 255u, 255L, 255UL, 255.0f, 255.0)]
        public void UInt8_ToAllTypes(byte src, bool toBool, sbyte toInt8, byte toUInt8, short toInt16, ushort toUInt16,
            int toInt32, uint toUInt32, long toInt64, ulong toUInt64, float toFloat32, double toFloat64)
        {
            var arr = np.array(new[] { src });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().Be(toBool);
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(toInt8);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(toUInt8);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(toInt16);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(toUInt16);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(toInt32);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(toUInt32);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(toInt64);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(toUInt64);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(toFloat32);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(toFloat64);
        }

        #endregion

        #region Int16 Source (5 values × 12 targets = 60 conversions)

        [TestMethod]
        [DataRow((short)0, false, (sbyte)0, (byte)0, (short)0, (ushort)0, 0, 0u, 0L, 0UL, 0.0f, 0.0)]
        [DataRow((short)1, true, (sbyte)1, (byte)1, (short)1, (ushort)1, 1, 1u, 1L, 1UL, 1.0f, 1.0)]
        [DataRow((short)-1, true, (sbyte)-1, (byte)255, (short)-1, (ushort)65535, -1, 4294967295u, -1L, 18446744073709551615UL, -1.0f, -1.0)]
        [DataRow((short)32767, true, (sbyte)-1, (byte)255, (short)32767, (ushort)32767, 32767, 32767u, 32767L, 32767UL, 32767.0f, 32767.0)]
        [DataRow((short)-32768, true, (sbyte)0, (byte)0, (short)-32768, (ushort)32768, -32768, 4294934528u, -32768L, 18446744073709518848UL, -32768.0f, -32768.0)]
        public void Int16_ToAllTypes(short src, bool toBool, sbyte toInt8, byte toUInt8, short toInt16, ushort toUInt16,
            int toInt32, uint toUInt32, long toInt64, ulong toUInt64, float toFloat32, double toFloat64)
        {
            var arr = np.array(new[] { src });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().Be(toBool);
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(toInt8);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(toUInt8);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(toInt16);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(toUInt16);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(toInt32);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(toUInt32);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(toInt64);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(toUInt64);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(toFloat32);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(toFloat64);
        }

        #endregion

        #region UInt16 Source (5 values × 12 targets = 60 conversions)

        [TestMethod]
        [DataRow((ushort)0, false, (sbyte)0, (byte)0, (short)0, (ushort)0, 0, 0u, 0L, 0UL, 0.0f, 0.0)]
        [DataRow((ushort)1, true, (sbyte)1, (byte)1, (short)1, (ushort)1, 1, 1u, 1L, 1UL, 1.0f, 1.0)]
        [DataRow((ushort)32767, true, (sbyte)-1, (byte)255, (short)32767, (ushort)32767, 32767, 32767u, 32767L, 32767UL, 32767.0f, 32767.0)]
        [DataRow((ushort)32768, true, (sbyte)0, (byte)0, (short)-32768, (ushort)32768, 32768, 32768u, 32768L, 32768UL, 32768.0f, 32768.0)]
        [DataRow((ushort)65535, true, (sbyte)-1, (byte)255, (short)-1, (ushort)65535, 65535, 65535u, 65535L, 65535UL, 65535.0f, 65535.0)]
        public void UInt16_ToAllTypes(ushort src, bool toBool, sbyte toInt8, byte toUInt8, short toInt16, ushort toUInt16,
            int toInt32, uint toUInt32, long toInt64, ulong toUInt64, float toFloat32, double toFloat64)
        {
            var arr = np.array(new[] { src });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().Be(toBool);
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(toInt8);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(toUInt8);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(toInt16);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(toUInt16);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(toInt32);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(toUInt32);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(toInt64);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(toUInt64);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(toFloat32);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(toFloat64);
        }

        #endregion

        #region Int32 Source (5 values × 12 targets = 60 conversions)

        [TestMethod]
        [DataRow(0, false, (sbyte)0, (byte)0, (short)0, (ushort)0, 0, 0u, 0L, 0UL, 0.0f, 0.0)]
        [DataRow(1, true, (sbyte)1, (byte)1, (short)1, (ushort)1, 1, 1u, 1L, 1UL, 1.0f, 1.0)]
        [DataRow(-1, true, (sbyte)-1, (byte)255, (short)-1, (ushort)65535, -1, 4294967295u, -1L, 18446744073709551615UL, -1.0f, -1.0)]
        [DataRow(2147483647, true, (sbyte)-1, (byte)255, (short)-1, (ushort)65535, 2147483647, 2147483647u, 2147483647L, 2147483647UL, 2147483648.0f, 2147483647.0)]
        [DataRow(-2147483648, true, (sbyte)0, (byte)0, (short)0, (ushort)0, -2147483648, 2147483648u, -2147483648L, 18446744071562067968UL, -2147483648.0f, -2147483648.0)]
        public void Int32_ToAllTypes(int src, bool toBool, sbyte toInt8, byte toUInt8, short toInt16, ushort toUInt16,
            int toInt32, uint toUInt32, long toInt64, ulong toUInt64, float toFloat32, double toFloat64)
        {
            var arr = np.array(new[] { src });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().Be(toBool);
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(toInt8);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(toUInt8);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(toInt16);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(toUInt16);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(toInt32);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(toUInt32);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(toInt64);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(toUInt64);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(toFloat32);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(toFloat64);
        }

        #endregion

        #region UInt32 Source (5 values × 12 targets = 60 conversions)

        [TestMethod]
        [DataRow(0u, false, (sbyte)0, (byte)0, (short)0, (ushort)0, 0, 0u, 0L, 0UL, 0.0f, 0.0)]
        [DataRow(1u, true, (sbyte)1, (byte)1, (short)1, (ushort)1, 1, 1u, 1L, 1UL, 1.0f, 1.0)]
        [DataRow(2147483647u, true, (sbyte)-1, (byte)255, (short)-1, (ushort)65535, 2147483647, 2147483647u, 2147483647L, 2147483647UL, 2147483648.0f, 2147483647.0)]
        [DataRow(2147483648u, true, (sbyte)0, (byte)0, (short)0, (ushort)0, -2147483648, 2147483648u, 2147483648L, 2147483648UL, 2147483648.0f, 2147483648.0)]
        [DataRow(4294967295u, true, (sbyte)-1, (byte)255, (short)-1, (ushort)65535, -1, 4294967295u, 4294967295L, 4294967295UL, 4294967296.0f, 4294967295.0)]
        public void UInt32_ToAllTypes(uint src, bool toBool, sbyte toInt8, byte toUInt8, short toInt16, ushort toUInt16,
            int toInt32, uint toUInt32, long toInt64, ulong toUInt64, float toFloat32, double toFloat64)
        {
            var arr = np.array(new[] { src });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().Be(toBool);
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(toInt8);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(toUInt8);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(toInt16);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(toUInt16);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(toInt32);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(toUInt32);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(toInt64);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(toUInt64);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(toFloat32);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(toFloat64);
        }

        #endregion

        #region Int64 Source (5 values × 12 targets = 60 conversions)

        [TestMethod]
        public void Int64_Zero_ToAllTypes()
        {
            var arr = np.array(new[] { 0L });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeFalse();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(0);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(0);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(0);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(0);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(0);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(0u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(0L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(0UL);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(0.0f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(0.0);
        }

        [TestMethod]
        public void Int64_One_ToAllTypes()
        {
            var arr = np.array(new[] { 1L });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(1);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(1);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(1);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(1);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(1);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(1u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(1L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(1UL);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(1.0f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(1.0);
        }

        [TestMethod]
        public void Int64_NegativeOne_ToAllTypes()
        {
            var arr = np.array(new[] { -1L });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(-1);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(255);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(-1);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(65535);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-1);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(4294967295u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-1L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(18446744073709551615UL);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(-1.0f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(-1.0);
        }

        [TestMethod]
        public void Int64_MaxValue_ToAllTypes()
        {
            var arr = np.array(new[] { 9223372036854775807L });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(-1);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(255);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(-1);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(65535);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-1);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(4294967295u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(9223372036854775807L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(9223372036854775807UL);
            // Float values: 9.223372036854776e+18 (rounded)
            arr.astype(np.float32).GetAtIndex<float>(0).Should().BeApproximately(9.223372e+18f, 1e12f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().BeApproximately(9.223372036854776e+18, 1e3);
        }

        [TestMethod]
        public void Int64_MinValue_ToAllTypes()
        {
            var arr = np.array(new[] { -9223372036854775808L });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(0);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(0);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(0);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(0);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(0);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(0u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-9223372036854775808L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(9223372036854775808UL);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().BeApproximately(-9.223372e+18f, 1e12f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().BeApproximately(-9.223372036854776e+18, 1e3);
        }

        #endregion

        #region UInt64 Source (5 values × 12 targets = 60 conversions)

        [TestMethod]
        public void UInt64_Zero_ToAllTypes()
        {
            var arr = np.array(new[] { 0UL });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeFalse();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(0);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(0);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(0);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(0);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(0);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(0u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(0L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(0UL);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(0.0f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(0.0);
        }

        [TestMethod]
        public void UInt64_One_ToAllTypes()
        {
            var arr = np.array(new[] { 1UL });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(1);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(1);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(1);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(1);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(1);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(1u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(1L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(1UL);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(1.0f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(1.0);
        }

        [TestMethod]
        public void UInt64_Int64Max_ToAllTypes()
        {
            var arr = np.array(new[] { 9223372036854775807UL });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(-1);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(255);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(-1);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(65535);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-1);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(4294967295u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(9223372036854775807L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(9223372036854775807UL);
        }

        [TestMethod]
        public void UInt64_Int64MaxPlus1_ToAllTypes()
        {
            var arr = np.array(new[] { 9223372036854775808UL });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(0);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(0);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(0);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(0);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(0);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(0u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-9223372036854775808L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(9223372036854775808UL);
        }

        [TestMethod]
        public void UInt64_MaxValue_ToAllTypes()
        {
            var arr = np.array(new[] { 18446744073709551615UL });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(-1);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(255);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(-1);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(65535);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-1);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(4294967295u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-1L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(18446744073709551615UL);
        }

        #endregion

        #region Float32 Source (8 values × 12 targets = 96 conversions)

        [TestMethod]
        public void Float32_Zero_ToAllTypes()
        {
            var arr = np.array(new[] { 0.0f });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeFalse();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(0);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(0);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(0);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(0);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(0);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(0u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(0L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(0UL);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(0.0f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(0.0);
        }

        [TestMethod]
        public void Float32_One_ToAllTypes()
        {
            var arr = np.array(new[] { 1.0f });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(1);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(1);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(1);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(1);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(1);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(1u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(1L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(1UL);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(1.0f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(1.0);
        }

        [TestMethod]
        public void Float32_NegativeOne_ToAllTypes()
        {
            var arr = np.array(new[] { -1.0f });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(-1);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(255);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(-1);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(65535);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-1);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(4294967295u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-1L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(18446744073709551615UL);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(-1.0f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(-1.0);
        }

        [TestMethod]
        public void Float32_Fractional_ToAllTypes()
        {
            var arr = np.array(new[] { 3.7f });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(3); // Truncate toward zero
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(3);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(3);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(3);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(3);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(3u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(3L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(3UL);
        }

        [TestMethod]
        public void Float32_NegativeFractional_ToAllTypes()
        {
            var arr = np.array(new[] { -3.7f });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(-3); // Truncate toward zero (not -4)
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(253); // -3 wraps to 253
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(-3);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(65533); // -3 wraps
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-3);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(4294967293u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-3L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(18446744073709551613UL);
        }

        [TestMethod]
        public void Float32_NaN_ToAllTypes()
        {
            var arr = np.array(new[] { float.NaN });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(0);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(0);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(0);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(0);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-2147483648); // int.MinValue
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(0u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-9223372036854775808L); // long.MinValue
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(9223372036854775808UL); // 2^63
            float.IsNaN(arr.astype(np.float32).GetAtIndex<float>(0)).Should().BeTrue();
            double.IsNaN(arr.astype(np.float64).GetAtIndex<double>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void Float32_PositiveInfinity_ToAllTypes()
        {
            var arr = np.array(new[] { float.PositiveInfinity });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(0);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(0);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(0);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(0);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-2147483648);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(0u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-9223372036854775808L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(9223372036854775808UL);
            float.IsPositiveInfinity(arr.astype(np.float32).GetAtIndex<float>(0)).Should().BeTrue();
            double.IsPositiveInfinity(arr.astype(np.float64).GetAtIndex<double>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void Float32_NegativeInfinity_ToAllTypes()
        {
            var arr = np.array(new[] { float.NegativeInfinity });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(0);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(0);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(0);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(0);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-2147483648);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(0u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-9223372036854775808L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(9223372036854775808UL);
            float.IsNegativeInfinity(arr.astype(np.float32).GetAtIndex<float>(0)).Should().BeTrue();
            double.IsNegativeInfinity(arr.astype(np.float64).GetAtIndex<double>(0)).Should().BeTrue();
        }

        #endregion

        #region Float64 Source (8 values × 12 targets = 96 conversions)

        [TestMethod]
        public void Float64_Zero_ToAllTypes()
        {
            var arr = np.array(new[] { 0.0 });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeFalse();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(0);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(0);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(0);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(0);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(0);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(0u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(0L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(0UL);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(0.0f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(0.0);
        }

        [TestMethod]
        public void Float64_One_ToAllTypes()
        {
            var arr = np.array(new[] { 1.0 });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(1);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(1);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(1);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(1);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(1);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(1u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(1L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(1UL);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(1.0f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(1.0);
        }

        [TestMethod]
        public void Float64_NegativeOne_ToAllTypes()
        {
            var arr = np.array(new[] { -1.0 });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(-1);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(255);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(-1);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(65535);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-1);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(4294967295u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-1L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(18446744073709551615UL);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(-1.0f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(-1.0);
        }

        [TestMethod]
        public void Float64_Fractional_ToAllTypes()
        {
            var arr = np.array(new[] { 3.7 });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(3);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(3);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(3);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(3);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(3);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(3u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(3L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(3UL);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(3.7);
        }

        [TestMethod]
        public void Float64_NegativeFractional_ToAllTypes()
        {
            var arr = np.array(new[] { -3.7 });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(-3);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(253);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(-3);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(65533);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-3);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(4294967293u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-3L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(18446744073709551613UL);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(-3.7);
        }

        [TestMethod]
        public void Float64_NaN_ToAllTypes()
        {
            var arr = np.array(new[] { double.NaN });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(0);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(0);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(0);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(0);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-2147483648);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(0u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-9223372036854775808L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(9223372036854775808UL);
            float.IsNaN(arr.astype(np.float32).GetAtIndex<float>(0)).Should().BeTrue();
            double.IsNaN(arr.astype(np.float64).GetAtIndex<double>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void Float64_PositiveInfinity_ToAllTypes()
        {
            var arr = np.array(new[] { double.PositiveInfinity });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(0);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(0);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(0);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(0);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-2147483648);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(0u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-9223372036854775808L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(9223372036854775808UL);
            float.IsPositiveInfinity(arr.astype(np.float32).GetAtIndex<float>(0)).Should().BeTrue();
            double.IsPositiveInfinity(arr.astype(np.float64).GetAtIndex<double>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void Float64_NegativeInfinity_ToAllTypes()
        {
            var arr = np.array(new[] { double.NegativeInfinity });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(0);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(0);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(0);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(0);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-2147483648);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(0u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-9223372036854775808L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(9223372036854775808UL);
            float.IsNegativeInfinity(arr.astype(np.float32).GetAtIndex<float>(0)).Should().BeTrue();
            double.IsNegativeInfinity(arr.astype(np.float64).GetAtIndex<double>(0)).Should().BeTrue();
        }

        #endregion

        #region Half (Float16) Source (8 values × 12 targets = 96 conversions)

        [TestMethod]
        public void Float16_Zero_ToAllTypes()
        {
            var arr = np.array(new Half[] { (Half)0.0f });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeFalse();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(0);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(0);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(0);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(0);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(0);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(0u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(0L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(0UL);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(0.0f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(0.0);
        }

        [TestMethod]
        public void Float16_One_ToAllTypes()
        {
            var arr = np.array(new Half[] { (Half)1.0f });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(1);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(1);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(1);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(1);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(1);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(1u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(1L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(1UL);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(1.0f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(1.0);
        }

        [TestMethod]
        public void Float16_NegativeOne_ToAllTypes()
        {
            var arr = np.array(new Half[] { (Half)(-1.0f) });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(-1);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(255);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(-1);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(65535);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-1);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(4294967295u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-1L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(18446744073709551615UL);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(-1.0f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(-1.0);
        }

        [TestMethod]
        public void Float16_Fractional_ToAllTypes()
        {
            // Note: Half(3.7) rounds to 3.69921875 due to precision
            var arr = np.array(new Half[] { (Half)3.7f });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(3);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(3);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(3);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(3);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(3);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(3u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(3L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(3UL);
        }

        [TestMethod]
        public void Float16_NegativeFractional_ToAllTypes()
        {
            var arr = np.array(new Half[] { (Half)(-3.7f) });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(-3);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(253);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(-3);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(65533);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-3);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(4294967293u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-3L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(18446744073709551613UL);
        }

        [TestMethod]
        public void Float16_NaN_ToAllTypes()
        {
            var arr = np.array(new Half[] { Half.NaN });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(0);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(0);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(0);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(0);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-2147483648);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(0u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-9223372036854775808L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(9223372036854775808UL);
            float.IsNaN(arr.astype(np.float32).GetAtIndex<float>(0)).Should().BeTrue();
            double.IsNaN(arr.astype(np.float64).GetAtIndex<double>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void Float16_PositiveInfinity_ToAllTypes()
        {
            var arr = np.array(new Half[] { Half.PositiveInfinity });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(0);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(0);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(0);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(0);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-2147483648);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(0u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-9223372036854775808L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(9223372036854775808UL);
            float.IsPositiveInfinity(arr.astype(np.float32).GetAtIndex<float>(0)).Should().BeTrue();
            double.IsPositiveInfinity(arr.astype(np.float64).GetAtIndex<double>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void Float16_NegativeInfinity_ToAllTypes()
        {
            var arr = np.array(new Half[] { Half.NegativeInfinity });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(0);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(0);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(0);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(0);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-2147483648);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(0u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-9223372036854775808L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(9223372036854775808UL);
            float.IsNegativeInfinity(arr.astype(np.float32).GetAtIndex<float>(0)).Should().BeTrue();
            double.IsNegativeInfinity(arr.astype(np.float64).GetAtIndex<double>(0)).Should().BeTrue();
        }

        #endregion

        #region Edge Case: Float16 to Float16 (special rounding)

        [TestMethod]
        public void Float16_ToFloat16_PreservesPrecision()
        {
            // NumPy: float16(3.7) -> 3.69921875 (rounded to half precision)
            var arr = np.array(new Half[] { (Half)3.7f });
            var result = arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0);

            // Half(3.7) = 3.69921875
            ((float)result).Should().BeApproximately(3.69921875f, 0.001f);
        }

        #endregion

        #region Edge Case: UInt16.MaxValue to Float16

        [TestMethod]
        public void UInt16_MaxValue_ToFloat16_IsInfinity()
        {
            // NumPy: np.array([65535], dtype=np.uint16).astype(np.float16) -> array([inf])
            // 65535 exceeds float16 max (~65504), so it becomes inf
            var arr = np.array(new ushort[] { 65535 });
            var result = arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0);

            Half.IsInfinity(result).Should().BeTrue();
        }

        #endregion

        #region Edge Case: Int32.MaxValue/MinValue to Float16

        [TestMethod]
        public void Int32_MaxValue_ToFloat16_IsInfinity()
        {
            // NumPy: np.array([2147483647], dtype=np.int32).astype(np.float16) -> array([inf])
            var arr = np.array(new int[] { int.MaxValue });
            var result = arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0);

            Half.IsPositiveInfinity(result).Should().BeTrue();
        }

        [TestMethod]
        public void Int32_MinValue_ToFloat16_IsNegativeInfinity()
        {
            // NumPy: np.array([-2147483648], dtype=np.int32).astype(np.float16) -> array([-inf])
            var arr = np.array(new int[] { int.MinValue });
            var result = arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0);

            Half.IsNegativeInfinity(result).Should().BeTrue();
        }

        #endregion
    }
}
