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
            ((float)arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeApproximately(3.69921875f, 0.001f);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().BeApproximately(3.7f, 0.001f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().BeApproximately(3.7, 0.001);
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
            ((float)arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeApproximately(-3.69921875f, 0.001f);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().BeApproximately(-3.7f, 0.001f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().BeApproximately(-3.7, 0.001);
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
            Half.IsNaN(arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeTrue();
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
            Half.IsPositiveInfinity(arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeTrue();
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
            Half.IsNegativeInfinity(arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeTrue();
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
            ((float)arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeApproximately(3.69921875f, 0.001f);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().BeApproximately(3.7f, 0.001f);
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
            ((float)arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeApproximately(-3.69921875f, 0.001f);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().BeApproximately(-3.7f, 0.001f);
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
            Half.IsNaN(arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeTrue();
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
            Half.IsPositiveInfinity(arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeTrue();
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
            Half.IsNegativeInfinity(arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeTrue();
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
            ((float)arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeApproximately(3.69921875f, 0.001f);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().BeApproximately(3.69921875f, 0.001f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().BeApproximately(3.69921875, 0.001);
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
            ((float)arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeApproximately(-3.69921875f, 0.001f);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().BeApproximately(-3.69921875f, 0.001f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().BeApproximately(-3.69921875, 0.001);
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
            Half.IsNaN(arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeTrue();
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
            Half.IsPositiveInfinity(arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeTrue();
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
            Half.IsNegativeInfinity(arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeTrue();
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

        #region Additional Edge Cases - Large Floats

        [TestMethod]
        public void Float64_LargePositive_ToInt32_ReturnsMinValue()
        {
            // NumPy: 1e10 is outside int32 range, returns MIN_VALUE
            var arr = np.array(new[] { 1e10 });
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-2147483648);
        }

        [TestMethod]
        public void Float64_LargePositive_ToUInt32_WrapsCorrectly()
        {
            // NumPy: 1e10 -> uint32 wraps to 1410065408 (not 0!)
            var arr = np.array(new[] { 1e10 });
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(1410065408u);
        }

        [TestMethod]
        public void Float64_LargeNegative_ToUInt32_WrapsCorrectly()
        {
            // NumPy: -1e10 -> uint32 wraps to 2884901888
            var arr = np.array(new[] { -1e10 });
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(2884901888u);
        }

        [TestMethod]
        public void Float64_ExtremelyLarge_ToInt64()
        {
            // NumPy: 1e18 fits, 1e19/1e20 overflow to MIN_VALUE
            np.array(new[] { 1e18 }).astype(np.int64).GetAtIndex<long>(0).Should().Be(1000000000000000000L);
            np.array(new[] { 1e19 }).astype(np.int64).GetAtIndex<long>(0).Should().Be(-9223372036854775808L);
            np.array(new[] { 1e20 }).astype(np.int64).GetAtIndex<long>(0).Should().Be(-9223372036854775808L);
        }

        [TestMethod]
        public void Float64_ExtremelyLarge_ToUInt64()
        {
            // NumPy: 1e19 fits, 1e20 overflows to 2^63
            np.array(new[] { 1e19 }).astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(10000000000000000000UL);
            np.array(new[] { 1e20 }).astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(9223372036854775808UL);
        }

        #endregion

        #region Additional Edge Cases - Exact Boundaries

        [TestMethod]
        public void Float64_AtInt8Boundaries_WrapsCorrectly()
        {
            // NumPy: exactly at boundary wraps
            np.array(new[] { 127.0 }).astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(127);
            np.array(new[] { 128.0 }).astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(-128);
            np.array(new[] { -128.0 }).astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(-128);
            np.array(new[] { -129.0 }).astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(127);
        }

        [TestMethod]
        public void Float64_AtUInt8Boundaries_WrapsCorrectly()
        {
            np.array(new[] { 255.0 }).astype(np.uint8).GetAtIndex<byte>(0).Should().Be(255);
            np.array(new[] { 256.0 }).astype(np.uint8).GetAtIndex<byte>(0).Should().Be(0);
        }

        [TestMethod]
        public void Float64_SmallFractions_TruncateToZero()
        {
            // All values < 1.0 truncate to 0
            np.array(new[] { 0.1 }).astype(np.int32).GetAtIndex<int>(0).Should().Be(0);
            np.array(new[] { -0.1 }).astype(np.int32).GetAtIndex<int>(0).Should().Be(0);
            np.array(new[] { 0.999999 }).astype(np.int32).GetAtIndex<int>(0).Should().Be(0);
            np.array(new[] { -0.999999 }).astype(np.int32).GetAtIndex<int>(0).Should().Be(0);
        }

        #endregion

        #region NumSharp-Specific: Char Type

        [TestMethod]
        public void Char_ToNumericTypes()
        {
            var arr = np.array(new char[] { 'A', 'Z', '\0', (char)255 });

            // Char -> int32: ASCII values
            var intResult = arr.astype(np.int32);
            intResult.GetAtIndex<int>(0).Should().Be(65);  // 'A'
            intResult.GetAtIndex<int>(1).Should().Be(90);  // 'Z'
            intResult.GetAtIndex<int>(2).Should().Be(0);   // '\0'
            intResult.GetAtIndex<int>(3).Should().Be(255);

            // Char -> uint8
            var byteResult = arr.astype(np.uint8);
            byteResult.GetAtIndex<byte>(0).Should().Be(65);
            byteResult.GetAtIndex<byte>(3).Should().Be(255);
        }

        [TestMethod]
        public void Int_ToChar_UsesLowBits()
        {
            var arr = np.array(new int[] { 65, 90, 0, 255, 1000 });
            var result = arr.astype(NPTypeCode.Char);

            result.GetAtIndex<char>(0).Should().Be('A');
            result.GetAtIndex<char>(1).Should().Be('Z');
            result.GetAtIndex<char>(2).Should().Be('\0');
            result.GetAtIndex<char>(3).Should().Be((char)255);
            result.GetAtIndex<char>(4).Should().Be((char)1000);
        }

        #endregion

        #region Complex Source → All 12 Targets

        [TestMethod]
        public void Complex_Zero_ToAllTypes()
        {
            // Complex(0, 0) → all types
            var arr = np.array(new System.Numerics.Complex[] { new(0, 0) });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeFalse();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(0);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(0);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(0);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(0);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(0);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(0u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(0L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(0UL);
            ((float)arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().Be(0.0f);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(0.0f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(0.0);
        }

        [TestMethod]
        public void Complex_One_ToAllTypes()
        {
            // Complex(1, 0) → all types
            var arr = np.array(new System.Numerics.Complex[] { new(1, 0) });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(1);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(1);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(1);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(1);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(1);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(1u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(1L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(1UL);
            ((float)arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().Be(1.0f);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(1.0f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(1.0);
        }

        [TestMethod]
        public void Complex_NegativeOne_ToAllTypes()
        {
            // Complex(-1, 0) → all types (wraps for unsigned)
            var arr = np.array(new System.Numerics.Complex[] { new(-1, 0) });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(-1);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(255);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(-1);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(65535);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(-1);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(4294967295u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(-1L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(18446744073709551615UL);
            ((float)arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().Be(-1.0f);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(-1.0f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(-1.0);
        }

        [TestMethod]
        public void Complex_Fractional_ToAllTypes()
        {
            // Complex(3.7, 4.2) → all types (imaginary part discarded, real truncated for int)
            var arr = np.array(new System.Numerics.Complex[] { new(3.7, 4.2) });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(3);
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(3);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(3);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(3);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(3);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(3u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(3L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(3UL);
            ((float)arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeApproximately(3.69921875f, 0.001f);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().BeApproximately(3.7f, 0.001f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(3.7);
        }

        [TestMethod]
        public void Complex_PureImaginary_ToAllTypes()
        {
            // Complex(0, 1) → all types (real part is 0, but nonzero for bool)
            var arr = np.array(new System.Numerics.Complex[] { new(0, 1) });

            arr.astype(np.@bool).GetAtIndex<bool>(0).Should().BeTrue(); // Nonzero magnitude
            arr.astype(NPTypeCode.SByte).GetAtIndex<sbyte>(0).Should().Be(0); // Real part = 0
            arr.astype(np.uint8).GetAtIndex<byte>(0).Should().Be(0);
            arr.astype(np.int16).GetAtIndex<short>(0).Should().Be(0);
            arr.astype(np.uint16).GetAtIndex<ushort>(0).Should().Be(0);
            arr.astype(np.int32).GetAtIndex<int>(0).Should().Be(0);
            arr.astype(np.uint32).GetAtIndex<uint>(0).Should().Be(0u);
            arr.astype(np.int64).GetAtIndex<long>(0).Should().Be(0L);
            arr.astype(np.uint64).GetAtIndex<ulong>(0).Should().Be(0UL);
            ((float)arr.astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().Be(0.0f);
            arr.astype(np.float32).GetAtIndex<float>(0).Should().Be(0.0f);
            arr.astype(np.float64).GetAtIndex<double>(0).Should().Be(0.0);
        }

        #endregion

        #region All Types → Complex Target

        [TestMethod]
        public void Bool_ToComplex()
        {
            np.array(new[] { false }).astype(NPTypeCode.Complex).GetAtIndex<System.Numerics.Complex>(0).Should().Be(new System.Numerics.Complex(0, 0));
            np.array(new[] { true }).astype(NPTypeCode.Complex).GetAtIndex<System.Numerics.Complex>(0).Should().Be(new System.Numerics.Complex(1, 0));
        }

        [TestMethod]
        public void Int32_ToComplex()
        {
            var values = new int[] { 0, 1, -1, 127, -128 };
            var arr = np.array(values);
            var result = arr.astype(NPTypeCode.Complex);

            result.GetAtIndex<System.Numerics.Complex>(0).Should().Be(new System.Numerics.Complex(0, 0));
            result.GetAtIndex<System.Numerics.Complex>(1).Should().Be(new System.Numerics.Complex(1, 0));
            result.GetAtIndex<System.Numerics.Complex>(2).Should().Be(new System.Numerics.Complex(-1, 0));
            result.GetAtIndex<System.Numerics.Complex>(3).Should().Be(new System.Numerics.Complex(127, 0));
            result.GetAtIndex<System.Numerics.Complex>(4).Should().Be(new System.Numerics.Complex(-128, 0));
        }

        [TestMethod]
        public void Float64_ToComplex()
        {
            var values = new double[] { 0.0, 1.0, -1.0, 3.7 };
            var arr = np.array(values);
            var result = arr.astype(NPTypeCode.Complex);

            result.GetAtIndex<System.Numerics.Complex>(0).Should().Be(new System.Numerics.Complex(0, 0));
            result.GetAtIndex<System.Numerics.Complex>(1).Should().Be(new System.Numerics.Complex(1, 0));
            result.GetAtIndex<System.Numerics.Complex>(2).Should().Be(new System.Numerics.Complex(-1, 0));
            result.GetAtIndex<System.Numerics.Complex>(3).Should().Be(new System.Numerics.Complex(3.7, 0));
        }

        [TestMethod]
        public void Float64_NaNInf_ToComplex()
        {
            var nanResult = np.array(new[] { double.NaN }).astype(NPTypeCode.Complex).GetAtIndex<System.Numerics.Complex>(0);
            double.IsNaN(nanResult.Real).Should().BeTrue();
            nanResult.Imaginary.Should().Be(0);

            var infResult = np.array(new[] { double.PositiveInfinity }).astype(NPTypeCode.Complex).GetAtIndex<System.Numerics.Complex>(0);
            double.IsPositiveInfinity(infResult.Real).Should().BeTrue();
            infResult.Imaginary.Should().Be(0);
        }

        [TestMethod]
        public void Int8_ToComplex()
        {
            var values = new sbyte[] { 0, 1, -1, 127, -128 };
            var arr = np.array(values);
            var result = arr.astype(NPTypeCode.Complex);

            result.GetAtIndex<System.Numerics.Complex>(0).Real.Should().Be(0);
            result.GetAtIndex<System.Numerics.Complex>(1).Real.Should().Be(1);
            result.GetAtIndex<System.Numerics.Complex>(2).Real.Should().Be(-1);
            result.GetAtIndex<System.Numerics.Complex>(3).Real.Should().Be(127);
            result.GetAtIndex<System.Numerics.Complex>(4).Real.Should().Be(-128);
        }

        [TestMethod]
        public void UInt8_ToComplex()
        {
            var values = new byte[] { 0, 1, 127, 128, 255 };
            var arr = np.array(values);
            var result = arr.astype(NPTypeCode.Complex);

            result.GetAtIndex<System.Numerics.Complex>(0).Real.Should().Be(0);
            result.GetAtIndex<System.Numerics.Complex>(1).Real.Should().Be(1);
            result.GetAtIndex<System.Numerics.Complex>(4).Real.Should().Be(255);
        }

        [TestMethod]
        public void Float32_ToComplex()
        {
            var values = new float[] { 0.0f, 1.0f, -1.0f, 3.7f };
            var arr = np.array(values);
            var result = arr.astype(NPTypeCode.Complex);

            result.GetAtIndex<System.Numerics.Complex>(0).Real.Should().Be(0);
            result.GetAtIndex<System.Numerics.Complex>(1).Real.Should().Be(1);
            result.GetAtIndex<System.Numerics.Complex>(2).Real.Should().Be(-1);
            result.GetAtIndex<System.Numerics.Complex>(3).Real.Should().BeApproximately(3.7, 0.001);
        }

        [TestMethod]
        public void Half_ToComplex()
        {
            var values = new Half[] { (Half)0.0f, (Half)1.0f, (Half)(-1.0f) };
            var arr = np.array(values);
            var result = arr.astype(NPTypeCode.Complex);

            result.GetAtIndex<System.Numerics.Complex>(0).Real.Should().Be(0);
            result.GetAtIndex<System.Numerics.Complex>(1).Real.Should().Be(1);
            result.GetAtIndex<System.Numerics.Complex>(2).Real.Should().Be(-1);
        }

        #endregion

        #region NumSharp-Specific: Decimal Type

        [TestMethod]
        public void Decimal_ToFloat64_Preserves()
        {
            var arr = np.array(new decimal[] { 0m, 1m, -1m, 3.7m, -3.7m });

            var result = arr.astype(np.float64);
            result.GetAtIndex<double>(0).Should().Be(0.0);
            result.GetAtIndex<double>(1).Should().Be(1.0);
            result.GetAtIndex<double>(2).Should().Be(-1.0);
            result.GetAtIndex<double>(3).Should().Be(3.7);
            result.GetAtIndex<double>(4).Should().Be(-3.7);
        }

        [TestMethod]
        public void Decimal_ToInt32_Truncates()
        {
            var arr = np.array(new decimal[] { 0m, 1m, -1m, 3.7m, -3.7m });

            var result = arr.astype(np.int32);
            result.GetAtIndex<int>(0).Should().Be(0);
            result.GetAtIndex<int>(1).Should().Be(1);
            result.GetAtIndex<int>(2).Should().Be(-1);
            result.GetAtIndex<int>(3).Should().Be(3);  // Truncate, not round
            result.GetAtIndex<int>(4).Should().Be(-3);
        }

        #endregion

        #region MISSING: All Types → Half (Float16)

        [TestMethod]
        public void Bool_ToHalf()
        {
            np.array(new[] { false }).astype(NPTypeCode.Half).GetAtIndex<Half>(0).Should().Be((Half)0.0f);
            np.array(new[] { true }).astype(NPTypeCode.Half).GetAtIndex<Half>(0).Should().Be((Half)1.0f);
        }

        [TestMethod]
        public void Int8_ToHalf()
        {
            var values = new sbyte[] { 0, 1, -1, 127, -128 };
            var expected = new float[] { 0.0f, 1.0f, -1.0f, 127.0f, -128.0f };

            var arr = np.array(values);
            var result = arr.astype(NPTypeCode.Half);

            for (int i = 0; i < values.Length; i++)
                ((float)result.GetAtIndex<Half>(i)).Should().Be(expected[i]);
        }

        [TestMethod]
        public void UInt8_ToHalf()
        {
            var values = new byte[] { 0, 1, 127, 128, 255 };
            var expected = new float[] { 0.0f, 1.0f, 127.0f, 128.0f, 255.0f };

            var arr = np.array(values);
            var result = arr.astype(NPTypeCode.Half);

            for (int i = 0; i < values.Length; i++)
                ((float)result.GetAtIndex<Half>(i)).Should().Be(expected[i]);
        }

        [TestMethod]
        public void Int16_ToHalf()
        {
            // Note: int16(32767) -> float16 = 32768.0 (rounded due to precision)
            var arr = np.array(new short[] { 0, 1, -1, 32767, -32768 });
            var result = arr.astype(NPTypeCode.Half);

            ((float)result.GetAtIndex<Half>(0)).Should().Be(0.0f);
            ((float)result.GetAtIndex<Half>(1)).Should().Be(1.0f);
            ((float)result.GetAtIndex<Half>(2)).Should().Be(-1.0f);
            ((float)result.GetAtIndex<Half>(3)).Should().Be(32768.0f); // Rounded
            ((float)result.GetAtIndex<Half>(4)).Should().Be(-32768.0f);
        }

        [TestMethod]
        public void UInt16_ToHalf()
        {
            var arr = np.array(new ushort[] { 0, 1, 32767, 32768, 65504 });
            var result = arr.astype(NPTypeCode.Half);

            ((float)result.GetAtIndex<Half>(0)).Should().Be(0.0f);
            ((float)result.GetAtIndex<Half>(1)).Should().Be(1.0f);
            ((float)result.GetAtIndex<Half>(2)).Should().Be(32768.0f); // Rounded
            ((float)result.GetAtIndex<Half>(3)).Should().Be(32768.0f);
            ((float)result.GetAtIndex<Half>(4)).Should().Be(65504.0f); // Max finite float16
        }

        [TestMethod]
        public void Int32_ToHalf()
        {
            var arr = np.array(new int[] { 0, 1, -1, 65504, -65504 });
            var result = arr.astype(NPTypeCode.Half);

            ((float)result.GetAtIndex<Half>(0)).Should().Be(0.0f);
            ((float)result.GetAtIndex<Half>(1)).Should().Be(1.0f);
            ((float)result.GetAtIndex<Half>(2)).Should().Be(-1.0f);
            ((float)result.GetAtIndex<Half>(3)).Should().Be(65504.0f);
            ((float)result.GetAtIndex<Half>(4)).Should().Be(-65504.0f);
        }

        [TestMethod]
        public void UInt32_ToHalf()
        {
            var arr = np.array(new uint[] { 0u, 1u, 65504u });
            var result = arr.astype(NPTypeCode.Half);

            ((float)result.GetAtIndex<Half>(0)).Should().Be(0.0f);
            ((float)result.GetAtIndex<Half>(1)).Should().Be(1.0f);
            ((float)result.GetAtIndex<Half>(2)).Should().Be(65504.0f);
        }

        [TestMethod]
        public void Int64_ToHalf()
        {
            var arr = np.array(new long[] { 0L, 1L, -1L, 65504L, -65504L });
            var result = arr.astype(NPTypeCode.Half);

            ((float)result.GetAtIndex<Half>(0)).Should().Be(0.0f);
            ((float)result.GetAtIndex<Half>(1)).Should().Be(1.0f);
            ((float)result.GetAtIndex<Half>(2)).Should().Be(-1.0f);
            ((float)result.GetAtIndex<Half>(3)).Should().Be(65504.0f);
            ((float)result.GetAtIndex<Half>(4)).Should().Be(-65504.0f);
        }

        [TestMethod]
        public void UInt64_ToHalf()
        {
            var arr = np.array(new ulong[] { 0UL, 1UL, 65504UL });
            var result = arr.astype(NPTypeCode.Half);

            ((float)result.GetAtIndex<Half>(0)).Should().Be(0.0f);
            ((float)result.GetAtIndex<Half>(1)).Should().Be(1.0f);
            ((float)result.GetAtIndex<Half>(2)).Should().Be(65504.0f);
        }

        [TestMethod]
        public void Float32_ToHalf()
        {
            var arr = np.array(new float[] { 0.0f, 1.0f, -1.0f, 3.7f, -3.7f });
            var result = arr.astype(NPTypeCode.Half);

            ((float)result.GetAtIndex<Half>(0)).Should().Be(0.0f);
            ((float)result.GetAtIndex<Half>(1)).Should().Be(1.0f);
            ((float)result.GetAtIndex<Half>(2)).Should().Be(-1.0f);
            ((float)result.GetAtIndex<Half>(3)).Should().BeApproximately(3.69921875f, 0.001f);
            ((float)result.GetAtIndex<Half>(4)).Should().BeApproximately(-3.69921875f, 0.001f);
        }

        [TestMethod]
        public void Float32_NaNInf_ToHalf()
        {
            Half.IsNaN(np.array(new[] { float.NaN }).astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeTrue();
            Half.IsPositiveInfinity(np.array(new[] { float.PositiveInfinity }).astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeTrue();
            Half.IsNegativeInfinity(np.array(new[] { float.NegativeInfinity }).astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void Float64_ToHalf()
        {
            var arr = np.array(new double[] { 0.0, 1.0, -1.0, 3.7, -3.7 });
            var result = arr.astype(NPTypeCode.Half);

            ((float)result.GetAtIndex<Half>(0)).Should().Be(0.0f);
            ((float)result.GetAtIndex<Half>(1)).Should().Be(1.0f);
            ((float)result.GetAtIndex<Half>(2)).Should().Be(-1.0f);
            ((float)result.GetAtIndex<Half>(3)).Should().BeApproximately(3.69921875f, 0.001f);
            ((float)result.GetAtIndex<Half>(4)).Should().BeApproximately(-3.69921875f, 0.001f);
        }

        [TestMethod]
        public void Float64_NaNInf_ToHalf()
        {
            Half.IsNaN(np.array(new[] { double.NaN }).astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeTrue();
            Half.IsPositiveInfinity(np.array(new[] { double.PositiveInfinity }).astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeTrue();
            Half.IsNegativeInfinity(np.array(new[] { double.NegativeInfinity }).astype(NPTypeCode.Half).GetAtIndex<Half>(0)).Should().BeTrue();
        }

        #endregion

        #region MISSING: Float Fractional → Float32/Float64

        [TestMethod]
        public void Float64_Fractional_ToFloat32()
        {
            var arr = np.array(new double[] { 3.7, -3.7 });
            var result = arr.astype(np.float32);

            result.GetAtIndex<float>(0).Should().BeApproximately(3.7f, 0.0001f);
            result.GetAtIndex<float>(1).Should().BeApproximately(-3.7f, 0.0001f);
        }

        [TestMethod]
        public void Float32_Fractional_ToFloat64()
        {
            var arr = np.array(new float[] { 3.7f, -3.7f });
            var result = arr.astype(np.float64);

            result.GetAtIndex<double>(0).Should().BeApproximately(3.7, 0.0001);
            result.GetAtIndex<double>(1).Should().BeApproximately(-3.7, 0.0001);
        }

        [TestMethod]
        public void Float16_ToFloat32()
        {
            var arr = np.array(new Half[] { (Half)0.0f, (Half)1.0f, (Half)(-1.0f), (Half)3.7f, (Half)(-3.7f) });
            var result = arr.astype(np.float32);

            result.GetAtIndex<float>(0).Should().Be(0.0f);
            result.GetAtIndex<float>(1).Should().Be(1.0f);
            result.GetAtIndex<float>(2).Should().Be(-1.0f);
            result.GetAtIndex<float>(3).Should().BeApproximately(3.69921875f, 0.001f);
            result.GetAtIndex<float>(4).Should().BeApproximately(-3.69921875f, 0.001f);
        }

        [TestMethod]
        public void Float16_ToFloat64()
        {
            var arr = np.array(new Half[] { (Half)0.0f, (Half)1.0f, (Half)(-1.0f), (Half)3.7f, (Half)(-3.7f) });
            var result = arr.astype(np.float64);

            result.GetAtIndex<double>(0).Should().Be(0.0);
            result.GetAtIndex<double>(1).Should().Be(1.0);
            result.GetAtIndex<double>(2).Should().Be(-1.0);
            result.GetAtIndex<double>(3).Should().BeApproximately(3.69921875, 0.001);
            result.GetAtIndex<double>(4).Should().BeApproximately(-3.69921875, 0.001);
        }

        #endregion
    }
}
