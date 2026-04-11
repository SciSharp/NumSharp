using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using TUnit.Core;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// Tests for float->int astype truncation behavior.
    ///
    /// NumPy uses truncation toward zero for float-to-int conversions:
    ///   >>> np.array([2.9, -2.9]).astype(int)
    ///   array([ 2, -2])
    ///
    /// This is different from banker's rounding (Convert.ToInt32) which rounds
    /// to nearest even: 2.5 -> 2, 3.5 -> 4.
    ///
    /// Fixed in commit 40a5c831 - changed Converts.Native.cs to use C# cast
    /// truncation instead of Convert.ToInt32.
    /// </summary>
    public class astypeTruncationTests
    {
        // ================================================================
        // BASIC TRUNCATION TESTS
        // ================================================================

        [Test]
        public void Float64_ToInt32_PositiveFractions_TruncateTowardZero()
        {
            // NumPy: np.array([2.1, 2.5, 2.9]).astype(np.int32) -> [2, 2, 2]
            var arr = np.array(new double[] { 2.1, 2.5, 2.9 });
            var result = arr.astype(np.int32);

            result.GetTypeCode.Should().Be(NPTypeCode.Int32);
            result.GetAtIndex<int>(0).Should().Be(2, "2.1 should truncate to 2");
            result.GetAtIndex<int>(1).Should().Be(2, "2.5 should truncate to 2 (not round to 2 or 3)");
            result.GetAtIndex<int>(2).Should().Be(2, "2.9 should truncate to 2");
        }

        [Test]
        public void Float64_ToInt32_NegativeFractions_TruncateTowardZero()
        {
            // NumPy: np.array([-2.1, -2.5, -2.9]).astype(np.int32) -> [-2, -2, -2]
            // Truncation toward zero means -2.9 -> -2, NOT -3
            var arr = np.array(new double[] { -2.1, -2.5, -2.9 });
            var result = arr.astype(np.int32);

            result.GetTypeCode.Should().Be(NPTypeCode.Int32);
            result.GetAtIndex<int>(0).Should().Be(-2, "-2.1 should truncate to -2");
            result.GetAtIndex<int>(1).Should().Be(-2, "-2.5 should truncate to -2 (not round to -2 or -3)");
            result.GetAtIndex<int>(2).Should().Be(-2, "-2.9 should truncate to -2 (NOT -3)");
        }

        [Test]
        public void Float32_ToInt32_TruncateTowardZero()
        {
            // Same behavior for float32
            var arr = np.array(new float[] { 2.9f, -2.9f, 1.5f, -1.5f });
            var result = arr.astype(np.int32);

            result.GetTypeCode.Should().Be(NPTypeCode.Int32);
            result.GetAtIndex<int>(0).Should().Be(2, "2.9f should truncate to 2");
            result.GetAtIndex<int>(1).Should().Be(-2, "-2.9f should truncate to -2");
            result.GetAtIndex<int>(2).Should().Be(1, "1.5f should truncate to 1");
            result.GetAtIndex<int>(3).Should().Be(-1, "-1.5f should truncate to -1");
        }

        // ================================================================
        // BANKER'S ROUNDING EDGE CASES
        // These are the cases where Convert.ToInt32 (banker's rounding)
        // would give different results than truncation
        // ================================================================

        [Test]
        public void Float64_ToInt32_BankersRoundingEdgeCases()
        {
            // Banker's rounding would round these to even:
            // 0.5 -> 0 (even), 1.5 -> 2 (even), 2.5 -> 2 (even), 3.5 -> 4 (even)
            // Truncation should give: 0, 1, 2, 3
            var arr = np.array(new double[] { 0.5, 1.5, 2.5, 3.5, 4.5 });
            var result = arr.astype(np.int32);

            result.GetAtIndex<int>(0).Should().Be(0, "0.5 truncates to 0");
            result.GetAtIndex<int>(1).Should().Be(1, "1.5 truncates to 1 (banker's would give 2)");
            result.GetAtIndex<int>(2).Should().Be(2, "2.5 truncates to 2");
            result.GetAtIndex<int>(3).Should().Be(3, "3.5 truncates to 3 (banker's would give 4)");
            result.GetAtIndex<int>(4).Should().Be(4, "4.5 truncates to 4");
        }

        [Test]
        public void Float64_ToInt32_NegativeBankersRoundingEdgeCases()
        {
            // Banker's rounding for negatives:
            // -0.5 -> 0 (even), -1.5 -> -2 (even), -2.5 -> -2 (even), -3.5 -> -4 (even)
            // Truncation toward zero: 0, -1, -2, -3
            var arr = np.array(new double[] { -0.5, -1.5, -2.5, -3.5, -4.5 });
            var result = arr.astype(np.int32);

            result.GetAtIndex<int>(0).Should().Be(0, "-0.5 truncates to 0");
            result.GetAtIndex<int>(1).Should().Be(-1, "-1.5 truncates to -1 (banker's would give -2)");
            result.GetAtIndex<int>(2).Should().Be(-2, "-2.5 truncates to -2");
            result.GetAtIndex<int>(3).Should().Be(-3, "-3.5 truncates to -3 (banker's would give -4)");
            result.GetAtIndex<int>(4).Should().Be(-4, "-4.5 truncates to -4");
        }

        // ================================================================
        // VERY SMALL VALUES
        // ================================================================

        [Test]
        public void Float64_ToInt32_VerySmallPositive_TruncateToZero()
        {
            var arr = np.array(new double[] { 0.0001, 0.1, 0.9, 0.99999 });
            var result = arr.astype(np.int32);

            result.GetAtIndex<int>(0).Should().Be(0, "0.0001 truncates to 0");
            result.GetAtIndex<int>(1).Should().Be(0, "0.1 truncates to 0");
            result.GetAtIndex<int>(2).Should().Be(0, "0.9 truncates to 0");
            result.GetAtIndex<int>(3).Should().Be(0, "0.99999 truncates to 0");
        }

        [Test]
        public void Float64_ToInt32_VerySmallNegative_TruncateToZero()
        {
            var arr = np.array(new double[] { -0.0001, -0.1, -0.9, -0.99999 });
            var result = arr.astype(np.int32);

            result.GetAtIndex<int>(0).Should().Be(0, "-0.0001 truncates to 0");
            result.GetAtIndex<int>(1).Should().Be(0, "-0.1 truncates to 0");
            result.GetAtIndex<int>(2).Should().Be(0, "-0.9 truncates to 0");
            result.GetAtIndex<int>(3).Should().Be(0, "-0.99999 truncates to 0");
        }

        // ================================================================
        // TYPE BOUNDARY VALUES
        // ================================================================

        [Test]
        public void Float64_ToInt32_AtBoundaries()
        {
            // Values just inside Int32 range
            var arr = np.array(new double[] {
                int.MaxValue - 0.5,
                int.MinValue + 0.5,
                2147483646.9,  // near int.MaxValue
                -2147483647.9  // near int.MinValue
            });
            var result = arr.astype(np.int32);

            result.GetAtIndex<int>(0).Should().Be(int.MaxValue - 1, "MaxValue-0.5 truncates");
            result.GetAtIndex<int>(1).Should().Be(int.MinValue + 1, "MinValue+0.5 truncates");
            result.GetAtIndex<int>(2).Should().Be(2147483646, "2147483646.9 truncates to 2147483646");
            result.GetAtIndex<int>(3).Should().Be(-2147483647, "-2147483647.9 truncates to -2147483647");
        }

        [Test]
        public void Float64_ToInt64_AtBoundaries()
        {
            // Int64 has a larger range
            var arr = np.array(new double[] {
                1e15 + 0.9,
                -1e15 - 0.9,
                9007199254740991.0  // Max safe integer in double (2^53 - 1)
            });
            var result = arr.astype(np.int64);

            result.GetTypeCode.Should().Be(NPTypeCode.Int64);
            // Note: due to double precision limits, exact values may vary
            result.GetAtIndex<long>(0).Should().BeGreaterThan(999999999999999L);
            result.GetAtIndex<long>(1).Should().BeLessThan(-999999999999999L);
            result.GetAtIndex<long>(2).Should().Be(9007199254740991L);
        }

        [Test]
        public void Float64_ToByte_AtBoundaries()
        {
            var arr = np.array(new double[] { 0.9, 254.9, 127.5, 255.0 });
            var result = arr.astype(np.uint8);

            result.GetTypeCode.Should().Be(NPTypeCode.Byte);
            result.GetAtIndex<byte>(0).Should().Be(0, "0.9 truncates to 0");
            result.GetAtIndex<byte>(1).Should().Be(254, "254.9 truncates to 254");
            result.GetAtIndex<byte>(2).Should().Be(127, "127.5 truncates to 127");
            result.GetAtIndex<byte>(3).Should().Be(255, "255.0 stays 255");
        }

        [Test]
        public void Float64_ToInt16_AtBoundaries()
        {
            var arr = np.array(new double[] {
                32766.9,
                -32767.9,
                short.MaxValue - 0.1,
                short.MinValue + 0.1
            });
            var result = arr.astype(np.int16);

            result.GetTypeCode.Should().Be(NPTypeCode.Int16);
            result.GetAtIndex<short>(0).Should().Be(32766, "32766.9 truncates to 32766");
            result.GetAtIndex<short>(1).Should().Be(-32767, "-32767.9 truncates to -32767");
            result.GetAtIndex<short>(2).Should().Be(32766, "32766.9 truncates to 32766");
            result.GetAtIndex<short>(3).Should().Be(-32767, "-32767.9 truncates to -32767");
        }

        // ================================================================
        // OVERFLOW BEHAVIOR
        // ================================================================

        [Test]
        public void Float64_ToInt32_Overflow_ThrowsException()
        {
            var arr = np.array(new double[] { (double)int.MaxValue + 1000 });

            Action act = () => arr.astype(np.int32);

            act.Should().Throw<OverflowException>(
                "Converting value > int.MaxValue should throw OverflowException");
        }

        [Test]
        public void Float64_ToInt32_Underflow_ThrowsException()
        {
            var arr = np.array(new double[] { (double)int.MinValue - 1000 });

            Action act = () => arr.astype(np.int32);

            act.Should().Throw<OverflowException>(
                "Converting value < int.MinValue should throw OverflowException");
        }

        // ================================================================
        // NaN AND INFINITY HANDLING
        // Note: NumPy behavior for NaN/Inf -> int is platform-dependent
        // and raises a warning. NumSharp should throw or have defined behavior.
        // ================================================================

        [Test]
        public void Float64_ToInt32_NaN_ThrowsOrReturnsZero()
        {
            // NumPy: RuntimeWarning and returns -2147483648 (or platform dependent)
            // In C#, (int)double.NaN is 0 or throws depending on checked context
            var arr = np.array(new double[] { double.NaN });

            try
            {
                var result = arr.astype(np.int32);
                // If it doesn't throw, document the behavior
                // C# (int)double.NaN gives int.MinValue in unchecked context
                // or 0 in some implementations
                var value = result.GetAtIndex<int>(0);
                // Accept either 0 or int.MinValue as valid implementation-defined behavior
                (value == 0 || value == int.MinValue).Should().BeTrue(
                    $"NaN conversion should yield 0 or int.MinValue, got {value}");
            }
            catch (OverflowException)
            {
                // Also acceptable - throwing on NaN conversion
            }
        }

        [Test]
        public void Float64_ToInt32_PositiveInfinity_Throws()
        {
            var arr = np.array(new double[] { double.PositiveInfinity });

            Action act = () => arr.astype(np.int32);

            // Should throw because infinity is outside int range
            act.Should().Throw<OverflowException>(
                "Converting +Infinity should throw OverflowException");
        }

        [Test]
        public void Float64_ToInt32_NegativeInfinity_Throws()
        {
            var arr = np.array(new double[] { double.NegativeInfinity });

            Action act = () => arr.astype(np.int32);

            // Should throw because infinity is outside int range
            act.Should().Throw<OverflowException>(
                "Converting -Infinity should throw OverflowException");
        }

        // ================================================================
        // ALL INTEGER TARGET TYPES
        // Verify truncation works for all float->int conversion paths
        // ================================================================

        [Test]
        public void Float64_ToAllIntTypes_Truncation()
        {
            var arr = np.array(new double[] { 2.9, -2.9 });

            // Note: NumSharp doesn't support sbyte (int8) - see BUG-7 in CLAUDE.md
            // So we skip Int8/sbyte tests

            // UInt8 (byte) - negative truncates toward zero then overflows
            // Actually, negative to unsigned should overflow/wrap
            var toByte = np.array(new double[] { 2.9, 254.9 }).astype(np.uint8);
            toByte.GetAtIndex<byte>(0).Should().Be(2);
            toByte.GetAtIndex<byte>(1).Should().Be(254);

            // Int16 (short)
            var toInt16 = arr.astype(np.int16);
            toInt16.GetTypeCode.Should().Be(NPTypeCode.Int16);
            toInt16.GetAtIndex<short>(0).Should().Be(2);
            toInt16.GetAtIndex<short>(1).Should().Be(-2);

            // UInt16 (ushort)
            var toUInt16 = np.array(new double[] { 2.9, 65534.9 }).astype(np.uint16);
            toUInt16.GetAtIndex<ushort>(0).Should().Be(2);
            toUInt16.GetAtIndex<ushort>(1).Should().Be(65534);

            // Int32
            var toInt32 = arr.astype(np.int32);
            toInt32.GetAtIndex<int>(0).Should().Be(2);
            toInt32.GetAtIndex<int>(1).Should().Be(-2);

            // UInt32
            var toUInt32 = np.array(new double[] { 2.9, 4294967294.9 }).astype(np.uint32);
            toUInt32.GetAtIndex<uint>(0).Should().Be(2);
            toUInt32.GetAtIndex<uint>(1).Should().Be(4294967294);

            // Int64
            var toInt64 = arr.astype(np.int64);
            toInt64.GetAtIndex<long>(0).Should().Be(2);
            toInt64.GetAtIndex<long>(1).Should().Be(-2);

            // UInt64
            var toUInt64 = np.array(new double[] { 2.9, 1000.9 }).astype(np.uint64);
            toUInt64.GetAtIndex<ulong>(0).Should().Be(2);
            toUInt64.GetAtIndex<ulong>(1).Should().Be(1000);
        }

        [Test]
        public void Float32_ToAllIntTypes_Truncation()
        {
            var arr = np.array(new float[] { 2.9f, -2.9f });

            // Int32
            var toInt32 = arr.astype(np.int32);
            toInt32.GetAtIndex<int>(0).Should().Be(2);
            toInt32.GetAtIndex<int>(1).Should().Be(-2);

            // Int64
            var toInt64 = arr.astype(np.int64);
            toInt64.GetAtIndex<long>(0).Should().Be(2);
            toInt64.GetAtIndex<long>(1).Should().Be(-2);

            // Int16
            var toInt16 = arr.astype(np.int16);
            toInt16.GetAtIndex<short>(0).Should().Be(2);
            toInt16.GetAtIndex<short>(1).Should().Be(-2);
        }

        // ================================================================
        // MULTI-DIMENSIONAL ARRAYS
        // ================================================================

        [Test]
        public void Float64_ToInt32_2DArray_TruncatesAllElements()
        {
            var arr = np.array(new double[,] {
                { 1.9, 2.9, 3.9 },
                { -1.9, -2.9, -3.9 }
            });
            var result = arr.astype(np.int32);

            result.shape.Should().BeEquivalentTo(new long[] { 2, 3 });
            result.GetTypeCode.Should().Be(NPTypeCode.Int32);

            // Row 0
            result[0, 0].GetInt32().Should().Be(1);
            result[0, 1].GetInt32().Should().Be(2);
            result[0, 2].GetInt32().Should().Be(3);

            // Row 1
            result[1, 0].GetInt32().Should().Be(-1);
            result[1, 1].GetInt32().Should().Be(-2);
            result[1, 2].GetInt32().Should().Be(-3);
        }

        [Test]
        public void Float64_ToInt32_3DArray_TruncatesAllElements()
        {
            var arr = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            arr = arr + 0.9; // Add 0.9 to all elements

            var result = arr.astype(np.int32);

            result.shape.Should().BeEquivalentTo(new long[] { 2, 3, 4 });

            // First element should be 0 (0.9 truncated)
            result[0, 0, 0].GetInt32().Should().Be(0);
            // Last element should be 23 (23.9 truncated)
            result[1, 2, 3].GetInt32().Should().Be(23);
        }

        // ================================================================
        // SLICED/VIEW ARRAYS
        // ================================================================

        [Test]
        public void Float64_ToInt32_SlicedArray_TruncatesCorrectly()
        {
            var arr = np.array(new double[] { 0.9, 1.9, 2.9, 3.9, 4.9, 5.9 });
            var sliced = arr["1:5"]; // [1.9, 2.9, 3.9, 4.9]

            var result = sliced.astype(np.int32);

            result.size.Should().Be(4);
            result.GetAtIndex<int>(0).Should().Be(1);
            result.GetAtIndex<int>(1).Should().Be(2);
            result.GetAtIndex<int>(2).Should().Be(3);
            result.GetAtIndex<int>(3).Should().Be(4);
        }

        [Test]
        public void Float64_ToInt32_SteppedSlice_TruncatesCorrectly()
        {
            var arr = np.array(new double[] { 0.9, 1.9, 2.9, 3.9, 4.9, 5.9 });
            var sliced = arr["::2"]; // [0.9, 2.9, 4.9]

            var result = sliced.astype(np.int32);

            result.size.Should().Be(3);
            result.GetAtIndex<int>(0).Should().Be(0);
            result.GetAtIndex<int>(1).Should().Be(2);
            result.GetAtIndex<int>(2).Should().Be(4);
        }

        // ================================================================
        // COPY PARAMETER
        // ================================================================

        [Test]
        public void Float64_ToInt32_WithCopy_CreatesNewArray()
        {
            var arr = np.array(new double[] { 2.9, -2.9 });
            var result = arr.astype(np.int32, copy: true);

            result.Array.Should().NotBeSameAs(arr.Array);
            result.GetAtIndex<int>(0).Should().Be(2);
            result.GetAtIndex<int>(1).Should().Be(-2);
        }

        // ================================================================
        // MIXED VALUES
        // ================================================================

        [Test]
        public void Float64_ToInt32_MixedValues_AllTruncateCorrectly()
        {
            // NumPy reference:
            // >>> np.array([1.7, 2.3, -1.7, -2.3, 0.0, -0.0, 100.9999]).astype(np.int32)
            // array([  1,   2,  -1,  -2,   0,   0, 100])
            var arr = np.array(new double[] { 1.7, 2.3, -1.7, -2.3, 0.0, -0.0, 100.9999 });
            var result = arr.astype(np.int32);

            result.GetAtIndex<int>(0).Should().Be(1);
            result.GetAtIndex<int>(1).Should().Be(2);
            result.GetAtIndex<int>(2).Should().Be(-1);
            result.GetAtIndex<int>(3).Should().Be(-2);
            result.GetAtIndex<int>(4).Should().Be(0);
            result.GetAtIndex<int>(5).Should().Be(0);
            result.GetAtIndex<int>(6).Should().Be(100);
        }

        // ================================================================
        // DECIMAL TO INT (should also truncate)
        // ================================================================

        [Test]
        public void Decimal_ToInt32_Truncation()
        {
            var arr = np.array(new decimal[] { 2.9m, -2.9m, 1.5m, -1.5m });
            var result = arr.astype(np.int32);

            result.GetAtIndex<int>(0).Should().Be(2, "2.9m should truncate to 2");
            result.GetAtIndex<int>(1).Should().Be(-2, "-2.9m should truncate to -2");
            result.GetAtIndex<int>(2).Should().Be(1, "1.5m should truncate to 1");
            result.GetAtIndex<int>(3).Should().Be(-1, "-1.5m should truncate to -1");
        }

        // ================================================================
        // EXACT INTEGER VALUES (no truncation needed)
        // ================================================================

        [Test]
        public void Float64_ToInt32_ExactIntegers_PreservedExactly()
        {
            var arr = np.array(new double[] { 0.0, 1.0, -1.0, 100.0, -100.0,
                int.MaxValue - 100.0, int.MinValue + 100.0 });
            var result = arr.astype(np.int32);

            result.GetAtIndex<int>(0).Should().Be(0);
            result.GetAtIndex<int>(1).Should().Be(1);
            result.GetAtIndex<int>(2).Should().Be(-1);
            result.GetAtIndex<int>(3).Should().Be(100);
            result.GetAtIndex<int>(4).Should().Be(-100);
            result.GetAtIndex<int>(5).Should().Be(int.MaxValue - 100);
            result.GetAtIndex<int>(6).Should().Be(int.MinValue + 100);
        }
    }
}
