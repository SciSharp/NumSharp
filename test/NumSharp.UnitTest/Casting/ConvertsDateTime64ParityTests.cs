using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Utilities;

namespace NumSharp.UnitTest.Casting
{
    /// <summary>
    /// NumPy-parity tests for <see cref="DateTime64"/> conversions in <see cref="Converts"/>.
    ///
    /// <para>
    /// DateTime64 stores a raw int64 tick count with full <c>long.MinValue…long.MaxValue</c>
    /// range. NaT == <c>long.MinValue</c>, matching NumPy's <c>datetime64</c> exactly.
    /// All parity values come from running NumPy 2.4.2 with the same input.
    /// </para>
    ///
    /// <para>
    /// These tests close the 64 diffs identified in the earlier battletest against
    /// <see cref="DateTime"/> which physically cannot hold ticks outside
    /// <c>[0, 3_155_378_975_999_999_999]</c>.
    /// </para>
    /// </summary>
    [TestClass]
    public class ConvertsDateTime64ParityTests
    {
        // DateTime(2024,1,1,0,0,0).Ticks = 638396640000000000
        private const long Jan1_2024_Ticks = 638396640000000000L;

        // ================================================================
        // DateTime64 → primitive (all 12 supported types)
        // Includes the "Group A" cases where raw int64 cannot be held by DateTime.
        // ================================================================

        [TestMethod]
        public void DateTime64_ToInt64_ReturnsRawTicks()
        {
            Converts.ToInt64(new DateTime64(Jan1_2024_Ticks)).Should().Be(Jan1_2024_Ticks);
            Converts.ToInt64(new DateTime64(0L)).Should().Be(0L);
            Converts.ToInt64(new DateTime64(-1L)).Should().Be(-1L);                        // Group A
            Converts.ToInt64(new DateTime64(int.MinValue)).Should().Be(int.MinValue);      // Group A
            Converts.ToInt64(DateTime64.NaT).Should().Be(long.MinValue);                   // Group A (NaT)
            Converts.ToInt64(new DateTime64(long.MaxValue)).Should().Be(long.MaxValue);
        }

        [TestMethod]
        public void DateTime64_ToUInt64_ReinterpretsTicks()
        {
            Converts.ToUInt64(new DateTime64(Jan1_2024_Ticks)).Should().Be((ulong)Jan1_2024_Ticks);
            Converts.ToUInt64(new DateTime64(-1L)).Should().Be(ulong.MaxValue);                   // NumPy uint64 of -1
            Converts.ToUInt64(DateTime64.NaT).Should().Be(9223372036854775808UL);                 // NumPy uint64 of long.MinValue
            Converts.ToUInt64(new DateTime64(int.MinValue)).Should().Be(18446744071562067968UL);  // NumPy
        }

        [TestMethod]
        public void DateTime64_ToInt32_WrapsLowBits()
        {
            // NumPy reference values
            Converts.ToInt32(new DateTime64(Jan1_2024_Ticks)).Should().Be(-1728004096);
            Converts.ToInt32(new DateTime64(-1L)).Should().Be(-1);                       // Group A
            Converts.ToInt32(new DateTime64(int.MinValue)).Should().Be(int.MinValue);    // Group A
            Converts.ToInt32(DateTime64.NaT).Should().Be(0);                             // Group A: low 32 of long.MinValue = 0
            Converts.ToInt32(new DateTime64(long.MaxValue)).Should().Be(-1);             // low 32 of long.MaxValue = -1
        }

        [TestMethod]
        public void DateTime64_ToUInt32_WrapsLowBits()
        {
            Converts.ToUInt32(new DateTime64(Jan1_2024_Ticks)).Should().Be(2566963200u);
            Converts.ToUInt32(new DateTime64(-1L)).Should().Be(uint.MaxValue);             // Group A
            Converts.ToUInt32(new DateTime64(int.MinValue)).Should().Be(2147483648u);      // Group A
            Converts.ToUInt32(DateTime64.NaT).Should().Be(0u);                             // Group A
        }

        [TestMethod]
        public void DateTime64_ToInt16_WrapsLowBits()
        {
            Converts.ToInt16(new DateTime64(Jan1_2024_Ticks)).Should().Be((short)-16384);
            Converts.ToInt16(new DateTime64(-1L)).Should().Be((short)-1);                  // Group A
            Converts.ToInt16(new DateTime64(int.MinValue)).Should().Be((short)0);          // Group A (low 16 = 0)
            Converts.ToInt16(DateTime64.NaT).Should().Be((short)0);                        // Group A
        }

        [TestMethod]
        public void DateTime64_ToUInt16_WrapsLowBits()
        {
            Converts.ToUInt16(new DateTime64(Jan1_2024_Ticks)).Should().Be((ushort)49152);
            Converts.ToUInt16(new DateTime64(-1L)).Should().Be(ushort.MaxValue);           // Group A
            Converts.ToUInt16(new DateTime64(int.MinValue)).Should().Be((ushort)0);        // Group A
            Converts.ToUInt16(DateTime64.NaT).Should().Be((ushort)0);                      // Group A
        }

        [TestMethod]
        public void DateTime64_ToSByte_WrapsLowByte()
        {
            Converts.ToSByte(new DateTime64(Jan1_2024_Ticks)).Should().Be((sbyte)0);
            Converts.ToSByte(new DateTime64(-1L)).Should().Be((sbyte)-1);                  // Group A
            Converts.ToSByte(new DateTime64(int.MinValue)).Should().Be((sbyte)0);          // Group A
            Converts.ToSByte(DateTime64.NaT).Should().Be((sbyte)0);                        // Group A
            Converts.ToSByte(new DateTime64(0xFFL)).Should().Be((sbyte)-1);
        }

        [TestMethod]
        public void DateTime64_ToByte_WrapsLowByte()
        {
            Converts.ToByte(new DateTime64(Jan1_2024_Ticks)).Should().Be((byte)0);
            Converts.ToByte(new DateTime64(-1L)).Should().Be((byte)255);                   // Group A
            Converts.ToByte(new DateTime64(int.MinValue)).Should().Be((byte)0);            // Group A
            Converts.ToByte(DateTime64.NaT).Should().Be((byte)0);                          // Group A
            Converts.ToByte(new DateTime64(0xFFL)).Should().Be((byte)0xFF);
        }

        [TestMethod]
        public void DateTime64_ToChar_WrapsLow16()
        {
            Converts.ToChar(new DateTime64(Jan1_2024_Ticks)).Should().Be((char)49152);
            Converts.ToChar(new DateTime64(-1L)).Should().Be((char)65535);                 // Group A
            Converts.ToChar(DateTime64.NaT).Should().Be((char)0);                          // Group A
        }

        [TestMethod]
        public void DateTime64_ToBoolean_TrueIfTicksNonzero()
        {
            Converts.ToBoolean(new DateTime64(Jan1_2024_Ticks)).Should().BeTrue();
            Converts.ToBoolean(new DateTime64(0L)).Should().BeFalse();
            Converts.ToBoolean(new DateTime64(-1L)).Should().BeTrue();                     // Group A
            Converts.ToBoolean(DateTime64.NaT).Should().BeTrue();                          // NumPy: bool(NaT) = True
            Converts.ToBoolean(new DateTime64(long.MaxValue)).Should().BeTrue();
        }

        [TestMethod]
        public void DateTime64_ToDouble_AsDouble()
        {
            Converts.ToDouble(new DateTime64(Jan1_2024_Ticks)).Should().Be((double)Jan1_2024_Ticks);
            Converts.ToDouble(new DateTime64(-1L)).Should().Be(-1.0);                      // Group A
            Converts.ToDouble(new DateTime64(long.MaxValue)).Should().Be(9.223372036854776e18);
            Converts.ToDouble(DateTime64.NaT).Should().Be(-9.223372036854776e18);          // Group A
        }

        [TestMethod]
        public void DateTime64_ToSingle_AsFloat()
        {
            Converts.ToSingle(new DateTime64(-1L)).Should().Be(-1f);
            Converts.ToSingle(new DateTime64(0L)).Should().Be(0f);
            Converts.ToSingle(DateTime64.NaT).Should().Be(-9.223372e18f);                  // Group A
        }

        [TestMethod]
        public void DateTime64_ToDecimal_AsDecimal()
        {
            Converts.ToDecimal(new DateTime64(Jan1_2024_Ticks)).Should().Be((decimal)Jan1_2024_Ticks);
            Converts.ToDecimal(new DateTime64(-1L)).Should().Be(-1m);
            Converts.ToDecimal(DateTime64.NaT).Should().Be((decimal)long.MinValue);
        }

        [TestMethod]
        public void DateTime64_ToHalf_ViaDouble()
        {
            Converts.ToHalf(new DateTime64(0L)).Should().Be((Half)0);
            Converts.ToHalf(new DateTime64(1L)).Should().Be((Half)1);
            Converts.ToHalf(new DateTime64(-1L)).Should().Be((Half)(-1));
            Half.IsInfinity(Converts.ToHalf(new DateTime64(Jan1_2024_Ticks))).Should().BeTrue();
            Half.IsNegativeInfinity(Converts.ToHalf(DateTime64.NaT)).Should().BeTrue();
        }

        [TestMethod]
        public void DateTime64_ToComplex_RealOnly()
        {
            var c = Converts.ToComplex(new DateTime64(Jan1_2024_Ticks));
            c.Real.Should().Be((double)Jan1_2024_Ticks);
            c.Imaginary.Should().Be(0);

            var cNaT = Converts.ToComplex(DateTime64.NaT);
            cNaT.Real.Should().Be((double)long.MinValue);
            cNaT.Imaginary.Should().Be(0);
        }

        // ================================================================
        // primitive → DateTime64 (the "Group B" diffs: dst=dt64)
        // ================================================================

        [TestMethod]
        public void ToDateTime64_FromInt64_Exact()
        {
            Converts.ToDateTime64(0L).Ticks.Should().Be(0L);
            Converts.ToDateTime64(-1L).Ticks.Should().Be(-1L);                             // Group B
            Converts.ToDateTime64(long.MinValue).IsNaT.Should().BeTrue();                  // Group B (NaT)
            Converts.ToDateTime64(long.MaxValue).Ticks.Should().Be(long.MaxValue);         // Group B
        }

        [TestMethod]
        public void ToDateTime64_FromInt32_SignExtend()
        {
            Converts.ToDateTime64(-1).Ticks.Should().Be(-1L);                              // Group B
            Converts.ToDateTime64(int.MinValue).Ticks.Should().Be(int.MinValue);           // Group B
            Converts.ToDateTime64(int.MaxValue).Ticks.Should().Be(int.MaxValue);
        }

        [TestMethod]
        public void ToDateTime64_FromSmallSignedInts_SignExtend()
        {
            Converts.ToDateTime64((sbyte)-1).Ticks.Should().Be(-1L);
            Converts.ToDateTime64((short)-1).Ticks.Should().Be(-1L);
            Converts.ToDateTime64((sbyte)sbyte.MinValue).Ticks.Should().Be(sbyte.MinValue);
            Converts.ToDateTime64((short)short.MinValue).Ticks.Should().Be(short.MinValue);
        }

        [TestMethod]
        public void ToDateTime64_FromUnsignedInts_ZeroExtend()
        {
            Converts.ToDateTime64((byte)255).Ticks.Should().Be(255L);
            Converts.ToDateTime64((ushort)65535).Ticks.Should().Be(65535L);
            Converts.ToDateTime64(uint.MaxValue).Ticks.Should().Be(4294967295L);
            Converts.ToDateTime64(ulong.MaxValue).Ticks.Should().Be(-1L);                  // reinterpret: matches NumPy
            // NumPy: uint64(9223372036854775808) → dt64 = long.MinValue = NaT
            Converts.ToDateTime64(9223372036854775808UL).IsNaT.Should().BeTrue();
        }

        [TestMethod]
        public void ToDateTime64_FromFloat_NaNInfOverflow_ToNaT()
        {
            // NumPy: NaN, ±Inf, overflow → NaT (long.MinValue)
            Converts.ToDateTime64(double.NaN).IsNaT.Should().BeTrue();                     // Group B
            Converts.ToDateTime64(double.PositiveInfinity).IsNaT.Should().BeTrue();        // Group B
            Converts.ToDateTime64(double.NegativeInfinity).IsNaT.Should().BeTrue();        // Group B
            Converts.ToDateTime64(1e20).IsNaT.Should().BeTrue();                           // Group B
            Converts.ToDateTime64(-1e20).IsNaT.Should().BeTrue();                          // Group B
            Converts.ToDateTime64(0.0).Ticks.Should().Be(0L);
            Converts.ToDateTime64(-1.0).Ticks.Should().Be(-1L);
            Converts.ToDateTime64(1234567890.0).Ticks.Should().Be(1234567890L);
        }

        [TestMethod]
        public void ToDateTime64_FromSingle_NaNInfOverflow_ToNaT()
        {
            Converts.ToDateTime64(float.NaN).IsNaT.Should().BeTrue();
            Converts.ToDateTime64(float.PositiveInfinity).IsNaT.Should().BeTrue();
            Converts.ToDateTime64(float.NegativeInfinity).IsNaT.Should().BeTrue();
            Converts.ToDateTime64(1e20f).IsNaT.Should().BeTrue();
        }

        [TestMethod]
        public void ToDateTime64_FromHalf_NaNInf_ToNaT()
        {
            Converts.ToDateTime64(Half.NaN).IsNaT.Should().BeTrue();
            Converts.ToDateTime64(Half.PositiveInfinity).IsNaT.Should().BeTrue();
            Converts.ToDateTime64(Half.NegativeInfinity).IsNaT.Should().BeTrue();
            Converts.ToDateTime64((Half)1).Ticks.Should().Be(1L);
        }

        [TestMethod]
        public void ToDateTime64_FromDecimal_Exact()
        {
            Converts.ToDateTime64((decimal)long.MaxValue).Ticks.Should().Be(long.MaxValue);
            Converts.ToDateTime64(-1m).Ticks.Should().Be(-1L);
            Converts.ToDateTime64(0m).Ticks.Should().Be(0L);
            // Out of range decimal → NaT
            Converts.ToDateTime64(1e28m).IsNaT.Should().BeTrue();
        }

        [TestMethod]
        public void ToDateTime64_FromBool_ZeroOrOne()
        {
            Converts.ToDateTime64(false).Ticks.Should().Be(0L);
            Converts.ToDateTime64(true).Ticks.Should().Be(1L);
        }

        [TestMethod]
        public void ToDateTime64_FromChar_ZeroExtend()
        {
            Converts.ToDateTime64('A').Ticks.Should().Be(65L);
            Converts.ToDateTime64('\0').Ticks.Should().Be(0L);
            Converts.ToDateTime64((char)65535).Ticks.Should().Be(65535L);
        }

        [TestMethod]
        public void ToDateTime64_FromComplex_RealPart()
        {
            Converts.ToDateTime64(new Complex(42, 99)).Ticks.Should().Be(42L);
            Converts.ToDateTime64(new Complex(double.NaN, 0)).IsNaT.Should().BeTrue();
            Converts.ToDateTime64(new Complex(1e20, 0)).IsNaT.Should().BeTrue();
        }

        // ================================================================
        // DateTime64 ↔ DateTime / DateTimeOffset interop
        // ================================================================

        [TestMethod]
        public void DateTime_To_DateTime64_Implicit()
        {
            DateTime64 d64 = new DateTime(2024, 1, 1);
            d64.Ticks.Should().Be(Jan1_2024_Ticks);
        }

        [TestMethod]
        public void DateTime64_To_DateTime_Explicit_Valid()
        {
            var d64 = new DateTime64(Jan1_2024_Ticks);
            DateTime dt = (DateTime)d64;
            dt.Ticks.Should().Be(Jan1_2024_Ticks);
        }

        [TestMethod]
        public void DateTime64_To_DateTime_Explicit_NaT_Throws()
        {
            Action act = () => { DateTime _ = (DateTime)DateTime64.NaT; };
            act.Should().Throw<InvalidOperationException>();
        }

        [TestMethod]
        public void DateTime64_To_DateTime_Explicit_OutOfRange_Throws()
        {
            Action act = () => { DateTime _ = (DateTime)new DateTime64(-1L); };
            act.Should().Throw<InvalidOperationException>();
        }

        [TestMethod]
        public void DateTime64_ToDateTime_WithFallback_ClampsNaT()
        {
            DateTime64.NaT.ToDateTime(DateTime.MinValue).Should().Be(DateTime.MinValue);
            new DateTime64(-1L).ToDateTime(DateTime.MinValue).Should().Be(DateTime.MinValue);
            new DateTime64(Jan1_2024_Ticks).ToDateTime(DateTime.MinValue).Should().Be(new DateTime(2024, 1, 1));
        }

        [TestMethod]
        public void DateTimeOffset_To_DateTime64_UsesUtcTicks()
        {
            var dto = new DateTimeOffset(2024, 1, 1, 12, 0, 0, TimeSpan.FromHours(5));
            DateTime64 d64 = dto;
            // UTC is 2024-01-01 07:00:00 — 7 hours after midnight 2024-01-01
            d64.Ticks.Should().Be(Jan1_2024_Ticks + 7 * TimeSpan.TicksPerHour);
        }

        [TestMethod]
        public void DateTime64_To_DateTimeOffset_Explicit_Valid()
        {
            var d64 = new DateTime64(Jan1_2024_Ticks);
            DateTimeOffset dto = (DateTimeOffset)d64;
            dto.UtcTicks.Should().Be(Jan1_2024_Ticks);
            dto.Offset.Should().Be(TimeSpan.Zero);
        }

        [TestMethod]
        public void Long_To_DateTime64_Implicit()
        {
            DateTime64 d64 = 12345L;
            d64.Ticks.Should().Be(12345L);
        }

        [TestMethod]
        public void DateTime64_To_Long_Explicit()
        {
            var d64 = new DateTime64(Jan1_2024_Ticks);
            long ticks = (long)d64;
            ticks.Should().Be(Jan1_2024_Ticks);

            ((long)DateTime64.NaT).Should().Be(long.MinValue);
        }

        // ================================================================
        // NaT semantics (NumPy parity)
        // ================================================================

        [TestMethod]
        public void NaT_EqualityFollowsNumPy()
        {
            // NumPy: NaT == NaT → False (NaN-like)
            DateTime64.NaT.Equals(DateTime64.NaT).Should().BeFalse();
            (DateTime64.NaT == DateTime64.NaT).Should().BeFalse();
            (DateTime64.NaT != DateTime64.NaT).Should().BeTrue();

            // NaT != value
            (DateTime64.NaT == new DateTime64(0L)).Should().BeFalse();
            (DateTime64.NaT != new DateTime64(0L)).Should().BeTrue();
        }

        [TestMethod]
        public void NaT_ComparisonsFalse()
        {
            // NumPy: any comparison with NaT → False
            (DateTime64.NaT < new DateTime64(0L)).Should().BeFalse();
            (DateTime64.NaT > new DateTime64(0L)).Should().BeFalse();
            (DateTime64.NaT <= new DateTime64(0L)).Should().BeFalse();
            (DateTime64.NaT >= new DateTime64(0L)).Should().BeFalse();
        }

        [TestMethod]
        public void NaT_ArithmeticPropagates()
        {
            (DateTime64.NaT + TimeSpan.FromDays(1)).IsNaT.Should().BeTrue();
            (DateTime64.NaT - TimeSpan.FromDays(1)).IsNaT.Should().BeTrue();
            (DateTime64.NaT.AddDays(1)).IsNaT.Should().BeTrue();
            (DateTime64.NaT.AddHours(1)).IsNaT.Should().BeTrue();
        }

        // ================================================================
        // Formatting
        // ================================================================

        [TestMethod]
        public void ToString_NaT_ReturnsNaT()
        {
            DateTime64.NaT.ToString().Should().Be("NaT");
        }

        [TestMethod]
        public void ToString_ValidDate_IsISO8601()
        {
            new DateTime64(Jan1_2024_Ticks).ToString().Should().Contain("2024-01-01");
        }

        [TestMethod]
        public void ToString_OutOfRange_IncludesTicks()
        {
            new DateTime64(-1L).ToString().Should().Contain("-1");
            new DateTime64(long.MaxValue).ToString().Should().Contain(long.MaxValue.ToString());
        }

        // ================================================================
        // object dispatcher
        // ================================================================

        [TestMethod]
        public void ObjectDispatch_ToDateTime64_HandlesAllSources()
        {
            Converts.ToDateTime64((object)-1L).Ticks.Should().Be(-1L);
            Converts.ToDateTime64((object)1.0).Ticks.Should().Be(1L);
            Converts.ToDateTime64((object)double.NaN).IsNaT.Should().BeTrue();
            Converts.ToDateTime64((object)"NaT").IsNaT.Should().BeTrue();
            Converts.ToDateTime64((object)new DateTime(2024, 1, 1)).Ticks.Should().Be(Jan1_2024_Ticks);
            Converts.ToDateTime64(null!).IsNaT.Should().BeTrue();
        }

        [TestMethod]
        public void ObjectDispatch_ToX_HandlesDateTime64Source()
        {
            object d64 = new DateTime64(-1L);
            Converts.ToInt64(d64).Should().Be(-1L);
            Converts.ToInt32(d64).Should().Be(-1);
            Converts.ToSByte(d64).Should().Be((sbyte)-1);
            Converts.ToBoolean(d64).Should().BeTrue();
            Converts.ToDouble(d64).Should().Be(-1.0);

            object nat = DateTime64.NaT;
            Converts.ToInt64(nat).Should().Be(long.MinValue);
            Converts.ToBoolean(nat).Should().BeTrue();
        }

        // ================================================================
        // InfoOf — verify InfoOf<DateTime64> doesn't throw and DateTime's
        // old collision with NPTypeCode.Half is gone.
        // ================================================================

        [TestMethod]
        public void InfoOf_DateTime_NotHalfAnymore()
        {
            InfoOf<DateTime>.NPTypeCode.Should().Be(NPTypeCode.Empty);
            InfoOf<DateTime>.Size.Should().Be(8);
        }

        [TestMethod]
        public void InfoOf_DateTime64_IsEmpty()
        {
            InfoOf<DateTime64>.NPTypeCode.Should().Be(NPTypeCode.Empty);
            InfoOf<DateTime64>.Size.Should().Be(8);
        }

        [TestMethod]
        public void InfoOf_TimeSpan_IsEmpty()
        {
            InfoOf<TimeSpan>.NPTypeCode.Should().Be(NPTypeCode.Empty);
            InfoOf<TimeSpan>.Size.Should().Be(8);
        }
    }
}
