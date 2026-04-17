using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Utilities;

namespace NumSharp.UnitTest.Casting
{
    /// <summary>
    /// NumPy-parity tests for DateTime and TimeSpan conversions in Converts.
    ///
    /// Background (not NumPy dtypes, but conversions must still behave):
    /// - NumPy datetime64 / timedelta64 are int64 internally; all casts go through int64.
    /// - NaT = int64.MinValue. bool(dt/td) = (internal_int64 != 0). NaN/Inf -> NaT.
    ///
    /// .NET mapping (cross-checked with live NumPy 2.4.2 output):
    /// - DateTime.Ticks IS the int64 representation (valid range [0, DateTime.MaxValue.Ticks]).
    /// - TimeSpan.Ticks IS the int64 representation (full long range — TimeSpan.MinValue
    ///   matches NumPy NaT exactly since TimeSpan.MinValue.Ticks == long.MinValue).
    /// - DateTime cannot represent negative ticks or NaT; collapses to DateTime.MinValue.
    /// - TimeSpan has full int64 range so maps NumPy semantics 1-to-1.
    ///
    /// All parity values come from running NumPy 2.4.2 with the same input.
    /// </summary>
    [TestClass]
    public class ConvertsDateTimeParityTests
    {
        // DateTime(2024,1,1,0,0,0).Ticks = 638396640000000000
        private static readonly DateTime Jan1_2024 = new DateTime(2024, 1, 1);
        private const long Jan1_2024_Ticks = 638396640000000000L;

        #region DateTime -> primitive (via Ticks, wrap on overflow)

        // NumPy parity table for datetime64 with int64=Ticks=638396640000000000:
        //   int8=0 (low byte is 0x00), uint8=0
        //   int16=-16384 (low 16 bits), uint16=49152
        //   int32=-1728004096 (low 32 bits), uint32=2566963200
        //   int64=638396640000000000, uint64=638396640000000000
        //   bool=True (nonzero), float/double=(double)ticks
        // Matches NumPy's int64->int32 and int64->int16 wrapping behavior exactly.

        [TestMethod]
        public void DateTime_ToInt64_ReturnsTicks()
        {
            Converts.ToInt64(Jan1_2024).Should().Be(Jan1_2024_Ticks);
            Converts.ToInt64(DateTime.MinValue).Should().Be(0L);
            Converts.ToInt64(DateTime.MaxValue).Should().Be(DateTime.MaxValue.Ticks);
        }

        [TestMethod]
        public void DateTime_ToUInt64_ReturnsTicksUnchecked()
        {
            Converts.ToUInt64(Jan1_2024).Should().Be((ulong)Jan1_2024_Ticks);
            Converts.ToUInt64(DateTime.MinValue).Should().Be(0UL);
        }

        [TestMethod]
        public void DateTime_ToInt32_WrapsLowBits()
        {
            // NumPy: int64 638396640000000000 -> int32 = -1728004096
            Converts.ToInt32(Jan1_2024).Should().Be(-1728004096);
            Converts.ToInt32(DateTime.MinValue).Should().Be(0);
        }

        [TestMethod]
        public void DateTime_ToUInt32_WrapsLowBits()
        {
            // NumPy: int64 638396640000000000 -> uint32 = 2566963200
            Converts.ToUInt32(Jan1_2024).Should().Be(2566963200u);
        }

        [TestMethod]
        public void DateTime_ToInt16_WrapsLowBits()
        {
            // NumPy: int64 638396640000000000 -> int16 = -16384
            Converts.ToInt16(Jan1_2024).Should().Be(-16384);
            Converts.ToInt16(DateTime.MinValue).Should().Be(0);
        }

        [TestMethod]
        public void DateTime_ToUInt16_WrapsLowBits()
        {
            // NumPy: int64 638396640000000000 -> uint16 = 49152
            Converts.ToUInt16(Jan1_2024).Should().Be((ushort)49152);
        }

        [TestMethod]
        public void DateTime_ToSByte_WrapsLowByte()
        {
            // 638396640000000000 & 0xFF = 0
            Converts.ToSByte(Jan1_2024).Should().Be((sbyte)0);
            Converts.ToSByte(DateTime.MinValue).Should().Be((sbyte)0);
            // DateTime with ticks ending in nonzero low byte
            Converts.ToSByte(new DateTime(1L)).Should().Be((sbyte)1);
            Converts.ToSByte(new DateTime(0xFFL)).Should().Be((sbyte)-1);
        }

        [TestMethod]
        public void DateTime_ToByte_WrapsLowByte()
        {
            Converts.ToByte(Jan1_2024).Should().Be((byte)0);
            Converts.ToByte(new DateTime(0xFFL)).Should().Be((byte)0xFF);
            Converts.ToByte(new DateTime(0x100L)).Should().Be((byte)0);
        }

        [TestMethod]
        public void DateTime_ToChar_WrapsLow16()
        {
            Converts.ToChar(Jan1_2024).Should().Be((char)49152);
            Converts.ToChar(DateTime.MinValue).Should().Be((char)0);
            Converts.ToChar(new DateTime(1L)).Should().Be((char)1);
        }

        [TestMethod]
        public void DateTime_ToBoolean_TrueIfTicksNonzero()
        {
            Converts.ToBoolean(Jan1_2024).Should().BeTrue();
            Converts.ToBoolean(DateTime.MinValue).Should().BeFalse();
            Converts.ToBoolean(new DateTime(1L)).Should().BeTrue();
        }

        [TestMethod]
        public void DateTime_ToDouble_AsDouble()
        {
            Converts.ToDouble(Jan1_2024).Should().Be((double)Jan1_2024_Ticks);
            Converts.ToDouble(DateTime.MinValue).Should().Be(0.0);
        }

        [TestMethod]
        public void DateTime_ToSingle_AsFloat()
        {
            Converts.ToSingle(Jan1_2024).Should().Be((float)Jan1_2024_Ticks);
            Converts.ToSingle(DateTime.MinValue).Should().Be(0f);
        }

        [TestMethod]
        public void DateTime_ToDecimal_AsDecimal()
        {
            Converts.ToDecimal(Jan1_2024).Should().Be((decimal)Jan1_2024_Ticks);
            Converts.ToDecimal(DateTime.MinValue).Should().Be(0m);
        }

        [TestMethod]
        public void DateTime_ToHalf_ViaDouble()
        {
            // DateTime.Ticks for modern dates overflows Half — NumPy returns inf
            Half.IsInfinity(Converts.ToHalf(Jan1_2024)).Should().BeTrue();
            Converts.ToHalf(DateTime.MinValue).Should().Be((Half)0);
            Converts.ToHalf(new DateTime(1L)).Should().Be((Half)1);
        }

        [TestMethod]
        public void DateTime_ToComplex_RealOnly()
        {
            var r = Converts.ToComplex(Jan1_2024);
            r.Real.Should().Be((double)Jan1_2024_Ticks);
            r.Imaginary.Should().Be(0);
        }

        #endregion

        #region TimeSpan -> primitive (via Ticks, full int64 range; NaT=long.MinValue)

        private static readonly TimeSpan Hundred_Sec = TimeSpan.FromSeconds(100);
        private const long Hundred_Sec_Ticks = 1000000000L;

        [TestMethod]
        public void TimeSpan_ToInt64_ReturnsTicks()
        {
            Converts.ToInt64(Hundred_Sec).Should().Be(Hundred_Sec_Ticks);
            Converts.ToInt64(TimeSpan.Zero).Should().Be(0L);
            Converts.ToInt64(TimeSpan.MinValue).Should().Be(long.MinValue);   // NaT parity
            Converts.ToInt64(TimeSpan.MaxValue).Should().Be(long.MaxValue);
        }

        [TestMethod]
        public void TimeSpan_ToUInt64_WrapsTicks()
        {
            Converts.ToUInt64(Hundred_Sec).Should().Be((ulong)Hundred_Sec_Ticks);
            Converts.ToUInt64(new TimeSpan(-1L)).Should().Be(ulong.MaxValue); // -1 wraps
        }

        [TestMethod]
        public void TimeSpan_ToBoolean_TrueIfTicksNonzero()
        {
            // NumPy: bool(timedelta64) = int64 != 0. NaT is also True (MinValue != 0).
            Converts.ToBoolean(TimeSpan.Zero).Should().BeFalse();
            Converts.ToBoolean(Hundred_Sec).Should().BeTrue();
            Converts.ToBoolean(new TimeSpan(-1L)).Should().BeTrue();
            Converts.ToBoolean(TimeSpan.MinValue).Should().BeTrue();  // NaT -> True
        }

        [TestMethod]
        public void TimeSpan_ToInt32_WrapsLowBits()
        {
            Converts.ToInt32(Hundred_Sec).Should().Be(unchecked((int)Hundred_Sec_Ticks));
            Converts.ToInt32(TimeSpan.MinValue).Should().Be(0);  // low 32 of int64.MinValue = 0
        }

        [TestMethod]
        public void TimeSpan_ToInt16_WrapsLowBits()
        {
            Converts.ToInt16(Hundred_Sec).Should().Be(unchecked((short)Hundred_Sec_Ticks));
            Converts.ToInt16(TimeSpan.MinValue).Should().Be(0);
        }

        [TestMethod]
        public void TimeSpan_ToByte_WrapsLowByte()
        {
            Converts.ToByte(Hundred_Sec).Should().Be(unchecked((byte)Hundred_Sec_Ticks));
            Converts.ToByte(TimeSpan.MinValue).Should().Be((byte)0);
        }

        [TestMethod]
        public void TimeSpan_ToDouble_AsDouble()
        {
            Converts.ToDouble(Hundred_Sec).Should().Be((double)Hundred_Sec_Ticks);
            Converts.ToDouble(TimeSpan.MinValue).Should().Be((double)long.MinValue);
        }

        [TestMethod]
        public void TimeSpan_ToDecimal_AsDecimal()
        {
            Converts.ToDecimal(Hundred_Sec).Should().Be((decimal)Hundred_Sec_Ticks);
            Converts.ToDecimal(TimeSpan.Zero).Should().Be(0m);
        }

        [TestMethod]
        public void TimeSpan_ToComplex_RealOnly()
        {
            var r = Converts.ToComplex(Hundred_Sec);
            r.Real.Should().Be((double)Hundred_Sec_Ticks);
            r.Imaginary.Should().Be(0);
        }

        #endregion

        #region primitive -> DateTime (interpret as Ticks, clamp on overflow)

        [TestMethod]
        public void IntegerToDateTime_InterpretAsTicks()
        {
            Converts.ToDateTime(0L).Should().Be(DateTime.MinValue);
            Converts.ToDateTime(1L).Ticks.Should().Be(1L);
            Converts.ToDateTime(Jan1_2024_Ticks).Should().Be(Jan1_2024);
        }

        [TestMethod]
        public void NegativeIntegerToDateTime_ClampsToMinValue()
        {
            // .NET DateTime cannot be negative — collapse to MinValue (NaT-like)
            Converts.ToDateTime(-1L).Should().Be(DateTime.MinValue);
            Converts.ToDateTime(long.MinValue).Should().Be(DateTime.MinValue);
        }

        [TestMethod]
        public void TooLargeIntegerToDateTime_ClampsToMinValue()
        {
            // ticks > DateTime.MaxValue.Ticks — invalid, map to MinValue (NaT-like)
            Converts.ToDateTime(long.MaxValue).Should().Be(DateTime.MinValue);
        }

        [TestMethod]
        public void BoolToDateTime()
        {
            Converts.ToDateTime(false).Should().Be(DateTime.MinValue);
            Converts.ToDateTime(true).Ticks.Should().Be(1L);
        }

        [TestMethod]
        public void DoubleToDateTime_NaNAndInfToMinValue()
        {
            Converts.ToDateTime(double.NaN).Should().Be(DateTime.MinValue);
            Converts.ToDateTime(double.PositiveInfinity).Should().Be(DateTime.MinValue);
            Converts.ToDateTime(double.NegativeInfinity).Should().Be(DateTime.MinValue);
        }

        [TestMethod]
        public void DoubleToDateTime_Normal()
        {
            Converts.ToDateTime(1.7d).Ticks.Should().Be(1L);  // truncate toward zero
            Converts.ToDateTime((double)Jan1_2024_Ticks).Should().BeOnOrAfter(Jan1_2024.AddSeconds(-1));
        }

        [TestMethod]
        public void HalfToDateTime_NaNAndInfToMinValue()
        {
            Converts.ToDateTime(Half.NaN).Should().Be(DateTime.MinValue);
            Converts.ToDateTime(Half.PositiveInfinity).Should().Be(DateTime.MinValue);
            Converts.ToDateTime((Half)42).Ticks.Should().Be(42L);
        }

        [TestMethod]
        public void ComplexToDateTime_UsesReal()
        {
            Converts.ToDateTime(new Complex(100, 99)).Ticks.Should().Be(100L);
            Converts.ToDateTime(new Complex(double.NaN, 0)).Should().Be(DateTime.MinValue);
        }

        [TestMethod]
        public void DecimalToDateTime_Truncates()
        {
            Converts.ToDateTime(1.7m).Ticks.Should().Be(1L);
            Converts.ToDateTime(-1m).Should().Be(DateTime.MinValue);
        }

        #endregion

        #region primitive -> TimeSpan (interpret as Ticks, full int64 range)

        [TestMethod]
        public void IntegerToTimeSpan_InterpretAsTicks()
        {
            Converts.ToTimeSpan(0L).Should().Be(TimeSpan.Zero);
            Converts.ToTimeSpan(1L).Ticks.Should().Be(1L);
            Converts.ToTimeSpan(-1L).Ticks.Should().Be(-1L);
            Converts.ToTimeSpan(long.MaxValue).Should().Be(TimeSpan.MaxValue);
            Converts.ToTimeSpan(long.MinValue).Should().Be(TimeSpan.MinValue);  // NaT parity
        }

        [TestMethod]
        public void BoolToTimeSpan()
        {
            Converts.ToTimeSpan(false).Should().Be(TimeSpan.Zero);
            Converts.ToTimeSpan(true).Ticks.Should().Be(1L);
        }

        [TestMethod]
        public void DoubleToTimeSpan_NaNAndInfToNaT()
        {
            // NumPy: NaN/Inf -> NaT (int64.MinValue) = TimeSpan.MinValue (EXACT parity)
            Converts.ToTimeSpan(double.NaN).Should().Be(TimeSpan.MinValue);
            Converts.ToTimeSpan(double.PositiveInfinity).Should().Be(TimeSpan.MinValue);
            Converts.ToTimeSpan(double.NegativeInfinity).Should().Be(TimeSpan.MinValue);
        }

        [TestMethod]
        public void DoubleToTimeSpan_Normal()
        {
            Converts.ToTimeSpan(1.7d).Ticks.Should().Be(1L);
            Converts.ToTimeSpan(-1.7d).Ticks.Should().Be(-1L);
        }

        [TestMethod]
        public void HalfToTimeSpan_NaNAndInf()
        {
            Converts.ToTimeSpan(Half.NaN).Should().Be(TimeSpan.MinValue);
            Converts.ToTimeSpan(Half.PositiveInfinity).Should().Be(TimeSpan.MinValue);
            Converts.ToTimeSpan(Half.NegativeInfinity).Should().Be(TimeSpan.MinValue);
            Converts.ToTimeSpan((Half)42).Ticks.Should().Be(42L);
        }

        [TestMethod]
        public void DecimalToTimeSpan_Truncates()
        {
            Converts.ToTimeSpan(1.7m).Ticks.Should().Be(1L);
            Converts.ToTimeSpan(-1m).Ticks.Should().Be(-1L);
        }

        [TestMethod]
        public void ComplexToTimeSpan_UsesReal()
        {
            Converts.ToTimeSpan(new Complex(100, 99)).Ticks.Should().Be(100L);
            Converts.ToTimeSpan(new Complex(double.NaN, 0)).Should().Be(TimeSpan.MinValue);
        }

        #endregion

        #region DateTime <-> TimeSpan cross conversion (parity with dt64 <-> td64)

        [TestMethod]
        public void DateTimeToTimeSpan_SharesTicks()
        {
            Converts.ToTimeSpan(Jan1_2024).Ticks.Should().Be(Jan1_2024_Ticks);
            Converts.ToTimeSpan(DateTime.MinValue).Should().Be(TimeSpan.Zero);
        }

        [TestMethod]
        public void TimeSpanToDateTime_SharesTicks()
        {
            Converts.ToDateTime(Hundred_Sec).Ticks.Should().Be(Hundred_Sec_Ticks);
            // TimeSpan with negative or oversized ticks collapses
            Converts.ToDateTime(TimeSpan.MinValue).Should().Be(DateTime.MinValue);
        }

        #endregion

        #region Object dispatch (ToXxx(object) covers DateTime/TimeSpan)

        [TestMethod]
        public void ObjectDispatch_DateTime_ToInt64()
        {
            Converts.ToInt64((object)Jan1_2024).Should().Be(Jan1_2024_Ticks);
        }

        [TestMethod]
        public void ObjectDispatch_TimeSpan_ToInt64()
        {
            Converts.ToInt64((object)Hundred_Sec).Should().Be(Hundred_Sec_Ticks);
        }

        [TestMethod]
        public void ObjectDispatch_DateTime_ToBoolean()
        {
            Converts.ToBoolean((object)DateTime.MinValue).Should().BeFalse();
            Converts.ToBoolean((object)Jan1_2024).Should().BeTrue();
        }

        [TestMethod]
        public void ObjectDispatch_TimeSpan_ToBoolean()
        {
            Converts.ToBoolean((object)TimeSpan.Zero).Should().BeFalse();
            Converts.ToBoolean((object)TimeSpan.MinValue).Should().BeTrue();  // NaT -> True
        }

        [TestMethod]
        public void ObjectDispatch_DateTime_ToDouble()
        {
            Converts.ToDouble((object)Jan1_2024).Should().Be((double)Jan1_2024_Ticks);
        }

        [TestMethod]
        public void ObjectDispatch_TimeSpan_ToDouble()
        {
            Converts.ToDouble((object)TimeSpan.MinValue).Should().Be((double)long.MinValue);
        }

        [TestMethod]
        public void ObjectDispatch_LongToDateTime()
        {
            var r = Converts.ToDateTime((object)Jan1_2024_Ticks);
            r.Should().Be(Jan1_2024);
        }

        [TestMethod]
        public void ObjectDispatch_DoubleToTimeSpanNaT()
        {
            var r = Converts.ToTimeSpan((object)double.NaN);
            r.Should().Be(TimeSpan.MinValue);
        }

        [TestMethod]
        public void ObjectDispatch_Null_ReturnsMinOrZero()
        {
            Converts.ToDateTime((object)null).Should().Be(DateTime.MinValue);
            Converts.ToTimeSpan((object)null).Should().Be(TimeSpan.Zero);
        }

        #endregion

        #region ChangeType integration

        [TestMethod]
        public void ChangeType_DateTimeToInt64_UsesTicks()
        {
            var r = Converts.ChangeType((object)Jan1_2024, NPTypeCode.Int64);
            r.Should().Be(Jan1_2024_Ticks);
        }

        [TestMethod]
        public void ChangeType_TimeSpanToInt64_UsesTicks()
        {
            var r = Converts.ChangeType((object)Hundred_Sec, NPTypeCode.Int64);
            r.Should().Be(Hundred_Sec_Ticks);
        }

        [TestMethod]
        public void ChangeType_TimeSpanNaTToBool()
        {
            // NumPy: bool(NaT) = True.
            var r = Converts.ChangeType((object)TimeSpan.MinValue, NPTypeCode.Boolean);
            r.Should().Be(true);
        }

        [TestMethod]
        public void ChangeType_DoubleNaNToDateTime()
        {
            var r = Converts.ChangeType((object)double.NaN, TypeCode.DateTime);
            r.Should().Be(DateTime.MinValue);
        }

        [TestMethod]
        public void ChangeType_LongToDateTime()
        {
            var r = Converts.ChangeType((object)Jan1_2024_Ticks, TypeCode.DateTime);
            r.Should().Be(Jan1_2024);
        }

        [TestMethod]
        public void ChangeType_BoolToDateTime()
        {
            var r = Converts.ChangeType((object)true, TypeCode.DateTime);
            r.Should().Be(new DateTime(1L));
        }

        #endregion

        #region Edge-case parity matrix (hand-verified against NumPy 2.4.2)

        [TestMethod]
        public void NumPyParity_DateTimeNaTAnalog_DateTimeMinValue()
        {
            // Best-effort NaT for DateTime: since Ticks cannot be long.MinValue,
            // DateTime.MinValue (Ticks=0) is the sentinel. This DIVERGES from NumPy
            // (where bool(NaT)=True) — document explicitly.
            Converts.ToBoolean(DateTime.MinValue).Should().BeFalse();
        }

        [TestMethod]
        public void NumPyParity_TimeSpanNaT_FullParity()
        {
            // TimeSpan.MinValue.Ticks == long.MinValue == NumPy NaT exactly.
            TimeSpan.MinValue.Ticks.Should().Be(long.MinValue);
            Converts.ToBoolean(TimeSpan.MinValue).Should().BeTrue();  // bool(NaT) = True
            Converts.ToInt64(TimeSpan.MinValue).Should().Be(long.MinValue);
            Converts.ToInt32(TimeSpan.MinValue).Should().Be(0);       // low 32 of MinValue
            Converts.ToDouble(TimeSpan.MinValue).Should().Be((double)long.MinValue);
        }

        [TestMethod]
        public void NumPyParity_RoundTrip_DateTimeIntegerTicks()
        {
            // DateTime -> long -> DateTime should round-trip for valid ticks.
            var original = Jan1_2024;
            var ticks = Converts.ToInt64(original);
            var restored = Converts.ToDateTime(ticks);
            restored.Should().Be(original);
        }

        [TestMethod]
        public void NumPyParity_RoundTrip_TimeSpanIntegerTicks()
        {
            // TimeSpan -> long -> TimeSpan should round-trip for ALL int64 values.
            foreach (var val in new[] { 0L, 1L, -1L, long.MaxValue, long.MinValue, 1000000000L })
            {
                var ts = new TimeSpan(val);
                var ticks = Converts.ToInt64(ts);
                var restored = Converts.ToTimeSpan(ticks);
                restored.Should().Be(ts, $"round-trip for ticks={val}");
            }
        }

        // Boundary bugs found via the battletest:
        //  1. ToDateTime((double)DateTime.MaxValue.Ticks) used to throw ArgumentOutOfRangeException
        //     because (double)MaxValue.Ticks rounds UP past the actual long value — the range
        //     guard missed it due to double precision. Fixed by routing through TicksToDateTime
        //     which re-validates after the long cast.
        //  2. ToInt64((double)long.MaxValue) used to return long.MaxValue instead of NaT.
        //     NumPy says `double 9.223372036854776e+18 -> int64 = -9223372036854775808`
        //     because (double)long.MaxValue rounds UP to 2^63 which is out of long range.
        //     The check `value > long.MaxValue` was comparing doubles (both == 2^63) and
        //     missed the overflow. Fixed by using exclusive upper bound at 2^63.
        //  3. Same fix applied to ToTimeSpan(double).

        [TestMethod]
        public void NumPyParity_DateTimeMaxValue_DoubleRoundTrip_DoesNotThrow()
        {
            var asDbl = Converts.ToDouble(DateTime.MaxValue);
            var back = Converts.ToDateTime(asDbl);
            // (double)DateTime.MaxValue.Ticks overshoots the actual ticks by rounding;
            // the NaT-equivalent (DateTime.MinValue) is the correct clamp here.
            back.Should().Be(DateTime.MinValue);
        }

        [TestMethod]
        public void NumPyParity_DoubleLongMaxValue_ToInt64_OverflowsToNaT()
        {
            // NumPy 2.4.2: np.float64(np.iinfo(np.int64).max).astype(np.int64) == int64.min
            // because (double)long.MaxValue rounds to 2^63 which is out of range.
            var result = Converts.ToInt64((double)long.MaxValue);
            result.Should().Be(long.MinValue);
        }

        [TestMethod]
        public void NumPyParity_DoubleLongMaxValue_ToTimeSpan_OverflowsToNaT()
        {
            var result = Converts.ToTimeSpan((double)long.MaxValue);
            result.Should().Be(TimeSpan.MinValue);
        }

        [TestMethod]
        public void NumPyParity_DoubleLongMinValue_ToInt64_DoesNotOverflow()
        {
            // (double)long.MinValue = -2^63 is EXACTLY representable as long.
            // NumPy: -9.223372036854776e+18 -> int64 = -9223372036854775808 (no overflow).
            Converts.ToInt64((double)long.MinValue).Should().Be(long.MinValue);
        }

        [TestMethod]
        public void NumPyParity_1e20_ToTimeSpan_OverflowsToNaT()
        {
            Converts.ToTimeSpan(1e20).Should().Be(TimeSpan.MinValue);
            Converts.ToTimeSpan(-1e20).Should().Be(TimeSpan.MinValue);
        }

        #endregion
    }
}
