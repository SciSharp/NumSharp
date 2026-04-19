using System;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    // =========================================================================
    // DateTime64 conversions — NumPy datetime64 parity.
    //
    // DateTime64 stores the raw int64 tick count (full long.MinValue…long.MaxValue
    // range). NaT == long.MinValue, matching NumPy exactly.
    //
    // These conversions mirror NumPy's datetime64↔numeric rules:
    //   • DateTime64 → primitive: uses Ticks as int64 (wrap/truncate/promote
    //     matches `datetime64.astype(dtype)`). bool(dt64) = (Ticks != 0), so
    //     NaT is True (long.MinValue ≠ 0) — same as NumPy.
    //   • primitive → DateTime64: sign-extends / reinterprets to int64, then
    //     wraps in DateTime64. Float NaN/Inf → NaT; float overflow → NaT.
    // =========================================================================
    public static partial class Converts
    {
        // ---------------------------------------------------------------------
        // DateTime64 → primitive (routed through Ticks; wrap/promote like int64)
        // ---------------------------------------------------------------------

        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(DateTime64 value) => value.Ticks != 0L;

        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(DateTime64 value) => unchecked((char)value.Ticks);

        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(DateTime64 value) => unchecked((sbyte)value.Ticks);

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(DateTime64 value) => unchecked((byte)value.Ticks);

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(DateTime64 value) => unchecked((short)value.Ticks);

        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(DateTime64 value) => unchecked((ushort)value.Ticks);

        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(DateTime64 value) => unchecked((int)value.Ticks);

        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(DateTime64 value) => unchecked((uint)value.Ticks);

        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(DateTime64 value) => value.Ticks;

        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(DateTime64 value) => unchecked((ulong)value.Ticks);

        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(DateTime64 value) => (float)value.Ticks;

        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(DateTime64 value) => (double)value.Ticks;

        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(DateTime64 value) => (decimal)value.Ticks;

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(DateTime64 value) => (Half)(double)value.Ticks;

        [MethodImpl(OptimizeAndInline)]
        public static Complex ToComplex(DateTime64 value) => new Complex((double)value.Ticks, 0);

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(DateTime64 value) => value.ToDateTime(DateTime.MinValue);

        [MethodImpl(OptimizeAndInline)]
        public static DateTimeOffset ToDateTimeOffset(DateTime64 value)
        {
            if (value.IsNaT || !value.IsValidDateTime)
                return new DateTimeOffset(DateTime.MinValue, TimeSpan.Zero);
            return value.ToDateTimeOffset();
        }

        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(DateTime64 value) => new TimeSpan(value.Ticks);

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(DateTime64 value) => value.ToString();

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(DateTime64 value, IFormatProvider provider)
            => value.ToString(null, provider);

        // ---------------------------------------------------------------------
        // DateTime → DateTime64 / DateTimeOffset → DateTime64 (lossless)
        // ---------------------------------------------------------------------

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(DateTime value) => new DateTime64(value);

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(DateTimeOffset value) => new DateTime64(value);

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(DateTime64 value) => value;

        // ---------------------------------------------------------------------
        // Primitive → DateTime64 (full int64 range; NaN/Inf/overflow → NaT)
        // ---------------------------------------------------------------------

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(bool value) => new DateTime64(value ? 1L : 0L);

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(sbyte value) => new DateTime64(value);

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(byte value) => new DateTime64(value);

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(short value) => new DateTime64(value);

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(ushort value) => new DateTime64(value);

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(int value) => new DateTime64(value);

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(uint value) => new DateTime64(value);

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(long value) => new DateTime64(value);

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(ulong value)
            => new DateTime64(unchecked((long)value));

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(char value) => new DateTime64(value);

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(float value) => ToDateTime64((double)value);

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(double value)
        {
            // NumPy: NaN, ±Inf → NaT (long.MinValue); overflow → NaT; else truncate.
            if (double.IsNaN(value) || double.IsInfinity(value))
                return DateTime64.NaT;
            if (value >= 9223372036854775808.0 || value < (double)long.MinValue)
                return DateTime64.NaT;
            return new DateTime64((long)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(Half value)
        {
            if (Half.IsNaN(value) || Half.IsInfinity(value))
                return DateTime64.NaT;
            return ToDateTime64((double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(decimal value)
        {
            decimal truncated = Math.Truncate(value);
            if (truncated < long.MinValue || truncated > long.MaxValue)
                return DateTime64.NaT;
            return new DateTime64((long)truncated);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(Complex value)
            => ToDateTime64(value.Real);

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(TimeSpan value) => new DateTime64(value.Ticks);

        public static DateTime64 ToDateTime64(string value)
        {
            if (string.IsNullOrEmpty(value) || value == "NaT")
                return DateTime64.NaT;
            return DateTime64.Parse(value, CultureInfo.CurrentCulture);
        }

        public static DateTime64 ToDateTime64(string value, IFormatProvider provider)
        {
            if (string.IsNullOrEmpty(value) || value == "NaT")
                return DateTime64.NaT;
            return DateTime64.Parse(value, provider);
        }

        // ---------------------------------------------------------------------
        // Object dispatcher for DateTime64
        // ---------------------------------------------------------------------

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(object value)
        {
            if (value is null) return DateTime64.NaT;
            return value switch
            {
                DateTime64 d64 => d64,
                DateTime dt => new DateTime64(dt),
                DateTimeOffset dto => new DateTime64(dto),
                TimeSpan ts => new DateTime64(ts.Ticks),
                bool b => ToDateTime64(b),
                sbyte sb => ToDateTime64(sb),
                byte by => ToDateTime64(by),
                short s => ToDateTime64(s),
                ushort us => ToDateTime64(us),
                int i => ToDateTime64(i),
                uint u => ToDateTime64(u),
                long l => ToDateTime64(l),
                ulong ul => ToDateTime64(ul),
                char c => ToDateTime64(c),
                float f => ToDateTime64(f),
                double d => ToDateTime64(d),
                Half h => ToDateTime64(h),
                decimal m => ToDateTime64(m),
                Complex cx => ToDateTime64(cx),
                string str => ToDateTime64(str),
                _ => new DateTime64(((IConvertible)value).ToInt64(null))
            };
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime64 ToDateTime64(object value, IFormatProvider provider)
        {
            if (value is string s) return ToDateTime64(s, provider);
            return ToDateTime64(value);
        }
    }
}
