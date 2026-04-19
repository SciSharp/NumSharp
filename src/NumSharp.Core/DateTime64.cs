// =============================================================================
// DateTime64 — NumPy datetime64 parity for .NET.
//
// ADAPTED FROM: .NET 10 System.DateTime
//   src/dotnet/src/libraries/System.Private.CoreLib/src/System/DateTime.cs
//
// Motivation:
//   NumPy's np.datetime64 is an int64-based scalar with full long.MinValue…
//   long.MaxValue range and a NaT sentinel at long.MinValue. .NET's
//   System.DateTime stores Ticks in the low 62 bits of a ulong (the top 2
//   bits hold DateTimeKind), so its Ticks range is [0, 3,155,378,975,999,999,999].
//   That leaves ~64 dtype-conversion cases where np.datetime64 can round-trip
//   int64 values that System.DateTime physically cannot. DateTime64 fills that
//   gap with the same public API shape as DateTime but without the Kind bits,
//   yielding full int64 Ticks and NaT semantics.
//
// Key differences from System.DateTime:
//   • Storage: `long _ticks` (no Kind; full int64 range) vs `ulong _dateData`.
//   • Range: long.MinValue … long.MaxValue vs [0, 3_155_378_975_999_999_999].
//   • NaT: long.MinValue sentinel — `IsNaT`, NumPy-style propagation through
//     arithmetic, and NumPy-style equality (NaT never equals anything).
//   • No Kind/timezone state: NumPy datetime64 has no timezone. Interop with
//     DateTime loses Kind; interop with DateTimeOffset uses `UtcTicks`.
//   • No leap-second or calendar machinery beyond what DateTime exposes —
//     year/month/day/… properties delegate to System.DateTime for values
//     inside [0, DateTime.MaxTicks] and throw for NaT / out-of-range.
//
// Interop:
//   • Implicit DateTime → DateTime64 (always lossless, drops Kind)
//   • Implicit DateTimeOffset → DateTime64 (via UtcTicks, drops offset)
//   • Implicit long → DateTime64 (raw tick count)
//   • Explicit DateTime64 → DateTime (throws for NaT / out-of-range)
//   • Explicit DateTime64 → DateTimeOffset (throws for NaT / out-of-range)
//   • Explicit DateTime64 → long (returns raw ticks; NaT = long.MinValue)
//
// Calendar methods (Year, Month, Day, Hour, …) delegate to System.DateTime
// when Ticks is in [0, DateTime.MaxTicks]; otherwise they throw
// InvalidOperationException. Use IsNaT / IsValidDateTime to guard.
// =============================================================================

using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NumSharp
{
    /// <summary>
    /// A 64-bit signed tick count representing a date/time value with full
    /// <c>long</c> range and a <see cref="NaT"/> sentinel, matching NumPy's
    /// <c>np.datetime64</c> semantics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// One "tick" equals 100 nanoseconds, matching <see cref="DateTime.Ticks"/>.
    /// The zero-tick value represents midnight on 1 January 0001 (the Gregorian
    /// epoch used by <see cref="DateTime"/>), which is <i>not</i> the Unix epoch.
    /// Use <see cref="UnixEpoch"/> for Unix-epoch-relative calculations.
    /// </para>
    /// <para>
    /// The <see cref="NaT"/> sentinel (<c>Ticks == long.MinValue</c>) is
    /// <c>Not-a-Time</c>. It propagates through all arithmetic operations and
    /// — following NumPy's rules — never compares equal to anything (including
    /// itself), analogous to IEEE 754 <c>NaN</c>.
    /// </para>
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    [Serializable]
    public readonly partial struct DateTime64
        : IComparable,
          IComparable<DateTime64>,
          IEquatable<DateTime64>,
          IConvertible,
          IFormattable,
          ISpanFormattable
    {
        // ---------------------------------------------------------------------
        // Constants (mirroring DateTime's layout, minus Kind bits)
        // ---------------------------------------------------------------------

        /// <summary>Ticks per 100-ns unit — for symmetry with DateTime constants.</summary>
        internal const long TicksPerMicrosecond = TimeSpan.TicksPerMicrosecond;
        internal const long TicksPerMillisecond = TimeSpan.TicksPerMillisecond;
        internal const long TicksPerSecond = TimeSpan.TicksPerSecond;
        internal const long TicksPerMinute = TimeSpan.TicksPerMinute;
        internal const long TicksPerHour = TimeSpan.TicksPerHour;
        internal const long TicksPerDay = TimeSpan.TicksPerDay;

        /// <summary>The minimum legal tick value for a <see cref="System.DateTime"/>.</summary>
        internal const long DotNetMinTicks = 0L;

        /// <summary>The maximum legal tick value for a <see cref="System.DateTime"/> (9999-12-31 23:59:59.9999999).</summary>
        internal const long DotNetMaxTicks = 3_155_378_975_999_999_999L;

        /// <summary>NaT sentinel tick value, matching NumPy (<c>long.MinValue</c>).</summary>
        internal const long NaTTicks = long.MinValue;

        /// <summary>Ticks at the Unix epoch (1970-01-01 UTC), matching <see cref="DateTime.UnixEpoch"/>.</summary>
        internal const long UnixEpochTicks = 621_355_968_000_000_000L;

        // ---------------------------------------------------------------------
        // Static Fields
        // ---------------------------------------------------------------------

        /// <summary>Not-a-Time sentinel (<c>Ticks == long.MinValue</c>), matching NumPy.</summary>
        public static readonly DateTime64 NaT = new DateTime64(NaTTicks);

        /// <summary>The smallest non-NaT representable value (<c>Ticks == long.MinValue + 1</c>).</summary>
        public static readonly DateTime64 MinValue = new DateTime64(long.MinValue + 1);

        /// <summary>The largest representable value (<c>Ticks == long.MaxValue</c>).</summary>
        public static readonly DateTime64 MaxValue = new DateTime64(long.MaxValue);

        /// <summary>The .NET calendar epoch (midnight 0001-01-01), same as <see cref="DateTime.MinValue"/>.</summary>
        public static readonly DateTime64 Epoch = default;

        /// <summary>The Unix epoch (midnight 1970-01-01 UTC).</summary>
        public static readonly DateTime64 UnixEpoch = new DateTime64(UnixEpochTicks);

        // ---------------------------------------------------------------------
        // Instance Field (single long — full int64 range, no Kind bits)
        // ---------------------------------------------------------------------

        /// <summary>
        /// Raw 100-ns tick count as a signed int64. Full <c>long</c> range is
        /// legal; <c>long.MinValue</c> is the NaT sentinel.
        /// </summary>
        private readonly long _ticks;

        // ---------------------------------------------------------------------
        // Constructors
        // ---------------------------------------------------------------------

        /// <summary>Constructs a <see cref="DateTime64"/> from a raw tick count (any int64, including NaT).</summary>
        public DateTime64(long ticks)
        {
            _ticks = ticks;
        }

        /// <summary>
        /// Constructs a <see cref="DateTime64"/> from a <see cref="System.DateTime"/>.
        /// The <see cref="DateTime.Kind"/> is discarded (NumPy datetime64 has no timezone).
        /// </summary>
        public DateTime64(DateTime dateTime)
        {
            _ticks = dateTime.Ticks;
        }

        /// <summary>
        /// Constructs a <see cref="DateTime64"/> from a <see cref="System.DateTimeOffset"/>.
        /// The value is stored as <see cref="DateTimeOffset.UtcTicks"/> (offset discarded).
        /// </summary>
        public DateTime64(DateTimeOffset dateTimeOffset)
        {
            _ticks = dateTimeOffset.UtcTicks;
        }

        /// <summary>Constructs a <see cref="DateTime64"/> from a <see cref="DateOnly"/> + <see cref="TimeOnly"/>.</summary>
        public DateTime64(DateOnly date, TimeOnly time)
        {
            _ticks = date.DayNumber * TicksPerDay + time.Ticks;
        }

        /// <summary>Constructs a <see cref="DateTime64"/> from year/month/day (Gregorian, midnight).</summary>
        public DateTime64(int year, int month, int day)
        {
            _ticks = new DateTime(year, month, day).Ticks;
        }

        /// <summary>Constructs a <see cref="DateTime64"/> from year/month/day/hour/minute/second.</summary>
        public DateTime64(int year, int month, int day, int hour, int minute, int second)
        {
            _ticks = new DateTime(year, month, day, hour, minute, second).Ticks;
        }

        /// <summary>Constructs a <see cref="DateTime64"/> from year/month/day/hour/minute/second/millisecond.</summary>
        public DateTime64(int year, int month, int day, int hour, int minute, int second, int millisecond)
        {
            _ticks = new DateTime(year, month, day, hour, minute, second, millisecond).Ticks;
        }

        // ---------------------------------------------------------------------
        // Core properties
        // ---------------------------------------------------------------------

        /// <summary>The raw 100-ns tick count (full int64, may equal <c>long.MinValue</c> for NaT).</summary>
        public long Ticks
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ticks;
        }

        /// <summary><see langword="true"/> iff this instance is Not-a-Time (<c>Ticks == long.MinValue</c>).</summary>
        public bool IsNaT
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ticks == NaTTicks;
        }

        /// <summary>
        /// <see langword="true"/> iff <see cref="Ticks"/> is inside the legal range
        /// of <see cref="System.DateTime"/>, i.e. <c>[0, DateTime.MaxValue.Ticks]</c>.
        /// </summary>
        public bool IsValidDateTime
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => (ulong)_ticks <= (ulong)DotNetMaxTicks;
        }

        // ---------------------------------------------------------------------
        // Calendar properties — delegate to System.DateTime when in range.
        // These throw InvalidOperationException for NaT / out-of-range values.
        // ---------------------------------------------------------------------

        /// <summary>Gets the year component [1..9999]. Throws for NaT / out-of-range.</summary>
        public int Year => RequireValidDateTime().Year;

        /// <summary>Gets the month component [1..12]. Throws for NaT / out-of-range.</summary>
        public int Month => RequireValidDateTime().Month;

        /// <summary>Gets the day component [1..31]. Throws for NaT / out-of-range.</summary>
        public int Day => RequireValidDateTime().Day;

        /// <summary>Gets the hour component [0..23]. Throws for NaT / out-of-range.</summary>
        public int Hour => RequireValidDateTime().Hour;

        /// <summary>Gets the minute component [0..59]. Throws for NaT / out-of-range.</summary>
        public int Minute => RequireValidDateTime().Minute;

        /// <summary>Gets the second component [0..59]. Throws for NaT / out-of-range.</summary>
        public int Second => RequireValidDateTime().Second;

        /// <summary>Gets the millisecond component [0..999]. Throws for NaT / out-of-range.</summary>
        public int Millisecond => RequireValidDateTime().Millisecond;

        /// <summary>Gets the microsecond component [0..999]. Throws for NaT / out-of-range.</summary>
        public int Microsecond => RequireValidDateTime().Microsecond;

        /// <summary>Gets the nanosecond component [0..900, step 100]. Throws for NaT / out-of-range.</summary>
        public int Nanosecond => RequireValidDateTime().Nanosecond;

        /// <summary>Gets the day-of-week. Throws for NaT / out-of-range.</summary>
        public DayOfWeek DayOfWeek => RequireValidDateTime().DayOfWeek;

        /// <summary>Gets the day-of-year [1..366]. Throws for NaT / out-of-range.</summary>
        public int DayOfYear => RequireValidDateTime().DayOfYear;

        /// <summary>Gets the date portion (time-of-day zeroed). Throws for NaT / out-of-range.</summary>
        public DateTime64 Date
        {
            get
            {
                var dt = RequireValidDateTime();
                return new DateTime64(dt.Date.Ticks);
            }
        }

        /// <summary>Gets the time-of-day component as a <see cref="TimeSpan"/>. Throws for NaT / out-of-range.</summary>
        public TimeSpan TimeOfDay
        {
            get
            {
                var dt = RequireValidDateTime();
                return dt.TimeOfDay;
            }
        }

        // ---------------------------------------------------------------------
        // Now / UtcNow / Today — mirror DateTime
        // ---------------------------------------------------------------------

        /// <summary>Current local time as a <see cref="DateTime64"/>.</summary>
        public static DateTime64 Now => new DateTime64(DateTime.Now);

        /// <summary>Current UTC time as a <see cref="DateTime64"/>.</summary>
        public static DateTime64 UtcNow => new DateTime64(DateTime.UtcNow);

        /// <summary>Current date (midnight) as a <see cref="DateTime64"/>.</summary>
        public static DateTime64 Today => new DateTime64(DateTime.Today);

        // ---------------------------------------------------------------------
        // Interop — implicit/explicit conversions
        // ---------------------------------------------------------------------

        /// <summary>Implicit widening from <see cref="System.DateTime"/> (drops Kind).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator DateTime64(DateTime value) => new DateTime64(value.Ticks);

        /// <summary>Implicit widening from <see cref="System.DateTimeOffset"/> (via UtcTicks; offset discarded).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator DateTime64(DateTimeOffset value) => new DateTime64(value.UtcTicks);

        /// <summary>Implicit widening from <see cref="long"/> (raw tick count).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator DateTime64(long ticks) => new DateTime64(ticks);

        /// <summary>Explicit narrowing to <see cref="System.DateTime"/>. Throws for NaT / out-of-range.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator DateTime(DateTime64 value) => value.ToDateTime();

        /// <summary>Explicit narrowing to <see cref="System.DateTimeOffset"/> (UTC). Throws for NaT / out-of-range.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator DateTimeOffset(DateTime64 value) => value.ToDateTimeOffset();

        /// <summary>Explicit extraction of the raw int64 tick count.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator long(DateTime64 value) => value._ticks;

        /// <summary>Convert to <see cref="System.DateTime"/>. Throws for NaT / out-of-range.</summary>
        public DateTime ToDateTime()
        {
            var dt = RequireValidDateTime();
            return dt;
        }

        /// <summary>
        /// Convert to <see cref="System.DateTime"/>, clamping NaT / out-of-range
        /// values to <paramref name="fallback"/> rather than throwing.
        /// </summary>
        public DateTime ToDateTime(DateTime fallback)
        {
            if (IsNaT || !IsValidDateTime)
                return fallback;
            return new DateTime(_ticks);
        }

        /// <summary>
        /// Try to convert to <see cref="System.DateTime"/>. Returns <see langword="false"/>
        /// for NaT / out-of-range values (<paramref name="result"/> is set to <see cref="DateTime.MinValue"/>).
        /// </summary>
        public bool TryToDateTime(out DateTime result)
        {
            if (IsNaT || !IsValidDateTime)
            {
                result = DateTime.MinValue;
                return false;
            }
            result = new DateTime(_ticks);
            return true;
        }

        /// <summary>Convert to <see cref="System.DateTimeOffset"/> at UTC offset. Throws for NaT / out-of-range.</summary>
        public DateTimeOffset ToDateTimeOffset()
        {
            var dt = RequireValidDateTime();
            return new DateTimeOffset(DateTime.SpecifyKind(dt, DateTimeKind.Utc));
        }

        /// <summary>Convert to <see cref="System.DateTimeOffset"/> at the given offset. Throws for NaT / out-of-range.</summary>
        public DateTimeOffset ToDateTimeOffset(TimeSpan offset)
        {
            var dt = RequireValidDateTime();
            return new DateTimeOffset(dt, offset);
        }

        /// <summary>
        /// Convert to Unix time in seconds (UTC), matching <see cref="DateTimeOffset.ToUnixTimeSeconds"/>.
        /// NaT → <see cref="long.MinValue"/>; out-of-.NET-range values use raw tick arithmetic.
        /// </summary>
        public long ToUnixTimeSeconds()
        {
            if (IsNaT) return long.MinValue;
            // Use raw tick math so we don't lose values outside DateTime's range.
            return (_ticks - UnixEpochTicks) / TicksPerSecond;
        }

        /// <summary>Convert to Unix time in milliseconds (UTC). NaT → <see cref="long.MinValue"/>.</summary>
        public long ToUnixTimeMilliseconds()
        {
            if (IsNaT) return long.MinValue;
            return (_ticks - UnixEpochTicks) / TicksPerMillisecond;
        }

        /// <summary>Construct from Unix time (seconds since 1970-01-01 UTC).</summary>
        public static DateTime64 FromUnixTimeSeconds(long seconds)
        {
            if (seconds == long.MinValue) return NaT;
            // Saturate overflow to NaT (NumPy behavior).
            try { return new DateTime64(checked(seconds * TicksPerSecond + UnixEpochTicks)); }
            catch (OverflowException) { return NaT; }
        }

        /// <summary>Construct from Unix time (milliseconds since 1970-01-01 UTC).</summary>
        public static DateTime64 FromUnixTimeMilliseconds(long milliseconds)
        {
            if (milliseconds == long.MinValue) return NaT;
            try { return new DateTime64(checked(milliseconds * TicksPerMillisecond + UnixEpochTicks)); }
            catch (OverflowException) { return NaT; }
        }

        // ---------------------------------------------------------------------
        // Arithmetic (NaT propagates; overflow saturates to NaT, matching NumPy)
        // ---------------------------------------------------------------------

        /// <summary>Add a raw tick delta. NaT propagates; overflow saturates to NaT.</summary>
        public DateTime64 AddTicks(long delta)
        {
            if (IsNaT) return NaT;
            long result;
            try { result = checked(_ticks + delta); }
            catch (OverflowException) { return NaT; }
            if (result == NaTTicks) return NaT;  // guard against accidental sentinel collision
            return new DateTime64(result);
        }

        /// <summary>Add a <see cref="TimeSpan"/>. NaT propagates; overflow saturates to NaT.</summary>
        public DateTime64 Add(TimeSpan value) => AddTicks(value.Ticks);

        /// <summary>Add whole and fractional days. NaT propagates; overflow saturates to NaT.</summary>
        public DateTime64 AddDays(double value) => AddTicks((long)(value * TicksPerDay));

        /// <summary>Add whole and fractional hours. NaT propagates.</summary>
        public DateTime64 AddHours(double value) => AddTicks((long)(value * TicksPerHour));

        /// <summary>Add whole and fractional minutes. NaT propagates.</summary>
        public DateTime64 AddMinutes(double value) => AddTicks((long)(value * TicksPerMinute));

        /// <summary>Add whole and fractional seconds. NaT propagates.</summary>
        public DateTime64 AddSeconds(double value) => AddTicks((long)(value * TicksPerSecond));

        /// <summary>Add whole and fractional milliseconds. NaT propagates.</summary>
        public DateTime64 AddMilliseconds(double value) => AddTicks((long)(value * TicksPerMillisecond));

        /// <summary>Add whole and fractional microseconds. NaT propagates.</summary>
        public DateTime64 AddMicroseconds(double value) => AddTicks((long)(value * TicksPerMicrosecond));

        /// <summary>Add the specified number of months. NaT / out-of-range propagate to NaT.</summary>
        public DateTime64 AddMonths(int months)
        {
            if (IsNaT || !IsValidDateTime) return NaT;
            try { return new DateTime64(new DateTime(_ticks).AddMonths(months)); }
            catch (ArgumentOutOfRangeException) { return NaT; }
        }

        /// <summary>Add the specified number of years. NaT / out-of-range propagate to NaT.</summary>
        public DateTime64 AddYears(int value)
        {
            if (IsNaT || !IsValidDateTime) return NaT;
            try { return new DateTime64(new DateTime(_ticks).AddYears(value)); }
            catch (ArgumentOutOfRangeException) { return NaT; }
        }

        /// <summary>Gets the number of days in the specified month of the specified year.</summary>
        public static int DaysInMonth(int year, int month) => DateTime.DaysInMonth(year, month);

        /// <summary>Returns whether the specified year is a leap year in the Gregorian calendar.</summary>
        public static bool IsLeapYear(int year) => DateTime.IsLeapYear(year);

        /// <summary>Subtract a <see cref="TimeSpan"/>. NaT propagates.</summary>
        public DateTime64 Subtract(TimeSpan value) => AddTicks(unchecked(-value.Ticks));

        /// <summary>
        /// Difference as a <see cref="TimeSpan"/>. If either operand is NaT,
        /// returns <see cref="TimeSpan.MinValue"/> (closest NaT-equivalent for TimeSpan).
        /// </summary>
        public TimeSpan Subtract(DateTime64 other)
        {
            if (IsNaT || other.IsNaT) return TimeSpan.MinValue;
            return new TimeSpan(unchecked(_ticks - other._ticks));
        }

        public static DateTime64 operator +(DateTime64 d, TimeSpan t) => d.Add(t);
        public static DateTime64 operator -(DateTime64 d, TimeSpan t) => d.Subtract(t);
        public static TimeSpan operator -(DateTime64 d1, DateTime64 d2) => d1.Subtract(d2);

        // ---------------------------------------------------------------------
        // Equality / Comparison (NumPy NaT semantics)
        // NumPy: NaT != NaT (NaN-like); ordering of NaT is implementation-defined
        //        but equality is the commonly-observed behavior. We follow that.
        // ---------------------------------------------------------------------

        /// <summary>
        /// Equality test following NumPy datetime64 semantics: <see cref="NaT"/>
        /// never equals anything (including itself).
        /// </summary>
        public bool Equals(DateTime64 other)
        {
            // NumPy: NaT == anything → False (NaN-like).
            if (IsNaT || other.IsNaT) return false;
            return _ticks == other._ticks;
        }

        public override bool Equals([NotNullWhen(true)] object? value)
            => value is DateTime64 d && Equals(d);

        public static bool Equals(DateTime64 t1, DateTime64 t2) => t1.Equals(t2);

        public override int GetHashCode() => _ticks.GetHashCode();

        /// <summary>Compare two <see cref="DateTime64"/> values by ticks (NaT ordering follows int64).</summary>
        public static int Compare(DateTime64 t1, DateTime64 t2)
        {
            long a = t1._ticks, b = t2._ticks;
            if (a < b) return -1;
            if (a > b) return 1;
            return 0;
        }

        public int CompareTo(DateTime64 value) => Compare(this, value);

        public int CompareTo(object? value)
        {
            if (value is null) return 1;
            if (value is DateTime64 d) return Compare(this, d);
            if (value is DateTime dt) return Compare(this, new DateTime64(dt));
            throw new ArgumentException("Object must be of type DateTime64 or DateTime.", nameof(value));
        }

        // Strict comparison operators: any NaT operand → False (NumPy semantics).
        public static bool operator ==(DateTime64 d1, DateTime64 d2) => d1.Equals(d2);
        public static bool operator !=(DateTime64 d1, DateTime64 d2) => !d1.Equals(d2);

        public static bool operator <(DateTime64 d1, DateTime64 d2)
            => !d1.IsNaT && !d2.IsNaT && d1._ticks < d2._ticks;

        public static bool operator >(DateTime64 d1, DateTime64 d2)
            => !d1.IsNaT && !d2.IsNaT && d1._ticks > d2._ticks;

        public static bool operator <=(DateTime64 d1, DateTime64 d2)
            => !d1.IsNaT && !d2.IsNaT && d1._ticks <= d2._ticks;

        public static bool operator >=(DateTime64 d1, DateTime64 d2)
            => !d1.IsNaT && !d2.IsNaT && d1._ticks >= d2._ticks;

        // ---------------------------------------------------------------------
        // Formatting
        // ---------------------------------------------------------------------

        /// <summary>
        /// Formats as ISO-8601 for in-range values, <c>"NaT"</c> for NaT, and
        /// <c>"DateTime64(ticks=N)"</c> for values outside <see cref="System.DateTime"/>'s range.
        /// </summary>
        public override string ToString()
        {
            if (IsNaT) return "NaT";
            if (!IsValidDateTime) return $"DateTime64(ticks={_ticks})";
            return new DateTime(_ticks).ToString("o", CultureInfo.InvariantCulture);
        }

        public string ToString(string? format) => ToString(format, CultureInfo.CurrentCulture);

        public string ToString(IFormatProvider? provider) => ToString(null, provider);

        public string ToString(string? format, IFormatProvider? provider)
        {
            if (IsNaT) return "NaT";
            if (!IsValidDateTime) return $"DateTime64(ticks={_ticks})";
            // Default to ISO-8601 (matches NumPy's datetime64 text representation).
            if (string.IsNullOrEmpty(format)) format = "o";
            return new DateTime(_ticks).ToString(format, provider ?? CultureInfo.InvariantCulture);
        }

        public bool TryFormat(Span<char> destination, out int charsWritten, ReadOnlySpan<char> format = default, IFormatProvider? provider = null)
        {
            string s = ToString(format.ToString(), provider);
            if (s.Length > destination.Length)
            {
                charsWritten = 0;
                return false;
            }
            s.AsSpan().CopyTo(destination);
            charsWritten = s.Length;
            return true;
        }

        // ---------------------------------------------------------------------
        // Parsing (delegate to DateTime for in-range values; "NaT" for NaT)
        // ---------------------------------------------------------------------

        public static DateTime64 Parse(string s)
        {
            if (s == "NaT") return NaT;
            return new DateTime64(DateTime.Parse(s, CultureInfo.CurrentCulture));
        }

        public static DateTime64 Parse(string s, IFormatProvider? provider)
        {
            if (s == "NaT") return NaT;
            return new DateTime64(DateTime.Parse(s, provider));
        }

        public static bool TryParse([NotNullWhen(true)] string? s, out DateTime64 result)
        {
            if (s == "NaT") { result = NaT; return true; }
            if (DateTime.TryParse(s, out var dt)) { result = new DateTime64(dt); return true; }
            result = default;
            return false;
        }

        public static bool TryParse([NotNullWhen(true)] string? s, IFormatProvider? provider, DateTimeStyles styles, out DateTime64 result)
        {
            if (s == "NaT") { result = NaT; return true; }
            if (DateTime.TryParse(s, provider, styles, out var dt)) { result = new DateTime64(dt); return true; }
            result = default;
            return false;
        }

        public static DateTime64 ParseExact(string s, string format, IFormatProvider? provider)
        {
            if (s == "NaT") return NaT;
            return new DateTime64(DateTime.ParseExact(s, format, provider));
        }

        public static DateTime64 ParseExact(string s, string[] formats, IFormatProvider? provider, DateTimeStyles style)
        {
            if (s == "NaT") return NaT;
            return new DateTime64(DateTime.ParseExact(s, formats, provider, style));
        }

        public static bool TryParseExact([NotNullWhen(true)] string? s, [NotNullWhen(true)] string? format,
            IFormatProvider? provider, DateTimeStyles style, out DateTime64 result)
        {
            if (s == "NaT") { result = NaT; return true; }
            if (DateTime.TryParseExact(s, format, provider, style, out var dt)) { result = new DateTime64(dt); return true; }
            result = default;
            return false;
        }

        // ---------------------------------------------------------------------
        // IConvertible — needed for Convert.ChangeType + NumSharp's type-switch paths.
        // Value is converted using the raw int64 tick count (matching NumPy).
        // ---------------------------------------------------------------------

        TypeCode IConvertible.GetTypeCode() => TypeCode.DateTime;

        bool IConvertible.ToBoolean(IFormatProvider? provider) => _ticks != 0L;  // NaT ticks=long.MinValue ≠ 0 → true (matches NumPy)
        sbyte IConvertible.ToSByte(IFormatProvider? provider) => unchecked((sbyte)_ticks);
        byte IConvertible.ToByte(IFormatProvider? provider) => unchecked((byte)_ticks);
        short IConvertible.ToInt16(IFormatProvider? provider) => unchecked((short)_ticks);
        ushort IConvertible.ToUInt16(IFormatProvider? provider) => unchecked((ushort)_ticks);
        int IConvertible.ToInt32(IFormatProvider? provider) => unchecked((int)_ticks);
        uint IConvertible.ToUInt32(IFormatProvider? provider) => unchecked((uint)_ticks);
        long IConvertible.ToInt64(IFormatProvider? provider) => _ticks;
        ulong IConvertible.ToUInt64(IFormatProvider? provider) => unchecked((ulong)_ticks);
        char IConvertible.ToChar(IFormatProvider? provider) => unchecked((char)_ticks);
        float IConvertible.ToSingle(IFormatProvider? provider) => (float)_ticks;
        double IConvertible.ToDouble(IFormatProvider? provider) => (double)_ticks;
        decimal IConvertible.ToDecimal(IFormatProvider? provider) => (decimal)_ticks;
        DateTime IConvertible.ToDateTime(IFormatProvider? provider) => ToDateTime(DateTime.MinValue);
        string IConvertible.ToString(IFormatProvider? provider) => ToString(null, provider);

        object IConvertible.ToType(Type conversionType, IFormatProvider? provider)
        {
            if (conversionType == typeof(DateTime64)) return this;
            if (conversionType == typeof(DateTime)) return ToDateTime(DateTime.MinValue);
            if (conversionType == typeof(DateTimeOffset)) return IsValidDateTime && !IsNaT ? ToDateTimeOffset() : (object)new DateTimeOffset(DateTime.MinValue);
            if (conversionType == typeof(long)) return _ticks;
            if (conversionType == typeof(ulong)) return unchecked((ulong)_ticks);
            if (conversionType == typeof(double)) return (double)_ticks;
            if (conversionType == typeof(int)) return unchecked((int)_ticks);
            if (conversionType == typeof(string)) return ToString(null, provider);
            return Convert.ChangeType(_ticks, conversionType, provider);
        }

        // ---------------------------------------------------------------------
        // Helpers
        // ---------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private DateTime RequireValidDateTime()
        {
            if (IsNaT)
                throw new InvalidOperationException("DateTime64 is NaT (Not a Time); cannot be converted to System.DateTime.");
            if (!IsValidDateTime)
                throw new InvalidOperationException($"DateTime64 ticks {_ticks} are outside System.DateTime's legal range [0, {DotNetMaxTicks}].");
            return new DateTime(_ticks);
        }
    }
}
