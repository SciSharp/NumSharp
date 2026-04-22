// =============================================================================
// DateTime64 — NumPy datetime64 parity for .NET.
//
// ADAPTED FROM: .NET 10 System.DateTime
//   src/dotnet/src/libraries/System.Private.CoreLib/src/System/DateTime.cs
//
// SCOPE:
//   DateTime64 is a CONVERSION HELPER TYPE, not a NumSharp NPTypeCode dtype.
//   It exists so Converts.ToDateTime64(X) / Converts.ToX(DateTime64) can match
//   NumPy's datetime64 semantics exactly (full int64 range, NaT sentinel).
//   Calendar arithmetic, parsing, formatting helpers, etc. are delegated to
//   System.DateTime (via interop) rather than duplicated here.
//
// Key differences from System.DateTime:
//   • Storage: `long _ticks` (no Kind bits; full int64 range) vs `ulong _dateData`.
//   • Range: long.MinValue…long.MaxValue vs [0, 3_155_378_975_999_999_999].
//   • NaT: long.MinValue sentinel — `IsNaT`, NumPy-style propagation through
//     arithmetic. Operators (==, !=, <, >, <=, >=) follow NumPy (NaT never
//     compares equal to anything, orderings involving NaT return false);
//     `Equals(DateTime64)` follows the .NET convention (bit-equal → true) so
//     the hash contract holds and NaT can be used as a dictionary key.
//   • No Kind / timezone: NumPy datetime64 has no timezone. Interop with
//     DateTime loses Kind; interop with DateTimeOffset uses `UtcTicks`.
//
// Interop:
//   • Implicit: DateTime → DateTime64 (drops Kind)
//   • Implicit: DateTimeOffset → DateTime64 (UtcTicks, offset discarded)
//   • Implicit: long → DateTime64 (raw tick count)
//   • Explicit: DateTime64 → DateTime / DateTimeOffset (throws if NaT / out-of-range)
//   • Explicit: DateTime64 → long (raw ticks; NaT = long.MinValue)
//   • Non-throwing alternatives: ToDateTime(fallback), TryToDateTime(out),
//     ToDateTimeOffset(fallback), TryToDateTimeOffset(out).
//
// This file intentionally does NOT expose: Year/Month/Day, AddMonths/AddYears,
// Parse/ParseExact, IsLeapYear, DaysInMonth, Now/UtcNow/Today, or Unix-time
// helpers. If you need calendar arithmetic, convert to System.DateTime first.
// =============================================================================

using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NumSharp
{
    /// <summary>
    /// A 64-bit signed tick count representing a date/time value with full
    /// <c>long</c> range and a <see cref="NaT"/> sentinel, matching NumPy's
    /// <c>np.datetime64</c> semantics. Used as a conversion-helper type in
    /// <see cref="Utilities.Converts"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// One tick equals 100 nanoseconds (same unit as <see cref="DateTime.Ticks"/>).
    /// <see cref="Ticks"/> <c>== 0</c> is midnight of 1 January 0001 (the
    /// Gregorian epoch of <see cref="DateTime"/>), which is <i>not</i> the Unix
    /// epoch.
    /// </para>
    /// <para>
    /// <b>NaT semantics.</b> <see cref="NaT"/> (<c>Ticks == long.MinValue</c>)
    /// is <c>Not-a-Time</c>, analogous to IEEE 754 <c>NaN</c>:
    /// <list type="bullet">
    /// <item><description>NaT propagates through arithmetic.</description></item>
    /// <item><description><c>operator ==</c> / <c>!=</c> / <c>&lt;</c> / <c>&gt;</c> / <c>&lt;=</c> / <c>&gt;=</c> follow NumPy: any comparison involving NaT is <c>false</c> for <c>==</c>/&lt;/&gt;/&lt;=/&gt;=, and <c>true</c> for <c>!=</c>.</description></item>
    /// <item><description><see cref="Equals(DateTime64)"/> follows the <see cref="IEquatable{T}"/> convention (two NaTs are considered equal bit-wise) so that <see cref="object.GetHashCode"/> is contract-compliant and NaT can be used as a <see cref="System.Collections.Generic.Dictionary{TKey, TValue}"/> key. This mirrors how .NET handles <see cref="double.NaN"/>: <c>double.NaN.Equals(double.NaN)</c> is <c>true</c> but <c>double.NaN == double.NaN</c> is <c>false</c>.</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    [Serializable]
    public readonly struct DateTime64
        : IComparable,
          IComparable<DateTime64>,
          IEquatable<DateTime64>,
          IConvertible,
          IFormattable,
          ISpanFormattable
    {
        // ---------------------------------------------------------------------
        // Constants
        // ---------------------------------------------------------------------

        /// <summary>The minimum legal tick value of a <see cref="System.DateTime"/>.</summary>
        internal const long DotNetMinTicks = 0L;

        /// <summary>The maximum legal tick value of a <see cref="System.DateTime"/> (9999-12-31 23:59:59.9999999).</summary>
        internal const long DotNetMaxTicks = 3_155_378_975_999_999_999L;

        /// <summary>NaT sentinel tick value, matching NumPy (<c>long.MinValue</c>).</summary>
        internal const long NaTTicks = long.MinValue;

        // NumPy datetime64 boundaries as doubles, for hardened float → int64 cast.
        // (double)long.MinValue is exactly representable; (double)long.MaxValue
        // rounds up to 2^63 which is NOT representable as a signed int64.
        private const double Int64MaxAsDoubleUpperExclusive = 9223372036854775808.0;   // 2^63
        private const double Int64MinAsDoubleLowerExclusive = -9223372036854775808.0;  // -2^63 = (double)long.MinValue

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

        // ---------------------------------------------------------------------
        // Instance field — single long, full int64 range, no Kind bits
        // ---------------------------------------------------------------------

        private readonly long _ticks;

        // ---------------------------------------------------------------------
        // Constructors (minimal surface; calendar construction goes via DateTime)
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
        /// Stored as <see cref="DateTimeOffset.UtcTicks"/> (offset discarded).
        /// </summary>
        public DateTime64(DateTimeOffset dateTimeOffset)
        {
            _ticks = dateTimeOffset.UtcTicks;
        }

        // ---------------------------------------------------------------------
        // Core properties
        // ---------------------------------------------------------------------

        /// <summary>The raw 100-ns tick count (full int64; <c>long.MinValue</c> for NaT).</summary>
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
        /// <see langword="true"/> iff <see cref="Ticks"/> is inside the legal range of
        /// <see cref="System.DateTime"/>, i.e. <c>[0, DateTime.MaxValue.Ticks]</c>.
        /// </summary>
        public bool IsValidDateTime
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => (ulong)_ticks <= (ulong)DotNetMaxTicks;
        }

        // ---------------------------------------------------------------------
        // Interop: implicit widening, explicit narrowing
        // ---------------------------------------------------------------------

        /// <summary>Implicit widening from <see cref="System.DateTime"/> (drops Kind).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator DateTime64(DateTime value) => new DateTime64(value.Ticks);

        /// <summary>Implicit widening from <see cref="System.DateTimeOffset"/> (via UtcTicks).</summary>
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

        /// <summary>Convert to <see cref="System.DateTime"/>. Throws <see cref="InvalidOperationException"/> for NaT / out-of-range.</summary>
        public DateTime ToDateTime()
        {
            if (IsNaT)
                throw new InvalidOperationException("DateTime64 is NaT (Not a Time); cannot be converted to System.DateTime.");
            if (!IsValidDateTime)
                throw new InvalidOperationException($"DateTime64 ticks {_ticks} are outside System.DateTime's legal range [0, {DotNetMaxTicks}].");
            return new DateTime(_ticks);
        }

        /// <summary>
        /// Convert to <see cref="System.DateTime"/>, returning <paramref name="fallback"/>
        /// for NaT / out-of-range values instead of throwing.
        /// </summary>
        public DateTime ToDateTime(DateTime fallback)
        {
            if (IsNaT || !IsValidDateTime) return fallback;
            return new DateTime(_ticks);
        }

        /// <summary>
        /// Try to convert to <see cref="System.DateTime"/>. Returns <see langword="false"/>
        /// for NaT / out-of-range values (<paramref name="result"/> set to <see cref="DateTime.MinValue"/>).
        /// </summary>
        public bool TryToDateTime(out DateTime result)
        {
            if (IsNaT || !IsValidDateTime) { result = DateTime.MinValue; return false; }
            result = new DateTime(_ticks);
            return true;
        }

        /// <summary>Convert to <see cref="System.DateTimeOffset"/> at UTC. Throws for NaT / out-of-range.</summary>
        public DateTimeOffset ToDateTimeOffset()
        {
            if (IsNaT)
                throw new InvalidOperationException("DateTime64 is NaT; cannot be converted to System.DateTimeOffset.");
            if (!IsValidDateTime)
                throw new InvalidOperationException($"DateTime64 ticks {_ticks} are outside System.DateTime's legal range.");
            return new DateTimeOffset(_ticks, TimeSpan.Zero);
        }

        /// <summary>
        /// Convert to <see cref="System.DateTimeOffset"/>, returning <paramref name="fallback"/>
        /// for NaT / out-of-range values instead of throwing.
        /// </summary>
        public DateTimeOffset ToDateTimeOffset(DateTimeOffset fallback)
        {
            if (IsNaT || !IsValidDateTime) return fallback;
            return new DateTimeOffset(_ticks, TimeSpan.Zero);
        }

        /// <summary>Try to convert to <see cref="System.DateTimeOffset"/>.</summary>
        public bool TryToDateTimeOffset(out DateTimeOffset result)
        {
            if (IsNaT || !IsValidDateTime)
            {
                result = new DateTimeOffset(DateTime.MinValue, TimeSpan.Zero);
                return false;
            }
            result = new DateTimeOffset(_ticks, TimeSpan.Zero);
            return true;
        }

        // ---------------------------------------------------------------------
        // Arithmetic — the minimum needed for NumPy-style dt64 + td64 math.
        // NaT propagates; overflow saturates to NaT.
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public DateTime64 Add(TimeSpan value) => AddTicks(value.Ticks);

        /// <summary>Subtract a <see cref="TimeSpan"/>. NaT propagates; overflow saturates to NaT.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public DateTime64 Subtract(TimeSpan value) => AddTicks(unchecked(-value.Ticks));

        /// <summary>
        /// Difference as a <see cref="TimeSpan"/>. If either operand is NaT,
        /// returns <see cref="TimeSpan.MinValue"/> (TimeSpan's NaT-equivalent,
        /// since <c>TimeSpan.MinValue.Ticks == long.MinValue</c>).
        /// </summary>
        public TimeSpan Subtract(DateTime64 other)
        {
            if (IsNaT || other.IsNaT) return TimeSpan.MinValue;
            return new TimeSpan(unchecked(_ticks - other._ticks));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DateTime64 operator +(DateTime64 d, TimeSpan t) => d.Add(t);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DateTime64 operator -(DateTime64 d, TimeSpan t) => d.Subtract(t);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TimeSpan operator -(DateTime64 d1, DateTime64 d2) => d1.Subtract(d2);

        // ---------------------------------------------------------------------
        // Equality / Comparison
        //
        // .NET convention vs NumPy semantics:
        //   • Equals() returns true for bit-equal ticks (NaT.Equals(NaT) == true)
        //     so GetHashCode honors the Equals → equal-hash contract, and NaT
        //     can be used as a Dictionary/HashSet key. Mirrors System.Double,
        //     where double.NaN.Equals(double.NaN) is true.
        //   • operator == / != / < / > / <= / >= follow NumPy (NaT vs anything
        //     → false for ==/</>/<=/>=, true for !=). Mirrors System.Double,
        //     where double.NaN == double.NaN is false.
        // ---------------------------------------------------------------------

        /// <summary>Bitwise tick equality (<c>NaT.Equals(NaT)</c> returns <see langword="true"/>).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Equals(DateTime64 other) => _ticks == other._ticks;

        public override bool Equals([NotNullWhen(true)] object? value)
            => value is DateTime64 d && Equals(d);

        public static bool Equals(DateTime64 t1, DateTime64 t2) => t1.Equals(t2);

        public override int GetHashCode() => _ticks.GetHashCode();

        /// <summary>Compares by ticks. NaT sorts before every other value (as the smallest int64).</summary>
        public static int Compare(DateTime64 t1, DateTime64 t2)
        {
            long a = t1._ticks, b = t2._ticks;
            if (a < b) return -1;
            if (a > b) return 1;
            return 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int CompareTo(DateTime64 value) => Compare(this, value);

        public int CompareTo(object? value)
        {
            if (value is null) return 1;
            if (value is DateTime64 d) return Compare(this, d);
            if (value is DateTime dt) return Compare(this, new DateTime64(dt));
            throw new ArgumentException("Object must be of type DateTime64 or DateTime.", nameof(value));
        }

        // Operator semantics follow NumPy (NaT vs anything = false for ordering / ==).
        public static bool operator ==(DateTime64 d1, DateTime64 d2)
            => !d1.IsNaT && !d2.IsNaT && d1._ticks == d2._ticks;

        public static bool operator !=(DateTime64 d1, DateTime64 d2)
            => d1.IsNaT || d2.IsNaT || d1._ticks != d2._ticks;

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

        private const string NaTString = "NaT";

        /// <summary>
        /// Formats as ISO-8601 (<see cref="DateTime"/>'s <c>"o"</c> format) for
        /// in-range values, <c>"NaT"</c> for NaT, and <c>"DateTime64(ticks=N)"</c>
        /// for values outside <see cref="System.DateTime"/>'s range.
        /// </summary>
        public override string ToString()
        {
            if (IsNaT) return NaTString;
            if (!IsValidDateTime) return $"DateTime64(ticks={_ticks})";
            return new DateTime(_ticks).ToString("o", CultureInfo.InvariantCulture);
        }

        public string ToString(string? format) => ToString(format, CultureInfo.CurrentCulture);

        public string ToString(IFormatProvider? provider) => ToString(null, provider);

        public string ToString(string? format, IFormatProvider? provider)
        {
            if (IsNaT) return NaTString;
            if (!IsValidDateTime) return $"DateTime64(ticks={_ticks})";
            // ISO-8601 by default (NumPy's datetime64 str() uses ISO-8601-like text).
            if (string.IsNullOrEmpty(format)) format = "o";
            return new DateTime(_ticks).ToString(format, provider ?? CultureInfo.InvariantCulture);
        }

        /// <summary>
        /// Non-allocating formatter: writes directly to <paramref name="destination"/>
        /// when possible via <see cref="DateTime.TryFormat"/>.
        /// </summary>
        public bool TryFormat(Span<char> destination, out int charsWritten,
                              ReadOnlySpan<char> format = default, IFormatProvider? provider = null)
        {
            if (IsNaT)
                return TryCopy(NaTString, destination, out charsWritten);

            if (!IsValidDateTime)
            {
                // Cold path for out-of-.NET-range values. Allocate here only — rare.
                string s = $"DateTime64(ticks={_ticks})";
                return TryCopy(s, destination, out charsWritten);
            }

            if (format.IsEmpty) format = "o";
            return new DateTime(_ticks).TryFormat(destination, out charsWritten, format,
                                                   provider ?? CultureInfo.InvariantCulture);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool TryCopy(string source, Span<char> destination, out int charsWritten)
        {
            if (destination.Length < source.Length)
            {
                charsWritten = 0;
                return false;
            }
            source.AsSpan().CopyTo(destination);
            charsWritten = source.Length;
            return true;
        }

        // ---------------------------------------------------------------------
        // Minimal Parse / TryParse — just enough to round-trip our ToString()
        // and handle the "NaT" literal. Full calendar parsing delegates to
        // System.DateTime, which already has exhaustive locale / format support.
        // ---------------------------------------------------------------------

        /// <summary>
        /// Parses a string produced by <see cref="ToString()"/>. Case-sensitive
        /// <c>"NaT"</c> literal returns <see cref="NaT"/>. Otherwise delegates
        /// to <see cref="DateTime.Parse(string, IFormatProvider?)"/>.
        /// </summary>
        public static DateTime64 Parse(string s)
        {
            if (s is null) throw new ArgumentNullException(nameof(s));
            if (s == NaTString) return NaT;
            return new DateTime64(DateTime.Parse(s, CultureInfo.InvariantCulture));
        }

        public static DateTime64 Parse(string s, IFormatProvider? provider)
        {
            if (s is null) throw new ArgumentNullException(nameof(s));
            if (s == NaTString) return NaT;
            return new DateTime64(DateTime.Parse(s, provider));
        }

        public static bool TryParse([NotNullWhen(true)] string? s, out DateTime64 result)
        {
            if (s is null) { result = default; return false; }
            if (s == NaTString) { result = NaT; return true; }
            if (DateTime.TryParse(s, CultureInfo.InvariantCulture, DateTimeStyles.None, out var dt))
            {
                result = new DateTime64(dt);
                return true;
            }
            result = default;
            return false;
        }

        public static bool TryParse([NotNullWhen(true)] string? s, IFormatProvider? provider,
                                    DateTimeStyles styles, out DateTime64 result)
        {
            if (s is null) { result = default; return false; }
            if (s == NaTString) { result = NaT; return true; }
            if (DateTime.TryParse(s, provider, styles, out var dt))
            {
                result = new DateTime64(dt);
                return true;
            }
            result = default;
            return false;
        }

        // ---------------------------------------------------------------------
        // IConvertible — needed for Convert.ChangeType + NumSharp's type-switch
        // paths. Values convert using the raw int64 tick count (NumPy parity).
        //
        // GetTypeCode returns TypeCode.Object (NOT TypeCode.DateTime) because
        // DateTime64 is NOT System.DateTime; we want Convert.ChangeType to treat
        // it as "unknown-to-IConvertible-fast-path" and fall back to ToType.
        // ---------------------------------------------------------------------

        TypeCode IConvertible.GetTypeCode() => TypeCode.Object;

        bool IConvertible.ToBoolean(IFormatProvider? provider) => _ticks != 0L;     // NaT.Ticks = long.MinValue ≠ 0 → True (NumPy parity)
        sbyte IConvertible.ToSByte(IFormatProvider? provider)  => unchecked((sbyte)_ticks);
        byte IConvertible.ToByte(IFormatProvider? provider)    => unchecked((byte)_ticks);
        short IConvertible.ToInt16(IFormatProvider? provider)  => unchecked((short)_ticks);
        ushort IConvertible.ToUInt16(IFormatProvider? provider)=> unchecked((ushort)_ticks);
        int IConvertible.ToInt32(IFormatProvider? provider)    => unchecked((int)_ticks);
        uint IConvertible.ToUInt32(IFormatProvider? provider)  => unchecked((uint)_ticks);
        long IConvertible.ToInt64(IFormatProvider? provider)   => _ticks;
        ulong IConvertible.ToUInt64(IFormatProvider? provider) => unchecked((ulong)_ticks);
        char IConvertible.ToChar(IFormatProvider? provider)    => unchecked((char)_ticks);
        float IConvertible.ToSingle(IFormatProvider? provider) => (float)_ticks;
        double IConvertible.ToDouble(IFormatProvider? provider)=> (double)_ticks;
        decimal IConvertible.ToDecimal(IFormatProvider? provider) => (decimal)_ticks;
        DateTime IConvertible.ToDateTime(IFormatProvider? provider) => ToDateTime(DateTime.MinValue);
        string IConvertible.ToString(IFormatProvider? provider) => ToString(null, provider);

        object IConvertible.ToType(Type conversionType, IFormatProvider? provider)
        {
            if (conversionType == typeof(DateTime64)) return this;
            if (conversionType == typeof(DateTime)) return ToDateTime(DateTime.MinValue);
            if (conversionType == typeof(DateTimeOffset))
                return ToDateTimeOffset(new DateTimeOffset(DateTime.MinValue, TimeSpan.Zero));
            if (conversionType == typeof(TimeSpan)) return new TimeSpan(_ticks);
            if (conversionType == typeof(long)) return _ticks;
            if (conversionType == typeof(ulong)) return unchecked((ulong)_ticks);
            if (conversionType == typeof(double)) return (double)_ticks;
            if (conversionType == typeof(string)) return ToString(null, provider);
            return Convert.ChangeType(_ticks, conversionType, provider);
        }

        // ---------------------------------------------------------------------
        // Hardened float → int64 bounds check
        //
        // Used by Converts.ToDateTime64(double). Keeping it here (as an
        // internal helper) ensures the rule stays in sync with the struct's
        // NaT semantics and is not duplicated across call sites.
        // ---------------------------------------------------------------------

        /// <summary>
        /// Converts a <see cref="double"/> to a <see cref="DateTime64"/> using
        /// NumPy's <c>float → datetime64</c> rules: NaN, ±Inf, and values
        /// outside <c>[long.MinValue, long.MaxValue]</c> → <see cref="NaT"/>;
        /// otherwise truncate toward zero and wrap in <see cref="DateTime64"/>.
        /// </summary>
        internal static DateTime64 FromDoubleOrNaT(double value)
        {
            if (double.IsNaN(value) || double.IsInfinity(value)) return NaT;
            // 2^63 is the exclusive upper bound: (double)long.MaxValue rounds up
            // to 2^63 which cannot be represented as a signed int64.
            if (value >= Int64MaxAsDoubleUpperExclusive) return NaT;
            // 2^63 negated is exactly (double)long.MinValue. Anything strictly
            // smaller overflows; anything ≥ long.MinValue is castable. We must
            // exclude the exact long.MinValue value too because a DateTime64
            // with that tick count is NaT — returning "valid dt64 == NaT" would
            // be indistinguishable from an actual overflow in downstream logic.
            if (value <= Int64MinAsDoubleLowerExclusive) return NaT;
            return new DateTime64((long)value);
        }
    }
}
