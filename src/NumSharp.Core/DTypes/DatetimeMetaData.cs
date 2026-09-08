using System;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    /// <summary>
    ///     NumPy's <c>NPY_DATETIMEUNIT</c> (<c>ndarraytypes.h</c>): the time unit a <c>datetime64</c> / <c>timedelta64</c>
    ///     value counts in. The enum values are NumPy's — including the gap at 3 where the 1.6 business-day unit sat —
    ///     because unit ORDER is semantic (a larger value is a finer unit) and NumPy's casting rules compare them.
    /// </summary>
    public enum NPY_DATETIMEUNIT
    {
        /// <summary>Error or undetermined.</summary>
        NPY_FR_ERROR = -1,

        /// <summary>Years.</summary>
        NPY_FR_Y = 0,
        /// <summary>Months.</summary>
        NPY_FR_M = 1,
        /// <summary>Weeks.</summary>
        NPY_FR_W = 2,
        /* Gap where 1.6 NPY_FR_B (value 3) was */
        /// <summary>Days.</summary>
        NPY_FR_D = 4,
        /// <summary>Hours.</summary>
        NPY_FR_h = 5,
        /// <summary>Minutes.</summary>
        NPY_FR_m = 6,
        /// <summary>Seconds.</summary>
        NPY_FR_s = 7,
        /// <summary>Milliseconds.</summary>
        NPY_FR_ms = 8,
        /// <summary>Microseconds.</summary>
        NPY_FR_us = 9,
        /// <summary>Nanoseconds.</summary>
        NPY_FR_ns = 10,
        /// <summary>Picoseconds.</summary>
        NPY_FR_ps = 11,
        /// <summary>Femtoseconds.</summary>
        NPY_FR_fs = 12,
        /// <summary>Attoseconds.</summary>
        NPY_FR_as = 13,
        /// <summary>Unbound units — can convert to anything (the unit of a bare <c>M8</c> / <c>m8</c> and of NaT).</summary>
        NPY_FR_GENERIC = 14,
    }

    /// <summary>
    ///     The per-INSTANCE parameter of a <c>datetime64</c> / <c>timedelta64</c> descriptor — NumPy's
    ///     <c>PyArray_DatetimeMetaData</c> (<c>{ NPY_DATETIMEUNIT base; int num; }</c>): the unit and an integer
    ///     multiplier, so <c>datetime64[5m]</c> counts five-minute ticks. This is what makes the datetime pair
    ///     PARAMETRIC: <c>M8[ns]</c> and <c>M8[s]</c> share one <see cref="DTypeMeta"/> and one 8-byte storage but are
    ///     different dtypes, and promotion / casting between units is arithmetic on these two fields.
    /// </summary>
    /// <remarks>
    ///     Everything here is a line-for-line port of <c>numpy/_core/src/multiarray/datetime.c</c> (2.4.2):
    ///     the metadata grammar <c>[&lt;num&gt;&lt;unit&gt;]</c> / <c>[&lt;unit&gt;/&lt;den&gt;]</c>
    ///     (<c>parse_datetime_metadata_from_metastr</c>, <c>parse_datetime_extended_unit_from_string</c>,
    ///     <c>parse_datetime_unit_from_string</c>, <c>convert_datetime_divisor_to_multiple</c>), the formatting
    ///     (<c>metastr_to_unicode</c>), the unit factors (<c>get_datetime_units_factor</c>,
    ///     <c>get_datetime_conversion_factor</c>), divisibility (<c>datetime_metadata_divides</c>), the GCD used by
    ///     promotion (<c>compute_datetime_metadata_greatest_common_divisor</c> — strict about the non-linear Y/M units
    ///     for timedelta operands, relaxed for datetime) and the unit/metadata casting rules
    ///     (<c>can_cast_datetime64_units</c> / <c>_metadata</c>, <c>can_cast_timedelta64_units</c> / <c>_metadata</c>).
    ///     Every error text is NumPy's verbatim, with one deliberate difference: a zero divisor (<c>M8[s/0]</c>) crashes
    ///     the CPython interpreter with an integer division by zero; NumSharp raises the metadata <see cref="TypeError"/>.
    /// </remarks>
    public readonly struct DatetimeMetaData : IEquatable<DatetimeMetaData>
    {
        /// <summary>NumPy's <c>NPY_DATETIME_NUMUNITS</c> (one more than the number of real units, because of the gap).</summary>
        public const int NumUnits = (int)NPY_DATETIMEUNIT.NPY_FR_GENERIC + 1;

        /// <summary>NumPy's <c>_datetime_strings</c>, indexed by <see cref="NPY_DATETIMEUNIT"/>.</summary>
        internal static readonly string[] UnitStrings =
        {
            "Y", "M", "W", "<invalid>", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as", "generic",
        };

        /// <summary>
        ///     NumPy's <c>_datetime_factors</c>: the scale factor from each unit to the next finer one (years and months
        ///     are non-linear and carry a placeholder 1; attoseconds are the smallest; generic has no conversion).
        /// </summary>
        private static readonly uint[] _datetimeFactors =
        {
            1,    /* Years - not used */
            1,    /* Months - not used */
            7,    /* Weeks -> Days */
            1,    /* Business Days - was removed but a gap still exists in the enum */
            24,   /* Days -> Hours */
            60,   /* Hours -> Minutes */
            60,   /* Minutes -> Seconds */
            1000, /* s -> ms */
            1000, /* ms -> us */
            1000, /* us -> ns */
            1000, /* ns -> ps */
            1000, /* ps -> fs */
            1000, /* fs -> as */
            1,    /* Attoseconds are the smallest base unit */
            0,    /* Generic units don't have a conversion */
        };

        /// <summary>The unit (<c>base</c> in NumPy's struct).</summary>
        public readonly NPY_DATETIMEUNIT Base;

        /// <summary>The unit multiplier (<c>num</c> in NumPy's struct); 1 for a plain unit.</summary>
        public readonly int Num;

        /// <summary>The generic (unbound) metadata — the parameter of a bare <c>M8</c> / <c>m8</c>.</summary>
        public static readonly DatetimeMetaData Generic = new DatetimeMetaData(NPY_DATETIMEUNIT.NPY_FR_GENERIC, 1);

        public DatetimeMetaData(NPY_DATETIMEUNIT @base, int num = 1)
        {
            Base = @base;
            Num = num;
        }

        /// <summary>True for the generic (unbound) unit.</summary>
        public bool IsGeneric => Base == NPY_DATETIMEUNIT.NPY_FR_GENERIC;

        /// <summary>The unit string NumPy's <c>np.datetime_data</c> returns (<c>"ns"</c>, <c>"Y"</c>, <c>"generic"</c>).</summary>
        public string UnitString
        {
            get
            {
                int b = (int)Base;
                if (b < 0 || b >= NumUnits)
                    throw new RuntimeError("NumPy datetime metadata is corrupted");
                return UnitStrings[b];
            }
        }

        /// <summary>
        ///     NumPy's <c>metastr_to_unicode</c>: <c>"[ns]"</c> / <c>"[10ns]"</c>, and <c>""</c> for generic units — or,
        ///     with <paramref name="skipBrackets"/>, <c>"ns"</c> / <c>"10ns"</c> / <c>"generic"</c>.
        /// </summary>
        public string ToString(bool skipBrackets)
        {
            if (IsGeneric)
                return skipBrackets ? "generic" : "";

            string basestr = UnitString;
            if (Num == 1)
                return skipBrackets ? basestr : "[" + basestr + "]";
            return skipBrackets ? Num + basestr : "[" + Num + basestr + "]";
        }

        /// <summary>The bracketed metadata string (<c>"[ns]"</c>, <c>"[10ns]"</c>, <c>""</c> for generic).</summary>
        public override string ToString() => ToString(skipBrackets: false);

        /// <inheritdoc/>
        public bool Equals(DatetimeMetaData other) => Base == other.Base && Num == other.Num;

        /// <inheritdoc/>
        public override bool Equals(object obj) => obj is DatetimeMetaData other && Equals(other);

        /// <inheritdoc/>
        public override int GetHashCode() => HashCode.Combine((int)Base, Num);

        public static bool operator ==(DatetimeMetaData left, DatetimeMetaData right) => left.Equals(right);
        public static bool operator !=(DatetimeMetaData left, DatetimeMetaData right) => !left.Equals(right);

        // ------------------------------------------------------------------------------------------------
        // Parsing — datetime.c: parse_datetime_metadata_from_metastr & friends
        // ------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Parses a metadata string — the part of a datetime typestr after <c>M8</c> / <c>datetime64</c>:
        ///     <c>""</c> (generic), <c>"[ns]"</c>, <c>"[10ns]"</c>, <c>"[s/2]"</c> (a divisor, rewritten to a multiple of a
        ///     finer unit: <c>[500ms]</c>). Port of <c>parse_datetime_metadata_from_metastr</c>.
        /// </summary>
        /// <exception cref="TypeError">NumPy's verbatim <c>Invalid datetime metadata string "…"[ at position N]</c> /
        /// <c>Invalid datetime unit in metadata string "…"</c>.</exception>
        /// <exception cref="ValueError">A divisor that is not a multiple of a lower unit, or a divisor on generic units.</exception>
        public static DatetimeMetaData Parse(string metastr)
        {
            if (metastr == null)
                throw new ArgumentNullException(nameof(metastr));

            int len = metastr.Length;

            /* Treat the empty string as generic units */
            if (len == 0)
                return Generic;

            int substr = 0;

            /* The metadata string must start with a '[' */
            if (len < 3 || metastr[substr++] != '[')
                throw BadInput(metastr, len < 3 ? 0 : substr);

            int substrend = substr;
            while (substrend < len && metastr[substrend] != ']')
                ++substrend;
            if (substrend == len || substr == substrend)
                throw BadInput(metastr, substrend);

            /* Parse the extended unit inside the [] */
            var meta = ParseExtendedUnit(metastr, substr, substrend, metastr);

            substr = substrend + 1;

            if (substr != len)
                throw BadInput(metastr, substr);

            return meta;
        }

        /// <summary>
        ///     Port of <c>parse_datetime_extended_unit_from_string</c>: <c>[num]unit[/den]</c> over
        ///     <paramref name="s"/>[<paramref name="start"/>, <paramref name="end"/>); <paramref name="metastr"/> is
        ///     the whole bracketed string (positions in error messages index into it).
        /// </summary>
        internal static DatetimeMetaData ParseExtendedUnit(string s, int start, int end, string metastr)
        {
            int substr = start;

            /* First comes an optional integer multiplier */
            long multiplier = StrtolConst(s, substr, end, out int substrend, out bool overflow);
            int num;
            if (substrend == substr)
            {
                num = 1;
            }
            else
            {
                // check for 32-bit integer overflow (NumPy re-reads the digits with strtoll for this test)
                if (overflow || multiplier > int.MaxValue || multiplier < 0)
                    throw BadInput(metastr, substr);
                num = (int)multiplier;
            }
            substr = substrend;

            /* Next comes the unit itself, followed by either '/' or the string end */
            substrend = substr;
            while (substrend < end && s[substrend] != '/')
                ++substrend;
            if (substr == substrend)
                throw BadInput(metastr, substr);

            var @base = ParseUnit(s, substr, substrend - substr, metastr);
            substr = substrend;

            /* Next comes an optional integer denominator */
            int den = 1;
            if (substr < end && s[substr] == '/')
            {
                substr++;
                long denValue = StrtolConst(s, substr, s.Length, out substrend, out _);
                /* If the '/' exists, there must be a number followed by ']' */
                if (substr == substrend || substrend >= s.Length || s[substrend] != ']')
                    throw BadInput(metastr, substr);
                den = (int)denValue;
                substr = substrend + 1;
            }
            else if (substr != end)
            {
                throw BadInput(metastr, substr);
            }

            var meta = new DatetimeMetaData(@base, num);
            if (den != 1)
                meta = ConvertDivisorToMultiple(meta, den, metastr);

            return meta;
        }

        /// <summary>
        ///     Parses a bare unit string (<c>"ns"</c>, <c>"Y"</c>, <c>"μs"</c>, <c>"generic"</c>) — port of
        ///     <c>parse_datetime_unit_from_string</c> with no surrounding metadata string.
        /// </summary>
        /// <exception cref="TypeError"><c>Invalid datetime unit "…" in metadata</c>.</exception>
        public static NPY_DATETIMEUNIT ParseUnit(string unit)
        {
            if (unit == null)
                throw new ArgumentNullException(nameof(unit));
            return ParseUnit(unit, 0, unit.Length, metastr: null);
        }

        /// <summary>Port of <c>parse_datetime_unit_from_string</c> over <paramref name="s"/>[<paramref name="start"/>, +<paramref name="len"/>).</summary>
        internal static NPY_DATETIMEUNIT ParseUnit(string s, int start, int len, string metastr)
        {
            /* Use switch statements so the compiler can make it fast */
            if (len == 1)
            {
                switch (s[start])
                {
                    case 'Y': return NPY_DATETIMEUNIT.NPY_FR_Y;
                    case 'M': return NPY_DATETIMEUNIT.NPY_FR_M;
                    case 'W': return NPY_DATETIMEUNIT.NPY_FR_W;
                    case 'D': return NPY_DATETIMEUNIT.NPY_FR_D;
                    case 'h': return NPY_DATETIMEUNIT.NPY_FR_h;
                    case 'm': return NPY_DATETIMEUNIT.NPY_FR_m;
                    case 's': return NPY_DATETIMEUNIT.NPY_FR_s;
                }
            }
            /* All the two-letter units are variants of seconds */
            else if (len == 2 && s[start + 1] == 's')
            {
                switch (s[start])
                {
                    case 'm': return NPY_DATETIMEUNIT.NPY_FR_ms;
                    case 'u': return NPY_DATETIMEUNIT.NPY_FR_us;
                    case 'n': return NPY_DATETIMEUNIT.NPY_FR_ns;
                    case 'p': return NPY_DATETIMEUNIT.NPY_FR_ps;
                    case 'f': return NPY_DATETIMEUNIT.NPY_FR_fs;
                    case 'a': return NPY_DATETIMEUNIT.NPY_FR_as;
                    /* greek small letter mu (NumPy matches its UTF-8 bytes; here it is one UTF-16 unit) */
                    case 'μ': return NPY_DATETIMEUNIT.NPY_FR_us;
                }
            }
            else if (len == 7 && string.CompareOrdinal(s, start, "generic", 0, 7) == 0)
            {
                return NPY_DATETIMEUNIT.NPY_FR_GENERIC;
            }

            /* If nothing matched, it's an error */
            if (metastr == null)
                throw new TypeError($"Invalid datetime unit \"{s.Substring(start, len)}\" in metadata");
            throw new TypeError($"Invalid datetime unit in metadata string \"{metastr}\"");
        }

        /// <summary>
        ///     NumPy's <c>_multiples_table</c>: for each coarse unit, the candidate multiples of finer units a divisor is
        ///     tried against, and the finer unit each candidate stands for. Rows come in pairs (multiples, base units);
        ///     the row for seconds-and-finer is completed per call (<c>base+1</c>, <c>base+2</c>). Trailing zeros are
        ///     NumPy's own (the arrays are <c>[16][4]</c>) and are reachable — kept so the answer stays identical.
        /// </summary>
        private static readonly int[][] _multiplesTable =
        {
            new[] { 12, 52, 365, 0 },                                                                    /* NPY_FR_Y */
            new[] { (int)NPY_DATETIMEUNIT.NPY_FR_M, (int)NPY_DATETIMEUNIT.NPY_FR_W, (int)NPY_DATETIMEUNIT.NPY_FR_D, 0 },
            new[] { 4, 30, 720, 0 },                                                                     /* NPY_FR_M */
            new[] { (int)NPY_DATETIMEUNIT.NPY_FR_W, (int)NPY_DATETIMEUNIT.NPY_FR_D, (int)NPY_DATETIMEUNIT.NPY_FR_h, 0 },
            new[] { 7, 168, 10080, 0 },                                                                  /* NPY_FR_W */
            new[] { (int)NPY_DATETIMEUNIT.NPY_FR_D, (int)NPY_DATETIMEUNIT.NPY_FR_h, (int)NPY_DATETIMEUNIT.NPY_FR_m, 0 },
            new[] { 0, 0, 0, 0 },                                                                        /* Gap for removed NPY_FR_B */
            new[] { 0, 0, 0, 0 },
            new[] { 24, 1440, 86400, 0 },                                                                /* NPY_FR_D */
            new[] { (int)NPY_DATETIMEUNIT.NPY_FR_h, (int)NPY_DATETIMEUNIT.NPY_FR_m, (int)NPY_DATETIMEUNIT.NPY_FR_s, 0 },
            new[] { 60, 3600, 0, 0 },                                                                    /* NPY_FR_h */
            new[] { (int)NPY_DATETIMEUNIT.NPY_FR_m, (int)NPY_DATETIMEUNIT.NPY_FR_s, 0, 0 },
            new[] { 60, 60000, 0, 0 },                                                                   /* NPY_FR_m */
            new[] { (int)NPY_DATETIMEUNIT.NPY_FR_s, (int)NPY_DATETIMEUNIT.NPY_FR_ms, 0, 0 },
            new[] { 1000, 1000000, 0, 0 },                                                               /* >=NPY_FR_s */
            new[] { 0, 0, 0, 0 },
        };

        /// <summary>
        ///     Port of <c>convert_datetime_divisor_to_multiple</c>: translates a divisor into a multiple of a smaller
        ///     unit (<c>[s/2]</c> → <c>[500ms]</c>, <c>[Y/2]</c> → <c>[6M]</c>, <c>[D/3]</c> → <c>[8h]</c>, <c>[W/2]</c> →
        ///     <c>[84h]</c>). <paramref name="metastr"/> is used for the error message and may be null.
        /// </summary>
        /// <exception cref="ValueError"><c>Can't use 'den' divisor with generic units</c> /
        /// <c>divisor (N) is not a multiple of a lower-unit in datetime metadata "…"</c>.</exception>
        /// <exception cref="TypeError">A zero divisor — where NumPy divides by zero and crashes.</exception>
        internal static DatetimeMetaData ConvertDivisorToMultiple(DatetimeMetaData meta, int den, string metastr)
        {
            if (meta.Base == NPY_DATETIMEUNIT.NPY_FR_GENERIC)
                throw new ValueError("Can't use 'den' divisor with generic units");

            if (den == 0)
            {
                // NumPy: integer division by zero (interpreter crash). Report it as malformed metadata instead.
                throw new TypeError(metastr == null
                    ? "Invalid datetime metadata string \"" + meta.ToString(false) + "\""
                    : $"Invalid datetime metadata string \"{metastr}\" at position {Math.Max(0, metastr.IndexOf('/') + 1)}");
            }

            int num = 3;
            if (meta.Base == NPY_DATETIMEUNIT.NPY_FR_W)
                num = 4;
            else if (meta.Base > NPY_DATETIMEUNIT.NPY_FR_D)
                num = 2;

            int[] totry;
            int[] baseunit;
            if (meta.Base >= NPY_DATETIMEUNIT.NPY_FR_s)
            {
                /* _multiplies_table only has entries up to NPY_FR_s */
                int ind = ((int)NPY_DATETIMEUNIT.NPY_FR_s - (int)NPY_DATETIMEUNIT.NPY_FR_Y) * 2;
                totry = _multiplesTable[ind];
                baseunit = new[] { (int)meta.Base + 1, (int)meta.Base + 2, 0, 0 };
                if (meta.Base == NPY_DATETIMEUNIT.NPY_FR_as - 1)
                    num = 1;
                if (meta.Base == NPY_DATETIMEUNIT.NPY_FR_as)
                    num = 0;
            }
            else
            {
                int ind = ((int)meta.Base - (int)NPY_DATETIMEUNIT.NPY_FR_Y) * 2;
                totry = _multiplesTable[ind];
                baseunit = _multiplesTable[ind + 1];
            }

            int i, q = 0;
            for (i = 0; i < num; i++)
            {
                q = totry[i] / den;
                int r = totry[i] % den;
                if (r == 0)
                    break;
            }

            if (i == num)
            {
                if (metastr == null)
                    throw new ValueError($"divisor ({den}) is not a multiple of a lower-unit in datetime metadata");
                throw new ValueError($"divisor ({den}) is not a multiple of a lower-unit in datetime metadata \"{metastr}\"");
            }

            return new DatetimeMetaData((NPY_DATETIMEUNIT)baseunit[i], meta.Num * q);
        }

        /// <summary>
        ///     C <c>strtol(str, &amp;endptr, 10)</c> over <paramref name="s"/>[<paramref name="pos"/>, <paramref name="limit"/>):
        ///     skips leading whitespace, accepts an optional sign and decimal digits. When no digits follow,
        ///     <paramref name="endPos"/> is <paramref name="pos"/> (nothing consumed). <paramref name="overflow"/> reports a
        ///     magnitude past <see cref="long"/> (NumPy's <c>strtoll</c> re-read would saturate).
        /// </summary>
        private static long StrtolConst(string s, int pos, int limit, out int endPos, out bool overflow)
        {
            overflow = false;
            int i = pos;
            while (i < limit && char.IsWhiteSpace(s[i]))
                i++;
            bool negative = false;
            if (i < limit && (s[i] == '+' || s[i] == '-'))
            {
                negative = s[i] == '-';
                i++;
            }
            int digitsStart = i;
            long value = 0;
            while (i < limit && s[i] >= '0' && s[i] <= '9')
            {
                int d = s[i] - '0';
                if (!overflow)
                {
                    if (value > (long.MaxValue - d) / 10)
                        overflow = true;
                    else
                        value = value * 10 + d;
                }
                i++;
            }
            if (i == digitsStart)
            {
                endPos = pos;
                return 0;
            }
            endPos = i;
            if (overflow)
                return negative ? long.MinValue : long.MaxValue;
            return negative ? -value : value;
        }

        private static TypeError BadInput(string metastr, int position)
        {
            if (position != 0)
                return new TypeError($"Invalid datetime metadata string \"{metastr}\" at position {position}");
            return new TypeError($"Invalid datetime metadata string \"{metastr}\"");
        }

        // ------------------------------------------------------------------------------------------------
        // Unit arithmetic — datetime.c: get_datetime_units_factor, datetime_metadata_divides, the GCD
        // ------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Port of <c>get_datetime_units_factor</c>: the scale factor between two units (the caller guarantees
        ///     <paramref name="bigbase"/> is the coarser unit and neither is generic). Returns 0 on overflow — NumPy
        ///     disallows the top 16 bits, a margin far larger than any single factor.
        /// </summary>
        internal static ulong UnitsFactor(NPY_DATETIMEUNIT bigbase, NPY_DATETIMEUNIT littlebase)
        {
            ulong factor = 1;
            var unit = bigbase;
            while (unit < littlebase)
            {
                factor *= _datetimeFactors[(int)unit];
                if ((factor & 0xff00000000000000UL) != 0)
                    return 0;
                ++unit;
            }
            return factor;
        }

        /* Euclidean algorithm on two positive numbers */
        private static ulong Gcd(ulong x, ulong y)
        {
            if (x > y)
                (x, y) = (y, x);
            while (x != y && y != 0)
            {
                ulong tmp = x % y;
                x = y;
                y = tmp;
            }
            return x;
        }

        /// <summary>
        ///     Port of <c>get_datetime_conversion_factor</c>: the reduced fraction <c>num/denom</c> that converts data
        ///     with <paramref name="src"/> metadata into <paramref name="dst"/> metadata. Year and month conversions use
        ///     the 400-year leap cycle average (365.2425 days per year). Overflow yields <c>(0, 0)</c>.
        /// </summary>
        /// <exception cref="ValueError">Converting specific units to generic units.</exception>
        /// <exception cref="OverflowException">The factor overflows (NumPy's <c>OverflowError</c>).</exception>
        internal static (long num, long denom) ConversionFactor(in DatetimeMetaData src, in DatetimeMetaData dst)
        {
            /* Generic units change to the destination with no conversion factor */
            if (src.Base == NPY_DATETIMEUNIT.NPY_FR_GENERIC)
                return (1, 1);
            /*
             * Converting to a generic unit from something other than a generic
             * unit is an error.
             */
            if (dst.Base == NPY_DATETIMEUNIT.NPY_FR_GENERIC)
                throw new ValueError("Cannot convert from specific units to generic units in NumPy datetimes or timedeltas");

            NPY_DATETIMEUNIT srcBase, dstBase;
            bool swapped;
            if (src.Base <= dst.Base)
            {
                srcBase = src.Base;
                dstBase = dst.Base;
                swapped = false;
            }
            else
            {
                srcBase = dst.Base;
                dstBase = src.Base;
                swapped = true;
            }

            ulong num = 1, denom = 1;
            if (srcBase != dstBase)
            {
                /*
                 * Conversions between years/months and other units use
                 * the factor averaged over the 400 year leap year cycle.
                 */
                if (srcBase == NPY_DATETIMEUNIT.NPY_FR_Y)
                {
                    if (dstBase == NPY_DATETIMEUNIT.NPY_FR_M)
                    {
                        num *= 12;
                    }
                    else if (dstBase == NPY_DATETIMEUNIT.NPY_FR_W)
                    {
                        num *= 97 + 400 * 365;
                        denom *= 400 * 7;
                    }
                    else
                    {
                        /* Year -> Day */
                        num *= 97 + 400 * 365;
                        denom *= 400;
                        /* Day -> dst_base */
                        num *= UnitsFactor(NPY_DATETIMEUNIT.NPY_FR_D, dstBase);
                    }
                }
                else if (srcBase == NPY_DATETIMEUNIT.NPY_FR_M)
                {
                    if (dstBase == NPY_DATETIMEUNIT.NPY_FR_W)
                    {
                        num *= 97 + 400 * 365;
                        denom *= 400 * 12 * 7;
                    }
                    else
                    {
                        /* Month -> Day */
                        num *= 97 + 400 * 365;
                        denom *= 400 * 12;
                        /* Day -> dst_base */
                        num *= UnitsFactor(NPY_DATETIMEUNIT.NPY_FR_D, dstBase);
                    }
                }
                else
                {
                    num *= UnitsFactor(srcBase, dstBase);
                }
            }

            /* If something overflowed, make both num and denom 0 */
            if (num == 0)
                throw new OverflowException(
                    $"Integer overflow while computing the conversion factor between NumPy datetime units {UnitStrings[(int)srcBase]} and {UnitStrings[(int)dstBase]}");

            /* Swap the numerator and denominator if necessary */
            if (swapped)
                (num, denom) = (denom, num);

            num *= (ulong)src.Num;
            denom *= (ulong)dst.Num;

            /* Return as a fraction in reduced form */
            ulong gcd = Gcd(num, denom);
            return ((long)(num / gcd), (long)(denom / gcd));
        }

        /// <summary>
        ///     Port of <c>datetime_metadata_divides</c>: whether <paramref name="divisor"/> divides evenly into
        ///     <paramref name="dividend"/> — the "is casting towards this finer unit exact" test. Generic dividends divide
        ///     into everything; nothing specific divides into generic. Years/months are incompatible with every other unit
        ///     (except each other) when <paramref name="strictWithNonlinearUnits"/> — the timedelta rule; the datetime rule
        ///     says "yes" there ("could do something complicated").
        /// </summary>
        public static bool Divides(in DatetimeMetaData dividend, in DatetimeMetaData divisor, bool strictWithNonlinearUnits)
        {
            /*
             * Any unit can always divide into generic units. In other words, we
             * should be able to convert generic units into any more specific unit.
             */
            if (dividend.Base == NPY_DATETIMEUNIT.NPY_FR_GENERIC)
                return true;
            /*
             * However, generic units cannot always divide into more specific units.
             * We cannot safely convert datetimes with units back into generic units.
             */
            if (divisor.Base == NPY_DATETIMEUNIT.NPY_FR_GENERIC)
                return false;

            ulong num1 = (ulong)(uint)dividend.Num;
            ulong num2 = (ulong)(uint)divisor.Num;

            /* If the bases are different, factor in a conversion */
            if (dividend.Base != divisor.Base)
            {
                /*
                 * Years and Months are incompatible with
                 * all other units (except years and months are compatible
                 * with each other).
                 */
                if (dividend.Base == NPY_DATETIMEUNIT.NPY_FR_Y)
                {
                    if (divisor.Base == NPY_DATETIMEUNIT.NPY_FR_M)
                        num1 *= 12;
                    else if (strictWithNonlinearUnits)
                        return false;
                    else
                        return true; /* Could do something complicated here */
                }
                else if (divisor.Base == NPY_DATETIMEUNIT.NPY_FR_Y)
                {
                    if (dividend.Base == NPY_DATETIMEUNIT.NPY_FR_M)
                        num2 *= 12;
                    else if (strictWithNonlinearUnits)
                        return false;
                    else
                        return true;
                }
                else if (dividend.Base == NPY_DATETIMEUNIT.NPY_FR_M || divisor.Base == NPY_DATETIMEUNIT.NPY_FR_M)
                {
                    if (strictWithNonlinearUnits)
                        return false;
                    return true;
                }

                /* Take the greater base (unit sizes are decreasing in enum) */
                if (dividend.Base > divisor.Base)
                {
                    num2 *= UnitsFactor(divisor.Base, dividend.Base);
                    if (num2 == 0)
                        return false;
                }
                else
                {
                    num1 *= UnitsFactor(dividend.Base, divisor.Base);
                    if (num1 == 0)
                        return false;
                }
            }

            /* Crude, incomplete check for overflow */
            if ((num1 & 0xff00000000000000UL) != 0 || (num2 & 0xff00000000000000UL) != 0)
                return false;

            return num2 != 0 && num1 % num2 == 0;
        }

        /// <summary>
        ///     Port of <c>compute_datetime_metadata_greatest_common_divisor</c> — the metadata of the promoted dtype of
        ///     two temporal operands: the GCD of the two multipliers expressed in the finer unit (<c>[h]</c> + <c>[30m]</c>
        ///     → <c>[30m]</c>, <c>[10s]</c> + <c>[15s]</c> → <c>[5s]</c>). A generic operand adopts the other's metadata.
        ///     Years/months are compatible only with each other; mixing them with a linear unit is an error when the
        ///     corresponding operand is strict (a <c>timedelta64</c>) and relaxed otherwise (a <c>datetime64</c>).
        /// </summary>
        /// <exception cref="TypeError"><c>Cannot get a common metadata divisor for Numpy datetime metadata [Y] and [D]
        /// because they have incompatible nonlinear base time units.</c></exception>
        /// <exception cref="OverflowException"><c>Integer overflow getting a common metadata divisor for NumPy datetime
        /// metadata [as] and [Y].</c></exception>
        public static DatetimeMetaData GreatestCommonDivisor(in DatetimeMetaData meta1, in DatetimeMetaData meta2,
            bool strictWithNonlinearUnits1, bool strictWithNonlinearUnits2)
        {
            /* If either unit is generic, adopt the metadata from the other one */
            if (meta1.Base == NPY_DATETIMEUNIT.NPY_FR_GENERIC)
                return meta2;
            if (meta2.Base == NPY_DATETIMEUNIT.NPY_FR_GENERIC)
                return meta1;

            ulong num1 = (ulong)(uint)meta1.Num;
            ulong num2 = (ulong)(uint)meta2.Num;
            NPY_DATETIMEUNIT @base;

            /* First validate that the units have a reasonable GCD */
            if (meta1.Base == meta2.Base)
            {
                @base = meta1.Base;
            }
            else
            {
                /*
                 * Years and Months are incompatible with
                 * all other units (except years and months are compatible
                 * with each other).
                 */
                if (meta1.Base == NPY_DATETIMEUNIT.NPY_FR_Y)
                {
                    if (meta2.Base == NPY_DATETIMEUNIT.NPY_FR_M)
                    {
                        @base = NPY_DATETIMEUNIT.NPY_FR_M;
                        num1 *= 12;
                    }
                    else if (strictWithNonlinearUnits1)
                        throw IncompatibleUnits(meta1, meta2);
                    else
                        @base = meta2.Base; /* Don't multiply num1 since there is no even factor */
                }
                else if (meta2.Base == NPY_DATETIMEUNIT.NPY_FR_Y)
                {
                    if (meta1.Base == NPY_DATETIMEUNIT.NPY_FR_M)
                    {
                        @base = NPY_DATETIMEUNIT.NPY_FR_M;
                        num2 *= 12;
                    }
                    else if (strictWithNonlinearUnits2)
                        throw IncompatibleUnits(meta1, meta2);
                    else
                        @base = meta1.Base;
                }
                else if (meta1.Base == NPY_DATETIMEUNIT.NPY_FR_M)
                {
                    if (strictWithNonlinearUnits1)
                        throw IncompatibleUnits(meta1, meta2);
                    @base = meta2.Base;
                }
                else if (meta2.Base == NPY_DATETIMEUNIT.NPY_FR_M)
                {
                    if (strictWithNonlinearUnits2)
                        throw IncompatibleUnits(meta1, meta2);
                    @base = meta1.Base;
                }
                else
                {
                    @base = meta1.Base; // overwritten below
                }

                /* Take the greater base (unit sizes are decreasing in enum) */
                if (meta1.Base > meta2.Base)
                {
                    @base = meta1.Base;
                    num2 *= UnitsFactor(meta2.Base, meta1.Base);
                    if (num2 == 0)
                        throw UnitsOverflow(meta1, meta2);
                }
                else
                {
                    @base = meta2.Base;
                    num1 *= UnitsFactor(meta1.Base, meta2.Base);
                    if (num1 == 0)
                        throw UnitsOverflow(meta1, meta2);
                }
            }

            /* Compute the GCD of the resulting multipliers */
            ulong num = Gcd(num1, num2);

            int outNum = (int)num;
            if (outNum <= 0 || num != (ulong)outNum)
                throw UnitsOverflow(meta1, meta2);

            return new DatetimeMetaData(@base, outNum);
        }

        private static TypeError IncompatibleUnits(in DatetimeMetaData meta1, in DatetimeMetaData meta2)
            => new TypeError("Cannot get a common metadata divisor for Numpy datetime metadata " +
                             $"{meta1.ToString(false)} and {meta2.ToString(false)} because they have incompatible nonlinear base time units.");

        private static OverflowException UnitsOverflow(in DatetimeMetaData meta1, in DatetimeMetaData meta2)
            => new OverflowException("Integer overflow getting a common metadata divisor for NumPy datetime metadata " +
                                     $"{meta1.ToString(false)} and {meta2.ToString(false)}.");

        // ------------------------------------------------------------------------------------------------
        // Casting rules — datetime.c: can_cast_datetime64_units / can_cast_timedelta64_units + the metadata forms
        // ------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Port of <c>can_cast_datetime64_units</c>: <c>unsafe</c> allows everything; <c>same_kind</c> allows any
        ///     unit pair (generic only as the SOURCE); <c>safe</c> only towards more precise units
        ///     (<c>src_unit &lt;= dst_unit</c>, generic only as the source); <c>no</c> / <c>equiv</c> require equality.
        /// </summary>
        public static bool CanCastDatetime64Units(NPY_DATETIMEUNIT srcUnit, NPY_DATETIMEUNIT dstUnit, NPY_CASTING casting)
        {
            switch (casting)
            {
                /* Allow anything with unsafe casting */
                case NPY_CASTING.NPY_UNSAFE_CASTING:
                    return true;

                /*
                 * Can cast between all units with 'same_kind' casting.
                 */
                case NPY_CASTING.NPY_SAME_KIND_CASTING:
                    if (srcUnit == NPY_DATETIMEUNIT.NPY_FR_GENERIC || dstUnit == NPY_DATETIMEUNIT.NPY_FR_GENERIC)
                        return srcUnit == NPY_DATETIMEUNIT.NPY_FR_GENERIC;
                    return true;

                /*
                 * Casting is only allowed towards more precise units with 'safe'
                 * casting.
                 */
                case NPY_CASTING.NPY_SAFE_CASTING:
                    if (srcUnit == NPY_DATETIMEUNIT.NPY_FR_GENERIC || dstUnit == NPY_DATETIMEUNIT.NPY_FR_GENERIC)
                        return srcUnit == NPY_DATETIMEUNIT.NPY_FR_GENERIC;
                    return srcUnit <= dstUnit;

                /* Enforce equality with 'no' or 'equiv' casting */
                default:
                    return srcUnit == dstUnit;
            }
        }

        /// <summary>
        ///     Port of <c>can_cast_timedelta64_units</c>: as the datetime rule, plus the hard barrier between the
        ///     "date units" (years, months) and the "time units" (weeks and finer) for every rule but <c>unsafe</c>.
        /// </summary>
        public static bool CanCastTimedelta64Units(NPY_DATETIMEUNIT srcUnit, NPY_DATETIMEUNIT dstUnit, NPY_CASTING casting)
        {
            switch (casting)
            {
                /* Allow anything with unsafe casting */
                case NPY_CASTING.NPY_UNSAFE_CASTING:
                    return true;

                /*
                 * Only enforce the 'date units' vs 'time units' barrier with
                 * 'same_kind' casting.
                 */
                case NPY_CASTING.NPY_SAME_KIND_CASTING:
                    if (srcUnit == NPY_DATETIMEUNIT.NPY_FR_GENERIC || dstUnit == NPY_DATETIMEUNIT.NPY_FR_GENERIC)
                        return srcUnit == NPY_DATETIMEUNIT.NPY_FR_GENERIC;
                    return (srcUnit <= NPY_DATETIMEUNIT.NPY_FR_M && dstUnit <= NPY_DATETIMEUNIT.NPY_FR_M) ||
                           (srcUnit > NPY_DATETIMEUNIT.NPY_FR_M && dstUnit > NPY_DATETIMEUNIT.NPY_FR_M);

                /*
                 * Enforce the 'date units' vs 'time units' barrier and that
                 * casting is only allowed towards more precise units with
                 * 'safe' casting.
                 */
                case NPY_CASTING.NPY_SAFE_CASTING:
                    if (srcUnit == NPY_DATETIMEUNIT.NPY_FR_GENERIC || dstUnit == NPY_DATETIMEUNIT.NPY_FR_GENERIC)
                        return srcUnit == NPY_DATETIMEUNIT.NPY_FR_GENERIC;
                    return srcUnit <= dstUnit &&
                           ((srcUnit <= NPY_DATETIMEUNIT.NPY_FR_M && dstUnit <= NPY_DATETIMEUNIT.NPY_FR_M) ||
                            (srcUnit > NPY_DATETIMEUNIT.NPY_FR_M && dstUnit > NPY_DATETIMEUNIT.NPY_FR_M));

                /* Enforce equality with 'no' or 'equiv' casting */
                default:
                    return srcUnit == dstUnit;
            }
        }

        /// <summary>Port of <c>can_cast_datetime64_metadata</c>: the unit rule, and under <c>safe</c> also exact divisibility.</summary>
        public static bool CanCastDatetime64Metadata(in DatetimeMetaData src, in DatetimeMetaData dst, NPY_CASTING casting)
        {
            switch (casting)
            {
                case NPY_CASTING.NPY_UNSAFE_CASTING:
                    return true;

                case NPY_CASTING.NPY_SAME_KIND_CASTING:
                    return CanCastDatetime64Units(src.Base, dst.Base, casting);

                case NPY_CASTING.NPY_SAFE_CASTING:
                    return CanCastDatetime64Units(src.Base, dst.Base, casting) && Divides(src, dst, false);

                default:
                    return src.Base == dst.Base && src.Num == dst.Num;
            }
        }

        /// <summary>Port of <c>can_cast_timedelta64_metadata</c>: the timedelta unit rule, and under <c>safe</c> also strict divisibility.</summary>
        public static bool CanCastTimedelta64Metadata(in DatetimeMetaData src, in DatetimeMetaData dst, NPY_CASTING casting)
        {
            switch (casting)
            {
                case NPY_CASTING.NPY_UNSAFE_CASTING:
                    return true;

                case NPY_CASTING.NPY_SAME_KIND_CASTING:
                    return CanCastTimedelta64Units(src.Base, dst.Base, casting);

                case NPY_CASTING.NPY_SAFE_CASTING:
                    return CanCastTimedelta64Units(src.Base, dst.Base, casting) && Divides(src, dst, true);

                default:
                    return src.Base == dst.Base && src.Num == dst.Num;
            }
        }
    }
}
