using System;
using System.Globalization;
using System.Text;

namespace NumSharp.Backends.Printing
{
    /// <summary>
    ///     The float-precision modes NumPy's Dragon4 trim characters map to.
    /// </summary>
    internal enum TrimMode
    {
        /// <summary>'k' — keep trailing zeros, keep decimal point (no trimming).</summary>
        Keep,

        /// <summary>'.' — trim all trailing zeros, leave decimal point.</summary>
        Zeros,

        /// <summary>'0' — trim trailing zeros but leave one zero after the point.</summary>
        LeaveOneZero,

        /// <summary>'-' — trim trailing zeros and any trailing decimal point.</summary>
        DptZeros
    }

    /// <summary>
    ///     Distinguishes the three real floating-point dtypes NumSharp formats with shortest
    ///     round-trip digit generation. Decimal is widened to <see cref="Double"/>.
    /// </summary>
    internal enum FloatKind
    {
        Half,
        Single,
        Double
    }

    /// <summary>
    ///     C# port of NumPy's Dragon4 string formatting (<c>numpy/_core/src/multiarray/dragon4.c</c>),
    ///     producing bit-identical output to <c>np.format_float_positional</c> /
    ///     <c>np.format_float_scientific</c> for the <see cref="float"/>, <see cref="double"/> and
    ///     <see cref="System.Half"/> dtypes.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Feasibility rests on a verified fact: .NET's shortest-round-trip formatting
    ///     (<c>ToString("R")</c> / default <c>ToString()</c> since .NET Core 3.0) emits the same
    ///     shortest decimal digit sequence as NumPy's Dragon4 <c>unique=True</c> mode.
    ///     </para>
    ///     <para>
    ///     The one trap (proven divergent ~50% of the time on adversarial ties) is rounding the
    ///     <em>shortest decimal string</em> when precision caps it: Dragon4 rounds the exact binary
    ///     value, not the shortest string. We avoid that entirely by routing every rounding through
    ///     .NET's <c>ToString("F"+p)</c> / <c>ToString("E"+p)</c>, which round the true value with
    ///     IEEE round-half-to-even. Shortest digits are used verbatim only when they already fit
    ///     within precision.
    ///     </para>
    /// </remarks>
    internal static class Dragon4
    {
        private static readonly CultureInfo CI = CultureInfo.InvariantCulture;

        internal static TrimMode ParseTrim(char trim) => trim switch
        {
            'k' => TrimMode.Keep,
            '.' => TrimMode.Zeros,
            '0' => TrimMode.LeaveOneZero,
            '-' => TrimMode.DptZeros,
            _ => throw new ArgumentException($"Invalid trim mode '{trim}'")
        };

        #region native ToString helpers

        // Shortest round-trip digits MUST be produced by the native type (widening Half->float or
        // float->double changes which decimal string is "shortest").
        private static string ShortestR(object v, FloatKind kind) => kind switch
        {
            FloatKind.Half => ((Half)v).ToString("R", CI),
            FloatKind.Single => ((float)v).ToString("R", CI),
            _ => ((double)v).ToString("R", CI)
        };

        // Rounding to a fixed number of digits is exact under widening (Half ⊂ float ⊂ double),
        // so we always round through double which has the most reliable F/E formatter.
        private static double ToDouble(object v, FloatKind kind) => kind switch
        {
            FloatKind.Half => (double)(Half)v,
            FloatKind.Single => (double)(float)v,
            _ => (double)v
        };

        #endregion

        /// <summary>
        ///     Significant digits (no point, no sign, leading/trailing zeros stripped) plus the
        ///     base-10 exponent of the first significant digit. value = 0 → ("0", 0).
        /// </summary>
        internal static (string digits, int decExp, bool negative) ExtractShortest(object v, FloatKind kind)
        {
            string s = ShortestR(v, kind);
            bool negative = s.Length > 0 && s[0] == '-';
            if (negative)
                s = s.Substring(1);

            return ParseDecimal(s, negative);
        }

        /// <summary>
        ///     Parse a plain or scientific decimal string ("3.14", "1E-16", "1000", "0.001",
        ///     "10.00000000") into (significant-digits, exponent-of-first-digit, sign).
        /// </summary>
        private static (string digits, int decExp, bool negative) ParseDecimal(string s, bool negative)
        {
            int expPart = 0;
            int eIdx = s.IndexOfAny(new[] { 'e', 'E' });
            if (eIdx >= 0)
            {
                expPart = int.Parse(s.Substring(eIdx + 1), CI);
                s = s.Substring(0, eIdx);
            }

            string intDigits, fracDigits;
            int dot = s.IndexOf('.');
            if (dot >= 0)
            {
                intDigits = s.Substring(0, dot);
                fracDigits = s.Substring(dot + 1);
            }
            else
            {
                intDigits = s;
                fracDigits = "";
            }

            string raw = intDigits + fracDigits;
            // exponent of raw[0] before stripping leading zeros
            int e0 = (intDigits.Length - 1) + expPart;

            // strip leading zeros (each one lowers the exponent of the surviving lead digit)
            int lead = 0;
            while (lead < raw.Length - 1 && raw[lead] == '0')
            {
                lead++;
                e0--;
            }
            raw = raw.Substring(lead);

            // strip trailing zeros (non-significant; magnitude already captured by e0)
            int end = raw.Length;
            while (end > 1 && raw[end - 1] == '0')
                end--;
            raw = raw.Substring(0, end);

            if (raw == "0" || raw.Length == 0)
                return ("0", 0, negative);

            return (raw, e0, negative);
        }

        /// <summary>
        ///     Build the integer and fractional digit strings (no point/sign) for the positional
        ///     representation of (digits, decExp).
        /// </summary>
        internal static (string intStr, string fracStr) SplitPositional(string digits, int decExp)
        {
            if (decExp >= 0)
            {
                int intLen = decExp + 1;
                string intStr, fracStr;
                if (digits.Length >= intLen)
                {
                    intStr = digits.Substring(0, intLen);
                    fracStr = digits.Substring(intLen);
                }
                else
                {
                    intStr = digits + new string('0', intLen - digits.Length);
                    fracStr = "";
                }

                return (intStr, fracStr);
            }
            else
            {
                string fracStr = new string('0', -decExp - 1) + digits;
                return ("0", fracStr);
            }
        }

        /// <summary>
        ///     The number of fractional/mantissa digits NumPy's Dragon4 prints for the given
        ///     (precision, minDigits, unique) combination and a value whose shortest form already has
        ///     <paramref name="shortestFrac"/> digits.
        /// </summary>
        private static int TargetDigits(int shortestFrac, int precision, int minDigits, bool unique)
        {
            int natural = unique ? shortestFrac : precision;
            if (precision >= 0)
                natural = Math.Min(natural, precision);
            return Math.Max(natural, minDigits);
        }

        /// <summary>
        ///     Produce (intStr, fracStr) for a value with exactly <c>N</c> fractional digits, where the
        ///     extra/rounded digits come from the true binary value (never the shortest string). The
        ///     returned fracStr keeps trailing zeros (the caller's trim mode removes them if needed).
        /// </summary>
        private static (string intStr, string fracStr, bool negative) PositionalDigits(
            object v, FloatKind kind, int precision, int minDigits, bool unique)
        {
            var (digits, decExp, negative) = ExtractShortest(v, kind);
            int shortestFrac = digits == "0" ? 0 : Math.Max(0, (digits.Length - 1) - decExp);
            int n = TargetDigits(shortestFrac, precision, minDigits, unique);

            if (n == shortestFrac && digits != "0")
            {
                var (i0, f0) = SplitPositional(digits, decExp);
                return (i0, f0, negative);
            }

            // Round/extend the true value to exactly n fractional digits (IEEE half-to-even).
            double dv = ToDouble(v, kind);
            string fixedStr = Math.Abs(dv).ToString("F" + n.ToString(CI), CI);
            int dot = fixedStr.IndexOf('.');
            string intStr = dot >= 0 ? fixedStr.Substring(0, dot) : fixedStr;
            string fracStr = dot >= 0 ? fixedStr.Substring(dot + 1) : "";
            return (intStr, fracStr, negative);
        }

        /// <summary>
        ///     Apply a trim mode to a fractional digit string, returning (fracStr, hasPoint).
        /// </summary>
        private static (string frac, bool hasPoint) ApplyTrim(string fracStr, TrimMode trim, int minDigits)
        {
            switch (trim)
            {
                case TrimMode.Keep:
                    // pad with zeros up to minDigits (used by fixed mode)
                    if (fracStr.Length < minDigits)
                        fracStr += new string('0', minDigits - fracStr.Length);
                    return (fracStr, true);

                case TrimMode.Zeros:
                    fracStr = fracStr.TrimEnd('0');
                    return (fracStr, true);

                case TrimMode.LeaveOneZero:
                    fracStr = fracStr.TrimEnd('0');
                    if (fracStr.Length == 0)
                        fracStr = "0";
                    return (fracStr, true);

                case TrimMode.DptZeros:
                    fracStr = fracStr.TrimEnd('0');
                    return (fracStr, fracStr.Length > 0);

                default:
                    return (fracStr, true);
            }
        }

        /// <summary>
        ///     Port of <c>dragon4_positional</c>. <paramref name="precision"/> may be negative for
        ///     unique mode. <paramref name="padLeft"/>/<paramref name="padRight"/> are space-pad
        ///     widths for the integer and fractional sides (-1 = no padding).
        /// </summary>
        internal static string FormatPositional(object v, FloatKind kind, int precision, int minDigits,
            bool unique, TrimMode trim, bool signPlus, int padLeft, int padRight)
        {
            var (intStr, fracStr, negative) = PositionalDigits(v, kind, precision, minDigits, unique);

            var (frac, hasPoint) = ApplyTrim(fracStr, trim, minDigits);

            // sign char: '-' for negatives; '+' for positives if signPlus. (' ' sign is emulated by
            // padLeft, matching NumPy's FloatingFormat.)
            string sign = negative ? "-" : (signPlus ? "+" : "");

            var sb = new StringBuilder();
            sb.Append(sign).Append(intStr);
            if (hasPoint)
                sb.Append('.');
            sb.Append(frac);

            string core = sb.ToString();

            // pad_right: NumPy counts the fractional field width as padRight digits after the point,
            // padding the fractional side with spaces until it reaches that width.
            if (padRight >= 0)
            {
                int deficit = padRight - frac.Length;
                if (deficit > 0)
                    core += new string(' ', deficit);
            }

            // pad_left: left-pad with spaces until the integer side (sign + intStr) reaches padLeft.
            if (padLeft >= 0)
            {
                int intChars = sign.Length + intStr.Length;
                int deficit = padLeft - intChars;
                if (deficit > 0)
                    core = new string(' ', deficit) + core;
            }

            return core;
        }

        /// <summary>
        ///     Port of <c>dragon4_scientific</c>. Produces a mantissa with <paramref name="precision"/>
        ///     fractional digits (capped/rounded against the true value), a lowercase 'e', a sign, and
        ///     an exponent zero-padded to at least <paramref name="expDigits"/> (min 2) digits.
        /// </summary>
        internal static string FormatScientific(object v, FloatKind kind, int precision, int minDigits,
            bool unique, TrimMode trim, bool signPlus, int padLeft, int expDigits)
        {
            var (digits, decExp, negative) = ExtractShortest(v, kind);
            int shortestFrac = digits == "0" ? 0 : digits.Length - 1;
            int n = TargetDigits(shortestFrac, precision, minDigits, unique);

            string lead, mantFracStr;
            int exp;
            if (n == shortestFrac && digits != "0")
            {
                lead = digits.Substring(0, 1);
                mantFracStr = digits.Length > 1 ? digits.Substring(1) : "";
                exp = decExp;
            }
            else
            {
                // Round/extend the true value to exactly n mantissa-fraction digits (keeping zeros).
                double dv = ToDouble(v, kind);
                string sci = Math.Abs(dv).ToString("E" + n.ToString(CI), CI);
                int epos = sci.IndexOf('E');
                string mant = sci.Substring(0, epos);
                exp = int.Parse(sci.Substring(epos + 1), CI);
                int dot = mant.IndexOf('.');
                lead = dot >= 0 ? mant.Substring(0, dot) : mant;
                mantFracStr = dot >= 0 ? mant.Substring(dot + 1) : "";
            }

            var (frac, hasPoint) = ApplyTrim(mantFracStr, trim, minDigits);

            string sign = negative ? "-" : (signPlus ? "+" : "");

            var sb = new StringBuilder();
            sb.Append(sign);

            // pad_left applies to the mantissa integer part (always 1 digit + optional sign).
            if (padLeft >= 0)
            {
                int deficit = padLeft - (sign.Length + lead.Length);
                if (deficit > 0)
                    sb.Insert(0, new string(' ', deficit));
            }

            sb.Append(lead);
            if (hasPoint)
                sb.Append('.');
            sb.Append(frac);

            // exponent
            sb.Append('e');
            sb.Append(exp >= 0 ? '+' : '-');
            int absExp = Math.Abs(exp);
            int width = Math.Max(expDigits < 0 ? 2 : expDigits, 2);
            sb.Append(absExp.ToString(CI).PadLeft(width, '0'));

            return sb.ToString();
        }

        /// <summary>
        ///     Number of fractional digits in the positional representation of the value's shortest
        ///     round-trip form. Used by <c>FloatingFormat.fillFormat</c> column sizing.
        /// </summary>
        internal static int ShortestFractionLength(object v, FloatKind kind)
        {
            var (digits, decExp, _) = ExtractShortest(v, kind);
            if (digits == "0")
                return 0;
            return Math.Max(0, (digits.Length - 1) - decExp);
        }
    }
}
