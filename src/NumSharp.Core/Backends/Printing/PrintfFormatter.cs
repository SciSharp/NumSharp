using System;
using System.Globalization;
using System.Numerics;
using System.Text;

namespace NumSharp.Backends.Printing
{
    /// <summary>
    ///     A focused port of CPython's <c>str.__mod__</c> (the <c>%</c> operator) as used by
    ///     <c>numpy.ndarray.tofile(sep, format)</c>: each array element is written as
    ///     <c>format % item</c> where <c>item</c> is the boxed scalar. NumPy's text writer applies the
    ///     format string to exactly one value per element, so this engine handles a format with the
    ///     surrounding literal text plus its conversion specifier(s), each fed the same value.
    ///
    ///     Supported conversions (matching Python semantics probed against 3.x / NumPy 2.4.2):
    ///     <c>d i u</c> (decimal; floats truncate toward zero), <c>x X o</c> (signed base-16/8),
    ///     <c>f F</c> (fixed, default precision 6), <c>e E</c> (scientific, 2-digit-min exponent),
    ///     <c>g G</c> (general), <c>s</c> (<see cref="ArrayFormatter.ScalarStr"/>), <c>c</c> (char),
    ///     <c>%%</c> (literal). Flags <c>- + space 0 #</c>, field width and <c>.precision</c> are honored.
    ///
    ///     Lenient by design where CPython/NumPy would RAISE: a conversion that does not match the value
    ///     type (e.g. <c>%x</c>/<c>%o</c> on a bool/float, <c>%c</c> on a float, <c>%d</c> on inf/nan) yields
    ///     a best-effort rendering (base of the truncated integer, the char, or "inf"/"nan") instead of a
    ///     TypeError/OverflowError, and a format whose conversion count != 1 is applied leniently rather than
    ///     rejected — a file writer must not abort mid-stream on a per-element formatting quirk. <c>%r</c>
    ///     maps to the scalar string rather than NumPy 2.x's typed scalar repr ("np.float64(1.5)").
    /// </summary>
    internal static class PrintfFormatter
    {
        private static readonly CultureInfo CI = CultureInfo.InvariantCulture;

        /// <summary>Apply a Python <c>%</c>-format string to a single boxed value.</summary>
        public static string Format(string format, object value, NPTypeCode tc)
        {
            var sb = new StringBuilder(format.Length + 16);
            int i = 0, n = format.Length;
            while (i < n)
            {
                char ch = format[i];
                if (ch != '%')
                {
                    sb.Append(ch);
                    i++;
                    continue;
                }

                // '%' begins a conversion spec.
                i++;
                if (i < n && format[i] == '%') // "%%" -> literal '%'
                {
                    sb.Append('%');
                    i++;
                    continue;
                }

                // flags
                bool left = false, plus = false, space = false, zero = false, alt = false;
                while (i < n)
                {
                    char f = format[i];
                    if (f == '-') left = true;
                    else if (f == '+') plus = true;
                    else if (f == ' ') space = true;
                    else if (f == '0') zero = true;
                    else if (f == '#') alt = true;
                    else break;
                    i++;
                }

                // width
                int width = 0;
                while (i < n && format[i] >= '0' && format[i] <= '9')
                {
                    width = width * 10 + (format[i] - '0');
                    i++;
                }

                // .precision
                int precision = -1;
                if (i < n && format[i] == '.')
                {
                    i++;
                    precision = 0;
                    while (i < n && format[i] >= '0' && format[i] <= '9')
                    {
                        precision = precision * 10 + (format[i] - '0');
                        i++;
                    }
                }

                // length modifiers are meaningless here (numpy passes a scalar) — skip them.
                while (i < n && (format[i] == 'l' || format[i] == 'h' || format[i] == 'L'))
                    i++;

                if (i >= n) // dangling '%...' with no conversion char — emit verbatim.
                {
                    sb.Append('%');
                    break;
                }

                char conv = format[i];
                i++;
                sb.Append(Apply(conv, left, plus, space, zero, alt, width, precision, value, tc));
            }

            return sb.ToString();
        }

        private static string Apply(char conv, bool left, bool plus, bool space, bool zero, bool alt,
            int width, int precision, object value, NPTypeCode tc)
        {
            switch (conv)
            {
                case 's':
                case 'r':
                {
                    // %s -> str(item); zero flag does not apply to strings.
                    string body = ArrayFormatter.ScalarStr(value, tc);
                    return Pad(body, "", width, left, false);
                }
                case 'd':
                case 'i':
                case 'u':
                    return FormatInteger(value, tc, 10, false, false, left, plus, space, zero, false, width);
                case 'x':
                    return FormatInteger(value, tc, 16, false, false, left, plus, space, zero, alt, width);
                case 'X':
                    return FormatInteger(value, tc, 16, true, false, left, plus, space, zero, alt, width);
                case 'o':
                    return FormatInteger(value, tc, 8, false, false, left, plus, space, zero, alt, width);
                case 'f':
                case 'F':
                    return FormatFixed(ToReal(value), conv == 'F', precision < 0 ? 6 : precision,
                        alt, left, plus, space, zero, width);
                case 'e':
                case 'E':
                    return FormatScientific(ToReal(value), conv == 'E', precision < 0 ? 6 : precision,
                        alt, left, plus, space, zero, width);
                case 'g':
                case 'G':
                    return FormatGeneral(ToReal(value), conv == 'G', precision, alt,
                        left, plus, space, zero, width);
                case 'c':
                {
                    string body = FormatChar(value, tc);
                    return Pad(body, "", width, left, false);
                }
                default:
                    // Unknown conversion: emit the '%' + conv verbatim (lenient).
                    return "%" + conv;
            }
        }

        // ---- numeric helpers ---------------------------------------------------------

        private static double ToReal(object v) => v switch
        {
            double d => d,
            float f => f,
            Half h => (double)h,
            decimal m => (double)m,
            Complex c => c.Real, // numpy discards the imaginary part for real conversions
            bool b => b ? 1.0 : 0.0,
            char ch => ch,
            _ => Convert.ToDouble(v, CI)
        };

        // Exact integer value as BigInteger (integers) or truncated-toward-zero (floats).
        private static bool TryToBigInteger(object v, NPTypeCode tc, out BigInteger big)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean: big = (bool)v ? 1 : 0; return true;
                case NPTypeCode.SByte: big = (sbyte)v; return true;
                case NPTypeCode.Byte: big = (byte)v; return true;
                case NPTypeCode.Int16: big = (short)v; return true;
                case NPTypeCode.UInt16: big = (ushort)v; return true;
                case NPTypeCode.Int32: big = (int)v; return true;
                case NPTypeCode.UInt32: big = (uint)v; return true;
                case NPTypeCode.Int64: big = (long)v; return true;
                case NPTypeCode.UInt64: big = (ulong)v; return true;
                case NPTypeCode.Char: big = (char)v; return true;
                default:
                {
                    double d = ToReal(v);
                    if (!double.IsFinite(d)) { big = BigInteger.Zero; return false; }
                    big = new BigInteger(Math.Truncate(d));
                    return true;
                }
            }
        }

        private static string FormatInteger(object value, NPTypeCode tc, int radix, bool upper, bool _,
            bool left, bool plus, bool space, bool zero, bool alt, int width)
        {
            // Non-finite floats have no integer form; Python raises, but a file writer should not abort —
            // fall back to the nan/inf text (still readable, never crashes mid-file).
            if (!TryToBigInteger(value, tc, out BigInteger big))
            {
                double d = ToReal(value);
                string t = double.IsNaN(d) ? "nan" : (d < 0 ? "-inf" : "inf");
                return Pad(t.TrimStart('-'), d < 0 ? "-" : Sign(false, plus, space), width, left, zero);
            }

            bool neg = big.Sign < 0;
            BigInteger mag = BigInteger.Abs(big);

            string digits = radix switch
            {
                16 => ToBaseString(mag, 16, upper),
                8 => ToBaseString(mag, 8, upper),
                _ => mag.ToString(CI)
            };

            string prefix = "";
            if (alt)
            {
                // Python's '#' flag prefixes even a zero value ("%#x" % 0 == "0x0", "%#o" % 0 == "0o0"),
                // unlike C printf which omits the prefix for zero.
                if (radix == 16) prefix = upper ? "0X" : "0x";
                else if (radix == 8) prefix = "0o";
            }

            string sign = neg ? "-" : Sign(false, plus, space);
            return Pad(prefix + digits, sign, width, left, zero);
        }

        private static string ToBaseString(BigInteger mag, int radix, bool upper)
        {
            if (mag.IsZero) return "0";
            const string lo = "0123456789abcdef";
            const string hi = "0123456789ABCDEF";
            string digitset = upper ? hi : lo;
            var sb = new StringBuilder();
            while (mag > 0)
            {
                sb.Insert(0, digitset[(int)(mag % radix)]);
                mag /= radix;
            }
            return sb.ToString();
        }

        private static string FormatFixed(double d, bool upper, int precision, bool alt,
            bool left, bool plus, bool space, bool zero, int width)
        {
            if (!double.IsFinite(d))
                return NonFinite(d, upper, plus, space, width, left, zero);

            string sign = (d < 0 || IsNegZero(d)) ? "-" : Sign(false, plus, space);
            string digits = Math.Abs(d).ToString("F" + precision, CI);
            // '#' forces a trailing decimal point even when no fractional digits are emitted ("%#.0f"%3=="3.").
            if (alt && digits.IndexOf('.') < 0) digits += ".";
            return Pad(digits, sign, width, left, zero);
        }

        private static string FormatScientific(double d, bool upper, int precision, bool alt,
            bool left, bool plus, bool space, bool zero, int width)
        {
            if (!double.IsFinite(d))
                return NonFinite(d, upper, plus, space, width, left, zero);

            string sign = (d < 0 || IsNegZero(d)) ? "-" : Sign(false, plus, space);
            string body = Math.Abs(d).ToString("E" + precision, CI); // e.g. "1.500000E+000"
            int ei = body.IndexOf('E');
            string mant = body.Substring(0, ei);
            // '#' keeps the mantissa's decimal point even at precision 0 ("%#.0e"%3=="3.e+00").
            if (alt && mant.IndexOf('.') < 0) mant += ".";
            string digits = mant + ExponentSuffix(body.Substring(ei + 1), upper);
            return Pad(digits, sign, width, left, zero);
        }

        private static string FormatGeneral(double d, bool upper, int precision, bool alt,
            bool left, bool plus, bool space, bool zero, int width)
        {
            if (!double.IsFinite(d))
                return NonFinite(d, upper, plus, space, width, left, zero);

            int P = precision < 0 ? 6 : (precision == 0 ? 1 : precision);
            string sign = (d < 0 || IsNegZero(d)) ? "-" : Sign(false, plus, space);
            double abs = Math.Abs(d);

            // decimal exponent X of the value written in %e style with P-1 fraction digits.
            string es = abs.ToString("E" + (P - 1), CI);
            int X = int.Parse(es.Substring(es.IndexOf('E') + 1), CI);

            string digits;
            if (X >= -4 && X < P)
            {
                digits = abs.ToString("F" + Math.Max(0, P - 1 - X), CI);
                // '#' keeps trailing zeros AND forces a decimal point ("%#g"%100000=="100000.");
                // the default form strips both.
                if (!alt) digits = StripTrailingZeros(digits);
                else if (digits.IndexOf('.') < 0) digits += ".";
            }
            else
            {
                string body = abs.ToString("E" + (P - 1), CI);
                int ei = body.IndexOf('E');
                string mant = body.Substring(0, ei);
                if (!alt) mant = StripTrailingZeros(mant);
                else if (mant.IndexOf('.') < 0) mant += ".";
                digits = mant + ExponentSuffix(body.Substring(ei + 1), upper);
            }

            return Pad(digits, sign, width, left, zero);
        }

        private static string FormatChar(object value, NPTypeCode tc)
        {
            if (tc == NPTypeCode.Char) return ((char)value).ToString();
            if (TryToBigInteger(value, tc, out BigInteger big) &&
                big >= 0 && big <= 0x10FFFF)
            {
                int cp = (int)big;
                return char.ConvertFromUtf32(cp);
            }
            return ArrayFormatter.ScalarStr(value, tc);
        }

        // ---- shared formatting primitives -------------------------------------------

        // .NET "E{p}" emits a 3-digit exponent with an uppercase 'E'; Python uses a 2-digit-minimum
        // exponent and matches the conversion's case. Rebuild the "±NN" tail.
        private static string ExponentSuffix(string netExpDigits, bool upper)
        {
            int exp = int.Parse(netExpDigits, CI);
            string es = Math.Abs(exp).ToString(CI);
            if (es.Length < 2) es = es.PadLeft(2, '0');
            return (upper ? "E" : "e") + (exp < 0 ? "-" : "+") + es;
        }

        private static string StripTrailingZeros(string s)
        {
            if (s.IndexOf('.') < 0) return s;
            s = s.TrimEnd('0');
            if (s.EndsWith(".")) s = s.Substring(0, s.Length - 1);
            return s;
        }

        private static string NonFinite(double d, bool upper, bool plus, bool space,
            int width, bool left, bool zero)
        {
            string sign = double.IsNaN(d) ? Sign(false, plus, space)
                                          : (d < 0 ? "-" : Sign(false, plus, space));
            string t = double.IsNaN(d) ? "nan" : "inf";
            if (upper) t = t.ToUpperInvariant();
            return Pad(t, sign, width, left, zero);
        }

        private static string Sign(bool neg, bool plus, bool space)
        {
            if (neg) return "-";
            if (plus) return "+";
            if (space) return " ";
            return "";
        }

        private static bool IsNegZero(double d) => d == 0.0 && double.IsNegative(d);

        // Combine sign + body into a width-padded field. Numeric zero-padding inserts the zeros
        // between the sign and the body (Python's behavior, including for inf/nan). Left-justify
        // ('-') always pads with spaces on the right and wins over the zero flag.
        private static string Pad(string body, string sign, int width, bool left, bool zero)
        {
            string full = sign + body;
            if (full.Length >= width)
                return full;

            int deficit = width - full.Length;
            if (left)
                return full + new string(' ', deficit);
            if (zero)
                return sign + new string('0', deficit) + body;
            return new string(' ', deficit) + full;
        }
    }
}
