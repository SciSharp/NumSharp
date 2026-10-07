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
                // A fixed-width string element is the Python object getitem returns (str / bytes): ndarray.tofile's
                // `format % item`. Its conversions are str.__mod__'s for text, not the numeric ones.
                sb.Append(tc == NPTypeCode.String
                    ? ApplyText(conv, left, width, precision, value, numpyScalar: false)
                    : Apply(conv, left, plus, space, zero, alt, width, precision, value, tc));
            }

            return sb.ToString();
        }

        /// <summary>
        ///     Sequential-argument variant of <see cref="Format"/> — the port of CPython's
        ///     <c>str.__mod__</c> applied to a TUPLE, as used by <c>numpy.savetxt</c>'s per-row
        ///     <c>format % tuple(row)</c>. Every conversion spec consumes the NEXT value from
        ///     <paramref name="args"/> (whereas <see cref="Format"/> feeds its single value to every
        ///     spec); <c>%%</c> is a literal and consumes nothing.
        /// </summary>
        /// <exception cref="PrintfArgumentException">
        ///     The number of conversion specs does not match <paramref name="args"/>.Count, or a
        ///     <c>%c</c> conversion targets a non-integral value — the two conditions under which
        ///     CPython's <c>%</c> operator raises a <see cref="TypeError"/> mid-row. NumPy's real
        ///     branch catches this and reports the dtype/format mismatch; its complex branch lets it
        ///     surface (a <see cref="PrintfArgumentException"/> IS a <see cref="TypeError"/>).
        /// </exception>
        public static string FormatRow(string format, System.Collections.Generic.IReadOnlyList<object> args,
            System.Collections.Generic.IReadOnlyList<NPTypeCode> tcs)
        {
            var sb = new StringBuilder(format.Length + 16);
            FormatRowInto(sb, format, args, tcs);
            return sb.ToString();
        }

        /// <summary>
        ///     <see cref="FormatRow"/> appending directly into <paramref name="sb"/> — the allocation-free
        ///     form used on <c>np.savetxt</c>'s hot per-row path (no intermediate row string, the caller's
        ///     batch buffer absorbs every row).
        /// </summary>
        public static void FormatRowInto(StringBuilder sb, string format,
            System.Collections.Generic.IReadOnlyList<object> args,
            System.Collections.Generic.IReadOnlyList<NPTypeCode> tcs)
        {
            int i = 0, n = format.Length, argIdx = 0, argCount = args.Count;
            while (i < n)
            {
                char ch = format[i];
                if (ch != '%')
                {
                    sb.Append(ch);
                    i++;
                    continue;
                }

                i++;
                if (i < n && format[i] == '%') // "%%" -> literal '%', consumes no argument
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

                // length modifiers are meaningless here (a scalar per spec) — skip them.
                while (i < n && (format[i] == 'l' || format[i] == 'h' || format[i] == 'L'))
                    i++;

                if (i >= n) // dangling '%...' with no conversion char — emit verbatim.
                {
                    sb.Append('%');
                    break;
                }

                char conv = format[i];
                i++;

                if (argIdx >= argCount)
                    throw new PrintfArgumentException("not enough arguments for format string");

                object value = args[argIdx];
                NPTypeCode tc = tcs[argIdx];
                argIdx++;

                // A fixed-width string element: savetxt formats tuple(row), whose items are NumPy scalars (np.str_ /
                // np.bytes_), so %r prints their NumPy repr and a numeric conversion's TypeError names numpy.str_.
                if (tc == NPTypeCode.String)
                {
                    sb.Append(ApplyText(conv, left, width, precision, value, numpyScalar: true));
                    continue;
                }

                // Integer/char conversions validate their operand exactly as CPython's `%` operator does on
                // a NumPy scalar — this is where savetxt on the wrong dtype raises. A TypeError is caught by
                // savetxt's real branch and reported as the dtype/format mismatch (PrintfArgumentException IS
                // a TypeError); a ValueError/OverflowError propagates uncaught.
                switch (conv)
                {
                    case 'c':
                        // %c takes a true integer (int/uint/char) whose value is a valid code point; bool and
                        // every non-integral scalar raise TypeError ("Mismatch..."), and an out-of-range int
                        // raises OverflowError. Probed: "%c"%np.bool_(True) and "%c"%np.float64(1) both raise
                        // TypeError; "%c"%np.int8(-1) raises OverflowError.
                        if (IsTrueIntegerTc(tc))
                        {
                            TryToBigInteger(value, tc, out BigInteger cp);
                            if (cp < 0 || cp > 0x10FFFF)
                                throw new OverflowException("%c arg not in range(0x110000)");
                            break;
                        }
                        throw new PrintfArgumentException("%c requires int or char");
                    case 'x':
                    case 'X':
                    case 'o':
                        // Hex/octal accept only true integers; bool AND non-integral floats raise TypeError.
                        if (IsTrueIntegerTc(tc))
                            break;
                        throw new PrintfArgumentException(
                            $"%{conv} format: an integer is required, not {(tc == NPTypeCode.Boolean ? "numpy.bool" : "float")}");
                    case 'd':
                    case 'i':
                    case 'u':
                        // Integers and bool pass; a FINITE float truncates, but nan/inf raise (uncaught).
                        if (!IsIntegralTc(tc))
                        {
                            double dv = ToReal(value);
                            if (double.IsNaN(dv))
                                throw new ValueError("cannot convert float NaN to integer");
                            if (double.IsInfinity(dv))
                                throw new OverflowException("cannot convert float infinity to integer");
                        }
                        break;
                }

                sb.Append(Apply(conv, left, plus, space, zero, alt, width, precision, value, tc));
            }

            if (argIdx != argCount)
                throw new PrintfArgumentException("not all arguments converted during string formatting");
        }

        private static bool IsIntegralTc(NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.SByte:
                case NPTypeCode.Byte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Char:
                    return true;
                default:
                    return false;
            }
        }

        // A "true" integer for the %x/%X/%o/%c operand rules — every integral dtype EXCEPT Boolean, which
        // CPython's `%` operator rejects for those conversions on a NumPy scalar (a bool is not an int there).
        private static bool IsTrueIntegerTc(NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.SByte:
                case NPTypeCode.Byte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Char:
                    return true;
                default:
                    return false;
            }
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
                    return FormatInteger(value, tc, 10, false, false, left, plus, space, zero, false, width, precision);
                case 'x':
                    return FormatInteger(value, tc, 16, false, false, left, plus, space, zero, alt, width, precision);
                case 'X':
                    return FormatInteger(value, tc, 16, true, false, left, plus, space, zero, alt, width, precision);
                case 'o':
                    return FormatInteger(value, tc, 8, false, false, left, plus, space, zero, alt, width, precision);
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

        // ---- text values (fixed-width string elements) --------------------------------

        /// <summary>
        ///     CPython's <c>str.__mod__</c> conversion of ONE text value — a <c>U</c> element (a C# <see cref="string"/>)
        ///     or an <c>S</c> element (a <c>byte[]</c>): <c>%s</c> is <c>str(value)</c> (the text; a bytes value prints as
        ///     its <c>b'...'</c> repr), <c>%r</c> is <c>repr(value)</c>, <c>%a</c> is <c>ascii(value)</c>; a precision
        ///     truncates and a width pads with spaces, both in code points (the <c>-</c> flag left-aligns, the numeric
        ///     flags do nothing); <c>%c</c> takes a one-character <c>str</c>; every numeric conversion raises CPython's
        ///     <c>TypeError</c>.
        /// </summary>
        /// <param name="conv">The conversion character.</param>
        /// <param name="left">The <c>-</c> flag.</param>
        /// <param name="width">The minimum width in code points (0 for none).</param>
        /// <param name="precision">The precision (-1 for none): the maximum number of code points kept.</param>
        /// <param name="value">The element: a <see cref="string"/> or a <c>byte[]</c>.</param>
        /// <param name="numpyScalar">True when the value stands for a NumPy scalar (<c>np.str_</c> / <c>np.bytes_</c>:
        ///     <c>np.savetxt</c> formats <c>tuple(row)</c>), false for the Python object <c>getitem</c> returns
        ///     (<c>str</c> / <c>bytes</c>: <c>ndarray.tofile</c>). The two differ in <c>repr</c>
        ///     (<c>np.str_('ab')</c> vs <c>'ab'</c>) and in the type name a <c>TypeError</c> reports.</param>
        /// <returns>The formatted text.</returns>
        /// <exception cref="PrintfArgumentException">CPython's <c>TypeError</c> for a numeric conversion of text
        ///     (<c>%d format: a real number is required, not str</c>, <c>%x format: an integer is required, not
        ///     bytes</c>, <c>must be real number, not numpy.str_</c>) or <c>%c</c> of anything but one character
        ///     (<c>%c requires int or char</c>).</exception>
        private static string ApplyText(char conv, bool left, int width, int precision, object value, bool numpyScalar)
        {
            bool isBytes = value is byte[];
            string typeName = isBytes ? (numpyScalar ? "numpy.bytes_" : "bytes") : (numpyScalar ? "numpy.str_" : "str");
            string body;
            switch (conv)
            {
                case 's':
                    // str(bytes) is its repr; str(np.bytes_) too. str(np.str_) / str(str) is the text itself.
                    body = isBytes ? PyRepr.Bytes((byte[])value) : (string)value;
                    break;
                case 'r':
                    body = TextRepr(value, numpyScalar);
                    break;
                case 'a':
                    body = PyRepr.Ascii(TextRepr(value, numpyScalar));
                    break;
                case 'c':
                    // A str of exactly one code point is its own %c; CPython refuses every other value with one text.
                    if (!isBytes && PyRepr.CodePointLength((string)value) == 1)
                        return PadCodePoints((string)value, width, left);
                    throw new PrintfArgumentException("%c requires int or char");
                case 'd':
                case 'i':
                case 'u':
                    throw new PrintfArgumentException($"%{conv} format: a real number is required, not {typeName}");
                case 'x':
                case 'X':
                case 'o':
                    throw new PrintfArgumentException($"%{conv} format: an integer is required, not {typeName}");
                case 'e':
                case 'E':
                case 'f':
                case 'F':
                case 'g':
                case 'G':
                    throw new PrintfArgumentException($"must be real number, not {typeName}");
                default:
                    // Unknown conversion: emit the '%' + conv verbatim (lenient, as the numeric path).
                    return "%" + conv;
            }

            if (precision >= 0)
                body = TruncateCodePoints(body, precision);
            return PadCodePoints(body, width, left);
        }

        /// <summary>
        ///     Python's <c>repr()</c> of a text element: the plain <c>'ab'</c> / <c>b'ab'</c> of the Python object
        ///     (<c>tofile</c>'s <c>getitem</c>), or <c>np.str_('ab')</c> / <c>np.bytes_(b'ab')</c> of a NumPy scalar.
        /// </summary>
        /// <param name="value">A <see cref="string"/> or a <c>byte[]</c>.</param>
        /// <param name="numpyScalar">Whether the value stands for a NumPy scalar.</param>
        /// <returns>The representation.</returns>
        private static string TextRepr(object value, bool numpyScalar)
        {
            string repr = value is byte[] bytes ? PyRepr.Bytes(bytes) : PyRepr.Str((string)value);
            if (!numpyScalar)
                return repr;
            return (value is byte[] ? "np.bytes_(" : "np.str_(") + repr + ")";
        }

        /// <summary>The first <paramref name="count"/> code points of <paramref name="s"/> (Python's <c>s[:count]</c>;
        ///     a surrogate pair is one code point and is never split).</summary>
        /// <param name="s">The text.</param>
        /// <param name="count">The number of code points to keep.</param>
        /// <returns>The truncated text (the text itself when it is not longer).</returns>
        private static string TruncateCodePoints(string s, int count)
        {
            int units = 0;
            for (int kept = 0; kept < count && units < s.Length; kept++)
                units += char.IsHighSurrogate(s[units]) && units + 1 < s.Length && char.IsLowSurrogate(s[units + 1]) ? 2 : 1;
            return units >= s.Length ? s : s.Substring(0, units);
        }

        /// <summary>Pads <paramref name="body"/> with spaces to <paramref name="width"/> CODE POINTS (Python's width
        ///     counts characters, not UTF-16 units), on the right when <paramref name="left"/>.</summary>
        /// <param name="body">The text.</param>
        /// <param name="width">The minimum width.</param>
        /// <param name="left">Left-align (pad on the right).</param>
        /// <returns>The padded text.</returns>
        private static string PadCodePoints(string body, int width, bool left)
        {
            int deficit = width - PyRepr.CodePointLength(body);
            if (deficit <= 0)
                return body;
            return left ? body + new string(' ', deficit) : new string(' ', deficit) + body;
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
            bool left, bool plus, bool space, bool zero, bool alt, int width, int precision)
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

            // Python's `.precision` on an integer conversion (d/i/u/x/X/o) is the MINIMUM number of digits:
            // the magnitude is left-zero-filled to `precision` chars ("%.5d" % 42 == "00042"). The digit run
            // for 0 is "0", so ".0" never shrinks it ("%.0d" % 0 == "0"). Unlike C, Python still honors the
            // '0' flag together with a precision ("%08.3d" % 7 == "00000007").
            if (precision >= 0 && digits.Length < precision)
                digits = new string('0', precision - digits.Length) + digits;

            string prefix = "";
            if (alt)
            {
                // Python's '#' flag prefixes even a zero value ("%#x" % 0 == "0x0", "%#o" % 0 == "0o0"),
                // unlike C printf which omits the prefix for zero.
                if (radix == 16) prefix = upper ? "0X" : "0x";
                else if (radix == 8) prefix = "0o";
            }

            string sign = neg ? "-" : Sign(false, plus, space);
            return PadInteger(sign, prefix, digits, width, left, zero);
        }

        // Width-pad an integer field. Zero-fill goes BETWEEN the alt prefix (0x/0o) and the digits, after the
        // sign — "%#08x" % 255 == "0x0000ff", "%08.3d" % -7 == "-0000007" — which the generic Pad (which would
        // place the zeros before the prefix) cannot express. Left-justify always space-pads on the right.
        private static string PadInteger(string sign, string prefix, string digits, int width, bool left, bool zero)
        {
            int len = sign.Length + prefix.Length + digits.Length;
            if (len >= width)
                return sign + prefix + digits;

            int deficit = width - len;
            if (left)
                return sign + prefix + digits + new string(' ', deficit);
            if (zero)
                return sign + prefix + new string('0', deficit) + digits;
            return new string(' ', deficit) + sign + prefix + digits;
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
