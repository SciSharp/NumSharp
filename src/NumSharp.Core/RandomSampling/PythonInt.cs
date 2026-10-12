using System;
using System.Globalization;
using System.Numerics;
using System.Text;
using NumSharp.Backends.Printing;

namespace NumSharp
{
    /// <summary>
    ///     Python's <c>int()</c> / <c>int(str, base)</c> / <c>str()</c> conversions, as NumPy's random module leans on
    ///     them when it coerces seeds, keys and counters (<c>_coerce_to_uint32_array</c>, <c>int_to_array</c>).
    /// </summary>
    /// <remarks>
    ///     These are the conversions whose accept/reject decisions — and whose exact error texts — reach the caller of
    ///     <see cref="SeedSequence"/> and <see cref="Philox"/>. A float truncates toward zero (<c>int(3.7) == 3</c>,
    ///     <c>int(-0.5) == 0</c>) rather than rounding, NaN and infinity are distinct errors, and a string parses with
    ///     Python's grammar (surrounding whitespace, a sign, single underscores between digits, a base prefix that must
    ///     agree with the base) — never with .NET's culture-sensitive number parsing.
    /// </remarks>
    internal static class PythonInt
    {
        /// <summary>
        ///     Python's <c>int(v)</c> of a scalar: integers (any C# width, <see cref="BigInteger"/>, bool, char) as-is,
        ///     real floats truncated toward zero, strings and bytes (an element of a <c>U</c> / <c>S</c> array) parsed in
        ///     base 10.
        /// </summary>
        /// <param name="v">The scalar.</param>
        /// <returns>The integer value.</returns>
        /// <exception cref="ValueError">A NaN (<c>cannot convert float NaN to integer</c>) or an unparsable string / bytes
        /// (<c>invalid literal for int() with base 10: '…'</c> / <c>b'…'</c>).</exception>
        /// <exception cref="OverflowException">An infinity (<c>cannot convert float infinity to integer</c>).</exception>
        /// <exception cref="TypeError">A complex or any other non-real value (<c>int() argument must be a string, a
        /// bytes-like object or a real number, not '…'</c>).</exception>
        internal static BigInteger From(object v)
        {
            switch (v)
            {
                case bool b:
                    return b ? BigInteger.One : BigInteger.Zero;
                case char c:
                    return c;
                case BigInteger bi:
                    return bi;
                case sbyte or byte or short or ushort or int or uint or long:
                    return Convert.ToInt64(v, CultureInfo.InvariantCulture);
                case ulong u:
                    return u;
                case double d:
                    return FromDouble(d);
                case float f:
                    return FromDouble(f);
                case Half h:
                    return FromDouble((double)h);
                case decimal m:
                    // Decimal has no NaN/infinity; truncation toward zero is exact.
                    return new BigInteger(decimal.Truncate(m));
                case string s:
                    return Parse(s, 10);
                case byte[] bytes:
                    // An S scalar: NumPy's 0-d bytes array converts through Python's int(bytes).
                    return ParseBytes(bytes, 10);
                case Complex:
                    throw new TypeError("int() argument must be a string, a bytes-like object or a real number, not 'complex'");
                case null:
                    throw new TypeError("int() argument must be a string, a bytes-like object or a real number, not 'NoneType'");
                default:
                    throw new TypeError($"int() argument must be a string, a bytes-like object or a real number, not '{v.GetType().Name}'");
            }
        }

        /// <summary>
        ///     Python's <c>float(int)</c>: the integer rounded to the NEAREST double, ties to even.
        /// </summary>
        /// <param name="v">The integer.</param>
        /// <returns>The correctly rounded double (±infinity past the double range, where Python raises instead).</returns>
        /// <remarks>
        ///     .NET's <c>(double)BigInteger</c> TRUNCATES the bits below the 53-bit mantissa (<c>2**63 + 1025</c> becomes
        ///     <c>2**63</c>, Python gives <c>2**63 + 2048</c>), and .NET 8's <c>(double)ulong</c> double-rounds at and above
        ///     2**63 — so a value that must match NumPy's int-list-to-float64 inference is rounded here, by hand: keep 54
        ///     bits (53 + a round bit), fold every lower bit into a sticky flag, round half to even, and rescale.
        /// </remarks>
        internal static double ToDouble(BigInteger v)
        {
            if (v.IsZero)
                return 0.0;
            bool negative = v.Sign < 0;
            BigInteger a = BigInteger.Abs(v);
            long bits = (long)a.GetBitLength();
            if (bits <= 53)
                return negative ? -(double)(ulong)a : (double)(ulong)a; // exact: fits the mantissa
            int shift = (int)(bits - 54);
            ulong top = (ulong)(a >> shift); // 54 significant bits
            bool sticky = !(a & ((BigInteger.One << shift) - 1)).IsZero;
            bool roundBit = (top & 1) != 0;
            ulong mantissa = top >> 1;
            if (roundBit && (sticky || (mantissa & 1) != 0))
                mantissa++; // may carry to 2**53 — still exact in a double
            double r = Math.ScaleB(mantissa, shift + 1);
            return negative ? -r : r;
        }

        /// <summary>Python's <c>int(float)</c>: truncation toward zero, exact for any finite magnitude.</summary>
        /// <param name="d">The float.</param>
        /// <returns>The truncated integer.</returns>
        /// <exception cref="ValueError"><paramref name="d"/> is NaN.</exception>
        /// <exception cref="OverflowException"><paramref name="d"/> is infinite.</exception>
        private static BigInteger FromDouble(double d)
        {
            if (double.IsNaN(d))
                throw new ValueError("cannot convert float NaN to integer");
            if (double.IsInfinity(d))
                throw new OverflowException("cannot convert float infinity to integer");
            // BigInteger(double) keeps the exact integral part of the binary value (2.0**64 is exactly 2**64).
            return new BigInteger(Math.Truncate(d));
        }

        /// <summary>
        ///     CPython's default <c>sys.get_int_max_str_digits()</c>: <c>int(str)</c> in a base that is not a power of two
        ///     refuses text with more digits than this (a guard against quadratic conversion), before converting anything.
        /// </summary>
        private const int MaxStrDigits = 4300;

        /// <summary>
        ///     Python's <c>int(s, base)</c> for bases 8, 10 and 16 — CPython's <c>PyLong_FromUnicodeObject</c>: the text is
        ///     first rewritten to ASCII (<c>_PyUnicode_TransformDecimalAndSpaceToASCII</c>: a non-ASCII whitespace becomes a
        ///     space, a non-ASCII DECIMAL DIGIT of any script becomes its ASCII digit, any other non-ASCII character ends the
        ///     text as an invalid character), then parsed by <c>PyLong_FromString</c>: surrounding whitespace ignored, one
        ///     sign, base 8/16 accept their <c>0o</c>/<c>0x</c> prefix (either case) followed by at most one underscore, and
        ///     single underscores between digits.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="radix">The base (8, 10 or 16).</param>
        /// <param name="numpyStr">True when the text is a NumPy <c>np.str_</c> scalar (an element of a <c>U</c> array):
        ///     only the error text differs, because CPython quotes the original OBJECT's repr, <c>np.str_('1a')</c>.</param>
        /// <returns>The parsed value.</returns>
        /// <exception cref="ValueError">The text is not a valid literal in that base (<c>invalid literal for int() with base
        ///     {base}: '…'</c>, the repr truncated to 200 code points as CPython's <c>%.200R</c> does), or a valid base-10
        ///     literal has more than 4300 digits (<c>Exceeds the limit (4300 digits) for integer string conversion: value has
        ///     N digits; use sys.set_int_max_str_digits() to increase the limit</c>).</exception>
        /// <remarks>
        ///     The ASCII rewrite is why <c>int('1٢')</c> is 12 and <c>int('1𝟏')</c> is 11 (an Arabic-Indic and a
        ///     mathematical-bold digit), while a non-ASCII letter or a zero-width space still makes the text invalid. The
        ///     digit limit applies after the grammar checks: text with an invalid character reports the invalid literal
        ///     however long it is, and base 16 (a power of two) has no limit.
        /// </remarks>
        internal static BigInteger Parse(string s, int radix, bool numpyStr = false)
        {
            string ascii = TransformDecimalAndSpaceToAscii(s);
            // PyLong_FromString reads a C string: a NUL ends the text it parses, and PyLong_FromUnicodeObject then
            // rejects the whole literal because the parse did not reach the true end (unless the digit limit fired).
            int nul = ascii.IndexOf('\0');
            ParseStatus status = ParseAscii(nul < 0 ? ascii.AsSpan() : ascii.AsSpan(0, nul), radix, out BigInteger value, out int digits);
            if (status == ParseStatus.TooManyDigits)
                throw DigitLimit(digits);
            if (status == ParseStatus.Invalid || nul >= 0)
                throw Invalid(s, radix, numpyStr);
            return value;
        }

        /// <summary>
        ///     Python's <c>int(b, base)</c> of a <c>bytes</c> value (CPython's <c>_PyLong_FromBytes</c>): the bytes go to
        ///     <c>PyLong_FromString</c> as they are — no Unicode rewrite, so a byte above <c>0x7F</c> is an invalid
        ///     character — with Python's grammar (see <see cref="Parse"/>); a NUL byte ends the text and makes the literal
        ///     invalid.
        /// </summary>
        /// <param name="b">The bytes (an <c>S</c> element or Python <c>bytes</c>).</param>
        /// <param name="radix">The base (8, 10 or 16).</param>
        /// <returns>The parsed value.</returns>
        /// <exception cref="ValueError">Not a valid literal (<c>invalid literal for int() with base 10: b'1a'</c>: the repr of
        ///     the first 200 bytes, cut at 200 characters), or more than 4300 decimal digits.</exception>
        internal static BigInteger ParseBytes(byte[] b, int radix)
        {
            int nul = Array.IndexOf(b, (byte)0);
            int n = nul < 0 ? b.Length : nul;
            // Every byte is its own character; '?' stands for a byte above 0x7F, which is no digit and no Py_ISSPACE
            // whitespace (a Latin-1 widening would not do: .NET calls U+0085 and U+00A0 whitespace).
            var chars = new char[n];
            for (int i = 0; i < n; i++)
                chars[i] = b[i] < 0x80 ? (char)b[i] : '?';
            ParseStatus status = ParseAscii(chars, radix, out BigInteger value, out int digits);
            if (status == ParseStatus.TooManyDigits)
                throw DigitLimit(digits);
            if (status == ParseStatus.Invalid || nul >= 0)
            {
                string repr = PyRepr.Bytes(b.AsSpan(0, Math.Min(b.Length, 200)));
                if (repr.Length > 200)
                    repr = repr.Substring(0, 200); // all ASCII: units are code points
                throw new ValueError($"invalid literal for int() with base {radix}: {repr}");
            }
            return value;
        }

        /// <summary>The outcome of <see cref="ParseAscii"/>.</summary>
        private enum ParseStatus
        {
            /// <summary>A valid literal; the value is set.</summary>
            Ok,

            /// <summary>Not a literal in the base (a syntax error).</summary>
            Invalid,

            /// <summary>A valid base-10 literal with more than 4300 digits (Python refuses to convert it).</summary>
            TooManyDigits,
        }

        /// <summary>
        ///     CPython's <c>PyLong_FromString</c> over ASCII text: surrounding whitespace, one sign, a base 8/16 prefix (either
        ///     case) followed by at most one underscore, then digits with single underscores between them. The whole run is
        ///     validated before anything converts (<c>long_from_string_base</c>), so a syntax error wins over the digit limit.
        /// </summary>
        /// <param name="ascii">The text, already free of non-ASCII characters and NULs.</param>
        /// <param name="radix">The base (8, 10 or 16).</param>
        /// <param name="value">The value when <see cref="ParseStatus.Ok"/>, else zero.</param>
        /// <param name="digits">The number of digits read (underscores excluded) — the count the digit-limit text names.</param>
        /// <returns>The outcome.</returns>
        private static ParseStatus ParseAscii(ReadOnlySpan<char> ascii, int radix, out BigInteger value, out int digits)
        {
            value = BigInteger.Zero;
            digits = 0;
            int start = 0, end = ascii.Length;
            // Py_ISSPACE: only ASCII whitespace is left here, where char.IsWhiteSpace is CPython's set (\t \n \v \f \r
            // and the space; the C0 separators U+001C..U+001F are whitespace to neither).
            while (start < end && char.IsWhiteSpace(ascii[start])) start++;
            while (end > start && char.IsWhiteSpace(ascii[end - 1])) end--;

            int i = start;
            bool negative = false;
            if (i < end && (ascii[i] == '+' || ascii[i] == '-'))
            {
                negative = ascii[i] == '-';
                i++;
            }

            // A base prefix is legal only for the base it names; an underscore may directly follow it.
            bool afterPrefix = false;
            if (radix != 10 && i + 1 < end && ascii[i] == '0')
            {
                char p = char.ToLowerInvariant(ascii[i + 1]);
                if ((radix == 16 && p == 'x') || (radix == 8 && p == 'o'))
                {
                    i += 2;
                    afterPrefix = true;
                }
            }

            int digitStart = i;
            bool lastUnderscore = false;
            for (; i < end; i++)
            {
                char c = ascii[i];
                if (c == '_')
                {
                    // Underscores separate digits: never doubled, never trailing, never before the first digit
                    // unless they follow a base prefix.
                    if (lastUnderscore || (digits == 0 && !afterPrefix))
                        return ParseStatus.Invalid;
                    lastUnderscore = true;
                    continue;
                }
                int dv = DigitValue(c);
                if (dv < 0 || dv >= radix)
                    return ParseStatus.Invalid;
                digits++;
                lastUnderscore = false;
            }
            if (digits == 0 || lastUnderscore)
                return ParseStatus.Invalid;
            // Only a base that is not a power of two converts quadratically, so only base 10 has the limit here.
            if (radix == 10 && digits > MaxStrDigits)
                return ParseStatus.TooManyDigits;

            for (int k = digitStart; k < end; k++)
                if (ascii[k] != '_')
                    value = value * radix + DigitValue(ascii[k]);
            if (negative)
                value = -value;
            return ParseStatus.Ok;
        }

        /// <summary>CPython's <c>_MAX_STR_DIGITS_ERROR_FMT_TO_INT</c> for the default limit.</summary>
        /// <param name="digits">The digit count of the refused literal.</param>
        /// <returns>The exception to throw.</returns>
        private static ValueError DigitLimit(int digits)
            => new ValueError($"Exceeds the limit ({MaxStrDigits} digits) for integer string conversion: value has " +
                              $"{digits} digits; use sys.set_int_max_str_digits() to increase the limit");

        /// <summary>
        ///     CPython's <c>_PyUnicode_TransformDecimalAndSpaceToASCII</c>: code points below 127 are kept, a non-ASCII
        ///     whitespace becomes a space, a non-ASCII decimal digit becomes its ASCII digit, and the first other
        ///     character becomes <c>?</c> and ENDS the text (it is invalid in every base, so nothing after it matters).
        /// </summary>
        /// <param name="s">The original text.</param>
        /// <returns><paramref name="s"/> itself when it is already ASCII (the common case), else the rewritten text.</returns>
        /// <remarks>
        ///     CPython walks code points; a C# string holds an astral digit (<c>U+1D7CF</c>) as a surrogate pair, so the pair
        ///     is read as one code point and becomes ONE ASCII digit. The whitespace and digit sets are CPython 3.12's
        ///     (Unicode 15.0.0, <see cref="NumSharp.Utilities.PyUnicode"/>), not .NET's: .NET 10 reads Unicode 16.0, whose new
        ///     decimal digits (Garay <c>U+10D40</c>, ...) CPython 3.12 does not parse. (Until 2026-10-12 this read
        ///     <see cref="CharUnicodeInfo.GetDecimalDigitValue(string, int)"/> and accepted them on .NET 10.)
        /// </remarks>
        private static string TransformDecimalAndSpaceToAscii(string s)
        {
            // CPython returns the text untouched when every code point is ASCII (PyUnicode_IS_ASCII, DEL included); the
            // per-character rule below (which turns DEL into '?') runs only for a text with a non-ASCII character. For
            // int() the two agree anyway: DEL is an invalid character either way.
            if (s.AsSpan().IndexOfAnyExceptInRange('\0', '\x7F') < 0)
                return s;

            var sb = new StringBuilder(s.Length);
            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];
                if (c < 127)
                {
                    sb.Append(c);
                    continue;
                }
                // A valid surrogate pair is ONE code point (an astral digit is one digit); a lone surrogate is its own.
                int cp = c;
                bool pair = char.IsHighSurrogate(c) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]);
                if (pair)
                    cp = char.ConvertToUtf32(c, s[i + 1]);
                if (NumSharp.Utilities.PyUnicode.IsSpace(cp))
                {
                    sb.Append(' ');
                    if (pair)
                        i++;
                    continue;
                }
                int d = NumSharp.Utilities.PyUnicode.DecimalValue(cp);
                if (d >= 0)
                {
                    sb.Append((char)('0' + d));
                    if (pair)
                        i++;
                    continue;
                }
                sb.Append('?');
                break;
            }
            return sb.ToString();
        }

        /// <summary>The value of an ASCII digit or letter digit (<c>a</c>/<c>A</c> = 10 …), or -1.</summary>
        /// <param name="c">The character.</param>
        /// <returns>The digit value, or -1 for a non-digit.</returns>
        private static int DigitValue(char c)
        {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'z') return c - 'a' + 10;
            if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
            return -1;
        }

        /// <summary>CPython's <c>invalid literal for int() with base %d: %.200R</c>, quoting the ORIGINAL text's repr.</summary>
        /// <param name="s">The original (unstripped, not rewritten) text.</param>
        /// <param name="radix">The base.</param>
        /// <param name="numpyStr">True for an <c>np.str_</c> scalar, whose repr is <c>np.str_('…')</c>.</param>
        /// <returns>The exception to throw.</returns>
        /// <remarks>
        ///     <c>%.200R</c> cuts the repr at 200 CODE POINTS (the <c>np.str_(</c> wrapper included), so an astral
        ///     character printed raw is never split into a lone surrogate.
        /// </remarks>
        private static ValueError Invalid(string s, int radix, bool numpyStr)
        {
            string repr = numpyStr ? "np.str_(" + Repr(s) + ")" : Repr(s);
            if (repr.Length > 200)
            {
                // Walk 200 code points: a surrogate pair is one code point and two UTF-16 units.
                int units = 0;
                for (int cps = 0; cps < 200 && units < repr.Length; cps++)
                    units += char.IsHighSurrogate(repr[units]) && units + 1 < repr.Length && char.IsLowSurrogate(repr[units + 1]) ? 2 : 1;
                repr = repr.Substring(0, units);
            }
            return new ValueError($"invalid literal for int() with base {radix}: {repr}");
        }

        /// <summary>
        ///     Python's <c>repr(str)</c> (CPython's <c>unicode_repr</c>, see <see cref="PyRepr.Str"/>): single quotes unless
        ///     the text holds a single quote and no double quote; the quote, backslash, <c>\t \n \r</c>, the other C0
        ///     controls and DEL escaped; non-printable non-ASCII characters (a no-break or zero-width space, a line
        ///     separator, an unassigned code point) escaped as <c>\xhh</c> / <c>\uhhhh</c> / <c>\Uhhhhhhhh</c>.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <returns>The quoted repr.</returns>
        internal static string Repr(string s) => PyRepr.Str(s);

        /// <summary>
        ///     Python's <c>str(v)</c> for the values that reach an error message: floats in Python's shortest repr
        ///     (<c>3.5</c>, <c>1e+20</c>, <c>nan</c>), bools as <c>True</c>/<c>False</c>, null as <c>None</c>.
        /// </summary>
        /// <param name="v">The value.</param>
        /// <returns>The text.</returns>
        internal static string Str(object v) => v switch
        {
            null => "None",
            bool b => b ? "True" : "False",
            double d => ArrayFormatter.PythonFloatRepr(d, FloatKind.Double),
            float f => ArrayFormatter.PythonFloatRepr((double)f, FloatKind.Double),
            Half h => ArrayFormatter.PythonFloatRepr((double)h, FloatKind.Double),
            _ => Convert.ToString(v, CultureInfo.InvariantCulture),
        };
    }
}
