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
        ///     real floats truncated toward zero, strings parsed in base 10.
        /// </summary>
        /// <param name="v">The scalar.</param>
        /// <returns>The integer value.</returns>
        /// <exception cref="ValueError">A NaN (<c>cannot convert float NaN to integer</c>) or an unparsable string
        /// (<c>invalid literal for int() with base 10: '…'</c>).</exception>
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
        ///     Python's <c>int(s, base)</c> for bases 8, 10 and 16: surrounding whitespace is ignored, one sign is allowed,
        ///     base 8/16 accept their <c>0o</c>/<c>0x</c> prefix (either case), and single underscores may separate digits
        ///     (or follow the prefix).
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="radix">The base (8, 10 or 16).</param>
        /// <returns>The parsed value.</returns>
        /// <exception cref="ValueError">The text is not a valid literal in that base (<c>invalid literal for int() with base
        /// {base}: '…'</c>, the repr truncated to 200 characters as CPython's <c>%.200R</c> does).</exception>
        internal static BigInteger Parse(string s, int radix)
        {
            int start = 0, end = s.Length;
            while (start < end && char.IsWhiteSpace(s[start])) start++;
            while (end > start && char.IsWhiteSpace(s[end - 1])) end--;

            int i = start;
            bool negative = false;
            if (i < end && (s[i] == '+' || s[i] == '-'))
            {
                negative = s[i] == '-';
                i++;
            }

            // A base prefix is legal only for the base it names; an underscore may directly follow it.
            bool afterPrefix = false;
            if (radix != 10 && i + 1 < end && s[i] == '0')
            {
                char p = char.ToLowerInvariant(s[i + 1]);
                if ((radix == 16 && p == 'x') || (radix == 8 && p == 'o'))
                {
                    i += 2;
                    afterPrefix = true;
                }
            }

            BigInteger value = BigInteger.Zero;
            int digits = 0;
            bool lastUnderscore = false;
            for (; i < end; i++)
            {
                char c = s[i];
                if (c == '_')
                {
                    // Underscores separate digits: never doubled, never trailing, never before the first digit
                    // unless they follow a base prefix.
                    if (lastUnderscore || (digits == 0 && !afterPrefix))
                        throw Invalid(s, radix);
                    lastUnderscore = true;
                    continue;
                }
                int dv = DigitValue(c);
                if (dv < 0 || dv >= radix)
                    throw Invalid(s, radix);
                value = value * radix + dv;
                digits++;
                lastUnderscore = false;
            }
            if (digits == 0 || lastUnderscore)
                throw Invalid(s, radix);
            return negative ? -value : value;
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

        /// <summary>CPython's <c>invalid literal for int() with base %d: %.200R</c>.</summary>
        /// <param name="s">The original (unstripped) text.</param>
        /// <param name="radix">The base.</param>
        /// <returns>The exception to throw.</returns>
        private static ValueError Invalid(string s, int radix)
        {
            string repr = Repr(s);
            if (repr.Length > 200)
                repr = repr.Substring(0, 200);
            return new ValueError($"invalid literal for int() with base {radix}: {repr}");
        }

        /// <summary>
        ///     Python's <c>repr(str)</c>: single quotes unless the text holds a single quote and no double quote,
        ///     backslash escapes for the quote, backslash, <c>\n \r \t</c> and other control characters.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <returns>The quoted repr.</returns>
        internal static string Repr(string s)
        {
            char quote = s.IndexOf('\'') >= 0 && s.IndexOf('"') < 0 ? '"' : '\'';
            var sb = new StringBuilder(s.Length + 2);
            sb.Append(quote);
            foreach (char c in s)
            {
                switch (c)
                {
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    default:
                        if (c == quote)
                            sb.Append('\\').Append(c);
                        else if (c < 0x20 || c == 0x7f)
                            sb.Append("\\x").Append(((int)c).ToString("x2", CultureInfo.InvariantCulture));
                        else
                            sb.Append(c);
                        break;
                }
            }
            sb.Append(quote);
            return sb.ToString();
        }

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
