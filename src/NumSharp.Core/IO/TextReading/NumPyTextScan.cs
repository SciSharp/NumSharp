using System;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace NumSharp.IO
{
    /// <summary>
    ///     How one element of NumPy's text reader (<c>array_from_text</c>) is scanned and staged: the C routine NumPy's
    ///     <c>ArrFuncs</c> table names for the dtype (<c>fromstr</c> for a string, <c>scanfunc</c> for a file), and the wide
    ///     value the routine produces before the dtype's C assignment narrows it.
    /// </summary>
    /// <remarks>
    ///     NumPy writes each scanned value straight into the element with a C cast (<c>*(npy_byte *)ip = result</c>); the
    ///     port stages the wide value and converts the whole buffer with the cast kernels afterwards, which apply the same C
    ///     conversions (an integer truncates, a double rounds to float32 / float16, a double is true when it is not 0).
    /// </remarks>
    internal enum TextScanFamily
    {
        /// <summary>A signed integer: <c>PyOS_strtol</c> (string) / glibc <c>%d</c>-family <c>fscanf</c> (file), staged as
        ///     <see cref="long"/>.</summary>
        Signed,

        /// <summary>An unsigned integer (and NumSharp's <c>Char</c>, NumPy's <c>uint16</c>): <c>PyOS_strtoul</c> / glibc
        ///     <c>%u</c>-family <c>fscanf</c>, staged as <see cref="ulong"/>.</summary>
        Unsigned,

        /// <summary>A bool, float16, float32 or float64: <c>NumPyOS_ascii_strtod</c> / <c>NumPyOS_ascii_ftolf</c>, staged as
        ///     <see cref="double"/>.</summary>
        Float,

        /// <summary>complex128: <c>CDOUBLE_fromstr</c> / <c>CDOUBLE_scan</c>, staged as <see cref="Complex"/>.</summary>
        Complex,

        /// <summary>NumSharp's Decimal (no NumPy analog): the float grammar of <see cref="Float"/>, its number converted to
        ///     <see cref="decimal"/> exactly.</summary>
        Decimal,
    }

    /// <summary>
    ///     Ports of the C routines behind the text modes of <c>np.fromstring</c> and <c>np.fromfile</c> (NumPy 2.4.2,
    ///     <c>ctors.c</c>, <c>arraytypes.c.src</c>, <c>numpyos.c</c>, and the CPython 3.12 / glibc routines they call),
    ///     reproducing their byte-level behavior: which characters each scanner consumes, the value it stores when it
    ///     consumes nothing, and how the separator is matched.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>The C long.</b> <c>np.fromstring</c>'s integer readers are <c>PyOS_strtol</c> / <c>PyOS_strtoul</c>, which work in
    ///     a C <c>long</c>: 64 bits on Linux and macOS, 32 on Windows, where an out-of-range number saturates at 2^31. NumSharp
    ///     pins the LP64 (Linux) answers, as its random module pins the LP64 integer types, so <c>"4294967296"</c> read as int8
    ///     is 0 (the 64-bit value truncated), never Windows NumPy's -1. <c>np.fromfile</c>'s integer readers are the C library's
    ///     <c>fscanf</c>; glibc's behavior is ported (probed through <c>fmemopen</c>, and matched by Linux NumPy). Windows
    ///     NumPy's MSVC <c>fscanf</c> reads the same values, but when an integer scan fails it also consumes the character
    ///     that ended the scan, so a stream read that way is left one byte further on than NumSharp and Linux NumPy leave
    ///     it.
    ///     </para>
    ///     <para>
    ///     <b>C strings.</b> Each <c>fromstr</c> routine reads a NUL-terminated C string: the scanners here read byte
    ///     <c>i</c> of the input as 0 at and past its end, so a scan stops at an embedded NUL exactly as it stops at the
    ///     terminator.
    ///     </para>
    /// </remarks>
    internal static partial class NumPyTextScan
    {
        /// <summary>NumPy's NaN: the positive quiet NaN (<c>0x7FF8...</c>); .NET's <see cref="double.NaN"/> is the negative one.</summary>
        internal static readonly double PositiveNaN = BitConverter.Int64BitsToDouble(0x7FF8000000000000L);

        /// <summary>
        ///     The scan family of a numeric dtype (<see cref="TextScanFamily"/>): the classification NumPy's <c>ArrFuncs</c>
        ///     table makes when it assigns each dtype its <c>fromstr</c> / <c>scanfunc</c>.
        /// </summary>
        /// <param name="tc">The dtype's type code (one of the 15 numeric NumSharp lanes).</param>
        /// <returns>The family.</returns>
        internal static TextScanFamily FamilyOf(NPTypeCode tc)
        {
            // Char is NumPy's uint16 everywhere NumSharp exchanges text (it scans with %hu / PyOS_strtoul).
            if (tc == NPTypeCode.Char)
                return TextScanFamily.Unsigned;
            if (tc.IsInteger())
                return tc.IsSigned() ? TextScanFamily.Signed : TextScanFamily.Unsigned;
            if (tc == NPTypeCode.Complex)
                return TextScanFamily.Complex;
            if (tc == NPTypeCode.Decimal)
                return TextScanFamily.Decimal;
            // Boolean, Half, Single, Double: NumPy's BOOL / HALF / FLOAT / DOUBLE readers all scan a double.
            return TextScanFamily.Float;
        }

        /// <summary>The lane a family stages its values in before they are converted to the dtype.</summary>
        /// <param name="family">The family.</param>
        /// <returns>The staging type code.</returns>
        internal static NPTypeCode StagingType(TextScanFamily family) => family switch
        {
            TextScanFamily.Signed => NPTypeCode.Int64,
            TextScanFamily.Unsigned => NPTypeCode.UInt64,
            TextScanFamily.Complex => NPTypeCode.Complex,
            TextScanFamily.Decimal => NPTypeCode.Decimal,
            _ => NPTypeCode.Double,
        };

        // ------------------------------------------------------------------------------------------------------------
        // Character classes (all of them the C locale's: NumPyOS_ascii_isspace, Py_ISSPACE and glibc's isspace agree on
        // ' ', \t, \n, \v, \f, \r; a byte >= 0x80 is no class).

        /// <summary>C's <c>isspace</c> in the C locale: space, <c>\t</c>, <c>\n</c>, <c>\v</c>, <c>\f</c>, <c>\r</c>.</summary>
        /// <param name="c">A byte value, or -1 (EOF).</param>
        /// <returns>True for the six whitespace bytes.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsCSpace(int c) => c == ' ' || (uint)(c - '\t') <= '\r' - '\t';

        /// <summary>An ASCII digit.</summary>
        /// <param name="c">A byte value, or -1.</param>
        /// <returns>True for '0'..'9'.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsDigit(int c) => (uint)(c - '0') <= 9;

        /// <summary>An ASCII letter or digit (<c>NumPyOS_ascii_isalnum</c>).</summary>
        /// <param name="c">A byte value, or -1.</param>
        /// <returns>True for 0-9, A-Z, a-z.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsAlnum(int c) => IsDigit(c) || (uint)((c | 0x20) - 'a') <= 'z' - 'a';

        /// <summary>Byte <paramref name="i"/> of a C string: 0 at and past the end (the terminator NumPy's buffers carry).</summary>
        /// <param name="s">The bytes.</param>
        /// <param name="i">The index (any non-negative value).</param>
        /// <returns>The byte, or 0.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int At(ReadOnlySpan<byte> s, int i) => (uint)i < (uint)s.Length ? s[i] : 0;

        /// <summary>
        ///     <c>NumPyOS_ascii_strncasecmp(s + i, word, n) == 0</c>: the lower-case ASCII <paramref name="word"/> starts at
        ///     <paramref name="i"/>, letters compared without case (a NUL, or the end, never matches).
        /// </summary>
        /// <param name="s">The bytes.</param>
        /// <param name="i">Where the comparison starts.</param>
        /// <param name="word">The lower-case word.</param>
        /// <returns>True when every byte matches.</returns>
        private static bool CaseMatch(ReadOnlySpan<byte> s, int i, string word)
        {
            for (int k = 0; k < word.Length; k++)
            {
                int c = At(s, i + k);
                // tolower maps only A-Z; '|' 0x20 maps exactly the one upper-case letter onto each lower-case one.
                if (c == 0 || (c | 0x20) != word[k] || !IsAlnum(c))
                    return false;
            }
            return true;
        }

        // ------------------------------------------------------------------------------------------------------------
        // fromstr routines

        /// <summary>
        ///     CPython 3.12's <c>PyOS_strtoul(str, &amp;end, 10)</c> with a 64-bit <c>unsigned long</c>: leading whitespace,
        ///     then leading zeros, then decimal digits. No sign is accepted (<c>"+1"</c> and <c>"-1"</c> consume nothing past
        ///     their whitespace); a value past 2^64 - 1 consumes all its digits and returns <see cref="ulong.MaxValue"/>.
        /// </summary>
        /// <param name="s">The C string.</param>
        /// <param name="i">Where the scan starts.</param>
        /// <param name="end">The index just past the last byte consumed: past the whitespace and zeros even when no digit
        ///     follows, which NumPy's reader counts as a read (<c>"  "</c> reads as 0).</param>
        /// <returns>The value; 0 when no digit was read.</returns>
        internal static ulong PyOSStrtoul(ReadOnlySpan<byte> s, int i, out int end)
        {
            while (IsCSpace(At(s, i)))
                i++;
            while (At(s, i) == '0')
                i++;
            ulong result = 0;
            // digitlimit[10] for a 64-bit long: the first 19 digits cannot overflow, the 20th is checked, a 21st overflows.
            int ovlimit = 19;
            int c;
            while ((uint)(c = At(s, i) - '0') <= 9)
            {
                if (ovlimit > 0)
                    result = result * 10 + (uint)c;
                else
                {
                    if (ovlimit < 0 || result > ulong.MaxValue / 10)
                        goto overflowed;
                    result *= 10;
                    ulong next = result + (uint)c;
                    if (next < result)
                        goto overflowed;
                    result = next;
                }
                i++;
                ovlimit--;
            }
            end = i;
            return result;

        overflowed:
            // spool through the remaining digits: they are part of the (overflowed) number
            while (IsDigit(At(s, i)))
                i++;
            end = i;
            return ulong.MaxValue;
        }

        /// <summary>
        ///     CPython 3.12's <c>PyOS_strtol(str, &amp;end, 10)</c> with a 64-bit <c>long</c>: leading whitespace, an optional
        ///     sign, then <see cref="PyOSStrtoul"/> (which skips whitespace AGAIN, so <c>"- 1"</c> is -1). A magnitude past
        ///     <see cref="long.MaxValue"/> saturates at <see cref="long.MaxValue"/> for either sign, except exactly 2^63 with a
        ///     minus sign, which is <see cref="long.MinValue"/>.
        /// </summary>
        /// <param name="s">The C string.</param>
        /// <param name="i">Where the scan starts.</param>
        /// <param name="end">The index just past the last byte consumed (past a sign even when no digit follows).</param>
        /// <returns>The value; 0 when no digit was read.</returns>
        internal static long PyOSStrtol(ReadOnlySpan<byte> s, int i, out int end)
        {
            while (IsCSpace(At(s, i)))
                i++;
            int sign = At(s, i);
            if (sign == '+' || sign == '-')
                i++;
            ulong u = PyOSStrtoul(s, i, out end);
            if (u <= long.MaxValue)
                return sign == '-' ? -(long)u : (long)u;
            if (sign == '-' && u == 0x8000000000000000UL)
                return long.MinValue;
            // PyOS_strtol's ERANGE answer is LONG_MAX whatever the sign (glibc's strtol would give LONG_MIN).
            return long.MaxValue;
        }

        /// <summary>
        ///     NumPy's <c>NumPyOS_ascii_strtod</c>: leading whitespace, then NumPy's own <c>nan</c> / <c>inf</c> forms
        ///     (case-insensitive, one optional sign; <c>nan(chars)</c> takes letters, digits and <c>_</c> between optional
        ///     parentheses; every NaN is the POSITIVE quiet NaN, <c>"-nan"</c> included), else <c>PyOS_string_to_double</c>'s
        ///     correctly rounded decimal grammar.
        /// </summary>
        /// <param name="s">The C string.</param>
        /// <param name="i">Where the scan starts.</param>
        /// <param name="end">The index just past the number. When no number follows the whitespace it is the index AFTER the
        ///     whitespace (<c>NumPyOS_ascii_strtod_plain</c> sets <c>*endptr = s</c> once the whitespace is skipped).</param>
        /// <returns>The value; -1.0 when no number was read (<c>PyOS_string_to_double</c>'s error return, which NumPy stores
        ///     in the element).</returns>
        internal static double NumPyOSStrtod(ReadOnlySpan<byte> s, int i, out int end)
        {
            while (IsCSpace(At(s, i)))
                i++;
            int p = i;
            bool negative = false;
            int c = At(s, p);
            if (c == '-')
            {
                negative = true;
                p++;
            }
            else if (c == '+')
                p++;

            if (CaseMatch(s, p, "nan"))
            {
                p += 3;
                if (At(s, p) == '(')
                {
                    p++;
                    while (IsAlnum(At(s, p)) || At(s, p) == '_')
                        p++;
                    if (At(s, p) == ')')
                        p++;
                }
                end = p;
                return PositiveNaN;
            }
            if (CaseMatch(s, p, "inf"))
            {
                p += 3;
                if (CaseMatch(s, p, "inity"))
                    p += 5;
                end = p;
                return negative ? double.NegativeInfinity : double.PositiveInfinity;
            }
            return StrtodPlain(s, i, out end);
        }

        /// <summary>
        ///     <c>PyOS_string_to_double(s, &amp;end, NULL)</c> over CPython's <c>_Py_dg_strtod</c>: an optional sign, digits
        ///     with an optional fraction (at least one digit in all; <c>"5."</c> and <c>".5"</c> are numbers, <c>"."</c> is not),
        ///     and an exponent only when a digit follows its optional sign (<c>"1e+"</c> stops before the <c>e</c>). The value
        ///     is correctly rounded; one past the double range is an infinity (no overflow exception is passed).
        /// </summary>
        /// <param name="s">The C string.</param>
        /// <param name="start">Where the number starts (its sign, if any).</param>
        /// <param name="end">The index just past the number, or <paramref name="start"/> when there is none.</param>
        /// <returns>The value, or -1.0 when there is no number.</returns>
        private static double StrtodPlain(ReadOnlySpan<byte> s, int start, out int end)
        {
            int p = MatchDecimalNumber(s, start);
            if (p == start)
            {
                end = start;
                return -1.0;
            }
            end = p;
            return ParseDoublePrefix(s.Slice(start, p - start));
        }

        /// <summary>
        ///     The extent of <c>_Py_dg_strtod</c>'s number at <paramref name="start"/>: <c>[+-]? digits [. digits]</c> (at least
        ///     one digit) followed by an exponent <c>[eE][+-]?digits</c> when it has a digit.
        /// </summary>
        /// <param name="s">The C string.</param>
        /// <param name="start">Where the number would start.</param>
        /// <returns>The index just past the number, or <paramref name="start"/> when none starts there.</returns>
        internal static int MatchDecimalNumber(ReadOnlySpan<byte> s, int start)
            // The C string ends at the span's end: NDDoubleParse reads past it as 0, like the terminator.
            => Utilities.NDDoubleParse.MatchDecimal(s, start);

        /// <summary>
        ///     The correctly rounded double of an ASCII decimal number (the text <see cref="MatchDecimalNumber"/> matched):
        ///     what <c>_Py_dg_strtod</c> returns, an infinity past the range and a (signed) zero below it.
        /// </summary>
        /// <param name="number">The number's bytes.</param>
        /// <returns>The value.</returns>
        internal static double ParseDoublePrefix(ReadOnlySpan<byte> number)
            // Clinger / Eisel-Lemire, verified against .NET's correctly rounded parser; the grammar was already matched.
            => Utilities.NDDoubleParse.Parse(number);

        /// <summary>
        ///     NumPy's <c>CDOUBLE_fromstr</c>: a real number, then — when the next byte is a sign — a second number that is the
        ///     imaginary part only if a <c>j</c> follows it. Two NumPy quirks are kept: a second number without <c>j</c> is still
        ///     CONSUMED (its <c>endptr = prev</c> restores nothing, so <c>"1+2"</c> reads as <c>1+0j</c>), and a <c>j</c> right
        ///     after the first scan makes it the imaginary part even when that scan read nothing (<c>"j"</c> is <c>-1j</c>,
        ///     the failed scan's -1.0).
        /// </summary>
        /// <param name="s">The C string.</param>
        /// <param name="i">Where the scan starts.</param>
        /// <param name="end">The index just past what was consumed.</param>
        /// <returns>The value (the parts a failed scan produced when nothing was read).</returns>
        internal static Complex CDoubleFromstr(ReadOnlySpan<byte> s, int i, out int end)
        {
            double real = NumPyOSStrtod(s, i, out int e);
            int c = At(s, e);
            if (c == '+' || c == '-')
            {
                double imag = NumPyOSStrtod(s, e, out int e2);
                if (At(s, e2) == 'j')
                {
                    end = e2 + 1;
                    return new Complex(real, imag);
                }
                end = e2;
                return new Complex(real, 0.0);
            }
            if (c == 'j')
            {
                end = e + 1;
                return new Complex(0.0, real);
            }
            end = e;
            return new Complex(real, 0.0);
        }

        /// <summary>
        ///     NumSharp's Decimal reader (no NumPy analog), shaped like a NumPy <c>fromstr</c> routine:
        ///     <see cref="NumPyOSStrtod"/>'s grammar, the decimal number it matches converted exactly. A NaN, an infinity, or a
        ///     number past the decimal range cannot be stored: the routine reports an error (NumPy's reader then reports
        ///     unmatched data unless the text has ended); a scan that consumed nothing stores -1, as the float readers store
        ///     -1.0.
        /// </summary>
        /// <param name="s">The C string.</param>
        /// <param name="i">Where the scan starts.</param>
        /// <param name="end">The index just past what was consumed (past the whitespace when no number follows it).</param>
        /// <param name="value">The value (-1 when nothing was read or the number cannot be stored).</param>
        /// <returns>0, or -1 when the matched text is a NaN, an infinity or out of the decimal range (a <c>fromstr</c>
        ///     routine's error return).</returns>
        internal static int DecimalFromstr(ReadOnlySpan<byte> s, int i, out int end, out decimal value)
        {
            while (IsCSpace(At(s, i)))
                i++;
            int p = MatchDecimalNumber(s, i);
            if (p == i)
            {
                // No decimal number here: NumPy's grammar may still read a nan / inf form (which a decimal cannot hold), or
                // nothing at all, which keeps the float readers' -1 failure value and is not an error by itself.
                NumPyOSStrtod(s, i, out end);
                value = -1m;
                return end == i ? 0 : -1;
            }
            end = p;
            return TryParseDecimal(s.Slice(i, p - i), out value) ? 0 : -1;
        }

        /// <summary>Converts an ASCII decimal number to <see cref="decimal"/> exactly (rounded to its 28-29 digits).</summary>
        /// <param name="number">The number's bytes (already matched by <see cref="MatchDecimalNumber"/>).</param>
        /// <param name="value">The value, or -1 when it is out of range.</param>
        /// <returns>False when the number is past <see cref="decimal.MaxValue"/>.</returns>
        internal static bool TryParseDecimal(ReadOnlySpan<byte> number, out decimal value)
        {
            if (decimal.TryParse(number, NumberStyles.Float, CultureInfo.InvariantCulture, out value))
                return true;
            value = -1m;
            return false;
        }

        // ------------------------------------------------------------------------------------------------------------
        // Separators

        /// <summary>
        ///     NumPy's <c>swab_separator</c>: every run of whitespace in the separator becomes ONE space (the wildcard), a space
        ///     is added in front when the separator does not start with whitespace, and — the code's behavior, whatever its
        ///     comment says — a second space is appended when it already ends with one. <c>","</c> is <c>" ,"</c>, <c>" "</c>
        ///     is <c>"  "</c>.
        /// </summary>
        /// <param name="sep">The separator's UTF-8 bytes (no NUL: the argument converter refused one).</param>
        /// <returns>The separator the skip routines match.</returns>
        internal static byte[] SwabSeparator(ReadOnlySpan<byte> sep)
        {
            var result = new byte[sep.Length + 2];
            int n = 0;
            if (sep.Length > 0 && !IsCSpace(sep[0]))
                result[n++] = (byte)' ';
            bool skipSpace = false;
            foreach (byte b in sep)
            {
                if (IsCSpace(b))
                {
                    if (!skipSpace)
                    {
                        result[n++] = (byte)' ';
                        skipSpace = true;
                    }
                }
                else
                {
                    result[n++] = b;
                    skipSpace = false;
                }
            }
            if (n > 0 && result[n - 1] == ' ')
                result[n++] = (byte)' ';
            return result.AsSpan(0, n).ToArray();
        }

        /// <summary>
        ///     NumPy's <c>fromstr_skip_separator</c>: matches the swabbed separator at <paramref name="pos"/>, a space of it
        ///     matching any run of whitespace (none included). Matching nothing at all fails, even for a separator of spaces.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="pos">The position; moved past what was matched (also on failure, as NumPy moves it).</param>
        /// <param name="sep">The swabbed separator (<see cref="SwabSeparator"/>).</param>
        /// <returns>0 when matched, -1 when the text ended first, -2 when it does not match.</returns>
        internal static int FromstrSkipSeparator(ReadOnlySpan<byte> s, ref int pos, ReadOnlySpan<byte> sep)
        {
            int str = pos, si = 0, result;
            while (true)
            {
                if (str >= s.Length)
                {
                    result = -1;
                    break;
                }
                int c = s[str];
                if (si == sep.Length)
                {
                    result = str != pos ? 0 : -2;
                    break;
                }
                if (sep[si] == ' ')
                {
                    // whitespace wildcard: a non-space moves on in the separator without consuming the byte
                    if (!IsCSpace(c))
                    {
                        si++;
                        continue;
                    }
                }
                else if (sep[si] != c)
                {
                    result = -2;
                    break;
                }
                else
                    si++;
                str++;
            }
            pos = str;
            return result;
        }
    }
}
