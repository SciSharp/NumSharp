using System;
using System.Globalization;
using System.Text;

namespace NumSharp.Backends.Printing
{
    /// <summary>
    ///     Python's text representations that NumPy's array printing is built from: <c>repr(str)</c>
    ///     (CPython's <c>unicode_repr</c>), <c>repr(bytes)</c> (<c>PyBytes_Repr</c> with smart quotes),
    ///     <c>str.isprintable()</c> per code point and <c>len()</c> of a string. NumPy prints a string array's
    ///     elements with <c>repr()</c> of the Python scalar (<c>arrayprint.repr_format</c>) and wraps lines by
    ///     <c>len()</c>, so these decide every byte of a printed string array.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Code points, not UTF-16 units.</b> Python strings are sequences of code points, C# strings of UTF-16
    ///     units. Every function here walks code points: a valid surrogate pair is one character (printed raw when
    ///     printable, else <c>\UXXXXXXXX</c>) and counts once in <see cref="CodePointLength"/>; a lone surrogate is a
    ///     character of category <c>Cs</c>, never printable, escaped as <c>\udXXX</c> exactly as Python escapes the
    ///     lone surrogates a <c>str</c> may hold.
    ///     </para>
    ///     <para>
    ///     <b>Unicode version.</b> <see cref="IsPrintable"/> reads the general category from .NET's tables, which
    ///     can be a newer Unicode version than the CPython build NumPy runs on; a code point assigned in between
    ///     (category <c>Cn</c> there, assigned here) is the only place the two can disagree.
    ///     </para>
    /// </remarks>
    internal static class PyRepr
    {
        /// <summary>CPython's <c>Py_hexdigits</c>: escapes use LOWERCASE hex digits (<c>'\x7f'</c>, <c>'\u2028'</c>).</summary>
        private const string HexDigits = "0123456789abcdef";

        /// <summary>
        ///     Python's <c>repr(s)</c> for a <c>str</c>: single quotes unless the text holds a single quote and no
        ///     double quote (then double quotes); backslash and the chosen quote escaped; <c>\t \n \r</c> by name;
        ///     other C0 controls and DEL as <c>\xhh</c>; non-ASCII characters raw when <see cref="IsPrintable"/>, else
        ///     <c>\xhh</c> / <c>\uhhhh</c> / <c>\Uhhhhhhhh</c> by magnitude.
        /// </summary>
        /// <param name="s">The text (null reads as the empty string).</param>
        /// <returns>The quoted, escaped representation.</returns>
        public static string Str(string s)
        {
            s ??= string.Empty;
            // CPython counts both quote kinds first: double quotes are used only when the text has single quotes and
            // no double quotes; with both present it keeps single quotes and escapes the single ones.
            bool hasSingle = s.IndexOf('\'') >= 0, hasDouble = s.IndexOf('"') >= 0;
            char quote = hasSingle && !hasDouble ? '"' : '\'';

            var sb = new StringBuilder(s.Length + 2);
            sb.Append(quote);
            for (int i = 0; i < s.Length; i++)
            {
                int cp = s[i];
                // A valid surrogate pair is ONE astral code point; a lone surrogate stays a code point of its own.
                if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
                {
                    cp = char.ConvertToUtf32(s[i], s[i + 1]);
                    i++;
                }
                AppendStrChar(sb, cp, quote);
            }
            sb.Append(quote);
            return sb.ToString();
        }

        /// <summary>
        ///     Python's <c>repr(s)</c> of a text given as CODE POINTS — a <c>U</c> element's UCS-4 units as NumPy's
        ///     <c>np.str_</c> holds them. Identical to <see cref="Str"/> for valid code points; a unit above U+10FFFF, which
        ///     no C# string can hold but a multi-character <c>np.str_</c> can (CPython's <c>PyUnicode_New</c> caps only the
        ///     maximum character it records), is non-printable and prints as <c>\Uhhhhhhhh</c> — NumPy's
        ///     <c>np.str_('1\U00110000')</c>.
        /// </summary>
        /// <param name="codePoints">The text's code points (trailing NULs already removed by the caller).</param>
        /// <returns>The quoted, escaped representation.</returns>
        public static string StrCodePoints(ReadOnlySpan<uint> codePoints)
        {
            bool hasSingle = false, hasDouble = false;
            foreach (uint u in codePoints)
            {
                hasSingle |= u == '\'';
                hasDouble |= u == '"';
            }
            char quote = hasSingle && !hasDouble ? '"' : '\'';

            var sb = new StringBuilder(codePoints.Length + 2);
            sb.Append(quote);
            foreach (uint u in codePoints)
            {
                if (u > 0x10FFFF)
                {
                    // Not a character: CPython's isprintable() is false for it, so the 8-digit escape (the value fits
                    // 32 bits; CPython prints the low 32 bits of a UCS-4 unit).
                    sb.Append("\\U");
                    for (int shift = 28; shift >= 0; shift -= 4)
                        sb.Append(HexDigits[(int)((u >> shift) & 0xF)]);
                    continue;
                }
                AppendStrChar(sb, (int)u, quote);
            }
            sb.Append(quote);
            return sb.ToString();
        }

        /// <summary>
        ///     Python's <c>ascii()</c> applied to a representation <paramref name="repr"/> already built (CPython's
        ///     <c>PyObject_ASCII</c>: the <c>repr</c>, then every non-ASCII code point escaped as <c>\xhh</c>,
        ///     <c>\uhhhh</c> or <c>\Uhhhhhhhh</c> by magnitude, lone surrogates as <c>\udXXX</c>). A bytes repr is
        ///     already ASCII and comes back unchanged.
        /// </summary>
        /// <param name="repr">A Python representation (the output of <see cref="Str"/> or <see cref="Bytes"/>).</param>
        /// <returns>The ASCII-only form.</returns>
        public static string Ascii(string repr)
        {
            // The all-ASCII repr is by far the common case: nothing to escape.
            if (repr.AsSpan().IndexOfAnyExceptInRange('\0', '\x7F') < 0)
                return repr;

            var sb = new StringBuilder(repr.Length + 8);
            for (int i = 0; i < repr.Length; i++)
            {
                int cp = repr[i];
                // A valid surrogate pair is ONE astral code point, escaped as \U; a lone surrogate escapes as \u.
                if (char.IsHighSurrogate(repr[i]) && i + 1 < repr.Length && char.IsLowSurrogate(repr[i + 1]))
                {
                    cp = char.ConvertToUtf32(repr[i], repr[i + 1]);
                    i++;
                }

                if (cp < 0x80)
                    sb.Append((char)cp);
                else if (cp < 0x100)
                    sb.Append("\\x").Append(HexDigits[cp >> 4]).Append(HexDigits[cp & 0xF]);
                else if (cp < 0x10000)
                    sb.Append("\\u").Append(HexDigits[(cp >> 12) & 0xF]).Append(HexDigits[(cp >> 8) & 0xF])
                      .Append(HexDigits[(cp >> 4) & 0xF]).Append(HexDigits[cp & 0xF]);
                else
                {
                    sb.Append("\\U");
                    for (int shift = 28; shift >= 0; shift -= 4)
                        sb.Append(HexDigits[(cp >> shift) & 0xF]);
                }
            }
            return sb.ToString();
        }

        /// <summary>
        ///     Python's <c>repr(b)</c> for a <c>bytes</c> value: the <c>b</c> prefix and the same quote choice as
        ///     <see cref="Str"/>; backslash and the chosen quote escaped; <c>\t \n \r</c> by name; every byte below
        ///     <c>0x20</c> or from <c>0x7f</c> up as <c>\xhh</c> (bytes have no "printable" notion past ASCII).
        /// </summary>
        /// <param name="b">The bytes.</param>
        /// <returns>The <c>b'...'</c> representation.</returns>
        public static string Bytes(ReadOnlySpan<byte> b)
        {
            bool hasSingle = b.IndexOf((byte)'\'') >= 0, hasDouble = b.IndexOf((byte)'"') >= 0;
            char quote = hasSingle && !hasDouble ? '"' : '\'';

            var sb = new StringBuilder(b.Length + 3);
            sb.Append('b').Append(quote);
            foreach (byte c in b)
            {
                if (c == quote || c == '\\')
                    sb.Append('\\').Append((char)c);
                else if (c == '\t')
                    sb.Append("\\t");
                else if (c == '\n')
                    sb.Append("\\n");
                else if (c == '\r')
                    sb.Append("\\r");
                else if (c < ' ' || c >= 0x7f)
                    sb.Append("\\x").Append(HexDigits[c >> 4]).Append(HexDigits[c & 0xF]);
                else
                    sb.Append((char)c);
            }
            sb.Append(quote);
            return sb.ToString();
        }

        /// <summary>
        ///     Python's <c>str.isprintable()</c> for one code point (<c>_PyUnicode_IsPrintable</c>): every character is
        ///     printable except the general categories <c>Cc</c>, <c>Cf</c>, <c>Cs</c>, <c>Co</c>, <c>Cn</c>,
        ///     <c>Zl</c>, <c>Zp</c> and <c>Zs</c> — with the ASCII space as the one printable <c>Zs</c>. So a
        ///     no-break space (<c>U+00A0</c>) and a BOM (<c>U+FEFF</c>) are escaped in a repr.
        /// </summary>
        /// <param name="cp">The code point (a lone surrogate is category <c>Cs</c>, values past U+10FFFF are refused by
        ///     the caller's decoder before they get here).</param>
        /// <returns>True when Python prints the character raw.</returns>
        public static bool IsPrintable(int cp)
        {
            if (cp == ' ')
                return true;
            // A lone surrogate has no scalar value for CharUnicodeInfo's int overload; it is category Cs.
            if (cp >= 0xD800 && cp <= 0xDFFF)
                return false;
            switch (CharUnicodeInfo.GetUnicodeCategory(cp))
            {
                case UnicodeCategory.Control:
                case UnicodeCategory.Format:
                case UnicodeCategory.Surrogate:
                case UnicodeCategory.PrivateUse:
                case UnicodeCategory.OtherNotAssigned:
                case UnicodeCategory.LineSeparator:
                case UnicodeCategory.ParagraphSeparator:
                case UnicodeCategory.SpaceSeparator:
                    return false;
                default:
                    return true;
            }
        }

        /// <summary>
        ///     Python's <c>len()</c> of the text a C# string holds: UTF-16 units minus one per valid surrogate pair, so
        ///     an astral character (an emoji) counts once. NumPy wraps printed lines by this length.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <returns>The number of code points.</returns>
        public static int CodePointLength(string s)
        {
            int n = s.Length;
            // Fast exit for the all-BMP text every numeric formatter produces: no high surrogate, nothing to subtract.
            int first = s.AsSpan().IndexOfAnyInRange('\uD800', '\uDBFF');
            if (first < 0)
                return n;
            for (int i = first; i < s.Length - 1; i++)
                if (char.IsHighSurrogate(s[i]) && char.IsLowSurrogate(s[i + 1]))
                {
                    n--;
                    i++;
                }
            return n;
        }

        /// <summary>Appends one code point of a <c>str</c> repr (the body of CPython's <c>unicode_repr</c> loop).</summary>
        /// <param name="sb">The output.</param>
        /// <param name="cp">The code point.</param>
        /// <param name="quote">The chosen quote character, which is escaped.</param>
        private static void AppendStrChar(StringBuilder sb, int cp, char quote)
        {
            if (cp == quote || cp == '\\')
            {
                sb.Append('\\').Append((char)cp);
                return;
            }
            switch (cp)
            {
                case '\t': sb.Append("\\t"); return;
                case '\n': sb.Append("\\n"); return;
                case '\r': sb.Append("\\r"); return;
            }
            // The ASCII range is decided by value alone: controls and DEL escaped, everything else raw.
            if (cp < ' ' || cp == 0x7F)
            {
                AppendHex(sb, 'x', cp, 2);
                return;
            }
            if (cp < 0x7F)
            {
                sb.Append((char)cp);
                return;
            }
            if (IsPrintable(cp))
            {
                if (cp > 0xFFFF)
                    sb.Append(char.ConvertFromUtf32(cp));
                else
                    sb.Append((char)cp);
                return;
            }
            // Non-printable: the shortest of CPython's three escape widths that holds the value.
            if (cp <= 0xFF)
                AppendHex(sb, 'x', cp, 2);
            else if (cp <= 0xFFFF)
                AppendHex(sb, 'u', cp, 4);
            else
                AppendHex(sb, 'U', cp, 8);
        }

        /// <summary>Appends <c>\</c>, the escape letter and <paramref name="digits"/> lowercase hex digits of the value.</summary>
        /// <param name="sb">The output.</param>
        /// <param name="letter"><c>x</c>, <c>u</c> or <c>U</c>.</param>
        /// <param name="value">The code point.</param>
        /// <param name="digits">2, 4 or 8.</param>
        private static void AppendHex(StringBuilder sb, char letter, int value, int digits)
        {
            sb.Append('\\').Append(letter);
            for (int shift = (digits - 1) * 4; shift >= 0; shift -= 4)
                sb.Append(HexDigits[(value >> shift) & 0xF]);
        }
    }
}
