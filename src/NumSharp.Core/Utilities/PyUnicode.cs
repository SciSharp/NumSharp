using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    ///     The two Unicode predicates CPython's number parsers read, pinned to the tables of CPython 3.12 (Unicode
    ///     15.0.0, the Python the NumPy 2.4.2 oracle runs on): <c>Py_UNICODE_ISSPACE</c> and <c>Py_UNICODE_TODECIMAL</c>,
    ///     plus <c>_PyUnicode_TransformDecimalAndSpaceToASCII</c>, the rewrite <c>int(str)</c>, <c>float(str)</c> and
    ///     <c>complex(str)</c> run before they parse.
    /// </summary>
    /// <remarks>
    ///     <b>Why pinned tables and not .NET's.</b> <see cref="System.Globalization.CharUnicodeInfo"/> follows the Unicode
    ///     version of the running .NET, and .NET 10 reads Unicode 16.0, which added decimal digits that Unicode 15.0 does not
    ///     have (Garay U+10D40, Sunuwar U+11BF0, Gurung Khema U+16130, Kirat Rai U+16D70, Outlined digits U+1CCF0, Ol Onal
    ///     U+1E5F1, Myanmar Extended-C U+116D0 / U+116DA). With .NET's tables <c>int('𐵀')</c> would parse on one runtime and
    ///     fail on another, while NumPy on CPython 3.12 always refuses it. Both predicates read
    ///     <see cref="NumSharp.Backends.Strings.PyUnicodeData"/>, the type records <c>tools/gen_unicode_tables.py</c>
    ///     generates from CPython 3.12 (the tables NumPy's string ufuncs and CPython's <c>str</c> methods read too).
    /// </remarks>
    internal static class PyUnicode
    {
        /// <summary>
        ///     <c>Py_UNICODE_TODECIMAL</c>: the decimal value of a code point that is a decimal digit in Unicode 15.0.0
        ///     (ASCII <c>0</c>..<c>9</c>, Arabic-Indic, Devanagari, fullwidth, mathematical digits, ...), else -1.
        /// </summary>
        /// <param name="cp">The code point (any value; values past U+10FFFF, negative values and lone surrogates are no
        ///     digit).</param>
        /// <returns>0..9, or -1 for a code point that is not a decimal digit.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int DecimalValue(int cp)
        {
            if ((uint)(cp - '0') <= 9)
                return cp - '0';
            // A negative value reads as a unit past U+10FFFF (record 0: no digit).
            return NumSharp.Backends.Strings.PyUnicodeData.DecimalValue((uint)cp);
        }

        /// <summary>
        ///     <c>Py_UNICODE_ISSPACE</c> (Unicode 15.0.0): the ASCII whitespace <c>\t \n \v \f \r</c> and the space, the four
        ///     C0 information separators U+001C..U+001F, NEL U+0085, the no-break space U+00A0, the Ogham space U+1680, the
        ///     spaces U+2000..U+200A, the line and paragraph separators U+2028 / U+2029, U+202F, U+205F and the ideographic
        ///     space U+3000.
        /// </summary>
        /// <param name="cp">The code point (any value; a negative one is no space).</param>
        /// <returns>True when CPython's <c>str.isspace</c> holds for the code point.</returns>
        /// <remarks>
        ///     Not .NET's <see cref="char.IsWhiteSpace(char)"/>: .NET does not count U+001C..U+001F as whitespace, CPython does
        ///     (their bidirectional class is B / S).
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsSpace(int cp) => NumSharp.Backends.Strings.PyUnicodeData.IsSpace((uint)cp);

        /// <summary>
        ///     <c>Py_ISSPACE</c>: the whitespace CPython's ASCII parsers skip after the Unicode rewrite (<c>PyLong_FromString</c>,
        ///     <c>float_from_string_inner</c>, <c>complex_from_string_inner</c>): <c>\t \n \v \f \r</c> and the space only.
        /// </summary>
        /// <param name="c">The ASCII character (a byte of the rewritten text).</param>
        /// <returns>True for the six ASCII whitespace characters.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsAsciiSpace(int c) => c == ' ' || (uint)(c - 0x09) <= 0x04;

        /// <summary>
        ///     <c>_PyUnicode_TransformDecimalAndSpaceToASCII</c> over code points: an ALL-ASCII text (every code point below
        ///     128) is returned unchanged; otherwise a code point below 127 is kept, a whitespace (<see cref="IsSpace"/>)
        ///     becomes a space, a decimal digit (<see cref="DecimalValue"/>) becomes its ASCII digit, and the first other code
        ///     point becomes <c>?</c> and ENDS the text.
        /// </summary>
        /// <param name="codePoints">The text as code points (a <c>U</c> element's units with the trailing NULs already
        ///     removed). A unit above U+10FFFF is just a code point that is neither space nor digit.</param>
        /// <param name="ascii">Receives the rewritten text; room for <c>codePoints.Length</c> bytes.</param>
        /// <returns>The rewritten length: <c>codePoints.Length</c>, or the position after the <c>?</c> when one was written.</returns>
        /// <remarks>
        ///     The two cases differ only for DEL (127): CPython returns an ASCII string as it is (<c>PyUnicode_IS_ASCII</c>),
        ///     so DEL survives there, while the per-character rule tests <c>ch &lt; 127</c> and turns DEL into <c>?</c>. That
        ///     and the truncation after <c>?</c> are observable only through which error a later step raises, never through
        ///     a value: <c>complex('1\x7f__0')</c> reports the underscores ("could not convert string to complex") while
        ///     <c>complex('é\x7f__0')</c> reports "complex() arg is a malformed string" (the text ended at the first <c>?</c>).
        /// </remarks>
        internal static int TransformDecimalAndSpaceToAscii(ReadOnlySpan<uint> codePoints, Span<byte> ascii)
        {
            // PyUnicode_IS_ASCII: the text is returned unchanged.
            bool allAscii = true;
            for (int i = 0; i < codePoints.Length; i++)
                if (codePoints[i] >= 128)
                {
                    allAscii = false;
                    break;
                }
            if (allAscii)
            {
                for (int i = 0; i < codePoints.Length; i++)
                    ascii[i] = (byte)codePoints[i];
                return codePoints.Length;
            }

            for (int i = 0; i < codePoints.Length; i++)
            {
                uint ch = codePoints[i];
                if (ch < 127)
                {
                    ascii[i] = (byte)ch;
                    continue;
                }
                // A unit past int.MaxValue cannot be a digit or a space; clamp so the int predicates stay total.
                int cp = ch > 0x7FFFFFFF ? int.MaxValue : (int)ch;
                if (IsSpace(cp))
                {
                    ascii[i] = (byte)' ';
                    continue;
                }
                int d = DecimalValue(cp);
                if (d < 0)
                {
                    ascii[i] = (byte)'?';
                    return i + 1;
                }
                ascii[i] = (byte)('0' + d);
            }
            return codePoints.Length;
        }

        /// <summary>
        ///     The same rewrite over a UTF-16 C# string: a valid surrogate pair is ONE code point (an astral digit such as
        ///     <c>U+1D7CF</c> becomes one ASCII digit), a lone surrogate is a code point of its own (no digit, no space).
        /// </summary>
        /// <param name="text">The text.</param>
        /// <param name="ascii">Receives the rewritten text; room for <c>text.Length</c> bytes (a pair writes one byte, so
        ///     the length never exceeds the unit count).</param>
        /// <returns>The rewritten length.</returns>
        internal static int TransformDecimalAndSpaceToAscii(ReadOnlySpan<char> text, Span<byte> ascii)
        {
            // PyUnicode_IS_ASCII: an all-ASCII text (DEL included) is returned unchanged.
            if (text.IndexOfAnyExceptInRange('\0', '\x7F') < 0)
            {
                for (int i = 0; i < text.Length; i++)
                    ascii[i] = (byte)text[i];
                return text.Length;
            }

            int n = 0;
            for (int i = 0; i < text.Length; i++)
            {
                int cp = text[i];
                if (cp < 127)
                {
                    ascii[n++] = (byte)cp;
                    continue;
                }
                if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
                {
                    cp = char.ConvertToUtf32(text[i], text[i + 1]);
                    i++;
                }
                if (IsSpace(cp))
                {
                    ascii[n++] = (byte)' ';
                    continue;
                }
                int d = DecimalValue(cp);
                if (d < 0)
                {
                    ascii[n++] = (byte)'?';
                    return n;
                }
                ascii[n++] = (byte)('0' + d);
            }
            return n;
        }
    }
}
