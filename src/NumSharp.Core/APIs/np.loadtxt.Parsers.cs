using System;
using System.Globalization;
using System.Numerics;
using System.Runtime.InteropServices;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        // The numeric field parsers of np.loadtxt (NumPy's per-dtype set_from_ucs4 functions, textreading/conversions.c
        // and str_to_int.c). They read one field's characters; the numeric sink (np.loadtxt.Sinks.cs) calls them for every
        // field of a numeric dtype that has no converter (a converter's value is stored by ReductionInitial.PackStore).

        #region numeric field parsers (loadtxt semantics)

        /// <summary>
        ///     Parses one field into the element slot of a numeric dtype — the dispatch NumPy makes through each field's
        ///     <c>set_from_ucs4</c> function pointer: float32 / float64 through <see cref="TryDoubleLT"/> (a float32 field is
        ///     parsed as a double and rounded once), integers range-checked, unsigned integers refusing a sign, bool through
        ///     int64 (any nonzero integer is True; <c>"True"</c> is not an integer and fails). float16 has no
        ///     <c>set_from_ucs4</c> of its own: NumPy's <c>npy_to_generic</c> packs the field's <c>str</c> with
        ///     <c>PyArray_Pack</c>, i.e. Python's <c>float()</c> (underscores between digits, any Unicode decimal digit and
        ///     space) rounded to float16 — <see cref="LoadtxtWriteGenericHalf"/>.
        /// </summary>
        /// <param name="tc">The result's storage lane.</param>
        /// <param name="tok">The field's characters (unstripped: each parser strips NumPy's whitespace itself).</param>
        /// <param name="dst">The element slot (the lane's item size, or more).</param>
        /// <returns>True when the field parsed and was written; false for any parse or range failure, which the caller
        ///     reports as NumPy's <c>could not convert string … to … at row r, column c.</c> without a cause (NumPy's
        ///     numeric parsers set no exception).</returns>
        /// <exception cref="ValueError">A float16 field <c>float()</c> refuses: the error NumPy's generic conversion raises,
        ///     which the caller chains as the cause of its <c>could not convert string</c> message.</exception>
        /// <remarks>A lane without a parser here (the string lanes) returns false: those dtypes have their own sinks.</remarks>
        private static bool LoadtxtWriteField(NPTypeCode tc, ReadOnlySpan<char> tok, Span<byte> dst)
        {
            switch (tc)
            {
                case NPTypeCode.Double: { if (!TryDoubleLT(tok, out double v)) return false; MemoryMarshal.Write(dst, in v); return true; }
                case NPTypeCode.Single: { if (!TryDoubleLT(tok, out double v)) return false; float f = (float)v; MemoryMarshal.Write(dst, in f); return true; }
                case NPTypeCode.Half: return LoadtxtWriteGenericHalf(tok, dst);
                case NPTypeCode.Decimal: { if (!TryDecimalLT(tok, out decimal v)) return false; MemoryMarshal.Write(dst, in v); return true; }
                case NPTypeCode.Complex: { if (!TryComplexLT(tok, out Complex v)) return false; MemoryMarshal.Write(dst, in v); return true; }
                case NPTypeCode.Int64: { if (!TryLongLT(tok, long.MinValue, long.MaxValue, out long v)) return false; MemoryMarshal.Write(dst, in v); return true; }
                case NPTypeCode.Int32: { if (!TryLongLT(tok, int.MinValue, int.MaxValue, out long v)) return false; int x = (int)v; MemoryMarshal.Write(dst, in x); return true; }
                case NPTypeCode.Int16: { if (!TryLongLT(tok, short.MinValue, short.MaxValue, out long v)) return false; short x = (short)v; MemoryMarshal.Write(dst, in x); return true; }
                case NPTypeCode.SByte: { if (!TryLongLT(tok, sbyte.MinValue, sbyte.MaxValue, out long v)) return false; sbyte x = (sbyte)v; MemoryMarshal.Write(dst, in x); return true; }
                case NPTypeCode.UInt64: { if (!TryULongLT(tok, ulong.MaxValue, out ulong v)) return false; MemoryMarshal.Write(dst, in v); return true; }
                case NPTypeCode.UInt32: { if (!TryULongLT(tok, uint.MaxValue, out ulong v)) return false; uint x = (uint)v; MemoryMarshal.Write(dst, in x); return true; }
                case NPTypeCode.UInt16: { if (!TryULongLT(tok, ushort.MaxValue, out ulong v)) return false; ushort x = (ushort)v; MemoryMarshal.Write(dst, in x); return true; }
                case NPTypeCode.Byte: { if (!TryULongLT(tok, byte.MaxValue, out ulong v)) return false; dst[0] = (byte)v; return true; }
                case NPTypeCode.Boolean: { if (!TryLongLT(tok, long.MinValue, long.MaxValue, out long v)) return false; dst[0] = (byte)(v != 0 ? 1 : 0); return true; }
                case NPTypeCode.Char: { if (!TryULongLT(tok, ushort.MaxValue, out ulong v)) return false; char x = (char)v; MemoryMarshal.Write(dst, in x); return true; }
                default: return false;
            }
        }

        /// <summary>
        ///     A float16 field, as NumPy's <c>npy_to_generic</c> stores it: <c>PyArray_Pack(float16, str(field))</c>, which
        ///     reaches <c>HALF_setitem</c> and Python's <c>float()</c> of the text (<see cref="ReductionInitial.PackStore"/>
        ///     carries that grammar), then <c>npy_double_to_half</c>.
        /// </summary>
        /// <param name="tok">The field's characters, unstripped (<c>float()</c> strips its own whitespace).</param>
        /// <param name="dst">The element slot (at least 2 bytes).</param>
        /// <returns>True (a failure raises).</returns>
        /// <exception cref="ValueError"><c>could not convert string to float: '…'</c>, which the reader chains as the cause of
        ///     its own message.</exception>
        private static bool LoadtxtWriteGenericHalf(ReadOnlySpan<char> tok, Span<byte> dst)
        {
            // Fast path, exact for the text it accepts: an ASCII field (below 127) without '_' is a str that float() reads
            // as PyOS_string_to_double of the field stripped of Py_ISSPACE (the six ASCII whitespace characters), or as
            // one of the inf / infinity / nan words. Anything else — underscores, Unicode digits or spaces, junk — takes
            // the full float() path, which also raises float()'s own error.
            if (TryPythonFloatAscii(tok, out double fast))
            {
                Half h = (Half)fast;
                MemoryMarshal.Write(dst, in h);
                return true;
            }
            Span<byte> slot = stackalloc byte[ReductionInitial.SlotBytes];
            ReductionInitial.PackStore(tok.ToString(), NPTypeCode.Half, slot);
            slot.Slice(0, 2).CopyTo(dst);
            return true;
        }

        /// <summary>
        ///     Python's <c>float(s)</c> for an ASCII <c>str</c> without underscores: the text stripped of <c>Py_ISSPACE</c>
        ///     (space, <c>\t</c>, <c>\n</c>, <c>\v</c>, <c>\f</c>, <c>\r</c> — not the C0 separators, which only a Unicode
        ///     <c>str</c> transform would map), then one decimal number or <c>inf</c> / <c>infinity</c> / <c>nan</c> with an
        ///     optional sign, case-insensitive. NaN keeps the sign (<c>float('-nan')</c> is <c>-fabs(Py_NAN)</c>).
        /// </summary>
        /// <param name="tok">The text.</param>
        /// <param name="value">The value, when accepted.</param>
        /// <returns>False when the text is not ASCII, holds an underscore, or is not one number: the caller then runs the
        ///     full <c>float()</c> (which accepts what this refuses or raises the error).</returns>
        private static bool TryPythonFloatAscii(ReadOnlySpan<char> tok, out double value)
        {
            value = 0;
            int start = 0, end = tok.Length;
            for (int i = 0; i < end; i++)
                if (tok[i] >= 127 || tok[i] == '_')
                    return false;
            while (start < end && IsPySpace(tok[start])) start++;
            while (end > start && IsPySpace(tok[end - 1])) end--;
            var s = tok.Slice(start, end - start);
            if (NDDoubleParse.TryParseWhole(s, out value))
                return true;
            return TryInfNanWord(s, out value);
        }

        /// <summary>CPython's <c>Py_ISSPACE</c> (and the C locale's <c>isspace</c>): the six ASCII whitespace characters.</summary>
        /// <param name="c">The character.</param>
        /// <returns>True for space, <c>\t</c>, <c>\n</c>, <c>\v</c>, <c>\f</c>, <c>\r</c>.</returns>
        private static bool IsPySpace(char c) => c == ' ' || (uint)(c - '\t') <= '\r' - '\t';

        /// <summary>
        ///     CPython's <c>_Py_parse_inf_or_nan</c> over the WHOLE text: an optional sign, then <c>inf</c>,
        ///     <c>infinity</c> or <c>nan</c> in any case. NaN takes the sign (<c>-fabs(Py_NAN)</c>); the unsigned NaN is the
        ///     positive quiet NaN.
        /// </summary>
        /// <param name="s">The stripped text.</param>
        /// <param name="value">The value, when it matched.</param>
        /// <returns>True when the text is exactly one of the words.</returns>
        private static bool TryInfNanWord(ReadOnlySpan<char> s, out double value)
        {
            value = 0;
            if (s.Length == 0)
                return false;
            ReadOnlySpan<char> body = s;
            bool negative = false;
            if (body[0] == '+') body = body.Slice(1);
            else if (body[0] == '-') { negative = true; body = body.Slice(1); }
            if (EqualsAsciiIgnoreCase(body, "inf") || EqualsAsciiIgnoreCase(body, "infinity"))
            {
                value = negative ? double.NegativeInfinity : double.PositiveInfinity;
                return true;
            }
            if (EqualsAsciiIgnoreCase(body, "nan"))
            {
                value = negative ? double.NaN : LoadtxtPositiveNaN;
                return true;
            }
            return false;
        }

        /// <summary>
        ///     CPython's <c>case_insensitive_match</c>: the text is exactly <paramref name="lowerWord"/>, each letter in either
        ///     ASCII case. Unlike <see cref="StringComparison.OrdinalIgnoreCase"/>, a non-ASCII letter whose upper case is ASCII
        ///     (the dotless <c>ı</c> of <c>"ınf"</c>) does not match: NumPy hands the parser ASCII only.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="lowerWord">The lower-case ASCII word.</param>
        /// <returns>True on a match.</returns>
        private static bool EqualsAsciiIgnoreCase(ReadOnlySpan<char> s, string lowerWord)
        {
            if (s.Length != lowerWord.Length)
                return false;
            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];
                if (c >= 128 || (c | 0x20) != lowerWord[i])
                    return false;
            }
            return true;
        }

        /// <summary>
        ///     The NaN NumPy's C parser produces for <c>"nan"</c>: the POSITIVE quiet NaN <c>0x7FF8…</c> (.NET's
        ///     <see cref="double.NaN"/> is the negative one, <c>0xFFF8…</c>, so a field must not parse to it). See np.fromfile.cs.
        /// </summary>
        private static readonly double LoadtxtPositiveNaN = BitConverter.Int64BitsToDouble(0x7FF8000000000000L);

        /// <summary>
        ///     NumPy's tokenizer and field whitespace (<c>Py_UNICODE_ISSPACE</c>): <see cref="char.IsWhiteSpace(char)"/> across
        ///     the whole BMP, PLUS the four C0 information separators U+001C..U+001F (verified to be the only difference), so
        ///     <c>"1\x1c2"</c> is two fields in whitespace mode and a leading / trailing separator is stripped from a field.
        /// </summary>
        /// <param name="c">The character.</param>
        /// <returns>True for NumPy's whitespace.</returns>
        private static bool IsLoadtxtWhitespace(char c)
            // ASCII first: a printable ASCII character (most field text) is past ' ' and below U+0085 (NEL), the first
            // non-ASCII whitespace, so the Unicode table is consulted only for the rare text beyond it.
            => c <= ' ' ? c == ' ' || (uint)(c - '\t') <= '\r' - '\t' || (uint)(c - 0x1c) <= 3
                        : c >= 0x85 && char.IsWhiteSpace(c);

        /// <summary>
        ///     Strips NumPy's whitespace (<see cref="IsLoadtxtWhitespace"/>) from both ends of a field — the strip
        ///     <c>double_from_ucs4</c> / <c>str_to_int64</c> apply before parsing (a superset of
        ///     <see cref="MemoryExtensions.Trim(ReadOnlySpan{char})"/>'s set).
        /// </summary>
        /// <param name="s">The field.</param>
        /// <returns>The stripped slice (no copy).</returns>
        private static ReadOnlySpan<char> TrimLoadtxtWs(ReadOnlySpan<char> s)
        {
            int start = 0, end = s.Length;
            while (start < end && IsLoadtxtWhitespace(s[start])) start++;
            while (end > start && IsLoadtxtWhitespace(s[end - 1])) end--;
            return s.Slice(start, end - start);
        }

        /// <summary>
        ///     Parses a float field as NumPy's <c>double_from_ucs4</c> does (<c>PyOS_string_to_double</c>): the stripped field
        ///     must be ONE float (hex, <c>_</c> and trailing characters are refused), with case-insensitive, optionally
        ///     signed <c>inf</c> / <c>infinity</c> / <c>nan</c>; <c>"nan"</c> is the positive quiet NaN, <c>"-nan"</c> the
        ///     negative one.
        /// </summary>
        /// <param name="tok">The field.</param>
        /// <param name="value">The value, when the field parsed.</param>
        /// <returns>True when the whole field is one float.</returns>
        private static bool TryDoubleLT(ReadOnlySpan<char> tok, out double value)
        {
            ReadOnlySpan<char> s = TrimLoadtxtWs(tok);
            // The whole stripped field must be one number (NumPy's p_end == end): hex, '_' and trailing text are refused.
            // The decimal grammar is _Py_dg_strtod's; the inf / nan words are _Py_parse_inf_or_nan's (no "nan(...)").
            if (NDDoubleParse.TryParseWhole(s, out value))
                return true;
            return TryInfNanWord(s, out value);
        }

        /// <summary>
        ///     NumPy's <c>str_to_int64</c> (textreading/str_to_int.h), range-checked against the target dtype while the digits
        ///     accumulate (<c>int8 "200"</c> fails, it does not wrap): <c>Py_UNICODE_ISSPACE</c> whitespace, one optional sign,
        ///     at least one ASCII digit, then only whitespace.
        /// </summary>
        /// <param name="tok">The field.</param>
        /// <param name="min">The dtype's smallest value.</param>
        /// <param name="max">The dtype's largest value.</param>
        /// <param name="value">The value (0 on failure).</param>
        /// <returns>True when the field is an integer within [<paramref name="min"/>, <paramref name="max"/>].</returns>
        private static bool TryLongLT(ReadOnlySpan<char> tok, long min, long max, out long value)
        {
            value = 0;
            int p = 0, end = tok.Length;
            while (p < end && IsLoadtxtWhitespace(tok[p])) p++;
            bool negative = false;
            if (p < end && tok[p] == '-') { negative = true; p++; }
            else if (p < end && tok[p] == '+') p++;
            if (p >= end || (uint)(tok[p] - '0') > 9)
                return false;
            long number = 0;
            if (negative)
            {
                // accumulate downwards so int_min itself is reachable
                long preMin = min / 10;
                int digPreMin = (int)-(min % 10);
                while (p < end && (uint)(tok[p] - '0') <= 9)
                {
                    int d = tok[p] - '0';
                    if (number > preMin || (number == preMin && d <= digPreMin))
                    {
                        number = number * 10 - d;
                        p++;
                    }
                    else
                        return false;
                }
            }
            else
            {
                long preMax = max / 10;
                int digPreMax = (int)(max % 10);
                while (p < end && (uint)(tok[p] - '0') <= 9)
                {
                    int d = tok[p] - '0';
                    if (number < preMax || (number == preMax && d <= digPreMax))
                    {
                        number = number * 10 + d;
                        p++;
                    }
                    else
                        return false;
                }
            }
            while (p < end && IsLoadtxtWhitespace(tok[p])) p++;
            if (p != end)
                return false;
            value = number;
            return true;
        }

        /// <summary>
        ///     NumPy's <c>str_to_uint64</c>: as <see cref="TryLongLT"/>, but a minus sign is refused outright (<c>uint8 "-1"</c>
        ///     fails) and a plus sign is allowed.
        /// </summary>
        /// <param name="tok">The field.</param>
        /// <param name="max">The dtype's largest value.</param>
        /// <param name="value">The value (0 on failure).</param>
        /// <returns>True when the field is an unsigned integer not above <paramref name="max"/>.</returns>
        private static bool TryULongLT(ReadOnlySpan<char> tok, ulong max, out ulong value)
        {
            value = 0;
            int p = 0, end = tok.Length;
            while (p < end && IsLoadtxtWhitespace(tok[p])) p++;
            if (p < end && tok[p] == '-')
                return false;
            if (p < end && tok[p] == '+') p++;
            if (p >= end || (uint)(tok[p] - '0') > 9)
                return false;
            ulong number = 0;
            ulong preMax = max / 10;
            int digPreMax = (int)(max % 10);
            while (p < end && (uint)(tok[p] - '0') <= 9)
            {
                int d = tok[p] - '0';
                if (number < preMax || (number == preMax && d <= digPreMax))
                {
                    number = number * 10 + (uint)d;
                    p++;
                }
                else
                    return false;
            }
            while (p < end && IsLoadtxtWhitespace(tok[p])) p++;
            if (p != end)
                return false;
            value = number;
            return true;
        }

        /// <summary>
        ///     Parses a decimal field (NumSharp's Decimal dtype has no NumPy analog): the stripped field in invariant float
        ///     syntax, exactly.
        /// </summary>
        /// <param name="tok">The field.</param>
        /// <param name="value">The value, when the field parsed.</param>
        /// <returns>True when the stripped field is one decimal number.</returns>
        private static bool TryDecimalLT(ReadOnlySpan<char> tok, out decimal value)
            => decimal.TryParse(TrimLoadtxtWs(tok), NumberStyles.Float, CI, out value);

        /// <summary>
        ///     Parses a complex field as NumPy's <c>to_complex_int</c> (conversions.c) does with its default
        ///     <c>imaginary_unit='j'</c>: <c>a</c>, <c>aj</c>, <c>a+bj</c> / <c>a-bj</c> (<c>1+-2j</c> reads the <c>-</c> as
        ///     the imaginary sign), optionally parenthesized; a capital <c>J</c> is not the unit, so <c>"1+2J"</c> fails.
        /// </summary>
        /// <param name="tok">The field.</param>
        /// <param name="value">The value (default on failure).</param>
        /// <returns>True when the whole stripped field is one complex number.</returns>
        private static bool TryComplexLT(ReadOnlySpan<char> tok, out Complex value)
        {
            value = default;
            ReadOnlySpan<char> s = TrimLoadtxtWs(tok);
            int i = 0, end = s.Length;
            bool paren = false;
            if (i < end && s[i] == '(') { paren = true; i++; while (i < end && IsLoadtxtWhitespace(s[i])) i++; }

            if (!ScanDouble(s, i, end, out double real, out int p)) return false;
            double imag;
            if (p == end)
            {
                if (paren) return false; // "(1" — unmatched paren
                value = new Complex(real, 0.0);
                return true;
            }

            if (s[p] == 'j')
            {
                imag = real; real = 0.0; p++;
            }
            else if (s[p] == '+' || s[p] == '-')
            {
                if (s[p] == '+') p++; // advance so "1+-2j" reads the '-' as the imaginary sign
                if (!ScanDouble(s, p, end, out imag, out p)) return false;
                if (p >= end || s[p] != 'j') return false;
                p++;
            }
            else imag = 0.0;

            if (paren)
            {
                while (p < end && IsLoadtxtWhitespace(s[p])) p++;
                if (p < end && s[p] == ')') p++;
                else return false;
            }
            while (p < end && IsLoadtxtWhitespace(s[p])) p++;
            if (p != end) return false;

            value = new Complex(real, imag);
            return true;
        }

        /// <summary>
        ///     Scans the longest float starting at <paramref name="i"/> (only to split a complex field into its components,
        ///     the way <c>PyOS_string_to_double</c> stops at the first character that ends a float).
        /// </summary>
        /// <param name="s">The stripped field.</param>
        /// <param name="i">Where the float starts.</param>
        /// <param name="end">The field's end.</param>
        /// <param name="value">The value scanned (the positive quiet NaN for an unsigned <c>nan</c>).</param>
        /// <param name="consumed">The index just past the float (<paramref name="i"/> when none).</param>
        /// <returns>True when a float starts at <paramref name="i"/>: digits (with an optional fraction and exponent), or
        ///     <c>inf</c> / <c>infinity</c> / <c>nan</c>, each optionally signed.</returns>
        private static bool ScanDouble(ReadOnlySpan<char> s, int i, int end, out double value, out int consumed)
        {
            value = 0; consumed = i;
            int begin = i;
            if (i < end && (s[i] == '+' || s[i] == '-')) i++;

            if (MatchWord(s, i, end, "infinity")) { value = s[begin] == '-' ? double.NegativeInfinity : double.PositiveInfinity; consumed = i + 8; return true; }
            if (MatchWord(s, i, end, "inf")) { value = s[begin] == '-' ? double.NegativeInfinity : double.PositiveInfinity; consumed = i + 3; return true; }
            if (MatchWord(s, i, end, "nan")) { value = s[begin] == '-' ? double.NaN : LoadtxtPositiveNaN; consumed = i + 3; return true; }

            bool anyDigit = false;
            while (i < end && s[i] >= '0' && s[i] <= '9') { i++; anyDigit = true; }
            if (i < end && s[i] == '.') { i++; while (i < end && s[i] >= '0' && s[i] <= '9') { i++; anyDigit = true; } }
            if (!anyDigit) return false;

            // An exponent counts only with at least one digit after its optional sign ("1e" stops before the 'e').
            if (i < end && (s[i] == 'e' || s[i] == 'E'))
            {
                int j = i + 1;
                if (j < end && (s[j] == '+' || s[j] == '-')) j++;
                if (j < end && s[j] >= '0' && s[j] <= '9') { j++; while (j < end && s[j] >= '0' && s[j] <= '9') j++; i = j; }
            }

            consumed = i;
            value = NDDoubleParse.Parse(s.Slice(begin, i - begin));
            return true;
        }

        /// <summary>Whether a lower-case word starts at <paramref name="i"/>, case-insensitively.</summary>
        /// <param name="s">The text.</param>
        /// <param name="i">The position.</param>
        /// <param name="end">The text's end.</param>
        /// <param name="word">The word, lower case.</param>
        /// <returns>True when the word fits before <paramref name="end"/> and matches.</returns>
        private static bool MatchWord(ReadOnlySpan<char> s, int i, int end, string word)
        {
            if (i + word.Length > end) return false;
            for (int k = 0; k < word.Length; k++)
                // ASCII letters only: NumPy hands its parser ASCII text, so a dotless 'ı' (upper case 'I') is no 'i'
                if (s[i + k] >= 128 || (s[i + k] | 0x20) != word[k]) return false;
            return true;
        }

        #endregion
    }
}
