using System;
using System.Globalization;
using System.Numerics;
using System.Runtime.InteropServices;
using NumSharp.Backends;

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
        ///     <c>set_from_ucs4</c> function pointer: floats through <see cref="TryDoubleLT"/> (then narrowed: a float32 /
        ///     float16 field is parsed as a double and rounded once), integers range-checked, unsigned integers refusing a
        ///     sign, bool through int64 (any nonzero integer is True; <c>"True"</c> is not an integer and fails).
        /// </summary>
        /// <param name="tc">The result's storage lane.</param>
        /// <param name="tok">The field's characters (unstripped: each parser strips NumPy's whitespace itself).</param>
        /// <param name="dst">The element slot (the lane's item size, or more).</param>
        /// <returns>True when the field parsed and was written; false for any parse or range failure, which the caller
        ///     reports as NumPy's <c>could not convert string … to … at row r, column c.</c> without a cause (NumPy's
        ///     numeric parsers set no exception).</returns>
        /// <remarks>A lane without a parser here (the string lanes) returns false: those dtypes have their own sinks.</remarks>
        private static bool LoadtxtWriteField(NPTypeCode tc, ReadOnlySpan<char> tok, Span<byte> dst)
        {
            switch (tc)
            {
                case NPTypeCode.Double: { if (!TryDoubleLT(tok, out double v)) return false; MemoryMarshal.Write(dst, in v); return true; }
                case NPTypeCode.Single: { if (!TryDoubleLT(tok, out double v)) return false; float f = (float)v; MemoryMarshal.Write(dst, in f); return true; }
                case NPTypeCode.Half: { if (!TryDoubleLT(tok, out double v)) return false; Half h = (Half)v; MemoryMarshal.Write(dst, in h); return true; }
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
        private static bool IsLoadtxtWhitespace(char c) => char.IsWhiteSpace(c) || (c >= '\u001c' && c <= '\u001f');

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
            if (s.Length > 0)
            {
                // inf / infinity / nan (case-insensitive, optional sign) — double.TryParse rejects these.
                ReadOnlySpan<char> body = s;
                int sign = 1;
                if (body[0] == '+') body = body.Slice(1);
                else if (body[0] == '-') { sign = -1; body = body.Slice(1); }
                if (body.Equals("inf", StringComparison.OrdinalIgnoreCase) || body.Equals("infinity", StringComparison.OrdinalIgnoreCase))
                { value = sign < 0 ? double.NegativeInfinity : double.PositiveInfinity; return true; }
                if (body.Equals("nan", StringComparison.OrdinalIgnoreCase))
                { value = sign < 0 ? double.NaN : LoadtxtPositiveNaN; return true; }
            }
            // Requires the whole token to be a valid float (NumPy's p_end == end): rejects hex / "_" / junk.
            return double.TryParse(s, NumberStyles.Float, CI, out value);
        }

        /// <summary>
        ///     Parses a signed integer field as NumPy's <c>str_to_int64</c> does, range-checked against the target dtype
        ///     (<c>int8 "200"</c> fails, it does not wrap).
        /// </summary>
        /// <param name="tok">The field.</param>
        /// <param name="min">The dtype's smallest value.</param>
        /// <param name="max">The dtype's largest value.</param>
        /// <param name="value">The value (0 on failure).</param>
        /// <returns>True when the stripped field is an integer within [<paramref name="min"/>, <paramref name="max"/>].</returns>
        private static bool TryLongLT(ReadOnlySpan<char> tok, long min, long max, out long value)
        {
            if (long.TryParse(TrimLoadtxtWs(tok), NumberStyles.Integer, CI, out value) && value >= min && value <= max)
                return true;
            value = 0;
            return false;
        }

        /// <summary>
        ///     Parses an unsigned integer field as NumPy's <c>str_to_uint64</c> does: a negative sign is refused outright
        ///     (<c>uint8 "-1"</c> fails), and the value is range-checked against the target dtype.
        /// </summary>
        /// <param name="tok">The field.</param>
        /// <param name="max">The dtype's largest value.</param>
        /// <param name="value">The value (0 on failure).</param>
        /// <returns>True when the stripped field is an unsigned integer not above <paramref name="max"/>.</returns>
        private static bool TryULongLT(ReadOnlySpan<char> tok, ulong max, out ulong value)
        {
            ReadOnlySpan<char> s = TrimLoadtxtWs(tok);
            value = 0;
            if (s.Length > 0 && s[0] == '-') return false; // unsigned rejects a negative sign outright
            return ulong.TryParse(s, NumberStyles.Integer, CI, out value) && value <= max;
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
            value = double.Parse(s.Slice(begin, i - begin), NumberStyles.Float, CI);
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
                if (char.ToLowerInvariant(s[i + k]) != word[k]) return false;
            return true;
        }

        #endregion
    }
}
