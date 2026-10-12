using System;
using System.Runtime.CompilerServices;
using System.Text;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The text codec of the fixed-width string dtypes — the C# side of NumPy's <c>STRING_setitem</c> /
    ///     <c>UNICODE_setitem</c> / <c>STRING_getitem</c> / <c>UNICODE_getitem</c> (<c>arraytypes.c.src</c>): how ONE element
    ///     value crosses between a C# <see cref="string"/> / <c>byte[]</c> and the NUL-padded bytes of an <c>S</c> element
    ///     (one byte per character) or a <c>U</c> element (one native-endian UCS-4 <c>uint32</c> per character).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Python string semantics over UTF-16.</b> A Python <c>str</c> is a sequence of CODE POINTS; a .NET
    ///     <see cref="string"/> is UTF-16. The mapping used everywhere here: a valid surrogate pair is ONE code point (so
    ///     <c>"😀"</c> is <c>U1</c>, as in NumPy), a lone surrogate is one code point of its own value (Python can hold a lone
    ///     surrogate too: <c>np.array(['\ud800'])</c> stores <c>0xD800</c>). The one representation gap: a <c>U</c> element
    ///     holding the two code points <c>0xD83D 0xDE00</c> (only reachable through a raw byte view) reads back as the .NET
    ///     string <c>"😀"</c>, which .NET then treats as one surrogate pair. Copies and casts between string arrays
    ///     never decode, so they are unaffected.
    ///     </para>
    ///     <para>
    ///     <b>Error texts are CPython's</b>, byte for byte: <c>'ascii' codec can't encode character '\xe9' in position 1:
    ///     ordinal not in range(128)</c> (a run of offending characters reads <c>characters in position 1-2</c>) and
    ///     <c>'ascii' codec can't decode byte 0xe9 in position 1: ordinal not in range(128)</c>; a stored unit above
    ///     <c>U+10FFFF</c> is <c>SystemError: invalid maximum character passed to PyUnicode_New</c>.
    ///     </para>
    ///     <para>
    ///     <b>Order of checks follows NumPy.</b> Storing a <see cref="string"/> into <c>S</c> encodes the WHOLE string before
    ///     truncating (an unencodable character past the kept width still raises); storing <c>byte[]</c> into <c>U</c>
    ///     decodes the whole value before truncating; storing a <see cref="string"/> into <c>U</c> truncates by code points
    ///     and cannot fail.
    ///     </para>
    /// </remarks>
    internal static unsafe class NDStringCodec
    {
        /// <summary>The largest Unicode code point; a UCS-4 unit above it is not a character.</summary>
        public const uint MaxCodePoint = 0x10FFFF;

        /// <summary>CPython's <c>PyUnicode_New</c> text for a UCS-4 unit above <see cref="MaxCodePoint"/>.</summary>
        public const string InvalidMaxCharMessage = "invalid maximum character passed to PyUnicode_New";

        // ---------------------------------------------------------------------------------------------
        // Lengths
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Python's <c>len(str)</c> of a .NET string: UTF-16 units minus valid surrogate pairs (each pair is one code
        ///     point, a lone surrogate is one). Vectorised for the common case — a string without any surrogate is its own
        ///     length after one SIMD scan.
        /// </summary>
        /// <param name="s">The value (not null).</param>
        /// <returns>The number of code points.</returns>
        public static int CodePointCount(string s) => CodePointCount(s.AsSpan());

        /// <summary>Python's <c>len(str)</c> of a UTF-16 span — see <see cref="CodePointCount(string)"/>.</summary>
        /// <param name="s">The value.</param>
        /// <returns>The number of code points.</returns>
        public static int CodePointCount(ReadOnlySpan<char> s)
        {
            int first = s.IndexOfAnyInRange('\uD800', '\uDFFF');
            if (first < 0)
                return s.Length;
            int pairs = 0;
            for (int i = first; i < s.Length - 1; i++)
            {
                if (char.IsHighSurrogate(s[i]) && char.IsLowSurrogate(s[i + 1]))
                {
                    pairs++;
                    i++; // the low half belongs to this code point
                }
            }
            return s.Length - pairs;
        }

        // ---------------------------------------------------------------------------------------------
        // Writing one element
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>UNICODE_setitem</c> for a <see cref="string"/>: writes the first <paramref name="chars"/> code points of
        ///     <paramref name="value"/> as UCS-4 units and zero-fills the rest of the element (truncation counts code points,
        ///     so a surrogate pair is never split). Cannot fail.
        /// </summary>
        /// <param name="value">The value to store.</param>
        /// <param name="dst">The element's first unit (native byte order).</param>
        /// <param name="chars">The element's width in characters (<c>itemsize / 4</c>).</param>
        public static void WriteUcs4(ReadOnlySpan<char> value, uint* dst, int chars)
        {
            int written = 0;
            if (value.IndexOfAnyInRange('\uD800', '\uDFFF') < 0)
            {
                // No surrogates: one unit per char, a plain widening copy.
                written = Math.Min(value.Length, chars);
                WidenUtf16(value.Slice(0, written), dst);
            }
            else
            {
                int i = 0;
                while (i < value.Length && written < chars)
                {
                    char c = value[i];
                    if (char.IsHighSurrogate(c) && i + 1 < value.Length && char.IsLowSurrogate(value[i + 1]))
                    {
                        dst[written++] = (uint)char.ConvertToUtf32(c, value[i + 1]);
                        i += 2;
                    }
                    else
                    {
                        dst[written++] = c; // BMP character or a lone surrogate (a code point of its own in Python)
                        i++;
                    }
                }
            }
            if (written < chars)
                new Span<uint>(dst + written, chars - written).Clear();
        }

        /// <summary>
        ///     <c>STRING_setitem</c> for a <see cref="string"/>: ASCII-encodes the WHOLE value (raising CPython's
        ///     <see cref="UnicodeEncodeError"/> even when the offending character lies past the kept width), then copies the
        ///     first <paramref name="length"/> bytes and zero-fills the rest.
        /// </summary>
        /// <param name="value">The value to store.</param>
        /// <param name="dst">The element's first byte.</param>
        /// <param name="length">The element's width in bytes (<c>itemsize</c>).</param>
        /// <exception cref="UnicodeEncodeError">The value holds a code point at or above 128.</exception>
        public static void WriteAscii(ReadOnlySpan<char> value, byte* dst, int length)
        {
            ThrowIfNotAscii(value);
            int written = Math.Min(value.Length, length);
            // Validated ASCII: the vectorised narrowing cannot meet a non-ASCII char.
            Ascii.FromUtf16(value.Slice(0, written), new Span<byte>(dst, written), out _);
            if (written < length)
                new Span<byte>(dst + written, length - written).Clear();
        }

        /// <summary>
        ///     <c>STRING_setitem</c> for bytes: copies the first <paramref name="length"/> bytes of <paramref name="value"/>
        ///     verbatim (any byte value, NULs included) and zero-fills the rest. Cannot fail.
        /// </summary>
        /// <param name="value">The value to store.</param>
        /// <param name="dst">The element's first byte.</param>
        /// <param name="length">The element's width in bytes.</param>
        public static void WriteBytes(ReadOnlySpan<byte> value, byte* dst, int length)
        {
            int written = Math.Min(value.Length, length);
            value.Slice(0, written).CopyTo(new Span<byte>(dst, written));
            if (written < length)
                new Span<byte>(dst + written, length - written).Clear();
        }

        /// <summary>
        ///     <c>UNICODE_setitem</c> for bytes: ASCII-decodes the WHOLE value (raising CPython's
        ///     <see cref="UnicodeDecodeError"/> at the first byte at or above <c>0x80</c>, wherever it lies), then widens the
        ///     first <paramref name="chars"/> bytes to UCS-4 units and zero-fills the rest.
        /// </summary>
        /// <param name="value">The value to store.</param>
        /// <param name="dst">The element's first unit.</param>
        /// <param name="chars">The element's width in characters.</param>
        /// <exception cref="UnicodeDecodeError">A byte is not ASCII.</exception>
        public static void WriteUcs4FromAscii(ReadOnlySpan<byte> value, uint* dst, int chars)
        {
            ThrowIfNotAscii(value);
            int written = Math.Min(value.Length, chars);
            for (int i = 0; i < written; i++)
                dst[i] = value[i];
            if (written < chars)
                new Span<uint>(dst + written, chars - written).Clear();
        }

        // ---------------------------------------------------------------------------------------------
        // Reading one element
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>UNICODE_getitem</c>: the element as a .NET string, trailing NUL units removed (NUL units inside the value are
        ///     kept, as in NumPy). Code points above <c>U+FFFF</c> become surrogate pairs; a surrogate unit becomes a lone
        ///     surrogate char.
        /// </summary>
        /// <param name="src">The element's first unit.</param>
        /// <param name="chars">The element's width in characters.</param>
        /// <returns>The value (possibly empty).</returns>
        /// <exception cref="SystemError">A unit is above <c>U+10FFFF</c> (<c>invalid maximum character passed to PyUnicode_New</c>).</exception>
        public static string ReadUcs4(uint* src, int chars)
        {
            int len = chars;
            /* trim trailing zeros */
            while (len > 0 && src[len - 1] == 0)
                len--;
            if (len == 0)
                return string.Empty;

            int astral = 0;
            for (int i = 0; i < len; i++)
            {
                uint u = src[i];
                if (u > 0xFFFF)
                {
                    if (u > MaxCodePoint)
                        throw new SystemError(InvalidMaxCharMessage);
                    astral++;
                }
            }

            if (astral == 0)
            {
                // Every unit is one UTF-16 char: a narrowing copy.
                return string.Create(len, (IntPtr)src, static (span, state) =>
                {
                    uint* p = (uint*)state;
                    for (int i = 0; i < span.Length; i++)
                        span[i] = (char)p[i];
                });
            }

            var sb = new StringBuilder(len + astral);
            for (int i = 0; i < len; i++)
            {
                uint u = src[i];
                if (u > 0xFFFF)
                    sb.Append(char.ConvertFromUtf32((int)u));
                else
                    sb.Append((char)u);
            }
            return sb.ToString();
        }

        /// <summary>
        ///     <c>STRING_getitem</c>: the element's bytes with trailing NUL bytes removed (NUL bytes inside the value are
        ///     kept, as in NumPy).
        /// </summary>
        /// <param name="src">The element's first byte.</param>
        /// <param name="length">The element's width in bytes.</param>
        /// <returns>A fresh array (possibly empty).</returns>
        public static byte[] ReadBytes(byte* src, int length)
        {
            int len = length;
            while (len > 0 && src[len - 1] == 0)
                len--;
            return len == 0 ? Array.Empty<byte>() : new ReadOnlySpan<byte>(src, len).ToArray();
        }

        // ---------------------------------------------------------------------------------------------
        // ASCII checks with CPython's error texts
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>PyUnicode_AsASCIIString</c>'s strict check over a whole value: returns when every code point is below 128,
        ///     else raises CPython's <see cref="UnicodeEncodeError"/> for the FIRST run of consecutive unencodable code points
        ///     (positions count code points).
        /// </summary>
        /// <param name="value">The value being encoded.</param>
        /// <exception cref="UnicodeEncodeError">The value holds a code point at or above 128.</exception>
        public static void ThrowIfNotAscii(ReadOnlySpan<char> value)
        {
            int first = value.IndexOfAnyExceptInRange('\0', '\x7F');
            if (first < 0)
                return;
            throw EncodeError(value, first);
        }

        /// <summary>
        ///     The same check over a <c>U</c> element's units (the trailing NUL units already excluded by
        ///     <paramref name="chars"/> or not — NUL is ASCII either way): the unit above <c>U+10FFFF</c> check comes first,
        ///     as NumPy reads the element into a Python string (<c>UNICODE_getitem</c>) before encoding it.
        /// </summary>
        /// <param name="src">The element's first unit.</param>
        /// <param name="chars">The element's width in characters.</param>
        /// <exception cref="SystemError">A unit is above <c>U+10FFFF</c>.</exception>
        /// <exception cref="UnicodeEncodeError">A unit is a code point at or above 128.</exception>
        public static void ThrowIfNotAscii(uint* src, int chars)
        {
            int first = -1;
            for (int i = 0; i < chars; i++)
            {
                if (src[i] > 0x7F)
                {
                    first = i;
                    break;
                }
            }
            if (first < 0)
                return;

            int len = chars;
            while (len > 0 && src[len - 1] == 0)
                len--;
            for (int i = first; i < len; i++)
                if (src[i] > MaxCodePoint)
                    throw new SystemError(InvalidMaxCharMessage);
            // Units ARE code points, so positions are unit indices; the run is the consecutive units >= 128.
            int end = first;
            while (end < len && src[end] > 0x7F)
                end++;
            throw EncodeErrorFor(src[first], first, end);
        }

        /// <summary>
        ///     <c>PyUnicode_FromEncodedObject(bytes, "ASCII", "strict")</c>'s check: returns when every byte is below
        ///     <c>0x80</c>, else raises CPython's <see cref="UnicodeDecodeError"/> naming the FIRST offending byte.
        /// </summary>
        /// <param name="value">The bytes being decoded.</param>
        /// <exception cref="UnicodeDecodeError">A byte is at or above <c>0x80</c>.</exception>
        public static void ThrowIfNotAscii(ReadOnlySpan<byte> value)
        {
            int first = value.IndexOfAnyExceptInRange((byte)0, (byte)0x7F);
            if (first < 0)
                return;
            throw DecodeError(value[first], first);
        }

        /// <summary>CPython's <c>UnicodeDecodeError</c> text for the ASCII codec at <paramref name="position"/>.</summary>
        /// <param name="value">The offending byte.</param>
        /// <param name="position">Its index in the value being decoded.</param>
        /// <returns>The exception (not thrown).</returns>
        public static UnicodeDecodeError DecodeError(byte value, long position)
            => new UnicodeDecodeError($"'ascii' codec can't decode byte 0x{value:x2} in position {position}: ordinal not in range(128)");

        /// <summary>
        ///     Builds CPython's <c>UnicodeEncodeError</c> for the run of unencodable code points starting at the UTF-16 index
        ///     <paramref name="firstIndex"/> (every code point before it is ASCII, so the UTF-16 index IS the code point
        ///     position there; inside the run a surrogate pair counts once).
        /// </summary>
        /// <param name="value">The value being encoded.</param>
        /// <param name="firstIndex">UTF-16 index of the first non-ASCII unit.</param>
        /// <returns>The exception (not thrown).</returns>
        private static UnicodeEncodeError EncodeError(ReadOnlySpan<char> value, int firstIndex)
        {
            int position = firstIndex; // all earlier chars are ASCII → one code point each
            int i = firstIndex;
            int endPosition = position;
            uint firstCodePoint = 0;
            while (i < value.Length && value[i] > '\x7F')
            {
                uint cp;
                if (char.IsHighSurrogate(value[i]) && i + 1 < value.Length && char.IsLowSurrogate(value[i + 1]))
                {
                    cp = (uint)char.ConvertToUtf32(value[i], value[i + 1]);
                    i += 2;
                }
                else
                {
                    cp = value[i];
                    i++;
                }
                if (endPosition == position)
                    firstCodePoint = cp;
                endPosition++;
            }
            return EncodeErrorFor(firstCodePoint, position, endPosition);
        }

        /// <summary>
        ///     CPython's <c>UnicodeEncodeError.__str__</c>: one bad character prints as <c>\xNN</c> / <c>\uNNNN</c> /
        ///     <c>\UNNNNNNNN</c> (lowercase hex); a longer run prints its position range.
        /// </summary>
        /// <param name="firstCodePoint">The first unencodable code point.</param>
        /// <param name="start">Code-point position of the run start.</param>
        /// <param name="end">Code-point position one past the run end.</param>
        /// <returns>The exception (not thrown).</returns>
        internal static UnicodeEncodeError EncodeErrorFor(uint firstCodePoint, long start, long end)
        {
            if (end == start + 1)
            {
                string escaped = firstCodePoint <= 0xFF ? $"\\x{firstCodePoint:x2}"
                    : firstCodePoint <= 0xFFFF ? $"\\u{firstCodePoint:x4}"
                    : $"\\U{firstCodePoint:x8}";
                return new UnicodeEncodeError($"'ascii' codec can't encode character '{escaped}' in position {start}: ordinal not in range(128)");
            }
            return new UnicodeEncodeError($"'ascii' codec can't encode characters in position {start}-{end - 1}: ordinal not in range(128)");
        }

        // ---------------------------------------------------------------------------------------------
        // Widening helper
        // ---------------------------------------------------------------------------------------------

        /// <summary>Widens UTF-16 units (no surrogates) to UCS-4 units, one per char.</summary>
        /// <param name="src">The chars (every one a BMP code point).</param>
        /// <param name="dst">The destination units (at least <c>src.Length</c>).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WidenUtf16(ReadOnlySpan<char> src, uint* dst)
        {
            for (int i = 0; i < src.Length; i++)
                dst[i] = src[i];
        }
    }
}
