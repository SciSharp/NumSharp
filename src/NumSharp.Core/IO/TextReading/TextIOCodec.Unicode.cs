using System;
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using NumSharp.Backends.Strings;

namespace NumSharp.IO
{
    internal sealed partial class TextIOCodec
    {
        /// <summary>
        ///     CPython's <c>PyUnicode_DecodeUTF16Stateful</c>: a leading BOM is read when <paramref name="byteorder"/> is 0
        ///     (and the order it names reported back), then the code units are decoded in that order (native little-endian
        ///     without a BOM).
        /// </summary>
        /// <param name="s">The bytes.</param>
        /// <param name="byteorder">In: 0 to read a BOM, -1 little-endian, 1 big-endian. Out: the order a BOM named (still 0
        ///     when there was none — CPython's <c>utf_16_ex_decode</c> reports that).</param>
        /// <param name="final">True when the input ends here (CPython's <c>consumed</c> is NULL): a trailing odd byte or
        ///     high surrogate is then an error; otherwise it is left for the next call.</param>
        /// <param name="dest">The characters: at least <c>s.Length / 2</c>.</param>
        /// <param name="consumed">The bytes decoded (the rest must be passed again, first, next time).</param>
        /// <returns>The number of characters written.</returns>
        /// <exception cref="UnicodeDecodeError">CPython's texts, under the codec name of the order (<c>'utf-16-le'</c> /
        ///     <c>'utf-16-be'</c>): a low surrogate first (<c>illegal encoding</c>, its two bytes), a high surrogate not
        ///     followed by a low one (<c>illegal UTF-16 surrogate</c>, the high surrogate's bytes), and at the end
        ///     <c>unexpected end of data</c> (a high surrogate cut off) or <c>truncated data</c> (an odd byte).</exception>
        internal static int DecodeUtf16(ReadOnlySpan<byte> s, ref int byteorder, bool final, Span<char> dest, out int consumed)
        {
            int q = 0, e = s.Length;
            int bo = byteorder;
            if (bo == 0 && e >= 2)
            {
                int bom = (s[1] << 8) | s[0];
                if (bom == 0xFEFF)
                {
                    q = 2;
                    bo = -1;
                }
                else if (bom == 0xFFFE)
                {
                    q = 2;
                    bo = 1;
                }
                byteorder = bo;
            }
            if (q == e)
            {
                consumed = e;
                return 0;
            }
            bool le = bo <= 0; // CPython's native ordering on a little-endian host
            string encoding = le ? "utf-16-le" : "utf-16-be";

            int w = 0;
            int status; // 0: fewer than 2 bytes left, 1: unexpected end, 2: illegal encoding, 3: illegal surrogate
            while (true)
            {
                // The units before the next surrogate decode as they are, instead of CPython's per-unit loop (the same
                // characters): host-order input is searched in place and copied up to the surrogate; swapped input is
                // byte-swapped into the destination 64 units at a time and searched there, so a surrogate wastes at most
                // one block of swapping (text dense in pairs stays linear).
                int units = (e - q) >> 1;
                int plain;
                if (le == BitConverter.IsLittleEndian)
                {
                    var src = MemoryMarshal.Cast<byte, char>(s.Slice(q, units * 2));
                    plain = src.IndexOfAnyInRange('\uD800', '\uDFFF');
                    if (plain < 0)
                        plain = units;
                    src.Slice(0, plain).CopyTo(dest.Slice(w));
                }
                else
                {
                    plain = 0;
                    while (plain < units)
                    {
                        int block = Math.Min(units - plain, 64);
                        var swapped = dest.Slice(w + plain, block);
                        BinaryPrimitives.ReverseEndianness(MemoryMarshal.Cast<byte, ushort>(s.Slice(q + plain * 2, block * 2)),
                            MemoryMarshal.Cast<char, ushort>(swapped));
                        int k = swapped.IndexOfAnyInRange('\uD800', '\uDFFF');
                        if (k >= 0)
                        {
                            plain += k;
                            break;
                        }
                        plain += block;
                    }
                }
                w += plain;
                q += plain * 2;
                if (e - q < 2)
                {
                    status = 0;
                    break;
                }
                uint ch = ReadUnit16(s, q, le);
                q += 2;
                if (ch >= 0xDC00)
                {
                    status = 2; // a low surrogate cannot start a pair
                    break;
                }
                if (e - q < 2)
                {
                    status = 1; // the high surrogate's partner is cut off
                    break;
                }
                uint ch2 = ReadUnit16(s, q, le);
                q += 2;
                if (ch2 < 0xDC00 || ch2 > 0xDFFF)
                {
                    status = 3;
                    break;
                }
                dest[w++] = (char)ch;
                dest[w++] = (char)ch2;
            }

            switch (status)
            {
                case 0:
                    if (q == e || !final)
                    {
                        consumed = q;
                        return w;
                    }
                    throw DecodeError(encoding, s, q, e, "truncated data");
                case 1:
                    q -= 2; // back to the high surrogate: it is what is incomplete
                    if (!final)
                    {
                        consumed = q;
                        return w;
                    }
                    throw DecodeError(encoding, s, q, e, "unexpected end of data");
                case 2:
                    throw DecodeError(encoding, s, q - 2, q, "illegal encoding");
                default:
                    throw DecodeError(encoding, s, q - 4, q - 2, "illegal UTF-16 surrogate");
            }
        }

        /// <summary>One UTF-16 code unit in the given order.</summary>
        /// <param name="s">The bytes.</param>
        /// <param name="q">The unit's first byte.</param>
        /// <param name="le">True for little-endian.</param>
        /// <returns>The unit.</returns>
        private static uint ReadUnit16(ReadOnlySpan<byte> s, int q, bool le)
            => le ? (uint)(s[q] | (s[q + 1] << 8)) : (uint)((s[q] << 8) | s[q + 1]);

        /// <summary>
        ///     CPython's <c>PyUnicode_DecodeUTF32Stateful</c>: a leading BOM is read when <paramref name="byteorder"/> is 0,
        ///     then four-byte code points in that order (native little-endian without a BOM).
        /// </summary>
        /// <param name="s">The bytes.</param>
        /// <param name="byteorder">In: 0 to read a BOM, -1 little-endian, 1 big-endian. Out: the order a BOM named (0 when
        ///     there was none).</param>
        /// <param name="final">True when the input ends here: 1–3 trailing bytes are then an error, else left over.</param>
        /// <param name="dest">The characters: at least <c>s.Length / 2</c> (a code point past the BMP is two).</param>
        /// <param name="consumed">The bytes decoded.</param>
        /// <returns>The number of characters written.</returns>
        /// <exception cref="UnicodeDecodeError">CPython's texts under <c>'utf-32-le'</c> / <c>'utf-32-be'</c>: <c>code point
        ///     in surrogate code point range(0xd800, 0xe000)</c>, <c>code point not in range(0x110000)</c> (the four bytes),
        ///     <c>truncated data</c> (the trailing bytes, when final).</exception>
        internal static int DecodeUtf32(ReadOnlySpan<byte> s, ref int byteorder, bool final, Span<char> dest, out int consumed)
        {
            int q = 0, e = s.Length;
            int bo = byteorder;
            if (bo == 0 && e >= 4)
            {
                uint bom = (uint)(s[3] << 24) | (uint)(s[2] << 16) | (uint)(s[1] << 8) | s[0];
                if (bom == 0x0000FEFF)
                {
                    bo = -1;
                    q = 4;
                }
                else if (bom == 0xFFFE0000)
                {
                    bo = 1;
                    q = 4;
                }
                byteorder = bo;
            }
            if (q == e)
            {
                consumed = e;
                return 0;
            }
            bool le = bo <= 0;
            string encoding = le ? "utf-32-le" : "utf-32-be";

            int w = 0;
            while (e - q >= 4)
            {
                uint ch = le
                    ? (uint)s[q] | (uint)(s[q + 1] << 8) | (uint)(s[q + 2] << 16) | (uint)(s[q + 3] << 24)
                    : (uint)(s[q] << 24) | (uint)(s[q + 1] << 16) | (uint)(s[q + 2] << 8) | s[q + 3];
                // CPython tests the surrogate range before the upper bound.
                if (ch >= 0xD800 && ch <= 0xDFFF)
                    throw DecodeError(encoding, s, q, q + 4, "code point in surrogate code point range(0xd800, 0xe000)");
                if (ch >= 0x110000)
                    throw DecodeError(encoding, s, q, q + 4, "code point not in range(0x110000)");
                if (ch < 0x10000)
                {
                    dest[w++] = (char)ch;
                }
                else
                {
                    ch -= 0x10000;
                    dest[w++] = (char)(0xD800 + (ch >> 10));
                    dest[w++] = (char)(0xDC00 + (ch & 0x3FF));
                }
                q += 4;
            }
            if (q == e || !final)
            {
                consumed = q;
                return w;
            }
            throw DecodeError(encoding, s, q, e, "truncated data");
        }

        /// <summary>
        ///     Raises CPython's UTF-16 / UTF-32 encode error for the first lone surrogate of <paramref name="text"/> (a valid
        ///     pair is one code point past the BMP, which both encode); CPython reports one surrogate at a time.
        /// </summary>
        /// <param name="text">The text.</param>
        /// <param name="codecName">The codec name the error quotes (<c>utf-16</c>, <c>utf-16-le</c>, <c>utf-32-be</c>, …).</param>
        /// <exception cref="UnicodeEncodeError"><c>'utf-16' codec can't encode character '\ud800' in position 1: surrogates
        ///     not allowed</c>, the position counting code points.</exception>
        private static void CheckNoLoneSurrogate(ReadOnlySpan<char> text, string codecName)
        {
            int i = text.IndexOfAnyInRange('\uD800', '\uDFFF');
            if (i < 0)
                return;
            // Before the first surrogate every character is one code point, so positions start as indices.
            long position = i;
            while (i < text.Length)
            {
                char c = text[i];
                if (char.IsHighSurrogate(c) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
                {
                    i += 2;
                    position++;
                    continue;
                }
                if (char.IsSurrogate(c))
                    throw NDStringCodec.EncodeErrorFor(c, position, position + 1, codecName, "surrogates not allowed");
                i++;
                position++;
            }
        }

        /// <summary>The UTF-16 bytes of validated text (no lone surrogate), after a little-endian BOM when asked.</summary>
        /// <param name="text">The text.</param>
        /// <param name="dest">At least <c>2 * text.Length + 2</c> bytes.</param>
        /// <param name="bigEndian">True for big-endian units.</param>
        /// <param name="bom">True to write <c>FF FE</c> first (the <c>utf-16</c> codec's native-order BOM).</param>
        /// <returns>The number of bytes written.</returns>
        private static int EncodeUtf16(ReadOnlySpan<char> text, Span<byte> dest, bool bigEndian, bool bom)
        {
            int head = 0;
            if (bom)
            {
                dest[0] = 0xFF;
                dest[1] = 0xFE;
                head = 2;
            }
            var units = MemoryMarshal.Cast<byte, ushort>(dest.Slice(head, text.Length * 2));
            var src = MemoryMarshal.Cast<char, ushort>(text);
            if (bigEndian == !BitConverter.IsLittleEndian)
                src.CopyTo(units);
            else
                BinaryPrimitives.ReverseEndianness(src, units);
            return head + text.Length * 2;
        }

        /// <summary>The UTF-32 bytes of validated text (a pair is one code point), after a little-endian BOM when asked.</summary>
        /// <param name="text">The text.</param>
        /// <param name="dest">At least <c>4 * text.Length + 4</c> bytes.</param>
        /// <param name="bigEndian">True for big-endian code points.</param>
        /// <param name="bom">True to write <c>FF FE 00 00</c> first.</param>
        /// <returns>The number of bytes written.</returns>
        private static int EncodeUtf32(ReadOnlySpan<char> text, Span<byte> dest, bool bigEndian, bool bom)
        {
            int w = 0;
            if (bom)
            {
                dest[0] = 0xFF;
                dest[1] = 0xFE;
                dest[2] = 0;
                dest[3] = 0;
                w = 4;
            }
            for (int i = 0; i < text.Length; i++)
            {
                uint cp = text[i];
                if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
                    cp = (uint)char.ConvertToUtf32(text[i], text[++i]);
                if (bigEndian)
                    BinaryPrimitives.WriteUInt32BigEndian(dest.Slice(w), cp);
                else
                    BinaryPrimitives.WriteUInt32LittleEndian(dest.Slice(w), cp);
                w += 4;
            }
            return w;
        }
    }
}
