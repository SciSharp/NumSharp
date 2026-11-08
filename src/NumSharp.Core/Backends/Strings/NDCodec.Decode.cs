using System;
using NumSharp.IO;
using System.Runtime.CompilerServices;

// =============================================================================
// NDCodec.Decode.cs — CPython 3.12's decoders into code points, error handling included (strings phase 2, P2.6 milestone 4)
// =============================================================================
//
// Ports, statement for statement where the output depends on it (Objects/unicodeobject.c, Objects/stringlib/codecs.h):
//   UTF-8    unicode_decode_utf8 → stringlib utf8_decode: the error RANGES are CPython's — "invalid start byte" (one byte),
//            "invalid continuation byte" (the lead and the valid continuation bytes before the bad one: E0 80, ED A0 (a
//            surrogate), F0 80, F4 90 count as bad second bytes), "unexpected end of data" (a sequence the input ends inside,
//            to the end); ignore / replace (ONE U+FFFD per range) / surrogateescape (a U+DC80.. per byte) inline, the
//            others generic.
//   ASCII    PyUnicode_DecodeASCII: every byte >= 0x80 is its own error ("ordinal not in range(128)"); replace /
//            surrogateescape / ignore inline.
//   Latin-1  no errors: each byte is its code point.
//   UTF-16   PyUnicode_DecodeUTF16Stateful (final): the BOM codec reads a BOM (FF FE little, FE FF big, skipped) and decodes
//            little-endian without one; the errors quote 'utf-16-le' / 'utf-16-be' (the order decoded, even for 'utf-16') —
//            "truncated data" (an odd byte left), "unexpected end of data" (a high surrogate the input ends after, to the
//            end), "illegal encoding" (a low surrogate first: its two bytes), "illegal UTF-16 surrogate" (a high surrogate
//            not followed by a low one: the HIGH unit's two bytes, so the decode resumes at the next unit). Every error is
//            generic.
//   UTF-32   PyUnicode_DecodeUTF32Stateful (final): the same BOM rule over four bytes; "code point in surrogate code point
//            range(0xd800, 0xe000)" and "code point not in range(0x110000)" (one unit each), "truncated data" (the 1-3
//            bytes left). Generic.
//   charmap  charmap_decode_string: a byte whose table entry is U+FFFE is one error ("character maps to <undefined>",
//            codec name 'charmap'). Generic.
// Positions count bytes of the decoded input (for utf-8-sig, of the input after its BOM).
// The coders, their run loops and the error paths are [MethodImpl(AggressiveOptimization)] and the byte writers
// AggressiveInlining: each runs per element, and tiering would keep a short-lived process's first calls on tier-0
// code (see NDCharObjectPath.CodecFast.cs's header).
// =============================================================================

namespace NumSharp.Backends.Strings
{
    internal sealed unsafe partial class NDCodec
    {
        /// <summary>
        ///     <c>bytes.decode</c> with this codec (see the file header): the code points of <paramref name="size"/> bytes
        ///     appended to a code-point output. Empty input decodes to nothing for every codec without looking at it (the
        ///     caller skips the codec lookup for it too, as <c>PyUnicode_FromEncodedObject</c> does).
        /// </summary>
        /// <param name="s">The bytes.</param>
        /// <param name="size">Their number.</param>
        /// <param name="errors">The errors policy.</param>
        /// <param name="sink">The code-point output.</param>
        /// <exception cref="UnicodeDecodeError">Bytes the codec cannot decode under <c>strict</c>, or a handler that re-raises.</exception>
        /// <exception cref="UnicodeError">The <c>undefined</c> codec over non-empty bytes.</exception>
        /// <exception cref="LookupError">An unknown handler name, at the first error.</exception>
        /// <exception cref="TypeError"><c>xmlcharrefreplace</c> / <c>namereplace</c> at the first error.</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal void Decode(byte* s, long size, NDCodecErrors errors, NDCodecSink sink)
        {
            if (size == 0)
                return;
            switch (Kind)
            {
                case PythonCodecKind.Utf8:
                    DecodeUtf8(s, size, errors, sink);
                    return;
                case PythonCodecKind.Utf8Sig:
                    // encodings/utf_8_sig.py: a leading BOM is dropped, then codecs.utf_8_decode(input[3:], errors, True) —
                    // the errors name 'utf-8' and their positions count from after the BOM.
                    if (size >= 3 && s[0] == 0xEF && s[1] == 0xBB && s[2] == 0xBF)
                    {
                        s += 3;
                        size -= 3;
                    }
                    DecodeUtf8(s, size, errors, sink);
                    return;
                case PythonCodecKind.Latin1:
                {
                    sink.Reserve(size);
                    uint* d = (uint*)sink.Data + sink.Count;
                    for (long i = 0; i < size; i++)
                        d[i] = s[i];
                    sink.Count += size;
                    return;
                }
                case PythonCodecKind.Ascii:
                    DecodeAscii(s, size, errors, sink);
                    return;
                case PythonCodecKind.Utf16:
                case PythonCodecKind.Utf16LE:
                case PythonCodecKind.Utf16BE:
                    DecodeUtf16(s, size, ByteOrder, errors, sink);
                    return;
                case PythonCodecKind.Utf32:
                case PythonCodecKind.Utf32LE:
                case PythonCodecKind.Utf32BE:
                    DecodeUtf32(s, size, ByteOrder, errors, sink);
                    return;
                case PythonCodecKind.Charmap:
                    DecodeCharmap(s, size, errors, sink);
                    return;
                default:
                    throw new UnicodeError("undefined encoding");
            }
        }

        /// <summary>Whether a byte is a UTF-8 continuation byte (0x80..0xBF).</summary>
        /// <param name="b">The byte.</param>
        /// <returns>True for a continuation byte.</returns>
        private static bool IsContinuation(uint b) => b - 0x80u < 0x40u;

        /// <summary>
        ///     CPython's UTF-8 decoder (<c>unicode_decode_utf8</c>) with its inline error handling (see the file header).
        /// </summary>
        /// <param name="s">The bytes.</param>
        /// <param name="size">Their number.</param>
        /// <param name="errors">The errors policy.</param>
        /// <param name="sink">The code-point output.</param>
        /// <exception cref="UnicodeDecodeError">An invalid sequence under <c>strict</c>, or a handler that re-raises.</exception>
        /// <exception cref="LookupError">An unknown handler name, at the first error.</exception>
        /// <exception cref="TypeError"><c>xmlcharrefreplace</c> / <c>namereplace</c> at the first error.</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static void DecodeUtf8(byte* s, long size, NDCodecErrors errors, NDCodecSink sink)
        {
            // At most one code point per byte without errors; a handler reserves its own room and the rest is reserved again.
            sink.Reserve(size);
            long pos = 0;
            while (pos < size)
            {
                int code = Utf8Run(s, ref pos, size, sink);
                long start = pos, end;
                string reason;
                switch (code)
                {
                    case 0:
                        if (pos == size)
                            return;
                        reason = "unexpected end of data";
                        end = size;
                        break;
                    case 1:
                        reason = "invalid start byte";
                        end = start + 1;
                        break;
                    default:
                        reason = "invalid continuation byte";
                        end = start + code - 1;
                        break;
                }
                switch (errors.Inline)
                {
                    case NDErrorHandler.Ignore:
                        pos = end;
                        break;
                    case NDErrorHandler.Replace:
                        sink.PutPoint(0xFFFD);
                        pos = end;
                        break;
                    case NDErrorHandler.SurrogateEscape:
                        // Every byte of the range (all >= 0x80: a lead byte and continuation bytes) as U+DC80..U+DCFF.
                        for (long i = start; i < end; i++)
                            sink.PutPoint(0xDC00u + s[i]);
                        pos = end;
                        break;
                    default:
                        pos = CallDecodeHandler(errors, "utf-8", reason, s, size, start, end, sink);
                        break;
                }
                sink.Reserve(size - pos);
            }
        }

        /// <summary>
        ///     CPython's <c>utf8_decode</c> (stringlib/codecs.h): decodes valid sequences from <paramref name="pos"/> into the
        ///     output until the input ends or an invalid sequence starts, leaving <paramref name="pos"/> at its first byte.
        /// </summary>
        /// <param name="s">The bytes.</param>
        /// <param name="pos">The position to start at; receives the stop position.</param>
        /// <param name="end">The input's end.</param>
        /// <param name="sink">The code-point output (room for one code point per remaining byte is reserved).</param>
        /// <returns>0 at the end or a sequence the input ends inside; 1 an invalid start byte; 2 / 3 / 4 an invalid
        ///     continuation byte at that position of the sequence.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static int Utf8Run(byte* s, ref long pos, long end, NDCodecSink sink)
        {
            uint* p = (uint*)sink.Data + sink.Count;
            uint* p0 = p;
            long i = pos;
            int code = 0;
            while (i < end)
            {
                uint ch = s[i];
                if (ch < 0x80)
                {
                    *p++ = ch;
                    i++;
                    continue;
                }
                if (ch < 0xE0)
                {
                    // C2 80 .. DF BF: U+0080 .. U+07FF (80 .. C1: a continuation byte or an overlong ASCII lead).
                    if (ch < 0xC2)
                    {
                        code = 1;
                        break;
                    }
                    if (end - i < 2)
                        break;
                    uint ch2 = s[i + 1];
                    if (!IsContinuation(ch2))
                    {
                        code = 2;
                        break;
                    }
                    *p++ = (ch << 6) + ch2 - ((0xC0u << 6) + 0x80u);
                    i += 2;
                    continue;
                }
                if (ch < 0xF0)
                {
                    // E0 A0 80 .. EF BF BF: U+0800 .. U+FFFF, without the overlong E0 80..9F and the surrogates ED A0..BF.
                    if (end - i < 3)
                    {
                        if (end - i < 2)
                            break;
                        uint t2 = s[i + 1];
                        if (!IsContinuation(t2) || (t2 < 0xA0 ? ch == 0xE0 : ch == 0xED))
                        {
                            code = 2;
                            break;
                        }
                        break;
                    }
                    uint ch2 = s[i + 1], ch3 = s[i + 2];
                    if (!IsContinuation(ch2))
                    {
                        code = 2;
                        break;
                    }
                    if (ch == 0xE0)
                    {
                        if (ch2 < 0xA0)
                        {
                            code = 2;
                            break;
                        }
                    }
                    else if (ch == 0xED && ch2 >= 0xA0)
                    {
                        code = 2;
                        break;
                    }
                    if (!IsContinuation(ch3))
                    {
                        code = 3;
                        break;
                    }
                    *p++ = (ch << 12) + (ch2 << 6) + ch3 - ((0xE0u << 12) + (0x80u << 6) + 0x80u);
                    i += 3;
                    continue;
                }
                if (ch < 0xF5)
                {
                    // F0 90 80 80 .. F4 8F BF BF: U+10000 .. U+10FFFF, without the overlong F0 80..8F and F4 90.. past U+10FFFF.
                    if (end - i < 4)
                    {
                        if (end - i < 2)
                            break;
                        uint t2 = s[i + 1];
                        if (!IsContinuation(t2) || (t2 < 0x90 ? ch == 0xF0 : ch == 0xF4))
                        {
                            code = 2;
                            break;
                        }
                        if (end - i < 3)
                            break;
                        if (!IsContinuation(s[i + 2]))
                        {
                            code = 3;
                            break;
                        }
                        break;
                    }
                    uint ch2 = s[i + 1], ch3 = s[i + 2], ch4 = s[i + 3];
                    if (!IsContinuation(ch2))
                    {
                        code = 2;
                        break;
                    }
                    if (ch == 0xF0)
                    {
                        if (ch2 < 0x90)
                        {
                            code = 2;
                            break;
                        }
                    }
                    else if (ch == 0xF4 && ch2 >= 0x90)
                    {
                        code = 2;
                        break;
                    }
                    if (!IsContinuation(ch3))
                    {
                        code = 3;
                        break;
                    }
                    if (!IsContinuation(ch4))
                    {
                        code = 4;
                        break;
                    }
                    *p++ = (ch << 18) + (ch2 << 12) + (ch3 << 6) + ch4 - ((0xF0u << 18) + (0x80u << 12) + (0x80u << 6) + 0x80u);
                    i += 4;
                    continue;
                }
                code = 1;   // F5 .. FF never start a sequence
                break;
            }
            pos = i;
            sink.Count += p - p0;
            return code;
        }

        /// <summary>CPython's <c>PyUnicode_DecodeASCII</c> (see the file header).</summary>
        /// <param name="s">The bytes.</param>
        /// <param name="size">Their number.</param>
        /// <param name="errors">The errors policy.</param>
        /// <param name="sink">The code-point output.</param>
        /// <exception cref="UnicodeDecodeError">A byte >= 0x80 under <c>strict</c>, or a handler that re-raises.</exception>
        /// <exception cref="LookupError">An unknown handler name, at the first such byte.</exception>
        /// <exception cref="TypeError"><c>xmlcharrefreplace</c> / <c>namereplace</c> at the first such byte.</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static void DecodeAscii(byte* s, long size, NDCodecErrors errors, NDCodecSink sink)
        {
            sink.Reserve(size);
            long pos = 0;
            while (pos < size)
            {
                byte c = s[pos];
                if (c < 0x80)
                {
                    ((uint*)sink.Data)[sink.Count++] = c;
                    pos++;
                    continue;
                }
                switch (errors.Inline)
                {
                    case NDErrorHandler.Replace:
                        ((uint*)sink.Data)[sink.Count++] = 0xFFFD;
                        pos++;
                        break;
                    case NDErrorHandler.SurrogateEscape:
                        ((uint*)sink.Data)[sink.Count++] = 0xDC00u + c;
                        pos++;
                        break;
                    case NDErrorHandler.Ignore:
                        pos++;
                        break;
                    default:
                        pos = CallDecodeHandler(errors, "ascii", "ordinal not in range(128)", s, size, pos, pos + 1, sink);
                        sink.Reserve(size - pos);
                        break;
                }
            }
        }

        /// <summary>CPython's <c>PyUnicode_DecodeUTF16Stateful</c> with <c>final</c> (see the file header).</summary>
        /// <param name="s">The bytes.</param>
        /// <param name="size">Their number.</param>
        /// <param name="byteorder">0 (read a BOM), -1 little-endian, 1 big-endian.</param>
        /// <param name="errors">The errors policy.</param>
        /// <param name="sink">The code-point output.</param>
        /// <exception cref="UnicodeDecodeError">An invalid unit under <c>strict</c>, or a handler that re-raises.</exception>
        /// <exception cref="LookupError">An unknown handler name, at the first error.</exception>
        /// <exception cref="TypeError"><c>xmlcharrefreplace</c> / <c>namereplace</c> at the first error.</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static void DecodeUtf16(byte* s, long size, int byteorder, NDCodecErrors errors, NDCodecSink sink)
        {
            long q = 0;
            int bo = byteorder;
            if (bo == 0 && size >= 2)
            {
                // The BOM codec: a leading U+FEFF read little-endian chooses the order and is skipped.
                uint bom = (uint)(s[1] << 8 | s[0]);
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
            }
            if (q == size)
                return;
            bool little = bo <= 0;
            string encoding = little ? "utf-16-le" : "utf-16-be";
            sink.Reserve((size - q + 1) / 2);
            while (true)
            {
                int code = 0;
                if (size - q >= 2)
                    code = Utf16Run(s, ref q, size, little, sink);
                long start, end;
                string reason;
                switch (code)
                {
                    case 0:
                        if (q == size)
                            return;
                        reason = "truncated data";
                        start = q;
                        end = size;
                        break;
                    case 1:
                        // The high surrogate the input ends after: back to it, the range to the end.
                        q -= 2;
                        reason = "unexpected end of data";
                        start = q;
                        end = size;
                        break;
                    case 2:
                        reason = "illegal encoding";
                        start = q - 2;
                        end = start + 2;
                        break;
                    default:
                        // A high surrogate followed by a unit that is not a low one: the range is the HIGH unit only.
                        reason = "illegal UTF-16 surrogate";
                        start = q - 4;
                        end = start + 2;
                        break;
                }
                q = CallDecodeHandler(errors, encoding, reason, s, size, start, end, sink);
                sink.Reserve((size - q + 1) / 2);
            }
        }

        /// <summary>
        ///     CPython's <c>utf16_decode</c> (stringlib/codecs.h): decodes units from <paramref name="q"/> while two or more bytes
        ///     remain, stopping at an invalid one.
        /// </summary>
        /// <param name="s">The bytes.</param>
        /// <param name="q">The position; receives the stop position (past the unit or units of an error, as CPython leaves it).</param>
        /// <param name="size">The input's end.</param>
        /// <param name="little">True for little-endian units.</param>
        /// <param name="sink">The code-point output (room for one code point per two remaining bytes is reserved).</param>
        /// <returns>0 when fewer than two bytes remain; 1 a high surrogate the input ends after; 2 a low surrogate first;
        ///     3 a high surrogate not followed by a low one.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static int Utf16Run(byte* s, ref long q, long size, bool little, NDCodecSink sink)
        {
            uint* p = (uint*)sink.Data + sink.Count;
            uint* p0 = p;
            int ihi = little ? 1 : 0, ilo = little ? 0 : 1;
            long e = size - 1;   // CPython's --e: a unit needs q < e
            long i = q;
            int code = 0;
            while (i < e)
            {
                uint ch = (uint)(s[i + ihi] << 8 | s[i + ilo]);
                i += 2;
                if (ch - 0xD800u > 0x7FFu)
                {
                    *p++ = ch;
                    continue;
                }
                if (ch >= 0xDC00)
                {
                    code = 2;
                    break;
                }
                if (i >= e)
                {
                    code = 1;
                    break;
                }
                uint ch2 = (uint)(s[i + ihi] << 8 | s[i + ilo]);
                i += 2;
                if (ch2 - 0xDC00u > 0x3FFu)
                {
                    code = 3;
                    break;
                }
                *p++ = (((ch & 0x3FF) << 10) | (ch2 & 0x3FF)) + 0x10000;
            }
            q = i;
            sink.Count += p - p0;
            return code;
        }

        /// <summary>CPython's <c>PyUnicode_DecodeUTF32Stateful</c> with <c>final</c> (see the file header).</summary>
        /// <param name="s">The bytes.</param>
        /// <param name="size">Their number.</param>
        /// <param name="byteorder">0 (read a BOM), -1 little-endian, 1 big-endian.</param>
        /// <param name="errors">The errors policy.</param>
        /// <param name="sink">The code-point output.</param>
        /// <exception cref="UnicodeDecodeError">An invalid unit under <c>strict</c>, or a handler that re-raises.</exception>
        /// <exception cref="LookupError">An unknown handler name, at the first error.</exception>
        /// <exception cref="TypeError"><c>xmlcharrefreplace</c> / <c>namereplace</c> at the first error.</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static void DecodeUtf32(byte* s, long size, int byteorder, NDCodecErrors errors, NDCodecSink sink)
        {
            long q = 0;
            int bo = byteorder;
            if (bo == 0 && size >= 4)
            {
                uint bom = (uint)s[3] << 24 | (uint)s[2] << 16 | (uint)s[1] << 8 | s[0];
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
            }
            if (q == size)
                return;
            bool little = bo <= 0;
            string encoding = little ? "utf-32-le" : "utf-32-be";
            sink.Reserve((size - q + 3) / 4);
            while (true)
            {
                string reason = null;
                uint* d = (uint*)sink.Data;
                long c = sink.Count;
                while (size - q >= 4)
                {
                    byte* u = s + q;
                    uint ch = little
                        ? (uint)u[3] << 24 | (uint)u[2] << 16 | (uint)u[1] << 8 | u[0]
                        : (uint)u[0] << 24 | (uint)u[1] << 16 | (uint)u[2] << 8 | u[3];
                    if (ch - 0xD800u <= 0x7FFu)
                    {
                        reason = "code point in surrogate code point range(0xd800, 0xe000)";
                        break;
                    }
                    if (ch >= 0x110000)
                    {
                        reason = "code point not in range(0x110000)";
                        break;
                    }
                    d[c++] = ch;
                    q += 4;
                }
                sink.Count = c;
                long start = q, end;
                if (reason is not null)
                {
                    end = q + 4;
                }
                else
                {
                    if (q == size)
                        return;
                    reason = "truncated data";
                    end = size;
                }
                q = CallDecodeHandler(errors, encoding, reason, s, size, start, end, sink);
                sink.Reserve((size - q + 3) / 4);
            }
        }

        /// <summary>CPython's <c>charmap_decode_string</c> (see the file header).</summary>
        /// <param name="s">The bytes.</param>
        /// <param name="size">Their number.</param>
        /// <param name="errors">The errors policy.</param>
        /// <param name="sink">The code-point output.</param>
        /// <exception cref="UnicodeDecodeError">An undefined byte under <c>strict</c>, or a handler that re-raises.</exception>
        /// <exception cref="LookupError">An unknown handler name, at the first undefined byte.</exception>
        /// <exception cref="TypeError"><c>xmlcharrefreplace</c> / <c>namereplace</c> at the first undefined byte.</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal void DecodeCharmap(byte* s, long size, NDCodecErrors errors, NDCodecSink sink)
        {
            sink.Reserve(size);
            long q = 0;
            var table = _decodeTable;
            while (q < size)
            {
                uint x = table[s[q]];
                if (x == 0xFFFE)
                {
                    q = CallDecodeHandler(errors, "charmap", UndefinedText, s, size, q, q + 1, sink);
                    sink.Reserve(size - q);
                    continue;
                }
                ((uint*)sink.Data)[sink.Count++] = x;
                q++;
            }
        }
    }
}
