using System;
using NumSharp.IO;
using System.Runtime.CompilerServices;

// =============================================================================
// NDCodec.Encode.cs — CPython 3.12's encoders over code points, error handling included (strings phase 2, P2.6 milestone 4)
// =============================================================================
//
// Ports, statement for statement where the output depends on it (Objects/unicodeobject.c, Objects/stringlib/codecs.h):
//   UTF-8     unicode_encode_utf8 → ucs4lib_utf8_encoder: 1-4 bytes per code point (a unit past U+10FFFF gets the four-byte
//             pattern, each byte truncated as C's (char) cast truncates); a RUN of consecutive surrogates is one error
//             ("surrogates not allowed") handled inline for replace / ignore / surrogatepass / backslashreplace /
//             xmlcharrefreplace / surrogateescape (its escapable prefix, then the generic call over the rest of the run),
//             generically otherwise (strict, namereplace, unknown names).
//   Latin-1 / unicode_encode_ucs1 with limit 256 / 128: a run of consecutive characters at or past the limit is one error
//   ASCII     ("ordinal not in range(256)" / "(128)"), strict inline, the others as in UTF-8; a generic str replacement must
//             be one-byte (Latin-1) or ASCII, else the ORIGINAL error is raised.
//   UTF-16 /  _PyUnicode_EncodeUTF16 / 32: a BOM codec writes the native (little-endian) BOM first; every SURROGATE is its
//   UTF-32    own error ([pos, pos + 1)) and goes through the generic call — a bytes replacement must fill whole units, a
//             str one must be ASCII (each character one unit). UTF-16 writes a code point past U+FFFF as CPython's surrogate
//             arithmetic does (Py_UNICODE_HIGH_SURROGATE / LOW, truncated to 16 bits — so even a unit past U+10FFFF writes
//             two units), UTF-32 any non-surrogate unit as it is.
//   charmap   _PyUnicode_EncodeCharmap + charmap_encoding_error: a run of consecutive unencodable characters is one error
//             (codec name 'charmap', "character maps to <undefined>"); strict / replace / ignore / xmlcharrefreplace inline,
//             every other handler generic; the replacement's characters ('?', the xmlcharref text, a generic str) are
//             themselves encoded through the table, and an unmapped one raises the ORIGINAL error.
// The coders, their run loops and the error paths are [MethodImpl(AggressiveOptimization)] and the byte writers
// AggressiveInlining: each runs per element, and tiering would keep a short-lived process's first calls on tier-0
// code (see NDCharObjectPath.CodecFast.cs's header).
// =============================================================================

namespace NumSharp.Backends.Strings
{
    internal sealed unsafe partial class NDCodec
    {
        /// <summary>The UTF-8 byte-order mark <c>utf-8-sig</c> writes before every encode.</summary>
        internal static ReadOnlySpan<byte> Utf8Bom => new byte[] { 0xEF, 0xBB, 0xBF };

        /// <summary>CPython's reason for a surrogate the UTF-8 / UTF-16 / UTF-32 encoders cannot write.</summary>
        private const string SurrogatesText = "surrogates not allowed";

        /// <summary>CPython's reason for a character the charmap codecs cannot encode or a byte they cannot decode.</summary>
        private const string UndefinedText = "character maps to <undefined>";

        /// <summary>
        ///     <c>str.encode</c> with this codec (see the file header): the codes of <paramref name="n"/> code points appended to a
        ///     byte output — a BOM codec's BOM first, so the empty str of <c>utf-16</c> is the two BOM bytes.
        /// </summary>
        /// <param name="s">The str's code points.</param>
        /// <param name="n">Their number.</param>
        /// <param name="errors">The errors policy.</param>
        /// <param name="sink">The byte output.</param>
        /// <exception cref="UnicodeEncodeError">A character the codec cannot encode under <c>strict</c>, or a handler that
        ///     re-raises.</exception>
        /// <exception cref="UnicodeError">The <c>undefined</c> codec (every encode, the empty str included).</exception>
        /// <exception cref="LookupError">An unknown handler name, at the first error.</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal void Encode(uint* s, long n, NDCodecErrors errors, NDCodecSink sink)
        {
            switch (Kind)
            {
                case PythonCodecKind.Utf8:
                    EncodeUtf8(s, n, errors, sink);
                    return;
                case PythonCodecKind.Utf8Sig:
                    // encodings/utf_8_sig.py: BOM_UTF8 + codecs.utf_8_encode(input, errors) — the errors name 'utf-8'.
                    sink.PutBytes(Utf8Bom);
                    EncodeUtf8(s, n, errors, sink);
                    return;
                case PythonCodecKind.Latin1:
                    EncodeUcs1(s, n, 256, "latin-1", "ordinal not in range(256)", errors, sink);
                    return;
                case PythonCodecKind.Ascii:
                    EncodeUcs1(s, n, 128, "ascii", "ordinal not in range(128)", errors, sink);
                    return;
                case PythonCodecKind.Utf16:
                case PythonCodecKind.Utf16LE:
                case PythonCodecKind.Utf16BE:
                    EncodeUtf16(s, n, ByteOrder, errors, sink);
                    return;
                case PythonCodecKind.Utf32:
                case PythonCodecKind.Utf32LE:
                case PythonCodecKind.Utf32BE:
                    EncodeUtf32(s, n, ByteOrder, errors, sink);
                    return;
                case PythonCodecKind.Charmap:
                    EncodeCharmap(s, n, errors, sink);
                    return;
                default:
                    // encodings/undefined.py raises before looking at the input.
                    throw new UnicodeError("undefined encoding");
            }
        }

        /// <summary>Whether every byte of a replacement is ASCII (<c>PyUnicode_IS_ASCII</c> of a generic str replacement).</summary>
        /// <param name="rep">The replacement.</param>
        /// <returns>True when every byte is below 0x80.</returns>
        private static bool IsAscii(ReadOnlySpan<byte> rep) => rep.IndexOfAnyExceptInRange((byte)0, (byte)0x7F) < 0;

        /// <summary>
        ///     CPython's UTF-8 encoder (<c>ucs4lib_utf8_encoder</c>) with its inline error handling (see the file header).
        /// </summary>
        /// <param name="s">The str's code points.</param>
        /// <param name="n">Their number.</param>
        /// <param name="errors">The errors policy.</param>
        /// <param name="sink">The byte output.</param>
        /// <exception cref="UnicodeEncodeError">A surrogate under <c>strict</c> (the run of consecutive surrogates), or a
        ///     handler that re-raises.</exception>
        /// <exception cref="LookupError">An unknown handler name, at the first surrogate.</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static void EncodeUtf8(uint* s, long n, NDCodecErrors errors, NDCodecSink sink)
        {
            // Four bytes per code point is the most any character writes; a handler reserves its own room, and the room
            // for the rest is reserved again after it (the buffer may have moved).
            sink.Reserve(n * 4);
            byte* d = sink.Data;
            long c = sink.Count;
            long i = 0;
            while (i < n)
            {
                uint ch = s[i++];
                if (ch < 0x80)
                {
                    d[c++] = (byte)ch;
                    continue;
                }
                if (ch < 0x800)
                {
                    d[c++] = (byte)(0xC0 | (ch >> 6));
                    d[c++] = (byte)(0x80 | (ch & 0x3F));
                    continue;
                }
                if (ch - 0xD800u <= 0x7FFu)
                {
                    // A run of consecutive surrogates is one error.
                    long startpos = i - 1, endpos = startpos + 1;
                    while (endpos < n && s[endpos] - 0xD800u <= 0x7FFu)
                        endpos++;
                    sink.Count = c;
                    switch (errors.Inline)
                    {
                        case NDErrorHandler.Replace:
                            for (long k = startpos; k < endpos; k++)
                                d[c++] = (byte)'?';
                            sink.Count = c;
                            i = endpos;
                            break;
                        case NDErrorHandler.Ignore:
                            i = endpos;
                            break;
                        case NDErrorHandler.SurrogatePass:
                            // Three bytes per surrogate: within the four reserved per character.
                            for (long k = startpos; k < endpos; k++)
                            {
                                uint sc = s[k];
                                d[c++] = (byte)(0xE0 | (sc >> 12));
                                d[c++] = (byte)(0x80 | ((sc >> 6) & 0x3F));
                                d[c++] = (byte)(0x80 | (sc & 0x3F));
                            }
                            sink.Count = c;
                            i = endpos;
                            break;
                        case NDErrorHandler.BackslashReplace:
                            PutBackslashReplace(s, startpos, endpos, sink);
                            i = endpos;
                            break;
                        case NDErrorHandler.XmlCharRefReplace:
                            PutXmlCharRefReplace(s, startpos, endpos, sink);
                            i = endpos;
                            break;
                        case NDErrorHandler.SurrogateEscape:
                        {
                            long k = startpos;
                            for (; k < endpos; k++)
                            {
                                uint sc = s[k];
                                if (sc < 0xDC80 || sc > 0xDCFF)
                                    break;
                                d[c++] = (byte)(sc & 0xFF);
                            }
                            sink.Count = c;
                            if (k >= endpos)
                            {
                                i = endpos;
                                break;
                            }
                            // The escapable prefix is written; the generic call sees the rest of the run (and re-raises at
                            // its first character, which is not U+DC80..U+DCFF).
                            i = Utf8Generic(errors, s, n, k, endpos, sink);
                            break;
                        }
                        default:
                            i = Utf8Generic(errors, s, n, startpos, endpos, sink);
                            break;
                    }
                    sink.Reserve((n - i) * 4);
                    d = sink.Data;
                    c = sink.Count;
                    continue;
                }
                if (ch < 0x10000)
                {
                    d[c++] = (byte)(0xE0 | (ch >> 12));
                    d[c++] = (byte)(0x80 | ((ch >> 6) & 0x3F));
                    d[c++] = (byte)(0x80 | (ch & 0x3F));
                    continue;
                }
                // The four-byte pattern, each byte truncated as CPython's (char) casts truncate (a unit past U+10FFFF
                // keeps only the low byte of 0xF0 | ch >> 18).
                d[c++] = (byte)(0xF0 | (ch >> 18));
                d[c++] = (byte)(0x80 | ((ch >> 12) & 0x3F));
                d[c++] = (byte)(0x80 | ((ch >> 6) & 0x3F));
                d[c++] = (byte)(0x80 | (ch & 0x3F));
            }
            sink.Count = c;
        }

        /// <summary>The UTF-8 encoder's generic call (its <c>default:</c> case): the handler's replacement appended — a str one
        ///     must be ASCII — and the position to resume at.</summary>
        /// <param name="errors">The errors policy.</param>
        /// <param name="s">The str's code points.</param>
        /// <param name="n">Their number.</param>
        /// <param name="start">The bad range's first character.</param>
        /// <param name="end">One past its last.</param>
        /// <param name="sink">The byte output (its <c>Count</c> current).</param>
        /// <returns>The position to resume at.</returns>
        /// <exception cref="UnicodeEncodeError">The handler re-raises, or its str is not ASCII.</exception>
        /// <exception cref="LookupError">An unknown handler name.</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static long Utf8Generic(NDCodecErrors errors, uint* s, long n, long start, long end, NDCodecSink sink)
        {
            var rep = CallEncodeHandler(errors, "utf-8", SurrogatesText, s, n, start, end, out bool isBytes, out long newpos);
            if (!isBytes && !IsAscii(rep))
                throw EncodeError("utf-8", s, start, end, SurrogatesText);
            sink.PutBytes(rep);
            return newpos;
        }

        /// <summary>
        ///     CPython's <c>unicode_encode_ucs1</c> (Latin-1 with limit 256, ASCII with limit 128) with its inline error
        ///     handling (see the file header).
        /// </summary>
        /// <param name="s">The str's code points.</param>
        /// <param name="n">Their number.</param>
        /// <param name="limit">The first code point the codec cannot write (256 or 128).</param>
        /// <param name="encoding">The codec name the errors quote (<c>latin-1</c>, <c>ascii</c>).</param>
        /// <param name="reason">CPython's reason (<c>ordinal not in range(256)</c> / <c>(128)</c>).</param>
        /// <param name="errors">The errors policy.</param>
        /// <param name="sink">The byte output.</param>
        /// <exception cref="UnicodeEncodeError">A character at or past the limit under <c>strict</c> (the run of them), or a
        ///     handler that re-raises.</exception>
        /// <exception cref="LookupError">An unknown handler name, at the first such character.</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static void EncodeUcs1(uint* s, long n, uint limit, string encoding, string reason, NDCodecErrors errors,
            NDCodecSink sink)
        {
            sink.Reserve(n);
            byte* d = sink.Data;
            long c = sink.Count;
            long pos = 0;
            while (pos < n)
            {
                uint ch = s[pos];
                if (ch < limit)
                {
                    d[c++] = (byte)ch;
                    pos++;
                    continue;
                }
                // The run of consecutive characters the codec cannot write is one error.
                long collstart = pos, collend = pos + 1;
                while (collend < n && s[collend] >= limit)
                    collend++;
                sink.Count = c;
                switch (errors.Inline)
                {
                    case NDErrorHandler.Strict:
                        throw EncodeError(encoding, s, collstart, collend, reason);
                    case NDErrorHandler.Replace:
                        // One byte per character: within the room reserved for the str.
                        for (long k = collstart; k < collend; k++)
                            d[c++] = (byte)'?';
                        sink.Count = c;
                        pos = collend;
                        break;
                    case NDErrorHandler.Ignore:
                        pos = collend;
                        break;
                    case NDErrorHandler.BackslashReplace:
                        PutBackslashReplace(s, collstart, collend, sink);
                        pos = collend;
                        break;
                    case NDErrorHandler.XmlCharRefReplace:
                        PutXmlCharRefReplace(s, collstart, collend, sink);
                        pos = collend;
                        break;
                    case NDErrorHandler.SurrogateEscape:
                    {
                        long k = collstart;
                        for (; k < collend; k++)
                        {
                            uint sc = s[k];
                            if (sc < 0xDC80 || sc > 0xDCFF)
                                break;
                            // (char)(ch - 0xdc00): the byte 0x80..0xFF, written even by the ASCII codec.
                            d[c++] = (byte)(sc - 0xDC00);
                            pos++;
                        }
                        sink.Count = c;
                        if (k >= collend)
                            break;
                        pos = Ucs1Generic(errors, s, n, pos, collend, limit, encoding, reason, sink);
                        break;
                    }
                    default:
                        pos = Ucs1Generic(errors, s, n, collstart, collend, limit, encoding, reason, sink);
                        break;
                }
                sink.Reserve(n - pos);
                d = sink.Data;
                c = sink.Count;
            }
            sink.Count = c;
        }

        /// <summary>The one-byte encoders' generic call: the handler's replacement appended — a str one must be one-byte
        ///     (Latin-1) or ASCII, else the ORIGINAL error is raised — and the position to resume at.</summary>
        /// <param name="errors">The errors policy.</param>
        /// <param name="s">The str's code points.</param>
        /// <param name="n">Their number.</param>
        /// <param name="start">The bad range's first character.</param>
        /// <param name="end">One past its last.</param>
        /// <param name="limit">The codec's limit (256 or 128).</param>
        /// <param name="encoding">The codec name the errors quote.</param>
        /// <param name="reason">CPython's reason.</param>
        /// <param name="sink">The byte output (its <c>Count</c> current).</param>
        /// <returns>The position to resume at.</returns>
        /// <exception cref="UnicodeEncodeError">The handler re-raises, or its str does not fit the codec.</exception>
        /// <exception cref="LookupError">An unknown handler name.</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static long Ucs1Generic(NDCodecErrors errors, uint* s, long n, long start, long end, uint limit, string encoding,
            string reason, NDCodecSink sink)
        {
            var rep = CallEncodeHandler(errors, encoding, reason, s, n, start, end, out bool isBytes, out long newpos);
            // A str replacement must be one-byte for Latin-1 (every byte of the span is) and ASCII for ASCII.
            if (!isBytes && limit == 128 && !IsAscii(rep))
                throw EncodeError(encoding, s, start, end, reason);
            sink.PutBytes(rep);
            return newpos;
        }

        /// <summary>
        ///     CPython's <c>_PyUnicode_EncodeUTF16</c> (see the file header): the BOM for the BOM codec, then every code point as
        ///     one or two 16-bit units in the codec's order, each surrogate through the generic call.
        /// </summary>
        /// <param name="s">The str's code points.</param>
        /// <param name="n">Their number.</param>
        /// <param name="byteorder">0 (the BOM codec: native little-endian with a BOM), -1 little-endian, 1 big-endian.</param>
        /// <param name="errors">The errors policy.</param>
        /// <param name="sink">The byte output.</param>
        /// <exception cref="UnicodeEncodeError">A surrogate under <c>strict</c>, a handler that re-raises, or a replacement that
        ///     does not fill whole units.</exception>
        /// <exception cref="LookupError">An unknown handler name, at the first surrogate.</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static void EncodeUtf16(uint* s, long n, int byteorder, NDCodecErrors errors, NDCodecSink sink)
        {
            string encoding = byteorder < 0 ? "utf-16-le" : byteorder > 0 ? "utf-16-be" : "utf-16";
            bool big = byteorder > 0;
            // Room for two units per code point (a code point past U+FFFF is a pair) and the BOM; a replacement reserves its
            // own, and the room for the rest is reserved again after it (the buffer may have moved).
            sink.Reserve(n * 4 + 2);
            if (byteorder == 0)
                Put16(sink, 0xFEFF, big: false);
            long pos = 0;
            while (pos < n)
            {
                // ucs4lib_utf16_encode: every character up to the first surrogate.
                byte* d = sink.Data;
                long c = sink.Count;
                for (; pos < n; pos++)
                {
                    uint ch = s[pos];
                    if (ch < 0xD800 || (ch >= 0xE000 && ch < 0x10000))
                    {
                        Write16(d, ref c, ch, big);
                    }
                    else if (ch < 0xE000)
                    {
                        break;   // a surrogate
                    }
                    else
                    {
                        // Py_UNICODE_HIGH_SURROGATE / LOW_SURROGATE, stored as unsigned shorts.
                        Write16(d, ref c, 0xD800 - (0x10000 >> 10) + (ch >> 10), big);
                        Write16(d, ref c, 0xDC00 + (ch & 0x3FF), big);
                    }
                }
                sink.Count = c;
                if (pos == n)
                    break;
                var rep = CallEncodeHandler(errors, encoding, SurrogatesText, s, n, pos, pos + 1, out bool isBytes, out long newpos);
                if (isBytes)
                {
                    if ((rep.Length & 1) != 0)
                        throw EncodeError(encoding, s, pos, pos + 1, SurrogatesText);
                    sink.PutBytes(rep);
                }
                else
                {
                    if (!IsAscii(rep))
                        throw EncodeError(encoding, s, pos, pos + 1, SurrogatesText);
                    // ucs1lib_utf16_encode of the str: each character one unit in the codec's order.
                    sink.Reserve(rep.Length * 2L);
                    foreach (byte b in rep)
                        Put16(sink, b, big);
                }
                pos = newpos;
                sink.Reserve((n - pos) * 4);
            }
        }

        /// <summary>Writes one 16-bit unit (the value truncated to 16 bits) in the given order.</summary>
        /// <param name="d">The output.</param>
        /// <param name="c">The write position, advanced by two.</param>
        /// <param name="unit">The unit.</param>
        /// <param name="big">True for big-endian.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Write16(byte* d, ref long c, uint unit, bool big)
        {
            if (big)
            {
                d[c++] = (byte)(unit >> 8);
                d[c++] = (byte)unit;
            }
            else
            {
                d[c++] = (byte)unit;
                d[c++] = (byte)(unit >> 8);
            }
        }

        /// <summary>Appends one 16-bit unit to a byte output (reserving its room).</summary>
        /// <param name="sink">The output.</param>
        /// <param name="unit">The unit.</param>
        /// <param name="big">True for big-endian.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Put16(NDCodecSink sink, uint unit, bool big)
        {
            sink.Reserve(2);
            long c = sink.Count;
            Write16(sink.Data, ref c, unit, big);
            sink.Count = c;
        }

        /// <summary>
        ///     CPython's <c>_PyUnicode_EncodeUTF32</c> (see the file header): the BOM for the BOM codec, then every code point as
        ///     one 32-bit unit in the codec's order, each surrogate through the generic call.
        /// </summary>
        /// <param name="s">The str's code points.</param>
        /// <param name="n">Their number.</param>
        /// <param name="byteorder">0 (the BOM codec), -1 little-endian, 1 big-endian.</param>
        /// <param name="errors">The errors policy.</param>
        /// <param name="sink">The byte output.</param>
        /// <exception cref="UnicodeEncodeError">A surrogate under <c>strict</c>, a handler that re-raises, or a replacement that
        ///     does not fill whole units.</exception>
        /// <exception cref="LookupError">An unknown handler name, at the first surrogate.</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static void EncodeUtf32(uint* s, long n, int byteorder, NDCodecErrors errors, NDCodecSink sink)
        {
            string encoding = byteorder < 0 ? "utf-32-le" : byteorder > 0 ? "utf-32-be" : "utf-32";
            bool big = byteorder > 0;
            sink.Reserve((n + 1) * 4);
            if (byteorder == 0)
                Put32(sink, 0xFEFF, big: false);
            long pos = 0;
            while (pos < n)
            {
                // ucs4lib_utf32_encode: every character up to the first surrogate, units past U+10FFFF as they are.
                byte* d = sink.Data;
                long c = sink.Count;
                for (; pos < n; pos++)
                {
                    uint ch = s[pos];
                    if (ch - 0xD800u <= 0x7FFu)
                        break;
                    Write32(d, ref c, ch, big);
                }
                sink.Count = c;
                if (pos == n)
                    break;
                var rep = CallEncodeHandler(errors, encoding, SurrogatesText, s, n, pos, pos + 1, out bool isBytes, out long newpos);
                if (isBytes)
                {
                    if ((rep.Length & 3) != 0)
                        throw EncodeError(encoding, s, pos, pos + 1, SurrogatesText);
                    sink.PutBytes(rep);
                }
                else
                {
                    if (!IsAscii(rep))
                        throw EncodeError(encoding, s, pos, pos + 1, SurrogatesText);
                    sink.Reserve(rep.Length * 4L);
                    foreach (byte b in rep)
                        Put32(sink, b, big);
                }
                pos = newpos;
                sink.Reserve((n - pos) * 4);
            }
        }

        /// <summary>Writes one 32-bit unit in the given order.</summary>
        /// <param name="d">The output.</param>
        /// <param name="c">The write position, advanced by four.</param>
        /// <param name="unit">The unit.</param>
        /// <param name="big">True for big-endian.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Write32(byte* d, ref long c, uint unit, bool big)
        {
            if (big)
            {
                d[c++] = (byte)(unit >> 24);
                d[c++] = (byte)(unit >> 16);
                d[c++] = (byte)(unit >> 8);
                d[c++] = (byte)unit;
            }
            else
            {
                d[c++] = (byte)unit;
                d[c++] = (byte)(unit >> 8);
                d[c++] = (byte)(unit >> 16);
                d[c++] = (byte)(unit >> 24);
            }
        }

        /// <summary>Appends one 32-bit unit to a byte output (reserving its room).</summary>
        /// <param name="sink">The output.</param>
        /// <param name="unit">The unit.</param>
        /// <param name="big">True for big-endian.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Put32(NDCodecSink sink, uint unit, bool big)
        {
            sink.Reserve(4);
            long c = sink.Count;
            Write32(sink.Data, ref c, unit, big);
            sink.Count = c;
        }

        /// <summary>
        ///     CPython's <c>_PyUnicode_EncodeCharmap</c> with <c>charmap_encoding_error</c> (see the file header): one byte per
        ///     code point through the table, a run of unencodable characters one error.
        /// </summary>
        /// <param name="s">The str's code points.</param>
        /// <param name="n">Their number.</param>
        /// <param name="errors">The errors policy.</param>
        /// <param name="sink">The byte output.</param>
        /// <exception cref="UnicodeEncodeError">An unencodable character under <c>strict</c> (the run of them), a handler that
        ///     re-raises, or a replacement character the table cannot encode (the original error).</exception>
        /// <exception cref="LookupError">An unknown handler name, at the first unencodable character.</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal void EncodeCharmap(uint* s, long n, NDCodecErrors errors, NDCodecSink sink)
        {
            const string encoding = "charmap";
            sink.Reserve(n);
            long inpos = 0;
            while (inpos < n)
            {
                int b = CharmapByte(s[inpos]);
                if (b >= 0)
                {
                    sink.Data[sink.Count++] = (byte)b;
                    inpos++;
                    continue;
                }
                long collstart = inpos, collend = inpos + 1;
                while (collend < n && CharmapByte(s[collend]) < 0)
                    collend++;
                switch (errors.Inline)
                {
                    case NDErrorHandler.Strict:
                        throw EncodeError(encoding, s, collstart, collend, UndefinedText);
                    case NDErrorHandler.Replace:
                    {
                        // '?' through the table per character (EBCDIC writes 0x6F; a table without '?' raises).
                        int q = CharmapByte('?');
                        if (q < 0)
                            throw EncodeError(encoding, s, collstart, collend, UndefinedText);
                        for (long k = collstart; k < collend; k++)
                            sink.PutByte((byte)q);
                        inpos = collend;
                        break;
                    }
                    case NDErrorHandler.Ignore:
                        inpos = collend;
                        break;
                    case NDErrorHandler.XmlCharRefReplace:
                    {
                        // sprintf("&#%d;", (int)ch) per character, every character of it through the table.
                        byte* text = stackalloc byte[16];
                        for (long k = collstart; k < collend; k++)
                        {
                            int w = 0;
                            text[w++] = (byte)'&';
                            text[w++] = (byte)'#';
                            FormatCInt((int)s[k], text, ref w);
                            text[w++] = (byte)';';
                            for (int t = 0; t < w; t++)
                            {
                                int q = CharmapByte(text[t]);
                                if (q < 0)
                                    throw EncodeError(encoding, s, collstart, collend, UndefinedText);
                                sink.PutByte((byte)q);
                            }
                        }
                        inpos = collend;
                        break;
                    }
                    default:
                    {
                        var rep = CallEncodeHandler(errors, encoding, UndefinedText, s, n, collstart, collend, out bool isBytes,
                            out long newpos);
                        if (isBytes)
                        {
                            // A bytes replacement is copied as it is (surrogateescape's raw bytes).
                            sink.PutBytes(rep);
                        }
                        else
                        {
                            foreach (byte r in rep)
                            {
                                int q = CharmapByte(r);
                                if (q < 0)
                                    throw EncodeError(encoding, s, collstart, collend, UndefinedText);
                                sink.PutByte((byte)q);
                            }
                        }
                        inpos = newpos;
                        break;
                    }
                }
                sink.Reserve(n - inpos);
            }
        }
    }
}
