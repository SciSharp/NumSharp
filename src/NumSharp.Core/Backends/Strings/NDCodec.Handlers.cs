using System;
using System.Globalization;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Unmanaged;
using NumSharp.IO;

// =============================================================================
// NDCodec.Handlers.cs — the output buffer of the codecs, CPython's generic error-handler calls and the inline replacement
// writers (strings phase 2, P2.6 milestone 4)
// =============================================================================
//
// THE GENERIC CALLS port unicode_encode_call_errorhandler / unicode_decode_call_errorhandler_writer (Objects/unicodeobject.c)
// with the built-in handlers of Python/codecs.c in place of the Python call:
//   * the registry lookup comes FIRST (an unknown name is LookupError before any error object exists);
//   * encode: the handler returns a replacement (a str for ignore / replace / xmlcharrefreplace / backslashreplace /
//     namereplace, bytes for surrogatepass / surrogateescape) and the position to resume at; the CODEC then validates it
//     (an 8-bit codec needs an ASCII — Latin-1: one-byte — str, a charmap codec maps every character, UTF-16 / UTF-32
//     need ASCII or a byte count that fills whole units), which is why the replacement is handed back rather than written;
//   * decode: the handler's str is written as it is and the decode resumes at the returned position (which may lie BEFORE
//     the end of the bad range: surrogateescape takes at most four bytes, surrogatepass one surrogate).
// The error objects are Python's Unicode{En,De}codeError over the WHOLE input of the call (an encode's str, a decode's bytes
// — for utf-8-sig the bytes after its BOM), so the texts read their character / byte at the error's start.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The growable output of one encode or decode: bytes (<see cref="UnitSize"/> 1) or code points (4), in a native
    ///     buffer from the buffer pool. The codecs append to it; the element just coded is its last <c>Count - start</c> units.
    /// </summary>
    /// <remarks>
    ///     Owns its buffer (returned to the pool by <see cref="Dispose"/>, or replaced — and the old one returned — when it
    ///     grows). Single-threaded: one call's walk uses it. <see cref="Data"/> moves when the buffer grows, so a caller holding
    ///     a pointer into it across an append must re-read it.
    /// </remarks>
    internal sealed unsafe class NDCodecSink : IDisposable
    {
        /// <summary>The size of a unit in bytes: 1 (an encode's bytes) or 4 (a decode's code points).</summary>
        internal readonly int UnitSize;

        /// <summary>The pool buffer behind <see cref="Data"/>.</summary>
        private NDArray _buffer;

        /// <summary>The first unit (moves when the buffer grows).</summary>
        internal byte* Data;

        /// <summary>The capacity in units.</summary>
        internal long Capacity;

        /// <summary>The number of units written.</summary>
        internal long Count;

        /// <summary>Creates an empty output.</summary>
        /// <param name="unitSize">1 for bytes, 4 for code points.</param>
        /// <param name="initialUnits">The first capacity (at least 64 units are allocated).</param>
        /// <exception cref="OutOfMemoryException">The buffer cannot be allocated.</exception>
        internal NDCodecSink(int unitSize, long initialUnits)
        {
            UnitSize = unitSize;
            Grow(Math.Max(64, initialUnits), exact: true);
        }

        /// <summary>Makes room for <paramref name="units"/> more units.</summary>
        /// <param name="units">The number of units about to be written.</param>
        /// <exception cref="OutOfMemoryException">The buffer cannot grow.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void Reserve(long units)
        {
            if (Count + units > Capacity)
                Grow(Count + units, exact: false);
        }

        /// <summary>Grows the buffer to at least <paramref name="units"/> units, keeping what was written (doubling unless
        ///     <paramref name="exact"/>, so appends cost O(total)).</summary>
        /// <param name="units">The capacity needed.</param>
        /// <param name="exact">Allocate exactly <paramref name="units"/> (the first allocation).</param>
        /// <exception cref="OutOfMemoryException">The buffer cannot be allocated.</exception>
        private void Grow(long units, bool exact)
        {
            long cap = exact ? units : Math.Max(units, Capacity * 2);
            var grown = new NDArray(NPTypeCode.Byte, Shape.Vector(cap * UnitSize), false);
            byte* p = (byte*)grown.Storage.Address;
            if (Count > 0)
                Buffer.MemoryCopy(Data, p, Count * UnitSize, Count * UnitSize);
            _buffer?.Dispose();
            _buffer = grown;
            Data = p;
            Capacity = cap;
        }

        /// <summary>Appends one byte (a byte output).</summary>
        /// <param name="b">The byte.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void PutByte(byte b)
        {
            Reserve(1);
            Data[Count++] = b;
        }

        /// <summary>Appends bytes (a byte output).</summary>
        /// <param name="src">The bytes.</param>
        internal void PutBytes(ReadOnlySpan<byte> src)
        {
            Reserve(src.Length);
            src.CopyTo(new Span<byte>(Data + Count, src.Length));
            Count += src.Length;
        }

        /// <summary>Appends one code point (a code-point output).</summary>
        /// <param name="cp">The code point.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void PutPoint(uint cp)
        {
            Reserve(1);
            ((uint*)Data)[Count++] = cp;
        }

        /// <summary>Appends ASCII bytes as code points (a code-point output: a decode handler's replacement).</summary>
        /// <param name="ascii">The bytes.</param>
        internal void PutAsciiPoints(ReadOnlySpan<byte> ascii)
        {
            Reserve(ascii.Length);
            uint* dst = (uint*)Data + Count;
            for (int i = 0; i < ascii.Length; i++)
                dst[i] = ascii[i];
            Count += ascii.Length;
        }

        /// <summary>Returns the buffer to the pool.</summary>
        public void Dispose()
        {
            _buffer?.Dispose();
            _buffer = null;
            Data = null;
            Capacity = Count = 0;
        }
    }

    internal sealed unsafe partial class NDCodec
    {
        /// <summary>CPython's lowercase hex digits (<c>Py_hexdigits</c>).</summary>
        private static ReadOnlySpan<byte> HexDigits => "0123456789abcdef"u8;

        /// <summary>The replacement scratch of a generic encode call (per thread: the engine runs on its caller's thread).</summary>
        [ThreadStatic] private static byte[] t_replacement;

        /// <summary>The replacement scratch with room for <paramref name="bytes"/> bytes.</summary>
        /// <param name="bytes">The capacity needed.</param>
        /// <returns>The scratch.</returns>
        private static byte[] Replacement(long bytes)
        {
            var r = t_replacement;
            if (r is null || r.Length < bytes)
                t_replacement = r = new byte[Math.Max(bytes, Math.Max(256, (r?.Length ?? 0) * 2L))];
            return r;
        }

        /// <summary>
        ///     CPython's <c>UnicodeEncodeError</c> over a str (its text, <see cref="NDStringCodec.EncodeErrorFor(uint, long, long, string, string)"/>):
        ///     one character prints as <c>'\xNN'</c> / <c>'\uNNNN'</c> / <c>'\UNNNNNNNN'</c>, a longer range by positions.
        /// </summary>
        /// <param name="encoding">The codec name the error quotes.</param>
        /// <param name="s">The str's code points.</param>
        /// <param name="start">The range's first character.</param>
        /// <param name="end">One past its last.</param>
        /// <param name="reason">CPython's reason text.</param>
        /// <returns>The exception (not thrown).</returns>
        internal static UnicodeEncodeError EncodeError(string encoding, uint* s, long start, long end, string reason)
            => NDStringCodec.EncodeErrorFor(s[start], start, end, encoding, reason);

        /// <summary>
        ///     CPython's <c>UnicodeDecodeError</c> over bytes (<see cref="TextIOCodec.DecodeError"/>): one byte prints as
        ///     <c>byte 0xNN in position P</c>, a longer range by positions.
        /// </summary>
        /// <param name="encoding">The codec name the error quotes.</param>
        /// <param name="s">The bytes the codec decodes.</param>
        /// <param name="size">Their number.</param>
        /// <param name="start">The range's first byte.</param>
        /// <param name="end">One past its last.</param>
        /// <param name="reason">CPython's reason text.</param>
        /// <returns>The exception (not thrown).</returns>
        internal static UnicodeDecodeError DecodeError(string encoding, byte* s, long size, long start, long end, string reason)
            => TextIOCodec.DecodeError(encoding, new ReadOnlySpan<byte>(s, checked((int)size)), checked((int)start), checked((int)end), reason);

        /// <summary>
        ///     CPython's <c>get_standard_encoding</c> over the codec names the encoders and decoders here report: the byte
        ///     length and order <c>surrogatepass</c> writes or reads a surrogate in, or 0 for a codec it does not support.
        /// </summary>
        /// <param name="encoding">The error's codec name.</param>
        /// <param name="bigEndian">Receives whether a UTF-16 / UTF-32 unit is big-endian (the BOM codecs are native: little).</param>
        /// <returns>3 (UTF-8), 2 (UTF-16), 4 (UTF-32), or 0 (Latin-1, ASCII, charmap: the error is re-raised).</returns>
        private static int StandardEncoding(string encoding, out bool bigEndian)
        {
            bigEndian = encoding is "utf-16-be" or "utf-32-be";
            return encoding switch
            {
                "utf-8" => 3,
                "utf-16" or "utf-16-le" or "utf-16-be" => 2,
                "utf-32" or "utf-32-le" or "utf-32-be" => 4,
                _ => 0,
            };
        }

        /// <summary>
        ///     The generic error call of an ENCODER (<c>unicode_encode_call_errorhandler</c> with the built-in handler): the
        ///     registry lookup, then the handler over the characters <c>[start, end)</c> of the str — a replacement the codec
        ///     must validate, and the position to resume at.
        /// </summary>
        /// <param name="errors">The call's errors policy.</param>
        /// <param name="encoding">The codec name the error quotes (<c>utf-8</c>, <c>latin-1</c>, <c>charmap</c>, <c>utf-16-le</c> …).</param>
        /// <param name="reason">CPython's reason text.</param>
        /// <param name="s">The str's code points.</param>
        /// <param name="n">Their number.</param>
        /// <param name="start">The range's first character.</param>
        /// <param name="end">One past its last.</param>
        /// <param name="isBytes">Receives true for a bytes replacement (written as it is), false for an ASCII str (each
        ///     character one byte of the result span: the codec encodes it).</param>
        /// <param name="newpos">Receives the position to resume at (<paramref name="end"/> for every built-in handler).</param>
        /// <returns>The replacement (valid until the next generic call on this thread).</returns>
        /// <exception cref="LookupError">An unknown handler name.</exception>
        /// <exception cref="UnicodeEncodeError"><c>strict</c>; <c>surrogatepass</c> for a codec that is not UTF-8 / 16 / 32 or a
        ///     non-surrogate in the range; <c>surrogateescape</c> for a character outside U+DC80..U+DCFF.</exception>
        internal static ReadOnlySpan<byte> CallEncodeHandler(NDCodecErrors errors, string encoding, string reason, uint* s, long n,
            long start, long end, out bool isBytes, out long newpos)
        {
            isBytes = false;
            newpos = end;
            switch (errors.Generic)
            {
                case NDGenericHandler.Strict:
                    throw EncodeError(encoding, s, start, end, reason);
                case NDGenericHandler.Ignore:
                    return ReadOnlySpan<byte>.Empty;
                case NDGenericHandler.Replace:
                {
                    // PyCodec_ReplaceErrors: one '?' per character of the range.
                    var r = Replacement(end - start);
                    r.AsSpan(0, checked((int)(end - start))).Fill((byte)'?');
                    return r.AsSpan(0, (int)(end - start));
                }
                case NDGenericHandler.XmlCharRefReplace:
                {
                    // PyCodec_XMLCharRefReplaceErrors: "&#" + the decimal digits + ";" per character (CPython's digit loop,
                    // capped at seven digits — only surrogates reach it, the 8-bit codecs format theirs inline).
                    var r = Replacement((end - start) * 10);
                    int w = 0;
                    for (long i = start; i < end; i++)
                    {
                        uint ch = s[i];
                        r[w++] = (byte)'&';
                        r[w++] = (byte)'#';
                        int digits;
                        uint @base;
                        if (ch < 10) { digits = 1; @base = 1; }
                        else if (ch < 100) { digits = 2; @base = 10; }
                        else if (ch < 1000) { digits = 3; @base = 100; }
                        else if (ch < 10000) { digits = 4; @base = 1000; }
                        else if (ch < 100000) { digits = 5; @base = 10000; }
                        else if (ch < 1000000) { digits = 6; @base = 100000; }
                        else { digits = 7; @base = 1000000; }
                        while (digits-- > 0)
                        {
                            // Py_UCS1 arithmetic: a leading quotient past 9 (a unit past 9,999,999) wraps as a byte does.
                            r[w++] = (byte)('0' + ch / @base);
                            ch %= @base;
                            @base /= 10;
                        }
                        r[w++] = (byte)';';
                    }
                    return r.AsSpan(0, w);
                }
                case NDGenericHandler.BackslashReplace:
                {
                    var r = Replacement((end - start) * 10);
                    int w = 0;
                    for (long i = start; i < end; i++)
                        w = WriteBackslash(s[i], r, w);
                    return r.AsSpan(0, w);
                }
                case NDGenericHandler.NameReplace:
                {
                    // PyCodec_NameReplaceErrors: \N{NAME} where unicodedata names the character (aliases and named sequences
                    // included), the backslash form otherwise.
                    var r = Replacement((end - start) * (PyUnicodeNames.LongestName + 4));
                    int w = 0;
                    for (long i = start; i < end; i++)
                    {
                        uint ch = s[i];
                        int len = PyUnicodeNames.GetName(ch, r.AsSpan(w + 3));
                        if (len > 0)
                        {
                            r[w] = (byte)'\\';
                            r[w + 1] = (byte)'N';
                            r[w + 2] = (byte)'{';
                            w += 3 + len;
                            r[w++] = (byte)'}';
                        }
                        else
                        {
                            w = WriteBackslash(ch, r, w);
                        }
                    }
                    return r.AsSpan(0, w);
                }
                case NDGenericHandler.SurrogatePass:
                {
                    int bytesPer = StandardEncoding(encoding, out bool big);
                    if (bytesPer == 0)
                        throw EncodeError(encoding, s, start, end, reason);
                    var r = Replacement((end - start) * bytesPer);
                    int w = 0;
                    for (long i = start; i < end; i++)
                    {
                        uint ch = s[i];
                        if (ch - 0xD800u > 0x7FFu)
                            throw EncodeError(encoding, s, start, end, reason);   // not a surrogate: the original error
                        switch (bytesPer)
                        {
                            case 3:
                                r[w++] = (byte)(0xE0 | (ch >> 12));
                                r[w++] = (byte)(0x80 | ((ch >> 6) & 0x3F));
                                r[w++] = (byte)(0x80 | (ch & 0x3F));
                                break;
                            case 2:
                                r[w++] = (byte)(big ? ch >> 8 : ch);
                                r[w++] = (byte)(big ? ch : ch >> 8);
                                break;
                            default:
                                if (big)
                                {
                                    r[w++] = 0;
                                    r[w++] = 0;
                                    r[w++] = (byte)(ch >> 8);
                                    r[w++] = (byte)ch;
                                }
                                else
                                {
                                    r[w++] = (byte)ch;
                                    r[w++] = (byte)(ch >> 8);
                                    r[w++] = 0;
                                    r[w++] = 0;
                                }
                                break;
                        }
                    }
                    isBytes = true;
                    return r.AsSpan(0, w);
                }
                default:
                {
                    // PyCodec_SurrogateEscapeErrors (encode): U+DC80..U+DCFF are the bytes 0x80..0xFF; anything else in the
                    // range re-raises the original error.
                    var r = Replacement(end - start);
                    int w = 0;
                    for (long i = start; i < end; i++)
                    {
                        uint ch = s[i];
                        if (ch < 0xDC80 || ch > 0xDCFF)
                            throw EncodeError(encoding, s, start, end, reason);
                        r[w++] = (byte)(ch - 0xDC00);
                    }
                    isBytes = true;
                    return r.AsSpan(0, w);
                }
            }
        }

        /// <summary>
        ///     The backslash form of one character (<c>PyCodec_BackslashReplaceErrors</c> and the inline
        ///     <c>backslashreplace</c>): <c>\UXXXXXXXX</c> from U+10000 (all eight hex digits of the 32-bit unit), <c>\uXXXX</c>
        ///     from U+0100, <c>\xXX</c> below; lowercase hex.
        /// </summary>
        /// <param name="ch">The character.</param>
        /// <param name="r">The destination.</param>
        /// <param name="w">Where to write.</param>
        /// <returns>The position after the form.</returns>
        private static int WriteBackslash(uint ch, Span<byte> r, int w)
        {
            r[w++] = (byte)'\\';
            if (ch >= 0x10000)
            {
                r[w++] = (byte)'U';
                r[w++] = HexDigits[(int)((ch >> 28) & 0xF)];
                r[w++] = HexDigits[(int)((ch >> 24) & 0xF)];
                r[w++] = HexDigits[(int)((ch >> 20) & 0xF)];
                r[w++] = HexDigits[(int)((ch >> 16) & 0xF)];
                r[w++] = HexDigits[(int)((ch >> 12) & 0xF)];
                r[w++] = HexDigits[(int)((ch >> 8) & 0xF)];
            }
            else if (ch >= 0x100)
            {
                r[w++] = (byte)'u';
                r[w++] = HexDigits[(int)((ch >> 12) & 0xF)];
                r[w++] = HexDigits[(int)((ch >> 8) & 0xF)];
            }
            else
            {
                r[w++] = (byte)'x';
            }
            r[w++] = HexDigits[(int)((ch >> 4) & 0xF)];
            r[w++] = HexDigits[(int)(ch & 0xF)];
            return w;
        }

        /// <summary>
        ///     The inline <c>backslashreplace</c> of the 8-bit encoders (<c>unicode_encode_ucs1</c>, the UTF-8 encoder): the
        ///     backslash form of each character of <c>[start, end)</c>, appended to a byte output.
        /// </summary>
        /// <param name="s">The str's code points.</param>
        /// <param name="start">The first character.</param>
        /// <param name="end">One past the last.</param>
        /// <param name="sink">The byte output.</param>
        internal static void PutBackslashReplace(uint* s, long start, long end, NDCodecSink sink)
        {
            sink.Reserve((end - start) * 10);
            var dst = new Span<byte>(sink.Data + sink.Count, checked((int)((end - start) * 10)));
            int w = 0;
            for (long i = start; i < end; i++)
                w = WriteBackslash(s[i], dst, w);
            sink.Count += w;
        }

        /// <summary>
        ///     The inline <c>xmlcharrefreplace</c> of the 8-bit encoders: <c>sprintf("&amp;#%d;", ch)</c> per character of
        ///     <c>[start, end)</c> — the unit read as a C <c>int</c>, so a unit past 0x7FFFFFFF prints negative — appended to a
        ///     byte output.
        /// </summary>
        /// <param name="s">The str's code points.</param>
        /// <param name="start">The first character.</param>
        /// <param name="end">One past the last.</param>
        /// <param name="sink">The byte output.</param>
        internal static void PutXmlCharRefReplace(uint* s, long start, long end, NDCodecSink sink)
        {
            // "&#" + at most 11 characters of a C int ("-2147483648") + ";".
            sink.Reserve((end - start) * 14);
            for (long i = start; i < end; i++)
            {
                byte* p = sink.Data + sink.Count;
                p[0] = (byte)'&';
                p[1] = (byte)'#';
                int w = 2;
                FormatCInt((int)s[i], p, ref w);
                p[w++] = (byte)';';
                sink.Count += w;
            }
        }

        /// <summary>The decimal text of a C <c>int</c> (<c>printf("%d")</c>), written as ASCII bytes.</summary>
        /// <param name="v">The value.</param>
        /// <param name="p">The destination.</param>
        /// <param name="w">The write position, advanced past the text.</param>
        internal static void FormatCInt(int v, byte* p, ref int w)
        {
            if (v == 0)
            {
                p[w++] = (byte)'0';
                return;
            }
            // The digits of |v| as an unsigned value (int.MinValue has no positive int), most significant first.
            uint u;
            if (v < 0)
            {
                p[w++] = (byte)'-';
                u = (uint)(-(long)v);
            }
            else
            {
                u = (uint)v;
            }
            Span<byte> digits = stackalloc byte[10];
            int d = 0;
            while (u != 0)
            {
                digits[d++] = (byte)('0' + u % 10);
                u /= 10;
            }
            while (d > 0)
                p[w++] = digits[--d];
        }

        /// <summary>
        ///     The generic error call of a DECODER (<c>unicode_decode_call_errorhandler_writer</c> with the built-in handler):
        ///     the registry lookup, then the handler over the bytes <c>[start, end)</c> — its str appended to the code-point
        ///     output — and the position the decode resumes at.
        /// </summary>
        /// <param name="errors">The call's errors policy.</param>
        /// <param name="encoding">The codec name the error quotes (<c>utf-8</c>, <c>ascii</c>, <c>charmap</c>, <c>utf-16-le</c> …).</param>
        /// <param name="reason">CPython's reason text.</param>
        /// <param name="s">The bytes being decoded (the error object's whole input).</param>
        /// <param name="size">Their number.</param>
        /// <param name="start">The range's first byte.</param>
        /// <param name="end">One past its last.</param>
        /// <param name="sink">The code-point output.</param>
        /// <returns>The position to resume at.</returns>
        /// <exception cref="LookupError">An unknown handler name.</exception>
        /// <exception cref="UnicodeDecodeError"><c>strict</c>; <c>surrogatepass</c> without a surrogate at the range's start (or for
        ///     a codec that is not UTF-8 / 16 / 32); <c>surrogateescape</c> at an ASCII byte.</exception>
        /// <exception cref="TypeError"><c>xmlcharrefreplace</c> / <c>namereplace</c>: <c>don't know how to handle
        ///     UnicodeDecodeError in error callback</c>.</exception>
        internal static long CallDecodeHandler(NDCodecErrors errors, string encoding, string reason, byte* s, long size, long start,
            long end, NDCodecSink sink)
        {
            switch (errors.Generic)
            {
                case NDGenericHandler.Strict:
                    throw DecodeError(encoding, s, size, start, end, reason);
                case NDGenericHandler.Ignore:
                    return end;
                case NDGenericHandler.Replace:
                    sink.PutPoint(0xFFFD);
                    return end;
                case NDGenericHandler.BackslashReplace:
                {
                    // \xNN per byte of the range.
                    sink.Reserve((end - start) * 4);
                    uint* dst = (uint*)sink.Data + sink.Count;
                    for (long i = start; i < end; i++, dst += 4)
                    {
                        byte b = s[i];
                        dst[0] = '\\';
                        dst[1] = 'x';
                        dst[2] = HexDigits[b >> 4];
                        dst[3] = HexDigits[b & 0xF];
                    }
                    sink.Count += (end - start) * 4;
                    return end;
                }
                case NDGenericHandler.XmlCharRefReplace:
                case NDGenericHandler.NameReplace:
                    // wrong_exception_type: these handlers only know UnicodeEncodeError.
                    throw new TypeError("don't know how to handle UnicodeDecodeError in error callback");
                case NDGenericHandler.SurrogatePass:
                {
                    int bytesPer = StandardEncoding(encoding, out bool big);
                    if (bytesPer == 0)
                        throw DecodeError(encoding, s, size, start, end, reason);
                    // One surrogate at the range's start, read from the whole input (the range may be shorter than its bytes).
                    uint ch = 0;
                    byte* p = s + start;
                    if (size - start >= bytesPer)
                    {
                        switch (bytesPer)
                        {
                            case 3:
                                if ((p[0] & 0xF0) == 0xE0 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80)
                                    ch = (uint)(((p[0] & 0x0F) << 12) + ((p[1] & 0x3F) << 6) + (p[2] & 0x3F));
                                break;
                            case 2:
                                ch = big ? (uint)(p[0] << 8 | p[1]) : (uint)(p[1] << 8 | p[0]);
                                break;
                            default:
                                ch = big
                                    ? (uint)p[0] << 24 | (uint)p[1] << 16 | (uint)p[2] << 8 | p[3]
                                    : (uint)p[3] << 24 | (uint)p[2] << 16 | (uint)p[1] << 8 | p[0];
                                break;
                        }
                    }
                    if (ch - 0xD800u > 0x7FFu)
                        throw DecodeError(encoding, s, size, start, end, reason);
                    sink.PutPoint(ch);
                    return start + bytesPer;
                }
                default:
                {
                    // PyCodec_SurrogateEscapeErrors (decode): at most four bytes >= 0x80 from the range's start become
                    // U+DC80..U+DCFF; an ASCII byte at the start re-raises.
                    int consumed = 0;
                    while (consumed < 4 && consumed < end - start)
                    {
                        byte b = s[start + consumed];
                        if (b < 0x80)
                            break;
                        sink.PutPoint(0xDC00u + b);
                        consumed++;
                    }
                    if (consumed == 0)
                        throw DecodeError(encoding, s, size, start, end, reason);
                    return start + consumed;
                }
            }
        }
    }
}
