using System;
using NumSharp.Backends.Strings;

namespace NumSharp.IO
{
    internal sealed partial class TextIOCodec
    {
        /// <summary>
        ///     The codec's incremental decoder (<c>codecs.getincrementaldecoder(encoding)()</c>), which a TextIOWrapper feeds
        ///     the file's bytes chunk by chunk.
        /// </summary>
        /// <returns>A fresh decoder (each file needs its own: it carries an incomplete sequence and BOM state).</returns>
        /// <exception cref="UnicodeError">Never here; the <c>undefined</c> codec's decoder raises at its first call.</exception>
        internal TextIOIncrementalDecoder NewIncrementalDecoder() => Kind switch
        {
            PythonCodecKind.Utf8 => new Utf8IncrementalDecoder(sig: false),
            PythonCodecKind.Utf8Sig => new Utf8IncrementalDecoder(sig: true),
            PythonCodecKind.Utf16 => new UnicodeIncrementalDecoder(utf16: true, byteorder: 0),
            PythonCodecKind.Utf16LE => new UnicodeIncrementalDecoder(utf16: true, byteorder: -1),
            PythonCodecKind.Utf16BE => new UnicodeIncrementalDecoder(utf16: true, byteorder: 1),
            PythonCodecKind.Utf32 => new UnicodeIncrementalDecoder(utf16: false, byteorder: 0),
            PythonCodecKind.Utf32LE => new UnicodeIncrementalDecoder(utf16: false, byteorder: -1),
            PythonCodecKind.Utf32BE => new UnicodeIncrementalDecoder(utf16: false, byteorder: 1),
            _ => new StatelessIncrementalDecoder(this),
        };

        /// <summary>
        ///     TextIOWrapper's encoder for a file NumPy opens itself for writing (at position 0): its fast
        ///     <c>encodefunc</c> picked by the codec's name, or the codec's incremental encoder.
        /// </summary>
        /// <returns>A fresh encoder (it carries the start-of-stream state).</returns>
        internal TextIOFileEncoder NewFileEncoder() => new TextIOFileEncoder(this);

        /// <summary>
        ///     The stateless incremental decoders (<c>latin_1</c>, <c>ascii</c>, every charmap module, <c>undefined</c>):
        ///     each chunk is decoded on its own (<c>codecs.charmap_decode(input, errors, table)[0]</c>), so a decode error's
        ///     position counts bytes of that chunk.
        /// </summary>
        private sealed class StatelessIncrementalDecoder : TextIOIncrementalDecoder
        {
            private readonly TextIOCodec _codec;

            /// <summary>Wraps the codec.</summary>
            /// <param name="codec">A Latin-1, ASCII, charmap or <c>undefined</c> codec.</param>
            internal StatelessIncrementalDecoder(TextIOCodec codec) => _codec = codec;

            /// <inheritdoc/>
            internal override int MaxChars(int inputLength) => inputLength;

            /// <inheritdoc/>
            internal override int Decode(ReadOnlySpan<byte> input, bool final, Span<char> dest)
            {
                switch (_codec.Kind)
                {
                    case PythonCodecKind.Latin1:
                        return System.Text.Encoding.Latin1.GetChars(input, dest);
                    case PythonCodecKind.Ascii:
                        NDStringCodec.ThrowIfNotAscii(input);
                        return System.Text.Encoding.ASCII.GetChars(input, dest);
                    case PythonCodecKind.Charmap:
                        _codec.DecodeCharmap(input, dest);
                        return input.Length;
                    default:
                        // encodings.undefined.IncrementalDecoder.decode raises on every call, an empty final one included.
                        throw new UnicodeError("undefined encoding");
                }
            }
        }

        /// <summary>
        ///     <c>codecs.BufferedIncrementalDecoder</c>: the bytes a call could not decode yet are prepended to the next
        ///     call's input (<c>data = self.buffer + input</c>), so a decode error's position counts bytes of that
        ///     concatenation.
        /// </summary>
        private abstract class BufferedIncrementalDecoder : TextIOIncrementalDecoder
        {
            private byte[] _buffer = Array.Empty<byte>();
            private int _buffered;
            private byte[] _joined = Array.Empty<byte>();

            /// <inheritdoc/>
            internal sealed override int MaxChars(int inputLength) => _buffered + inputLength + 4;

            /// <inheritdoc/>
            internal sealed override int Decode(ReadOnlySpan<byte> input, bool final, Span<char> dest)
            {
                ReadOnlySpan<byte> data = input;
                if (_buffered > 0)
                {
                    // Only a few bytes are ever left over (an incomplete sequence, an odd byte, a partial BOM), so joining
                    // copies the chunk once; the common case — nothing left over — decodes the chunk in place.
                    int total = _buffered + input.Length;
                    if (_joined.Length < total)
                        _joined = new byte[Math.Max(total, _joined.Length * 2)];
                    _buffer.AsSpan(0, _buffered).CopyTo(_joined);
                    input.CopyTo(_joined.AsSpan(_buffered));
                    data = _joined.AsSpan(0, total);
                }
                int written = BufferDecode(data, final, dest, out int consumed);
                // self.buffer = data[consumed:]
                int rest = data.Length - consumed;
                if (rest > 0)
                {
                    if (_buffer.Length < rest)
                        _buffer = new byte[Math.Max(rest, 8)];
                    data.Slice(consumed).CopyTo(_buffer);
                }
                _buffered = rest;
                return written;
            }

            /// <summary>The codec's stateful decode function (<c>_buffer_decode(data, errors, final)</c>).</summary>
            /// <param name="data">The left-over bytes followed by the new input.</param>
            /// <param name="final">True at the end of the file.</param>
            /// <param name="dest">The characters.</param>
            /// <param name="consumed">The bytes decoded.</param>
            /// <returns>The number of characters written.</returns>
            protected abstract int BufferDecode(ReadOnlySpan<byte> data, bool final, Span<char> dest, out int consumed);
        }

        /// <summary>
        ///     The <c>utf_8</c> and <c>utf_8_sig</c> incremental decoders: <c>codecs.utf_8_decode(data, errors, final)</c>, which
        ///     keeps a valid sequence the input ends in the middle of for the next call; <c>utf_8_sig</c> first decides
        ///     whether the stream starts with a BOM (waiting while fewer than three bytes could still be one).
        /// </summary>
        private sealed class Utf8IncrementalDecoder : BufferedIncrementalDecoder
        {
            private bool _first;

            /// <summary>Creates the decoder.</summary>
            /// <param name="sig">True for <c>utf_8_sig</c>.</param>
            internal Utf8IncrementalDecoder(bool sig) => _first = sig;

            /// <inheritdoc/>
            protected override int BufferDecode(ReadOnlySpan<byte> data, bool final, Span<char> dest, out int consumed)
            {
                if (_first)
                {
                    if (data.Length < 3)
                    {
                        // BOM_UTF8.startswith(input) (true for the empty input too): not enough to decide, try again — at
                        // the end of the file the partial BOM is then silently dropped, as CPython drops it.
                        if (Utf8Bom.StartsWith(data))
                        {
                            consumed = 0;
                            return 0;
                        }
                        _first = false;
                    }
                    else
                    {
                        _first = false;
                        if (data.StartsWith(Utf8Bom))
                        {
                            int n = DecodeUtf8Stateful(data.Slice(3), final, dest, out int used);
                            consumed = used + 3;
                            return n;
                        }
                    }
                }
                return DecodeUtf8Stateful(data, final, dest, out consumed);
            }
        }

        /// <summary>
        ///     <c>codecs.utf_8_decode(data, errors, final)</c>: valid UTF-8 goes through the BCL's vectorized transcoder (which
        ///     agrees with CPython's strict codec on what is well formed); anything it refuses runs the reference decoder,
        ///     which raises CPython's exact error. A valid sequence cut off by the end of the input is left over unless final.
        /// </summary>
        /// <param name="data">The bytes.</param>
        /// <param name="final">True at the end of the input.</param>
        /// <param name="dest">The characters: at least <c>data.Length</c>.</param>
        /// <param name="consumed">The bytes decoded.</param>
        /// <returns>The number of characters written.</returns>
        /// <exception cref="UnicodeDecodeError">CPython's text, positions counting bytes of <paramref name="data"/>.</exception>
        internal static int DecodeUtf8Stateful(ReadOnlySpan<byte> data, bool final, Span<char> dest, out int consumed)
        {
            var status = System.Text.Unicode.Utf8.ToUtf16(data, dest, out int read, out int written, replaceInvalidSequences: false,
                isFinalBlock: final);
            if (status == System.Buffers.OperationStatus.Done || status == System.Buffers.OperationStatus.NeedMoreData)
            {
                consumed = read;
                return written;
            }
            // InvalidData: CPython reports an invalid sequence at once and defers only a tail the input cuts off (a valid start,
            // or the two bytes of a truncated surrogate .NET already calls invalid), so the strict decode of everything but
            // such a tail raises the first error with CPython's text — or, when the deferred tail was the only invalid data,
            // decodes everything before it.
            int cut = final ? data.Length : data.Length - IncompleteUtf8Tail(data);
            string text = NDUtf8.DecodeStrict(data.Slice(0, cut));
            text.AsSpan().CopyTo(dest);
            consumed = cut;
            return text.Length;
        }

        /// <summary>
        ///     The length of an incomplete UTF-8 sequence at the end of <paramref name="b"/> (0 when it ends on a sequence
        ///     boundary): a lead byte followed by fewer continuation bytes than it announces, every one of them valid for that
        ///     lead. A tail that is already invalid is not "incomplete" — it is left in, so the strict decode reports it now,
        ///     as CPython's incremental decoder reports an invalid continuation at once and defers only an unexpected end —
        ///     and the first two bytes of a surrogate (<c>0xED 0xA0..0xBF</c>), which CPython defers too.
        /// </summary>
        /// <param name="b">The bytes.</param>
        /// <returns>The number of trailing bytes to carry.</returns>
        private static int IncompleteUtf8Tail(ReadOnlySpan<byte> b)
        {
            // Walk back over at most three continuation bytes to the lead byte.
            for (int back = 1; back <= Math.Min(4, b.Length); back++)
            {
                byte lead = b[b.Length - back];
                if ((lead & 0xC0) == 0x80)
                    continue; // a continuation byte: keep looking for the lead
                int need = lead >= 0xF5 ? 1 : lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC2 ? 2 : 1;
                if (need <= back)
                    return 0; // complete (or not a lead byte at all): nothing to carry
                // The second byte's range depends on the lead (no overlong forms, no surrogates, nothing past U+10FFFF).
                if (back >= 2)
                {
                    byte second = b[b.Length - back + 1];
                    // unicode_decode_utf8's case 2: an incremental decode that ends on the first two bytes of a SURROGATE
                    // (0xED then 0xA0..0xBF, exactly two bytes before the end) is "a truncated surrogate code": carried
                    // like a valid start, so the error comes from the next decode, at its position 0.
                    if (lead == 0xED && back == 2 && second >= 0xA0 && second <= 0xBF)
                        return 2;
                    byte lo = lead == 0xE0 ? (byte)0xA0 : lead == 0xF0 ? (byte)0x90 : (byte)0x80;
                    byte hi = lead == 0xED ? (byte)0x9F : lead == 0xF4 ? (byte)0x8F : (byte)0xBF;
                    if (second < lo || second > hi)
                        return 0;
                }
                return back;
            }
            return 0;
        }

        /// <summary>
        ///     The UTF-16 and UTF-32 incremental decoders: an explicit byte order decodes from the first chunk (the BOM, if
        ///     any, is then an ordinary U+FEFF); the BOM codecs read one first and raise CPython's <c>UTF-16 stream does not
        ///     start with BOM</c> once a whole code unit arrived without one (<c>encodings.utf_16.IncrementalDecoder</c>).
        /// </summary>
        private sealed class UnicodeIncrementalDecoder : BufferedIncrementalDecoder
        {
            private readonly bool _utf16;
            private int _byteorder; // 0 until a BOM named the order (BOM codecs only)

            /// <summary>Creates the decoder.</summary>
            /// <param name="utf16">True for UTF-16, false for UTF-32.</param>
            /// <param name="byteorder">0 for the BOM codec, -1 little-endian, 1 big-endian.</param>
            internal UnicodeIncrementalDecoder(bool utf16, int byteorder)
            {
                _utf16 = utf16;
                _byteorder = byteorder;
            }

            /// <inheritdoc/>
            protected override int BufferDecode(ReadOnlySpan<byte> data, bool final, Span<char> dest, out int consumed)
            {
                if (_byteorder != 0)
                {
                    int fixedOrder = _byteorder;
                    return _utf16 ? DecodeUtf16(data, ref fixedOrder, final, dest, out consumed)
                                  : DecodeUtf32(data, ref fixedOrder, final, dest, out consumed);
                }
                // utf_16_ex_decode(input, errors, 0, final): reads a BOM if one is there.
                int order = 0;
                int n = _utf16 ? DecodeUtf16(data, ref order, final, dest, out consumed)
                               : DecodeUtf32(data, ref order, final, dest, out consumed);
                if (order != 0)
                    _byteorder = order;
                else if (consumed >= (_utf16 ? 2 : 4))
                    throw new UnicodeError(_utf16 ? "UTF-16 stream does not start with BOM" : "UTF-32 stream does not start with BOM");
                return n;
            }
        }
    }

    /// <summary>
    ///     A codec's incremental decoder as a TextIOWrapper drives it: <c>decoder.decode(input, final)</c> for each chunk read
    ///     from the file, <paramref name="final"/> once the file is exhausted.
    /// </summary>
    internal abstract class TextIOIncrementalDecoder
    {
        /// <summary>The most characters <see cref="Decode"/> writes for an input of <paramref name="inputLength"/> bytes
        ///     (the bytes left over from the previous call included).</summary>
        /// <param name="inputLength">The new input's length.</param>
        /// <returns>The character capacity <see cref="Decode"/> needs.</returns>
        internal abstract int MaxChars(int inputLength);

        /// <summary>Decodes the next chunk.</summary>
        /// <param name="input">The new bytes.</param>
        /// <param name="final">True at the end of the file.</param>
        /// <param name="dest">The characters: at least <see cref="MaxChars"/> of the input's length.</param>
        /// <returns>The number of characters written.</returns>
        /// <exception cref="UnicodeDecodeError">Bytes the codec cannot decode, with CPython's text.</exception>
        /// <exception cref="UnicodeError">A BOM codec whose stream does not start with a BOM; the <c>undefined</c> codec.</exception>
        internal abstract int Decode(ReadOnlySpan<byte> input, bool final, Span<char> dest);
    }

    /// <summary>
    ///     CPython's <c>IncrementalNewlineDecoder</c> with <c>translate=True</c> (a text file opened with <c>newline=None</c>):
    ///     the codec's output with <c>"\r\n"</c> and <c>'\r'</c> turned into <c>'\n'</c>, a <c>'\r'</c> that ends a
    ///     non-final chunk held back until the next one shows whether a <c>'\n'</c> follows it.
    /// </summary>
    internal sealed class TextIONewlineDecoder
    {
        private readonly TextIOIncrementalDecoder _inner;
        private bool _pendingCR;

        /// <summary>Wraps a codec's incremental decoder.</summary>
        /// <param name="inner">The decoder.</param>
        internal TextIONewlineDecoder(TextIOIncrementalDecoder inner) => _inner = inner;

        /// <summary>The character capacity <see cref="Decode"/> needs for an input of <paramref name="inputLength"/>
        ///     bytes (one more than the codec's, for a held-back <c>'\r'</c>).</summary>
        /// <param name="inputLength">The input's length.</param>
        /// <returns>The capacity.</returns>
        internal int MaxChars(int inputLength) => _inner.MaxChars(inputLength) + 1;

        /// <summary>Decodes and translates the next chunk.</summary>
        /// <param name="input">The new bytes.</param>
        /// <param name="final">True at the end of the file.</param>
        /// <param name="dest">The characters: at least <see cref="MaxChars"/> of the input's length.</param>
        /// <param name="start">The index of the first character in <paramref name="dest"/> (0 or 1: the codec writes after
        ///     a slot kept for a held-back <c>'\r'</c>).</param>
        /// <returns>The number of characters from <paramref name="start"/>.</returns>
        /// <exception cref="UnicodeDecodeError">The codec's decode errors.</exception>
        /// <exception cref="UnicodeError">The codec's BOM / undefined errors.</exception>
        internal int Decode(ReadOnlySpan<byte> input, bool final, Span<char> dest, out int start)
        {
            int n = _inner.Decode(input, final, dest.Slice(1));
            start = 1;
            // The '\r' held back by the previous call comes first, once there is output or the stream ends.
            if (_pendingCR && (n > 0 || final))
            {
                dest[0] = '\r';
                start = 0;
                n++;
                _pendingCR = false;
            }
            // Retain a trailing '\r' (unless final): the next chunk may start with its '\n'.
            if (!final && n > 0 && dest[start + n - 1] == '\r')
            {
                n--;
                _pendingCR = true;
            }
            var chars = dest.Slice(start, n);
            int first = chars.IndexOf('\r');
            if (first < 0)
                return n;
            // output.replace("\r\n", "\n").replace("\r", "\n"), in one pass: a '\r' before '\n' is dropped, any other '\r'
            // becomes '\n'. The output only shrinks, so it is rewritten in place.
            int w = first;
            for (int r = first; r < n; r++)
            {
                char c = chars[r];
                if (c == '\r')
                {
                    if (r + 1 < n && chars[r + 1] == '\n')
                        continue;
                    c = '\n';
                }
                chars[w++] = c;
            }
            return w;
        }
    }

    /// <summary>
    ///     TextIOWrapper's encoding of the writes to a file NumPy opened itself (seekable, at position 0): the fast
    ///     <c>encodefunc</c> its codec's name selects (<c>ascii</c>, <c>iso8859-1</c>, <c>utf-8</c>, the UTF-16 / UTF-32
    ///     names) or the codec's incremental encoder (<c>utf-8-sig</c>, a charmap). A BOM codec writes its BOM before the
    ///     first write's bytes only, and the UTF-16 / UTF-32 BOM codecs name themselves <c>utf-16</c> in the first write's
    ///     error and <c>utf-16-le</c> in any later one (the first write ends the start of the stream, even when it fails).
    /// </summary>
    internal sealed class TextIOFileEncoder
    {
        private readonly TextIOCodec _codec;
        private readonly string _name;
        private bool _startOfStream = true;
        private bool _bomPending;

        /// <summary>Creates the encoder.</summary>
        /// <param name="codec">The file's codec.</param>
        internal TextIOFileEncoder(TextIOCodec codec)
        {
            _codec = codec;
            _name = codec.Name;
            _bomPending = codec.WritesBomPerEncode;
        }

        /// <summary>True until the first write has been checked (TextIOWrapper's <c>encoding_start_of_stream</c>).</summary>
        internal bool StartOfStream => _startOfStream;

        /// <summary>
        ///     Raises what <c>TextIOWrapper.write(text)</c> raises for one write, and ends the start of the stream (even when
        ///     the write fails, as TextIOWrapper clears the flag whatever the encode returned).
        /// </summary>
        /// <param name="text">The write. CPython translates its newlines before it encodes, so an error's position counts
        ///     the translated text; a caller that checks the untranslated write re-checks a failing one translated
        ///     (<see cref="Check"/>).</param>
        /// <exception cref="UnicodeEncodeError">An unencodable character, with the encoder's codec name.</exception>
        /// <exception cref="UnicodeError">The <c>undefined</c> codec.</exception>
        internal void CheckWrite(ReadOnlySpan<char> text)
        {
            bool first = _startOfStream;
            _startOfStream = false;
            Check(text, first);
        }

        /// <summary>The check of <see cref="CheckWrite"/> without its state change: as the first write of the file when
        ///     <paramref name="first"/>, as a later one otherwise.</summary>
        /// <param name="text">The write.</param>
        /// <param name="first">Whether it is the file's first write (the UTF-16 / UTF-32 BOM codecs name themselves
        ///     <c>utf-16</c> then, <c>utf-16-le</c> after).</param>
        /// <exception cref="UnicodeEncodeError">An unencodable character.</exception>
        /// <exception cref="UnicodeError">The <c>undefined</c> codec.</exception>
        internal void Check(ReadOnlySpan<char> text, bool first)
        {
            switch (_name)
            {
                case "ascii":
                    NDStringCodec.ThrowIfNotAscii(text);
                    return;
                case "iso8859-1":
                    // latin1_encode: the latin_1 module and the iso8859_1 charmap share this name, so both encode as Latin-1.
                    NDStringCodec.ThrowIfNotEncodable(text, (char)0xFF, "latin-1", "ordinal not in range(256)");
                    return;
                case "utf-16":
                    _codec.CheckEncode(text, first ? "utf-16" : "utf-16-le");
                    return;
                case "utf-32":
                    _codec.CheckEncode(text, first ? "utf-32" : "utf-32-le");
                    return;
                default:
                    // utf-8, utf-16-le/-be, utf-32-le/-be (fast, fixed names) and the incremental encoders.
                    _codec.CheckEncode(text);
                    return;
            }
        }

        /// <summary>True when the file still owes its BOM (no write has been encoded yet).</summary>
        internal bool BomPending => _bomPending;

        /// <summary>The most bytes <see cref="Encode"/> writes for <paramref name="chars"/> characters (BOM included).</summary>
        /// <param name="chars">The number of characters.</param>
        /// <returns>The byte capacity.</returns>
        internal int MaxByteCount(int chars) => _codec.MaxByteCount(chars) + 4;

        /// <summary>The bytes of checked writes, after the BOM when this is the first encoded output.</summary>
        /// <param name="text">The writes (each passed <see cref="CheckWrite"/>).</param>
        /// <param name="dest">At least the codec's <see cref="TextIOCodec.MaxByteCount"/> of the text's length.</param>
        /// <returns>The number of bytes written.</returns>
        internal int Encode(ReadOnlySpan<char> text, Span<byte> dest)
        {
            bool bom = _bomPending;
            _bomPending = false;
            if (_name == "iso8859-1")
                return System.Text.Encoding.Latin1.GetBytes(text, dest); // checked as Latin-1 above
            return _codec.Encode(text, dest, bom);
        }
    }
}
