using System;
using System.Collections.Concurrent;
using System.Text;
using NumSharp.Backends.Strings;

namespace NumSharp.IO
{
    /// <summary>
    ///     A Python text codec as <c>np.loadtxt</c> / <c>np.savetxt</c> use it: CPython 3.12's codec of that name, strict
    ///     (Python's default <c>errors='strict'</c>), with CPython's error texts and its byte-order-mark rules.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>What NumPy calls.</b> NumPy's text I/O never decodes itself: a line of a byte stream is
    ///     <c>bytes.decode(encoding)</c> (<c>PyUnicode_FromEncodedObject</c>, <see cref="Decode"/>), a write to a byte
    ///     stream is <c>str.encode(encoding)</c> (savetxt's <c>WriteWrap</c>, <see cref="CheckEncode(ReadOnlySpan{char})"/>
    ///     / <see cref="Encode"/> — a BOM codec writes a BOM on EVERY write), and a file NumPy opens from a name is a
    ///     <c>TextIOWrapper</c> over the codec's incremental decoder (<see cref="NewIncrementalDecoder"/>) or its encoder
    ///     (<see cref="NewFileEncoder"/> — a BOM once, at the start of the file).
    ///     </para>
    ///     <para>
    ///     <b>Which codecs.</b> Exact: UTF-8 (and <c>utf-8-sig</c>), UTF-16 and UTF-32 (with and without a BOM, either byte
    ///     order), Latin-1, ASCII, <c>undefined</c> and every single-byte charmap codec of the <c>encodings</c> package
    ///     (<c>cp1252</c>, <c>iso8859_15</c>, <c>koi8_r</c>, <c>mac_roman</c>, … — 70 tables generated from CPython). A
    ///     name is resolved the way <c>codecs.lookup</c> resolves it (<see cref="PythonCodecs"/>), so every alias CPython
    ///     accepts works (<c>"Windows-1252"</c>, <c>"utf_16le"</c>, <c>"L1"</c>). The multi-byte codecs (Shift-JIS, GBK,
    ///     Big5, the ISO-2022 family, …), <c>utf-7</c>, the escape codecs and IDNA are known by name and refused with
    ///     <see cref="NotSupportedException"/>; a bytes-to-bytes codec (<c>base64</c>, <c>hex</c>, …) is CPython's
    ///     <see cref="LookupError"/>.
    ///     </para>
    ///     <para>
    ///     <b>Defaults.</b> NumPy reads and writes a binary file object (NumSharp's <see cref="System.IO.Stream"/>) with
    ///     Latin-1 when no encoding is given — <c>loadtxt</c>'s <c>getattr(fname, 'encoding', 'latin1')</c> and
    ///     <c>savetxt</c>'s <c>WriteWrap(fh, encoding or 'latin1')</c> — and a file NAME with the locale's encoding; NumSharp
    ///     pins that one to UTF-8 (the Linux / macOS answer; Windows NumPy uses the ANSI code page there).
    ///     </para>
    /// </remarks>
    internal sealed partial class TextIOCodec
    {
        /// <summary>CPython's strict UTF-8 codec (the <c>utf_8</c> module).</summary>
        internal static readonly TextIOCodec Utf8 = new(PythonCodecKind.Utf8, "utf-8");

        /// <summary>CPython's Latin-1 codec (the <c>latin_1</c> module; <c>codecs.lookup</c> names it <c>iso8859-1</c>).</summary>
        internal static readonly TextIOCodec Latin1 = new(PythonCodecKind.Latin1, "iso8859-1");

        /// <summary>CPython's strict ASCII codec.</summary>
        internal static readonly TextIOCodec Ascii = new(PythonCodecKind.Ascii, "ascii");

        /// <summary>UTF-16 with a BOM (the <c>utf_16</c> module): the C fast paths' <c>utf16</c>.</summary>
        private static readonly TextIOCodec Utf16 = new(PythonCodecKind.Utf16, "utf-16");

        /// <summary>UTF-32 with a BOM (the <c>utf_32</c> module): the C fast paths' <c>utf32</c>.</summary>
        private static readonly TextIOCodec Utf32 = new(PythonCodecKind.Utf32, "utf-32");

        /// <summary>One codec per <c>encodings</c> module, created at its first lookup (a charmap builds its tables once).</summary>
        private static readonly ConcurrentDictionary<string, TextIOCodec> ByModule = new(StringComparer.Ordinal)
        {
            ["utf_8"] = Utf8,
            ["latin_1"] = Latin1,
            ["ascii"] = Ascii,
            ["utf_16"] = Utf16,
            ["utf_32"] = Utf32,
        };

        /// <summary>The codec's kind.</summary>
        internal PythonCodecKind Kind { get; }

        /// <summary>The name <c>codecs.lookup</c> reports (<c>utf-8</c>, <c>iso8859-1</c>, <c>cp1252</c>): what TextIOWrapper
        ///     picks its fast encoder by.</summary>
        internal string Name { get; }

        // A charmap codec's decoding table (256 characters, U+FFFE for a byte it cannot decode), and its inverse: the byte of
        // each encodable BMP character (the last byte wins where two decode to one character) with a bit per character
        // saying whether it is encodable at all.
        private readonly string _table;
        private byte[] _encodeBytes;
        private ulong[] _encodable;

        private TextIOCodec(PythonCodecKind kind, string name, string table = null)
        {
            Kind = kind;
            Name = name;
            _table = table;
        }

        /// <summary>
        ///     The codec of an encoding name in a context, or <paramref name="default"/> for no name: null (NumPy's
        ///     <c>None</c>), and for a file NumPy opens itself also <c>'locale'</c> (Python's spelling of the locale encoding,
        ///     which only <c>open()</c> understands).
        /// </summary>
        /// <param name="name">The name as given, or null.</param>
        /// <param name="default">The codec when none is named (Latin-1 for a byte stream, UTF-8 for a file name).</param>
        /// <param name="context">Where the codec is looked up.</param>
        /// <returns>The codec.</returns>
        /// <exception cref="LookupError">An unknown name (<c>unknown encoding: name</c>, the empty name included) or a
        ///     codec that is not a text encoding.</exception>
        /// <exception cref="NotSupportedException">A Python text codec NumSharp does not implement.</exception>
        internal static TextIOCodec Resolve(string name, TextIOCodec @default, CodecLookupContext context)
        {
            if (name == null)
                return @default;
            if (context == CodecLookupContext.Open && name == "locale")
                return @default;
            return Lookup(name, context);
        }

        /// <summary>
        ///     CPython's lookup of a codec by name: the C fast paths for a one-shot decode or encode, then
        ///     <c>codecs.lookup</c> (<see cref="PythonCodecs.Normalize"/>, the alias table, the module), then the text-encoding
        ///     check of the caller (<c>_PyCodec_LookupTextEncoding</c>).
        /// </summary>
        /// <param name="name">The name as given.</param>
        /// <param name="context">Where the codec is looked up.</param>
        /// <returns>The codec.</returns>
        /// <exception cref="LookupError"><c>unknown encoding: name</c> (the name as given), or <c>'base64' is not a text
        ///     encoding; use codecs.decode() to handle arbitrary codecs</c> (<c>codecs.encode()</c> / <c>codecs.open()</c> by
        ///     context).</exception>
        /// <exception cref="NotSupportedException">A known codec NumSharp does not implement.</exception>
        internal static TextIOCodec Lookup(string name, CodecLookupContext context)
        {
            string normalized = PythonCodecs.Normalize(name);
            if (context != CodecLookupContext.Open && PythonCodecs.TryFastPath(normalized, out PythonCodecKind fast))
            {
                return fast switch
                {
                    PythonCodecKind.Utf8 => Utf8,
                    PythonCodecKind.Utf16 => Utf16,
                    PythonCodecKind.Utf32 => Utf32,
                    PythonCodecKind.Ascii => Ascii,
                    _ => Latin1,
                };
            }
            string module = PythonCodecs.FindModule(normalized) ?? throw new LookupError($"unknown encoding: {name}");
            var (kind, codecName) = PythonCodecs.Modules[module];
            switch (kind)
            {
                case PythonCodecKind.NotText:
                {
                    string advice = context switch
                    {
                        CodecLookupContext.Decode => "codecs.decode()",
                        CodecLookupContext.Encode => "codecs.encode()",
                        _ => "codecs.open()",
                    };
                    throw new LookupError($"'{name}' is not a text encoding; use {advice} to handle arbitrary codecs");
                }
                case PythonCodecKind.Unsupported:
                    throw new NotSupportedException(
                        $"NumSharp's text I/O does not implement Python's '{codecName}' codec (named by '{name}'): it supports " +
                        "UTF-8, UTF-16, UTF-32 (with or without a BOM), Latin-1, ASCII and the single-byte code pages (cp1252, " +
                        "iso8859_15, koi8_r, …).");
            }
            return ByModule.GetOrAdd(module, static m =>
            {
                var (k, n) = PythonCodecs.Modules[m];
                return new TextIOCodec(k, n, k == PythonCodecKind.Charmap ? PythonCodecs.CharmapTables[m] : null);
            });
        }

        /// <summary>
        ///     <c>bytes.decode(encoding)</c>: the whole input decoded at once (final), strictly. A BOM codec reads its BOM
        ///     (UTF-16 / UTF-32: the order it names, native little-endian without one; <c>utf-8-sig</c>: dropped when present).
        /// </summary>
        /// <param name="bytes">The bytes.</param>
        /// <returns>The text.</returns>
        /// <exception cref="UnicodeDecodeError">Bytes the codec cannot decode, with CPython's text (positions count bytes of
        ///     <paramref name="bytes"/>; for <c>utf-8-sig</c>, of the bytes after a BOM).</exception>
        /// <exception cref="UnicodeError">The <c>undefined</c> codec.</exception>
        internal string Decode(ReadOnlySpan<byte> bytes)
        {
            switch (Kind)
            {
                case PythonCodecKind.Utf8:
                    return NDUtf8.DecodeStrict(bytes);
                case PythonCodecKind.Utf8Sig:
                    return NDUtf8.DecodeStrict(bytes.StartsWith(Utf8Bom) ? bytes.Slice(3) : bytes);
                case PythonCodecKind.Latin1:
                    return Encoding.Latin1.GetString(bytes);
                case PythonCodecKind.Ascii:
                    NDStringCodec.ThrowIfNotAscii(bytes);
                    return Encoding.ASCII.GetString(bytes);
                case PythonCodecKind.Charmap:
                {
                    var chars = bytes.Length <= 512 ? stackalloc char[bytes.Length] : new char[bytes.Length];
                    DecodeCharmap(bytes, chars);
                    return new string(chars);
                }
                case PythonCodecKind.Undefined:
                    throw new UnicodeError("undefined encoding");
                default:
                {
                    // UTF-16 / UTF-32: CPython's stateful decoders with final=True (consumed is NULL), the order given or read.
                    int order = UnicodeByteOrder(Kind);
                    var dest = new char[bytes.Length / 2 + 1];
                    int n = IsUtf16(Kind)
                        ? DecodeUtf16(bytes, ref order, final: true, dest, out _)
                        : DecodeUtf32(bytes, ref order, final: true, dest, out _);
                    return new string(dest, 0, n);
                }
            }
        }

        /// <summary>
        ///     Raises what <c>str.encode(encoding)</c> raises for <paramref name="text"/> — the codec's first unencodable
        ///     character (or run of them), at its code-point position in <paramref name="text"/>.
        /// </summary>
        /// <param name="text">The text one write would encode.</param>
        /// <exception cref="UnicodeEncodeError">An unencodable character: <c>'latin-1' codec can't encode character
        ///     'Ā' in position 0: ordinal not in range(256)</c>, <c>'charmap' codec can't encode characters in position
        ///     1-2: character maps to &lt;undefined&gt;</c>, <c>'utf-16' codec can't encode character '\ud800' in position 1:
        ///     surrogates not allowed</c>.</exception>
        /// <exception cref="UnicodeError">The <c>undefined</c> codec.</exception>
        internal void CheckEncode(ReadOnlySpan<char> text) => CheckEncode(text, OneShotErrorName(Kind));

        /// <summary>The same check, reporting <paramref name="codecName"/> in a UTF-16 / UTF-32 error (TextIOWrapper's
        ///     encoders name <c>utf-16</c> on a file's first write and <c>utf-16-le</c> after it).</summary>
        /// <param name="text">The text.</param>
        /// <param name="codecName">The codec name a UTF-16 / UTF-32 error quotes.</param>
        /// <exception cref="UnicodeEncodeError">An unencodable character.</exception>
        /// <exception cref="UnicodeError">The <c>undefined</c> codec.</exception>
        internal void CheckEncode(ReadOnlySpan<char> text, string codecName)
        {
            switch (Kind)
            {
                case PythonCodecKind.Utf8:
                case PythonCodecKind.Utf8Sig:
                    NDUtf8.GetByteCount(text); // a run of lone surrogates, CPython's text and positions
                    return;
                case PythonCodecKind.Latin1:
                    NDStringCodec.ThrowIfNotEncodable(text, (char)0xFF, "latin-1", "ordinal not in range(256)");
                    return;
                case PythonCodecKind.Ascii:
                    NDStringCodec.ThrowIfNotAscii(text);
                    return;
                case PythonCodecKind.Charmap:
                    CheckCharmapEncode(text);
                    return;
                case PythonCodecKind.Undefined:
                    throw new UnicodeError("undefined encoding");
                default:
                    // UTF-16 / UTF-32 encode every code point but a lone surrogate, reported one at a time.
                    CheckNoLoneSurrogate(text, codecName);
                    return;
            }
        }

        /// <summary>
        ///     The bytes <c>str.encode(encoding)</c> returns for <paramref name="text"/>, which must have passed
        ///     <see cref="CheckEncode(ReadOnlySpan{char})"/>; a BOM codec starts with its BOM when <paramref name="bom"/>.
        /// </summary>
        /// <param name="text">The text.</param>
        /// <param name="dest">The destination: at least <see cref="MaxByteCount"/> of the text's length.</param>
        /// <param name="bom">True to write the BOM of a BOM codec first (every one-shot encode does; TextIOWrapper only at
        ///     the start of the file).</param>
        /// <returns>The number of bytes written.</returns>
        /// <exception cref="UnicodeError">The <c>undefined</c> codec.</exception>
        internal int Encode(ReadOnlySpan<char> text, Span<byte> dest, bool bom)
        {
            switch (Kind)
            {
                case PythonCodecKind.Utf8:
                    return Encoding.UTF8.GetBytes(text, dest);
                case PythonCodecKind.Utf8Sig:
                {
                    int head = 0;
                    if (bom)
                    {
                        Utf8Bom.CopyTo(dest);
                        head = 3;
                    }
                    return head + Encoding.UTF8.GetBytes(text, dest.Slice(head));
                }
                case PythonCodecKind.Latin1:
                case PythonCodecKind.Ascii:
                    // Validated: every character is below the codec's limit, so it IS its byte.
                    return Encoding.Latin1.GetBytes(text, dest);
                case PythonCodecKind.Charmap:
                    return EncodeCharmap(text, dest);
                case PythonCodecKind.Utf16:
                    return EncodeUtf16(text, dest, bigEndian: false, bom);
                case PythonCodecKind.Utf16LE:
                    return EncodeUtf16(text, dest, bigEndian: false, bom: false);
                case PythonCodecKind.Utf16BE:
                    return EncodeUtf16(text, dest, bigEndian: true, bom: false);
                case PythonCodecKind.Utf32:
                    return EncodeUtf32(text, dest, bigEndian: false, bom);
                case PythonCodecKind.Utf32LE:
                    return EncodeUtf32(text, dest, bigEndian: false, bom: false);
                case PythonCodecKind.Utf32BE:
                    return EncodeUtf32(text, dest, bigEndian: true, bom: false);
                default:
                    throw new UnicodeError("undefined encoding");
            }
        }

        /// <summary>The most bytes <see cref="Encode"/> writes for <paramref name="chars"/> characters (BOM included).</summary>
        /// <param name="chars">The number of UTF-16 characters.</param>
        /// <returns>The byte capacity.</returns>
        internal int MaxByteCount(int chars) => Kind switch
        {
            PythonCodecKind.Utf8 or PythonCodecKind.Utf8Sig => chars * 3 + 3,
            PythonCodecKind.Utf16 or PythonCodecKind.Utf16LE or PythonCodecKind.Utf16BE => chars * 2 + 2,
            PythonCodecKind.Utf32 or PythonCodecKind.Utf32LE or PythonCodecKind.Utf32BE => chars * 4 + 4,
            _ => chars,
        };

        /// <summary>True when every one-shot encode starts with a BOM (<c>utf-16</c>, <c>utf-32</c>, <c>utf-8-sig</c>):
        ///     savetxt's byte-stream writes then each carry one, so they cannot be encoded as one batch.</summary>
        internal bool WritesBomPerEncode => Kind is PythonCodecKind.Utf16 or PythonCodecKind.Utf32 or PythonCodecKind.Utf8Sig;

        /// <summary>Whether <paramref name="kind"/> is one of the UTF-16 codecs.</summary>
        /// <param name="kind">The kind.</param>
        /// <returns>True for <c>utf-16</c>, <c>utf-16-le</c>, <c>utf-16-be</c>.</returns>
        private static bool IsUtf16(PythonCodecKind kind)
            => kind is PythonCodecKind.Utf16 or PythonCodecKind.Utf16LE or PythonCodecKind.Utf16BE;

        /// <summary>The codec name a one-shot encode error quotes (<c>utf-16</c> for the BOM codec, <c>charmap</c> for a
        ///     table, …).</summary>
        /// <param name="kind">The codec kind.</param>
        /// <returns>The name.</returns>
        private static string OneShotErrorName(PythonCodecKind kind) => kind switch
        {
            PythonCodecKind.Utf8 or PythonCodecKind.Utf8Sig => "utf-8",
            PythonCodecKind.Utf16 => "utf-16",
            PythonCodecKind.Utf16LE => "utf-16-le",
            PythonCodecKind.Utf16BE => "utf-16-be",
            PythonCodecKind.Utf32 => "utf-32",
            PythonCodecKind.Utf32LE => "utf-32-le",
            PythonCodecKind.Utf32BE => "utf-32-be",
            PythonCodecKind.Latin1 => "latin-1",
            PythonCodecKind.Ascii => "ascii",
            _ => "charmap",
        };

        /// <summary>The byte order CPython's UTF-16 / UTF-32 decoders start with: 0 (read a BOM) for the BOM codecs, -1
        ///     (little-endian) or 1 (big-endian) for the explicit ones.</summary>
        /// <param name="kind">A UTF-16 / UTF-32 kind.</param>
        /// <returns>The byte order.</returns>
        private static int UnicodeByteOrder(PythonCodecKind kind) => kind switch
        {
            PythonCodecKind.Utf16LE or PythonCodecKind.Utf32LE => -1,
            PythonCodecKind.Utf16BE or PythonCodecKind.Utf32BE => 1,
            _ => 0,
        };

        /// <summary>The UTF-8 byte-order mark.</summary>
        private static ReadOnlySpan<byte> Utf8Bom => new byte[] { 0xEF, 0xBB, 0xBF };

        /// <summary>
        ///     CPython's <c>UnicodeDecodeError.__str__</c> for any codec: one byte reads <c>byte 0x81 in position 4</c>, a range
        ///     <c>bytes in position 0-1</c>; the five arguments are kept for the error's <c>repr()</c>.
        /// </summary>
        /// <param name="codec">The codec name the error quotes.</param>
        /// <param name="data">The whole input the decoder was given.</param>
        /// <param name="start">The first byte of the bad range.</param>
        /// <param name="end">One past its last byte.</param>
        /// <param name="reason">CPython's reason text.</param>
        /// <returns>The exception (not thrown).</returns>
        internal static UnicodeDecodeError DecodeError(string codec, ReadOnlySpan<byte> data, int start, int end, string reason)
            => new UnicodeDecodeError(end - start == 1
                ? $"'{codec}' codec can't decode byte 0x{data[start]:x2} in position {start}: {reason}"
                : $"'{codec}' codec can't decode bytes in position {start}-{end - 1}: {reason}",
                codec, data.ToArray(), start, end, reason);

        // ---- charmap ------------------------------------------------------------------------------------------------

        /// <summary>CPython's <c>charmap_decode</c> over the decoding table: a byte the table marks U+FFFE is undefined.</summary>
        /// <param name="bytes">The bytes.</param>
        /// <param name="dest">One character per byte.</param>
        /// <exception cref="UnicodeDecodeError"><c>'charmap' codec can't decode byte 0x81 in position 4: character maps to
        ///     &lt;undefined&gt;</c> (positions count bytes of <paramref name="bytes"/>).</exception>
        private void DecodeCharmap(ReadOnlySpan<byte> bytes, Span<char> dest)
        {
            string table = _table;
            for (int i = 0; i < bytes.Length; i++)
            {
                char c = table[bytes[i]];
                if (c == '\uFFFE')
                    throw DecodeError("charmap", bytes, i, i + 1, "character maps to <undefined>");
                dest[i] = c;
            }
        }

        /// <summary>Builds the encode map at the first encode: the inverse of the table, a later byte overwriting an
        ///     earlier one (<c>PyUnicode_BuildEncodingMap</c>).</summary>
        private void EnsureEncodeMap()
        {
            if (System.Threading.Volatile.Read(ref _encodable) != null)
                return;
            var bytes = new byte[0x10000];
            var defined = new ulong[0x10000 / 64];
            for (int b = 0; b < 256; b++)
            {
                char c = _table[b];
                if (c == '\uFFFE')
                    continue;
                bytes[c] = (byte)b;
                defined[c >> 6] |= 1UL << (c & 63);
            }
            _encodeBytes = bytes;
            // Published last: a thread that sees the bitmap sees the finished byte map (both are complete before it). Two
            // threads may build it at once; both build the same maps, so either publication is right.
            System.Threading.Volatile.Write(ref _encodable, defined);
        }

        /// <summary>Whether the charmap encodes <paramref name="c"/> (a surrogate never: a pair is a code point past the
        ///     BMP, which no single-byte table holds, and a lone one is in no table).</summary>
        /// <param name="c">The character.</param>
        /// <returns>True when encodable.</returns>
        private bool CharmapEncodes(char c) => (_encodable[c >> 6] & (1UL << (c & 63))) != 0;

        /// <summary>
        ///     CPython's <c>charmap_encoding_error</c> under <c>strict</c>: the first unencodable character starts a RUN of
        ///     consecutive unencodable code points, reported together (<c>characters in position 1-2</c>).
        /// </summary>
        /// <param name="text">The text.</param>
        /// <exception cref="UnicodeEncodeError">An unencodable character.</exception>
        private void CheckCharmapEncode(ReadOnlySpan<char> text)
        {
            EnsureEncodeMap();
            int i = 0;
            // Every character before the first unencodable one is a BMP character, so its index IS its code-point position.
            while (i < text.Length && CharmapEncodes(text[i]))
                i++;
            if (i == text.Length)
                return;
            long start = i, position = i;
            uint first = 0;
            while (i < text.Length && !CharmapEncodes(text[i]))
            {
                uint cp;
                if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
                {
                    cp = (uint)char.ConvertToUtf32(text[i], text[i + 1]);
                    i += 2;
                }
                else
                {
                    cp = text[i];
                    i++;
                }
                if (position == start)
                    first = cp;
                position++;
            }
            throw NDStringCodec.EncodeErrorFor(first, start, position, "charmap", "character maps to <undefined>");
        }

        /// <summary>The charmap encode of validated text: one byte per character.</summary>
        /// <param name="text">The text (every character encodable).</param>
        /// <param name="dest">The destination.</param>
        /// <returns>The number of bytes written.</returns>
        private int EncodeCharmap(ReadOnlySpan<char> text, Span<byte> dest)
        {
            EnsureEncodeMap();
            byte[] map = _encodeBytes;
            for (int i = 0; i < text.Length; i++)
                dest[i] = map[text[i]];
            return text.Length;
        }
    }
}
