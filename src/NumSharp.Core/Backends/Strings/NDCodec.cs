using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Threading;
using NumSharp.IO;

// =============================================================================
// NDCodec.cs — CPython 3.12's text codecs over code points, with every built-in error handler (strings phase 2, P2.6
// milestone 4: np.strings.encode / np.strings.decode)
// =============================================================================
//
// WHAT NUMPY CALLS
//   np.strings.encode(a, encoding, errors) runs str.encode per element and np.strings.decode(a, encoding, errors)
//   bytes.decode (numpy/_core/strings.py: _vec_string over _clean_args(encoding, errors)). Those are CPython's
//   PyUnicode_AsEncodedString and PyUnicode_FromEncodedObject → PyUnicode_Decode (Objects/unicodeobject.c):
//     * no encoding: UTF-8 (strict unless errors given — _clean_args never passes errors without an encoding);
//     * the C FAST PATHS, chosen on _Py_normalize_encoding of the name into an 11-byte buffer (so a name normalizing to
//       more than 10 characters never takes one): utf8 / utf_8, utf16 / utf_16, utf32 / utf_32, ascii / us_ascii, and
//       latin1 / latin_1 / iso_8859_1 / iso8859_1 (mbcs on Windows, which NumSharp refuses);
//     * otherwise the codec registry (codecs.lookup: encodings.search_function over the alias table), whose text-encoding
//       check refuses a bytes-to-bytes codec ("'base64' is not a text encoding; use codecs.encode() to handle arbitrary
//       codecs") and whose miss is "unknown encoding: nope".
//   bytes.decode of EMPTY bytes returns '' before any of this (PyUnicode_FromEncodedObject's size-0 shortcut), so an
//   unknown codec name is not an error for an empty element; str.encode has no such shortcut.
//
// THE CODECS (NumSharp implements what IO/TextReading/PythonCodecs.Tables.cs classifies — generated from CPython)
//   UTF-8 (unicode_encode_utf8 / unicode_decode_utf8 with stringlib's utf8_encoder / utf8_decode), utf-8-sig (a BOM, then
//   UTF-8 — the errors name 'utf-8' and count positions after the BOM), UTF-16 and UTF-32 with or without a BOM
//   (_PyUnicode_EncodeUTF16 / 32, PyUnicode_DecodeUTF16Stateful / 32 with final=True), Latin-1 and ASCII
//   (unicode_encode_ucs1 with limits 256 / 128, PyUnicode_DecodeASCII), the 70 single-byte charmap codecs
//   (_PyUnicode_EncodeCharmap with charmap_encoding_error over the EncodingMap built from the decoding table, the LAST byte
//   winning where two bytes decode to one character; charmap_decode_string) and 'undefined' (UnicodeError("undefined
//   encoding") for every encode, and for every decode of non-empty bytes). The multi-byte CJK codecs, utf-7, the escape
//   codecs, idna, punycode and mbcs / oem are refused with NotSupportedException (the phase-2 plan's codec scope).
//
// THE ERROR HANDLERS (Python/codecs.c, and the inline copies the codecs carry)
//   A codec meets its first error, then classifies the handler NAME (_Py_GetErrorHandler: exact, case-sensitive
//   'strict', 'surrogateescape', 'replace', 'ignore', 'backslashreplace', 'surrogatepass', 'xmlcharrefreplace', anything
//   else OTHER). The 8-bit encoders (UTF-8, Latin-1, ASCII, charmap) and the UTF-8 / ASCII decoders handle the common
//   names INLINE; everything else goes through the GENERIC call (unicode_encode_call_errorhandler /
//   unicode_decode_call_errorhandler_writer), which first looks the name up in the error registry (PyCodec_LookupError:
//   the built-ins plus 'namereplace'; an unknown name is "LookupError: unknown error handler name 'x'" — raised only when
//   an error occurs, never for clean text) and then calls the handler with a Unicode{En,De}codeError over the bad range:
//     strict            raises the error;
//     ignore            nothing, resume after the range;
//     replace           encode: '?' per character; decode: ONE U+FFFD for the range;
//     backslashreplace  encode: \xNN / \uNNNN / \UNNNNNNNN per character; decode: \xNN per byte;
//     xmlcharrefreplace encode: &#NNNN; per character (decimal); decode: TypeError "don't know how to handle
//                       UnicodeDecodeError in error callback";
//     namereplace       encode: \N{NAME} per character (unicodedata's names WITH aliases and named sequences —
//                       PyUnicodeNames), the backslash form for an unnamed one; decode: the TypeError above;
//     surrogatepass     only for a UTF-8 / UTF-16 / UTF-32 codec (get_standard_encoding of the error's codec name, else
//                       the error is re-raised): encode writes each SURROGATE of the range in that codec's form (a
//                       non-surrogate re-raises); decode reads ONE surrogate at the range's start (any range) and
//                       resumes after its bytes, or re-raises;
//     surrogateescape   encode: U+DC80..U+DCFF become the bytes 0x80..0xFF (anything else re-raises); decode: up to four
//                       bytes >= 0x80 become U+DC80..U+DCFF (an ASCII byte at the start re-raises).
//   The inline copies give the same bytes as the generic handlers except where CPython's own code differs, which the
//   encoders below reproduce: each codec's error RANGES (the 8-bit encoders collect a run of consecutive unencodable
//   characters, UTF-16 / UTF-32 report one surrogate at a time), surrogateescape's partial run (the escapable prefix is
//   written, the generic call then sees the rest of the run and re-raises), the replacement's own encodability (a
//   charmap codec maps '?' and the generic replacement's characters through its table, raising the ORIGINAL error when
//   one is unmapped; the UTF-16 / UTF-32 encoders refuse a bytes replacement of the wrong length), and the decimal
//   formatting of an xmlcharref for a code unit past U+10FFFF (sprintf("%d") inline, the digit loop in the handler).
//
// ERROR TEXTS are CPython's (UnicodeEncodeError_str / UnicodeDecodeError_str): "'ascii' codec can't encode character
//   '\xe9' in position 1: ordinal not in range(128)", "... characters in position 1-3: ...", "'utf-8' codec can't decode
//   byte 0xff in position 0: invalid start byte", "... bytes in position 2-4: unexpected end of data". The codec name a
//   charmap codec's errors quote is 'charmap', a UTF-16 / UTF-32 decoder's the order it decoded in ('utf-16-le' even
//   for 'utf-16').
//
// UNITS. A str is a sequence of code points (uint), as NumPy's U element is: a lone surrogate is a code point of its own,
// and a unit past U+10FFFF (only a raw view of a U array holds one) is carried through with CPython's arithmetic on it
// (UTF-8 writes its four-byte pattern truncated to bytes, UTF-16 its "surrogates", UTF-32 the value itself).
// =============================================================================

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The inline classification of an error handler NAME (CPython's <c>_Py_error_handler</c>, from
    ///     <c>_Py_GetErrorHandler</c>): the names the 8-bit codecs handle without calling the handler, and
    ///     <see cref="Other"/> for every other name — <c>namereplace</c> and the unknown ones alike, which go through the
    ///     generic call (<see cref="NDCodecErrors.Generic"/>).
    /// </summary>
    internal enum NDErrorHandler : byte
    {
        /// <summary><c>strict</c>, or no errors argument.</summary>
        Strict,

        /// <summary><c>surrogateescape</c>.</summary>
        SurrogateEscape,

        /// <summary><c>replace</c>.</summary>
        Replace,

        /// <summary><c>ignore</c>.</summary>
        Ignore,

        /// <summary><c>backslashreplace</c>.</summary>
        BackslashReplace,

        /// <summary><c>surrogatepass</c>.</summary>
        SurrogatePass,

        /// <summary><c>xmlcharrefreplace</c>.</summary>
        XmlCharRefReplace,

        /// <summary>Any other name: <c>namereplace</c>, or a name the registry does not know.</summary>
        Other,
    }

    /// <summary>
    ///     The handler a GENERIC error call runs (<c>PyCodec_LookupError</c> over the error registry's built-ins): every
    ///     inline name plus <c>namereplace</c>.
    /// </summary>
    internal enum NDGenericHandler : byte
    {
        /// <summary><c>PyCodec_StrictErrors</c>: raises the error.</summary>
        Strict,

        /// <summary><c>PyCodec_IgnoreErrors</c>.</summary>
        Ignore,

        /// <summary><c>PyCodec_ReplaceErrors</c>.</summary>
        Replace,

        /// <summary><c>PyCodec_XMLCharRefReplaceErrors</c>.</summary>
        XmlCharRefReplace,

        /// <summary><c>PyCodec_BackslashReplaceErrors</c>.</summary>
        BackslashReplace,

        /// <summary><c>PyCodec_NameReplaceErrors</c>.</summary>
        NameReplace,

        /// <summary><c>PyCodec_SurrogatePassErrors</c>.</summary>
        SurrogatePass,

        /// <summary><c>PyCodec_SurrogateEscapeErrors</c>.</summary>
        SurrogateEscape,
    }

    /// <summary>
    ///     The <c>errors</c> argument of one <c>str.encode</c> / <c>bytes.decode</c> call: its name, its inline class, and the
    ///     generic handler the registry gives it — looked up only when a codec first needs the generic call, because an
    ///     unknown name raises <see cref="LookupError"/> only then (clean text never consults the registry).
    /// </summary>
    /// <remarks>
    ///     Immutable apart from the memoized registry lookup (a benign race: two threads compute the same answer). One
    ///     instance serves every element of a call whose <c>errors</c> is a scalar; <see cref="Strict"/> is the shared policy
    ///     of a call without one.
    /// </remarks>
    internal sealed class NDCodecErrors
    {
        /// <summary>The policy of a call without an <c>errors</c> argument (CPython's <c>errors == NULL</c>: strict).</summary>
        internal static readonly NDCodecErrors Strict = new NDCodecErrors(null);

        /// <summary>The name as given (an exact, case-sensitive Python string), or null when none was given.</summary>
        internal readonly string Name;

        /// <summary>The name's inline class (<c>_Py_GetErrorHandler</c>).</summary>
        internal readonly NDErrorHandler Inline;

        /// <summary>The memoized registry answer: <see cref="NDGenericHandler"/> + 1, 0 before the first lookup.</summary>
        private int _generic;

        /// <summary>Creates the policy of a name.</summary>
        /// <param name="name">The name (already validated as a Python str without NUL characters or surrogates), or null.</param>
        internal NDCodecErrors(string name)
        {
            Name = name;
            Inline = Classify(name);
        }

        /// <summary>
        ///     CPython's <c>_Py_GetErrorHandler</c>: an exact comparison against the seven inline names (no normalization:
        ///     <c>'Strict'</c> is <see cref="NDErrorHandler.Other"/>, and so an unknown name at the first error).
        /// </summary>
        /// <param name="name">The name, or null (strict).</param>
        /// <returns>The inline class.</returns>
        internal static NDErrorHandler Classify(string name) => name switch
        {
            null or "strict" => NDErrorHandler.Strict,
            "surrogateescape" => NDErrorHandler.SurrogateEscape,
            "replace" => NDErrorHandler.Replace,
            "ignore" => NDErrorHandler.Ignore,
            "backslashreplace" => NDErrorHandler.BackslashReplace,
            "surrogatepass" => NDErrorHandler.SurrogatePass,
            "xmlcharrefreplace" => NDErrorHandler.XmlCharRefReplace,
            _ => NDErrorHandler.Other,
        };

        /// <summary>
        ///     The handler of a generic error call: <c>PyCodec_LookupError(name)</c> over the built-in registry (a null name is
        ///     <c>strict</c>), memoized after the first lookup.
        /// </summary>
        /// <exception cref="LookupError"><c>unknown error handler name 'x'</c> — the name as CPython's <c>%.400s</c> prints it
        ///     (its first 400 UTF-8 bytes, a cut character read as U+FFFD).</exception>
        internal NDGenericHandler Generic
        {
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            get
            {
                int memo = Volatile.Read(ref _generic);
                if (memo != 0)
                    return (NDGenericHandler)(memo - 1);
                NDGenericHandler h = (Name ?? "strict") switch
                {
                    "strict" => NDGenericHandler.Strict,
                    "ignore" => NDGenericHandler.Ignore,
                    "replace" => NDGenericHandler.Replace,
                    "xmlcharrefreplace" => NDGenericHandler.XmlCharRefReplace,
                    "backslashreplace" => NDGenericHandler.BackslashReplace,
                    "namereplace" => NDGenericHandler.NameReplace,
                    "surrogatepass" => NDGenericHandler.SurrogatePass,
                    "surrogateescape" => NDGenericHandler.SurrogateEscape,
                    // The failure is not memoized: every call that needs the handler raises it again, as CPython looks the
                    // name up again for every encode call.
                    _ => throw new LookupError($"unknown error handler name '{NDCodec.CFormatString(Name, 400)}'"),
                };
                Volatile.Write(ref _generic, (int)h + 1);
                return h;
            }
        }
    }

    /// <summary>
    ///     One of CPython 3.12's text codecs as <c>str.encode</c> / <c>bytes.decode</c> resolve it by name (see the file
    ///     header): the codec kind, a charmap codec's tables, and the codec names its errors quote. The encoders and decoders
    ///     themselves are <c>NDCodec.Encode.cs</c> / <c>NDCodec.Decode.cs</c>; the generic error handlers
    ///     <c>NDCodec.Handlers.cs</c>.
    /// </summary>
    /// <remarks>
    ///     Instances are immutable apart from a charmap codec's encode map, built at its first encode and published with a
    ///     volatile write (two threads may build it at once; both build the same map). A fast-path codec and every
    ///     <c>encodings</c> module have one shared instance each.
    /// </remarks>
    internal sealed unsafe partial class NDCodec
    {
        /// <summary>CPython's strict-by-default UTF-8 codec (no encoding argument, the <c>utf8</c> fast path, the <c>utf_8</c>
        ///     module).</summary>
        internal static readonly NDCodec Utf8 = new NDCodec(PythonCodecKind.Utf8, null);

        /// <summary>The Latin-1 codec (the <c>latin1</c> fast path, the <c>latin_1</c> module).</summary>
        internal static readonly NDCodec Latin1 = new NDCodec(PythonCodecKind.Latin1, null);

        /// <summary>The ASCII codec (the <c>ascii</c> fast path, the <c>ascii</c> module).</summary>
        internal static readonly NDCodec Ascii = new NDCodec(PythonCodecKind.Ascii, null);

        /// <summary>UTF-16 with a BOM (the <c>utf16</c> fast path, the <c>utf_16</c> module).</summary>
        internal static readonly NDCodec Utf16 = new NDCodec(PythonCodecKind.Utf16, null);

        /// <summary>UTF-32 with a BOM (the <c>utf32</c> fast path, the <c>utf_32</c> module).</summary>
        internal static readonly NDCodec Utf32 = new NDCodec(PythonCodecKind.Utf32, null);

        /// <summary>One codec per <c>encodings</c> module, created at its first lookup (a charmap builds its tables once).</summary>
        private static readonly ConcurrentDictionary<string, NDCodec> ByModule = new ConcurrentDictionary<string, NDCodec>(StringComparer.Ordinal)
        {
            ["utf_8"] = Utf8,
            ["latin_1"] = Latin1,
            ["ascii"] = Ascii,
            ["utf_16"] = Utf16,
            ["utf_32"] = Utf32,
        };

        /// <summary>The codec's kind.</summary>
        internal readonly PythonCodecKind Kind;

        /// <summary>A charmap codec's decoding table: the code point of each byte, <c>0xFFFE</c> for a byte it cannot decode
        ///     (CPython's own marker); null for the other kinds.</summary>
        private readonly uint[] _decodeTable;

        /// <summary>A charmap codec's encode map, built at its first encode: the byte of each encodable BMP code point.</summary>
        private byte[] _encodeBytes;

        /// <summary>The bit set of the encodable BMP code points (published after <see cref="_encodeBytes"/>, so a thread
        ///     that sees it sees the finished byte map).</summary>
        private ulong[] _encodable;

        /// <summary>Creates a codec.</summary>
        /// <param name="kind">The kind.</param>
        /// <param name="table">A charmap codec's decoding table (256 characters, U+FFFE undefined), else null.</param>
        private NDCodec(PythonCodecKind kind, string table)
        {
            Kind = kind;
            if (table is not null)
            {
                _decodeTable = new uint[256];
                for (int b = 0; b < 256; b++)
                    _decodeTable[b] = table[b];
            }
        }

        /// <summary>
        ///     CPython's codec lookup for one <c>str.encode</c> / <c>bytes.decode</c> call: the C fast paths first
        ///     (<see cref="PythonCodecs.TryFastPath"/> over <see cref="PythonCodecs.Normalize"/>), then <c>codecs.lookup</c>
        ///     (<see cref="PythonCodecs.FindModule"/>), then the text-encoding check of <c>_PyCodec_LookupTextEncoding</c>.
        /// </summary>
        /// <param name="name">The encoding name as given (the str CPython read; its NUL and surrogate checks already
        ///     passed).</param>
        /// <param name="encode">True for <c>str.encode</c> (a non-text codec's advice names <c>codecs.encode()</c>), false
        ///     for <c>bytes.decode</c> (<c>codecs.decode()</c>).</param>
        /// <returns>The codec (a shared instance).</returns>
        /// <exception cref="LookupError"><c>unknown encoding: nope</c> (the name in full), or <c>'base64' is not a text
        ///     encoding; use codecs.encode() to handle arbitrary codecs</c> (the name as <c>%.400s</c> prints it).</exception>
        /// <exception cref="NotSupportedException">A Python text codec NumSharp does not implement (the multi-byte CJK
        ///     codecs, <c>utf-7</c>, the escape codecs, <c>idna</c>, <c>mbcs</c> / <c>oem</c> …).</exception>
        internal static NDCodec Lookup(string name, bool encode)
        {
            string normalized = PythonCodecs.Normalize(name);
            if (PythonCodecs.TryFastPath(normalized, out PythonCodecKind fast))
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
            // On Windows 'mbcs' is one more fast path (the ANSI code page); NumSharp implements no code page of the host,
            // and the registry classifies the module as unsupported, which the switch below refuses.
            string module = PythonCodecs.FindModule(normalized) ?? throw new LookupError($"unknown encoding: {name}");
            var (kind, codecName) = PythonCodecs.Modules[module];
            switch (kind)
            {
                case PythonCodecKind.NotText:
                    throw new LookupError($"'{CFormatString(name, 400)}' is not a text encoding; use " +
                                          (encode ? "codecs.encode()" : "codecs.decode()") + " to handle arbitrary codecs");
                case PythonCodecKind.Unsupported:
                    throw new NotSupportedException(
                        $"NumSharp does not implement Python's '{codecName}' codec (named by '{name}') for np.strings.encode / " +
                        "np.strings.decode: it implements UTF-8 (and utf-8-sig), UTF-16 and UTF-32 (with or without a BOM, either " +
                        "byte order), Latin-1, ASCII, the single-byte code pages (cp1252, iso8859_15, koi8_r, mac_roman, ...) " +
                        "and 'undefined'.");
            }
            return ByModule.GetOrAdd(module, static m =>
            {
                var (k, _) = PythonCodecs.Modules[m];
                return new NDCodec(k, k == PythonCodecKind.Charmap ? PythonCodecs.CharmapTables[m] : null);
            });
        }

        /// <summary>
        ///     The byte order of a UTF-16 / UTF-32 codec as CPython's coders take it: 0 for the BOM codecs (the encoder writes
        ///     a native little-endian BOM; the decoder reads one to choose the order), -1 little-endian, 1 big-endian.
        /// </summary>
        internal int ByteOrder => Kind switch
        {
            PythonCodecKind.Utf16LE or PythonCodecKind.Utf32LE => -1,
            PythonCodecKind.Utf16BE or PythonCodecKind.Utf32BE => 1,
            _ => 0,
        };

        /// <summary>
        ///     The byte a charmap codec writes for a code point, CPython's <c>encoding_map_lookup</c> over the map
        ///     <c>PyUnicode_BuildEncodingMap</c> builds from the decoding table: U+0000 is byte 0, a code point past U+FFFF or
        ///     absent from the table is -1 (unencodable), and where two bytes decode to one character the LAST wins
        ///     (<c>tools/gen_python_codecs.py</c> verifies that rule against <c>str.encode</c> for every codec over the BMP).
        /// </summary>
        /// <param name="cp">The code point.</param>
        /// <returns>The byte, or -1.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal int CharmapByte(uint cp)
        {
            if (cp > 0xFFFF)
                return -1;
            if (cp == 0)
                return 0;
            var encodable = _encodable;
            if (encodable is null)
                encodable = BuildEncodeMap();
            return (encodable[cp >> 6] & (1UL << (int)(cp & 63))) != 0 ? _encodeBytes[cp] : -1;
        }

        /// <summary>A charmap codec's decoding table entry: the code point of a byte, <c>0xFFFE</c> when undefined.</summary>
        /// <param name="b">The byte.</param>
        /// <returns>The code point.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal uint CharmapPoint(byte b) => _decodeTable[b];

        /// <summary>
        ///     Builds the encode map at the first encode: the inverse of the decoding table, a later byte overwriting an
        ///     earlier one (<c>PyUnicode_BuildEncodingMap</c>'s level-3 store), U+FFFE (undefined) skipped.
        /// </summary>
        /// <returns>The published bit set of encodable code points.</returns>
        private ulong[] BuildEncodeMap()
        {
            var bytes = new byte[0x10000];
            var defined = new ulong[0x10000 / 64];
            for (int b = 0; b < 256; b++)
            {
                uint c = _decodeTable[b];
                if (c == 0xFFFE)
                    continue;
                bytes[c] = (byte)b;
                defined[c >> 6] |= 1UL << (int)(c & 63);
            }
            _encodeBytes = bytes;
            // Published last (a release write): a thread that reads the bit set also reads the finished byte map.
            Volatile.Write(ref _encodable, defined);
            return defined;
        }

        /// <summary>
        ///     CPython's <c>PyUnicode_FromFormat("%.Ns", s)</c> of a UTF-8 C string: the string's first <paramref name="limit"/>
        ///     UTF-8 bytes, decoded with the <c>replace</c> handler (a character the cut splits reads as U+FFFD).
        /// </summary>
        /// <param name="s">The string (a Python str's text; its surrogates were refused before it became a C string).</param>
        /// <param name="limit">The precision in bytes (400 for the error-handler and text-encoding messages).</param>
        /// <returns>The printed text (<paramref name="s"/> itself when it is short enough).</returns>
        internal static string CFormatString(string s, int limit)
        {
            if (s.Length * 3 <= limit)
                return s;   // at most three UTF-8 bytes per UTF-16 unit: certainly within the precision
            byte[] utf8 = System.Text.Encoding.UTF8.GetBytes(s);
            if (utf8.Length <= limit)
                return s;
            // The cut can split a character: Python decodes the prefix with 'replace', which gives ONE U+FFFD for an
            // incomplete sequence at the end (unexpected end of data) — .NET's decoder does the same for a truncated tail.
            return System.Text.Encoding.UTF8.GetString(utf8, 0, limit);
        }
    }
}
