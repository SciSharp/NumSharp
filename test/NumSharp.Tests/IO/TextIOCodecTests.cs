using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using NumSharp.IO;

namespace NumSharp.Tests.IO
{
    /// <summary>
    ///     The Python codec layer of <c>np.loadtxt</c> / <c>np.savetxt</c> (<see cref="TextIOCodec"/>) and NumPy's
    ///     <c>_datasource</c> file rules, against CPython 3.12 / NumPy 2.4.2 (every expected value probed on 2026-10-15):
    ///     codec lookup by any CPython spelling, the one-shot <c>bytes.decode</c> / <c>str.encode</c> of each family with
    ///     CPython's error texts, TextIOWrapper's chunking (which decides a decode error's position) and BOM rules, the
    ///     compressed-file fall-backs and the comment objects of <c>loadtxt</c>.
    /// </summary>
    /// <remarks>
    ///     The string oracle's section tio replays the NumPy cases of these paths; these tests pin the exact exception
    ///     TYPES, the platform-dependent newline translation of a file target, the gzip round trip (Python's and .NET's
    ///     gzip bytes differ, so a corpus cannot compare them) and NumSharp's own refusals.
    /// </remarks>
    [TestClass]
    public class TextIOCodecTests
    {
        /// <summary>A fresh temp file path, deleted by the caller's finally.</summary>
        /// <param name="suffix">The file name's suffix.</param>
        /// <returns>The path.</returns>
        private static string TempFile(string suffix = ".txt")
            => Path.Combine(Path.GetTempPath(), "ns_codec_" + Guid.NewGuid().ToString("N") + suffix);

        /// <summary>Asserts the exact exception type and its message.</summary>
        /// <typeparam name="T">The exception type (exactly).</typeparam>
        /// <param name="f">The call.</param>
        /// <param name="msg">CPython's message.</param>
        /// <returns>The exception.</returns>
        private static T Throws<T>(Action f, string msg) where T : Exception
        {
            var e = Assert.ThrowsException<T>(f);
            Assert.AreEqual(msg, e.Message);
            return e;
        }

        /// <summary>The elements of a StringDType / U array in C order.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The texts.</returns>
        private static string[] Texts(NDArray a)
        {
            using var c = a.copy();
            return Enumerable.Range(0, (int)c.size).Select(i => (string)c.Storage.GetAtIndex(i)).ToArray();
        }

        // ---- lookup -------------------------------------------------------------------------------------------------

        /// <summary><c>codecs.lookup(name).name</c> for CPython's spellings: case, separators, aliases, the dot rule.</summary>
        [TestMethod]
        public void Lookup_ResolvesNamesAsCodecsLookup()
        {
            string Name(string n) => TextIOCodec.Lookup(n, CodecLookupContext.Open).Name;
            Assert.AreEqual("utf-16", Name("UTF-16"));
            Assert.AreEqual("utf-16", Name("u16"));
            Assert.AreEqual("utf-16-le", Name("UTF_16LE"));
            Assert.AreEqual("utf-8", Name("  utf--8 "));
            Assert.AreEqual("utf-8", Name("cp65001"));
            Assert.AreEqual("utf-8-sig", Name("utf_8_sig"));
            Assert.AreEqual("cp1252", Name("Windows-1252"));
            Assert.AreEqual("cp1252", Name("1252"));
            Assert.AreEqual("cp037", Name("ibm037"));
            Assert.AreEqual("iso8859-15", Name("iso-8859-15"));
            // latin_1's codec names itself iso8859-1, as does the iso8859_1 charmap module.
            Assert.AreEqual("iso8859-1", Name("L1"));
            Assert.AreEqual("iso8859-1", Name("8859"));
            Assert.AreEqual("iso8859-1", Name("iso.8859.1")); // an alias looked up with '.' read as '_'
            // A '.' in a module name is never imported; an unknown spelling reports the name as given.
            Throws<LookupError>(() => Name("utf.8"), "unknown encoding: utf.8");
            Throws<LookupError>(() => Name("cp_1252"), "unknown encoding: cp_1252");
            Throws<LookupError>(() => Name("UNICODELITTLEUNFRAMED"), "unknown encoding: UNICODELITTLEUNFRAMED");
            Throws<LookupError>(() => Name(""), "unknown encoding: ");
            Throws<LookupError>(() => Name("-"), "unknown encoding: -");
            Throws<LookupError>(() => Name("latin_9"), "unknown encoding: latin_9");
        }

        /// <summary>A bytes-to-bytes codec is CPython's LookupError, its advice naming the caller's context.</summary>
        [TestMethod]
        public void Lookup_NonTextCodec_AdviceFollowsTheContext()
        {
            Throws<LookupError>(() => TextIOCodec.Lookup("base64", CodecLookupContext.Decode),
                "'base64' is not a text encoding; use codecs.decode() to handle arbitrary codecs");
            Throws<LookupError>(() => TextIOCodec.Lookup("rot13", CodecLookupContext.Encode),
                "'rot13' is not a text encoding; use codecs.encode() to handle arbitrary codecs");
            Throws<LookupError>(() => TextIOCodec.Lookup("hex", CodecLookupContext.Open),
                "'hex' is not a text encoding; use codecs.open() to handle arbitrary codecs");
        }

        /// <summary>A Python text codec NumSharp does not implement is refused by name, not mistaken for an unknown one.</summary>
        [TestMethod]
        public void Lookup_UnimplementedCodec_IsNotSupported()
        {
            foreach (string name in new[] { "shift_jis", "x-mac-japanese", "gbk", "utf-7", "idna", "unicode_escape" })
                Assert.ThrowsException<NotSupportedException>(() => TextIOCodec.Lookup(name, CodecLookupContext.Open), name);
            using var ms = new MemoryStream(Encoding.ASCII.GetBytes("a b\n"));
            Assert.ThrowsException<NotSupportedException>(() => np.loadtxt(ms, np.dtypes.StringDType(), encoding: "big5"));
        }

        // ---- one-shot decode ---------------------------------------------------------------------------------------

        /// <summary><c>bytes.decode('utf-16')</c>: the BOM picks the order (native little-endian without one), and the
        ///     four CPython errors with their byte ranges, under the order's codec name.</summary>
        [TestMethod]
        public void OneShot_Utf16Decode_IsCPythons()
        {
            var c = TextIOCodec.Lookup("utf-16", CodecLookupContext.Decode);
            Assert.AreEqual("a", c.Decode(new byte[] { 0xFF, 0xFE, 0x61, 0x00 }));
            Assert.AreEqual("a", c.Decode(new byte[] { 0xFE, 0xFF, 0x00, 0x61 }));
            Assert.AreEqual("ab", c.Decode(new byte[] { 0x61, 0x00, 0x62, 0x00 }));
            Assert.AreEqual("", c.Decode(new byte[] { 0xFF, 0xFE }));
            Throws<UnicodeDecodeError>(() => c.Decode(new byte[] { 0x61, 0x00, 0x62 }),
                "'utf-16-le' codec can't decode byte 0x62 in position 2: truncated data");
            Throws<UnicodeDecodeError>(() => c.Decode(new byte[] { 0x00, 0xDC }),
                "'utf-16-le' codec can't decode bytes in position 0-1: illegal encoding");
            Throws<UnicodeDecodeError>(() => c.Decode(new byte[] { 0x00, 0xD8, 0x61, 0x00 }),
                "'utf-16-le' codec can't decode bytes in position 0-1: illegal UTF-16 surrogate");
            Throws<UnicodeDecodeError>(() => c.Decode(new byte[] { 0x00, 0xD8 }),
                "'utf-16-le' codec can't decode bytes in position 0-1: unexpected end of data");
            Throws<UnicodeDecodeError>(() => c.Decode(new byte[] { 0xFE, 0xFF, 0xDC, 0x00 }),
                "'utf-16-be' codec can't decode bytes in position 2-3: illegal encoding");
            // An explicit order keeps a BOM as U+FEFF; a pair decodes to its code point.
            var le = TextIOCodec.Lookup("utf-16-le", CodecLookupContext.Decode);
            Assert.AreEqual("﻿a", le.Decode(new byte[] { 0xFF, 0xFE, 0x61, 0x00 }));
            var be = TextIOCodec.Lookup("utf-16-be", CodecLookupContext.Decode);
            Assert.AreEqual("\U00010000", be.Decode(new byte[] { 0xD8, 0x00, 0xDC, 0x00 }));
            Throws<UnicodeDecodeError>(() => le.Decode(new byte[] { 0x61, 0x00, 0x00, 0xD8, 0x00, 0xDC, 0x62 }),
                "'utf-16-le' codec can't decode byte 0x62 in position 6: truncated data");
        }

        /// <summary><c>bytes.decode('utf-32')</c>: the BOM, the surrogate range checked before the upper bound, truncation.</summary>
        [TestMethod]
        public void OneShot_Utf32Decode_IsCPythons()
        {
            var c = TextIOCodec.Lookup("utf-32", CodecLookupContext.Decode);
            Assert.AreEqual("a", c.Decode(new byte[] { 0xFF, 0xFE, 0, 0, 0x61, 0, 0, 0 }));
            Assert.AreEqual("a", c.Decode(new byte[] { 0, 0, 0xFE, 0xFF, 0, 0, 0, 0x61 }));
            Throws<UnicodeDecodeError>(() => c.Decode(new byte[] { 0x61, 0, 0 }),
                "'utf-32-le' codec can't decode bytes in position 0-2: truncated data");
            Throws<UnicodeDecodeError>(() => c.Decode(new byte[] { 0, 0, 0x11, 0 }),
                "'utf-32-le' codec can't decode bytes in position 0-3: code point not in range(0x110000)");
            Throws<UnicodeDecodeError>(() => c.Decode(new byte[] { 0, 0xD8, 0, 0 }),
                "'utf-32-le' codec can't decode bytes in position 0-3: code point in surrogate code point range(0xd800, 0xe000)");
            var be = TextIOCodec.Lookup("utf-32-be", CodecLookupContext.Decode);
            Throws<UnicodeDecodeError>(() => be.Decode(new byte[] { 0, 0, 0xD8, 0 }),
                "'utf-32-be' codec can't decode bytes in position 0-3: code point in surrogate code point range(0xd800, 0xe000)");
        }

        /// <summary><c>utf-8-sig</c> drops one leading BOM (a second is a U+FEFF), errors count bytes after it; ASCII and a
        ///     charmap report one byte; Latin-1 decodes everything.</summary>
        [TestMethod]
        public void OneShot_OtherDecodes_AreCPythons()
        {
            var sig = TextIOCodec.Lookup("utf-8-sig", CodecLookupContext.Decode);
            Assert.AreEqual("ab", sig.Decode(new byte[] { 0xEF, 0xBB, 0xBF, 0x61, 0x62 }));
            Assert.AreEqual("﻿a", sig.Decode(new byte[] { 0xEF, 0xBB, 0xBF, 0xEF, 0xBB, 0xBF, 0x61 }));
            Throws<UnicodeDecodeError>(() => sig.Decode(new byte[] { 0xEF, 0xBB, 0xBF, 0xFF }),
                "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte");
            Throws<UnicodeDecodeError>(() => sig.Decode(new byte[] { 0xEF, 0xBB }),
                "'utf-8' codec can't decode bytes in position 0-1: unexpected end of data");
            Throws<UnicodeDecodeError>(() => TextIOCodec.Lookup("ascii", CodecLookupContext.Decode).Decode(new byte[] { 0x80, 0x81 }),
                "'ascii' codec can't decode byte 0x80 in position 0: ordinal not in range(128)");
            var cp1252 = TextIOCodec.Lookup("cp1252", CodecLookupContext.Decode);
            Assert.AreEqual("€", cp1252.Decode(new byte[] { 0x80 }));
            Throws<UnicodeDecodeError>(() => cp1252.Decode(new byte[] { 0x61, 0x81, 0x62 }),
                "'charmap' codec can't decode byte 0x81 in position 1: character maps to <undefined>");
            Throws<UnicodeDecodeError>(() => TextIOCodec.Lookup("iso8859_6", CodecLookupContext.Decode).Decode(new byte[] { 0xA1 }),
                "'charmap' codec can't decode byte 0xa1 in position 0: character maps to <undefined>");
            Assert.AreEqual("A", TextIOCodec.Lookup("cp037", CodecLookupContext.Decode).Decode(new byte[] { 0xC1 }));
            Assert.AreEqual("\u0080", TextIOCodec.Lookup("latin-1", CodecLookupContext.Decode).Decode(new byte[] { 0x80 }));
            Throws<UnicodeError>(() => TextIOCodec.Lookup("undefined", CodecLookupContext.Decode).Decode(Array.Empty<byte>()),
                "undefined encoding");
        }

        // ---- one-shot encode ---------------------------------------------------------------------------------------

        /// <summary>The encode errors: UTF-16 / UTF-32 one surrogate at a time under the codec's name; ASCII, Latin-1 and
        ///     a charmap the RUN of unencodable characters; utf-8-sig as UTF-8.</summary>
        [TestMethod]
        public void OneShot_EncodeErrors_AreCPythons()
        {
            void Check(string name, string text) => TextIOCodec.Lookup(name, CodecLookupContext.Encode).CheckEncode(text);
            Throws<UnicodeEncodeError>(() => Check("utf-16", "a\ud800"),
                "'utf-16' codec can't encode character '\\ud800' in position 1: surrogates not allowed");
            Throws<UnicodeEncodeError>(() => Check("utf-16-be", "a\udc00\ud800b"),
                "'utf-16-be' codec can't encode character '\\udc00' in position 1: surrogates not allowed");
            Throws<UnicodeEncodeError>(() => Check("utf-32", "\ud800"),
                "'utf-32' codec can't encode character '\\ud800' in position 0: surrogates not allowed");
            Throws<UnicodeEncodeError>(() => Check("utf-32-le", "a\udfffb"),
                "'utf-32-le' codec can't encode character '\\udfff' in position 1: surrogates not allowed");
            // A pair before the surrogate is one code point.
            Throws<UnicodeEncodeError>(() => Check("utf-16", "\U00010000\ud800"),
                "'utf-16' codec can't encode character '\\ud800' in position 1: surrogates not allowed");
            Throws<UnicodeEncodeError>(() => Check("utf-8-sig", "\ud800"),
                "'utf-8' codec can't encode character '\\ud800' in position 0: surrogates not allowed");
            Throws<UnicodeEncodeError>(() => Check("ascii", "éèx"),
                "'ascii' codec can't encode characters in position 0-1: ordinal not in range(128)");
            Throws<UnicodeEncodeError>(() => Check("latin-1", "aĀābĂ"),
                "'latin-1' codec can't encode characters in position 1-2: ordinal not in range(256)");
            Throws<UnicodeEncodeError>(() => Check("cp1252", "a中丮b"),
                "'charmap' codec can't encode characters in position 1-2: character maps to <undefined>");
            Throws<UnicodeEncodeError>(() => Check("cp037", "中"),
                "'charmap' codec can't encode character '\\u4e2d' in position 0: character maps to <undefined>");
            // The C fast paths: an iso8859_1 spelling encodes as Latin-1 (the registry would pick the charmap module).
            Throws<UnicodeEncodeError>(() => Check("iso8859_1", "Ā"),
                "'latin-1' codec can't encode character '\\u0100' in position 0: ordinal not in range(256)");
            Throws<UnicodeEncodeError>(() => Check("iso-8859-1", "Ā"),
                "'latin-1' codec can't encode character '\\u0100' in position 0: ordinal not in range(256)");
            Throws<UnicodeError>(() => Check("undefined", ""), "undefined encoding");
        }

        /// <summary>A one-shot encode of a BOM codec starts with the BOM (an empty text is the BOM alone); a charmap whose
        ///     table decodes two bytes to one character encodes it to the LATER byte.</summary>
        [TestMethod]
        public void OneShot_EncodeBytes_AreCPythons()
        {
            byte[] Encode(string name, string text)
            {
                var c = TextIOCodec.Lookup(name, CodecLookupContext.Encode);
                c.CheckEncode(text);
                var buf = new byte[c.MaxByteCount(text.Length)];
                return buf.AsSpan(0, c.Encode(text, buf, bom: true)).ToArray();
            }
            CollectionAssert.AreEqual(new byte[] { 0xFF, 0xFE, 0x61, 0, 0x62, 0 }, Encode("utf-16", "ab"));
            CollectionAssert.AreEqual(new byte[] { 0x61, 0, 0x62, 0 }, Encode("utf-16-le", "ab"));
            CollectionAssert.AreEqual(new byte[] { 0, 0x61, 0, 0x62 }, Encode("utf-16-be", "ab"));
            CollectionAssert.AreEqual(new byte[] { 0, 0xD8, 0, 0xDC, 0x78, 0 }, Encode("utf-16-le", "\U00010000x"));
            CollectionAssert.AreEqual(new byte[] { 0xFF, 0xFE, 0, 0, 0x61, 0, 0, 0 }, Encode("utf-32", "a"));
            CollectionAssert.AreEqual(new byte[] { 0xEF, 0xBB, 0xBF, 0x61, 0x62 }, Encode("utf-8-sig", "ab"));
            CollectionAssert.AreEqual(new byte[] { 0xFF, 0xFE }, Encode("utf-16", ""));
            CollectionAssert.AreEqual(new byte[] { 0xFF, 0xFE, 0, 0 }, Encode("utf-32", ""));
            CollectionAssert.AreEqual(new byte[] { 0xEF, 0xBB, 0xBF }, Encode("utf-8-sig", ""));
            CollectionAssert.AreEqual(new byte[] { 0x80 }, Encode("cp1252", "€"));
            // mac_arabic decodes 0x20 and 0xA0 to ' ': CPython encodes ' ' to 0xA0.
            CollectionAssert.AreEqual(new byte[] { 0xA0 }, Encode("mac_arabic", " "));
            CollectionAssert.AreEqual(new byte[] { 0xFD }, Encode("cp875", "\u001A"));
        }

        // ---- loadtxt over the codecs -------------------------------------------------------------------------------

        /// <summary>A file read with a BOM codec requires the BOM (TextIOWrapper's incremental decoder: a UnicodeError,
        ///     not a decode error); with one it decodes in the order named; CRLF is translated after decoding.</summary>
        [TestMethod]
        public void LoadTxt_File_BomCodecs_FollowTextIOWrapper()
        {
            string path = TempFile();
            try
            {
                File.WriteAllBytes(path, Encoding.Unicode.GetBytes("ab\n"));
                Throws<UnicodeError>(() => np.loadtxt(path, np.dtypes.StringDType(), encoding: "utf-16"),
                    "UTF-16 stream does not start with BOM");
                File.WriteAllBytes(path, new byte[] { 0xFE, 0xFF }.Concat(Encoding.BigEndianUnicode.GetBytes("a b\r\nc d\r\n")).ToArray());
                using (var a = np.loadtxt(path, np.dtypes.StringDType(), encoding: "UTF-16"))
                    CollectionAssert.AreEqual(new[] { "a", "b", "c", "d" }, Texts(a));
                File.WriteAllBytes(path, Encoding.UTF32.GetBytes("ab"));
                Throws<UnicodeError>(() => np.loadtxt(path, np.dtypes.StringDType(), encoding: "utf-32"),
                    "UTF-32 stream does not start with BOM");
                // utf-8-sig: a BOM is dropped, a partial one at the end of the file is dropped silently, as CPython does.
                File.WriteAllBytes(path, new byte[] { 0xEF, 0xBB, 0xBF, 0x61, 0x62, 0x0A });
                using (var a = np.loadtxt(path, np.dtypes.StringDType(), encoding: "utf-8-sig"))
                    CollectionAssert.AreEqual(new[] { "ab" }, Texts(a));
                File.WriteAllBytes(path, new byte[] { 0xEF, 0xBB });
                using (var a = np.loadtxt(path, np.dtypes.StringDType(), encoding: "utf-8-sig"))
                    Assert.AreEqual(0, a.size);
                // The undefined codec raises at the first chunk, an empty file's included.
                File.WriteAllBytes(path, Array.Empty<byte>());
                Throws<UnicodeError>(() => np.loadtxt(path, np.dtypes.StringDType(), encoding: "undefined"), "undefined encoding");
            }
            finally
            {
                File.Delete(path);
            }
        }

        /// <summary>A byte stream decodes each line on its own (<c>bytes.decode</c>): with UTF-16 the 0x0A byte ends a line
        ///     in the middle of a code unit, which CPython reports as truncated data.</summary>
        [TestMethod]
        public void LoadTxt_Stream_DecodesEachLineOnItsOwn()
        {
            using var u16 = new MemoryStream(Encoding.Unicode.GetPreamble().Concat(Encoding.Unicode.GetBytes("a b\n")).ToArray());
            Throws<UnicodeDecodeError>(() => np.loadtxt(u16, np.dtypes.StringDType(), encoding: "utf-16"),
                "'utf-16-le' codec can't decode byte 0x0a in position 8: truncated data");
            using var cp = new MemoryStream(new byte[] { 0x61, 0x20, 0x62, 0x0A, 0x80, 0x20, 0x63, 0x0A });
            using var a = np.loadtxt(cp, np.dtypes.StringDType(), encoding: "Windows-1252");
            CollectionAssert.AreEqual(new[] { "a", "b", "€", "c" }, Texts(a));
            using var bad = new MemoryStream(new byte[] { 0x61, 0x0A, 0x81, 0x0A });
            Throws<UnicodeDecodeError>(() => np.loadtxt(bad, np.dtypes.StringDType(), encoding: "cp1252"),
                "'charmap' codec can't decode byte 0x81 in position 0: character maps to <undefined>");
        }

        /// <summary>
        ///     A decode error's position counts within the chunk TextIOWrapper decoded: <c>read(16384)</c> reads 16384
        ///     bytes first (an error at byte 10000 is at position 10000, at byte 20000 at 3616), later chunks from the
        ///     bytes-per-character ratio; the line iteration of multi-character comments reads 8192-byte chunks.
        /// </summary>
        [TestMethod]
        public void LoadTxt_File_DecodeErrorPosition_FollowsTextIOWrapperChunks()
        {
            string path = TempFile();
            try
            {
                void At(byte[] prefix, int position, string encoding = "utf-8", object comments = null, byte bad = 0xFF)
                {
                    File.WriteAllBytes(path, prefix.Concat(new[] { bad, (byte)'\n' }).ToArray());
                    var e = Assert.ThrowsException<UnicodeDecodeError>(() =>
                        np.loadtxt(path, np.dtypes.StringDType(), comments ?? "#", encoding: encoding));
                    StringAssert.Contains(e.Message, $"in position {position}:");
                }
                At(Enumerable.Repeat((byte)'a', 10000).ToArray(), 10000);
                At(Enumerable.Repeat((byte)'a', 16384).ToArray(), 0);
                At(Enumerable.Repeat((byte)'a', 20000).ToArray(), 3616);
                At(Enumerable.Repeat((byte)'a', 40000).ToArray(), 7232);
                // 9000 'é' (18000 bytes) then 100 'a': chunk 1 is 16384 bytes = 8192 chars, chunk 2 b2cratio 2 * 8192.
                At(Encoding.UTF8.GetBytes(new string('é', 9000)).Concat(Enumerable.Repeat((byte)'a', 100)).ToArray(), 1716);
                At(Enumerable.Repeat((byte)'a', 20000).ToArray(), 3616, encoding: "cp1252", bad: 0x81);
                // The line iteration's 8192-byte chunks.
                At(Enumerable.Repeat((byte)'a', 10000).ToArray(), 1808, comments: new[] { "#", "%" });
            }
            finally
            {
                File.Delete(path);
            }
        }

        // ---- the file name: DataSource -----------------------------------------------------------------------------

        /// <summary>A missing name falls back to the name with a compressed extension; the extension is compared
        ///     case-sensitively; bz2 / LZMA names are refused; nothing found is NumPy's not-found text.</summary>
        [TestMethod]
        public void LoadTxt_DataSource_FindsAndOpensAsNumPy()
        {
            string stem = TempFile("");
            try
            {
                // data.txt missing, data.txt.gz present: NumPy opens the gzip file.
                using (var gz = new GZipStream(File.Create(stem + ".txt.gz"), CompressionMode.Compress))
                    gz.Write(Encoding.UTF8.GetBytes("a b\n"));
                using (var a = np.loadtxt(stem + ".txt", np.dtypes.StringDType()))
                    CollectionAssert.AreEqual(new[] { "a", "b" }, Texts(a));
                // .GZ is not '.gz': plain text, whose gzip bytes are not UTF-8.
                File.Copy(stem + ".txt.gz", stem + ".GZ");
                Assert.ThrowsException<UnicodeDecodeError>(() => np.loadtxt(stem + ".GZ", np.dtypes.StringDType()));
                File.WriteAllBytes(stem + ".txt.bz2", new byte[] { 0x42, 0x5A, 0x68 });
                Assert.ThrowsException<NotSupportedException>(() => np.loadtxt(stem + ".txt.bz2", np.dtypes.StringDType()));
                Throws<FileNotFoundException>(() => np.loadtxt(stem + ".nope", np.dtypes.StringDType()), $"{stem}.nope not found.");
            }
            finally
            {
                foreach (string f in Directory.GetFiles(Path.GetDirectoryName(stem), Path.GetFileName(stem) + "*"))
                    File.Delete(f);
            }
        }

        // ---- savetxt over the codecs -------------------------------------------------------------------------------

        /// <summary>A byte stream encodes each write with <c>str.encode</c>: a BOM codec's BOM leads every write.</summary>
        [TestMethod]
        public void SaveTxt_Stream_BomCodec_WritesABomPerWrite()
        {
            using var ms = new MemoryStream();
            np.savetxt(ms, np.array(new[] { "a", "b" }, np.dtypes.StringDType()), "%s", header: "h", encoding: "utf-16");
            CollectionAssert.AreEqual(new byte[]
            {
                0xFF, 0xFE, 0x23, 0, 0x20, 0, 0x68, 0, 0x0A, 0, 0xFF, 0xFE, 0x61, 0, 0x0A, 0, 0xFF, 0xFE, 0x62, 0, 0x0A, 0,
            }, ms.ToArray());
            // Python's `encoding or 'latin1'`: the empty name is Latin-1 for a stream.
            using var latin = new MemoryStream();
            np.savetxt(latin, np.array(new[] { "é" }, np.dtypes.StringDType()), "%s", encoding: "");
            CollectionAssert.AreEqual(new byte[] { 0xE9, 0x0A }, latin.ToArray());
        }

        /// <summary>A file's TextIOWrapper writes one BOM, names <c>utf-16</c> in the first write's error and
        ///     <c>utf-16-le</c> after it, and encodes an empty first write to the BOM alone.</summary>
        [TestMethod]
        public void SaveTxt_File_BomCodec_FollowsTextIOWrapper()
        {
            string path = TempFile();
            try
            {
                np.savetxt(path, np.array(new[] { "a", "b" }, np.dtypes.StringDType()), "%s", newline: "|", encoding: "utf-16");
                CollectionAssert.AreEqual(new byte[] { 0xFF, 0xFE, 0x61, 0, 0x7C, 0, 0x62, 0, 0x7C, 0 }, File.ReadAllBytes(path));
                Throws<UnicodeEncodeError>(() => np.savetxt(path, np.array(new[] { "x\ud800" }, "U2"), "%s", newline: "|", encoding: "utf-16"),
                    "'utf-16' codec can't encode character '\\ud800' in position 1: surrogates not allowed");
                Assert.AreEqual(0, File.ReadAllBytes(path).Length, "the failed first write leaves the file empty");
                Throws<UnicodeEncodeError>(() => np.savetxt(path, np.array(new[] { "ab", "x\ud800" }, "U2"), "%s", newline: "|", encoding: "utf-16"),
                    "'utf-16-le' codec can't encode character '\\ud800' in position 1: surrogates not allowed");
                CollectionAssert.AreEqual(new byte[] { 0xFF, 0xFE, 0x61, 0, 0x62, 0, 0x7C, 0 }, File.ReadAllBytes(path));
                // Two rows of no columns with newline="": two empty writes, the first of which is encoded to the BOM.
                np.savetxt(path, np.zeros(new Shape(2, 0), np.dtypes.StringDType()), "%s", newline: "", encoding: "utf-16");
                CollectionAssert.AreEqual(new byte[] { 0xFF, 0xFE }, File.ReadAllBytes(path));
                np.savetxt(path, np.array(new[] { "a" }, np.dtypes.StringDType()), "%s", newline: "|", encoding: "utf-8-sig");
                CollectionAssert.AreEqual(new byte[] { 0xEF, 0xBB, 0xBF, 0x61, 0x7C }, File.ReadAllBytes(path));
                // TextIOWrapper's encodefunc for 'iso8859-1' is Latin-1, whatever module the name resolved to.
                Throws<UnicodeEncodeError>(() => np.savetxt(path, np.array(new[] { "Ā" }, np.dtypes.StringDType()), "%s", encoding: "iso8859_1"),
                    "'latin-1' codec can't encode character '\\u0100' in position 0: ordinal not in range(256)");
            }
            finally
            {
                File.Delete(path);
            }
        }

        /// <summary>TextIOWrapper translates <c>'\n'</c> to the platform separator before it encodes, so an encode error's
        ///     position counts the translated write (one more per newline before it on Windows).</summary>
        [TestMethod]
        public void SaveTxt_File_EncodeErrorPosition_CountsTheTranslatedWrite()
        {
            string path = TempFile();
            try
            {
                // The header write is "# h\n# 中\n": '中' is at index 6, after one '\n'.
                int position = 6 + (Environment.NewLine.Length - 1);
                Throws<UnicodeEncodeError>(() => np.savetxt(path, np.array(new[] { "a" }, np.dtypes.StringDType()), "%s",
                        header: "h\n中", encoding: "ascii"),
                    $"'ascii' codec can't encode character '\\u4e2d' in position {position}: ordinal not in range(128)");
            }
            finally
            {
                File.Delete(path);
            }
        }

        /// <summary>A gzip name round-trips (Python's and .NET's gzip bytes differ, so this is not a corpus case); a bz2 /
        ///     LZMA name is refused before the file is touched.</summary>
        [TestMethod]
        public void SaveTxt_File_CompressedNames()
        {
            string gz = TempFile(".txt.gz"), bz2 = TempFile(".txt.bz2");
            try
            {
                np.savetxt(gz, np.array(new[] { "a", "é" }, np.dtypes.StringDType()), "%s");
                using (var a = np.loadtxt(gz, np.dtypes.StringDType()))
                    CollectionAssert.AreEqual(new[] { "a", "é" }, Texts(a));
                File.WriteAllText(bz2, "keep");
                Assert.ThrowsException<NotSupportedException>(() =>
                    np.savetxt(bz2, np.array(new[] { "a" }, np.dtypes.StringDType()), "%s"));
                Assert.AreEqual("keep", File.ReadAllText(bz2), "a refused name is not truncated");
                // A file name's empty encoding is a lookup like any other (only a stream's is Python's `or 'latin1'`).
                Throws<LookupError>(() => np.savetxt(gz, np.array(new[] { "a" }, np.dtypes.StringDType()), "%s", encoding: ""),
                    "unknown encoding: ");
            }
            finally
            {
                File.Delete(gz);
                File.Delete(bz2);
            }
        }

        // ---- loadtxt's comments -------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>comments</c> as Python objects: a non-iterable argument is <c>'int' object is not iterable</c> (before the
        ///     ndmin check); a None element keeps each line's first word, a blank line then being <c>str.split</c>'s
        ///     IndexError; another object raises at the first line; <c>delimiter in comments</c> is Python's <c>in</c>.
        /// </summary>
        [TestMethod]
        public void LoadTxt_Comments_AsPythonObjects()
        {
            var T = np.dtypes.StringDType();
            Throws<TypeError>(() => np.loadtxt(new[] { "a" }, T, 5, ndmin: 3), "'int' object is not iterable");
            Throws<TypeError>(() => np.loadtxt(new[] { "a" }, T, 2.5), "'float' object is not iterable");
            Throws<ValueError>(() => np.loadtxt(new[] { "a" }, T, new object[] { 5 }, ndmin: 3), "Illegal value of ndmin keyword: 3");
            Throws<TypeError>(() => np.loadtxt(new[] { "a b" }, T, new object[] { 5 }), "must be str or None, not int");
            Throws<TypeError>(() => np.loadtxt(new[] { "a b" }, T, new object[] { true }), "must be str or None, not bool");
            using (var a = np.loadtxt(new string[0], T, new object[] { 5 }))
                Assert.AreEqual(0, a.size, "no line, no split");
            Throws<TypeError>(() => np.loadtxt(new[] { "a b" }, T, new object[] { "#", null }),
                "Comment characters '('#', None)' cannot include the delimiter 'None'");
            Throws<TypeError>(() => np.loadtxt(new[] { "a b" }, T, new object[] { 5, "," }, ","),
                "Comment characters '(5, ',')' cannot include the delimiter ','");
            using (var a = np.loadtxt(new[] { "a b # c", "  d e" }, T, new object[] { null }))
                CollectionAssert.AreEqual(new[] { "a", "d" }, Texts(a));
            Throws<IndexError>(() => np.loadtxt(new[] { "", "a b" }, T, new object[] { null }, skiprows: 1), "list index out of range");
            Throws<AttributeError>(() => np.loadtxt(new[] { "a", null }, T, new[] { "#", "%" }), "'NoneType' object has no attribute 'split'");
            Throws<TypeError>(() => np.loadtxt(new[] { "a", null }, T, "#"), "non-string returned while reading data");
            // A value tuple is a Python tuple; a byte[] element is decoded as Latin-1; a char is a one-character str.
            using (var a = np.loadtxt(new[] { "a b % c" }, T, ValueTuple.Create("%")))
                CollectionAssert.AreEqual(new[] { "a", "b" }, Texts(a));
            using (var a = np.loadtxt(new[] { "a b % c # d" }, T, new object[] { new byte[] { (byte)'#' }, '%' }))
                CollectionAssert.AreEqual(new[] { "a", "b" }, Texts(a));
        }

        // ---- fromfile ----------------------------------------------------------------------------------------------

        /// <summary><c>np.fromfile</c> over an open stream checks offset-with-a-separator before the StringDType refusal,
        ///     as <c>array_fromfile</c> does for a file object.</summary>
        [TestMethod]
        public void FromFile_Stream_OffsetWithSeparator_IsCheckedFirst()
        {
            using var ms = new MemoryStream(Encoding.ASCII.GetBytes("a b"));
            Throws<TypeError>(() => np.fromfile(ms, np.dtypes.StringDType(), sep: " ", offset: 1),
                "'offset' argument only permitted for binary files");
            Throws<ValueError>(() => np.fromfile(ms, np.dtypes.StringDType(), sep: " "), "Cannot read into object array");
        }
    }
}
