using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Numerics;
using System.Text;

namespace NumSharp.Tests.IO
{
    /// <summary>
    ///     Text I/O of NEP 55 <c>StringDType</c> against NumPy 2.4.2 (strings phase 1, P1.7c; every expected value probed on
    ///     2026-10-15): <c>np.loadtxt</c> into a <c>StringDType</c> (the NA objects, <c>coerce=False</c>, converters of every
    ///     value kind, the four sources, large inputs and fields across the reader's chunks), <c>np.savetxt</c> of one (the NA
    ///     element, the Latin-1 default of a byte stream, an unknown codec), <c>ndarray.tofile</c> and the refusals of
    ///     <c>np.fromfile</c> / <c>np.fromstring</c> / <c>np.frombuffer</c>.
    /// </summary>
    /// <remarks>
    ///     The string oracle's section tio replays 5,488 NumPy cases of the same functions; these tests pin what a corpus
    ///     cannot carry: C#-only argument kinds (a <see cref="Func{T, TResult}"/> returning a value tuple or a dictionary, a
    ///     <c>.gz</c> file name, a <see cref="Stream"/> target of <c>tofile</c>), inputs too large for the corpus, the exact
    ///     exception TYPES (a test with <c>ThrowsException&lt;TypeError&gt;</c> requires the class itself), the inner
    ///     exceptions NumPy chains as <c>__cause__</c>, and the files left behind after an error.
    /// </remarks>
    [TestClass]
    public class StringDTypeTextIOTests
    {
        /// <summary>The plain <c>StringDType()</c>.</summary>
        private static DType TD => np.dtypes.StringDType();

        /// <summary><c>StringDType(na_object=None)</c>.</summary>
        private static DType TDN => np.dtypes.StringDType(na_object: null);

        /// <summary><c>StringDType(na_object=float('nan'))</c>.</summary>
        private static DType TDNAN => np.dtypes.StringDType(na_object: double.NaN);

        /// <summary><c>StringDType(na_object='NA')</c>.</summary>
        private static DType TDS => np.dtypes.StringDType(na_object: "NA");

        /// <summary><c>StringDType(coerce=False)</c>.</summary>
        private static DType TDC => np.dtypes.StringDType(false);

        /// <summary>A fresh temp file path, deleted by the caller's finally.</summary>
        /// <param name="suffix">The file name's suffix (<c>.gz</c> selects gzip in loadtxt / savetxt).</param>
        /// <returns>The path.</returns>
        private static string TempFile(string suffix = ".txt")
            => Path.Combine(Path.GetTempPath(), "ns_tio_" + Guid.NewGuid().ToString("N") + suffix);

        /// <summary>The elements of an array in C order: a <see cref="string"/>, or the descriptor's NA object for an NA cell.</summary>
        /// <param name="a">A StringDType array.</param>
        /// <returns>The elements.</returns>
        private static object[] Values(NDArray a)
        {
            using var c = a.copy();
            return Enumerable.Range(0, (int)c.size).Select(i => c.Storage.GetAtIndex(i)).ToArray();
        }

        /// <summary>Whether each element of an array is an NA cell (its MISSING flag, which tells a string NA object apart
        ///     from the same text).</summary>
        /// <param name="a">A StringDType array.</param>
        /// <returns>One flag per element in C order.</returns>
        private static bool[] Missing(NDArray a)
        {
            var cells = a.tobytes('C');
            return Enumerable.Range(0, (int)a.size).Select(i => (cells[16 * i + 15] & 0x80) != 0).ToArray();
        }

        /// <summary>Asserts the exact exception type and its message.</summary>
        /// <typeparam name="T">The exception type (exactly).</typeparam>
        /// <param name="f">The call.</param>
        /// <param name="msg">NumPy's message.</param>
        /// <returns>The exception, for its inner one.</returns>
        private static T Throws<T>(Action f, string msg) where T : Exception
        {
            var e = Assert.ThrowsException<T>(f);
            Assert.AreEqual(msg, e.Message);
            return e;
        }

        // ---- np.loadtxt ------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Guards that every field is stored whole in a <c>StringDType</c> result, whatever its length (short cells, the
        ///     medium and the long arena kinds) or script, and that the result carries the requested descriptor.
        /// </summary>
        [TestMethod]
        public void LoadTxt_StringDType_StoresEveryFieldWhole()
        {
            var a = np.loadtxt(new[] { "ab " + new string('m', 20), "é " + new string('z', 300) }, TD);
            a.shape.Should().Equal(2L, 2L);
            a.dtype.Should().Be(TD);
            Values(a).Should().Equal("ab", new string('m', 20), "é", new string('z', 300));
            a.Shape.IsContiguous.Should().BeTrue();
            a.@base.Should().BeNull("read_rows' result owns its data when no axis is squeezed");
        }

        /// <summary>
        ///     Guards NumPy's NA rule for a field: a field EQUAL to a string <c>na_object</c> is an NA cell, while a field that
        ///     reads like another kind of NA object (<c>"None"</c>, <c>"nan"</c>) is just that text.
        /// </summary>
        [TestMethod]
        public void LoadTxt_NAObject_OnlyAnEqualStringFieldIsMissing()
        {
            var s = np.loadtxt(new[] { "ab NA", "NA cd" }, TDS);
            Missing(s).Should().Equal(false, true, true, false);
            s.dtype.Should().Be(TDS);
            Values(np.loadtxt(new[] { "None x" }, TDN)).Should().Equal("None", "x");
            Values(np.loadtxt(new[] { "nan x" }, TDNAN)).Should().Equal("nan", "x");
            Missing(np.loadtxt(new[] { "nan x" }, TDNAN)).Should().Equal(false, false);
        }

        /// <summary>
        ///     Guards stringdtype_setitem for a converter's value: the NA object first (None for na_object=None, a NaN for
        ///     na_object=nan), then <c>str()</c> of the Python object — a list, a tuple, a dict, a float's repr, an int past
        ///     int64, a complex, a bool, an np.float16 — bytes decoded as UTF-8, and a 0-d array CAST into the descriptor (so
        ///     even <c>coerce=False</c> takes it) where an N-d array is <c>str()</c>-ed.
        /// </summary>
        [TestMethod]
        public void LoadTxt_Converters_StoreEveryValueKindLikeSetitem()
        {
            object One(DType dt, Func<string, object> f) => Values(np.loadtxt(new[] { "a" }, dt, converters: f))[0];

            One(TD, _ => null).Should().Be("None");
            One(TDN, _ => null).Should().BeNull();
            Missing(np.loadtxt(new[] { "a" }, TDN, converters: (Func<string, object>)(_ => null))).Should().Equal(true);
            double.IsNaN((double)One(TDNAN, _ => double.NaN)).Should().BeTrue("the NaN NA object reads back as the NaN");
            Missing(np.loadtxt(new[] { "a" }, TDNAN, converters: (Func<string, object>)(_ => double.NaN))).Should().Equal(true);
            One(TD, s => Encoding.UTF8.GetBytes(s)).Should().Be("a");
            One(TD, s => new object[] { s, s }).Should().Be("['a', 'a']");
            One(TD, s => ValueTuple.Create(s)).Should().Be("('a',)");
            One(TD, s => new Dictionary<string, long> { [s] = 1 }).Should().Be("{'a': 1}");
            One(TDC, _ => np.array(5L)).Should().Be("5", "a 0-d array is cast, not coerced");
            One(TD, _ => np.array(new long[] { 5, 6 })).Should().Be("[5 6]");
            One(TD, _ => (Half)0.5).Should().Be("0.5");
            One(TD, _ => 1.0 / 3).Should().Be("0.3333333333333333");
            One(TD, _ => BigInteger.Pow(10, 20)).Should().Be("100000000000000000000");
            One(TD, _ => new Complex(1, -2)).Should().Be("(1-2j)");
            One(TD, _ => true).Should().Be("True");
        }

        /// <summary>
        ///     Guards the conversion error of a converter's value: NumPy's <c>could not convert string … to StringDType(…) at
        ///     row r, column c.</c> with setitem's own error as the cause — coerce=False refusing a number, bytes that are not
        ///     UTF-8, a str with a lone surrogate.
        /// </summary>
        [TestMethod]
        public void LoadTxt_ConverterValueRefused_ChainsSetitemsError()
        {
            var e = Throws<ValueError>(() => np.loadtxt(new[] { "1 2" }, TDC, converters: (Func<string, object>)(_ => 1L)),
                "could not convert string '1' to StringDType(coerce=False) at row 0, column 1.");
            e.InnerException.Should().BeOfType<ValueError>()
                .Which.Message.Should().Be("StringDType only allows string data when string coercion is disabled.");

            e = Throws<ValueError>(() => np.loadtxt(new[] { "a" }, TD, converters: (Func<string, object>)(_ => new byte[] { 0x80 })),
                "could not convert string 'a' to StringDType() at row 0, column 1.");
            e.InnerException.Should().BeOfType<UnicodeDecodeError>()
                .Which.Message.Should().Be("'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte");

            e = Throws<ValueError>(() => np.loadtxt(new[] { "a" }, TD, converters: (Func<string, object>)(_ => "\ud800")),
                "could not convert string 'a' to StringDType() at row 0, column 1.");
            e.InnerException.Should().BeOfType<UnicodeEncodeError>()
                .Which.Message.Should().Be("'utf-8' codec can't encode character '\\ud800' in position 0: surrogates not allowed");
        }

        /// <summary>
        ///     Guards <c>encoding='bytes'</c>: a converter's input must survive NumPy's Latin-1 encode of the field, so a field
        ///     past U+00FF fails its conversion with the UnicodeEncodeError as the cause, while a Latin-1 field reaches the
        ///     converter. (NumPy hands the converter the BYTES; a C# converter receives the same characters as a string — the
        ///     documented divergence, so the probe is a converter that ignores the kind of its input.)
        /// </summary>
        [TestMethod]
        public void LoadTxt_EncodingBytes_FieldMustEncodeAsLatin1()
        {
            Values(np.loadtxt(new[] { "é b" }, TD, converters: (Func<string, object>)(s => (long)s.Length), encoding: "bytes"))
                .Should().Equal("1", "1");
            var e = Throws<ValueError>(
                () => np.loadtxt(new[] { "中 b" }, TD, converters: (Func<string, object>)(s => (long)s.Length), encoding: "bytes"),
                "could not convert string '中' to StringDType() at row 0, column 1.");
            e.InnerException.Should().BeOfType<UnicodeEncodeError>().Which.Message.Should()
                .Be("'latin-1' codec can't encode character '\\u4e2d' in position 0: ordinal not in range(256)");
        }

        /// <summary>
        ///     Guards that a byte stream meets an unknown codec where NumPy does — at its FIRST line (NumPy decodes each line as
        ///     it reads it) — so a stream without lines loads its empty result, while a file name meets it at the open.
        /// </summary>
        [TestMethod]
        public void LoadTxt_UnknownEncoding_StreamAtTheFirstLine_FileAtTheOpen()
        {
            using (var empty = new MemoryStream(Array.Empty<byte>()))
            {
                var r = np.loadtxt(empty, TD, encoding: "nosuchcodec");
                r.shape.Should().Equal(0L);
            }
            using (var one = new MemoryStream(Encoding.ASCII.GetBytes("a b\n")))
                Throws<LookupError>(() => np.loadtxt(one, TD, encoding: "nosuchcodec"), "unknown encoding: nosuchcodec");

            var path = TempFile();
            try
            {
                File.WriteAllText(path, "a b\n");
                Throws<LookupError>(() => np.loadtxt(path, TD, encoding: "nosuchcodec"), "unknown encoding: nosuchcodec");
            }
            finally
            {
                File.Delete(path);
            }
        }

        /// <summary>
        ///     Guards NumPy's text for a missing file name (<c>DataSource.open</c>'s <c>&lt;path&gt; not found.</c>, a
        ///     <see cref="FileNotFoundException"/>), reported before an unknown encoding is looked up.
        /// </summary>
        [TestMethod]
        public void LoadTxt_MissingFile_IsNumPysNotFound()
        {
            var path = TempFile();
            Throws<FileNotFoundException>(() => np.loadtxt(path, TD), $"{path} not found.");
            Throws<FileNotFoundException>(() => np.loadtxt(path, TD, encoding: "nosuchcodec"), $"{path} not found.");
        }

        /// <summary>Guards a <c>.gz</c> file name: decompressed transparently, then decoded with the given encoding.</summary>
        [TestMethod]
        public void LoadTxt_GzipFile_IsDecompressed()
        {
            var path = TempFile(".txt.gz");
            try
            {
                using (var fs = File.Create(path))
                using (var gz = new GZipStream(fs, CompressionLevel.Fastest))
                    gz.Write(Encoding.UTF8.GetBytes("ab cd\né x\n"));
                Values(np.loadtxt(path, TD, encoding: "utf-8")).Should().Equal("ab", "cd", "é", "x");
            }
            finally
            {
                File.Delete(path);
            }
        }

        /// <summary>
        ///     Guards a large input from every source: 50,000 rows whose second field cycles through every length from 1 to
        ///     300 (short cells and both arena kinds), so the staging buffers grow many times and the file reader crosses
        ///     many chunks — the values must still be NumPy's, row for row.
        /// </summary>
        [TestMethod]
        public void LoadTxt_LargeInput_FromEverySource()
        {
            var lines = Enumerable.Range(0, 50000).Select(i => $"r{i} {new string('q', i % 300 + 1)}").ToArray();
            var expected = lines.SelectMany(l => l.Split(' ')).Cast<object>().ToArray();
            string text = string.Join("\n", lines) + "\n";

            Values(np.loadtxt(lines, TD)).Should().Equal(expected);
            using (var reader = new StringReader(text))
                Values(np.loadtxt(reader, TD)).Should().Equal(expected);
            using (var stream = new MemoryStream(Encoding.ASCII.GetBytes(text)))
                Values(np.loadtxt(stream, TD)).Should().Equal(expected);
            var path = TempFile();
            try
            {
                File.WriteAllText(path, text);
                var a = np.loadtxt(path, TD, encoding: "utf-8");
                a.shape.Should().Equal(50000L, 2L);
                Values(a).Should().Equal(expected);
            }
            finally
            {
                File.Delete(path);
            }
        }

        /// <summary>
        ///     Guards a quoted field that spans lines AND the file reader's 16 KiB chunks: 20,000 characters, a newline, 20,000
        ///     more — one field of 40,001 characters, the newline inside it kept.
        /// </summary>
        [TestMethod]
        public void LoadTxt_QuotedFieldAcrossChunks_IsOneField()
        {
            string big = new string('a', 20000);
            var path = TempFile();
            try
            {
                File.WriteAllText(path, $"x \"{big}\n{big}\" y\nz w w\n");
                var a = np.loadtxt(path, TD, quotechar: "\"", encoding: "utf-8");
                a.shape.Should().Equal(2L, 3L);
                var v = Values(a);
                v[1].Should().Be(big + "\n" + big);
                v.Skip(3).Should().Equal("z", "w", "w");
            }
            finally
            {
                File.Delete(path);
            }
        }

        // ---- np.savetxt ------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Guards that a row is formatted as Python objects: the NA element is the descriptor's na_object itself (None, a
        ///     float NaN, the string <c>'NA'</c>), so <c>%s</c> prints its <c>str()</c> and <c>%r</c> its <c>repr()</c>.
        /// </summary>
        [TestMethod]
        public void SaveTxt_NAElement_IsTheNAObject()
        {
            string Save(NDArray a, string fmt)
            {
                var sw = new StringWriter();
                np.savetxt(sw, a, fmt);
                return sw.ToString();
            }
            var none = np.array(new object[] { "ab", null, "c" }, TDN);
            var nan = np.array(new object[] { "ab", double.NaN, "c" }, TDNAN);
            var na = np.array(new object[] { "ab", "NA", "c" }, TDS);
            Save(none, "%s").Should().Be("ab\nNone\nc\n");
            Save(none, "%r").Should().Be("'ab'\nNone\n'c'\n");
            Save(nan, "%s").Should().Be("ab\nnan\nc\n");
            Save(nan, "%r").Should().Be("'ab'\nnan\n'c'\n");
            Save(na, "%s").Should().Be("ab\nNA\nc\n");
            Save(na, "%r").Should().Be("'ab'\n'NA'\n'c'\n");
        }

        /// <summary>
        ///     Guards the default <c>%.18e</c> over text: NumPy's Mismatch TypeError (the class itself), naming the descriptor's
        ///     <c>str()</c> and the row format, with CPython's <c>must be real number, not str</c> as the cause.
        /// </summary>
        [TestMethod]
        public void SaveTxt_DefaultFormat_IsTheMismatch()
        {
            var e = Throws<TypeError>(() => np.savetxt(new StringWriter(), np.array(new[,] { { "a", "b" } }, TD)),
                "Mismatch between array dtype ('StringDType()') and format specifier ('%.18e %.18e')");
            e.InnerException!.Message.Should().Be("must be real number, not str");
        }

        /// <summary>
        ///     Guards a byte stream's Latin-1 default (NumPy's <c>WriteWrap(fh, encoding or 'latin1')</c>): rows encode one by
        ///     one, so a row past U+00FF raises UnicodeEncodeError with the header and the rows before it in the stream.
        /// </summary>
        [TestMethod]
        public void SaveTxt_ByteStream_Latin1_KeepsTheRowsBeforeTheError()
        {
            var ms = new MemoryStream();
            Throws<UnicodeEncodeError>(() => np.savetxt(ms, np.array(new[] { "ab", "é", "中", "x" }, TD), "%s", header: "H"),
                "'latin-1' codec can't encode character '\\u4e2d' in position 0: ordinal not in range(256)");
            ms.ToArray().Should().Equal(new byte[] { (byte)'#', (byte)' ', (byte)'H', 10, (byte)'a', (byte)'b', 10, 0xE9, 10 });
        }

        /// <summary>
        ///     Guards an unknown codec: a byte stream meets it at its first write (an empty array writes nothing, so nothing
        ///     raises), a text writer never (NumPy encodes only for a stream that refuses str), a file name at the open — after
        ///     the file was created, which stays behind empty (NumPy's <c>open(fname, 'wt').close()</c> runs first).
        /// </summary>
        [TestMethod]
        public void SaveTxt_UnknownEncoding_MetWhereNumPyMeetsIt()
        {
            var ms = new MemoryStream();
            Throws<LookupError>(() => np.savetxt(ms, np.array(new[] { "a" }, TD), "%s", encoding: "nosuchcodec"),
                "unknown encoding: nosuchcodec");
            ms.Length.Should().Be(0);
            var empty = new MemoryStream();
            np.savetxt(empty, np.array(Array.Empty<string>(), TD), "%s", encoding: "nosuchcodec");
            empty.Length.Should().Be(0);

            var sw = new StringWriter();
            np.savetxt(sw, np.array(new[] { "é" }, TD), "%s", encoding: "nosuchcodec");
            sw.ToString().Should().Be("é\n");

            var path = TempFile();
            try
            {
                Throws<LookupError>(() => np.savetxt(path, np.array(new[] { "a" }, TD), "%s", encoding: "bytes"),
                    "unknown encoding: bytes");
                File.Exists(path).Should().BeTrue();
                new FileInfo(path).Length.Should().Be(0);
            }
            finally
            {
                File.Delete(path);
            }
        }

        /// <summary>
        ///     Guards that savetxt's complex branch raises CPython's plain TypeError (NumPy does not wrap it there), the class
        ///     itself rather than the formatter's internal subclass.
        /// </summary>
        [TestMethod]
        public void SaveTxt_ComplexBranch_RaisesThePlainTypeError()
        {
            Throws<TypeError>(() => np.savetxt(new StringWriter(), np.array(new[] { new Complex(1, 2) }), "%c"),
                "%c requires int or char");
        }

        // ---- ndarray.tofile and the readers -----------------------------------------------------------------------------

        /// <summary>
        ///     Guards the binary refusal: NumPy opens (creates / truncates) the file, then raises OSError, so an empty file is
        ///     left behind.
        /// </summary>
        [TestMethod]
        public void ToFile_Binary_IsOSError_AfterTheFileIsCreated()
        {
            var path = TempFile(".bin");
            try
            {
                Throws<OSError>(() => np.array(new[] { "a" }, TD).tofile(path),
                    "cannot write object arrays to a file in binary mode");
                File.ReadAllBytes(path).Should().BeEmpty();
            }
            finally
            {
                File.Delete(path);
            }
        }

        /// <summary>
        ///     Guards the text mode: each element is <c>format % obj</c> of the Python object (the NA object included), joined
        ///     by the separator; a conversion that does not apply raises CPython's plain TypeError (the class itself) before
        ///     anything is written — through a file name and through a <see cref="Stream"/> alike.
        /// </summary>
        [TestMethod]
        public void ToFile_Text_FormatsPythonObjects()
        {
            var a = np.array(new object[] { "ab", null, "c" }, TDN);
            string Write(string sep, string fmt)
            {
                var path = TempFile();
                try
                {
                    a.tofile(path, sep, fmt);
                    return Encoding.Latin1.GetString(File.ReadAllBytes(path));
                }
                finally
                {
                    File.Delete(path);
                }
            }
            Write(",", "%s").Should().Be("ab,None,c");
            Write(" ", "%r").Should().Be("'ab' None 'c'");
            Write("|", "%5s").Should().Be("   ab| None|    c");
            using (var ms = new MemoryStream())
            {
                a.tofile(ms, ",", "%s");
                Encoding.ASCII.GetString(ms.ToArray()).Should().Be("ab,None,c");
            }

            var p = TempFile();
            try
            {
                Throws<TypeError>(() => a.tofile(p, ",", "%d"), "%d format: a real number is required, not str");
                File.ReadAllBytes(p).Should().BeEmpty();
            }
            finally
            {
                File.Delete(p);
            }
        }

        /// <summary>Guards the readers' refusals of a reference dtype, with NumPy's texts.</summary>
        [TestMethod]
        public void Readers_RefuseStringDType()
        {
            var path = TempFile();
            try
            {
                File.WriteAllText(path, "a b");
                Throws<ValueError>(() => np.fromfile(path, TD, sep: " "), "Cannot read into object array");
                Throws<TypeError>(() => np.fromfile(path, TD, sep: " ", offset: 1), "'offset' argument only permitted for binary files");
            }
            finally
            {
                File.Delete(path);
            }
            Throws<ValueError>(() => np.fromstring("a b", TD, sep: " "), "Cannot create an object array from a string");
            Throws<ValueError>(() => np.frombuffer(Encoding.ASCII.GetBytes(new string('a', 16)), TD),
                "cannot create an OBJECT array from memory buffer");
        }
    }
}
