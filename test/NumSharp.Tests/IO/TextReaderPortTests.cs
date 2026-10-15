using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;

namespace NumSharp.Tests.IO
{
    /// <summary>
    ///     The text readers ported from NumPy 2.4.2: <c>np.fromstring</c> / <c>np.fromfile</c>'s <c>array_from_text</c>
    ///     (<c>fromstr</c> = CPython's <c>PyOS_strtol</c> / <c>PyOS_strtoul</c> / NumPy's <c>NumPyOS_ascii_strtod</c>,
    ///     <c>scan</c> = glibc's <c>fscanf</c>), the binary <c>fread</c> / <c>fwrite</c> paths, and <c>np.loadtxt</c>'s
    ///     tokenizer fast path and per-row stores. Every expected value was probed against NumPy 2.4.2 on Linux (LP64: a C
    ///     <c>long</c> is 64 bits; Windows NumPy's 32-bit <c>long</c> differs in the integer cells, see the CLAUDE.md
    ///     entry), the probe script being <c>probe_tests.py</c> of the validation pass.
    /// </summary>
    [TestClass]
    public class TextReaderPortTests
    {
        // ---------------------------------------------------------------------------------------------
        // np.fromstring
        // ---------------------------------------------------------------------------------------------

        [TestMethod]
        public void FromString_CountPastTheData_KeepsTheCount()
        {
            // NumPy never shrinks a counted text read: (4,), the elements read first (NumPy leaves the rest uninitialized,
            // NumSharp zeroes it).
            var a = np.fromstring("1,2", np.int64, 4, ",");
            Assert.AreEqual(4, a.size);
            CollectionAssert.AreEqual(new long[] { 1, 2, 0, 0 }, a.ToArray<long>());
        }

        [TestMethod]
        public void FromString_FailedScanWritesItsValue()
        {
            // fromstr writes its result before the reader judges the progress: the scan of an empty string stores 0, -1.0
            // or -1+0j in the first slot (probed).
            Assert.AreEqual(-1.0, np.fromstring("", np.float64, 2, ",").GetDouble(0));
            Assert.AreEqual(0L, np.fromstring("", np.int64, 2, ",").GetInt64(0));
            Assert.AreEqual(new Complex(-1, 0), (Complex)np.fromstring("", np.complex128, 2, ",").GetAtIndex(0));
        }

        [TestMethod]
        public void FromString_TrailingWhitespace_IsAnElement()
        {
            // The scanners skip leading whitespace and report the position after it, so whitespace alone reads as one
            // element holding the failed scan's value (probed: [1., -1.], [1, 0], [-1.], [0]).
            CollectionAssert.AreEqual(new[] { 1.0, -1.0 }, np.fromstring("1, ", np.float64, -1, ",").ToArray<double>());
            CollectionAssert.AreEqual(new long[] { 1, 0 }, np.fromstring("1, ", np.int64, -1, ",").ToArray<long>());
            CollectionAssert.AreEqual(new[] { -1.0 }, np.fromstring("   ", np.float64, -1, ",").ToArray<double>());
            CollectionAssert.AreEqual(new long[] { 0 }, np.fromstring("   ", np.int64, -1, ",").ToArray<long>());
            Assert.AreEqual(0, np.fromstring("", np.float64, -1, ",").size);
        }

        [TestMethod]
        public void FromString_SeparatorRules()
        {
            // A trailing separator ends the read (the end is tested before the separator's match), a leading one fails,
            // whitespace in the separator is a wildcard and runs of it collapse.
            CollectionAssert.AreEqual(new[] { 1.0, 2.0 }, np.fromstring("1,2,", np.float64, -1, ",").ToArray<double>());
            AssertUnmatched(() => np.fromstring(",1,2", np.float64, -1, ","));
            CollectionAssert.AreEqual(new long[] { 1, 2, 3 }, np.fromstring("1 \t\n 2\n3", np.int64, -1, " ").ToArray<long>());
            CollectionAssert.AreEqual(new long[] { 1, 2, 3 }, np.fromstring("1 ,2,  3", np.int64, -1, ",").ToArray<long>());
            CollectionAssert.AreEqual(new long[] { 1, 2 }, np.fromstring("1 , 2", np.int64, -1, " , ").ToArray<long>());
            // The separator after the last requested element is optional, and count 0 reads nothing.
            CollectionAssert.AreEqual(new long[] { 1, 2 }, np.fromstring("1 2", np.int64, 2, " ").ToArray<long>());
            Assert.AreEqual(0, np.fromstring("1 2", np.int64, 0, " ").size);
        }

        [TestMethod]
        public void FromString_IntegersFollowCPythonsStrtolWithA64BitLong()
        {
            // PyOS_strtol saturates at LONG_MAX for either sign, except exactly 2^63 with a minus sign.
            Assert.AreEqual(long.MaxValue, np.fromstring("99999999999999999999", np.int64, -1, " ").GetInt64(0));
            Assert.AreEqual(long.MinValue, np.fromstring("-9223372036854775808", np.int64, -1, " ").GetInt64(0));
            Assert.AreEqual(long.MaxValue, np.fromstring("-9223372036854775809", np.int64, -1, " ").GetInt64(0));
            // PyOS_strtoul saturates at ULONG_MAX and takes no sign.
            Assert.AreEqual(ulong.MaxValue, np.fromstring("18446744073709551616", np.uint64, -1, " ").GetUInt64(0));
            Assert.AreEqual(ulong.MaxValue, np.fromstring("18446744073709551615", np.uint64, -1, " ").GetUInt64(0));
            AssertUnmatched(() => np.fromstring("-1", np.uint64, -1, " "));
            AssertUnmatched(() => np.fromstring("-1", np.uint8, -1, " "));
            // A narrower dtype takes the C long's low bits; strtol skips whitespace after the sign; an exponent is junk.
            CollectionAssert.AreEqual(new sbyte[] { 44, 127 }, np.fromstring("300 -129", np.int8, -1, " ").ToArray<sbyte>());
            Assert.AreEqual(1, np.fromstring("4294967297", np.int32, -1, " ").GetInt32(0));
            Assert.AreEqual(-1L, np.fromstring("- 1", np.int64, -1, ",").GetInt64(0));
            AssertUnmatched(() => np.fromstring("1e3", np.int64, -1, " "));
        }

        [TestMethod]
        public void FromString_FloatForms()
        {
            // NumPyOS_ascii_strtod: every NaN form is the POSITIVE quiet NaN ("-nan" included), inf/infinity in any case.
            var nan = np.fromstring("nan(abc) -nan NaN", np.float64, -1, " ").ToArray<double>();
            Assert.IsTrue(nan.All(double.IsNaN));
            Assert.IsTrue(nan.All(d => BitConverter.DoubleToInt64Bits(d) >= 0), "every NaN is positive");
            CollectionAssert.AreEqual(new[] { double.PositiveInfinity, double.NegativeInfinity, double.PositiveInfinity },
                np.fromstring("inf -Infinity +INF", np.float64, -1, " ").ToArray<double>());
            AssertUnmatched(() => np.fromstring("infinit", np.float64, -1, " "));
            AssertUnmatched(() => np.fromstring("0x10", np.float64, -1, " "));
            // A float32 / float16 element is the double rounded once (C's assignment): 0.1, 2^24 + 1, 65520 -> inf.
            CollectionAssert.AreEqual(new uint[] { 1036831949, 1266679808 },
                np.fromstring("0.1 16777217", np.float32, -1, " ").ToArray<float>().Select(BitConverter.SingleToUInt32Bits).ToArray());
            CollectionAssert.AreEqual(new ushort[] { 11878, 31744 },
                np.fromstring("0.1 65520", np.float16, -1, " ").ToArray<Half>().Select(BitConverter.HalfToUInt16Bits).ToArray());
            CollectionAssert.AreEqual(new[] { false, true, true, true }, np.fromstring("0 1 2 -1", np.bool_, -1, " ").ToArray<bool>());
        }

        [TestMethod]
        public void FromString_ComplexQuirks()
        {
            // CDOUBLE_fromstr: "1+2" is 1+0j (the "+2" consumed without its j), a bare "j" is -1j (probed).
            Assert.AreEqual(new Complex(1, 0), (Complex)np.fromstring("1+2", np.complex128, -1, " ").GetAtIndex(0));
            Assert.AreEqual(new Complex(0, -1), (Complex)np.fromstring("j", np.complex128, -1, " ").GetAtIndex(0));
        }

        [TestMethod]
        public void FromString_BytesArgument()
        {
            CollectionAssert.AreEqual(new[] { 1.0, 2.0 }, np.fromstring(Encoding.ASCII.GetBytes("1 2"), np.float64, -1, " ").ToArray<double>());
        }

        [TestMethod]
        public void FromString_ArgumentAndDtypeErrors()
        {
            Assert.AreEqual("embedded null character",
                Assert.ThrowsException<ValueError>(() => np.fromstring("1 2", np.float64, -1, "\0")).Message);
            Assert.AreEqual("Cannot create an object array from a string",
                Assert.ThrowsException<ValueError>(() => np.fromstring("1 2", np.dtypes.StringDType(), -1, " ")).Message);
            Assert.AreEqual("zero-valued itemsize",
                Assert.ThrowsException<ValueError>(() => np.fromstring("1 2", np.dtype("U"), -1, " ")).Message);
            Assert.AreEqual("don't know how to read character strings with that array type",
                Assert.ThrowsException<ValueError>(() => np.fromstring("1 2", np.dtype("U5"), -1, " ")).Message);
        }

        // ---------------------------------------------------------------------------------------------
        // np.fromfile — text (glibc fscanf) and binary (fread)
        // ---------------------------------------------------------------------------------------------

        [TestMethod]
        public void FromFile_TextIntegersFollowGlibcScanf()
        {
            // A sign followed by a non-digit is a matching failure; strtoul negates a sign modulo 2^64; strtol clamps by
            // sign (unlike CPython's strtol); a narrower dtype takes the low bits.
            WithFile("1 -x", p => AssertUnmatched(() => np.fromfile(p, np.int64, -1, " ")));
            WithFile("-1 5", p => CollectionAssert.AreEqual(new[] { ulong.MaxValue, 5UL }, np.fromfile(p, np.uint64, -1, " ").ToArray<ulong>()));
            WithFile("-18446744073709551615", p => CollectionAssert.AreEqual(new[] { 1UL }, np.fromfile(p, np.uint64, -1, " ").ToArray<ulong>()));
            WithFile("-99999999999999999999 99999999999999999999",
                p => CollectionAssert.AreEqual(new[] { long.MinValue, long.MaxValue }, np.fromfile(p, np.int64, -1, " ").ToArray<long>()));
            WithFile("300", p => CollectionAssert.AreEqual(new sbyte[] { 44 }, np.fromfile(p, np.int8, -1, " ").ToArray<sbyte>()));
        }

        [TestMethod]
        public void FromFile_TextScanQuirks()
        {
            // A byte of the separator at or past 0x80 never matches (fgetc's int against a signed char); a dangling
            // exponent or point ends the number; a complex without its j fails.
            WithFile("1é2", p => AssertUnmatched(() => np.fromfile(p, np.int64, -1, "é")));
            WithFile("1e 2", p => CollectionAssert.AreEqual(new[] { 1.0, 2.0 }, np.fromfile(p, np.float64, -1, " ").ToArray<double>()));
            WithFile("1. 2", p => CollectionAssert.AreEqual(new[] { 1.0, 2.0 }, np.fromfile(p, np.float64, -1, " ").ToArray<double>()));
            WithFile("1+2 3", p => AssertUnmatched(() => np.fromfile(p, np.complex128, -1, " ")));
        }

        [TestMethod]
        public void FromFile_TextLeavesTheStreamWhereTheScanStopped()
        {
            // count=2 over "1,2,3,4": both elements and the separator after the second are consumed (position 4).
            WithFile("1,2,3,4", p =>
            {
                using var fs = File.OpenRead(p);
                CollectionAssert.AreEqual(new long[] { 1, 2 }, np.fromfile(fs, np.int64, 2, ",").ToArray<long>());
                Assert.AreEqual(4, fs.Position);
            });
        }

        [TestMethod]
        public void FromFile_BinaryConsumesAsFreadDoes()
        {
            byte[] ten = Encoding.ASCII.GetBytes("abcdefghij");
            // count=-1: the whole items between the position and the end, so a partial trailing item stays unread.
            WithBytes(ten, p =>
            {
                using var fs = File.OpenRead(p);
                CollectionAssert.AreEqual(new[] { 1684234849, 1751606885 }, np.fromfile(fs, np.int32).ToArray<int>());
                Assert.AreEqual(8, fs.Position);
            });
            // A count past the end: fread reads every byte there is, the partial item included.
            WithBytes(ten, p =>
            {
                using var fs = File.OpenRead(p);
                Assert.AreEqual(2, np.fromfile(fs, np.int32, 5).size);
                Assert.AreEqual(10, fs.Position);
            });
        }

        [TestMethod]
        public void FromFile_BinaryOffsetAndByteOrder()
        {
            WithBytes(new byte[] { 1, 0, 2, 0, 3, 0 }, p =>
                CollectionAssert.AreEqual(new short[] { 2, 3 }, np.fromfile(p, np.int16, -1, "", 2).ToArray<short>()));
            WithBytes(new byte[] { 1, 0 }, p => Assert.AreEqual("negative dimensions are not allowed",
                Assert.ThrowsException<ValueError>(() => np.fromfile(p, np.int16, -1, "", 6)).Message));
            WithBytes(new byte[] { 0, 1, 0, 2 }, p =>
                CollectionAssert.AreEqual(new short[] { 1, 2 }, np.fromfile(p, np.dtype(">i2")).ToArray<short>()));
            WithFile("1 2", p => Assert.AreEqual("'offset' argument only permitted for binary files",
                Assert.ThrowsException<TypeError>(() => np.fromfile(p, np.int64, -1, " ", 1)).Message));
        }

        [TestMethod]
        public void FromFile_BinaryUnseekableStream_ReadsToTheEnd()
        {
            var values = Enumerable.Range(0, 70_000).Select(i => (double)i * 0.5).ToArray(); // past the first 64 KiB chunk
            var bytes = new byte[values.Length * 8];
            Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            using var s = new ForwardOnlyStream(new MemoryStream(bytes));
            CollectionAssert.AreEqual(values, np.fromfile(s, np.float64).ToArray<double>());
        }

        [TestMethod]
        public void ToFile_Binary_ContiguousViewAtAnOffset_WritesItsOwnElements()
        {
            // a[2:] is C-contiguous at a non-zero offset: written straight from its first element (no copy).
            var a = np.arange(10).astype(np.int32);
            string p = Path.Combine(Path.GetTempPath(), $"ns_trp_{Guid.NewGuid():N}.bin");
            try
            {
                a["2:"].tofile(p);
                CollectionAssert.AreEqual(Enumerable.Range(2, 8).ToArray(), np.fromfile(p, np.int32).ToArray<int>());
                a["::3"].tofile(p);
                CollectionAssert.AreEqual(new[] { 0, 3, 6, 9 }, np.fromfile(p, np.int32).ToArray<int>());
            }
            finally { File.Delete(p); }
        }

        // ---------------------------------------------------------------------------------------------
        // np.loadtxt
        // ---------------------------------------------------------------------------------------------

        [TestMethod]
        public void LoadTxt_Float16_IsPythonsFloatOfTheField()
        {
            // npy_to_generic: PyArray_Pack of the str, i.e. float(str) — underscores between digits, Unicode digits and
            // Unicode whitespace are accepted (float64's own parser takes none of them).
            CollectionAssert.AreEqual(new[] { (Half)10, (Half)1000.5 }, np.loadtxt(new[] { "1_0", "1_000.5" }, np.float16).ToArray<Half>());
            CollectionAssert.AreEqual(new[] { (Half)12, (Half)12 },
                np.loadtxt(new[] { "١٢", "１２" }, np.float16, delimiter: ",").ToArray<Half>());
            Assert.AreEqual((Half)1.5, (Half)np.loadtxt(new[] { "　1.5　" }, np.float16, delimiter: ",").item());
            Assert.AreEqual("could not convert string '1__0' to float16 at row 0, column 1.",
                Assert.ThrowsException<ValueError>(() => np.loadtxt(new[] { "1__0" }, np.float16)).Message);
            Assert.AreEqual("could not convert string '1_0' to float64 at row 0, column 1.",
                Assert.ThrowsException<ValueError>(() => np.loadtxt(new[] { "1_0" }, np.float64)).Message);
        }

        [TestMethod]
        public void LoadTxt_MaxRowsZero_UnsizedTextClass_ConcatenatesNothing()
        {
            // _read's chunked object read never runs: np.concatenate([]) raises before the C reader checks anything.
            foreach (string dt in new[] { "U", "S" })
                Assert.AreEqual("need at least one array to concatenate",
                    Assert.ThrowsException<ValueError>(() => np.loadtxt(new[] { "a b" }, np.dtype(dt), max_rows: 0)).Message);
            Assert.AreEqual(0, np.loadtxt(new[] { "a b" }, np.dtype("U5"), max_rows: 0).size);
        }

        [TestMethod]
        public void LoadTxt_UnsizedText_WidthAndErrors()
        {
            // The cast of the object array: the widest element's code points (a surrogate pair is one).
            var u = np.loadtxt(new[] { "a\U0001F600b cd" }, np.dtype("U"));
            Assert.AreEqual("<U3", u.dtype.str);
            Assert.AreEqual("a\U0001F600b", (string)u.item(0));
            Assert.AreEqual("cd", (string)u.item(1));
            // An unsized S field is encoded WHOLE as Latin-1: '日' raises, chained into the row / column message.
            Assert.AreEqual("could not convert string '日' to object at row 0, column 2.",
                Assert.ThrowsException<ValueError>(() => np.loadtxt(new[] { "a 日" }, np.dtype("S"))).Message);
        }

        [TestMethod]
        public void LoadTxt_LineSource_NewlinesAndBlankLines()
        {
            Assert.AreEqual("Found an unquoted embedded newline within a single line of input.  This is currently not supported.",
                Assert.ThrowsException<ValueError>(() => np.loadtxt(new[] { "1 2\r3 4" })).Message);
            var expect = new[] { 1.0, 2, 3, 4 };
            CollectionAssert.AreEqual(expect, np.loadtxt(new[] { "1 2\r\n", "3 4\n" }).ToArray<double>());
            CollectionAssert.AreEqual(expect, np.loadtxt(new[] { "1 2\r", "3 4\r" }).ToArray<double>());
            CollectionAssert.AreEqual(expect, np.loadtxt(new[] { "1 2", "   ", "# c", "\t", "3 4 # t" }).ToArray<double>());
        }

        [TestMethod]
        public void LoadTxt_WhitespaceSplit_IsPythonsIsspace()
        {
            // U+3000, U+00A0 and U+2000 separate fields, U+200B (zero-width space) does not, U+001C..U+001F do.
            CollectionAssert.AreEqual(new[] { "1", "2", "3", "4" }, Texts(np.loadtxt(new[] { "1　2 3 4" }, np.dtype("U"))));
            CollectionAssert.AreEqual(new[] { "1​2", "3" }, Texts(np.loadtxt(new[] { "1​2 3" }, np.dtype("U"))));
            CollectionAssert.AreEqual(new[] { "1", "2", "3" }, Texts(np.loadtxt(new[] { "1\u001C2\u001F3" }, np.dtype("U"))));
            CollectionAssert.AreEqual(new[] { "1", "2" }, Texts(np.loadtxt(new[] { "1 2 " }, np.dtype("U"))));
            // A comment character ends the field it is in; a long field continues past the scalar scan.
            Assert.AreEqual("ab", (string)np.loadtxt(new[] { "ab#cd ef" }, np.dtype("U")).item());
            CollectionAssert.AreEqual(new[] { "abcdefghijklmnopqrstuvwxyz0123", "b" },
                Texts(np.loadtxt(new[] { "abcdefghijklmnopqrstuvwxyz0123 b" }, np.dtype("U"))));
        }

        [TestMethod]
        public void LoadTxt_DelimiterSplit_KeepsEmptyFields()
        {
            CollectionAssert.AreEqual(new[] { "1", "2", "" }, Texts(np.loadtxt(new[] { "1,2," }, np.dtype("U"), delimiter: ",")));
            CollectionAssert.AreEqual(new[] { "", "1", "2" }, Texts(np.loadtxt(new[] { ",1,2" }, np.dtype("U"), delimiter: ",")));
            CollectionAssert.AreEqual(new[] { "", "" }, Texts(np.loadtxt(new[] { "," }, np.dtype("U"), delimiter: ",")));
        }

        [TestMethod]
        public void LoadTxt_QuotedLines_TakeTheStateMachine()
        {
            CollectionAssert.AreEqual(new[] { "a b", "c" }, Texts(np.loadtxt(new[] { "\"a b\" c" }, np.dtype("U"), quotechar: "\"")));
            CollectionAssert.AreEqual(new[] { "a\"b", "c" }, Texts(np.loadtxt(new[] { "a\"b c" }, np.dtype("U"), quotechar: "\"")));
        }

        [TestMethod]
        public void LoadTxt_ListGrowingDuringTheRead_IsReadToItsNewEnd()
        {
            // Python's list iterator tests the length before every element: lines a converter appends are read too.
            var lines = new List<string> { "1", "2" };
            Func<string, object> conv = s =>
            {
                if (lines.Count < 4)
                    lines.Add((lines.Count + 1).ToString());
                return double.Parse(s);
            };
            CollectionAssert.AreEqual(new[] { 1.0, 2, 3, 4 }, np.loadtxt(lines, converters: conv).ToArray<double>());
        }

        [TestMethod]
        public void LoadTxt_RowPathAndFieldPath_Agree()
        {
            // StoreRow (no converter, no usecols) and the per-field Store (usecols selecting every column) give the same
            // bytes for every sink family.
            string[] lines = { "w1 1.5", "w22 -2", "w333 3e2" };
            foreach (var dt in new DType[] { np.dtype("S4"), np.dtype("U4"), np.dtype("U"), np.dtype("S"), np.dtypes.StringDType() })
            {
                var all = np.loadtxt(lines, dt);
                var cols = np.loadtxt(lines, dt, usecols: new[] { 0, 1 });
                CollectionAssert.AreEqual(Texts(all), Texts(cols), dt.ToString());
            }
            var nums = new[] { "1 2.5", "3 -4" };
            CollectionAssert.AreEqual(np.loadtxt(nums).ToArray<double>(), np.loadtxt(nums, usecols: new[] { 0, 1 }).ToArray<double>());
        }

        // ---------------------------------------------------------------------------------------------
        // Helpers
        // ---------------------------------------------------------------------------------------------

        private static void AssertUnmatched(Func<NDArray> read)
            => Assert.AreEqual("string or file could not be read to its end due to unmatched data",
                Assert.ThrowsException<ValueError>(() => read()).Message);

        private static string[] Texts(NDArray a)
        {
            var r = a.ravel();
            return Enumerable.Range(0, (int)r.size).Select(i => r.item(i) is byte[] b ? Encoding.Latin1.GetString(b) : (string)r.item(i)).ToArray();
        }

        private static void WithFile(string text, Action<string> body) => WithBytes(Encoding.UTF8.GetBytes(text), body);

        private static void WithBytes(byte[] bytes, Action<string> body)
        {
            string p = Path.Combine(Path.GetTempPath(), $"ns_trp_{Guid.NewGuid():N}.txt");
            File.WriteAllBytes(p, bytes);
            try { body(p); }
            finally { File.Delete(p); }
        }

        /// <summary>A stream that can only be read forward and reports no length (NumPy cannot read one; NumSharp
        ///     reads it to its end).</summary>
        private sealed class ForwardOnlyStream : Stream
        {
            private readonly Stream _inner;

            internal ForwardOnlyStream(Stream inner) => _inner = inner;

            public override bool CanRead => true;
            public override bool CanSeek => false;
            public override bool CanWrite => false;
            public override long Length => throw new NotSupportedException();
            public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
            public override void Flush() { }
            // Short reads on purpose: at most 1000 bytes a call.
            public override int Read(byte[] buffer, int offset, int count) => _inner.Read(buffer, offset, System.Math.Min(count, 1000));
            public override int Read(Span<byte> buffer) => _inner.Read(buffer.Slice(0, System.Math.Min(buffer.Length, 1000)));
            public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
            public override void SetLength(long value) => throw new NotSupportedException();
            public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        }
    }
}
