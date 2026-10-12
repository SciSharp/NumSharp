using System;
using System.Linq;
using System.Numerics;
using System.Text;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;
using NumSharp.Utilities;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     Conversion between text and numbers (strings phase 1, P1.1): NumPy's legacy casts write a number as its
    ///     scalar's <c>str()</c> and parse a text with CPython's <c>int()</c> / <c>float()</c> / <c>complex()</c>. Every
    ///     expected value and error text was probed against NumPy 2.4.2 (win-amd64; the C <c>long</c> overflow texts
    ///     against Linux NumPy 2.4.2, the LP64 texts NumSharp pins). The <c>strconv</c> section of the string oracle holds
    ///     the full matrix; these tests pin what the oracle cannot see — partial writes of a failed cast, the LP64 texts,
    ///     the C# boundary map (a C# string is a Python <c>str</c>), NumSharp's own <c>char</c> and <c>decimal</c> dtypes,
    ///     and the CPython character tables.
    /// </summary>
    [TestClass]
    public class TextNumberConversionTests
    {
        /// <summary>A 1-D <c>U</c> array of the given texts (NumPy's <c>np.array(list_of_str)</c>).</summary>
        /// <param name="s">The elements.</param>
        /// <returns>A new <c>&lt;Un</c> array as wide as the longest text.</returns>
        private static NDArray U(params string[] s) => np.array(s);

        /// <summary>A 1-D <c>S</c> array of the given texts' Latin-1 bytes (every char below 256 is one byte).</summary>
        /// <param name="s">The elements.</param>
        /// <returns>A new <c>|Sn</c> array as wide as the longest element.</returns>
        private static NDArray S(params string[] s) =>
            np.array(s.Select(x => (object)Encoding.Latin1.GetBytes(x)).ToArray(), dtype: "S");

        // ================================================================================== numbers to text

        /// <summary>A number becomes its NumPy scalar's <c>str()</c>: the float positional / scientific windows per type,
        /// float32 and float16 shortest digits, complex forms with signed zeros and specials, bool, and a sized request's
        /// cut.</summary>
        [TestMethod]
        public void NumbersToText_WriteTheScalarStr()
        {
            NDArray.AsStringArray(np.array(new[] { 1e16, 9999999999999998.0, 1e-5, 1e-4, 5e-324, -0.0, 123456789012345678.0 }).astype("U"))
                .Should().Equal("1e+16", "9999999999999998.0", "1e-05", "0.0001", "5e-324", "-0.0", "1.2345678901234568e+17");
            NDArray.AsStringArray(np.array(new[] { 1e16f, 1e-5f, 1.4e-45f, 16777217.0f, 0.1f }).astype("U"))
                .Should().Equal("1e+16", "1e-05", "1e-45", "1.6777216e+07", "0.1");
            NDArray.AsStringArray(np.array(new[] { (Half)65504, (Half)6e-8, (Half)0.1, (Half)2049, (Half)1e-4 }).astype("U"))
                .Should().Equal("6.55e+04", "6e-08", "0.1", "2.048e+03", "0.0001");
            var c = np.array(new[]
            {
                new Complex(0, -0.0), new Complex(-0.0, 0.0), new Complex(double.PositiveInfinity, double.NaN),
                new Complex(double.NaN, double.NegativeInfinity), new Complex(0, 1.5), new Complex(1, -0.0),
            });
            NDArray.AsStringArray(c.astype("U")).Should().Equal("-0j", "(-0+0j)", "(inf+nanj)", "(nan-infj)", "1.5j", "(1-0j)");
            np.array(new[] { true, false }).astype("S").ToString(true).Should().Be("array([b'True', b'False'], dtype='|S5')");
            np.array(new long[] { 12345, -7 }).astype("S3").ToString(true).Should().Be("array([b'123', b'-7'], dtype='|S3')");
            np.array(new[] { 1.5, 2.0 }).astype("S2").ToString(true).Should().Be("array([b'1.', b'2.'], dtype='|S2')");
        }

        /// <summary>An unsized <c>'U'</c> / <c>'S'</c> request takes NumPy's <c>REQUIRED_STR_LEN</c> width for every numeric
        /// dtype, NumSharp's <c>char</c> (uint16) and <c>decimal</c> (float64) included.</summary>
        [TestMethod]
        public void NumbersToText_UnsizedRequestTakesRequiredStrLen()
        {
            // NumPy's REQUIRED_STR_LEN: the digits of the type's widest value plus a sign; 32 for every float, 64 for complex.
            (NDArray a, string u)[] cases =
            {
                (np.array(new[] { true }), "<U5"), (np.array(new sbyte[] { 1 }), "<U4"), (np.array(new byte[] { 1 }), "<U3"),
                (np.array(new short[] { 1 }), "<U6"), (np.array(new ushort[] { 1 }), "<U5"), (np.array(new[] { 1 }), "<U11"),
                (np.array(new uint[] { 1 }), "<U10"), (np.array(new long[] { 1 }), "<U21"), (np.array(new ulong[] { 1 }), "<U20"),
                (np.array(new[] { (Half)1 }), "<U32"), (np.array(new[] { 1f }), "<U32"), (np.array(new[] { 1.0 }), "<U32"),
                (np.array(new[] { Complex.One }), "<U64"),
                // NumSharp's own dtypes: char is NumPy's uint16, decimal formats as a float64.
                (np.array(new[] { 'a' }), "<U5"), (np.array(new[] { 1.5m }), "<U32"),
            };
            foreach (var (a, u) in cases)
            {
                a.astype("U").dtype.str.Should().Be(u, a.dtype.ToString());
                a.astype("S").dtype.str.Should().Be("|S" + u.Substring(2), a.dtype.ToString());
            }
        }

        /// <summary>NumSharp's <c>char</c> converts as NumPy's uint16 (its code unit as a number, the uint16 range check)
        /// and <c>decimal</c> as a float64, both ways.</summary>
        [TestMethod]
        public void CharAndDecimal_ConvertAsTheirNumPyStandIns()
        {
            // char is NumPy's uint16 (its code unit as a number); decimal writes and parses as a float64.
            NDArray.AsStringArray(np.array(new[] { 'a', 'Z' }).astype("U")).Should().Equal("97", "90");
            np.array(new[] { "65", "7" }).astype(np.@char).ToArray<char>().Should().Equal('A', (char)7);
            Action charOverflow = () => np.array(new[] { "65536" }).astype(np.@char);
            charOverflow.Should().Throw<OverflowException>().WithMessage("Python integer 65536 out of bounds for uint16");
            NDArray.AsStringArray(np.array(new[] { 1.5m, -0.25m }).astype("U")).Should().Equal("1.5", "-0.25");
            np.array(new[] { "1.5", " 2 " }).astype(np.@decimal).ToArray<decimal>().Should().Equal(1.5m, 2m);
        }

        // ================================================================================== text to numbers

        /// <summary>Text parses by CPython's <c>int()</c> / <c>float()</c> / <c>complex()</c>: signs, whitespace,
        /// underscores, Unicode decimal digits, the float specials and overflow to inf, complex forms with parentheses;
        /// bytes alike; trailing NULs dropped.</summary>
        [TestMethod]
        public void TextToNumbers_ParseWithCPythonsGrammars()
        {
            np.array(new[] { "1", " 22 ", "-3", "1_000", "+5", "\t7\n", "١٢٣", "１２" }).astype(np.int64).ToArray<long>()
              .Should().Equal(1L, 22L, -3L, 1000L, 5L, 7L, 123L, 12L);
            var f = np.array(new[] { "1.5", "inf", "-Infinity", "nan", "1_000.5", ".5", "5.", "1e309", "٣.٥" }).astype(np.float64)
                      .ToArray<double>();
            f[0].Should().Be(1.5);
            double.IsPositiveInfinity(f[1]).Should().BeTrue();
            double.IsNegativeInfinity(f[2]).Should().BeTrue();
            double.IsNaN(f[3]).Should().BeTrue();
            f.Skip(4).Take(3).Should().Equal(1000.5, 0.5, 5.0);
            double.IsPositiveInfinity(f[7]).Should().BeTrue();
            f[8].Should().Be(3.5);
            np.array(new[] { "1+2j", "(3-4j)", "-j", "2", " (5+0j) " }).astype(np.complex128).ToArray<Complex>()
              .Should().Equal(new Complex(1, 2), new Complex(3, -4), new Complex(0, -1), new Complex(2, 0), new Complex(5, 0));
            // Bytes parse the same way; trailing NULs never count.
            S("1+2j", " (3-4j) ").astype(np.complex128).ToArray<Complex>().Should().Equal(new Complex(1, 2), new Complex(3, -4));
            U("12\0").astype(np.int64).ToArray<long>().Should().Equal(12L);
        }

        /// <summary>A text that does not parse raises NumPy's ValueError quoting the NumPy scalar (<c>np.str_</c> /
        /// <c>np.bytes_</c>), except <c>int()</c> of bytes (a plain bytes repr); complex() of bytes decodes UTF-8 first.</summary>
        [TestMethod]
        public void TextToNumbers_ErrorsQuoteTheNumPyScalar()
        {
            Action badInt = () => U("1", "x").astype(np.int64);
            badInt.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: np.str_('x')");
            Action floatText = () => U("1.5").astype(np.int64);
            floatText.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: np.str_('1.5')");
            // int() of bytes runs on the C string and quotes a plain bytes object; float() quotes the NumPy scalar.
            Action bytesInt = () => S("1\u00002").astype(np.int64);
            bytesInt.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: b'1\\x002'");
            Action bytesFloat = () => S("1.5\u00002").astype(np.float64);
            bytesFloat.Should().Throw<ValueError>().WithMessage("could not convert string to float: np.bytes_(b'1.5\\x002')");
            Action complexText = () => U("1 + 2j").astype(np.complex128);
            complexText.Should().Throw<ValueError>().WithMessage("complex() arg is a malformed string");
            // complex() of bytes decodes them as UTF-8 first.
            Action notUtf8 = () => S("\xff").astype(np.complex128);
            notUtf8.Should().Throw<UnicodeDecodeError>()
                   .WithMessage("'utf-8' codec can't decode byte 0xff in position 0: invalid start byte");
            Action surrogate = () => U("\ud800").astype(np.float64);
            surrogate.Should().Throw<ValueError>().WithMessage("could not convert string to float: np.str_('\\ud800')");
        }

        /// <summary>
        ///     <c>complex()</c> of an <c>S</c> element decodes the whole value as strict UTF-8 first (<c>CDOUBLE_setitem</c>),
        ///     with CPython 3.12's reason and byte range for the first bad sequence: a byte that cannot start one, a bad or
        ///     out-of-range continuation (overlong forms, surrogates, values past U+10FFFF) over the bytes before it, a cut
        ///     sequence to the end of the value — unless an early byte already proves it invalid. A valid value is then
        ///     rewritten (Unicode digits and spaces) and parsed; an underscore error quotes the DECODED text.
        /// </summary>
        [TestMethod]
        public void TextToComplex_BytesDecodeAsStrictUtf8_WithCPythonsRanges()
        {
            // One S element holding exactly these bytes, cast to complex128.
            static NDArray C(params byte[] b) =>
                np.array(new object[] { b }, dtype: "S" + System.Math.Max(b.Length, 1)).astype(np.complex128);
            const string Codec = "'utf-8' codec can't decode ";
            var errors = new (byte[] Bytes, string Text)[]
            {
                (new byte[] { 0xE2, 0x28, 0xA1 }, "byte 0xe2 in position 0: invalid continuation byte"),
                (new byte[] { 0xE2, 0x82, 0x28 }, "bytes in position 0-1: invalid continuation byte"),
                (new byte[] { 0xE2, 0x82 }, "bytes in position 0-1: unexpected end of data"),
                (new byte[] { 0xE0, 0x80, 0x80 }, "byte 0xe0 in position 0: invalid continuation byte"),
                (new byte[] { 0xED, 0xA0, 0x80 }, "byte 0xed in position 0: invalid continuation byte"),
                (new byte[] { 0xF0, 0x80, 0x80, 0x80 }, "byte 0xf0 in position 0: invalid continuation byte"),
                (new byte[] { 0xF4, 0x90, 0x80, 0x80 }, "byte 0xf4 in position 0: invalid continuation byte"),
                (new byte[] { 0xF0, 0x28, 0x8C, 0xBC }, "byte 0xf0 in position 0: invalid continuation byte"),
                (new byte[] { 0xF0, 0x9F, 0x98 }, "bytes in position 0-2: unexpected end of data"),
                (new byte[] { 0xF0, 0x90, 0x28, 0xBC }, "bytes in position 0-1: invalid continuation byte"),
                (new byte[] { 0xF0, 0x90, 0x8C, 0x28 }, "bytes in position 0-2: invalid continuation byte"),
                (new byte[] { 0xF5 }, "byte 0xf5 in position 0: invalid start byte"),
                (new byte[] { 0xC3 }, "byte 0xc3 in position 0: unexpected end of data"),
                (new byte[] { 0xC3, 0x28 }, "byte 0xc3 in position 0: invalid continuation byte"),
                (new byte[] { 0xF0, 0x9F }, "bytes in position 0-1: unexpected end of data"),
                (new byte[] { 0xC0, 0x80 }, "byte 0xc0 in position 0: invalid start byte"),
                (new byte[] { 0xC1 }, "byte 0xc1 in position 0: invalid start byte"),
                (new byte[] { 0x80 }, "byte 0x80 in position 0: invalid start byte"),
                (new byte[] { 0x31, 0xFF }, "byte 0xff in position 1: invalid start byte"),
                (new byte[] { 0x31, 0x32, 0xE2, 0x82 }, "bytes in position 2-3: unexpected end of data"),
                (new byte[] { 0xE0, 0x9F }, "byte 0xe0 in position 0: invalid continuation byte"),
                (new byte[] { 0xED, 0xA0 }, "byte 0xed in position 0: invalid continuation byte"),
                (new byte[] { 0xF0, 0x8F }, "byte 0xf0 in position 0: invalid continuation byte"),
                (new byte[] { 0xF4, 0x90 }, "byte 0xf4 in position 0: invalid continuation byte"),
                (new byte[] { 0xF0, 0x90, 0x28 }, "bytes in position 0-1: invalid continuation byte"),
            };
            foreach (var (bytes, text) in errors)
            {
                Action cast = () => C(bytes);
                cast.Should().Throw<UnicodeDecodeError>(BitConverter.ToString(bytes)).WithMessage(Codec + text);
            }
            // Valid UTF-8 that is not a number: the rewrite turns the first other non-ASCII character into '?' and stops.
            foreach (var bytes in new[] { new byte[] { 0xE2, 0x82, 0xAC }, new byte[] { 0xF0, 0x9F, 0x98, 0x80 }, Encoding.UTF8.GetBytes("é__1j") })
            {
                Action cast = () => C(bytes);
                cast.Should().Throw<ValueError>(BitConverter.ToString(bytes)).WithMessage("complex() arg is a malformed string");
            }
            using (var r = C(Encoding.UTF8.GetBytes("١+٢j")))
                r.GetValue(0).Should().Be(new Complex(1, 2));
            // U+2003 EM SPACE is a Unicode space: the rewrite makes it ' '.
            using (var r = C(Encoding.UTF8.GetBytes("\u2003 3j")))
                r.GetValue(0).Should().Be(new Complex(0, 3));
            using (var r = C(Encoding.UTF8.GetBytes("١_٠j")))
                r.GetValue(0).Should().Be(new Complex(0, 10));
            using (var r = C(Encoding.ASCII.GetBytes("(1+2j)")))
                r.GetValue(0).Should().Be(new Complex(1, 2));
            Action underscore = () => C(Encoding.UTF8.GetBytes("١__٠j"));
            underscore.Should().Throw<ValueError>().WithMessage("could not convert string to complex: '١__٠j'");
            Action asciiUnderscore = () => C(Encoding.ASCII.GetBytes("1__0j"));
            asciiUnderscore.Should().Throw<ValueError>().WithMessage("could not convert string to complex: '1__0j'");
        }

        /// <summary>
        ///     A <c>U</c> element of ONE code unit above U+10FFFF is CPython's <c>SystemError</c> for every target (NumPy builds
        ///     the one-character <c>np.str_</c> first and PyUnicode_New refuses it); a longer element only records its maximum
        ///     character, so it parses (and fails) as text, quoting the unit, and its truth value is True.
        /// </summary>
        [TestMethod]
        public void LoneUnitAboveU10FFFF_IsTheSystemError_ForEveryTarget()
        {
            using var lone = np.array(new uint[] { 0x110000 }).view(np.dtype("<U1"));
            foreach (var target in new[] { np.int64, np.float64, np.complex128, np.@bool })
            {
                Action cast = () => lone.astype(target);
                cast.Should().Throw<SystemError>(target.name).WithMessage("invalid maximum character passed to PyUnicode_New");
            }
            using var two = np.array(new uint[] { 0x31, 0x110000 }).view(np.dtype("<U2"));
            Action parse = () => two.astype(np.int64);
            parse.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: np.str_('1\\U00110000')");
            using (var truth = two.astype(np.@bool))
                truth.GetBoolean(0).Should().BeTrue();
            using var huge = np.array(new uint[] { 0x31, 0xFFFFFFFF }).view(np.dtype("<U2"));
            Action parseHuge = () => huge.astype(np.int64);
            parseHuge.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: np.str_('1\\Uffffffff')");
        }

        /// <summary>
        ///     The truth value of an <c>S</c> element is whether ANY of its bytes is nonzero (a leading NUL with a later byte
        ///     is True, an empty element False).
        /// </summary>
        [TestMethod]
        public void BytesToBool_IsAnyNonzeroByte()
        {
            using var s = np.array(new object[] { new byte[0], new byte[] { (byte)'a' }, new byte[] { 0, (byte)'b' } }, dtype: "S2");
            using var r = s.astype(np.@bool);
            r.ToArray<bool>().Should().Equal(false, true, true);
        }

        /// <summary>
        ///     CPython's <c>complex()</c> grammar takes a bare <c>j</c> after a sign as a unit imaginary part (<c>1+j</c>,
        ///     <c>-j</c>), and its float grammar takes leading zeros in the exponent (<c>1e-005</c>).
        /// </summary>
        [TestMethod]
        public void TextToNumbers_GrammarCorners()
        {
            using (var c = U("1+j", "1-j", "+j", "-j", "j").astype(np.complex128))
                Enumerable.Range(0, 5).Select(i => (Complex)c.GetValue(i)).Should().Equal(
                    new Complex(1, 1), new Complex(1, -1), new Complex(0, 1), new Complex(0, -1), new Complex(0, 1));
            using (var f = U("1e05", "1e-005", "1E+00", "2e0").astype(np.float64))
                f.ToArray<double>().Should().Equal(1e5, 1e-5, 1.0, 2.0);
        }

        /// <summary>
        ///     The integer texts at the C <c>long</c> boundary (LP64 texts, probed against Linux NumPy 2.4.2): 2**63 and
        ///     -2**63-1 do not fit a C long; -2**63 does; a negative value into an unsigned target is out of bounds while it
        ///     fits a C long and "too large" past it; <c>-0</c> is 0 for every unsigned target.
        /// </summary>
        [TestMethod]
        public void TextToNumbers_CLongBoundary_IsLP64()
        {
            const string CLong = "Python int too large to convert to C long";
            Action i64Plus = () => U("9223372036854775808").astype(np.int64);
            i64Plus.Should().Throw<OverflowException>().WithMessage(CLong);
            Action i64Minus = () => U("-9223372036854775809").astype(np.int64);
            i64Minus.Should().Throw<OverflowException>().WithMessage(CLong);
            using (var min = U("-9223372036854775808").astype(np.int64))
                min.GetInt64(0).Should().Be(long.MinValue);
            Action u64Minus = () => U("-9223372036854775809").astype(np.uint64);
            u64Minus.Should().Throw<OverflowException>().WithMessage(CLong);
            Action u64Min = () => U("-9223372036854775808").astype(np.uint64);
            u64Min.Should().Throw<OverflowException>().WithMessage("Python integer -9223372036854775808 out of bounds for uint64");
            using (var z = U("-0").astype(np.uint64))
                z.GetUInt64(0).Should().Be(0UL);
            using (var z = U("-0").astype(np.uint32))
                z.GetUInt32(0).Should().Be(0U);
            Action u32Minus = () => U("-1").astype(np.uint32);
            u32Minus.Should().Throw<OverflowException>().WithMessage("Python integer -1 out of bounds for uint32");
            Action i8 = () => U("9223372036854775808").astype(np.int8);
            i8.Should().Throw<OverflowException>().WithMessage(CLong);
        }

        /// <summary>
        ///     A C# string holding astral characters (UTF-16 surrogate pairs) is one Python <c>str</c> of code points: Unicode
        ///     decimal digits beyond the BMP parse (MATHEMATICAL BOLD DIGIT ONE, TWO is 12), and an error quotes the character
        ///     whole.
        /// </summary>
        [TestMethod]
        public void CSharpString_WithSurrogatePairs_IsOneStrOfCodePoints()
        {
            using var a = np.zeros(new Shape(2), np.int64);
            a[0] = "\U0001D7CF\U0001D7D0";
            a.ToArray<long>().Should().Equal(12L, 0L);
            Action emoji = () => a[0] = "1\U0001F600";
            emoji.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: '1\U0001F600'");
            using var b = np.zeros(new Shape(1), np.float64);
            Action floatEmoji = () => b[0] = "\U0001F600";
            floatEmoji.Should().Throw<ValueError>().WithMessage("could not convert string to float: '\U0001F600'");
        }

        /// <summary>
        ///     A text-to-number copy whose source overlaps the destination (both views of one buffer, through
        ///     <c>view(dtype)</c>) reads the source as it was before the copy: NumPy copies an overlapping source first. The
        ///     masked form does too, and so does a number-to-text copy.
        /// </summary>
        [TestMethod]
        public void Copyto_OverlappingTextAndNumbers_ReadsTheSourceFirst()
        {
            using (var x = np.array(new[] { 0x31, 0x32, 0x33, 0x34 }))
            using (var u = x.view(np.dtype("<U1")))
            {
                np.copyto(x["1:"], u[":-1"], "unsafe");
                x.ToArray<int>().Should().Equal(0x31, 1, 2, 3);
            }
            using (var x = np.array(new[] { 0x31, 0x32, 0x33, 0x34 }))
            using (var u = x.view(np.dtype("<U1")))
            {
                np.copyto(x["1:"], u[":-1"], "unsafe", np.array(new[] { true, false, true }));
                x.ToArray<int>().Should().Equal(0x31, 1, 0x33, 3);
            }
            using (var x = np.array(new[] { 1, 2, 3, 4 }))
            using (var u = x.view(np.dtype("<U1")))
            {
                np.copyto(u["1:"], x[":-1"]);
                x.ToArray<int>().Should().Equal(1, 0x31, 0x32, 0x33);
            }
        }

        /// <summary>
        ///     A masked text-to-number copy into a strided destination of two and three axes converts each run of selected
        ///     elements and walks the outer axes with the mask (NumPy's <c>raw_array_wheremasked_assign_array</c>).
        /// </summary>
        [TestMethod]
        public void Copyto_MaskedTextToNumbers_WalksTheOuterAxes()
        {
            using (var d = np.zeros(new Shape(3, 4), np.int64))
            {
                var view = d[":, ::2"];
                np.copyto(view, np.array(new[,] { { "1", "2" }, { "3", "4" }, { "5", "6" } }), "unsafe",
                    np.array(new[,] { { true, false }, { false, true }, { true, true } }));
                view.ToArray<long>().Should().Equal(1L, 0L, 0L, 4L, 5L, 6L);
            }
            using (var d = np.zeros(new Shape(2, 3, 4), np.int64))
            {
                var view = d[":, ::2, ::2"];
                np.copyto(view, np.array(new[, ,] { { { "1", "2" }, { "3", "4" } }, { { "5", "6" }, { "7", "8" } } }), "unsafe",
                    np.array(new[, ,] { { { true, false }, { false, true } }, { { true, true }, { false, false } } }));
                view.ToArray<long>().Should().Equal(1L, 0L, 0L, 4L, 5L, 6L, 0L, 0L);
            }
        }

        /// <summary>An integer text outside the target raises the LP64 (Linux / macOS) OverflowError texts NumSharp pins
        /// for every width.</summary>
        [TestMethod]
        public void TextToNumbers_OverflowTextsAreLP64()
        {
            // NumSharp pins the Linux / macOS (LP64) C long: the Windows texts differ ("int too big to convert", and int32
            // reports the C long first) — probed against both.
            Action i8 = () => U("99999999999999999999").astype(np.int64);
            i8.Should().Throw<OverflowException>().WithMessage("Python int too large to convert to C long");
            Action u8 = () => U("18446744073709551616").astype(np.uint64);
            u8.Should().Throw<OverflowException>().WithMessage("Python int too large to convert to C long");
            Action u8Negative = () => U("-1").astype(np.uint64);
            u8Negative.Should().Throw<OverflowException>().WithMessage("Python integer -1 out of bounds for uint64");
            Action i4 = () => U("3000000000").astype(np.int32);
            i4.Should().Throw<OverflowException>().WithMessage("Python integer 3000000000 out of bounds for int32");
            Action u4 = () => U("5000000000").astype(np.uint32);
            u4.Should().Throw<OverflowException>().WithMessage("Python integer 5000000000 out of bounds for uint32");
            Action i1 = () => U("300").astype(np.int8);
            i1.Should().Throw<OverflowException>().WithMessage("Python integer 300 out of bounds for int8");
        }

        /// <summary>CPython's 4300-digit limit applies to <c>int()</c> (its ValueError text), not to <c>float()</c>.</summary>
        [TestMethod]
        public void TextToNumbers_IntegerDigitLimit()
        {
            Action limit = () => U(new string('1', 4301)).astype(np.int64);
            limit.Should().Throw<ValueError>().WithMessage(
                "Exceeds the limit (4300 digits) for integer string conversion: value has 4301 digits; use " +
                "sys.set_int_max_str_digits() to increase the limit");
            // float() has no limit.
            U(new string('1', 4301)).astype(np.float64).GetDouble().Should().Be(double.PositiveInfinity);
        }

        // ================================================================================== stores

        /// <summary>A C# string stored into a numeric array is a Python <c>str</c> (its error quotes the str); np.full's
        /// copy of a str scalar is the array cast (its error quotes the NumPy scalar); text into bool is its truth.</summary>
        [TestMethod]
        public void CSharpString_StoredIntoNumbers_IsAPythonStr()
        {
            // A Python str reaches the target's setitem: its error quotes the str itself, not a NumPy scalar.
            var a = np.zeros(new Shape(3), np.int32);
            a[0] = "12";
            a.ToArray<int>().Should().Equal(12, 0, 0);
            Action hello = () => a[0] = "hello";
            hello.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: 'hello'");
            var b = np.zeros(new Shape(2), np.int64);
            Action big = () => b[0] = "99999999999999999999";
            big.Should().Throw<OverflowException>().WithMessage("Python int too large to convert to C long");
            // np.full / np.full_like copy a str scalar through np.asarray: the array cast, quoting the NumPy scalar.
            Action full = () => np.full(new Shape(2), "x", np.int64);
            full.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: np.str_('x')");
            np.full(new Shape(2), "12", np.int64).ToArray<long>().Should().Equal(12L, 12L);
            np.full(new Shape(2), "", np.@bool).ToArray<bool>().Should().Equal(false, false);
        }

        /// <summary>Numbers stored into a text array (setitem, fill, full, ones, pad's constant) are their str(), cut to
        /// the element width.</summary>
        [TestMethod]
        public void NumbersStoredIntoText_AreCutToTheWidth()
        {
            var u = U("ab", "cd");
            u[0] = 5;
            u[1] = 2.5;
            NDArray.AsStringArray(u).Should().Equal("5", "2.");
            u.fill(true);
            NDArray.AsStringArray(u).Should().Equal("Tr", "Tr");
            NDArray.AsStringArray(np.full(new Shape(2), 7, "U")).Should().Equal("7", "7");
            np.ones(new Shape(2), "S").ToString(true).Should().Be("array([b'1', b'1'], dtype='|S1')");
            NDArray.AsStringArray(np.pad(U("ab", "c"), 1, constant_values: 7)).Should().Equal("7", "ab", "c", "7");
        }

        /// <summary>A failing text-to-number copy leaves NumPy's partial result: whole 128-element blocks before the
        /// failing one are written, in NumPy's raw assignment order (a reversed destination walks memory order).</summary>
        [TestMethod]
        public void Copyto_TextToNumbers_CommitsWholeBlocksBeforeTheFailingOne()
        {
            // NumPy's wrapped legacy cast converts 128 elements a block and writes a block only when all of it parsed:
            // 'x' at 150 leaves elements 0..127 written and 128.. untouched (probed).
            var src = np.array(Enumerable.Range(0, 200).Select(i => i == 150 ? "x" : i.ToString()).ToArray());
            var dst = np.full(new Shape(200), -1L);
            Action act = () => np.copyto(dst, src, "unsafe");
            act.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: np.str_('x')");
            dst.GetInt64(127).Should().Be(127L);
            dst.GetInt64(128).Should().Be(-1L);
            dst.GetInt64(149).Should().Be(-1L);
            // A reversed destination is walked in memory order: the bad element falls in the FIRST block, nothing is written.
            var dst2 = np.full(new Shape(200), -1L);
            Action reversed = () => np.copyto(dst2["::-1"], src, "unsafe");
            reversed.Should().Throw<ValueError>();
            dst2.GetInt64(0).Should().Be(-1L);
            dst2.GetInt64(49).Should().Be(-1L);
        }

        /// <summary>A masked copy converts only the selected elements: an unparsable masked-off text is never read.</summary>
        [TestMethod]
        public void Copyto_UnderAMask_ParsesOnlySelectedText()
        {
            var d = np.zeros(new Shape(3), np.int64);
            np.copyto(d, U("1", "x", "3"), "unsafe", np.array(new[] { true, false, true }));
            d.ToArray<long>().Should().Equal(1L, 0L, 3L);
        }

        // ================================================================================== inside other calls

        /// <summary>A numeric ufunc (and clip, matmul, modf) into a text <c>out=</c> writes each loop result as text, cut
        /// to the width; a comparison of text into a text out writes 'True' / 'False'.</summary>
        [TestMethod]
        public void UfuncTextOut_FormatsTheLoopResult()
        {
            NDArray f = np.array(new[] { 1.0, 2.5 }), g = np.array(new[] { 1.0, 3.0 });
            NDArray.AsStringArray(np.add(f, g, np.zeros(new Shape(2), "U5"))).Should().Equal("2.0", "5.5");
            NDArray.AsStringArray(np.sqrt(f, np.zeros(new Shape(2), "U5"))).Should().Equal("1.0", "1.581");
            NDArray.AsStringArray(np.equal(f, g, np.zeros(new Shape(2), "U1"))).Should().Equal("T", "F");
            NDArray.AsStringArray(np.equal(U("a", "b"), U("a", "c"), np.zeros(new Shape(2), "U5"))).Should().Equal("True", "False");
            NDArray.AsStringArray(np.clip(f, 0, 2, np.zeros(new Shape(2), "U32"))).Should().Equal("1.0", "2.0");
            NDArray.AsStringArray(np.matmul(np.array(new long[,] { { 1, 2 } }), np.array(new long[,] { { 3 }, { 4 } }),
                np.zeros(new Shape(1, 1), "U3"))).Should().Equal("11");
            var (frac, whole) = np.modf(f, np.zeros(new Shape(2), "U5"), np.zeros(new Shape(2), "U5"));
            NDArray.AsStringArray(frac).Should().Equal("0.0", "0.5");
            NDArray.AsStringArray(whole).Should().Equal("1.0", "2.0");
        }

        /// <summary>Under <c>where=</c> a text out is READWRITE: its text is parsed into the loop dtype first (masked-off
        /// elements included; a bool loop reads the truth), selected slots are written, masked-off ones keep their text,
        /// and the parse precedes the loop's own errors.</summary>
        [TestMethod]
        public void UfuncTextOut_UnderWhere_ReadsTheOutsTextFirst()
        {
            NDArray f = np.array(new[] { 1.0, 2.5 }), g = np.array(new[] { 1.0, 3.0 });
            var mask = np.array(new[] { true, false });
            // Selected slots are written, a masked-off '8' keeps its text.
            NDArray.AsStringArray(np.add(f, g, U("7", "8").astype("U32"), mask)).Should().Equal("2.0", "8");
            // The out's text is parsed into the loop dtype first — masked-off 'x' included.
            Action bad = () => np.sqrt(f, U("4", "x").astype("U32"), mask);
            bad.Should().Throw<ValueError>().WithMessage("could not convert string to float: np.str_('x')");
            // A bool loop reads the text's truth, which never fails.
            NDArray.AsStringArray(np.equal(f, g, U("zz", "yy").astype("U5"), mask)).Should().Equal("True", "yy");
            // The parse precedes power's in-loop negative-exponent error.
            Action order = () => np.power(np.array(new long[] { 2, -1 }), np.array(new long[] { 2, -1 }), U("7", "x"), mask);
            order.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: np.str_('x')");
        }

        /// <summary>The integer power loop raises for a negative exponent only where it runs: a masked-off negative
        /// exponent is never met.</summary>
        [TestMethod]
        public void Power_NegativeExponentRaisesOnlyWhereTheLoopRuns()
        {
            // NumPy raises inside the integer loop, so a masked-off negative exponent is never met (probed: [4 0]).
            var o = np.zeros(new Shape(2), np.int64);
            np.power(np.array(new long[] { 2, -1 }), np.array(new long[] { 2, -1 }), o, np.array(new[] { true, false }));
            o.ToArray<long>().Should().Equal(4L, 0L);
            Action selected = () => np.power(np.array(new long[] { 2, -1 }), np.array(new long[] { 2, -1 }), np.zeros(new Shape(2), np.int64),
                np.array(new[] { true, true }));
            selected.Should().Throw<ArgumentException>().WithMessage("Integers to negative integer powers are not allowed.");
        }

        /// <summary><c>np.add</c> under a string <c>dtype=</c> request writes numbers as text (REQUIRED_STR_LEN widths, a C#
        /// literal as its Python scalar) and concatenates; U text never reaches S; a sized request is refused.</summary>
        [TestMethod]
        public void AddUnderAStringRequest_FormatsNumbersThenConcatenates()
        {
            var r = np.add(np.array(new[] { 1.5, 2.0 }), np.array(new long[] { 1, 2 }), dtype: "U");
            r.dtype.str.Should().Be("<U53");
            NDArray.AsStringArray(r).Should().Equal("1.51", "2.02");
            // A C# literal is the Python scalar it stands for: an int reads as int64 (21 characters), a float as float64.
            np.add(U("a", "b"), 1, dtype: "U").dtype.str.Should().Be("<U22");
            NDArray.AsStringArray(np.add(U("a"), 0.1f, dtype: "U")).Should().Equal("a0.10000000149011612");
            // S text reaches a U request (ASCII decode); U text never reaches S.
            np.add(S("ab"), np.array(new[] { 1.5 }), dtype: "U").ToString(true).Should().Be("array(['ab1.5'], dtype='<U34')");
            Action narrow = () => np.add(U("a"), np.array(new[] { 1.5 }), dtype: "S");
            narrow.Should().Throw<ArgumentException>()
                  .WithMessage("Cannot cast ufunc 'add' input 0 from dtype('<U1') to dtype('S1') with casting rule 'same_kind'");
            Action sized = () => np.add(U("a"), np.array(new[] { 1.5 }), dtype: "U5");
            sized.Should().Throw<TypeError>().WithMessage("The `dtype` and `signature` arguments to ufuncs only select*");
        }

        /// <summary>A multiply / maximum reduction with a numeric <c>dtype=</c> parses the text first; nancumprod keeps a
        /// parsed 'nan'; np.cov with a numeric dtype parses too.</summary>
        [TestMethod]
        public void NumericDtypeReductions_ParseTheTextFirst()
        {
            np.prod(U("1.5", "2", "4"), (int?)null, np.float64).GetDouble().Should().Be(12.0);
            np.amax(U("3", "-1", "2"), (int?)null, dtype: np.int64).GetInt64().Should().Be(3L);
            np.cumprod(U("2", "3"), (int?)null, np.int64).ToArray<long>().Should().Equal(2L, 6L);
            // nancumprod's _replace_nan leaves text alone: a parsed 'nan' is multiplied, not replaced by 1.
            var nan = np.nancumprod(U("inf", "nan"), (int?)null, np.float64).ToArray<double>();
            double.IsPositiveInfinity(nan[0]).Should().BeTrue();
            double.IsNaN(nan[1]).Should().BeTrue();
            Action bad = () => np.prod(U("1", "x"), (int?)null, np.float64);
            bad.Should().Throw<ValueError>().WithMessage("could not convert string to float: np.str_('x')");
            // np.cov with a numeric dtype parses too.
            np.cov(U("1", "2", "4"), dtype: np.float64).GetDouble().Should().Be(2.333333333333333);   // NumPy's bits (probed)
        }

        /// <summary>numpy.ma's astype converts the stored fill (at allocation) before the data cast; a masked fill value
        /// prints with its mask in the conversion error.</summary>
        [TestMethod]
        public void MaskedAstype_ConvertsTheFillBeforeTheData()
        {
            // __array_finalize__ runs at allocation, before the data cast: the fill's error wins.
            var m = np.ma.array(U("ab", "1", "2"), mask: new[] { false, true, false }, fill_value: "zz");
            Action fill = () => m.astype(np.float64);
            fill.Should().Throw<TypeError>().WithMessage("Cannot convert fill_value zz to dtype float64");
            var m2 = np.ma.array(U("ab", "1", "2"), mask: new[] { false, true, false }, fill_value: "3");
            Action data = () => m2.astype(np.float64);
            data.Should().Throw<ValueError>().WithMessage("could not convert string to float: np.str_('ab')");
            // A masked fill value prints with its mask in the error.
            var x = np.ma.array(np.array(new[] { 1.0, 2.0, 3.0 }), mask: new[] { false, false, false });
            var badFill = np.ma.array(U("ab", "c", "de"), mask: new[] { true, false, false });
            Action masked = () => np.ma.masked_values(x, badFill);
            masked.Should().Throw<TypeError>().WithMessage("Cannot convert fill_value [-- 'c' 'de'] to dtype float64");
        }

        /// <summary><c>matrix_power(text, 0)</c> is the identity written in the operand's text dtype: the Python int 1 as
        /// '1', the zeros empty.</summary>
        [TestMethod]
        public void MatrixPowerZero_OfText_IsTheIdentityWrittenAsText()
        {
            var r = np.linalg.matrix_power(np.array(new[,] { { "a", "b" }, { "c", "d" } }), 0);
            r.ToString(true).Should().Be("array([['1', ''],\n       ['', '1']], dtype='<U1')");
        }

        // ================================================================================== creation and element stores

        /// <summary>
        ///     <c>np.array</c> of text mixed with numbers: each number widens the text dtype by its discovered descriptor's
        ///     <c>REQUIRED_STR_LEN</c> (int64 21, uint64 20, a Python int past int64 but within uint64 20, float16 / float64
        ///     32, complex 64); a Python int past 64 bits (NumPy's object array) and a value of no NumPy type are refused.
        /// </summary>
        [TestMethod]
        public void TextWithNumbers_DiscoversTheNumbersWidths()
        {
            static string Width(object v) => np.array(new object[] { "x", v }).dtype.str;
            Width(ulong.MaxValue).Should().Be("<U20");
            Width(long.MaxValue).Should().Be("<U21");
            Width(new BigInteger(long.MaxValue)).Should().Be("<U21");
            Width(BigInteger.Pow(2, 63)).Should().Be("<U20");
            Width((Half)1.5).Should().Be("<U32");
            Width(1.5m).Should().Be("<U32");
            Width(new Complex(1, 2)).Should().Be("<U64");
            Action objectArray = () => np.array(new object[] { "x", BigInteger.Pow(2, 70) });
            objectArray.Should().Throw<NotSupportedException>().WithMessage("*object array*");
            Action noDtype = () => np.array(new object[] { "x", DateTime.Now });
            noDtype.Should().Throw<NotSupportedException>().WithMessage("A DateTime value has no NumPy dtype*");
        }

        /// <summary>
        ///     <c>np.array(values, numeric dtype)</c> over text and <c>None</c>: None is NaN for a float dtype (NumPy's
        ///     POSITIVE quiet NaN), NaN+NaNj for complex, False for bool, and int()'s TypeError for an integer dtype; an F
        ///     order request lays the result out column-major.
        /// </summary>
        [TestMethod]
        public void TextAndNoneIntoNumbers_FollowEachDtypesSetitem()
        {
            using (var f = np.array(new object[] { "1", null }, np.float64))
            {
                f.GetDouble(0).Should().Be(1.0);
                BitConverter.DoubleToUInt64Bits(f.GetDouble(1)).Should().Be(0x7FF8_0000_0000_0000UL);
            }
            using (var c = np.array(new object[] { "1", null }, np.complex128))
            {
                ((Complex)c.GetValue(0)).Should().Be(new Complex(1, 0));
                var z = (Complex)c.GetValue(1);
                double.IsNaN(z.Real).Should().BeTrue();
                double.IsNaN(z.Imaginary).Should().BeTrue();
            }
            using (var b = np.array(new object[] { "1", null }, np.@bool))
                b.ToArray<bool>().Should().Equal(true, false);
            Action integer = () => np.array(new object[] { "1", null }, np.int64);
            integer.Should().Throw<TypeError>()
                   .WithMessage("int() argument must be a string, a bytes-like object or a real number, not 'NoneType'");
            using var fortran = np.array(new object[,] { { "1", "2" }, { "3", "4" } }, np.int64, order: 'F');
            fortran.Shape.IsFContiguous.Should().BeTrue();
            fortran.ToArray<long>().Should().Equal(1L, 2L, 3L, 4L);
        }

        /// <summary>
        ///     A typed numeric C# array under a string dtype is an ndarray cast to text: <c>np.array(int32[], 'U', ndmin=2)</c>
        ///     is <c>&lt;U11</c> (the unsized class adapted to int32), raised to two axes.
        /// </summary>
        [TestMethod]
        public void TypedNumbersUnderAStringDtype_AreCastWithNdmin()
        {
            using var a = np.array(new[] { 1, 22, 333 }, np.dtype("U"), ndmin: 2);
            a.dtype.str.Should().Be("<U11");
            a.shape.Should().Equal(1L, 3L);
            NDArray.AsStringArray(a).Should().Equal("1", "22", "333");
        }

        /// <summary>
        ///     One element stored into a fixed-width text array through the object route (<c>SetValue(object, …)</c>): a 0-d
        ///     array as its own dtype's scalar <c>str()</c> (float32 shortest digits), a Python int of any size, a C# float as
        ///     the Python float of its widened value, <c>np.float16</c> / <c>np.float64</c> (Half, decimal) by their scalar's
        ///     digits, a complex; a C# object with no Python <c>str()</c> is refused. An unsized request discovers a big
        ///     int's width.
        /// </summary>
        [TestMethod]
        public void ElementStoresIntoText_AreTheValuesStr()
        {
            using var u = np.array(new[] { new string('x', 26) });
            string Store(object v)
            {
                u.SetValue(v, 0L);
                return (string)u.GetValue(0);
            }
            using (var f32 = NDArray.Scalar(0.1f))
                Store(f32).Should().Be("0.1");
            Store(BigInteger.Pow(2, 70)).Should().Be("1180591620717411303424");
            Store((object)ulong.MaxValue).Should().Be("18446744073709551615");
            Store((object)0.1f).Should().Be("0.10000000149011612");
            Store((object)(Half)0.1).Should().Be("0.1");
            Store((object)0.1m).Should().Be("0.1");
            Store((object)new Complex(1, 2)).Should().Be("(1+2j)");
            Action unknown = () => u.SetValue((object)DateTime.Now, 0L);
            unknown.Should().Throw<NotSupportedException>().WithMessage("Cannot store a DateTime value in a fixed-width string array*");
            using var big = np.array(new object[] { BigInteger.Pow(2, 70) }, np.dtype("U"));
            big.dtype.str.Should().Be("<U22");
            using var u2 = np.array(new[,] { { "a", "b" }, { "c", "d" } });
            u2.GetValue(new[] { 1, 0 }).Should().Be("c");
        }

        /// <summary>
        ///     <c>np.copyto</c> with a scalar: a 0-d text array into numbers is the array cast (parsed); a Python int into text
        ///     is 'same_kind' (its <c>str()</c>, cut to the width), refused under 'safe' with NumPy's scalar text, and stored
        ///     only where a mask selects.
        /// </summary>
        [TestMethod]
        public void Copyto_ScalarsBetweenTextAndNumbers()
        {
            using (var d = np.zeros(new Shape(2), np.int64))
            using (var seven = np.array("7"))
            {
                np.copyto(d, (NDArray)seven, "unsafe");
                d.ToArray<long>().Should().Equal(7L, 7L);
            }
            using (var t = np.array(new[] { "ab", "cd" }))
            {
                np.copyto(t, 5);
                NDArray.AsStringArray(t).Should().Equal("5", "5");
                Action safe = () => np.copyto(t, 5, "safe");
                safe.Should().Throw<InvalidCastException>()
                    .WithMessage("Cannot cast scalar from dtype('int64') to dtype('<U2') according to the rule 'safe'");
            }
            using (var t = np.array(new[] { "ab", "cd" }))
            {
                np.copyto(t, 5, "unsafe", np.array(new[] { true, false }));
                NDArray.AsStringArray(t).Should().Equal("5", "cd");
                np.copyto(t, 6, "unsafe", np.array(new[] { false, false }));
                NDArray.AsStringArray(t).Should().Equal("5", "cd");
            }
        }

        /// <summary>
        ///     A fancy assignment of text into a numeric array: a 0-d value is parsed before the index is read, an array value
        ///     after the index and broadcast checks, and not at all for an empty selection (<c>d[[]] = ['x']</c> succeeds).
        /// </summary>
        [TestMethod]
        public void FancySet_TextIntoNumbers()
        {
            using var d = np.zeros(new Shape(3), np.int64);
            d[np.array(new[] { 0, 2 })] = "5";
            d.ToArray<long>().Should().Equal(5L, 0L, 5L);
            d[np.array(new[] { 0, 2 })] = np.array(new[] { "1", "2" });
            d.ToArray<long>().Should().Equal(1L, 0L, 2L);
            d[np.array(new long[0])] = np.array(new[] { "x" });
            d.ToArray<long>().Should().Equal(1L, 0L, 2L);
            Action bad = () => d[np.array(new[] { 1 })] = np.array(new[] { "x" });
            bad.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: np.str_('x')");
        }

        /// <summary>
        ///     The C# conversions of a 0-d text array (NumPy's <c>bool()</c> / <c>float()</c> / <c>int()</c> of the scalar):
        ///     truth is non-emptiness, bytes parse as float, an integer text above <c>long.MaxValue</c> is a ulong, and one past
        ///     64 bits is refused.
        /// </summary>
        [TestMethod]
        public void ScalarConversions_OfZeroDText()
        {
            ((bool)np.array("")).Should().BeFalse();
            ((bool)np.array("x")).Should().BeTrue();
            ((bool)NDArray.Scalar(new byte[] { (byte)'x' })).Should().BeTrue();
            ((double)NDArray.Scalar(Encoding.ASCII.GetBytes("1.5"))).Should().Be(1.5);
            ((ulong)np.array("18446744073709551615")).Should().Be(ulong.MaxValue);
            Action tooBig = () => _ = (ulong)np.array("18446744073709551616");
            tooBig.Should().Throw<OverflowException>().WithMessage("Python int too large to convert to C long");
        }

        /// <summary>
        ///     A text <c>out=</c> receives the formatted results of every output: <c>np.divmod</c> and <c>np.modf</c> (two
        ///     outputs), and <c>np.clip</c> without bounds (positive's typed copy); with <c>where=</c> the out's current text is
        ///     parsed into the loop dtype first (NumPy's READWRITE masked out), so text that does not parse raises.
        /// </summary>
        [TestMethod]
        public void TextOut_MultiOutputAndClip()
        {
            var (q, r) = np.divmod(np.array(new[] { 7 }), np.array(new[] { 2 }), (np.empty(new Shape(1), "U5"), np.empty(new Shape(1), "U5")));
            NDArray.AsStringArray(q).Should().Equal("3");
            NDArray.AsStringArray(r).Should().Equal("1");
            var (frac, integral) = np.modf(np.array(new[] { 2.5 }), np.empty(new Shape(1), "U5"), np.empty(new Shape(1), "U5"));
            NDArray.AsStringArray(frac).Should().Equal("0.5");
            NDArray.AsStringArray(integral).Should().Equal("2.0");
            using var clipped = np.clip(np.array(new[] { 1, 2 }), null, null, @out: np.empty(new Shape(2), "U3"));
            NDArray.AsStringArray(clipped).Should().Equal("1", "2");
            Action masked = () => np.clip(np.array(new[] { 1, 2 }), null, null, @out: np.array(new[] { "a", "b" }, "U3"),
                where: np.array(new[] { true, false }));
            masked.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: np.str_('a')");
        }

        /// <summary>
        ///     A text array read out as C# numbers (<c>ToArray&lt;T&gt;</c> / <c>ToMuliDimArray&lt;T&gt;</c>) is parsed into T's
        ///     dtype first, as <c>astype</c> parses it.
        /// </summary>
        [TestMethod]
        public void TextReadOutAsNumbers_IsParsed()
        {
            np.array(new[] { "1", "2" }).ToArray<int>().Should().Equal(1, 2);
            var m = (int[,])np.array(new[,] { { "1", "2" }, { "3", "4" } }).ToMuliDimArray<int>();
            m[1, 0].Should().Be(3);
        }

        /// <summary>
        ///     A MASKED text-to-number copy whose source is a reversed view of the destination's buffer converts the whole
        ///     source first (NumPy's overlap rule for opposite directions), then copies under the mask.
        /// </summary>
        [TestMethod]
        public void Copyto_MaskedOverlap_ReversedSource_ConvertsFirst()
        {
            using var x = np.array(new[] { 0x31, 0x32, 0x33, 0x34 });
            using var u = x.view(np.dtype("<U1"));
            np.copyto(x["1:"], u[":-1"]["::-1"], "unsafe", np.array(new[] { true, false, true }));
            x.ToArray<int>().Should().Equal(0x31, 3, 0x33, 1);
        }

        /// <summary>
        ///     Numeric functions over text: <c>np.cov</c> parses a <c>dtype=</c> request (rowvar=False transposes the parsed
        ///     dims), <c>np.i0</c> parses into float64, <c>np.cumprod</c> with a numeric <c>dtype=</c> parses first; a bad
        ///     <c>fill_value</c> bytes of a masked array is NumPy's TypeError; an integer power whose mask does not broadcast
        ///     raises the broadcast error, not the negative-exponent one; <c>np.add</c> of text with an empty broadcast is the
        ///     empty concatenation, and with a Python complex NumPy's no-loop TypeError.
        /// </summary>
        [TestMethod]
        public void NumericFunctionsOverText()
        {
            using (var cov = np.cov(np.array(new[,] { { "1", "2" }, { "3", "5" } }), rowvar: false, dtype: np.float64))
                cov.ToArray<double>().Should().Equal(2.0, 3.0, 3.0, 4.5);
            using (var i0 = np.i0(np.array(new[] { "0", "1" })))
            {
                i0.GetDouble(0).Should().Be(1.0);
                i0.GetDouble(1).Should().BeApproximately(1.2660658777520082, 1e-15);
            }
            using (var cp = np.cumprod(np.array(new[] { "2", "3" }), typeCode: np.int64))
                cp.ToArray<long>().Should().Equal(2L, 6L);
            Action fill = () => np.ma.array(np.array(new[] { 1L, 2L }), fill_value: new byte[] { (byte)'x' });
            fill.Should().Throw<TypeError>().WithMessage("Cannot convert fill_value b'x' to dtype int64");
            Action power = () => np.power(np.array(new[] { 2, 3 }), np.array(new[] { -1, 1 }), where: np.array(new[] { true, false, true }));
            power.Should().Throw<ArgumentException>().WithMessage("operands could not be broadcast together with shapes (2,) (2,) (3,) ");
            using (var empty = np.add(np.array(new string[0], "U1"), "x"))
            {
                empty.dtype.str.Should().Be("<U2");
                empty.size.Should().Be(0);
            }
            Action complex = () => np.add(np.array(new[] { "a" }), new Complex(0, 1));
            complex.Should().Throw<TypeError>()
                   .WithMessage("ufunc 'add' did not contain a loop with signature matching types (dtype('<U1'), dtype('complex128')) -> None");
        }

        /// <summary>
        ///     <c>np.place</c> with a Python list of strings into numbers builds the values in the target's dtype (each
        ///     parsed by setitem, quoting the str): a bad one raises before anything is placed.
        /// </summary>
        [TestMethod]
        public void Place_ListOfTextIntoNumbers_ParsesEveryValueFirst()
        {
            using var d = np.zeros(new Shape(2), np.int64);
            Action place = () => np.place(d, np.array(new[] { true, true }), new object[] { "1", "x" });
            place.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: 'x'");
            d.ToArray<long>().Should().Equal(0L, 0L);
        }

        /// <summary>
        ///     The routes left over: <c>np.full</c> with a 0-d text ARRAY into numbers (the array cast), a text copy into bool
        ///     (the truth of each element; a lone unit above U+10FFFF is the SystemError), astral digits in Python's
        ///     <c>float(str)</c> / <c>int(str)</c> of a C# string (a surrogate pair is one code point), <c>np.add</c> of numbers
        ///     under a text <c>dtype=</c> (an empty broadcast, a Python complex), <c>np.frexp</c> into text outputs, the
        ///     engine's <c>cumprod</c> over text with a numeric <c>dtype=</c>, and an unsized text dtype over a typed numeric
        ///     array without <c>ndmin</c>.
        /// </summary>
        [TestMethod]
        public void RemainingTextNumberRoutes()
        {
            using (var full = np.full(new Shape(2), np.array("7"), np.int64))
                full.ToArray<long>().Should().Equal(7L, 7L);
            using (var truth = np.zeros(new Shape(3), np.@bool))
            {
                np.copyto(truth, np.array(new object[] { new byte[0], new byte[] { (byte)'a' }, new byte[] { 0, (byte)'b' } }, dtype: "S2"), "unsafe");
                truth.ToArray<bool>().Should().Equal(false, true, true);
                using var lone = np.array(new uint[] { 0x110000 }).view(np.dtype("<U1"));
                Action loneUnit = () => np.copyto(truth[":1"], lone, "unsafe");
                loneUnit.Should().Throw<SystemError>().WithMessage("invalid maximum character passed to PyUnicode_New");
                // An assignment of text into a bool array converts through the same truth loop (NumPy: SystemError too).
                Action assign = () => truth[":1"] = lone;
                assign.Should().Throw<SystemError>().WithMessage("invalid maximum character passed to PyUnicode_New");
                truth[":"] = np.array(new object[] { new byte[0], new byte[] { (byte)'a' }, new byte[] { 0, (byte)'b' } }, dtype: "S2");
                truth.ToArray<bool>().Should().Equal(false, true, true);
            }
            NumSharp.Backends.ReductionInitial.ParsePythonFloat("\U0001D7CF.5").Should().Be(1.5);
            ((long)np.array("\U0001D7CF\U0001D7D0")).Should().Be(12L);
            using (var empty = np.add(np.array(new double[0]), np.array(new double[0]), dtype: np.dtype("U")))
            {
                empty.dtype.str.Should().Be("<U64");
                empty.size.Should().Be(0);
            }
            using (var complexText = np.add(new Complex(1, 2), np.array(new[] { 1.5 }), dtype: np.dtype("U")))
            {
                complexText.dtype.str.Should().Be("<U96");
                NDArray.AsStringArray(complexText).Should().Equal("(1+2j)1.5");
            }
            var (mantissa, exponent) = np.frexp(np.array(new[] { 4.0 }), np.empty(new Shape(1), "U5"), np.empty(new Shape(1), "U5"));
            NDArray.AsStringArray(mantissa).Should().Equal("0.5");
            NDArray.AsStringArray(exponent).Should().Equal("3");
            using (var text = np.array(new[] { "2", "3" }))
            using (var cp = text.TensorEngine.ReduceCumMul(text, null, np.int64))
                cp.ToArray<long>().Should().Equal(2L, 6L);
            using (var plain = np.array((Array)new[] { 1, 22 }, np.dtype("U")))
            {
                plain.dtype.str.Should().Be("<U11");
                plain.ndim.Should().Be(1);
            }
        }

        /// <summary>
        ///     The conversion core's own checks (its callers run NumPy's checks first): exactly one operand holds text, a
        ///     mask is bool; an empty destination converts nothing.
        /// </summary>
        [TestMethod]
        public void ConversionCore_Checks()
        {
            using var n1 = np.array(new[] { 1, 2 });
            using var n2 = np.array(new[] { 1, 2 });
            using var t1 = U("1", "2");
            using var t2 = U("1", "2");
            Action numbers = () => NDTextConversion.Convert(n1.Storage, n2.Storage);
            numbers.Should().Throw<ArgumentException>().WithMessage("A text <-> number conversion needs exactly one string operand.");
            Action texts = () => NDTextConversion.Convert(t1.Storage, t2.Storage, (NumSharp.Backends.UnmanagedStorage)null);
            texts.Should().Throw<ArgumentException>();
            using var mask = np.array(new[] { 1, 0 });
            Action maskedTexts = () => NDTextConversion.Convert(t1.Storage, t2.Storage, mask.Storage);
            maskedTexts.Should().Throw<ArgumentException>().WithMessage("A text <-> number conversion needs exactly one string operand.");
            Action intMask = () => NDTextConversion.Convert(n1.Storage, t1.Storage, mask.Storage);
            intMask.Should().Throw<ArgumentException>().WithMessage("A where mask is bool, not int32.*");
            using var emptyNumbers = np.zeros(new Shape(0), np.int64);
            using var emptyText = np.array(new string[0], "U1");
            NDTextConversion.Convert(emptyNumbers.Storage, emptyText.Storage);
            using var noneSelected = np.array(new bool[0]);
            NDTextConversion.Convert(emptyNumbers.Storage, emptyText.Storage, noneSelected.Storage);
            // The error builder of a failed run re-parses the element; an element that parses after all has no error.
            using var valid = U("12");
            unsafe
            {
                NDTextParse.ElementError((byte*)valid.Storage.Address, true, 2, NPTypeCode.Int64, TextReprStyle.ArrayElement)
                    .Should().BeNull();
            }
        }

        /// <summary>
        ///     Where dynamic code is unavailable (the IL kernels switched off), the legacy text ↔ number casts convert
        ///     through the scalar cast loop instead of the house IL cast kernel, with the same answers.
        /// </summary>
        [TestMethod]
        public void TextNumberCasts_WithoutDynamicCode_GiveTheSameAnswers()
        {
            using var ints = np.array(new[] { 1, -22, 333 });
            using var text = U("4", "-55", "666");
            using var formatted = ints.astype(np.dtype("U"));
            using var parsed = text.astype(np.int32);
            DirectILKernelGenerator.Enabled = false;
            try
            {
                using var f = ints.astype(np.dtype("U"));
                NDArray.AsStringArray(f).Should().Equal(NDArray.AsStringArray(formatted));
                using var p = text.astype(np.int32);
                p.ToArray<int>().Should().Equal(parsed.ToArray<int>());
            }
            finally
            {
                DirectILKernelGenerator.Enabled = true;
            }
        }

        /// <summary>
        ///     The internal guards of the conversion layer: the numeric type code of a non-number, the promotion tables'
        ///     refusal of the string lane (and their own miss for any other pair), Python's <c>float(str)</c> grammar for an
        ///     <c>initial=</c> text, the <c>str()</c> length of a canonical number, the legacy string class's discovery of
        ///     None and numbers, and a C# null string.
        /// </summary>
        [TestMethod]
        public unsafe void InternalGuards()
        {
            Action kind = () => NDTextFormat.KindOf(NPTypeCode.String);
            kind.Should().Throw<NotSupportedException>().WithMessage("String has no number <-> text conversion.");
            Action family = () => NDTextParse.FamilyOf(NPTypeCode.String);
            family.Should().Throw<NotSupportedException>().WithMessage("String is not a parse target.");
            Action promote = () => np._FindCommonArrayType(NPTypeCode.String, NPTypeCode.Int32);
            promote.Should().Throw<NotSupportedException>().WithMessage("This operation is not supported for fixed-width string arrays*");
            Action promoteScalar = () => np._FindCommonArrayScalarType(NPTypeCode.Int32, NPTypeCode.String);
            promoteScalar.Should().Throw<NotSupportedException>();
            Action miss = () => np._FindCommonArrayType(NPTypeCode.Empty, NPTypeCode.Int32);
            miss.Should().Throw<System.Collections.Generic.KeyNotFoundException>();
            NumSharp.Backends.ReductionInitial.ParsePythonFloat("1_5.0").Should().Be(15.0);
            double v = 12.5;
            NDTextFormat.StrLength(TextNumberKind.Double, (byte*)&v).Should().Be(4);
            var strMeta = np.dtype("U").Meta;
            strMeta.DiscoverDescrFromObject(null).str.Should().Be("<U4");
            strMeta.DiscoverDescrFromObject(12345L).str.Should().Be("<U5");
            strMeta.IsKnownScalarType(typeof(string)).Should().BeTrue();
            strMeta.IsKnownScalarType(typeof(Half)).Should().BeFalse("a NumPy scalar is discovered through its own dtype");
            Action nullString = () => { NDArray _ = (string)null; };
            nullString.Should().Throw<ArgumentNullException>();
        }

        // ================================================================================== CPython character tables

        /// <summary>The pinned CPython 3.12 tables: Unicode decimal digits of every script, and str.isspace() (the C0
        /// separators U+001C..U+001F included, U+200B / U+180E / U+FEFF excluded).</summary>
        [TestMethod]
        public void PyUnicode_DecimalAndSpaceTables_AreCPython312s()
        {
            PyUnicode.DecimalValue('7').Should().Be(7);
            PyUnicode.DecimalValue(0x0663).Should().Be(3);      // ARABIC-INDIC DIGIT THREE
            PyUnicode.DecimalValue(0xFF10).Should().Be(0);      // FULLWIDTH DIGIT ZERO
            PyUnicode.DecimalValue(0x1D7CE).Should().Be(0);     // MATHEMATICAL BOLD DIGIT ZERO
            PyUnicode.DecimalValue('a').Should().Be(-1);
            PyUnicode.DecimalValue(0x00B2).Should().Be(-1);     // SUPERSCRIPT TWO is a digit, not a decimal
            foreach (var cp in new[] { 0x1C, 0x1D, 0x1E, 0x1F, 0x85, 0xA0, 0x1680, 0x2000, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 })
                PyUnicode.IsSpace(cp).Should().BeTrue($"U+{cp:X4} is whitespace to str.isspace()");
            foreach (var cp in new[] { 0x200B, 0x180E, 0xFEFF, (int)'a' })
                PyUnicode.IsSpace(cp).Should().BeFalse($"U+{cp:X4} is not whitespace to str.isspace()");
        }

        /// <summary>CPython's decimal-and-space rewrite: an all-ASCII text unchanged, Unicode digits to ASCII, Unicode
        /// spaces to ' ', and any other non-ASCII character to '?' with the text ending there.</summary>
        [TestMethod]
        public void PyUnicode_Transform_RewritesDigitsAndSpaces_StopsAtTheFirstOtherCharacter()
        {
            // The rewrite of one text, as ASCII.
            static string T(string s)
            {
                var buf = new byte[s.Length + 1];
                int n = PyUnicode.TransformDecimalAndSpaceToAscii(s.AsSpan(), buf);
                return Encoding.ASCII.GetString(buf, 0, n);
            }
            T("12 ab").Should().Be("12 ab");                // all ASCII: unchanged
            T("١٢٣").Should().Be("123");
            T("1 2").Should().Be("1 2");
            T("1é2").Should().Be("1?");                     // CPython writes '?' and ends the text there
            T("\U0001D7CF\U0001D7D0").Should().Be("12");    // a surrogate pair is ONE code point: an astral digit
        }
    }
}
