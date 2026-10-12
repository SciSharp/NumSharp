using System;
using System.Linq;
using System.Numerics;
using System.Text;
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
        }
    }
}
