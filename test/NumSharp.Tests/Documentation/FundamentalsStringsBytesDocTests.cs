using System;
using System.IO;
using System.Linq;
using System.Text;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Utilities;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for every code example in
    ///     <c>docs/website-src/docs/fundamentals/strings-and-bytes.md</c> (the "Working with arrays of
    ///     strings and bytes" fundamentals article). Each test mirrors one documented snippet and
    ///     asserts the behaviour the page claims: NumPy's fixed-width <c>U</c> / <c>S</c> dtypes store, convert (to and from
    ///     numbers included), move and print like NumPy 2.4.2, and computing on text gives NumPy's results (comparisons,
    ///     truth values, concatenation, repetition), NumPy's refusal (NumPy's text), or the next-phase exception.
    /// </summary>
    [TestClass]
    public class FundamentalsStringsBytesDocTests
    {
        private static NDArray Names() => np.array(new[] { "cat", "dog", "bird" });

        // ── The two string dtypes ─────────────────────────────────────────────────────────────────

        [TestMethod]
        public void TwoStringDtypes_WidthAndItemSize()
        {
            var names = Names();
            names.dtype.ToString(true).Should().Be("dtype('<U4')", "the longest value sets the width");
            names.dtype.itemsize.Should().Be(16, "4 characters x 4 bytes");
            np.dtype("S5").str.Should().Be("|S5");
            np.dtype("U5").name.Should().Be("str160", "NumPy's name counts bits");
            np.dtype("U5").itemsize.Should().Be(20);
            np.dtype("S5").itemsize.Should().Be(5);
            np.array(new[] { "\U0001F600" }).dtype.str.Should().Be("<U1", "an emoji is one UCS-4 code point");
        }

        [TestMethod]
        public void TwoStringDtypes_RawBytes()
        {
            np.array(new[] { "ab" }).tobytes().Should().Equal(0x61, 0, 0, 0, 0x62, 0, 0, 0);
            np.array(new[] { "ab" }).astype("S3").tobytes().Should().Equal(0x61, 0x62, 0);
        }

        // ── Creating string arrays ────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Creating_FromCSharpValues()
        {
            np.array(new[] { "cat", "dog" }).ToString(true).Should().Be("array(['cat', 'dog'], dtype='<U3')");
            np.array(new[] { "cat", "dog" }, np.dtype("S5")).ToString(true).Should().Be("array([b'cat', b'dog'], dtype='|S5')");
            np.zeros(new Shape(2), np.dtype("U3")).ToString(true).Should().Be("array(['', ''], dtype='<U3')");
            np.full(new Shape(2), "ab").ToString(true).Should().Be("array(['ab', 'ab'], dtype='<U2')");
            NDArray.Scalar("abc").ToString(true).Should().Be("array('abc', dtype='<U3')");
            np.frombuffer(Encoding.ASCII.GetBytes("abcdef"), "S3").ToString(true)
              .Should().Be("array([b'abc', b'def'], dtype='|S3')");
        }

        [TestMethod]
        public void Creating_BytesNeedObjectArray()
        {
            np.array(new object[] { Encoding.ASCII.GetBytes("ab"), Encoding.ASCII.GetBytes("cde") }).ToString(true)
              .Should().Be("array([b'ab', b'cde'], dtype='|S3')");
            np.array(new object[] { Encoding.ASCII.GetBytes("hi") }, np.dtype("S2")).ToString(true)
              .Should().Be("array([b'hi'], dtype='|S2')");

            // The footgun: a jagged byte[][] is numeric data in C#.
            var jagged = np.array(new[] { Encoding.ASCII.GetBytes("ab"), Encoding.ASCII.GetBytes("cd") });
            jagged.typecode.Should().Be(NPTypeCode.Byte);
            jagged.Should().BeShaped(2, 2);
        }

        // ── Reading and writing elements ──────────────────────────────────────────────────────────

        [TestMethod]
        public void Elements_ReadAndWrite()
        {
            var names = Names();
            names.item(1).Should().Be("dog");
            ((byte[])names.astype("S4").item(1)).Should().Equal((byte)'d', (byte)'o', (byte)'g');
            NDArray.AsStringArray(names).Should().Equal("cat", "dog", "bird");
            names[1].ToString(true).Should().Be("array('dog', dtype='<U4')");
            ((System.Collections.IEnumerable)names.tolist()).Cast<object>().Should().Equal("cat", "dog", "bird");

            names[1] = "zebras";
            names.ToString(true).Should().Be("array(['cat', 'zebr', 'bird'], dtype='<U4')", "a longer value is cut to the width");
        }

        [TestMethod]
        public void CharVectorApi_RefusesStringArrays()
        {
            var names = np.array(new[,] { { "ab", "c" } });
            Action get = () => names.GetString(0);
            get.Should().Throw<InvalidOperationException>().WithMessage("GetString works on a vector of chars*");
            Action set = () => names.SetString("x", 0);
            set.Should().Throw<InvalidOperationException>().WithMessage("SetString works on a vector of chars*");

            // The Char dtype keeps its char-vector API.
            var chars = NDArray.FromString("abc");
            chars.GetString().Should().Be("abc");
        }

        // ── Dtype integration ─────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Dtypes_PromotionAndCasting()
        {
            np.promote_types(np.dtype("S3"), np.dtype("U2")).ToString(true).Should().Be("dtype('<U3')");
            np.result_type(new[] { "U3", "S7" }).ToString(true).Should().Be("dtype('<U7')");
            np.can_cast(np.dtype("S3"), np.dtype("U3"), "safe").Should().BeTrue();
            np.can_cast(np.dtype("U3"), np.dtype("S3"), "same_kind").Should().BeFalse();
            np.can_cast(np.dtype("U5"), np.dtype("U3"), "safe").Should().BeFalse();
        }

        // ── Moving data ───────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Moving_ViewsCopiesAndCasts()
        {
            np.concatenate(new[] { np.array(new[] { "a" }), np.array(new[] { "long" }) }).ToString(true)
              .Should().Be("array(['a', 'long'], dtype='<U4')");
            Names().astype("U2").ToString(true).Should().Be("array(['ca', 'do', 'bi'], dtype='<U2')");
            np.array(new[,] { { "a", "b" }, { "c", "d" } }).T.ToString(true)
              .Should().Be("array([['a', 'c'],\n       ['b', 'd']], dtype='<U1')");
        }

        [TestMethod]
        public void Moving_NonAsciiToBytes_Throws()
        {
            Action act = () => np.array(new[] { "é" }).astype("S1");
            act.Should().Throw<UnicodeEncodeError>()
               .WithMessage("'ascii' codec can't encode character '\\xe9' in position 0: ordinal not in range(128)");
        }

        // ── Converting between text and numbers ──────────────────────────────────────────────────

        /// <summary>The "Numbers to text" snippets: a number is its scalar's <c>str()</c>, at NumPy's width for its
        /// type when the request is unsized, and cut to a sized request's width; the same text goes in through every
        /// store.</summary>
        [TestMethod]
        public void Converting_NumbersToText()
        {
            np.arange(3).astype("U").ToString(true).Should().Be("array(['0', '1', '2'], dtype='<U21')");
            np.array(new[] { 1.5, 1e16, 1e-5 }).astype("U").ToString(true).Should().Be("array(['1.5', '1e+16', '1e-05'], dtype='<U32')");
            np.array(new[] { true, false }).astype("U").ToString(true).Should().Be("array(['True', 'False'], dtype='<U5')");
            np.array(new[] { 0.1f }).astype("U").ToString(true).Should().Be("array(['0.1'], dtype='<U32')");
            np.array(new[] { 3.14159 }).astype("U3").ToString(true).Should().Be("array(['3.1'], dtype='<U3')");

            var u = np.array(new[] { "ab", "cd" });
            u[0] = 5;
            NDArray.AsStringArray(u).Should().Equal("5", "cd");
            u[1] = 2.5;
            NDArray.AsStringArray(u).Should().Equal("5", "2.");
            np.full(new Shape(2), 7, "U").ToString(true).Should().Be("array(['7', '7'], dtype='<U1')");
            np.ones(new Shape(2), "U").ToString(true).Should().Be("array(['1', '1'], dtype='<U1')");
            np.pad(np.array(new[] { "ab", "c" }), 1).ToString().Should().Be("['0' 'ab' 'c' '0']");
        }

        /// <summary>The "Text to numbers" snippets: each element parsed by <c>int()</c> / <c>float()</c> /
        /// <c>complex()</c>, NumPy's ValueError for a text that does not parse, NumPy's OverflowError for an integer that
        /// does not fit.</summary>
        [TestMethod]
        public void Converting_TextToNumbers()
        {
            np.array(new[] { "1", " 22 ", "-3", "1_000" }).astype(np.int64).ToArray<long>().Should().Equal(1L, 22L, -3L, 1000L);
            var f = np.array(new[] { "1.5", "inf", "-nan", "1e3" }).astype(np.float64).ToArray<double>();
            f[0].Should().Be(1.5);
            double.IsPositiveInfinity(f[1]).Should().BeTrue();
            double.IsNaN(f[2]).Should().BeTrue();
            f[3].Should().Be(1000.0);
            np.array(new[] { "1+2j", "3", "(4-5j)" }).astype(np.complex128).ToString().Should().Be("[1.+2.j 3.+0.j 4.-5.j]");

            Action badInt = () => np.array(new[] { "1", "x" }).astype(np.int64);
            badInt.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: np.str_('x')");
            Action badFloat = () => np.array(new[] { "1", "a" }).astype(np.float64);
            badFloat.Should().Throw<ValueError>().WithMessage("could not convert string to float: np.str_('a')");
            Action overflow = () => np.array(new[] { "300" }).astype(np.int8);
            overflow.Should().Throw<OverflowException>().WithMessage("Python integer 300 out of bounds for int8");

            var a = np.zeros(new Shape(3), np.int32);
            a[0] = "12";
            a.ToArray<int>().Should().Equal(12, 0, 0);
            Action hello = () => a[0] = "hello";
            hello.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: 'hello'");
        }

        /// <summary>The "Conversions inside other calls" snippets: a string <c>dtype=</c> request of <c>np.add</c>, a text
        /// <c>out=</c> (of a comparison of text, and of a numeric ufunc), a reduction with a numeric <c>dtype=</c>, and the
        /// out's text parsed first under <c>where=</c>.</summary>
        [TestMethod]
        public void Converting_StoresAndUfuncs()
        {
            np.add(np.array(new[] { 1.5, 2.0 }), np.array(new long[] { 1, 2 }), dtype: "U").ToString(true)
              .Should().Be("array(['1.51', '2.02'], dtype='<U53')");
            np.equal(Names(), np.array(new[] { "cat", "cow", "bird" }), np.empty(new Shape(3), "U5")).ToString()
              .Should().Be("['True' 'False' 'True']");
            np.add(np.array(new[] { 1.0, 2.5 }), np.array(new[] { 1.0, 3.0 }), np.zeros(new Shape(2), "U5")).ToString()
              .Should().Be("['2.0' '5.5']");
            np.prod(np.array(new[] { "1.5", "2", "4" }), (int?)null, np.float64).GetDouble().Should().Be(12.0);

            var outText = np.array(new[] { "1", "x" }).astype("U32");
            Action masked = () => np.sqrt(np.array(new[] { 1.0, 2.5 }), outText, np.array(new[] { true, false }));
            masked.Should().Throw<ValueError>().WithMessage("could not convert string to float: np.str_('x')");
        }

        /// <summary>The breaking change: a C# string is NumPy's <c>np.array(str)</c>, a 0-d <c>U</c> array, never the
        /// numbers the text spells.</summary>
        [TestMethod]
        public void Converting_CSharpStringIsText()
        {
            NDArray x = "[1, 2, 3]";
            x.ToString(true).Should().Be("array('[1, 2, 3]', dtype='<U9')");
            NDArray y = "1.5";
            y.astype(np.float64).GetDouble().Should().Be(1.5);
        }

        // ── Printing ──────────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Printing_StrAndRepr()
        {
            var names = Names();
            names.ToString().Should().Be("['cat' 'dog' 'bird']");
            names.ToString(true).Should().Be("array(['cat', 'dog', 'bird'], dtype='<U4')");
            np.array2string(np.array(new[] { "a'b", "c\"d" })).Should().Be("[\"a'b\" 'c\"d']");
        }

        // ── Computing on text (the next phase) ────────────────────────────────────────────────────

        [TestMethod]
        public void Computing_NumPyRefusals()
        {
            var names = Names();
            Action sum = () => np.sum(names);
            sum.Should().Throw<TypeError>().WithMessage(
                "the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U4'), dtype('<U4'), dtype('<U8'))");
            Action sqrt = () => np.sqrt(names);
            sqrt.Should().Throw<TypeError>().WithMessage("ufunc 'sqrt' not supported for the input types, *");
        }

        /// <summary>
        ///     The comparisons compute on text (strings phase 2, P2.1): <c>names == names</c> is <c>[True, True, True]</c>,
        ///     <c>names &lt; ['dog']</c> is <c>[True, False, True]</c> (probed against NumPy 2.4.2).
        /// </summary>
        [TestMethod]
        public void Computing_Comparisons()
        {
            var names = Names();
            (names == names).ToArray<bool>().Should().Equal(true, true, true);
            (names < np.array(new[] { "dog" })).ToArray<bool>().Should().Equal(true, false, true);
            (names > "cat").ToArray<bool>().Should().Equal(false, true, false);
        }

        /// <summary>
        ///     A string is True when it is not empty (any code unit not NUL): <c>['', 'a', ' '].astype(bool)</c> is
        ///     <c>[False, True, True]</c>, <c>count_nonzero(names)</c> is 3 (probed against NumPy 2.4.2).
        /// </summary>
        [TestMethod]
        public void Computing_TruthValues()
        {
            np.array(new[] { "", "a", " " }).astype(np.@bool).ToArray<bool>().Should().Equal(false, true, true);
            np.count_nonzero(Names()).Should().Be(3L);
        }

        /// <summary>
        ///     The "Concatenation and repetition" snippets (strings phase 2, P2.3; every value probed against NumPy 2.4.2):
        ///     <c>names + 's'</c> is <c>&lt;U5</c>, a string on the left stays the first operand, <c>np.strings.multiply</c>
        ///     sizes its result to the longest repetition, and the <c>np.multiply</c> ufunc cuts at <c>out=</c>'s width and
        ///     refuses without one.
        /// </summary>
        [TestMethod]
        public void Computing_ConcatenationAndRepetition()
        {
            var names = Names();
            (names + "s").ToString(true).Should().Be("array(['cats', 'dogs', 'birds'], dtype='<U5')");
            ("a " + names).ToString(true).Should().Be("array(['a cat', 'a dog', 'a bird'], dtype='<U6')");
            np.add(names, np.array(new[] { "!", "?", "" })).ToString(true).Should().Be("array(['cat!', 'dog?', 'bird'], dtype='<U5')");

            np.strings.multiply(names, 2).ToString(true).Should().Be("array(['catcat', 'dogdog', 'birdbird'], dtype='<U8')");
            np.strings.multiply(names, np.array(new long[] { 1, 0, -1 })).ToString(true)
              .Should().Be("array(['cat', '', ''], dtype='<U3')");

            np.multiply(names, 2, np.zeros(new Shape(3L), "U5")).ToString(true).Should().Be("array(['catca', 'dogdo', 'birdb'], dtype='<U5')");
            Action noOut = () => { var _ = names * 2; };
            noOut.Should().Throw<TypeError>().WithMessage("The 'out' kwarg is necessary when using the string multiply ufunc directly.*");
        }

        /// <summary>
        ///     What the doc page still lists as a later phase: sorting and the unique values (a sort) refuse with
        ///     <see cref="StringComputationNotSupportedException"/> (a <see cref="NotSupportedException"/>); a string
        ///     <c>dtype=</c> request of <c>np.add</c> over numbers computes (the numbers written as text, then concatenated).
        /// </summary>
        [TestMethod]
        public void Computing_NextPhase()
        {
            var names = Names();
            Action sort = () => np.sort(names);
            sort.Should().Throw<StringComputationNotSupportedException>();
            Action unique = () => np.unique(names);
            unique.Should().Throw<StringComputationNotSupportedException>();
            NDArray.AsStringArray(np.add(np.array(new[] { 1.5 }), np.array(new[] { 1.5 }), dtype: "U")).Should().Equal("1.51.5");
            // StringComputationNotSupportedException is a NotSupportedException.
            sort.Should().Throw<NotSupportedException>();

            // The .NET route until then.
            NDArray.AsStringArray(names).Select(s => s.ToUpper()).Should().Equal("CAT", "DOG", "BIRD");
        }

        [TestMethod]
        public void Computing_NeverEqualToNumbers()
        {
            var eq = Names() == np.array(new[] { 1, 2, 3 });
            eq.ToString().Should().Be("[False False False]");
        }

        /// <summary>The "Masked text arrays" snippets: the fill value stored at its own width, cut by
        /// <c>filled()</c>; a masked reduction refuses; a number fill is written as its text.</summary>
        [TestMethod]
        public void Computing_MaskedTextArrays()
        {
            var m = np.ma.array(np.array(new[] { "ab", "c", "de" }), mask: new[] { true, false, false });
            ((string)m.fill_value).Should().Be("N/A");
            NDArray.AsStringArray(m.filled()).Should().Equal("N/", "c", "de");
            NDArray.AsStringArray(m.filled("zz")).Should().Equal("zz", "c", "de");
            np.ma.count(m).GetInt64().Should().Be(2L);
            Action sum = () => np.ma.sum(m);
            sum.Should().Throw<TypeError>().WithMessage(
                "the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U2'), dtype('<U2'), dtype('<U4'))");
            NDArray.AsStringArray(m.filled(5)).Should().Equal("5", "c", "de");
        }

        [TestMethod]
        public void Computing_ChecksInNumPysOrder()
        {
            Action axis = () => np.sum(Names(), 9);
            axis.Should().Throw<AxisError>().WithMessage("axis 9 is out of bounds for array of dimension 1*");
            Action noLoop = () => np.sqrt(np.array(new[] { 1.5 }), dtype: "U");
            noLoop.Should().Throw<IncorrectTypeException>()
                  .WithMessage("No loop matching the specified signature and casting was found for ufunc sqrt");
        }

        // ── .npy files ────────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Npy_RoundTrip()
        {
            var stream = new MemoryStream();
            np.save(stream, Names());
            stream.Position = 0;
            np.load_npy(stream).ToString(true).Should().Be("array(['cat', 'dog', 'bird'], dtype='<U4')");
        }

        [TestMethod]
        public void Npy_CharWritesAsU1()
        {
            var stream = new MemoryStream();
            np.save(stream, np.array(new[] { 'a', 'b' }));
            stream.Position = 0;
            // The <U1 file reads back as NumPy's own <U1 string array (breaking: it used to read as Char).
            np.load_npy(stream).ToString(true).Should().Be("array(['a', 'b'], dtype='<U1')");
        }

        // ── The Char dtype ────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Char_IsTwoByteUtf16CodeUnit()
        {
            np.array(new[] { 'a', 'b', 'c' }).typecode.Should().Be(NPTypeCode.Char);
            InfoOf<char>.Size.Should().Be(2, "actual memory footprint is 2 bytes (UTF-16 code unit)");
        }

        [TestMethod]
        public void UnsupportedDtypes_StillThrow()
        {
            ((Action)(() => np.dtype("V7"))).Should().Throw<NotSupportedException>();
            ((Action)(() => np.dtype("O"))).Should().Throw<NotSupportedException>();
        }

        // ── Variable-width text: StringDType ──────────────────────────────────────────────────────

        /// <summary>The "Creating StringDType arrays" snippet: the descriptor, the 16-byte cell, zeros and ones.</summary>
        [TestMethod]
        public void StringDType_Creating()
        {
            var t = np.array(new[] { "a", "a much longer text" }, np.dtype("T"));
            t.dtype.ToString(true).Should().Be("StringDType()");
            t.dtype.itemsize.Should().Be(16, "the cell, whatever the length of the text");
            np.zeros(new Shape(2), np.dtypes.StringDType()).ToString(true).Should().Be("array(['', ''], dtype=StringDType())");
            np.ones(new Shape(2), np.dtypes.StringDType()).ToString(true).Should().Be("array(['1', '1'], dtype=StringDType())");
        }

        /// <summary>The "Missing data: the NA object" snippet: None as the NA object, and None as text without one.</summary>
        [TestMethod]
        public void StringDType_NAObjects()
        {
            var withNA = np.dtypes.StringDType(na_object: null);
            var a = np.array(new object[] { "x", null }, withNA);
            a.item(1).Should().BeNull();
            a.ToString(true).Should().Be("array(['x', None], dtype=StringDType(na_object=None))");
            np.array(new object[] { "x", null }, np.dtype("T")).item(1).Should().Be("None");
        }

        /// <summary>The "Writing elements" snippet: a value is its str(); coerce=False accepts only text.</summary>
        [TestMethod]
        public void StringDType_WritingElements()
        {
            var t = np.array(new[] { "a", "b" }, np.dtypes.StringDType());
            t[0] = 3.25;
            t.item(0).Should().Be("3.25");
            var strict = np.array(new[] { "a" }, np.dtypes.StringDType(coerce: false));
            ((Action)(() => strict[0] = 5)).Should().Throw<ValueError>()
                .WithMessage("StringDType only allows string data when string coercion is disabled.");
        }

        /// <summary>The "Moving and casting" snippet: parse to numbers, cut to a sized U, numbers to StringDType.</summary>
        [TestMethod]
        public void StringDType_MovingAndCasting()
        {
            var t = np.array(new[] { "12", "-3" }, np.dtypes.StringDType());
            t.astype(np.int64).ToArray<long>().Should().Equal(12L, -3L);
            NDArray.AsStringArray(t.astype("U1")).Should().Equal("1", "-");
            np.array(new[] { 1.5, 2.0 }).astype(np.dtypes.StringDType()).ToString(true)
                .Should().Be("array(['1.5', '2.0'], dtype=StringDType())");
        }

        /// <summary>The "What StringDType refuses" table: the raw-buffer routes, buffering np.nditer, computation.</summary>
        [TestMethod]
        public void StringDType_Refusals()
        {
            var t = np.array(new[] { "a" }, np.dtypes.StringDType());
            ((Action)(() => t.view(np.dtype("U1")))).Should().Throw<TypeError>()
                .WithMessage("Cannot change data-type for array of references.");
            ((Action)(() => { var _ = t.data; })).Should().Throw<ValueError>()
                .WithMessage("cannot include dtype 'numpy.dtypes.StringDType' in a buffer");
            ((Action)(() => np.frombuffer(new byte[16], "T"))).Should().Throw<ValueError>()
                .WithMessage("cannot create an OBJECT array from memory buffer");
            ((Action)(() => np.ndarray(new Shape(1), np.dtype("T"), new byte[16]))).Should().Throw<ArgumentException>();
            ((Action)(() => np.nditer(t, flags: new[] { "refs_ok", "buffered" }))).Should().Throw<NotSupportedException>();
            ((Action)(() => { var _ = t == t; })).Should().Throw<StringComputationNotSupportedException>();
        }

        // ── Byte streams via np.frombuffer ────────────────────────────────────────────────────────

        [TestMethod]
        public void Bytes_ViaFrombuffer()
        {
            byte[] raw = { 1, 2, 3 };
            var asBytes = np.frombuffer(raw, np.uint8);
            asBytes.typecode.Should().Be(NPTypeCode.Byte);
            asBytes.ToArray<byte>().Should().Equal((byte)1, (byte)2, (byte)3);

            // reinterpret 16 bytes as 4 float32
            byte[] fbytes = new byte[16];
            Buffer.BlockCopy(new[] { 1f, 2f, 3f, 4f }, 0, fbytes, 0, 16);
            np.frombuffer(fbytes, np.float32).ToArray<float>().Should().Equal(1f, 2f, 3f, 4f);
        }
    }
}
