using System;
using System.IO;
using System.Linq;
using System.Text;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
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
    ///     truth values, concatenation, repetition, sorting), NumPy's refusal (NumPy's text), or the next-phase exception.
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
            names[2] = null;
            names.ToString(true).Should().Be("array(['cat', 'zebr', 'None'], dtype='<U4')", "None is stored as its str()");
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

        // ── Computing on text ─────────────────────────────────────────────────────────────────────

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
        ///     The "Sorting" snippets (strings phase 2, P2.2; every value probed against NumPy 2.4.2): code-point order, the
        ///     in-place sort, NumPy's introsort order of equal strings for the default kind and the input order for the
        ///     stable kind (20 single letters, past the 16 elements NumPy insertion-sorts), the partitions as full sorts,
        ///     and NumPy's two keyword errors with its texts.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
        [TestMethod]
        public void Computing_Sorting()
        {
            var words = np.array(new[] { "pear", "fig", "apple", "fig" });
            NDArray.AsStringArray(np.sort(words)).Should().Equal("apple", "fig", "fig", "pear");
            np.argsort(words).ToArray<long>().Should().Equal(2L, 1L, 3L, 0L);
            NDArray.AsStringArray(np.sort(np.array(new[] { "b", "A", "a", "é", "Z" }))).Should().Equal("A", "Z", "a", "b", "é");
            NDArray.AsStringArray(np.partition(words, 1)).Should().Equal("apple", "fig", "fig", "pear");
            np.argpartition(words, 1).ToArray<long>().Should().Equal(2L, 1L, 3L, 0L);
            words.sort();
            NDArray.AsStringArray(words).Should().Equal("apple", "fig", "fig", "pear");

            var grades = np.array("babacabcabacbacbabac".Select(c => c.ToString()).ToArray());
            np.argsort(grades).ToArray<long>().Should().Equal(1L, 3L, 16L, 5L, 13L, 8L, 18L, 10L, 0L, 17L, 15L, 12L, 9L, 6L, 2L, 7L, 14L, 4L, 11L, 19L);
            np.argsort(grades, kind: "stable").ToArray<long>()
              .Should().Equal(1L, 3L, 5L, 8L, 10L, 13L, 16L, 18L, 0L, 2L, 6L, 9L, 12L, 15L, 17L, 4L, 7L, 11L, 14L, 19L);
            np.argsort(grades, stable: true).ToArray<long>().Should().Equal(np.argsort(grades, kind: "stable").ToArray<long>());

            Action badKind = () => np.sort(words, kind: "timsort");
            badKind.Should().Throw<ValueError>().WithMessage("sort kind must be one of 'quick', 'heap', or 'stable' (got 'timsort')");
            Action both = () => np.sort(words, kind: "stable", stable: true);
            both.Should().Throw<ValueError>()
                .WithMessage("`kind` and keyword parameters can't be provided at the same time. Use only one of them.");
        }

        /// <summary>
        ///     The doc page's Searching section: the arg-reductions return the first extreme (flat and along an axis, into an
        ///     int32 <c>out=</c>), <c>np.searchsorted</c> over sorted text on both sides and with a number key (written as
        ///     text), and <c>np.digitize</c> of text, whose bins follow text order.
        /// </summary>
        [TestMethod]
        public void Computing_Searching()
        {
            var words = np.array(new[] { "pear", "fig", "apple", "fig" });
            np.argmax(words).Should().Be(0L);
            np.argmin(words).Should().Be(2L);
            var m = np.array(new[,] { { "b", "a" }, { "a", "c" } });
            np.argmax(m, 1).ToArray<long>().Should().Equal(0L, 1L);
            var o = np.zeros(new Shape(2L), np.int32);
            np.argmax(m, (int?)1, o);
            o.ToArray<int>().Should().Equal(0, 1);
            np.nanargmax(words).Should().Be(0L);

            var sorted = np.sort(words);
            NDArray.AsStringArray(sorted).Should().Equal("apple", "fig", "fig", "pear");
            np.searchsorted(sorted, np.array(new[] { "fig", "kiwi" })).ToArray<long>().Should().Equal(1L, 3L);
            np.searchsorted(sorted, np.array(new[] { "fig", "kiwi" }), "right").ToArray<long>().Should().Equal(3L, 3L);
            np.searchsorted(np.array(new[] { "1", "10", "2", "5" }), 3).Should().Be(3L);

            np.digitize(np.array(new[] { "10", "3", "25" }), np.array(new[] { 5, 20 })).ToArray<long>().Should().Equal(0L, 2L, 2L);
            Action uint64Out = () => np.argmax(m, (int?)1, np.zeros(new Shape(2L), np.uint64));
            uint64Out.Should().Throw<TypeError>();
        }

        /// <summary>
        ///     The doc page's "Unique values, set routines and statistics" section (every value probed against NumPy 2.4.2):
        ///     np.unique with counts, np.isin, intersect1d / setdiff1d, the take path of np.quantile and the interpolation's
        ///     TypeError; np.isin's two paths for numbers beside text; np.sort_complex's text order; a negative histogram count;
        ///     np.quantile with an integer q.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
        [TestMethod]
        public void Computing_UniqueSetRoutinesAndStatistics()
        {
            var fruit = np.array(new[] { "pear", "fig", "apple", "fig", "pear" });
            var (values, counts) = np.unique(fruit, return_counts: true);
            NDArray.AsStringArray(values).Should().Equal("apple", "fig", "pear");
            counts.ToArray<long>().Should().Equal(1L, 2L, 2L);
            np.isin(fruit, np.array(new[] { "fig", "kiwi" })).ToArray<bool>().Should().Equal(false, true, false, true, false);
            NDArray.AsStringArray(np.intersect1d(fruit, np.array(new[] { "pear", "plum" }))).Should().Equal("pear");
            NDArray.AsStringArray(np.setdiff1d(fruit, np.array(new[] { "pear" }))).Should().Equal("apple", "fig");
            ((string)np.quantile(fruit, 0.5, method: "lower").item()).Should().Be("fig");
            Action lerp = () => np.quantile(fruit, 0.5);
            lerp.Should().Throw<TypeError>().WithMessage("unsupported operand type(s) for -: 'numpy.str_' and 'numpy.str_'");

            np.isin(np.array(new[] { "1", "2" }), np.arange(20)).ToArray<bool>().Should().Equal(true, true);
            np.isin(np.array(new[] { "1", "2" }), np.array(new long[] { 1 })).ToArray<bool>().Should().Equal(false, false);
            np.sort_complex(np.array(new[] { "10", "9", "2+1j" })).ToArray<System.Numerics.Complex>().Should().Equal(
                new System.Numerics.Complex(10, 0), new System.Numerics.Complex(2, 1), new System.Numerics.Complex(9, 0));
            np.histogram(np.array(new[] { "1", "5", "10", "30", "2" }), np.array(new long[] { 0, 5, 10 })).hist
                .ToArray<long>().Should().Equal(4L, -2L);
            ((string)np.quantile(fruit, 1).item()).Should().Be("pear");
        }

        /// <summary>
        ///     The "The <c>numpy.char</c> module" snippets (the <c>numpy.char</c> plan's phase 1; every value probed against
        ///     NumPy 2.4.2): a comparison that strips trailing whitespace against the plain ufunc, <c>compare_chararrays</c>,
        ///     <c>np.char.add</c>, <c>np.char.multiply</c> and its ValueError rule, the <c>S</c> / <c>U</c> pair's
        ///     <c>NotImplemented</c> (null), and <c>np.@char</c> still converting to the <c>Char</c> dtype.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
        [TestMethod]
        public void Computing_CharModule()
        {
            var padded = np.array(new[] { "cat  ", "dog\t", "bird" });
            np.@char.equal(padded, "cat").ToArray<bool>().Should().Equal(true, false, false);
            np.equal(padded, "cat").ToArray<bool>().Should().Equal(false, false, false);
            np.@char.compare_chararrays(padded, "dog", "<", true).ToArray<bool>().Should().Equal(true, false, true);
            np.@char.add(padded, "!").ToString(true).Should().Be("array(['cat  !', 'dog\\t!', 'bird!'], dtype='<U6')");
            np.@char.multiply(np.array(new[] { "ab", "c" }), 3).ToString(true).Should().Be("array(['ababab', 'ccc'], dtype='<U6')");
            Action notInteger = () => np.@char.multiply(padded, 2.5);
            notInteger.Should().Throw<ValueError>().WithMessage("Can only multiply by integers");

            ((bool)np.@char.equal("a ", "a")).Should().BeTrue();
            np.@char.less(padded, new[] { "dog"u8.ToArray() }).Should().BeNull();   // NumPy's NotImplemented
            Action vstring = () => np.@char.equal(np.array(new[] { "a" }, np.dtypes.StringDType()), "a");
            vstring.Should().Throw<TypeError>().WithMessage("comparison of non-string arrays");

            np.ones(new Shape(2L), np.@char).dtype.Should().Be(np.dtype("char"));
            ((DType)np.@char).itemsize.Should().Be(2);
        }

        /// <summary>
        ///     The "The <c>chararray</c> class" snippets (the <c>numpy.char</c> plan's phase 2; every value probed against
        ///     NumPy 2.4.2): the stripped element, the unstripped <c>item()</c>, iteration, the repr, a view that stays a
        ///     chararray, the finalize ValueError, the float32 text of the <c>tolist()</c> route and <c>view(typeof(...))</c>.
        /// </summary>
        [TestMethod]
        public void Computing_CharArrayClass()
        {
            var c = np.@char.array(new[] { "cat  ", "dog\t", "bird" });
            c[0].item().Should().Be("cat");
            c.item(0).Should().Be("cat  ");
            var items = new System.Collections.Generic.List<object>();
            foreach (var s in c)
                items.Add(s);
            items.Should().Equal("cat", "dog", "bird");
            c.ToString(true).Should().Be("chararray(['cat', 'dog', 'bird'], dtype='<U5')");
            (c.T is NDCharArray).Should().BeTrue();
            Action numbers = () => c.astype(np.float64);
            numbers.Should().Throw<ValueError>().Where(e => e.Message == "Can only create a chararray from string data.");
            ((byte[])np.@char.array(np.array(new[] { 0.1f })).item(0)).Should().Equal("0.10000000149011612"u8.ToArray());
            (np.array(new[] { "a", "b" }).view(typeof(NDCharArray)) is NDCharArray).Should().BeTrue();
        }

        /// <summary>
        ///     The "What is not computed yet" examples: the comparison and the unique values of <c>StringDType</c> text compute
        ///     (strings phase 2, P2.T and P2.2 over <c>StringDType</c>), as the same calls over <c>U</c> text do; a string
        ///     <c>dtype=</c> request of <c>np.add</c> over numbers computes (the numbers written as text, then concatenated);
        ///     <c>np.evaluate</c> over text, a NumSharp route without a text form, throws
        ///     <see cref="StringComputationNotSupportedException"/> (a <see cref="NotSupportedException"/>).
        /// </summary>
        [TestMethod]
        public void Computing_NextPhase()
        {
            var names = Names();
            var t = names.astype(np.dtypes.StringDType());
            using (var eq = t == t)
                eq.ToArray<bool>().Should().Equal(true, true, true);
            NDArray.AsStringArray((NDArray)np.unique(t)).Should().Equal("bird", "cat", "dog");
            NDArray.AsStringArray(np.sort(t)).Should().Equal("bird", "cat", "dog");
            NDArray.AsStringArray((NDArray)np.unique(names)).Should().Equal("bird", "cat", "dog");
            np.isin(names, np.array(new[] { "cow" })).ToArray<bool>().Should().Equal(false, false, false);
            NDArray.AsStringArray(np.add(np.array(new[] { 1.5 }), np.array(new[] { 1.5 }), dtype: "U")).Should().Equal("1.51.5");
            Action fused = () => np.evaluate((NDExpr)names + names);
            fused.Should().Throw<StringComputationNotSupportedException>();
            // StringComputationNotSupportedException is a NotSupportedException.
            fused.Should().Throw<NotSupportedException>();

            // The .NET route for the np.strings functions NumSharp does not have yet.
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

        /// <summary>
        ///     The ".npy files" <c>StringDType</c> snippet: <c>np.save</c> writes the bytes <c>numpy.save</c> writes
        ///     (NumPy's protocol-4 pickle behind a <c>'|O'</c> header, captured from NumPy 2.4.2), <c>allow_pickle: true</c>
        ///     reads them back with the NA object, and the three refusals the paragraph names carry NumPy's texts.
        /// </summary>
        [TestMethod]
        public void Npy_StringDTypeIsNumPysPickle()
        {
            // numpy.save(f, np.array(['cat', None], dtype=np.dtypes.StringDType(na_object=None))) under NumPy 2.4.2.
            const string NumPysBytes =
                "k05VTVBZAQB2AHsnZGVzY3InOiAnfE8nLCAnZm9ydHJhbl9vcmRlcic6IEZhbHNlLCAnc2hhcGUnOiAoMiwpLCB9ICAgICAgICAgICAg" +
                "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAqABJWjAAAAAAAAAIwWbnVtcHkuX2NvcmUubXVs" +
                "dGlhcnJheZSMDF9yZWNvbnN0cnVjdJSTlIwFbnVtcHmUjAduZGFycmF5lJOUSwCFlEMBYpSHlFKUKEsBSwKFlIwVbnVtcHkuX2NvcmUu" +
                "X2ludGVybmFslIweX2NvbnZlcnRfdG9fc3RyaW5nZHR5cGVfa3dhcmdzlJOUSwFOhpRSlIldlCiMA2NhdJROZXSUYi4=";
            var words = np.array(new object[] { "cat", null }, np.dtypes.StringDType(na_object: null));
            var stream = new MemoryStream();
            np.save(stream, words);
            Convert.ToBase64String(stream.ToArray()).Should().Be(NumPysBytes);

            stream.Position = 0;
            np.load_npy(stream, allow_pickle: true).ToString(true)
              .Should().Be("array(['cat', None], dtype=StringDType(na_object=None))");
            stream.Position = 0;
            ((Action)(() => np.load_npy(stream))).Should().Throw<FormatException>()
                .WithMessage("Object arrays cannot be loaded when allow_pickle=False");
            ((Action)(() => np.save(new MemoryStream(), words, allow_pickle: false))).Should().Throw<ValueError>()
                .WithMessage("Object arrays cannot be saved when allow_pickle=False");

            // mmap_mode needs a file on disk; NumPy refuses before it reads the pickle, whatever allow_pickle says.
            var path = Path.Combine(Path.GetTempPath(), $"numsharp-doc-{Guid.NewGuid():N}.npy");
            try
            {
                np.save(path, words);
                ((Action)(() => np.load(path, mmap_mode: "r", allow_pickle: true))).Should().Throw<ValueError>()
                    .WithMessage("Array can't be memory-mapped: Python objects in dtype.");
            }
            finally
            {
                File.Delete(path);
            }
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

        /// <summary>
        ///     The "Iterating with np.nditer" snippet: a buffered cast to <c>U5</c> cuts the long text, and a buffered
        ///     <c>readwrite</c> pass writes every string back.
        /// </summary>
        [TestMethod]
        public void StringDType_Nditer()
        {
            var t = np.array(new[] { "a", "a long string over fifteen" }, np.dtypes.StringDType());
            var seen = new System.Collections.Generic.List<object>();
            using (var it = np.nditer(t, new[] { "refs_ok", "buffered" }, op_dtypes: new DType[] { "U5" }, casting: "unsafe"))
                foreach (var x in it)
                    seen.Add(x[0].item());
            seen.Should().Equal("a", "a lon");
            using (var it = np.nditer(t, new[] { "refs_ok", "buffered" }, new[] { "readwrite" }))
                foreach (var x in it)
                    x[0][Slice.Ellipsis] = (string)x[0].item() + "!";
            t.ToString(true).Should().Be("array(['a!', 'a long string over fifteen!'], dtype=StringDType())");
        }

        /// <summary>
        ///     The "Computing on StringDType text" examples, with a NaN-like NA: the comparisons (every one False with the
        ///     NA, <c>!=</c> included), <c>add</c> / <c>multiply</c> propagating the NA, <c>maximum</c> / <c>minimum</c> and
        ///     the reductions ordering it last, <c>isnan</c>, the default <c>StringDType()</c> result cast into a <c>U</c> out
        ///     (the NA becomes <c>''</c>) and <c>maximum</c>'s missing <c>U</c> loop. Every value probed from NumPy 2.4.2.
        /// </summary>
        [TestMethod]
        public void StringDType_Computing()
        {
            var nan = np.dtypes.StringDType(na_object: double.NaN);
            var t = np.array(new object[] { "pear", "fig", double.NaN }, nan);
            var u = np.array(new object[] { "kiwi", "apple", "plum" }, nan);
            (t < "grape").ToArray<bool>().Should().Equal(false, true, false);
            (t != u).ToArray<bool>().Should().Equal(true, true, false);
            (t + "!").ToString(true).Should().Be("array(['pear!', 'fig!', nan], dtype=StringDType(na_object=nan))");
            (t * 2).ToString(true).Should().Be("array(['pearpear', 'figfig', nan], dtype=StringDType(na_object=nan))");
            np.maximum(t, u).ToString(true).Should().Be("array(['pear', 'fig', nan], dtype=StringDType(na_object=nan))");
            np.minimum(t, u).ToString(true).Should().Be("array(['kiwi', 'apple', 'plum'], dtype=StringDType(na_object=nan))");
            np.sum(t["0:2"]).item().Should().Be("pearfig");
            np.cumsum(t).ToString(true).Should().Be("array(['pear', 'pearfig', nan], dtype=StringDType(na_object=nan))");
            np.argmax(t).Should().Be(2);
            np.isnan(t).ToArray<bool>().Should().Equal(false, false, true);
            Action noLoop = () => np.maximum(t, "x");
            noLoop.Should().Throw<TypeError>().WithMessage(
                "ufunc 'maximum' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.StrDType'>) -> None");

            var o = np.empty(new Shape(3), np.dtype("U8"));
            np.add(t, "!", o).ToString(true).Should().Be("array(['pear!', 'fig!', ''], dtype='<U8')");
        }

        /// <summary>
        ///     The "Ordering StringDType text" examples (strings phase 2, P2.2 over <c>StringDType</c>): a NaN-like NA sorting
        ///     after every text in the sort, the stable argsort, the unique values, the search and <c>np.isin</c>; a None-like
        ///     NA refused at the first comparison; the heapsort kind <c>StringDType</c> does not register; the order
        ///     statistics picking an element or refusing. Every value and message probed from NumPy 2.4.2.
        /// </summary>
        [TestMethod]
        public void StringDType_Ordering()
        {
            var nan = np.dtypes.StringDType(na_object: double.NaN);
            var t = np.array(new object[] { "pear", double.NaN, "fig", "apple", "fig" }, nan);
            // NumPy's repr wraps here (the line would pass 75 characters): the dtype goes on its own line.
            np.sort(t).ToString(true).Should().Be(
                "array(['apple', 'fig', 'fig', 'pear', nan],\n      dtype=StringDType(na_object=nan))");
            np.argsort(t, kind: "stable").ToArray<long>().Should().Equal(3L, 2L, 4L, 0L, 1L);
            ((NDArray)np.unique(t)).ToString(true).Should().Be(
                "array(['apple', 'fig', 'pear', nan], dtype=StringDType(na_object=nan))");
            np.searchsorted(np.sort(t), np.array(new object[] { "grape" }, nan)).ToArray<long>().Should().Equal(3L);
            np.isin(t, np.array(new object[] { "fig" }, nan)).ToArray<bool>().Should().Equal(false, false, true, false, true);
            var none = np.dtypes.StringDType(na_object: null);
            ((Action)(() => np.sort(np.array(new object[] { "b", null }, none)))).Should().Throw<ValueError>()
                .WithMessage("Cannot compare null that is not a nan-like value");
            ((Action)(() => np.sort(t, kind: "heapsort"))).Should().Throw<RuntimeError>().WithMessage("unsupported sort kind");

            var s = np.array(new[] { "b", "a", "c" }, np.dtypes.StringDType());
            np.quantile(s, 0.5, method: "lower").item().Should().Be("b");
            ((Action)(() => np.quantile(s, 0.5))).Should().Throw<TypeError>()
                .WithMessage("unsupported operand type(s) for -: 'str' and 'str'");
            ((Action)(() => np.median(s))).Should().Throw<TypeError>()
                .WithMessage("ufunc 'divide' not supported for the input types, *");
        }

        /// <summary>
        ///     The "Text files" examples: loadtxt with a converter whose null is the NA object and a whole-line field;
        ///     savetxt formatting the NA object itself (its <c>str()</c>, its own type's <c>%d</c>) and the Mismatch error;
        ///     tofile's text mode and fromfile's refusal.
        /// </summary>
        [TestMethod]
        public void StringDType_TextFiles()
        {
            var na = np.dtypes.StringDType(na_object: null);
            var t = np.loadtxt(new[] { "a b", "- c" }, na, converters: (Func<string, object>)(s => s == "-" ? null : s));
            t.item(1, 0).Should().BeNull();
            np.loadtxt(new[] { "a  b" }, np.dtypes.StringDType(), delimiter: ",").item().Should().Be("a  b");

            var w = new System.IO.StringWriter();
            np.savetxt(w, np.array(new object[] { "x", null }, na), "%s");
            w.ToString().Should().Be("x\nNone\n");
            var nanNA = np.dtypes.StringDType(na_object: double.NaN);
            ((Action)(() => np.savetxt(new System.IO.StringWriter(), np.array(new object[] { double.NaN }, nanNA), "%d")))
                .Should().Throw<ValueError>().WithMessage("cannot convert float NaN to integer");
            ((Action)(() => np.savetxt(new System.IO.StringWriter(), np.array(new[] { "x" }, na), "%d")))
                .Should().Throw<TypeError>()
                .WithMessage("Mismatch between array dtype ('StringDType(na_object=None)') and format specifier ('%d')");

            string path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "ns_doc_tofile_" + Guid.NewGuid().ToString("N"));
            try
            {
                t.tofile(path, sep: ",");
                System.IO.File.ReadAllText(path).Should().Be("a,b,None,c");
                ((Action)(() => np.fromfile(path, np.dtypes.StringDType(), sep: ",")))
                    .Should().Throw<ValueError>().WithMessage("Cannot read into object array");
            }
            finally
            {
                System.IO.File.Delete(path);
            }
        }

        /// <summary>The "What StringDType refuses" table: the raw-buffer routes and computation.</summary>
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
            // The structured path of np.unique along an axis cannot hold StringDType fields (NumPy's TypeError).
            ((Action)(() => np.unique(np.array(new object[,] { { "a", "b" }, { "a", "b" } }, np.dtypes.StringDType()), axis: 0)))
                .Should().Throw<TypeError>().WithMessage("The axis argument to unique is not supported for dtype StringDType()");

            // The NumPy-bug list: a 1-D accumulation into a 0-d out (NumPy crashes) is the wrong-size error, and maximum
            // with a None-like NA into an S out of non-ASCII texts (NumPy's SystemError) is the loop's own ValueError.
            using var one = np.array("", np.dtypes.StringDType());
            ((Action)(() => np.cumsum(np.array(new[] { "a", "b" }, np.dtypes.StringDType()), null, null, one)))
                .Should().Throw<IncorrectShapeException>().WithMessage("provided out is the wrong size for the accumulation.");
            var none = np.dtypes.StringDType(na_object: null);
            var withNA = np.array(new object[] { null, "été" }, none);
            ((Action)(() => np.maximum(withNA, np.array(new object[] { "x", "y" }, none), np.empty(new Shape(2), np.dtype("S4")))))
                .Should().Throw<ValueError>().WithMessage("Cannot compare null that is not a nan-like value");

            // A nan-quantile row whose result is a None NA: NumPy builds an object array, NumSharp has no object dtype.
            ((Action)(() => np.nanquantile(np.array(new object[,] { { null }, { null } }, none), 0.5, 1, method: "lower")))
                .Should().Throw<NotSupportedException>();

            // The new entries of the NumPy-bug list: a non-contiguous lexsort key (NumPy crashes), an arena haystack
            // (NumPy's MemoryError) and the complex cast (NumPy copies the real part into the imaginary part).
            var keys = np.array(new object[,] { { "b", "a" }, { "a", "b" }, { "a", "a" } }, np.dtypes.StringDType());
            np.lexsort(keys.T, -1).ToArray<long>().Should().Equal(2L, 0L, 1L);
            var hay = np.array(new object[] { new string('a', 20), new string('b', 20), new string('c', 20) }, np.dtypes.StringDType());
            np.searchsorted(hay, np.array(new object[] { "b" }, np.dtypes.StringDType())).ToArray<long>().Should().Equal(1L);
            np.sort_complex(np.array(new[] { "2", "1" }, np.dtypes.StringDType())).ToString(true)
                .Should().Be("array([1.+0.j, 2.+0.j])");
        }

        /// <summary>
        ///     The NumPy-bug list's <c>byteswap</c> line: NumPy 2.4.2 crashes; a cell has no byte order (the descriptor's
        ///     <c>byteorder</c> is <c>'|'</c>), so NumSharp returns a copy with its own buffer and the same texts, and with
        ///     <c>inplace: true</c> the array itself.
        /// </summary>
        [TestMethod]
        public void StringDType_ByteswapIsACopy()
        {
            var t = np.array(new[] { "a", "a long string over fifteen" }, np.dtypes.StringDType());
            t.dtype.byteorder.Should().Be('|');
            var copy = t.byteswap();
            copy.ToString(true).Should().Be(t.ToString(true));
            np.shares_memory(copy, t).Should().BeFalse("a copy owns its cells and its arena");
            ReferenceEquals(t.byteswap(inplace: true), t).Should().BeTrue();
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
