using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     <c>numpy.char</c> phase 4 (docs/plans/np-char.md, C3): the object path — <c>np.@char.split</c>, <c>rsplit</c>,
    ///     <c>splitlines</c> and <c>join</c>, the chararray members over them, and <see cref="NDListArray"/>, NumSharp's
    ///     stand-in for NumPy's object array of lists (owner decision C-D3). Every literal expectation was probed against
    ///     NumPy 2.4.2; the string oracle's section <c>strcharobj</c> (3,625 cases) replays the full matrices over every
    ///     layout, text class and argument kind. The <see cref="MisalignedAttribute"/> tests pin the documented divergences.
    /// </summary>
    [TestClass]
    public class CharObjectPathTests
    {
        /// <summary>CPython's SystemError for a one-character str above U+10FFFF.</summary>
        private const string InvalidMaxChar = "invalid maximum character passed to PyUnicode_New";

        /// <summary>NumPy 2.4.2's TypeError when join's object result is cast by <c>ObjectDType(n)</c>.</summary>
        private const string ObjectDType = "currently only the no-argument instantiation is supported; use `np.dtype` instead.";

        /// <summary>The five texts of NumPy's probes: runs at either end, an empty element, mixed whitespace.</summary>
        private static readonly string[] Texts = { "a b  c", "  lead trail  ", "", "x\ty\nz", "one" };

        /// <summary>A bytes value from Latin-1 text.</summary>
        /// <param name="text">The text (every char below U+0100).</param>
        /// <returns>The bytes.</returns>
        private static byte[] B(string text) => text.Select(c => (byte)c).ToArray();

        /// <summary>An <c>S</c> array of Latin-1 texts (NumPy's <c>np.array([b'...', ...])</c>).</summary>
        /// <param name="texts">The texts.</param>
        /// <returns>The array.</returns>
        private static NDArray S(params string[] texts) => np.array(texts.Select(t => (object)B(t)).ToArray());

        /// <summary>The lists of a 1-d result as .NET values (NumPy's <c>tolist()</c>).</summary>
        /// <param name="r">The result.</param>
        /// <returns>One list per element.</returns>
        private static List<object>[] Lists(NDListArray r) => ((IList)r.tolist()).Cast<List<object>>().ToArray();

        /// <summary>The lists of a 1-d result of str pieces.</summary>
        /// <param name="r">The result.</param>
        /// <returns>One string array per element.</returns>
        private static string[][] Words(NDListArray r) => Lists(r).Select(l => l.Cast<string>().ToArray()).ToArray();

        // ------------------------------------------------------------------ split / rsplit

        /// <summary>
        ///     Without a separator the words are the runs of non-whitespace (no empty word), with one every occurrence
        ///     splits (empty words appear); rsplit differs only with a limit, which it counts from the end.
        /// </summary>
        [TestMethod]
        public void Split_WordsAndSeparators_AsPythonsStrSplit()
        {
            var u = np.array(Texts);
            using var words = np.@char.split(u);
            Words(words).Should().BeEquivalentTo(new[]
            {
                new[] { "a", "b", "c" }, new[] { "lead", "trail" }, Array.Empty<string>(), new[] { "x", "y", "z" }, new[] { "one" },
            }, o => o.WithStrictOrdering());
            words.shape.Should().Equal(5L);

            using var bySpace = np.@char.split(u, " ");
            Words(bySpace)[0].Should().Equal("a", "b", "", "c");
            Words(bySpace)[1].Should().Equal("", "", "lead", "trail", "", "");

            using var right = np.@char.rsplit(u, maxsplit: 1);
            Words(right)[0].Should().Equal("a b", "c");
            Words(right)[1].Should().Equal("  lead", "trail");
            using var left = np.@char.split(u, maxsplit: 1);
            Words(left)[1].Should().Equal("lead", "trail  ");
            using var none = np.@char.split(u, maxsplit: 0);
            Words(none)[1].Should().Equal("lead trail  ");
            using var rnone = np.@char.rsplit(u, maxsplit: 0);
            Words(rnone)[1].Should().Equal("  lead trail");

            using var rcomma = np.@char.rsplit("a,b,,c,", ",", 2);
            ((IList)rcomma.tolist()).Cast<string>().Should().Equal("a,b,", "c", "");
            using var raaa = np.@char.rsplit("aaa", "aa");
            ((IList)raaa.tolist()).Cast<string>().Should().Equal("a", "");
            using var saaa = np.@char.split("aaa", "aa");
            ((IList)saaa.tolist()).Cast<string>().Should().Equal("", "a");
        }

        /// <summary>
        ///     The whitespace classes differ by method: str.split's 29 code points include <c>\x1c</c>–<c>\x1f</c>,
        ///     <c>\x85</c>, <c>\xa0</c> and <c>\u3000</c>; bytes.split's 6 bytes do not; U+200B is no whitespace at all.
        /// </summary>
        [TestMethod]
        public void Split_WhitespaceClasses_StrAndBytes()
        {
            int count = Enumerable.Range(0, 0x3001).Count(c => NDCharArray.IsPyWhitespace((uint)c));
            count.Should().Be(29);

            using var u = np.@char.split(np.array(new[] { "a\u001cb\u0085c\u00a0d\u3000e\u200bf" }));
            Words(u)[0].Should().Equal("a", "b", "c", "d", "e\u200bf");

            using var s = np.@char.split(S("a\u001cb\u0085c\u000bd\u000ce"));
            Lists(s)[0].Cast<byte[]>().Select(Encoding.Latin1.GetString).Should().Equal("a\u001cb\u0085c", "d", "e");
        }

        /// <summary>
        ///     A <c>U</c> element shorter than the separator, or of a narrower kind (<c>PyUnicode_KIND</c>), is CPython's
        ///     <c>[self]</c>: the list holds the element's own <c>np.str_</c>, which the repr prints as <c>np.str_('...')</c>
        ///     — a StringDType element is an exact str and prints plainly; a bytes split has no such shortcut.
        /// </summary>
        [TestMethod]
        public void Split_SelfShortcut_PrintsTheNumPyScalar()
        {
            np.@char.split("ab", "abc").ToString(true).Should().Be("array(list([np.str_('ab')]), dtype=object)");
            np.@char.split("abc", "Ā").ToString().Should().Be("[np.str_('abc')]");
            np.@char.split("ab\U0001F600", "Ā").ToString().Should().Be("['ab\U0001F600']");
            np.@char.split("abc", "abc").ToString().Should().Be("['', '']");
            np.@char.split(np.array(new[] { "ab" }, np.dtypes.StringDType()), "abc").ToString(true)
                .Should().Be("array([list(['ab'])], dtype=object)");
            np.@char.split(np.array(new[] { "a b  c", "" }), " ").ToString(true)
                .Should().Be("array([list(['a', 'b', '', 'c']), list([np.str_('')])], dtype=object)");
            np.@char.split(S("ab"), B("abc")).ToString().Should().Be("[list([b'ab'])]");
        }

        /// <summary>
        ///     bytes.split reads its separator through the buffer protocol, so a NumPy number's raw bytes are a separator:
        ///     <c>np.int64(5)</c> splits on 8 bytes, <c>True</c> on <c>\x01</c>, a str on its UCS-4 bytes — while str.split
        ///     refuses anything but a str or None.
        /// </summary>
        [TestMethod]
        public void Split_BytesSeparator_IsAnyBuffer()
        {
            var s5 = np.array(new object[] { B("a\u0005\0\0\0\0\0\0\0b") }, np.dtype("S10"));
            using var byInt = np.@char.split(s5, 5);
            Lists(byInt)[0].Cast<byte[]>().Select(Encoding.Latin1.GetString).Should().Equal("a", "b");
            using var byBool = np.@char.split(S("a\u0001b"), true);
            Lists(byBool)[0].Cast<byte[]>().Select(Encoding.Latin1.GetString).Should().Equal("a", "b");
            using var byStr = np.@char.split(S("xa\0\0\0y", "a"), "a");
            Lists(byStr)[0].Cast<byte[]>().Select(Encoding.Latin1.GetString).Should().Equal("x", "y");

            var u = np.array(new[] { "ab" });
            new Action(() => np.@char.split(u, 5)).Should().Throw<TypeError>().WithMessage("must be str or None, not numpy.int64");
            new Action(() => np.@char.split(u, B("a"))).Should().Throw<TypeError>().WithMessage("must be str or None, not numpy.bytes_");
            new Action(() => np.@char.split(S("ab"), np.array(new[] { "x" }, np.dtypes.StringDType())))
                .Should().Throw<TypeError>().WithMessage("a bytes-like object is required, not 'str'");
            new Action(() => np.@char.split(S("ab"), BigInteger.Pow(2, 64)))
                .Should().Throw<TypeError>().WithMessage("a bytes-like object is required, not 'int'");
        }

        /// <summary>
        ///     NumPy's errors in NumPy's order: the first argument's class, the broadcast (the multi-iterator's text), then per
        ///     element (C order) maxsplit's <c>__index__</c>, the separator's type and the empty separator.
        /// </summary>
        [TestMethod]
        public void Split_Errors_InNumPysOrder()
        {
            var u2 = np.array(new[] { "a", "b" });
            new Action(() => np.@char.split(np.array(new long[] { 1 }), "")).Should().Throw<TypeError>()
                .WithMessage("string operation on non-string array");
            new Action(() => np.@char.split(u2, "", new long[] { 0, 1, 2 })).Should().Throw<IncorrectShapeException>()
                .WithMessage("shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (2,) and arg 2 with shape (3,).");
            new Action(() => np.@char.split(u2, new[] { "", "x" }, new[] { 1.5, 1 })).Should().Throw<TypeError>()
                .WithMessage("'numpy.float64' object cannot be interpreted as an integer");
            new Action(() => np.@char.split(u2, new[] { "x", "" })).Should().Throw<ValueError>().WithMessage("empty separator");
            new Action(() => np.@char.split(u2, " ", true)).Should().Throw<TypeError>()
                .WithMessage("'numpy.bool' object cannot be interpreted as an integer");
            new Action(() => np.@char.split(u2, " ", "1")).Should().Throw<TypeError>()
                .WithMessage("'numpy.str_' object cannot be interpreted as an integer");
            new Action(() => np.@char.split(u2, " ", new object[] { 1, null })).Should().Throw<TypeError>()
                .WithMessage("'NoneType' object cannot be interpreted as an integer");
            new Action(() => np.@char.split(u2, " ", 1UL << 63)).Should().Throw<OverflowException>()
                .WithMessage("Python int too large to convert to C ssize_t");
            using var huge = np.@char.split(np.array(new[] { "a b c" }), " ", long.MinValue);
            Words(huge)[0].Should().Equal("a", "b", "c");
            // An empty array raises none of the element errors.
            using var empty = np.@char.split(np.zeros(new Shape(0), np.dtype("U3")), "");
            empty.shape.Should().Equal(0L);
        }

        /// <summary>
        ///     A <c>U</c> element, separator or piece that is ONE code unit above U+10FFFF is CPython's SystemError
        ///     (<c>PyUnicode_New</c>); a longer text holding such a unit is accepted (its maximum is capped).
        /// </summary>
        [TestMethod]
        public void Split_LoneInvalidCodeUnit_IsCPythonsSystemError()
        {
            var bad = np.array(new uint[] { 0x61, 0x20, 0x110000, 0x62 }).view(np.dtype("U4"));
            using var ok = np.@char.split(bad);
            ok.ToString().Should().Be("[list(['a', '\\U00110000b'])]");
            var lone = np.array(new uint[] { 0x110000, 0, 0, 0 }).view(np.dtype("U4"));
            new Action(() => np.@char.split(lone)).Should().Throw<SystemError>().WithMessage(InvalidMaxChar);
            var piece = np.array(new uint[] { 0x61, 0x20, 0x110000, 0 }).view(np.dtype("U4"));
            new Action(() => np.@char.split(piece)).Should().Throw<SystemError>().WithMessage(InvalidMaxChar);
            new Action(() => np.@char.split(np.array(new[] { "ab" }), np.array(new uint[] { 0x110000 }).view(np.dtype("U1"))))
                .Should().Throw<SystemError>().WithMessage(InvalidMaxChar);
        }

        /// <summary>
        ///     StringDType text: an NA that is not a str fails str.split's descriptor check; a string NA splits as itself; an
        ///     NA None separator is whitespace; trailing NULs of an element are kept (they are not padding).
        /// </summary>
        [TestMethod]
        public void Split_StringDType_NAAndNULs()
        {
            var tn = np.array(new object[] { "a b", null }, np.dtypes.StringDType(na_object: null));
            new Action(() => np.@char.split(tn)).Should().Throw<TypeError>()
                .WithMessage("descriptor 'split' for 'str' objects doesn't apply to a 'NoneType' object");
            var tnan = np.array(new object[] { "a b", double.NaN }, np.dtypes.StringDType(na_object: double.NaN));
            new Action(() => np.@char.split(tnan)).Should().Throw<TypeError>()
                .WithMessage("descriptor 'split' for 'str' objects doesn't apply to a 'float' object");
            var tstr = np.array(new object[] { "a b", "NA" }, np.dtypes.StringDType(na_object: "NA"));
            using var named = np.@char.split(tstr);
            Words(named)[1].Should().Equal("NA");
            using var naSep = np.@char.split(np.array(new[] { "a b", "c d" }), np.array(new object[] { " ", null }, np.dtypes.StringDType(na_object: null)));
            Words(naSep)[1].Should().Equal("c", "d");
            using var nul = np.@char.split(np.array(new[] { "w\0 x\0" }, np.dtypes.StringDType()));
            Words(nul)[0].Should().Equal("w\0", "x\0");
            using var nulU = np.@char.split(np.array(new[] { "w\0 x\0" }));
            Words(nulU)[0].Should().Equal("w\0", "x");
        }

        // ------------------------------------------------------------------ splitlines

        /// <summary>
        ///     str lines end at the eleven boundaries (\r\n one), bytes lines at \n \r \r\n only; a trailing boundary ends the
        ///     last line; keepends follows Python's truth rule (0.5, 'x' and NaN are true; '' and None false).
        /// </summary>
        [TestMethod]
        public void Splitlines_Boundaries_AndTruth()
        {
            var lines = np.array(new[] { "a\nb\r\nc\rd", "x\n", "", "\n", "a\u000bb\u000cc\u001cd\u001de\u001ef\u0085g\u2028h\u2029i" });
            using var plain = np.@char.splitlines(lines);
            Words(plain)[0].Should().Equal("a", "b", "c", "d");
            Words(plain)[1].Should().Equal("x");
            Words(plain)[2].Should().BeEmpty();
            Words(plain)[3].Should().Equal("");
            Words(plain)[4].Should().Equal("a", "b", "c", "d", "e", "f", "g", "h", "i");
            foreach (var truthy in new object[] { true, 2, 0.5, "x", double.NaN, new Complex(0, 1) })
            {
                using var kept = np.@char.splitlines(lines, truthy);
                Words(kept)[0].Should().Equal("a\n", "b\r\n", "c\r", "d");
            }
            foreach (var falsy in new object[] { false, 0, "", -0.0, Complex.Zero })
            {
                using var dropped = np.@char.splitlines(lines, falsy);
                Words(dropped)[0].Should().Equal("a", "b", "c", "d");
            }
            using var bytes = np.@char.splitlines(S("a\nb\r\nc\rd", "a\u000bb\u000cc\u001cd\u0085e"));
            Lists(bytes)[1].Cast<byte[]>().Select(Encoding.Latin1.GetString).Should().Equal("a\u000bb\u000cc\u001cd\u0085e");
            using var crlf = np.@char.splitlines("a\r\n\r\nb", true);
            ((IList)crlf.tolist()).Cast<string>().Should().Equal("a\r\n", "\r\n", "b");
        }

        // ------------------------------------------------------------------ join

        /// <summary>
        ///     join joins the CHARACTERS of each sequence string with its separator; the result takes the sequence's class,
        ///     as wide as the longest result (at least 1), an empty broadcast the sequence's dtype; a StringDType sequence
        ///     gives the default StringDType().
        /// </summary>
        [TestMethod]
        public void Join_TheCharacters_WithNumPysResultDtype()
        {
            var r = np.@char.join(new[] { "-", "." }, new[] { "ghc", "osd" });
            r.Strs().Should().Equal("g-h-c", "o.s.d");
            r.dtype.str.Should().Be("<U5");
            np.@char.join("ab", new[] { "", "x" }).dtype.str.Should().Be("<U1");
            var zero = np.@char.join("-", "osd");
            zero.ndim.Should().Be(0);
            zero.ToString(true).Should().Be("array('o-s-d', dtype='<U5')");
            var nul = np.@char.join("-", np.array(new[] { "ab\0", "c\0d" }, np.dtype("U4")));
            nul.Strs().Should().Equal("a-b", "c-\0-d");
            var t = np.@char.join("-", np.array(new object[] { "ab", "" }, np.dtypes.StringDType(na_object: "")));
            t.dtype.ToString().Should().Be("StringDType()");
            np.@char.join("-", np.zeros(new Shape(0), np.dtype("U3"))).dtype.str.Should().Be("<U3");
            np.@char.join("-", np.zeros(new Shape(0), np.dtype("S3"))).dtype.str.Should().Be("|S3");
            var sEmpty = np.@char.join(B("-"), new object[] { B("") });
            sEmpty.dtype.str.Should().Be("|S1");
        }

        /// <summary>
        ///     NumPy's join errors: bytes.join iterates a bytes object as ints; a number or None sequence is no iterable; a
        ///     non-text separator is the class error; an NA separator fails str.join's descriptor check.
        /// </summary>
        [TestMethod]
        public void Join_Errors_InNumPysOrder()
        {
            new Action(() => np.@char.join(B("-"), new object[] { B("ab") })).Should().Throw<TypeError>()
                .WithMessage("sequence item 0: expected a bytes-like object, int found");
            new Action(() => np.@char.join("-", new object[] { B("ab") })).Should().Throw<TypeError>()
                .WithMessage("sequence item 0: expected str instance, int found");
            new Action(() => np.@char.join(B("-"), new[] { "ab" })).Should().Throw<TypeError>()
                .WithMessage("sequence item 0: expected a bytes-like object, str found");
            new Action(() => np.@char.join(5, "ab")).Should().Throw<TypeError>().WithMessage("string operation on non-string array");
            new Action(() => np.@char.join("-", 5)).Should().Throw<TypeError>().WithMessage("can only join an iterable");
            new Action(() => np.@char.join("-", new long[] { 1, 2 })).Should().Throw<TypeError>().WithMessage("can only join an iterable");
            new Action(() => np.@char.join(np.array(new object[] { "-", null }, np.dtypes.StringDType(na_object: null)), "ab"))
                .Should().Throw<TypeError>().WithMessage("descriptor 'join' for 'str' objects doesn't apply to a 'NoneType' object");
            new Action(() => np.@char.join("-", np.array(new uint[] { 0x78, 0x110000 }).view(np.dtype("U2"))))
                .Should().Throw<SystemError>().WithMessage(InvalidMaxChar);
            np.@char.join(np.array(new uint[] { 0x78, 0x110000 }).view(np.dtype("U2")), "ab").dtype.str.Should().Be("<U4");
        }

        /// <summary>
        ///     NumPy 2.4.2 cannot finish <c>np.char.join(' ', np.char.split(a))</c>: every list joins, then
        ///     <c>_to_bytes_or_str_array</c> calls <c>ObjectDType(n)</c>, which takes no argument — NumSharp raises the same
        ///     TypeError (after the per-element errors); an empty broadcast is an empty object array.
        /// </summary>
        [TestMethod]
        public void Join_OverListArrays_NumPys2_4_2Failure()
        {
            using var words = np.@char.split(np.array(new[] { "a b c", "d  e", "" }));
            new Action(() => np.@char.join(" ", words)).Should().Throw<TypeError>().WithMessage(ObjectDType);
            using var bytesWords = np.@char.split(S("a b", "c"));
            new Action(() => np.@char.join(" ", bytesWords)).Should().Throw<TypeError>()
                .WithMessage("sequence item 0: expected str instance, bytes found");
            using var none = np.@char.split(np.zeros(new Shape(0, 2), np.dtype("U2")));
            using var empty = np.@char.join(np.array(new[] { "-", "+" }), none);
            empty.shape.Should().Equal(0L, 2L);
            // Through an object reference the empty result has no NDArray form.
            new Action(() => np.@char.join("-", (object)none)).Should().Throw<NotSupportedException>();
        }

        // ------------------------------------------------------------------ Python's None

        /// <summary>
        ///     The object path models NumPy's object arrays, so a null argument is Python's None: the text argument's class
        ///     error, join's non-iterable sequence, split's whitespace separator.
        /// </summary>
        [TestMethod]
        public void Null_IsPythonsNone()
        {
            new Action(() => np.@char.split(null)).Should().Throw<TypeError>().WithMessage("string operation on non-string array");
            new Action(() => np.@char.join(null, "ab")).Should().Throw<TypeError>().WithMessage("string operation on non-string array");
            new Action(() => np.@char.join("-", (object)null)).Should().Throw<TypeError>().WithMessage("can only join an iterable");
            new Action(() => np.@char.join("-", (NDListArray)null)).Should().Throw<TypeError>().WithMessage("can only join an iterable");
            using var bySep = np.@char.split(np.array(new[] { "a b", "c d" }), new object[] { " ", null });
            Words(bySep)[1].Should().Equal("c", "d");
            new Action(() => np.@char.split(new object[] { "a b", null })).Should().Throw<TypeError>()
                .WithMessage("string operation on non-string array");
        }

        // ------------------------------------------------------------------ chararray members

        /// <summary>
        ///     The members read the elements UNSTRIPPED (_vec_string makes NumPy scalars, not the chararray's stripped
        ///     elements), and their results are not chararrays: join's text is a plain array.
        /// </summary>
        [TestMethod]
        public void CharArrayMembers_ReadUnstrippedElements()
        {
            var c = np.@char.array(new[] { "a b ", "c  d" });
            using var bySpace = c.split(" ");
            Words(bySpace)[0].Should().Equal("a", "b", "");
            Words(bySpace)[1].Should().Equal("c", "", "d");
            using var right = c.rsplit(null, 1);
            Words(right)[0].Should().Equal("a", "b");
            var joined = c.join(new[] { "xy", "z" });
            joined.Should().NotBeOfType<NDCharArray>();
            joined.Strs().Should().Equal("xa b y", "z");
            np.@char.join("-", c).Strs().Should().Equal("a- -b- ", "c- - -d");
            using var lines = np.@char.array(new[] { "a\nb", "c" }).splitlines();
            Words(lines)[0].Should().Equal("a", "b");
        }

        // ------------------------------------------------------------------ NDListArray

        /// <summary>
        ///     The list array's surface: shape / ndim / size, a basic index as a VIEW (only the element ranges are sliced),
        ///     <c>item</c> and <c>tolist</c> with exact pieces, iteration over the first axis, NumPy's repr / str, and
        ///     disposal.
        /// </summary>
        [TestMethod]
        public void ListArray_Surface()
        {
            using var r = np.@char.split(np.array(new[,] { { "a b", "c" }, { "d", "" } }));
            r.shape.Should().Equal(2L, 2L);
            r.ndim.Should().Be(2);
            r.size.Should().Be(4);
            r.ToString(true).Should().Be("array([[list(['a', 'b']), list(['c'])],\n       [list(['d']), list([])]], dtype=object)");
            r.ToString().Should().Be("[[list(['a', 'b']) list(['c'])]\n [list(['d']) list([])]]");
            r.item(0, 0).Cast<string>().Should().Equal("a", "b");
            ((IList)((IList)r.tolist())[1]).Cast<List<object>>().Select(l => l.Count).Should().Equal(1, 0);

            using var row = r["1"];
            row.shape.Should().Equal(2L);
            Words(row)[0].Should().Equal("d");
            using var back = r[":, ::-1"];
            back.ToString().Should().Be("[[list(['c']) list(['a', 'b'])]\n [list([]) list(['d'])]]");
            var pieces = r[0, 0];
            pieces.Strs().Should().Equal("a", "b");
            pieces.dtype.str.Should().Be("<U1");
            new Action(() => _ = r[5, 0]).Should().Throw<IndexError>().WithMessage("index 5 is out of bounds for axis 0 with size 2");

            int rows = 0;
            foreach (NDListArray sub in r)
            {
                sub.ndim.Should().Be(1);
                rows++;
            }
            rows.Should().Be(2);

            using var zero = np.@char.split(np.array("a b"));
            zero.ToString().Should().Be("['a', 'b']");
            zero.ToString(true).Should().Be("array(list(['a', 'b']), dtype=object)");
            new Action(() => { foreach (var _ in zero) { } }).Should().Throw<TypeError>().WithMessage("iteration over a 0-d array");
            np.@char.split(np.zeros(new Shape(0), np.dtype("U1"))).ToString(true).Should().Be("array([], dtype=object)");
            np.@char.split(np.zeros(new Shape(2, 0), np.dtype("U1"))).ToString(true).Should().Be("array([], shape=(2, 0), dtype=object)");

            var disposed = np.@char.split(np.array(new[] { "a" }));
            disposed.Dispose();
            new Action(() => disposed.tolist()).Should().Throw<ObjectDisposedException>();
        }

        /// <summary>
        ///     A piece that ends in a NUL keeps it in <c>tolist</c> / <c>item</c> (NumPy's Python str), while the 1-D piece
        ///     array of an element access drops it, as every fixed-width element does — NumPy returns the list itself.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void ListArray_ElementAccess_IsAPieceArray_NotAList()
        {
            using var r = np.@char.split(np.array(new[] { "w\0 x" }));
            r.item(0).Cast<string>().Should().Equal("w\0", "x");
            r[0].Strs().Should().Equal("w", "x");
        }

        // ------------------------------------------------------------------ layouts and the search paths

        /// <summary>
        ///     The split runs over every layout through NDIter (strided, reversed, broadcast, offset views) and its searches
        ///     take vector paths on long texts: the results equal a reference split built from .NET's string search, for a
        ///     one-unit and a two-unit separator, from either end.
        /// </summary>
        [TestMethod]
        public void Split_LongTextsAndLayouts_MatchAReference()
        {
            var rng = new System.Random(20261016);
            var texts = Enumerable.Range(0, 64).Select(_ => new string(Enumerable.Range(0, rng.Next(0, 300))
                .Select(__ => "ab ,c"[rng.Next(5)]).ToArray())).ToArray();
            var u = np.array(texts);
            foreach (var view in new[] { u, u["::-1"], u["3:40"], u["::3"] })
            {
                var values = view.Strs();
                foreach (var sep in new[] { " ", ",c", "ab" })
                {
                    using var got = np.@char.split(view, sep);
                    var lists = Words(got);
                    for (int i = 0; i < values.Length; i++)
                        lists[i].Should().Equal(values[i].Split(sep), $"split of element {i} on '{sep}'");
                    using var gotRight = np.@char.rsplit(view, sep, 3);
                    var right = Words(gotRight);
                    for (int i = 0; i < values.Length; i++)
                        right[i].Should().Equal(ReferenceRSplit(values[i], sep, 3), $"rsplit of element {i} on '{sep}'");
                }
            }
        }

        /// <summary>A reference <c>str.rsplit(sep, maxsplit)</c>: the rightmost occurrences first.</summary>
        /// <param name="text">The text.</param>
        /// <param name="sep">The separator.</param>
        /// <param name="maxsplit">The limit.</param>
        /// <returns>The pieces in text order.</returns>
        private static string[] ReferenceRSplit(string text, string sep, int maxsplit)
        {
            var pieces = new List<string>();
            int end = text.Length;
            while (maxsplit-- > 0)
            {
                int at = end >= sep.Length ? text.LastIndexOf(sep, end - 1, end, StringComparison.Ordinal) : -1;
                if (at < 0)
                    break;
                pieces.Add(text.Substring(at + sep.Length, end - at - sep.Length));
                end = at;
            }
            pieces.Add(text.Substring(0, end));
            pieces.Reverse();
            return pieces.ToArray();
        }
    }

    /// <summary>Reads text arrays as .NET strings in the tests (NumPy's <c>tolist()</c> of a 1-d array).</summary>
    internal static class CharObjectPathTestText
    {
        /// <summary>The elements of a 1-d text array.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The strings (NumPy's <c>tolist()</c>).</returns>
        internal static string[] Strs(this NDArray a)
            => a.ndim == 0 ? new[] { (string)a.tolist() } : ((IList)a.tolist()).Cast<string>().ToArray();
    }
}
