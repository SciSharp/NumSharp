using System;
using System.Collections;
using System.Linq;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     <c>numpy.char</c>'s string functions and the chararray string members over them (docs/plans/np-char.md, C4 / C5):
    ///     numpy.char's own <c>partition</c> / <c>rpartition</c> (the three parts stacked on a new last axis), the functions
    ///     it re-exports from numpy.strings, and NumPy's three return rules for the members — <c>asarray(...)</c> (a <c>U</c>
    ///     result owns its data, an <c>S</c> result views the computed array), the ufunc's chararray view, a plain int64 /
    ///     bool array. Every literal expectation was probed against NumPy 2.4.2; the string oracle's section
    ///     <c>strcharmeth</c> replays the full matrices over every layout, class and argument kind.
    /// </summary>
    [TestClass]
    public class CharStringMethodsTests
    {
        /// <summary>The texts of NumPy's probes: a separator in the middle, none, one inside trailing whitespace.</summary>
        private static readonly string[] Words = { "ab cd", "x", "a b c " };

        /// <summary>A <c>U</c> array of <see cref="Words"/>.</summary>
        /// <returns>The array.</returns>
        private static NDArray U() => np.array(Words);

        /// <summary>A <c>U</c> chararray view of <see cref="Words"/> (NumPy's <c>u.view(np.char.chararray)</c>).</summary>
        /// <returns>The chararray.</returns>
        private static NDCharArray CU() => (NDCharArray)np.array(Words).view(typeof(NDCharArray));

        /// <summary>An <c>S</c> chararray view of <see cref="Words"/>.</summary>
        /// <returns>The chararray.</returns>
        private static NDCharArray CS()
            => (NDCharArray)np.array(Words.Select(w => (object)System.Text.Encoding.Latin1.GetBytes(w)).ToArray()).view(typeof(NDCharArray));

        /// <summary>The elements of a text array in C order, unstripped (NumPy's <c>tolist()</c>; bytes read as Latin-1).</summary>
        /// <param name="a">The array.</param>
        /// <returns>The texts.</returns>
        private static string[] Texts(NDArray a)
        {
            object list = a.ndim == 0 ? a.tolist() : a.flatten().tolist();
            static string one(object x) => x is byte[] b ? System.Text.Encoding.Latin1.GetString(b) : (string)x;
            return list is IList items ? items.Cast<object>().Select(one).ToArray() : new[] { one(list) };
        }

        // ------------------------------------------------------------------ np.char.partition / rpartition (C4)

        /// <summary>
        ///     <c>np.char.partition</c> is <c>np.stack(np.strings.partition(a, sep), axis=-1)</c>: one PLAIN array (a
        ///     chararray input too: np.stack keeps no subclass), a last axis of the three parts, as wide as the widest part,
        ///     a fresh C-ordered copy; a 0-d input gives shape (3,).
        /// </summary>
        [TestMethod]
        public void Partition_StacksTheThreeParts_IntoOnePlainArray()
        {
            var r = np.@char.partition(U(), " ");
            r.Should().NotBeOfType<NDCharArray>();
            r.dtype.ToString().Should().Be("<U4");
            r.shape.Should().Equal(3L, 3L);
            r.flags.owndata.Should().BeTrue();
            r.strides.Should().Equal(48L, 16L);
            Texts(r).Should().Equal("ab", " ", "cd", "x", "", "", "a", " ", "b c ");

            var right = np.@char.rpartition(U(), " ");
            right.dtype.ToString().Should().Be("<U5");
            Texts(right).Should().Equal("ab", " ", "cd", "", "", "x", "a b c", " ", "");

            np.@char.partition(CU(), " ").Should().NotBeOfType<NDCharArray>();
            var scalar = np.@char.partition(np.array("ab cd"), " ");
            scalar.shape.Should().Equal(3L);
            Texts(scalar).Should().Equal("ab", " ", "cd");
        }

        /// <summary>
        ///     An F-ordered input: NumPy's np.stack of the three field views orders the axes by stride, giving strides
        ///     (24, 48, 8) for <c>&lt;U2</c> — neither C nor F order. NumSharp's result has the same layout.
        /// </summary>
        [TestMethod]
        public void Partition_FOrderedInput_HasNumPysStackLayout()
        {
            var uf = np.asfortranarray(np.array(new[,] { { "ab", "c d" }, { "e f", "g" } }));
            var r = np.@char.partition(uf, " ");
            r.shape.Should().Equal(2L, 2L, 3L);
            r.strides.Should().Equal(24L, 48L, 8L);
            r.Shape.IsContiguous.Should().BeFalse();
            r.Shape.IsFContiguous.Should().BeFalse();
            Texts(r).Should().Equal("ab", "", "", "c", " ", "d", "e", " ", "f", "g", "", "");
        }

        /// <summary>
        ///     The separator is cast to the text's dtype first (<c>sep.astype(a.dtype)</c>): a number separator is its text,
        ///     a str separator beside bytes is encoded; a number TEXT meets the reverse cast and NumPy's int() error.
        /// </summary>
        [TestMethod]
        public void Partition_CastsTheSeparator_AndRefusesANumberText()
        {
            var byOne = CU().partition(1);
            byOne.dtype.ToString().Should().Be("<U6");
            Texts(byOne).Should().Equal("ab cd", "", "", "x", "", "", "a b c ", "", "");

            Action number = () => np.@char.partition(5, " ");
            number.Should().Throw<ValueError>().Where(e => e.Message == "invalid literal for int() with base 10: np.str_(' ')");
        }

        /// <summary>
        ///     StringDType text partitions around a StringDType separator; NumPy 2.4.2 has no loop for a str separator beside
        ///     StringDType (its TypeError, verbatim).
        /// </summary>
        [TestMethod]
        public void Partition_StringDType_NeedsAStringDTypeSeparator()
        {
            var t = np.array(new[] { "a b", "c" }, np.dtypes.StringDType());
            var tsep = np.array(" ", np.dtypes.StringDType());
            var r = np.@char.partition(t, tsep);
            r.dtype.@char.Should().Be('T');
            r.shape.Should().Equal(2L, 3L);
            Texts(r).Should().Equal("a", " ", "b", "c", "", "");
            Texts(np.@char.rpartition(t, tsep)).Should().Equal("a", " ", "b", "", "", "c");

            Action str = () => np.@char.partition(t, " ");
            str.Should().Throw<TypeError>().Where(e => e.Message ==
                "ufunc '_partition' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, " +
                "<class 'numpy.dtypes.StrDType'>) -> (None, None, None)");
        }

        // ------------------------------------------------------------------ the chararray members: asarray's rule

        /// <summary>
        ///     The members are <c>asarray(f(self, ...))</c>: <c>np.char.asarray</c> compares a <c>U</c> result's width in
        ///     characters with its itemsize in bytes and COPIES it (an owning chararray), and views an <c>S</c> result (a
        ///     chararray over the computed array). The layout of the input is kept (an F chararray gives an F result).
        /// </summary>
        [TestMethod]
        public void Members_UResultOwnsItsData_SResultIsAView()
        {
            var u = CU().center(9);
            u.Should().BeOfType<NDCharArray>();
            u.flags.owndata.Should().BeTrue();
            Texts(u).Should().Equal("  ab cd  ", "    x    ", "  a b c  ");

            var s = CS().center(9);
            s.Should().BeOfType<NDCharArray>();
            s.flags.owndata.Should().BeFalse();
            s.dtype.ToString().Should().Be("|S9");
            Texts(s).Should().Equal("  ab cd  ", "    x    ", "  a b c  ");

            var rp = CU().rpartition(" ");
            rp.flags.owndata.Should().BeTrue();
            Texts(rp).Should().Equal("ab", " ", "cd", "", "", "x", "a b c", " ", "");
            var srp = CS().rpartition(new byte[] { 32 });
            srp.flags.owndata.Should().BeFalse();
            srp.dtype.ToString().Should().Be("|S5");

            var cf = (NDCharArray)np.asfortranarray(np.array(new[,] { { "ab", "c d" }, { "e f", "g" } })).view(typeof(NDCharArray));
            var fr = cf.center(5);
            fr.Shape.IsFContiguous.Should().BeTrue();
            fr.Shape.IsContiguous.Should().BeFalse();
            Texts(fr).Should().Equal("  ab ", " c d ", " e f ", "  g  ");
        }

        /// <summary>
        ///     The members read the STORED text: <c>c[0]</c> strips <c>'ab  '</c> to <c>'ab'</c>, but <c>center</c> /
        ///     <c>rjust</c> / <c>ljust</c> pad the four stored characters.
        /// </summary>
        [TestMethod]
        public void Members_PadTheStoredText_NotTheStrippedElement()
        {
            var c = np.@char.array(new[] { "ab  " });
            ((string)c[0].item()).Should().Be("ab");
            Texts(c.center(6, "*")).Should().Equal("*ab  *");
            Texts(c.rjust(6)).Should().Equal("  ab  ");
            Texts(c.ljust(6, "-")).Should().Equal("ab  --");
        }

        /// <summary>
        ///     zfill moves a sign in front of the zeros; expandtabs sizes a <c>U</c> result as NumPy does (a tab's columns plus
        ///     every other character's bytes: <c>&lt;U15</c> for two short texts) while an <c>S</c> result is a view.
        /// </summary>
        [TestMethod]
        public void Members_ZFillAndExpandTabs_AsNumPy()
        {
            var z = np.@char.array(new[] { "-7", "+3", "12" }).zfill(4);
            Texts(z).Should().Equal("-007", "+003", "0012");
            z.flags.owndata.Should().BeTrue();

            var tabs = np.@char.array(new[] { "a\tbc", "\tx" }).expandtabs(4);
            tabs.dtype.ToString().Should().Be("<U15");
            Texts(tabs).Should().Equal("a   bc", "    x");
            var stabs = np.@char.array(new object[] { System.Text.Encoding.Latin1.GetBytes("a\tbc") }).expandtabs();
            stabs.dtype.ToString().Should().Be("|S10");
            stabs.flags.owndata.Should().BeFalse();
            Texts(stabs).Should().Equal("a       bc");

            var widths = CU().center(np.array(new long[] { 3, 9, 12 }));
            widths.dtype.ToString().Should().Be("<U12");
            Texts(widths).Should().Equal("ab cd", "    x    ", "   a b c    ");
        }

        /// <summary>
        ///     The padding functions size their result as the longer of the widest width and the longest element: a width
        ///     below the element width keeps a full-width element whole (<c>np.char.center(['abcd', 'x'], 3)</c> is
        ///     <c>&lt;U4</c>, <c>'abcd'</c> unchanged). The size pass skips its length scan only when the widest width
        ///     reaches the element width; one unit lower, it must still scan, or the full-width element is cut.
        /// </summary>
        [TestMethod]
        public void Padding_WidthBelowTheElementWidth_KeepsTheLongestElement()
        {
            var a = np.array(new[] { "abcd", "x" });
            var center = np.@char.center(a, 3);
            center.dtype.ToString().Should().Be("<U4");
            Texts(center).Should().Equal("abcd", " x ");
            Texts(np.@char.ljust(a, 3)).Should().Equal("abcd", "x  ");
            Texts(np.@char.rjust(a, 3)).Should().Equal("abcd", "  x");
            Texts(np.@char.zfill(a, 3)).Should().Equal("abcd", "00x");
            Texts(np.@char.center(a, 4)).Should().Equal("abcd", " x  ");
            var wider = np.@char.center(a, 5);
            wider.dtype.ToString().Should().Be("<U5");
            Texts(wider).Should().Equal(" abcd", "  x  ");
            Texts(np.@char.center(a, np.array(new long[] { 3, 1 }))).Should().Equal("abcd", "x");

            var s = np.array(new object[] { System.Text.Encoding.Latin1.GetBytes("abcd"), System.Text.Encoding.Latin1.GetBytes("x") });
            var scenter = np.@char.center(s, 3);
            scenter.dtype.ToString().Should().Be("|S4");
            Texts(scenter).Should().Equal("abcd", " x ");
            Texts(np.@char.ljust(s, 2, new byte[] { (byte)'*' })).Should().Equal("abcd", "x*");
        }

        /// <summary>
        ///     A result past the buffer pool's cap (here 250,000 <c>&lt;U80</c> elements, 80 MB: NumPy sizes a <c>U</c>
        ///     expandtabs result in bytes) comes from fresh OS memory rather than a recycled buffer; its bytes, zero tails
        ///     included, must equal those of a 10-element result of the same pattern (a pooled buffer), tiled to the same
        ///     length.
        /// </summary>
        [TestMethod]
        public void ExpandTabs_PastThePoolCap_EqualsThePooledResult()
        {
            var pattern = new[] { "alpha\tbe\tgamma\tdelta", "x", "\t", "a\tb", "", "ab\tc\td\te", "q\nr\ts", "\r\t", "tab\t", "z" };
            var big = np.array(Enumerable.Range(0, 250_000).Select(i => pattern[i % 10]).ToArray());
            var r = np.@char.expandtabs(big);
            var small = np.@char.expandtabs(np.array(pattern));
            r.dtype.ToString().Should().Be("<U80");
            small.dtype.ToString().Should().Be("<U80");
            ((long)r.nbytes).Should().BeGreaterThan(global::NumSharp.Backends.Unmanaged.Pooling.SizeBucketedBufferPool.MaxPoolableBytes);
            var one = small.tobytes();
            var all = r.tobytes();
            for (int k = 0; k < 250_000; k += 9_999)
                all.AsSpan(k / 10 * one.Length, one.Length).SequenceEqual(one).Should().BeTrue($"block {k / 10}");
            all.AsSpan(all.Length - one.Length).SequenceEqual(one).Should().BeTrue("the last block");
        }

        // ------------------------------------------------------------------ the module functions: numpy.strings' rules

        /// <summary>
        ///     The re-exports ARE numpy.strings' functions: a chararray text gives a chararray VIEW of the result
        ///     (<c>chararray.__array_wrap__</c> over the <c>np.empty_like</c> out), a chararray in another position does not
        ///     (the out is <c>np.empty_like(a)</c>); C# values convert as Python's (a <c>byte[]</c> fill is bytes, cast to
        ///     text for a <c>U</c> operand).
        /// </summary>
        [TestMethod]
        public void Functions_FollowTheClassOfTheText()
        {
            var r = np.@char.center(CU(), 9);
            r.Should().BeOfType<NDCharArray>();
            r.flags.owndata.Should().BeFalse();

            var fill = (NDCharArray)np.array(new[] { "*" }).view(typeof(NDCharArray));
            var plain = np.@char.center(U(), 9, fill);
            plain.Should().NotBeOfType<NDCharArray>();
            Texts(plain).Should().Equal("**ab cd**", "****x****", "**a b c *");

            Texts(np.@char.ljust(new[] { "ab", "c" }, 4)).Should().Equal("ab  ", "c   ");
            var zeros = np.@char.rjust("ab", 5, new byte[] { 48 });
            zeros.ndim.Should().Be(0);
            Texts(zeros).Should().Equal("000ab");

            var t = np.array(new[] { "-5", "x" }, np.dtypes.StringDType());
            Texts(np.@char.zfill(t, 4)).Should().Equal("-005", "000x");
            Texts(np.@char.center(np.array(new[] { "ab" }, np.dtypes.StringDType()), 5, "*")).Should().Equal("**ab*");
        }

        /// <summary>NumPy's refusals, verbatim, through the members and the functions.</summary>
        [TestMethod]
        public void Refusals_AreNumPys()
        {
            var cu = CU();
            Action floatWidth = () => cu.center(2.5);
            floatWidth.Should().Throw<TypeError>().Where(e => e.Message == "unsupported type float64 for operand 'width'");
            Action longFill = () => cu.center(9, "ab");
            longFill.Should().Throw<TypeError>().Where(e => e.Message == "The fill character must be exactly one character long");
            Action textWidth = () => cu.zfill("3");
            textWidth.Should().Throw<TypeError>().Where(e => e.Message == "unsupported type <U1 for operand 'width'");
            Action floatTab = () => cu.expandtabs(2.0);
            floatTab.Should().Throw<ArgumentException>().Where(e => e.Message.StartsWith(
                "Cannot cast ufunc '_expandtabs_length' input 1 from dtype('float64') to dtype('int64') with casting rule 'same_kind'"));
            Action empty = () => ((NDCharArray)cu["0:0"]).center(4);
            empty.Should().Throw<ValueError>().Where(e => e.Message == "zero-size array to reduction operation maximum which has no identity");
            Action nonAscii = () => CS().center(9, "é");
            nonAscii.Should().Throw<UnicodeEncodeError>();
            Action emptySep = () => np.@char.partition(U(), "");
            emptySep.Should().Throw<ValueError>().Where(e => e.Message == "empty separator");
        }

        /// <summary>
        ///     The module's null convention: a required argument (a width, a separator) is an
        ///     <see cref="ArgumentNullException"/>; an optional one (a fill, a tab size) is NumPy's default.
        /// </summary>
        [TestMethod]
        public void Null_IsTheDefaultOfAnOptionalArgument_AndRefusedForARequiredOne()
        {
            var cu = CU();
            Action width = () => cu.center(null);
            width.Should().Throw<ArgumentNullException>();
            Action sep = () => np.@char.rpartition(U(), null);
            sep.Should().Throw<ArgumentNullException>();
            Texts(cu.center(7, null)).Should().Equal(Texts(cu.center(7, " ")));
            Texts(np.@char.expandtabs(np.array(new[] { "a\tb" }), null)).Should().Equal("a       b");
        }

        // ------------------------------------------------------------------ the length and search family: plain results

        /// <summary>
        ///     The search members return PLAIN int64 / bool arrays (<c>chararray.__array_wrap__</c> keeps the class only for
        ///     a text result) over the STORED text: <c>'ab  '</c> holds two trailing spaces that <c>c[0]</c> strips, and
        ///     <c>count(' ')</c> counts them. A 0-d chararray gives a 0-d array, not a scalar.
        /// </summary>
        [TestMethod]
        public void SearchMembers_ReturnPlainArrays_OverTheStoredText()
        {
            var c = np.@char.array(new[] { "ab  ", "cab", "b b " });
            var count = c.count("b");
            count.Should().NotBeOfType<NDCharArray>();
            count.dtype.ToString().Should().Be("int64");
            count.ToArray<long>().Should().Equal(1L, 1L, 2L);
            c.count(" ").ToArray<long>().Should().Equal(2L, 0L, 2L);
            c.find(" ").ToArray<long>().Should().Equal(2L, -1L, 1L);
            c.rfind(" ").ToArray<long>().Should().Equal(3L, -1L, 3L);
            c.find("b", 0, 2).ToArray<long>().Should().Equal(1L, -1L, 0L);
            c.rfind("b", end: -1).ToArray<long>().Should().Equal(1L, -1L, 2L);
            c.index("b", 1).ToArray<long>().Should().Equal(1L, 2L, 2L);
            c.rindex("b").ToArray<long>().Should().Equal(1L, 2L, 2L);
            var ends = c.endswith(" ");
            ends.Should().NotBeOfType<NDCharArray>();
            ends.dtype.ToString().Should().Be("bool");
            ends.ToArray<bool>().Should().Equal(true, false, true);
            c.startswith("b").ToArray<bool>().Should().Equal(false, false, true);
            c.startswith("b", 1).ToArray<bool>().Should().Equal(true, false, false);
            c.endswith("b", 0, 2).ToArray<bool>().Should().Equal(true, false, false);

            var s = np.@char.array(new object[]
            {
                System.Text.Encoding.Latin1.GetBytes("ab  "), System.Text.Encoding.Latin1.GetBytes("cab"),
                System.Text.Encoding.Latin1.GetBytes("b b "),
            });
            s.count(new byte[] { 98 }).ToArray<long>().Should().Equal(1L, 1L, 2L);

            var zero = (NDCharArray)((NDCharArray)c["0:1"]).reshape(Shape.Scalar);
            var one = zero.count("b");
            one.ndim.Should().Be(0);
            one.Should().NotBeOfType<NDCharArray>();
            one.ToArray<long>().Should().Equal(1L);
        }

        /// <summary>
        ///     <c>np.char.str_len</c> IS numpy.strings' ufunc: the stored length (trailing whitespace counts), a plain result
        ///     for a chararray, <c>out=</c> returned itself (any int64-castable dtype), <c>where=</c> leaving the masked-off
        ///     elements of the out, and a Python <c>str</c> giving a 0-d result.
        /// </summary>
        [TestMethod]
        public void StrLen_IsTheUfunc_OverTheStoredText()
        {
            var c = np.@char.array(new[] { "ab  ", "cab", "b b " });
            var lengths = np.@char.str_len(c);
            lengths.Should().NotBeOfType<NDCharArray>();
            lengths.ToArray<long>().Should().Equal(4L, 3L, 4L);
            var o = np.zeros(new Shape(3), np.int32);
            np.@char.str_len(c, o).Should().BeSameAs(o);
            o.ToArray<int>().Should().Equal(4, 3, 4);
            var masked = np.zeros(new Shape(3), np.int64);
            np.@char.str_len(c, masked, np.array(new[] { true, false, true }));
            masked.ToArray<long>().Should().Equal(4L, 0L, 4L);
            var scalar = np.@char.str_len("abc");
            scalar.ndim.Should().Be(0);
            scalar.ToArray<long>().Should().Equal(3L);
        }

        /// <summary>The search family's refusals, verbatim: no loop for a substring of the other class or for a number, the
        ///     start's same_kind cast, index's missing substring, str_len's int64-only loop.</summary>
        [TestMethod]
        public void Search_Refusals_AreNumPys()
        {
            var cu = CU();
            Action alien = () => cu.count(new byte[] { 98 });
            alien.Should().Throw<TypeError>().Where(e => e.Message ==
                "ufunc 'count' did not contain a loop with signature matching types (<class 'numpy.dtypes.StrDType'>, " +
                "<class 'numpy.dtypes.BytesDType'>, <class 'numpy.dtypes._PyLongDType'>, <class 'numpy.dtypes._PyLongDType'>) -> None");
            Action floatStart = () => cu.find("b", 1.5);
            floatStart.Should().Throw<ArgumentException>().Where(e => e.Message.StartsWith(
                "Cannot cast ufunc 'find' input 2 from dtype('float64') to dtype('int64') with casting rule 'same_kind'"));
            Action missing = () => cu.index("z");
            missing.Should().Throw<ValueError>().Where(e => e.Message == "substring not found");
            Action int32Loop = () => np.@char.str_len(U(), dtype: np.int32);
            int32Loop.Should().Throw<TypeError>().Where(e => e.Message ==
                "ufunc 'str_len' did not contain a loop with signature matching types <class 'numpy.dtypes.StrDType'> -> " +
                "<class 'numpy.dtypes.Int32DType'>");
            Action number = () => np.@char.str_len(5);
            number.Should().Throw<TypeError>().Where(e => e.Message ==
                "ufunc 'str_len' did not contain a loop with signature matching types <class 'numpy.dtypes.Int64DType'> -> None");
        }

        /// <summary>
        ///     A ONE-input text ufunc names a C# literal operand by its default DType in its no-loop refusal (NumPy's
        ///     <c>convert_ufunc_arguments</c> marks no operand of a one-input ufunc weak): <c>Int64DType</c>,
        ///     <c>UInt64DType</c> past int64, <c>Float64DType</c>, <c>Complex128DType</c>. A ufunc of more inputs names the
        ///     abstract <c>_PyLongDType</c>. The re-exports share the texts (<c>np.char.str_len</c> IS
        ///     <c>np.strings.str_len</c>). Every text was probed against NumPy 2.4.2.
        /// </summary>
        [TestMethod]
        public void UnaryTextUfuncs_NameALiteralByItsDefaultDType()
        {
            static string NoLoop(string ufunc, string cls, string output = "None")
                => $"ufunc '{ufunc}' did not contain a loop with signature matching types <class 'numpy.dtypes.{cls}'> -> {output}";
            static void Raises(Func<NDArray> call, string text)
            {
                Action act = () => call();
                act.Should().Throw<TypeError>().Where(e => e.Message == text);
            }

            Raises(() => np.strings.str_len(5), NoLoop("str_len", "Int64DType"));
            Raises(() => np.@char.str_len(5), NoLoop("str_len", "Int64DType"));
            Raises(() => np.strings.str_len(5.0), NoLoop("str_len", "Float64DType"));
            Raises(() => np.strings.str_len(new System.Numerics.Complex(0, 1)), NoLoop("str_len", "Complex128DType"));
            Raises(() => np.strings.isalpha(5), NoLoop("isalpha", "Int64DType"));
            Raises(() => np.strings.isdecimal(5.0), NoLoop("isdecimal", "Float64DType"));
            Raises(() => np.strings.isspace(9223372036854775808UL), NoLoop("isspace", "UInt64DType"));
            Raises(() => np.strings.lstrip(5), NoLoop("_lstrip_whitespace", "Int64DType"));
            Raises(() => np.strings.rstrip(new System.Numerics.Complex(0, 1)), NoLoop("_rstrip_whitespace", "Complex128DType"));
            Raises(() => np.strings.strip(5.5), NoLoop("_strip_whitespace", "Float64DType"));
            Raises(() => np.strings.str_len(5, dtype: np.int32), NoLoop("str_len", "Int64DType", "<class 'numpy.dtypes.Int32DType'>"));
            Raises(() => np.strings.isalpha(5, dtype: np.int32), NoLoop("isalpha", "Int64DType", "<class 'numpy.dtypes.Int32DType'>"));
            // Four inputs: the literal stays weak.
            Raises(() => np.strings.count(5, " "),
                "ufunc 'count' did not contain a loop with signature matching types (<class 'numpy.dtypes._PyLongDType'>, " +
                "<class 'numpy.dtypes.StrDType'>, <class 'numpy.dtypes._PyLongDType'>, <class 'numpy.dtypes._PyLongDType'>) -> None");
        }
    }
}
