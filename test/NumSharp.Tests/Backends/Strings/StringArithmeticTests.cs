using System;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     Strings phase 2, P2.3: text concatenation (<c>np.add</c> over two strings of one class, the <c>+</c> operator) and
    ///     repetition (<c>np.multiply</c> of a string and an integer into <c>out=</c>), and their <c>np.strings</c> wrappers.
    ///     Every expected value, descriptor, layout and message was probed against NumPy 2.4.2 (probe scripts of the session
    ///     that wrote them). The string oracle (section <c>strarith</c>) replays the full matrices; these tests pin one
    ///     representative of each rule, and the kernel tests check the IL kernels and the managed fallbacks against a naive
    ///     reference written from NumPy's <c>string_add</c> / <c>string_multiply</c> / <c>str_len</c> loops.
    /// </summary>
    [TestClass]
    public class StringArithmeticTests
    {
        /// <summary>A <c>U</c> array of the given descriptor.</summary>
        /// <param name="dtype">The descriptor (its width).</param>
        /// <param name="s">The values.</param>
        /// <returns>The array.</returns>
        private static NDArray U(string dtype, params string[] s) => np.array(s, dtype: dtype);

        /// <summary>An <c>S</c> array of the given width from Latin-1 strings (each char is one byte, so <c>\xff</c> is 0xFF).</summary>
        /// <param name="dtype">The <c>S</c> descriptor (its width).</param>
        /// <param name="s">The values, one char per byte.</param>
        /// <returns>The array.</returns>
        private static NDArray S(string dtype, params string[] s) =>
            np.array(s.Select(x => (object)x.Select(c => (byte)c).ToArray()).ToArray(), dtype: dtype);

        /// <summary>The first concatenation operand: <c>&lt;U3</c> with an embedded NUL, an empty string and a trailing NUL.</summary>
        /// <returns>The array.</returns>
        private static NDArray U3() => U("U3", "ab", "a\0b", "", "xyz", "a\0");

        /// <summary>The second concatenation operand: <c>&lt;U4</c> (a trailing NUL inside, an empty string).</summary>
        /// <returns>The array.</returns>
        private static NDArray U4() => U("U4", "c", "dd\0", "e", "", "f");

        /// <summary>The <c>S</c> twin of <see cref="U3"/> with high bytes (0xFF 0x80).</summary>
        /// <returns>The array.</returns>
        private static NDArray S3() => S("S3", "ab", "a\0b", "", "\xff\x80", "a\0");

        /// <summary>The <c>S</c> twin of <see cref="U4"/>.</summary>
        /// <returns>The array.</returns>
        private static NDArray S2() => S("S2", "c", "d\0", "", "\x01", "e");

        /// <summary>The 2-D operand of the layout tests: <c>[['a', 'bb'], ['ccc', '']]</c> as <c>&lt;U3</c>.</summary>
        /// <returns>The array.</returns>
        private static NDArray M() => np.array(new[,] { { "a", "bb" }, { "ccc", "" } }, dtype: "U3");

        /// <summary>
        ///     Every element of a string array in C order: a <c>U</c> element as its string, an <c>S</c> element decoded as
        ///     Latin-1 (one char per byte); trailing NULs are dropped as NumPy's element read drops them, embedded ones kept.
        /// </summary>
        /// <param name="a">The array (any rank, 0-d included).</param>
        /// <returns>The elements.</returns>
        private static string[] Text(NDArray a)
        {
            var flat = a.ndim == 0 ? a.reshape(1) : a.flatten();
            if (a.dtype.kind == 'U')
                return NDArray.AsStringArray(flat);
            return Enumerable.Range(0, (int)flat.size)
                             .Select(i => new string(((byte[])flat.item(i)).Select(b => (char)b).ToArray()))
                             .ToArray();
        }

        /// <summary>The code units of a <c>U</c> array's raw buffer, in C order (NUL units included).</summary>
        /// <param name="a">The array.</param>
        /// <returns>The units.</returns>
        private static uint[] Units(NDArray a)
        {
            var bytes = a.tobytes();
            return Enumerable.Range(0, bytes.Length / 4).Select(i => BitConverter.ToUInt32(bytes, i * 4)).ToArray();
        }

        /// <summary>A <c>U{width}</c> array over raw code units (units above U+10FFFF are reachable only through such views).</summary>
        /// <param name="width">The element width in characters.</param>
        /// <param name="units">Every element's units, <paramref name="width"/> per element.</param>
        /// <returns>The array over the pinned unit buffer.</returns>
        private static NDArray UnitArray(int width, params uint[] units) =>
            np.ndarray(new Shape(units.Length / width), $"U{width}", buffer: units);

        // ============================================================================================ concatenation

        /// <summary>
        ///     <c>np.add</c> over two <c>U</c> arrays: the result is <c>&lt;U7</c> (the sum of the widths); trailing NULs never
        ///     count and embedded ones do (<c>'a\0b' + 'dd\0'</c> is <c>'a\0bdd'</c>).
        /// </summary>
        [TestMethod]
        public void Add_U_WidthIsTheSum_TrailingNulsDropped_EmbeddedKept()
        {
            var r = np.add(U3(), U4());
            r.dtype.str.Should().Be("<U7");
            Text(r).Should().Equal("abc", "a\0bdd", "e", "xyz", "af");
            // Every unit past the two texts is zero (the loop zero-fills to the output width).
            Units(r).Skip(7).Take(7).Should().Equal(new uint[] { 'a', 0, 'b', 'd', 'd', 0, 0 });
        }

        /// <summary><c>np.add</c> over two <c>S</c> arrays copies bytes verbatim (high bytes included): <c>|S5</c>.</summary>
        [TestMethod]
        public void Add_S_CopiesTheBytes()
        {
            var r = np.add(S3(), S2());
            r.dtype.str.Should().Be("|S5");
            Text(r).Should().Equal("abc", "a\0bd", "", "\xff\x80\x01", "ae");
        }

        /// <summary>
        ///     A bytes / str mix and a string with a number have no loop: NumPy's two UFuncTypeError texts, verbatim (a NumPy
        ///     integer is named by its descriptor).
        /// </summary>
        [TestMethod]
        public void Add_MixedClassesAndNumbers_AreRefusedWithNumPysText()
        {
            Action mixed = () => np.add(U3(), S2());
            mixed.Should().Throw<TypeError>().WithMessage("ufunc 'add' cannot use operands with types dtype('<U3') and dtype('S2')");
            Action number = () => np.add(U3(), NDArray.Scalar(1L));
            number.Should().Throw<TypeError>().WithMessage(
                "ufunc 'add' did not contain a loop with signature matching types (dtype('<U3'), dtype('int64')) -> None");
            Action numberFirst = () => np.add(NDArray.Scalar(1L), U3());
            numberFirst.Should().Throw<TypeError>().WithMessage(
                "ufunc 'add' did not contain a loop with signature matching types (dtype('int64'), dtype('<U3')) -> None");
        }

        /// <summary>NumPy's string add promoter ignores <c>dtype=</c> for two strings of one class (any class requested).</summary>
        [TestMethod]
        public void Add_TwoStrings_TheDtypeArgumentIsIgnored()
        {
            foreach (DType requested in new DType[] { "U", np.float64, "S" })
            {
                var r = np.add(U3(), U4(), dtype: requested);
                r.dtype.str.Should().Be("<U7");
                Text(r).Should().Equal("abc", "a\0bdd", "e", "xyz", "af");
            }
        }

        /// <summary>Two 0-d operands give a 0-d result; an empty operand gives an empty result of the summed width.</summary>
        [TestMethod]
        public void Add_ZeroD_AndEmpty()
        {
            var r = np.add(NDArray.Scalar("ab"), NDArray.Scalar("c"));
            r.ndim.Should().Be(0);
            r.dtype.str.Should().Be("<U3");
            Text(r).Should().Equal("abc");

            var e = np.add(np.zeros(new Shape(0L), "U1"), U("U2", "ab"));
            e.shape.Should().Equal(0L);
            e.dtype.str.Should().Be("<U3");
        }

        /// <summary>
        ///     <c>out=</c> of the loop's descriptor is written and returned itself; a narrower or wider text <c>out=</c>
        ///     receives NumPy's output cast (truncated / zero-padded); an <c>S</c> result into a <c>U</c> out is ASCII-decoded.
        /// </summary>
        [TestMethod]
        public void Add_IntoOut_CastsTheLoopResult()
        {
            var exact = np.zeros(new Shape(5L), "U7");
            np.add(U3(), U4(), exact).Should().BeSameAs(exact);
            Text(exact).Should().Equal("abc", "a\0bdd", "e", "xyz", "af");

            var narrow = np.zeros(new Shape(5L), "U3");
            np.add(U3(), U4(), narrow).Should().BeSameAs(narrow);
            Text(narrow).Should().Equal("abc", "a\0b", "e", "xyz", "af");

            var wide = np.zeros(new Shape(5L), "U10");
            Text(np.add(U3(), U4(), wide)).Should().Equal("abc", "a\0bdd", "e", "xyz", "af");

            var decoded = np.zeros(new Shape(3L), "U5");
            Text(np.add(S3()["0:3"], S2()["0:3"], decoded)).Should().Equal("abc", "a\0bd", "");
        }

        /// <summary>
        ///     The output-cast refusals: a byte ≥ 0x80 cast into <c>U</c> (UnicodeDecodeError), a <c>U</c> result into an
        ///     <c>S</c> or numeric out (NumPy's same_kind text, raised as the house <see cref="ArgumentException"/>).
        /// </summary>
        [TestMethod]
        public void Add_IntoOut_CastRefusals()
        {
            Action highByte = () => np.add(S3(), S2(), np.zeros(new Shape(5L), "U5"));
            highByte.Should().Throw<UnicodeDecodeError>()
                    .WithMessage("'ascii' codec can't decode byte 0xff in position 0: ordinal not in range(128)");
            Action toBytes = () => np.add(U3(), U4(), np.zeros(new Shape(5L), "S7"));
            toBytes.Should().Throw<ArgumentException>()
                   .WithMessage("Cannot cast ufunc 'add' output from dtype('<U7') to dtype('S7') with casting rule 'same_kind'");
            Action toNumber = () => np.add(U3(), U4(), np.zeros(new Shape(5L)));
            toNumber.Should().Throw<ArgumentException>()
                    .WithMessage("Cannot cast ufunc 'add' output from dtype('<U7') to dtype('float64') with casting rule 'same_kind'");
        }

        /// <summary>A <c>where=</c> mask leaves the masked-off elements of <c>out=</c> untouched.</summary>
        [TestMethod]
        public void Add_Where_KeepsTheMaskedOffElements()
        {
            var o = U("U7", "k", "k", "k", "k", "k");
            np.add(U3(), U4(), o, np.array(new[] { true, false, true, false, true }));
            Text(o).Should().Equal("abc", "k", "e", "k", "af");
        }

        /// <summary>
        ///     A fresh result takes NumPy's ufunc output layout (an F pair gives an F result, a transposed or reversed
        ///     operand a C one), and broadcasting follows the usual rules.
        /// </summary>
        [TestMethod]
        public void Add_Layouts_AndBroadcasting()
        {
            var m = M();
            var cross = np.add(m, m.T);
            cross.dtype.str.Should().Be("<U6");
            Text(cross).Should().Equal("aa", "bbccc", "cccbb", "");
            cross.flags.c_contiguous.Should().BeTrue();

            var f = np.add(np.asfortranarray(m), np.asfortranarray(m));
            Text(f).Should().Equal("aa", "bbbb", "cccccc", "");
            f.flags.f_contiguous.Should().BeTrue();
            f.flags.c_contiguous.Should().BeFalse();

            var outer = np.add(np.array(new[,] { { "a" }, { "b" } }), np.array(new[] { "x", "yy", "z" }));
            outer.shape.Should().Equal(2L, 3L);
            outer.dtype.str.Should().Be("<U3");
            Text(outer).Should().Equal("ax", "ayy", "az", "bx", "byy", "bz");

            var reversed = np.add(m[":, ::-1"], NDArray.Scalar("q"));
            reversed.dtype.str.Should().Be("<U4");
            Text(reversed).Should().Equal("bbq", "aq", "q", "cccq");
            reversed.flags.c_contiguous.Should().BeTrue();

            Action mismatch = () => np.add(m, np.array(new[] { "a", "b", "c" }));
            mismatch.Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with shapes (2,2) (3,) ");
        }

        /// <summary>
        ///     The summed width must fit <c>int</c> bytes: NumPy's TypeError (it prints the length in BYTES for both
        ///     classes), raised from the descriptors alone.
        /// </summary>
        [TestMethod]
        public void Add_ResultTooLarge_IsNumPysTypeError()
        {
            Action huge = () => np.add(np.zeros(new Shape(0L), "S2147483647"), np.zeros(new Shape(0L), "S1"));
            huge.Should().Throw<TypeError>().WithMessage("addition result string of length 2147483648 is too large to store inside array.");
        }

        /// <summary>
        ///     <c>out=</c> overlapping an input across elements (a reversed narrow view of out's own buffer) is computed as if
        ///     from a copy (NDIter's COPY_IF_OVERLAP), and the out itself is returned.
        /// </summary>
        [TestMethod]
        public void Add_OutOverlappingAnInput_ReadsTheOriginal()
        {
            var o = U("U4", "abcd", "efgh", "ijkl");
            var x = o.view("U2")["::-2"];
            np.add(x, U("U2", "X", "YY", ""), o).Should().BeSameAs(o);
            Text(o).Should().Equal("klX", "ghYY", "cd");
        }

        /// <summary>Strided, reversed, F-ordered and broadcast outputs receive the concatenation at their own positions.</summary>
        [TestMethod]
        public void Add_IntoReversedFAndBroadcastOuts()
        {
            var b = U("U7", Enumerable.Repeat("z", 5).ToArray());
            np.add(U3(), U4(), b["::-1"]);
            Text(b).Should().Equal("af", "xyz", "e", "a\0bdd", "abc");

            var m = M();
            var f = np.add(m, m, np.asfortranarray(np.zeros(new Shape(2, 2), "U6")));
            Text(f).Should().Equal("aa", "bbbb", "cccccc", "");
            f.flags.f_contiguous.Should().BeTrue();

            var rows = np.add(U3(), U4(), np.zeros(new Shape(2, 5), "U7"));
            Text(rows).Should().Equal("abc", "a\0bdd", "e", "xyz", "af", "abc", "a\0bdd", "e", "xyz", "af");
        }

        /// <summary>
        ///     The <c>+</c> operator with a C# string reads it as NumPy's <c>str</c> (a 0-d <c>U</c>); a string on the left
        ///     stays the FIRST operand (Python's <c>__radd__</c> is <c>np.add(s, a)</c>); bytes plus str is refused.
        /// </summary>
        [TestMethod]
        public void PlusOperator_WithStrings()
        {
            var right = U3() + "xy";
            right.dtype.str.Should().Be("<U5");
            Text(right).Should().Equal("abxy", "a\0bxy", "xy", "xyzxy", "axy");

            var left = "p" + U3();
            left.dtype.str.Should().Be("<U4");
            Text(left).Should().Equal("pab", "pa\0b", "p", "pxyz", "pa");

            Text(U3() + U4()).Should().Equal("abc", "a\0bdd", "e", "xyz", "af");

            Action bytesPlusText = () => { var _ = S3() + "xy"; };
            bytesPlusText.Should().Throw<TypeError>().WithMessage("ufunc 'add' cannot use operands with types dtype('S3') and dtype('<U2')");
        }

        // ============================================================================================ repetition

        /// <summary>
        ///     The string multiply loop cannot size its result: without <c>out=</c> NumPy raises its TypeError, for the ufunc
        ///     and for both operator orders.
        /// </summary>
        [TestMethod]
        public void Multiply_WithoutOut_IsNumPysTypeError()
        {
            const string text = "The 'out' kwarg is necessary when using the string multiply ufunc directly. Use numpy.strings.multiply " +
                                "to multiply strings without specifying 'out'.";
            Action ufunc = () => np.multiply(U3(), 2);
            ufunc.Should().Throw<TypeError>().WithMessage(text);
            Action right = () => { var _ = U3() * 2; };
            right.Should().Throw<TypeError>().WithMessage(text);
            Action left = () => { var _ = 2 * U3(); };
            left.Should().Throw<TypeError>().WithMessage(text);
        }

        /// <summary>Two strings, a bool count and a float count have no multiply loop: NumPy's no-loop text.</summary>
        [TestMethod]
        public void Multiply_NoLoop_ForTextTextBoolAndFloat()
        {
            Action text = () => { var _ = U3() * "ab"; };
            text.Should().Throw<TypeError>().WithMessage(
                "ufunc 'multiply' did not contain a loop with signature matching types (dtype('<U3'), dtype('<U2')) -> None");
            Action flag = () => np.multiply(U3(), true, np.zeros(new Shape(5L), "U6"));
            flag.Should().Throw<TypeError>().WithMessage(
                "ufunc 'multiply' did not contain a loop with signature matching types (dtype('<U3'), dtype('bool')) -> None");
            Action real = () => np.multiply(U3(), 2.0, np.zeros(new Shape(5L), "U6"));
            real.Should().Throw<TypeError>().WithMessage(
                "ufunc 'multiply' did not contain a loop with signature matching types (dtype('<U3'), dtype('float64')) -> None");
        }

        /// <summary>
        ///     The repeat into <c>out=</c>: each text repeated, cut at the output width (a one-unit text is a fill); a count
        ///     below 1 gives an empty string; the text may be either operand; any integer count dtype works.
        /// </summary>
        [TestMethod]
        public void Multiply_IntoOut_RepeatsAndCuts()
        {
            var o = np.zeros(new Shape(5L), "U6");
            np.multiply(U3(), 2, o).Should().BeSameAs(o);
            Text(o).Should().Equal("abab", "a\0ba\0b", "", "xyzxyz", "aa");

            Text(np.multiply(2, U3(), np.zeros(new Shape(5L), "U6"))).Should().Equal("abab", "a\0ba\0b", "", "xyzxyz", "aa");
            Text(np.multiply(U3(), np.array(new long[] { 3, 2, 1, 0, -1 }), np.zeros(new Shape(5L), "U5")))
                .Should().Equal("ababa", "a\0ba", "", "", "");
            Text(np.multiply(U3(), NDArray.Scalar((byte)2), np.zeros(new Shape(5L), "U6")))
                .Should().Equal("abab", "a\0ba\0b", "", "xyzxyz", "aa");
        }

        /// <summary>
        ///     <c>S</c> repeats copy bytes verbatim; an <c>S</c> loop into a <c>U</c> out is ASCII-decoded; a <c>U</c> loop into
        ///     an <c>S</c> or numeric out is refused (the numeric out's loop descriptor is <c>&lt;U32</c>, NumPy's text).
        /// </summary>
        [TestMethod]
        public void Multiply_Bytes_AndTheOutputCast()
        {
            Text(np.multiply(S3(), 3, np.zeros(new Shape(5L), "S9")))
                .Should().Equal("ababab", "a\0ba\0ba\0b", "", "\xff\x80\xff\x80\xff\x80", "aaa");
            Text(np.multiply(S3()["0:3"], 3, np.zeros(new Shape(3L), "U9"))).Should().Equal("ababab", "a\0ba\0ba\0b", "");

            Action toBytes = () => np.multiply(U3(), 3, np.zeros(new Shape(5L), "S9"));
            toBytes.Should().Throw<ArgumentException>()
                   .WithMessage("Cannot cast ufunc 'multiply' output from dtype('<U9') to dtype('S9') with casting rule 'same_kind'");
            Action toNumber = () => np.multiply(U3(), 3, np.zeros(new Shape(5L)));
            toNumber.Should().Throw<ArgumentException>()
                    .WithMessage("Cannot cast ufunc 'multiply' output from dtype('<U32') to dtype('float64') with casting rule 'same_kind'");
        }

        /// <summary>
        ///     NumPy's promoter casts the count to int64: a uint64 count past 2^63 wraps to a negative count, which repeats
        ///     nothing.
        /// </summary>
        [TestMethod]
        public void Multiply_UInt64CountPast2To63_RepeatsNothing()
        {
            var o = U("U6", "k", "k", "k", "k", "k");
            np.multiply(U3(), NDArray.Scalar((1UL << 63) + 5), o);
            Text(o).Should().Equal("", "", "", "", "");
        }

        /// <summary>
        ///     A repetition past 2^63 - 1 units leaves THAT element untouched, writes every other one, and raises NumPy's
        ///     OverflowError after the loop (a one-unit text is a capped fill and never overflows).
        /// </summary>
        [TestMethod]
        public void Multiply_Overflow_LeavesTheElementAndRaisesAfterTheLoop()
        {
            var o = U("U6", "k", "k", "k", "k", "k");
            Action act = () => np.multiply(U3(), NDArray.Scalar(1L << 62), o);
            act.Should().Throw<OverflowException>().WithMessage("Overflow detected in string multiply");
            Text(o).Should().Equal("k", "k", "", "k", "aaaaaa");
        }

        /// <summary>
        ///     A <c>where=</c> mask leaves masked-off elements untouched — and a masked-off <c>S</c> element with a high byte
        ///     never reaches the ASCII decode of a <c>U</c> out.
        /// </summary>
        [TestMethod]
        public void Multiply_Where_KeepsTheMaskedOffElements()
        {
            var o = U("U6", "k", "k", "k", "k", "k");
            np.multiply(U3(), 2, o, np.array(new[] { true, false, true, false, true }));
            Text(o).Should().Equal("abab", "k", "", "k", "aa");

            var d = U("U2", "k", "k");
            np.multiply(S("S1", "\xff", "a"), 2, d, np.array(new[] { false, true }));
            Text(d).Should().Equal("k", "aa");
        }

        /// <summary>
        ///     <c>out=</c> overlapping the text across elements (a reversed narrow view of out's own buffer) is computed as if
        ///     from a copy, with and without a mask.
        /// </summary>
        [TestMethod]
        public void Multiply_OutOverlappingTheText_ReadsTheOriginal()
        {
            var o = U("U6", "abcdef", "ghijkl", "mnopqr");
            np.multiply(o.view("U3")["::-2"], 2, o).Should().BeSameAs(o);
            Text(o).Should().Equal("pqrpqr", "jkljkl", "defdef");

            var masked = U("U6", "abcdef", "ghijkl", "mnopqr");
            np.multiply(masked.view("U3")["::-2"], np.array(new long[] { 1, 2, 3 }), masked, np.array(new[] { true, false, true }));
            Text(masked).Should().Equal("pqr", "ghijkl", "defdef");
        }

        /// <summary>A strided out and a column-count broadcast into a 2-D out.</summary>
        [TestMethod]
        public void Multiply_IntoStridedAndBroadcastOuts()
        {
            var b = U("U6", Enumerable.Repeat("z", 10).ToArray());
            np.multiply(U3(), 2, b["::2"]);
            Text(b).Should().Equal("abab", "z", "a\0ba\0b", "z", "", "z", "xyzxyz", "z", "aa", "z");

            var grid = np.multiply(U3(), np.array(new long[,] { { 1 }, { 2 } }), np.zeros(new Shape(2, 5), "U6"));
            Text(grid).Should().Equal("ab", "a\0b", "", "xyz", "a", "abab", "a\0ba\0b", "", "xyzxyz", "aa");
        }

        /// <summary>
        ///     NumPy's checks before the loop, in its order and texts: the out's broadcast, a read-only out, a non-bool mask
        ///     (house exception types: <see cref="ArgumentException"/> for the cast / broadcast refusals).
        /// </summary>
        [TestMethod]
        public void ArgumentChecks_HappenBeforeTheLoop()
        {
            Action repeatShape = () => np.multiply(U3(), 2, np.zeros(new Shape(4L), "U6"));
            repeatShape.Should().Throw<ArgumentException>().WithMessage("operands could not be broadcast together with shapes (5,) () (4,) ");
            Action addShape = () => np.add(U3(), U3(), np.zeros(new Shape(4L), "U6"));
            addShape.Should().Throw<ArgumentException>().WithMessage("operands could not be broadcast together with shapes (5,) (5,) (4,) ");
            Action readOnly = () =>
            {
                var ro = np.zeros(new Shape(5L), "U6");
                ro.flags.writeable = false;
                np.multiply(U3(), 2, ro);
            };
            readOnly.Should().Throw<ValueError>().WithMessage("output array is read-only");
            Action intMask = () => np.add(U3(), U3(), np.zeros(new Shape(5L), "U6"), np.array(new long[] { 1, 0, 1, 0, 1 }));
            intMask.Should().Throw<ArgumentException>()
                   .WithMessage("Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'");
        }

        // ============================================================================================ np.strings

        /// <summary><c>np.strings.add</c> IS <c>np.add</c>; its string overloads read a C# string as NumPy's <c>str</c>.</summary>
        [TestMethod]
        public void StringsAdd_IsNpAdd_WithStringOverloads()
        {
            Text(np.strings.add(U3(), U4())).Should().Equal("abc", "a\0bdd", "e", "xyz", "af");
            Text(np.strings.add(U3(), "xy")).Should().Equal("abxy", "a\0bxy", "xy", "xyzxy", "axy");
            Text(np.strings.add("p", U3())).Should().Equal("pab", "pa\0b", "p", "pxyz", "pa");
            var two = np.strings.add("a", "b");
            two.ndim.Should().Be(0);
            two.dtype.str.Should().Be("<U2");
            Text(two).Should().Equal("ab");

            var o = U("U6", "k", "k", "k", "k");
            var u = U("U3", "ab", "c", "", "xyz");
            np.strings.add(u, u, o, np.array(new[] { true, false, true, false }));
            Text(o).Should().Equal("abab", "k", "", "k");
        }

        /// <summary>
        ///     <c>np.strings.multiply</c> sizes the result to the longest repetition (<c>max(len(a) * max(i, 0))</c>, at least
        ///     1): counts below 1 give empty strings, the class follows the text, a 0-d pair (or a C# string) gives a 0-d
        ///     result.
        /// </summary>
        [TestMethod]
        public void StringsMultiply_SizesTheResult()
        {
            var r = np.strings.multiply(U3(), 2);
            r.dtype.str.Should().Be("<U6");
            Text(r).Should().Equal("abab", "a\0ba\0b", "", "xyzxyz", "aa");

            var counts = np.strings.multiply(U3(), np.array(new long[] { 3, 2, 1, 0, -1 }));
            counts.dtype.str.Should().Be("<U6");
            Text(counts).Should().Equal("ababab", "a\0ba\0b", "", "", "");

            np.strings.multiply(U("U2", "a", "bb"), np.array(new long[] { 3, 2 })).dtype.str.Should().Be("<U4");

            var bytes = np.strings.multiply(S3(), 2);
            bytes.dtype.str.Should().Be("|S6");
            Text(bytes).Should().Equal("abab", "a\0ba\0b", "", "\xff\x80\xff\x80", "aa");

            var u = U("U3", "ab", "c", "", "xyz");
            np.strings.multiply(U3(), 0).dtype.str.Should().Be("<U1");
            Text(np.strings.multiply(u, -3)).Should().Equal("", "", "", "");
            np.strings.multiply(u, NDArray.Scalar((sbyte)3)).dtype.str.Should().Be("<U9");
            Text(np.strings.multiply(u, NDArray.Scalar(2u))).Should().Equal("abab", "cc", "", "xyzxyz");

            var zeroD = np.strings.multiply(NDArray.Scalar("ab"), 3);
            zeroD.ndim.Should().Be(0);
            zeroD.dtype.str.Should().Be("<U6");
            Text(zeroD).Should().Equal("ababab");
            Text(np.strings.multiply("ab", 3)).Should().Equal("ababab");
        }

        /// <summary>
        ///     The result is <c>np.empty_like(a, shape=…)</c>: it keeps <c>a</c>'s memory order when the broadcast keeps its
        ///     rank and shape (F stays F, a transpose is F, a reversed view is C); a broadcast count gives the broadcast shape.
        /// </summary>
        [TestMethod]
        public void StringsMultiply_Layout()
        {
            var m = M();
            var f = np.strings.multiply(np.asfortranarray(m), 2);
            Text(f).Should().Equal("aa", "bbbb", "cccccc", "");
            f.flags.f_contiguous.Should().BeTrue();
            f.flags.c_contiguous.Should().BeFalse();

            var t = np.strings.multiply(m.T, 2);
            Text(t).Should().Equal("aa", "cccccc", "bbbb", "");
            t.flags.f_contiguous.Should().BeTrue();

            var rev = np.strings.multiply(m[":, ::-1"], 2);
            Text(rev).Should().Equal("bbbb", "aa", "", "cccccc");
            rev.flags.c_contiguous.Should().BeTrue();

            var grid = np.strings.multiply(U("U2", "a", "bb"), np.array(new long[,] { { 1 }, { 3 } }));
            grid.shape.Should().Equal(2L, 2L);
            grid.dtype.str.Should().Be("<U6");
            Text(grid).Should().Equal("a", "bb", "aaa", "bbbbbb");

            var fRow = np.strings.multiply(np.asfortranarray(m), np.array(new long[] { 1, 2 }));
            fRow.dtype.str.Should().Be("<U4");
            Text(fRow).Should().Equal("a", "bbbb", "ccc", "");
            fRow.flags.f_contiguous.Should().BeTrue();
        }

        /// <summary>
        ///     <c>np.strings.multiply</c>'s checks in NumPy's order and texts: the count's dtype first, then the text, then the
        ///     float64 overflow test, then the descriptor NumPy formats from the largest size (a uint64 count makes it
        ///     <c>'U6.0'</c>, which NumPy 2.4.2 cannot parse), then the empty-result reduction.
        /// </summary>
        [TestMethod]
        public void StringsMultiply_Errors()
        {
            var u = U("U3", "ab", "c", "", "xyz");
            Action real = () => np.strings.multiply(U3(), 2.0);
            real.Should().Throw<TypeError>().WithMessage("unsupported type float64 for operand 'i'");
            Action flag = () => np.strings.multiply(U3(), true);
            flag.Should().Throw<TypeError>().WithMessage("unsupported type bool for operand 'i'");
            Action number = () => np.strings.multiply(np.array(new long[] { 1 }), 2);
            number.Should().Throw<TypeError>().WithMessage(
                "ufunc 'str_len' did not contain a loop with signature matching types <class 'numpy.dtypes.Int64DType'> -> None");
            Action overflow = () => np.strings.multiply(U3(), NDArray.Scalar(1L << 62));
            overflow.Should().Throw<OverflowException>().WithMessage("Overflow encountered in string multiply");
            Action wrapped = () => np.strings.multiply(u, NDArray.Scalar((1UL << 63) + 1));
            wrapped.Should().Throw<OverflowException>().WithMessage("Overflow encountered in string multiply");
            Action unsigned = () => np.strings.multiply(u, NDArray.Scalar(2UL));
            unsigned.Should().Throw<TypeError>().WithMessage("data type 'U6.0' not understood");
            Action tooWide = () => np.strings.multiply(u, NDArray.Scalar(3074457345618258602L));
            tooWide.Should().Throw<TypeError>().WithMessage("data type 'U9223372036854775806' not understood");
            Action empty = () => np.strings.multiply(np.zeros(new Shape(0L), "U3"), 2);
            empty.Should().Throw<ValueError>().WithMessage("zero-size array to reduction operation maximum which has no identity");
            Action shapes = () => np.strings.multiply(u, np.array(new long[] { 1, 2 }));
            shapes.Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with shapes (4,) (2,) ");
        }

        // ============================================================================================ special code units

        /// <summary>
        ///     Units above U+10FFFF (reachable through views) are copied like any nonzero unit by both loops; a leading NUL
        ///     unit before a nonzero one is embedded, so it is kept.
        /// </summary>
        [TestMethod]
        public void UnitsAboveTheLastCodePoint_PassThrough()
        {
            var big = UnitArray(2, 0x110000, 0x41, 0, 0x10FFFF);
            Units(np.add(big, big)).Should().Equal(0x110000u, 0x41u, 0x110000u, 0x41u, 0u, 0x10FFFFu, 0u, 0x10FFFFu);
            Units(np.multiply(big, 2, np.zeros(new Shape(2L), "U4")))
                .Should().Equal(0x110000u, 0x41u, 0x110000u, 0x41u, 0u, 0x10FFFFu, 0u, 0x10FFFFu);
        }

        // ============================================================================================ the managed route end to end

        /// <summary>
        ///     The managed kernels (the route on hosts without dynamic code) give the same bytes as the IL kernels through
        ///     every driver path of the two text loops: the trivial loop, the iterator over F / transposed / reversed /
        ///     broadcast operands, a mask, and a cast into <c>out=</c>.
        /// </summary>
        /// <remarks>
        ///     <c>np.strings.multiply</c> is left out: its size pass composes NUMERIC ufuncs (<c>np.maximum</c>,
        ///     <c>np.true_divide</c>), which have no route with the IL generator switched off; its text loop is the
        ///     <c>np.multiply</c> route checked here.
        /// </remarks>
        [TestMethod]
        public void ManagedRoute_MatchesTheILRoute()
        {
            var calls = new Func<NDArray>[]
            {
                () => np.add(U3(), U4()),
                () => np.add(S3(), S2()),
                () => np.add(M(), M().T),
                () => np.add(np.asfortranarray(M()), np.asfortranarray(M())),
                () => np.add(M()[":, ::-1"], NDArray.Scalar("q")),
                () => np.add(U3(), U4(), U("U7", "k", "k", "k", "k", "k"), np.array(new[] { true, false, true, false, true })),
                () => np.add(U3(), U4(), np.zeros(new Shape(5L), "U3")),
                () => np.multiply(U3(), np.array(new long[] { 3, 2, 1, 0, -1 }), np.zeros(new Shape(5L), "U5")),
                () => np.multiply(S3(), 3, np.zeros(new Shape(5L), "S9")),
                () => np.multiply(U3(), 2, U("U6", "k", "k", "k", "k", "k"), np.array(new[] { true, false, true, false, true })),
                () => np.multiply(np.asfortranarray(M()), np.array(new long[] { 1, 2 }), np.asfortranarray(np.zeros(new Shape(2, 2), "U4"))),
            };
            var withIL = calls.Select(c => c().tobytes()).ToArray();
            DirectILKernelGenerator.Enabled = false;
            try
            {
                for (int i = 0; i < calls.Length; i++)
                    calls[i]().tobytes().Should().Equal(withIL[i], $"call {i} must not depend on the kernel route");
            }
            finally
            {
                DirectILKernelGenerator.Enabled = true;
            }
        }

        // ============================================================================================ the kernels against a naive reference

        /// <summary>
        ///     The concatenation kernels — IL (when dynamic code is available) and managed — against a naive reference over
        ///     both classes, widths across every chunk-plan boundary, and the four broadcast patterns (stride 0 per input).
        ///     The output buffer starts as a sentinel and is compared WHOLE, so a store outside an element's
        ///     <c>widthA + widthB</c> bytes (gaps, tail) fails too.
        /// </summary>
        [TestMethod]
        public unsafe void ConcatKernels_MatchTheNaiveReference()
        {
            var rng = new System.Random(20261009);
            int[] widths1 = { 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 257 };
            int[] widths4 = { 4, 8, 12, 28, 32, 36, 64, 68, 128, 132, 260, 520 };
            const int n = 19;
            void** p = stackalloc void*[3];
            long* st = stackalloc long[3];
            long checkedElements = 0;
            foreach (int unit in new[] { 1, 4 })
            foreach (int wa in unit == 1 ? widths1 : widths4)
            foreach (int wb in unit == 1 ? widths1 : widths4)
            {
                var kernels = new[] { ILKernelGenerator.GetTextConcatKernel(unit, wa, wb), NDStringArithmetic.ManagedConcatKernel(unit, wa, wb) };
                for (int pattern = 0; pattern < 4; pattern++)
                {
                    long sa = (pattern & 1) != 0 ? 0 : wa + unit, sb = (pattern & 2) != 0 ? 0 : wb + 2 * unit, so = wa + wb + unit;
                    var a = new byte[sa * (n - 1) + wa];
                    var b = new byte[sb * (n - 1) + wb];
                    FillText(rng, a, sa == 0 ? 1 : n, sa, wa, unit);
                    FillText(rng, b, sb == 0 ? 1 : n, sb, wb, unit);
                    var expected = Sentinel(so * n + 16);
                    for (int i = 0; i < n; i++)
                        NaiveConcat(a, i * sa, wa, b, i * sb, wb, unit, expected, i * so);
                    foreach (var kernel in kernels)
                    {
                        if (kernel is null)
                            continue;
                        var got = Sentinel(so * n + 16);
                        fixed (byte* pa = a, pb = b, po = got)
                        {
                            p[0] = pa; p[1] = pb; p[2] = po;
                            st[0] = sa; st[1] = sb; st[2] = so;
                            kernel(p, st, n, null);
                        }
                        got.Should().Equal(expected, $"unit {unit}, widths {wa}+{wb}, pattern {pattern}");
                        checkedElements += n;
                    }
                }
            }
            checkedElements.Should().BeGreaterThan(0);
        }

        /// <summary>The length kernels (IL and managed) against the naive length, int64 units, over widths and strides.</summary>
        [TestMethod]
        public unsafe void LengthKernels_MatchTheNaiveReference()
        {
            var rng = new System.Random(11);
            const int n = 23;
            void** p = stackalloc void*[2];
            long* st = stackalloc long[2];
            foreach (int unit in new[] { 1, 4 })
            foreach (int w in unit == 1 ? new[] { 1, 2, 7, 8, 9, 16, 17, 31, 32, 33, 64, 300 } : new[] { 4, 8, 12, 32, 36, 64, 300 })
            {
                long s = w + unit;
                var data = new byte[s * n];
                FillText(rng, data, n, s, w, unit);
                var expected = Enumerable.Range(0, n).Select(i => (long)NaiveLength(data, i * s, w, unit)).ToArray();
                foreach (var kernel in new[] { ILKernelGenerator.GetTextLengthKernel(unit, w), NDStringArithmetic.ManagedLengthKernel(unit, w) })
                {
                    if (kernel is null)
                        continue;
                    var got = new long[n];
                    fixed (byte* pd = data)
                    fixed (long* po = got)
                    {
                        p[0] = pd; p[1] = po;
                        st[0] = s; st[1] = sizeof(long);
                        kernel(p, st, n, null);
                    }
                    got.Should().Equal(expected, $"unit {unit}, width {w}");
                }
            }
        }

        /// <summary>
        ///     The repetition kernels (IL and managed, both text slots) against a naive repeat over widths, output widths
        ///     shorter and longer than the repetition, counts from negative through the overflow boundary of each length, and
        ///     a broadcast count (stride 0). The widths straddle the NARROW kernel's limits (text
        ///     <see cref="ILKernelGenerator.NarrowRepeatMaxWidth"/>, output <see cref="ILKernelGenerator.NarrowRepeatMaxOutWidth"/>
        ///     bytes), so both IL forms are checked. An overflowing element must keep its sentinel bytes, and the flag must be
        ///     set exactly when one overflowed.
        /// </summary>
        [TestMethod]
        public unsafe void RepeatKernels_MatchTheNaiveReference()
        {
            var rng = new System.Random(5);
            const int n = 31;
            void** p = stackalloc void*[3];
            long* st = stackalloc long[3];
            int overflowCalls = 0, cleanCalls = 0;
            foreach (int unit in new[] { 1, 4 })
            foreach (int w in unit == 1 ? new[] { 1, 2, 3, 8, 17, 33, 128, 129 } : new[] { 4, 8, 12, 36, 68, 128, 132 })
            foreach (int outW in unit == 1 ? new[] { 1, 5, 8, 12, 16, 64, 512, 513, 600 } : new[] { 4, 8, 12, 20, 64, 256, 512, 516, 2400 })
            foreach (int textIndex in new[] { 0, 1 })
            foreach (bool broadcastCount in new[] { false, true })
            for (int trial = 0; trial < 2; trial++)
            {
                long sText = w + unit, sOut = outW + unit, sReps = broadcastCount ? 0 : sizeof(long);
                var text = new byte[sText * n];
                FillText(rng, text, n, sText, w, unit);
                var reps = new long[broadcastCount ? 1 : n];
                for (int i = 0; i < reps.Length; i++)
                    reps[i] = RepeatCount(rng, NaiveLength(text, i * sText, w, unit), trial);
                var expected = Sentinel(sOut * n + 16);
                bool expectOverflow = false;
                for (int i = 0; i < n; i++)
                    expectOverflow |= NaiveRepeat(text, i * sText, w, unit, reps[broadcastCount ? 0 : i], expected, i * sOut, outW);
                if (expectOverflow) overflowCalls++; else cleanCalls++;
                var kernels = new[]
                {
                    ILKernelGenerator.GetTextRepeatKernel(unit, w, outW, textIndex),
                    NDStringArithmetic.ManagedRepeatKernel(unit, w, outW, textIndex),
                };
                foreach (var kernel in kernels)
                {
                    if (kernel is null)
                        continue;
                    var got = Sentinel(sOut * n + 16);
                    int flag = 0;
                    fixed (byte* pt = text, po = got)
                    fixed (long* pr = reps)
                    {
                        p[textIndex] = pt; p[1 - textIndex] = pr; p[2] = po;
                        st[textIndex] = sText; st[1 - textIndex] = sReps; st[2] = sOut;
                        kernel(p, st, n, &flag);
                    }
                    got.Should().Equal(expected, $"unit {unit}, width {w}, out {outW}, slot {textIndex}");
                    flag.Should().Be(expectOverflow ? 1 : 0);
                }
            }
            // Both outcomes were reached (a vacuous comparison would pass with every count small).
            overflowCalls.Should().BeGreaterThan(0);
            cleanCalls.Should().BeGreaterThan(0);
        }

        // ---------------------------------------------------------------- the naive reference (NumPy's loops, unit by unit)

        /// <summary>A buffer filled with the sentinel byte 0xCC (bytes a kernel must not touch keep it).</summary>
        /// <param name="length">The length in bytes.</param>
        /// <returns>The buffer.</returns>
        private static byte[] Sentinel(long length)
        {
            var b = new byte[length];
            Array.Fill(b, (byte)0xCC);
            return b;
        }

        /// <summary>
        ///     Fills <paramref name="count"/> elements of a text buffer so the interesting cases dominate: empty elements,
        ///     full-width ones, trailing NULs, embedded NULs, high bytes (S) and units above U+10FFFF (U).
        /// </summary>
        /// <param name="rng">The random source.</param>
        /// <param name="buf">The buffer.</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="stride">The element stride in bytes.</param>
        /// <param name="width">The element width in bytes.</param>
        /// <param name="unit">1 for S, 4 for U.</param>
        private static void FillText(System.Random rng, byte[] buf, long count, long stride, int width, int unit)
        {
            int units = width / unit;
            for (long i = 0; i < count; i++)
            {
                int kind = rng.Next(6);
                int length = kind switch { 0 => 0, 1 => units, _ => rng.Next(units + 1) };
                for (int u = 0; u < length; u++)
                {
                    // One unit in five is NUL (embedded unless it is the last), the rest random, with special units.
                    uint value = rng.Next(5) == 0 ? 0u
                        : unit == 1 ? (uint)rng.Next(1, 256)
                        : rng.Next(8) == 0 ? 0x110000u + (uint)rng.Next(16) : (uint)rng.Next(1, 0x110000);
                    WriteUnit(buf, i * stride + u * unit, unit, value);
                }
            }
        }

        /// <summary>Writes one code unit little-endian.</summary>
        /// <param name="buf">The buffer.</param>
        /// <param name="offset">The byte offset.</param>
        /// <param name="unit">1 or 4.</param>
        /// <param name="value">The unit.</param>
        private static void WriteUnit(byte[] buf, long offset, int unit, uint value)
        {
            if (unit == 1)
                buf[offset] = (byte)value;
            else
                BitConverter.GetBytes(value).CopyTo(buf, offset);
        }

        /// <summary>NumPy's <c>num_codepoints</c>: the index of the last nonzero code unit plus one.</summary>
        /// <param name="buf">The buffer.</param>
        /// <param name="offset">The element's byte offset.</param>
        /// <param name="width">The element width in bytes.</param>
        /// <param name="unit">1 or 4.</param>
        /// <returns>The length in units.</returns>
        private static int NaiveLength(byte[] buf, long offset, int width, int unit)
        {
            for (int u = width / unit - 1; u >= 0; u--)
                for (int b = 0; b < unit; b++)
                    if (buf[offset + u * unit + b] != 0)
                        return u + 1;
            return 0;
        }

        /// <summary>NumPy's <c>string_add</c> for one element: A's text, then B's, zero-filled to <c>wa + wb</c> bytes.</summary>
        /// <param name="a">A's buffer.</param>
        /// <param name="ao">A's element offset.</param>
        /// <param name="wa">A's width in bytes.</param>
        /// <param name="b">B's buffer.</param>
        /// <param name="bo">B's element offset.</param>
        /// <param name="wb">B's width in bytes.</param>
        /// <param name="unit">1 or 4.</param>
        /// <param name="o">The output buffer.</param>
        /// <param name="oo">The output element offset.</param>
        private static void NaiveConcat(byte[] a, long ao, int wa, byte[] b, long bo, int wb, int unit, byte[] o, long oo)
        {
            int la = NaiveLength(a, ao, wa, unit) * unit, lb = NaiveLength(b, bo, wb, unit) * unit;
            for (int k = 0; k < wa + wb; k++)
                o[oo + k] = k < la ? a[ao + k] : k < la + lb ? b[bo + k - la] : (byte)0;
        }

        /// <summary>
        ///     NumPy's <c>string_multiply</c> for one element, unit by unit: an empty text or a count below 1 writes an empty
        ///     element; a one-unit text fills <c>min(reps, outUnits)</c> units; a longer one overflows when <c>reps * len</c>
        ///     exceeds 2^63 - 1 (the element untouched), else is written cut at the output width.
        /// </summary>
        /// <param name="t">The text buffer.</param>
        /// <param name="to">The text element offset.</param>
        /// <param name="w">The text width in bytes.</param>
        /// <param name="unit">1 or 4.</param>
        /// <param name="reps">The count.</param>
        /// <param name="o">The output buffer.</param>
        /// <param name="oo">The output element offset.</param>
        /// <param name="outW">The output width in bytes.</param>
        /// <returns>True when the element overflowed.</returns>
        private static bool NaiveRepeat(byte[] t, long to, int w, int unit, long reps, byte[] o, long oo, int outW)
        {
            int len = NaiveLength(t, to, w, unit);
            int outUnits = outW / unit;
            if (len > 1 && reps >= 1 && reps > long.MaxValue / len)
                return true;
            long written = reps < 1 || len == 0 ? 0 : len == 1 ? System.Math.Min(reps, outUnits) : System.Math.Min(reps * len, outUnits);
            for (int u = 0; u < outUnits; u++)
                for (int b = 0; b < unit; b++)
                    o[oo + u * unit + b] = u < written ? t[to + (u % System.Math.Max(len, 1)) * unit + b] : (byte)0;
            return false;
        }

        /// <summary>
        ///     A repeat count for an element of the given length: mostly small (negative, zero, one, a few, many), and on the
        ///     second trial often at the overflow boundary of the length (just below it, at it, just past it).
        /// </summary>
        /// <param name="rng">The random source.</param>
        /// <param name="length">The element's length in units.</param>
        /// <param name="trial">0 for ordinary counts, 1 to include boundary counts.</param>
        /// <returns>The count.</returns>
        private static long RepeatCount(System.Random rng, int length, int trial)
        {
            if (trial == 1 && length > 1 && rng.Next(3) == 0)
                return long.MaxValue / length + rng.Next(-1, 2);
            return rng.Next(9) switch
            {
                0 => -rng.Next(1, 5),
                1 => 0,
                2 => 1,
                3 => long.MinValue,
                4 => rng.Next(1000, 100000),
                _ => rng.Next(2, 9),
            };
        }
    }
}
