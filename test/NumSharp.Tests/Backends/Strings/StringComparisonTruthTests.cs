using System;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     Strings phase 2, P2.1: the six comparison ufuncs over two fixed-width string arrays of one class, the C# string
    ///     operators, and the truth value of a string (its two NumPy routes). Every expected value was probed against NumPy
    ///     2.4.2 (a probe script of the session that wrote them); the string oracle replays the full matrices, these
    ///     tests pin one representative of each rule, and the kernel tests check the IL kernels against the managed
    ///     reference (NumPy's <c>Buffer::strcmp</c> ported) over widths, broadcast patterns and special code units.
    /// </summary>
    [TestClass]
    public class StringComparisonTruthTests
    {
        /// <summary>The first comparison operand: <c>&lt;U3</c> with a prefix pair, an embedded NUL, an empty string and a one-unit difference.</summary>
        /// <returns>The array.</returns>
        private static NDArray A() => np.array(new[] { "ab", "a", "a\0b", "", "b", "abc" }, dtype: "U3");
        /// <summary>The second comparison operand: <c>&lt;U4</c>, so every pair also compares across widths (trailing NULs, a nonzero tail).</summary>
        /// <returns>The array.</returns>
        private static NDArray B() => np.array(new[] { "a", "ab", "a", "\0", "a\0", "abd" }, dtype: "U4");
        /// <summary>An <c>S</c> array of the given width from Latin-1 strings (each char is one byte, so <c>\x80</c> is the byte 0x80).</summary>
        /// <param name="dtype">The <c>S</c> dtype (its width).</param>
        /// <param name="s">The values, one char per byte.</param>
        /// <returns>The array.</returns>
        private static NDArray Bytes(string dtype, params string[] s) =>
            np.array(s.Select(x => (object)x.Select(c => (byte)c).ToArray()).ToArray(), dtype: dtype);

        /// <summary>A U1 array over raw code units (units above U+10FFFF are reachable only through such views).</summary>
        /// <param name="units">The code units, one per element.</param>
        /// <returns>A U1 array over the pinned unit buffer.</returns>
        private static NDArray Units(params uint[] units) => np.ndarray(new Shape(units.Length), "U1", buffer: units);

        /// <summary>A 1-D <c>U{width}</c> array over raw code units, one row of <paramref name="width"/> units per element.</summary>
        /// <param name="width">The element width in characters.</param>
        /// <param name="rows">The elements' units (each row exactly <paramref name="width"/> long).</param>
        /// <returns>The array over the pinned unit buffer.</returns>
        private static NDArray UnitRows(int width, params uint[][] rows) =>
            np.ndarray(new Shape(rows.Length), $"U{width}", buffer: rows.SelectMany(r => r).ToArray());

        // ---------------------------------------------------------------- the six comparisons

        /// <summary>The six comparisons over <c>U</c> of two widths: trailing NULs never count, embedded ones do, a nonzero tail unit decides.</summary>
        [TestMethod]
        public void Comparisons_U_DifferentWidths_TrailingAndEmbeddedNuls()
        {
            // 'a' == 'a\0' (trailing NULs never count), 'a\0b' > 'a' (an embedded NUL does), '' == '\0'.
            np.equal(A(), B()).ToArray<bool>().Should().Equal(false, false, false, true, false, false);
            np.not_equal(A(), B()).ToArray<bool>().Should().Equal(true, true, true, false, true, true);
            np.less(A(), B()).ToArray<bool>().Should().Equal(false, true, false, false, false, true);
            np.less_equal(A(), B()).ToArray<bool>().Should().Equal(false, true, false, true, false, true);
            np.greater(A(), B()).ToArray<bool>().Should().Equal(true, false, true, false, true, false);
            np.greater_equal(A(), B()).ToArray<bool>().Should().Equal(true, false, true, true, true, false);
        }

        /// <summary><c>S</c> compares bytes UNSIGNED (NumPy sign-extends a char to npy_ucs4, which keeps that order).</summary>
        [TestMethod]
        public void Comparisons_S_AreUnsignedBytes()
        {
            // b'\x80' > b'\x7f': NumPy compares the bytes unsigned; b'zz' < b'zz\x01' (a nonzero tail byte).
            var s1 = Bytes("S2", "\x80", "\x7f", "a\0", "", "zz");
            var s2 = Bytes("S3", "\x7f", "\x80", "a", "\0\0\0", "zz\x01");
            np.less(s1, s2).ToArray<bool>().Should().Equal(false, true, false, false, true);
            np.equal(s1, s2).ToArray<bool>().Should().Equal(false, false, true, true, false);
        }

        /// <summary>Units above U+10FFFF (only reachable through views) compare as unsigned 32-bit values, without an error.</summary>
        [TestMethod]
        public void Comparisons_U_UnitsAboveTheLastCodePoint_CompareUnsigned()
        {
            var big = Units(0x110000, 0x41, 0xFFFFFFFF);
            var other = Units(0x41, 0x110000, 0x10FFFF);
            np.less(big, other).ToArray<bool>().Should().Equal(false, true, false);
            np.greater_equal(big, other).ToArray<bool>().Should().Equal(true, false, true);
        }

        /// <summary>F, transposed, reversed, strided, column-broadcast, 5-D and 0-d operands: NumPy's values and its result layout.</summary>
        [TestMethod]
        public void Comparisons_FollowNumPysLayouts()
        {
            var m = np.array(new[,] { { "b", "a", "c" }, { "a", "c", "b" } });
            var f = np.asfortranarray(m);
            var fl = np.less(f, np.asfortranarray(np.array(new[,] { { "a", "b", "b" }, { "b", "b", "b" } })));
            fl.ToArray<bool>().Should().Equal(false, true, false, true, false, false);
            fl.flags.f_contiguous.Should().BeTrue();   // an F pair gives an F result (NumPy's ufunc output layout)
            fl.flags.c_contiguous.Should().BeFalse();
            var t = np.greater(m.T, np.array(new[] { "b", "a" }));
            t.ToArray<bool>().Should().Equal(false, false, false, true, true, true);
            t.flags.f_contiguous.Should().BeTrue();
            np.equal(m[":, ::-1"], m).ToArray<bool>().Should().Equal(false, true, false, false, true, false);
            np.less(np.array(new[] { "a", "x", "c", "y", "b" })["::2"], np.array("b")).ToArray<bool>().Should().Equal(true, false, false);
            var col = np.less_equal(np.array(new[,] { { "a" }, { "c" } }), np.array(new[] { "b", "c", "a" }));
            col.ToArray<bool>().Should().Equal(true, true, true, false, true, false);
            col.flags.c_contiguous.Should().BeTrue();
            var five = np.equal(np.array(new[] { "a", "b" }).reshape(1, 1, 2, 1, 1), np.array(new[] { "a", "a" }).reshape(1, 1, 1, 1, 2));
            five.shape.Should().Equal(new long[] { 1, 1, 2, 1, 2 });
            five.ToArray<bool>().Should().Equal(true, true, false, false);
            var zero = np.equal(np.array("ab"), np.array("ab"));
            zero.ndim.Should().Be(0);
            zero.GetBoolean().Should().BeTrue();
        }

        /// <summary><c>out=</c> of another dtype receives 0 / 1 and is returned itself; <c>where=</c> keeps the masked-off slots; a non-bool <c>dtype=</c> has no loop; an empty broadcast is an empty bool result.</summary>
        [TestMethod]
        public void Comparisons_Out_Where_Dtype_Empty()
        {
            var o = np.zeros(new Shape(6L), NPTypeCode.SByte);
            var r = np.less(A(), B(), o);
            ReferenceEquals(r, o).Should().BeTrue();
            o.ToArray<sbyte>().Should().Equal(0, 1, 0, 0, 0, 1);
            var keep = np.ones(new Shape(6L), NPTypeCode.Boolean);
            np.less(A(), B(), keep, np.array(new[] { true, false, true, false, true, false }));
            keep.ToArray<bool>().Should().Equal(false, true, false, true, false, true);
            Action loop = () => np.less(A(), B(), dtype: "i1");
            loop.Should().Throw<IncorrectTypeException>()
                .WithMessage("No loop matching the specified signature and casting was found for ufunc less");
            var empty = np.less(np.zeros(new Shape(0L, 2L), "U2"), np.zeros(new Shape(2L), "U3"));
            empty.shape.Should().Equal(new long[] { 0, 2 });
            empty.dtype.Should().Be(np.@bool);
        }

        /// <summary>The C# operators read a string operand as NumPy's <c>str</c> scalar against a text array; bytes against str follow NumPy's no-loop rules; a null string keeps the null-array answer.</summary>
        [TestMethod]
        public void Operators_TakeACSharpStringAsNumPysStrScalar()
        {
            var u = np.array(new[] { "a", "b", "c" });
            (u < "b").ToArray<bool>().Should().Equal(true, false, false);
            ("b" > u).ToArray<bool>().Should().Equal(true, false, false);
            (u == "b").ToArray<bool>().Should().Equal(false, true, false);
            (u != "b").ToArray<bool>().Should().Equal(true, false, true);
            // Bytes against str: == is all False, an ordering has no loop.
            var s = Bytes("S1", "a", "b");
            (s == "a").ToArray<bool>().Should().Equal(false, false);
            Action less = () => _ = s < "a";
            less.Should().Throw<TypeError>().WithMessage(
                "ufunc 'less' did not contain a loop with signature matching types (<class 'numpy.dtypes.BytesDType'>, " +
                "<class 'numpy.dtypes.StrDType'>) -> None");
            // A null string keeps the null-array answer.
            (u == (string)null).GetBoolean().Should().BeFalse();
        }

        /// <summary>
        ///     A string on the LEFT is Python's reflection: <c>"b" &lt; a</c> runs <c>np.greater(a, "b")</c>, so the values match
        ///     and the bytes-against-str error names the reflected ufunc with the array's dtype first (probed 2.4.2).
        /// </summary>
        [TestMethod]
        public void Operators_AStringOnTheLeft_IsPythonsReflection()
        {
            var u = np.array(new[] { "a", "b", "c" });
            ("b" < u).ToArray<bool>().Should().Equal(false, false, true);
            ("b" <= u).ToArray<bool>().Should().Equal(false, true, true);
            ("b" >= u).ToArray<bool>().Should().Equal(true, true, false);
            ("b" == u).ToArray<bool>().Should().Equal(false, true, false);
            ("b" != u).ToArray<bool>().Should().Equal(true, false, true);

            var s = Bytes("S3", "a", "b");
            ("a" == s).ToArray<bool>().Should().Equal(false, false);
            ("a" != s).ToArray<bool>().Should().Equal(true, true);
            const string types = " did not contain a loop with signature matching types (<class 'numpy.dtypes.BytesDType'>, " +
                                 "<class 'numpy.dtypes.StrDType'>) -> None";
            Action lt = () => _ = "" < s;
            lt.Should().Throw<TypeError>().WithMessage("ufunc 'greater'" + types);
            Action le = () => _ = "" <= s;
            le.Should().Throw<TypeError>().WithMessage("ufunc 'greater_equal'" + types);
            Action gt = () => _ = "" > s;
            gt.Should().Throw<TypeError>().WithMessage("ufunc 'less'" + types);
            Action ge = () => _ = "" >= s;
            ge.Should().Throw<TypeError>().WithMessage("ufunc 'less_equal'" + types);
        }

        // ---------------------------------------------------------------- truth values

        /// <summary>The truth operand: empty, NUL-only, whitespace, a leading NUL and the text 'False'.</summary>
        /// <returns>The array.</returns>
        private static NDArray T() => np.array(new[] { "", "a", "\0", " ", "\0a", "False" }, dtype: "U5");

        /// <summary>Every truth reader over text: a string is True when any code unit is nonzero (whitespace and 'False' included).</summary>
        [TestMethod]
        public void Truth_ANonEmptyStringIsTrue()
        {
            T().astype(np.@bool).ToArray<bool>().Should().Equal(false, true, false, true, true, true);
            Bytes("S2", "", "\0", "\0a", "0").astype(np.@bool).ToArray<bool>().Should().Equal(false, false, true, true);
            np.nonzero(T())[0].ToArray<long>().Should().Equal(1L, 3L, 4L, 5L);
            np.count_nonzero(T()).Should().Be(4L);
            np.count_nonzero(T().reshape(2, 3), 0).ToArray<long>().Should().Equal(1L, 2L, 1L);
            np.any(T().reshape(2, 3), 1, true).ToArray<bool>().Should().Equal(true, true);
            np.all(T()).Should().BeFalse();
            np.argwhere(T().reshape(2, 3)).ToArray<long>().Should().Equal(0L, 1L, 1L, 0L, 1L, 1L, 1L, 2L);
            np.flatnonzero(T()).ToArray<long>().Should().Equal(1L, 3L, 4L, 5L);
            np.compress(T(), np.arange(6)).ToArray<long>().Should().Equal(1L, 3L, 4L, 5L);
            np.extract(T(), np.arange(6)).ToArray<long>().Should().Equal(1L, 3L, 4L, 5L);
            np.logical_and(T(), np.array(new long[] { 1, 1, 1, 0, 1, 1 })).ToArray<bool>().Should().Equal(false, true, false, false, true, true);
            np.logical_xor(T(), T()["::-1"]).ToArray<bool>().Should().Equal(true, false, true, true, false, true);
        }

        /// <summary><c>np.trim_zeros</c> trims empty strings (NumPy's argwhere route) and returns a view of the text.</summary>
        [TestMethod]
        public void Truth_TrimZeros_IsAViewOfTheText()
        {
            var r = np.trim_zeros(np.array(new[] { "", "a", "", "b", "", "" }));
            r.ToString(false).Should().Be("['a' '' 'b']");
            r.flags.owndata.Should().BeFalse();
            var r2 = np.trim_zeros(np.array(new[,] { { "", "", "" }, { "", "x", "" }, { "", "", "" } }));
            r2.shape.Should().Equal(new long[] { 1, 1 });
        }

        /// <summary>The bool cast builds a Python <c>str</c> per element, so a unit above U+10FFFF raises SystemError; the nonzero route reads the units and never raises.</summary>
        [TestMethod]
        public void Truth_TheCastRouteRaisesForUnitsAboveTheLastCodePoint_TheNonzeroRouteDoesNot()
        {
            const string message = "invalid maximum character passed to PyUnicode_New";
            Action astype = () => Units(0x110000, 0x41, 0xFFFFFFFF).astype(np.@bool);
            astype.Should().Throw<SystemError>().WithMessage(message);
            Action any = () => np.any(Units(0x110000, 0x41, 0xFFFFFFFF));
            any.Should().Throw<SystemError>().WithMessage(message);
            Action logical = () => np.logical_or(Units(0x110000, 0x41, 0xFFFFFFFF), np.array(0L));
            logical.Should().Throw<SystemError>().WithMessage(message);
            np.nonzero(Units(0x110000, 0x41, 0xFFFFFFFF))[0].ToArray<long>().Should().Equal(0L, 1L, 2L);
            np.count_nonzero(Units(0x110000, 0x41, 0xFFFFFFFF)).Should().Be(3L);
            np.compress(Units(0x110000, 0x41, 0xFFFFFFFF), np.arange(3)).ToArray<long>().Should().Equal(0L, 1L, 2L);
        }

        /// <summary>
        ///     The cast raises only for an element that is ONE unit above U+10FFFF once its trailing NULs are stripped (CPython's
        ///     <c>unicode_char</c>); a longer string builds with its maximum character capped at U+10FFFF and is True. Every
        ///     row below is a probe of NumPy 2.4.2's <c>astype(bool)</c> and <c>np.any</c>.
        /// </summary>
        [TestMethod]
        public void Truth_TheCastRouteRaisesOnlyForALoneUnitAboveTheLastCodePoint()
        {
            const uint big = 0x110000;
            const string message = "invalid maximum character passed to PyUnicode_New";
            var raising = new (int width, uint[] units)[]
            {
                (1, new[] { big }), (2, new[] { big, 0u }), (3, new[] { big, 0u, 0u }), (1, new[] { 0xFFFFFFFFu }),
                (2, new[] { 0xFFFFFFFFu, 0u }), (8, new[] { big, 0u, 0u, 0u, 0u, 0u, 0u, 0u }),
            };
            foreach (var (width, units) in raising)
            {
                Action astype = () => UnitRows(width, units).astype(np.@bool);
                astype.Should().Throw<SystemError>().WithMessage(message, $"U{width} [{string.Join(",", units)}] is one unit");
                Action any = () => np.any(UnitRows(width, units));
                any.Should().Throw<SystemError>().WithMessage(message);
                // The nonzero route never builds a str.
                np.count_nonzero(UnitRows(width, units)).Should().Be(1L);
            }

            var castable = new (int width, uint[] units)[]
            {
                (2, new[] { 0u, big }), (2, new[] { big, 0x41u }), (2, new[] { 0x41u, big }), (2, new[] { big, big }),
                (3, new[] { big, 0u, 0x41u }), (1, new[] { 0x10FFFFu }), (1, new[] { 0xD800u }),
                (8, new[] { 0u, 0u, 0u, 0u, 0u, 0u, 0u, big }), (9, new[] { big, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u }),
            };
            foreach (var (width, units) in castable)
            {
                UnitRows(width, units).astype(np.@bool).ToArray<bool>().Should().Equal(new[] { true }, $"U{width} [{string.Join(",", units)}]");
                np.any(UnitRows(width, units)).Should().BeTrue();
            }

            // The other cast readers follow the same rule: [big, 'A'] is True everywhere.
            var two = UnitRows(2, new[] { big, 0x41u }, new[] { 0x41u, 0u });
            np.logical_or(two, np.array(0L)).ToArray<bool>().Should().Equal(true, true);
            np.where(two, np.array(1L), np.array(2L)).ToArray<long>().Should().Equal(1L, 1L);
            var placed = np.zeros(new Shape(2L));
            np.place(placed, two, np.array(new[] { 7.0 }));
            placed.ToArray<double>().Should().Equal(7.0, 7.0);
            var copied = np.zeros(new Shape(2L), NPTypeCode.Boolean);
            np.copyto(copied, two, "unsafe", np.array(new[] { true, false }));
            copied.ToArray<bool>().Should().Equal(true, false);
            Action maskedLone = () => np.copyto(np.zeros(new Shape(2L), NPTypeCode.Boolean), UnitRows(2, new[] { big, 0u }, new[] { 0x41u, 0u }),
                "unsafe", np.array(new[] { true, false }));
            maskedLone.Should().Throw<SystemError>().WithMessage(message);

            // ma.flatten_mask iterates the first non-empty str: a first unit above U+10FFFF fails (as the str or as its first
            // character); a valid first character recurses.
            Action flattenBig = () => np.ma.flatten_mask(UnitRows(2, new[] { 0u, 0u }, new[] { big, 0x41u }));
            flattenBig.Should().Throw<SystemError>().WithMessage(message);
            Action flattenRecursion = () => np.ma.flatten_mask(UnitRows(2, new[] { 0x41u, big }));
            flattenRecursion.Should().Throw<RuntimeError>().WithMessage("maximum recursion depth exceeded");
        }

        /// <summary><c>np.copyto</c> into bool, <c>np.place</c> and <c>np.putmask</c> read a text source / mask through the bool cast, in NumPy's order (the mask cast before the size check; a masked copy converts only the selected elements).</summary>
        [TestMethod]
        public void Truth_Copyto_Place_Putmask()
        {
            var d = np.zeros(new Shape(3L), NPTypeCode.Boolean);
            np.copyto(d, np.array(new[] { "", "x", "\0" }), "unsafe");
            d.ToArray<bool>().Should().Equal(false, true, false);
            // The masked cast converts only the selected elements: a bad unit outside the mask does not raise.
            var masked = np.zeros(new Shape(3L), NPTypeCode.Boolean);
            np.copyto(masked, Units(0x110000, 0x41, 0xFFFFFFFF), "unsafe", np.array(new[] { false, true, false }));
            masked.ToArray<bool>().Should().Equal(false, true, false);
            Action inside = () => np.copyto(np.zeros(new Shape(3L), NPTypeCode.Boolean), Units(0x110000, 0x41, 0xFFFFFFFF),
                "unsafe", np.array(new[] { true, false, false }));
            inside.Should().Throw<SystemError>();
            Action sameKind = () => np.copyto(np.zeros(new Shape(2L), NPTypeCode.Boolean), np.array(new[] { "a", "" }));
            sameKind.Should().Throw<InvalidCastException>()
                    .WithMessage("Cannot cast array data from dtype('<U1') to dtype('bool') according to the rule 'same_kind'");

            var pa = np.zeros(new Shape(4L));
            np.place(pa, np.array(new[] { "", "x", "", "y" }), np.array(new[] { 7.0, 8.0 }));
            pa.ToArray<double>().Should().Equal(0.0, 7.0, 0.0, 8.0);
            var pm = np.zeros(new Shape(4L));
            np.putmask(pm, np.array(new[] { "", "x", "", "y" }), np.array(new[] { 1.0, 2.0, 3.0, 4.0 }));
            pm.ToArray<double>().Should().Equal(0.0, 2.0, 0.0, 4.0);
            // NumPy casts the mask before its size check: the SystemError comes first.
            Action placeBig = () => np.place(np.zeros(new Shape(4L)), Units(0x110000, 0x41, 0xFFFFFFFF), np.array(new[] { 1.0 }));
            placeBig.Should().Throw<SystemError>();
        }

        /// <summary>numpy.ma's text truth members: <c>flatten_mask</c> (a RecursionError for non-empty <c>U</c>, byte iteration for <c>S</c>) and <c>mask_or</c> (the result is the MaskedArray's data view).</summary>
        [TestMethod]
        public void Truth_MaskedArrayMembers()
        {
            Action recursion = () => np.ma.flatten_mask(np.array(new[] { "", "a" }));
            recursion.Should().Throw<RuntimeError>().WithMessage("maximum recursion depth exceeded");
            np.ma.flatten_mask(np.array(new[] { "", "" })).size.Should().Be(0);
            np.ma.flatten_mask(Bytes("S3", "ab", "", "a\0b")).ToArray<bool>().Should().Equal(true, true, true, false, true);
            np.ma.flatten_mask(np.array("x")).ToArray<bool>().Should().Equal(true);
            var x = np.ma.array(np.array(new[] { "ab", "", "c" }));
            var m = np.ma.mask_or(x, x);
            m.ToArray<bool>().Should().Equal(true, false, true);
            m.flags.owndata.Should().BeFalse();   // NumPy hands back the MaskedArray's _data view
        }

        /// <summary><c>np.array_equal</c> compares text across widths, and the histogram's monotonic check compares text edges.</summary>
        [TestMethod]
        public void Equality_AndHistogramEdges()
        {
            np.array_equal(np.array(new[] { "a", "b" }), np.array(new[] { "a", "b\0" }, dtype: "U3")).Should().BeTrue();
            Action hist = () => np.histogram(np.zeros(new Shape(0L)), np.array(new[] { "b", "a" }));
            hist.Should().Throw<ValueError>().WithMessage("`bins` must increase monotonically, when an array");
        }

        // ---------------------------------------------------------------- the IL kernels against the reference

        /// <summary>
        ///     Every comparison kernel shape against <see cref="NDStringComparison.ManagedKernel"/> (NumPy's strcmp, unit by
        ///     unit): both classes, widths across the unrolled / loop boundary, both operand orders, the four broadcast
        ///     patterns, and elements built to hit equal prefixes, tails and single-unit differences (high bytes, units above
        ///     U+10FFFF, 0xFFFFFFFF).
        /// </summary>
        [TestMethod]
        public unsafe void CompareKernels_MatchTheReference()
        {
            if (!ILKernelGenerator.TextKernelsAvailable)
                Assert.Inconclusive("Dynamic code is unavailable: the managed reference is the kernel.");
            var rng = new System.Random(20261009);
            int[] widths1 = { 1, 3, 8, 15, 16, 31, 33, 64, 257, 300 };
            int[] widths4 = { 4, 12, 32, 36, 64, 128, 260, 520 };
            long mismatches = 0;
            void** p = stackalloc void*[3];
            long* st = stackalloc long[3];
            const int n = 23;
            foreach (int unit in new[] { 1, 4 })
            foreach (int wa in unit == 1 ? widths1 : widths4)
            foreach (int wb in unit == 1 ? widths1 : widths4)
            foreach (TextCompareOp op in Enum.GetValues<TextCompareOp>())
            foreach (int aIndex in new[] { 0, 1 })
            {
                var il = ILKernelGenerator.GetTextCompareKernel(op, unit, wa, wb, aIndex);
                var reference = NDStringComparison.ManagedKernel(op, unit, wa, wb, aIndex);
                int w0 = aIndex == 0 ? wa : wb, w1 = aIndex == 0 ? wb : wa;
                for (int pattern = 0; pattern < 4; pattern++)
                {
                    long s0 = (pattern & 1) != 0 ? 0 : w0 + 1, s1 = (pattern & 2) != 0 ? 0 : w1 + 3;
                    var b0 = new byte[s0 * n + w0];
                    var b1 = new byte[s1 * n + w1];
                    var o1 = new byte[n];
                    var o2 = new byte[n];
                    FillRelated(rng, b0, s0 == 0 ? 1 : n, s0, w0, b1, s1 == 0 ? 1 : n, s1, w1, unit);
                    fixed (byte* pb0 = b0, pb1 = b1, po1 = o1, po2 = o2)
                    {
                        p[0] = pb0; p[1] = pb1; st[0] = s0; st[1] = s1; st[2] = 1;
                        p[2] = po1;
                        il(p, st, n, null);
                        p[2] = po2;
                        reference(p, st, n, null);
                    }
                    for (int i = 0; i < n; i++)
                        if (o1[i] != o2[i])
                            mismatches++;
                }
            }
            mismatches.Should().Be(0);
        }

        /// <summary>
        ///     The truth kernels (both flavors) against <see cref="NDStringTruth.ManagedKernel"/>, flag included. The U
        ///     elements mix the cases the cast flavor's lone-unit test branches on: a lone first unit above U+10FFFF (fails),
        ///     such a first unit followed by a nonzero unit anywhere (builds), and an invalid unit past the first (builds).
        /// </summary>
        [TestMethod]
        public unsafe void TruthKernels_MatchTheReference()
        {
            if (!ILKernelGenerator.TextKernelsAvailable)
                Assert.Inconclusive("Dynamic code is unavailable: the managed reference is the kernel.");
            var rng = new System.Random(7);
            void** p = stackalloc void*[2];
            long* st = stackalloc long[2];
            const int n = 29;
            int failingCalls = 0;
            foreach (int unit in new[] { 1, 4 })
            foreach (int w in unit == 1 ? new[] { 1, 2, 7, 8, 17, 32, 33, 300 } : new[] { 4, 8, 12, 32, 36, 300 })
            foreach (bool check in new[] { false, true })
            for (int trial = 0; trial < 8; trial++)
            {
                long s = w + 2;
                var data = new byte[s * n];
                // One U trial in four has no lone unit at all, so a call that must NOT set the flag is checked too.
                bool lonesAllowed = unit == 4 && trial % 4 != 0;
                for (int i = 0; i < n; i++)
                {
                    int kind = rng.Next(8);
                    if (kind == 0) continue;                                                     // empty
                    if (unit == 4 && kind == 1 && lonesAllowed)
                    {
                        // A lone first unit above U+10FFFF: the one-unit str CPython refuses.
                        BitConverter.GetBytes(0x110000u + (uint)rng.Next(9)).CopyTo(data, i * s);
                        continue;
                    }
                    data[i * s + rng.Next(w)] = (byte)rng.Next(1, 256);
                    if (unit == 4 && kind == 2)
                    {
                        // A first unit above U+10FFFF plus a nonzero unit after it: a longer str, which builds.
                        BitConverter.GetBytes(0x110000u + (uint)rng.Next(9)).CopyTo(data, i * s);
                        if (w > 4)
                            data[i * s + 4 + rng.Next(w - 4)] = (byte)rng.Next(1, 256);
                    }
                    else if (unit == 4 && kind == 3 && w > 4)
                        BitConverter.GetBytes(0x110000u + (uint)rng.Next(9)).CopyTo(data, i * s + (1 + rng.Next(w / 4 - 1)) * 4);
                }
                var o1 = new byte[n];
                var o2 = new byte[n];
                int f1 = 0, f2 = 0;
                fixed (byte* pd = data, po1 = o1, po2 = o2)
                {
                    p[0] = pd; st[0] = s; st[1] = 1;
                    p[1] = po1;
                    ILKernelGenerator.GetTextTruthKernel(unit, w, check)(p, st, n, &f1);
                    p[1] = po2;
                    NDStringTruth.ManagedKernel(unit, w, check)(p, st, n, &f2);
                }
                o1.Should().Equal(o2);
                f1.Should().Be(f2);
                failingCalls += f1;
            }
            // The data did reach the failing branch (a vacuous comparison would pass with every flag zero).
            failingCalls.Should().BeGreaterThan(0);
        }

        /// <summary>
        ///     Fills the operands so the interesting cases dominate: element B is often a copy of element A (equal, a strict
        ///     prefix, one changed unit, a nonzero tail unit), with special units (high bytes, units above U+10FFFF).
        /// </summary>
        private static void FillRelated(System.Random rng, byte[] a, long na, long sa, int wa, byte[] b, long nb, long sb, int wb, int unit)
        {
            for (long i = 0; i < na; i++)
                for (int u = 0; u < wa / unit; u++)
                    WriteUnit(a, i * sa + u * unit, unit, u < rng.Next(wa / unit + 1) ? RandomUnit(rng, unit) : 0);
            for (long i = 0; i < nb; i++)
            {
                long src = (sa == 0 ? 0 : System.Math.Min(i, na - 1) * sa), dst = i * sb;
                int copy = System.Math.Min(wa, wb);
                if (rng.Next(4) != 0)
                    Array.Copy(a, src, b, dst, copy);
                else
                    for (int u = 0; u < copy / unit; u++)
                        WriteUnit(b, dst + u * unit, unit, RandomUnit(rng, unit));
                for (int k = copy; k < wb; k++)
                    b[dst + k] = 0;
                int mode = rng.Next(4);
                if (mode == 0 && wb / unit > 0)
                    WriteUnit(b, dst + rng.Next(wb / unit) * unit, unit, RandomUnit(rng, unit));
                else if (mode == 1 && wb > wa)
                    WriteUnit(b, dst + (wa / unit + rng.Next((wb - wa) / unit)) * unit, unit, RandomUnit(rng, unit));
            }
        }

        /// <summary>A random code unit, biased to the cases the kernels branch on.</summary>
        private static uint RandomUnit(System.Random rng, int unit)
        {
            int k = rng.Next(10);
            if (unit == 1)
                return k < 2 ? (uint)rng.Next(0x80, 0x100) : k < 3 ? 0u : (uint)rng.Next(0x61, 0x64);
            return k < 1 ? 0x110000u + (uint)rng.Next(4) : k < 2 ? 0xFFFFFFFFu : k < 3 ? 0u : (uint)rng.Next(0x61, 0x64);
        }

        /// <summary>Writes one code unit (little-endian for U).</summary>
        private static void WriteUnit(byte[] buffer, long offset, int unit, uint value)
        {
            if (unit == 1)
                buffer[offset] = (byte)value;
            else
                BitConverter.GetBytes(value).CopyTo(buffer, offset);
        }
    }
}
