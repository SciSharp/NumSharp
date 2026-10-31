using System;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     <c>numpy.char</c> phase 1 (docs/plans/np-char.md, C0 / C1): the <c>np.@char</c> facade, the six comparisons and
    ///     <c>compare_chararrays</c> that strip trailing whitespace, and <c>np.char.add</c> / <c>np.char.multiply</c>. The
    ///     literal expectations were probed against NumPy 2.4.2 (the session's <c>probe_char.py</c> / <c>probe_char2.py</c>);
    ///     the string oracle's section <c>strchar</c> replays the full matrices. The kernel tests check every form of the
    ///     rstrip kernel (1-D with its broadcast stages, rows, masked) against a literal port of NumPy's
    ///     <c>Buffer::rstrip</c> + <c>Buffer::strcmp</c>, which shares no code with the kernels' bit-mask method.
    /// </summary>
    [TestClass]
    public unsafe class CharModuleTests
    {
        // ------------------------------------------------------------------ the facade

        /// <summary>
        ///     The stripped units are NUL and the six ASCII whitespace characters, for <c>U</c> and <c>S</c> alike;
        ///     <c>'\x1c'</c>, <c>'\xa0'</c>, U+3000 and a leading space stay (probed: <c>[True]*7 + [False]*3 + [True, True,
        ///     False]</c>).
        /// </summary>
        [TestMethod]
        public void Equal_StripsNulAndAsciiWhitespaceOnly()
        {
            var u = np.array(new[] { "a", "a ", "a\t", "a\n", "a\v", "a\f", "a\r", "a\x1c", "a\xa0", "a　", "a\0", "a \0 ", " a" });
            np.@char.equal(u, "a").ToArray<bool>().Should().Equal(
                true, true, true, true, true, true, true, false, false, false, true, true, false);
            var s = Bytes("S9", "a\t\n\v\f\r \0", "a\x1c", "a\xa0", "a\x85");
            np.@char.equal(s, Bytes("S1", "a")).ToArray<bool>().Should().Equal(true, false, false, false);
        }

        /// <summary>
        ///     The six comparisons on stripped strings: an embedded NUL still counts, the longer stripped string is the
        ///     greater, a strip unit inside the string is an ordinary unit (probed 2.4.2).
        /// </summary>
        [TestMethod]
        public void Comparisons_CompareTheStrippedStrings()
        {
            np.@char.less(np.array(new[] { "a ", "a\t", "a!", "a" }), np.array(new[] { "a\0", "a ", "a", "a!" }))
                .ToArray<bool>().Should().Equal(false, false, false, true);
            np.@char.greater_equal(np.array(new[] { "b ", "a" }), np.array(new[] { "b", "a " })).ToArray<bool>().Should().Equal(true, true);
            np.@char.not_equal(np.array(new[] { "b ", "a" }), np.array(new[] { "b", "a\x01" })).ToArray<bool>().Should().Equal(false, true);
            ((bool)np.@char.less("a ", "a!")).Should().BeTrue();
            ((bool)np.@char.equal("a\0b ", "a\0b")).Should().BeTrue();
            ((bool)np.@char.greater("a\0b", "a")).Should().BeTrue();
            ((bool)np.@char.less_equal("a b", "a")).Should().BeFalse();
        }

        /// <summary>
        ///     Two scalars give a 0-d array (NumPy's iterator allocates it), an empty operand an empty bool array, an
        ///     F-ordered pair an F result, a column against a row the broadcast (probed 2.4.2).
        /// </summary>
        [TestMethod]
        public void Comparisons_ResultShapesAndLayouts()
        {
            var zero = np.@char.equal("a", "a  ");
            zero.ndim.Should().Be(0);
            zero.dtype.Should().Be(np.@bool);
            ((bool)zero).Should().BeTrue();
            var empty = np.@char.equal(np.zeros(new Shape(0L), "U2"), "a");
            empty.shape.Should().Equal(new long[] { 0 });
            var f = np.@char.equal(np.array(new[,] { { "a", "b" }, { "c", "d" } }).T, np.array(new[,] { { "a ", "b " }, { "c ", "d " } }).T);
            f.flags.f_contiguous.Should().BeTrue();
            f.flags.c_contiguous.Should().BeFalse();
            f.ToArray<bool>().Should().Equal(true, true, true, true);
            var b = np.@char.equal(np.array(new[,] { { "a" }, { "b" } }), np.array(new[] { "a", "b", "c" }));
            b.shape.Should().Equal(new long[] { 2, 3 });
            b.ToArray<bool>().Should().Equal(true, false, false, false, true, false);
        }

        /// <summary>
        ///     NumPy's checks in its order: a non-<c>S</c>/<c>U</c> operand (a number, a <c>StringDType</c> array) is a
        ///     TypeError; an <c>S</c> / <c>U</c> pair answers <c>NotImplemented</c> (C# null) even when the shapes do not
        ///     broadcast; one class that does not broadcast is NumPy's ValueError text.
        /// </summary>
        [TestMethod]
        public void Comparisons_ErrorsAndNotImplemented()
        {
            var u = np.array(new[] { "a ", "b" });
            np.@char.equal(Bytes("S2", "a ", "b"), u).Should().BeNull();
            np.@char.equal(u, new[] { "a"u8.ToArray() }).Should().BeNull();
            np.@char.less(new[] { "a"u8.ToArray(), "b"u8.ToArray() }, np.array(new[] { "a", "b", "c" })).Should().BeNull();
            Action number = () => np.@char.equal(u, np.array(new[] { 1, 2 }));
            number.Should().Throw<TypeError>().WithMessage("comparison of non-string arrays");
            Action boolean = () => np.@char.equal(np.array(new[] { true }), np.array(new[] { true }));
            boolean.Should().Throw<TypeError>().WithMessage("comparison of non-string arrays");
            var t = np.array(new[] { "a ", "b" }, dtype: "T");
            Action vstring = () => np.@char.equal(t, t);
            vstring.Should().Throw<TypeError>().WithMessage("comparison of non-string arrays");
            Action mixed = () => np.@char.equal(u, t);
            mixed.Should().Throw<TypeError>().WithMessage("comparison of non-string arrays");
            Action shapes = () => np.@char.equal(np.array(new[] { "a", "b" }), np.array(new[] { "a", "b", "c" }));
            shapes.Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with shapes (2,) (3,) ");
            Action none = () => np.@char.equal(null, u);
            none.Should().Throw<ArgumentNullException>();
        }

        /// <summary>
        ///     <c>compare_chararrays</c>' <c>cmp</c>: the six spellings as str or bytes, every other byte string a ValueError
        ///     — checked BEFORE the operands, so it wins over a <c>StringDType</c>, a number or a broadcast error — a lone
        ///     surrogate CPython's UnicodeEncodeError, null NumPy's <c>None</c> text (probed 2.4.2).
        /// </summary>
        [TestMethod]
        public void CompareChararrays_ParsesCmpLikeNumPy()
        {
            var a = np.array(new[] { "a" });
            var b = np.array(new[] { "a " });
            var want = new[] { ("==", true), ("!=", false), ("<", false), ("<=", true), (">", false), (">=", true) };
            foreach (var (cmp, result) in want)
            {
                ((bool)np.@char.compare_chararrays(a, b, cmp, true)[0]).Should().Be(result, cmp);
                ((bool)np.@char.compare_chararrays(a, b, System.Text.Encoding.ASCII.GetBytes(cmp), true)[0]).Should().Be(result, cmp);
            }
            ((bool)np.@char.compare_chararrays(a, b, "==", false)[0]).Should().BeFalse();
            const string message = "comparison must be '==', '!=', '<', '>', '<=', '>='";
            foreach (var bad in new[] { "=", "=<", "<>", "", "===", "! ", "é", "=é", "\0", "=\0", "==\0", " =", "=>" })
            {
                Action call = () => np.@char.compare_chararrays(a, b, bad, true);
                call.Should().Throw<ValueError>().WithMessage(message);
            }
            Action badBytes = () => np.@char.compare_chararrays(a, b, new byte[] { 0xFF }, true);
            badBytes.Should().Throw<ValueError>().WithMessage(message);
            Action surrogate = () => np.@char.compare_chararrays(a, b, "\ud800", true);
            surrogate.Should().Throw<UnicodeEncodeError>()
                     .WithMessage("'utf-8' codec can't encode character '\\ud800' in position 0: surrogates not allowed");
            Action none = () => np.@char.compare_chararrays(a, b, (string)null, true);
            none.Should().Throw<TypeError>().WithMessage("a bytes-like object is required, not 'NoneType'");
            // cmp first: a bad cmp wins over a StringDType operand, a number and a broadcast error.
            Action badCmpT = () => np.@char.compare_chararrays(np.array(new[] { "a" }, dtype: "T"), a, "=", true);
            badCmpT.Should().Throw<ValueError>().WithMessage(message);
            Action badCmpShape = () => np.@char.compare_chararrays(np.array(new[] { "a", "b" }), np.array(new[] { "a", "b", "c" }), "=", true);
            badCmpShape.Should().Throw<ValueError>().WithMessage(message);
        }

        /// <summary>
        ///     The text argument rule: a C# string is NumPy's <c>str</c>, a <c>byte[]</c> its <c>bytes</c>, a
        ///     <c>byte[][]</c> a list of bytes, a <c>string[]</c> a list of str.
        /// </summary>
        [TestMethod]
        public void Arguments_FollowTheTextRule()
        {
            ((bool)np.@char.equal(new byte[] { (byte)'a', (byte)' ' }, new byte[] { (byte)'a' })).Should().BeTrue();
            np.@char.equal(new[] { "a"u8.ToArray(), "b \t"u8.ToArray() }, new[] { "a"u8.ToArray(), "b"u8.ToArray() })
                .ToArray<bool>().Should().Equal(true, true);
            np.@char.greater(new[] { "b", "a" }, "a ").ToArray<bool>().Should().Equal(true, false);
            Text(np.@char.add(new[] { "x", "y" }, "z")).Should().Equal("xz", "yz");
        }

        /// <summary>
        ///     <c>np.char.add</c> is <c>np.add</c>: concatenation of one class, NumPy's mix error, a 0-d result for two
        ///     strings (probed 2.4.2).
        /// </summary>
        [TestMethod]
        public void Add_IsNpAdd()
        {
            var u = np.array(new[] { "a ", "b" });
            var r = np.@char.add(u, u);
            r.dtype.Should().Be(np.dtype("<U4"));
            Text(r).Should().Equal("a a ", "bb");
            var ab = np.@char.add("a", "b");
            ab.ndim.Should().Be(0);
            ab.dtype.Should().Be(np.dtype("<U2"));
            Action mix = () => np.@char.add(u, Bytes("S2", "a ", "b"));
            mix.Should().Throw<TypeError>().WithMessage("ufunc 'add' cannot use operands with types dtype('<U2') and dtype('S2')");
        }

        /// <summary>
        ///     <c>np.char.multiply</c> re-raises every TypeError of <c>np.strings.multiply</c> as
        ///     <c>ValueError("Can only multiply by integers")</c>; its ValueErrors and OverflowError pass through (probed
        ///     2.4.2: a float, a bool, a str count, a numeric text operand, a uint64 count, a count of 2**62 all give the
        ///     ValueError; an empty result is NumPy's reduction ValueError).
        /// </summary>
        [TestMethod]
        public void Multiply_TypeErrorsBecomeTheValueError()
        {
            var u = np.array(new[] { "a ", "b" });
            Text(np.@char.multiply(u, 2)).Should().Equal("a a ", "bb");
            var neg = np.@char.multiply(u, np.array(new[] { -1, 2 }));
            neg.dtype.Should().Be(np.dtype("<U2"));
            Text(neg).Should().Equal("", "bb");
            var s = np.@char.multiply("ab", 3);
            s.ndim.Should().Be(0);
            s.dtype.Should().Be(np.dtype("<U6"));
            foreach (var count in new object[] { 2.5, true, "x", (ulong)2, 1L << 62 })
            {
                Action bad = () => np.@char.multiply(u, count);
                bad.Should().Throw<ValueError>().WithMessage("Can only multiply by integers");
            }
            Action numeric = () => np.@char.multiply(np.array(new[] { 1, 2 }), 2);
            numeric.Should().Throw<ValueError>().WithMessage("Can only multiply by integers");
            Action empty = () => np.@char.multiply(np.zeros(new Shape(0L), "U3"), 2);
            empty.Should().Throw<ValueError>().WithMessage("zero-size array to reduction operation maximum which has no identity");
            Action overflow = () => np.@char.multiply(np.array(new[] { "abc" }), long.MaxValue / 2);
            overflow.Should().Throw<OverflowException>().WithMessage("Overflow encountered in string multiply");
        }

        /// <summary>
        ///     <c>np.@char</c> still stands for the <c>Char</c> dtype wherever a <see cref="DType"/> is expected: an
        ///     <c>astype</c> / creation argument, <c>==</c> in both orders, a <c>DType[]</c> initializer, a tuple key.
        /// </summary>
        [TestMethod]
        public void TheModule_ConvertsToTheCharDtype()
        {
            var ones = np.ones(new Shape(2L), np.@char);
            (ones.dtype == np.@char).Should().BeTrue();
            (np.@char == ones.dtype).Should().BeTrue();
            ((DType)np.@char).Should().BeSameAs(np.dtype("char"));
            var list = new[] { np.@char, np.int16 };
            list.GetType().GetElementType().Should().Be(typeof(DType));
            var table = new System.Collections.Generic.Dictionary<(DType, DType), DType> { { (np.int16, np.@char), np.@char } };
            table[(np.int16, (DType)np.@char)].Should().Be(np.dtype("char"));
            np.@char.ToString().Should().Be("<module 'numpy.char'>");
        }

        // ------------------------------------------------------------------ the rstrip kernels against NumPy's code

        /// <summary>
        ///     Every 1-D rstrip comparison kernel (the inline pattern and the three broadcast stages) against
        ///     <see cref="NumPyStripCompare"/>: both classes, widths on both sides of the bit-mask limit (64 units) and the
        ///     unrolled-chunk limit, both operand orders, every op. The managed kernel (no dynamic code) is checked too.
        /// </summary>
        [TestMethod]
        public void StripKernels_MatchNumPysRstripAndStrcmp()
        {
            var rng = new System.Random(20261015);
            int[] chars1 = { 1, 2, 3, 7, 8, 15, 16, 33, 64, 65, 300 };
            int[] chars4 = { 1, 2, 3, 8, 9, 16, 63, 64, 65, 70 };
            long mismatches = 0, cases = 0, stripped = 0;
            void** p = stackalloc void*[3];
            long* st = stackalloc long[3];
            const int n = 29;
            bool il = ILKernelGenerator.TextKernelsAvailable;
            foreach (int unit in new[] { 1, 4 })
            foreach (int ca in unit == 1 ? chars1 : chars4)
            foreach (int cb in unit == 1 ? chars1 : chars4)
            foreach (TextCompareOp op in Enum.GetValues<TextCompareOp>())
            foreach (int aIndex in new[] { 0, 1 })
            {
                int wa = ca * unit, wb = cb * unit;
                var kernel = il ? ILKernelGenerator.GetTextCompareKernel(op, unit, wa, wb, aIndex, rstrip: true) : null;
                var managed = NDStringComparison.ManagedKernel(op, unit, wa, wb, aIndex, rstrip: true);
                int w0 = aIndex == 0 ? wa : wb, w1 = aIndex == 0 ? wb : wa;
                for (int pattern = 0; pattern < 4; pattern++)
                {
                    long s0 = (pattern & 1) != 0 ? 0 : w0 + 1, s1 = (pattern & 2) != 0 ? 0 : w1 + 3;
                    long n0 = s0 == 0 ? 1 : n, n1 = s1 == 0 ? 1 : n;
                    var b0 = new byte[s0 * n + w0];
                    var b1 = new byte[s1 * n + w1];
                    FillStripPair(rng, b0, n0, s0, w0, b1, n1, s1, w1, unit);
                    var got = new byte[n];
                    var viaManaged = new byte[n];
                    fixed (byte* pb0 = b0, pb1 = b1, pg = got, pm = viaManaged)
                    {
                        p[0] = pb0; p[1] = pb1; st[0] = s0; st[1] = s1; st[2] = 1;
                        if (kernel is not null)
                        {
                            p[2] = pg;
                            kernel(p, st, n, null);
                        }
                        p[2] = pm;
                        managed(p, st, n, null);
                    }
                    for (int i = 0; i < n; i++)
                    {
                        long o0 = (s0 == 0 ? 0 : i) * s0, o1 = (s1 == 0 ? 0 : i) * s1;
                        // Operand slot aIndex is A: the expected order is A against B.
                        int cmp = aIndex == 0
                            ? NumPyStripCompare(b0, o0, w0, b1, o1, w1, unit)
                            : NumPyStripCompare(b1, o1, w1, b0, o0, w0, unit);
                        byte want = Decide(op, cmp);
                        cases++;
                        if (cmp == 0 && !BytesEqual(b0, o0, w0, b1, o1, w1))
                            stripped++;
                        if (viaManaged[i] != want || (kernel is not null && got[i] != want))
                            mismatches++;
                    }
                }
            }
            mismatches.Should().Be(0);
            // The data reached the rule under test: pairs equal only once stripped.
            stripped.Should().BeGreaterThan(cases / 50);
        }

        /// <summary>
        ///     The rows and masked forms of the rstrip kernels against <see cref="NumPyStripCompare"/>: five stride
        ///     patterns per operand (dense, column broadcast, row broadcast, scalar, reversed rows), any mask byte selects.
        /// </summary>
        [TestMethod]
        public void StripRowsAndMaskedKernels_MatchNumPysRstripAndStrcmp()
        {
            if (!ILKernelGenerator.TextKernelsAvailable)
                Assert.Inconclusive("Dynamic code is unavailable: the managed kernel is the only form.");
            var rng = new System.Random(15);
            const int rows = 4, inner = 5;
            long mismatches = 0, maskedOff = 0;
            void** p = stackalloc void*[4];
            long* si = stackalloc long[4];
            long* so = stackalloc long[4];
            foreach (int unit in new[] { 1, 4 })
            foreach (int ca in unit == 1 ? new[] { 1, 8, 64, 65 } : new[] { 2, 9, 64, 66 })
            foreach (int cb in unit == 1 ? new[] { 3, 33 } : new[] { 3, 65 })
            foreach (TextCompareOp op in Enum.GetValues<TextCompareOp>())
            foreach (int aIndex in new[] { 0, 1 })
            foreach (bool masked in new[] { false, true })
            {
                int wa = ca * unit, wb = cb * unit;
                var kernel = ILKernelGenerator.GetTextCompareRowsKernel(op, unit, wa, wb, aIndex, masked, rstrip: true);
                int w0 = aIndex == 0 ? wa : wb, w1 = aIndex == 0 ? wb : wa;
                for (int pa = 0; pa < 5; pa++)
                for (int pb = 0; pb < 5; pb++)
                {
                    var (i0, o0) = Strides(pa, w0, inner);
                    var (i1, o1) = Strides(pb, w1, inner);
                    var x0 = new byte[Span(rows, inner, w0, i0, o0, out long z0)];
                    var x1 = new byte[Span(rows, inner, w1, i1, o1, out long z1)];
                    // Every element sits at a whole multiple of its width from the buffer start (Span), so the buffers
                    // are filled element by element; a broadcast operand's positions alias one element.
                    FillStripPair(rng, x0, x0.Length / w0, w0, w0, x1, x1.Length / w1, w1, w1, unit);
                    var outBuf = new byte[rows * inner];
                    for (int k = 0; k < outBuf.Length; k++)
                        outBuf[k] = (byte)rng.Next(2);
                    var start = (byte[])outBuf.Clone();
                    var mask = new byte[rows * inner];
                    for (int k = 0; k < mask.Length; k++)
                        mask[k] = new byte[] { 0, 1, 2, 255 }[rng.Next(4)];
                    fixed (byte* b0 = x0, b1 = x1, bo = outBuf, bm = mask)
                    {
                        p[0] = b0 + z0; p[1] = b1 + z1; p[2] = bo; p[3] = bm;
                        si[0] = i0; si[1] = i1; si[2] = 1; si[3] = 1;
                        so[0] = o0; so[1] = o1; so[2] = inner; so[3] = inner;
                        kernel(p, si, inner, so, rows, null);
                    }
                    for (int r = 0; r < rows; r++)
                    for (int i = 0; i < inner; i++)
                    {
                        int k = r * inner + i;
                        long e0 = z0 + r * o0 + i * i0, e1 = z1 + r * o1 + i * i1;
                        int cmp = aIndex == 0
                            ? NumPyStripCompare(x0, e0, w0, x1, e1, w1, unit)
                            : NumPyStripCompare(x1, e1, w1, x0, e0, w0, unit);
                        byte want = !masked || mask[k] != 0 ? Decide(op, cmp) : start[k];
                        if (masked && mask[k] == 0)
                            maskedOff++;
                        if (outBuf[k] != want)
                            mismatches++;
                    }
                }
            }
            mismatches.Should().Be(0);
            maskedOff.Should().BeGreaterThan(0);
        }

        /// <summary>
        ///     The public route over NDArray layouts (the trivial loop, the iterator, its rows blocks, broadcast operands)
        ///     against <see cref="NumPyStripCompare"/> element by element, with and without the rows kernels.
        /// </summary>
        [TestMethod]
        [DoNotParallelize] // flips the process-wide string-comparison route switches (NDStringComparison)
        public void Comparisons_OverLayouts_MatchTheElementwisePort()
        {
            var rng = new System.Random(3);
            string[] pool = { "a", "a ", "a\t\0", "ab", " a", "a\x1c", "", " ", "\0", "b\n", "a\0b", "abc  " };
            NDArray Random(int n, string dtype) => np.array(Enumerable.Range(0, n).Select(_ => pool[rng.Next(pool.Length)]).ToArray(), dtype: dtype);
            // Every layout keeps the logical shape (4, 12) or broadcasts to it: C, F, a transposed buffer, a stepped view, a
            // reversed view, a row, a column, a scalar.
            var layouts = new (string Name, Func<string, NDArray> Make)[]
            {
                ("c", d => Random(48, d).reshape(4, 12)),
                ("f", d => np.asfortranarray(Random(48, d).reshape(4, 12))),
                ("t", d => Random(48, d).reshape(12, 4).T),
                ("step", d => Random(96, d).reshape(4, 24)[":, ::2"]),
                ("rev", d => Random(48, d).reshape(4, 12)[":, ::-1"]),
                ("row", d => Random(12, d).reshape(1, 12)),
                ("col", d => Random(4, d).reshape(4, 1)),
                ("scalar", d => Random(1, d).reshape()),
            };
            bool saved = NDStringComparison.RowsDisabled;
            try
            {
                foreach (bool rowsOff in new[] { false, true })
                {
                    NDStringComparison.RowsDisabled = rowsOff;
                    foreach (var (n1, m1) in layouts)
                    foreach (var (n2, m2) in layouts)
                    {
                        var a = m1("U6");
                        var b = m2("U5");
                        foreach (var name in new[] { "equal", "not_equal", "less", "less_equal", "greater", "greater_equal" })
                        {
                            var r = Call(name, a, b);
                            var dims = r.shape;
                            var got = r.flatten().ToArray<bool>();
                            var av = Text(np.broadcast_to(a, new Shape(dims)));
                            var bv = Text(np.broadcast_to(b, new Shape(dims)));
                            for (int i = 0; i < got.Length; i++)
                                got[i].Should().Be(Decide(Op(name, out bool swap), swap ? StrCmp(bv[i], av[i]) : StrCmp(av[i], bv[i])) == 1,
                                                   $"{name} {n1}/{n2} rows {(rowsOff ? "off" : "on")} [{i}]: '{av[i]}' vs '{bv[i]}'");
                        }
                    }
                }
            }
            finally
            {
                NDStringComparison.RowsDisabled = saved;
            }
        }

        // ------------------------------------------------------------------ helpers

        /// <summary>The elements of a <c>U</c> array as C# strings, C order (Python's str: trailing NULs dropped).</summary>
        private static string[] Text(NDArray a) => NDArray.AsStringArray(a.ndim == 0 ? a.reshape(1) : a.flatten());

        /// <summary>The six public comparisons by name.</summary>
        private static NDArray Call(string name, NDArray a, NDArray b) => name switch
        {
            "equal" => np.@char.equal(a, b),
            "not_equal" => np.@char.not_equal(a, b),
            "less" => np.@char.less(a, b),
            "less_equal" => np.@char.less_equal(a, b),
            "greater" => np.@char.greater(a, b),
            _ => np.@char.greater_equal(a, b),
        };

        /// <summary>A comparison name as the kernel op and whether its operands swap (greater / greater_equal).</summary>
        private static TextCompareOp Op(string name, out bool swap)
        {
            swap = name is "greater" or "greater_equal";
            return name switch
            {
                "equal" => TextCompareOp.Equal,
                "not_equal" => TextCompareOp.NotEqual,
                "less" or "greater" => TextCompareOp.Less,
                _ => TextCompareOp.LessEqual,
            };
        }

        /// <summary>NumPy's strip-and-compare of two C# strings (UCS-4 code points, as a <c>U</c> element holds them).</summary>
        private static int StrCmp(string a, string b)
        {
            var ua = a.EnumerateRunes().Select(r => (uint)r.Value).ToArray();
            var ub = b.EnumerateRunes().Select(r => (uint)r.Value).ToArray();
            var ba = new byte[ua.Length * 4];
            var bb = new byte[ub.Length * 4];
            Buffer.BlockCopy(ua, 0, ba, 0, ba.Length);
            Buffer.BlockCopy(ub, 0, bb, 0, bb.Length);
            return NumPyStripCompare(ba, 0, ba.Length, bb, 0, bb.Length, 4);
        }

        /// <summary>The bool an op gives for an order: Equal 0, NotEqual != 0, Less &lt; 0, LessEqual &lt;= 0.</summary>
        private static byte Decide(TextCompareOp op, int cmp) => op switch
        {
            TextCompareOp.Equal => (byte)(cmp == 0 ? 1 : 0),
            TextCompareOp.NotEqual => (byte)(cmp != 0 ? 1 : 0),
            TextCompareOp.Less => (byte)(cmp < 0 ? 1 : 0),
            _ => (byte)(cmp <= 0 ? 1 : 0),
        };

        /// <summary>
        ///     A literal port of NumPy 2.4.2's <c>Buffer&lt;enc&gt;::rstrip</c> followed by <c>Buffer&lt;enc&gt;::strcmp</c>
        ///     (string_buffer.h): walk back over NUL and <c>NumPyOS_ascii_isspace</c>, then compare the common prefix unit by
        ///     unit (unsigned) and let a nonzero unit in the longer remainder decide.
        /// </summary>
        private static int NumPyStripCompare(byte[] a, long ao, int wa, byte[] b, long bo, int wb, int unit)
        {
            int la = wa / unit, lb = wb / unit;
            while (la > 0 && IsSpaceOrNul(UnitAt(a, ao, la - 1, unit)))
                la--;
            while (lb > 0 && IsSpaceOrNul(UnitAt(b, bo, lb - 1, unit)))
                lb--;
            int i = 0;
            for (; i < la && i < lb; i++)
            {
                uint x = UnitAt(a, ao, i, unit), y = UnitAt(b, bo, i, unit);
                if (x < y)
                    return -1;
                if (x > y)
                    return 1;
            }
            for (int k = i; k < la; k++)
                if (UnitAt(a, ao, k, unit) != 0)
                    return 1;
            for (int k = i; k < lb; k++)
                if (UnitAt(b, bo, k, unit) != 0)
                    return -1;
            return 0;
        }

        /// <summary>NumPy's rstrip test: <c>*tmp == '\0' || NumPyOS_ascii_isspace(*tmp)</c>.</summary>
        private static bool IsSpaceOrNul(uint c) => c is 0 or ' ' or '\t' or '\n' or '\v' or '\f' or '\r';

        /// <summary>The code unit <paramref name="index"/> of the element at <paramref name="at"/>.</summary>
        private static uint UnitAt(byte[] buf, long at, int index, int unit)
            => unit == 1 ? buf[at + index] : BitConverter.ToUInt32(buf, (int)(at + index * 4L));

        /// <summary>Whether two elements hold identical bytes (equal widths required).</summary>
        private static bool BytesEqual(byte[] a, long ao, int wa, byte[] b, long bo, int wb)
        {
            if (wa != wb)
                return false;
            for (int k = 0; k < wa; k++)
                if (a[ao + k] != b[bo + k])
                    return false;
            return true;
        }

        /// <summary>
        ///     Fills two operands so NumPy's strip rule decides many pairs: each element is content (mostly letters, now and
        ///     then a strip unit inside, special units), a trailing run of strip units, then NULs; an element of B is often
        ///     A's element with another trailing run (equal once stripped), one changed unit or one more content unit.
        /// </summary>
        private static void FillStripPair(System.Random rng, byte[] a, long na, long sa, int wa, byte[] b, long nb, long sb, int wb,
                                          int unit)
        {
            for (long i = 0; i < na; i++)
                FillStripElement(rng, a, i * sa, wa, unit);
            for (long i = 0; i < nb; i++)
            {
                long dst = i * sb;
                if (wa == 0 || wb == 0 || rng.Next(3) == 0)
                {
                    FillStripElement(rng, b, dst, wb, unit);
                    continue;
                }
                // Derive from A's element at the same index (A's last when A has fewer).
                long src = System.Math.Min(i, na - 1) * sa;
                int la = wa / unit, lb = wb / unit;
                int content = la;
                while (content > 0 && IsSpaceOrNul(UnitAt(a, src, content - 1, unit)))
                    content--;
                content = System.Math.Min(content, lb);
                for (int u = 0; u < lb; u++)
                    WriteUnit(b, dst + u * unit, unit, 0);
                for (int u = 0; u < content; u++)
                    WriteUnit(b, dst + u * unit, unit, UnitAt(a, src, u, unit));
                int mode = rng.Next(4);
                if (mode == 1 && content > 0)
                    WriteUnit(b, dst + rng.Next(content) * unit, unit, ContentUnit(rng, unit));
                else if (mode == 2 && content < lb)
                    WriteUnit(b, dst + content++ * unit, unit, ContentUnit(rng, unit));
                // A fresh trailing strip run (mode 0 keeps the content: equal only once stripped, when it fits).
                int trail = rng.Next(lb - content + 1);
                for (int u = 0; u < trail; u++)
                    WriteUnit(b, dst + (content + u) * unit, unit, StripUnit(rng));
            }
        }

        /// <summary>One element: content, a trailing strip run, NUL padding.</summary>
        private static void FillStripElement(System.Random rng, byte[] buf, long at, int width, int unit)
        {
            int n = width / unit;
            int content = rng.Next(n + 1);
            int trail = rng.Next(n - content + 1);
            for (int u = 0; u < n; u++)
            {
                uint v = u < content ? ContentUnit(rng, unit) : u < content + trail ? StripUnit(rng) : 0u;
                WriteUnit(buf, at + u * unit, unit, v);
            }
        }

        /// <summary>A content unit: mostly 'a' / 'b', sometimes a strip unit inside the string, sometimes a unit the strip must keep.</summary>
        private static uint ContentUnit(System.Random rng, int unit)
        {
            int k = rng.Next(12);
            if (k < 7)
                return k % 2 == 0 ? 'a' : 'b';
            if (k < 9)
                return StripUnit(rng);
            // Units next to the strip set ('\b' 0x08, 0x0E, '!' 0x21), the other whitespace NumPy keeps (0x1C, 0x85, 0xA0,
            // U+3000, U+2028), high bytes, and U units past U+10FFFF (unsigned values only).
            var keep = unit == 1
                ? new uint[] { 0x1C, 0x85, 0xA0, 0x80, 0xFF, 0x08, 0x0E, 0x1F, 0x21 }
                : new uint[] { 0x1C, 0x85, 0xA0, 0x3000, 0x2028, 0x110000, 0xFFFFFFFF, 0x08, 0x0E, 0x21 };
            return keep[rng.Next(keep.Length)];
        }

        /// <summary>One of the seven units NumPy's rstrip removes.</summary>
        private static uint StripUnit(System.Random rng) => new uint[] { 0, ' ', '\t', '\n', '\v', '\f', '\r' }[rng.Next(7)];

        /// <summary>Writes one code unit (little-endian for U).</summary>
        private static void WriteUnit(byte[] buffer, long offset, int unit, uint value)
        {
            if (unit == 1)
                buffer[offset] = (byte)value;
            else
                BitConverter.GetBytes(value).CopyTo(buffer, offset);
        }

        /// <summary>The strides (inner, outer) a raw operand is tested with: dense, inner broadcast, outer broadcast, scalar, reversed rows.</summary>
        private static (long Inner, long Outer) Strides(int pattern, int width, int inner) => pattern switch
        {
            0 => (width, (long)width * inner),
            1 => (0, width),
            2 => (width, 0),
            3 => (0, 0),
            _ => (width, -(long)width * inner),
        };

        /// <summary>
        ///     The byte length a raw operand of <paramref name="rows"/> × <paramref name="inner"/> elements needs at the given
        ///     strides, and the offset of element (0, 0) (past any negative extent). Every element sits at a whole multiple of
        ///     the width from the buffer start, so the buffer can be filled element by element.
        /// </summary>
        private static long Span(int rows, int inner, int width, long innerStride, long outerStride, out long origin)
        {
            long lo = 0, hi = 0;
            foreach (long r in new long[] { 0, rows - 1 })
                foreach (long i in new long[] { 0, inner - 1 })
                {
                    long o = r * outerStride + i * innerStride;
                    lo = System.Math.Min(lo, o);
                    hi = System.Math.Max(hi, o);
                }
            origin = -lo;
            return hi - lo + width;
        }

        /// <summary>An <c>S</c> array from Latin-1 strings (one byte per char).</summary>
        private static NDArray Bytes(string dtype, params string[] s) =>
            np.array(s.Select(x => (object)x.Select(c => (byte)c).ToArray()).ToArray(), dtype: dtype);
    }
}
