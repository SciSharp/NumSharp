using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Sorting;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     Strings phase 2, P2.2 milestone B (docs/plans/strings-phase2.md): <c>np.argmax</c> / <c>np.argmin</c> /
    ///     <c>np.nanargmax</c> / <c>np.nanargmin</c> and the ndarray methods, <c>np.searchsorted</c> /
    ///     <c>ndarray.searchsorted</c>, <c>np.digitize</c> and the bin search of <c>np.histogram2d</c> over <c>S</c> /
    ///     <c>U</c> text. The expectations are NumPy 2.4.2 literals (first extreme element, NumPy's exact search path over
    ///     unsorted haystacks with its monotonic carry, sorter values reached and not reached, the argument checks in NumPy's
    ///     order), plus references written from NumPy's algorithms (the first extreme by ordinal order; NumPy's binary search
    ///     step for step) for the randomized route checks. The routes — the GATHER and KEYS filters of the arg-reductions,
    ///     the prefix skip, the short rows, the DIRECT / KEYED / SORTED searches — are held to one another with counters
    ///     proving which ran, and every read to the operands' bytes with fenced pages.
    /// </summary>
    [TestClass]
    public unsafe class StringSearchTests
    {
        // ------------------------------------------------------------------ helpers

        /// <summary>A <c>U</c> array of the given strings (as wide as the longest).</summary>
        /// <param name="values">The strings.</param>
        /// <returns>The array.</returns>
        private static NDArray U(params string[] values) => np.array(values);

        /// <summary>The int64 elements of an index array, in C order.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The values.</returns>
        private static long[] Ints(NDArray a) => a.ToArray<long>();

        /// <summary>Random strings of 0 to <paramref name="width"/> letters from <paramref name="alphabet"/>.</summary>
        /// <param name="rng">The generator.</param>
        /// <param name="n">The count.</param>
        /// <param name="width">The most letters.</param>
        /// <param name="alphabet">The letters.</param>
        /// <param name="minLength">The fewest letters.</param>
        /// <returns>The strings.</returns>
        private static string[] Random(System.Random rng, int n, int width, string alphabet, int minLength = 0)
            => Enumerable.Range(0, n)
                .Select(_ => new string(Enumerable.Range(0, rng.Next(minLength, width + 1)).Select(_ => alphabet[rng.Next(alphabet.Length)]).ToArray()))
                .ToArray();

        /// <summary>NumPy's argmax / argmin of strings without NULs: the first extreme element by ordinal order.</summary>
        /// <param name="values">The strings.</param>
        /// <param name="max">True for argmax.</param>
        /// <returns>The index.</returns>
        private static long FirstExtreme(IReadOnlyList<string> values, bool max)
        {
            long best = 0;
            for (int i = 1; i < values.Count; i++)
            {
                int c = string.CompareOrdinal(values[i], values[(int)best]);
                if (max ? c > 0 : c < 0)
                    best = i;
            }
            return best;
        }

        /// <summary>
        ///     NumPy's <c>npy_binsearch</c> step for step (its monotonic carry included) over strings without NULs: the
        ///     reference for sorted and unsorted haystacks alike.
        /// </summary>
        /// <param name="hay">The haystack.</param>
        /// <param name="keys">The keys, in C order.</param>
        /// <param name="left">True for <c>side='left'</c>.</param>
        /// <param name="sorter">The sorter, or null.</param>
        /// <returns>The insertion points.</returns>
        private static long[] NumPySearch(string[] hay, string[] keys, bool left, long[] sorter = null)
        {
            long n = hay.Length, minIdx = 0, maxIdx = n;
            var ret = new long[keys.Length];
            string last = keys.Length > 0 ? keys[0] : null;
            for (int i = 0; i < keys.Length; i++)
            {
                int carry = string.CompareOrdinal(last, keys[i]);
                if (left ? carry < 0 : carry <= 0)
                    maxIdx = n;
                else
                {
                    minIdx = 0;
                    maxIdx = maxIdx < n ? maxIdx + 1 : n;
                }
                last = keys[i];
                while (minIdx < maxIdx)
                {
                    long mid = minIdx + ((maxIdx - minIdx) >> 1);
                    string h = hay[sorter is null ? mid : sorter[mid]];
                    int c = string.CompareOrdinal(h, keys[i]);
                    if (left ? c < 0 : c <= 0)
                        minIdx = mid + 1;
                    else
                        maxIdx = mid;
                }
                ret[i] = minIdx;
            }
            return ret;
        }

        /// <summary>
        ///     Runs <paramref name="body"/> with the routes' test hooks set, restoring them after (they are process-wide).
        /// </summary>
        /// <param name="gatherOff">Sets <see cref="TextArgReduce.GatherDisabled"/>.</param>
        /// <param name="prefixOff">Sets <see cref="TextArgReduce.PrefixSkipDisabled"/> and <see cref="TextSearch.PrefixSkipDisabled"/>.</param>
        /// <param name="keyed">Sets <see cref="TextSearch.ForceKeyed"/>.</param>
        /// <param name="sortedOff">Sets <see cref="TextSearch.SortedRouteDisabled"/>.</param>
        /// <param name="managed">Turns the IL generator off, so every key kernel is its managed twin.</param>
        /// <param name="body">The calls to make.</param>
        private static void WithRoutes(bool gatherOff, bool prefixOff, bool? keyed, bool sortedOff, bool managed, Action body)
        {
            bool gather = TextArgReduce.GatherDisabled, prefixA = TextArgReduce.PrefixSkipDisabled;
            bool prefixS = TextSearch.PrefixSkipDisabled, sorted = TextSearch.SortedRouteDisabled, il = DirectILKernelGenerator.Enabled;
            bool? forced = TextSearch.ForceKeyed;
            try
            {
                TextArgReduce.GatherDisabled = gatherOff;
                TextArgReduce.PrefixSkipDisabled = prefixOff;
                TextSearch.PrefixSkipDisabled = prefixOff;
                TextSearch.ForceKeyed = keyed;
                TextSearch.SortedRouteDisabled = sortedOff;
                DirectILKernelGenerator.Enabled = !managed;
                body();
            }
            finally
            {
                TextArgReduce.GatherDisabled = gather;
                TextArgReduce.PrefixSkipDisabled = prefixA;
                TextSearch.PrefixSkipDisabled = prefixS;
                TextSearch.SortedRouteDisabled = sorted;
                TextSearch.ForceKeyed = forced;
                DirectILKernelGenerator.Enabled = il;
            }
        }

        // ------------------------------------------------------------------ arg-reductions: NumPy's answers

        /// <summary>The first of equal extremes wins, for every function and both text kinds (NumPy literals).</summary>
        [TestMethod]
        public void ArgReductions_ReturnTheFirstExtreme()
        {
            var words = U("pear", "fig", "apple", "fig", "pear", "kiwi");
            np.argmax(words).Should().Be(0L);
            np.argmin(words).Should().Be(2L);
            np.nanargmax(words).Should().Be(0L);
            np.nanargmin(words).Should().Be(2L);
            var bytes = words.astype(np.dtype("S5"));
            np.argmax(bytes).Should().Be(0L);
            np.argmin(bytes).Should().Be(2L);
            words.argmax().Should().Be(0L);
            words.argmin().Should().Be(2L);
        }

        /// <summary>Every axis, keepdims, and the methods (NumPy literals).</summary>
        [TestMethod]
        public void ArgReductions_Axes_AreNumPys()
        {
            var m = np.array(new[,] { { "b", "a", "c" }, { "a", "c", "c" } });
            Ints(np.argmax(m, 0)).Should().Equal(0L, 1L, 0L);
            Ints(np.argmin(m, 0)).Should().Equal(1L, 0L, 0L);
            Ints(np.argmax(m, 1)).Should().Equal(2L, 1L);
            Ints(np.argmin(m, 1)).Should().Equal(1L, 0L);
            var kd = np.argmax(m, (int?)1, null, keepdims: true);
            kd.shape.Should().Equal(new long[] { 2, 1 });
            Ints(kd).Should().Equal(2L, 1L);
            var flatKd = np.argmax(m, (int?)null, null, keepdims: true);
            flatKd.shape.Should().Equal(new long[] { 1, 1 });
            Ints(flatKd).Should().Equal(2L);
            Ints(m.argmin((int?)0, null, false)).Should().Equal(1L, 0L, 0L);
            Ints(np.nanargmax(m, (int?)1)).Should().Equal(2L, 1L);
        }

        /// <summary>
        ///     <c>out=</c>: NumPy writes the indices into an integer array it can cast them into 'safely' from bool through
        ///     uint32 (unsafe value cast: a bool out reads index != 0), refuses uint64 and floats with its TypeError, a wrong
        ///     shape with its ValueError, and checks an empty line first.
        /// </summary>
        [TestMethod]
        public void ArgReductions_Out_FollowsNumPysChecks()
        {
            var m = np.array(new[,] { { "b", "a", "c" }, { "a", "c", "c" } });
            foreach (var dt in new[] { "int64", "int32", "uint8", "int8" })
            {
                var o = np.zeros(new Shape(2L), np.dtype(dt));
                var r = np.argmax(m, (int?)1, o);
                ReferenceEquals(r, o).Should().BeTrue();
                o.astype(np.int64).ToArray<long>().Should().Equal(2L, 1L);
            }
            var b = np.zeros(new Shape(2L), np.@bool);
            np.argmax(m, (int?)1, b);
            b.ToArray<bool>().Should().Equal(true, true);
            Action u64 = () => np.argmax(m, (int?)1, np.zeros(new Shape(2L), np.uint64));
            u64.Should().Throw<TypeError>().WithMessage("Cannot cast array data from dtype('uint64') to dtype('int64') according to the rule 'safe'");
            Action f64 = () => np.argmax(m, (int?)1, np.zeros(new Shape(2L), np.float64));
            f64.Should().Throw<TypeError>().WithMessage("Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'");
            Action shape = () => np.argmax(m, (int?)1, np.zeros(new Shape(3L), np.int64));
            shape.Should().Throw<ValueError>().WithMessage("output array does not match result of np.argmax.");
            Action empty = () => np.argmin(np.zeros(new Shape(0L, 3L), np.dtype("U2")), (int?)0, np.zeros(new Shape(3L), np.float64));
            empty.Should().Throw<ValueError>().WithMessage("attempt to get argmin of an empty sequence");
        }

        /// <summary>
        ///     The GATHER filter takes a group of eight units against the group's starting best: an earlier lane of the group
        ///     can raise the best above a later lane that passed the mask, which must not replace it (NumPy literals).
        /// </summary>
        [TestMethod]
        public void ArgReductions_ALaterLaneOfAGroup_FacesTheRaisedBest()
        {
            var g = U(new[] { "a", "c", "b" }.Concat(Enumerable.Repeat("a", 14)).ToArray());
            np.argmax(g).Should().Be(1L);
            np.argmin(g).Should().Be(0L);
            var g2 = U(new[] { "m", "c", "d" }.Concat(Enumerable.Repeat("n", 14)).ToArray());
            np.argmax(g2).Should().Be(3L);
            np.argmin(g2).Should().Be(1L);
        }

        /// <summary>Every <c>U</c> key packing, units above U+10FFFF (compared unsigned), NULs and high bytes (NumPy literals).</summary>
        [TestMethod]
        public void ArgReductions_EveryPackingAndUnit_AreNumPys()
        {
            var lat = U("é", "z", "ÿ", "a", "ÿ");
            (np.argmax(lat), np.argmin(lat)).Should().Be((2L, 3L));
            var bmp = U("中", "文", "字", "a", "文");
            (np.argmax(bmp), np.argmin(bmp)).Should().Be((1L, 3L));
            var astral = U("\U0001F600", "a", "\U00010000", "\U0001F600");
            (np.argmax(astral), np.argmin(astral)).Should().Be((0L, 1L));
            var raw = new uint[] { 0x41, 0, 0x110000, 0, 0xFFFFFFFF, 0, 0x10FFFF, 0, 0xFFFFFFFF, 1 };
            var bytes = raw.SelectMany(BitConverter.GetBytes).ToArray();
            var hi = np.frombuffer(bytes, np.dtype("<U2")).copy();
            (np.argmax(hi), np.argmin(hi)).Should().Be((4L, 0L));
            var nul = np.array(new object[] { "a"u8.ToArray(), "a\0b"u8.ToArray(), "a\0"u8.ToArray(), new byte[] { 0xff }, new byte[] { 0xff, 0 }, Array.Empty<byte>() }, dtype: "S3");
            (np.argmax(nul), np.argmin(nul)).Should().Be((3L, 5L));
        }

        // ------------------------------------------------------------------ arg-reductions: routes

        /// <summary>
        ///     The routes give one answer: the GATHER and KEYS filters, with and without the prefix skip, with the emitted and
        ///     the managed key kernels — over random, categorical, long-prefix and block-straddling lines, flat and along both
        ///     axes; the counters prove each route ran.
        /// </summary>
        [TestMethod]
        public void ArgReductions_Routes_GiveOneAnswer()
        {
            var rng = new System.Random(16);
            var sets = new List<(string name, string[] values)>
            {
                ("rand", Random(rng, 2500, 8, "abcdefghijklmnop")),
                ("cat", Enumerable.Range(0, 2500).Select(_ => new[] { "apple", "banana", "cherry", "honeydew" }[rng.Next(4)]).ToArray()),
                ("prefix", Random(rng, 2500, 3, "abc").Select(s => "https://site.example/" + s).ToArray()),
                ("equal", Enumerable.Repeat("same", 1500).ToArray()),
            };
            var blocks = Random(rng, 2049, 6, "abcde");
            foreach (int pos in new[] { 0, 1023, 1024, 2048 })
                blocks[pos] = "zzzzzz";
            sets.Add(("blocks", blocks));
            long gatherBefore = TextArgReduce.GatherRuns, keysBefore = TextArgReduce.BlockRuns, shortBefore = TextArgReduce.ShortRowRuns;
            foreach (var (name, values) in sets)
            {
                foreach (var kind in new[] { "U", "S" })
                {
                    var a = U(values).astype(np.dtype($"{kind}{values.Max(v => v.Length)}"));
                    var rows = a[$":{values.Length - values.Length % 10}"].reshape(-1, 10);
                    foreach (bool max in new[] { true, false })
                    {
                        long expect = FirstExtreme(values, max);
                        var expectRows = Enumerable.Range(0, (int)rows.shape[0])
                            .Select(r => FirstExtreme(values.Skip(r * 10).Take(10).ToArray(), max)).ToArray();
                        var expectCols = Enumerable.Range(0, 10)
                            .Select(c => FirstExtreme(Enumerable.Range(0, (int)rows.shape[0]).Select(r => values[r * 10 + c]).ToArray(), max)).ToArray();
                        foreach (var (gatherOff, prefixOff, managed) in new[] { (false, false, false), (true, false, false), (false, true, false), (true, true, true) })
                        {
                            WithRoutes(gatherOff, prefixOff, null, false, managed, () =>
                            {
                                (max ? np.argmax(a) : np.argmin(a)).Should().Be(expect, $"{name} {kind} max={max} routes {gatherOff}/{prefixOff}/{managed}");
                                Ints(max ? np.argmax(rows, 1) : np.argmin(rows, 1)).Should().Equal(expectRows);
                                Ints(max ? np.argmax(rows, 0) : np.argmin(rows, 0)).Should().Equal(expectCols);
                            });
                        }
                    }
                }
            }
            // The GATHER filter needs AVX2 (a host without it, or DOTNET_EnableAVX2=0, keys every block).
            if (System.Runtime.Intrinsics.X86.Avx2.IsSupported)
                (TextArgReduce.GatherRuns - gatherBefore).Should().BePositive("the GATHER filter ran");
            (TextArgReduce.BlockRuns - keysBefore).Should().BePositive("the KEYS filter ran");
            (TextArgReduce.ShortRowRuns - shortBefore).Should().BePositive("the short rows ran");
        }

        /// <summary>
        ///     Text whose best string recurs (one element in eight) switches the call's later blocks from the GATHER filter to
        ///     the KEYS filter; the answer is the first copy either way.
        /// </summary>
        [TestMethod]
        public void ArgReductions_TieHeavyText_SwitchesFilters_AndKeepsTheFirstCopy()
        {
            var rng = new System.Random(4);
            string[] words = { "apple", "banana", "cherry", "date", "elder", "fig", "grape", "honeydew" };
            var values = Enumerable.Range(0, 4000).Select(_ => words[rng.Next(words.Length)]).ToArray();
            var a = U(values);
            long gather = TextArgReduce.GatherRuns, keys = TextArgReduce.BlockRuns;
            np.argmax(a).Should().Be(FirstExtreme(values, true));
            np.argmin(a).Should().Be(FirstExtreme(values, false));
            (TextArgReduce.BlockRuns - keys).Should().BePositive("the later blocks are keyed");
            // The GATHER filter needs AVX2 (a host without it keys every block, the same answer).
            if (System.Runtime.Intrinsics.X86.Avx2.IsSupported)
                (TextArgReduce.GatherRuns - gather).Should().BePositive("the first block gathers");
        }

        // ------------------------------------------------------------------ searches: NumPy's answers

        /// <summary>Sorted and unsorted haystacks, both sides, a sorter (NumPy literals: its exact path over the unsorted one).</summary>
        [TestMethod]
        public void SearchSorted_IsNumPys()
        {
            var words = U("pear", "fig", "apple", "fig", "pear", "kiwi");
            var sorted = np.sort(words);
            var keys = U("fig", "kiwi", "a", "zz", "pear");
            Ints(np.searchsorted(sorted, keys)).Should().Equal(1L, 3L, 0L, 6L, 4L);
            Ints(np.searchsorted(sorted, keys, "right")).Should().Equal(3L, 4L, 0L, 6L, 6L);
            Ints(np.searchsorted(words, keys)).Should().Equal(0L, 4L, 0L, 6L, 6L);
            Ints(np.searchsorted(words, keys, "right")).Should().Equal(4L, 6L, 0L, 6L, 6L);
            var order = np.argsort(words, kind: "stable");
            Ints(order).Should().Equal(2L, 1L, 3L, 5L, 0L, 4L);
            Ints(np.searchsorted(words, keys, "left", order)).Should().Equal(1L, 3L, 0L, 6L, 4L);
            Ints(words.searchsorted(keys, "left", order)).Should().Equal(1L, 3L, 0L, 6L, 4L);
            // A 0-d key gives a 0-d answer.
            var zeroD = np.searchsorted(sorted, np.array("kiwi"));
            zeroD.ndim.Should().Be(0);
            zeroD.GetInt64().Should().Be(3L);
            // NumPy's carry over an unsorted haystack: an ascending key keeps the lower bound.
            var u = U("m", "c", "x", "a", "q", "b");
            Ints(np.searchsorted(u, U("n", "b", "z", "c", "a"))).Should().Equal(6L, 4L, 6L, 6L, 0L);
            Ints(np.searchsorted(u, U("n", "b", "z", "c", "a"), "right")).Should().Equal(6L, 6L, 6L, 6L, 4L);
        }

        /// <summary>A sorter value outside the haystack raises only when the search reaches it (NumPy literals).</summary>
        [TestMethod]
        public void SearchSorted_SorterChecks_FireWhenReached()
        {
            var hay = U("d", "a", "c", "b", "e");
            Ints(np.searchsorted(hay, U("c"), "left", np.array(new long[] { 1, 3, 2, 0, 4 }))).Should().Equal(2L);
            Ints(np.searchsorted(hay, U("a"), "left", np.array(new long[] { 1, 3, 2, 0, 9 }))).Should().Equal(0L);
            Ints(np.searchsorted(hay, U("c"), "left", np.array(new long[] { 1, 3, 2, 0, 9 }))).Should().Equal(2L);
            Action reached = () => np.searchsorted(hay, U("z"), "left", np.array(new long[] { 1, 3, 2, 0, 9 }));
            reached.Should().Throw<ValueError>().WithMessage("Sorter index out of range.");
            Action mid = () => np.searchsorted(hay, U("a"), "left", np.array(new long[] { 1, 3, 9, 0, 4 }));
            mid.Should().Throw<ValueError>().WithMessage("Sorter index out of range.");
        }

        /// <summary>Widths, kinds, numbers and the side check (NumPy literals).</summary>
        [TestMethod]
        public void SearchSorted_WidthsKindsNumbers_AreNumPys()
        {
            Ints(np.searchsorted(U("ab", "abc", "b"), np.array(new[] { "ab", "abcd", "a" }).astype(np.dtype("U4")))).Should().Equal(0L, 2L, 0L);
            np.searchsorted(U("1", "10", "2", "5"), 3).Should().Be(3L);
            Ints(np.searchsorted(U("1", "10", "2", "5"), np.array(new long[] { 3, 10, 100 }))).Should().Equal(3L, 1L, 2L);
            var bytesHay = np.array(new object[] { "ab"u8.ToArray(), "b"u8.ToArray(), "cd"u8.ToArray() }, dtype: "S2");
            Ints(np.searchsorted(bytesHay, U("b", "ba", "c"))).Should().Equal(1L, 2L, 2L);
            Action nonAscii = () => np.searchsorted(U("a", "b"), np.array(new object[] { new byte[] { 0xff } }, dtype: "S1"));
            nonAscii.Should().Throw<UnicodeDecodeError>();
            Action side = () => np.searchsorted(U("a", "b"), U("a"), "lft");
            side.Should().Throw<ArgumentException>().WithMessage("search side must be one of 'left' or 'right'*");
        }

        /// <summary>np.digitize of text searches the bins converted to text: the bins follow text order (NumPy literals).</summary>
        [TestMethod]
        public void Digitize_OfText_IsNumPys()
        {
            var x = U("10", "3", "25");
            Ints(np.digitize(x, np.array(new long[] { 5, 20 }))).Should().Equal(0L, 2L, 2L);
            Ints(np.digitize(x, np.array(new long[] { 5, 20 }), true)).Should().Equal(0L, 2L, 2L);
            Ints(np.digitize(x, np.array(new long[] { 20, 5 }))).Should().Equal(2L, 0L, 0L);
        }

        /// <summary>histogram2d of text: NumPy searches the edges as text (the point "1" sorts before the edge "1.0").</summary>
        [TestMethod]
        public void Histogram2d_OfText_SearchesTheEdgesAsText()
        {
            var r = np.histogram2d(U("a", "bc", "1"), np.array(new[] { 0.5, 0.2, 0.9 }), 3, new (double, double)?[] { (0.0, 1.0), (0.0, 1.0) });
            r.H.ToArray<double>().Should().Equal(0, 0, 0, 0, 0, 0, 0, 0, 1);
        }

        /// <summary>
        ///     histogramdd's first statement stacks the sample (<c>atleast_2d(sample).T</c>): an <c>S</c> column beside a
        ///     <c>U</c> one is decoded there, so a byte that is not ASCII raises <see cref="UnicodeDecodeError"/> before a bad bin
        ///     count, decreasing edges or good edges are read (NumPy 2.4.2). ASCII bytes stack into <c>U</c> and are searched in
        ///     the numeric edges converted to text; the caller's sample array keeps its own columns.
        /// </summary>
        [TestMethod]
        public void Histogramdd_OfMixedKindText_StacksFirst_AndKeepsTheCallersColumns()
        {
            var nonAscii = np.array(new object[] { new byte[] { (byte)'a', 0xff }, "b"u8.ToArray() }, dtype: "S2");
            var y = U("c", "d");
            Action count = () => np.histogram2d(nonAscii, y, 2);
            count.Should().Throw<UnicodeDecodeError>();
            Action decreasing = () => np.histogram2d(nonAscii, y, new[] { np.array(new[] { 2.0, 1.0 }), np.array(new[] { 1.0, 2.0 }) });
            decreasing.Should().Throw<UnicodeDecodeError>();
            Action edges = () => np.histogram2d(nonAscii, y, new[] { np.array(new[] { 1.0, 2.0 }), np.array(new[] { 1.0, 2.0 }) });
            edges.Should().Throw<UnicodeDecodeError>();

            // "10" sorts between the text edges "1.0" and "2.0", "3" before "30.0"; "1.5" and "0.5" land in bins 1 and 0.
            var ascii = np.array(new object[] { "10"u8.ToArray(), "3"u8.ToArray() }, dtype: "S2");
            var r = np.histogram2d(ascii, U("1.5", "0.5"), np.array(new[] { 0.0, 1.0, 2.0, 30.0 }));
            r.H.ToArray<double>().Should().Equal(0, 0, 0, 0, 1, 0, 1, 0, 0);

            // The text edges "0.0", "5.0", "20.0" are not sorted as text: NumPy's exact search, step for step.
            var text = U("10", "3", "25");
            var number = np.array(new[] { 1.5, 2.5, 0.5 });
            var sample = new[] { text, number };
            var dd = np.histogramdd(sample, new[] { np.array(new[] { 0.0, 5, 20 }), np.array(new[] { 0.0, 1, 2, 3 }) });
            dd.H.ToArray<double>().Should().Equal(1, 1, 0, 0, 0, 0);
            ReferenceEquals(sample[0], text).Should().BeTrue();
            ReferenceEquals(sample[1], number).Should().BeTrue();
            sample[1].dtype.Should().Be(np.float64);
        }

        // ------------------------------------------------------------------ searches: routes

        /// <summary>
        ///     The routes give NumPy's answers: DIRECT, KEYED with and without the SORTED route and the prefix skip, with the
        ///     emitted and the managed key kernels — over sorted and unsorted haystacks, sorted / descending / random keys,
        ///     with and without a sorter, keys that end before the strings (ties in the haystack past the key) and shared
        ///     prefixes; the counters prove each route ran.
        /// </summary>
        [TestMethod]
        public void SearchSorted_Routes_GiveNumPysAnswers()
        {
            var rng = new System.Random(32);
            var sets = new List<(string name, string[] hay, string[] keys)>
            {
                ("rand", Random(rng, 700, 6, "abcdefg"), Random(rng, 300, 6, "abcdefgh")),
                ("cat", Enumerable.Range(0, 700).Select(_ => new[] { "ab", "b", "abc", "c" }[rng.Next(4)]).ToArray(),
                        Enumerable.Range(0, 300).Select(_ => new[] { "ab", "b", "a", "abcd", "d" }[rng.Next(5)]).ToArray()),
                ("prefix", Random(rng, 700, 3, "abc").Select(s => "https://site.example/" + s).ToArray(),
                           Random(rng, 300, 3, "abcd").Select(s => "https://site.example/" + s).ToArray()),
                ("ties", Random(rng, 700, 12, "ab", 11), Random(rng, 300, 12, "ab", 11)),
            };
            long direct = TextSearch.DirectRuns, keyed = TextSearch.KeyedRuns, sortedRuns = TextSearch.SortedRuns, exact = TextSearch.ExactRuns;
            foreach (var (name, hayValues, keyValues) in sets)
            {
                foreach (var kind in new[] { "U", "S" })
                {
                    int width = System.Math.Max(hayValues.Max(v => v.Length), keyValues.Max(v => v.Length));
                    var sortedValues = hayValues.OrderBy(v => v, StringComparer.Ordinal).ToArray();
                    var sorterValues = Enumerable.Range(0, hayValues.Length).OrderBy(i => hayValues[i], StringComparer.Ordinal).Select(i => (long)i).ToArray();
                    NDArray Arr(string[] v) => U(v).astype(np.dtype($"{kind}{width}"));
                    var cases = new (string label, string[] hay, string[] keys, long[] sorter)[]
                    {
                        ("sorted", sortedValues, keyValues, null),
                        ("sortedkeys", sortedValues, keyValues.OrderBy(v => v, StringComparer.Ordinal).ToArray(), null),
                        ("desckeys", hayValues, keyValues.OrderByDescending(v => v, StringComparer.Ordinal).ToArray(), null),
                        ("unsorted", hayValues, keyValues, null),
                        ("sorter", hayValues, keyValues, sorterValues),
                    };
                    foreach (var (label, hay, keys, sorter) in cases)
                    {
                        var hayArr = Arr(hay);
                        var keyArr = Arr(keys);
                        var sorterArr = sorter is null ? null : np.array(sorter);
                        foreach (bool left in new[] { true, false })
                        {
                            long[] expect = NumPySearch(hay, keys, left, sorter);
                            foreach (var (forced, sortedOff, prefixOff, managed) in new (bool?, bool, bool, bool)[]
                                     { (null, false, false, false), (false, false, false, false), (true, false, false, false),
                                       (true, true, false, false), (true, false, true, false), (true, false, false, true) })
                            {
                                WithRoutes(false, prefixOff, forced, sortedOff, managed, () =>
                                    Ints(np.searchsorted(hayArr, keyArr, left ? "left" : "right", sorterArr))
                                        .Should().Equal(expect, $"{name} {kind} {label} left={left} routes {forced}/{sortedOff}/{prefixOff}/{managed}"));
                            }
                        }
                    }
                }
            }
            (TextSearch.DirectRuns - direct).Should().BePositive("the DIRECT route ran");
            (TextSearch.KeyedRuns - keyed).Should().BePositive("the KEYED route ran");
            (TextSearch.SortedRuns - sortedRuns).Should().BePositive("the SORTED route ran");
            (TextSearch.ExactRuns - exact).Should().BePositive("NumPy's exact loop over keys ran");
        }

        /// <summary>
        ///     The shared prefix the KEYED search skips must hold for EVERY element: a haystack whose elements share a long
        ///     prefix but for one element in the middle (and keys that break it in one place) is searched from unit 0 there.
        /// </summary>
        [TestMethod]
        public void SearchSorted_ThePrefixBreakerInTheMiddle_IsSeen()
        {
            var rng = new System.Random(5);
            var clean = Random(rng, 400, 3, "abc").Select(s => "https://site.example/" + s).ToArray();
            var keys = Random(rng, 300, 3, "abcd").Select(s => "https://site.example/" + s).ToArray();
            // The breaker in the middle of an unsorted haystack only (its first and last elements and every key share the
            // prefix): only the scan of the haystack's middle sees it.
            var hay = (string[])clean.Clone();
            hay[200] = "https://site.examplf/";
            foreach (bool left in new[] { true, false })
            {
                WithRoutes(false, false, true, false, false, () =>
                    Ints(np.searchsorted(U(hay), U(keys), left ? "left" : "right"))
                        .Should().Equal(NumPySearch(hay, keys, left), $"left={left}"));
            }
            // The breaker in the middle of the keys only (a sorted, clean haystack).
            var sortedHay = clean.OrderBy(v => v, StringComparer.Ordinal).ToArray();
            var keys2 = (string[])keys.Clone();
            keys2[150] = "https://site.examplf/b";
            WithRoutes(false, false, true, false, false, () =>
                Ints(np.searchsorted(U(sortedHay), U(keys2))).Should().Equal(NumPySearch(sortedHay, keys2, true)));
        }

        /// <summary>
        ///     Short rows whose keys tie BEFORE the row's last element: the elements tied on the extreme key are decided by
        ///     the units after the key (here the 9th of nine), whatever comes after them in the row.
        /// </summary>
        [TestMethod]
        public void ShortRows_TiesBeforeTheEnd_AreDecidedByTheTail()
        {
            foreach (var kind in new[] { "U9", "S9" })
            {
                var rows = U("aaaaaaaab", "aaaaaaaac", "a", "zzzzzzzzb", "zzzzzzzza", "b", "m", "aaaaaaaaz", "aaaaaaaay")
                    .astype(np.dtype(kind)).reshape(3, 3);
                // argmax per row: 'aaaaaaaac' (1), 'zzzzzzzzb' (0), 'm' (0); argmin: 'a' (2), 'b' (2), 'aaaaaaaay' (2)
                Ints(np.argmax(rows, 1)).Should().Equal(1L, 0L, 0L);
                Ints(np.argmin(rows, 1)).Should().Equal(2L, 2L, 2L);
            }
        }

        /// <summary>
        ///     Two block winners that differ in a unit's HIGH byte (U+0101 against U+0201) with the next units ordered the
        ///     other way: the comparison reads the whole first differing unit, not four bytes from the differing byte.
        /// </summary>
        [TestMethod]
        public void ArgReductions_UnitsDifferingInAHighByte_CompareWhole()
        {
            var values = Enumerable.Repeat("a", 2000).ToArray();
            values[10] = "āz";
            values[1500] = "ȁa";
            np.argmax(U(values)).Should().Be(1500L);
            var hay = U("āz", "ȁa");
            Ints(np.searchsorted(hay, U("ȁ\u0001", "ā\u007f", "ǿz"))).Should().Equal(1L, 1L, 1L);
        }

        /// <summary>
        ///     A haystack sorted but for ONE descent, at every position modulo the vector check's stride: the SORTED route
        ///     must refuse it, so NumPy's exact path answers.
        /// </summary>
        [TestMethod]
        public void SearchSorted_OneDescent_IsNotSorted()
        {
            var rng = new System.Random(9);
            var sorted = Random(rng, 64, 4, "abcdefgh", 4).Distinct().OrderBy(v => v, StringComparer.Ordinal).ToArray();
            var keys = Random(rng, 200, 4, "abcdefgh", 4);
            for (int p = 0; p + 1 < sorted.Length; p++)
            {
                var hay = (string[])sorted.Clone();
                (hay[p], hay[p + 1]) = (hay[p + 1], hay[p]);
                Ints(np.searchsorted(U(hay), U(keys))).Should().Equal(NumPySearch(hay, keys, true), $"descent at {p}");
            }
        }

        /// <summary>
        ///     Keys that end before the strings, with equal keys whose tails DESCEND: the haystack is not sorted (its sortedness
        ///     check compares the tails of equal keys), so NumPy's exact path answers.
        /// </summary>
        [TestMethod]
        public void SearchSorted_EqualKeysWithDescendingTails_AreNotSorted()
        {
            foreach (var kind in new[] { "U12", "S12" })
            {
                var hay = new[] { "aaaaaaaa", "bbbbbbbbz", "bbbbbbbba", "cccccccc", "dddddddd" };
                var keys = new[] { "bbbbbbbbm", "bbbbbbbbz", "c", "bbbbbbbba", "e" };
                WithRoutes(false, false, true, false, false, () =>
                {
                    Ints(np.searchsorted(U(hay).astype(np.dtype(kind)), U(keys).astype(np.dtype(kind))))
                        .Should().Equal(NumPySearch(hay, keys, true));
                    Ints(np.searchsorted(U(hay).astype(np.dtype(kind)), U(keys).astype(np.dtype(kind)), "right"))
                        .Should().Equal(NumPySearch(hay, keys, false));
                });
            }
        }

        /// <summary>
        ///     Nine-unit strings tied on their first eight units: an 8-unit key does not reach the 9th unit, so equal keys
        ///     defer to it (sorted and unsorted haystacks, keys sorted and not).
        /// </summary>
        [TestMethod]
        public void SearchSorted_KeysOneUnitShort_DeferToTheLastUnit()
        {
            var rng = new System.Random(3);
            var hay = Enumerable.Range(0, 300).Select(_ => "aaaaaaaa" + "abc"[rng.Next(3)]).ToArray();
            var keys = Enumerable.Range(0, 100).Select(_ => "aaaaaaaa" + "abcd"[rng.Next(4)]).ToArray();
            var sortedHay = hay.OrderBy(v => v, StringComparer.Ordinal).ToArray();
            hay[0] = "b";
            foreach (var kind in new[] { "U9", "S9" })
            {
                foreach (var h in new[] { sortedHay, hay })
                {
                    foreach (var k in new[] { keys, keys.OrderBy(v => v, StringComparer.Ordinal).ToArray() })
                    {
                        WithRoutes(false, true, true, false, false, () =>
                            Ints(np.searchsorted(U(h).astype(np.dtype(kind)), U(k).astype(np.dtype(kind))))
                                .Should().Equal(NumPySearch(h, k, true)));
                    }
                }
            }
        }

        // ------------------------------------------------------------------ fenced memory

        /// <summary>
        ///     Every route reads only its operands' bytes: arrays laid against an inaccessible page (at the start and at the end
        ///     of the readable span) — <c>U8</c> (one masked or vector load an element), <c>U9</c>, <c>U3</c>, <c>U1</c>,
        ///     <c>S8</c>, <c>S5</c> — arg-reduced flat, along both axes and as short rows, and searched as haystack and keys on
        ///     every route.
        /// </summary>
        [TestMethod]
        public void Searches_ReadNothingOutsideTheirOperands()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("FencedMemory needs Windows (reserved pages).");
            var rng = new System.Random(64);
            foreach (var dtype in new[] { "U8", "U9", "U3", "U1", "S8", "S5" })
            {
                var d = np.dtype(dtype);
                int chars = dtype[0] == 'U' ? d.itemsize / 4 : d.itemsize;
                foreach (var shape in new[] { new long[] { 1100 }, new long[] { 17 }, new long[] { 60, 10 } })
                {
                    long count = shape.Aggregate(1L, (x, y) => x * y);
                    var values = Random(rng, (int)count, chars, "abc");
                    var data = U(values).astype(d).reshape(new Shape(shape));
                    long bytes = count * d.itemsize;
                    var keys = U(Random(rng, 300, chars, "abcd")).astype(d);
                    var sortedData = np.sort(data.flatten());
                    foreach (bool atEnd in new[] { false, true })
                    {
                        using var fence = new FencedMemory(pages: (int)((bytes + FencedMemory.PageBytes - 1) / FencedMemory.PageBytes) + 1);
                        byte* p = fence.Place(bytes, atEnd);
                        using var fenced = np.frombuffer((IntPtr)p, bytes, d).reshape(new Shape(shape));
                        foreach (var (gatherOff, prefixOff) in new[] { (false, false), (true, false), (false, true) })
                        {
                            WithRoutes(gatherOff, prefixOff, null, false, false, () =>
                            {
                                np.copyto(fenced, data);
                                np.argmax(fenced).Should().Be(np.argmax(data.copy()));
                                np.argmin(fenced).Should().Be(np.argmin(data.copy()));
                                if (shape.Length == 2)
                                {
                                    Ints(np.argmax(fenced, 1)).Should().Equal(Ints(np.argmax(data.copy(), 1)));
                                    Ints(np.argmin(fenced, 0)).Should().Equal(Ints(np.argmin(data.copy(), 0)));
                                }
                            });
                        }
                        if (shape.Length == 1)
                        {
                            np.copyto(fenced, sortedData);
                            foreach (var forced in new bool?[] { false, true })
                            {
                                foreach (bool sortedOff in new[] { false, true })
                                {
                                    WithRoutes(false, false, forced, sortedOff, false, () =>
                                    {
                                        Ints(np.searchsorted(fenced, keys)).Should().Equal(Ints(np.searchsorted(sortedData.copy(), keys)));
                                        Ints(np.searchsorted(fenced, keys, "right")).Should().Equal(Ints(np.searchsorted(sortedData.copy(), keys, "right")));
                                    });
                                }
                            }
                        }
                    }
                    // The keys fenced too: the haystack copied, the keys laid against the inaccessible page.
                    long keyBytes = keys.size * d.itemsize;
                    foreach (bool atEnd in new[] { false, true })
                    {
                        using var fence = new FencedMemory(pages: (int)((keyBytes + FencedMemory.PageBytes - 1) / FencedMemory.PageBytes) + 1);
                        byte* p = fence.Place(keyBytes, atEnd);
                        using var fencedKeys = np.frombuffer((IntPtr)p, keyBytes, d);
                        np.copyto(fencedKeys, keys);
                        foreach (var forced in new bool?[] { false, true })
                        {
                            WithRoutes(false, false, forced, false, false, () =>
                                Ints(np.searchsorted(sortedData, fencedKeys)).Should().Equal(Ints(np.searchsorted(sortedData, keys.copy()))));
                        }
                    }
                }
            }
        }
    }
}
