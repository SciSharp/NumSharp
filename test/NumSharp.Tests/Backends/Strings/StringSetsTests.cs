using System;
using System.Linq;
using NumSharp.Backends.Sorting;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     Strings phase 2, P2.2 milestone C (docs/plans/strings-phase2.md): <c>np.unique</c> with every flag and axis, the
    ///     Array-API unique family, <c>np.isin</c>, <c>intersect1d</c> / <c>union1d</c> / <c>setdiff1d</c> /
    ///     <c>setxor1d</c>, <c>np.lexsort</c>, <c>np.sort_complex</c>, <c>np.histogram</c> /
    ///     <c>np.histogram_bin_edges</c> and the take path of <c>np.quantile</c> / <c>np.percentile</c> over <c>S</c> /
    ///     <c>U</c> text. Every expectation is a NumPy 2.4.2 literal (probed), one representative of each rule; the string
    ///     oracle's <c>strsets</c> section replays the full matrix. The functions are NumPy's statements over the sort and
    ///     search engines of milestones A and B, so the reads past an operand those engines could make are checked here
    ///     against fenced pages too.
    /// </summary>
    [TestClass]
    public unsafe class StringSetsTests
    {
        // ------------------------------------------------------------------ helpers

        /// <summary>A <c>U</c> array of the given strings (as wide as the longest).</summary>
        /// <param name="values">The strings.</param>
        /// <returns>The array.</returns>
        private static NDArray U(params string[] values) => np.array(values);

        /// <summary>An <c>S</c> array of the given byte strings (as wide as the longest).</summary>
        /// <param name="values">The byte strings.</param>
        /// <returns>The array.</returns>
        private static NDArray S(params byte[][] values) => np.array(values.Cast<object>().ToArray(), dtype: "S");

        /// <summary>The int64 elements of an index array, in C order.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The values.</returns>
        private static long[] Ints(NDArray a) => a.ToArray<long>();

        /// <summary>The elements of a <c>U</c> array, in C order.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The strings.</returns>
        private static string[] Texts(NDArray a) => NDArray.AsStringArray(a);

        /// <summary>The elements of a 1-D <c>S</c> array as Latin-1 strings (trailing NULs removed, as item() reads them).</summary>
        /// <param name="a">The array.</param>
        /// <returns>One string per element, each byte one character.</returns>
        private static string[] ByteTexts(NDArray a)
            => Enumerable.Range(0, (int)a.size).Select(i => new string(((byte[])a.item(i)).Select(b => (char)b).ToArray())).ToArray();

        // ------------------------------------------------------------------ np.unique

        /// <summary>
        ///     np.unique(['b', 'a\x00b', 'a', 'b', '', 'a'], return_index, return_inverse, return_counts) is
        ///     (['' 'a' 'a\x00b' 'b'], [4 2 1 0], [3 2 1 3 0 1], [1 2 1 2]): an embedded NUL is text, the FIRST occurrence
        ///     gives each index (NumPy's mergesort argsort).
        /// </summary>
        [TestMethod]
        public void Unique_EveryFlag_FirstOccurrence_EmbeddedNul()
        {
            var r = (NDArray[])np.unique(U("b", "a\0b", "a", "b", "", "a"), true, true, true);
            r[0].dtype.str.Should().Be("<U3");
            Texts(r[0]).Should().Equal("", "a", "a\0b", "b");
            Ints(r[1]).Should().Equal(4L, 2L, 1L, 0L);
            Ints(r[2]).Should().Equal(3L, 2L, 1L, 3L, 0L, 1L);
            Ints(r[3]).Should().Equal(1L, 2L, 1L, 2L);
        }

        /// <summary>np.unique of <c>S</c> text orders bytes unsigned: [b'\x80', b'z', b'a', b'\x80', b'\xff'] gives
        ///     [b'a' b'z' b'\x80' b'\xff'] with indices [2 1 0 4].</summary>
        [TestMethod]
        public void Unique_Bytes_OrderedUnsigned()
        {
            var r = (NDArray[])np.unique(S(new byte[] { 0x80 }, new byte[] { (byte)'z' }, new byte[] { (byte)'a' },
                                            new byte[] { 0x80 }, new byte[] { 0xff }), return_index: true);
            r[0].dtype.str.Should().Be("|S1");
            ByteTexts(r[0]).Should().Equal("a", "z", "", "ÿ");
            Ints(r[1]).Should().Equal(2L, 1L, 0L, 4L);
        }

        /// <summary>
        ///     np.unique along an axis compares whole rows (NumPy's structured view): [['b','a','b'],['a','b','a'],
        ///     ['b','a','b']] along axis 0 is [['a','b','a'],['b','a','b']] with index [1 0], inverse [1 0 1], counts [1 2];
        ///     along axis 1 the columns ['b','a','b'] / ['a','b','a'] the same way.
        /// </summary>
        [TestMethod]
        public void Unique_AlongAnAxis_ComparesRows()
        {
            var m = np.array(new[,] { { "b", "a", "b" }, { "a", "b", "a" }, { "b", "a", "b" } });
            var rows = (NDArray[])np.unique(m, true, true, true, 0);
            rows[0].shape.Should().Equal(new long[] { 2, 3 });
            Texts(rows[0]).Should().Equal("a", "b", "a", "b", "a", "b");
            Ints(rows[1]).Should().Equal(1L, 0L);
            Ints(rows[2]).Should().Equal(1L, 0L, 1L);
            Ints(rows[3]).Should().Equal(1L, 2L);
            var cols = (NDArray[])np.unique(m, true, true, true, 1);
            cols[0].shape.Should().Equal(new long[] { 3, 2 });
            Texts(cols[0]).Should().Equal("a", "b", "b", "a", "a", "b");
            Ints(cols[1]).Should().Equal(1L, 0L);
            Ints(cols[2]).Should().Equal(1L, 0L, 1L);
            Ints(cols[3]).Should().Equal(1L, 2L);
        }

        /// <summary>
        ///     The Array-API family over 2-D text: unique_all is (['a' 'b'], [1 0], [[1 0 1] [0 1 0] [1 0 1]], [4 5]) — the
        ///     inverse in the input's shape; unique_values returns the set SORTED where NumPy returns its hash order
        ///     (['pear' 'apple' 'fig'] for ['pear','fig','apple','fig']), the documented order divergence.
        /// </summary>
        [TestMethod]
        public void UniqueFamily_InverseKeepsTheInputShape_ValuesSorted()
        {
            var m = np.array(new[,] { { "b", "a", "b" }, { "a", "b", "a" }, { "b", "a", "b" } });
            var all = np.unique_all(m);
            Texts(all.values).Should().Equal("a", "b");
            Ints(all.indices).Should().Equal(1L, 0L);
            all.inverse_indices.shape.Should().Equal(new long[] { 3, 3 });
            Ints(all.inverse_indices).Should().Equal(1L, 0L, 1L, 0L, 1L, 0L, 1L, 0L, 1L);
            Ints(all.counts).Should().Equal(4L, 5L);
            Texts(np.unique_values(U("pear", "fig", "apple", "fig"))).Should().Equal("apple", "fig", "pear");
        }

        /// <summary>
        ///     The values-only route hashes like NumPy while few strings are distinct (TextDistinct: the distinct strings,
        ///     then a sort of only those) and sorts every string once too many are; both routes give the same values. Every
        ///     tail width of the hash (S1 … S9: 1 to 7 bytes after the 8-byte steps; U widths: 4 bytes) over NUL-holding and
        ///     high-byte data, few distinct values (the hash route, proven by its counter) and mostly distinct ones (its
        ///     bailout, counted too).
        /// </summary>
        [TestMethod]
        public void Unique_HashRoute_MatchesTheSortRoute()
        {
            var rng = new System.Random(19);
            string alphabet = "ab\0ÿéz";
            foreach (var dtype in new[] { "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S15", "U1", "U2", "U3", "U5" })
            {
                var d = np.dtype(dtype);
                int chars = dtype[0] == 'U' ? d.itemsize / 4 : d.itemsize;
                foreach (var (count, distinct) in new[] { (64, 3), (500, 40), (5000, 200), (5000, 5000), (100, 100) })
                {
                    var pool = Enumerable.Range(0, distinct)
                        .Select(_ => new string(Enumerable.Range(0, rng.Next(0, chars + 1)).Select(_ => alphabet[rng.Next(alphabet.Length)]).ToArray()))
                        .ToArray();
                    var values = Enumerable.Range(0, count).Select(_ => pool[rng.Next(pool.Length)]).ToArray();
                    // A U array of the values, cast to the target width (the latin-1 code points fit S as bytes).
                    var data = dtype[0] == 'U'
                        ? U(values).astype(d)
                        : np.array(values.Select(s => (object)s.Select(c => (byte)c).ToArray()).ToArray(), dtype: "S").astype(d);
                    long runs = TextDistinct.Runs, bailouts = TextDistinct.Bailouts;
                    var hashed = (NDArray)np.unique(data);
                    bool hashRan = TextDistinct.Runs > runs, bailed = TextDistinct.Bailouts > bailouts;
                    NDArray sorted;
                    TextDistinct.Disabled = true;
                    try { sorted = (NDArray)np.unique(data); }
                    finally { TextDistinct.Disabled = false; }
                    hashed.dtype.Should().Be(sorted.dtype);
                    np.array_equal(hashed, sorted).Should().BeTrue($"{dtype} n={count} distinct={distinct}");
                    // The distinct count decides the route: at most max(256, n / 16) distinct strings complete the hash pass.
                    if (sorted.size <= TextDistinct.Limit(count))
                        hashRan.Should().BeTrue($"{dtype} n={count}: {sorted.size} distinct strings take the hash route");
                    else
                        bailed.Should().BeTrue($"{dtype} n={count}: {sorted.size} distinct strings give up the hash route");
                }
            }
        }

        // ------------------------------------------------------------------ np.isin

        /// <summary>
        ///     _isin's two paths decide text against numbers: with fewer than <c>10 * len(ar1) ** 0.145</c> test elements
        ///     NumPy loops <c>ar1 == a</c>, which has no loop for text and numbers (never equal: np.isin(['1'],
        ///     np.arange(9)) is [False]); from that count on it sorts the concatenation, which writes the numbers as text
        ///     (np.isin(['1'], np.arange(10)) is [True]). <c>S</c> against <c>U</c> the same way: [False False] on the loop,
        ///     [True] on the sort path (the bytes decoded).
        /// </summary>
        [TestMethod]
        public void Isin_LoopAndSortPaths_DecideTextAgainstNumbers()
        {
            np.isin(U("1"), np.arange(9)).ToArray<bool>().Should().Equal(false);
            np.isin(U("1"), np.arange(10)).ToArray<bool>().Should().Equal(true);
            np.isin(U("a", "b"), S(new[] { (byte)'a' })).ToArray<bool>().Should().Equal(false, false);
            np.isin(U("a"), S(Enumerable.Repeat(new[] { (byte)'a' }, 12).ToArray())).ToArray<bool>().Should().Equal(true);
        }

        /// <summary>
        ///     assume_unique with duplicates is NumPy's garbage-in answer, reproduced exactly: np.isin(['b','a','b','c','a'],
        ///     ['a','b']*6, assume_unique=True) is [T T T F T], with invert [F F F T F].
        /// </summary>
        [TestMethod]
        public void Isin_AssumeUniqueWithDuplicates_IsNumPysExactAnswer()
        {
            var x = U("b", "a", "b", "c", "a");
            var y = U(Enumerable.Range(0, 12).Select(i => i % 2 == 0 ? "a" : "b").ToArray());
            np.isin(x, y, assume_unique: true).ToArray<bool>().Should().Equal(true, true, true, false, true);
            np.isin(x, y, assume_unique: true, invert: true).ToArray<bool>().Should().Equal(false, false, false, true, false);
        }

        // ------------------------------------------------------------------ set routines

        /// <summary>
        ///     intersect1d(['b','a','b','c'], ['c','b','c'], return_indices=True) is (['b' 'c'], [0 3], [1 0]): the first
        ///     occurrences; with assume_unique over (['b','a','c'], ['c','b']) the indices are [0 2] and [1 0].
        /// </summary>
        [TestMethod]
        public void Intersect1d_ReturnIndices_FirstOccurrences()
        {
            var r = np.intersect1d(U("b", "a", "b", "c"), U("c", "b", "c"), return_indices: true);
            Texts(r[0]).Should().Equal("b", "c");
            Ints(r[1]).Should().Equal(0L, 3L);
            Ints(r[2]).Should().Equal(1L, 0L);
            var au = np.intersect1d(U("b", "a", "c"), U("c", "b"), assume_unique: true, return_indices: true);
            Ints(au[1]).Should().Equal(0L, 2L);
            Ints(au[2]).Should().Equal(1L, 0L);
        }

        /// <summary>
        ///     setdiff1d keeps ar1's order and duplicates under assume_unique (['b' 'a' 'b'] for ['b','a','b','c'] minus
        ///     ['c']) and sorts the unique values otherwise (['a' 'b']); setxor1d is ['a' 'c']; union1d promotes the widths
        ///     (['a' 'abc' 'abcde'], &lt;U5) and decodes bytes, raising NumPy's UnicodeDecodeError for a byte past ASCII.
        /// </summary>
        [TestMethod]
        public void SetRoutines_OrderWidthsAndDecoding()
        {
            Texts(np.setdiff1d(U("b", "a", "b", "c"), U("c"), assume_unique: true)).Should().Equal("b", "a", "b");
            Texts(np.setdiff1d(U("b", "a", "b", "c"), U("c"))).Should().Equal("a", "b");
            Texts(np.setxor1d(U("b", "a", "b"), U("c", "b"))).Should().Equal("a", "c");
            var union = np.union1d(U("abc", "a"), U("abcde"));
            union.dtype.str.Should().Be("<U5");
            Texts(union).Should().Equal("a", "abc", "abcde");
            Texts(np.union1d(U("b"), S(new[] { (byte)'a' }))).Should().Equal("a", "b");
            Action bad = () => np.union1d(U("b"), S(new byte[] { 0xff }));
            bad.Should().Throw<UnicodeDecodeError>()
                .WithMessage("'ascii' codec can't decode byte 0xff in position 0: ordinal not in range(128)");
        }

        // ------------------------------------------------------------------ np.lexsort / np.sort_complex

        /// <summary>
        ///     np.lexsort((['b','a','b','a'], [1,1,0,0])) is [3 2 1 0] (the LAST key primary, text and numbers mixed); a 2-D
        ///     text key along axis 0 is [[1 0] [0 1]].
        /// </summary>
        [TestMethod]
        public void Lexsort_TextAndMixedKeys()
        {
            Ints(np.lexsort(new[] { U("b", "a", "b", "a"), np.array(new long[] { 1, 1, 0, 0 }) })).Should().Equal(3L, 2L, 1L, 0L);
            var r = np.lexsort(new[] { np.array(new[,] { { "b", "a" }, { "a", "b" } }) }, 0);
            Ints(r).Should().Equal(1L, 0L, 0L, 1L);
        }

        /// <summary>
        ///     np.sort_complex sorts the TEXT, then parses it: ['10','9','2+1j'] is [10+0j, 2+1j, 9+0j] (the text's order);
        ///     a malformed element raises NumPy's ValueError after the sort; <c>S</c> text parses the same way.
        /// </summary>
        [TestMethod]
        public void SortComplex_SortsTheTextThenParses()
        {
            var r = np.sort_complex(U("10", "9", "2+1j"));
            r.dtype.Should().Be(np.complex128);
            r.ToArray<System.Numerics.Complex>().Should().Equal(new System.Numerics.Complex(10, 0),
                new System.Numerics.Complex(2, 1), new System.Numerics.Complex(9, 0));
            Action bad = () => np.sort_complex(U("1", "x", "2"));
            bad.Should().Throw<ValueError>().WithMessage("complex() arg is a malformed string");
            np.sort_complex(S(new[] { (byte)'3' }, "1+1j"u8.ToArray(), "-2"u8.ToArray()))
                .ToArray<System.Numerics.Complex>().Should().Equal(new System.Numerics.Complex(-2, 0),
                    new System.Numerics.Complex(1, 1), new System.Numerics.Complex(3, 0));
        }

        // ------------------------------------------------------------------ np.histogram

        /// <summary>
        ///     Number edges are searched as text: histogram(['1','5','10','30','2'], [0, 5, 10]) is [4 -2] — a NEGATIVE count,
        ///     NumPy's answer ('10' &lt; '5' as text); text edges with weights sum them ([0.2 0.8]); number data with text
        ///     edges are searched as their text ([1 4]).
        /// </summary>
        [TestMethod]
        public void Histogram_EdgesSearchedAsText()
        {
            var neg = np.histogram(U("1", "5", "10", "30", "2"), np.array(new long[] { 0, 5, 10 }));
            Ints(neg.hist).Should().Equal(4L, -2L);
            var w = np.histogram(U("b", "a", "c", "bb"), U("a", "b", "c"), weights: np.array(new[] { 0.1, 0.2, 0.3, 0.4 }));
            w.hist.ToArray<double>().Should().Equal(0.2, 0.8);
            Texts(w.bin_edges).Should().Equal("a", "b", "c");
            Ints(np.histogram(np.array(new long[] { 1, 2, 3, 10, 20 }), U("1", "2", "3")).hist).Should().Equal(1L, 4L);
            Texts(np.histogram_bin_edges(U("a", "b"), U("a", "c"))).Should().Equal("a", "c");
        }

        /// <summary>
        ///     NumPy's cumulative loop runs per 65,536-sample block: ['a','b','c'] * 21846 (65,538 samples) is [21846 43692],
        ///     and with int64 weights (i % 3) [0 65538]; bool weights keep a bool running total, whose in-place add of the
        ///     int64 cumsum NumPy refuses (<c>Cannot cast ufunc 'add' output from dtype('int64') to dtype('bool') with
        ///     casting rule 'same_kind'</c>).
        /// </summary>
        [TestMethod]
        public void Histogram_BlocksAndWeightKinds()
        {
            var data = U(Enumerable.Range(0, 65538).Select(i => i % 3 == 0 ? "a" : i % 3 == 1 ? "b" : "c").ToArray());
            var edges = U("a", "b", "c");
            Ints(np.histogram(data, edges).hist).Should().Equal(21846L, 43692L);
            Ints(np.histogram(data, edges, weights: (np.arange(65538) % 3).astype(np.int64)).hist).Should().Equal(0L, 65538L);
            Action boolWeights = () => np.histogram(U("a", "b"), U("a", "c"), weights: np.array(new[] { true, false }));
            boolWeights.Should().Throw<Exception>()
                .WithMessage("Cannot cast ufunc 'add' output from dtype('int64') to dtype('bool') with casting rule 'same_kind'");
        }

        // ------------------------------------------------------------------ np.quantile / np.percentile

        /// <summary>
        ///     The take-path methods over ['d','b','a','c','e'] at q = [0, .2, .25, .5, .75, .9, 1]: lower
        ///     [a a b c d d e], higher [a b b c d e e], nearest [a b b c d e e], inverted_cdf [a a b c d e e],
        ///     closest_observation [a a a b d d e].
        /// </summary>
        [TestMethod]
        public void Quantile_SelectionMethods()
        {
            var t = U("d", "b", "a", "c", "e");
            var q = new[] { 0.0, 0.2, 0.25, 0.5, 0.75, 0.9, 1.0 };
            Texts(np.quantile(t, q, method: "lower")).Should().Equal("a", "a", "b", "c", "d", "d", "e");
            Texts(np.quantile(t, q, method: "higher")).Should().Equal("a", "b", "b", "c", "d", "e", "e");
            Texts(np.quantile(t, q, method: "nearest")).Should().Equal("a", "b", "b", "c", "d", "e", "e");
            Texts(np.quantile(t, q, method: "inverted_cdf")).Should().Equal("a", "a", "b", "c", "d", "e", "e");
            Texts(np.quantile(t, q, method: "closest_observation")).Should().Equal("a", "a", "a", "b", "d", "d", "e");
        }

        /// <summary>
        ///     NumPy 2.4's integer q: <c>q = np.asanyarray(q)</c> keeps a Python int an integer, so 'linear' TAKES the element
        ///     at (n - 1) * q — over text ('e' for q = 1) and over floats too (np.quantile([1., inf], 1) is inf, where the
        ///     float q 1.0 interpolates inf - inf into NaN); a float q interpolates text into NumPy's TypeError.
        /// </summary>
        [TestMethod]
        public void Quantile_IntegerQ_TakesTheElement()
        {
            var t = U("d", "b", "a", "c", "e");
            ((string)np.quantile(t, 1).item()).Should().Be("e");
            double.IsPositiveInfinity(np.quantile(np.array(new[] { 1.0, double.PositiveInfinity }), 1).GetDouble()).Should().BeTrue();
            double.IsNaN(np.quantile(np.array(new[] { 1.0, double.PositiveInfinity }), 1.0).GetDouble()).Should().BeTrue();
            Ints(np.quantile(np.array(new long[] { 3, 1, 2 }), np.array(new long[] { 0, 1 }))).Should().Equal(1L, 3L);
            Action lerp = () => np.quantile(t, 0.5);
            lerp.Should().Throw<TypeError>().WithMessage("unsupported operand type(s) for -: 'numpy.str_' and 'numpy.str_'");
        }

        /// <summary>
        ///     NumPy 2.4's <c>weak_q = type(q) in (int, float)</c>: a C# number keeps float32 data's dtype (2.5 as float32),
        ///     a 0-d ARRAY q is a strong float64 (2.5 as float64).
        /// </summary>
        [TestMethod]
        public void Quantile_WeakQ_KeepsFloat32_ArrayQPromotes()
        {
            var f32 = np.array(new float[] { 3, 1, 4, 2 });
            np.quantile(f32, 0.5).dtype.Should().Be(np.float32);
            np.quantile(f32, np.array(0.5)).dtype.Should().Be(np.float64);
            np.quantile(f32, np.array(0.5)).GetDouble().Should().Be(2.5);
        }

        /// <summary>
        ///     NumPy's errors in its order: keepdims over a 0-d text result indexes NumPy's scalar (<c>string indices must be
        ///     integers, not 'tuple'</c> for <c>U</c>, <c>byte indices must be integers or slices, not tuple</c> for <c>S</c>);
        ///     a 0-d out that does not cast safely is a "scalar"; several reduced axes beside a kept axis of length 0 cannot
        ///     merge; an interpolating method over an empty line reports its first neighbour (-1 for linear at q = 0 and
        ///     weibull at q = 1, 0 otherwise); a selection method its take's index (inverted_cdf clips -1 to 0).
        /// </summary>
        [TestMethod]
        public void Quantile_ErrorsInNumPysOrder()
        {
            var t = U("d", "b", "a", "c", "e");
            Action kdU = () => np.quantile(t, 0.5, method: "lower", keepdims: true);
            kdU.Should().Throw<TypeError>().WithMessage("string indices must be integers, not 'tuple'");
            Action kdS = () => np.quantile(t.astype("S"), 0.5, method: "lower", keepdims: true);
            kdS.Should().Throw<TypeError>().WithMessage("byte indices must be integers or slices, not tuple");
            Action out0d = () => np.quantile(t, 0.5, method: "lower", @out: np.zeros(new Shape(), np.dtype("U4")));
            out0d.Should().Throw<TypeError>().WithMessage("Cannot cast scalar from dtype('<U4') to dtype('<U1') according to the rule 'safe'");
            var o = np.zeros(new Shape(), np.dtype("U1"));
            ReferenceEquals(np.quantile(t, 0.5, method: "lower", @out: o), o).Should().BeTrue();
            ((string)o.item()).Should().Be("c");
            Action merge = () => np.quantile(np.zeros(new Shape(0, 2, 3), np.dtype("U1")), 0.5, new[] { 1, 2 }, method: "lower");
            merge.Should().Throw<IncorrectShapeException>().WithMessage("cannot reshape array of size 0 into shape (0,newaxis)");
            var e0 = np.zeros(new Shape(0), np.dtype("U1"));
            Action lin0 = () => np.quantile(e0, 0.0);
            lin0.Should().Throw<IndexError>().WithMessage("index -1 is out of bounds for axis 0 with size 0");
            Action weibull1 = () => np.quantile(e0, 1.0, method: "weibull");
            weibull1.Should().Throw<IndexError>().WithMessage("index -1 is out of bounds for axis 0 with size 0");
            var z = np.zeros(new Shape(0, 0), np.dtype("U1"));
            Action icdf = () => np.quantile(z, 1, 1, method: "inverted_cdf");
            icdf.Should().Throw<IndexError>().WithMessage("index 0 is out of bounds for axis 0 with size 0");
            Action lower = () => np.quantile(z, 1, 1, method: "lower");
            lower.Should().Throw<IndexError>().WithMessage("index -1 is out of bounds for axis 0 with size 0");
            Action bad = () => np.quantile(t, 0.5, method: "foo");
            bad.Should().Throw<ValueError>().WithMessage("'foo' is not a valid method. Use one of: dict_keys(*");
        }

        // ------------------------------------------------------------------ fenced reads

        /// <summary>
        ///     The routines read their operands only inside them: every call over text placed against an inaccessible page
        ///     (below and above) returns the answer of the same call over a copy.
        /// </summary>
        [TestMethod]
        public void SetRoutines_ReadNothingOutsideTheirOperands()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("FencedMemory needs Windows (reserved pages).");
            var rng = new System.Random(71);
            foreach (var dtype in new[] { "U3", "U1", "S3", "S1", "S5", "S7", "S9" })
            {
                var d = np.dtype(dtype);
                int chars = dtype[0] == 'U' ? d.itemsize / 4 : d.itemsize;
                foreach (int count in new[] { 1, 17, 1100 })
                {
                    var values = Enumerable.Range(0, count)
                        .Select(_ => new string(Enumerable.Range(0, rng.Next(0, chars + 1)).Select(_ => "abc"[rng.Next(3)]).ToArray()))
                        .ToArray();
                    var data = U(values).astype(d);
                    long bytes = count * (long)d.itemsize;
                    var other = U(values.Take(5).Concat(new[] { "zz" }).ToArray()).astype(d);
                    foreach (bool atEnd in new[] { false, true })
                    {
                        using var fence = new FencedMemory(pages: (int)((bytes + FencedMemory.PageBytes - 1) / FencedMemory.PageBytes) + 1);
                        byte* p = fence.Place(bytes, atEnd);
                        using var fenced = np.frombuffer((IntPtr)p, bytes, d);
                        np.copyto(fenced, data);
                        var copy = data.copy();
                        // The values-only route (the hash pass from 64 elements on) and the flags route (the stable sort).
                        np.array_equal((NDArray)np.unique(fenced), (NDArray)np.unique(copy)).Should().BeTrue();
                        var a = (NDArray[])np.unique(fenced, true, true, true);
                        var b = (NDArray[])np.unique(copy, true, true, true);
                        for (int i = 0; i < 4; i++)
                            np.array_equal(a[i], b[i]).Should().BeTrue();
                        np.array_equal(np.isin(fenced, other), np.isin(copy, other)).Should().BeTrue();
                        np.array_equal(np.isin(other, fenced), np.isin(other, copy)).Should().BeTrue();
                        np.array_equal(np.setxor1d(fenced, other), np.setxor1d(copy, other)).Should().BeTrue();
                        np.array_equal(np.lexsort(new[] { fenced }), np.lexsort(new[] { copy })).Should().BeTrue();
                        np.array_equal(np.quantile(fenced, new[] { 0.0, 0.5, 1.0 }, method: "nearest"),
                                       np.quantile(copy, new[] { 0.0, 0.5, 1.0 }, method: "nearest")).Should().BeTrue();
                    }
                }
            }
        }
    }
}
