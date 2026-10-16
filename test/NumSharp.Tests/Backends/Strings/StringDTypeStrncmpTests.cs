using System;
using System.Linq;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     NEP 55 <c>StringDType</c> compares through NumPy's <c>NpyString_cmp</c>, which is <c>strncmp</c> over the shorter
    ///     length and then the sizes: <c>strncmp</c> stops at the first byte that differs OR at a NUL both texts share, so
    ///     <c>'a\0c'</c> and <c>'a\0b'</c> are EQUAL (same bytes up to the shared NUL, same size) while <c>'a\0'</c> is smaller
    ///     than <c>'ab'</c> (the NUL meets <c>'b'</c>). Every family that orders or matches text follows it — the comparisons,
    ///     the sorts, <c>np.unique</c>, the extremes, the searches, the set routines, <c>np.array_equal</c> — and fixed-width
    ///     <c>U</c> text does not (it compares every code point). Every expected value here was probed against
    ///     <c>numpy==2.4.2</c>; the string oracle's randomized soak holds the same contract over random texts and layouts.
    /// </summary>
    [TestClass]
    public class StringDTypeStrncmpTests
    {
        /// <summary>A fresh <c>StringDType()</c> descriptor.</summary>
        private static DType T => np.dtypes.StringDType();

        /// <summary>A fresh <c>StringDType(na_object=nan)</c> descriptor.</summary>
        private static DType TNaN => np.dtypes.StringDType(na_object: double.NaN);

        /// <summary>A fresh <c>StringDType(na_object=None)</c> descriptor.</summary>
        private static DType TNone => np.dtypes.StringDType(na_object: null);

        /// <summary>The six texts NumPy's probe orders: two tied pairs and the NUL-against-a-byte case.</summary>
        private static readonly object[] Six = { "a\0c", "a\0b", "a", "a\0", "a\0bb", "ab" };

        /// <summary>A 1-D array of the given elements in the given descriptor.</summary>
        /// <param name="d">The descriptor.</param>
        /// <param name="values">The elements (texts and NA objects).</param>
        /// <returns>The new array.</returns>
        private static NDArray A(DType d, params object[] values) => np.array(values, d);

        /// <summary>The elements of an array in C order (a dense copy is read, so any layout works).</summary>
        /// <param name="a">The array.</param>
        /// <returns>The values.</returns>
        private static object[] V(NDArray a)
        {
            using var dense = a.copy();
            return Enumerable.Range(0, (int)dense.size).Select(i => dense.Storage.GetAtIndex(i)).ToArray();
        }

        /// <summary>The int64 elements of an index array in C order.</summary>
        /// <param name="a">An integer array.</param>
        /// <returns>The values.</returns>
        private static long[] L(NDArray a) => V(a).Select(Convert.ToInt64).ToArray();

        /// <summary>The 0 / 1 matrix of a 2-D bool array.</summary>
        /// <param name="m">The bool array.</param>
        /// <returns>The rows.</returns>
        private static int[][] M(NDArray m)
        {
            using var dense = m.copy();
            int rows = (int)dense.shape[0], cols = (int)dense.shape[1];
            return Enumerable.Range(0, rows)
                .Select(r => Enumerable.Range(0, cols).Select(c => (bool)dense.Storage.GetAtIndex(r * cols + c) ? 1 : 0).ToArray())
                .ToArray();
        }

        /// <summary>
        ///     <c>==</c> and <c>&lt;</c> over every pair of the six texts: the tied pairs are equal and never less than each
        ///     other, a shorter text equal up to its end is smaller, and a NUL is smaller than any byte it meets.
        /// </summary>
        [TestMethod]
        public void Comparisons_StopAtASharedNul_ThenCompareTheSizes()
        {
            using var a = A(T, Six);
            using var col = a.reshape(6, 1);
            using var row = a.reshape(1, 6);
            using var eq = np.equal(col, row);
            M(eq).Should().BeEquivalentTo(new[]
            {
                new[] { 1, 1, 0, 0, 0, 0 }, new[] { 1, 1, 0, 0, 0, 0 }, new[] { 0, 0, 1, 0, 0, 0 },
                new[] { 0, 0, 0, 1, 0, 0 }, new[] { 0, 0, 0, 0, 1, 0 }, new[] { 0, 0, 0, 0, 0, 1 },
            }, o => o.WithStrictOrdering());
            using var lt = np.less(col, row);
            M(lt).Should().BeEquivalentTo(new[]
            {
                new[] { 0, 0, 0, 0, 1, 1 }, new[] { 0, 0, 0, 0, 1, 1 }, new[] { 1, 1, 0, 1, 1, 1 },
                new[] { 1, 1, 0, 0, 1, 1 }, new[] { 0, 0, 0, 0, 0, 1 }, new[] { 0, 0, 0, 0, 0, 0 },
            }, o => o.WithStrictOrdering());
            using var nulVsByte = np.less(A(T, "a\0"), A(T, "ab"));
            V(nulVsByte).Should().Equal(true);
            using var shorterFirst = np.less(A(T, "x\0y"), A(T, "x\0yz"));
            V(shorterFirst).Should().Equal(true);
            using var tied = np.equal(A(T, "x\0y"), A(T, "x\0z"));
            V(tied).Should().Equal(true);
        }

        /// <summary>
        ///     The sorts order tied texts as equal: the stable and the default kind keep <c>'a\0c'</c> before
        ///     <c>'a\0b'</c> (their input order), and the extremes are <c>'ab'</c> and <c>'a'</c>.
        /// </summary>
        [TestMethod]
        public void Sorts_AndExtremes_TreatTiedTextsAsEqual()
        {
            using var a = A(T, Six);
            object[] sorted = { "a", "a\0", "a\0c", "a\0b", "a\0bb", "ab" };
            using (var s = np.sort(a))
                V(s).Should().Equal(sorted);
            using (var s = np.sort(a, kind: "stable"))
                V(s).Should().Equal(sorted);
            using (var o = np.argsort(a))
                L(o).Should().Equal(2L, 3L, 0L, 1L, 4L, 5L);
            using (var o = np.argsort(a, kind: "stable"))
                L(o).Should().Equal(2L, 3L, 0L, 1L, 4L, 5L);
            using (var mx = np.max(a))
                mx.GetAtIndex(0).Should().Be("ab");
            using (var mn = np.min(a))
                mn.GetAtIndex(0).Should().Be("a");
            np.argmax(a).Should().Be(5L);
            np.argmin(a).Should().Be(2L);
            // maximum keeps its first operand on a tie, minimum its second (NumPy's loops).
            using (var mx = np.maximum(A(T, "a\0c"), A(T, "a\0b")))
                V(mx).Should().Equal("a\0c");
            using (var mn = np.minimum(A(T, "a\0c"), A(T, "a\0b")))
                V(mn).Should().Equal("a\0b");
        }

        /// <summary>
        ///     <c>np.unique</c> with counts runs NumPy's sort route, whose <c>!=</c> mask joins the tied texts (the first of
        ///     the run is kept, counted twice); the values-only route hashes the FULL bytes, so it keeps both tied texts, in
        ///     NumSharp's input order (NumPy's hash order is unspecified; here it is the same).
        /// </summary>
        [TestMethod]
        public void Unique_SortRouteJoinsTiedTexts_HashRouteKeepsBoth()
        {
            using var a = A(T, Six);
            var counted = np.unique(a, return_counts: true);
            V(counted.values).Should().Equal("a", "a\0", "a\0c", "a\0bb", "ab");
            L(counted.counts).Should().Equal(1L, 1L, 2L, 1L, 1L);
            var values = np.unique(a);
            V(values.values).Should().Equal("a", "a\0", "a\0c", "a\0b", "a\0bb", "ab");
        }

        /// <summary>
        ///     The searches and the set routines match tied texts: <c>np.isin</c> (any kind) finds <c>'a\0c'</c> in
        ///     <c>['a\0b']</c>, <c>np.searchsorted</c> places both keys first, <c>np.intersect1d</c> keeps the first operand's
        ///     text and <c>np.setdiff1d</c> removes it.
        /// </summary>
        [TestMethod]
        public void Searches_AndSetRoutines_MatchTiedTexts()
        {
            using var x = A(T, "a\0c");
            using var y = A(T, "a\0b");
            using (var r = np.isin(x, y))
                V(r).Should().Equal(true);
            using (var r = np.isin(x, y, kind: "sort"))
                V(r).Should().Equal(true);
            using (var r = np.searchsorted(A(T, "a\0b", "b"), A(T, "a\0c", "a\0")))
                L(r).Should().Equal(0L, 0L);
            using (var r = np.intersect1d(x, y))
                V(r).Should().Equal("a\0c");
            using (var r = np.setdiff1d(x, y))
                V(r).Should().BeEmpty();
            // union1d keeps both texts (the values-only np.unique: full bytes); their order is NumPy's hash order.
            using (var r = np.union1d(x, y))
                V(r).Should().BeEquivalentTo(new object[] { "a\0c", "a\0b" });
        }

        /// <summary>
        ///     Fixed-width <c>U</c> text compares every code point: the same two texts are different and both unique.
        /// </summary>
        [TestMethod]
        public void FixedWidthUnicode_ComparesEveryCodePoint()
        {
            using var u = np.array(new[] { "a\0c", "a\0b" });
            using (var eq = np.equal(u["0"], u["1"]))
                ((bool)eq.GetAtIndex(0)).Should().BeFalse();
            V(np.unique(u).values).Should().Equal("a\0b", "a\0c");
        }

        /// <summary>
        ///     <c>np.array_equal</c> / <c>np.array_equiv</c> run <c>(a1 == a2).all()</c> through the <c>StringDType</c>
        ///     comparison: tied texts are equal, <c>U</c> text compares, a pair <c>equal</c> has no loop for (<c>S</c>,
        ///     numbers) is NumPy's all-False array (True only when empty), a NaN-like NA is never equal unless
        ///     <c>equal_nan</c> matches the positions, a None-like NA reads as <c>''</c>, and two different NA objects raise.
        /// </summary>
        [TestMethod]
        public void ArrayEqual_AndEquiv_UseTheStringDTypeComparison()
        {
            np.array_equal(A(T, "a\0c", "x"), A(T, "a\0b", "x")).Should().BeTrue();
            np.array_equal(A(T, "a", "x"), A(T, "a", "y")).Should().BeFalse();
            np.array_equal(A(T, "a"), A(T, "a", "a")).Should().BeFalse();
            np.array_equal(A(T, "a", "b"), np.array(new[] { "a", "b" })).Should().BeTrue();
            np.array_equal(A(T, "a", "b"), np.array(new[] { "a", "b" }, np.dtype("S1"))).Should().BeFalse();
            np.array_equal(A(T, "1", "2"), np.array(new long[] { 1, 2 })).Should().BeFalse();
            np.array_equal(np.zeros(new Shape(0), T), np.zeros(new Shape(0), np.int64)).Should().BeTrue();
            np.array_equal(A(TNaN, "a", double.NaN), A(TNaN, "a", double.NaN)).Should().BeFalse();
            np.array_equal(A(TNaN, "a", double.NaN), A(TNaN, "a", double.NaN), equal_nan: true).Should().BeTrue();
            np.array_equal(A(TNaN, "a", double.NaN), A(TNaN, double.NaN, "a"), equal_nan: true).Should().BeFalse();
            np.array_equal(A(TNaN, double.NaN), np.array(new[] { double.NaN }), equal_nan: true).Should().BeTrue();
            np.array_equal(A(TNone, "a", null), A(TNone, "a", "")).Should().BeTrue();
            Action naConflict = () => np.array_equal(A(TNaN, "a"), A(TNone, "a"));
            naConflict.Should().Throw<TypeError>()
                .WithMessage("Cannot find a compatible null string value for null strings 'nan' and 'None'");
            // equal_nan meets isnan, which refuses S / U text (the first operand first).
            Action isnanU = () => np.array_equal(np.array(new[] { "a" }), A(T, "a"), equal_nan: true);
            isnanU.Should().Throw<TypeError>().WithMessage("ufunc 'isnan' not supported for the input types*");
            np.array_equiv(A(T, "a\0c", "a\0q"), A(T, "a\0b")).Should().BeTrue();
            np.array_equiv(A(T, "a", "a"), A(T, "a", "b", "c")).Should().BeFalse();
            np.array_equiv(A(TNaN, "a", double.NaN), A(TNaN, "a", double.NaN)).Should().BeFalse();
            np.array_equiv(A(T, "a"), np.array(new[] { "a" }, np.dtype("S1"))).Should().BeFalse();
            np.array_equiv(np.zeros(new Shape(0), T), np.zeros(new Shape(2, 0), np.int64)).Should().BeTrue();
        }
    }
}
