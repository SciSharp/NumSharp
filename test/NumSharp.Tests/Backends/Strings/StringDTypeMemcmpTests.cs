using System;
using System.Linq;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     NEP 55 <c>StringDType</c> compares through NumPy's <c>NpyString_cmp</c>, which is <c>memcmp</c> over the shorter
    ///     length and then the sizes since NumPy 2.5.0 (gh-31673): every byte counts, an embedded NUL included, so
    ///     <c>'a\0c'</c> and <c>'a\0b'</c> are DIFFERENT (<c>'a\0b' &lt; 'a\0c'</c>) and <c>'a\0'</c> is smaller than <c>'ab'</c>
    ///     (the NUL meets <c>'b'</c>). NumPy 2.4's <c>strncmp</c> stopped at a NUL both texts shared and tied such texts.
    ///     Every family that orders or matches text follows it — the comparisons, the sorts, <c>np.unique</c>, the extremes,
    ///     the searches, the set routines, <c>np.array_equal</c> — and fixed-width <c>U</c> text compares every code point
    ///     too. Every expected value here was probed against <c>numpy==2.5.4</c>; the string oracle holds the same contract
    ///     over its corpus.
    /// </summary>
    [TestClass]
    public class StringDTypeMemcmpTests
    {
        /// <summary>A fresh <c>StringDType()</c> descriptor.</summary>
        private static DType T => np.dtypes.StringDType();

        /// <summary>A fresh <c>StringDType(na_object=nan)</c> descriptor.</summary>
        private static DType TNaN => np.dtypes.StringDType(na_object: double.NaN);

        /// <summary>A fresh <c>StringDType(na_object=None)</c> descriptor.</summary>
        private static DType TNone => np.dtypes.StringDType(na_object: null);

        /// <summary>The six texts NumPy's probe orders: two pairs that differ only after a shared NUL, and the NUL-against-a-byte
        ///     case.</summary>
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
        ///     <c>==</c> and <c>&lt;</c> over every pair of the six texts: only identical texts are equal, the bytes after a
        ///     shared NUL decide, a shorter text equal up to its end is smaller, and a NUL is smaller than any byte it meets.
        /// </summary>
        [TestMethod]
        public void Comparisons_CompareEveryByte_ThenTheSizes()
        {
            using var a = A(T, Six);
            using var col = a.reshape(6, 1);
            using var row = a.reshape(1, 6);
            using var eq = np.equal(col, row);
            M(eq).Should().BeEquivalentTo(new[]
            {
                new[] { 1, 0, 0, 0, 0, 0 }, new[] { 0, 1, 0, 0, 0, 0 }, new[] { 0, 0, 1, 0, 0, 0 },
                new[] { 0, 0, 0, 1, 0, 0 }, new[] { 0, 0, 0, 0, 1, 0 }, new[] { 0, 0, 0, 0, 0, 1 },
            }, o => o.WithStrictOrdering());
            using var lt = np.less(col, row);
            M(lt).Should().BeEquivalentTo(new[]
            {
                new[] { 0, 0, 0, 0, 0, 1 }, new[] { 1, 0, 0, 0, 1, 1 }, new[] { 1, 1, 0, 1, 1, 1 },
                new[] { 1, 1, 0, 0, 1, 1 }, new[] { 1, 0, 0, 0, 0, 1 }, new[] { 0, 0, 0, 0, 0, 0 },
            }, o => o.WithStrictOrdering());
            using var nulVsByte = np.less(A(T, "a\0"), A(T, "ab"));
            V(nulVsByte).Should().Equal(true);
            using var shorterFirst = np.less(A(T, "x\0y"), A(T, "x\0yz"));
            V(shorterFirst).Should().Equal(true);
            using var pastTheNul = np.equal(A(T, "x\0y"), A(T, "x\0z"));
            V(pastTheNul).Should().Equal(false);
        }

        /// <summary>
        ///     The sorts and the extremes order the texts by their bytes: <c>'a\0b' &lt; 'a\0bb' &lt; 'a\0c'</c>, both kinds;
        ///     <c>np.maximum</c> / <c>np.minimum</c> pick by the bytes past the NUL.
        /// </summary>
        [TestMethod]
        public void Sorts_AndExtremes_OrderByEveryByte()
        {
            using var a = A(T, Six);
            object[] sorted = { "a", "a\0", "a\0b", "a\0bb", "a\0c", "ab" };
            using (var s = np.sort(a))
                V(s).Should().Equal(sorted);
            using (var s = np.sort(a, kind: "stable"))
                V(s).Should().Equal(sorted);
            using (var o = np.argsort(a))
                L(o).Should().Equal(2L, 3L, 1L, 4L, 0L, 5L);
            using (var o = np.argsort(a, kind: "stable"))
                L(o).Should().Equal(2L, 3L, 1L, 4L, 0L, 5L);
            using (var mx = np.max(a))
                mx.GetAtIndex(0).Should().Be("ab");
            using (var mn = np.min(a))
                mn.GetAtIndex(0).Should().Be("a");
            np.argmax(a).Should().Be(5L);
            np.argmin(a).Should().Be(2L);
            using (var mx = np.maximum(A(T, "a\0c"), A(T, "a\0b")))
                V(mx).Should().Equal("a\0c");
            using (var mn = np.minimum(A(T, "a\0c"), A(T, "a\0b")))
                V(mn).Should().Equal("a\0b");
        }

        /// <summary>
        ///     <c>np.unique</c>'s sort route and its values-only hash route agree now: every byte content is one value, each
        ///     counted once.
        /// </summary>
        [TestMethod]
        public void Unique_KeepsEveryByteContent()
        {
            using var a = A(T, Six);
            var counted = np.unique(a, return_counts: true);
            V(counted.values).Should().Equal("a", "a\0", "a\0b", "a\0bb", "a\0c", "ab");
            L(counted.counts).Should().Equal(1L, 1L, 1L, 1L, 1L, 1L);
            var values = np.unique(a);
            V(values.values).Should().Equal("a", "a\0", "a\0b", "a\0bb", "a\0c", "ab");
        }

        /// <summary>
        ///     The searches and the set routines compare every byte: <c>np.isin</c> (any kind) does not find <c>'a\0c'</c> in
        ///     <c>['a\0b']</c>, <c>np.searchsorted</c> places <c>'a\0c'</c> after <c>'a\0b'</c>, <c>np.intersect1d</c> is empty,
        ///     <c>np.setdiff1d</c> keeps the text and <c>np.union1d</c> holds both in byte order.
        /// </summary>
        [TestMethod]
        public void Searches_AndSetRoutines_CompareEveryByte()
        {
            using var x = A(T, "a\0c");
            using var y = A(T, "a\0b");
            using (var r = np.isin(x, y))
                V(r).Should().Equal(false);
            using (var r = np.isin(x, y, kind: "sort"))
                V(r).Should().Equal(false);
            using (var r = np.searchsorted(A(T, "a\0b", "b"), A(T, "a\0c", "a\0")))
                L(r).Should().Equal(1L, 0L);
            using (var r = np.intersect1d(x, y))
                V(r).Should().BeEmpty();
            using (var r = np.setdiff1d(x, y))
                V(r).Should().Equal("a\0c");
            using (var r = np.union1d(x, y))
                V(r).Should().Equal("a\0b", "a\0c");
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
        ///     comparison: texts that differ past a NUL are different, <c>U</c> text compares, a pair <c>equal</c> has no loop
        ///     for (<c>S</c>, numbers) is NumPy's all-False array (True only when empty), a NaN-like NA is never equal unless
        ///     <c>equal_nan</c> matches the positions, a None-like NA reads as <c>''</c>, and two different NA objects raise.
        /// </summary>
        [TestMethod]
        public void ArrayEqual_AndEquiv_UseTheStringDTypeComparison()
        {
            np.array_equal(A(T, "a\0c", "x"), A(T, "a\0b", "x")).Should().BeFalse();
            np.array_equal(A(T, "a\0b", "x"), A(T, "a\0b", "x")).Should().BeTrue();
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
            np.array_equiv(A(T, "a\0c", "a\0q"), A(T, "a\0b")).Should().BeFalse();
            np.array_equiv(A(T, "a\0b", "a\0b"), A(T, "a\0b")).Should().BeTrue();
            np.array_equiv(A(T, "a", "a"), A(T, "a", "b", "c")).Should().BeFalse();
            np.array_equiv(A(TNaN, "a", double.NaN), A(TNaN, "a", double.NaN)).Should().BeFalse();
            np.array_equiv(A(T, "a"), np.array(new[] { "a" }, np.dtype("S1"))).Should().BeFalse();
            np.array_equiv(np.zeros(new Shape(0), T), np.zeros(new Shape(2, 0), np.int64)).Should().BeTrue();
        }

        /// <summary>
        ///     <c>np.unique</c>'s NA entries (NumPy 2.5.4's <c>unique_vstring</c>): every NA cell of a descriptor whose NA object
        ///     is not NaN-like is ONE entry even with <c>equal_nan=False</c> (<c>equal_nulls = equal_nan || !has_nan_na</c>;
        ///     NumPy 2.4 kept each), a None-like NA among texts is still the sort's refusal, and a NaN-like NA keeps one entry
        ///     per NA only without <c>equal_nan</c>.
        /// </summary>
        [TestMethod]
        public void Unique_NAEntries_FollowEqualNulls()
        {
            using (var none = A(TNone, null, null))
                V(np.unique(none, equal_nan: false).values).Should().HaveCount(1);
            Action noneAndText = () => np.unique(A(TNone, null, "a", null), equal_nan: false);
            noneAndText.Should().Throw<ValueError>().WithMessage("Cannot compare null that is not a nan-like value");
            using (var nan = A(TNaN, double.NaN, "a", double.NaN))
            {
                V(np.unique(nan, equal_nan: false).values).Should().HaveCount(3);
                V(np.unique(nan).values).Should().HaveCount(2);
            }
        }

        /// <summary>The byte loop <c>memcmp</c> answers from: the first index whose bytes differ, else n.</summary>
        /// <param name="a">The first text.</param>
        /// <param name="b">The second text.</param>
        /// <param name="n">The bytes to compare.</param>
        /// <returns>The first differing index.</returns>
        private static unsafe int ReferenceMismatch(byte* a, byte* b, int n)
        {
            for (int i = 0; i < n; i++)
                if (a[i] != b[i])
                    return i;
            return n;
        }

        /// <summary>
        ///     <see cref="NumSharp.Backends.Strings.NDVStringCells.MismatchIndex"/> (its 32- and 16-byte vector steps, the step
        ///     moved back to end on the last byte, the 8-byte SWAR words) finds the byte loop's first difference at every length
        ///     0–300: none, a difference at every position, a NUL in both texts at every position (an ordinary byte: no stop),
        ///     a NUL in A alone (a difference), and a difference right after a shared NUL.
        /// </summary>
        [TestMethod]
        public unsafe void MismatchIndex_EqualsTheByteLoop_AtEveryLengthAndPosition()
        {
            var rng = new System.Random(4401);
            var a = new byte[300];
            var b = new byte[300];
            fixed (byte* pa = a)
            fixed (byte* pb = b)
                for (int n = 0; n <= 300; n++)
                {
                    for (int k = 0; k < 300; k++)
                        a[k] = b[k] = (byte)rng.Next(1, 256);
                    NumSharp.Backends.Strings.NDVStringCells.MismatchIndex(pa, pb, n).Should().Be(n, $"length {n}, equal");
                    for (int p = 0; p < n; p++)
                    {
                        byte keepA = a[p], keepB = b[p];
                        b[p] = (byte)(a[p] ^ (1 << rng.Next(8)));
                        NumSharp.Backends.Strings.NDVStringCells.MismatchIndex(pa, pb, n).Should().Be(ReferenceMismatch(pa, pb, n), $"length {n}, difference at {p}");
                        a[p] = b[p] = 0;
                        NumSharp.Backends.Strings.NDVStringCells.MismatchIndex(pa, pb, n).Should().Be(n, $"length {n}, shared NUL at {p}");
                        b[p] = 1;
                        NumSharp.Backends.Strings.NDVStringCells.MismatchIndex(pa, pb, n).Should().Be(p, $"length {n}, NUL in A at {p}");
                        if (p + 1 < n)
                        {
                            // A shared NUL at p and a difference right above it: memcmp reads past the NUL.
                            b[p] = 0;
                            byte above = a[p + 1];
                            a[p + 1] = (byte)(b[p + 1] ^ 1);
                            NumSharp.Backends.Strings.NDVStringCells.MismatchIndex(pa, pb, n).Should().Be(p + 1, $"length {n}, NUL then a difference at {p}");
                            a[p + 1] = above;
                        }
                        a[p] = keepA;
                        b[p] = keepB;
                    }
                }
        }

        /// <summary>
        ///     <see cref="NumSharp.Backends.Strings.NDVStringCells.MismatchIndex"/> reads nothing outside its texts: two texts of
        ///     every length to 300 that never differ placed against inaccessible pages, ending on the last accessible byte and
        ///     starting on the first.
        /// </summary>
        [TestMethod]
        public unsafe void MismatchIndex_ReadsOnlyItsTexts()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("FencedMemory needs Windows (reserved pages).");
            using var fa = new FencedMemory(pages: 1);
            using var fb = new FencedMemory(pages: 1);
            for (int n = 0; n <= 300; n++)
                foreach (bool atEnd in new[] { true, false })
                {
                    byte* a = fa.Place(n, atEnd), b = fb.Place(n, atEnd);
                    for (int k = 0; k < n; k++)
                        a[k] = b[k] = (byte)('a' + k % 26);
                    NumSharp.Backends.Strings.NDVStringCells.MismatchIndex(a, b, n).Should().Be(n, $"length {n}, at end {atEnd}");
                    if (n > 0)
                    {
                        b[n - 1] ^= 1;
                        NumSharp.Backends.Strings.NDVStringCells.MismatchIndex(a, b, n).Should().Be(n - 1, $"length {n}, last byte differs, at end {atEnd}");
                    }
                }
        }
    }
}
