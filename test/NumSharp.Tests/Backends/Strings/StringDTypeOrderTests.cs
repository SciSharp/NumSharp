using System;
using System.Linq;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Sorting;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     Ordering NEP 55 <c>StringDType</c> text (strings phase 2, P2.2 over <c>T</c>): the sorts, partitions and
    ///     <c>np.lexsort</c>, the searches, <c>np.unique</c> and the set routines, and the order statistics (<c>np.median</c>,
    ///     the quantile family, their nan forms, numpy.ma's median), with NumPy's ordering of missing values: a NaN-like NA
    ///     sorts after every text and keeps its input order, a None-like NA raises at its first comparison, a string NA reads
    ///     as its text. Every expected value and message was probed against <c>numpy==2.4.2</c>; the string oracle's section
    ///     <c>tsort</c> holds the same contract over every layout. The <see cref="MisalignedAttribute"/> tests pin where
    ///     NumSharp deliberately answers differently: NumPy's crashes and misorders (NumSharp gives the answer NumPy gives for a
    ///     contiguous copy), its hash order (NumSharp sorts), and an object-array result NumSharp has no dtype for.
    /// </summary>
    [TestClass]
    public class StringDTypeOrderTests
    {
        /// <summary>A fresh <c>StringDType()</c> descriptor.</summary>
        private static DType T => np.dtypes.StringDType();

        /// <summary>A fresh <c>StringDType(na_object=nan)</c> descriptor (a NaN-like NA: it sorts after every text).</summary>
        private static DType TNaN => np.dtypes.StringDType(na_object: double.NaN);

        /// <summary>A fresh <c>StringDType(na_object=None)</c> descriptor (a None-like NA: any ordering of it raises).</summary>
        private static DType TNone => np.dtypes.StringDType(na_object: null);

        /// <summary>A 1-D array of the given elements in the given descriptor (texts and NA objects).</summary>
        /// <param name="d">The descriptor.</param>
        /// <param name="values">The elements.</param>
        /// <returns>The new array.</returns>
        private static NDArray A(DType d, params object[] values) => np.array(values, d);

        /// <summary>The elements of an array in C order (texts and NA objects; numbers for a numeric array).</summary>
        /// <param name="a">The array.</param>
        /// <returns>The values.</returns>
        private static object[] V(NDArray a)
        {
            // A view (a broadcast or transposed result) reads in C order through its own strides: copy to a dense array
            // first so GetAtIndex walks C order.
            using var dense = a.copy();
            return Enumerable.Range(0, (int)dense.size).Select(i => dense.Storage.GetAtIndex(i)).ToArray();
        }

        /// <summary>The int64 elements of an index array in C order.</summary>
        /// <param name="a">An int64 array.</param>
        /// <returns>The indices.</returns>
        private static long[] L(NDArray a) => V(a).Select(Convert.ToInt64).ToArray();

        // ------------------------------------------------------------------------------------------------------------
        // sorts
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     A NaN-like NA sorts after every text (NumPy's <c>_compare</c> returns 1 whenever the first operand is NA), so
        ///     <c>np.sort</c> puts it last and <c>np.argsort</c> lists it last.
        /// </summary>
        [TestMethod]
        public void Sort_NaNLikeNA_SortsAfterEveryText()
        {
            using var a = A(TNaN, "b", double.NaN, "a", "ab");
            using var sorted = np.sort(a);
            V(sorted).Should().Equal("a", "ab", "b", double.NaN);
            sorted.dtype.Should().Be(TNaN);
            using var order = np.argsort(a);
            L(order).Should().Equal(2L, 3L, 0L, 1L);
        }

        /// <summary>
        ///     A None-like NA in a line of two or more elements raises NumPy's ValueError at the sort's first comparison; a
        ///     line of one element compares nothing and copies.
        /// </summary>
        [TestMethod]
        public void Sort_NoneLikeNA_RaisesInALineOfTwo_CopiesALineOfOne()
        {
            using var two = A(TNone, "b", null);
            Action sort = () => np.sort(two);
            sort.Should().Throw<ValueError>().WithMessage("Cannot compare null that is not a nan-like value");
            using var one = A(TNone, new object[] { null });
            using var sortedOne = np.sort(one);
            V(sortedOne).Should().Equal(new object[] { null });
        }

        /// <summary>The stable kind keeps equal texts in input order (NumPy's mergesort only tests <c>&lt; 0</c>).</summary>
        [TestMethod]
        public void ArgSort_StableKind_KeepsEqualTextsInInputOrder()
        {
            using var a = A(T, "b", "a", "b", "a");
            using var order = np.argsort(a, kind: "stable");
            L(order).Should().Equal(1L, 3L, 0L, 2L);
        }

        /// <summary>
        ///     <c>np.sort</c> along axis 0 of a 2-D array sorts each column, and <c>np.partition</c> (no selection loop for
        ///     <c>StringDType</c>: NumPy's fallback is the full introsort) gives the sorted line.
        /// </summary>
        [TestMethod]
        public void Sort_AlongAxis0_AndPartition_AreTheSortedLines()
        {
            using var m = np.array(new object[,] { { "b", "a" }, { "a", "c" } }, T);
            using var columns = np.sort(m, 0);
            V(columns).Should().Equal("a", "a", "b", "c");
            using var p = np.partition(A(T, "d", "a", "c", "b"), 1);
            V(p).Should().Equal("a", "b", "c", "d");
        }

        /// <summary>
        ///     NumPy 2.4.2 crashes (an access violation) in <c>np.lexsort</c> when a <c>StringDType</c> key is not contiguous
        ///     along the axis — a transposed 2-D key, or a stepped 1-D key in a key list. NumSharp sorts and gives NumPy's
        ///     answer for contiguous copies of the keys.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void LexSort_NonContiguousKeys_GiveTheContiguousAnswer()
        {
            using var keys = np.array(new object[,] { { "b", "a" }, { "a", "b" }, { "a", "a" } }, T);
            // keys.T is (2, 3): key 0 = ['b', 'a', 'a'], key 1 (primary) = ['a', 'b', 'a'] — NumPy: [2, 0, 1] on a copy.
            using var transposed = np.lexsort(keys.T, -1);
            L(transposed).Should().Equal(2L, 0L, 1L);
            using var base6 = A(T, "b", "z", "a", "z", "a", "z");
            using var stepped = base6["::2"];
            using var primary = A(T, "a", "b", "a");
            using var listed = np.lexsort(new[] { stepped, primary });
            L(listed).Should().Equal(2L, 0L, 1L);
        }

        /// <summary>
        ///     NumPy 2.4.2's argsort of a BROADCAST <c>StringDType</c> view misorders arena texts (a 16-byte text lands before
        ///     shorter, smaller ones); the same array copied sorts right. NumSharp sorts the view like its copy.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void ArgSort_BroadcastView_SortsLikeItsCopy()
        {
            using var b = A(T, new string('a', 16), "b", new string('a', 17), "c");
            using var view = np.broadcast_to(b, new Shape(3, 4));
            using var order = np.argsort(view, null);
            // NumPy's argsort of view.copy(); NumPy's own argsort of the view gives [4, 6, 10, 0, 2, 1, 5, 9, 3, 7, 11, 8].
            L(order).Should().Equal(0L, 4L, 8L, 2L, 6L, 10L, 1L, 5L, 9L, 3L, 7L, 11L);
        }

        // ------------------------------------------------------------------------------------------------------------
        // searches
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>np.searchsorted</c> over <c>StringDType</c>: NumPy's binary search on both sides, a NaN-like NA searched as
        ///     the greatest value, a None-like NA refused at its first comparison.
        /// </summary>
        [TestMethod]
        public void SearchSorted_BothSides_AndTheNARules()
        {
            using var hay = A(T, "a", "c", "e");
            using var left = np.searchsorted(hay, A(T, "b", "c", "z"));
            L(left).Should().Equal(1L, 1L, 3L);
            using var right = np.searchsorted(hay, A(T, "c"), "right");
            L(right).Should().Equal(2L);
            using var nan = np.searchsorted(A(TNaN, "a", "c", double.NaN), A(TNaN, double.NaN, "b"));
            L(nan).Should().Equal(2L, 1L);
            Action none = () => np.searchsorted(A(TNone, "a", null), A(TNone, "b"));
            none.Should().Throw<ValueError>().WithMessage("Cannot compare null that is not a nan-like value");
        }

        /// <summary>
        ///     NumPy 2.4.2's searchsorted loads the HAYSTACK cells with the keys' allocator, so a haystack text in its arena
        ///     raises <c>MemoryError: Failed to load string in string comparison</c> (or reads another array's arena). NumSharp
        ///     reads each array's cells with its own allocator and searches.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void SearchSorted_ArenaHaystackTexts_Search()
        {
            using var hay = A(T, new string('a', 20), new string('b', 20), new string('c', 20));
            using var at = np.searchsorted(hay, A(T, "b"));
            L(at).Should().Equal(1L);
        }

        /// <summary><c>np.digitize</c> converts its bins to float64 for the monotonicity test: NumPy's safe-cast refusal.</summary>
        [TestMethod]
        public void Digitize_StringDType_RefusesTheFloatCast()
        {
            Action digitize = () => np.digitize(A(T, "b", "d"), A(T, "a", "c"));
            digitize.Should().Throw<InvalidCastException>()
                .WithMessage("Cannot cast array data from StringDType() to dtype('float64') according to the rule 'safe'");
        }

        // ------------------------------------------------------------------------------------------------------------
        // unique values and set routines
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>np.unique</c> sorts the distinct texts; <c>equal_nan</c> collapses the NaN-like NAs into one entry (the last)
        ///     or keeps each; the counts follow the sorted values.
        /// </summary>
        [TestMethod]
        public void Unique_ValuesCountsAndEqualNan()
        {
            V(np.unique(A(T, "b", "a", "b", "c")).values).Should().Equal("a", "b", "c");
            var counted = np.unique(A(T, "b", "a", "b"), return_counts: true);
            V(counted.values).Should().Equal("a", "b");
            L(counted.counts).Should().Equal(1L, 2L);
            V(np.unique(A(TNaN, double.NaN, "a", double.NaN)).values).Should().Equal("a", double.NaN);
            V(np.unique(A(TNaN, double.NaN, "a", double.NaN), equal_nan: false).values).Should().Equal("a", double.NaN, double.NaN);
        }

        /// <summary>
        ///     The sorted path compares a None-like NA and raises; an axis over two dimensions takes NumPy's structured path,
        ///     which refuses <c>StringDType</c> fields.
        /// </summary>
        [TestMethod]
        public void Unique_NoneLikeNA_Raises_AndAnAxisIsRefused()
        {
            Action none = () => np.unique(A(TNone, "a", null, "b"));
            none.Should().Throw<ValueError>().WithMessage("Cannot compare null that is not a nan-like value");
            Action axis = () => np.unique(np.array(new object[,] { { "a", "b" }, { "a", "b" } }, T), axis: 0);
            axis.Should().Throw<TypeError>().WithMessage("The axis argument to unique is not supported for dtype StringDType()");
        }

        /// <summary>
        ///     <c>np.unique_values</c> runs NumPy's <c>sorted=False</c> hash pass, which compares no NA: a None-like NA does not
        ///     raise. NumPy returns its hash set's order (<c>['a', None, 'b']</c>); NumSharp returns the same values sorted, the
        ///     NA last.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void UniqueValues_NoneLikeNA_DoesNotRaise_AndIsSorted()
        {
            using var values = np.unique_values(A(TNone, "a", null, "b"));
            V(values).Should().Equal("a", "b", null);
        }

        /// <summary>
        ///     The values-only route hashes texts of at most 15 bytes from their two cell words masked to the size: a short
        ///     text written over a longer one leaves stale bytes past its size (<c>t[i] = 'q'</c> over an arena text keeps the
        ///     freed offset's 0x01), and those must not make it a second entry beside an equal text in a fresh cell.
        /// </summary>
        [TestMethod]
        public void Unique_HashRoute_IgnoresStaleCellBytes()
        {
            // 128 elements (the hash route needs 64): long texts overwritten by "q" (stale bytes), fresh "q", "r".
            var values = new object[128];
            for (int i = 0; i < values.Length; i++)
                values[i] = i % 3 == 0 ? new string('L', 40) : i % 3 == 1 ? "q" : "r";
            using var t = A(T, values);
            for (int i = 0; i < values.Length; i += 3)
                t[i] = "q";
            V(np.unique(t).values).Should().Equal("q", "r");
            V(np.unique_values(t)).Should().Equal("q", "r");
        }

        /// <summary>
        ///     The hash route and the sort route give the same distinct values (and the flag route's): few distinct texts, a
        ///     fifth distinct with arena texts, mostly distinct, and NaN-like NAs (one entry under <c>equal_nan</c>, one each
        ///     without).
        /// </summary>
        [TestMethod]
        public void Unique_HashAndSortRoutes_Agree()
        {
            var rng = new System.Random(11);
            string Word(int len) => new string(Enumerable.Range(0, len).Select(_ => (char)('a' + rng.Next(26))).ToArray());
            var cases = new[]
            {
                Enumerable.Range(0, 5000).Select(_ => (object)new[] { "apple", "fig", "kiwi" }[rng.Next(3)]).ToArray(),
                Enumerable.Range(0, 5000).Select(_ => (object)("https://site.example/" + Word(2))).ToArray(),
                Enumerable.Range(0, 5000).Select(_ => (object)Word(10)).ToArray(),
                Enumerable.Range(0, 5000).Select(i => i % 7 == 3 ? (object)double.NaN : Word(2)).ToArray(),
            };
            foreach (var values in cases)
            {
                using var t = A(TNaN, values);
                foreach (bool equalNan in new[] { true, false })
                {
                    var hashed = V(np.unique(t, equal_nan: equalNan).values);
                    VStringDistinct.Disabled = true;
                    try
                    {
                        V(np.unique(t, equal_nan: equalNan).values).Should().Equal(hashed);
                    }
                    finally
                    {
                        VStringDistinct.Disabled = false;
                    }
                }
                // The flag route (argsort, run starts) gives the same values under its own equal_nan rule (ignored for
                // StringDType: the NAs join the greatest run), so compare without NAs there.
                var flagged = V(np.unique(t, return_counts: true).values);
                V(np.unique(t).values).Where(v => v is string).Should().Equal(flagged.Where(v => v is string));
            }
        }

        /// <summary>
        ///     The hash pass's estimate of the whole array's distinct count from its first window: no repeat reads as every
        ///     text distinct; many repeats far under half the array; a few repeats in a window of 4,096 among 100,000 texts
        ///     (the benchmark's 6-letter non-Latin words) over half.
        /// </summary>
        [TestMethod]
        public void EstimateDistinct_FromTheRepeatsOfTheFirstWindow()
        {
            VStringDistinct.EstimateDistinct(4096, 4096, 100_000).Should().Be(100_000);
            VStringDistinct.EstimateDistinct(4096, 8, 100_000).Should().BeLessThan(50_000);
            VStringDistinct.EstimateDistinct(4096, 3655, 100_000).Should().BeLessThan(50_000, "a fifth distinct: hash");
            VStringDistinct.EstimateDistinct(4096, 4064, 100_000).Should().BeGreaterThan(50_000, "mostly distinct: sort");
        }

        /// <summary><c>np.isin</c> of <c>StringDType</c> against <c>U</c> text, and a NaN-like NA that never matches.</summary>
        [TestMethod]
        public void IsIn_AgainstStr_AndNaNNeverMatches()
        {
            using var hit = np.isin(A(T, "a", "b", "c"), np.array(new[] { "c", "a" }));
            V(hit).Should().Equal(true, false, true);
            using var nan = np.isin(A(TNaN, "a", double.NaN), A(TNaN, double.NaN));
            V(nan).Should().Equal(false, false);
        }

        /// <summary>
        ///     The concatenating set routines promote the operands: a <c>StringDType</c> beside a number has no common DType
        ///     (NumPy's DTypePromotionError), beside another <c>StringDType</c> the result takes the common NA object.
        /// </summary>
        [TestMethod]
        public void SetRoutines_PromoteLikeTheConcatenation()
        {
            Action number = () => np.intersect1d(A(T, "a"), np.array(new long[] { 1 }));
            number.Should().Throw<DTypePromotionError>().WithMessage(
                "The DType <class 'numpy.dtypes.StringDType'> could not be promoted by <class 'numpy.dtypes.Int64DType'>. This " +
                "means that no common DType exists for the given inputs. For example they cannot be stored in a single array " +
                "unless the dtype is `object`. The full list of DTypes is: (<class 'numpy.dtypes.StringDType'>, <class " +
                "'numpy.dtypes.Int64DType'>)");
            using var mixed = np.intersect1d(A(T, "b", "a"), A(TNone, "a"));
            V(mixed).Should().Equal("a");
            mixed.dtype.Should().Be(TNone);
            V(np.union1d(A(T, "b", "a"), A(T, "c", "a"))).Should().Equal("a", "b", "c");
            V(np.setxor1d(A(T, "b", "a"), A(T, "c", "a"))).Should().Equal("b", "c");
            V(np.setdiff1d(A(T, "b", "a"), A(T, "a"))).Should().Equal("b");
        }

        // ------------------------------------------------------------------------------------------------------------
        // order statistics
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>np.median</c> sorts and then means the middle: the sum of texts is a Python str, which the legacy divide
        ///     refuses; two NaN-like NA middles give <c>np.float64(nan)</c>; an empty input meets the mean's no-identity refusal.
        /// </summary>
        [TestMethod]
        public void Median_RefusesText_GivesNaNForNaNMiddles()
        {
            Action text = () => np.median(A(T, "b", "a", "c"));
            text.Should().Throw<TypeError>().WithMessage(
                "ufunc 'divide' not supported for the input types, and the inputs could not be safely coerced to any supported " +
                "types according to the casting rule ''safe''");
            using var nan = np.median(A(TNaN, double.NaN, double.NaN));
            nan.dtype.Should().Be(np.float64);
            nan.ndim.Should().Be(0);
            double.IsNaN((double)nan.Storage.GetAtIndex(0)).Should().BeTrue();
            nan.Shape.IsWriteable.Should().BeFalse();
            Action empty = () => np.median(A(T));
            empty.Should().Throw<ValueError>().WithMessage("zero-size array to reduction operation add which has no identity");
        }

        /// <summary>
        ///     The quantile methods that pick an element compute (<c>lower</c> gives NumPy's str as a read-only 0-d array);
        ///     the interpolating ones read the two neighbours and refuse their Python subtraction.
        /// </summary>
        [TestMethod]
        public void Quantile_PicksAnElement_OrRefusesTheSubtraction()
        {
            using var lower = np.quantile(A(T, "b", "a", "c"), 0.5, method: "lower");
            lower.ndim.Should().Be(0);
            lower.Storage.GetAtIndex(0).Should().Be("b");
            lower.Shape.IsWriteable.Should().BeFalse();
            using var nearest = np.percentile(A(T, "d", "a", "c", "b"), new double[] { 25, 75 }, method: "nearest");
            V(nearest).Should().Equal("b", "c");
            Action linear = () => np.quantile(A(T, "b", "a", "c"), 0.5);
            linear.Should().Throw<TypeError>().WithMessage("unsupported operand type(s) for -: 'str' and 'str'");
        }

        /// <summary>
        ///     The nan forms remove the NaN-like NAs: an all-NaN input gives the NA object itself (a read-only 0-d array holding
        ///     it); <c>np.nanmedian</c> along an axis shorter than 600 goes through numpy.ma's median, whose masked sort refuses.
        /// </summary>
        [TestMethod]
        public void NanForms_AllNaNGivesTheNA_ShortAxisMedianRefuses()
        {
            using var q = np.nanquantile(A(TNaN, double.NaN, double.NaN), 0.5);
            q.dtype.Should().Be(TNaN);
            double.IsNaN((double)q.Storage.GetAtIndex(0)).Should().BeTrue();
            q.Shape.IsWriteable.Should().BeFalse();
            using var m = np.nanmedian(A(TNaN, double.NaN, double.NaN));
            double.IsNaN((double)m.Storage.GetAtIndex(0)).Should().BeTrue();
            Action axis = () => np.nanmedian(np.array(new object[,] { { "a", double.NaN }, { "b", "c" } }, TNaN), 1);
            axis.Should().Throw<TypeError>().WithMessage("Unsuitable type StringDType(na_object=nan) for calculating minimum.");
        }

        /// <summary>
        ///     A row whose nan-quantile is a None-like NA (NumPy keeps None: it is not NaN) makes NumPy's
        ///     <c>np.apply_along_axis</c> build an OBJECT array (<c>array([None, None], dtype=object)</c>); NumSharp has no
        ///     object dtype and refuses.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void NanQuantile_NoneRows_NeedAnObjectArray()
        {
            using var a = np.array(new object[,] { { null }, { null } }, TNone);
            Action rows = () => np.nanquantile(a, 0.5, 1, method: "lower");
            rows.Should().Throw<NotSupportedException>();
        }

        /// <summary>
        ///     numpy.ma's median of a masked <c>StringDType</c> array without a mask sorts the data and ends in the masked
        ///     mean's divide, whose sum is an ndarray: the new-style dispatcher's refusal.
        /// </summary>
        [TestMethod]
        public void MaskedMedian_NoMask_RefusesTheDivide()
        {
            Action median = () => np.ma.median(np.ma.array(A(T, "a", "b")));
            median.Should().Throw<TypeError>().WithMessage(
                "ufunc 'divide' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, " +
                "<class 'numpy.dtypes.Int64DType'>) -> None");
        }

        /// <summary><c>np.histogram</c> with <c>StringDType</c> bins compares them with the float data: Python's refusal.</summary>
        [TestMethod]
        public void Histogram_StringDTypeBins_RefuseTheComparison()
        {
            Action histogram = () => np.histogram(np.array(new[] { 1.0, 2.0 }), A(T, "a", "b"));
            histogram.Should().Throw<TypeError>().WithMessage("'<' not supported between instances of 'float' and 'str'");
        }

        /// <summary>
        ///     NumPy 2.4.2's <c>StringDType</c> to complex cast writes the real part into the imaginary part too, so its
        ///     <c>np.sort_complex(['2', '1'])</c> is <c>[1+1j, 2+2j]</c>. NumSharp's cast parses the text as Python's
        ///     <c>complex()</c> does.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void SortComplex_ParsesTheText_RealPartOnly()
        {
            using var sorted = np.sort_complex(A(T, "2", "1"));
            V(sorted).Should().Equal(new Complex(1, 0), new Complex(2, 0));
        }

        /// <summary>
        ///     NumPy 2.4.2's <c>np.argsort(view, axis=None)</c> of a non-contiguous <c>StringDType</c> view misorders its arena
        ///     texts (a 17-byte text before a 16-byte prefix of it), while <c>np.sort(view, axis=None)</c>, an axis, the
        ///     partitions and <c>ndarray.sort</c> order them right. NumSharp's argsort gives the answer NumPy gives for the
        ///     view's copy.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void ArgSort_NonContiguousView_AxisNone_SortsLikeItsCopy()
        {
            using var b = A(T, "b", "c", new string('a', 17), new string('a', 16));
            using var square = b.reshape(2, 2);
            using var view = square["::-1, ::-1"];
            using var order = np.argsort(view, null, kind: "stable");
            // NumPy's argsort of view.copy(); NumPy's own argsort of the view gives [1, 0, 3, 2].
            L(order).Should().Equal(0L, 1L, 3L, 2L);
            using var sorted = np.sort(view, null);
            V(sorted).Should().Equal(new string('a', 16), new string('a', 17), "b", "c");
        }

        /// <summary>
        ///     The nan forms remove the NaN-like NAs as NumPy's <c>_remove_nan_1d</c> does — the first NaN slots filled with the
        ///     non-NaN elements of the last ones, not a mask selection in input order — and the partition that picks the
        ///     element keeps texts tied under <c>NpyString_cmp</c> in that order: <c>['\0x', nan, '\0y', 'z', nan, '\0w']</c>
        ///     becomes <c>['\0x', '\0w', '\0y', 'z']</c>, so the element at index 1 is <c>'\0w'</c>.
        /// </summary>
        [TestMethod]
        public void NanQuantile_TiedTexts_FollowNumPysNaNRemovalOrder()
        {
            using var a = A(TNaN, "\0x", double.NaN, "\0y", "z", double.NaN, "\0w");
            using (var r = np.nanquantile(a, 0.34, method: "lower"))
                r.GetAtIndex(0).Should().Be("\0w");
            using (var r = np.nanquantile(a, 0.67, method: "nearest"))
                r.GetAtIndex(0).Should().Be("\0y");
            using (var r = np.nanpercentile(a, 50.0, method: "lower"))
                r.GetAtIndex(0).Should().Be("\0w");
        }

        /// <summary>
        ///     <c>np.searchsorted</c> converts the keys to <c>PyArray_PromoteTypes(keys, haystack)</c>, whose
        ///     <c>common_instance</c> takes the present NA object and the FIRST operand's <c>coerce</c>, a <c>U</c> operand
        ///     counting as <c>StringDType()</c> — so a <c>U</c>-array key into a <c>coerce=False</c> haystack is refused naming
        ///     <c>StringDType()</c>, not the haystack's descriptor.
        /// </summary>
        [TestMethod]
        public void SearchSorted_UKeys_AreRefusedNamingThePromotedDescriptor()
        {
            using var hay = np.array(new object[] { "a", "b" }, np.dtypes.StringDType(coerce: false));
            Action plain = () => np.searchsorted(hay, np.array(new[] { "a" }));
            plain.Should().Throw<TypeError>()
                .WithMessage("Cannot cast array data from dtype('<U1') to StringDType() according to the rule 'safe'");
            using var hayNA = np.array(new object[] { "a", "b" }, np.dtypes.StringDType(na_object: null, coerce: false));
            Action withNA = () => np.searchsorted(hayNA, np.array(new[] { "a" }));
            withNA.Should().Throw<TypeError>()
                .WithMessage("Cannot cast array data from dtype('<U1') to StringDType(na_object=None) according to the rule 'safe'");
        }
    }
}
