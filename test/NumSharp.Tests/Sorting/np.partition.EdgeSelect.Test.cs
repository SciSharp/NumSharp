using System;
using System.Linq;

namespace NumSharp.Tests.Sorting
{
    /// <summary>
    ///     np.partition / np.argpartition on the edge selection (<c>QuickSelect.TryEdgePartition</c> /
    ///     <c>TryEdgeArgPartition</c>): a single kth within 64 positions of either end of a contiguous line of at least 256
    ///     elements, for the ten dtypes with vector lanes. The contract is NumPy's (the kth value equals the sorted array's,
    ///     the two sides bound it, the output is a permutation / the values a rearrangement); the data are permutations
    ///     <c>(i * 7919) mod n</c>, so <c>np.partition(a, kth)[kth] == kth</c> in NumPy 2.4.2 by construction, plus
    ///     duplicate-heavy, sorted, reversed, NaN and signed-zero lines that exercise the path's declines and its tie
    ///     handling.
    /// </summary>
    [TestClass]
    public class PartitionEdgeSelectTests
    {
        /// <summary>The permutation <c>(i * 7919) mod n</c> of <c>0 .. n-1</c> (7919 is prime, so any n it does not divide
        ///     gives a permutation), as float64 for the caller to cast.</summary>
        /// <param name="n">The length.</param>
        /// <returns>The values, each of 0 .. n-1 once.</returns>
        private static NDArray Permutation(int n)
            => np.array(Enumerable.Range(0, n).Select(i => (double)((long)i * 7919 % n)).ToArray());

        /// <summary>Asserts argpartition's contract on a 1-D NaN-free <paramref name="a"/>: <paramref name="g"/> is an
        ///     int64 permutation, <c>a[g[kth]]</c> is the kth order statistic, and every other gathered value lies on its
        ///     side of it.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="a">The input line.</param>
        /// <param name="g">np.argpartition's result.</param>
        /// <param name="kth">The kth.</param>
        private static void AssertArgPartition<T>(NDArray a, NDArray g, int kth) where T : unmanaged, IComparable<T>
        {
            g.typecode.Should().Be(NPTypeCode.Int64);
            T[] values = a.ToArray<T>();
            long[] idx = g.ToArray<long>();
            int n = values.Length;
            var seen = new bool[n];
            foreach (long i in idx)
            {
                (i >= 0 && i < n && !seen[i]).Should().BeTrue("argpartition must return a permutation");
                seen[i] = true;
            }
            T[] sorted = (T[])values.Clone();
            Array.Sort(sorted);
            T kv = values[idx[kth]];
            kv.CompareTo(sorted[kth]).Should().Be(0, $"a[g[{kth}]] must be the kth order statistic");
            int bad = -1;
            for (int i = 0; i < n && bad < 0; i++)
                if (i < kth ? values[idx[i]].CompareTo(kv) > 0 : i > kth && values[idx[i]].CompareTo(kv) < 0)
                    bad = i;
            bad.Should().Be(-1, $"every gathered value must lie on its side of kth={kth}");
        }

        /// <summary>Asserts partition's contract on a 1-D NaN-free <paramref name="a"/>: <paramref name="p"/> holds the
        ///     same values, <c>p[kth]</c> is the kth order statistic, and every other value lies on its side of it.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="a">The input line.</param>
        /// <param name="p">np.partition's result.</param>
        /// <param name="kth">The kth.</param>
        private static void AssertPartition<T>(NDArray a, NDArray p, int kth) where T : unmanaged, IComparable<T>
        {
            T[] values = a.ToArray<T>();
            T[] result = p.ToArray<T>();
            T[] sortedIn = (T[])values.Clone();
            Array.Sort(sortedIn);
            T[] sortedOut = (T[])result.Clone();
            Array.Sort(sortedOut);
            sortedOut.SequenceEqual(sortedIn).Should().BeTrue("partition must only rearrange the values");
            result[kth].CompareTo(sortedIn[kth]).Should().Be(0, $"p[{kth}] must be the kth order statistic");
            int bad = -1;
            for (int i = 0; i < result.Length && bad < 0; i++)
                if (i < kth ? result[i].CompareTo(result[kth]) > 0 : i > kth && result[i].CompareTo(result[kth]) < 0)
                    bad = i;
            bad.Should().Be(-1, $"every value must lie on its side of kth={kth}");
        }

        /// <summary>Runs both functions on <paramref name="a"/> at each kth and checks their contracts.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="a">The input line.</param>
        /// <param name="kths">The kths, one call each.</param>
        private static void AssertBoth<T>(NDArray a, params int[] kths) where T : unmanaged, IComparable<T>
        {
            foreach (int kth in kths)
            {
                using NDArray g = np.argpartition(a, kth);
                AssertArgPartition<T>(a, g, kth);
                using NDArray p = np.partition(a, kth);
                AssertPartition<T>(a, p, kth);
            }
        }

        /// <summary>
        ///     np.partition(a, kth)[kth] == kth over a float32 permutation of 0..n-1 (NumPy 2.4.2's answer by
        ///     construction) at both ends: m = 1..65 on the largest side (the float kth = n - 1 keeps NumPy's
        ///     last-maximum pass, m = 65 the quickselect) and m = 4..65 on the smallest (kth &lt; 3 keeps NumPy's
        ///     dumb_select).
        /// </summary>
        [TestMethod]
        public void TopAndBottom_Float32_KthValueIsKth()
        {
            // np.partition(a, kth)[kth] == kth for a permutation of 0..n-1 (NumPy 2.4.2), at both ends: m = 1..64 on the
            // largest side (kth = n - m; the float kth = n - 1 keeps NumPy's last-maximum pass) and m = 4..64 on the
            // smallest (kth = m - 1; kth < 3 keeps NumPy's dumb_select).
            const int n = 8192 + 1;
            using NDArray a = Permutation(n).astype(np.float32);
            foreach (int kth in new[] { n - 1, n - 2, n - 5, n - 16, n - 63, n - 64, n - 65, 3, 4, 15, 63, 64 })
            {
                using NDArray p = np.partition(a, kth);
                p.GetSingle(kth).Should().Be(kth);
                using NDArray g = np.argpartition(a, kth);
                a.GetSingle((int)g.GetInt64(kth)).Should().Be(kth);
            }
            AssertBoth<float>(a, n - 2, n - 5, n - 64, 3, 63);
        }

        /// <summary>
        ///     The ten dtypes with vector lanes, signed and unsigned, each over a permutation shifted into its range;
        ///     the 8-bit lines are 256 long, the shortest the path serves (m up to 8 there).
        /// </summary>
        [TestMethod]
        public void EveryVectorDtype_TopAndBottom()
        {
            // The ten dtypes with vector lanes, signed and unsigned, each over a permutation shifted to its range.
            const int n = 4096 + 3;
            using NDArray perm = Permutation(n);
            using NDArray shifted = perm - 2000.0;
            AssertBoth<double>(perm.astype(np.float64), n - 5, 4);
            AssertBoth<int>(shifted.astype(np.int32), n - 5, n - 40, 4, 39);
            AssertBoth<uint>(perm.astype(np.uint32), n - 5, 4);
            AssertBoth<long>(shifted.astype(np.int64), n - 5, 4);
            AssertBoth<ulong>(perm.astype(np.uint64), n - 5, 4);
            AssertBoth<short>(shifted.astype(np.int16), n - 5, 4);
            AssertBoth<ushort>(perm.astype(np.uint16), n - 5, 4);
            // 8-bit lines: 256 elements, the shortest the path serves (m up to 8 there).
            using NDArray perm256 = Permutation(256 + 1)["0:256"];
            using NDArray perm256Copy = perm256.copy();
            using NDArray signed256 = perm256Copy - 128.0;
            AssertBoth<sbyte>(signed256.astype(np.int8), 255 - 4, 250, 4, 7);
            AssertBoth<byte>(perm256Copy.astype(np.uint8), 255 - 4, 250, 4, 7);
        }

        /// <summary>
        ///     uint64 values spread over the whole range (a wrapping Fibonacci hash): the five largest are above
        ///     long.MaxValue, which a signed lane compare would rank below every small value.
        /// </summary>
        [TestMethod]
        public void UInt64_AboveInt64Max_ComparesUnsigned()
        {
            // Values spread over the whole uint64 range (a wrapping Fibonacci hash): the top five are above
            // long.MaxValue, which a signed lane compare would rank below every small value.
            const int n = 1000;
            ulong[] v = Enumerable.Range(0, n).Select(i => unchecked((ulong)i * 0x9E3779B97F4A7C15UL)).ToArray();
            using NDArray a = np.array(v);
            AssertBoth<ulong>(a, n - 5, 4);
            using NDArray g = np.argpartition(a, n - 5);
            ulong[] top = g.ToArray<long>().Skip(n - 5).Select(i => v[i]).ToArray();
            top.Should().OnlyContain(x => x > long.MaxValue);
        }

        /// <summary>
        ///     Ties everywhere: a line of three distinct values, and a constant line (nothing is admitted past the
        ///     fill).
        /// </summary>
        [TestMethod]
        public void DuplicatesAndConstantLines()
        {
            // Ties everywhere: three distinct values, and a constant line (nothing is ever admitted past the fill).
            const int n = 3000;
            using NDArray dups = np.array(Enumerable.Range(0, n).Select(i => (double)(i * 7 % 3)).ToArray()).astype(np.int32);
            AssertBoth<int>(dups, n - 5, n - 64, 4, 63);
            using NDArray constant = np.full(new Shape(n), 7.5, np.float32);
            AssertBoth<float>(constant, n - 5, 4);
        }

        /// <summary>
        ///     An ascending line beats the kept set at every element when the largest are wanted (a descending one when
        ///     the smallest are): the selection gives up after its admission budget and the quickselect answers,
        ///     correctly.
        /// </summary>
        [TestMethod]
        public void SortedLines_TheAdmissionBudgetFallsBack()
        {
            // An ascending line beats the kept set at every element when the largest are wanted (and a descending one
            // when the smallest are): the selection gives up after its admission budget and the quickselect answers.
            const int n = 51200;
            using NDArray ascending = np.arange(n).astype(np.float32);
            using NDArray descending = np.arange(n, 0, -1).astype(np.int64);
            AssertBoth<float>(ascending, n - 5, n - 64, 4);
            AssertBoth<long>(descending, 4, 63, n - 5);
        }

        /// <summary>
        ///     A NaN makes the selection decline before it writes anything, so the float line kernels' compaction runs:
        ///     argpartition's last indices are the NaN positions in encounter order, partition's last values are NaN,
        ///     and the kth value is the non-NaN order statistic.
        /// </summary>
        [TestMethod]
        public void NaNLine_KeepsTheEncounterOrderNaNTail()
        {
            // A NaN makes the selection decline before it writes anything, so the float line kernels' compaction runs:
            // argpartition's last indices are the NaN positions in encounter order, partition's last values are NaN.
            const int n = 1000;
            double[] v = Enumerable.Range(0, n).Select(i => (double)((long)i * 7919 % n)).ToArray();
            int[] nanAt = { 17, 500, 999 };
            foreach (int i in nanAt)
                v[i] = double.NaN;
            foreach (NPTypeCode tc in new[] { NPTypeCode.Single, NPTypeCode.Double })
            {
                using NDArray a = np.array(v).astype(tc);
                foreach (int kth in new[] { n - 5, 4 })
                {
                    using NDArray g = np.argpartition(a, kth);
                    long[] idx = g.ToArray<long>();
                    idx.Skip(n - 3).Should().Equal(17L, 500L, 999L);
                    // The non-NaN values are 0..999 minus the three at the NaN positions; the kth is among them.
                    double[] finite = v.Where(x => !double.IsNaN(x)).OrderBy(x => x).ToArray();
                    double want = kth < finite.Length ? finite[kth] : double.NaN;
                    using NDArray gathered = a.astype(np.float64)[g];
                    gathered.GetDouble(kth).Should().Be(want);
                    using NDArray p = np.partition(a, kth);
                    using NDArray p64 = p.astype(np.float64);
                    p64.GetDouble(kth).Should().Be(want);
                    p64["-3:"].ToArray<double>().Should().OnlyContain(x => double.IsNaN(x));
                }
            }
        }

        /// <summary>
        ///     Values move as whole float64s: the counts of the -0.0 and +0.0 bit patterns survive a partition at
        ///     either end.
        /// </summary>
        [TestMethod]
        public void SignedZeros_PartitionMovesWholeValues()
        {
            // A zero keeps its sign through the moves: the counts of -0.0 and +0.0 bit patterns are unchanged.
            const int n = 512;
            double[] v = Enumerable.Range(0, n).Select(i => i % 3 == 0 ? -0.0 : i % 3 == 1 ? 0.0 : -(double)i).ToArray();
            using NDArray a = np.array(v);
            foreach (int kth in new[] { n - 5, 4 })
            {
                using NDArray p = np.partition(a, kth);
                double[] r = p.ToArray<double>();
                r.Count(x => BitConverter.DoubleToInt64Bits(x) == BitConverter.DoubleToInt64Bits(-0.0))
                    .Should().Be(v.Count(x => BitConverter.DoubleToInt64Bits(x) == BitConverter.DoubleToInt64Bits(-0.0)));
                r.Count(x => BitConverter.DoubleToInt64Bits(x) == 0L).Should().Be(v.Count(x => BitConverter.DoubleToInt64Bits(x) == 0L));
                AssertPartition<double>(a, p, kth);
            }
        }

        /// <summary>
        ///     Rows of a C-contiguous 2-D array are contiguous lines (the edge selection), its columns strided ones
        ///     (the quickselect); both match the sorted array at kth.
        /// </summary>
        [TestMethod]
        public void TwoDimensional_RowsAndColumns()
        {
            // Rows are contiguous lines (the edge selection); columns are strided (the quickselect). Both match the
            // sorted array at kth.
            using NDArray perm = Permutation(8 * 4096 + 1)["0:32768"];
            using NDArray m = perm.copy().astype(np.float32).reshape(8, 4096);
            using NDArray sortedRows = np.sort(m, 1);
            using NDArray sortedCols = np.sort(m, 0);
            foreach ((int axis, int kth, NDArray sorted) in new[] { (1, 4096 - 5, sortedRows), (1, 4, sortedRows), (0, 8 - 5, sortedCols) })
            {
                using NDArray g = np.argpartition(m, kth, axis);
                using NDArray gathered = np.take_along_axis(m, g, axis);
                using NDArray p = np.partition(m, kth, axis);
                using NDArray kthIndex = np.array(new[] { kth });
                using NDArray want = np.take(sorted, kthIndex, axis);
                using NDArray fromArg = np.take(gathered, kthIndex, axis);
                using NDArray fromPart = np.take(p, kthIndex, axis);
                np.array_equal(fromArg, want).Should().BeTrue($"argpartition kth values, axis {axis}");
                np.array_equal(fromPart, want).Should().BeTrue($"partition kth values, axis {axis}");
            }
        }

        /// <summary>
        ///     ndarray.partition runs the same line kernel in place.
        /// </summary>
        [TestMethod]
        public void InPlacePartition_ContiguousLine()
        {
            // ndarray.partition runs the same line kernel in place.
            const int n = 2048 + 1;
            using NDArray a = Permutation(n).astype(np.int64);
            using NDArray before = a.copy();
            a.partition(n - 5);
            a.GetInt64(n - 5).Should().Be(n - 5);
            AssertPartition<long>(before, a, n - 5);
        }

        /// <summary>
        ///     The edge selection's own argpartition arrangement (NumPy's is not contractual): every index outside the
        ///     kth's block in ascending order, which a quickselect scatters, so this pins that a top-5 over a long
        ///     random line takes the path.
        /// </summary>
        [TestMethod]
        public void EngagementSignature_UnselectedIndicesAscend()
        {
            // How the edge selection arranges an argpartition (not NumPy's arrangement, which is not contractual): every
            // index outside the kth's block in ascending order. A quickselect scatters them, so this pins that a top-5
            // over a long random line takes the path.
            const int n = 51200;
            np.random.seed(42);
            using NDArray r = np.random.randn(n);
            using NDArray a = r.astype(np.float32);
            using NDArray g = np.argpartition(a, n - 5);
            long[] idx = g.ToArray<long>();
            bool ascending = true;
            for (int i = 1; i < n - 5 && ascending; i++)
                ascending = idx[i] > idx[i - 1];
            ascending.Should().BeTrue();
            AssertArgPartition<float>(a, g, n - 5);
        }
    }
}
