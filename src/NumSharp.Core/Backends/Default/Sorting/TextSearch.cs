using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends.Unmanaged.Pooling;

// =============================================================================
// TextSearch.cs — np.searchsorted over fixed-width text (S / U): NumPy's npy_binsearch / npy_argbinsearch answers
// =============================================================================
//
// WHAT NUMPY DOES (2.4.2, npysort/binsearch.cpp; probed)
//   PyArray_SearchSorted converts the keys and the haystack to their common descriptor, then runs the GENERIC binary search
//   with the dtype's compare function (STRING_compare / UNICODE_compare: unsigned code units over the common width). For
//   each key, in C order:
//       if (cmp(last_key, key) < 0)  max_idx = arr_len;            // ('right': <= 0) — the monotonic carry: an ascending
//       else { min_idx = 0; max_idx = max_idx < arr_len ? max_idx + 1 : arr_len; }   // key keeps the last lower bound
//       last_key = key;
//       while (min_idx < max_idx) {
//           mid = min_idx + ((max_idx - min_idx) >> 1);
//           if (cmp(arr[mid], key) < 0) min_idx = mid + 1;       // ('right': <= 0)
//           else max_idx = mid;
//       }
//       result = min_idx;
//   With a sorter, arr[mid] is arr[sorter[mid]], and a sorter index outside [0, arr_len) raises "Sorter index out of
//   range." — but only when the search reaches it. For a SORTED haystack every correct search gives one answer; for an
//   unsorted one the answer is this exact sequence of comparisons, carry included, so the port follows it step for step.
//
// HOW THIS FILE GETS THE SAME ANSWERS
//   Each comparison is answered as STRING_compare / UNICODE_compare answers it, cheaply. Two routes, by a cost rule (key the
//   haystack once the searches compare more often than there are haystack elements to key, by a margin):
//   * KEYED (many keys): the units every haystack element and every key share are measured first (vector compares against
//     haystack element 0, the haystack's far end and the keys first; a haystack of URLs sharing "https://site.example/"
//     shares 21 units), and the strings are keyed from the unit after them: order-preserving 64-bit prefix keys with ONE
//     packing chosen from both arrays (TextKeyer: S 8 bytes; U 8 units of 8 bits, else 16 / 21 / 32 bits). Different keys
//     decide a comparison; equal keys decide it only when they reach the end of both strings (EXACT keys), else the units
//     after them do. Then:
//       - SORTED: when the haystack never decreases in the search's order (one vector pass over its keys; through the
//         sorter, when there is one, whose values must all lie in the haystack), NumPy's search returns every key's bound
//         (the elements below it for 'left', not above it for 'right') whatever its carry does, so any correct search gives
//         its answers. Keys that never decrease are MERGED with the haystack (a galloping walk forward); other keys are
//         searched four at a time, branch-free and interleaved (the searches share their step lengths, which depend only
//         on the haystack's length, so their loads overlap).
//       - otherwise NumPy's loop step for step over the keys, its outcomes as branches (over an unsorted haystack they
//         repeat from key to key often enough that branches beat masks: 1.6x over 100K random keys).
//   * DIRECT (few keys against a long haystack): keying the whole haystack would cost more than the searches, so NumPy's
//     loop compares the strings themselves — the head first (a U string's first unit, an S string's first 8 bytes read
//     big-endian), the rest only on an equal head — with the next step's two candidates prefetched past about half of L2.
//     Each step's address depends on the previous comparison, as in NumPy's loop: this route runs at the haystack's
//     memory latency, near NumPy's speed.
//   Strings of different widths of one kind are compared as NumPy compares them after its cast to the common width: the
//   shorter is zero-padded (a key of a narrower element has zeros below its last unit, exactly that padding).
// =============================================================================

namespace NumSharp.Backends.Sorting
{
    /// <summary>
    ///     <c>np.searchsorted</c> over fixed-width text with NumPy 2.4.2's answers: the exact port of its generic binary
    ///     search (monotonic carry and sorter checks included), every comparison answered from prefix keys or the strings
    ///     (see the file header).
    /// </summary>
    internal static unsafe class TextSearch
    {
        /// <summary>Test hook: the number of calls that keyed the haystack (the KEYED route ran).</summary>
        internal static long KeyedRuns;

        /// <summary>Test hook: the number of KEYED calls that ran NumPy's search over exact keys (the haystack not sorted).</summary>
        internal static long ExactRuns;

        /// <summary>Test hook: the number of KEYED calls that took the SORTED route (the haystack sorted in search order).</summary>
        internal static long SortedRuns;

        /// <summary>Test hook: when true, the KEYED route always runs NumPy's search step for step (the SORTED route is
        ///     skipped), so tests can hold both to one answer.</summary>
        internal static bool SortedRouteDisabled;

        /// <summary>Test hook: the number of calls that compared strings directly (the DIRECT route ran).</summary>
        internal static long DirectRuns;

        /// <summary>Test hook: forces the KEYED route (true), the DIRECT route (false), or lets the cost rule choose (null).</summary>
        internal static bool? ForceKeyed;

        /// <summary>
        ///     Test hook: when true, the KEYED route keys from unit 0 instead of from the end of the shared prefix (the tie
        ///     comparisons then do the prefix's work), so tests can hold both paths to one answer.
        /// </summary>
        internal static bool PrefixSkipDisabled;

        /// <summary>
        ///     The haystack size (bytes the elements span) above which the DIRECT route prefetches: about half of a core's L2
        ///     (2 MB on the measured host), past which the deep steps of a search read L3.
        /// </summary>
        private const long DirectPrefetchMinBytes = 1L << 20;

        /// <summary>
        ///     The search of <paramref name="keys"/> in <paramref name="haystack"/>, both of ONE text kind (<c>S</c> or
        ///     <c>U</c>; the caller converted a number or the other kind to the common descriptor first).
        /// </summary>
        /// <param name="haystack">The 1-D haystack (any stride), at least one element.</param>
        /// <param name="keys">The keys (any shape and layout), at least one element.</param>
        /// <param name="left">True for <c>side='left'</c>, false for <c>'right'</c>.</param>
        /// <param name="sorter">The validated sorter as a C-contiguous int64 array of the haystack's length, or null.</param>
        /// <returns>A fresh C-contiguous int64 array of the keys' shape (0-d for a 0-d key).</returns>
        /// <exception cref="ValueError">A sorter index outside the haystack that the search reaches (<c>Sorter index out of
        ///     range.</c>, NumPy's text).</exception>
        /// <exception cref="OutOfMemoryException">The key scratch cannot be allocated.</exception>
        [NDScoped]
        internal static NDArray SearchSorted(NDArray haystack, NDArray keys, bool left, NDArray sorter)
        {
            int unit = haystack.dtype.kind == 'U' ? 4 : 1;
            int hayUnits = haystack.dtype.itemsize / unit;
            int keyUnits = keys.dtype.itemsize / unit;
            long n = haystack.size;
            // NumPy's keys are a C-contiguous array (PyArray_CheckFromAny with NPY_ARRAY_CARRAY_RO): read them in C order.
            NDArray k = keys.Shape.IsContiguous ? keys : NDArray.ascontiguousarray(keys, null);
            long m = k.size;
            var result = new NDArray(NPTypeCode.Int64,
                keys.ndim == 0 ? Shape.NewScalar() : Shape.Adopt((long[])keys.Shape.dimensions.Clone()), false);

            byte* hay = (byte*)haystack.Address + haystack.Shape.offset * (long)haystack.dtype.itemsize;
            long hayStride = haystack.Shape.strides[0] * (long)haystack.dtype.itemsize;
            byte* key = (byte*)k.Address + k.Shape.offset * (long)k.dtype.itemsize;
            long keyStride = k.dtype.itemsize;
            long* sort = sorter is null ? null : (long*)sorter.Address + sorter.Shape.offset;
            long* ret = (long*)result.Address;
            using var search = new Search(unit, hay, hayStride, hayUnits, n, key, keyStride, keyUnits, m, sort, left, ret);
            if (UseKeys(n, m))
            {
                KeyedRuns++;
                search.Keyed();
            }
            else
            {
                DirectRuns++;
                search.Direct();
            }
            return result;
        }

        /// <summary>
        ///     The cost rule: key the haystack when the keys' binary searches compare more often than there are haystack
        ///     elements to key, by a margin. A full string comparison costs about eight key comparisons (it reads two strings
        ///     at random places), and keying an element about one, so keying pays once <c>m · 8 · log2(n) &gt; n</c>.
        /// </summary>
        /// <param name="n">The haystack length.</param>
        /// <param name="m">The number of keys.</param>
        /// <returns>True for the KEYED route.</returns>
        private static bool UseKeys(long n, long m)
        {
            if (ForceKeyed is bool forced)
                return forced;
            int log = 64 - BitOperations.LeadingZeroCount((ulong)n);
            return m * 8L * log > n;
        }

        /// <summary>
        ///     NumPy's <c>STRING_compare</c> / <c>UNICODE_compare</c> after the cast to the common width, from unit
        ///     <paramref name="fromUnit"/> on: the first differing code unit decides (unsigned); past the shorter string's end
        ///     its zero padding is compared with the longer one's units.
        /// </summary>
        /// <param name="a">The first string.</param>
        /// <param name="unitsA">Its width in code units.</param>
        /// <param name="b">The second string.</param>
        /// <param name="unitsB">Its width in code units.</param>
        /// <param name="fromUnit">The first unit compared (the units before it are known equal).</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <returns>-1, 0 or 1.</returns>
        internal static int CompareMixed(byte* a, int unitsA, byte* b, int unitsB, int fromUnit, int unit)
        {
            int common = Math.Min(unitsA, unitsB);
            if (fromUnit < common)
            {
                int bytes = (common - fromUnit) * unit;
                byte* x = a + fromUnit * unit, y = b + fromUnit * unit;
                int d = TextArgReduce.CommonPrefixBytes(x, y, bytes);
                if (d < bytes)
                {
                    if (unit == 1)
                        return x[d] < y[d] ? -1 : 1;
                    int u = d & ~3;
                    return Unsafe.ReadUnaligned<uint>(x + u) < Unsafe.ReadUnaligned<uint>(y + u) ? -1 : 1;
                }
                fromUnit = common;
            }
            // Equal over the common width: the longer string is greater exactly when it holds a non-zero unit past it.
            if (unitsA > unitsB)
                return new ReadOnlySpan<byte>(a + fromUnit * unit, (unitsA - fromUnit) * unit).IndexOfAnyExcept((byte)0) >= 0 ? 1 : 0;
            if (unitsB > unitsA)
                return new ReadOnlySpan<byte>(b + fromUnit * unit, (unitsB - fromUnit) * unit).IndexOfAnyExcept((byte)0) >= 0 ? -1 : 0;
            return 0;
        }

        /// <summary>
        ///     One comparison of two strings of one kind from unit <paramref name="fromUnit"/> on: of one width, the inlined
        ///     vector compare (<see cref="TextArgReduce.CompareFrom"/>); of two widths, <see cref="CompareMixed"/>.
        /// </summary>
        /// <param name="a">The first string.</param>
        /// <param name="unitsA">Its width in code units.</param>
        /// <param name="b">The second string.</param>
        /// <param name="unitsB">Its width in code units.</param>
        /// <param name="fromUnit">The first unit compared (the units before it are known equal).</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <returns>-1, 0 or 1.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int CompareStrings(byte* a, int unitsA, byte* b, int unitsB, int fromUnit, int unit)
            => unitsA == unitsB
                ? TextArgReduce.CompareFrom(a, b, fromUnit * unit, unitsA * unit, unit)
                : CompareMixed(a, unitsA, b, unitsB, fromUnit, unit);

        /// <summary>
        ///     One comparison of the DIRECT route (two whole strings): of one width, the HEAD decides first — a <c>U</c>
        ///     string's first unit, an <c>S</c> string's first 8 (or 4) bytes read big-endian — in a few instructions, and
        ///     only an equal head goes on to the vector compare of the rest; of two widths, <see cref="CompareMixed"/>.
        /// </summary>
        /// <param name="a">The first string.</param>
        /// <param name="unitsA">Its width in code units.</param>
        /// <param name="b">The second string.</param>
        /// <param name="unitsB">Its width in code units.</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <returns>-1, 0 or 1.</returns>
        /// <remarks>Most comparisons of a binary search are decided by the head; the vector compare it skips was a call
        ///     whose fixed cost exceeded the comparison (the search over a one-unit haystack took 6.4 ns a step).</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int CompareDirect(byte* a, int unitsA, byte* b, int unitsB, int unit)
        {
            if (unitsA == unitsB)
            {
                if (unit == 4)
                {
                    if (unitsA == 0)
                        return 0;
                    uint ua = Unsafe.ReadUnaligned<uint>(a), ub = Unsafe.ReadUnaligned<uint>(b);
                    if (ua != ub)
                        return ua < ub ? -1 : 1;
                    return unitsA == 1 ? 0 : TextArgReduce.CompareFrom(a, b, 4, unitsA * 4, 4);
                }
                if (unitsA >= 8)
                {
                    ulong ka = System.Buffers.Binary.BinaryPrimitives.ReverseEndianness(Unsafe.ReadUnaligned<ulong>(a));
                    ulong kb = System.Buffers.Binary.BinaryPrimitives.ReverseEndianness(Unsafe.ReadUnaligned<ulong>(b));
                    if (ka != kb)
                        return ka < kb ? -1 : 1;
                    return unitsA == 8 ? 0 : TextArgReduce.CompareFrom(a, b, 8, unitsA, 1);
                }
                if (unitsA >= 4)
                {
                    uint ka = System.Buffers.Binary.BinaryPrimitives.ReverseEndianness(Unsafe.ReadUnaligned<uint>(a));
                    uint kb = System.Buffers.Binary.BinaryPrimitives.ReverseEndianness(Unsafe.ReadUnaligned<uint>(b));
                    if (ka != kb)
                        return ka < kb ? -1 : 1;
                    return unitsA == 4 ? 0 : TextArgReduce.CompareFrom(a, b, 4, unitsA, 1);
                }
            }
            return CompareStrings(a, unitsA, b, unitsB, 0, unit);
        }

        /// <summary>Raises NumPy's error for a sorter index outside the haystack (kept out of line: the loops stay small).</summary>
        /// <exception cref="ValueError">Always: <c>Sorter index out of range.</c></exception>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowSorterOutOfRange() => throw new ValueError("Sorter index out of range.");

        /// <summary>An all-ones mask when <paramref name="condition"/> holds, else zero (a setcc and a negation, no branch).</summary>
        /// <param name="condition">The condition.</param>
        /// <returns><c>-1</c> or <c>0</c>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long Mask(bool condition) => -(long)Unsafe.As<bool, byte>(ref condition);

        /// <summary>
        ///     One search call: the operands, the key scratch of the KEYED route, and the two routes. Disposing returns the
        ///     scratch.
        /// </summary>
        private sealed class Search : IDisposable
        {
            /// <summary>1 for <c>S</c>, 4 for <c>U</c>.</summary>
            private readonly int _unit;

            /// <summary>Haystack element 0.</summary>
            private readonly byte* _hay;

            /// <summary>The haystack's byte stride (any sign).</summary>
            private readonly long _hayStride;

            /// <summary>The haystack's width in code units.</summary>
            private readonly int _hayUnits;

            /// <summary>The haystack length.</summary>
            private readonly long _n;

            /// <summary>Key 0 (the keys are C-contiguous).</summary>
            private readonly byte* _key;

            /// <summary>The keys' byte stride (their item size).</summary>
            private readonly long _keyStride;

            /// <summary>The keys' width in code units.</summary>
            private readonly int _keyUnits;

            /// <summary>The number of keys.</summary>
            private readonly long _m;

            /// <summary>The sorter (int64, contiguous), or null.</summary>
            private readonly long* _sort;

            /// <summary>True for <c>side='left'</c>.</summary>
            private readonly bool _left;

            /// <summary>The result slots, one per key, in C order.</summary>
            private readonly long* _ret;

            /// <summary>The haystack's prefix keys (KEYED route), in haystack order.</summary>
            private ulong* _hayKeys;

            /// <summary>The keys' prefix keys (KEYED route).</summary>
            private ulong* _keyKeys;

            /// <summary>The bytes taken for <see cref="_hayKeys"/>.</summary>
            private long _hayKeyBytes;

            /// <summary>The bytes taken for <see cref="_keyKeys"/>.</summary>
            private long _keyKeyBytes;

            /// <summary>The haystack's keys in the sorter's order (SORTED route with a sorter), or null.</summary>
            private ulong* _sortedKeys;

            /// <summary>The bytes taken for <see cref="_sortedKeys"/>.</summary>
            private long _sortedKeyBytes;

            /// <summary>The units the KEYED route's keys reach (the shared prefix included): equal keys are decided from here.</summary>
            private int _covered;

            /// <summary>Captures one call's operands.</summary>
            /// <param name="unit">1 or 4.</param>
            /// <param name="hay">Haystack element 0.</param>
            /// <param name="hayStride">The haystack's byte stride.</param>
            /// <param name="hayUnits">The haystack's width in units.</param>
            /// <param name="n">The haystack length.</param>
            /// <param name="key">Key 0.</param>
            /// <param name="keyStride">The keys' byte stride.</param>
            /// <param name="keyUnits">The keys' width in units.</param>
            /// <param name="m">The number of keys.</param>
            /// <param name="sort">The sorter, or null.</param>
            /// <param name="left">True for <c>side='left'</c>.</param>
            /// <param name="ret">The result slots.</param>
            internal Search(int unit, byte* hay, long hayStride, int hayUnits, long n, byte* key, long keyStride, int keyUnits,
                            long m, long* sort, bool left, long* ret)
            {
                _unit = unit;
                _hay = hay;
                _hayStride = hayStride;
                _hayUnits = hayUnits;
                _n = n;
                _key = key;
                _keyStride = keyStride;
                _keyUnits = keyUnits;
                _m = m;
                _sort = sort;
                _left = left;
                _ret = ret;
            }

            /// <summary>Returns the KEYED route's scratch to the pool (idempotent).</summary>
            public void Dispose()
            {
                if (_hayKeys != null) { SizeBucketedBufferPool.Return((IntPtr)_hayKeys, _hayKeyBytes); _hayKeys = null; }
                if (_keyKeys != null) { SizeBucketedBufferPool.Return((IntPtr)_keyKeys, _keyKeyBytes); _keyKeys = null; }
                if (_sortedKeys != null) { SizeBucketedBufferPool.Return((IntPtr)_sortedKeys, _sortedKeyBytes); _sortedKeys = null; }
            }

            /// <summary>
            ///     The DIRECT route: NumPy's search step for step, every comparison made on the strings.
            /// </summary>
            /// <exception cref="ValueError">A sorter index outside the haystack that the search reaches.</exception>
            internal void Direct()
            {
                // A haystack beyond about half of a core's L2 is read from L3 at the deep steps: the two elements the next
                // step may compare are prefetched there. A smaller one is in L2 already, and its prefetches only took load
                // slots (a categorical haystack's search predicts its own path, so it loses most from them).
                bool prefetch = Sse.IsSupported && _n * Math.Abs(_hayStride) > DirectPrefetchMinBytes;
                // One copy of the loop per (side, prefetch) pair, each with its constants folded in.
                if (_left)
                {
                    if (prefetch) DirectLoop(true, true);
                    else DirectLoop(true, false);
                }
                else
                {
                    if (prefetch) DirectLoop(false, true);
                    else DirectLoop(false, false);
                }
            }

            /// <summary>
            ///     The DIRECT loop for one side: NumPy's carry and bound updates as branches (a categorical haystack, or keys
            ///     that follow each other, repeat their comparison outcomes, which the branch predictor follows ahead of the
            ///     loads), every comparison decided by the strings' heads where it can (<see cref="CompareDirect"/>).
            /// </summary>
            /// <param name="left">True for <c>side='left'</c> (a constant at each call site).</param>
            /// <param name="prefetch">Whether to prefetch the next step's two candidate elements (a constant at each call
            ///     site).</param>
            /// <exception cref="ValueError">A sorter index outside the haystack that the search reaches.</exception>
            /// <remarks>
            ///     Measured on 100 keys into a 100K haystack (median µs, branch-free with prefetch two steps ahead / branchy /
            ///     branchy with the next step prefetched): U8 random 35.8 / 39.7 / 35.7, U8 categorical 20.6 / 14.7 / 16.4,
            ///     U1 categorical 12.8 / 7.0 / 9.3, U24 shared prefix 48.9 / 54.4 / 47.3. Each step's address depends on the
            ///     previous comparison, as in NumPy's loop, so the route runs at the haystack's memory latency.
            /// </remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private void DirectLoop(bool left, bool prefetch)
            {
                int unit = _unit, hayUnits = _hayUnits, keyUnits = _keyUnits;
                byte* hay = _hay;
                long hayStride = _hayStride, n = _n, m = _m, keyStride = _keyStride;
                long* sort = _sort;
                long* ret = _ret;
                long minIdx = 0, maxIdx = n;
                byte* lastKey = _key;
                for (long i = 0; i < m; i++)
                {
                    byte* key = _key + i * keyStride;
                    int carry = CompareDirect(lastKey, keyUnits, key, keyUnits, unit);
                    if (left ? carry < 0 : carry <= 0)
                    {
                        maxIdx = n;
                    }
                    else
                    {
                        minIdx = 0;
                        maxIdx = maxIdx < n ? maxIdx + 1 : n;
                    }
                    lastKey = key;
                    while (minIdx < maxIdx)
                    {
                        long mid = minIdx + ((maxIdx - minIdx) >> 1);
                        long at = mid;
                        if (sort != null)
                        {
                            at = sort[mid];
                            if ((ulong)at >= (ulong)n)
                                ThrowSorterOutOfRange();
                        }
                        else if (prefetch)
                        {
                            // The midpoints of [mid + 1, max) and [min, mid): one of them is the next step's element.
                            Sse.Prefetch0(hay + ((mid + 1) + ((maxIdx - mid - 1) >> 1)) * hayStride);
                            Sse.Prefetch0(hay + (minIdx + ((mid - minIdx) >> 1)) * hayStride);
                        }
                        int c = CompareDirect(hay + at * hayStride, hayUnits, key, keyUnits, unit);
                        if (left ? c < 0 : c <= 0)
                            minIdx = mid + 1;
                        else
                            maxIdx = mid;
                    }
                    ret[i] = minIdx;
                }
            }

            /// <summary>
            ///     The KEYED route: the shared prefix measured, prefix keys for the haystack and the keys with one packing
            ///     chosen from both, then the SORTED route when the haystack is sorted in search order, else NumPy's search
            ///     step for step — over the keys alone when they are exact, else each comparison decided by the keys and, on a
            ///     tie, by the strings' units after them.
            /// </summary>
            /// <exception cref="ValueError">A sorter index outside the haystack that the search reaches.</exception>
            /// <exception cref="OutOfMemoryException">The key scratch cannot be allocated.</exception>
            internal void Keyed()
            {
                _hayKeyBytes = _n * sizeof(ulong);
                _keyKeyBytes = _m * sizeof(ulong);
                _hayKeys = (ulong*)SizeBucketedBufferPool.Take(_hayKeyBytes);
                _keyKeys = (ulong*)SizeBucketedBufferPool.Take(_keyKeyBytes);
                int lcp = PrefixSkipDisabled ? 0 : SharedPrefixUnits();
                int covered = ComputeSharedKeys(lcp);
                // Keys that reach the end of both strings are their order; otherwise an equal key defers to the units after.
                bool exact = covered >= Math.Max(_hayUnits, _keyUnits);
                _covered = covered;
                if (!SortedRouteDisabled && (exact ? SortedSearch(true) : SortedSearch(false)))
                {
                    SortedRuns++;
                    return;
                }
                if (exact)
                {
                    ExactRuns++;
                    if (_left)
                        ExactLoop(true);
                    else
                        ExactLoop(false);
                }
                else if (_left)
                    TieLoop(true);
                else
                    TieLoop(false);
            }

            /// <summary>
            ///     The SORTED route: when the haystack never decreases in the search's order (through the sorter, when there is
            ///     one), NumPy's search returns, for every key, the number of haystack elements below it ('left') or not above
            ///     it ('right') — whatever its carry did — so any correct search gives NumPy's answers. Sorted keys are then
            ///     MERGED with the haystack (a galloping walk that only moves forward), and other keys are searched four at a
            ///     time, interleaved and branch-free (the four searches share their step lengths, which depend only on the
            ///     haystack's length, so their loads overlap). With keys that end before the strings do, equal keys are
            ///     decided by the strings' units after them — in the sortedness checks and in the searches.
            /// </summary>
            /// <param name="exact">Whether the keys reach the end of every string (a constant at each call site).</param>
            /// <returns>True when the haystack is sorted in search order and the results are written; false (nothing written)
            ///     otherwise — an unsorted haystack, or a sorter value outside the haystack (NumPy raises only if its search
            ///     reaches that value, so the exact port decides).</returns>
            /// <exception cref="OutOfMemoryException">The sorter-ordered key scratch cannot be allocated.</exception>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private bool SortedSearch(bool exact)
            {
                ulong* sk = _hayKeys;
                if (_sort != null)
                {
                    _sortedKeyBytes = _n * sizeof(ulong);
                    _sortedKeys = (ulong*)SizeBucketedBufferPool.Take(_sortedKeyBytes);
                    for (long i = 0; i < _n; i++)
                    {
                        long s = _sort[i];
                        if ((ulong)s >= (ulong)_n)
                            return false;
                        _sortedKeys[i] = _hayKeys[s];
                    }
                    sk = _sortedKeys;
                }
                if (!Nondecreasing(sk, _n) || (!exact && !TiesAscending(sk, _n, haystack: true)))
                    return false;
                bool sortedKeys = Nondecreasing(_keyKeys, _m) && (exact || TiesAscending(_keyKeys, _m, haystack: false));
                // One copy of each search per (side, exactness) pair, each with its constants folded in.
                if (_left)
                {
                    if (sortedKeys) MergeBounds(sk, true, exact);
                    else InterleavedBounds(sk, true, exact);
                }
                else
                {
                    if (sortedKeys) MergeBounds(sk, false, exact);
                    else InterleavedBounds(sk, false, exact);
                }
                return true;
            }

            /// <summary>
            ///     Whether the strings behind equal adjacent keys ascend (keys that end before the strings do): every pair of
            ///     equal neighbours, found 4 pairs a 256-bit compare, is compared from the units after the keys.
            /// </summary>
            /// <param name="keys">The keys in order (nondecreasing).</param>
            /// <param name="count">Their number.</param>
            /// <param name="haystack">True for the haystack in search order, false for the keys.</param>
            /// <returns>True when no pair of equal keys has a greater string first.</returns>
            private bool TiesAscending(ulong* keys, long count, bool haystack)
            {
                long i = 0;
                if (Vector256.IsHardwareAccelerated)
                {
                    for (; i + 4 < count; i += 4)
                    {
                        uint equal = Vector256.Equals(Vector256.Load(keys + i), Vector256.Load(keys + i + 1)).ExtractMostSignificantBits();
                        while (equal != 0)
                        {
                            long j = i + BitOperations.TrailingZeroCount(equal);
                            equal &= equal - 1;
                            if (PairDescends(j, haystack))
                                return false;
                        }
                    }
                }
                for (; i + 1 < count; i++)
                    if (keys[i] == keys[i + 1] && PairDescends(i, haystack))
                        return false;
                return true;
            }

            /// <summary>Whether element <paramref name="j"/> (in search order) is greater than element j + 1, compared from
            ///     the units after the keys.</summary>
            /// <param name="j">The first element's position (in the haystack's search order, or among the keys).</param>
            /// <param name="haystack">True for the haystack in search order, false for the keys.</param>
            /// <returns>True when the pair is out of order.</returns>
            private bool PairDescends(long j, bool haystack)
            {
                if (haystack)
                    return CompareStrings(HaystackAt(j), _hayUnits, HaystackAt(j + 1), _hayUnits, _covered, _unit) > 0;
                return CompareStrings(_key + j * _keyStride, _keyUnits, _key + (j + 1) * _keyStride, _keyUnits, _covered, _unit) > 0;
            }

            /// <summary>The haystack element at position <paramref name="j"/> of the search order (through the sorter).</summary>
            /// <param name="j">The position (the sorter's values are known to be in range on the SORTED route).</param>
            /// <returns>A pointer to the element.</returns>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private byte* HaystackAt(long j) => _hay + (_sort != null ? _sort[j] : j) * _hayStride;

            /// <summary>
            ///     NumPy's comparison of the haystack element at search position <paramref name="j"/> with key
            ///     <paramref name="i"/> when their prefix keys are equal and do not reach the strings' ends: the units after
            ///     the keys decide. Kept out of line: in most data equal keys are rare, and the searches stay small.
            /// </summary>
            /// <param name="j">The haystack's search position.</param>
            /// <param name="i">The key's index.</param>
            /// <param name="left">True for <c>side='left'</c>.</param>
            /// <returns>Whether the haystack element moves the lower bound past it (below the key, or not above it for 'right').</returns>
            [MethodImpl(MethodImplOptions.NoInlining)]
            private bool TieLess(long j, long i, bool left)
            {
                int c = CompareStrings(HaystackAt(j), _hayUnits, _key + i * _keyStride, _keyUnits, _covered, _unit);
                return left ? c < 0 : c <= 0;
            }

            /// <summary>
            ///     NumPy's comparison of a string with key string <paramref name="key"/> when their prefix keys are equal and do
            ///     not reach the strings' ends: the units after the keys decide. Kept out of line, like <see cref="TieLess"/>.
            /// </summary>
            /// <param name="a">The string compared with the key (a haystack element, or the previous key for the carry).</param>
            /// <param name="unitsA">Its width in code units.</param>
            /// <param name="key">The key string.</param>
            /// <param name="left">True for <c>side='left'</c>.</param>
            /// <returns>Whether <paramref name="a"/> is below the key ('left') or not above it ('right').</returns>
            [MethodImpl(MethodImplOptions.NoInlining)]
            private bool TieBelow(byte* a, int unitsA, byte* key, bool left)
            {
                int c = CompareStrings(a, unitsA, key, _keyUnits, _covered, _unit);
                return left ? c < 0 : c <= 0;
            }

            /// <summary>
            ///     Whether the haystack element at search position <paramref name="j"/>, with key <paramref name="h"/>, moves
            ///     the lower bound past it for key <paramref name="i"/> (key <paramref name="k"/>): below the key ('left') or
            ///     not above it ('right'). Different keys decide; equal exact keys are equal strings; equal keys that end before
            ///     the strings do defer to <see cref="TieLess"/>.
            /// </summary>
            /// <param name="h">The haystack element's key.</param>
            /// <param name="k">The search key's key.</param>
            /// <param name="j">The haystack's search position.</param>
            /// <param name="i">The key's index.</param>
            /// <param name="left">True for <c>side='left'</c> (a constant).</param>
            /// <param name="exact">Whether the keys are exact (a constant).</param>
            /// <returns>True when the lower bound moves past the element.</returns>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private bool Below(ulong h, ulong k, long j, long i, bool left, bool exact)
            {
                if (exact || h != k)
                    return left ? h < k : h <= k;
                return TieLess(j, i, left);
            }

            /// <summary>
            ///     The bounds of nondecreasing keys in a sorted haystack: one forward walk. From the previous key's bound, the
            ///     walk gallops (steps of 1, 2, 4, …) to the first element that is not below the key and binary-searches the
            ///     last step, so a key far from the previous one costs a logarithm of the distance, not the distance.
            /// </summary>
            /// <param name="sk">The haystack's keys in search order (nondecreasing).</param>
            /// <param name="left">True for <c>side='left'</c> (a constant at each call site).</param>
            /// <param name="exact">Whether the keys are exact (a constant at each call site).</param>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private void MergeBounds(ulong* sk, bool left, bool exact)
            {
                ulong* kk = _keyKeys;
                long* ret = _ret;
                long n = _n, m = _m;
                long pos = 0;
                for (long i = 0; i < m; i++)
                {
                    ulong key = kk[i];
                    if (pos < n && Below(sk[pos], key, pos, i, left, exact))
                    {
                        // sk[lo] is below the key; gallop until sk[hi] is not (or hi passes the end).
                        long lo = pos, step = 1, hi = pos + 1;
                        while (hi < n && Below(sk[hi], key, hi, i, left, exact))
                        {
                            lo = hi;
                            step <<= 1;
                            hi = lo + step;
                        }
                        if (hi > n)
                            hi = n;
                        // The bound lies in (lo, hi].
                        long a = lo + 1, b = hi;
                        while (a < b)
                        {
                            long mid = a + ((b - a) >> 1);
                            if (Below(sk[mid], key, mid, i, left, exact))
                                a = mid + 1;
                            else
                                b = mid;
                        }
                        pos = a;
                    }
                    ret[i] = pos;
                }
            }

            /// <summary>
            ///     The bounds of keys in any order in a sorted haystack: four searches at a time, each the branch-free halving
            ///     search (the window <c>[p, p + len)</c> holds the bound; a step moves <c>p</c> past the lower half when its
            ///     last element is below the key, selected by a mask). The four searches take the same step lengths, so the
            ///     loop runs once for all four and their loads are independent.
            /// </summary>
            /// <param name="sk">The haystack's keys in search order (nondecreasing).</param>
            /// <param name="left">True for <c>side='left'</c> (a constant at each call site).</param>
            /// <param name="exact">Whether the keys are exact (a constant at each call site).</param>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private void InterleavedBounds(ulong* sk, bool left, bool exact)
            {
                ulong* kk = _keyKeys;
                long* ret = _ret;
                long n = _n, m = _m;
                long i = 0;
                for (; i + 4 <= m; i += 4)
                {
                    ulong k0 = kk[i], k1 = kk[i + 1], k2 = kk[i + 2], k3 = kk[i + 3];
                    long b0 = 0, b1 = 0, b2 = 0, b3 = 0;
                    long len = n;
                    while (len > 1)
                    {
                        long half = len >> 1;
                        b0 += half & Mask(Below(sk[b0 + half - 1], k0, b0 + half - 1, i, left, exact));
                        b1 += half & Mask(Below(sk[b1 + half - 1], k1, b1 + half - 1, i + 1, left, exact));
                        b2 += half & Mask(Below(sk[b2 + half - 1], k2, b2 + half - 1, i + 2, left, exact));
                        b3 += half & Mask(Below(sk[b3 + half - 1], k3, b3 + half - 1, i + 3, left, exact));
                        len -= half;
                    }
                    ret[i] = b0 + (Below(sk[b0], k0, b0, i, left, exact) ? 1 : 0);
                    ret[i + 1] = b1 + (Below(sk[b1], k1, b1, i + 1, left, exact) ? 1 : 0);
                    ret[i + 2] = b2 + (Below(sk[b2], k2, b2, i + 2, left, exact) ? 1 : 0);
                    ret[i + 3] = b3 + (Below(sk[b3], k3, b3, i + 3, left, exact) ? 1 : 0);
                }
                for (; i < m; i++)
                {
                    ulong k = kk[i];
                    long b = 0;
                    long len = n;
                    while (len > 1)
                    {
                        long half = len >> 1;
                        b += half & Mask(Below(sk[b + half - 1], k, b + half - 1, i, left, exact));
                        len -= half;
                    }
                    ret[i] = b + (Below(sk[b], k, b, i, left, exact) ? 1 : 0);
                }
            }

            /// <summary>
            ///     Whether <paramref name="count"/> keys never decrease (unsigned): four adjacent pairs a 256-bit compare,
            ///     biased so that the signed compare orders them as unsigned values; stops at the first decrease.
            /// </summary>
            /// <param name="keys">The keys.</param>
            /// <param name="count">Their number.</param>
            /// <returns>True when every key is at most the next one.</returns>
            private static bool Nondecreasing(ulong* keys, long count)
            {
                long i = 0;
                if (Vector256.IsHardwareAccelerated)
                {
                    var bias = Vector256.Create(long.MinValue);
                    for (; i + 8 < count; i += 8)
                    {
                        var a0 = Vector256.Load((long*)keys + i) ^ bias;
                        var b0 = Vector256.Load((long*)keys + i + 1) ^ bias;
                        var a1 = Vector256.Load((long*)keys + i + 4) ^ bias;
                        var b1 = Vector256.Load((long*)keys + i + 5) ^ bias;
                        if ((Vector256.GreaterThan(a0, b0) | Vector256.GreaterThan(a1, b1)) != Vector256<long>.Zero)
                            return false;
                    }
                }
                for (; i + 1 < count; i++)
                    if (keys[i] > keys[i + 1])
                        return false;
                return true;
            }

            /// <summary>
            ///     NumPy's search over exact keys, step for step (the haystack is not known to be sorted, so the comparison
            ///     sequence decides the answer): every comparison is a key comparison. The outcomes are branches: over an
            ///     unsorted haystack and correlated keys they repeat from one key to the next and predict well, where masks
            ///     would put every step's load on the critical path.
            /// </summary>
            /// <param name="left">True for <c>side='left'</c> (a constant at each call site).</param>
            /// <exception cref="ValueError">A sorter index outside the haystack that the search reaches.</exception>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private void ExactLoop(bool left)
            {
                ulong* hk = _hayKeys;
                ulong* kk = _keyKeys;
                long* sort = _sort;
                long* ret = _ret;
                long n = _n, m = _m;
                long minIdx = 0, maxIdx = n;
                ulong lastKey = kk[0];
                for (long i = 0; i < m; i++)
                {
                    ulong key = kk[i];
                    // The monotonic carry: an ascending key keeps the lower bound and searches to the end; any other key
                    // restarts from 0 with the old upper bound plus one (capped at n).
                    if (left ? lastKey < key : lastKey <= key)
                    {
                        maxIdx = n;
                    }
                    else
                    {
                        minIdx = 0;
                        maxIdx = maxIdx < n ? maxIdx + 1 : n;
                    }
                    lastKey = key;
                    while (minIdx < maxIdx)
                    {
                        long mid = minIdx + ((maxIdx - minIdx) >> 1);
                        long at = mid;
                        if (sort != null)
                        {
                            at = sort[mid];
                            if ((ulong)at >= (ulong)n)
                                ThrowSorterOutOfRange();
                        }
                        if (left ? hk[at] < key : hk[at] <= key)
                            minIdx = mid + 1;
                        else
                            maxIdx = mid;
                    }
                    ret[i] = minIdx;
                }
            }


            /// <summary>
            ///     NumPy's search over keys that end before the strings do, step for step: a comparison is decided by the keys
            ///     when they differ, else by the strings from the units after the keys.
            /// </summary>
            /// <param name="left">True for <c>side='left'</c> (a constant at each call site).</param>
            /// <exception cref="ValueError">A sorter index outside the haystack that the search reaches.</exception>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private void TieLoop(bool left)
            {
                ulong* hk = _hayKeys;
                ulong* kk = _keyKeys;
                long* sort = _sort;
                long* ret = _ret;
                byte* hay = _hay;
                long hayStride = _hayStride, keyStride = _keyStride, n = _n, m = _m;
                int hayUnits = _hayUnits, keyUnits = _keyUnits;
                long minIdx = 0, maxIdx = n;
                long last = 0;
                for (long i = 0; i < m; i++)
                {
                    byte* key = _key + i * keyStride;
                    ulong k = kk[i];
                    ulong lk = kk[last];
                    bool ascend = lk != k ? lk < k : TieBelow(_key + last * keyStride, keyUnits, key, left);
                    if (ascend)
                    {
                        maxIdx = n;
                    }
                    else
                    {
                        minIdx = 0;
                        maxIdx = maxIdx < n ? maxIdx + 1 : n;
                    }
                    last = i;
                    while (minIdx < maxIdx)
                    {
                        long mid = minIdx + ((maxIdx - minIdx) >> 1);
                        long at = mid;
                        if (sort != null)
                        {
                            at = sort[mid];
                            if ((ulong)at >= (ulong)n)
                                ThrowSorterOutOfRange();
                        }
                        ulong h = hk[at];
                        // Different keys decide (the common case: one branch on the comparison itself); equal keys are
                        // decided by the strings' units after the keys, out of line.
                        if (h == k ? TieBelow(hay + at * hayStride, hayUnits, key, left) : h < k)
                            minIdx = mid + 1;
                        else
                            maxIdx = mid;
                    }
                    ret[i] = minIdx;
                }
            }

            /// <summary>
            ///     The code units every haystack element and every key share with haystack element 0, within the narrower of
            ///     the two widths: every comparison of the search agrees on them, so the keys can start after them. The
            ///     elements most likely to differ are visited first — the haystack's last element (the far end of a sorted
            ///     haystack), then the keys — so the measure usually ends after a few vector compares.
            /// </summary>
            /// <returns>The shared units (0 when none).</returns>
            private int SharedPrefixUnits()
            {
                int common = Math.Min(_hayUnits, _keyUnits);
                if (common == 0)
                    return 0;
                int bytes = common * _unit;
                byte* first = _hay;
                bytes = TextArgReduce.CommonPrefixBytes(first, _hay + (_n - 1) * _hayStride, bytes);
                for (long i = 0; i < _m && bytes >= _unit; i++)
                    bytes = TextArgReduce.CommonPrefixBytes(first, _key + i * _keyStride, bytes);
                for (long i = 1; i < _n - 1 && bytes >= _unit; i++)
                    bytes = TextArgReduce.CommonPrefixBytes(first, _hay + i * _hayStride, bytes);
                return bytes / _unit;
            }

            /// <summary>
            ///     Keys the haystack and the keys from unit <paramref name="lcp"/> on with one packing: <c>S</c> 8 bytes;
            ///     <c>U</c> the 8-bit packing when every keyed unit of both fits 8 bits, else the narrowest wider packing that
            ///     holds both (both re-keyed). An array narrower than <paramref name="lcp"/> plus one unit has nothing left to
            ///     key: its keys are zero, the zero padding.
            /// </summary>
            /// <param name="lcp">The shared prefix in units (below or at the narrower width).</param>
            /// <returns>The units the keys reach (the shared prefix included; of the wider width — a narrower string's key is
            ///     zero-padded).</returns>
            private int ComputeSharedKeys(int lcp)
            {
                var hayKeyer = new TextKeyer(_unit, _hayUnits);
                var keyKeyer = new TextKeyer(_unit, _keyUnits);
                int widest = Math.Max(_hayUnits, _keyUnits) - lcp;
                if (widest <= 0)
                {
                    // Nothing after the shared prefix in either array: every string is equal, every key zero.
                    new Span<ulong>(_hayKeys, checked((int)_n)).Clear();
                    new Span<ulong>(_keyKeys, checked((int)_m)).Clear();
                    return lcp;
                }
                uint* ors = stackalloc uint[4];
                ors[0] = ors[1] = ors[2] = ors[3] = 0;
                bool hayKeyed = _hayUnits > lcp, keyKeyed = _keyUnits > lcp;
                if (hayKeyed)
                    hayKeyer.ProbeUnits8(_hay, _hayStride, null, _n, lcp, _hayKeys, ors);
                else
                    new Span<ulong>(_hayKeys, checked((int)_n)).Clear();
                if (keyKeyed)
                    keyKeyer.ProbeUnits8(_key, _keyStride, null, _m, lcp, _keyKeys, ors);
                else
                    new Span<ulong>(_keyKeys, checked((int)_m)).Clear();
                if (_unit == 1)
                    return lcp + Math.Min(8, widest);
                int bits = TextKeyer.PackingFor(ors, widest, out int units);
                if (bits == 8)
                    return lcp + units;
                if (hayKeyed)
                    hayKeyer.KeysWith(bits, _hay, _hayStride, null, _n, lcp, _hayKeys);
                if (keyKeyed)
                    keyKeyer.KeysWith(bits, _key, _keyStride, null, _m, lcp, _keyKeys);
                return lcp + units;
            }
        }
    }
}
