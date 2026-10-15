using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged.Pooling;

// =============================================================================
// TextSort.cs — sort / argsort / partition / argpartition of fixed-width text (S / U), along an axis
// =============================================================================
//
// WHAT NUMPY DOES (2.4.2, probed)
//   * The default kind (kind=None / 'quicksort') and 'heapsort' (PyArray_Sort / PyArray_ArgSort clear the heapsort bit for
//     a legacy dtype) run string_quicksort_ / string_aquicksort_ (npysort/quicksort.cpp): introsort — median of three,
//     insertion sort below SMALL_QUICKSORT + 1 = 16 elements, string_aheapsort_ once the depth budget 2 * msb(n) runs out —
//     comparing with STRING_LT / UNICODE_LT (unsigned code units over the whole item width). It is not stable: with
//     duplicates the index order is the algorithm's ([19, 37, 33, 30, ...] for 40 elements of three values, where a stable
//     sort gives [4, 5, 8, 9, ...]). A line of at most 16 elements is a plain insertion sort, so it IS stable.
//   * The stable kinds ('stable', 'mergesort', stable=True) run the string timsort: one permutation, ties in input order.
//   * np.partition / np.argpartition have no selection loop for strings, so PyArray_(Arg)Partition falls back to the generic
//     npy_quicksort / npy_aquicksort with the dtype's compare function: the same introsort over the same comparisons (its
//     extra `pi < pj` scan guards stop exactly where the unguarded scans stop), so argpartition returns argsort's
//     permutation (probed), and partition the sorted values.
//   * Items of size zero are never compared (`if (len == 0) return 0;`): every kind returns the identity.
//
// HOW THIS FILE GETS THE SAME ANSWERS (plan D3 / D19)
//   1. Keys. Each element of a line gets an order-preserving 64-bit prefix key (the first 8 bytes of an S element; the first
//      8, 4, 3 or 2 units of a U element, by the largest unit seen — ILKernelGenerator.GetTextSortKeyKernel).
//   2. A stable radix sort of the keys (RadixSort.ArgSortU64; insertion below 121 elements), then a stable TIE PASS over the
//      runs of equal keys whose elements are wider than the key: the next units are keyed and radix-sorted inside the run,
//      recursively; runs of at most 16 elements are insertion-sorted by comparing their remaining units. The result is the
//      stable order of the full strings: the answer of the stable kinds, and (equal strings being byte-identical at one
//      width) of every value sort.
//   3. The unstable kinds of an argsort need NumPy's own tie order. When no two adjacent strings of the stable order are
//      equal, every correct sort gives that one order, so it is returned (the DISTINCT-STRINGS SHORTCUT). Otherwise each
//      element gets its dense RANK in the stable order and NumPy's string_aquicksort_, ported statement for statement, runs
//      over (rank, index) records: rank order is string order and rank equality string equality, so each of its comparisons
//      has NumPy's answer, its swaps are NumPy's, and the permutation it leaves is NumPy's — ties included. A rank compare is
//      one integer compare, where NumPy compares strings through the index array.
//
// LINES
//   DriveLines walks every line along the axis with NDIter: the axis is dropped (op_axes) and the iteration runs with
//   EXTERNAL_LOOP, so ONE kernel call receives a whole block of lines (the first line's start per operand, the byte
//   distance between consecutive lines, their count) — rows of 10 strings cost ~140 ns each when the iterator called the
//   kernel once per line, the cost of the numeric argsort's driver. The Job owns one line's scratch (keys, indices, radix
//   histograms, the duplicate flags, a line copy for the in-place value sort) and reuses it for every line.
// =============================================================================

namespace NumSharp.Backends.Sorting
{
    /// <summary>
    ///     Sorting of fixed-width text along an axis with NumPy 2.4.2's answers: the values sort, the stable argsort and the
    ///     default argsort with NumPy's introsort tie order (see the file header for the algorithm).
    /// </summary>
    internal static unsafe class TextSort
    {
        /// <summary>
        ///     NumPy's <c>SMALL_QUICKSORT</c>: a range of more than this many elements beyond its first is partitioned, a
        ///     shorter one insertion-sorted. A whole line of at most <c>SmallQuicksort + 1</c> elements is therefore one
        ///     insertion sort, which is stable.
        /// </summary>
        internal const int SmallQuicksort = 15;

        /// <summary>NumPy's <c>PYA_QS_STACK</c> (<c>NPY_BITSOF_INTP * 2</c>): the introsort's pending-range stack.</summary>
        private const int QuicksortStack = 128;

        /// <summary>
        ///     The largest run of equal keys the tie pass insertion-sorts by comparing the remaining units; longer runs key
        ///     their next units and radix-sort them.
        /// </summary>
        private const int TieInsertionMax = 16;

        /// <summary>
        ///     The longest line whose stable order is found by counting ranks (each element's slot is the number of keys
        ///     below it plus the equal keys before it): branch-free, where an insertion sort of random keys mispredicts about
        ///     every other compare (107 ns for 10 keys, against ~15 ns counted). It is also every line NumPy's introsort
        ///     orders by insertion alone (at most <c>SmallQuicksort + 1</c> elements), whose order is therefore the stable one.
        /// </summary>
        private const int CountRankMax = SmallQuicksort + 1;

        /// <summary>
        ///     The longest line whose unstable argsort over exact keys runs NumPy's introsort port straight over the key
        ///     records whatever the data (the duplicate sample would read the whole line anyway); a longer line does so only
        ///     when its duplicates are certain (more elements than its keys' window holds values) or a sample of its keys holds
        ///     a repeat. Otherwise the stable radix order and the distinct-strings shortcut are cheaper: over FRESH distinct
        ///     keys the port is a branchy introsort that mispredicts (median 30.7 µs for 512 keys, 68 µs for 1,000) where the
        ///     radix is not (11.9 and 22 µs). A benchmark that sorts one input over and over hides this — the branch
        ///     predictor memorizes a repeated sort of up to ~1,000 elements, NumPy's included (NumPy 2.4.2 argsorts the same
        ///     1,000 random <c>U8</c> strings in 31 µs, and fresh ones in 81 µs at best and 88 µs in the median call).
        /// </summary>
        private const int DirectPortMax = DuplicateSampleSize;

        /// <summary>The keys a long line's duplicate sample reads (evenly spaced).</summary>
        private const int DuplicateSampleSize = 128;

        /// <summary>
        ///     The keys one pass of the short-row route computes: the rows of a block, laid end to end, are keyed this many
        ///     elements at a time by ONE key-kernel call (32 KB of keys, L1-resident while their rows are ranked).
        /// </summary>
        private const int RowKeyChunk = 4096;

        /// <summary>
        ///     Test hook: when true, an unstable argsort runs NumPy's introsort port even when every string of the line is
        ///     distinct (the shortcut gives the same permutation then; this proves it). Process-wide; set it only from
        ///     single-threaded test code and restore it in a <c>finally</c>.
        /// </summary>
        internal static bool ShortcutDisabled;

        /// <summary>
        ///     Test hook: lines an unstable argsort ordered through the introsort port since the process started. A plain
        ///     counter, not synchronized: tests read the difference around calls they make on one thread.
        /// </summary>
        internal static long PortRuns;

        /// <summary>Test hook: lines an unstable argsort answered with the distinct-strings shortcut. See <see cref="PortRuns"/>.</summary>
        internal static long ShortcutRuns;

        /// <summary>
        ///     Test hook: lines an unstable argsort ordered by running the introsort port straight over (key, index) records
        ///     (exact keys, no stable pass first). Counted in <see cref="PortRuns"/> too. See <see cref="PortRuns"/>.
        /// </summary>
        internal static long DirectPortRuns;

        /// <summary>
        ///     Test hook: when true, an unstable argsort over exact keys takes the stable order first (then the shortcut or the
        ///     rank port) instead of the direct key-record port. The permutation is the same. Process-wide; set it only from
        ///     single-threaded test code and restore it in a <c>finally</c>.
        /// </summary>
        internal static bool DirectPortDisabled;

        /// <summary>
        ///     Test hook: lines the direct port ordered over WINDOW-packed records (the bits in which the line's keys differ
        ///     fit 32, so each record is that window over its index in one ulong). Counted in <see cref="DirectPortRuns"/>
        ///     too. See <see cref="PortRuns"/>.
        /// </summary>
        internal static long PackedDirectRuns;

        /// <summary>
        ///     Test hook: when true, the direct port always takes the two-ulong (key, index) records, never the window-packed
        ///     ones. The permutation is the same. Process-wide; set it only from single-threaded test code and restore it in a
        ///     <c>finally</c>.
        /// </summary>
        internal static bool PackedWindowDisabled;

        /// <summary>
        ///     The most repeats a sample of a line's prefix keys may hold for the unstable argsort to run NumPy's introsort
        ///     straight over (prefix key, index) records with the tail compare (<see cref="IntroArgSortTextRecords"/>) when the
        ///     keys do not cover the strings. With so few equal keys almost every compare is a key compare, and the stable
        ///     pass, the tie pass and the ranks the rank route needs before its port are saved; more repeats (a common
        ///     prefix, categorical data) would make most compares tail compares, which the rank route's keyed tie pass does
        ///     faster. The route also needs at least one repeat (see <see cref="TailPortMinRepeats"/>) and a line of at most
        ///     <see cref="TailPortMax"/> elements.
        /// </summary>
        private const int TailPortMaxRepeats = DuplicateSampleSize / 16;

        /// <summary>
        ///     The fewest sampled key repeats for the tail-compare port. Without one the strings are most likely distinct,
        ///     and the stable radix order with the distinct-strings shortcut answers far cheaper than any introsort (fresh
        ///     random 32-byte strings: ~20 µs for 1,000 against ~55 through the port); a repeat of the prefix makes equal
        ///     strings, which force NumPy's tie order, likely enough to skip the stable pass.
        /// </summary>
        private const int TailPortMinRepeats = 1;

        /// <summary>
        ///     The longest line the tail-compare port takes. Measured pinned over fresh random lines of six-unit BMP strings
        ///     (rare duplicates): 4–10 % faster than the rank route from 1,000 to 16,000 elements, 6 % slower at 64,000, where
        ///     the radix's per-element cost has fallen below the extra branches of an introsort on n log n compares.
        /// </summary>
        private const int TailPortMax = 16384;

        /// <summary>
        ///     Test hook: lines the unstable argsort ordered with <see cref="IntroArgSortTextRecords"/> (keys that do not cover
        ///     the strings, few repeats). Counted in <see cref="DirectPortRuns"/> too. See <see cref="PortRuns"/>.
        /// </summary>
        internal static long TailPortRuns;

        /// <summary>
        ///     Test hook: when true, a line whose keys do not cover its strings always takes the stable order first (then the
        ///     shortcut or the rank port), never <see cref="IntroArgSortTextRecords"/>. The permutation is the same.
        ///     Process-wide; set it only from single-threaded test code and restore it in a <c>finally</c>.
        /// </summary>
        internal static bool TailPortDisabled;

        /// <summary>
        ///     Test hook: block-kernel calls the line driver made since the process started (one per run of lines NDIter hands
        ///     out; a C-contiguous array along its last axis is one call). See <see cref="PortRuns"/>.
        /// </summary>
        internal static long BlockCalls;

        /// <summary>
        ///     Test hook: ranges the introsort port handed to its heapsort fallback (NumPy's <c>cdepth &lt; 0</c> branch) since
        ///     the process started. See <see cref="PortRuns"/>.
        /// </summary>
        internal static long HeapFallbackRuns;

        /// <summary>
        ///     Sorts every line of <paramref name="target"/> along <paramref name="axis"/> in place (<c>ndarray.sort</c> over
        ///     text): each line is copied into a contiguous scratch buffer, ordered, and written back sorted.
        /// </summary>
        /// <param name="target">A writeable <c>S</c> or <c>U</c> array.</param>
        /// <param name="axis">The normalised axis.</param>
        /// <remarks>Equal strings are equal bytes at one width, so every correct sort writes NumPy's bytes.</remarks>
        internal static void SortLines(NDArray target, int axis)
        {
            long n = target.Shape.dimensions[axis];
            if (n <= 1 || target.size == 0)
                return;
            int width = target.dtype.itemsize;
            if (width == 0)
                return;
            long stride = target.Shape.strides[axis] * (long)width;
            using var job = new Job(UnitOf(target), width, n, stride, stride, stable: true, trackSame: false, lineCopy: true,
                                    target.size);
            DriveLines(new[] { target }, new[] { NDIterPerOpFlags.READWRITE }, axis, job.SortLinesInPlace);
        }

        /// <summary>
        ///     Writes the sorted lines of <paramref name="src"/> into <paramref name="dst"/> (<c>np.sort</c> over text: the
        ///     result's lines are gathered straight from the input, with no copy of the input first).
        /// </summary>
        /// <param name="src">The <c>S</c> or <c>U</c> array to sort (only read).</param>
        /// <param name="dst">A fresh array of <paramref name="src"/>'s dtype and dimensions (any layout); every element is
        ///     written. Must not overlap <paramref name="src"/>.</param>
        /// <param name="axis">The normalised axis.</param>
        internal static void SortLinesInto(NDArray src, NDArray dst, int axis)
        {
            long n = src.Shape.dimensions[axis];
            if (n == 0 || src.size == 0)
                return;
            int width = src.dtype.itemsize;
            if (width == 0)
                return;
            long srcStride = src.Shape.strides[axis] * (long)width;
            long dstStride = dst.Shape.strides[axis] * (long)width;
            using var job = new Job(UnitOf(src), width, n, srcStride, dstStride, stable: true, trackSame: false, lineCopy: false,
                                    src.size);
            DriveLines(new[] { src, dst }, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }, axis,
                job.SortLinesInto);
        }

        /// <summary>
        ///     Writes the indices that sort every line of <paramref name="src"/> along <paramref name="axis"/> into
        ///     <paramref name="dst"/>: NumPy's stable order for the stable kinds, NumPy's introsort order otherwise.
        /// </summary>
        /// <param name="src">The <c>S</c> or <c>U</c> array (only read; any layout, broadcast included).</param>
        /// <param name="dst">An int64 array of <paramref name="src"/>'s dimensions; every line is overwritten.</param>
        /// <param name="axis">The normalised axis.</param>
        /// <param name="stable">True for the stable kinds (<c>kind='stable'</c> / <c>'mergesort'</c>, <c>stable=True</c>);
        ///     false for the default and heapsort kinds and for <c>argpartition</c> (the generic introsort NumPy falls back
        ///     to compares as the string introsort does).</param>
        internal static void ArgSortLines(NDArray src, NDArray dst, int axis, bool stable)
        {
            long n = src.Shape.dimensions[axis];
            if (n == 0 || src.size == 0)
                return;
            int width = src.dtype.itemsize;
            long textStride = src.Shape.strides[axis] * (long)width;
            long outStride = dst.Shape.strides[axis] * (long)sizeof(long);
            // Only an unstable sort of more than SMALL_QUICKSORT + 1 elements partitions, so only it can order ties
            // otherwise than the stable sort does; it alone needs the duplicate flags.
            bool trackSame = !stable && n > SmallQuicksort + 1;
            using var job = new Job(UnitOf(src), width, n, textStride, outStride, stable, trackSame, lineCopy: false, src.size);
            DriveLines(new[] { src, dst }, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }, axis,
                job.ArgSortLines);
        }

        /// <summary>
        ///     NumPy's comparison of two elements of one width from unit <paramref name="from"/> on: the first differing code
        ///     unit decides, compared unsigned (bytes for <c>S</c>, 32-bit units for <c>U</c>).
        /// </summary>
        /// <param name="a">The first element.</param>
        /// <param name="b">The second element.</param>
        /// <param name="from">The first byte compared (a unit boundary).</param>
        /// <param name="width">The element width in bytes.</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <returns>-1, 0 or 1.</returns>
        internal static int CompareTail(byte* a, byte* b, int from, int width, int unit)
        {
            int length = width - from;
            var x = new ReadOnlySpan<byte>(a + from, length);
            var y = new ReadOnlySpan<byte>(b + from, length);
            // The vectorized common-prefix search finds the first differing BYTE; its unit decides.
            int d = x.CommonPrefixLength(y);
            if (d == length)
                return 0;
            if (unit == 1)
                return x[d] < y[d] ? -1 : 1;
            int u = d & ~3;
            uint ua = System.Runtime.CompilerServices.Unsafe.ReadUnaligned<uint>(a + from + u);
            uint ub = System.Runtime.CompilerServices.Unsafe.ReadUnaligned<uint>(b + from + u);
            return ua < ub ? -1 : 1;
        }

        /// <summary>
        ///     NumPy's <c>string_aquicksort_</c> (<c>npysort/quicksort.cpp</c>), statement for statement, over records of
        ///     (rank, index): two ulongs per record, the rank first. It orders the records as NumPy orders its index array,
        ///     comparing ranks where NumPy compares strings, so with ranks that are the strings' dense ranks the indices it
        ///     leaves in the records are NumPy's argsort.
        /// </summary>
        /// <param name="rec">The records, <paramref name="num"/> of them (ranks in even slots, indices in odd ones), in
        ///     NumPy's initial order (index i at record i).</param>
        /// <param name="num">The number of records.</param>
        /// <remarks>
        ///     The pivot is captured as a rank before it is swapped to <c>pr - 1</c> (NumPy keeps a pointer to the pivot's
        ///     string, whose value does not move); the scans need no bounds checks for the reason NumPy's do not (the median
        ///     of three leaves an element not below the pivot at <c>pr</c> and one not above it at <c>pl</c>). The pending
        ///     ranges stay below <see cref="QuicksortStack"/> because the larger part is always pushed.
        /// </remarks>
        internal static void IntroArgSortRecords(ulong* rec, long num)
        {
            if (num <= 1)
                return;
            long* stack = stackalloc long[QuicksortStack];
            int* depth = stackalloc int[QuicksortStack];
            int sptr = 0, psdepth = 0;
            long pl = 0, pr = num - 1;
            int cdepth = BitOperations.Log2((ulong)num) * 2;   // npy_get_msb(num) * 2

            for (;;)
            {
                if (cdepth < 0)
                {
                    HeapArgSortRecords(rec + 2 * pl, pr - pl + 1);
                    HeapFallbackRuns++;
                    goto stack_pop;
                }
                while (pr - pl > SmallQuicksort)
                {
                    // Quicksort partition, median of three.
                    long pm = pl + ((pr - pl) >> 1);
                    if (rec[2 * pm] < rec[2 * pl])
                        Swap(rec, pm, pl);
                    if (rec[2 * pr] < rec[2 * pm])
                        Swap(rec, pr, pm);
                    if (rec[2 * pm] < rec[2 * pl])
                        Swap(rec, pm, pl);
                    ulong vp = rec[2 * pm];
                    long pi = pl, pj = pr - 1;
                    Swap(rec, pm, pj);
                    for (;;)
                    {
                        do
                            ++pi;
                        while (rec[2 * pi] < vp);
                        do
                            --pj;
                        while (vp < rec[2 * pj]);
                        if (pi >= pj)
                            break;
                        Swap(rec, pi, pj);
                    }
                    long pk = pr - 1;
                    Swap(rec, pi, pk);
                    // Push the larger part, continue with the smaller one.
                    if (pi - pl < pr - pi)
                    {
                        stack[sptr++] = pi + 1;
                        stack[sptr++] = pr;
                        pr = pi - 1;
                    }
                    else
                    {
                        stack[sptr++] = pl;
                        stack[sptr++] = pi - 1;
                        pl = pi + 1;
                    }
                    depth[psdepth++] = --cdepth;
                }

                // Insertion sort (strict less: equal ranks keep their order).
                for (long pi = pl + 1; pi <= pr; ++pi)
                {
                    ulong vr = rec[2 * pi], vi = rec[2 * pi + 1];
                    long pj = pi, pk = pi - 1;
                    while (pj > pl && vr < rec[2 * pk])
                    {
                        rec[2 * pj] = rec[2 * pk];
                        rec[2 * pj + 1] = rec[2 * pk + 1];
                        pj--;
                        pk--;
                    }
                    rec[2 * pj] = vr;
                    rec[2 * pj + 1] = vi;
                }
            stack_pop:
                if (sptr == 0)
                    break;
                pr = stack[--sptr];
                pl = stack[--sptr];
                cdepth = depth[--psdepth];
            }
        }

        /// <summary>
        ///     NumPy's <c>string_aheapsort_</c> (<c>npysort/npysort_heapsort.h</c>) over (rank, index) records: the introsort's
        ///     fallback once its depth budget is spent, heap-sorting one range.
        /// </summary>
        /// <param name="rec">The range's first record.</param>
        /// <param name="n">The range's length.</param>
        /// <remarks>NumPy indexes the heap from 1 (<c>a = tosort - 1</c>); so does this port, through a base one record
        ///     before the range (never dereferenced at index 0).</remarks>
        internal static void HeapArgSortRecords(ulong* rec, long n)
        {
            ulong* a = rec - 2;
            for (long l = n >> 1; l > 0; --l)
            {
                ulong tr = a[2 * l], ti = a[2 * l + 1];
                long i = l, j = l << 1;
                while (j <= n)
                {
                    if (j < n && a[2 * j] < a[2 * (j + 1)])
                        j += 1;
                    if (tr < a[2 * j])
                    {
                        a[2 * i] = a[2 * j];
                        a[2 * i + 1] = a[2 * j + 1];
                        i = j;
                        j += j;
                    }
                    else
                        break;
                }
                a[2 * i] = tr;
                a[2 * i + 1] = ti;
            }

            for (; n > 1;)
            {
                ulong tr = a[2 * n], ti = a[2 * n + 1];
                a[2 * n] = a[2];
                a[2 * n + 1] = a[3];
                n -= 1;
                long i = 1, j = 2;
                while (j <= n)
                {
                    if (j < n && a[2 * j] < a[2 * (j + 1)])
                        j++;
                    if (tr < a[2 * j])
                    {
                        a[2 * i] = a[2 * j];
                        a[2 * i + 1] = a[2 * j + 1];
                        i = j;
                        j += j;
                    }
                    else
                        break;
                }
                a[2 * i] = tr;
                a[2 * i + 1] = ti;
            }
        }

        /// <summary>
        ///     <see cref="IntroArgSortRecords"/> over (prefix key, index) records whose keys do NOT cover the strings: NumPy's
        ///     <c>string_aquicksort_</c> statement for statement, comparing two records as NumPy compares their strings —
        ///     by key, and on equal keys by the units after the keyed ones (<see cref="CompareTail"/>). With keys that rarely
        ///     tie this is NumPy's sort at the cost of a key compare, with no stable pass and no ranks first.
        /// </summary>
        /// <param name="rec">The records (keys in even slots, element indices in odd ones), in NumPy's initial order.</param>
        /// <param name="num">The number of records.</param>
        /// <param name="text">The line's element 0.</param>
        /// <param name="stride">The byte distance between its elements.</param>
        /// <param name="from">The first byte the keys do not cover (a unit boundary).</param>
        /// <param name="width">The element width in bytes.</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <remarks>
        ///     The pivot is kept as a record copy (its key and the element it names), as NumPy keeps a pointer to the pivot's
        ///     string; the scans need no bounds checks for the reason NumPy's do not (the median of three).
        /// </remarks>
        internal static void IntroArgSortTextRecords(ulong* rec, long num, byte* text, long stride, int from, int width, int unit)
        {
            if (num <= 1)
                return;
            long* stack = stackalloc long[QuicksortStack];
            int* depth = stackalloc int[QuicksortStack];
            // The pivot and the inserted record as records of their own (allocated once: a stackalloc inside the loops
            // would grow the frame on every iteration).
            ulong* vp = stackalloc ulong[2];
            ulong* v = stackalloc ulong[2];
            int sptr = 0, psdepth = 0;
            long pl = 0, pr = num - 1;
            int cdepth = BitOperations.Log2((ulong)num) * 2;   // npy_get_msb(num) * 2

            for (;;)
            {
                if (cdepth < 0)
                {
                    HeapArgSortTextRecords(rec + 2 * pl, pr - pl + 1, text, stride, from, width, unit);
                    HeapFallbackRuns++;
                    goto stack_pop;
                }
                while (pr - pl > SmallQuicksort)
                {
                    long pm = pl + ((pr - pl) >> 1);
                    if (TextLess(rec + 2 * pm, rec + 2 * pl, text, stride, from, width, unit))
                        Swap(rec, pm, pl);
                    if (TextLess(rec + 2 * pr, rec + 2 * pm, text, stride, from, width, unit))
                        Swap(rec, pr, pm);
                    if (TextLess(rec + 2 * pm, rec + 2 * pl, text, stride, from, width, unit))
                        Swap(rec, pm, pl);
                    // The pivot as a record of its own: its slot moves below, its string does not.
                    vp[0] = rec[2 * pm];
                    vp[1] = rec[2 * pm + 1];
                    long pi = pl, pj = pr - 1;
                    Swap(rec, pm, pj);
                    for (;;)
                    {
                        do
                            ++pi;
                        while (TextLess(rec + 2 * pi, vp, text, stride, from, width, unit));
                        do
                            --pj;
                        while (TextLess(vp, rec + 2 * pj, text, stride, from, width, unit));
                        if (pi >= pj)
                            break;
                        Swap(rec, pi, pj);
                    }
                    long pk = pr - 1;
                    Swap(rec, pi, pk);
                    if (pi - pl < pr - pi)
                    {
                        stack[sptr++] = pi + 1;
                        stack[sptr++] = pr;
                        pr = pi - 1;
                    }
                    else
                    {
                        stack[sptr++] = pl;
                        stack[sptr++] = pi - 1;
                        pl = pi + 1;
                    }
                    depth[psdepth++] = --cdepth;
                }

                // Insertion sort (strict less: equal strings keep their order).
                for (long pi = pl + 1; pi <= pr; ++pi)
                {
                    ulong vk = rec[2 * pi], vi = rec[2 * pi + 1];
                    v[0] = vk;
                    v[1] = vi;
                    long pj = pi, pk = pi - 1;
                    while (pj > pl && TextLess(v, rec + 2 * pk, text, stride, from, width, unit))
                    {
                        rec[2 * pj] = rec[2 * pk];
                        rec[2 * pj + 1] = rec[2 * pk + 1];
                        pj--;
                        pk--;
                    }
                    rec[2 * pj] = vk;
                    rec[2 * pj + 1] = vi;
                }
            stack_pop:
                if (sptr == 0)
                    break;
                pr = stack[--sptr];
                pl = stack[--sptr];
                cdepth = depth[--psdepth];
            }
        }

        /// <summary>NumPy's <c>string_aheapsort_</c> over (prefix key, index) records with the tail compare of
        ///     <see cref="IntroArgSortTextRecords"/>: the introsort's fallback once its depth budget is spent.</summary>
        /// <param name="rec">The range's first record.</param>
        /// <param name="n">The range's length.</param>
        /// <param name="text">The line's element 0.</param>
        /// <param name="stride">The byte distance between its elements.</param>
        /// <param name="from">The first byte the keys do not cover.</param>
        /// <param name="width">The element width in bytes.</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        internal static void HeapArgSortTextRecords(ulong* rec, long n, byte* text, long stride, int from, int width, int unit)
        {
            ulong* a = rec - 2;
            ulong* tmp = stackalloc ulong[2];
            for (long l = n >> 1; l > 0; --l)
            {
                tmp[0] = a[2 * l];
                tmp[1] = a[2 * l + 1];
                long i = l, j = l << 1;
                while (j <= n)
                {
                    if (j < n && TextLess(a + 2 * j, a + 2 * (j + 1), text, stride, from, width, unit))
                        j += 1;
                    if (TextLess(tmp, a + 2 * j, text, stride, from, width, unit))
                    {
                        a[2 * i] = a[2 * j];
                        a[2 * i + 1] = a[2 * j + 1];
                        i = j;
                        j += j;
                    }
                    else
                        break;
                }
                a[2 * i] = tmp[0];
                a[2 * i + 1] = tmp[1];
            }

            for (; n > 1;)
            {
                tmp[0] = a[2 * n];
                tmp[1] = a[2 * n + 1];
                a[2 * n] = a[2];
                a[2 * n + 1] = a[3];
                n -= 1;
                long i = 1, j = 2;
                while (j <= n)
                {
                    if (j < n && TextLess(a + 2 * j, a + 2 * (j + 1), text, stride, from, width, unit))
                        j++;
                    if (TextLess(tmp, a + 2 * j, text, stride, from, width, unit))
                    {
                        a[2 * i] = a[2 * j];
                        a[2 * i + 1] = a[2 * j + 1];
                        i = j;
                        j += j;
                    }
                    else
                        break;
                }
                a[2 * i] = tmp[0];
                a[2 * i + 1] = tmp[1];
            }
        }

        /// <summary>
        ///     NumPy's string compare of two (prefix key, index) records: the keys decide when they differ (key order is the
        ///     order of the keyed units); equal keys compare the elements' units after them.
        /// </summary>
        /// <param name="a">One record (key, element index).</param>
        /// <param name="b">The other.</param>
        /// <param name="text">The line's element 0.</param>
        /// <param name="stride">The byte distance between its elements.</param>
        /// <param name="from">The first byte the keys do not cover.</param>
        /// <param name="width">The element width in bytes.</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <returns>True when a's string is strictly less than b's.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool TextLess(ulong* a, ulong* b, byte* text, long stride, int from, int width, int unit)
        {
            if (a[0] != b[0])
                return a[0] < b[0];
            return CompareTail(text + (long)a[1] * stride, text + (long)b[1] * stride, from, width, unit) < 0;
        }

        /// <summary>The low half of a packed record: its index.</summary>
        private const ulong PackedIndexMask = 0xFFFFFFFFUL;

        /// <summary>The high half of a packed record: its rank.</summary>
        private const ulong PackedRankMask = ~PackedIndexMask;

        /// <summary>
        ///     <see cref="IntroArgSortRecords"/> over PACKED records: one ulong each, the rank in the high 32 bits and the
        ///     index in the low 32 (lines of at most 2^32 elements, where both fit). Half the bytes per swap, and each of
        ///     NumPy's comparisons is ONE unsigned compare against a precomputed bound:
        ///     <c>rank(x) &lt; rank(y)</c> ⟺ <c>x &lt; (y &amp; PackedRankMask)</c> ⟺ <c>(x | PackedIndexMask) &lt; y</c>.
        /// </summary>
        /// <param name="rec">The records, <paramref name="num"/> of them, in NumPy's initial order (index i at record i).</param>
        /// <param name="num">The number of records (at most 2^32).</param>
        /// <remarks>
        ///     The two compare forms follow from the packing: when the ranks differ, the rank halves decide whatever the
        ///     index halves hold; when they are equal, <c>x &lt; (y &amp; mask)</c> compares x against y with a zero index
        ///     (false: x's index is at least zero) and <c>(x | mask) &lt; y</c> x with the largest index against y (false).
        ///     So equal ranks are never "less", as equal strings are not in NumPy.
        /// </remarks>
        internal static void IntroArgSortPacked(ulong* rec, long num)
        {
            if (num <= 1)
                return;
            long* stack = stackalloc long[QuicksortStack];
            int* depth = stackalloc int[QuicksortStack];
            int sptr = 0, psdepth = 0;
            long pl = 0, pr = num - 1;
            int cdepth = BitOperations.Log2((ulong)num) * 2;   // npy_get_msb(num) * 2

            for (;;)
            {
                if (cdepth < 0)
                {
                    HeapArgSortPacked(rec + pl, pr - pl + 1);
                    HeapFallbackRuns++;
                    goto stack_pop;
                }
                while (pr - pl > SmallQuicksort)
                {
                    long pm = pl + ((pr - pl) >> 1);
                    if (rec[pm] < (rec[pl] & PackedRankMask))
                        (rec[pm], rec[pl]) = (rec[pl], rec[pm]);
                    if (rec[pr] < (rec[pm] & PackedRankMask))
                        (rec[pr], rec[pm]) = (rec[pm], rec[pr]);
                    if (rec[pm] < (rec[pl] & PackedRankMask))
                        (rec[pm], rec[pl]) = (rec[pl], rec[pm]);
                    // less(x, pivot) ⟺ x < pivotLow; less(pivot, x) ⟺ pivotHigh < x.
                    ulong pivotLow = rec[pm] & PackedRankMask, pivotHigh = rec[pm] | PackedIndexMask;
                    long pi = pl, pj = pr - 1;
                    (rec[pm], rec[pj]) = (rec[pj], rec[pm]);
                    for (;;)
                    {
                        do
                            ++pi;
                        while (rec[pi] < pivotLow);
                        do
                            --pj;
                        while (pivotHigh < rec[pj]);
                        if (pi >= pj)
                            break;
                        (rec[pi], rec[pj]) = (rec[pj], rec[pi]);
                    }
                    long pk = pr - 1;
                    (rec[pi], rec[pk]) = (rec[pk], rec[pi]);
                    if (pi - pl < pr - pi)
                    {
                        stack[sptr++] = pi + 1;
                        stack[sptr++] = pr;
                        pr = pi - 1;
                    }
                    else
                    {
                        stack[sptr++] = pl;
                        stack[sptr++] = pi - 1;
                        pl = pi + 1;
                    }
                    depth[psdepth++] = --cdepth;
                }

                for (long pi = pl + 1; pi <= pr; ++pi)
                {
                    ulong v = rec[pi], vHigh = v | PackedIndexMask;
                    long pj = pi, pk = pi - 1;
                    while (pj > pl && vHigh < rec[pk])
                    {
                        rec[pj] = rec[pk];
                        pj--;
                        pk--;
                    }
                    rec[pj] = v;
                }
            stack_pop:
                if (sptr == 0)
                    break;
                pr = stack[--sptr];
                pl = stack[--sptr];
                cdepth = depth[--psdepth];
            }
        }

        /// <summary><see cref="HeapArgSortRecords"/> over packed records (see <see cref="IntroArgSortPacked"/> for the compares).</summary>
        /// <param name="rec">The range's first record.</param>
        /// <param name="n">The range's length.</param>
        internal static void HeapArgSortPacked(ulong* rec, long n)
        {
            ulong* a = rec - 1;
            for (long l = n >> 1; l > 0; --l)
            {
                ulong tmp = a[l], tmpHigh = tmp | PackedIndexMask;
                long i = l, j = l << 1;
                while (j <= n)
                {
                    if (j < n && a[j] < (a[j + 1] & PackedRankMask))
                        j += 1;
                    if (tmpHigh < a[j])
                    {
                        a[i] = a[j];
                        i = j;
                        j += j;
                    }
                    else
                        break;
                }
                a[i] = tmp;
            }

            for (; n > 1;)
            {
                ulong tmp = a[n], tmpHigh = tmp | PackedIndexMask;
                a[n] = a[1];
                n -= 1;
                long i = 1, j = 2;
                while (j <= n)
                {
                    if (j < n && a[j] < (a[j + 1] & PackedRankMask))
                        j++;
                    if (tmpHigh < a[j])
                    {
                        a[i] = a[j];
                        i = j;
                        j += j;
                    }
                    else
                        break;
                }
                a[i] = tmp;
            }
        }

        /// <summary>Swaps records <paramref name="i"/> and <paramref name="j"/> (NumPy's <c>std::swap(*pi, *pj)</c> of two indices).</summary>
        /// <param name="rec">The records.</param>
        /// <param name="i">One record.</param>
        /// <param name="j">The other.</param>
        private static void Swap(ulong* rec, long i, long j)
        {
            ulong r = rec[2 * i], x = rec[2 * i + 1];
            rec[2 * i] = rec[2 * j];
            rec[2 * i + 1] = rec[2 * j + 1];
            rec[2 * j] = r;
            rec[2 * j + 1] = x;
        }

        /// <summary>
        ///     The stable order of at most <see cref="CountRankMax"/> keys by counting: element i lands in slot
        ///     (keys below it) + (equal keys before it), which is its position in the stable order, so no two elements share
        ///     a slot. Every count is a compare turned into 0 / 1, with no branch on the data (an insertion sort of random
        ///     keys mispredicts about every other compare).
        /// </summary>
        /// <param name="keys">The keys, element i at slot i, followed by readable slots up to slot 15 (the vector form
        ///     loads 16 keys).</param>
        /// <param name="n">The number of keys (at most <see cref="CountRankMax"/>).</param>
        /// <param name="sortedKeys">Receives the keys in stable order.</param>
        /// <param name="sortedIdx">Receives the element of each sorted slot.</param>
        /// <remarks>
        ///     With 256-bit vectors four elements are ranked at once against every key: the keys get their sign bit
        ///     flipped, which turns unsigned order into the signed order the one-instruction 64-bit compare tests. The
        ///     portable form ranks one element at a time over a fixed trip count (no loop exit to mispredict).
        /// </remarks>
        [SkipLocalsInit]
        internal static void CountRankOrder(ulong* keys, long n, ulong* sortedKeys, long* sortedIdx)
        {
            int m = (int)n;
            if (Vector256.IsHardwareAccelerated)
            {
                // The keys with their sign bit flipped (a signed compare then orders as the unsigned key does), one slot
                // each for the broadcasts below.
                const ulong Sign = 0x8000000000000000UL;
                long* flipped = stackalloc long[CountRankMax];
                for (int j = 0; j < m; j++)
                    flipped[j] = (long)(keys[j] ^ Sign);
                var sign = Vector256.Create(unchecked((long)Sign));
                var lane = Vector256.Create(0L, 1, 2, 3);
                long* ranks = stackalloc long[CountRankMax];
                // Four elements (one group g, lanes 4g..4g+3) are ranked at once. A true compare lane is all ones (-1), so
                // adding it subtracts one and subtracting it adds one. Element j counts for lane i when its key is smaller,
                // or equal with j before i — which splits by where j lies:
                //   j before the group: every lane comes after j, so j counts unless its key is GREATER (1 + GT(kj, ki));
                //   j inside the group: the general rule, the "after j" lanes from a lane-index compare;
                //   j after the group:  only a SMALLER key counts (GT(ki, kj)).
                // One 64-bit compare per (group, j) outside the group, where the general rule needs three (the index
                // compare and the key compare both run on one port on current Intel cores): 37 -> ~10 ns for 10 keys.
                // The lanes past m (the last group's tail) hold whatever follows the keys (the key block is padded to 16
                // readable slots): ranked but never scattered.
                for (int lo = 0; lo < m; lo += 4)
                {
                    var ki = Vector256.Load((long*)keys + lo) ^ sign;
                    int hi = Math.Min(lo + 4, m);
                    var c = Vector256.Create((long)lo);
                    for (int j = 0; j < lo; j++)
                        c += Vector256.GreaterThan(Vector256.Create(flipped[j]), ki);
                    for (int j = lo; j < hi; j++)
                    {
                        var kj = Vector256.Create(flipped[j]);
                        var after = Vector256.GreaterThan(lane, Vector256.Create((long)(j - lo)));
                        c -= Vector256.GreaterThan(ki, kj) | (Vector256.Equals(ki, kj) & after);
                    }
                    for (int j = hi; j < m; j++)
                        c -= Vector256.GreaterThan(ki, Vector256.Create(flipped[j]));
                    c.Store(ranks + lo);
                }
                for (int i = 0; i < m; i++)
                {
                    long r = ranks[i];
                    sortedKeys[r] = keys[i];
                    sortedIdx[r] = i;
                }
                return;
            }

            for (int i = 0; i < m; i++)
            {
                ulong ki = keys[i];
                int rank = 0;
                for (int j = 0; j < m; j++)
                    rank += Unsafe.BitCast<bool, byte>(keys[j] < ki) + Unsafe.BitCast<bool, byte>((keys[j] == ki) & (j < i));
                sortedKeys[rank] = ki;
                sortedIdx[rank] = i;
            }
        }

        /// <summary>
        ///     The bits in which some two of <paramref name="n"/> keys differ: the OR of the keys XOR their AND (four keys a
        ///     step with 256-bit vectors). A bit outside the result is the same in every key, so the keys order exactly as
        ///     their varying bits do: the radix sorts only the bytes those bits touch, and when they span at most 32 bits the
        ///     direct port packs that window over each element's index into one ulong.
        /// </summary>
        /// <param name="keys">The keys.</param>
        /// <param name="n">Their number (at least 1).</param>
        /// <returns>The varying bits (0 when every key is the same).</returns>
        private static ulong VaryingBits(ulong* keys, long n)
        {
            ulong or = 0, and = ulong.MaxValue;
            long i = 0;
            if (Vector256.IsHardwareAccelerated && n >= 8)
            {
                var vor = Vector256<ulong>.Zero;
                var vand = Vector256<ulong>.AllBitsSet;
                for (; i + 4 <= n; i += 4)
                {
                    var v = Vector256.Load(keys + i);
                    vor |= v;
                    vand &= v;
                }
                or = vor.GetElement(0) | vor.GetElement(1) | vor.GetElement(2) | vor.GetElement(3);
                and = vand.GetElement(0) & vand.GetElement(1) & vand.GetElement(2) & vand.GetElement(3);
            }
            for (; i < n; i++)
            {
                or |= keys[i];
                and &= keys[i];
            }
            return or ^ and;
        }

        /// <summary>
        ///     Whether two adjacent keys of a sorted key array are equal (with exact keys: whether the line holds equal
        ///     strings), comparing four neighbours a step with 256-bit vectors.
        /// </summary>
        /// <param name="k">The sorted keys.</param>
        /// <param name="n">Their number.</param>
        /// <returns>True at the first equal pair.</returns>
        private static bool HasAdjacentEqual(ulong* k, long n)
        {
            long i = 1;
            if (Vector256.IsHardwareAccelerated)
            {
                // k[i..i+3] against k[i-1..i+2]: both loads stay inside [0, n).
                for (; i + 4 <= n; i += 4)
                {
                    if (Vector256.EqualsAny(Vector256.Load(k + i), Vector256.Load(k + i - 1)))
                        return true;
                }
            }
            for (; i < n; i++)
            {
                if (k[i] == k[i - 1])
                    return true;
            }
            return false;
        }

        /// <summary>The duplicate flags past the first <see cref="int.MaxValue"/> positions (a line longer than a span).</summary>
        /// <param name="same">The flags.</param>
        /// <param name="n">The line length.</param>
        /// <returns>True when a flag at position <c>int.MaxValue + 1</c> or later is set.</returns>
        private static bool AnyFlagPast(byte* same, long n)
        {
            for (long i = (long)int.MaxValue + 1; i < n; i++)
            {
                if (same[i] != 0)
                    return true;
            }
            return false;
        }

        /// <summary>The code-unit size of a text array: 4 for <c>U</c>, 1 for <c>S</c>.</summary>
        /// <param name="a">An <c>S</c> or <c>U</c> array.</param>
        /// <returns>4 or 1.</returns>
        private static int UnitOf(NDArray a) => a.dtype.kind == 'U' ? 4 : 1;

        /// <summary>
        ///     Hands every line of <paramref name="ops"/> along <paramref name="axis"/> to <paramref name="kernel"/> in blocks:
        ///     NDIter over every axis BUT the line axis (dropped through <c>op_axes</c>, NumPy's <c>IterAllButAxis</c>) with
        ///     <c>EXTERNAL_LOOP</c>, so each call carries a run of lines — <c>dataptrs[k]</c> the first line's element 0 in
        ///     operand k, <c>strides[k]</c> the byte distance to the next line, <c>count</c> the lines in the run.
        /// </summary>
        /// <param name="ops">The operands whose lines are visited together (all with <c>ops[0]</c>'s dimensions; text
        ///     operands are sized by their descriptors).</param>
        /// <param name="flags">Per-operand access flags.</param>
        /// <param name="axis">The normalised line axis.</param>
        /// <param name="kernel">The block kernel.</param>
        /// <remarks>
        ///     A zero-size <c>ops[0]</c> has no lines and returns at once. With at most two dimensions there is at most one
        ///     kept axis, so every operand's lines are evenly spaced — exactly the single run the iterator would hand out — and
        ///     the kernel is called directly with that run (the trivial-loop bypass the ufuncs take; building the iterator
        ///     costs about a microsecond, most of a 1,000-element call): one line for 1-D, the kept axis's extent and strides
        ///     for 2-D, whatever the layout (stepped, reversed, transposed, broadcast).
        /// </remarks>
        [NDScoped]
        internal static void DriveLines(NDArray[] ops, NDIterPerOpFlags[] flags, int axis, NDInnerLoopFunc kernel)
        {
            if (ops[0].size == 0)
                return;
            int ndim = ops[0].ndim;
            if (ndim <= 2)
            {
                int keptAxis = ndim == 2 ? 1 - axis : -1;
                long count = keptAxis >= 0 ? ops[0].Shape.dimensions[keptAxis] : 1;
                void** pointers = stackalloc void*[ops.Length];
                long* steps = stackalloc long[ops.Length];
                for (int k = 0; k < ops.Length; k++)
                {
                    long elementBytes = NDIterRef.ElementBytes(ops[k]);
                    pointers[k] = (byte*)ops[k].Address + ops[k].Shape.offset * elementBytes;
                    steps[k] = keptAxis >= 0 ? ops[k].Shape.strides[keptAxis] * elementBytes : 0;
                }
                kernel(pointers, steps, count, null);
                return;
            }
            var kept = new int[ndim - 1];
            for (int d = 0, w = 0; d < ndim; d++)
                if (d != axis)
                    kept[w++] = d;
            var opAxes = new int[ops.Length][];
            for (int i = 0; i < ops.Length; i++)
                opAxes[i] = kept;
            // Null descriptors keep each operand's own: a text operand's width is its descriptor's, and nothing is cast.
            var descrs = new DType[ops.Length];
            using var iter = NDIterRef.AdvancedNew(ops.Length, ops, NDIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_CORDER,
                NPY_CASTING.NPY_NO_CASTING, flags, null, ndim - 1, opAxes, opDescrs: descrs);
            iter.ForEach(kernel);
        }

        /// <summary>
        ///     One sort call's state: its constants, its kernels and ONE line's scratch, reused for every line the driver
        ///     hands it. Disposing frees the scratch.
        /// </summary>
        private sealed class Job : IDisposable
        {
            /// <summary>The code-unit size (1 or 4).</summary>
            private readonly int _unit;

            /// <summary>The element width in bytes.</summary>
            private readonly int _width;

            /// <summary>The element width in code units.</summary>
            private readonly int _units;

            /// <summary>The line length.</summary>
            private readonly long _n;

            /// <summary>The byte distance between consecutive elements of a source line.</summary>
            private readonly long _textStride;

            /// <summary>The byte distance between consecutive slots of an output line (text for a sort, int64 for an argsort).</summary>
            private readonly long _outStride;

            /// <summary>Whether this is the unstable argsort of a long line, which must know where the stable order holds equal strings.</summary>
            private readonly bool _trackSame;

            /// <summary>
            ///     Whether the current line keeps the duplicate flags: an unstable argsort whose keys do not cover the strings
            ///     (with exact keys two strings are equal exactly when their keys are, which the sorted keys show).
            /// </summary>
            private bool _sameLive;

            /// <summary>
            ///     The bits in which some two of the current line's first keys differ (<see cref="VaryingBits"/>), set with the
            ///     keys: the radix of <see cref="OrderKeyed"/> sorts only the bytes they touch.
            /// </summary>
            private ulong _varying;

            /// <summary>The gather kernel of the width, or null on a host without dynamic code (the managed twin runs).</summary>
            private readonly TextGatherFunc _gather;

            /// <summary>The prefix keys of this job's dtype (the kernels it resolves are cached in it for every line).</summary>
            private readonly TextKeyer _keyer;

            /// <summary>2n ulongs: the keys and the radix's key buffer; after a line's stable order, the introsort records.</summary>
            private ulong* _keyBlock;

            /// <summary>2n longs: the indices and the radix's index buffer.</summary>
            private long* _idxBlock;

            /// <summary>
            ///     The radix histograms (8 x 256 counts), allocated on the first radix (<see cref="Hist"/>): short rows and
            ///     lines the direct port orders never run one, and 16 KB a call was a fifth of a small argsort's fixed cost.
            /// </summary>
            private long* _hist;

            /// <summary>n flags: sorted position i holds the same string as position i - 1 (unstable argsort only).</summary>
            private byte* _same;

            /// <summary>n elements: a contiguous copy of the line (in-place value sort only).</summary>
            private byte* _lineCopy;

            /// <summary>
            ///     The short-row route's keys: <see cref="_rowKeyCapacity"/> slots plus <see cref="CountRankMax"/> readable
            ///     padding (the count-rank's vector loads read 16 keys from a row's first); allocated for argsorts of rows of at
            ///     most <see cref="CountRankMax"/> elements.
            /// </summary>
            private ulong* _rowKeys;

            /// <summary>The key slots of <see cref="_rowKeys"/>: <see cref="RowKeyChunk"/>, or the whole array's elements when
            ///     it has fewer (a small array keys all its rows in one chunk without allocating 32 KB).</summary>
            private readonly long _rowKeyCapacity;

            /// <summary>After a line's stable pass: the buffer holding the keys of the sorted order (inside a run of equal
            ///     keys, the keys of the last level the tie pass reached).</summary>
            private ulong* _k;

            /// <summary>The other key buffer (the radix's scratch for a run).</summary>
            private ulong* _kt;

            /// <summary>After a line's stable pass: the buffer holding its stable order (element indices).</summary>
            private long* _x;

            /// <summary>The other index buffer (the radix's scratch for a run).</summary>
            private long* _xt;

            /// <summary>Allocates one line's scratch and resolves the gather kernel.</summary>
            /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
            /// <param name="width">The element width in bytes.</param>
            /// <param name="n">The line length (at least 1).</param>
            /// <param name="textStride">The source line's byte stride.</param>
            /// <param name="outStride">The output line's byte stride.</param>
            /// <param name="stable">Whether the argsort asks for the stable order (unused by the value sorts).</param>
            /// <param name="trackSame">Whether to keep the duplicate flags (the unstable argsort of a long line).</param>
            /// <param name="lineCopy">Whether to allocate the line copy (the in-place value sort).</param>
            /// <param name="totalElements">The elements of the whole array (sizes the short-row route's key chunk).</param>
            /// <exception cref="OutOfMemoryException">The scratch cannot be allocated (the partial allocation is freed).</exception>
            internal Job(int unit, int width, long n, long textStride, long outStride, bool stable, bool trackSame, bool lineCopy,
                         long totalElements)
            {
                _unit = unit;
                _width = width;
                _units = width / unit;
                _keyer = new TextKeyer(unit, _units);
                _n = n;
                _textStride = textStride;
                _outStride = outStride;
                _trackSame = trackSame && !stable;
                _gather = ILKernelGenerator.GetTextGatherKernel(width);
                try
                {
                    // CountRankMax extra slots: the count-rank's vector loads read 16 keys whatever the line's length.
                    _keyBlock = (ulong*)SizeBucketedBufferPool.Take(KeyBlockBytes);
                    _idxBlock = (long*)SizeBucketedBufferPool.Take(IdxBlockBytes);
                    if (_trackSame)
                        _same = (byte*)SizeBucketedBufferPool.Take(n);
                    if (lineCopy && width > 0)
                        _lineCopy = (byte*)SizeBucketedBufferPool.Take(n * width);
                    if (!lineCopy && n <= CountRankMax && width > 0)
                    {
                        _rowKeyCapacity = Math.Max(n, Math.Min(RowKeyChunk, totalElements));
                        _rowKeys = (ulong*)SizeBucketedBufferPool.Take(RowKeysBytes);
                    }
                }
                catch
                {
                    Dispose();
                    throw;
                }
            }

            /// <summary>The bytes of <see cref="_keyBlock"/>: 2n keys plus the count-rank's <see cref="CountRankMax"/> padding.</summary>
            private long KeyBlockBytes => (_n * 2 + CountRankMax) * sizeof(ulong);

            /// <summary>The bytes of <see cref="_idxBlock"/>: 2n indices.</summary>
            private long IdxBlockBytes => _n * 2 * sizeof(long);

            /// <summary>The bytes of <see cref="_rowKeys"/>: its capacity plus the count-rank's padding.</summary>
            private long RowKeysBytes => (_rowKeyCapacity + CountRankMax) * sizeof(ulong);

            /// <summary>The bytes of <see cref="_hist"/>: 8 histograms of 256 counts.</summary>
            private const long HistBytes = 8 * 256 * sizeof(long);

            /// <summary>
            ///     Returns the scratch to the buffer pool (idempotent). Every block came from
            ///     <see cref="SizeBucketedBufferPool.Take"/> and goes back with the size it was taken with (the pool reads an
            ///     alignment header in front of a buffer of a page or more, chosen by that size); repeated sorts reuse warm
            ///     blocks, and the guard-page diagnostic mode fences them like every other pooled buffer.
            /// </summary>
            public void Dispose()
            {
                if (_keyBlock != null) { SizeBucketedBufferPool.Return((IntPtr)_keyBlock, KeyBlockBytes); _keyBlock = null; }
                if (_idxBlock != null) { SizeBucketedBufferPool.Return((IntPtr)_idxBlock, IdxBlockBytes); _idxBlock = null; }
                if (_hist != null) { SizeBucketedBufferPool.Return((IntPtr)_hist, HistBytes); _hist = null; }
                if (_same != null) { SizeBucketedBufferPool.Return((IntPtr)_same, _n); _same = null; }
                if (_lineCopy != null) { SizeBucketedBufferPool.Return((IntPtr)_lineCopy, _n * _width); _lineCopy = null; }
                if (_rowKeys != null) { SizeBucketedBufferPool.Return((IntPtr)_rowKeys, RowKeysBytes); _rowKeys = null; }
            }

            /// <summary>The radix histograms, taken from the pool on first use (see <see cref="_hist"/>); returned by
            ///     <see cref="Dispose"/>.</summary>
            /// <exception cref="OutOfMemoryException">The 16 KB cannot be allocated.</exception>
            private long* Hist => _hist != null ? _hist : (_hist = (long*)SizeBucketedBufferPool.Take(HistBytes));

            /// <summary>
            ///     The block kernel of an argsort (an <see cref="NDInnerLoopFunc"/> run by <see cref="DriveLines"/>): operand 0
            ///     holds the text lines, operand 1 the int64 index lines.
            /// </summary>
            /// <param name="dataptrs">The first line's start in each operand.</param>
            /// <param name="strides">The byte distance between consecutive lines in each operand.</param>
            /// <param name="count">The number of lines in the block.</param>
            /// <param name="aux">Unused.</param>
            internal void ArgSortLines(void** dataptrs, long* strides, long count, void* aux)
            {
                BlockCalls++;
                byte* text = (byte*)dataptrs[0], output = (byte*)dataptrs[1];
                long textStep = strides[0], outputStep = strides[1];
                // Short rows laid end to end (a C-contiguous array along its last axis): the whole block is one run of
                // elements, keyed chunk by chunk with one kernel call each.
                if (_rowKeys != null && count > 1 && textStep == _n * _textStride)
                {
                    ArgSortShortRows(text, output, outputStep, count);
                    return;
                }
                // Short rows that do not lie end to end (an F-ordered or stepped array) are keyed one row at a time; each
                // row's counted order is written straight into a contiguous index row.
                if (_rowKeys != null && _outStride == sizeof(long))
                {
                    for (long l = 0; l < count; l++)
                        ArgSortShortRow(text + l * textStep, output + l * outputStep);
                    return;
                }
                for (long l = 0; l < count; l++)
                    ArgSortLine(text + l * textStep, output + l * outputStep);
            }

            /// <summary>
            ///     Argsorts a block of rows of at most <see cref="CountRankMax"/> elements that lie end to end: every chunk of
            ///     rows is keyed by ONE key-kernel call over its elements, then each row is ranked from its keys. A row this
            ///     short is an insertion sort in NumPy (stable whatever the kind), so its counted stable ranks are the answer;
            ///     they are scattered straight into a contiguous index row. A row whose keys tie without covering its strings
            ///     (or a strided index row) takes the per-line route instead.
            /// </summary>
            /// <param name="text">Row 0's element 0.</param>
            /// <param name="output">Index row 0's slot 0.</param>
            /// <param name="outputStep">The byte distance between index rows.</param>
            /// <param name="count">The number of rows.</param>
            /// <remarks>The packing is chosen per chunk (from all its units), which keeps every row's keys order-preserving.</remarks>
            private void ArgSortShortRows(byte* text, byte* output, long outputStep, long count)
            {
                long n = _n;
                long rowBytes = n * _textStride;
                long rowsPerChunk = _rowKeyCapacity / n;
                ulong* sortedKeys = _keyBlock;
                for (long first = 0; first < count; first += rowsPerChunk)
                {
                    long rows = Math.Min(rowsPerChunk, count - first);
                    byte* chunk = text + first * rowBytes;
                    int covered = ComputeKeys(chunk, _textStride, null, rows * n, 0, _rowKeys);
                    bool exact = covered == _units;
                    for (long r = 0; r < rows; r++)
                    {
                        byte* row = output + (first + r) * outputStep;
                        if (_outStride == sizeof(long))
                        {
                            CountRankOrder(_rowKeys + r * n, n, sortedKeys, (long*)row);
                            // Exact keys: the counted order is the answer. Keys that do not cover the strings give it too
                            // when no two of the row's keys are equal (the strings then differ inside the keyed units,
                            // which order them); a tie goes to the per-line route, which keys the following units.
                            if (exact || !HasAdjacentEqual(sortedKeys, n))
                                continue;
                        }
                        ArgSortLine(chunk + r * rowBytes, row);
                    }
                }
            }

            /// <summary>
            ///     Argsorts one row of at most <see cref="CountRankMax"/> elements into a contiguous index row: the row keyed
            ///     alone (any element stride), then counted straight into the output. The answer for the same reason as in
            ///     <see cref="ArgSortShortRows"/> (NumPy insertion-sorts such a row, which is the stable order); a row whose keys
            ///     tie without covering its strings goes to the per-line route.
            /// </summary>
            /// <param name="text">The row's element 0.</param>
            /// <param name="output">The index row's slot 0 (contiguous).</param>
            private void ArgSortShortRow(byte* text, byte* output)
            {
                long n = _n;
                if (_units == 0)
                {
                    // NumPy's sorts leave items of size zero in place: the identity.
                    for (long i = 0; i < n; i++)
                        ((long*)output)[i] = i;
                    return;
                }
                int covered = ComputeKeys(text, _textStride, null, n, 0, _rowKeys);
                CountRankOrder(_rowKeys, n, _keyBlock, (long*)output);
                if (covered != _units && HasAdjacentEqual(_keyBlock, n))
                    ArgSortLine(text, output);
            }

            /// <summary>
            ///     The block kernel of the in-place value sort: operand 0 holds the text lines, each copied out, ordered and
            ///     written back sorted.
            /// </summary>
            /// <param name="dataptrs">The first line's start.</param>
            /// <param name="strides">The byte distance between consecutive lines.</param>
            /// <param name="count">The number of lines in the block.</param>
            /// <param name="aux">Unused.</param>
            internal void SortLinesInPlace(void** dataptrs, long* strides, long count, void* aux)
            {
                BlockCalls++;
                byte* line = (byte*)dataptrs[0];
                long step = strides[0];
                for (long l = 0; l < count; l++)
                    SortLineInPlace(line + l * step);
            }

            /// <summary>
            ///     The block kernel of the out-of-place value sort: operand 0 holds the source lines, operand 1 the result lines.
            /// </summary>
            /// <param name="dataptrs">The first line's start in each operand.</param>
            /// <param name="strides">The byte distance between consecutive lines in each operand.</param>
            /// <param name="count">The number of lines in the block.</param>
            /// <param name="aux">Unused.</param>
            internal void SortLinesInto(void** dataptrs, long* strides, long count, void* aux)
            {
                BlockCalls++;
                byte* src = (byte*)dataptrs[0], dst = (byte*)dataptrs[1];
                long srcStep = strides[0], dstStep = strides[1];
                for (long l = 0; l < count; l++)
                    SortLineInto(src + l * srcStep, dst + l * dstStep);
            }

            /// <summary>Argsorts one line: the stable order, then (for the unstable kinds) the shortcut or NumPy's introsort.</summary>
            /// <param name="text">The text line's element 0.</param>
            /// <param name="output">The index line's slot 0.</param>
            private void ArgSortLine(byte* text, byte* output)
            {
                long n = _n;
                if (_units == 0)
                {
                    // NumPy's sorts return before comparing items of size zero: every kind leaves the identity.
                    for (long i = 0; i < n; i++)
                        *(long*)(output + i * _outStride) = i;
                    return;
                }

                // Exact keys (the whole string fits in its key) and an unstable sort that must find NumPy's tie order: run
                // the port straight over the key records — key order is string order, key equality string equality — when
                // the line is short, when its duplicates are certain, or when a sample finds a repeat; otherwise the stable
                // radix order is cheaper, and with no duplicate it is the answer.
                int covered = KeyLine(text, _textStride);
                bool exact = covered == _units;
                // A line of at most CountRankMax elements is counted, never radix-sorted: it needs no varying bits.
                _varying = n > CountRankMax ? VaryingBits(_keyBlock, n) : 0;
                if (_trackSame && !exact && n <= TailPortMax && !DirectPortDisabled && !TailPortDisabled
                    && SampleRepeats(_keyBlock, n) is >= TailPortMinRepeats and <= TailPortMaxRepeats)
                {
                    DirectPortTail(text, output, covered);
                    TailPortRuns++;
                    DirectPortRuns++;
                    PortRuns++;
                    return;
                }
                if (_trackSame && exact && !DirectPortDisabled)
                {
                    int low = _varying == 0 ? 0 : BitOperations.TrailingZeroCount(_varying);
                    int window = _varying == 0 ? 0 : 64 - BitOperations.LeadingZeroCount(_varying) - low;
                    // More elements than the window has values: two of them hold the same key, so the port is needed.
                    bool duplicatesCertain = window < 32 && n > (1L << window);
                    if (n <= DirectPortMax || duplicatesCertain || SampleHasDuplicates(_keyBlock, n))
                    {
                        if (window <= 32 && (ulong)n <= PackedIndexMask + 1 && !PackedWindowDisabled)
                            DirectPortPacked(output, low, window);
                        else
                            DirectPortRecords(output);
                        DirectPortRuns++;
                        PortRuns++;
                        return;
                    }
                }
                _sameLive = _trackSame && !exact;
                OrderKeyed(text, _textStride, covered);
                long* order = _x;
                if (_trackSame)
                {
                    // Equal strings sit next to each other in the stable order: exact keys show them as equal adjacent
                    // keys; otherwise the tie pass flagged them.
                    bool duplicates = ShortcutDisabled || HasDuplicates(exact);
                    if (duplicates)
                    {
                        // Dense ranks of the stable order, first per sorted position into the key half the sorted keys do
                        // NOT occupy (the equality test may still read them), then scattered to each element's initial
                        // slot: record i holds (rank of element i, i), NumPy's initial index array.
                        ulong* ranks = _kt;
                        ulong rank = 0;
                        for (long i = 0; i < n; i++)
                        {
                            if (i > 0 && !SameAsPrevious(exact, i))
                                rank++;
                            ranks[i] = rank;
                        }
                        if ((ulong)n <= PackedIndexMask + 1)
                        {
                            // Ranks and indices below 2^32: one ulong per record, in the sorted keys' half (read no more).
                            ulong* rec = _k;
                            for (long i = 0; i < n; i++)
                            {
                                long element = order[i];
                                rec[element] = (ranks[i] << 32) | (ulong)element;
                            }
                            IntroArgSortPacked(rec, n);
                            for (long i = 0; i < n; i++)
                                *(long*)(output + i * _outStride) = (long)(rec[i] & PackedIndexMask);
                        }
                        else
                        {
                            // A line of more than 2^32 elements: two ulongs per record, in a buffer of their own (every
                            // block of the job still holds an input of the scatter).
                            long recBytes = n * 2 * sizeof(ulong);
                            ulong* rec = (ulong*)SizeBucketedBufferPool.Take(recBytes);
                            try
                            {
                                for (long i = 0; i < n; i++)
                                {
                                    long element = order[i];
                                    rec[2 * element] = ranks[i];
                                    rec[2 * element + 1] = (ulong)element;
                                }
                                IntroArgSortRecords(rec, n);
                                for (long i = 0; i < n; i++)
                                    *(long*)(output + i * _outStride) = (long)rec[2 * i + 1];
                            }
                            finally
                            {
                                SizeBucketedBufferPool.Return((IntPtr)rec, recBytes);
                            }
                        }
                        PortRuns++;
                        return;
                    }
                    ShortcutRuns++;
                }
                for (long i = 0; i < n; i++)
                    *(long*)(output + i * _outStride) = order[i];
            }

            /// <summary>
            ///     The direct port over WINDOW-packed records: element i's record is the line's varying key bits (at most 32,
            ///     shifted down by <paramref name="low"/>) in the high half and i in the low half. Window order is key order
            ///     (the bits outside the window are the same in every key), so <see cref="IntroArgSortPacked"/> makes NumPy's
            ///     comparisons over records half the size of (key, index) pairs — 2× the two-ulong port on tie-heavy lines.
            /// </summary>
            /// <param name="output">The index line's slot 0.</param>
            /// <param name="low">The lowest varying key bit.</param>
            /// <param name="window">The number of key bits from <paramref name="low"/> to the highest varying one (at most 32;
            ///     0 when every key is the same).</param>
            private void DirectPortPacked(byte* output, int low, int window)
            {
                long n = _n;
                ulong mask = window == 0 ? 0 : ulong.MaxValue >> (64 - window);
                // The indices the key kernel read are no longer needed: their block holds the records.
                ulong* rec = (ulong*)_idxBlock;
                ulong* keys = _keyBlock;
                for (long i = 0; i < n; i++)
                    rec[i] = (((keys[i] >> low) & mask) << 32) | (ulong)i;
                IntroArgSortPacked(rec, n);
                for (long i = 0; i < n; i++)
                    *(long*)(output + i * _outStride) = (long)(rec[i] & PackedIndexMask);
                PackedDirectRuns++;
            }

            /// <summary>
            ///     The direct port over (prefix key, index) records whose keys do not cover the strings: NumPy's introsort with
            ///     its string compare, keys first (<see cref="IntroArgSortTextRecords"/>).
            /// </summary>
            /// <param name="text">The text line's element 0.</param>
            /// <param name="output">The index line's slot 0.</param>
            /// <param name="covered">The units the keys cover (fewer than the element's).</param>
            private void DirectPortTail(byte* text, byte* output, int covered)
            {
                long n = _n;
                // The indices the key kernel read are no longer needed: their block (2n slots) holds the records.
                ulong* rec = (ulong*)_idxBlock;
                for (long i = 0; i < n; i++)
                {
                    rec[2 * i] = _keyBlock[i];
                    rec[2 * i + 1] = (ulong)i;
                }
                IntroArgSortTextRecords(rec, n, text, _textStride, covered * _unit, _width, _unit);
                for (long i = 0; i < n; i++)
                    *(long*)(output + i * _outStride) = (long)rec[2 * i + 1];
            }

            /// <summary>The direct port over two-ulong (key, index) records, for keys whose varying bits exceed 32.</summary>
            /// <param name="output">The index line's slot 0.</param>
            private void DirectPortRecords(byte* output)
            {
                long n = _n;
                // The indices the key kernel read are no longer needed: their block (2n slots) holds the records.
                ulong* rec = (ulong*)_idxBlock;
                for (long i = 0; i < n; i++)
                {
                    rec[2 * i] = _keyBlock[i];
                    rec[2 * i + 1] = (ulong)i;
                }
                IntroArgSortRecords(rec, n);
                for (long i = 0; i < n; i++)
                    *(long*)(output + i * _outStride) = (long)rec[2 * i + 1];
            }

            /// <summary>
            ///     Whether sorted position <paramref name="i"/> holds the same string as position i - 1 (after
            ///     <see cref="OrderKeyed"/>): equal adjacent keys when the keys are exact, the tie pass's flag otherwise.
            /// </summary>
            /// <param name="exact">Whether the line's keys cover its strings.</param>
            /// <param name="i">A sorted position (at least 1).</param>
            /// <returns>True for equal strings.</returns>
            private bool SameAsPrevious(bool exact, long i) => exact ? _k[i] == _k[i - 1] : _same[i] != 0;

            /// <summary>Whether the current line's stable order holds two equal strings (see <see cref="SameAsPrevious"/>).</summary>
            /// <param name="exact">Whether the line's keys cover its strings.</param>
            /// <returns>True when some sorted position holds the string of the position before it.</returns>
            private bool HasDuplicates(bool exact)
            {
                if (exact)
                    return HasAdjacentEqual(_k, _n);
                long flags = _n - 1;
                // The flags of positions 1..n-1, scanned with the vectorized span search (a span holds int.MaxValue bytes).
                if (new ReadOnlySpan<byte>(_same + 1, (int)Math.Min(flags, int.MaxValue)).IndexOfAnyExcept((byte)0) >= 0)
                    return true;
                return flags > int.MaxValue && AnyFlagPast(_same, _n);
            }

            /// <summary>Sorts one line in place: copied out, ordered, and written back sorted.</summary>
            /// <param name="line">The line's element 0.</param>
            private void SortLineInPlace(byte* line)
            {
                // The gather back reads every element from the copy, so the line can be overwritten in any order.
                Gather(line, _textStride, null, _n, _lineCopy, _width);
                StableOrder(_lineCopy, _width);
                Gather(_lineCopy, _width, _x, _n, line, _textStride);
            }

            /// <summary>Sorts one line out of place: the source line's stable order, gathered into the result line.</summary>
            /// <param name="src">The source line's element 0.</param>
            /// <param name="dst">The result line's element 0.</param>
            private void SortLineInto(byte* src, byte* dst)
            {
                StableOrder(src, _textStride);
                Gather(src, _textStride, _x, _n, dst, _outStride);
            }

            /// <summary>
            ///     Orders a line's element indices stably by their full strings into <see cref="_x"/>: prefix keys, a stable
            ///     radix sort, then the tie pass over runs of equal keys; keeps the duplicate flags when the job tracks them.
            /// </summary>
            /// <param name="text">The line's element 0.</param>
            /// <param name="stride">The byte distance between its elements.</param>
            private void StableOrder(byte* text, long stride)
            {
                _sameLive = false;
                int covered = KeyLine(text, stride);
                _varying = _n > CountRankMax ? VaryingBits(_keyBlock, _n) : 0;
                OrderKeyed(text, stride, covered);
            }

            /// <summary>
            ///     Keys every element of a line into slot i of the key block (element i at slot i), walking the line with the
            ///     key kernel's sequential form (no index array).
            /// </summary>
            /// <param name="text">The line's element 0.</param>
            /// <param name="stride">The byte distance between its elements.</param>
            /// <returns>The units the keys cover (the element width in units when the keys are exact).</returns>
            private int KeyLine(byte* text, long stride) => ComputeKeys(text, stride, null, _n, 0, _keyBlock);

            /// <summary>
            ///     The stable order of a line whose keys <see cref="KeyLine"/> computed: counted ranks for a short line, a
            ///     stable radix sort otherwise; then the tie pass when the keys do not cover the strings, and the duplicate
            ///     flags when the job keeps them.
            /// </summary>
            /// <param name="text">The line's element 0.</param>
            /// <param name="stride">The byte distance between its elements.</param>
            /// <param name="covered">The units the keys cover.</param>
            private void OrderKeyed(byte* text, long stride, int covered)
            {
                long n = _n;
                long* idx = _idxBlock, idxTmp = _idxBlock + n;
                ulong* keys = _keyBlock, keyTmp = _keyBlock + n;
                if (n <= CountRankMax)
                {
                    CountRankOrder(keys, n, keyTmp, idxTmp);
                    _x = idxTmp;
                    _xt = idx;
                    _k = keyTmp;
                    _kt = keys;
                }
                else
                {
                    // The radix carries the indices with the keys: element i starts at slot i. It sorts only the key bytes
                    // that hold a varying bit (_varying, computed with the keys).
                    for (long i = 0; i < n; i++)
                        idx[i] = i;
                    long* x = RadixSort.ArgSortU64(keys, keyTmp, idx, idxTmp, n, Hist, _varying, out ulong* k);
                    // The radix ping-pongs between the two buffers of each block; remember which holds the result.
                    _x = x;
                    _xt = x == idx ? idxTmp : idx;
                    _k = k;
                    _kt = k == keys ? keyTmp : keys;
                }
                if (_sameLive)
                {
                    _same[0] = 0;
                    for (long i = 1; i < n; i++)
                        _same[i] = _k[i] == _k[i - 1] ? (byte)1 : (byte)0;
                }
                if (covered < _units)
                    RefineRuns(text, stride, 0, n, covered);
            }

            /// <summary>
            ///     The tie pass over <c>[start, end)</c> of the sorted order: every run of at least two equal keys is ordered
            ///     by the units after <paramref name="fromUnit"/>.
            /// </summary>
            /// <param name="text">The line's element 0.</param>
            /// <param name="stride">The byte distance between its elements.</param>
            /// <param name="start">The first sorted position.</param>
            /// <param name="end">One past the last.</param>
            /// <param name="fromUnit">The units the current keys cover: inside a run the elements are equal up to here.</param>
            private void RefineRuns(byte* text, long stride, long start, long end, int fromUnit)
            {
                long s = start;
                while (s < end)
                {
                    ulong key = _k[s];
                    long e = s + 1;
                    while (e < end && _k[e] == key)
                        e++;
                    // RefineRun rewrites the keys of [s, e) only, so the scan continues from e unharmed.
                    if (e - s >= 2)
                        RefineRun(text, stride, s, e, fromUnit);
                    s = e;
                }
            }

            /// <summary>
            ///     Orders one run of elements equal up to <paramref name="fromUnit"/> by their remaining units, stably: a short
            ///     run by insertion with tail compares, a longer one by keying its next units and radix-sorting them, then
            ///     recursing into the runs those keys leave.
            /// </summary>
            /// <param name="text">The line's element 0.</param>
            /// <param name="stride">The byte distance between its elements.</param>
            /// <param name="s">The run's first sorted position.</param>
            /// <param name="e">One past its last.</param>
            /// <param name="fromUnit">The first unit not yet compared.</param>
            private void RefineRun(byte* text, long stride, long s, long e, int fromUnit)
            {
                // Equal over the whole width: equal strings, already flagged as such.
                if (fromUnit >= _units)
                    return;
                long m = e - s;
                if (m <= TieInsertionMax)
                {
                    InsertionByTail(text, stride, s, e, fromUnit);
                    return;
                }
                int covered = ComputeKeys(text, stride, _x + s, m, fromUnit, _k + s);
                long* x = RadixSort.ArgSortU64(_k + s, _kt + s, _x + s, _xt + s, m, Hist, VaryingBits(_k + s, m), out ulong* k);
                // A radix that ended in the scratch halves: bring the run's order and keys back to the main buffers.
                if (x != _x + s)
                    Buffer.MemoryCopy(x, _x + s, m * sizeof(long), m * sizeof(long));
                if (k != _k + s)
                    Buffer.MemoryCopy(k, _k + s, m * sizeof(ulong), m * sizeof(ulong));
                if (_sameLive)
                {
                    for (long i = s + 1; i < e; i++)
                        _same[i] = _k[i] == _k[i - 1] ? (byte)1 : (byte)0;
                }
                RefineRuns(text, stride, s, e, fromUnit + covered);
            }

            /// <summary>
            ///     Stable insertion sort of the run <c>[s, e)</c> by the units from <paramref name="fromUnit"/> on, then (when
            ///     tracked) the duplicate flags inside it.
            /// </summary>
            /// <param name="text">The line's element 0.</param>
            /// <param name="stride">The byte distance between its elements.</param>
            /// <param name="s">The run's first sorted position.</param>
            /// <param name="e">One past its last.</param>
            /// <param name="fromUnit">The first unit not yet compared.</param>
            private void InsertionByTail(byte* text, long stride, long s, long e, int fromUnit)
            {
                int from = fromUnit * _unit;
                for (long i = s + 1; i < e; i++)
                {
                    long v = _x[i];
                    byte* pv = text + v * stride;
                    long j = i;
                    while (j > s && CompareTail(pv, text + _x[j - 1] * stride, from, _width, _unit) < 0)
                    {
                        _x[j] = _x[j - 1];
                        j--;
                    }
                    _x[j] = v;
                }
                if (_sameLive)
                {
                    for (long i = s + 1; i < e; i++)
                        _same[i] = CompareTail(text + _x[i - 1] * stride, text + _x[i] * stride, from, _width, _unit) == 0
                            ? (byte)1 : (byte)0;
                }
            }


            /// <summary>
            ///     Whether an evenly spaced sample of a long line's keys holds a repeated key: the cheap evidence that the
            ///     line has duplicates, so the direct key-record port beats the stable radix order (whose distinct-strings
            ///     shortcut a line with duplicates cannot take). Either route gives the same permutation.
            /// </summary>
            /// <param name="keys">The keys, element i at slot i.</param>
            /// <param name="n">The line length (more than <see cref="DuplicateSampleSize"/>).</param>
            /// <returns>True when two sampled keys are equal.</returns>
            [SkipLocalsInit]
            private static bool SampleHasDuplicates(ulong* keys, long n)
            {
                // An open-addressing set of twice the sample's size (a power of two), keyed by a multiplicative hash; a slot
                // holds key + 1 so that 0 marks it empty (a key of ulong.MaxValue wraps to 0 and is simply never stored:
                // missing one repeat only steers the route, never the answer).
                const int Slots = DuplicateSampleSize * 2;
                ulong* set = stackalloc ulong[Slots];
                new Span<ulong>(set, Slots).Clear();
                long step = n / DuplicateSampleSize;
                for (int i = 0; i < DuplicateSampleSize; i++)
                {
                    ulong stored = keys[i * step] + 1;
                    if (stored == 0)
                        continue;
                    int slot = (int)((stored * 0x9E3779B97F4A7C15UL) >> 56) & (Slots - 1);
                    while (set[slot] != 0)
                    {
                        if (set[slot] == stored)
                            return true;
                        slot = (slot + 1) & (Slots - 1);
                    }
                    set[slot] = stored;
                }
                return false;
            }

            /// <summary>
            ///     How many keys of a sample of the line repeat an earlier sampled key: the whole line when it holds at most
            ///     <see cref="DuplicateSampleSize"/> keys, else that many evenly spaced ones. Steers the route only.
            /// </summary>
            /// <param name="keys">The keys, element i at slot i.</param>
            /// <param name="n">The line length (at least 1).</param>
            /// <returns>The repeats found (0 to the sample's size - 1).</returns>
            [SkipLocalsInit]
            private static int SampleRepeats(ulong* keys, long n)
            {
                // The open-addressing set of SampleHasDuplicates: twice the sample's size, key + 1 stored (0 marks an empty
                // slot; a key of ulong.MaxValue wraps to 0 and is skipped, which only steers the route).
                const int Slots = DuplicateSampleSize * 2;
                ulong* set = stackalloc ulong[Slots];
                new Span<ulong>(set, Slots).Clear();
                long count = Math.Min(n, DuplicateSampleSize);
                long step = n / count;
                int repeats = 0;
                for (long i = 0; i < count; i++)
                {
                    ulong stored = keys[i * step] + 1;
                    if (stored == 0)
                        continue;
                    int slot = (int)((stored * 0x9E3779B97F4A7C15UL) >> 56) & (Slots - 1);
                    while (set[slot] != 0 && set[slot] != stored)
                        slot = (slot + 1) & (Slots - 1);
                    if (set[slot] == stored)
                        repeats++;
                    else
                        set[slot] = stored;
                }
                return repeats;
            }

            /// <summary>
            ///     Keys the selected elements from unit <paramref name="fromUnit"/> on (<see cref="TextKeyer.ComputeKeys"/>: <c>S</c>
            ///     takes the next 8 bytes; <c>U</c> first takes 8 units of 8 bits, and when a unit does not fit 8 bits re-keys
            ///     with the widest packing its units allow).
            /// </summary>
            /// <param name="text">The line's element 0.</param>
            /// <param name="stride">The byte distance between its elements.</param>
            /// <param name="index">The element of each key slot.</param>
            /// <param name="count">The number of keys.</param>
            /// <param name="fromUnit">The first unit keyed (below the width).</param>
            /// <param name="keys">The key slots.</param>
            /// <returns>The units the keys cover (fewer than the packing holds when the element ends first).</returns>
            private int ComputeKeys(byte* text, long stride, long* index, long count, int fromUnit, ulong* keys)
                => _keyer.ComputeKeys(text, stride, index, count, fromUnit, keys);

            /// <summary>Runs the gather kernel, or its managed twin on a host without dynamic code.</summary>
            /// <param name="src">See <see cref="TextGatherFunc"/>.</param>
            /// <param name="srcStride">See <see cref="TextGatherFunc"/>.</param>
            /// <param name="index">See <see cref="TextGatherFunc"/>.</param>
            /// <param name="count">See <see cref="TextGatherFunc"/>.</param>
            /// <param name="dst">See <see cref="TextGatherFunc"/>.</param>
            /// <param name="dstStride">See <see cref="TextGatherFunc"/>.</param>
            private void Gather(byte* src, long srcStride, long* index, long count, byte* dst, long dstStride)
            {
                if (_gather is not null)
                    _gather(src, srcStride, index, count, dst, dstStride);
                else
                    ILKernelGenerator.TextGatherManaged(_width, src, srcStride, index, count, dst, dstStride);
            }
        }
    }
}
