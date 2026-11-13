using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    // ── the select of a SINGLE kth (np.partition / np.argpartition with one kth) ──────────────────────────────────
    //
    // NumPy hands a single kth of a 32- or 64-bit type to x86-simd-sort (`introselect_noarg`: `nkth == 1 &&
    // quickselect_dispatch(...)`) on AVX2 / AVX-512 hosts. Its arrangement is that library's: a window of up to its sorting
    // network's size (128-256 elements) comes back fully SORTED, longer lines are partitioned with vector compress-stores.
    // No select reproduces it, and the unit tests pin NumPy's small sorted results (np.partition([3, 4, 2, 1], 3) is
    // [1, 2, 3, 4]). So a single kth keeps the select NumSharp has always run for it, unchanged: NumPy's pivot stack and
    // linear shortcuts, a window of up to 16 elements insertion-sorted (sorted, as x86-simd-sort leaves a short line),
    // the branchless block partition with a pdqsort-style drain, and a heap sort once its depth budget is spent. The
    // edge selection (QuickSelect.EdgeSelect.cs) runs before it on long contiguous lines.
    //
    // Several kth run NumPy's scalar introselect_ on every host, and there NumSharp runs its exact port
    // (QuickSelect.Introselect.cs), which this file is NOT: a kth list longer than one comes here only through
    // PartitionAtManyTieBlind, for the quantile kernels' rows whose bits cannot show the arrangement.
    internal static partial class QuickSelect
    {
        /// <summary>
        ///     np.partition's value select of one line around a SINGLE kth: the select NumSharp has always run for one kth
        ///     (see the block above), which leaves a short line sorted, as x86-simd-sort leaves NumPy's.
        /// </summary>
        /// <typeparam name="T">The element type (a float line must be NaN-free: the line kernels compact NaNs first).</typeparam>
        /// <param name="buf">The line, partitioned in place.</param>
        /// <param name="n">The line's length.</param>
        /// <param name="kth">The position to place; outside <c>[0, n)</c> nothing moves.</param>
        /// <remarks>FOOTGUN: not NumPy's introselect_ (that is <see cref="PartitionAtMany{T}(T*, long, long*, int)"/>);
        ///     calling this once per kth of a list does not give NumPy's arrangement for several kth.
        ///     [SkipLocalsInit]: the pivot stack is written before it is read.</remarks>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void SelectSingleKth<T>(T* buf, long n, long kth) where T : unmanaged, IComparable<T>
        {
            if (n <= 1 || kth < 0 || kth >= n) return;
            long* pivots = stackalloc long[PivotStackMax];
            int npiv = 0;
            IntroSelectBlock(buf, n, kth, pivots, ref npiv);
        }

        /// <summary>
        ///     np.argpartition's select of one line around a SINGLE kth, the index column moving with every swap: the
        ///     index-tracking twin of <see cref="SelectSingleKth{T}(T*, long, long)"/>.
        /// </summary>
        /// <typeparam name="T">The element type (a float line must be NaN-free: the line kernels compact NaNs first).</typeparam>
        /// <param name="buf">The line's values (scratch).</param>
        /// <param name="idx">The original-index column beside <paramref name="buf"/> (the answer).</param>
        /// <param name="n">The line's length.</param>
        /// <param name="kth">The position to place; outside <c>[0, n)</c> nothing moves.</param>
        /// <remarks>[SkipLocalsInit]: the pivot stack is written before it is read.</remarks>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void SelectSingleKth<T>(T* buf, long* idx, long n, long kth) where T : unmanaged, IComparable<T>
        {
            if (n <= 1 || kth < 0 || kth >= n) return;
            long* pivots = stackalloc long[PivotStackMax];
            int npiv = 0;
            IntroSelectBlock(buf, idx, n, kth, pivots, ref npiv);
        }

        /// <summary>
        ///     The quantile kernels' TIE-BLIND select: places the values of every position in <paramref name="sortedKs"/>
        ///     with the select the family always ran (a line of up to <see cref="InsertionSortThreshold"/> elements
        ///     insertion-sorted whole, a longer one selected kth by kth with a shared pivot stack). The arrangement among
        ///     equal values is NOT NumPy's.
        /// </summary>
        /// <typeparam name="T">The element type (NaN-free).</typeparam>
        /// <param name="buf">The line (the kernel's scratch copy).</param>
        /// <param name="n">The line's length.</param>
        /// <param name="sortedKs">The positions the kernel reads, ascending.</param>
        /// <param name="nKs">How many.</param>
        /// <remarks>
        ///     FOOTGUN: only for a caller that reads nothing but the values at <paramref name="sortedKs"/> and whose line
        ///     cannot show the arrangement: every correct partition puts the same VALUE at a kth, and equal values have
        ///     equal bits except <c>+0.0</c> / <c>-0.0</c>, so a line holding both zero signs (or NaNs, which change the
        ///     order itself) takes NumPy's <see cref="PartitionAtMany{T}(T*, int, int*, int)"/> instead. The kernel's
        ///     prescan decides that; this route keeps the short-row speed (np.quantile over axis 1 of a (100000, 5)
        ///     array: one inlined insertion sort per row instead of NumPy's four select passes) and lets a long line skip
        ///     NumPy's extra 0 / -1 positions. [SkipLocalsInit]: the pivot stack is written before it is read;
        ///     [AggressiveOptimization]: tier-1 from the first call, so the whole-line insertion sort inlines into the
        ///     per-row path.
        /// </remarks>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void PartitionAtManyTieBlind<T>(T* buf, int n, int* sortedKs, int nKs)
            where T : unmanaged, IComparable<T>
        {
            if (nKs == 0 || n <= 1) return;
            if (n <= InsertionSortThreshold)
            {
                InsertionSort(buf, 0, n - 1);
                return;
            }
            long* pivots = stackalloc long[PivotStackMax];
            int npiv = 0;
            for (int i = 0; i < nKs; i++)
            {
                long k = sortedKs[i];
                if (k < 0 || k >= n) continue;
                IntroSelectBlock(buf, n, k, pivots, ref npiv);
            }
        }

        /// <summary>
        ///     <c>store_pivot</c> over a by-reference depth, for the single-kth select (whose one call never reads what it
        ///     stores; kept so its statements stay the ones it always ran).
        /// </summary>
        /// <param name="pivot">The placed position.</param>
        /// <param name="kth">The kth being selected.</param>
        /// <param name="pivots">The stack.</param>
        /// <param name="npiv">The stack's depth.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void StorePivot(long pivot, long kth, long* pivots, ref int npiv)
        {
            if (pivot == kth && npiv == PivotStackMax) pivots[npiv - 1] = pivot;
            else if (pivot >= kth && npiv < PivotStackMax) { pivots[npiv] = pivot; npiv++; }
        }

        /// <summary>
        ///     The single-kth select: pops the stack to narrow <c>[low, high]</c>, insertion-sorts a window of up to
        ///     <see cref="InsertionSortThreshold"/> elements, takes NumPy's two linear shortcuts
        ///     (<see cref="DumbSelect{T, TLess, TCol}"/> for a kth within 3 of low, <see cref="LastMaxIndex{T, TLess}"/> for a
        ///     float kth at the line's end), then median-of-3 + branchless block partition down to the insertion-sort /
        ///     heap-sort thresholds.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="v">The line.</param>
        /// <param name="n">The line's length.</param>
        /// <param name="kth">The position to place.</param>
        /// <param name="pivots">The pivot stack.</param>
        /// <param name="npiv">The stack's depth.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void IntroSelectBlock<T>(T* v, long n, long kth, long* pivots, ref int npiv)
            where T : unmanaged, IComparable<T>
        {
            long low = 0, high = n - 1;
            // Narrow from already-placed pivots: pop everything ≤ kth (raising low), stop at the
            // first pivot > kth (lowering high). An exact hit means kth is already in place.
            while (npiv > 0)
            {
                long top = pivots[npiv - 1];
                if (top > kth) { high = top - 1; break; }
                if (top == kth) return;
                low = top + 1;
                npiv--;
            }

            // A window the loop below would insertion-sort on its first step is sorted before NumPy's shortcuts, which
            // NumPy takes for every window: on a few elements one insertion sort beat up to three minimum passes
            // (100,000 rows of 5, np.quantile(m, 0.5, axis=1): 7.8 ms against 8.7 ms). The values placed are the same
            // order statistics; only the arrangement of a short window differs.
            if (high - low + 1 <= InsertionSortThreshold)
            {
                InsertionSort(v, low, high);
                StorePivot(kth, kth, pivots, ref npiv);
                return;
            }

            // NumPy's linear shortcuts, in NumPy's order (see the block above DumbSelect).
            if (kth - low < 3)
            {
                DumbSelect<T, RawLess<T>, ValueColumn<T>>(v, default, low, high, kth);
                StorePivot(kth, kth, pivots, ref npiv);
                return;
            }
            if (IsInexact<T>() && kth == n - 1)
            {
                // No pivot above kth can exist here, so the window ends at the line's end, which NumPy scans to.
                Swap(v, kth, LastMaxIndex<T, RawLess<T>>(v, low, n - 1));
                return;
            }

            int depthLimit = 2 * Log2(n);
            while (low < high)
            {
                long len = high - low + 1;
                if (len <= InsertionSortThreshold)
                {
                    InsertionSort(v, low, high);
                    StorePivot(kth, kth, pivots, ref npiv);
                    return;
                }
                if (depthLimit == 0)
                {
                    // Median-of-3 degraded (adversarial input) — heap-sort the window for the
                    // O(n log n) worst-case guarantee, matching NumPy's med-of-median-5 fallback.
                    HeapSort(v, low, high);
                    StorePivot(kth, kth, pivots, ref npiv);
                    return;
                }
                depthLimit--;

                long p = PartitionBlock(v, low, high);
                if (p != kth) StorePivot(p, kth, pivots, ref npiv);
                // Both may fire when p == kth: the window collapses and the loop exits.
                if (p >= kth) high = p - 1;
                if (p <= kth) low = p + 1;
            }
            StorePivot(kth, kth, pivots, ref npiv);
        }

        /// <summary>
        ///     Median-of-3 pivot setup (NumPy <c>median3_swap_</c>): moves the median of
        ///     <c>{low, mid, high}</c> to <paramref name="low"/> (the pivot) and the smallest of
        ///     the three to <c>low+1</c>. This leaves <c>v[low+1] ≤ pivot</c> and
        ///     <c>v[high] ≥ pivot</c> as sentinels, letting the partition scan run unguarded.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="v">The line.</param>
        /// <param name="low">The window's first position.</param>
        /// <param name="mid">The window's middle.</param>
        /// <param name="high">The window's last position.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void Median3Swap<T>(T* v, long low, long mid, long high)
            where T : unmanaged, IComparable<T>
        {
            if (LtV(v[high], v[mid])) Swap(v, high, mid);
            if (LtV(v[high], v[low])) Swap(v, high, low);
            if (LtV(v[low], v[mid])) Swap(v, low, mid);
            Swap(v, mid, low + 1);
        }

        /// <summary>
        ///     Scalar unguarded Hoare crossing used to finish the small middle a block partition
        ///     leaves (or a whole small window). The caller guarantees a sentinel ≤ pivot at
        ///     <paramref name="ll"/> and a sentinel ≥ pivot at <paramref name="hh"/>, so neither
        ///     do/while can run off the region — no bounds test in the hot loop.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="v">The line.</param>
        /// <param name="pivot">The pivot's address.</param>
        /// <param name="ll">The position before the left scan's first look.</param>
        /// <param name="hh">The position after the right scan's first look.</param>
        /// <returns>The right scan's last stop, the pivot's final position.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe long ScalarPartitionFinish<T>(T* v, T* pivot, long ll, long hh)
            where T : unmanaged, IComparable<T>
        {
            for (;;)
            {
                do { ll++; } while (LtPtr(v + ll, pivot));
                do { hh--; } while (LtPtr(pivot, v + hh));
                if (hh < ll) break;
                Swap(v, ll, hh);
            }
            return hh;
        }

        /// <summary>
        ///     Branchless block-Hoare partition of <c>[low, high]</c> around the median-of-3
        ///     pivot, returning the pivot's final index (BlockQuicksort / pdqsort scheme). The
        ///     bulk of each side is scanned into fixed <see cref="BlockSize"/> offset blocks with
        ///     a data-independent write (<c>numX += cmp ? 0 : 1</c>), then paired offsets are
        ///     swapped — eliminating the ~50% branch-mispredict of a scalar Hoare inner loop on
        ///     random data. The sub-block-size middle is finished by
        ///     <see cref="ScalarPartitionFinish{T}"/>. Callers must supply a window larger than
        ///     <see cref="InsertionSortThreshold"/> so the sentinels at <c>low+1</c>/<c>high</c>
        ///     exist. The line kernels compact NaNs out before this runs, so the raw
        ///     <see cref="LtV{T}"/> ordering is total here.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="v">The line.</param>
        /// <param name="low">The window's first position.</param>
        /// <param name="high">The window's last position.</param>
        /// <returns>The pivot's final position.</returns>
        /// <remarks>Its drain pairs a partly consumed block's LAST offset first (pdqsort), which is not Hoare's pairing:
        ///     the multi-kth select (<see cref="BlockScan{T, TLess, TCol}"/>) hands the crossing to Hoare's own loop
        ///     instead.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe long PartitionBlock<T>(T* v, long low, long high)
            where T : unmanaged, IComparable<T>
        {
            long mid = low + ((high - low) >> 1);
            Median3Swap(v, low, mid, high);

            byte* offL = stackalloc byte[BlockSize];
            byte* offR = stackalloc byte[BlockSize];
            // The pivot lives at v[low] for the whole partition; comparing against its address
            // (rather than a by-value copy) is what keeps the block scan in registers.
            T* pivot = v + low;
            // v[low+1] ≤ pivot and v[high] ≥ pivot are the sentinels, so the unknown span is
            // (low+1, high): scan up from ll and down from hh.
            long ll = low + 2, hh = high - 1;
            int numL = 0, numR = 0, startL = 0, startR = 0;
            long baseL = ll, baseR = hh;

            while (hh - ll + 1 > 2 * BlockSize)
            {
                if (numL == 0)
                {
                    startL = 0; baseL = ll;
                    for (int i = 0; i < BlockSize; i++) { offL[numL] = (byte)i; numL += LtPtr(v + ll + i, pivot) ? 0 : 1; }
                }
                if (numR == 0)
                {
                    startR = 0; baseR = hh;
                    for (int i = 0; i < BlockSize; i++) { offR[numR] = (byte)i; numR += LtPtr(pivot, v + hh - i) ? 0 : 1; }
                }
                int num = Math.Min(numL, numR);
                for (int i = 0; i < num; i++) Swap(v, baseL + offL[startL + i], baseR - offR[startR + i]);
                numL -= num; numR -= num; startL += num; startR += num;
                if (numL == 0) ll += BlockSize;
                if (numR == 0) hh -= BlockSize;
            }

            // Drain the one partially-consumed block (at most one is non-empty here) against a
            // scalar scan of the opposite side, keeping the confirmed regions as sentinels.
            while (numL > 0)
            {
                while (LtPtr(pivot, v + hh)) hh--;
                long li = baseL + offL[startL + numL - 1];
                if (li >= hh) break;
                Swap(v, li, hh);
                hh--; numL--;
            }
            while (numR > 0)
            {
                while (LtPtr(v + ll, pivot)) ll++;
                long ri = baseR - offR[startR + numR - 1];
                if (ri <= ll) break;
                Swap(v, ri, ll);
                ll++; numR--;
            }

            long p = ScalarPartitionFinish(v, pivot, ll - 1, hh + 1);
            Swap(v, low, p);   // move pivot from low into its final crossing position
            return p;
        }

        // ── its index-tracking twin (values + parallel idx column swap together) ──

        /// <summary>
        ///     Index-tracking single-kth select: mirrors the value-only
        ///     <see cref="IntroSelectBlock{T}(T*, long, long, long*, ref int)"/> exactly, moving
        ///     <paramref name="idx"/> through every swap — NumPy's linear shortcuts included.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="v">The line's values.</param>
        /// <param name="idx">The original-index column.</param>
        /// <param name="n">The line's length.</param>
        /// <param name="kth">The position to place.</param>
        /// <param name="pivots">The pivot stack.</param>
        /// <param name="npiv">The stack's depth.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void IntroSelectBlock<T>(T* v, long* idx, long n, long kth, long* pivots, ref int npiv)
            where T : unmanaged, IComparable<T>
        {
            long low = 0, high = n - 1;
            // Narrow from already-placed pivots: pop everything ≤ kth (raising low), stop at the
            // first pivot > kth (lowering high). An exact hit means kth is already in place.
            while (npiv > 0)
            {
                long top = pivots[npiv - 1];
                if (top > kth) { high = top - 1; break; }
                if (top == kth) return;
                low = top + 1;
                npiv--;
            }

            // A short window is insertion-sorted first, as the value path does (see there).
            if (high - low + 1 <= InsertionSortThreshold)
            {
                InsertionSort(v, idx, low, high);
                StorePivot(kth, kth, pivots, ref npiv);
                return;
            }

            // NumPy's linear shortcuts (see DumbSelect / LastMaxIndex): the values decide, the index column follows.
            if (kth - low < 3)
            {
                for (long i = low; i <= kth; i++)
                    Swap(v, idx, i, FirstMinIndex<T, RawLess<T>>(v, i, high));
                StorePivot(kth, kth, pivots, ref npiv);
                return;
            }
            if (IsInexact<T>() && kth == n - 1)
            {
                Swap(v, idx, kth, LastMaxIndex<T, RawLess<T>>(v, low, n - 1));
                return;
            }

            int depthLimit = 2 * Log2(n);
            while (low < high)
            {
                long len = high - low + 1;
                if (len <= InsertionSortThreshold)
                {
                    InsertionSort(v, idx, low, high);
                    StorePivot(kth, kth, pivots, ref npiv);
                    return;
                }
                if (depthLimit == 0)
                {
                    HeapSort(v, idx, low, high);
                    StorePivot(kth, kth, pivots, ref npiv);
                    return;
                }
                depthLimit--;

                long p = PartitionBlock(v, idx, low, high);
                if (p != kth) StorePivot(p, kth, pivots, ref npiv);
                if (p >= kth) high = p - 1;
                if (p <= kth) low = p + 1;
            }
            StorePivot(kth, kth, pivots, ref npiv);
        }

        /// <summary>
        ///     Index-tracking median-of-3 setup (mirror of
        ///     <see cref="Median3Swap{T}(T*, long, long, long)"/>): moves the median of
        ///     <c>{low, mid, high}</c> to <paramref name="low"/> and the smallest to <c>low+1</c>,
        ///     carrying <paramref name="idx"/> through each swap.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="v">The line's values.</param>
        /// <param name="idx">The original-index column.</param>
        /// <param name="low">The window's first position.</param>
        /// <param name="mid">The window's middle.</param>
        /// <param name="high">The window's last position.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void Median3Swap<T>(T* v, long* idx, long low, long mid, long high)
            where T : unmanaged, IComparable<T>
        {
            if (LtV(v[high], v[mid])) Swap(v, idx, high, mid);
            if (LtV(v[high], v[low])) Swap(v, idx, high, low);
            if (LtV(v[low], v[mid])) Swap(v, idx, low, mid);
            Swap(v, idx, mid, low + 1);
        }

        /// <summary>
        ///     Index-tracking scalar unguarded Hoare crossing (mirror of
        ///     <see cref="ScalarPartitionFinish{T}(T*, T*, long, long)"/>). Compares against the pivot
        ///     address (never a by-value copy) and carries <paramref name="idx"/> through each swap.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="v">The line's values.</param>
        /// <param name="idx">The original-index column.</param>
        /// <param name="pivot">The pivot's address.</param>
        /// <param name="ll">The position before the left scan's first look.</param>
        /// <param name="hh">The position after the right scan's first look.</param>
        /// <returns>The right scan's last stop, the pivot's final position.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe long ScalarPartitionFinish<T>(T* v, long* idx, T* pivot, long ll, long hh)
            where T : unmanaged, IComparable<T>
        {
            for (;;)
            {
                do { ll++; } while (LtPtr(v + ll, pivot));
                do { hh--; } while (LtPtr(pivot, v + hh));
                if (hh < ll) break;
                Swap(v, idx, ll, hh);
            }
            return hh;
        }

        /// <summary>
        ///     Index-tracking branchless block-Hoare partition (mirror of
        ///     <see cref="PartitionBlock{T}(T*, long, long)"/>): identical offset-block scan and swaps,
        ///     with <paramref name="idx"/> carried through every <see cref="Swap{T}(T*, long*, long, long)"/>.
        ///     The pivot stays at <c>v[low]</c> for the whole partition (the scan begins at
        ///     <c>low+2</c>, so <paramref name="idx"/> at <c>low</c> is untouched until the final
        ///     placement swap), keeping the compares against its hoisted address in registers.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="v">The line's values.</param>
        /// <param name="idx">The original-index column.</param>
        /// <param name="low">The window's first position.</param>
        /// <param name="high">The window's last position.</param>
        /// <returns>The pivot's final position.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe long PartitionBlock<T>(T* v, long* idx, long low, long high)
            where T : unmanaged, IComparable<T>
        {
            long mid = low + ((high - low) >> 1);
            Median3Swap(v, idx, low, mid, high);

            byte* offL = stackalloc byte[BlockSize];
            byte* offR = stackalloc byte[BlockSize];
            T* pivot = v + low;
            long ll = low + 2, hh = high - 1;
            int numL = 0, numR = 0, startL = 0, startR = 0;
            long baseL = ll, baseR = hh;

            while (hh - ll + 1 > 2 * BlockSize)
            {
                if (numL == 0)
                {
                    startL = 0; baseL = ll;
                    for (int i = 0; i < BlockSize; i++) { offL[numL] = (byte)i; numL += LtPtr(v + ll + i, pivot) ? 0 : 1; }
                }
                if (numR == 0)
                {
                    startR = 0; baseR = hh;
                    for (int i = 0; i < BlockSize; i++) { offR[numR] = (byte)i; numR += LtPtr(pivot, v + hh - i) ? 0 : 1; }
                }
                int num = Math.Min(numL, numR);
                for (int i = 0; i < num; i++) Swap(v, idx, baseL + offL[startL + i], baseR - offR[startR + i]);
                numL -= num; numR -= num; startL += num; startR += num;
                if (numL == 0) ll += BlockSize;
                if (numR == 0) hh -= BlockSize;
            }

            while (numL > 0)
            {
                while (LtPtr(pivot, v + hh)) hh--;
                long li = baseL + offL[startL + numL - 1];
                if (li >= hh) break;
                Swap(v, idx, li, hh);
                hh--; numL--;
            }
            while (numR > 0)
            {
                while (LtPtr(v + ll, pivot)) ll++;
                long ri = baseR - offR[startR + numR - 1];
                if (ri <= ll) break;
                Swap(v, idx, ri, ll);
                ll++; numR--;
            }

            long p = ScalarPartitionFinish(v, idx, pivot, ll - 1, hh + 1);
            Swap(v, idx, low, p);   // move pivot from low into its final crossing position
            return p;
        }
    }
}
