using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    ///     IntroSelect (QuickSelect + HeapSelect fallback) — places the k-th smallest element at
    ///     index k with everything left of it ≤ pivot and everything right ≥ pivot.
    ///     Mirrors NumPy's <c>np.partition</c> primitive, which backs <c>np.median</c> /
    ///     <c>np.percentile</c> hot paths: O(n) average / O(n log n) worst-case vs the
    ///     O(n log n) of a full sort.
    /// </summary>
    /// <remarks>
    ///     The multi-pivot overload partitions around an entire sorted list of k-values in
    ///     one pass. After <c>PartitionAt(buf, n, [k0, k1, k2])</c> each <c>buf[k_i]</c> is
    ///     in its final sorted position; adjacent ranges are mutually ordered. Net cost is
    ///     roughly O(n + k·n) average — far better than O(n log n) for small k.
    /// </remarks>
    internal static class QuickSelect
    {
        // ── IComparable<T> path (used for int dtypes + ones where NaN is impossible) ──

        public static unsafe void PartitionAt<T>(T* buf, int n, int k) where T : unmanaged, IComparable<T>
        {
            if (n <= 1 || k < 0 || k >= n) return;
            IntroSelect(buf, 0, n - 1, k, 2 * Log2(n));
        }

        public static unsafe void PartitionAt<T>(T* buf, int n, int[] sortedKs) where T : unmanaged, IComparable<T>
        {
            if (sortedKs.Length == 0) return;
            fixed (int* p = sortedKs) PartitionAtMany(buf, n, p, sortedKs.Length);
        }

        /// <summary>
        ///     Pointer-+-length variant suitable for IL-emitted callers that prefer to avoid
        ///     managed-array allocation per row. <paramref name="sortedKs"/> must already be
        ///     sorted ascending and within <c>[0, n-1]</c>.
        /// </summary>
        /// <remarks>
        ///     Port of NumPy's <c>introselect_</c> multi-select driver (<c>selection.cpp</c>):
        ///     a <b>pivot stack</b> (<see cref="StorePivot"/>) records every placed pivot ≥ the
        ///     current kth, so a subsequent (usually adjacent) kth — the <c>prev,next</c> pairs
        ///     that <c>np.percentile</c>/<c>np.median</c> partition around — narrows from
        ///     <b>both</b> ends instead of re-selecting the min of a half-array each time. The
        ///     inner partition is a branchless <b>block-Hoare</b> pass
        ///     (<see cref="PartitionBlock{T}"/>): the outer regions are scanned into fixed
        ///     offset blocks with no data-dependent branch, so random-data partitions no longer
        ///     pay ~50% branch-mispredict — the dominant cost of a scalar Hoare loop at scale.
        ///     Together these are ~3–4.4× the previous scalar/narrow-lo path on 10M rows
        ///     (single median, even median, and multi-quantile alike), and back the whole
        ///     <c>median</c>/<c>percentile</c>/<c>quantile</c>/<c>nan*</c> family plus
        ///     <c>np.partition</c>'s value path.
        /// </remarks>
        [SkipLocalsInit]   // the pivot stack is written before read (StorePivot); skipping the
                           // per-call zero-init of the stackalloc matters for many-tiny-rows axis
                           // reductions, where each short row would otherwise pay to zero 256 bytes.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]   // tier-1 from the first call so the
                           // per-row InsertionSort/IntroSelectBlock inline — the hot axis-reduce loop.
        public static unsafe void PartitionAtMany<T>(T* buf, int n, int* sortedKs, int nKs)
            where T : unmanaged, IComparable<T>
        {
            if (nKs == 0 || n <= 1) return;
            int* pivots = stackalloc int[PivotStackMax];
            int npiv = 0;
            for (int i = 0; i < nKs; i++)
            {
                int k = sortedKs[i];
                if (k < 0 || k >= n) continue;
                IntroSelectBlock(buf, n, k, pivots, ref npiv);
            }
        }

        /// <summary>
        ///     Pointer-ks <see cref="Comparison{T}"/> variant of <see cref="PartitionAtMany{T}(T*, int, int*, int)"/>
        ///     (np.partition's Complex path, whose NumPy CDOUBLE_LT NaN ordering cannot ride the raw
        ///     <c>&lt;</c> fast path). <paramref name="sortedKs"/> must be sorted ascending.
        /// </summary>
        public static unsafe void PartitionAtMany<T>(T* buf, int n, int* sortedKs, int nKs, Comparison<T> cmp)
            where T : unmanaged
        {
            if (nKs == 0) return;
            int lo = 0;
            int hi = n - 1;
            for (int i = 0; i < nKs; i++)
            {
                int k = sortedKs[i];
                if (k < lo || k > hi) continue;
                IntroSelect(buf, lo, hi, k, 2 * Log2(hi - lo + 1), cmp);
                lo = k + 1;
            }
        }

        // ── index-tracking path (np.argpartition): a parallel long[] of original indices is
        //    swapped in lockstep with the values, exactly the way RadixSort.ArgSortU32/U64 move
        //    their idx column with the keys. The value array is scratch; the indices are the answer.

        /// <summary>
        ///     Index-tracking introselect: partitions <paramref name="buf"/> around every k in
        ///     <paramref name="sortedKs"/> (sorted ascending, within <c>[0, n-1]</c>) while moving
        ///     <paramref name="idx"/> — the elements' original indices — through the identical swaps.
        ///     After the call <c>idx[k]</c> indexes the k-th smallest source element (NumPy
        ///     <c>np.argpartition</c> semantics).
        /// </summary>
        public static unsafe void PartitionAtMany<T>(T* buf, long* idx, int n, int* sortedKs, int nKs)
            where T : unmanaged, IComparable<T>
        {
            if (nKs == 0) return;
            int lo = 0;
            int hi = n - 1;
            for (int i = 0; i < nKs; i++)
            {
                int k = sortedKs[i];
                if (k < lo || k > hi) continue;
                IntroSelect(buf, idx, lo, hi, k, 2 * Log2(hi - lo + 1));
                lo = k + 1;
            }
        }

        /// <summary>
        ///     Index-tracking introselect with a <see cref="Comparison{T}"/> (argpartition's Complex path).
        /// </summary>
        public static unsafe void PartitionAtMany<T>(T* buf, long* idx, int n, int* sortedKs, int nKs, Comparison<T> cmp)
            where T : unmanaged
        {
            if (nKs == 0) return;
            int lo = 0;
            int hi = n - 1;
            for (int i = 0; i < nKs; i++)
            {
                int k = sortedKs[i];
                if (k < lo || k > hi) continue;
                IntroSelect(buf, idx, lo, hi, k, 2 * Log2(hi - lo + 1), cmp);
                lo = k + 1;
            }
        }

        // ── Comparison<T> path (used for float/double with NaN-at-end semantics) ──

        public static unsafe void PartitionAt<T>(T* buf, int n, int k, Comparison<T> cmp) where T : unmanaged
        {
            if (n <= 1 || k < 0 || k >= n) return;
            IntroSelect(buf, 0, n - 1, k, 2 * Log2(n), cmp);
        }

        public static unsafe void PartitionAt<T>(T* buf, int n, int[] sortedKs, Comparison<T> cmp) where T : unmanaged
        {
            if (sortedKs.Length == 0) return;
            int lo = 0;
            int hi = n - 1;
            for (int i = 0; i < sortedKs.Length; i++)
            {
                int k = sortedKs[i];
                if (k < lo || k > hi) continue;
                IntroSelect(buf, lo, hi, k, 2 * Log2(hi - lo + 1), cmp);
                lo = k + 1;
            }
        }

        // ── IComparable internals ─────────────────────────────────────────────────

        private const int InsertionSortThreshold = 16;

        // ── pivot-stack block-select internals (np.introselect_ + branchless block partition) ──
        //
        // BlockSize is the offset-block width: each pass scans this many elements from each
        // outer edge into a byte[] of misplaced-element offsets, then swaps the paired offsets.
        // Kept ≤ 256 so an offset fits a byte. PivotStackMax mirrors NumPy's NPY_MAX_PIVOT_STACK.
        private const int BlockSize = 128;
        private const int PivotStackMax = 64;

        /// <summary>
        ///     Records a placed pivot for reuse by later kths (NumPy <c>store_pivot</c>). Only
        ///     pivots at or above the current kth are useful as upper bounds, so smaller ones are
        ///     never pushed; the stack therefore stays descending and its top is the tightest
        ///     upper bound for the next (larger) kth.
        /// </summary>
        private static unsafe void StorePivot(int pivot, int kth, int* pivots, ref int npiv)
        {
            if (pivot == kth && npiv == PivotStackMax) pivots[npiv - 1] = pivot;
            else if (pivot >= kth && npiv < PivotStackMax) { pivots[npiv] = pivot; npiv++; }
        }

        /// <summary>
        ///     Single-kth introselect threading the shared pivot stack (NumPy <c>introselect_</c>).
        ///     Pops the stack to narrow <c>[low, high]</c> before selecting, then median-of-3 +
        ///     branchless block partition down to the insertion-sort / heap-sort thresholds,
        ///     pushing every placed pivot back so the next kth reuses this work.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void IntroSelectBlock<T>(T* v, int n, int kth, int* pivots, ref int npiv)
            where T : unmanaged, IComparable<T>
        {
            int low = 0, high = n - 1;
            // Narrow from already-placed pivots: pop everything ≤ kth (raising low), stop at the
            // first pivot > kth (lowering high). An exact hit means kth is already in place.
            while (npiv > 0)
            {
                int top = pivots[npiv - 1];
                if (top > kth) { high = top - 1; break; }
                if (top == kth) return;
                low = top + 1;
                npiv--;
            }

            int depthLimit = 2 * Log2(n);
            while (low < high)
            {
                int len = high - low + 1;
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

                int p = PartitionBlock(v, low, high);
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void Median3Swap<T>(T* v, int low, int mid, int high)
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
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe int ScalarPartitionFinish<T>(T* v, T* pivot, int ll, int hh)
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
        ///     exist. Any NaNs are stripped by the quantile kernel before this runs, so the raw
        ///     <see cref="LtV{T}"/> ordering is total here.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe int PartitionBlock<T>(T* v, int low, int high)
            where T : unmanaged, IComparable<T>
        {
            int mid = low + ((high - low) >> 1);
            Median3Swap(v, low, mid, high);

            byte* offL = stackalloc byte[BlockSize];
            byte* offR = stackalloc byte[BlockSize];
            // The pivot lives at v[low] for the whole partition; comparing against its address
            // (rather than a by-value copy) is what keeps the block scan in registers.
            T* pivot = v + low;
            // v[low+1] ≤ pivot and v[high] ≥ pivot are the sentinels, so the unknown span is
            // (low+1, high): scan up from ll and down from hh.
            int ll = low + 2, hh = high - 1;
            int numL = 0, numR = 0, startL = 0, startR = 0, baseL = ll, baseR = hh;

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
                int li = baseL + offL[startL + numL - 1];
                if (li >= hh) break;
                Swap(v, li, hh);
                hh--; numL--;
            }
            while (numR > 0)
            {
                while (LtPtr(v + ll, pivot)) ll++;
                int ri = baseR - offR[startR + numR - 1];
                if (ri <= ll) break;
                Swap(v, ri, ll);
                ll++; numR--;
            }

            int p = ScalarPartitionFinish(v, pivot, ll - 1, hh + 1);
            Swap(v, low, p);   // move pivot from low into its final crossing position
            return p;
        }

        private static unsafe void IntroSelect<T>(T* buf, int lo, int hi, int k, int depthLimit)
            where T : unmanaged, IComparable<T>
        {
            while (lo < hi)
            {
                int len = hi - lo + 1;
                if (len <= InsertionSortThreshold)
                {
                    InsertionSort(buf, lo, hi);
                    return;
                }
                if (depthLimit == 0)
                {
                    // Recursion went too deep — fall back to heap-sort for O(n log n) worst case.
                    HeapSort(buf, lo, hi);
                    return;
                }
                depthLimit--;

                int p = Partition(buf, lo, hi);
                if (k == p) return;
                if (k < p) hi = p - 1;
                else lo = p + 1;
            }
        }

        private static unsafe int Partition<T>(T* buf, int lo, int hi) where T : unmanaged, IComparable<T>
        {
            int mid = lo + ((hi - lo) >> 1);
            SwapIfGreater(buf, lo, mid);
            SwapIfGreater(buf, lo, hi);
            SwapIfGreater(buf, mid, hi);

            T pivot = buf[mid];
            Swap(buf, mid, hi - 1);

            int left = lo;
            int right = hi - 1;
            while (left < right)
            {
                while (LtV(buf[++left], pivot)) { }
                while (LtV(pivot, buf[--right])) { }
                if (left >= right) break;
                Swap(buf, left, right);
            }
            if (left != hi - 1) Swap(buf, left, hi - 1);
            return left;
        }

        private static unsafe void InsertionSort<T>(T* buf, int lo, int hi) where T : unmanaged, IComparable<T>
        {
            for (int i = lo; i < hi; i++)
            {
                int j = i;
                T t = buf[i + 1];
                while (j >= lo && LtV(t, buf[j]))
                {
                    buf[j + 1] = buf[j];
                    j--;
                }
                buf[j + 1] = t;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SwapIfGreater<T>(T* buf, int i, int j) where T : unmanaged, IComparable<T>
        {
            if (LtV(buf[j], buf[i])) Swap(buf, i, j);
        }

        // ── Comparison<T> internals ───────────────────────────────────────────────

        private static unsafe void IntroSelect<T>(T* buf, int lo, int hi, int k, int depthLimit, Comparison<T> cmp)
            where T : unmanaged
        {
            while (lo < hi)
            {
                int len = hi - lo + 1;
                if (len <= InsertionSortThreshold)
                {
                    InsertionSort(buf, lo, hi, cmp);
                    return;
                }
                if (depthLimit == 0)
                {
                    HeapSort(buf, lo, hi, cmp);
                    return;
                }
                depthLimit--;

                int p = Partition(buf, lo, hi, cmp);
                if (k == p) return;
                if (k < p) hi = p - 1;
                else lo = p + 1;
            }
        }

        private static unsafe int Partition<T>(T* buf, int lo, int hi, Comparison<T> cmp) where T : unmanaged
        {
            int mid = lo + ((hi - lo) >> 1);
            SwapIfGreater(buf, lo, mid, cmp);
            SwapIfGreater(buf, lo, hi, cmp);
            SwapIfGreater(buf, mid, hi, cmp);

            T pivot = buf[mid];
            Swap(buf, mid, hi - 1);

            int left = lo;
            int right = hi - 1;
            while (left < right)
            {
                while (cmp(buf[++left], pivot) < 0) { }
                while (cmp(pivot, buf[--right]) < 0) { }
                if (left >= right) break;
                Swap(buf, left, right);
            }
            if (left != hi - 1) Swap(buf, left, hi - 1);
            return left;
        }

        private static unsafe void InsertionSort<T>(T* buf, int lo, int hi, Comparison<T> cmp) where T : unmanaged
        {
            for (int i = lo; i < hi; i++)
            {
                int j = i;
                T t = buf[i + 1];
                while (j >= lo && cmp(t, buf[j]) < 0)
                {
                    buf[j + 1] = buf[j];
                    j--;
                }
                buf[j + 1] = t;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SwapIfGreater<T>(T* buf, int i, int j, Comparison<T> cmp) where T : unmanaged
        {
            if (cmp(buf[i], buf[j]) > 0) Swap(buf, i, j);
        }

        // ── heap-sort fallback (used when introselect recurses too deep) ──────────

        private static unsafe void HeapSort<T>(T* buf, int lo, int hi) where T : unmanaged, IComparable<T>
        {
            int n = hi - lo + 1;
            for (int i = n >> 1; i >= 1; i--) DownHeap(buf, i, n, lo);
            for (int i = n; i > 1; i--) { Swap(buf, lo, lo + i - 1); DownHeap(buf, 1, i - 1, lo); }
        }

        private static unsafe void DownHeap<T>(T* buf, int i, int n, int lo) where T : unmanaged, IComparable<T>
        {
            T d = buf[lo + i - 1];
            while (i <= n >> 1)
            {
                int child = 2 * i;
                if (child < n && LtV(buf[lo + child - 1], buf[lo + child])) child++;
                if (!LtV(d, buf[lo + child - 1])) break;
                buf[lo + i - 1] = buf[lo + child - 1];
                i = child;
            }
            buf[lo + i - 1] = d;
        }

        private static unsafe void HeapSort<T>(T* buf, int lo, int hi, Comparison<T> cmp) where T : unmanaged
        {
            int n = hi - lo + 1;
            for (int i = n >> 1; i >= 1; i--) DownHeap(buf, i, n, lo, cmp);
            for (int i = n; i > 1; i--) { Swap(buf, lo, lo + i - 1); DownHeap(buf, 1, i - 1, lo, cmp); }
        }

        private static unsafe void DownHeap<T>(T* buf, int i, int n, int lo, Comparison<T> cmp) where T : unmanaged
        {
            T d = buf[lo + i - 1];
            while (i <= n >> 1)
            {
                int child = 2 * i;
                if (child < n && cmp(buf[lo + child - 1], buf[lo + child]) < 0) child++;
                if (cmp(d, buf[lo + child - 1]) >= 0) break;
                buf[lo + i - 1] = buf[lo + child - 1];
                i = child;
            }
            buf[lo + i - 1] = d;
        }

        // ── index-tracking internals (values + parallel idx column swap together) ──

        private static unsafe void IntroSelect<T>(T* buf, long* idx, int lo, int hi, int k, int depthLimit)
            where T : unmanaged, IComparable<T>
        {
            while (lo < hi)
            {
                int len = hi - lo + 1;
                if (len <= InsertionSortThreshold)
                {
                    InsertionSort(buf, idx, lo, hi);
                    return;
                }
                if (depthLimit == 0)
                {
                    HeapSort(buf, idx, lo, hi);
                    return;
                }
                depthLimit--;

                int p = Partition(buf, idx, lo, hi);
                if (k == p) return;
                if (k < p) hi = p - 1;
                else lo = p + 1;
            }
        }

        private static unsafe int Partition<T>(T* buf, long* idx, int lo, int hi) where T : unmanaged, IComparable<T>
        {
            int mid = lo + ((hi - lo) >> 1);
            SwapIfGreater(buf, idx, lo, mid);
            SwapIfGreater(buf, idx, lo, hi);
            SwapIfGreater(buf, idx, mid, hi);

            T pivot = buf[mid];
            Swap(buf, idx, mid, hi - 1);

            int left = lo;
            int right = hi - 1;
            while (left < right)
            {
                while (LtV(buf[++left], pivot)) { }
                while (LtV(pivot, buf[--right])) { }
                if (left >= right) break;
                Swap(buf, idx, left, right);
            }
            if (left != hi - 1) Swap(buf, idx, left, hi - 1);
            return left;
        }

        private static unsafe void InsertionSort<T>(T* buf, long* idx, int lo, int hi) where T : unmanaged, IComparable<T>
        {
            for (int i = lo; i < hi; i++)
            {
                int j = i;
                T t = buf[i + 1];
                long ti = idx[i + 1];
                while (j >= lo && LtV(t, buf[j]))
                {
                    buf[j + 1] = buf[j];
                    idx[j + 1] = idx[j];
                    j--;
                }
                buf[j + 1] = t;
                idx[j + 1] = ti;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SwapIfGreater<T>(T* buf, long* idx, int i, int j) where T : unmanaged, IComparable<T>
        {
            if (LtV(buf[j], buf[i])) Swap(buf, idx, i, j);
        }

        private static unsafe void HeapSort<T>(T* buf, long* idx, int lo, int hi) where T : unmanaged, IComparable<T>
        {
            int n = hi - lo + 1;
            for (int i = n >> 1; i >= 1; i--) DownHeap(buf, idx, i, n, lo);
            for (int i = n; i > 1; i--) { Swap(buf, idx, lo, lo + i - 1); DownHeap(buf, idx, 1, i - 1, lo); }
        }

        private static unsafe void DownHeap<T>(T* buf, long* idx, int i, int n, int lo) where T : unmanaged, IComparable<T>
        {
            T d = buf[lo + i - 1];
            long di = idx[lo + i - 1];
            while (i <= n >> 1)
            {
                int child = 2 * i;
                if (child < n && LtV(buf[lo + child - 1], buf[lo + child])) child++;
                if (!LtV(d, buf[lo + child - 1])) break;
                buf[lo + i - 1] = buf[lo + child - 1];
                idx[lo + i - 1] = idx[lo + child - 1];
                i = child;
            }
            buf[lo + i - 1] = d;
            idx[lo + i - 1] = di;
        }

        private static unsafe void IntroSelect<T>(T* buf, long* idx, int lo, int hi, int k, int depthLimit, Comparison<T> cmp)
            where T : unmanaged
        {
            while (lo < hi)
            {
                int len = hi - lo + 1;
                if (len <= InsertionSortThreshold)
                {
                    InsertionSort(buf, idx, lo, hi, cmp);
                    return;
                }
                if (depthLimit == 0)
                {
                    HeapSort(buf, idx, lo, hi, cmp);
                    return;
                }
                depthLimit--;

                int p = Partition(buf, idx, lo, hi, cmp);
                if (k == p) return;
                if (k < p) hi = p - 1;
                else lo = p + 1;
            }
        }

        private static unsafe int Partition<T>(T* buf, long* idx, int lo, int hi, Comparison<T> cmp) where T : unmanaged
        {
            int mid = lo + ((hi - lo) >> 1);
            SwapIfGreater(buf, idx, lo, mid, cmp);
            SwapIfGreater(buf, idx, lo, hi, cmp);
            SwapIfGreater(buf, idx, mid, hi, cmp);

            T pivot = buf[mid];
            Swap(buf, idx, mid, hi - 1);

            int left = lo;
            int right = hi - 1;
            while (left < right)
            {
                while (cmp(buf[++left], pivot) < 0) { }
                while (cmp(pivot, buf[--right]) < 0) { }
                if (left >= right) break;
                Swap(buf, idx, left, right);
            }
            if (left != hi - 1) Swap(buf, idx, left, hi - 1);
            return left;
        }

        private static unsafe void InsertionSort<T>(T* buf, long* idx, int lo, int hi, Comparison<T> cmp) where T : unmanaged
        {
            for (int i = lo; i < hi; i++)
            {
                int j = i;
                T t = buf[i + 1];
                long ti = idx[i + 1];
                while (j >= lo && cmp(t, buf[j]) < 0)
                {
                    buf[j + 1] = buf[j];
                    idx[j + 1] = idx[j];
                    j--;
                }
                buf[j + 1] = t;
                idx[j + 1] = ti;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SwapIfGreater<T>(T* buf, long* idx, int i, int j, Comparison<T> cmp) where T : unmanaged
        {
            if (cmp(buf[i], buf[j]) > 0) Swap(buf, idx, i, j);
        }

        private static unsafe void HeapSort<T>(T* buf, long* idx, int lo, int hi, Comparison<T> cmp) where T : unmanaged
        {
            int n = hi - lo + 1;
            for (int i = n >> 1; i >= 1; i--) DownHeap(buf, idx, i, n, lo, cmp);
            for (int i = n; i > 1; i--) { Swap(buf, idx, lo, lo + i - 1); DownHeap(buf, idx, 1, i - 1, lo, cmp); }
        }

        private static unsafe void DownHeap<T>(T* buf, long* idx, int i, int n, int lo, Comparison<T> cmp) where T : unmanaged
        {
            T d = buf[lo + i - 1];
            long di = idx[lo + i - 1];
            while (i <= n >> 1)
            {
                int child = 2 * i;
                if (child < n && cmp(buf[lo + child - 1], buf[lo + child]) < 0) child++;
                if (cmp(d, buf[lo + child - 1]) >= 0) break;
                buf[lo + i - 1] = buf[lo + child - 1];
                idx[lo + i - 1] = idx[lo + child - 1];
                i = child;
            }
            buf[lo + i - 1] = d;
            idx[lo + i - 1] = di;
        }

        // ── shared ────────────────────────────────────────────────────────────────

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void Swap<T>(T* buf, int i, int j) where T : unmanaged
        {
            T t = buf[i];
            buf[i] = buf[j];
            buf[j] = t;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void Swap<T>(T* buf, long* idx, int i, int j) where T : unmanaged
        {
            T t = buf[i];
            buf[i] = buf[j];
            buf[j] = t;
            long ti = idx[i];
            idx[i] = idx[j];
            idx[j] = ti;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static int Log2(int v)
        {
            int r = 0;
            while (v > 0) { r++; v >>= 1; }
            return r;
        }

        /// <summary>
        ///     Direct typed "a &lt; b" for the IComparable&lt;T&gt; partition path. The
        ///     <c>typeof(T) == typeof(X)</c> chain is JIT-folded per specialization, so each
        ///     instantiation compiles to a single native comparison — far cheaper than
        ///     <see cref="IComparable{T}.CompareTo"/>, which returns a tri-state int the
        ///     caller must then re-test. The quantile kernel strips NaNs before partitioning
        ///     (prescan on the plain path, compaction on the nan path), so IEEE NaN ordering
        ///     never reaches here and a raw <c>&lt;</c> is safe for floats.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe bool LtV<T>(T a, T b) where T : unmanaged, IComparable<T>
        {
            if (typeof(T) == typeof(byte))    return *(byte*)&a    < *(byte*)&b;
            if (typeof(T) == typeof(sbyte))   return *(sbyte*)&a   < *(sbyte*)&b;
            if (typeof(T) == typeof(short))   return *(short*)&a   < *(short*)&b;
            if (typeof(T) == typeof(ushort))  return *(ushort*)&a  < *(ushort*)&b;
            if (typeof(T) == typeof(int))     return *(int*)&a     < *(int*)&b;
            if (typeof(T) == typeof(uint))    return *(uint*)&a    < *(uint*)&b;
            if (typeof(T) == typeof(long))    return *(long*)&a    < *(long*)&b;
            if (typeof(T) == typeof(ulong))   return *(ulong*)&a   < *(ulong*)&b;
            if (typeof(T) == typeof(char))    return *(char*)&a    < *(char*)&b;
            if (typeof(T) == typeof(float))   return *(float*)&a   < *(float*)&b;
            if (typeof(T) == typeof(double))  return *(double*)&a  < *(double*)&b;
            if (typeof(T) == typeof(Half))    return *(Half*)&a    < *(Half*)&b;
            if (typeof(T) == typeof(decimal)) return *(decimal*)&a < *(decimal*)&b;
            if (typeof(T) == typeof(bool))    return !*(bool*)&a && *(bool*)&b;  // false < true
            return a.CompareTo(b) < 0;   // fallback for any other IComparable type
        }

        /// <summary>
        ///     Pointer form of <see cref="LtV{T}"/> — <c>*a &lt; *b</c> reading both operands in
        ///     place. The by-value <see cref="LtV{T}"/> takes <c>&amp;a</c> of its parameters,
        ///     which forces each operand to a stack slot; in the branchless block-partition scan
        ///     that is a per-element store+reload that erases the whole win. Passing the array
        ///     element pointer directly (and the pivot's hoisted address) keeps the compare in
        ///     registers, so the emitted <c>numX += (*a &lt; *b) ? 0 : 1</c> is a single load +
        ///     compare + branchless increment — matching a hand-written typed loop.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe bool LtPtr<T>(T* a, T* b) where T : unmanaged, IComparable<T>
        {
            if (typeof(T) == typeof(byte))    return *(byte*)a    < *(byte*)b;
            if (typeof(T) == typeof(sbyte))   return *(sbyte*)a   < *(sbyte*)b;
            if (typeof(T) == typeof(short))   return *(short*)a   < *(short*)b;
            if (typeof(T) == typeof(ushort))  return *(ushort*)a  < *(ushort*)b;
            if (typeof(T) == typeof(int))     return *(int*)a     < *(int*)b;
            if (typeof(T) == typeof(uint))    return *(uint*)a    < *(uint*)b;
            if (typeof(T) == typeof(long))    return *(long*)a    < *(long*)b;
            if (typeof(T) == typeof(ulong))   return *(ulong*)a   < *(ulong*)b;
            if (typeof(T) == typeof(char))    return *(char*)a    < *(char*)b;
            if (typeof(T) == typeof(float))   return *(float*)a   < *(float*)b;
            if (typeof(T) == typeof(double))  return *(double*)a  < *(double*)b;
            if (typeof(T) == typeof(Half))    return *(Half*)a    < *(Half*)b;
            if (typeof(T) == typeof(decimal)) return *(decimal*)a < *(decimal*)b;
            if (typeof(T) == typeof(bool))    return !*(bool*)a && *(bool*)b;
            return (*a).CompareTo(*b) < 0;
        }
    }
}
