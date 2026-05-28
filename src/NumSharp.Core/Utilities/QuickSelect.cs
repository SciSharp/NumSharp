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
        public static unsafe void PartitionAtMany<T>(T* buf, int n, int* sortedKs, int nKs)
            where T : unmanaged, IComparable<T>
        {
            if (nKs == 0) return;
            int lo = 0;
            int hi = n - 1;
            for (int i = 0; i < nKs; i++)
            {
                int k = sortedKs[i];
                if (k < lo || k > hi) continue;
                IntroSelect(buf, lo, hi, k, 2 * Log2(hi - lo + 1));
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
                while (buf[++left].CompareTo(pivot) < 0) { }
                while (pivot.CompareTo(buf[--right]) < 0) { }
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
                while (j >= lo && t.CompareTo(buf[j]) < 0)
                {
                    buf[j + 1] = buf[j];
                    j--;
                }
                buf[j + 1] = t;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void SwapIfGreater<T>(T* buf, int i, int j) where T : unmanaged, IComparable<T>
        {
            if (buf[i].CompareTo(buf[j]) > 0) Swap(buf, i, j);
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
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
                if (child < n && buf[lo + child - 1].CompareTo(buf[lo + child]) < 0) child++;
                if (d.CompareTo(buf[lo + child - 1]) >= 0) break;
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

        // ── shared ────────────────────────────────────────────────────────────────

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void Swap<T>(T* buf, int i, int j) where T : unmanaged
        {
            T t = buf[i];
            buf[i] = buf[j];
            buf[j] = t;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int Log2(int v)
        {
            int r = 0;
            while (v > 0) { r++; v >>= 1; }
            return r;
        }
    }
}
