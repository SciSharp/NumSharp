using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Backends.Sorting
{
    /// <summary>
    /// Pure LSD radix sort over unsigned integer keys — the high-performance inner
    /// "sort function" that <see cref="DefaultEngine"/>'s axis-sort driver runs on each
    /// 1-D line (NumPy's npysort analog; the iterator drives the all-but-axis outer loop).
    ///
    /// Two native-width cores (no widening): <see cref="SortU32"/> for 1/2/4-byte dtypes,
    /// <see cref="SortU64"/> for 8-byte dtypes. 8-bit digits, one count + one scatter per
    /// byte, double-buffered. Stable — so the argsort variants are stable and the value sort
    /// preserves equal-key order (irrelevant for plain dtypes, load-bearing for argsort ties).
    ///
    /// Why radix (POC, i9-13900K, vs NumPy 2.4.2):
    ///   int32 sort  0.72-0.80x NumPy, 8-9x over BCL Span.Sort;
    ///   int32 argsort 2.1-4.3x FASTER than NumPy (NumPy uses indirect quicksort; radix
    ///   carries the index column for the cost of one extra scatter stream).
    ///
    /// The caller fills the key buffer with a MONOTONIC unsigned transform of the dtype
    /// (so unsigned ascending order == NumPy's value order) and un-transforms on the way out.
    /// NaN floats are partitioned out by the caller before keys are built (NumPy sorts NaN last).
    /// </summary>
    internal static unsafe class RadixSort
    {
        /// <summary>
        /// Sorts <paramref name="n"/> unsigned keys ascending. <paramref name="nbytes"/> is the
        /// number of low bytes that carry information (1, 2, or 4). Double-buffered between
        /// <paramref name="keys"/> and <paramref name="tmp"/>; returns the pointer to the buffer
        /// holding the sorted result (either <paramref name="keys"/> or <paramref name="tmp"/>).
        /// <paramref name="count"/> is a reusable scratch histogram of length 256.
        /// </summary>
        internal static uint* SortU32(uint* keys, uint* tmp, int n, int nbytes, int* count)
        {
            if (n <= 1) return keys; // 0 or 1 element: already sorted (and guards null fixed-ptr on n==0)
            uint* src = keys, dst = tmp;
            for (int shift = 0, pass = 0; pass < nbytes; pass++, shift += 8)
            {
                if (!Histogram32(src, n, shift, count))
                    continue; // all keys share this digit -> nothing to scatter
                Prefix(count);
                for (int i = 0; i < n; i++)
                {
                    int d = (int)((src[i] >> shift) & 0xFF);
                    dst[count[d]++] = src[i];
                }
                uint* t = src; src = dst; dst = t;
            }
            return src;
        }

        /// <summary>
        /// Stable argsort: co-sorts <paramref name="idx"/> by <paramref name="keys"/>.
        /// Returns the index buffer (<paramref name="idx"/> or <paramref name="idxTmp"/>) holding
        /// the result; <paramref name="keys"/>/<paramref name="keyTmp"/> are scratch.
        /// </summary>
        internal static long* ArgSortU32(uint* keys, uint* keyTmp, long* idx, long* idxTmp,
                                         int n, int nbytes, int* count)
        {
            if (n <= 1) return idx;
            uint* ks = keys, kd = keyTmp;
            long* xs = idx, xd = idxTmp;
            for (int shift = 0, pass = 0; pass < nbytes; pass++, shift += 8)
            {
                if (!Histogram32(ks, n, shift, count))
                    continue;
                Prefix(count);
                for (int i = 0; i < n; i++)
                {
                    int d = (int)((ks[i] >> shift) & 0xFF);
                    int p = count[d]++;
                    kd[p] = ks[i];
                    xd[p] = xs[i];
                }
                uint* tk = ks; ks = kd; kd = tk;
                long* tx = xs; xs = xd; xd = tx;
            }
            return xs;
        }

        internal static ulong* SortU64(ulong* keys, ulong* tmp, int n, int* count)
        {
            if (n <= 1) return keys;
            ulong* src = keys, dst = tmp;
            for (int shift = 0, pass = 0; pass < 8; pass++, shift += 8)
            {
                if (!Histogram64(src, n, shift, count))
                    continue;
                Prefix(count);
                for (int i = 0; i < n; i++)
                {
                    int d = (int)((src[i] >> shift) & 0xFF);
                    dst[count[d]++] = src[i];
                }
                ulong* t = src; src = dst; dst = t;
            }
            return src;
        }

        internal static long* ArgSortU64(ulong* keys, ulong* keyTmp, long* idx, long* idxTmp,
                                         int n, int* count)
        {
            if (n <= 1) return idx;
            ulong* ks = keys, kd = keyTmp;
            long* xs = idx, xd = idxTmp;
            for (int shift = 0, pass = 0; pass < 8; pass++, shift += 8)
            {
                if (!Histogram64(ks, n, shift, count))
                    continue;
                Prefix(count);
                for (int i = 0; i < n; i++)
                {
                    int d = (int)((ks[i] >> shift) & 0xFF);
                    int p = count[d]++;
                    kd[p] = ks[i];
                    xd[p] = xs[i];
                }
                ulong* tk = ks; ks = kd; kd = tk;
                long* tx = xs; xs = xd; xd = tx;
            }
            return xs;
        }

        /// <summary>Counts the 8-bit digit at <paramref name="shift"/>. Returns false (skip pass)
        /// when every key shares the same digit.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool Histogram32(uint* src, int n, int shift, int* count)
        {
            for (int b = 0; b < 256; b++) count[b] = 0;
            for (int i = 0; i < n; i++) count[(int)((src[i] >> shift) & 0xFF)]++;
            // trivial-pass detection: if one bucket holds all n, the scatter is a no-op copy.
            int first = (int)((src[0] >> shift) & 0xFF);
            return count[first] != n;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool Histogram64(ulong* src, int n, int shift, int* count)
        {
            for (int b = 0; b < 256; b++) count[b] = 0;
            for (int i = 0; i < n; i++) count[(int)((src[i] >> shift) & 0xFF)]++;
            int first = (int)((src[0] >> shift) & 0xFF);
            return count[first] != n;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Prefix(int* count)
        {
            int sum = 0;
            for (int b = 0; b < 256; b++) { int c = count[b]; count[b] = sum; sum += c; }
        }
    }
}
