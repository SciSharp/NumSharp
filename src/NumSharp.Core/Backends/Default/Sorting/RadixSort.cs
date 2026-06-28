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
        /// Below these line lengths, a binary-insertion sort beats LSD radix: radix pays a fixed
        /// 256-bucket histogram + prefix per byte-pass (≈ <c>nbytes·(512 + 2n)</c> ops + a cold
        /// 1 KB histogram sweep), which dwarfs the O(n²/4) of insertion for small n. NumPy makes the
        /// same call (quicksort falls to insertion below 16); radix's far larger constant pushes our
        /// crossover much higher, and it differs by key width — the 8-byte core runs twice the passes
        /// (8 vs 4), so insertion stays ahead to a longer line. Measured crossovers on an i9 (200k
        /// lines, random keys): 4-byte ≈ n80, 8-byte ≈ n120. Picking those exactly keeps each core
        /// out of its bad regime. This is THE fix for the short-line pathology (e.g. sort along the
        /// length-10 axis of a (1e6,10) array, where every one of a million lines paid the histogram
        /// tax: 624 ms → 72 ms).
        /// </summary>
        private const int InsertionThreshold32 = 80;
        private const int InsertionThreshold64 = 120;

        /// <summary>Stable in-place binary-insertion sort of <paramref name="n"/> u32 keys ascending.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void InsertionU32(uint* a, int n)
        {
            for (int i = 1; i < n; i++)
            {
                uint v = a[i];
                int j = i - 1;
                // strict '>' shift keeps equal keys in source order (stability is load-bearing for argsort ties)
                while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; }
                a[j + 1] = v;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void InsertionU64(ulong* a, int n)
        {
            for (int i = 1; i < n; i++)
            {
                ulong v = a[i];
                int j = i - 1;
                while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; }
                a[j + 1] = v;
            }
        }

        /// <summary>Stable insertion argsort: co-moves <paramref name="idx"/> while ordering by <paramref name="a"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void ArgInsertionU32(uint* a, long* idx, int n)
        {
            for (int i = 1; i < n; i++)
            {
                uint v = a[i]; long ix = idx[i];
                int j = i - 1;
                while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; idx[j + 1] = idx[j]; j--; }
                a[j + 1] = v; idx[j + 1] = ix;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void ArgInsertionU64(ulong* a, long* idx, int n)
        {
            for (int i = 1; i < n; i++)
            {
                ulong v = a[i]; long ix = idx[i];
                int j = i - 1;
                while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; idx[j + 1] = idx[j]; j--; }
                a[j + 1] = v; idx[j + 1] = ix;
            }
        }

        /// <summary>
        /// Sorts <paramref name="n"/> unsigned keys ascending. <paramref name="nbytes"/> is the
        /// number of low bytes that carry information (1, 2, or 4). Double-buffered between
        /// <paramref name="keys"/> and <paramref name="tmp"/>; returns the pointer to the buffer
        /// holding the sorted result (either <paramref name="keys"/> or <paramref name="tmp"/>).
        /// <paramref name="hist"/> is reusable scratch of length <c>nbytes·256</c> — ALL byte
        /// histograms, built in a SINGLE read pass (see <see cref="BuildHist32"/>).
        /// </summary>
        internal static uint* SortU32(uint* keys, uint* tmp, int n, int nbytes, int* hist)
        {
            if (n <= 1) return keys; // 0 or 1 element: already sorted (and guards null fixed-ptr on n==0)
            if (n <= InsertionThreshold32) { InsertionU32(keys, n); return keys; }
            BuildHist32(keys, n, nbytes, hist);   // one read pass builds every byte's histogram
            uint* src = keys, dst = tmp;
            for (int shift = 0, pass = 0; pass < nbytes; pass++, shift += 8)
            {
                int* h = hist + pass * 256;
                // trivial-pass skip: a byte whose value is uniform across all keys needs no scatter.
                // (The histograms are order-invariant, so this stays correct after earlier swaps.)
                if (h[(int)((src[0] >> shift) & 0xFF)] == n)
                    continue;
                Prefix(h);
                for (int i = 0; i < n; i++)
                {
                    int d = (int)((src[i] >> shift) & 0xFF);
                    dst[h[d]++] = src[i];
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
                                         int n, int nbytes, int* hist)
        {
            if (n <= 1) return idx;
            if (n <= InsertionThreshold32) { ArgInsertionU32(keys, idx, n); return idx; }
            BuildHist32(keys, n, nbytes, hist);
            uint* ks = keys, kd = keyTmp;
            long* xs = idx, xd = idxTmp;
            for (int shift = 0, pass = 0; pass < nbytes; pass++, shift += 8)
            {
                int* h = hist + pass * 256;
                if (h[(int)((ks[0] >> shift) & 0xFF)] == n)
                    continue;
                Prefix(h);
                for (int i = 0; i < n; i++)
                {
                    int d = (int)((ks[i] >> shift) & 0xFF);
                    int p = h[d]++;
                    kd[p] = ks[i];
                    xd[p] = xs[i];
                }
                uint* tk = ks; ks = kd; kd = tk;
                long* tx = xs; xs = xd; xd = tx;
            }
            return xs;
        }

        internal static ulong* SortU64(ulong* keys, ulong* tmp, int n, int* hist)
        {
            if (n <= 1) return keys;
            if (n <= InsertionThreshold64) { InsertionU64(keys, n); return keys; }
            BuildHist64(keys, n, hist);   // one read pass builds all 8 byte histograms
            ulong* src = keys, dst = tmp;
            for (int shift = 0, pass = 0; pass < 8; pass++, shift += 8)
            {
                int* h = hist + pass * 256;
                if (h[(int)((src[0] >> shift) & 0xFF)] == n)
                    continue;
                Prefix(h);
                for (int i = 0; i < n; i++)
                {
                    int d = (int)((src[i] >> shift) & 0xFF);
                    dst[h[d]++] = src[i];
                }
                ulong* t = src; src = dst; dst = t;
            }
            return src;
        }

        internal static long* ArgSortU64(ulong* keys, ulong* keyTmp, long* idx, long* idxTmp,
                                         int n, int* hist)
        {
            if (n <= 1) return idx;
            if (n <= InsertionThreshold64) { ArgInsertionU64(keys, idx, n); return idx; }
            BuildHist64(keys, n, hist);
            ulong* ks = keys, kd = keyTmp;
            long* xs = idx, xd = idxTmp;
            for (int shift = 0, pass = 0; pass < 8; pass++, shift += 8)
            {
                int* h = hist + pass * 256;
                if (h[(int)((ks[0] >> shift) & 0xFF)] == n)
                    continue;
                Prefix(h);
                for (int i = 0; i < n; i++)
                {
                    int d = (int)((ks[i] >> shift) & 0xFF);
                    int p = h[d]++;
                    kd[p] = ks[i];
                    xd[p] = xs[i];
                }
                ulong* tk = ks; ks = kd; kd = tk;
                long* tx = xs; xs = xd; xd = tx;
            }
            return xs;
        }

        /// <summary>
        /// Builds ALL <paramref name="nbytes"/> byte-histograms of <paramref name="src"/> in ONE
        /// read pass into <paramref name="hist"/> (<c>nbytes·256</c> ints, the b-th 256-block = byte
        /// b's counts). Standard LSD radix re-reads the key array once per byte-pass purely to count;
        /// since each byte's value distribution is invariant under reordering by the OTHER bytes, all
        /// counts can be gathered up front, dropping <c>nbytes-1</c> full read passes (the bulk of the
        /// non-scatter traffic). Measured: u32 ≈1.08×, u64 ≈1.17× over the per-pass count (the scatter,
        /// a random-access write, is the bandwidth floor SIMD/restructuring can't move). The 4-byte
        /// case is hand-unrolled so the four counter streams hit four distinct cache lines per key.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void BuildHist32(uint* src, int n, int nbytes, int* hist)
        {
            int total = nbytes * 256;
            for (int b = 0; b < total; b++) hist[b] = 0;
            if (nbytes == 4)
            {
                for (int i = 0; i < n; i++)
                {
                    uint v = src[i];
                    hist[(int)(v & 0xFF)]++;
                    hist[256 + (int)((v >> 8) & 0xFF)]++;
                    hist[512 + (int)((v >> 16) & 0xFF)]++;
                    hist[768 + (int)((v >> 24) & 0xFF)]++;
                }
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    uint v = src[i];
                    for (int p = 0, sh = 0; p < nbytes; p++, sh += 8)
                        hist[p * 256 + (int)((v >> sh) & 0xFF)]++;
                }
            }
        }

        /// <summary>8-byte twin of <see cref="BuildHist32"/>: all 8 histograms in one read pass.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void BuildHist64(ulong* src, int n, int* hist)
        {
            for (int b = 0; b < 8 * 256; b++) hist[b] = 0;
            for (int i = 0; i < n; i++)
            {
                ulong v = src[i];
                hist[(int)(v & 0xFF)]++;
                hist[256 + (int)((v >> 8) & 0xFF)]++;
                hist[512 + (int)((v >> 16) & 0xFF)]++;
                hist[768 + (int)((v >> 24) & 0xFF)]++;
                hist[1024 + (int)((v >> 32) & 0xFF)]++;
                hist[1280 + (int)((v >> 40) & 0xFF)]++;
                hist[1536 + (int)((v >> 48) & 0xFF)]++;
                hist[1792 + (int)((v >> 56) & 0xFF)]++;
            }
        }

        // (legacy per-pass Histogram32/Histogram64 removed — superseded by the single-pass BuildHist above)

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Prefix(int* count)
        {
            int sum = 0;
            for (int b = 0; b < 256; b++) { int c = count[b]; count[b] = sum; sum += c; }
        }
    }
}
