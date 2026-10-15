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
    ///
    /// <b>64-bit line lengths.</b> <paramref name="n"/> and every element position are <see cref="long"/>,
    /// so a single line may exceed <see cref="int.MaxValue"/> (the caller supplies unmanaged key/tmp/idx
    /// scratch, which is not bound by the ~2^31 managed-array element cap). The histograms are
    /// <c>long*</c> because ONE byte-value bucket can hold up to <paramref name="n"/> counts, which can
    /// itself exceed int range. The digit index, byte-pass counter and shift stay <see cref="int"/>
    /// (0-255, 0-8, 0-56). The insertion-sort fallbacks stay int-indexed — they run only for
    /// <c>n &lt;= InsertionThreshold</c> (≤120), far below int range.
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
        internal static uint* SortU32(uint* keys, uint* tmp, long n, int nbytes, long* hist)
        {
            if (n <= 1) return keys; // 0 or 1 element: already sorted (and guards null fixed-ptr on n==0)
            if (n <= InsertionThreshold32) { InsertionU32(keys, (int)n); return keys; }
            BuildHist32(keys, n, nbytes, hist);   // one read pass builds every byte's histogram
            uint* src = keys, dst = tmp;
            for (int shift = 0, pass = 0; pass < nbytes; pass++, shift += 8)
            {
                long* h = hist + (long)pass * 256;
                // trivial-pass skip: a byte whose value is uniform across all keys needs no scatter.
                // (The histograms are order-invariant, so this stays correct after earlier swaps.)
                if (h[(int)((src[0] >> shift) & 0xFF)] == n)
                    continue;
                Prefix(h);
                for (long i = 0; i < n; i++)
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
                                         long n, int nbytes, long* hist)
        {
            if (n <= 1) return idx;
            if (n <= InsertionThreshold32) { ArgInsertionU32(keys, idx, (int)n); return idx; }
            BuildHist32(keys, n, nbytes, hist);
            uint* ks = keys, kd = keyTmp;
            long* xs = idx, xd = idxTmp;
            for (int shift = 0, pass = 0; pass < nbytes; pass++, shift += 8)
            {
                long* h = hist + (long)pass * 256;
                if (h[(int)((ks[0] >> shift) & 0xFF)] == n)
                    continue;
                Prefix(h);
                for (long i = 0; i < n; i++)
                {
                    int d = (int)((ks[i] >> shift) & 0xFF);
                    long p = h[d]++;
                    kd[p] = ks[i];
                    xd[p] = xs[i];
                }
                uint* tk = ks; ks = kd; kd = tk;
                long* tx = xs; xs = xd; xd = tx;
            }
            return xs;
        }

        /// <summary>
        /// Keyed variant of <see cref="ArgSortU32"/>: co-sorts <paramref name="idx"/> by
        /// <paramref name="keys"/> AND exposes the buffer holding the sorted KEYS via
        /// <paramref name="sortedKeys"/> (either <paramref name="keys"/> or <paramref name="keyTmp"/>,
        /// whichever the double-buffer parity landed on — the trivial-pass skip makes it unpredictable,
        /// so it is tracked, not computed). Used by <c>np.unique</c>'s perm path so the sorted values
        /// come back through the monotonic key's inverse transform with NO gather (the sorted key
        /// column is already in memory-order), while the sorted index column supplies the permutation.
        /// </summary>
        internal static long* ArgSortU32(uint* keys, uint* keyTmp, long* idx, long* idxTmp,
                                         long n, int nbytes, long* hist, out uint* sortedKeys)
        {
            if (n <= 1) { sortedKeys = keys; return idx; }
            if (n <= InsertionThreshold32) { ArgInsertionU32(keys, idx, (int)n); sortedKeys = keys; return idx; }
            BuildHist32(keys, n, nbytes, hist);
            uint* ks = keys, kd = keyTmp;
            long* xs = idx, xd = idxTmp;
            for (int shift = 0, pass = 0; pass < nbytes; pass++, shift += 8)
            {
                long* h = hist + (long)pass * 256;
                if (h[(int)((ks[0] >> shift) & 0xFF)] == n)
                    continue;
                Prefix(h);
                for (long i = 0; i < n; i++)
                {
                    int d = (int)((ks[i] >> shift) & 0xFF);
                    long p = h[d]++;
                    kd[p] = ks[i];
                    xd[p] = xs[i];
                }
                uint* tk = ks; ks = kd; kd = tk;
                long* tx = xs; xs = xd; xd = tx;
            }
            sortedKeys = ks;
            return xs;
        }

        internal static ulong* SortU64(ulong* keys, ulong* tmp, long n, long* hist)
        {
            if (n <= 1) return keys;
            if (n <= InsertionThreshold64) { InsertionU64(keys, (int)n); return keys; }
            BuildHist64(keys, n, hist);   // one read pass builds all 8 byte histograms
            ulong* src = keys, dst = tmp;
            for (int shift = 0, pass = 0; pass < 8; pass++, shift += 8)
            {
                long* h = hist + (long)pass * 256;
                if (h[(int)((src[0] >> shift) & 0xFF)] == n)
                    continue;
                Prefix(h);
                for (long i = 0; i < n; i++)
                {
                    int d = (int)((src[i] >> shift) & 0xFF);
                    dst[h[d]++] = src[i];
                }
                ulong* t = src; src = dst; dst = t;
            }
            return src;
        }

        internal static long* ArgSortU64(ulong* keys, ulong* keyTmp, long* idx, long* idxTmp,
                                         long n, long* hist)
        {
            if (n <= 1) return idx;
            if (n <= InsertionThreshold64) { ArgInsertionU64(keys, idx, (int)n); return idx; }
            BuildHist64(keys, n, hist);
            ulong* ks = keys, kd = keyTmp;
            long* xs = idx, xd = idxTmp;
            for (int shift = 0, pass = 0; pass < 8; pass++, shift += 8)
            {
                long* h = hist + (long)pass * 256;
                if (h[(int)((ks[0] >> shift) & 0xFF)] == n)
                    continue;
                Prefix(h);
                for (long i = 0; i < n; i++)
                {
                    int d = (int)((ks[i] >> shift) & 0xFF);
                    long p = h[d]++;
                    kd[p] = ks[i];
                    xd[p] = xs[i];
                }
                ulong* tk = ks; ks = kd; kd = tk;
                long* tx = xs; xs = xd; xd = tx;
            }
            return xs;
        }

        /// <summary>8-byte twin of the keyed <see cref="ArgSortU32(uint*,uint*,long*,long*,long,int,long*,out uint*)"/>:
        /// exposes the sorted-key buffer alongside the sorted index column (see that overload).</summary>
        internal static long* ArgSortU64(ulong* keys, ulong* keyTmp, long* idx, long* idxTmp,
                                         long n, long* hist, out ulong* sortedKeys)
        {
            if (n <= 1) { sortedKeys = keys; return idx; }
            if (n <= InsertionThreshold64) { ArgInsertionU64(keys, idx, (int)n); sortedKeys = keys; return idx; }
            BuildHist64(keys, n, hist);
            ulong* ks = keys, kd = keyTmp;
            long* xs = idx, xd = idxTmp;
            for (int shift = 0, pass = 0; pass < 8; pass++, shift += 8)
            {
                long* h = hist + (long)pass * 256;
                if (h[(int)((ks[0] >> shift) & 0xFF)] == n)
                    continue;
                Prefix(h);
                for (long i = 0; i < n; i++)
                {
                    int d = (int)((ks[i] >> shift) & 0xFF);
                    long p = h[d]++;
                    kd[p] = ks[i];
                    xd[p] = xs[i];
                }
                ulong* tk = ks; ks = kd; kd = tk;
                long* tx = xs; xs = xd; xd = tx;
            }
            sortedKeys = ks;
            return xs;
        }

        /// <summary>
        ///     The keyed 8-byte argsort when the caller knows which key bits differ between the keys
        ///     (<paramref name="varyingBits"/> = OR of the keys XOR AND of the keys): only the bytes that hold a varying bit
        ///     get a histogram and a scatter pass. A byte in which every key agrees cannot reorder anything, and its
        ///     histogram would still cost one increment per key — for text keys, whose short strings leave whole bytes zero
        ///     (a one-character <c>U1</c> key varies in one byte of eight), that is most of the counting work.
        /// </summary>
        /// <param name="keys">The keys (the first buffer of the ping-pong pair).</param>
        /// <param name="keyTmp">The second key buffer, <paramref name="n"/> slots.</param>
        /// <param name="idx">The index column, ordered with the keys (element i at slot i on entry).</param>
        /// <param name="idxTmp">The second index buffer.</param>
        /// <param name="n">The number of keys.</param>
        /// <param name="hist">Histogram scratch of at least 8 · 256 longs.</param>
        /// <param name="varyingBits">The bits in which some two keys differ; a bit outside it is the same in every key.</param>
        /// <param name="sortedKeys">Receives the buffer that holds the sorted keys.</param>
        /// <returns>The buffer that holds the index column in sorted order (stable: equal keys keep their order).</returns>
        /// <remarks>Gives the same order as <see cref="ArgSortU64(ulong*, ulong*, long*, long*, long, long*, out ulong*)"/>:
        ///     that overload skips a byte whose histogram puts every key in one bucket, which is exactly a byte with no
        ///     varying bit, so both run the same scatter passes in the same order.</remarks>
        internal static long* ArgSortU64(ulong* keys, ulong* keyTmp, long* idx, long* idxTmp,
                                         long n, long* hist, ulong varyingBits, out ulong* sortedKeys)
        {
            if (n <= 1 || varyingBits == 0) { sortedKeys = keys; return idx; }
            if (n <= InsertionThresholdForBytes(varyingBits))
            {
                ArgInsertionU64(keys, idx, (int)n);
                sortedKeys = keys;
                return idx;
            }
            return ArgSortU64Radix(keys, keyTmp, idx, idxTmp, n, hist, varyingBits, out sortedKeys);
        }

        /// <summary>
        ///     The longest line the varying-bits argsort orders by insertion: the radix costs a fixed histogram setup plus a
        ///     pass per VARYING byte, so with few of them it overtakes insertion on much shorter lines than the 120 elements
        ///     of eight varying bytes. Measured over fresh random lines (median µs, insertion / radix): one varying byte
        ///     0.8 / 0.5 at 48 elements and 4.0 / 1.0 at 100; two, 2.3 / 1.4 at 64; eight, 6.7 / 6.5 at 120.
        /// </summary>
        /// <param name="varyingBits">The bits in which some two keys differ.</param>
        /// <returns>The longest line insertion keeps.</returns>
        private static int InsertionThresholdForBytes(ulong varyingBits)
        {
            int bytes = 0;
            for (int b = 0; b < 8; b++)
                if (((varyingBits >> (8 * b)) & 0xFF) != 0)
                    bytes++;
            return bytes switch
            {
                1 => 40,
                2 => 56,
                3 => 90,
                _ => InsertionThreshold64,
            };
        }

        /// <summary>
        ///     The radix passes of <see cref="ArgSortU64(ulong*, ulong*, long*, long*, long, long*, ulong, out ulong*)"/>
        ///     whatever the line's length (no insertion fallback): a counting pass and a stable scatter per varying byte.
        /// </summary>
        /// <param name="keys">The keys (the first buffer of the ping-pong pair).</param>
        /// <param name="keyTmp">The second key buffer, <paramref name="n"/> slots.</param>
        /// <param name="idx">The index column, ordered with the keys.</param>
        /// <param name="idxTmp">The second index buffer.</param>
        /// <param name="n">The number of keys (at least 2).</param>
        /// <param name="hist">Histogram scratch of at least 8 · 256 longs.</param>
        /// <param name="varyingBits">The bits in which some two keys differ (not 0).</param>
        /// <param name="sortedKeys">Receives the buffer that holds the sorted keys.</param>
        /// <returns>The buffer that holds the index column in sorted order (stable).</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static long* ArgSortU64Radix(ulong* keys, ulong* keyTmp, long* idx, long* idxTmp,
                                              long n, long* hist, ulong varyingBits, out ulong* sortedKeys)
        {
            // The varying bytes, lowest first (the LSD order).
            int* shifts = stackalloc int[8];
            int passes = 0;
            for (int b = 0; b < 8; b++)
                if (((varyingBits >> (8 * b)) & 0xFF) != 0)
                    shifts[passes++] = 8 * b;

            // Their histograms in one read pass. One or two varying bytes count only those (a U1 key: one increment per key
            // instead of eight); three or more take the unrolled all-bytes pass, whose eight fixed increments run faster
            // than a loop over a variable number of bytes (measured: 1,000 keys of eight varying bytes, 21.9 against
            // 18.6 µs through a variable inner loop). histBase[p] is pass p's histogram either way.
            long** histBase = stackalloc long*[8];
            if (passes <= 2)
            {
                for (int b = 0; b < passes * 256; b++)
                    hist[b] = 0;
                for (int p = 0; p < passes; p++)
                    histBase[p] = hist + p * 256;
                int s0 = shifts[0];
                if (passes == 1)
                {
                    for (long i = 0; i < n; i++)
                        hist[(int)((keys[i] >> s0) & 0xFF)]++;
                }
                else
                {
                    int s1 = shifts[1];
                    for (long i = 0; i < n; i++)
                    {
                        ulong v = keys[i];
                        hist[(int)((v >> s0) & 0xFF)]++;
                        hist[256 + (int)((v >> s1) & 0xFF)]++;
                    }
                }
            }
            else
            {
                BuildHist64(keys, n, hist);
                for (int p = 0; p < passes; p++)
                    histBase[p] = hist + (shifts[p] >> 3) * 256;
            }

            ulong* ks = keys, kd = keyTmp;
            long* xs = idx, xd = idxTmp;
            for (int pass = 0; pass < passes; pass++)
            {
                int shift = shifts[pass];
                long* h = histBase[pass];
                Prefix(h);
                for (long i = 0; i < n; i++)
                {
                    int d = (int)((ks[i] >> shift) & 0xFF);
                    long p = h[d]++;
                    kd[p] = ks[i];
                    xd[p] = xs[i];
                }
                ulong* tk = ks; ks = kd; kd = tk;
                long* tx = xs; xs = xd; xd = tx;
            }
            sortedKeys = ks;
            return xs;
        }

        /// <summary>
        /// Builds ALL <paramref name="nbytes"/> byte-histograms of <paramref name="src"/> in ONE
        /// read pass into <paramref name="hist"/> (<c>nbytes·256</c> longs, the b-th 256-block = byte
        /// b's counts). Standard LSD radix re-reads the key array once per byte-pass purely to count;
        /// since each byte's value distribution is invariant under reordering by the OTHER bytes, all
        /// counts can be gathered up front, dropping <c>nbytes-1</c> full read passes (the bulk of the
        /// non-scatter traffic). Measured: u32 ≈1.08×, u64 ≈1.17× over the per-pass count (the scatter,
        /// a random-access write, is the bandwidth floor SIMD/restructuring can't move). The 4-byte
        /// case is hand-unrolled so the four counter streams hit four distinct cache lines per key.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void BuildHist32(uint* src, long n, int nbytes, long* hist)
        {
            int total = nbytes * 256;
            for (int b = 0; b < total; b++) hist[b] = 0;
            if (nbytes == 4)
            {
                for (long i = 0; i < n; i++)
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
                for (long i = 0; i < n; i++)
                {
                    uint v = src[i];
                    for (int p = 0, sh = 0; p < nbytes; p++, sh += 8)
                        hist[p * 256 + (int)((v >> sh) & 0xFF)]++;
                }
            }
        }

        /// <summary>8-byte twin of <see cref="BuildHist32"/>: all 8 histograms in one read pass.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void BuildHist64(ulong* src, long n, long* hist)
        {
            for (int b = 0; b < 8 * 256; b++) hist[b] = 0;
            for (long i = 0; i < n; i++)
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
        private static void Prefix(long* count)
        {
            long sum = 0;
            for (int b = 0; b < 256; b++) { long c = count[b]; count[b] = sum; sum += c; }
        }
    }
}
