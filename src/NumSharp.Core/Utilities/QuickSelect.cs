using System;
using System.Numerics;
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
    ///
    ///     <b>Three selects, by what NumPy runs.</b> Several kth are NumPy's <c>introselect_</c> on every host, and
    ///     <c>PartitionAtMany</c> / <c>PartitionAtManyNaN</c> / <c>PartitionAtManyComplex</c> run its statement-for-statement
    ///     port (<c>QuickSelect.Introselect.cs</c>), so equal values land where NumPy puts them (QB43). A single kth is
    ///     x86-simd-sort's in NumPy; <c>SelectSingleKth</c> keeps the select NumSharp always ran for it
    ///     (<c>QuickSelect.SingleKth.cs</c>), after the edge selection (<c>QuickSelect.EdgeSelect.cs</c>).
    ///     <c>PartitionAtManyTieBlind</c> is that select over a kth list, for the quantile kernels' rows whose result bits
    ///     cannot show the arrangement.
    ///
    ///     <b>64-bit line lengths.</b> <c>n</c>, every element position (lo/hi/left/right/mid/pivot),
    ///     the kth targets and the pivot stack are <see cref="long"/>, so a single line may exceed
    ///     <see cref="int.MaxValue"/> (the caller supplies unmanaged scratch, which is not bound by the
    ///     ~2^31 managed-array element cap). Block-local counters (numL/numR/startL/startR, the
    ///     0..BlockSize offset scan), the pivot-stack DEPTH (npiv), depthLimit and nKs stay
    ///     <see cref="int"/> — all bounded by small constants. A thin int-taking overload forwards the
    ///     &lt;=int.MaxValue quantile-kernel callers to the long core.
    /// </remarks>
    internal static partial class QuickSelect
    {
        // ── IComparable<T> path (used for int dtypes + ones where NaN is impossible) ──

        public static unsafe void PartitionAt<T>(T* buf, long n, long k) where T : unmanaged, IComparable<T>
        {
            if (n <= 1 || k < 0 || k >= n) return;
            IntroSelect(buf, 0, n - 1, k, 2 * Log2(n));
        }

        public static unsafe void PartitionAt<T>(T* buf, long n, long[] sortedKs) where T : unmanaged, IComparable<T>
        {
            if (sortedKs.Length == 0) return;
            fixed (long* p = sortedKs) PartitionAtMany(buf, n, p, sortedKs.Length);
        }

        /// <summary>
        ///     np.partition's value select of one NaN-free line around several positions: NumPy's <c>introselect_</c>
        ///     (<see cref="Introselect{T, TLess, TCol}"/>) once per kth, in the given order, sharing one pivot stack, so the
        ///     line ends in NumPy's arrangement, equal values (the zero signs of a ±0 tie) included.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="buf">The line, partitioned in place.</param>
        /// <param name="n">The line's length.</param>
        /// <param name="sortedKs">The positions, sorted ascending (repeats allowed: a repeat is a no-op, as in NumPy);
        ///     a position outside <c>[0, n)</c> is skipped.</param>
        /// <param name="nKs">How many positions.</param>
        /// <remarks>
        ///     FOOTGUN: the comparison is the type's raw <c>&lt;</c>, NumPy's <c>Tag::less</c> only when the line holds no
        ///     NaN; a float line that may hold one goes to <see cref="PartitionAtManyNaN{T}(T*, long, long*, int)"/>, which
        ///     partitions it in place as NumPy does. Long windows run the branchless block partition, whose swaps are
        ///     Hoare's (<see cref="BlockScan{T, TLess, TCol}"/>): 10M-element selects keep their speed.
        ///     <para>[SkipLocalsInit]: the pivot stack is written before it is read, and many-short-rows reductions would
        ///     otherwise zero it per row; [AggressiveOptimization]: tier-1 from the first call, so the per-row select inlines
        ///     into the hot axis-reduce loop.</para>
        /// </remarks>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void PartitionAtMany<T>(T* buf, long n, long* sortedKs, int nKs)
            where T : unmanaged, IComparable<T>
        {
            if (nKs == 0 || n <= 1) return;
            long* pivots = stackalloc long[PivotStackMax];
            int npiv = 0;
            for (int i = 0; i < nKs; i++)
            {
                long k = sortedKs[i];
                if (k < 0 || k >= n) continue;
                Introselect<T, RawLess<T>, ValueColumn<T>>(buf, default, n, k, pivots, &npiv);
            }
        }

        /// <summary>
        ///     np.partition's value select of a float line that may hold NaN, around several positions: NumPy's
        ///     <c>introselect_</c> under its NaN-aware comparison (<see cref="NaNLess{T}"/>: every NaN after every number),
        ///     in place, so the NaNs take part in the swaps exactly as in NumPy and end the line, each with its own bits.
        /// </summary>
        /// <typeparam name="T"><see cref="double"/>, <see cref="float"/> or <see cref="Half"/>.</typeparam>
        /// <param name="buf">The line, partitioned in place.</param>
        /// <param name="n">The line's length.</param>
        /// <param name="sortedKs">The positions, sorted ascending (repeats allowed); a position outside <c>[0, n)</c> is
        ///     skipped.</param>
        /// <param name="nKs">How many positions.</param>
        /// <remarks>Slower per comparison than the raw order, so a caller takes it only for a line it found a NaN in.
        ///     [SkipLocalsInit]: the pivot stack is written before it is read.</remarks>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void PartitionAtManyNaN<T>(T* buf, long n, long* sortedKs, int nKs)
            where T : unmanaged
        {
            if (nKs == 0 || n <= 1) return;
            long* pivots = stackalloc long[PivotStackMax];
            int npiv = 0;
            for (int i = 0; i < nKs; i++)
            {
                long k = sortedKs[i];
                if (k < 0 || k >= n) continue;
                Introselect<T, NaNLess<T>, ValueColumn<T>>(buf, default, n, k, pivots, &npiv);
            }
        }

        /// <summary>
        ///     The quantile kernels' entry (their line lengths and positions fit an int): NumPy's <c>introselect_</c> over a
        ///     NaN-free line, once per position, exactly as <see cref="PartitionAtMany{T}(T*, long, long*, int)"/>.
        /// </summary>
        /// <typeparam name="T">The element type (NaN-free: the kernels strip or skip NaN lines first).</typeparam>
        /// <param name="buf">The line (the kernel's scratch copy: the values it reads are the placed positions).</param>
        /// <param name="n">The line's length.</param>
        /// <param name="sortedKs">The positions to place, ascending (NumPy's kth list for the call: see the quantile
        ///     engine, which adds NumPy's <c>0</c> / <c>-1</c> for float data).</param>
        /// <param name="nKs">How many.</param>
        /// <remarks>
        ///     A short line used to be insertion-sorted whole here (a stable order, not NumPy's swaps), so the zero a ±0
        ///     tie left at a neighbour was not NumPy's: np.percentile([0., -0., -0., -1.], 50) read +0.0 where NumPy reads
        ///     -0.0 (QB43). Every line now runs NumPy's statements; for short lines they are mostly dumb_select passes.
        ///     [SkipLocalsInit]: the pivot stack is written before it is read; [AggressiveOptimization]: tier-1 from the
        ///     first call, so the select inlines into the per-row path.
        /// </remarks>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void PartitionAtMany<T>(T* buf, int n, int* sortedKs, int nKs)
            where T : unmanaged, IComparable<T>
        {
            if (nKs == 0 || n <= 1) return;
            long* pivots = stackalloc long[PivotStackMax];
            int npiv = 0;
            for (int i = 0; i < nKs; i++)
            {
                long k = sortedKs[i];
                if (k < 0 || k >= n) continue;
                Introselect<T, RawLess<T>, ValueColumn<T>>(buf, default, n, k, pivots, &npiv);
            }
        }

        /// <summary>
        ///     The quantile kernels' NaN-aware entry: NumPy's <c>introselect_</c> under its NaN-aware comparison over a line
        ///     that holds NaN, so the NaN NumPy's partition leaves at the line's end (<c>arr[-1]</c>, the value a NaN slice
        ///     answers) is found by the same swaps.
        /// </summary>
        /// <typeparam name="T"><see cref="double"/>, <see cref="float"/> or <see cref="Half"/>.</typeparam>
        /// <param name="buf">The line (the kernel's scratch copy).</param>
        /// <param name="n">The line's length.</param>
        /// <param name="sortedKs">The positions to place, ascending.</param>
        /// <param name="nKs">How many.</param>
        /// <remarks>[SkipLocalsInit]: the pivot stack is written before it is read.</remarks>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void PartitionAtManyNaN<T>(T* buf, int n, int* sortedKs, int nKs)
            where T : unmanaged
        {
            if (nKs == 0 || n <= 1) return;
            long* pivots = stackalloc long[PivotStackMax];
            int npiv = 0;
            for (int i = 0; i < nKs; i++)
            {
                long k = sortedKs[i];
                if (k < 0 || k >= n) continue;
                Introselect<T, NaNLess<T>, ValueColumn<T>>(buf, default, n, k, pivots, &npiv);
            }
        }

        /// <summary>
        ///     np.partition's complex128 select: NumPy's <c>introselect_</c> under <c>CDOUBLE_LT</c>
        ///     (<see cref="CDoubleLess"/>), in place, NaN parts included (complex128 is inexact: a kth at the line's end
        ///     takes the last-maximum pass).
        /// </summary>
        /// <param name="buf">The line, partitioned in place.</param>
        /// <param name="n">The line's length.</param>
        /// <param name="sortedKs">The positions, sorted ascending (repeats allowed); a position outside <c>[0, n)</c> is
        ///     skipped.</param>
        /// <param name="nKs">How many positions.</param>
        /// <remarks>[SkipLocalsInit]: the pivot stack is written before it is read.</remarks>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void PartitionAtManyComplex(Complex* buf, long n, long* sortedKs, int nKs)
        {
            if (nKs == 0 || n <= 1) return;
            long* pivots = stackalloc long[PivotStackMax];
            int npiv = 0;
            for (int i = 0; i < nKs; i++)
            {
                long k = sortedKs[i];
                if (k < 0 || k >= n) continue;
                Introselect<Complex, CDoubleLess, ValueColumn<Complex>>(buf, default, n, k, pivots, &npiv);
            }
        }

        /// <summary>
        ///     Pointer-ks <see cref="Comparison{T}"/> variant of <see cref="PartitionAtMany{T}(T*, long, long*, int)"/>
        ///     (np.partition's Complex path, whose NumPy CDOUBLE_LT NaN ordering cannot ride the raw
        ///     <c>&lt;</c> fast path). <paramref name="sortedKs"/> must be sorted ascending.
        /// </summary>
        public static unsafe void PartitionAtMany<T>(T* buf, long n, long* sortedKs, int nKs, Comparison<T> cmp)
            where T : unmanaged
        {
            if (nKs == 0) return;
            long lo = 0;
            long hi = n - 1;
            for (int i = 0; i < nKs; i++)
            {
                long k = sortedKs[i];
                if (k < lo || k > hi) continue;
                IntroSelect(buf, lo, hi, k, 2 * Log2(hi - lo + 1), cmp);
                lo = k + 1;
            }
        }

        // ── index-tracking path (np.argpartition): a parallel long[] of original indices is
        //    swapped in lockstep with the values, exactly the way RadixSort.ArgSortU32/U64 move
        //    their idx column with the keys. The value array is scratch; the indices are the answer.

        /// <summary>
        ///     Index-tracking introselect (NumPy <c>np.argpartition</c>): partitions
        ///     <paramref name="buf"/> around every k in <paramref name="sortedKs"/> (sorted ascending,
        ///     within <c>[0, n-1]</c>) while moving <paramref name="idx"/> — the elements' original
        ///     indices — through the identical swaps. After the call <c>idx[k]</c> indexes the k-th
        ///     smallest source element.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="buf">The line's values (scratch: moved with the indices).</param>
        /// <param name="idx">The original-index column beside <paramref name="buf"/> (the answer).</param>
        /// <param name="n">The line's length.</param>
        /// <param name="sortedKs">The positions, sorted ascending (repeats allowed); a position outside <c>[0, n)</c> is
        ///     skipped.</param>
        /// <param name="nKs">How many positions.</param>
        /// <remarks>
        ///     NumPy's single <c>introselect_&lt;Tag, arg&gt;</c> template with <c>arg = true</c>: the same statements and the
        ///     same pivot stack as the value select, the index column moving with every swap
        ///     (<see cref="IndexColumn{T}"/>), so a tie reports NumPy's index. Every comparison reads
        ///     <paramref name="buf"/> alone: the indices never steer the select. FOOTGUN: the comparison is the raw
        ///     <c>&lt;</c>; a float line that may hold NaN goes to
        ///     <see cref="PartitionAtManyNaN{T}(T*, long*, long, long*, int)"/>. [SkipLocalsInit]: the pivot stack is
        ///     written before it is read.
        /// </remarks>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void PartitionAtMany<T>(T* buf, long* idx, long n, long* sortedKs, int nKs)
            where T : unmanaged, IComparable<T>
        {
            if (nKs == 0 || n <= 1) return;
            long* pivots = stackalloc long[PivotStackMax];
            int npiv = 0;
            var col = new IndexColumn<T>(idx);
            for (int i = 0; i < nKs; i++)
            {
                long k = sortedKs[i];
                if (k < 0 || k >= n) continue;
                Introselect<T, RawLess<T>, IndexColumn<T>>(buf, col, n, k, pivots, &npiv);
            }
        }

        /// <summary>
        ///     np.argpartition's select of a float line that may hold NaN: NumPy's <c>introselect_</c> under its NaN-aware
        ///     comparison (<see cref="NaNLess{T}"/>), in place, the index column moving with every swap, so the NaN indices
        ///     end the line in NumPy's order (not the encounter order of a compaction).
        /// </summary>
        /// <typeparam name="T"><see cref="double"/>, <see cref="float"/> or <see cref="Half"/>.</typeparam>
        /// <param name="buf">The line's values (scratch).</param>
        /// <param name="idx">The original-index column (the answer).</param>
        /// <param name="n">The line's length.</param>
        /// <param name="sortedKs">The positions, sorted ascending (repeats allowed); a position outside <c>[0, n)</c> is
        ///     skipped.</param>
        /// <param name="nKs">How many positions.</param>
        /// <remarks>[SkipLocalsInit]: the pivot stack is written before it is read.</remarks>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void PartitionAtManyNaN<T>(T* buf, long* idx, long n, long* sortedKs, int nKs)
            where T : unmanaged
        {
            if (nKs == 0 || n <= 1) return;
            long* pivots = stackalloc long[PivotStackMax];
            int npiv = 0;
            var col = new IndexColumn<T>(idx);
            for (int i = 0; i < nKs; i++)
            {
                long k = sortedKs[i];
                if (k < 0 || k >= n) continue;
                Introselect<T, NaNLess<T>, IndexColumn<T>>(buf, col, n, k, pivots, &npiv);
            }
        }

        /// <summary>
        ///     np.argpartition's complex128 select: NumPy's <c>introselect_</c> under <c>CDOUBLE_LT</c>, the index column
        ///     moving with every swap.
        /// </summary>
        /// <param name="buf">The line's values (scratch).</param>
        /// <param name="idx">The original-index column (the answer).</param>
        /// <param name="n">The line's length.</param>
        /// <param name="sortedKs">The positions, sorted ascending (repeats allowed); a position outside <c>[0, n)</c> is
        ///     skipped.</param>
        /// <param name="nKs">How many positions.</param>
        /// <remarks>[SkipLocalsInit]: the pivot stack is written before it is read.</remarks>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void PartitionAtManyComplex(Complex* buf, long* idx, long n, long* sortedKs, int nKs)
        {
            if (nKs == 0 || n <= 1) return;
            long* pivots = stackalloc long[PivotStackMax];
            int npiv = 0;
            var col = new IndexColumn<Complex>(idx);
            for (int i = 0; i < nKs; i++)
            {
                long k = sortedKs[i];
                if (k < 0 || k >= n) continue;
                Introselect<Complex, CDoubleLess, IndexColumn<Complex>>(buf, col, n, k, pivots, &npiv);
            }
        }

        /// <summary>
        ///     Index-tracking introselect with a <see cref="Comparison{T}"/> (argpartition's Complex path).
        /// </summary>
        public static unsafe void PartitionAtMany<T>(T* buf, long* idx, long n, long* sortedKs, int nKs, Comparison<T> cmp)
            where T : unmanaged
        {
            if (nKs == 0) return;
            long lo = 0;
            long hi = n - 1;
            for (int i = 0; i < nKs; i++)
            {
                long k = sortedKs[i];
                if (k < lo || k > hi) continue;
                IntroSelect(buf, idx, lo, hi, k, 2 * Log2(hi - lo + 1), cmp);
                lo = k + 1;
            }
        }

        // ── Comparison<T> path (used for float/double with NaN-at-end semantics) ──

        public static unsafe void PartitionAt<T>(T* buf, long n, long k, Comparison<T> cmp) where T : unmanaged
        {
            if (n <= 1 || k < 0 || k >= n) return;
            IntroSelect(buf, 0, n - 1, k, 2 * Log2(n), cmp);
        }

        public static unsafe void PartitionAt<T>(T* buf, long n, long[] sortedKs, Comparison<T> cmp) where T : unmanaged
        {
            if (sortedKs.Length == 0) return;
            long lo = 0;
            long hi = n - 1;
            for (int i = 0; i < sortedKs.Length; i++)
            {
                long k = sortedKs[i];
                if (k < lo || k > hi) continue;
                IntroSelect(buf, lo, hi, k, 2 * Log2(hi - lo + 1), cmp);
                lo = k + 1;
            }
        }

        // ── IComparable internals ─────────────────────────────────────────────────

        private const int InsertionSortThreshold = 16;

        // ── introselect_ constants (QuickSelect.Introselect.cs holds the select itself) ──

        /// <summary>The block partition's scan width per side (see <see cref="BlockScan{T, TLess, TCol}"/>); at most 256,
        ///     so an offset fits a byte.</summary>
        private const int BlockSize = 128;

        /// <summary>
        ///     NumPy's <c>NPY_MAX_PIVOT_STACK</c>: the depth of a line's pivot stack. FOOTGUN: the depth is part of the
        ///     arrangement, not a tuning knob: once the stack is full NumPy drops further pivots (and lets only the kth
        ///     overwrite the top), so a deeper stack narrows later kths' windows differently and moves ties.
        /// </summary>
        private const int PivotStackMax = 50;

        // ── NumPy introselect_'s two linear shortcuts (selection.cpp) ──────────────────────
        //
        // Once the pivot stack has narrowed [low, high], NumPy's introselect_ tries two O(n) placements before its
        // quickselect (QuickSelect.Introselect.cs runs them in NumPy's order):
        //   * kth - low < 3 → dumb_select_: kth - low + 1 passes, pass i moving the window's FIRST minimum (a strict less
        //     never replaces an equal candidate) to position low + i, then store_pivot(kth);
        //   * an inexact line (float32 / float64 / complex128, IsInexact) with kth == n - 1 → one pass moving the window's
        //     LAST maximum ("not less" replaces an equal candidate) to kth; nothing is stored.
        // The quantile family partitions at both ends constantly (NumPy's own 0 and -1, q = 0 and 1, the neighbours of a
        // virtual index near 0, every index clipped to -1), where a quickselect pays several partitioning passes with
        // swaps for one order statistic (np.quantile(a, [0, 1], method='midpoint') over 100,000 doubles: 394 µs as a
        // quickselect, 83 µs as these passes, NumPy 282 µs). Over a NaN-free line of a type with vector lanes the passes
        // search with vectors (the minimum, then its first position; the maximum, then its last position), which finds
        // NumPy's element: equality there is the type's ==.

        /// <summary>Whether NumPy's introselect_ treats the element type as inexact (its kth == n - 1 maximum pass):
        ///     float32, float64 and complex128. NumPy's test is <c>!std::is_integral&lt;type&gt;</c>: its complex types are
        ///     structs (inexact), while its float16 element type <c>npy_half</c> is a <c>uint16_t</c>, so a float16 line
        ///     takes the quickselect there; so it does here, which keeps NumPy's argpartition index for a float16 tie at the
        ///     line's end. JIT-folded to a constant per instantiation.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <returns>True for <see cref="float"/>, <see cref="double"/> and <see cref="Complex"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsInexact<T>()
            => typeof(T) == typeof(double) || typeof(T) == typeof(float) || typeof(T) == typeof(Complex);

        /// <summary>Whether <see cref="Vector{T}"/> has lanes of the element type, so the extreme-value passes run vector
        ///     minimum / maximum and equality scans (<see cref="Vector{T}"/> throws for char, Half, decimal and bool, which
        ///     take NumPy's scalar loop). JIT-folded to a constant per instantiation.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <returns>True for the eight integer types, float and double.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool HasVectorLanes<T>()
            => typeof(T) == typeof(double) || typeof(T) == typeof(float) || typeof(T) == typeof(long)
               || typeof(T) == typeof(ulong) || typeof(T) == typeof(int) || typeof(T) == typeof(uint)
               || typeof(T) == typeof(short) || typeof(T) == typeof(ushort) || typeof(T) == typeof(sbyte)
               || typeof(T) == typeof(byte);

        /// <summary>
        ///     The lane-wise minimum as the hardware computes it (<c>vminpd</c> / <c>vpminsd</c> ...): on .NET 9 and later
        ///     <see cref="Vector.Min{T}"/> is the IEEE 754-2019 minimum, a compare-and-blend sequence whose latency, chained
        ///     through an accumulator, ran 100K doubles in ~110 µs; the native instruction is one cycle of throughput. Its
        ///     only difference, which zero of a ±0 pair (or which NaN) it keeps, is irrelevant here: the windows are NaN-free
        ///     and the minimum's position is found by equality, under which every zero equals every other.
        /// </summary>
        /// <typeparam name="T">A type with <see cref="Vector{T}"/> lanes.</typeparam>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>Per lane, a minimum of the two.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector<T> LaneMin<T>(Vector<T> a, Vector<T> b) where T : unmanaged
#if NET9_0_OR_GREATER
            => Vector.MinNative(a, b);
#else
            => Vector.Min(a, b);   // .NET 8's Vector.Min is already the hardware instruction
#endif

        /// <summary>The lane-wise maximum as the hardware computes it (see <see cref="LaneMin{T}"/>).</summary>
        /// <typeparam name="T">A type with <see cref="Vector{T}"/> lanes.</typeparam>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>Per lane, a maximum of the two.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector<T> LaneMax<T>(Vector<T> a, Vector<T> b) where T : unmanaged
#if NET9_0_OR_GREATER
            => Vector.MaxNative(a, b);
#else
            => Vector.Max(a, b);
#endif

        /// <summary>
        ///     <see cref="FirstMinIndex{T}"/> in two vector passes: the minimum value (four independent lane-wise minimum
        ///     accumulators, then the lanes), then the first position holding it. The second pass compares with "not above
        ///     the minimum", which for the minimum is equality, every zero equal to every other: so the first zero of either
        ///     sign is found whichever sign the lane reduction kept, as NumPy's strict-less scan finds it.
        /// </summary>
        /// <typeparam name="T">A type with <see cref="Vector{T}"/> lanes (<see cref="HasVectorLanes{T}"/>).</typeparam>
        /// <param name="p">The window's first element.</param>
        /// <param name="count">The window's length, at least two vectors.</param>
        /// <returns>The offset of the first minimum from <paramref name="p"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe long FirstMinIndexVector<T>(T* p, long count) where T : unmanaged
        {
            int w = Vector<T>.Count;
            // Four accumulators break the dependency chain of one: four minimums in flight per step.
            var a0 = Unsafe.ReadUnaligned<Vector<T>>(p);
            Vector<T> a1 = a0, a2 = a0, a3 = a0;
            long i = w;
            for (; i <= count - 4L * w; i += 4L * w)
            {
                a0 = LaneMin(a0, Unsafe.ReadUnaligned<Vector<T>>(p + i));
                a1 = LaneMin(a1, Unsafe.ReadUnaligned<Vector<T>>(p + i + w));
                a2 = LaneMin(a2, Unsafe.ReadUnaligned<Vector<T>>(p + i + 2L * w));
                a3 = LaneMin(a3, Unsafe.ReadUnaligned<Vector<T>>(p + i + 3L * w));
            }
            for (; i <= count - w; i += w)
                a0 = LaneMin(a0, Unsafe.ReadUnaligned<Vector<T>>(p + i));
            // The overlapping last vector re-reads a few elements, which cannot change a minimum.
            if (i < count)
                a0 = LaneMin(a0, Unsafe.ReadUnaligned<Vector<T>>(p + count - w));
            a0 = LaneMin(LaneMin(a0, a1), LaneMin(a2, a3));
            T e = a0[0];
            for (int l = 1; l < w; l++)
            {
                T x = a0[l];
                if (LtPtr(&x, &e)) e = x;
            }

            // The first group of four vectors holding the minimum, then the first such vector, then its lane; nothing
            // before them held it.
            var ve = new Vector<T>(e);
            i = 0;
            for (; i <= count - 4L * w; i += 4L * w)
            {
                var hit = Vector.Equals(Unsafe.ReadUnaligned<Vector<T>>(p + i), ve)
                          | Vector.Equals(Unsafe.ReadUnaligned<Vector<T>>(p + i + w), ve)
                          | Vector.Equals(Unsafe.ReadUnaligned<Vector<T>>(p + i + 2L * w), ve)
                          | Vector.Equals(Unsafe.ReadUnaligned<Vector<T>>(p + i + 3L * w), ve);
                if (hit != Vector<T>.Zero)
                    break;
            }
            for (; i <= count - w; i += w)
                if (Vector.EqualsAny(Unsafe.ReadUnaligned<Vector<T>>(p + i), ve))
                    break;
            for (; i < count - 1; i++)
                if (!LtPtr(&e, p + i))
                    return i;
            // The minimum is an element of the window: when no earlier position holds it, the last one does.
            return count - 1;
        }

        /// <summary>
        ///     <see cref="LastMaxIndex{T}"/> in two vector passes: the maximum value (four independent accumulators), then
        ///     the last position holding it, scanning backwards ("not below the maximum" is equality, every zero equal to
        ///     every other).
        /// </summary>
        /// <typeparam name="T">A type with <see cref="Vector{T}"/> lanes (<see cref="HasVectorLanes{T}"/>).</typeparam>
        /// <param name="p">The window's first element.</param>
        /// <param name="count">The window's length, at least two vectors.</param>
        /// <returns>The offset of the last maximum from <paramref name="p"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe long LastMaxIndexVector<T>(T* p, long count) where T : unmanaged
        {
            int w = Vector<T>.Count;
            var a0 = Unsafe.ReadUnaligned<Vector<T>>(p);
            Vector<T> a1 = a0, a2 = a0, a3 = a0;
            long i = w;
            for (; i <= count - 4L * w; i += 4L * w)
            {
                a0 = LaneMax(a0, Unsafe.ReadUnaligned<Vector<T>>(p + i));
                a1 = LaneMax(a1, Unsafe.ReadUnaligned<Vector<T>>(p + i + w));
                a2 = LaneMax(a2, Unsafe.ReadUnaligned<Vector<T>>(p + i + 2L * w));
                a3 = LaneMax(a3, Unsafe.ReadUnaligned<Vector<T>>(p + i + 3L * w));
            }
            for (; i <= count - w; i += w)
                a0 = LaneMax(a0, Unsafe.ReadUnaligned<Vector<T>>(p + i));
            if (i < count)
                a0 = LaneMax(a0, Unsafe.ReadUnaligned<Vector<T>>(p + count - w));
            a0 = LaneMax(LaneMax(a0, a1), LaneMax(a2, a3));
            T e = a0[0];
            for (int l = 1; l < w; l++)
            {
                T x = a0[l];
                if (LtPtr(&e, &x)) e = x;
            }

            // The last group of four vectors holding the maximum (j is the end of the part not yet scanned), then the
            // last such vector, then its lane; nothing after them held it.
            var ve = new Vector<T>(e);
            long j = count;
            for (; j >= 4L * w; j -= 4L * w)
            {
                var hit = Vector.Equals(Unsafe.ReadUnaligned<Vector<T>>(p + j - w), ve)
                          | Vector.Equals(Unsafe.ReadUnaligned<Vector<T>>(p + j - 2L * w), ve)
                          | Vector.Equals(Unsafe.ReadUnaligned<Vector<T>>(p + j - 3L * w), ve)
                          | Vector.Equals(Unsafe.ReadUnaligned<Vector<T>>(p + j - 4L * w), ve);
                if (hit != Vector<T>.Zero)
                    break;
            }
            for (; j >= w; j -= w)
                if (Vector.EqualsAny(Unsafe.ReadUnaligned<Vector<T>>(p + j - w), ve))
                    break;
            for (j--; j > 0; j--)
                if (!LtPtr(p + j, &e))
                    return j;
            // The maximum is an element of the window: when no later position holds it, the first one does.
            return 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void IntroSelect<T>(T* buf, long lo, long hi, long k, int depthLimit)
            where T : unmanaged, IComparable<T>
        {
            while (lo < hi)
            {
                long len = hi - lo + 1;
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

                long p = Partition(buf, lo, hi);
                if (k == p) return;
                if (k < p) hi = p - 1;
                else lo = p + 1;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe long Partition<T>(T* buf, long lo, long hi) where T : unmanaged, IComparable<T>
        {
            long mid = lo + ((hi - lo) >> 1);
            SwapIfGreater(buf, lo, mid);
            SwapIfGreater(buf, lo, hi);
            SwapIfGreater(buf, mid, hi);

            T pivot = buf[mid];
            Swap(buf, mid, hi - 1);

            long left = lo;
            long right = hi - 1;
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

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void InsertionSort<T>(T* buf, long lo, long hi) where T : unmanaged, IComparable<T>
        {
            for (long i = lo; i < hi; i++)
            {
                long j = i;
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
        private static unsafe void SwapIfGreater<T>(T* buf, long i, long j) where T : unmanaged, IComparable<T>
        {
            if (LtV(buf[j], buf[i])) Swap(buf, i, j);
        }

        // ── Comparison<T> internals ───────────────────────────────────────────────

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void IntroSelect<T>(T* buf, long lo, long hi, long k, int depthLimit, Comparison<T> cmp)
            where T : unmanaged
        {
            while (lo < hi)
            {
                long len = hi - lo + 1;
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

                long p = Partition(buf, lo, hi, cmp);
                if (k == p) return;
                if (k < p) hi = p - 1;
                else lo = p + 1;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe long Partition<T>(T* buf, long lo, long hi, Comparison<T> cmp) where T : unmanaged
        {
            long mid = lo + ((hi - lo) >> 1);
            SwapIfGreater(buf, lo, mid, cmp);
            SwapIfGreater(buf, lo, hi, cmp);
            SwapIfGreater(buf, mid, hi, cmp);

            T pivot = buf[mid];
            Swap(buf, mid, hi - 1);

            long left = lo;
            long right = hi - 1;
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

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void InsertionSort<T>(T* buf, long lo, long hi, Comparison<T> cmp) where T : unmanaged
        {
            for (long i = lo; i < hi; i++)
            {
                long j = i;
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
        private static unsafe void SwapIfGreater<T>(T* buf, long i, long j, Comparison<T> cmp) where T : unmanaged
        {
            if (cmp(buf[i], buf[j]) > 0) Swap(buf, i, j);
        }

        // ── heap-sort fallback (used when introselect recurses too deep) ──────────

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HeapSort<T>(T* buf, long lo, long hi) where T : unmanaged, IComparable<T>
        {
            long n = hi - lo + 1;
            for (long i = n >> 1; i >= 1; i--) DownHeap(buf, i, n, lo);
            for (long i = n; i > 1; i--) { Swap(buf, lo, lo + i - 1); DownHeap(buf, 1, i - 1, lo); }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void DownHeap<T>(T* buf, long i, long n, long lo) where T : unmanaged, IComparable<T>
        {
            T d = buf[lo + i - 1];
            while (i <= n >> 1)
            {
                long child = 2 * i;
                if (child < n && LtV(buf[lo + child - 1], buf[lo + child])) child++;
                if (!LtV(d, buf[lo + child - 1])) break;
                buf[lo + i - 1] = buf[lo + child - 1];
                i = child;
            }
            buf[lo + i - 1] = d;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HeapSort<T>(T* buf, long lo, long hi, Comparison<T> cmp) where T : unmanaged
        {
            long n = hi - lo + 1;
            for (long i = n >> 1; i >= 1; i--) DownHeap(buf, i, n, lo, cmp);
            for (long i = n; i > 1; i--) { Swap(buf, lo, lo + i - 1); DownHeap(buf, 1, i - 1, lo, cmp); }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void DownHeap<T>(T* buf, long i, long n, long lo, Comparison<T> cmp) where T : unmanaged
        {
            T d = buf[lo + i - 1];
            while (i <= n >> 1)
            {
                long child = 2 * i;
                if (child < n && cmp(buf[lo + child - 1], buf[lo + child]) < 0) child++;
                if (cmp(d, buf[lo + child - 1]) >= 0) break;
                buf[lo + i - 1] = buf[lo + child - 1];
                i = child;
            }
            buf[lo + i - 1] = d;
        }

        // ── index-tracking scalar internals (values + parallel idx column swap together) ──
        //
        // The single-k PartitionAt-style scalar select with an index column and its Comparison<T> twin. np.argpartition
        // runs NumPy's introselect_ instead (QuickSelect.Introselect.cs, IndexColumn<T>).

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void IntroSelect<T>(T* buf, long* idx, long lo, long hi, long k, int depthLimit)
            where T : unmanaged, IComparable<T>
        {
            while (lo < hi)
            {
                long len = hi - lo + 1;
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

                long p = Partition(buf, idx, lo, hi);
                if (k == p) return;
                if (k < p) hi = p - 1;
                else lo = p + 1;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe long Partition<T>(T* buf, long* idx, long lo, long hi) where T : unmanaged, IComparable<T>
        {
            long mid = lo + ((hi - lo) >> 1);
            SwapIfGreater(buf, idx, lo, mid);
            SwapIfGreater(buf, idx, lo, hi);
            SwapIfGreater(buf, idx, mid, hi);

            T pivot = buf[mid];
            Swap(buf, idx, mid, hi - 1);

            long left = lo;
            long right = hi - 1;
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

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void InsertionSort<T>(T* buf, long* idx, long lo, long hi) where T : unmanaged, IComparable<T>
        {
            for (long i = lo; i < hi; i++)
            {
                long j = i;
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
        private static unsafe void SwapIfGreater<T>(T* buf, long* idx, long i, long j) where T : unmanaged, IComparable<T>
        {
            if (LtV(buf[j], buf[i])) Swap(buf, idx, i, j);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HeapSort<T>(T* buf, long* idx, long lo, long hi) where T : unmanaged, IComparable<T>
        {
            long n = hi - lo + 1;
            for (long i = n >> 1; i >= 1; i--) DownHeap(buf, idx, i, n, lo);
            for (long i = n; i > 1; i--) { Swap(buf, idx, lo, lo + i - 1); DownHeap(buf, idx, 1, i - 1, lo); }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void DownHeap<T>(T* buf, long* idx, long i, long n, long lo) where T : unmanaged, IComparable<T>
        {
            T d = buf[lo + i - 1];
            long di = idx[lo + i - 1];
            while (i <= n >> 1)
            {
                long child = 2 * i;
                if (child < n && LtV(buf[lo + child - 1], buf[lo + child])) child++;
                if (!LtV(d, buf[lo + child - 1])) break;
                buf[lo + i - 1] = buf[lo + child - 1];
                idx[lo + i - 1] = idx[lo + child - 1];
                i = child;
            }
            buf[lo + i - 1] = d;
            idx[lo + i - 1] = di;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void IntroSelect<T>(T* buf, long* idx, long lo, long hi, long k, int depthLimit, Comparison<T> cmp)
            where T : unmanaged
        {
            while (lo < hi)
            {
                long len = hi - lo + 1;
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

                long p = Partition(buf, idx, lo, hi, cmp);
                if (k == p) return;
                if (k < p) hi = p - 1;
                else lo = p + 1;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe long Partition<T>(T* buf, long* idx, long lo, long hi, Comparison<T> cmp) where T : unmanaged
        {
            long mid = lo + ((hi - lo) >> 1);
            SwapIfGreater(buf, idx, lo, mid, cmp);
            SwapIfGreater(buf, idx, lo, hi, cmp);
            SwapIfGreater(buf, idx, mid, hi, cmp);

            T pivot = buf[mid];
            Swap(buf, idx, mid, hi - 1);

            long left = lo;
            long right = hi - 1;
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

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void InsertionSort<T>(T* buf, long* idx, long lo, long hi, Comparison<T> cmp) where T : unmanaged
        {
            for (long i = lo; i < hi; i++)
            {
                long j = i;
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
        private static unsafe void SwapIfGreater<T>(T* buf, long* idx, long i, long j, Comparison<T> cmp) where T : unmanaged
        {
            if (cmp(buf[i], buf[j]) > 0) Swap(buf, idx, i, j);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HeapSort<T>(T* buf, long* idx, long lo, long hi, Comparison<T> cmp) where T : unmanaged
        {
            long n = hi - lo + 1;
            for (long i = n >> 1; i >= 1; i--) DownHeap(buf, idx, i, n, lo, cmp);
            for (long i = n; i > 1; i--) { Swap(buf, idx, lo, lo + i - 1); DownHeap(buf, idx, 1, i - 1, lo, cmp); }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void DownHeap<T>(T* buf, long* idx, long i, long n, long lo, Comparison<T> cmp) where T : unmanaged
        {
            T d = buf[lo + i - 1];
            long di = idx[lo + i - 1];
            while (i <= n >> 1)
            {
                long child = 2 * i;
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
        private static unsafe void Swap<T>(T* buf, long i, long j) where T : unmanaged
        {
            T t = buf[i];
            buf[i] = buf[j];
            buf[j] = t;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void Swap<T>(T* buf, long* idx, long i, long j) where T : unmanaged
        {
            T t = buf[i];
            buf[i] = buf[j];
            buf[j] = t;
            long ti = idx[i];
            idx[i] = idx[j];
            idx[j] = ti;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static int Log2(long v)
        {
            int r = 0;
            while (v > 0) { r++; v >>= 1; }
            return r;
        }

        /// <summary>
        ///     A float16's position in IEEE order as an int, from its bits: the magnitude for a positive value, minus the
        ///     magnitude for a negative one, so -0.0 and +0.0 share the key 0 and the key order is exactly IEEE <c>&lt;</c>
        ///     for every value that is not NaN (the only inputs QuickSelect sees: its callers compact NaNs out).
        /// </summary>
        /// <param name="bits">The float16's bit pattern.</param>
        /// <returns>The order key.</returns>
        /// <remarks>
        ///     .NET's <see cref="Half"/> <c>&lt;</c> tests NaN and the signs with branches, which turned the branchless
        ///     block partition's scan back into a branchy one: float16 lines partitioned at 1.05–1.3× NumPy's speed. This
        ///     is four integer operations without a branch.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int HalfOrderKey(ushort bits)
        {
            int magnitude = bits & 0x7FFF;
            int negative = -(bits >> 15);              // 0 for a positive value, -1 for a negative one
            return (magnitude ^ negative) - negative;  // -magnitude when negative (two's complement), else magnitude
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
            if (typeof(T) == typeof(Half))    return HalfOrderKey(*(ushort*)&a) < HalfOrderKey(*(ushort*)&b);
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
        /// <typeparam name="T">The element type: any of the 14 non-complex dtypes folds to one native compare; another
        ///     type falls back to <see cref="System.Collections.Generic.Comparer{T}.Default"/> (slow, and an exception for a
        ///     type with no ordering).</typeparam>
        /// <param name="a">The first element.</param>
        /// <param name="b">The second element.</param>
        /// <returns>The type's raw <c>*a &lt; *b</c> (IEEE for the floats: false whenever either is NaN).</returns>
        /// <exception cref="ArgumentException">Only for a type outside the dtype list that has no
        ///     <see cref="IComparable{T}"/> / <see cref="IComparable"/> ordering (no NumSharp dtype reaches the fallback).</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe bool LtPtr<T>(T* a, T* b) where T : unmanaged
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
            if (typeof(T) == typeof(Half))    return HalfOrderKey(*(ushort*)a) < HalfOrderKey(*(ushort*)b);
            if (typeof(T) == typeof(decimal)) return *(decimal*)a < *(decimal*)b;
            if (typeof(T) == typeof(bool))    return !*(bool*)a && *(bool*)b;
            // Reached by no dtype (complex takes CDoubleLess); kept so the constraint can stay `unmanaged` for the
            // policy structs, which instantiate this for every element type.
            return System.Collections.Generic.Comparer<T>.Default.Compare(*a, *b) < 0;
        }
    }
}
