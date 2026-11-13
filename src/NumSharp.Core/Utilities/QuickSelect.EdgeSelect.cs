using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    // ── edge selection: one kth within a few positions of either end of a long line ──────────────────────────────
    //
    // np.argpartition(x, -k)[-k:] is how NumPy code takes a top-k (a classifier's best classes, a decoder's candidate
    // tokens), np.partition(x, -k) its value twin. For a kth that close to an end the quickselect still partitions the
    // whole line, pass after pass, moving the values and (for argpartition) an int64 index column with them: 51,200
    // float32 at kth = n - 5 took 247 µs here, against NumPy's 96 µs (x86-simd-sort's AVX2 argselect). The m elements on
    // the kth's side are instead found in ONE read: the m best seen so far are kept sorted, and every element is tested
    // against the worst of them, a vector compare rejecting Vector<T>.Count elements at a time (once the kept set is
    // warm, almost nothing beats it). The arrangement is then written directly: argpartition writes every other index in
    // ascending order (a vector iota with m holes) and the kept ones from the kth outwards; partition moves only the
    // kept values that were outside the kth's block, O(m) writes. Measured at n = 51,200, m = 5: argpartition
    // 206 -> 23 µs, partition 170 -> 17 µs; at n = 1M 5.7 -> 0.6 ms and 4.4 -> 0.56 ms.
    //
    // NumPy has no such path (its introselect_ carries "FIXME: useful to check if kth is near the end?"). The arrangement
    // around a single kth was not NumPy's here before either: NumPy hands a single kth of a 32- or 64-bit type to
    // x86-simd-sort's argselect / qselect on AVX2 and AVX-512 hosts, and the quickselect above arranges its own way. Only
    // the kth's value and the two-sided invariant are contractual, which is what the oracle compares (G12, issue #623).
    // The two linear shortcuts that keep NumPy's choice among equal values (DumbSelect for kth < 3, the float
    // last-maximum pass for kth == n - 1) keep serving their lines: the gate leaves them out.
    internal static partial class QuickSelect
    {
        /// <summary>
        ///     The capacity of an edge selection's kept set, and the most elements one serves (m, the kth's side of the line,
        ///     kth included): the block <see cref="EdgeArrange{T, TSide}"/> fills is marked in one 64-bit mask, and keeping
        ///     the set sorted costs O(m) per admitted element, which a larger m would turn into the scan's main cost.
        /// </summary>
        private const int EdgeSelectCapacity = 64;

        /// <summary>
        ///     The shortest line the edge selection serves. On shorter lines the quickselect's whole cost is a few
        ///     microseconds of per-call work that the selection's fixed costs (the kept set's fill, the sort of the kept
        ///     positions) would not undercut by much, and the arrangements unit tests pin on small arrays stay the
        ///     quickselect's.
        /// </summary>
        private const long EdgeSelectMinLength = 256;

        /// <summary>
        ///     The most elements on the kth's side of a line that the edge selection serves, at most
        ///     <see cref="EdgeSelectCapacity"/>: 64 by default; the environment variable <c>NS_EDGE_SELECT_MAX</c> lowers it
        ///     for tuning, and <c>0</c> turns the path off (every line then takes the quickselect, as before the path
        ///     existed). Read once, at the type's first use.
        /// </summary>
        internal static readonly int EdgeSelectMaxCount =
            int.TryParse(Environment.GetEnvironmentVariable("NS_EDGE_SELECT_MAX"), out int t) && t >= 0
                ? Math.Min(t, EdgeSelectCapacity)
                : EdgeSelectCapacity;

        /// <summary>Which end of the line an edge selection works from; a struct type argument, so the JIT folds every
        ///     <see cref="Top"/> test to a constant and each side gets its own branch-free code.</summary>
        private interface IEdgeSide
        {
            /// <summary>True: the kth's side runs to the line's end (the m largest elements); false: it starts at 0 (the
            ///     m smallest).</summary>
            bool Top { get; }
        }

        /// <summary>The m largest elements: a kth at <c>n - m</c>.</summary>
        private struct TopSide : IEdgeSide
        {
            /// <inheritdoc/>
            public bool Top => true;
        }

        /// <summary>The m smallest elements: a kth at <c>m - 1</c>.</summary>
        private struct BottomSide : IEdgeSide
        {
            /// <inheritdoc/>
            public bool Top => false;
        }

        /// <summary>
        ///     Whether the edge selection serves a single <paramref name="kth"/> of an <paramref name="n"/>-element line of
        ///     <typeparamref name="T"/>, and from which end. Cheap (a few compares), so the line kernels ask it per line.
        /// </summary>
        /// <typeparam name="T">The element type: one with <see cref="Vector{T}"/> lanes (the eight integer types, float and
        ///     double); bool, char, Half, decimal and complex lines keep the quickselect.</typeparam>
        /// <param name="n">The line's length.</param>
        /// <param name="kth">The position to place, validated to <c>[0, n)</c>.</param>
        /// <param name="top">True when the kth's side runs to the line's end, false when it starts at 0.</param>
        /// <param name="m">The kth's side's length, kth included: the elements the selection finds.</param>
        /// <returns>True when the line takes the edge selection.</returns>
        /// <remarks>
        ///     The line must be long for its side: the selection's cost beyond its read is the elements it admits into the
        ///     kept set, each at O(m), about <c>m (ln(n / m) + 0.58)</c> of them on random data, while the quickselect
        ///     it replaces costs a few nanoseconds per element. <c>n ≥ max(256, 32 m, m²)</c> keeps the admissions a small
        ///     part of the work. Measured against the quickselect on random float32 (argpartition): m = 5 runs 1.2×
        ///     faster at n = 256, 2.4× at 1,024, 6.1× at 4,096 and 9× at 51,200; m = 16 1.4× at 512; m = 32 1.7× at 2,048;
        ///     m = 64 1.6× at 4,096 but 0.95× at 2,048, which the m² term keeps out.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsEdgeSelect<T>(long n, long kth, out bool top, out int m) where T : unmanaged, IComparable<T>
        {
            top = false;
            m = 0;
            if (!HasVectorLanes<T>() || !Vector.IsHardwareAccelerated)
                return false;
            long above = n - kth, below = kth + 1;
            top = above <= below;
            long side = top ? above : below;
            if (side > EdgeSelectMaxCount)
                return false;
            // NumPy's own linear shortcuts already serve these lines in one or three passes, choosing among equal values as
            // NumPy does (IntroSelectBlock takes them before its quickselect).
            if (top ? IsInexact<T>() && kth == n - 1 : kth < 3)
                return false;
            if (n < EdgeSelectMinLength || n < 32L * side || n < side * side)
                return false;
            m = (int)side;
            return true;
        }

        /// <summary>
        ///     np.argpartition of one contiguous line around a single kth near one of its ends: writes the full index
        ///     permutation to <paramref name="outLine"/> when the edge selection serves the line, and returns false (having
        ///     written nothing) when it does not, so the caller runs the quickselect instead.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="line">The line's first element; the elements are contiguous and only read.</param>
        /// <param name="n">The line's length.</param>
        /// <param name="kth">The position to place, validated to <c>[0, n)</c>.</param>
        /// <param name="outLine">The int64 output line's first element.</param>
        /// <param name="outStride">The output line's stride in bytes.</param>
        /// <returns>True when the output was written; false when the line is not one the path serves (see
        ///     <see cref="IsEdgeSelect{T}"/>), holds a NaN (the caller's compaction moves NaN indices to the tail), or gave
        ///     too many admissions (a mostly ascending line for the largest elements, descending for the smallest).</returns>
        /// <remarks>
        ///     The output: position kth holds the kth order statistic's index, the other kept indices follow it outwards in
        ///     value order (ascending values from the kth to the line's end, or from position 0 to the kth), and the
        ///     remaining positions hold every other index in ascending order. Among equal values the lower index is kept,
        ///     which is ONNX TopK's tie rule; nothing promises it to callers (argpartition's arrangement is not
        ///     contractual).
        /// </remarks>
        [SkipLocalsInit]
        internal static unsafe bool TryEdgeArgPartition<T>(T* line, long n, long kth, byte* outLine, long outStride)
            where T : unmanaged, IComparable<T>
        {
            if (!IsEdgeSelect<T>(n, kth, out bool top, out int m))
                return false;
            // Stack scratch, not zeroed ([SkipLocalsInit]): the fill writes every kept slot before anything reads it, and
            // this runs once per line, so a many-row argpartition would pay the zeroing on every row.
            T* keptV = stackalloc T[EdgeSelectCapacity];
            long* keptI = stackalloc long[EdgeSelectCapacity];
            if (top)
            {
                if (!TryEdgeSelect<T, TopSide>(line, n, m, keptV, keptI))
                    return false;
                EdgeWriteIndices<TopSide>(outLine, outStride, n, m, keptI);
            }
            else
            {
                if (!TryEdgeSelect<T, BottomSide>(line, n, m, keptV, keptI))
                    return false;
                EdgeWriteIndices<BottomSide>(outLine, outStride, n, m, keptI);
            }
            return true;
        }

        /// <summary>
        ///     np.partition of one contiguous line, in place, around a single kth near one of its ends: arranges the line
        ///     when the edge selection serves it, and returns false (having moved nothing) when it does not, so the caller
        ///     runs the quickselect instead.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="line">The line's first element; the elements are contiguous and writable.</param>
        /// <param name="n">The line's length.</param>
        /// <param name="kth">The position to place, validated to <c>[0, n)</c>.</param>
        /// <returns>True when the line was partitioned; false as <see cref="TryEdgeArgPartition{T}"/> returns it (a NaN
        ///     included: the float line kernels' compaction keeps the NaN bit patterns in the tail).</returns>
        /// <remarks>
        ///     The kth's block (the line's last m positions, or its first m) ends sorted, its order statistic at kth; of
        ///     the other elements only those that were in the block move, into the positions the block's elements came
        ///     from. Values move as whole <typeparamref name="T"/>s, so a zero keeps its sign.
        /// </remarks>
        [SkipLocalsInit]
        internal static unsafe bool TryEdgePartition<T>(T* line, long n, long kth) where T : unmanaged, IComparable<T>
        {
            if (!IsEdgeSelect<T>(n, kth, out bool top, out int m))
                return false;
            // Stack scratch, not zeroed ([SkipLocalsInit]): the fill writes every kept slot before anything reads it.
            T* keptV = stackalloc T[EdgeSelectCapacity];
            long* keptI = stackalloc long[EdgeSelectCapacity];
            if (top)
            {
                if (!TryEdgeSelect<T, TopSide>(line, n, m, keptV, keptI))
                    return false;
                EdgeArrange<T, TopSide>(line, n, m, keptV, keptI);
            }
            else
            {
                if (!TryEdgeSelect<T, BottomSide>(line, n, m, keptV, keptI))
                    return false;
                EdgeArrange<T, BottomSide>(line, n, m, keptV, keptI);
            }
            return true;
        }

        /// <summary>
        ///     The selection pass: finds the <paramref name="m"/> elements of <paramref name="v"/> on
        ///     <typeparamref name="TSide"/>'s end, in one read, without moving anything.
        /// </summary>
        /// <typeparam name="T">The element type (with <see cref="Vector{T}"/> lanes).</typeparam>
        /// <typeparam name="TSide"><see cref="TopSide"/> for the largest elements, <see cref="BottomSide"/> for the
        ///     smallest.</typeparam>
        /// <param name="v">The line (contiguous, at least <paramref name="m"/> + one vector long).</param>
        /// <param name="n">The line's length.</param>
        /// <param name="m">How many elements to keep, 1 to <see cref="EdgeSelectCapacity"/>.</param>
        /// <param name="keptV">Receives the kept values, worst first: <c>keptV[0]</c> is the kth order statistic.</param>
        /// <param name="keptI">Receives the kept values' positions, in the same order.</param>
        /// <returns>False on a NaN, or once more than <c>m (log2(n) + 8)</c> elements were admitted after the fill; the
        ///     line is then untouched, and the kept arrays are scratch.</returns>
        /// <remarks>
        ///     An element is admitted when it is strictly better than the worst kept one, which it evicts; it is inserted
        ///     below the kept values equal to it. Elements arrive in position order, so among equal values the lower
        ///     position is kept and is the last to be evicted.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe bool TryEdgeSelect<T, TSide>(T* v, long n, int m, T* keptV, long* keptI)
            where T : unmanaged, IComparable<T>
            where TSide : struct, IEdgeSide
        {
            long i = 0;
            // Fill with the first m elements, sorted worst first. An element goes below its equals: it came later, so it
            // is the first of them to go.
            for (; i < m; i++)
            {
                T x = v[i];
                if (IsNaNValue(&x))
                    return false;
                long p = i;
                for (; p > 0 && !EdgeBetter<T, TSide>(&x, keptV + p - 1); p--)
                {
                    keptV[p] = keptV[p - 1];
                    keptI[p] = keptI[p - 1];
                }
                keptV[p] = x;
                keptI[p] = i;
            }

            // Bounds the admissions. Random data admits about m (ln(n / m) + 0.58) elements, well under this whatever n is
            // (ln(n / m) < log2(n), plus 8 of margin: at least 2.4× the expectation). A line that keeps beating the kept
            // set (sorted the wrong way for this side) gives up after about that many elements, so it costs the
            // quickselect plus a sliver: a first budget of 4 n / m let an ascending line pay 0.8 n slow admissions first,
            // which doubled its time (51,200 float32, top 5: 239 -> 482 µs). Each vector with a hit admits at least its
            // first hit, so the per-lane rechecks are bounded by the budget too.
            long budget = (long)m * (Log2(n) + 8);
            int w = Vector<T>.Count;
            var worst = new Vector<T>(keptV[0]);
            for (; i <= n - w; i += w)
            {
                var x = Unsafe.ReadUnaligned<Vector<T>>(v + i);
                // A NaN compares false either way, so it would pass silently; the float line kernels' NaN-tail policy
                // needs the quickselect route, so a NaN anywhere ends the selection.
                if (IsInexact<T>() && !Vector.EqualsAll(x, x))
                    return false;
                if (default(TSide).Top ? !Vector.GreaterThanAny(x, worst) : !Vector.LessThanAny(x, worst))
                    continue;
                for (long j = i; j < i + w; j++)
                {
                    if (!EdgeBetter<T, TSide>(v + j, keptV))
                        continue;
                    if (--budget < 0)
                        return false;
                    EdgeAdmit<T, TSide>(v[j], j, keptV, keptI, m);
                }
                worst = new Vector<T>(keptV[0]);
            }
            for (; i < n; i++)
            {
                T x = v[i];
                if (IsNaNValue(&x))
                    return false;
                if (!EdgeBetter<T, TSide>(&x, keptV))
                    continue;
                if (--budget < 0)
                    return false;
                EdgeAdmit<T, TSide>(x, i, keptV, keptI, m);
            }
            return true;
        }

        /// <summary>
        ///     Admits <paramref name="x"/> into a full kept set: evicts the worst (<c>keptV[0]</c>) and slides every kept
        ///     value strictly worse than <paramref name="x"/> down one place, so <paramref name="x"/> lands below its equals.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TSide">The selection's end.</typeparam>
        /// <param name="x">The admitted value, strictly better than <c>keptV[0]</c>.</param>
        /// <param name="pos">Its position in the line.</param>
        /// <param name="keptV">The kept values, worst first.</param>
        /// <param name="keptI">Their positions.</param>
        /// <param name="m">The kept set's size.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void EdgeAdmit<T, TSide>(T x, long pos, T* keptV, long* keptI, int m)
            where T : unmanaged, IComparable<T>
            where TSide : struct, IEdgeSide
        {
            int p = 0;
            for (; p + 1 < m && EdgeBetter<T, TSide>(&x, keptV + p + 1); p++)
            {
                keptV[p] = keptV[p + 1];
                keptI[p] = keptI[p + 1];
            }
            keptV[p] = x;
            keptI[p] = pos;
        }

        /// <summary>Whether <c>*a</c> is strictly better than <c>*b</c> for <typeparamref name="TSide"/>: larger for the
        ///     top side, smaller for the bottom one (NaN-free operands: the selection refuses NaN lines).</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TSide">The selection's end.</typeparam>
        /// <param name="a">The candidate.</param>
        /// <param name="b">The value it is measured against.</param>
        /// <returns>True when <c>*a</c> ranks strictly before <c>*b</c>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool EdgeBetter<T, TSide>(T* a, T* b)
            where T : unmanaged, IComparable<T>
            where TSide : struct, IEdgeSide
            => default(TSide).Top ? LtPtr(b, a) : LtPtr(a, b);

        /// <summary>Whether <c>*p</c> is a float or double NaN; false for every other type (JIT-folded per
        ///     instantiation).</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="p">The value.</param>
        /// <returns>True for a NaN.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool IsNaNValue<T>(T* p) where T : unmanaged
            => typeof(T) == typeof(float) ? float.IsNaN(*(float*)p)
               : typeof(T) == typeof(double) && double.IsNaN(*(double*)p);

        /// <summary>
        ///     Writes the argpartition permutation from a finished selection: every unselected index in ascending order,
        ///     and the kept ones in value order with the worst at kth (<c>n - m</c> for the top side, <c>m - 1</c> for the
        ///     bottom one).
        /// </summary>
        /// <typeparam name="TSide">The selection's end.</typeparam>
        /// <param name="outLine">The int64 output line's first element.</param>
        /// <param name="outStride">The output line's stride in bytes.</param>
        /// <param name="n">The line's length.</param>
        /// <param name="m">The kept set's size.</param>
        /// <param name="keptI">The kept positions, worst first.</param>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void EdgeWriteIndices<TSide>(byte* outLine, long outStride, long n, int m, long* keptI)
            where TSide : struct, IEdgeSide
        {
            // The kept positions ascending, so the iota below can step over them in one forward walk (an insertion sort:
            // m is at most 64; the stack copy is not zeroed, [SkipLocalsInit], and each slot is written before it is read).
            long* holes = stackalloc long[EdgeSelectCapacity];
            for (int k = 0; k < m; k++)
            {
                long x = keptI[k];
                int p = k;
                for (; p > 0 && holes[p - 1] > x; p--)
                    holes[p] = holes[p - 1];
                holes[p] = x;
            }

            // The unselected indices fill the side away from kth: positions [0, n - m) for the top side, [m, n) for the
            // bottom one, each run between two holes written as one ascending stretch.
            long at = default(TSide).Top ? 0 : m;
            long next = 0;
            for (int k = 0; k <= m; k++)
            {
                long stop = k < m ? holes[k] : n;
                EdgeFillIota(outLine, outStride, at, next, stop - next);
                at += stop - next;
                next = stop + 1;
            }

            // The kept indices, worst at kth and value order outwards.
            if (default(TSide).Top)
            {
                for (int k = 0; k < m; k++)
                    *(long*)(outLine + (n - m + k) * outStride) = keptI[k];
            }
            else
            {
                for (int k = 0; k < m; k++)
                    *(long*)(outLine + k * outStride) = keptI[m - 1 - k];
            }
        }

        /// <summary>
        ///     Writes <c>first, first + 1, …</c> (<paramref name="count"/> values) to output positions <paramref name="at"/>
        ///     onwards: whole vectors of int64 when the output is contiguous, one element per step otherwise.
        /// </summary>
        /// <param name="outLine">The int64 output line's first element.</param>
        /// <param name="outStride">The output line's stride in bytes.</param>
        /// <param name="at">The first output position written.</param>
        /// <param name="first">The first value written.</param>
        /// <param name="count">How many values; zero writes nothing.</param>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void EdgeFillIota(byte* outLine, long outStride, long at, long first, long count)
        {
            long k = 0;
            if (outStride == sizeof(long) && Vector.IsHardwareAccelerated && count >= 2L * Vector<long>.Count)
            {
                long* dst = (long*)outLine + at;
                // Scalar stores up to the first vector-aligned element: the bottom side's fill starts m elements into the
                // output, and every unaligned 32-byte store there that crossed a cache line made its whole fill ~1.7×
                // slower than the top side's (51,200 float32, m = 5: 39.9 against 23.4 µs). An int64 output is 8-aligned,
                // so the peel reaches the vector boundary.
                int vectorBytes = Vector<byte>.Count;
                for (; k < count && ((nuint)(dst + k) & (nuint)(vectorBytes - 1)) != 0; k++)
                    dst[k] = first + k;
                // The lane numbers 0..w-1 (Vector<T>.Indices is .NET 9+; this file builds for net8.0 too), written into
                // unzeroed stack memory ([SkipLocalsInit]) before the one read.
                int w = Vector<long>.Count;
                long* lanes = stackalloc long[w];
                for (int l = 0; l < w; l++)
                    lanes[l] = l;
                var cur = new Vector<long>(first + k) + Unsafe.ReadUnaligned<Vector<long>>(lanes);
                var step = new Vector<long>(w);
                for (; k <= count - w; k += w)
                {
                    Unsafe.Write(dst + k, cur);   // aligned by the peel above
                    cur += step;
                }
            }
            for (; k < count; k++)
                *(long*)(outLine + (at + k) * outStride) = first + k;
        }

        /// <summary>
        ///     Arranges a line in place from a finished selection: the kept values fill the kth's block (the last m
        ///     positions for the top side, the first m for the bottom one) in value order with the worst at kth, and each
        ///     element the block held that was not kept moves into the position a kept value left.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TSide">The selection's end.</typeparam>
        /// <param name="v">The line (contiguous, writable).</param>
        /// <param name="n">The line's length.</param>
        /// <param name="m">The kept set's size, at most 64 (one mask bit per block slot).</param>
        /// <param name="keptV">The kept values, worst first (copies: the line's own slots are overwritten below).</param>
        /// <param name="keptI">Their positions.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void EdgeArrange<T, TSide>(T* v, long n, int m, T* keptV, long* keptI)
            where T : unmanaged, IComparable<T>
            where TSide : struct, IEdgeSide
        {
            long block = default(TSide).Top ? n - m : 0;
            // The block slots already holding a kept value.
            ulong keptInBlock = 0;
            for (int k = 0; k < m; k++)
            {
                ulong slot = (ulong)(keptI[k] - block);
                if (slot < (ulong)m)
                    keptInBlock |= 1UL << (int)slot;
            }
            // Every kept value outside the block leaves a position free, and every block slot not holding a kept value
            // holds an element that has to leave: both counts are m minus the kept values already in the block, so they
            // pair up one to one. The kept values themselves are safe in keptV.
            int free = 0;
            for (int k = 0; k < m; k++)
            {
                long pos = keptI[k];
                if ((ulong)(pos - block) < (ulong)m)
                    continue;
                while (((keptInBlock >> free) & 1) != 0)
                    free++;
                v[pos] = v[block + free];
                free++;
            }
            // The block in value order: ascending from the kth for the top side, ascending up to the kth for the bottom one.
            if (default(TSide).Top)
            {
                for (int k = 0; k < m; k++)
                    v[block + k] = keptV[k];
            }
            else
            {
                for (int k = 0; k < m; k++)
                    v[k] = keptV[m - 1 - k];
            }
        }
    }
}
