using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Utilities;

// =============================================================================
// Default.Reduction.MinMax.Exact.cs — NumPy-EXACT axis AND flat Min / Max
// (plan docs/plans/ndexpr-evaluate.md, "Perf review 2026-09-23", axis min/max lever)
// =============================================================================
//
// np.max(x, axis=k) is np.maximum.reduce: PyUFunc_ReduceWrapper (numpy/_core/src/umath/reduction.c) copies the
// FIRST element along the axis into the freshly allocated result (maximum has no identity), then drives
// loops_minmax.dispatch.c.src::<TYPE>_maximum over the rest through a K-order NpyIter built with
// NPY_ITER_DONT_NEGATE_STRIDES — so the reduced axis is ALWAYS walked in increasing logical index, and the
// iterator's innermost axis is the extent > 1 axis with the smallest |stride| (a stable sort from reversed C
// order, so a tie goes to the later axis; the allocated result has no strides yet and does not vote). Per result
// element that leaves exactly three schedules, and NumPy's answer — for floats, which zero sign survives a ±0 tie
// and whether a NaN comes back canonical or with its payload — depends on which one runs:
//
//   * ROW, contiguous (the reduced axis is innermost and its stride is one element): the inner loop is an
//     IS_BINARY_REDUCE call over the row after its first element, i.e. simd_reduce_c seeded with splat(x[0]) —
//     NumPyMinMaxReduce.Reduce, the same schedule the flat reduction runs (8-vector groups, single vectors, the
//     canonical-NaN horizontal reduce, the SSE scalar tail). Rows are not split at the 8192-element buffer size:
//     no operand needs buffering, and GROWINNER lets the inner loop take the whole row (probed with 20000-long rows).
//   * ROW, strided (innermost reduced axis with any other stride — a stepped or reversed row): the SIMD branch
//     needs is2 == sizeof(T), so the loop falls to its 8-accumulator SCALAR unroll (m0..m7, pairwise combine,
//     out = N(out, m0), then the tail one by one) — ReduceStridedRow. No lane reduce, so no NaN canonicalization.
//   * SLAB (the innermost axis is a non-reduced one): every inner-loop call is ELEMENTWISE, out[i] = N(out[i],
//     x[k, i]) for k = 1, 2, … — simd_binary_ccc / simd_binary / the scalar loop all apply the same per-lane rule, so
//     the result is the plain sequential fold and independent of vectorization: the LAST equal zero wins, the FIRST
//     NaN (with its payload) is kept.
//
// N(a, b) = isnan(a) ? a : (a > b ? a : b)   (min: a < b) — _mm256_max_pd / _mm_max_sd, second operand wins a tie
// and an unordered compare, blended back to a when a is NaN. Integer lanes carry neither ties that differ in bits
// nor NaN: any schedule is exact for them, so for integers this is purely a vectorized reduction.
//
// A broadcast (stride-0, extent > 1) input declines: a zero stride makes NpyIter's axis sort ambiguous for that
// operand, which moves axes differently than the |stride| rule above, and nothing measured needs it.
//
// FLAT (np.max(x) with axis=None) is a different iterator, and NOT "the same thing along every axis": the 0-d result
// has a zero stride everywhere, so it never votes in the axis sort, never blocks coalescing and never reaches an outer
// reduce dimension — NumPy runs the iterator WITHOUT its reduce logic, and npyiter_find_buffering_setup
// (nditer_constr.c) decides on cost alone whether the input is iterated in place or copied into the 8192-element
// (np.getbufsize()) buffer:
//
//   * the axes (extent > 1 only, sorted by |stride| exactly as above) are COALESCED — signed, so a fully reversed
//     block is one run with a negative stride;
//   * ONE run left (any dense block, any evenly strided or reversed view): one inner-loop call over the run after the
//     copied first element — simd_reduce_c when the stride is one element, the 8-accumulator unroll otherwise;
//   * two or more runs: the cost model picks the "outer" dimension. When the innermost run is longer than
//     buffersize / 2 it keeps dimension 0 — the input stays UNBUFFERED and every axis-0 run is its own call (chained:
//     each call seeds with the running result); otherwise it grows the core over as many dimensions as fit and the
//     input is BUFFERED — copied, in iteration order, into chunks of coresize · ⌊buffersize / coresize⌋ elements
//     (never across the outer dimension's end) — and every chunk is ONE contiguous simd_reduce_c call. So a reversed
//     or column-sliced 2-D view is reduced as contiguous chunks, never row by row: the chunk boundaries decide which
//     ±0 survives and where a NaN turns canonical.
//
// Every chained contiguous call re-seeds splat(running result), so a NaN the result already holds comes back as the
// canonical NaN from the next contiguous call however short, while the strided unroll keeps its payload.
// =============================================================================

namespace NumSharp.Backends
{
    internal static partial class NumPyMinMaxReduce
    {
        /// <summary>
        /// One contiguous ROW of an axis reduction — <see cref="Reduce{T,TLane}"/> with a short-row shortcut. When the
        /// row after its first element is shorter than one vector, NumPy's <c>simd_reduce_c</c> folds no vector at
        /// all, so its horizontal reduce sees <c>splat(x[0])</c>: the canonical NaN when <c>x[0]</c> is NaN, else
        /// <c>x[0]</c> itself (every lane is the same value) — computed here without building the vector.
        /// </summary>
        /// <remarks>
        /// The shortcut is what keeps a many-short-rows reduction (e.g. <c>(100000, 3)</c> along axis 1) from paying the
        /// lane spill of <see cref="ReduceLanes{T,TLane}"/> per row; the answer is bit-identical to the vector path.
        /// Note the consequence it preserves: a NaN FIRST element of a row longer than one element comes back as the
        /// canonical NaN (it went through the lane reduce), while a single-element row keeps its payload.
        /// </remarks>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">Max or min.</typeparam>
        /// <param name="x">The row's first element; the row is <paramref name="n"/> consecutive elements.</param>
        /// <param name="n">Row length (&gt;= 1).</param>
        /// <returns>The row's reduction, NumPy's bits.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe T ReduceRow<T, TLane>(T* x, long n)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
            => n == 1
                ? x[0]   // nothing after the copied first element: NumPy never calls the loop
                : ChainContiguous<T, TLane>(x[0], x + 1, n - 1);

        /// <summary>
        /// One <c>simd_reduce_c</c> CALL over a contiguous run: <c>acc = splat(r)</c> — <paramref name="r"/> being the
        /// output's current value (a row's first element, or the running result of a chained flat reduction) — then the
        /// run's <paramref name="len"/> elements through the schedule (<see cref="Finish{T,TLane}"/>).
        /// </summary>
        /// <remarks>
        /// When the run is shorter than one vector no vector op runs, so the horizontal reduce sees <c>splat(r)</c>: the
        /// canonical NaN when <paramref name="r"/> is NaN, else <paramref name="r"/> itself (every lane equal) — computed
        /// without building the vector. The consequence it preserves: EVERY call re-canonicalizes a NaN the output already
        /// holds, so a chained reduction returns the canonical NaN once any contiguous run follows the NaN, however short.
        /// IsNaN is always false for integer <typeparamref name="T"/>, so <see cref="CanonicalNaN{T}"/> is never reached there.
        /// </remarks>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">Max or min.</typeparam>
        /// <param name="r">The output's current value (the seed of the call's accumulator).</param>
        /// <param name="x">The run's first element.</param>
        /// <param name="len">The run's element count (&gt;= 1 — NumPy never makes an empty call).</param>
        /// <returns>The output's value after the call.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe T ChainContiguous<T, TLane>(T r, T* x, long len)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
        {
            if (len >= Vector256<T>.Count)
                return Finish<T, TLane>(Vector256.Create(r), x, len);

            r = T.IsNaN(r) ? CanonicalNaN<T>() : r;
            for (long k = 0; k < len; k++)
                r = N<T, TLane>(r, x[k]);
            return r;
        }

        /// <summary>
        /// One STRIDED row — NumPy's 8-accumulator scalar unroll for a non-contiguous <c>IS_BINARY_REDUCE</c> inner
        /// loop (the SIMD reduce requires a one-element stride, so a stepped or reversed row lands here): with
        /// <c>len = n - 1</c> elements after the copied first one, when <c>len &gt;= 8</c> eight running values
        /// <c>m0..m7</c> take the first eight, fold every further whole group of eight lane by lane, combine as
        /// <c>(0,1)(2,3)(4,5)(6,7) → (0,2)(4,6) → (0,4)</c> and fold into the result; the remaining elements then fold
        /// one by one.
        /// </summary>
        /// <remarks>
        /// All steps use the scalar <see cref="N{T,TLane}(T,T)"/> — no lane reduce, so a NaN keeps its payload. For
        /// floats the grouping is observable (which ±0 survives, which NaN), so it must be exactly this order.
        /// </remarks>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">Max or min.</typeparam>
        /// <param name="x">The row's first element (logical index 0).</param>
        /// <param name="stride">The row's element stride (any value, negative included — the row is walked in
        /// increasing logical index, as NumPy's reduce iterator never negates strides).</param>
        /// <param name="n">Row length (&gt;= 1).</param>
        /// <returns>The row's reduction, NumPy's bits.</returns>
        internal static unsafe T ReduceStridedRow<T, TLane>(T* x, long stride, long n)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
            => ChainStrided<T, TLane>(x[0], x + stride, stride, n - 1);   // element 1 on: the copied first element seeds

        /// <summary>
        /// One strided <c>IS_BINARY_REDUCE</c> CALL — the 8-accumulator scalar unroll — folding a run of
        /// <paramref name="len"/> elements into the output's current value <paramref name="r"/>: when <c>len &gt;= 8</c>,
        /// <c>m0..m7</c> take the run's first eight, fold every further whole group of eight lane by lane, combine as
        /// <c>(0,1)(2,3)(4,5)(6,7) → (0,2)(4,6) → (0,4)</c> and fold into <paramref name="r"/>; the rest then fold one by
        /// one. No lane reduce, so a NaN keeps its payload.
        /// </summary>
        /// <remarks>
        /// The eight running values ARE eight independent lanes — <c>m_j</c> only ever meets elements <c>j, j+8,
        /// j+16, …</c> — so for 4- and 8-byte elements they run as <see cref="Vector256{T}"/> lanes (one vector of eight
        /// 4-byte lanes, or two of four 8-byte lanes) fed by a strided gather (<see cref="GatherRun{T}"/>, one reversed
        /// load for a stride of -1), with the vector form of the same per-lane rule. The combine and the tail stay
        /// scalar, in NumPy's order. Bit-identical to the scalar unroll by construction: the lane op is NumPy's scalar op
        /// (<c>_mm_max_sd</c> + NaN blend) applied per lane, and no lane ever meets another before the combine.
        /// </remarks>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">Max or min.</typeparam>
        /// <param name="r">The output's current value.</param>
        /// <param name="ip">The run's first element.</param>
        /// <param name="stride">The run's element stride (any sign; walked in increasing logical index).</param>
        /// <param name="len">The run's element count (&gt;= 0).</param>
        /// <returns>The output's value after the call.</returns>
        internal static unsafe T ChainStrided<T, TLane>(T r, T* ip, long stride, long len)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
        {
            long i = 0;
            if (len >= 8)
            {
                T m0, m1, m2, m3, m4, m5, m6, m7;
                if (Vector256.IsHardwareAccelerated && sizeof(T) == 4)
                {
                    // m0..m7 = the eight lanes of one vector.
                    var acc = GatherRun(ip, stride);
                    for (i = 8; i + 8 <= len; i += 8)
                        acc = TLane.N(acc, GatherRun(ip + i * stride, stride));
                    m0 = acc.GetElement(0); m1 = acc.GetElement(1); m2 = acc.GetElement(2); m3 = acc.GetElement(3);
                    m4 = acc.GetElement(4); m5 = acc.GetElement(5); m6 = acc.GetElement(6); m7 = acc.GetElement(7);
                }
                else if (Vector256.IsHardwareAccelerated && sizeof(T) == 8)
                {
                    // m0..m3 = the low vector's lanes, m4..m7 the high vector's.
                    var lo = GatherRun(ip, stride);
                    var hi = GatherRun(ip + 4 * stride, stride);
                    for (i = 8; i + 8 <= len; i += 8)
                    {
                        T* g = ip + i * stride;
                        lo = TLane.N(lo, GatherRun(g, stride));
                        hi = TLane.N(hi, GatherRun(g + 4 * stride, stride));
                    }

                    m0 = lo.GetElement(0); m1 = lo.GetElement(1); m2 = lo.GetElement(2); m3 = lo.GetElement(3);
                    m4 = hi.GetElement(0); m5 = hi.GetElement(1); m6 = hi.GetElement(2); m7 = hi.GetElement(3);
                }
                else
                {
                    m0 = ip[0]; m1 = ip[stride]; m2 = ip[2 * stride]; m3 = ip[3 * stride];
                    m4 = ip[4 * stride]; m5 = ip[5 * stride]; m6 = ip[6 * stride]; m7 = ip[7 * stride];
                    for (i = 8; i + 8 <= len; i += 8)
                    {
                        T* g = ip + i * stride;
                        m0 = N<T, TLane>(m0, g[0]);
                        m1 = N<T, TLane>(m1, g[stride]);
                        m2 = N<T, TLane>(m2, g[2 * stride]);
                        m3 = N<T, TLane>(m3, g[3 * stride]);
                        m4 = N<T, TLane>(m4, g[4 * stride]);
                        m5 = N<T, TLane>(m5, g[5 * stride]);
                        m6 = N<T, TLane>(m6, g[6 * stride]);
                        m7 = N<T, TLane>(m7, g[7 * stride]);
                    }
                }

                m0 = N<T, TLane>(m0, m1);
                m2 = N<T, TLane>(m2, m3);
                m4 = N<T, TLane>(m4, m5);
                m6 = N<T, TLane>(m6, m7);
                m0 = N<T, TLane>(m0, m2);
                m4 = N<T, TLane>(m4, m6);
                m0 = N<T, TLane>(m0, m4);
                r = N<T, TLane>(r, m0);
            }

            for (; i < len; i++)
                r = N<T, TLane>(r, ip[i * stride]);
            return r;
        }

        /// <summary>
        /// <c>o[i] = x[i]</c> over a run of <paramref name="n"/> elements — the first-visit copy NumPy's
        /// <c>PyArray_CopyInitialReduceValues</c> makes (a straight copy: a NaN keeps its payload, -0 its sign).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="o">First output element.</param>
        /// <param name="os">Output element stride.</param>
        /// <param name="x">First input element.</param>
        /// <param name="xs">Input element stride.</param>
        /// <param name="n">Run length.</param>
        internal static unsafe void CopyRun<T>(T* o, long os, T* x, long xs, long n) where T : unmanaged
        {
            if (os == 1 && xs == 1)
            {
                Buffer.MemoryCopy(x, o, n * sizeof(T), n * sizeof(T));
                return;
            }

            long i = 0;
            if (os == 1 && Vector256.IsHardwareAccelerated && (sizeof(T) == 4 || sizeof(T) == 8))
            {
                // A strided source into a contiguous destination (the flat reduction's buffer fill): whole vectors
                // gathered — one reversed load for a stride of -1 — then stored. A copy has no order to preserve, so
                // this is exact whatever the width.
                int w = Vector256<T>.Count;
                for (; i + w <= n; i += w)
                    Vector256.Store(GatherRun(x + i * xs, xs), o + i);
            }

            for (; i < n; i++)
                o[i * os] = x[i * xs];
        }

        /// <summary>
        /// The SLAB step: <c>o[i] = N(o[i], x[i])</c> over a run — one reduced-axis index folded into every output of
        /// the run. Per element this is exactly NumPy's elementwise loop (<c>simd_binary_ccc</c> / <c>simd_binary</c> /
        /// scalar all apply the same rule), so the vector body here is exact whatever its width.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">Max or min.</typeparam>
        /// <param name="o">First output element (read-modify-write).</param>
        /// <param name="os">Output element stride.</param>
        /// <param name="x">First input element of this reduced index.</param>
        /// <param name="xs">Input element stride.</param>
        /// <param name="n">Run length.</param>
        internal static unsafe void CombineRun<T, TLane>(T* o, long os, T* x, long xs, long n)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
        {
            long i = 0;
            if (os == 1 && xs == 1)
            {
                // Vector256 only where it is hardware — the portable Vector256 path is correct (the rule is per lane)
                // but a software emulation would be slower than the scalar loop below.
                if (Vector256.IsHardwareAccelerated)
                {
                    int w = Vector256<T>.Count;
                    for (; i + 2 * w <= n; i += 2 * w)
                    {
                        var r0 = TLane.N(Vector256.Load(o + i), Vector256.Load(x + i));
                        var r1 = TLane.N(Vector256.Load(o + i + w), Vector256.Load(x + i + w));
                        Vector256.Store(r0, o + i);
                        Vector256.Store(r1, o + i + w);
                    }

                    for (; i + w <= n; i += w)
                        Vector256.Store(TLane.N(Vector256.Load(o + i), Vector256.Load(x + i)), o + i);
                }

                for (; i < n; i++)
                    o[i] = N<T, TLane>(o[i], x[i]);
                return;
            }

            if (os == 1 && Vector256.IsHardwareAccelerated && (sizeof(T) == 4 || sizeof(T) == 8))
            {
                // A contiguous output run over a STRIDED input run (a stepped or reversed view reduced along an outer
                // axis): gather each vector's lanes with scalar loads — NumPy's own npyv_loadn does exactly this — and
                // apply the same per-lane rule. Elementwise, so any grouping is exact.
                int w = Vector256<T>.Count;
                for (; i + w <= n; i += w)
                    Vector256.Store(TLane.N(Vector256.Load(o + i), GatherRun(x + i * xs, xs)), o + i);
            }

            for (; i < n; i++)
                o[i * os] = N<T, TLane>(o[i * os], x[i * xs]);
        }

        /// <summary>
        /// One vector of a strided run, lanes in logical order <c>x[0], x[s], x[2s], …</c>: a stride of -1 (a reversed
        /// view) is ONE contiguous load ending at <paramref name="x"/> with its lanes reversed; any other stride goes
        /// through <see cref="GatherStrided{T}"/>. Only 4- and 8-byte lanes are served — the caller gates on it.
        /// </summary>
        /// <typeparam name="T">A 4- or 8-byte element type.</typeparam>
        /// <param name="x">The run's first element for this vector (logical lane 0).</param>
        /// <param name="s">The run's element stride (any sign; 1 is served too, as a plain load).</param>
        /// <returns>The gathered vector.</returns>
        /// <exception cref="NotSupportedException">A lane width other than 4 or 8 bytes — a caller bug.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<T> GatherRun<T>(T* x, long s) where T : unmanaged
        {
            if (s == 1)
                return Vector256.Load(x);
            if (s == -1)
                return ReverseLanes(Vector256.Load(x - (Vector256<T>.Count - 1)));
            return GatherStrided(x, s);
        }

        /// <summary>
        /// The lanes of <paramref name="v"/> in reverse order — <c>vpermq 0x1B</c> / <c>vpermd</c> under AVX2, the portable
        /// <see cref="Vector256.Shuffle(Vector256{ulong}, Vector256{ulong})"/> otherwise (a data move, so any form is exact).
        /// </summary>
        /// <typeparam name="T">A 4- or 8-byte element type.</typeparam>
        /// <param name="v">The vector to reverse.</param>
        /// <returns><paramref name="v"/> with lane <c>k</c> moved to lane <c>Count - 1 - k</c>.</returns>
        /// <exception cref="NotSupportedException">A lane width other than 4 or 8 bytes — a caller bug.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<T> ReverseLanes<T>(Vector256<T> v) where T : unmanaged
        {
            if (Unsafe.SizeOf<T>() == 8)
                return Avx2.IsSupported
                    ? Avx2.Permute4x64(v.AsUInt64(), 0b00_01_10_11).As<ulong, T>()
                    : Vector256.Shuffle(v.AsUInt64(), Vector256.Create(3ul, 2, 1, 0)).As<ulong, T>();
            if (Unsafe.SizeOf<T>() == 4)
            {
                var rev = Vector256.Create(7u, 6, 5, 4, 3, 2, 1, 0);
                return Avx2.IsSupported
                    ? Avx2.PermuteVar8x32(v.AsUInt32(), rev).As<uint, T>()
                    : Vector256.Shuffle(v.AsUInt32(), rev).As<uint, T>();
            }

            throw new NotSupportedException("ReverseLanes serves 4- and 8-byte lanes only.");
        }

        /// <summary>
        /// One vector of a strided run: lanes <c>x[0], x[s], x[2s], …</c> built from scalar loads (the JIT lowers the typed
        /// <c>Vector256.Create</c> to load + insert sequences, as NumPy's <c>npyv_loadn</c> does). Only 4- and 8-byte
        /// lanes are served — the caller gates on it.
        /// </summary>
        /// <typeparam name="T">A 4- or 8-byte element type.</typeparam>
        /// <param name="x">The run's first element for this vector.</param>
        /// <param name="s">The run's element stride (any sign).</param>
        /// <returns>The gathered vector.</returns>
        /// <exception cref="NotSupportedException">A lane width other than 4 or 8 bytes — a caller bug.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<T> GatherStrided<T>(T* x, long s) where T : unmanaged
        {
            if (sizeof(T) == 8)
            {
                ulong* p = (ulong*)x;
                return Vector256.Create(p[0], p[s], p[2 * s], p[3 * s]).As<ulong, T>();
            }

            if (sizeof(T) == 4)
            {
                uint* p = (uint*)x;
                return Vector256.Create(p[0], p[s], p[2 * s], p[3 * s], p[4 * s], p[5 * s], p[6 * s], p[7 * s]).As<uint, T>();
            }

            throw new NotSupportedException("GatherStrided serves 4- and 8-byte lanes only.");
        }

        /// <summary>
        /// The whole axis reduction over an arbitrary (non-broadcast) strided input, NumPy's schedule per output element
        /// (see the file header): ROW mode when the reduced axis is the iterator's innermost axis — contiguous rows
        /// through <see cref="ReduceRow{T,TLane}"/>, strided ones through <see cref="ReduceStridedRow{T,TLane}"/> —
        /// otherwise SLAB mode, the first-visit copy then one <see cref="CombineRun{T,TLane}"/> per further reduced index.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The innermost axis is picked by NpyIter's rule for a single voting operand: the extent &gt; 1 axis with the
        /// smallest |stride|, a tie going to the LATER axis (the stable insertion sort starts from reversed C order).
        /// Extent-1 axes never matter (NpyIter zeroes their strides and coalesces them away).
        /// </para>
        /// <para>
        /// Outer axes are walked in memory order (largest |stride| outermost) purely for locality — each output
        /// element's operation sequence is fixed by the mode, not by the walk. In SLAB mode, outer axes that continue
        /// the run contiguously in BOTH input and output are merged into it (NumPy's coalescing does the same), so a
        /// C <c>(K, A, B)</c> reduction along axis 0 folds each slab as one <c>A·B</c> run.
        /// </para>
        /// </remarks>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">Max or min.</typeparam>
        /// <param name="x">Logical element 0 of the input.</param>
        /// <param name="dims">Input extents (rank nd, nd &lt;= 64).</param>
        /// <param name="xs">Input element strides (rank nd); no extent &gt; 1 axis may have stride 0.</param>
        /// <param name="o">Logical element 0 of the output.</param>
        /// <param name="os">Output element strides indexed by INPUT axis (rank nd; the reduced axis' entry is ignored).</param>
        /// <param name="axis">The reduced axis; <c>dims[axis] &gt;= 1</c>.</param>
        internal static unsafe void ReduceAxis<T, TLane>(T* x, ReadOnlySpan<long> dims, ReadOnlySpan<long> xs, T* o,
            ReadOnlySpan<long> os, int axis)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
        {
            int nd = dims.Length;
            long K = dims[axis];
            long sK = xs[axis];

            // NpyIter's innermost axis ("<=": a tie moves to the later axis, as the stable sort from reversed C
            // order leaves it). -1 when no axis has extent > 1 (a single output of a K == 1 reduction).
            int inner = -1;
            long best = long.MaxValue;
            for (int d = 0; d < nd; d++)
            {
                if (dims[d] <= 1)
                    continue;
                long a = Math.Abs(xs[d]);
                if (a <= best)
                {
                    best = a;
                    inner = d;
                }
            }

            bool rows = inner == axis;

            // Run (SLAB mode only): the innermost non-reduced axis, grown by every outer axis that continues it
            // contiguously in both operands.
            long runN = 1, runXs = 1, runOs = 1;
            Span<bool> used = stackalloc bool[nd];
            used.Clear();   // explicit: correctness must not hinge on the method never gaining [SkipLocalsInit]
            if (!rows && inner >= 0)
            {
                runN = dims[inner];
                runXs = xs[inner];
                runOs = os[inner];
                used[inner] = true;
                for (bool grew = true; grew;)
                {
                    grew = false;
                    for (int d = 0; d < nd; d++)
                    {
                        if (used[d] || d == axis || dims[d] <= 1)
                            continue;
                        if (xs[d] == runXs * runN && os[d] == runOs * runN)
                        {
                            runN *= dims[d];
                            used[d] = true;
                            grew = true;
                        }
                    }
                }
            }

            // Outer walk axes: every remaining extent > 1 non-reduced axis, outermost (largest |stride|) first.
            Span<long> wDim = stackalloc long[nd];
            Span<long> wXs = stackalloc long[nd];
            Span<long> wOs = stackalloc long[nd];
            int w = 0;
            for (int d = 0; d < nd; d++)
            {
                if (d == axis || used[d] || dims[d] <= 1)
                    continue;
                long a = Math.Abs(xs[d]);
                int j = w++;
                while (j > 0 && Math.Abs(wXs[j - 1]) < a)
                {
                    wDim[j] = wDim[j - 1];
                    wXs[j] = wXs[j - 1];
                    wOs[j] = wOs[j - 1];
                    j--;
                }

                wDim[j] = dims[d];
                wXs[j] = xs[d];
                wOs[j] = os[d];
            }

            long total = 1;
            for (int j = 0; j < w; j++)
                total *= wDim[j];

            Span<long> coord = stackalloc long[Math.Max(w, 1)];
            coord.Clear();
            long offX = 0, offO = 0;
            for (long it = 0; it < total; it++)
            {
                T* xb = x + offX;
                T* ob = o + offO;
                if (rows)
                {
                    *ob = sK == 1 ? ReduceRow<T, TLane>(xb, K) : ReduceStridedRow<T, TLane>(xb, sK, K);
                }
                else
                {
                    CopyRun(ob, runOs, xb, runXs, runN);
                    for (long k = 1; k < K; k++)
                        CombineRun<T, TLane>(ob, runOs, xb + k * sK, runXs, runN);
                }

                // Odometer over the walk axes, innermost (last) fastest.
                for (int j = w - 1; j >= 0; j--)
                {
                    if (++coord[j] < wDim[j])
                    {
                        offX += wXs[j];
                        offO += wOs[j];
                        break;
                    }

                    coord[j] = 0;
                    offX -= (wDim[j] - 1) * wXs[j];
                    offO -= (wDim[j] - 1) * wOs[j];
                }
            }
        }

        /// <summary>
        /// Dtype dispatch for <see cref="ReduceAxis{T,TLane}"/> over raw byte pointers.
        /// </summary>
        /// <param name="t">The element dtype (must satisfy <see cref="Supports"/>).</param>
        /// <param name="isMax">Max when true, min when false.</param>
        /// <param name="x">Logical element 0 of the input.</param>
        /// <param name="dims">Input extents.</param>
        /// <param name="xs">Input element strides.</param>
        /// <param name="o">Logical element 0 of the output.</param>
        /// <param name="os">Output element strides indexed by input axis (reduced entry ignored).</param>
        /// <param name="axis">The reduced axis.</param>
        /// <exception cref="NotSupportedException"><paramref name="t"/> is not a supported dtype.</exception>
        internal static unsafe void ReduceAxis(NPTypeCode t, bool isMax, byte* x, ReadOnlySpan<long> dims, ReadOnlySpan<long> xs,
            byte* o, ReadOnlySpan<long> os, int axis)
        {
            switch (t)
            {
                case NPTypeCode.Double: if (isMax) ReduceAxis<double, MaxLane<double>>((double*)x, dims, xs, (double*)o, os, axis); else ReduceAxis<double, MinLane<double>>((double*)x, dims, xs, (double*)o, os, axis); break;
                case NPTypeCode.Single: if (isMax) ReduceAxis<float, MaxLane<float>>((float*)x, dims, xs, (float*)o, os, axis); else ReduceAxis<float, MinLane<float>>((float*)x, dims, xs, (float*)o, os, axis); break;
                case NPTypeCode.SByte: if (isMax) ReduceAxis<sbyte, MaxLane<sbyte>>((sbyte*)x, dims, xs, (sbyte*)o, os, axis); else ReduceAxis<sbyte, MinLane<sbyte>>((sbyte*)x, dims, xs, (sbyte*)o, os, axis); break;
                case NPTypeCode.Byte: if (isMax) ReduceAxis<byte, MaxLane<byte>>(x, dims, xs, o, os, axis); else ReduceAxis<byte, MinLane<byte>>(x, dims, xs, o, os, axis); break;
                case NPTypeCode.Int16: if (isMax) ReduceAxis<short, MaxLane<short>>((short*)x, dims, xs, (short*)o, os, axis); else ReduceAxis<short, MinLane<short>>((short*)x, dims, xs, (short*)o, os, axis); break;
                case NPTypeCode.UInt16: if (isMax) ReduceAxis<ushort, MaxLane<ushort>>((ushort*)x, dims, xs, (ushort*)o, os, axis); else ReduceAxis<ushort, MinLane<ushort>>((ushort*)x, dims, xs, (ushort*)o, os, axis); break;
                case NPTypeCode.Int32: if (isMax) ReduceAxis<int, MaxLane<int>>((int*)x, dims, xs, (int*)o, os, axis); else ReduceAxis<int, MinLane<int>>((int*)x, dims, xs, (int*)o, os, axis); break;
                case NPTypeCode.UInt32: if (isMax) ReduceAxis<uint, MaxLane<uint>>((uint*)x, dims, xs, (uint*)o, os, axis); else ReduceAxis<uint, MinLane<uint>>((uint*)x, dims, xs, (uint*)o, os, axis); break;
                case NPTypeCode.Int64: if (isMax) ReduceAxis<long, MaxLane<long>>((long*)x, dims, xs, (long*)o, os, axis); else ReduceAxis<long, MinLane<long>>((long*)x, dims, xs, (long*)o, os, axis); break;
                case NPTypeCode.UInt64: if (isMax) ReduceAxis<ulong, MaxLane<ulong>>((ulong*)x, dims, xs, (ulong*)o, os, axis); else ReduceAxis<ulong, MinLane<ulong>>((ulong*)x, dims, xs, (ulong*)o, os, axis); break;
                default: throw new NotSupportedException($"NumPy axis min/max schedule: dtype {t} is not served.");
            }
        }

        // -------------------------------------------------------------------------------------------------------------
        // FLAT (all axes reduced) — NumPy's iterator reproduced: axis order, coalescing, and the buffering decision.
        // -------------------------------------------------------------------------------------------------------------

        /// <summary>
        /// Bytes of stack scratch <see cref="ReduceFlatBuffered{T,TLane}"/> uses before borrowing a pooled buffer —
        /// enough for every small flat reduction (256 float64), so only a large strided one touches the pool.
        /// </summary>
        private const int FlatStackScratchBytes = 2048;

        /// <summary>
        /// One inner-loop call of the flat reduction, dispatched as NumPy's loop dispatches it: a one-element stride is the
        /// SIMD reduce (<see cref="ChainContiguous{T,TLane}"/>), any other stride the 8-accumulator unroll
        /// (<see cref="ChainStrided{T,TLane}"/>); an empty call is skipped, as NumPy's <c>count &gt; 0</c> guard skips it.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">Max or min.</typeparam>
        /// <param name="r">The running result.</param>
        /// <param name="p">The call's first element.</param>
        /// <param name="stride">The call's element stride (any sign).</param>
        /// <param name="len">The call's element count (&lt;= 0 means no call).</param>
        /// <returns>The running result after the call.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe T Chain<T, TLane>(T r, T* p, long stride, long len)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
            => len <= 0 ? r
                : stride == 1 ? ChainContiguous<T, TLane>(r, p, len)
                : ChainStrided<T, TLane>(r, p, stride, len);

        /// <summary>
        /// NpyIter's axis setup for a FLAT reduction of one non-broadcast operand: the extent &gt; 1 axes in iteration
        /// order (innermost first), coalesced — the axes <c>npyiter_find_best_axis_ordering</c> and
        /// <c>npyiter_coalesce_axes</c> leave behind for <c>np.max(x)</c>.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Order: a stable insertion sort by |stride| starting from reversed C order, so the smallest |stride| is
        /// innermost and a tie keeps the LATER logical axis inner. Only the input votes — the result has a zero stride on
        /// every axis, and NpyIter skips a comparison unless both strides are non-zero. NpyIter gives extent-1 axes a zero
        /// stride too, so they neither vote nor block coalescing; dropping them up front is equivalent.
        /// </para>
        /// <para>
        /// Coalescing: the next axis merges into the running innermost one when <c>stride · extent == next stride</c> —
        /// SIGNED, which is how a fully reversed block becomes one run with a negative stride (NumPy's reduce iterator
        /// never negates strides, so every axis is walked in increasing logical index).
        /// </para>
        /// </remarks>
        /// <param name="dims">Operand extents.</param>
        /// <param name="xs">Operand element strides (no extent &gt; 1 axis may have stride 0 — the caller declines
        /// broadcast operands).</param>
        /// <param name="cd">Receives the coalesced extents, innermost first (capacity &gt;= <c>dims.Length</c>).</param>
        /// <param name="cs">Receives the coalesced element strides, innermost first (same capacity).</param>
        /// <returns>The number of coalesced axes written; 0 when the operand holds a single element.</returns>
        internal static int FlatIteratorAxes(ReadOnlySpan<long> dims, ReadOnlySpan<long> xs, Span<long> cd, Span<long> cs)
        {
            // Stable insertion sort, fed in reversed C order: each axis moves inward past every placed axis with a
            // STRICTLY larger |stride| and stops at the first one that is not larger, so equal |strides| keep their
            // reversed-C order (NumPy's shouldswap stays 0 on "<=").
            int m = 0;
            for (int d = dims.Length - 1; d >= 0; d--)
            {
                if (dims[d] <= 1)
                    continue;
                long a = Math.Abs(xs[d]);
                int j = m++;
                while (j > 0 && Math.Abs(cs[j - 1]) > a)
                {
                    cd[j] = cd[j - 1];
                    cs[j] = cs[j - 1];
                    j--;
                }

                cd[j] = dims[d];
                cs[j] = xs[d];
            }

            // NumPy's pairwise sweep, compressing into the running axis: the merged extent grows, the stride stays the
            // innermost one's. In place — the write index never passes the read index.
            int c = 0;
            for (int k = 0; k < m; k++)
            {
                if (c > 0 && cs[c - 1] * cd[c - 1] == cs[k])
                {
                    cd[c - 1] *= cd[k];
                    continue;
                }

                cd[c] = cd[k];
                cs[c] = cs[k];
                c++;
            }

            return c;
        }

        /// <summary>
        /// <c>npyiter_find_buffering_setup</c>'s cost model for a flat reduction whose operand did NOT coalesce to one run
        /// (<paramref name="cd"/> has at least two axes): the "outer" dimension NumPy picks. 0 means the input is iterated
        /// in place, one inner loop per axis-0 run; anything larger means the input is BUFFERED, the core being every axis
        /// below the returned one.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The model, specialized: <c>cost = 1 + buffered operands</c>. The 0-d result stays single-strided through every
        /// dimension (all its strides are 0), so it never adds cost and never becomes an outer reduce dimension — NumPy
        /// runs this iterator without its reduce logic. The input, already coalesced, stops being single-strided at
        /// dimension 1, so cost is 2 from there on. Dimension <c>i</c> replaces the best one when
        /// <c>cost · bestSize &lt;= bestCost · min(size, maximumSize)</c>; the scan stops once the running size reaches
        /// <paramref name="maximumSize"/> (GROWINNER lets it continue only while nothing is buffered, i.e. at dimension 1).
        /// </para>
        /// <para>
        /// Consequence: an innermost run longer than half the buffer keeps dimension 0 (row by row, in place); a shorter
        /// one is buffered with its neighbours — which is why the flat schedule depends on <see cref="np.getbufsize"/>
        /// exactly as NumPy's does on <c>np.setbufsize</c>.
        /// </para>
        /// </remarks>
        /// <param name="cd">Coalesced extents, innermost first (length &gt;= 2).</param>
        /// <param name="maximumSize">The buffer size in elements (<c>np.getbufsize()</c>).</param>
        /// <param name="coreSize">The core (product of the extents below the chosen dimension); 1 for dimension 0.</param>
        /// <param name="bestSize">The core times the chosen dimension's extent — the elements one outer index spans.</param>
        /// <returns>The chosen outer dimension.</returns>
        internal static int FlatBufferingDim(ReadOnlySpan<long> cd, long maximumSize, out long coreSize, out long bestSize)
        {
            int cost = 1;
            long size = cd[0];
            int bestDim = 0, bestCost = 1;
            bestSize = size;
            coreSize = 1;
            for (int idim = 1; idim < cd.Length; idim++)
            {
                // "Exceeded buffer size, can only improve without buffers and growinner": GROWINNER is set, so only a
                // buffered operand (cost > 1, i.e. from dimension 2 on) stops the growth.
                if (size >= maximumSize && cost > 1)
                    break;

                // The coalesced input stops collapsing here (at dimension 1 — later dimensions add nothing more).
                cost = 2;
                long core = size;
                size *= cd[idim];

                // A buffered iteration can use at most maximumSize of it.
                double bufsize = size > maximumSize ? maximumSize : size;
                if (cost * (double)bestSize <= bestCost * bufsize)
                {
                    bestCost = cost;
                    coreSize = core;
                    bestSize = size;
                    bestDim = idim;
                }
            }

            return bestDim;
        }

        /// <summary>
        /// The whole FLAT reduction of an arbitrary non-broadcast strided operand, with NumPy's schedule (the file
        /// header's FLAT section): element 0 seeds the result, then — after <see cref="FlatIteratorAxes"/> — one call over
        /// the single coalesced run, or <see cref="FlatBufferingDim"/>'s choice between one call per axis-0 run in place
        /// (<see cref="ReduceFlatRows{T,TLane}"/>) and contiguous calls over buffered chunks
        /// (<see cref="ReduceFlatBuffered{T,TLane}"/>).
        /// </summary>
        /// <remarks>
        /// Integers carry no bit-distinct ties and no NaN, so for them every schedule is exact and this is purely the
        /// vectorized route; for floats the chosen calls decide which ±0 survives a tie and whether a NaN comes back
        /// canonical, so the choice must be NumPy's to the element.
        /// </remarks>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">Max or min.</typeparam>
        /// <param name="x">Logical element 0 of the operand.</param>
        /// <param name="dims">Operand extents (rank &lt;= 64; an empty span is a 0-d operand).</param>
        /// <param name="xs">Operand element strides (non-broadcast).</param>
        /// <param name="maximumSize">The buffer size in elements (<c>np.getbufsize()</c>).</param>
        /// <returns>The reduction, NumPy's bits.</returns>
        internal static unsafe T ReduceFlat<T, TLane>(T* x, ReadOnlySpan<long> dims, ReadOnlySpan<long> xs, long maximumSize)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
        {
            int nd = dims.Length;
            Span<long> cd = stackalloc long[Math.Max(nd, 1)];
            Span<long> cs = stackalloc long[Math.Max(nd, 1)];
            int c = FlatIteratorAxes(dims, xs, cd, cs);

            // PyArray_CopyInitialReduceValues: the result starts as element 0, bits untouched (a NaN keeps its payload).
            T r = x[0];
            if (c == 0)
                return r;   // a single element: the loop is never called

            // One run (the coalesced whole): one call over everything after the copied element — nothing buffers, and
            // GROWINNER lets the inner loop span the whole run however long.
            if (c == 1)
                return Chain<T, TLane>(r, x + cs[0], cs[0], cd[0] - 1);

            var axes = cd[..c];
            var strides = cs[..c];
            int bestDim = FlatBufferingDim(axes, maximumSize, out long coreSize, out long bestSize);
            if (bestDim == 0)
                return ReduceFlatRows<T, TLane>(x, axes, strides);

            // Buffered: a fill holds whole cores, capped to the buffer size (rounded down to a multiple of the core).
            long chunk = bestSize > maximumSize ? coreSize * Math.Max(1, maximumSize / coreSize) : bestSize;
            return ReduceFlatBuffered<T, TLane>(x, axes, strides, bestSize, chunk);
        }

        /// <summary>
        /// The UNBUFFERED multi-run flat schedule: one inner-loop call per axis-0 run in iteration order, each seeded with
        /// the running result, the first skipping the copied element.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">Max or min.</typeparam>
        /// <param name="x">Logical element 0 of the operand.</param>
        /// <param name="cd">Coalesced extents, innermost first (length &gt;= 2, rank &lt;= 64).</param>
        /// <param name="cs">Coalesced element strides, innermost first.</param>
        /// <returns>The reduction.</returns>
        private static unsafe T ReduceFlatRows<T, TLane>(T* x, ReadOnlySpan<long> cd, ReadOnlySpan<long> cs)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
        {
            int c = cd.Length;
            long n0 = cd[0], s0 = cs[0];
            Span<long> coord = stackalloc long[c];
            coord.Clear();   // explicit: the odometer must start at the origin whatever the method's locals-init policy

            T r = Chain<T, TLane>(x[0], x + s0, s0, n0 - 1);
            long off = 0;
            while (true)
            {
                // Odometer over the axes above the run, axis 1 fastest (iteration order).
                int j = 1;
                for (; j < c; j++)
                {
                    if (++coord[j] < cd[j])
                    {
                        off += cs[j];
                        break;
                    }

                    coord[j] = 0;
                    off -= (cd[j] - 1) * cs[j];
                }

                if (j == c)
                    return r;
                r = Chain<T, TLane>(r, x + off, s0, n0);
            }
        }

        /// <summary>
        /// The BUFFERED multi-run flat schedule: the operand's elements, in iteration order, are copied into chunks and
        /// every chunk is one contiguous reduce call (<see cref="ChainContiguous{T,TLane}"/>) — exactly NumPy's buffer
        /// fills. A chunk holds <paramref name="chunkElems"/> elements (whole axis-0 runs), except the last of each outer
        /// block, which ends where the block does (NumPy never fills a buffer across the outer dimension's end).
        /// </summary>
        /// <remarks>
        /// The scratch is <see cref="FlatStackScratchBytes"/> of stack when a chunk fits, otherwise a buffer borrowed from
        /// <see cref="SizeBucketedBufferPool"/> and returned on every exit (a raw allocation here would be invisible to
        /// the pool accounting — the chokepoint gate forbids it). The copy is the same one NumPy makes; the reduce then
        /// reads it from L1/L2.
        /// </remarks>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">Max or min.</typeparam>
        /// <param name="x">Logical element 0 of the operand.</param>
        /// <param name="cd">Coalesced extents, innermost first (length &gt;= 2, rank &lt;= 64).</param>
        /// <param name="cs">Coalesced element strides, innermost first.</param>
        /// <param name="blockElems">Elements one outer index spans (the core times the outer extent).</param>
        /// <param name="chunkElems">Elements one buffer fill holds (a multiple of the axis-0 extent).</param>
        /// <returns>The reduction.</returns>
        [SkipLocalsInit]
        private static unsafe T ReduceFlatBuffered<T, TLane>(T* x, ReadOnlySpan<long> cd, ReadOnlySpan<long> cs,
            long blockElems, long chunkElems)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
        {
            int c = cd.Length;
            long n0 = cd[0], s0 = cs[0];
            long rowsPerBlock = blockElems / n0;
            long rowsPerChunk = chunkElems / n0;

            long bytes = chunkElems * sizeof(T);
            byte* stackScratch = stackalloc byte[FlatStackScratchBytes];
            // A chunk too big for the stack scratch borrows a pooled buffer (never a raw allocation — the
            // chokepoint gate pins those): at most np.getbufsize() elements, so the pool's size buckets serve it
            // warm on the next call. Returned in the finally below on every exit, the reduction's own included.
            T* buf = bytes <= FlatStackScratchBytes ? (T*)stackScratch : (T*)SizeBucketedBufferPool.Take(bytes);
            try
            {
                Span<long> coord = stackalloc long[c];
                coord.Clear();   // [SkipLocalsInit]: the odometer's origin must be written, not assumed

                T r = x[0];
                bool first = true;
                long off = 0, rowInBlock = 0, rowsInChunk = 0;
                T* d = buf;
                while (true)
                {
                    CopyRun(d, 1, x + off, s0, n0);
                    d += n0;
                    rowsInChunk++;
                    rowInBlock++;

                    bool blockEnd = rowInBlock == rowsPerBlock;
                    if (rowsInChunk == rowsPerChunk || blockEnd)
                    {
                        // One buffer fill = one contiguous call. The first skips the copied element (NumPy's
                        // skip_first_count), which is always the first element of the first fill.
                        long t = rowsInChunk * n0;
                        r = first
                            ? Chain<T, TLane>(r, buf + 1, 1, t - 1)
                            : ChainContiguous<T, TLane>(r, buf, t);
                        first = false;
                        rowsInChunk = 0;
                        d = buf;
                        if (blockEnd)
                            rowInBlock = 0;
                    }

                    // Next axis-0 run in iteration order (axis 1 fastest).
                    int j = 1;
                    for (; j < c; j++)
                    {
                        if (++coord[j] < cd[j])
                        {
                            off += cs[j];
                            break;
                        }

                        coord[j] = 0;
                        off -= (cd[j] - 1) * cs[j];
                    }

                    if (j == c)
                        return r;   // the last run always ends a block, so its fill was reduced above
                }
            }
            finally
            {
                if (buf != (T*)stackScratch)
                    SizeBucketedBufferPool.Return((IntPtr)buf, bytes);
            }
        }

        /// <summary>
        /// Dtype dispatch for <see cref="ReduceFlat{T,TLane}"/>: reduce the operand at <paramref name="x"/> and write the
        /// result (at dtype <paramref name="t"/>) to offset 0 of <paramref name="slot"/>.
        /// </summary>
        /// <param name="t">The element dtype (must satisfy <see cref="Supports"/>).</param>
        /// <param name="isMax">Max when true, min when false.</param>
        /// <param name="x">Logical element 0 of the operand.</param>
        /// <param name="dims">Operand extents.</param>
        /// <param name="xs">Operand element strides (non-broadcast).</param>
        /// <param name="maximumSize">The buffer size in elements (<c>np.getbufsize()</c>).</param>
        /// <param name="slot">Receives the result (at least 8 bytes).</param>
        /// <exception cref="NotSupportedException"><paramref name="t"/> is not a supported dtype.</exception>
        internal static unsafe void ReduceFlat(NPTypeCode t, bool isMax, byte* x, ReadOnlySpan<long> dims, ReadOnlySpan<long> xs,
            long maximumSize, byte* slot)
        {
            switch (t)
            {
                case NPTypeCode.Double: *(double*)slot = isMax ? ReduceFlat<double, MaxLane<double>>((double*)x, dims, xs, maximumSize) : ReduceFlat<double, MinLane<double>>((double*)x, dims, xs, maximumSize); break;
                case NPTypeCode.Single: *(float*)slot = isMax ? ReduceFlat<float, MaxLane<float>>((float*)x, dims, xs, maximumSize) : ReduceFlat<float, MinLane<float>>((float*)x, dims, xs, maximumSize); break;
                case NPTypeCode.SByte: *(sbyte*)slot = isMax ? ReduceFlat<sbyte, MaxLane<sbyte>>((sbyte*)x, dims, xs, maximumSize) : ReduceFlat<sbyte, MinLane<sbyte>>((sbyte*)x, dims, xs, maximumSize); break;
                case NPTypeCode.Byte: *slot = isMax ? ReduceFlat<byte, MaxLane<byte>>(x, dims, xs, maximumSize) : ReduceFlat<byte, MinLane<byte>>(x, dims, xs, maximumSize); break;
                case NPTypeCode.Int16: *(short*)slot = isMax ? ReduceFlat<short, MaxLane<short>>((short*)x, dims, xs, maximumSize) : ReduceFlat<short, MinLane<short>>((short*)x, dims, xs, maximumSize); break;
                case NPTypeCode.UInt16: *(ushort*)slot = isMax ? ReduceFlat<ushort, MaxLane<ushort>>((ushort*)x, dims, xs, maximumSize) : ReduceFlat<ushort, MinLane<ushort>>((ushort*)x, dims, xs, maximumSize); break;
                case NPTypeCode.Int32: *(int*)slot = isMax ? ReduceFlat<int, MaxLane<int>>((int*)x, dims, xs, maximumSize) : ReduceFlat<int, MinLane<int>>((int*)x, dims, xs, maximumSize); break;
                case NPTypeCode.UInt32: *(uint*)slot = isMax ? ReduceFlat<uint, MaxLane<uint>>((uint*)x, dims, xs, maximumSize) : ReduceFlat<uint, MinLane<uint>>((uint*)x, dims, xs, maximumSize); break;
                case NPTypeCode.Int64: *(long*)slot = isMax ? ReduceFlat<long, MaxLane<long>>((long*)x, dims, xs, maximumSize) : ReduceFlat<long, MinLane<long>>((long*)x, dims, xs, maximumSize); break;
                case NPTypeCode.UInt64: *(ulong*)slot = isMax ? ReduceFlat<ulong, MaxLane<ulong>>((ulong*)x, dims, xs, maximumSize) : ReduceFlat<ulong, MinLane<ulong>>((ulong*)x, dims, xs, maximumSize); break;
                default: throw new NotSupportedException($"NumPy flat min/max schedule: dtype {t} is not served.");
            }
        }
    }

    public partial class DefaultEngine
    {
        /// <summary>
        /// Axis <c>Min</c> / <c>Max</c> of an existing array with NumPy's exact per-element schedule
        /// (<see cref="NumPyMinMaxReduce.ReduceAxis{T,TLane}"/>), into a fresh result allocated in NumPy's K order
        /// (<see cref="AllocateReductionResult"/> — F for an F-contiguous input). Shared by <c>np.max</c> / <c>np.min</c>
        /// (<see cref="ReduceAMax"/> / <see cref="ReduceAMin"/>) and np.evaluate's axis Min / Max over a bare leaf or a
        /// materialized child.
        /// </summary>
        /// <remarks>
        /// Declines (null, nothing allocated) when <see cref="NDExpr.DisableExactMinMax"/> is set, for a dtype outside
        /// <see cref="NumPyMinMaxReduce.Supports"/> (Half, Char, Decimal, Complex, Boolean keep their kernels), for a
        /// broadcast input (see the file header), for rank &gt; 64, and for an empty input or result — the callers'
        /// own empty handling (NumPy's "zero-size array to reduction" error, the empty result) stays in charge.
        /// Consequence for floats: the ±0 tie and NaN-payload bits equal NumPy's on every non-broadcast layout —
        /// the previous kernels matched only the value.
        /// </remarks>
        /// <param name="arr">The array to reduce (any non-broadcast layout; views honored through offset and strides).</param>
        /// <param name="axis">The already-normalized reduction axis.</param>
        /// <param name="isMax">Max when true, min when false.</param>
        /// <returns>The fresh reduced array (the reduced axis removed; a 0-d result for a 1-D input), or null.</returns>
        internal unsafe NDArray TryExactAxisMinMax(NDArray arr, int axis, bool isMax)
        {
            if (NDExpr.DisableExactMinMax)
                return null;
            var t = arr.GetTypeCode;
            var shape = arr.Shape;
            int nd = shape.NDim;
            if (!NumPyMinMaxReduce.Supports(t) || shape.IsBroadcasted || nd == 0 || nd > 64 || shape.size == 0)
                return null;

            var dims = shape.dimensions;
            var outputDims = new long[nd - 1];
            for (int d = 0, od = 0; d < nd; d++)
                if (d != axis)
                    outputDims[od++] = dims[d];

            var result = AllocateReductionResult(t, outputDims, shape);
            try
            {
                // Output strides indexed by INPUT axis (the reduced axis' entry is never read).
                Span<long> os = stackalloc long[nd];
                var rs = result.Shape.strides;
                for (int d = 0, od = 0; d < nd; d++)
                    os[d] = d == axis ? 0 : (outputDims.Length == 0 ? 0 : rs[od++]);

                int es = t.SizeOf();
                // Logical element 0 of each: a contiguous slice re-seats Address (offset 0), a strided view keeps its
                // base with a non-zero offset — Address + offset·itemsize is right for both.
                byte* x = (byte*)arr.Address + shape.offset * es;
                byte* o = (byte*)result.Address + result.Shape.offset * es;
                NumPyMinMaxReduce.ReduceAxis(t, isMax, x, dims, shape.strides, o, os, axis);
                NDExpr.ExactMinMaxRuns++;
                return result;
            }
            catch
            {
                // A failing reduction must not strand the fresh buffer.
                result.Dispose();
                throw;
            }
        }

        /// <summary>
        /// FLAT <c>Min</c> / <c>Max</c> (<c>axis=None</c>) of an existing array with NumPy's exact schedule
        /// (<see cref="NumPyMinMaxReduce.ReduceFlat{T,TLane}"/> — coalesced runs, NumPy's buffering decision, chained
        /// contiguous / strided calls), written to <paramref name="slot"/>. Shared by <c>np.max</c> / <c>np.min</c>
        /// (<see cref="ReduceAMax"/> / <see cref="ReduceAMin"/>) and np.evaluate's flat <c>Min</c> / <c>Max</c> over a
        /// bare leaf that is not one dense block.
        /// </summary>
        /// <remarks>
        /// Declines (false, <paramref name="slot"/> untouched — the caller keeps its previous kernel) when
        /// <see cref="NDExpr.DisableExactMinMax"/> is set, for a dtype outside <see cref="NumPyMinMaxReduce.Supports"/>, for
        /// a broadcast input (a zero stride changes NpyIter's axis sort — the ambiguous-comparison rule — which this does
        /// not port), for rank &gt; 64 and for an empty array (the caller's "zero-size array to reduction" handling stays in
        /// charge). A buffered layout (see the file header) copies its chunks through a scratch of at most
        /// <see cref="np.getbufsize"/> elements — the copy NumPy makes too.
        /// </remarks>
        /// <param name="arr">The array to reduce (any non-broadcast layout; views honored through offset and strides).</param>
        /// <param name="isMax">Max when true, min when false.</param>
        /// <param name="slot">Receives the result at the array's dtype (at least 8 bytes).</param>
        /// <returns>True when the reduction was computed here.</returns>
        internal unsafe bool TryExactFlatMinMaxArray(NDArray arr, bool isMax, byte* slot)
        {
            if (NDExpr.DisableExactMinMax)
                return false;
            var t = arr.GetTypeCode;
            var shape = arr.Shape;
            if (!NumPyMinMaxReduce.Supports(t) || shape.IsBroadcasted || shape.NDim > 64 || shape.size == 0)
                return false;

            // Logical element 0 (Address + offset·itemsize is right for a re-seated contiguous slice and a strided view).
            byte* x = (byte*)arr.Address + shape.offset * t.SizeOf();
            NumPyMinMaxReduce.ReduceFlat(t, isMax, x, shape.dimensions, shape.strides, np.getbufsize(), slot);
            NDExpr.ExactMinMaxRuns++;
            return true;
        }

        /// <summary>
        /// <see cref="TryExactFlatMinMaxArray(NDArray, bool, byte*)"/> as a fresh 0-d array of the input's dtype — the shape
        /// <c>np.max(x)</c> returns before its keepdims / scalar-marking step.
        /// </summary>
        /// <param name="arr">The array to reduce.</param>
        /// <param name="isMax">Max when true, min when false.</param>
        /// <returns>The 0-d result, or null when the exact schedule declined (nothing allocated).</returns>
        internal unsafe NDArray TryExactFlatMinMaxScalar(NDArray arr, bool isMax)
        {
            ulong slot = 0;   // 8 bytes: the widest served dtype
            if (!TryExactFlatMinMaxArray(arr, isMax, (byte*)&slot))
                return null;

            var t = arr.GetTypeCode;
            var r = new NDArray(t, Shape.Scalar, false);
            Buffer.MemoryCopy(&slot, (byte*)r.Address + r.Shape.offset * t.SizeOf(), t.SizeOf(), t.SizeOf());
            return r;
        }
    }
}
