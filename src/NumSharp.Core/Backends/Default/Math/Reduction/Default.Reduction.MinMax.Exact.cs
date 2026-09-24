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
//
// THE P RULES — np.nanmax / np.nanmin (MinMaxOp.FMax / FMin). For a float32/float64 ndarray NumPy's nanmax IS
// np.fmax.reduce (numpy/lib/_nanfunctions_impl.py), so it runs the SAME iterator and the SAME three schedules as
// np.max — only the loop is <TYPE>_fmax, and that loop has TWO different per-element ops where maximum has one:
//
//   * the VECTOR op, npyv_maxp  = blendv(a, max(a, b), ord(b, b)) — lane-wise: a NaN b keeps a (so of two NaNs the
//     FIRST survives, payload intact), otherwise max(a, b), whose tie keeps b (the LAST of equal zeros wins);
//   * the SCALAR op, npy_fmax   = the MSVC CRT fmax — a NaN a gives b, a NaN b gives a, two NaNs give the LAST, and an
//     ordered tie returns the bitwise AND of the operands for fmax (+0 beats -0 whatever the order) and the bitwise OR
//     for fmin (-0 beats +0);
//   * the lane reduce npyv_reduce_maxp: an all-NaN vector returns LANE 0 AS IS (no canonical NaN — unlike the N rules'
//     reduce), otherwise NaN lanes are blanked to -inf (+inf for minp) and the lanes max-reduced.
//
// So for the P rules the SCHEDULE decides not only which ±0 survives but which of several NaNs comes back from an
// all-NaN slice — and a NaN is never canonicalized anywhere. Per schedule:
//
//   * FLAT and ROW-contiguous: simd_reduce_c with the P lane reduce — the same NumPyMinMaxReduce code the N rules run,
//     TLane.PropagatesNaN == false switching off every canonicalization (ChainContiguous, ReduceLanes). Only the vector
//     section differs in HOW it is computed: maxp is an associative selection, so NumPy's group tree + single-vector loop
//     equal one order-free per-lane fold, which NumPyMinMaxReduce.FoldVectorsP runs as an aligned backward scan (one
//     plain vmaxp per vector) — the same bits, without the per-group NaN proof and the split loads of NumPy's x[1:] start;
//   * ROW-strided: the 8-accumulator unroll with the SCALAR op (ChainStrided folds through TLane.SV / TLane.S, the CRT
//     rule lane by lane);
//   * SLAB: the elementwise calls apply the VECTOR op to the first L - L % lanes elements of every call and the SCALAR
//     op to the rest, so the per-element answer depends on where NumPy's calls CUT the region below the reduced axis.
//     SlabCallTiling.Build ports that cut (npyiter_find_buffering_setup: the reduce-outer decision, the per-operand
//     buffering, the buffer capped to whole cores, np.getbufsize() included) and Segment returns each call's vector /
//     scalar boundary; CombineRunTiled / CombineRun8Tiled fold every piece with its call's op (VectorOpStep /
//     ScalarOpStep). A float32 input iterated IN PLACE with a negative innermost stride runs every call scalar
//     (SlabCallTiling.AllScalar: npyv_loadable_stride_f32 divides the signed stride by an unsigned size).
//
// Both P ops are associative SELECTIONS (see CombineRun8), so the eight-slab balanced tree stays exact under either.
// float16 and complex128 are NOT here: NumPy's HALF_fmax / CDOUBLE_fmax are plain sequential BINARY_LOOPs with their
// own comparisons, ported in Default.Reduction.Nan.Sequential.cs. A broadcast input declines to the old value-exact
// kernels (its zero stride makes NpyIter's axis sort ambiguous — the same gap np.max has), as do rank > 64 and dtypes
// other than float32/float64 (integers never reach here: nanmax of an integer array IS np.max).
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
        /// <typeparam name="TLane">The lane rule (max / min, or the NaN-suppressing fmax / fmin).</typeparam>
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
        /// When the run is shorter than one vector no vector op runs, so the horizontal reduce sees <c>splat(r)</c>: for the
        /// N rules the canonical NaN when <paramref name="r"/> is NaN, else <paramref name="r"/> itself (every lane equal) —
        /// computed without building the vector. The consequence it preserves: EVERY <c>np.max</c> call re-canonicalizes a
        /// NaN the output already holds, so a chained reduction returns the canonical NaN once any contiguous run follows the
        /// NaN, however short. The P rules (<c>np.nanmax</c>) never canonicalize here: <c>npyv_reduce_maxp</c> of an all-NaN
        /// splat is lane 0 as is, i.e. <paramref name="r"/> itself. IsNaN is always false for integer
        /// <typeparamref name="T"/>, so <see cref="CanonicalNaN{T}"/> is never reached there.
        /// </remarks>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">The lane rule (max / min, or the NaN-suppressing fmax / fmin).</typeparam>
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

            // The horizontal reduce of splat(r) (see remarks): only the NaN-propagating rules canonicalize.
            if (TLane.PropagatesNaN)
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
        /// <typeparam name="TLane">The lane rule (max / min, or the NaN-suppressing fmax / fmin).</typeparam>
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
        /// scalar, in NumPy's order. Bit-identical to the scalar unroll by construction: the lane op is NumPy's SCALAR op
        /// applied per lane (<see cref="ILane{T}.SV"/> — <c>_mm_max_sd</c> + NaN blend for the N rules, the CRT
        /// <c>fmax</c> / <c>fmin</c> rule for the P rules, which is NOT the P rules' vector op), and no lane ever meets
        /// another before the combine.
        /// </remarks>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">The lane rule (max / min, or the NaN-suppressing fmax / fmin).</typeparam>
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
                        acc = TLane.SV(acc, GatherRun(ip + i * stride, stride));
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
                        lo = TLane.SV(lo, GatherRun(g, stride));
                        hi = TLane.SV(hi, GatherRun(g + 4 * stride, stride));
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
        /// One of the two per-element ops of an elementwise SLAB call. NumPy's binary loop applies its VECTOR op to the
        /// elements a call's vector section covers and its SCALAR op to the rest (<see cref="VectorOpStep{T,TLane}"/> /
        /// <see cref="ScalarOpStep{T,TLane}"/>); a step fixes WHICH op, and both of its forms compute that op's per-element
        /// rule, so a range folded through one step may be vectorized in any grouping — the op an element gets is the
        /// step's choice, never the vector width's.
        /// </summary>
        /// <remarks>
        /// The distinction only matters for the P rules (<c>np.nanmax</c> / <c>np.nanmin</c>), whose vector op
        /// (<c>maxp</c>) and scalar op (the CRT <c>fmax</c>) disagree on a ±0 tie and on which of two NaNs survives. For the
        /// N rules both steps are the same rule, so any step is exact.
        /// </remarks>
        /// <typeparam name="T">The element type.</typeparam>
        internal interface IFoldStep<T> where T : unmanaged, INumber<T>
        {
            /// <summary>The step's rule applied lane by lane.</summary>
            /// <param name="a">The running outputs (NumPy's first operand).</param>
            /// <param name="b">The incoming elements.</param>
            /// <returns>The per-lane result.</returns>
            static abstract Vector256<T> Vec(Vector256<T> a, Vector256<T> b);

            /// <summary>The step's rule on one element.</summary>
            /// <param name="a">The running output (NumPy's first operand).</param>
            /// <param name="b">The incoming element.</param>
            /// <returns>The result.</returns>
            static abstract T One(T a, T b);
        }

        /// <summary>
        /// The loop's VECTOR op (<see cref="ILane{T}.N"/> lane-wise, <see cref="ILane{T}.V"/> per element) — what the
        /// elements inside a call's vector section get.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">The lane rule.</typeparam>
        internal readonly struct VectorOpStep<T, TLane> : IFoldStep<T>
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
        {
            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> Vec(Vector256<T> a, Vector256<T> b) => TLane.N(a, b);

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static T One(T a, T b) => TLane.V(a, b);
        }

        /// <summary>
        /// The loop's SCALAR op (<see cref="ILane{T}.SV"/> lane-wise, <see cref="ILane{T}.S"/> per element) — what the
        /// elements in a call's scalar tail get (and every element of a call NumPy runs without SIMD).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">The lane rule.</typeparam>
        internal readonly struct ScalarOpStep<T, TLane> : IFoldStep<T>
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
        {
            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> Vec(Vector256<T> a, Vector256<T> b) => TLane.SV(a, b);

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static T One(T a, T b) => TLane.S(a, b);
        }

        /// <summary>
        /// The SLAB step: <c>o[i] = N(o[i], x[i])</c> over a run — one reduced-axis index folded into every output of
        /// the run — with the loop's VECTOR op for every element. That is exactly NumPy's per-element result for any
        /// N-rule (<c>np.max</c> / <c>np.min</c>) call, whose vector and scalar ops agree, so the vector body here is
        /// exact whatever its width; for the P rules it is right only over a range wholly inside a call's vector section
        /// (a P-rule SLAB goes through <see cref="CombineRunTiled{T,TLane}"/>, which splits each run where NumPy's calls do).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">The lane rule.</typeparam>
        /// <param name="o">First output element (read-modify-write).</param>
        /// <param name="os">Output element stride.</param>
        /// <param name="x">First input element of this reduced index.</param>
        /// <param name="xs">Input element stride.</param>
        /// <param name="n">Run length.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void CombineRun<T, TLane>(T* o, long os, T* x, long xs, long n)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
            => CombineRunStep<T, VectorOpStep<T, TLane>>(o, os, x, xs, n);

        /// <summary>
        /// The SLAB step with an explicit per-element op: <c>o[i] = TStep(o[i], x[i])</c> over a run. The body vectorizes
        /// wherever the layout allows (contiguous runs, or a contiguous output over a gathered strided input) — exact for
        /// either step, since the rule is per element and the step, not the grouping, picks the op.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TStep">The per-element op (<see cref="VectorOpStep{T,TLane}"/> or <see cref="ScalarOpStep{T,TLane}"/>).</typeparam>
        /// <param name="o">First output element (read-modify-write).</param>
        /// <param name="os">Output element stride.</param>
        /// <param name="x">First input element of this reduced index.</param>
        /// <param name="xs">Input element stride.</param>
        /// <param name="n">Run length (&lt;= 0 does nothing).</param>
        internal static unsafe void CombineRunStep<T, TStep>(T* o, long os, T* x, long xs, long n)
            where T : unmanaged, INumber<T> where TStep : struct, IFoldStep<T>
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
                        var r0 = TStep.Vec(Vector256.Load(o + i), Vector256.Load(x + i));
                        var r1 = TStep.Vec(Vector256.Load(o + i + w), Vector256.Load(x + i + w));
                        Vector256.Store(r0, o + i);
                        Vector256.Store(r1, o + i + w);
                    }

                    for (; i + w <= n; i += w)
                        Vector256.Store(TStep.Vec(Vector256.Load(o + i), Vector256.Load(x + i)), o + i);
                }

                for (; i < n; i++)
                    o[i] = TStep.One(o[i], x[i]);
                return;
            }

            if (os == 1 && Vector256.IsHardwareAccelerated && (sizeof(T) == 4 || sizeof(T) == 8))
            {
                // A contiguous output run over a STRIDED input run (a stepped or reversed view reduced along an outer
                // axis): gather each vector's lanes with scalar loads — NumPy's own npyv_loadn does exactly this — and
                // apply the same per-lane rule. Elementwise, so any grouping is exact.
                int w = Vector256<T>.Count;
                for (; i + w <= n; i += w)
                    Vector256.Store(TStep.Vec(Vector256.Load(o + i), GatherRun(x + i * xs, xs)), o + i);
            }

            for (; i < n; i++)
                o[i * os] = TStep.One(o[i * os], x[i * xs]);
        }

        /// <summary>
        /// Reduced indices folded per pass by <see cref="CombineRun8{T,TLane}"/> — the SLAB step fused over this many
        /// consecutive slabs.
        /// </summary>
        /// <remarks>
        /// Measured on the (1000, 4000) f64 <c>max(axis=0)</c> / F <c>max(axis=1)</c> probes (DRAM-bound): per-slab
        /// passes 1.4–2.2 ms, 4 per pass 0.95–1.18 ms, 8 per pass 0.75–0.90 ms — each pass re-reads and re-writes the
        /// output run once instead of once per slab, and the eight concurrent input streams keep more misses in flight.
        /// </remarks>
        internal const int SlabFuse = 8;

        /// <summary>
        /// The SLAB step fused over <see cref="SlabFuse"/> consecutive reduced indices of CONTIGUOUS runs:
        /// <c>o[i] = N(…N(N(o[i], x_0[i]), x_1[i])…, x_7[i])</c> with <c>x_q = x + q·sK</c> — bit-identical to eight
        /// separate <see cref="CombineRun{T,TLane}"/> calls for k, k+1, …, k+7, and so to NumPy's sequential fold (the
        /// last equal zero wins, the first NaN keeps its payload). Every element takes the loop's VECTOR op (see
        /// <see cref="CombineRun{T,TLane}"/> for when that is NumPy's answer; the P rules go through
        /// <see cref="CombineRun8Tiled{T,TLane}"/>).
        /// </summary>
        /// <remarks>
        /// <para>
        /// The win is memory traffic: the output run is loaded and stored once per eight slabs instead of once per slab,
        /// and the eight slabs stream concurrently. Callers must hand eight slabs that all exist (the remainder of a
        /// reduced axis goes through <see cref="CombineRun{T,TLane}"/>) and unit-stride runs in both operands.
        /// </para>
        /// <para>
        /// The vector body evaluates the fold as a BALANCED TREE, <c>N(o, N(N(N(x0,x1), N(x2,x3)), N(N(x4,x5), N(x6,x7))))</c>,
        /// not as the left-leaning chain it equals. That is exact because <c>N</c> is associative: it returns <c>a</c> when
        /// <c>a</c> is NaN, else <c>b</c> when <c>b</c> is NaN, else <c>a</c> when <c>a</c> strictly beats <c>b</c>, else
        /// <c>b</c> — "the FIRST NaN of the sequence, else the LAST of its extreme values", a selection that depends only
        /// on the operands' order, never on how they are parenthesized, so every grouping picks the same element with
        /// the same bits (integer lanes have no bit-distinct ties at all). It is the same fact the flat strided unroll
        /// rests on (<c>Flat_StridedRun_LaterLaneWinsEveryTie</c>). The P rules' two ops are associative selections too —
        /// <c>maxp</c> keeps "the LAST extreme non-NaN value, else the FIRST NaN", the CRT <c>fmax</c> "the extreme value
        /// with <c>+0</c> above <c>-0</c> (<c>fmin</c>: <c>-0</c> below <c>+0</c>), else the LAST NaN" — so the same tree is
        /// exact under either step (<see cref="CombineRun8Step{T,TStep}"/>). The chain was measured SLOWER than per-slab
        /// passes on L2-resident floats — eight dependent <c>vmaxp</c> + <c>vcmpordp</c> + <c>vblendvp</c> steps per vector
        /// (f64 (100, 1000): chain 12.2 µs, per-slab 9.8 µs, tree 9.4 µs; f32: 6.5 / 4.8 / 4.8 µs) — while the tree's depth
        /// of four keeps the DRAM win (f64 (4000, 1000): per-slab 1.86 ms, chain 0.81 ms, tree 0.76 ms).
        /// </para>
        /// </remarks>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">The lane rule.</typeparam>
        /// <param name="o">First output element of the run (read-modify-write, contiguous).</param>
        /// <param name="x">First element of the first of the eight slabs' runs (contiguous).</param>
        /// <param name="sK">Element distance between consecutive slabs' runs (any sign).</param>
        /// <param name="n">Run length.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void CombineRun8<T, TLane>(T* o, T* x, long sK, long n)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
            => CombineRun8Step<T, VectorOpStep<T, TLane>>(o, x, sK, n);

        /// <summary>
        /// <see cref="CombineRun8{T,TLane}"/> with an explicit per-element op: the eight-slab fused SLAB step where every
        /// element takes <typeparamref name="TStep"/> (exact under the balanced tree for either step — see the remarks of
        /// <see cref="CombineRun8{T,TLane}"/>).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TStep">The per-element op (<see cref="VectorOpStep{T,TLane}"/> or <see cref="ScalarOpStep{T,TLane}"/>).</typeparam>
        /// <param name="o">First output element of the run (read-modify-write, contiguous).</param>
        /// <param name="x">First element of the first of the eight slabs' runs (contiguous).</param>
        /// <param name="sK">Element distance between consecutive slabs' runs (any sign).</param>
        /// <param name="n">Run length (&lt;= 0 does nothing).</param>
        internal static unsafe void CombineRun8Step<T, TStep>(T* o, T* x, long sK, long n)
            where T : unmanaged, INumber<T> where TStep : struct, IFoldStep<T>
        {
            long i = 0;
            if (Vector256.IsHardwareAccelerated)
            {
                // Vector256 only where it is hardware (as in CombineRunStep: an emulated vector would lose to the scalar
                // loop). The eight slabs fold as a tree in k order (see CombineRun8's remarks: exact because the step's
                // rule is associative), the output vector joins LAST as the leftmost operand, then is stored once.
                int w = Vector256<T>.Count;
                for (; i + w <= n; i += w)
                {
                    T* p = x + i;
                    var t0 = TStep.Vec(Vector256.Load(p), Vector256.Load(p + sK));
                    var t1 = TStep.Vec(Vector256.Load(p + 2 * sK), Vector256.Load(p + 3 * sK));
                    var t2 = TStep.Vec(Vector256.Load(p + 4 * sK), Vector256.Load(p + 5 * sK));
                    var t3 = TStep.Vec(Vector256.Load(p + 6 * sK), Vector256.Load(p + 7 * sK));
                    var v = TStep.Vec(TStep.Vec(t0, t1), TStep.Vec(t2, t3));
                    Vector256.Store(TStep.Vec(Vector256.Load(o + i), v), o + i);
                }
            }

            // Scalar tail (and the whole run without SIMD): the k-ordered left fold per element.
            for (; i < n; i++)
            {
                T v = o[i];
                T* p = x + i;
                for (int q = 0; q < SlabFuse; q++, p += sK)
                    v = TStep.One(v, *p);
                o[i] = v;
            }
        }

        /// <summary>
        /// The P-rule SLAB step over a run whose elements sit at CONSECUTIVE positions <c>p0, p0 + 1, …</c> of NumPy's
        /// P region: split where NumPy's inner-loop calls split it (<see cref="SlabCallTiling.Segment"/>) and fold each
        /// piece with the op NumPy's call gives it — the vector section through <see cref="VectorOpStep{T,TLane}"/>, the
        /// scalar tail through <see cref="ScalarOpStep{T,TLane}"/>.
        /// </summary>
        /// <remarks>
        /// Consequence: which ±0 survives a tie and which NaN comes back from an all-NaN column of <c>np.nanmax(x, axis)</c>
        /// follow NumPy's call boundaries — the tail <c>len % lanes</c> elements of every call (or all of a call's elements
        /// for a float32 input iterated in place with a negative stride, <see cref="SlabCallTiling.AllScalar"/>) keep the CRT
        /// <c>fmax</c> answer. The piece count is per segment, not per element, so a run inside one call costs two calls.
        /// </remarks>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">The lane rule (a P rule; an N rule would be exact too, just slower).</typeparam>
        /// <param name="o">First output element (read-modify-write).</param>
        /// <param name="os">Output element stride.</param>
        /// <param name="x">First input element of this reduced index.</param>
        /// <param name="xs">Input element stride.</param>
        /// <param name="p0">P-region position of the run's first element.</param>
        /// <param name="n">Run length.</param>
        /// <param name="tiling">NumPy's call tiling of the P region.</param>
        internal static unsafe void CombineRunTiled<T, TLane>(T* o, long os, T* x, long xs, long p0, long n, in SlabCallTiling tiling)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
        {
            long i = 0;
            while (i < n)
            {
                tiling.Segment(p0 + i, out long vEnd, out long cEnd);
                // Relative bounds of this call's vector section and of the call itself, clipped to the run.
                long v = Math.Min(vEnd - p0, n);
                long c = Math.Min(cEnd - p0, n);
                if (i < v)
                {
                    CombineRunStep<T, VectorOpStep<T, TLane>>(o + i * os, os, x + i * xs, xs, v - i);
                    i = v;
                }

                if (i < c)
                {
                    CombineRunStep<T, ScalarOpStep<T, TLane>>(o + i * os, os, x + i * xs, xs, c - i);
                    i = c;
                }
            }
        }

        /// <summary>
        /// <see cref="CombineRunTiled{T,TLane}"/> fused over <see cref="SlabFuse"/> consecutive reduced indices of
        /// CONTIGUOUS runs: each call piece is folded by <see cref="CombineRun8Step{T,TStep}"/> with its call's op.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">The lane rule.</typeparam>
        /// <param name="o">First output element of the run (read-modify-write, contiguous).</param>
        /// <param name="x">First element of the first of the eight slabs' runs (contiguous).</param>
        /// <param name="sK">Element distance between consecutive slabs' runs (any sign).</param>
        /// <param name="p0">P-region position of the run's first element.</param>
        /// <param name="n">Run length.</param>
        /// <param name="tiling">NumPy's call tiling of the P region.</param>
        internal static unsafe void CombineRun8Tiled<T, TLane>(T* o, T* x, long sK, long p0, long n, in SlabCallTiling tiling)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
        {
            long i = 0;
            while (i < n)
            {
                tiling.Segment(p0 + i, out long vEnd, out long cEnd);
                long v = Math.Min(vEnd - p0, n);
                long c = Math.Min(cEnd - p0, n);
                if (i < v)
                {
                    CombineRun8Step<T, VectorOpStep<T, TLane>>(o + i, x + i, sK, v - i);
                    i = v;
                }

                if (i < c)
                {
                    CombineRun8Step<T, ScalarOpStep<T, TLane>>(o + i, x + i, sK, c - i);
                    i = c;
                }
            }
        }

        /// <summary>
        /// How NumPy's reduce iterator cuts the P region of a SLAB-mode axis reduction into inner-loop CALLS — the part of
        /// the schedule a NaN-suppressing reduction (<c>np.nanmax</c> / <c>np.nanmin</c>) makes observable. The P region is
        /// the non-reduced axes the iterator walks BELOW the reduced one (smaller |stride|), enumerated in iteration order
        /// (position <c>p = Σ idx·place</c>, <see cref="Build"/>); every reduced index <c>k &gt;= 1</c> is folded in by the
        /// same calls, each covering a contiguous range of positions.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Inside a call of <c>L</c> elements NumPy's binary loop (<c>simd_binary_ccc</c> / <c>simd_binary</c>) applies the
        /// vector op to the first <c>L - L % lanes</c> and the scalar op to the rest — or the scalar op to all of them when
        /// the loop's SIMD branch refuses the strides (<see cref="AllScalar"/>). For <c>np.max</c> both ops are the same
        /// rule and none of this is visible; for <c>fmax</c> the CRT scalar op keeps <c>+0</c> on a ±0 tie and the LAST of
        /// two NaNs where the vector op keeps the second operand and the FIRST NaN.
        /// </para>
        /// <para>
        /// Calls follow <c>npyiter_find_buffering_setup</c> (numpy/_core/src/multiarray/nditer_constr.c) for the two-operand
        /// reduction iterator: when the chosen outer dimension is the reduce dimension (<c>using_reduce</c>) one call covers
        /// the whole P region; otherwise the region is cut into blocks of <see cref="Block"/> positions (one outer index of
        /// the core) and each block into calls of <see cref="Chunk"/> (the buffer, capped to whole cores). Validated
        /// against NumPy 2.4.2 on 7,430 random layouts (0 mismatches) before this port.
        /// </para>
        /// </remarks>
        internal readonly struct SlabCallTiling
        {
            /// <summary>Positions one outer index of the iterator core spans; calls never cross a block boundary.</summary>
            internal readonly long Block;

            /// <summary>Positions one call covers inside a block (the last call of a block may be shorter).</summary>
            internal readonly long Chunk;

            /// <summary>
            /// True when every call runs NumPy's scalar loop: a float32 input iterated IN PLACE (unbuffered) with a
            /// negative innermost stride — <c>npyv_loadable_stride_f32</c> divides the signed stride by the unsigned
            /// <c>sizeof(float)</c>, so a negative stride turns into a huge one and fails the AVX2 gather limit.
            /// </summary>
            internal readonly bool AllScalar;

            /// <summary>NumPy's lanes per vector for the element type (<see cref="Vector256{T}.Count"/> on AVX2).</summary>
            internal readonly int Lanes;

            /// <summary>Creates a tiling from its four parameters (see the fields).</summary>
            /// <param name="block">Positions per block (&gt;= 1).</param>
            /// <param name="chunk">Positions per call inside a block (&gt;= 1).</param>
            /// <param name="allScalar">Whether every call takes the scalar loop.</param>
            /// <param name="lanes">Lanes per vector (&gt;= 1).</param>
            internal SlabCallTiling(long block, long chunk, bool allScalar, int lanes)
            {
                Block = block;
                Chunk = chunk;
                AllScalar = allScalar;
                Lanes = lanes;
            }

            /// <summary>
            /// The call holding position <paramref name="p"/>: where its vector section ends and where the call ends.
            /// </summary>
            /// <param name="p">A P-region position (&gt;= 0).</param>
            /// <param name="vEnd">Receives the first position of the call's scalar tail (the call's start when
            /// <see cref="AllScalar"/>).</param>
            /// <param name="cEnd">Receives the first position after the call.</param>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal void Segment(long p, out long vEnd, out long cEnd)
            {
                long bStart = p - p % Block;
                long off = p - bStart;
                long cStart = bStart + (off - off % Chunk);
                cEnd = Math.Min(cStart + Chunk, bStart + Block);
                long len = cEnd - cStart;
                vEnd = AllScalar ? cStart : cEnd - len % Lanes;
            }

            /// <summary>
            /// NumPy's call tiling for a SLAB-mode reduction of one non-broadcast operand along <paramref name="axis"/>: the
            /// iteration order (a stable insertion sort by |stride| from reversed C order — only the input votes, the
            /// allocated result has no strides yet), the P axes and their place factors, the coalesced groups, and
            /// <c>npyiter_find_buffering_setup</c>'s choice for the two operands (the result, whose stride is zero on the
            /// reduce dimension, and the input).
            /// </summary>
            /// <remarks>
            /// A line-for-line port of the validated model (<c>fmax_tiling_model.py</c>): <c>cost = 1 + multi-strided
            /// operands</c> grown dimension by dimension until the reduce dimension is reached or the buffer overflows, the
            /// best dimension kept by <c>cost · bestSize &lt;= bestCost · min(size, maximumSize)</c>, the reduce-outer
            /// decision, the per-operand buffering flags, and the buffer capped to whole cores when anything is buffered.
            /// </remarks>
            /// <typeparam name="T">The element type (float32 decides <see cref="AllScalar"/>).</typeparam>
            /// <param name="dims">Input extents (rank &lt;= 64).</param>
            /// <param name="xs">Input element strides (no extent &gt; 1 axis may have stride 0).</param>
            /// <param name="axis">The reduced axis: extent &gt; 1 and NOT the iterator's innermost axis (SLAB mode).</param>
            /// <param name="maximumSize">The buffer size in elements (<see cref="np.getbufsize"/>).</param>
            /// <param name="place">Receives each input axis' place factor in the P region (0 for an axis outside it);
            /// length <c>dims.Length</c>.</param>
            /// <param name="pRegion">Receives the P region's size (the product of the P axes' extents).</param>
            /// <returns>The tiling.</returns>
            /// <exception cref="InvalidOperationException"><paramref name="axis"/> is not a SLAB-mode reduced axis — a caller bug.</exception>
            internal static SlabCallTiling Build<T>(ReadOnlySpan<long> dims, ReadOnlySpan<long> xs, int axis, long maximumSize,
                Span<long> place, out long pRegion) where T : unmanaged
            {
                int nd = dims.Length;

                // Iteration order, innermost first: extent > 1 axes by |stride|, a tie keeping the LATER axis inner.
                Span<int> perm = stackalloc int[Math.Max(nd, 1)];
                int m = 0;
                for (int d = nd - 1; d >= 0; d--)
                {
                    if (dims[d] <= 1)
                        continue;
                    long a = Math.Abs(xs[d]);
                    int j = m++;
                    while (j > 0 && Math.Abs(xs[perm[j - 1]]) > a)
                    {
                        perm[j] = perm[j - 1];
                        j--;
                    }

                    perm[j] = d;
                }

                int rpos = -1;
                for (int i = 0; i < m; i++)
                {
                    if (perm[i] == axis)
                    {
                        rpos = i;
                        break;
                    }
                }

                if (rpos <= 0)
                    throw new InvalidOperationException($"SlabCallTiling: axis {axis} is not a SLAB-mode reduced axis — caller bug.");

                // P region: the axes below the reduced one, positions enumerated in iteration order.
                place.Clear();
                long run = 1;
                for (int i = 0; i < rpos; i++)
                {
                    place[perm[i]] = run;
                    run *= dims[perm[i]];
                }

                pRegion = run;

                // Coalesced groups: adjacent non-reduced axes merge when the INPUT continues contiguously (the allocated
                // result is contiguous across adjacent non-reduced axes by construction); the reduced axis stays alone.
                Span<long> ext = stackalloc long[m];
                Span<long> st = stackalloc long[m];
                Span<bool> isRed = stackalloc bool[m];
                int g = 0;
                for (int i = 0; i < m; i++)
                {
                    int d = perm[i];
                    if (i > 0 && d != axis && perm[i - 1] != axis && xs[d] == st[g - 1] * ext[g - 1])
                    {
                        ext[g - 1] *= dims[d];
                        continue;
                    }

                    ext[g] = dims[d];
                    st[g] = xs[d];
                    isRed[g] = d == axis;
                    g++;
                }

                // npyiter_find_buffering_setup: op 0 = the result (single-strided until the reduce dimension), op 1 = the input.
                long size = ext[0];
                int cost = 1, ssd0 = 1, ssd1 = 1, red0 = 0, outerReduceDim = 0;
                int bestDim = 0, bestCost = 1;
                long bestSize = size, bestCore = 1;
                for (int idim = 1; idim < g; idim++)
                {
                    if (outerReduceDim != 0)
                        break;
                    if (size >= maximumSize && cost > 1)
                        break;

                    bool prev0 = !isRed[idim - 1], cur0 = !isRed[idim];   // the result's stride is non-zero there
                    if (ssd0 == idim)
                    {
                        if (prev0 && cur0)
                            ssd0++;
                        else
                            cost++;
                    }

                    if (!prev0 || !cur0)
                    {
                        red0 = idim;
                        outerReduceDim = idim;
                    }

                    if (ssd1 == idim)
                    {
                        if (st[idim - 1] * ext[idim - 1] == st[idim])
                            ssd1++;
                        else
                            cost++;
                    }

                    long core = size;
                    size *= ext[idim];
                    if (size == 0)
                        break;

                    long bufSize = size > maximumSize && cost > 1 ? maximumSize : size;
                    if (cost * bestSize <= bestCost * bufSize)
                    {
                        bestCost = cost;
                        bestCore = core;
                        bestSize = size;
                        bestDim = idim;
                    }
                }

                bool usingReduce = outerReduceDim != 0 && bestDim == outerReduceDim;
                bool redZeroAtOuter = outerReduceDim < g && isRed[outerReduceDim];
                bool isRed0 = usingReduce && (red0 == bestDim || ssd0 == bestDim || (redZeroAtOuter && ssd0 <= bestDim));
                bool buf0 = !(ssd0 + (isRed0 ? 1 : 0) > bestDim);
                bool isRed1 = usingReduce && ssd1 == bestDim;
                bool buf1 = !(ssd1 + (isRed1 ? 1 : 0) > bestDim);

                long bufsz = bestSize;
                if ((buf0 || buf1) && maximumSize < bestSize)
                    bufsz = bestCore * Math.Max(1, maximumSize / bestCore);   // whole cores (never 0 — see ReduceFlat)

                long block, chunk;
                if (usingReduce)
                {
                    block = run;
                    chunk = run;
                }
                else
                {
                    block = bestSize;
                    chunk = Math.Min(bufsz, bestSize);
                }

                bool allScalar = typeof(T) == typeof(float) && !buf1 && st[0] < 0;
                return new SlabCallTiling(block, chunk, allScalar, Vector256<T>.Count);
            }
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
        /// otherwise SLAB mode, the first-visit copy then one <see cref="CombineRun{T,TLane}"/> per further reduced index
        /// (contiguous runs fused eight indices per pass by <see cref="CombineRun8{T,TLane}"/> — the same per-element
        /// result as the one-slab steps, bit for bit). For the P rules (<c>np.nanmax</c> / <c>np.nanmin</c>) every SLAB
        /// step is split where NumPy's inner-loop calls split the P region (<see cref="SlabCallTiling"/>), since their
        /// vector and scalar ops differ.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The innermost axis is picked by NpyIter's rule for a single voting operand: the extent &gt; 1 axis with the
        /// smallest |stride|, a tie going to the LATER axis (the stable insertion sort starts from reversed C order).
        /// Extent-1 axes never matter (NpyIter zeroes their strides and coalesces them away).
        /// </para>
        /// <para>
        /// Outer axes are walked in memory order (largest |stride| outermost) purely for locality — each output
        /// element's operation sequence is fixed by the mode (and, for the P rules, by its position in NumPy's call
        /// tiling), not by the walk. In SLAB mode, outer axes that continue the run contiguously in BOTH input and output
        /// are merged into it (NumPy's coalescing does the same), so a C <c>(K, A, B)</c> reduction along axis 0 folds each
        /// slab as one <c>A·B</c> run. For the P rules a merged axis must also be the NEXT axis of NumPy's P region (place
        /// factor equal to the run so far), so that a run's elements keep consecutive P positions.
        /// </para>
        /// </remarks>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">The lane rule.</typeparam>
        /// <param name="x">Logical element 0 of the input.</param>
        /// <param name="dims">Input extents (rank nd, nd &lt;= 64).</param>
        /// <param name="xs">Input element strides (rank nd); no extent &gt; 1 axis may have stride 0.</param>
        /// <param name="o">Logical element 0 of the output.</param>
        /// <param name="os">Output element strides indexed by INPUT axis (rank nd; the reduced axis' entry is ignored).</param>
        /// <param name="axis">The reduced axis; <c>dims[axis] &gt;= 1</c>.</param>
        /// <param name="maximumSize">The buffer size in elements (<see cref="np.getbufsize"/>) — it shapes NumPy's call
        /// tiling, so it only matters for the P rules.</param>
        internal static unsafe void ReduceAxis<T, TLane>(T* x, ReadOnlySpan<long> dims, ReadOnlySpan<long> xs, T* o,
            ReadOnlySpan<long> os, int axis, long maximumSize)
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

            // P rules in SLAB mode: NumPy's call tiling of the P region and each axis' place factor in it. `inner` is the
            // iterator's innermost axis by the same rule Build sorts by, so it is the P region's first axis (place 1).
            // K == 1 folds nothing (the first-visit copy is the whole answer), so it needs no tiling.
            Span<long> place = stackalloc long[nd];
            place.Clear();   // explicit: an N rule never writes it, and the walk reads it (as zeros) either way
            SlabCallTiling tiling = default;
            bool tiled = !TLane.PropagatesNaN && !rows && inner >= 0 && K > 1;
            if (tiled)
                tiling = SlabCallTiling.Build<T>(dims, xs, axis, maximumSize, place, out _);

            // Run (SLAB mode only): the innermost non-reduced axis, grown by every outer axis that continues it
            // contiguously in both operands (and, tiled, continues its P positions).
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
                        if (xs[d] == runXs * runN && os[d] == runOs * runN && (!tiled || place[d] == runN))
                        {
                            runN *= dims[d];
                            used[d] = true;
                            grew = true;
                        }
                    }
                }
            }

            // Outer walk axes: every remaining extent > 1 non-reduced axis, outermost (largest |stride|) first, each
            // carrying its P place factor (0 outside the P region, and always 0 for the N rules).
            Span<long> wDim = stackalloc long[nd];
            Span<long> wXs = stackalloc long[nd];
            Span<long> wOs = stackalloc long[nd];
            Span<long> wP = stackalloc long[nd];
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
                    wP[j] = wP[j - 1];
                    j--;
                }

                wDim[j] = dims[d];
                wXs[j] = xs[d];
                wOs[j] = os[d];
                wP[j] = place[d];
            }

            long total = 1;
            for (int j = 0; j < w; j++)
                total *= wDim[j];

            Span<long> coord = stackalloc long[Math.Max(w, 1)];
            coord.Clear();
            long offX = 0, offO = 0, offP = 0;
            for (long it = 0; it < total; it++)
            {
                T* xb = x + offX;
                T* ob = o + offO;
                if (rows)
                {
                    *ob = sK == 1 ? ReduceRow<T, TLane>(xb, K) : ReduceStridedRow<T, TLane>(xb, sK, K);
                }
                else if (tiled)
                {
                    // P rules: the same fold, each run split at NumPy's call boundaries (offP = the run's first position).
                    CopyRun(ob, runOs, xb, runXs, runN);
                    long k = 1;
                    if (runOs == 1 && runXs == 1)
                        for (; k + SlabFuse <= K; k += SlabFuse)
                            CombineRun8Tiled<T, TLane>(ob, xb + k * sK, sK, offP, runN, tiling);
                    for (; k < K; k++)
                        CombineRunTiled<T, TLane>(ob, runOs, xb + k * sK, runXs, offP, runN, tiling);
                }
                else
                {
                    CopyRun(ob, runOs, xb, runXs, runN);
                    long k = 1;
                    // Contiguous runs fold eight reduced indices per pass (a regrouping of the same k-ordered fold —
                    // exact, see CombineRun8); a strided run and the axis' remainder take the one-slab step.
                    if (runOs == 1 && runXs == 1)
                        for (; k + SlabFuse <= K; k += SlabFuse)
                            CombineRun8<T, TLane>(ob, xb + k * sK, sK, runN);
                    for (; k < K; k++)
                        CombineRun<T, TLane>(ob, runOs, xb + k * sK, runXs, runN);
                }

                // Odometer over the walk axes, innermost (last) fastest.
                for (int j = w - 1; j >= 0; j--)
                {
                    if (++coord[j] < wDim[j])
                    {
                        offX += wXs[j];
                        offO += wOs[j];
                        offP += wP[j];
                        break;
                    }

                    coord[j] = 0;
                    offX -= (wDim[j] - 1) * wXs[j];
                    offO -= (wDim[j] - 1) * wOs[j];
                    offP -= (wDim[j] - 1) * wP[j];
                }
            }
        }

        /// <summary>
        /// Dtype dispatch for <see cref="ReduceAxis{T,TLane}"/> over raw byte pointers.
        /// </summary>
        /// <remarks>
        /// Floats get one lane rule per op; an integer dtype folds <see cref="MinMaxOp.FMax"/> / <see cref="MinMaxOp.FMin"/>
        /// onto the max / min lanes (identical results — see <see cref="MinMaxOp"/>).
        /// </remarks>
        /// <param name="t">The element dtype (must satisfy <see cref="Supports"/>).</param>
        /// <param name="op">The reduction op.</param>
        /// <param name="x">Logical element 0 of the input.</param>
        /// <param name="dims">Input extents.</param>
        /// <param name="xs">Input element strides.</param>
        /// <param name="o">Logical element 0 of the output.</param>
        /// <param name="os">Output element strides indexed by input axis (reduced entry ignored).</param>
        /// <param name="axis">The reduced axis.</param>
        /// <param name="maximumSize">The buffer size in elements (<see cref="np.getbufsize"/>).</param>
        /// <exception cref="NotSupportedException"><paramref name="t"/> is not a supported dtype.</exception>
        internal static unsafe void ReduceAxis(NPTypeCode t, MinMaxOp op, byte* x, ReadOnlySpan<long> dims, ReadOnlySpan<long> xs,
            byte* o, ReadOnlySpan<long> os, int axis, long maximumSize)
        {
            bool isMax = IsMaxLike(op);
            switch (t)
            {
                case NPTypeCode.Double:
                    switch (op)
                    {
                        case MinMaxOp.Max: ReduceAxis<double, MaxLane<double>>((double*)x, dims, xs, (double*)o, os, axis, maximumSize); break;
                        case MinMaxOp.Min: ReduceAxis<double, MinLane<double>>((double*)x, dims, xs, (double*)o, os, axis, maximumSize); break;
                        case MinMaxOp.FMax: ReduceAxis<double, FMaxLane<double>>((double*)x, dims, xs, (double*)o, os, axis, maximumSize); break;
                        default: ReduceAxis<double, FMinLane<double>>((double*)x, dims, xs, (double*)o, os, axis, maximumSize); break;
                    }

                    break;
                case NPTypeCode.Single:
                    switch (op)
                    {
                        case MinMaxOp.Max: ReduceAxis<float, MaxLane<float>>((float*)x, dims, xs, (float*)o, os, axis, maximumSize); break;
                        case MinMaxOp.Min: ReduceAxis<float, MinLane<float>>((float*)x, dims, xs, (float*)o, os, axis, maximumSize); break;
                        case MinMaxOp.FMax: ReduceAxis<float, FMaxLane<float>>((float*)x, dims, xs, (float*)o, os, axis, maximumSize); break;
                        default: ReduceAxis<float, FMinLane<float>>((float*)x, dims, xs, (float*)o, os, axis, maximumSize); break;
                    }

                    break;
                case NPTypeCode.SByte: if (isMax) ReduceAxis<sbyte, MaxLane<sbyte>>((sbyte*)x, dims, xs, (sbyte*)o, os, axis, maximumSize); else ReduceAxis<sbyte, MinLane<sbyte>>((sbyte*)x, dims, xs, (sbyte*)o, os, axis, maximumSize); break;
                case NPTypeCode.Byte: if (isMax) ReduceAxis<byte, MaxLane<byte>>(x, dims, xs, o, os, axis, maximumSize); else ReduceAxis<byte, MinLane<byte>>(x, dims, xs, o, os, axis, maximumSize); break;
                case NPTypeCode.Int16: if (isMax) ReduceAxis<short, MaxLane<short>>((short*)x, dims, xs, (short*)o, os, axis, maximumSize); else ReduceAxis<short, MinLane<short>>((short*)x, dims, xs, (short*)o, os, axis, maximumSize); break;
                case NPTypeCode.UInt16: if (isMax) ReduceAxis<ushort, MaxLane<ushort>>((ushort*)x, dims, xs, (ushort*)o, os, axis, maximumSize); else ReduceAxis<ushort, MinLane<ushort>>((ushort*)x, dims, xs, (ushort*)o, os, axis, maximumSize); break;
                case NPTypeCode.Int32: if (isMax) ReduceAxis<int, MaxLane<int>>((int*)x, dims, xs, (int*)o, os, axis, maximumSize); else ReduceAxis<int, MinLane<int>>((int*)x, dims, xs, (int*)o, os, axis, maximumSize); break;
                case NPTypeCode.UInt32: if (isMax) ReduceAxis<uint, MaxLane<uint>>((uint*)x, dims, xs, (uint*)o, os, axis, maximumSize); else ReduceAxis<uint, MinLane<uint>>((uint*)x, dims, xs, (uint*)o, os, axis, maximumSize); break;
                case NPTypeCode.Int64: if (isMax) ReduceAxis<long, MaxLane<long>>((long*)x, dims, xs, (long*)o, os, axis, maximumSize); else ReduceAxis<long, MinLane<long>>((long*)x, dims, xs, (long*)o, os, axis, maximumSize); break;
                case NPTypeCode.UInt64: if (isMax) ReduceAxis<ulong, MaxLane<ulong>>((ulong*)x, dims, xs, (ulong*)o, os, axis, maximumSize); else ReduceAxis<ulong, MinLane<ulong>>((ulong*)x, dims, xs, (ulong*)o, os, axis, maximumSize); break;
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
        /// <typeparam name="TLane">The lane rule (max / min, or the NaN-suppressing fmax / fmin).</typeparam>
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
        /// <typeparam name="TLane">The lane rule (max / min, or the NaN-suppressing fmax / fmin).</typeparam>
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
        /// <typeparam name="TLane">The lane rule (max / min, or the NaN-suppressing fmax / fmin).</typeparam>
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
        /// <typeparam name="TLane">The lane rule (max / min, or the NaN-suppressing fmax / fmin).</typeparam>
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
        /// <remarks>
        /// Floats get one lane rule per op; an integer dtype folds <see cref="MinMaxOp.FMax"/> / <see cref="MinMaxOp.FMin"/>
        /// onto the max / min lanes (identical results — see <see cref="MinMaxOp"/>).
        /// </remarks>
        /// <param name="t">The element dtype (must satisfy <see cref="Supports"/>).</param>
        /// <param name="op">The reduction op.</param>
        /// <param name="x">Logical element 0 of the operand.</param>
        /// <param name="dims">Operand extents.</param>
        /// <param name="xs">Operand element strides (non-broadcast).</param>
        /// <param name="maximumSize">The buffer size in elements (<c>np.getbufsize()</c>).</param>
        /// <param name="slot">Receives the result (at least 8 bytes).</param>
        /// <exception cref="NotSupportedException"><paramref name="t"/> is not a supported dtype.</exception>
        internal static unsafe void ReduceFlat(NPTypeCode t, MinMaxOp op, byte* x, ReadOnlySpan<long> dims, ReadOnlySpan<long> xs,
            long maximumSize, byte* slot)
        {
            bool isMax = IsMaxLike(op);
            switch (t)
            {
                case NPTypeCode.Double:
                    *(double*)slot = op switch
                    {
                        MinMaxOp.Max => ReduceFlat<double, MaxLane<double>>((double*)x, dims, xs, maximumSize),
                        MinMaxOp.Min => ReduceFlat<double, MinLane<double>>((double*)x, dims, xs, maximumSize),
                        MinMaxOp.FMax => ReduceFlat<double, FMaxLane<double>>((double*)x, dims, xs, maximumSize),
                        _ => ReduceFlat<double, FMinLane<double>>((double*)x, dims, xs, maximumSize),
                    };
                    break;
                case NPTypeCode.Single:
                    *(float*)slot = op switch
                    {
                        MinMaxOp.Max => ReduceFlat<float, MaxLane<float>>((float*)x, dims, xs, maximumSize),
                        MinMaxOp.Min => ReduceFlat<float, MinLane<float>>((float*)x, dims, xs, maximumSize),
                        MinMaxOp.FMax => ReduceFlat<float, FMaxLane<float>>((float*)x, dims, xs, maximumSize),
                        _ => ReduceFlat<float, FMinLane<float>>((float*)x, dims, xs, maximumSize),
                    };
                    break;
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
        /// <param name="op">The reduction op (<see cref="MinMaxOp.FMax"/> / <see cref="MinMaxOp.FMin"/> for
        /// <c>np.nanmax</c> / <c>np.nanmin</c>).</param>
        /// <returns>The fresh reduced array (the reduced axis removed; a 0-d result for a 1-D input), or null.</returns>
        internal unsafe NDArray TryExactAxisMinMax(NDArray arr, int axis, MinMaxOp op)
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
                NumPyMinMaxReduce.ReduceAxis(t, op, x, dims, shape.strides, o, os, axis, np.getbufsize());
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
        /// <param name="op">The reduction op.</param>
        /// <param name="slot">Receives the result at the array's dtype (at least 8 bytes).</param>
        /// <returns>True when the reduction was computed here.</returns>
        internal unsafe bool TryExactFlatMinMaxArray(NDArray arr, MinMaxOp op, byte* slot)
        {
            if (NDExpr.DisableExactMinMax)
                return false;
            var t = arr.GetTypeCode;
            var shape = arr.Shape;
            if (!NumPyMinMaxReduce.Supports(t) || shape.IsBroadcasted || shape.NDim > 64 || shape.size == 0)
                return false;

            // Logical element 0 (Address + offset·itemsize is right for a re-seated contiguous slice and a strided view).
            byte* x = (byte*)arr.Address + shape.offset * t.SizeOf();
            NumPyMinMaxReduce.ReduceFlat(t, op, x, shape.dimensions, shape.strides, np.getbufsize(), slot);
            NDExpr.ExactMinMaxRuns++;
            return true;
        }

        /// <summary>
        /// <see cref="TryExactFlatMinMaxArray(NDArray, MinMaxOp, byte*)"/> as a fresh 0-d array of the input's dtype — the
        /// shape <c>np.max(x)</c> / <c>np.nanmax(x)</c> returns before its keepdims / scalar-marking step.
        /// </summary>
        /// <param name="arr">The array to reduce.</param>
        /// <param name="op">The reduction op.</param>
        /// <returns>The 0-d result, or null when the exact schedule declined (nothing allocated).</returns>
        internal unsafe NDArray TryExactFlatMinMaxScalar(NDArray arr, MinMaxOp op)
        {
            ulong slot = 0;   // 8 bytes: the widest served dtype
            if (!TryExactFlatMinMaxArray(arr, op, (byte*)&slot))
                return null;

            var t = arr.GetTypeCode;
            var r = new NDArray(t, Shape.Scalar, false);
            Buffer.MemoryCopy(&slot, (byte*)r.Address + r.Shape.offset * t.SizeOf(), t.SizeOf(), t.SizeOf());
            return r;
        }
    }
}
