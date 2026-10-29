using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Kernels;

// =============================================================================
// Default.Reduction.MinMax.Half.cs — np.max / np.min / np.nanmax / np.nanmin for float16, NumPy's bits on every layout
// =============================================================================
//
// NumPy has no SIMD loop for float16 comparisons. HALF_maximum / HALF_minimum and HALF_fmax / HALF_fmin
// (numpy/_core/src/umath/loops.c.src) are plain BINARY_LOOPs, and a reduction runs them with the result as in1:
//
//     maximum:  acc = (npy_half_ge(acc, x) || npy_half_isnan(acc)) ? acc : x        (minimum: npy_half_le)
//     fmax:     acc = (npy_half_ge(acc, x) || npy_half_isnan(x))   ? acc : x        (fmin:    npy_half_le)
//
// so every result is a SEQUENTIAL fold, and its BITS depend on the order the iterator visits the elements:
//
//   * maximum / minimum return the FIRST NaN verbatim (payload, sign, a signalling NaN unquieted), else the first
//     extreme element — npy_half_ge / le call -0 == +0, so a ±0 tie keeps the earlier zero;
//   * fmax / fmin skip NaN, keep the first extreme non-NaN element, and an all-NaN reduction returns its FIRST element.
//
// The visiting order is NumPy's reduction iterator's (PyUFunc_ReduceWrapper, numpy/_core/src/umath/reduction.c: a K-order
// NpyIter with NPY_ITER_DONT_NEGATE_STRIDES, the first element of every output copied in first):
//
//   * AXIS: each output folds its reduced elements in increasing logical index, whatever the memory layout — so the answer
//     is layout-independent and the walk below only picks the cache-friendly order;
//   * FLAT: one fold over the K order — the extent > 1 axes insertion-sorted by |stride| from reversed C order, a zero
//     stride ambiguous (NumPySequentialReduce.FlatVisitOrder), each axis in increasing logical index.
//
// The kernels (DirectILKernelGenerator.Reduction.MinMax.Half.cs) work on the raw bits, never converting a float16. The fold
// step is an associative selection ("the first NaN, else the first extreme element"), so a run reduced by SIMD and folded
// into the running winner equals folding its elements one by one — which is what lets every layout run vectorized:
//
//   * FLAT: the visit axes, minus broadcast ones (a revisited element never changes a winner — only first occurrences
//     decide), coalesced into runs; ±1-stride runs of >= HalfSimdRunMin elements go to the SIMD cores, the rest fold
//     scalar by the branch-light priority key. Under maximum / minimum the first NaN ends the walk.
//   * AXIS, ROW mode (the reduced axis is the tighter one, or the non-reduced runs are too short to vectorize): each output
//     is the winner of its row — a SIMD core for a ±1-stride row, the scalar priority-key fold otherwise.
//   * AXIS, SLAB mode (a non-reduced axis is tighter): the first slab is copied into the output (or a scratch block when
//     the output run is strided), then every further slab folds in elementwise by the pair-select kernel, in blocks of
//     SlabBlock lanes so the running winners stay in L1.
//
// It replaces three paths, each value-exact only: the axis kernel (AxisReductionHalfMinMaxHelper) let a LATER NaN replace an
// earlier one (`x > acc || IsNaN(x)`), the flat non-contiguous NDIter kernel returned the canonical Half.NaN for any NaN and
// visited reversed / permuted views in another order (a ±0 tie took another zero), and the scalar nanmax / nanmin fold
// converted every element to float (0.4–0.9× NumPy).
// =============================================================================

namespace NumSharp.Backends
{
    /// <summary>
    /// NumPy's float16 <c>maximum</c> / <c>minimum</c> / <c>fmax</c> / <c>fmin</c> reductions, flat and per axis, walked
    /// in NumPy's visiting order over the bit-level kernels (see the file header). Stateless; the caller owns every buffer.
    /// </summary>
    internal static class NumPyHalfMinMaxReduce
    {
        /// <summary>
        /// Lanes per SLAB block: the running winners (8 KB of float16) stay in L1 while every slab's block streams past them,
        /// so the output is read and written once per block instead of once per slab.
        /// </summary>
        internal const long SlabBlock = 4096;

        /// <summary>
        /// The shortest non-reduced run the SLAB mode takes on a contiguous slab; below it a slab step is mostly call and
        /// loop overhead, and the ROW mode's per-output fold is cheaper.
        /// </summary>
        internal const long SlabMinRun = 16;

        /// <summary>
        /// The FLAT reduction: one fold over NumPy's K visiting order, starting at the operand's logical element 0.
        /// </summary>
        /// <remarks>
        /// The visit axes come from <see cref="NumPySequentialReduce.FlatVisitOrder"/>, which needs the zero strides (an
        /// ambiguous comparison can keep another axis from moving inward). Only afterwards are the broadcast axes dropped —
        /// revisiting an element never changes a winner, and the other elements keep their first-occurrence order — and the
        /// rest coalesced into runs (an axis whose stride is the run's stride times its length continues it, in the same
        /// order). Each run either goes to <see cref="DirectILKernelGenerator.HalfRunWinnerBits"/> (a long ±1-stride run,
        /// SIMD) or folds inline by the priority key, so short runs pay no call.
        /// </remarks>
        /// <param name="x">The operand's logical element 0 (raw float16 bits).</param>
        /// <param name="dims">The operand's extents (non-empty).</param>
        /// <param name="strides">The operand's element strides (any sign, zero for a broadcast axis).</param>
        /// <param name="isMax">maximum / fmax when true, minimum / fmin when false.</param>
        /// <param name="nanIgnore">fmax / fmin (np.nanmax / np.nanmin) when true, maximum / minimum when false.</param>
        /// <returns>The reduction's raw bits, NumPy's to the bit.</returns>
        [SkipLocalsInit]
        internal static unsafe ushort Flat(ushort* x, ReadOnlySpan<long> dims, ReadOnlySpan<long> strides, bool isMax, bool nanIgnore)
        {
            int nd = dims.Length;
            // [SkipLocalsInit] spans: every slot read below is written first (perm by FlatVisitOrder, rd / rs / coord here).
            Span<int> perm = nd <= 64 ? stackalloc int[64] : new int[nd];
            Span<long> rd = nd <= 64 ? stackalloc long[64] : new long[nd];
            Span<long> rs = nd <= 64 ? stackalloc long[64] : new long[nd];
            int m = NumPySequentialReduce.FlatVisitOrder(dims, strides, perm);

            int r = 0;
            for (int j = 0; j < m; j++)
            {
                int d = perm[j];
                long len = dims[d], s = strides[d];
                if (s == 0)
                    continue;                       // a broadcast axis only revisits elements
                if (r > 0 && s == rs[r - 1] * rd[r - 1])
                {
                    rd[r - 1] *= len;               // continues the run below it, in the same visiting order
                    continue;
                }

                rd[r] = len;
                rs[r] = s;
                r++;
            }

            uint inv = isMax ? 0u : 0xFFFFu;
            uint nanKey = nanIgnore ? 0u : 0xFFFFu;
            ushort bestBits = x[0];
            uint best = DirectILKernelGenerator.HalfPriorityKey(bestBits, inv, nanKey);
            // One distinct element, or a NaN first element under maximum / minimum (it sticks): the seed is the answer.
            if (r == 0 || best == 0xFFFF)
                return bestBits;

            long n0 = rd[0], s0 = rs[0];
            bool simdRuns = n0 >= DirectILKernelGenerator.HalfSimdRunMin && (s0 == 1 || s0 == -1);
            Span<long> coord = nd <= 64 ? stackalloc long[64] : new long[nd];
            coord.Slice(0, r).Clear();
            long off = 0;
            while (true)
            {
                ushort* p = x + off;
                if (simdRuns)
                {
                    // The run's own winner, then one fold step: the selection is associative, so this equals folding the
                    // run element by element (the first run re-includes the seed, which ties with itself and changes nothing).
                    ushort w = DirectILKernelGenerator.HalfRunWinnerBits(p, s0, n0, isMax, nanIgnore);
                    uint k = DirectILKernelGenerator.HalfPriorityKey(w, inv, nanKey);
                    if (k > best)
                    {
                        best = k;
                        bestBits = w;
                    }
                }
                else
                {
                    ushort* q = p;
                    for (long i = 0; i < n0; i++, q += s0)
                    {
                        ushort b = *q;
                        uint k = DirectILKernelGenerator.HalfPriorityKey(b, inv, nanKey);
                        if (k > best)
                        {
                            best = k;
                            bestBits = b;
                            if (k == 0xFFFF)
                                return bestBits;    // the first NaN under maximum / minimum: nothing outranks it
                        }
                    }
                }

                if (best == 0xFFFF)
                    return bestBits;

                // Odometer over the outer runs 1 .. r-1 (run 1 fastest), stepping the offset by stride differences.
                int j2 = 1;
                for (; j2 < r; j2++)
                {
                    if (++coord[j2] < rd[j2])
                    {
                        off += rs[j2];
                        break;
                    }

                    off -= (rd[j2] - 1) * rs[j2];
                    coord[j2] = 0;
                }

                if (j2 == r)
                    return bestBits;
            }
        }

        /// <summary>
        /// The AXIS reduction: <c>o[..] = fold over k = 0 … K-1 of x[.., k, ..]</c>, every output in increasing k — NumPy's
        /// per-output sequence under <c>NPY_ITER_DONT_NEGATE_STRIDES</c> — in the ROW or SLAB walk (see the file header).
        /// </summary>
        /// <remarks>
        /// The walk axes are the non-reduced extent &gt; 1 axes, innermost (smallest |input stride|, a tie keeping the later
        /// axis inner) first, coalesced where both the input and the output strides continue the run below; the choice
        /// between ROW and SLAB is locality only, the answer is the same. A broadcast reduced axis (zero stride) repeats one
        /// element, so its first element is every output's answer under every rule.
        /// </remarks>
        /// <param name="x">The operand's logical element 0 (raw float16 bits).</param>
        /// <param name="dims">The operand's extents (non-empty; the reduced extent &gt;= 1).</param>
        /// <param name="xs">The operand's element strides.</param>
        /// <param name="o">The result's logical element 0.</param>
        /// <param name="os">The result's element strides indexed by INPUT axis (the reduced axis' entry is never read).</param>
        /// <param name="axis">The reduced axis (normalized).</param>
        /// <param name="isMax">maximum / fmax when true, minimum / fmin when false.</param>
        /// <param name="nanIgnore">fmax / fmin when true, maximum / minimum when false.</param>
        [SkipLocalsInit]
        internal static unsafe void Axis(ushort* x, ReadOnlySpan<long> dims, ReadOnlySpan<long> xs, ushort* o, ReadOnlySpan<long> os,
            int axis, bool isMax, bool nanIgnore)
        {
            int nd = dims.Length;
            long K = dims[axis], sK = xs[axis];
            if (sK == 0)
                K = 1;

            Span<long> wd = nd <= 64 ? stackalloc long[64] : new long[nd];
            Span<long> wx = nd <= 64 ? stackalloc long[64] : new long[nd];
            Span<long> wo = nd <= 64 ? stackalloc long[64] : new long[nd];
            int w = 0;
            for (int d = nd - 1; d >= 0; d--)
            {
                if (d == axis || dims[d] <= 1)
                    continue;
                long a = Math.Abs(xs[d]);
                int j = w++;
                while (j > 0 && Math.Abs(wx[j - 1]) > a)
                {
                    wd[j] = wd[j - 1];
                    wx[j] = wx[j - 1];
                    wo[j] = wo[j - 1];
                    j--;
                }

                wd[j] = dims[d];
                wx[j] = xs[d];
                wo[j] = os[d];
            }

            if (w > 1)
            {
                int c = 0;
                for (int j = 1; j < w; j++)
                {
                    if (wx[j] == wx[c] * wd[c] && wo[j] == wo[c] * wd[c])
                    {
                        wd[c] *= wd[j];
                        continue;
                    }

                    c++;
                    wd[c] = wd[j];
                    wx[c] = wx[j];
                    wo[c] = wo[j];
                }

                w = c + 1;
            }

            Span<long> coord = nd <= 64 ? stackalloc long[64] : new long[nd];

            // ROW when the rows vectorize, or when no slab does and the rows are at least as tight (or the slab runs are
            // too short to amortize a call per slab); SLAB otherwise.
            bool rowSimd = (sK == 1 || sK == -1) && K >= DirectILKernelGenerator.HalfSimdRunMin;
            bool slabSimd = w > 0 && wx[0] == 1 && wd[0] >= SlabMinRun;
            bool rows = w == 0 || rowSimd || (!slabSimd && (wd[0] < SlabMinRun || Math.Abs(sK) <= Math.Abs(wx[0])));
            if (rows)
            {
                coord.Slice(0, w).Clear();
                long offX = 0, offO = 0;
                if (rowSimd)
                {
                    while (true)
                    {
                        o[offO] = DirectILKernelGenerator.HalfRunWinnerBits(x + offX, sK, K, isMax, nanIgnore);
                        if (!Advance(wd, wx, wo, 0, w, coord, ref offX, ref offO))
                            return;
                    }
                }

                // Short or strided rows: the priority-key fold — inlined and branch-free for short rows (an unpredictable
                // "winner improves" branch per element, and no call per row), the branchy out-of-line one with its NaN
                // early exit for long ones.
                if (K <= DirectILKernelGenerator.HalfShortRunMax)
                {
                    while (true)
                    {
                        o[offO] = DirectILKernelGenerator.HalfShortRunWinnerBits(x + offX, sK, K, isMax, nanIgnore);
                        if (!Advance(wd, wx, wo, 0, w, coord, ref offX, ref offO))
                            return;
                    }
                }

                while (true)
                {
                    o[offO] = DirectILKernelGenerator.HalfStridedWinnerBits(x + offX, sK, K, isMax, nanIgnore);
                    if (!Advance(wd, wx, wo, 0, w, coord, ref offX, ref offO))
                        return;
                }
            }

            // SLAB: walk axis 0 is the inner run; the rest form the outer odometer. A strided inner run is gathered into a
            // contiguous block per slab first, so every fold step is the SIMD pair-select (a gather is one load and one
            // store per element, the scalar select several compares).
            long nIn = wd[0], si = wx[0], oi = wo[0];
            // Two 8 KB blocks (never zeroed: [SkipLocalsInit], and every lane is written before it is read).
            ushort* scratch = stackalloc ushort[(int)SlabBlock];
            ushort* gathered = stackalloc ushort[(int)SlabBlock];
            coord.Slice(0, w).Clear();
            long offXs = 0, offOs = 0;
            while (true)
            {
                for (long j0 = 0; j0 < nIn; j0 += SlabBlock)
                {
                    long len = Math.Min(SlabBlock, nIn - j0);
                    // A contiguous output run is folded in place; a strided one through the scratch block.
                    ushort* acc = oi == 1 ? o + offOs + j0 : scratch;
                    ushort* xb = x + offXs + j0 * si;
                    DirectILKernelGenerator.HalfGatherRun(acc, xb, si, len);           // k = 0: NumPy copies it in
                    for (long k = 1; k < K; k++)
                    {
                        ushort* slab = xb + k * sK;
                        if (si != 1)
                        {
                            DirectILKernelGenerator.HalfGatherRun(gathered, slab, si, len);
                            slab = gathered;
                        }

                        DirectILKernelGenerator.HalfFoldSlab(acc, slab, 1, len, isMax, nanIgnore);
                    }

                    if (oi != 1)
                        DirectILKernelGenerator.HalfScatterRun(o + offOs + j0 * oi, oi, acc, len);
                }

                if (!Advance(wd, wx, wo, 1, w, coord, ref offXs, ref offOs))
                    return;
            }
        }

        /// <summary>
        /// One odometer step over the walk axes <c>[from, w)</c> (axis <paramref name="from"/> fastest), advancing the input
        /// and output offsets together by stride differences (no div / mod).
        /// </summary>
        /// <param name="wd">Walk extents.</param>
        /// <param name="wx">Walk input element strides.</param>
        /// <param name="wo">Walk output element strides.</param>
        /// <param name="from">The first (fastest) odometer axis.</param>
        /// <param name="w">One past the last walk axis.</param>
        /// <param name="coord">The odometer digits (indices <c>[from, w)</c> start at zero).</param>
        /// <param name="offX">The input offset (updated).</param>
        /// <param name="offO">The output offset (updated).</param>
        /// <returns>False when the walk is complete (every digit wrapped).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool Advance(ReadOnlySpan<long> wd, ReadOnlySpan<long> wx, ReadOnlySpan<long> wo, int from, int w,
            Span<long> coord, ref long offX, ref long offO)
        {
            for (int j = from; j < w; j++)
            {
                if (++coord[j] < wd[j])
                {
                    offX += wx[j];
                    offO += wo[j];
                    return true;
                }

                coord[j] = 0;
                offX -= (wd[j] - 1) * wx[j];
                offO -= (wd[j] - 1) * wo[j];
            }

            return false;
        }
    }

    public partial class DefaultEngine
    {
        /// <summary>
        /// <c>np.max</c> / <c>np.min</c> / <c>np.nanmax</c> / <c>np.nanmin</c> of a float16 array — NumPy's sequential HALF
        /// loops, bit for bit on every layout (see the file header): maximum / minimum keep the first NaN verbatim, fmax /
        /// fmin skip NaN and return an all-NaN reduction's first element, a ±0 tie keeps the earlier zero, "first" meaning
        /// NumPy's visiting order.
        /// </summary>
        /// <remarks>
        /// The caller has handled the empty reduction (NumPy's "zero-size array" error) and the 0-d / one-element input; a
        /// reduced axis of extent 1 is served here too (each output is its lone element). The result is fresh: a read-only
        /// numpy-scalar 0-d for a flat reduction without keepdims (<see cref="NDArray.MarkReductionScalar"/>), else the
        /// reduced array in NumPy's K order (<see cref="AllocateReductionResult"/>) with keepdims applied.
        /// </remarks>
        /// <param name="arr">A non-empty float16 array (any layout, broadcast included).</param>
        /// <param name="axis">The axis to reduce (null = all axes; negative counts from the end).</param>
        /// <param name="keepdims">Keep the reduced axes as size 1.</param>
        /// <param name="isMax">maximum / fmax when true, minimum / fmin when false.</param>
        /// <param name="nanIgnore">fmax / fmin (np.nanmax / np.nanmin) when true, maximum / minimum (np.max / np.min) when false.</param>
        /// <returns>The reduced array.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="NotSupportedException"><paramref name="arr"/> is not float16 — a caller bug.</exception>
        internal unsafe NDArray HalfMinMaxReduce(NDArray arr, int? axis, bool keepdims, bool isMax, bool nanIgnore)
        {
            if (arr.GetTypeCode != NPTypeCode.Half)
                throw new NotSupportedException($"HalfMinMaxReduce serves float16, got {arr.GetTypeCode} — caller bug.");

            var shape = arr.Shape;
            int nd = shape.NDim;
            // Logical element 0: a contiguous slice re-seats Address (offset 0), a strided view keeps its base with a
            // non-zero offset — Address + offset·itemsize is right for both.
            ushort* x = (ushort*)((byte*)arr.Address + shape.offset * sizeof(ushort));

            if (axis == null)
            {
                ushort bits = NumPyHalfMinMaxReduce.Flat(x, shape.dimensions, shape.strides, isMax, nanIgnore);
                var r = new NDArray(NPTypeCode.Half, Shape.Scalar, false);
                *(ushort*)((byte*)r.Address + r.Shape.offset * sizeof(ushort)) = bits;
                if (keepdims)
                {
                    var ks = new long[nd];
                    for (int i = 0; i < nd; i++)
                        ks[i] = 1;
                    r.Storage.Reshape(Shape.Adopt(ks));
                }

                return r.MarkReductionScalar();
            }

            int ax = NormalizeAxis(axis.Value, nd);
            var dims = shape.dimensions;
            var outputDims = new long[nd - 1];
            for (int d = 0, od = 0; d < nd; d++)
                if (d != ax)
                    outputDims[od++] = dims[d];

            var result = AllocateReductionResult(NPTypeCode.Half, outputDims, shape, ax);
            try
            {
                // Output strides indexed by INPUT axis (the reduced axis' entry is never read).
                Span<long> os = nd <= 64 ? stackalloc long[nd] : new long[nd];
                var rs = result.Shape.strides;
                for (int d = 0, od = 0; d < nd; d++)
                    os[d] = d == ax ? 0 : (outputDims.Length == 0 ? 0 : rs[od++]);
                ushort* o = (ushort*)((byte*)result.Address + result.Shape.offset * sizeof(ushort));
                NumPyHalfMinMaxReduce.Axis(x, dims, shape.strides, o, os, ax, isMax, nanIgnore);
            }
            catch
            {
                // A failing reduction must not strand the fresh buffer.
                result.Dispose();
                throw;
            }

            if (keepdims)
                result.Storage.ExpandDimension(ax);
            return result.MarkReductionScalar();
        }
    }
}
