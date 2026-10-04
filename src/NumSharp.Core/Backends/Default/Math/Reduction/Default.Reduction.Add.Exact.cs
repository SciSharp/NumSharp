using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp.Backends.Kernels;

// =============================================================================
// Default.Reduction.Add.Exact.cs — NumPy-EXACT flat add.reduce for the float loops
// =============================================================================
//
// np.sum(a) / np.sum(a, dtype=d) with axis=None is NumPy's add.reduce over every axis, and its bits are decided by
// three things this file reproduces exactly:
//
//   1. THE START. PyUFunc_ReduceWrapper assigns the loop's identity (+0.0) to the result before the loop runs, so the
//      sum is +0.0 + … — never a copy of the first element (a sum of only -0.0 is +0.0).
//   2. THE CALLS. The reduce iterator hands the add loop the operand in inner-loop CALLS, and every call does
//      `*out += pairwise_sum(call)` (FLOAT/DOUBLE: in the loop dtype; HALF: in float32, the running result narrowed
//      back to float16 after EACH call; CDOUBLE: per part). Which elements share a call is NpyIter's schedule, the
//      same one np.max's flat reduction runs (Default.Reduction.MinMax.Exact.cs, FLAT section): the axes sorted by
//      |stride| (reversed C order, stable) and coalesced SIGNED (NumPyMinMaxReduce.FlatIteratorAxes), then
//      npyiter_find_buffering_setup's cost model (NumPyMinMaxReduce.FlatBufferingDim):
//        * one coalesced run, no cast — ONE call over the whole run, in place, any stride (a reversed view is summed
//          in its logical order, a transposed dense block in memory order);
//        * several runs, no cast — the innermost run longer than half the buffer keeps dimension 0: one call per run,
//          in place; a shorter one is BUFFERED: runs copied in iteration order into fills of
//          coresize·⌊bufsize/coresize⌋ elements (never across the outer block's end), one call per fill;
//        * a CAST operand (dtype= naming another loop, np.mean's float64 loop over integers, its float32 loop over
//          float16) is buffered from the start: the cost model starts at 2, so a run that alone fills the buffer is
//          cut into fills of bufsize elements (never across the run's end), and shorter runs share fills as above.
//      The fills follow np.getbufsize(), exactly as NumPy's answer follows np.setbufsize().
//   3. THE LEAF. pairwise_sum itself — 8 accumulators per 128-element block, split at a multiple of 8, a -0.0 seed
//      below 8 elements — runs through ILKernelGenerator's emitted SIMD folds (bit-identical to NumPy's C), or the
//      scalar ports where dynamic code is unavailable.
//
// A buffered fill is never materialized. A pairwise sum over a range is fixed by the range alone, so the sum of a
// fill is the fill's own recursion down to sub-ranges of at most StreamBlockElements, each copied (and cast) into an
// L1-resident block and folded there, the sub-results combined as the recursion combines them — the same bits as
// folding the whole fill, without NumPy's 64 KB buffer round trip through L2 (measured on a stride-2 uint8 mean:
// 52.9 µs per 100K elements through the buffer). A sub-range that lies inside one run and reads as the loop dtype
// without converting (the same dtype, or the real parts of a complex128 operand in a float64 loop) is folded where it
// is, when the fold reads its stride with SIMD loads (FillCursor.TryTakeInPlace): np.sum(c128, dtype=float64) over
// 10M elements 8.7 → 7.1 ms on one pinned P-core (NumPy: 11.3 ms), the stride-2 real-part fold replacing a
// deinterleaving copy. 10M complex128 elements are 160 MB, so this cell is bound by DRAM reads, not by the fold.
//
// Validated before porting: a Python model of exactly this schedule reproduced np.sum on 198/198 (layout × dtype ×
// cast) cases — contiguous 1-D across the 8192 boundary, strided, reversed, transposed, F-order 3-D, sliced 2-D/3-D in
// both regimes — float64 / float32 / float16 / complex128 loops and the casts f32→f64, f64→f32, f16→f32, f64→f16,
// i32→f64, f64→c128, f16→f64.
//
// Declines (the caller keeps its previous route): a broadcast operand (a stride-0 axis of extent > 1 changes NpyIter's
// axis sort, which FlatIteratorAxes does not model — np.max declines it too), rank > 64, an empty operand, a cast with
// no house cast kernel (Decimal sources; any cast when IL generation is disabled).
// =============================================================================

namespace NumSharp.Backends
{
    /// <summary>
    ///     NumPy's FLAT <c>add.reduce</c> (<c>np.sum(a)</c>, <c>np.sum(a, dtype=…)</c> and the sum inside <c>np.mean(a)</c>)
    ///     for the float16 / float32 / float64 / complex128 loops, bit for bit on every non-broadcast layout — see the
    ///     file header for the schedule it reproduces and why each piece is observable.
    /// </summary>
    internal static unsafe class NumPyAddReduce
    {
        /// <summary>
        ///     The largest sub-range of a buffered fill that is materialized at once (and the scratch block's capacity):
        ///     at least one 128-element leaf and a multiple of 8, so every sub-range the stream recursion folds is a whole
        ///     subtree of NumPy's recursion; 16 KB for complex128, which keeps the block in L1.
        /// </summary>
        private const int StreamBlockElements = 1024;

        /// <summary>
        ///     Block, in elements, that a strided CONVERTING copy is staged through (a same-dtype SIMD strided copy into an
        ///     L1-resident block, then the SIMD contiguous cast out of it) — NDPolySeries.CopyInto's measured choice.
        /// </summary>
        private const int CastStageElements = 1024;

        /// <summary>
        ///     The per-call step of a loop over memory the add loop reads in place: <c>*acc += pairwise_sum(a[0..n) at
        ///     stride)</c> with the loop's own arithmetic. <paramref name="stride"/> is in ELEMENTS of the loop dtype.
        /// </summary>
        /// <param name="a">Address of the call's first element (loop dtype).</param>
        /// <param name="n">The call's element count (&gt;= 1).</param>
        /// <param name="stride">Element stride of the call.</param>
        /// <param name="acc">The running result (loop dtype), updated in place.</param>
        private delegate void Accumulate(byte* a, long n, long stride, byte* acc);

        /// <summary>
        ///     Whether this engine reproduces NumPy's add loop for <paramref name="loop"/>: the inexact loops, whose bits
        ///     depend on the summation ORDER. Integer and bool loops are exact in any order (modular / logical), so the
        ///     existing SIMD kernels already are NumPy's answer there.
        /// </summary>
        /// <param name="loop">The reduce loop dtype.</param>
        /// <returns>True for float16, float32, float64 and complex128.</returns>
        internal static bool SupportsLoop(NPTypeCode loop)
            => loop == NPTypeCode.Half || loop == NPTypeCode.Single || loop == NPTypeCode.Double || loop == NPTypeCode.Complex;

        /// <summary>
        ///     Reduces EVERY element of <paramref name="arr"/> into <paramref name="slot"/> in loop dtype
        ///     <paramref name="loop"/>, exactly as NumPy's <c>np.add.reduce(arr, axis=None, dtype=loop)</c> does — the
        ///     identity start, NpyIter's call schedule, a pairwise sum per call.
        /// </summary>
        /// <param name="arr">The operand; any dtype castable to <paramref name="loop"/>, any non-broadcast layout.</param>
        /// <param name="loop">The loop dtype (see <see cref="SupportsLoop"/>); the operand is cast into it per element,
        /// unsafely, when its own dtype differs — NumPy's buffered cast.</param>
        /// <param name="maximumSize">The buffer size in elements, <see cref="np.getbufsize"/>: it decides the fills, as
        /// NumPy's buffer size does.</param>
        /// <param name="slot">Receives the result in the loop dtype (at least 16 bytes; written only on success).</param>
        /// <returns>False — nothing written — when the operand is outside this engine's model (see the file header's
        /// declines); the caller then keeps its previous route.</returns>
        [SkipLocalsInit]
        internal static bool TryReduceFlat(NDArray arr, NPTypeCode loop, long maximumSize, byte* slot)
        {
            if (!SupportsLoop(loop) || maximumSize < 1)
                return false;

            var shape = arr.Shape;
            int nd = shape.NDim;
            if (nd > 64 || shape.size == 0)
                return false;

            var dims = shape.dimensions;
            var strides = shape.strides;
            // A broadcast axis (stride 0, extent > 1) makes NpyIter's axis sort skip that operand's comparisons, an
            // order FlatIteratorAxes does not reproduce; extent-1 axes are dropped by it and are harmless.
            for (int d = 0; d < nd; d++)
                if (strides[d] == 0 && dims[d] > 1)
                    return false;

            NPTypeCode src = arr.GetTypeCode;
            bool cast = src != loop;
            // The casts below are house IL kernels: without dynamic code there is no exact cast to run, and Decimal (no
            // NumPy dtype) has no NumPy cast semantics to reproduce — both keep their previous route.
            if (cast && (src == NPTypeCode.Decimal || !DirectILKernelGenerator.Enabled))
                return false;

            int srcSize = src.SizeOf();
            byte* x = (byte*)arr.Address + shape.offset * srcSize;

            Span<long> cd = stackalloc long[Math.Max(nd, 1)];
            Span<long> cs = stackalloc long[Math.Max(nd, 1)];
            int c = NumPyMinMaxReduce.FlatIteratorAxes(dims, strides, cd, cs);
            if (c == 0)
            {
                // A single element (every extent 1): one call of one element. Its stride is irrelevant.
                cd[0] = 1;
                cs[0] = 1;
                c = 1;
            }

            // The identity: +0.0 for every supported loop is all-zero bits (complex: both parts).
            Unsafe.InitBlockUnaligned(slot, 0, 16);

            var axes = cd[..c];
            var axisStrides = cs[..c];

            if (!cast)
            {
                if (c == 1)
                {
                    // One run, no cast: one call over the whole run, in place (GROWINNER lets it span any length).
                    AccumulatorFor(loop)(x, axes[0], axisStrides[0], slot);
                    return true;
                }

                int bestDim = NumPyMinMaxReduce.FlatBufferingDim(axes, maximumSize, out long core, out long best);
                if (bestDim == 0)
                {
                    RunsInPlace(x, axes, axisStrides, srcSize, AccumulatorFor(loop), slot);
                    return true;
                }

                Buffered(x, axes, axisStrides, src, loop, core, best, maximumSize, slot);
                return true;
            }

            if (c == 1)
            {
                // One run with a cast: fills of bufsize elements along the run.
                var run = new FillCursor(x, src, srcSize, axes[..1], axisStrides[..1], stackalloc long[1]);
                RunInFills(ref run, axes[0], loop, maximumSize, slot);
                return true;
            }

            int castDim = NumPyMinMaxReduce.FlatBufferingDim(axes, maximumSize, out long castCore, out long castBest, castOperand: true);
            if (castDim == 0)
            {
                // Each run alone reaches the buffer size: fills of bufsize elements, never across a run's end — the cursor
                // walks every run in iteration order, and each run is filled on its own.
                var runs = new FillCursor(x, src, srcSize, axes, axisStrides, stackalloc long[c]);
                long runCount = 1;
                for (int j = 1; j < c; j++) runCount *= axes[j];
                for (long r = 0; r < runCount; r++)
                    RunInFills(ref runs, axes[0], loop, maximumSize, slot);
                return true;
            }

            Buffered(x, axes, axisStrides, src, loop, castCore, castBest, maximumSize, slot);
            return true;
        }

        /// <summary>
        ///     The UNBUFFERED multi-run schedule: one call per axis-0 run, in place, in iteration order (axis 1 fastest).
        /// </summary>
        /// <param name="x">Logical element 0 of the operand.</param>
        /// <param name="cd">Coalesced extents, innermost first (length &gt;= 2).</param>
        /// <param name="cs">Coalesced element strides, innermost first.</param>
        /// <param name="elemSize">Element size in bytes (operand dtype == loop dtype here).</param>
        /// <param name="accumulate">The loop's per-call step.</param>
        /// <param name="slot">The running result.</param>
        [SkipLocalsInit]
        private static void RunsInPlace(byte* x, ReadOnlySpan<long> cd, ReadOnlySpan<long> cs, int elemSize, Accumulate accumulate, byte* slot)
        {
            Span<long> coord = stackalloc long[cd.Length];
            coord.Clear();   // [SkipLocalsInit]: the odometer's origin must be written
            long off = 0;
            while (true)
            {
                accumulate(x + off * elemSize, cd[0], cs[0], slot);
                if (!NextRun(cd, cs, coord, ref off))
                    return;
            }
        }

        /// <summary>
        ///     The BUFFERED multi-run schedule: the operand's runs, in iteration order, form fills of
        ///     <c>coreSize·⌊maximumSize/coreSize⌋</c> elements — or the outer block when it is smaller — never across the
        ///     outer block's end, each fill ONE call: NumPy's buffer fills. Each fill is streamed (see the file header),
        ///     never materialized whole.
        /// </summary>
        /// <param name="x">Logical element 0 of the operand.</param>
        /// <param name="cd">Coalesced extents, innermost first (length &gt;= 2).</param>
        /// <param name="cs">Coalesced element strides, innermost first.</param>
        /// <param name="src">The operand dtype.</param>
        /// <param name="loop">The loop dtype the fills hold.</param>
        /// <param name="coreSize">Elements of the core (whole axis-0 runs).</param>
        /// <param name="blockElems">Elements one outer index spans (the core times the outer extent).</param>
        /// <param name="maximumSize">The buffer size in elements.</param>
        /// <param name="slot">The running result.</param>
        [SkipLocalsInit]
        private static void Buffered(byte* x, ReadOnlySpan<long> cd, ReadOnlySpan<long> cs, NPTypeCode src, NPTypeCode loop,
            long coreSize, long blockElems, long maximumSize, byte* slot)
        {
            long n0 = cd[0];
            long chunkElems = blockElems > maximumSize ? coreSize * Math.Max(1, maximumSize / coreSize) : blockElems;
            long rowsPerChunk = chunkElems / n0;
            long rowsPerBlock = blockElems / n0;
            long totalRows = 1;
            for (int j = 1; j < cd.Length; j++) totalRows *= cd[j];

            var cursor = new FillCursor(x, src, src.SizeOf(), cd, cs, stackalloc long[cd.Length]);
            // Every block holds rowsPerBlock whole runs and the runs come in iteration order, so the fills are: per
            // block, rowsPerChunk-row fills and a final shorter one at the block's end.
            for (long row = 0; row < totalRows; row += rowsPerBlock)
            {
                for (long done = 0; done < rowsPerBlock; done += rowsPerChunk)
                {
                    long rows = Math.Min(rowsPerChunk, rowsPerBlock - done);
                    AccumulateFill(ref cursor, rows * n0, loop, slot);
                }
            }
        }

        /// <summary>
        ///     The cursor's current run, reduced in fills of at most <paramref name="maximumSize"/> elements along it — the
        ///     buffered schedule of a CAST run that alone reaches the buffer size (or is the operand's only run).
        /// </summary>
        /// <param name="cursor">Positioned at the run's first element; left at the next run's.</param>
        /// <param name="n">The run's element count.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="maximumSize">The buffer size in elements.</param>
        /// <param name="slot">The running result.</param>
        private static void RunInFills(ref FillCursor cursor, long n, NPTypeCode loop, long maximumSize, byte* slot)
        {
            for (long k = 0; k < n; k += maximumSize)
                AccumulateFill(ref cursor, Math.Min(maximumSize, n - k), loop, slot);
        }

        /// <summary>
        ///     One buffered call: <c>*slot += pairwise_sum(the next <paramref name="n"/> elements of the cursor)</c> in the
        ///     loop's arithmetic, the sum streamed through an L1 block (see the file header).
        /// </summary>
        /// <param name="cursor">The fill's source; advanced past the fill.</param>
        /// <param name="n">The fill's element count.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="slot">The running result.</param>
        [SkipLocalsInit]
        private static void AccumulateFill(ref FillCursor cursor, long n, NPTypeCode loop, byte* slot)
        {
            byte* block = stackalloc byte[StreamBlockElements * 16];
            switch (loop)
            {
                case NPTypeCode.Double:
                    *(double*)slot = *(double*)slot + StreamDouble(ref cursor, n, loop, block);
                    return;
                case NPTypeCode.Single:
                    *(float*)slot = *(float*)slot + StreamSingle(ref cursor, n, loop, block);
                    return;
                case NPTypeCode.Half:
                {
                    // HALF_add: the float32 pairwise sum added to the widened result, narrowed back per call.
                    float io1 = DirectILKernelGenerator.HalfToFloatScalarExact(*(ushort*)slot) + StreamHalf(ref cursor, n, loop, block);
                    *(ushort*)slot = DirectILKernelGenerator.SingleToHalfBits(io1);
                    return;
                }
                default:
                {
                    StreamComplex(ref cursor, n, loop, block, out double rr, out double ri);
                    double* r = (double*)slot;
                    r[0] = r[0] + rr;
                    r[1] = r[1] + ri;
                    return;
                }
            }
        }

        /// <summary>
        ///     NumPy's DOUBLE_pairwise_sum of the cursor's next <paramref name="n"/> elements: the recursion's own split
        ///     down to sub-ranges of at most <see cref="StreamBlockElements"/>, each folded in place when it can be
        ///     (<see cref="FillCursor.TryTakeInPlace"/>) or else read into <paramref name="block"/> and folded there; the
        ///     left sub-range is read (and summed) before the right one, the cursor's order.
        /// </summary>
        /// <param name="cursor">The source; advanced past the range.</param>
        /// <param name="n">Element count.</param>
        /// <param name="loop">The loop dtype (float64) the block holds.</param>
        /// <param name="block">The L1 scratch block.</param>
        /// <returns>The pairwise sum.</returns>
        private static double StreamDouble(ref FillCursor cursor, long n, NPTypeCode loop, byte* block)
        {
            if (n <= StreamBlockElements)
            {
                var fold = ILKernelGenerator.TryGetPairwiseFoldDouble();
                if (cursor.TryTakeInPlace(n, loop, out byte* run, out long stride))
                    return fold != null ? fold(run, n, stride) : ILKernelGenerator.PairwiseFold((double*)run, n, stride);
                cursor.Read(n, loop, block);
                return fold != null ? fold(block, n, 1) : ILKernelGenerator.PairwiseFold((double*)block, n, 1);
            }

            long n2 = n / 2;
            n2 -= n2 % 8;
            double left = StreamDouble(ref cursor, n2, loop, block);
            return left + StreamDouble(ref cursor, n - n2, loop, block);
        }

        /// <summary>NumPy's FLOAT_pairwise_sum of the cursor's next <paramref name="n"/> elements, streamed (see <see cref="StreamDouble"/>).</summary>
        /// <param name="cursor">The source; advanced past the range.</param>
        /// <param name="n">Element count.</param>
        /// <param name="loop">The loop dtype (float32) the block holds.</param>
        /// <param name="block">The L1 scratch block.</param>
        /// <returns>The float32 pairwise sum.</returns>
        private static float StreamSingle(ref FillCursor cursor, long n, NPTypeCode loop, byte* block)
        {
            if (n <= StreamBlockElements)
            {
                var fold = ILKernelGenerator.TryGetPairwiseFoldSingle();
                if (cursor.TryTakeInPlace(n, loop, out byte* run, out long stride))
                    return fold != null ? fold(run, n, stride) : ILKernelGenerator.PairwiseFold((float*)run, n, stride);
                cursor.Read(n, loop, block);
                return fold != null ? fold(block, n, 1) : ILKernelGenerator.PairwiseFold((float*)block, n, 1);
            }

            long n2 = n / 2;
            n2 -= n2 % 8;
            float left = StreamSingle(ref cursor, n2, loop, block);
            return left + StreamSingle(ref cursor, n - n2, loop, block);
        }

        /// <summary>
        ///     NumPy's HALF_pairwise_sum (float32 accumulation of widened float16 elements) of the cursor's next
        ///     <paramref name="n"/> elements, streamed (see <see cref="StreamDouble"/>).
        /// </summary>
        /// <param name="cursor">The source; advanced past the range.</param>
        /// <param name="n">Element count.</param>
        /// <param name="loop">The loop dtype (float16) the block holds.</param>
        /// <param name="block">The L1 scratch block.</param>
        /// <returns>The float32 pairwise sum.</returns>
        private static float StreamHalf(ref FillCursor cursor, long n, NPTypeCode loop, byte* block)
        {
            if (n <= StreamBlockElements)
            {
                if (cursor.TryTakeInPlace(n, loop, out byte* run, out long stride))
                    return ILKernelGenerator.PairwiseFoldHalf((ushort*)run, n, stride);
                cursor.Read(n, loop, block);
                return ILKernelGenerator.PairwiseFoldHalf((ushort*)block, n, 1);
            }

            long n2 = n / 2;
            n2 -= n2 % 8;
            float left = StreamHalf(ref cursor, n2, loop, block);
            return left + StreamHalf(ref cursor, n - n2, loop, block);
        }

        /// <summary>
        ///     NumPy's CDOUBLE_pairwise_sum of the cursor's next <paramref name="n"/> complexes, streamed with the complex
        ///     split rule (<c>(n - n%8)/2</c> complexes on the left).
        /// </summary>
        /// <param name="cursor">The source; advanced past the range.</param>
        /// <param name="n">Complex count.</param>
        /// <param name="loop">The loop dtype (complex128) the block holds.</param>
        /// <param name="block">The L1 scratch block.</param>
        /// <param name="rr">Receives the real-part sum.</param>
        /// <param name="ri">Receives the imaginary-part sum.</param>
        private static void StreamComplex(ref FillCursor cursor, long n, NPTypeCode loop, byte* block, out double rr, out double ri)
        {
            if (n <= StreamBlockElements)
            {
                if (cursor.TryTakeInPlace(n, loop, out byte* run, out long stride))
                {
                    FoldComplex(run, n, stride, out rr, out ri);
                    return;
                }

                cursor.Read(n, loop, block);
                FoldComplex(block, n, 1, out rr, out ri);
                return;
            }

            long left = (n - n % 8) / 2;
            StreamComplex(ref cursor, left, loop, block, out double lr, out double li);
            StreamComplex(ref cursor, n - left, loop, block, out double rr2, out double ri2);
            rr = lr + rr2;
            ri = li + ri2;
        }

        /// <summary>
        ///     A sequential reader over the operand's runs in iteration order (axis 0 the run, axis 1 fastest across runs):
        ///     each <see cref="Read"/> continues where the previous one stopped, crossing run ends as the buffer fill does.
        /// </summary>
        private ref struct FillCursor
        {
            private readonly byte* _x;
            private readonly NPTypeCode _src;
            private readonly int _srcSize;
            private readonly ReadOnlySpan<long> _cd;
            private readonly ReadOnlySpan<long> _cs;
            private readonly Span<long> _coord;
            private long _runOffset;   // element offset of the current run's first element
            private long _pos;         // elements already read from the current run

            /// <summary>Positions a cursor at the operand's first element.</summary>
            /// <param name="x">Logical element 0 of the operand.</param>
            /// <param name="src">The operand dtype.</param>
            /// <param name="srcSize">Its element size in bytes.</param>
            /// <param name="cd">Coalesced extents, innermost first (cd[0] is the run length).</param>
            /// <param name="cs">Coalesced element strides, innermost first.</param>
            /// <param name="coord">Odometer storage, one slot per axis (cleared here).</param>
            public FillCursor(byte* x, NPTypeCode src, int srcSize, ReadOnlySpan<long> cd, ReadOnlySpan<long> cs, Span<long> coord)
            {
                _x = x;
                _src = src;
                _srcSize = srcSize;
                _cd = cd;
                _cs = cs;
                _coord = coord;
                _coord.Clear();
                _runOffset = 0;
                _pos = 0;
            }

            /// <summary>
            ///     Takes the next <paramref name="n"/> elements IN PLACE, as a run of the loop dtype, when a fold can read
            ///     them where they are: they lie inside the current run, the operand reads as the loop dtype without a
            ///     conversion — the same dtype, or a complex128 operand in a float64 loop, whose NumPy cast keeps the real
            ///     part, every other double — and the fold reads that stride with SIMD loads
            ///     (<see cref="ILKernelGenerator.PairwiseFoldIsVectorized"/>). Otherwise nothing moves and the caller copies.
            /// </summary>
            /// <param name="n">Element count (the caller never asks past the operand's end).</param>
            /// <param name="loop">The loop dtype the fold works in.</param>
            /// <param name="first">Receives the first element's address (operand memory).</param>
            /// <param name="loopStride">Receives the element stride in LOOP-dtype elements (twice the complex stride for
            /// a real part).</param>
            /// <returns>True — the cursor advanced past the elements — when they can be folded in place.</returns>
            /// <remarks>
            ///     A pairwise sum is fixed by the sequence of values alone, and the SIMD run folds are bit-identical to the
            ///     contiguous one, so folding in place gives the copy's bits. It saves the copy: for the real part of a
            ///     complex128 array (np.sum(c, dtype=float64)) the stride-2 fold replaces a strided deinterleave into the
            ///     block.
            /// </remarks>
            public bool TryTakeInPlace(long n, NPTypeCode loop, out byte* first, out long loopStride)
            {
                first = null;
                loopStride = 0;
                long n0 = _cd[0], s0 = _cs[0];
                if (n > n0 - _pos)
                    return false;   // the elements cross into the next run: only a copy makes them one sequence

                long stride;
                if (_src == loop)
                    stride = s0;
                else if (_src == NPTypeCode.Complex && loop == NPTypeCode.Double)
                    stride = 2 * s0;
                else
                    return false;   // a converting cast: the fold needs the converted values
                if (!ILKernelGenerator.PairwiseFoldIsVectorized(loop, stride))
                    return false;

                first = _x + (_runOffset + _pos * s0) * _srcSize;
                loopStride = stride;
                _pos += n;
                if (_pos == n0)
                {
                    _pos = 0;
                    NextRun(_cd, _cs, _coord, ref _runOffset);
                }

                return true;
            }

            /// <summary>
            ///     Copies (casting to <paramref name="loop"/>) the next <paramref name="n"/> elements into the contiguous
            ///     <paramref name="dst"/>, moving to the next run whenever one ends.
            /// </summary>
            /// <param name="n">Element count (the caller never asks past the operand's end).</param>
            /// <param name="loop">The loop dtype <paramref name="dst"/> holds.</param>
            /// <param name="dst">Destination, <paramref name="n"/> loop-dtype elements.</param>
            public void Read(long n, NPTypeCode loop, byte* dst)
            {
                int loopSize = loop.SizeOf();
                long n0 = _cd[0], s0 = _cs[0];
                while (n > 0)
                {
                    long take = Math.Min(n, n0 - _pos);
                    CopyOrCastRun(_x + (_runOffset + _pos * s0) * _srcSize, _src, s0, take, loop, dst);
                    dst += take * loopSize;
                    n -= take;
                    _pos += take;
                    if (_pos == n0)
                    {
                        _pos = 0;
                        NextRun(_cd, _cs, _coord, ref _runOffset);
                    }
                }
            }
        }

        /// <summary>
        ///     Advances the axis-0 run odometer (axes 1.. of the coalesced operand, axis 1 fastest) — NumPy's iteration
        ///     order over the runs.
        /// </summary>
        /// <param name="cd">Coalesced extents, innermost first.</param>
        /// <param name="cs">Coalesced element strides, innermost first.</param>
        /// <param name="coord">The odometer (one slot per axis; slot 0 unused).</param>
        /// <param name="off">The current run's element offset, moved to the next run's.</param>
        /// <returns>False when every run has been visited.</returns>
        private static bool NextRun(ReadOnlySpan<long> cd, ReadOnlySpan<long> cs, Span<long> coord, ref long off)
        {
            for (int j = 1; j < cd.Length; j++)
            {
                if (++coord[j] < cd[j])
                {
                    off += cs[j];
                    return true;
                }

                coord[j] = 0;
                off -= (cd[j] - 1) * cs[j];
            }

            return false;
        }

        /// <summary>
        ///     Copies <paramref name="n"/> operand elements at element stride <paramref name="stride"/> into the contiguous
        ///     loop-dtype buffer <paramref name="dst"/>, casting each one exactly as NumPy's buffered cast does (unsafe:
        ///     a float truncates into an integer loop, a complex keeps its real part in a real loop, a float64 narrows to
        ///     float16 in one rounding). Same-dtype runs are a plain (SIMD) copy.
        /// </summary>
        /// <remarks>
        ///     The dispatch mirrors <c>NDPolySeries.CopyInto</c>'s measured choices over the same house kernels: a
        ///     contiguous source takes the SIMD contiguous cast; a strided source whose contiguous cast vectorizes is
        ///     staged through an L1 block (same-dtype SIMD strided copy, then the contiguous cast); float16 pairs take the
        ///     house strided cast; everything else the scalar convert-copy, which is exact for every pair.
        /// </remarks>
        /// <param name="srcRun">The run's first element (operand dtype).</param>
        /// <param name="src">The operand dtype.</param>
        /// <param name="stride">Element stride of the run (operand elements; may be negative).</param>
        /// <param name="n">Element count.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="dst">Contiguous destination, <paramref name="n"/> loop-dtype elements.</param>
        /// <exception cref="NotSupportedException">A same-dtype copy of an element width other than 1/2/4/8/16 bytes —
        /// unreachable through <see cref="TryReduceFlat"/>, which declines Decimal before any copy.</exception>
        [SkipLocalsInit]
        private static void CopyOrCastRun(byte* srcRun, NPTypeCode src, long stride, long n, NPTypeCode loop, byte* dst)
        {
            int srcSize = src.SizeOf();
            if (src == loop)
            {
                // A plain element copy by width. The 1-byte case is reached by the cast staging below (a bool / int8 /
                // uint8 source copied into the stage before its contiguous cast), never as a loop dtype.
                switch (srcSize)
                {
                    case 1: NumPyMinMaxReduce.CopyRun(dst, 1, srcRun, stride, n); return;
                    case 2: NumPyMinMaxReduce.CopyRun((ushort*)dst, 1, (ushort*)srcRun, stride, n); return;
                    case 4: NumPyMinMaxReduce.CopyRun((float*)dst, 1, (float*)srcRun, stride, n); return;
                    case 8: NumPyMinMaxReduce.CopyRun((double*)dst, 1, (double*)srcRun, stride, n); return;
                    case 16: NumPyMinMaxReduce.CopyRun((Complex*)dst, 1, (Complex*)srcRun, stride, n); return;
                    default: throw new NotSupportedException($"no element copy for {src} ({srcSize} bytes)");
                }
            }

            long strideBytes = stride * srcSize;
            if (stride == 1)
            {
                var contiguous = DirectILKernelGenerator.GetPolyContiguousCast(src, loop);
                if (contiguous != null)
                {
                    contiguous(srcRun, dst, n);
                    return;
                }
            }
            else if (n >= DirectILKernelGenerator.PolyHouseKernelThreshold)
            {
                if (srcSize <= 8 && DirectILKernelGenerator.IsVectorizedContiguousCast(src, loop))
                {
                    // Both halves vectorize: the same-dtype strided copy into an L1 block, then the contiguous cast.
                    var cast = DirectILKernelGenerator.GetPolyContiguousCast(src, loop);
                    if (cast != null)
                    {
                        byte* stage = stackalloc byte[CastStageElements * 8];
                        int loopSize = loop.SizeOf();
                        for (long k = 0; k < n; k += CastStageElements)
                        {
                            long m = Math.Min(CastStageElements, n - k);
                            CopyOrCastRun(srcRun + k * strideBytes, src, stride, m, src, stage);
                            cast(stage, dst + k * loopSize, m);
                        }

                        return;
                    }
                }
                else if (DirectILKernelGenerator.PolyScalarIsEmulated(src) || DirectILKernelGenerator.PolyScalarIsEmulated(loop))
                {
                    // float16 has no scalar hardware conversion: the house strided cast converts 8 lanes at a time.
                    var strided = DirectILKernelGenerator.GetPolyStridedCast(src, loop);
                    if (strided != null)
                    {
                        long srcStride = stride, dstStride = 1, extent = n;
                        strided(srcRun, dst, &srcStride, &dstStride, &extent, 1);
                        return;
                    }
                }
            }

            DirectILKernelGenerator.GetPolyCastKernel(src, loop)(srcRun, n, strideBytes, dst);
        }

        /// <summary>The per-call step of NumPy's loop for <paramref name="loop"/> over memory read in place.</summary>
        /// <param name="loop">A supported loop dtype.</param>
        /// <returns>The step.</returns>
        private static Accumulate AccumulatorFor(NPTypeCode loop) => loop switch
        {
            NPTypeCode.Double => s_double,
            NPTypeCode.Single => s_single,
            NPTypeCode.Half => s_half,
            _ => s_complex,
        };

        private static readonly Accumulate s_double = AccumulateDouble;
        private static readonly Accumulate s_single = AccumulateSingle;
        private static readonly Accumulate s_half = AccumulateHalf;
        private static readonly Accumulate s_complex = AccumulateComplex;

        /// <summary>
        ///     DOUBLE_add's reduce step: <c>*acc += DOUBLE_pairwise_sum(call)</c>. The SIMD fold when the host has one,
        ///     else the scalar port — the same bits.
        /// </summary>
        /// <param name="a">The call's first float64.</param>
        /// <param name="n">Element count.</param>
        /// <param name="stride">Element stride.</param>
        /// <param name="acc">The running float64 result.</param>
        private static void AccumulateDouble(byte* a, long n, long stride, byte* acc)
        {
            var fold = ILKernelGenerator.TryGetPairwiseFoldDouble();
            double s = fold != null ? fold(a, n, stride) : ILKernelGenerator.PairwiseFold((double*)a, n, stride);
            *(double*)acc = *(double*)acc + s;
        }

        /// <summary>FLOAT_add's reduce step: <c>*acc += FLOAT_pairwise_sum(call)</c>, accumulated in float32.</summary>
        /// <param name="a">The call's first float32.</param>
        /// <param name="n">Element count.</param>
        /// <param name="stride">Element stride.</param>
        /// <param name="acc">The running float32 result.</param>
        private static void AccumulateSingle(byte* a, long n, long stride, byte* acc)
        {
            var fold = ILKernelGenerator.TryGetPairwiseFoldSingle();
            float s = fold != null ? fold(a, n, stride) : ILKernelGenerator.PairwiseFold((float*)a, n, stride);
            *(float*)acc = *(float*)acc + s;
        }

        /// <summary>
        ///     HALF_add's reduce step: <c>io1 = half_to_float(*acc); io1 += HALF_pairwise_sum(call); *acc =
        ///     float_to_half(io1)</c> — the pairwise sum accumulates in float32 and the running result is narrowed back
        ///     to float16 after EVERY call, which is why the fills are observable for float16.
        /// </summary>
        /// <param name="a">The call's first float16.</param>
        /// <param name="n">Element count.</param>
        /// <param name="stride">Element stride.</param>
        /// <param name="acc">The running float16 result (raw bits).</param>
        private static void AccumulateHalf(byte* a, long n, long stride, byte* acc)
        {
            float s = ILKernelGenerator.PairwiseFoldHalf((ushort*)a, n, stride);
            float io1 = DirectILKernelGenerator.HalfToFloatScalarExact(*(ushort*)acc) + s;
            *(ushort*)acc = DirectILKernelGenerator.SingleToHalfBits(io1);
        }

        /// <summary>
        ///     CDOUBLE_add's reduce step: <c>*or += rr; *oi += ri</c> with <c>(rr, ri)</c> from CDOUBLE_pairwise_sum
        ///     (4 accumulators per part over the interleaved doubles — not the real loops' 8).
        /// </summary>
        /// <param name="a">The call's first complex128.</param>
        /// <param name="n">Complex element count.</param>
        /// <param name="stride">Complex element stride.</param>
        /// <param name="acc">The running complex128 result (real, imaginary).</param>
        private static void AccumulateComplex(byte* a, long n, long stride, byte* acc)
        {
            FoldComplex(a, n, stride, out double rr, out double ri);
            double* r = (double*)acc;
            r[0] = r[0] + rr;
            r[1] = r[1] + ri;
        }

        /// <summary>CDOUBLE_pairwise_sum of <paramref name="n"/> complexes at <paramref name="stride"/> complexes.</summary>
        /// <param name="a">The first complex.</param>
        /// <param name="n">Complex count.</param>
        /// <param name="stride">Complex stride.</param>
        /// <param name="rr">Receives the real-part sum.</param>
        /// <param name="ri">Receives the imaginary-part sum.</param>
        private static void FoldComplex(byte* a, long n, long stride, out double rr, out double ri)
        {
            var fold = ILKernelGenerator.TryGetPairwiseFoldComplex();
            if (fold != null)
            {
                Vector128<double> v = fold(a, n, stride * 16);
                rr = v.GetElement(0);
                ri = v.GetElement(1);
            }
            else
            {
                PairwiseComplexScalar((double*)a, n, stride * 2, out rr, out ri);
            }
        }

        /// <summary>
        ///     Scalar port of NumPy's <c>CDOUBLE_pairwise_sum</c> (loops_utils.h.src), the fallback where the emitted SIMD
        ///     complex fold is unavailable. NumPy counts DOUBLES: <c>n = 2·count</c>, a block of up to 128 doubles (64
        ///     complexes) keeps four accumulators per part, and a longer range splits at <c>n/2</c> rounded down to a
        ///     multiple of 8 doubles.
        /// </summary>
        /// <param name="a">First complex, as its real double.</param>
        /// <param name="count">Complex count.</param>
        /// <param name="strideDoubles">Distance between consecutive complexes, in doubles (2 = contiguous).</param>
        /// <param name="rr">Receives the real-part sum.</param>
        /// <param name="ri">Receives the imaginary-part sum.</param>
        private static void PairwiseComplexScalar(double* a, long count, long strideDoubles, out double rr, out double ri)
        {
            long n = 2 * count;
            if (n < 8)
            {
                rr = -0.0;
                ri = -0.0;
                for (long c = 0; c < count; c++)
                {
                    rr += a[c * strideDoubles];
                    ri += a[c * strideDoubles + 1];
                }

                return;
            }

            if (n <= 128)
            {
                double r0 = a[0], r1 = a[1];
                double r2 = a[strideDoubles], r3 = a[strideDoubles + 1];
                double r4 = a[2 * strideDoubles], r5 = a[2 * strideDoubles + 1];
                double r6 = a[3 * strideDoubles], r7 = a[3 * strideDoubles + 1];
                long i = 8;
                for (; i < n - (n % 8); i += 8)
                {
                    long c = i / 2;
                    r0 += a[c * strideDoubles]; r1 += a[c * strideDoubles + 1];
                    r2 += a[(c + 1) * strideDoubles]; r3 += a[(c + 1) * strideDoubles + 1];
                    r4 += a[(c + 2) * strideDoubles]; r5 += a[(c + 2) * strideDoubles + 1];
                    r6 += a[(c + 3) * strideDoubles]; r7 += a[(c + 3) * strideDoubles + 1];
                }

                rr = (r0 + r2) + (r4 + r6);
                ri = (r1 + r3) + (r5 + r7);
                for (; i < n; i += 2)
                {
                    long c = i / 2;
                    rr += a[c * strideDoubles];
                    ri += a[c * strideDoubles + 1];
                }

                return;
            }

            long n2 = n / 2;
            n2 -= n2 % 8;
            long split = n2 / 2;
            PairwiseComplexScalar(a, split, strideDoubles, out double rr1, out double ri1);
            PairwiseComplexScalar(a + split * strideDoubles, count - split, strideDoubles, out double rr2, out double ri2);
            rr = rr1 + rr2;
            ri = ri1 + ri2;
        }
    }

    public partial class DefaultEngine
    {
        /// <summary>
        ///     The flat (<c>axis=None</c>) <c>np.sum</c> of a float loop through <see cref="NumPyAddReduce"/> — NumPy's exact
        ///     add.reduce, the dtype= cast loops included — shaped like every other flat reduction result.
        /// </summary>
        /// <param name="arr">The operand (size &gt;= 2; the one-element case keeps <see cref="SumOfOneElement"/>).</param>
        /// <param name="keepdims">When true the result keeps the input rank with every axis length 1.</param>
        /// <param name="loopType">The loop dtype (float16 / float32 / float64 / complex128).</param>
        /// <param name="out">Optional output; when given, the sum is written to its first element and it is returned.</param>
        /// <returns>The sum — a read-only 0-d scalar, a <c>(1,)*ndim</c> array under <paramref name="keepdims"/>, or
        /// <paramref name="out"/> — or null when the engine declines (a broadcast operand, rank &gt; 64, a Decimal cast,
        /// no dynamic code for a cast), so the caller keeps its previous route.</returns>
        private unsafe NDArray TryExactFlatSum(NDArray arr, bool keepdims, NPTypeCode loopType, NDArray @out)
        {
            byte* slot = stackalloc byte[16];
            if (!NumPyAddReduce.TryReduceFlat(arr, loopType, np.getbufsize(), slot))
                return null;

            if (@out is not null)
            {
                @out.SetAtIndex(BoxLoopValue(slot, loopType), 0);
                return @out;
            }

            var r = NewFlatReductionResult(loopType, arr.ndim, keepdims);
            int size = loopType.SizeOf();
            Buffer.MemoryCopy(slot, (byte*)r.Address, size, size);
            return r.MarkReductionScalar();
        }

        /// <summary>
        ///     The flat (<c>axis=None</c>) <c>np.mean</c> through <see cref="NumPyAddReduce"/>, NumPy's <c>_methods._mean</c>
        ///     statement for statement: the sum runs in float64 for bool / integer / char input, in float32 for float16
        ///     input (the result narrowed back to float16), in the input's own dtype for float32 / float64 / complex128, or
        ///     in <c>dtype=</c> when given; then <c>ret / rcount</c> divides by the element count.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     <c>rcount</c> is an <c>np.intp</c> — a STRONG int64 scalar, not a Python int — so for every real loop the
        ///     division is a float64 one (<c>result_type(float32, int64)</c> is float64) whose quotient is then rounded to
        ///     the result dtype: a float32 mean of 2²⁴ + 1 ones is 0.99999994, not 1. A complex loop divides with NumPy's
        ///     complex loop by <c>n + 0j</c> (<see cref="ComplexDivideByCountLikeNumPy"/>).
        ///     </para>
        ///     <para>
        ///     A float16 input's default mean rounds DIFFERENTLY with and without <paramref name="keepdims"/>, because
        ///     <c>_mean</c> takes two different statements: the scalar sum gives <c>float16(sum / n)</c> — one rounding
        ///     from float64 — while the keepdims array is divided IN PLACE into its float32 sum
        ///     (<c>true_divide(ret, rcount, out=ret)</c>) and only then cast to float16, two roundings. They differ where
        ///     the float32 quotient lands exactly on a float16 tie (probed: 4096 ones and 4097 × 1.0009765625 give
        ///     <c>0x3c01</c> flat and <c>0x3c00</c> with keepdims).
        ///     </para>
        /// </remarks>
        /// <param name="arr">The operand (size &gt;= 2; one element keeps <see cref="SumOfOneElement"/>).</param>
        /// <param name="keepdims">When true the result keeps the input rank with every axis length 1.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c>, or null.</param>
        /// <returns>The mean, or null when the engine declines (a non-float <paramref name="dtype"/>, a Decimal input, a
        /// broadcast operand, …) so the caller keeps its previous route.</returns>
        private unsafe NDArray TryExactFlatMean(NDArray arr, bool keepdims, NPTypeCode? dtype)
        {
            var src = arr.GetTypeCode;
            NPTypeCode loop, result;
            if (dtype.HasValue)
            {
                if (!NumPyAddReduce.SupportsLoop(dtype.Value))
                    return null;
                loop = result = dtype.Value;
            }
            else if (src == NPTypeCode.Half)
            {
                // _mean: `dtype = mu.dtype('f4'); is_float16_result = True` — a float32 sum, a float16 result.
                loop = NPTypeCode.Single;
                result = NPTypeCode.Half;
            }
            else if (src == NPTypeCode.Single || src == NPTypeCode.Double || src == NPTypeCode.Complex)
            {
                loop = result = src;
            }
            else if (src == NPTypeCode.Decimal)
            {
                return null;   // no NumPy dtype: the previous decimal route stays
            }
            else
            {
                // _mean: `issubclass(arr.dtype.type, (nt.integer, nt.bool))` → dtype f8, a buffered cast per element.
                loop = result = NPTypeCode.Double;
            }

            byte* slot = stackalloc byte[16];
            if (!NumPyAddReduce.TryReduceFlat(arr, loop, np.getbufsize(), slot))
                return null;

            long count = arr.size;
            var r = NewFlatReductionResult(result, arr.ndim, keepdims);
            byte* dst = (byte*)r.Address;
            if (loop == NPTypeCode.Complex)
            {
                *(Complex*)dst = ComplexDivideByCountLikeNumPy(*(Complex*)slot, count);
            }
            else
            {
                // Widening the loop's sum to float64 is exact for all three real loops; the quotient is NumPy's float64 one.
                double sum = loop == NPTypeCode.Double ? *(double*)slot
                    : loop == NPTypeCode.Single ? *(float*)slot
                    : DirectILKernelGenerator.HalfToFloatScalarExact(*(ushort*)slot);
                double q = sum / count;
                if (result == NPTypeCode.Double)
                    *(double*)dst = q;
                else if (result == NPTypeCode.Single)
                    *(float*)dst = (float)q;
                else if (loop == NPTypeCode.Single && keepdims)
                    // The keepdims statement: the quotient stored into the float32 sum array, then that array cast.
                    *(ushort*)dst = DirectILKernelGenerator.SingleToHalfBits((float)q);
                else
                    *(ushort*)dst = DirectILKernelGenerator.DoubleToHalfBits(q);
            }

            return r.MarkReductionScalar();
        }

        /// <summary>
        ///     A fresh, uninitialized flat-reduction result: 0-d, or <c>(1,)*ndim</c> under <paramref name="keepdims"/>.
        /// </summary>
        /// <param name="type">The result dtype.</param>
        /// <param name="ndim">The reduced operand's rank.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <returns>The array; the caller writes its one element.</returns>
        private static NDArray NewFlatReductionResult(NPTypeCode type, int ndim, bool keepdims)
        {
            if (!keepdims || ndim == 0)
                return new NDArray(type, Shape.Scalar, false);
            var ones = new long[ndim];
            Array.Fill(ones, 1L);
            return new NDArray(type, new Shape(ones), false);
        }

        /// <summary>The loop-dtype value in <paramref name="slot"/>, boxed (for an <c>out=</c> element write).</summary>
        /// <param name="slot">The value's bytes.</param>
        /// <param name="loopType">float16 / float32 / float64 / complex128.</param>
        /// <returns>The boxed value.</returns>
        private static unsafe object BoxLoopValue(byte* slot, NPTypeCode loopType) => loopType switch
        {
            NPTypeCode.Double => *(double*)slot,
            NPTypeCode.Single => *(float*)slot,
            NPTypeCode.Half => BitConverter.UInt16BitsToHalf(*(ushort*)slot),
            _ => *(Complex*)slot,
        };
    }
}
