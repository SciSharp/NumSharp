using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged.Pooling;

// =============================================================================
// NumPyReduceIterator.cs — the iterator NumPy's PyUFunc_ReduceWrapper builds, ported for [out, input, where]
// =============================================================================
//
// A ufunc reduction with keywords (np.max(a, axis, out=, initial=, where=)) is not "the reduction, then a copy": NumPy runs
// ONE buffered NpyIter over the output, the input and the mask (numpy/_core/src/umath/reduction.c), and every observable
// detail falls out of how that iterator cuts the work into inner-loop CALLS:
//
//   * the AXIS ORDER is a stable insertion sort by |stride| over every operand (npyiter_find_best_axis_ordering): a mask
//     votes, and a given out votes, so an F-ordered input with a C-ordered mask iterates — and allocates its result — in C
//     order (probed: np.max(xF, axis=0, where=mC, initial=0) is C-contiguous, without the mask F);
//   * COALESCING merges adjacent axes only where every operand continues contiguously;
//   * the BUFFERING SETUP (npyiter_find_buffering_setup, NumPy 2.3+) picks the outer dimension and core by a cost model,
//     decides which operands are copied into buffers (a cast operand always is: an out of another dtype, an input whose
//     dtype is not the loop's), and whether the "reduce outer" double loop runs;
//   * each FILL then covers whole cores up to np.getbufsize() elements, never past the outer dimension's block, and is cut
//     into calls: one per fill, or one per core in reduce mode.
//
// Why the calls matter: the loop's answer depends on them. A cast out is read into the buffer (cast to the loop dtype) at
// every fill start and written back at its end, so the running maximum round-trips through out's dtype at every fill
// (probed: np.max(f64 [100]*20000 with 300 at index 100, out=int8) is 100, not int8(300) = 44). A where= mask splits
// every call into runs of selected elements, each its own loop call (generic_masked_strided_loop), so the float bits — a
// ±0 tie, a NaN's payload — follow the runs (probed: np.max([1, nan_payload, 3, 4], where=[T,T,F,T], initial=0) returns
// the canonical NaN, where=[T,T,T,F] the payload).
//
// The port keeps NumPy's structure: Build mirrors NpyIter_AdvancedNew for the reduction's flags (BUFFERED | EXTERNAL_LOOP |
// GROWINNER | DONT_NEGATE_STRIDES, the out READWRITE + ALLOCATE, the input NO_BROADCAST, the mask READONLY), and Run
// walks the fills and calls the way npyiter_copy_to_buffers / npyiter_buffered_reduce_iternext / reduce_loop do. The
// element work of each call is a caller-supplied loop: the min / max family hands in NumPy's own loop rules (the SIMD
// schedule of loops_minmax for float32 / float64 / integers, the sequential BINARY_LOOPs for float16, complex, bool).
// Strides are in ELEMENTS of each operand (NumPy's are bytes; every comparison and coalescing test is per operand, so the
// unit cancels).
// =============================================================================

namespace NumSharp.Backends
{
    /// <summary>
    ///     NumPy's reduction iterator over <c>[out, input, where]</c>, ported: the axis order, the allocated result's layout,
    ///     the coalesced axes, the buffering setup, and the walk of fills and inner-loop calls (see the file header).
    /// </summary>
    /// <remarks>
    ///     Built once per reduction by <see cref="Build"/>; <see cref="Run{T}"/> then drives a loop over the calls. The object
    ///     holds only the schedule (shapes and strides), no array references, so it can be built before the result exists.
    /// </remarks>
    internal sealed unsafe class NumPyReduceIterator
    {
        /// <summary>Operand index of the output (the accumulator NumPy reads and writes back).</summary>
        internal const int OpOut = 0;

        /// <summary>Operand index of the input being reduced.</summary>
        internal const int OpIn = 1;

        /// <summary>Operand index of the <c>where=</c> mask (present only when <see cref="NOp"/> is 3).</summary>
        internal const int OpMask = 2;

        /// <summary>Operand count: 2 without a mask, 3 with one.</summary>
        internal int NOp;

        /// <summary>Iteration rank after coalescing (at least 1 — NumPy keeps one axis even for a 0-d operand).</summary>
        internal int NDim;

        /// <summary>Iteration extents after coalescing, innermost first (<see cref="NDim"/> entries are valid).</summary>
        internal long[] Shape;

        /// <summary>
        ///     Per operand, the element stride along each coalesced iteration axis (innermost first), into the operand's ARRAY
        ///     (for the out: the given out, or the allocated result laid out by <see cref="ResultStrides"/>).
        /// </summary>
        internal long[][] Strides;

        /// <summary>Total iteration size (the product of <see cref="Shape"/>; 0 for an empty iteration).</summary>
        internal long IterSize;

        /// <summary>Elements of one core: the product of the extents below <see cref="OuterDim"/>.</summary>
        internal long CoreSize;

        /// <summary>The buffering setup's outer dimension (NumPy's <c>outerdim</c>).</summary>
        internal int OuterDim;

        /// <summary>Elements one fill may cover (NumPy's <c>buffersize</c> after the setup: a multiple of <see cref="CoreSize"/>).</summary>
        internal long BufferSize;

        /// <summary>True when the reduce double loop runs: each fill is cut into one call per core.</summary>
        internal bool UsingReduce;

        /// <summary>Per operand: copied into a buffer (cast, or not single-strided over the core).</summary>
        internal bool[] Buffered;

        /// <summary>Per operand: iterated as a reduce operand in reduce mode (its pointer repeats or steps per core).</summary>
        internal bool[] IsReduceOp;

        /// <summary>
        ///     Per operand, the stride a call sees: the array's innermost stride when unbuffered, 1 or 0 (in buffer elements)
        ///     when buffered.
        /// </summary>
        internal long[] InnerStride;

        /// <summary>Per operand, the advance from one core's call to the next in reduce mode (array or buffer elements).</summary>
        internal long[] ReduceOuterStride;

        /// <summary>The result's dims in result-axis order (the keepdims shape when keepdims is set).</summary>
        internal long[] ResultDims;

        /// <summary>
        ///     The allocated result's element strides in result-axis order — <c>npyiter_new_temp_array</c>'s layout, the
        ///     iteration order of the non-reduced axes — or <c>null</c> when the caller gave an out.
        /// </summary>
        internal long[] ResultStrides;

        /// <summary>
        ///     Builds the iterator NumPy's <c>PyUFunc_ReduceWrapper</c> builds for one reduction, up to the buffering setup.
        /// </summary>
        /// <param name="input">The input's shape (dims and element strides; the offset is the caller's).</param>
        /// <param name="axisFlags">One flag per input axis: true where the axis is reduced (<see cref="ReductionArguments.Validate(NDArray, int?, bool, NDArray, NDArray, string)"/>).</param>
        /// <param name="keepdims">Whether the reduced axes stay in the result with length 1.</param>
        /// <param name="outShape">The given out's shape, or <c>null</c> when the result is allocated (it then has no
        ///     strides when the axes are sorted, so it does not vote).</param>
        /// <param name="maskShape">The mask's shape (already validated to broadcast to the input), or <c>null</c>.</param>
        /// <param name="outCast">True when the out's dtype is not the loop dtype (NumPy's <c>NPY_OP_ITFLAG_CAST</c>: the out
        ///     is buffered).</param>
        /// <param name="inCast">True when the input's dtype is not the loop dtype (the input is buffered).</param>
        /// <param name="bufferSize">NumPy's maximum buffer size in elements (<see cref="np.getbufsize"/>).</param>
        /// <returns>The iterator.</returns>
        /// <exception cref="ArgumentException"><paramref name="axisFlags"/> does not have one entry per input axis.</exception>
        internal static NumPyReduceIterator Build(Shape input, bool[] axisFlags, bool keepdims, Shape? outShape, Shape? maskShape,
            bool outCast, bool inCast, long bufferSize)
        {
            int nd = input.NDim;
            if (axisFlags.Length != nd)
                throw new ArgumentException("NumPyReduceIterator: one axis flag per input axis is required.", nameof(axisFlags));

            var it = new NumPyReduceIterator { NOp = maskShape.HasValue ? 3 : 2 };
            int nop = it.NOp;
            long[] dims = input.dimensions;
            long[] xs = input.strides;

            // result_axes (PyUFunc_ReduceWrapper): the result axis each input axis maps to, -1 for a reduced axis keepdims
            // drops; a kept reduced axis has length 1.
            var resultAxis = new int[nd];
            int curr = 0;
            bool anyReduced = false;
            for (int c = 0; c < nd; c++)
            {
                anyReduced |= axisFlags[c];
                resultAxis[c] = axisFlags[c] && !keepdims ? -1 : curr++;
            }

            it.ResultDims = new long[curr];
            for (int c = 0; c < nd; c++)
                if (resultAxis[c] >= 0)
                    it.ResultDims[resultAxis[c]] = axisFlags[c] ? 1 : dims[c];

            // AXISDATA in reversed C order (npyiter_fill_axisdata): j indexes axisdata, C axis = nd-1-j. An extent-1 axis
            // gets stride 0 for EVERY operand (the iterator's bshape == 1 rule), so it never votes and always coalesces.
            // A 0-d input still has one axisdata of extent 1.
            int na = Math.Max(nd, 1);
            var adShape = new long[na];
            var adStride = new long[nop][];
            for (int op = 0; op < nop; op++)
                adStride[op] = new long[na];
            adShape[0] = 1;

            long[] outDims = outShape?.dimensions;
            long[] outStr = outShape?.strides;
            long[] maskDims = maskShape?.dimensions;
            long[] maskStr = maskShape?.strides;
            for (int j = 0; j < nd; j++)
            {
                int c = nd - 1 - j;
                long len = dims[c];
                adShape[j] = len;
                if (len == 1)
                    continue;

                adStride[OpIn][j] = xs[c];
                // The out: zero along a reduction axis; a given out's own stride elsewhere (validated to have the length);
                // an allocated out has no strides yet.
                int r = resultAxis[c];
                if (outShape.HasValue && r >= 0 && !axisFlags[c] && outDims[r] != 1)
                    adStride[OpOut][j] = outStr[r];
                if (nop == 3)
                {
                    int md = c - (nd - maskDims.Length);
                    if (md >= 0 && maskDims[md] != 1)
                        adStride[OpMask][j] = maskStr[md];
                }
            }

            // npyiter_find_best_axis_ordering: a custom stable insertion sort of the axisdata (smallest |stride| first).
            // Per pair, every operand whose two strides are both non-zero compares; the first comparison decides "swap",
            // any later operand that disagrees resets it ("in the case of conflicts between different operands, C-order
            // wins"); an all-ambiguous pair is skipped without stopping the scan.
            var perm = new int[na];
            for (int i = 0; i < na; i++)
                perm[i] = i;
            for (int i0 = 1; i0 < na; i0++)
            {
                int ipos = i0;
                int j0 = perm[i0];
                for (int i1 = i0 - 1; i1 >= 0; i1--)
                {
                    int j1 = perm[i1];
                    bool ambig = true, shouldSwap = false;
                    for (int op = 0; op < nop; op++)
                    {
                        long s0 = adStride[op][j0], s1 = adStride[op][j1];
                        if (s0 == 0 || s1 == 0)
                            continue;
                        if (Math.Abs(s1) <= Math.Abs(s0))
                            shouldSwap = false;
                        else if (ambig)
                            shouldSwap = true;
                        ambig = false;
                    }

                    if (!ambig)
                    {
                        if (shouldSwap)
                            ipos = i1;
                        else
                            break;
                    }
                }

                if (ipos != i0)
                {
                    for (int i1 = i0; i1 > ipos; i1--)
                        perm[i1] = perm[i1 - 1];
                    perm[ipos] = j0;
                }
            }

            // npyiter_new_temp_array (an allocated out): strides assigned walking the iteration order innermost first, each
            // result axis taking the running stride, which grows by its length (a kept reduced axis is 1 long). Then
            // npyiter_replace_axisdata gives the operand those strides wherever its length is not 1.
            if (!outShape.HasValue)
            {
                it.ResultStrides = new long[curr];
                long stride = 1;
                for (int i = 0; i < na && nd > 0; i++)
                {
                    int c = nd - 1 - perm[i];
                    int r = resultAxis[c];
                    if (r < 0)
                        continue;
                    it.ResultStrides[r] = stride;
                    stride *= it.ResultDims[r];
                }

                for (int j = 0; j < nd; j++)
                {
                    int c = nd - 1 - j;
                    int r = resultAxis[c];
                    adStride[OpOut][j] = r >= 0 && !axisFlags[c] && it.ResultDims[r] != 1 && dims[c] != 1 ? it.ResultStrides[r] : 0;
                }
            }

            // The iteration order: axisdata permuted.
            var shape = new long[na];
            var strides = new long[nop][];
            for (int op = 0; op < nop; op++)
                strides[op] = new long[na];
            for (int i = 0; i < na; i++)
            {
                shape[i] = adShape[perm[i]];
                for (int op = 0; op < nop; op++)
                    strides[op][i] = adStride[op][perm[i]];
            }

            // npyiter_coalesce_axes: merge the next axis into the running one when, for every operand, one of the two has
            // length 1 with stride 0 or the strides continue (stride0 * shape0 == stride1); a zero stride of the running
            // axis takes the merged axis' stride.
            int ndim = 1;
            for (int i = 1; i < na; i++)
            {
                int c0 = ndim - 1;
                long shape0 = shape[c0], shape1 = shape[i];
                bool can = true;
                for (int op = 0; op < nop && can; op++)
                {
                    long s0 = strides[op][c0], s1 = strides[op][i];
                    if (!((shape0 == 1 && s0 == 0) || (shape1 == 1 && s1 == 0)) && s0 * shape0 != s1)
                        can = false;
                }

                if (can)
                {
                    shape[c0] *= shape1;
                    for (int op = 0; op < nop; op++)
                        if (strides[op][c0] == 0)
                            strides[op][c0] = strides[op][i];
                }
                else
                {
                    shape[ndim] = shape1;
                    for (int op = 0; op < nop; op++)
                        strides[op][ndim] = strides[op][i];
                    ndim++;
                }
            }

            it.NDim = ndim;
            it.Shape = shape;
            it.Strides = strides;
            long size = 1;
            for (int i = 0; i < ndim; i++)
                size *= shape[i];
            it.IterSize = size;

            var cast = new bool[nop];
            cast[OpOut] = outCast;
            cast[OpIn] = inCast;
            var reduceFlag = new bool[nop];
            reduceFlag[OpOut] = anyReduced;
            it.FindBufferingSetup(cast, reduceFlag, bufferSize <= 0 ? 8192 : bufferSize);
            return it;
        }

        /// <summary>
        ///     <c>npyiter_find_buffering_setup</c>, ported line for line for this iterator (GROWINNER set, no operand asking for
        ///     contiguity): the outer dimension and core by NumPy's cost model, whether the reduce double loop runs, and per
        ///     operand whether it is buffered and which strides a call sees.
        /// </summary>
        /// <param name="cast">Per operand: needs a cast into the loop dtype (always buffered).</param>
        /// <param name="reduceFlag">Per operand: carries NumPy's <c>NPY_OP_ITFLAG_REDUCE</c> (the out of a reduction with a
        ///     reduced axis).</param>
        /// <param name="maximumSize">The buffer size cap in elements.</param>
        private void FindBufferingSetup(bool[] cast, bool[] reduceFlag, long maximumSize)
        {
            int nop = NOp;
            var singleStrideDims = new int[nop];
            var reduceOuterDim = new int[nop];
            int cost = 1;
            for (int op = 0; op < nop; op++)
            {
                singleStrideDims[op] = 1;
                if (cast[op])
                    cost++;
            }

            int outerReduceDim = 0;
            long size = Shape[0];
            int bestDim = 0, bestCost = cost;
            long bestSize = size, bestCore = 1;
            for (int idim = 1; idim < NDim; idim++)
            {
                if (outerReduceDim != 0)
                    break;   // "Cannot currently expand beyond reduce dim!"
                if (size >= maximumSize && cost > 1)
                    break;   // exceeded the buffer: only GROWINNER without buffers could still improve

                for (int op = 0; op < nop; op++)
                {
                    long prev = Strides[op][idim - 1], cur = Strides[op][idim];
                    if (singleStrideDims[op] == idim)
                    {
                        if (prev * Shape[idim - 1] == cur)
                        {
                            singleStrideDims[op]++;
                            continue;   // NumPy's `continue` also skips the reduce-outer test below
                        }

                        if (!cast[op])
                            cost++;
                    }

                    if (reduceFlag[op] && (cur == 0 || prev == 0))
                    {
                        reduceOuterDim[op] = idim;
                        outerReduceDim = idim;
                    }
                }

                long core = size;
                size *= Shape[idim];
                if (size == 0)
                    break;

                double bufsize = size;
                if (size > maximumSize && cost > 1)
                    bufsize = maximumSize;
                if (cost * (double)bestSize <= bestCost * bufsize)
                {
                    bestCost = cost;
                    bestCore = core;
                    bestSize = size;
                    bestDim = idim;
                }
            }

            bool usingReduce = outerReduceDim != 0 && bestDim == outerReduceDim;
            bool mustBuffer = false;
            Buffered = new bool[nop];
            IsReduceOp = new bool[nop];
            InnerStride = new long[nop];
            ReduceOuterStride = new long[nop];
            for (int op = 0; op < nop; op++)
            {
                bool buffered = cast[op];
                bool isReduce;
                long outerStride = Strides[op][outerReduceDim];
                if (!usingReduce)
                    isReduce = false;
                else if (reduceOuterDim[op] == bestDim)
                    isReduce = true;                                   // this op MUST use reduce semantics
                else if (singleStrideDims[op] == bestDim && !buffered)
                    isReduce = true;                                   // unbuffered: iterate it as a reduce operand anyway
                else if (outerStride == 0 && singleStrideDims[op] <= bestDim)
                    isReduce = true;                                   // buffer the core only, repeat it per outer index
                else
                    isReduce = false;

                bool singleStride = singleStrideDims[op] + (isReduce ? 1 : 0) > bestDim;
                if (!singleStride)
                    buffered = true;

                long inner, outer;
                if (buffered)
                {
                    if (isReduce)
                    {
                        if (outerStride == 0)
                        {
                            inner = 1;
                            outer = 0;
                        }
                        else
                        {
                            inner = 0;
                            outer = 1;
                        }
                    }
                    else if (singleStride && Strides[op][0] == 0)
                    {
                        inner = 0;   // always the same element: even the buffer holds one
                        outer = 0;
                    }
                    else
                    {
                        inner = 1;
                        outer = bestCore;
                    }
                }
                else
                {
                    inner = Strides[op][0];
                    outer = outerStride;
                }

                Buffered[op] = buffered;
                IsReduceOp[op] = isReduce;
                InnerStride[op] = inner;
                ReduceOuterStride[op] = usingReduce ? outer : 0;
                mustBuffer |= buffered;
            }

            // A buffered iteration caps the fill at the maximum size, rounded down to whole cores (never below one core).
            if (mustBuffer && maximumSize < bestSize)
                bestSize = bestCore * Math.Max(1, maximumSize / bestCore);

            UsingReduce = usingReduce;
            CoreSize = bestCore;
            OuterDim = bestDim;
            BufferSize = bestSize;
        }

        /// <summary>
        ///     One inner-loop call's element work: the NumPy loop's IS_BINARY_REDUCE form (<paramref name="reduce"/>, the out
        ///     stride is 0) or its elementwise form (<paramref name="binary"/>), both handed exact NumPy rules by the caller.
        /// </summary>
        /// <typeparam name="T">The loop dtype's element type.</typeparam>
        internal readonly struct Loop<T> where T : unmanaged
        {
            /// <summary>The reduce form: the out's running value, a run of input elements at a stride, their count → the
            ///     out's new value.</summary>
            internal readonly delegate*<T, T*, long, long, T> Reduce;

            /// <summary>The elementwise form: <c>o[i] = op(o[i], x[i])</c> over a run (out pointer and stride, input pointer
            ///     and stride, count).</summary>
            internal readonly delegate*<T*, long, T*, long, long, void> Binary;

            /// <summary>
            ///     The reduce form over a MASKED call in one piece (the out's running value, the input pointer and stride, the
            ///     mask pointer and byte stride, the count → the out's new value), or <c>null</c> (the call is then cut into
            ///     runs here, one <see cref="Reduce"/> call per run). A loop whose result cannot depend on where NumPy cuts
            ///     the call — an integer, bool or sequential loop — folds the selected elements however it likes; a loop whose
            ///     result does depend on the cuts (the float SIMD loops) must make NumPy's per-run calls itself, only faster
            ///     (a cheaper run scan, work moved off the out's dependency chain).
            /// </summary>
            internal readonly delegate*<T, T*, long, byte*, long, long, T> MaskedReduce;

            /// <summary>The elementwise form over a masked call in one piece (<c>o[i] = op(o[i], x[i])</c> where
            ///     <c>m[i]</c>), or <c>null</c> — the same rule as <see cref="MaskedReduce"/>.</summary>
            internal readonly delegate*<T*, long, T*, long, byte*, long, long, void> MaskedBinary;

            /// <summary>
            ///     The elementwise form over several CONSECUTIVE calls into the same out (the reduce double loop's rows when
            ///     the out does not move along the outer dimension): out pointer and stride, input pointer and stride, the
            ///     input's row stride, the row count, the count — <c>o[i] = op(...op(op(o[i], x0[i]), x1[i])..., xk[i])</c>, or
            ///     <c>null</c> (one <see cref="Binary"/> call per row). Each element must get exactly the sequence of ops its
            ///     per-row calls would apply (the same op per row position, rows in order); the form only saves the out's
            ///     load and store per row and the call per row.
            /// </summary>
            internal readonly delegate*<T*, long, T*, long, long, long, long, void> BinaryRows;

            /// <summary>Wraps the two unmasked forms (masked calls are cut into runs).</summary>
            /// <param name="reduce">The reduce form.</param>
            /// <param name="binary">The elementwise form.</param>
            internal Loop(delegate*<T, T*, long, long, T> reduce, delegate*<T*, long, T*, long, long, void> binary)
            {
                Reduce = reduce;
                Binary = binary;
                MaskedReduce = null;
                MaskedBinary = null;
                BinaryRows = null;
            }

            /// <summary>Wraps the unmasked and the whole-call masked forms.</summary>
            /// <param name="reduce">The reduce form.</param>
            /// <param name="binary">The elementwise form.</param>
            /// <param name="maskedReduce">The masked reduce form (see <see cref="MaskedReduce"/> for when it is exact).</param>
            /// <param name="maskedBinary">The masked elementwise form.</param>
            internal Loop(delegate*<T, T*, long, long, T> reduce, delegate*<T*, long, T*, long, long, void> binary,
                delegate*<T, T*, long, byte*, long, long, T> maskedReduce, delegate*<T*, long, T*, long, byte*, long, long, void> maskedBinary)
            {
                Reduce = reduce;
                Binary = binary;
                MaskedReduce = maskedReduce;
                MaskedBinary = maskedBinary;
                BinaryRows = null;
            }

            /// <summary>Wraps the unmasked, the whole-call masked and the several-rows forms.</summary>
            /// <param name="reduce">The reduce form.</param>
            /// <param name="binary">The elementwise form.</param>
            /// <param name="maskedReduce">The masked reduce form (see <see cref="MaskedReduce"/> for when it is exact).</param>
            /// <param name="maskedBinary">The masked elementwise form.</param>
            /// <param name="binaryRows">The several-rows elementwise form (see <see cref="BinaryRows"/>).</param>
            internal Loop(delegate*<T, T*, long, long, T> reduce, delegate*<T*, long, T*, long, long, void> binary,
                delegate*<T, T*, long, byte*, long, long, T> maskedReduce, delegate*<T*, long, T*, long, byte*, long, long, void> maskedBinary,
                delegate*<T*, long, T*, long, long, long, long, void> binaryRows)
            {
                Reduce = reduce;
                Binary = binary;
                MaskedReduce = maskedReduce;
                MaskedBinary = maskedBinary;
                BinaryRows = binaryRows;
            }
        }

        /// <summary>
        ///     Walks the iteration the way NumPy's reduction does — fills, the reduce double loop, the first-visit skip, the
        ///     masked runs — and runs <paramref name="loop"/> on every call.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     Buffered operands are copied in iteration order: the input into a scratch of the loop dtype (cast from its own
        ///     dtype), the mask into a byte scratch, and the out read into a loop-dtype scratch at the fill's start and
        ///     written back (cast to its own dtype) at the fill's end — the round trip that makes a narrowing out lossy per
        ///     fill, as in NumPy. Unbuffered operands are read in place with their array strides.
        ///     </para>
        ///     <para>
        ///     <paramref name="skipFirst"/> is NumPy's <c>skip_first_count</c>: with no initial value the result already holds
        ///     each output's first element (<c>PyArray_CopyInitialReduceValues</c>), so a first-visit call skips it — one input
        ///     element of a reduce call, or the whole of an elementwise call. A mask never meets a skip (NumPy requires an
        ///     initial value with <c>where=</c>).
        ///     </para>
        /// </remarks>
        /// <typeparam name="T">The loop dtype's element type.</typeparam>
        /// <param name="outBase">The out array's logical element 0 (the given out, or the allocated result).</param>
        /// <param name="outType">The out array's dtype (the loop dtype unless the out is cast).</param>
        /// <param name="inBase">The input's logical element 0.</param>
        /// <param name="inType">The input's dtype.</param>
        /// <param name="maskBase">The mask's logical element 0 (bool), or <c>null</c> without a mask.</param>
        /// <param name="loopType">The loop dtype (<typeparamref name="T"/>'s).</param>
        /// <param name="skipFirst">NumPy's skip count: the result's size without an initial value, 0 with one.</param>
        /// <param name="loop">The loop's two forms.</param>
        /// <exception cref="NotSupportedException">A cast pair has no conversion (cannot happen for the 15 dtypes).</exception>
        internal void Run<T>(byte* outBase, NPTypeCode outType, byte* inBase, NPTypeCode inType, byte* maskBase,
            NPTypeCode loopType, long skipFirst, in Loop<T> loop) where T : unmanaged
        {
            if (IterSize == 0)
                return;

            int nd = NDim;
            long C = CoreSize;
            int outSz = outType.SizeOf(), inSz = inType.SizeOf(), tSz = Unsafe.SizeOf<T>();
            bool masked = NOp == 3;

            // Scratch per buffered operand, at most one fill (BufferSize elements); pooled, returned on every exit.
            long cap = Math.Max(BufferSize, 1);
            T* inBuf = null, outBuf = null;
            byte* maskBuf = null;
            long inBytes = cap * tSz, outBytes = cap * tSz, maskBytes = cap;
            try
            {
                if (Buffered[OpIn])
                    inBuf = (T*)SizeBucketedBufferPool.Take(inBytes);
                if (Buffered[OpOut])
                    outBuf = (T*)SizeBucketedBufferPool.Take(outBytes);
                if (masked && Buffered[OpMask])
                    maskBuf = (byte*)SizeBucketedBufferPool.Take(maskBytes);

                var toLoopIn = Buffered[OpIn] ? new Conversion(inType, loopType) : null;
                var toLoopOut = Buffered[OpOut] ? new Conversion(outType, loopType) : null;
                var fromLoopOut = Buffered[OpOut] ? new Conversion(loopType, outType) : null;
                var copyMask = masked && Buffered[OpMask] ? new Conversion(NPTypeCode.Boolean, NPTypeCode.Boolean) : null;

                Span<long> coords = nd <= 32 ? stackalloc long[32] : new long[nd];
                coords = coords[..nd];
                long iterIndex = 0;
                // npyiter_copy_to_buffers' buffer reuse (NPY_OP_ITFLAG_BUF_REUSABLE): in the reduce double loop NumPy
                // passes the previous fill's data pointers, and an out whose pointer did not move keeps its buffer — the
                // running value stays in the loop dtype instead of round-tripping through a cast out's dtype. The buffer is
                // reusable after a fill that copied a whole buffer, or at least a core of a reduce out whose stride along the
                // outer dimension is 0. Outside the reduce double loop NumPy passes no previous pointers: never reused.
                long prevOffOut = long.MinValue;
                bool outReusable = false;
                while (iterIndex < IterSize)
                {
                    // npyiter_goto_iterindex: the fill's multi-index (innermost fastest).
                    long rem = iterIndex;
                    for (int i = 0; i < nd; i++)
                    {
                        coords[i] = rem % Shape[i];
                        rem /= Shape[i];
                    }

                    // npyiter_copy_to_buffers: never past the outer dimension's block, never past the end.
                    long transfer = Math.Min(BufferSize, C * (Shape[OuterDim] - coords[OuterDim]));
                    transfer = Math.Min(transfer, IterSize - iterIndex);
                    long reduceOuterSize = UsingReduce ? transfer / C : 1;
                    long callSize = UsingReduce && reduceOuterSize > 1 ? C : transfer;
                    if (!UsingReduce)
                        reduceOuterSize = 1;

                    long offOut = 0, offIn = 0, offMask = 0;
                    for (int i = 0; i < nd; i++)
                    {
                        offOut += coords[i] * Strides[OpOut][i];
                        offIn += coords[i] * Strides[OpIn][i];
                        if (masked)
                            offMask += coords[i] * Strides[OpMask][i];
                    }

                    // Fill the buffers of the read operands (the out is READWRITE: it is read too).
                    if (inBuf != null)
                        Gather(OpIn, coords, transfer, inBase + offIn * inSz, inSz, (byte*)inBuf, tSz, toLoopIn);
                    if (maskBuf != null)
                        Gather(OpMask, coords, transfer, maskBase + offMask, 1, maskBuf, 1, copyMask);
                    if (outBuf != null)
                    {
                        bool keep = UsingReduce && outReusable && prevOffOut == offOut;
                        if (!keep)
                        {
                            outReusable = transfer == BufferSize
                                          || (transfer >= C && IsReduceOp[OpOut] && Strides[OpOut][OuterDim] == 0);
                            Gather(OpOut, coords, transfer, outBase + offOut * outSz, outSz, (byte*)outBuf, tSz, toLoopOut);
                        }
                    }

                    prevOffOut = offOut;

                    // The rows of the reduce double loop all fold into one out row when the out does not move along the outer
                    // dimension: a loop with a several-rows form takes them in one call once no first visit remains.
                    bool fuseRows = loop.BinaryRows != null && !masked && reduceOuterSize > 1
                                    && InnerStride[OpOut] != 0 && ReduceOuterStride[OpOut] == 0;
                    for (long q = 0; q < reduceOuterSize; q++)
                    {
                        T* o = outBuf != null ? outBuf + q * ReduceOuterStride[OpOut] : (T*)(outBase + (offOut + q * ReduceOuterStride[OpOut]) * outSz);
                        T* x = inBuf != null ? inBuf + q * ReduceOuterStride[OpIn] : (T*)(inBase + (offIn + q * ReduceOuterStride[OpIn]) * inSz);
                        byte* m = !masked ? null : maskBuf != null ? maskBuf + q * ReduceOuterStride[OpMask] : maskBase + offMask + q * ReduceOuterStride[OpMask];
                        long count = callSize;
                        long os = InnerStride[OpOut], xsI = InnerStride[OpIn];
                        if (fuseRows && skipFirst == 0)
                        {
                            loop.BinaryRows(o, os, x, xsI, ReduceOuterStride[OpIn], reduceOuterSize - q, count);
                            break;
                        }

                        if (skipFirst > 0 && IsFirstVisit(coords, q))
                        {
                            // reduce_loop's first-visit skip: the result already holds this output's first element.
                            if (os == 0)
                            {
                                count--;
                                skipFirst--;
                                x += xsI;
                            }
                            else
                            {
                                skipFirst -= count;
                                count = 0;
                            }
                        }

                        if (count > 0)
                        {
                            if (m == null)
                                Call(loop, o, os, x, xsI, count);
                            else
                                CallMasked(loop, o, os, x, xsI, m, InnerStride[OpMask], count);
                        }
                    }

                    // npyiter_copy_from_buffers: the out's buffer goes back, cast to the out's dtype.
                    if (outBuf != null)
                        Scatter(OpOut, coords, transfer, outBase + offOut * outSz, outSz, (byte*)outBuf, tSz, fromLoopOut);

                    iterIndex += transfer;
                }
            }
            finally
            {
                if (inBuf != null)
                    SizeBucketedBufferPool.Return((IntPtr)inBuf, inBytes);
                if (outBuf != null)
                    SizeBucketedBufferPool.Return((IntPtr)outBuf, outBytes);
                if (maskBuf != null)
                    SizeBucketedBufferPool.Return((IntPtr)maskBuf, maskBytes);
            }
        }

        /// <summary>
        ///     <c>NpyIter_IsFirstVisit</c> for the out at a call: no reduction axis (a zero out stride) is past its first index
        ///     at the fill's start, and in reduce mode the core is the fill's first or the out advances per core.
        /// </summary>
        /// <param name="coords">The fill's multi-index.</param>
        /// <param name="q">The core's index inside the fill (NumPy's <c>NBF_REDUCE_POS</c>).</param>
        /// <returns>Whether the out's elements of this call are visited for the first time.</returns>
        private bool IsFirstVisit(ReadOnlySpan<long> coords, long q)
        {
            for (int i = 0; i < NDim; i++)
                if (Strides[OpOut][i] == 0 && coords[i] != 0)
                    return false;
            return !(UsingReduce && q != 0 && ReduceOuterStride[OpOut] == 0);
        }

        /// <summary>
        ///     One unmasked call: the reduce form when the out does not move (NumPy's IS_BINARY_REDUCE), else the elementwise
        ///     form.
        /// </summary>
        /// <typeparam name="T">The loop element type.</typeparam>
        /// <param name="loop">The loop.</param>
        /// <param name="o">The call's first out element.</param>
        /// <param name="os">The out's stride in the call.</param>
        /// <param name="x">The call's first input element.</param>
        /// <param name="xs">The input's stride in the call.</param>
        /// <param name="n">The element count (&gt;= 1).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Call<T>(in Loop<T> loop, T* o, long os, T* x, long xs, long n) where T : unmanaged
        {
            if (os == 0)
                *o = loop.Reduce(*o, x, xs, n);
            else
                loop.Binary(o, os, x, xs, n);
        }

        /// <summary>
        ///     One masked call — <c>generic_masked_strided_loop</c>: the call is cut into runs of selected elements (a nonzero
        ///     mask byte), and every run is its own unmasked call with all pointers advanced to it.
        /// </summary>
        /// <typeparam name="T">The loop element type.</typeparam>
        /// <param name="loop">The loop.</param>
        /// <param name="o">The call's first out element.</param>
        /// <param name="os">The out's stride.</param>
        /// <param name="x">The call's first input element.</param>
        /// <param name="xs">The input's stride.</param>
        /// <param name="m">The call's first mask byte.</param>
        /// <param name="ms">The mask's stride (bytes; 0 for a mask broadcast along the call).</param>
        /// <param name="n">The element count.</param>
        private static void CallMasked<T>(in Loop<T> loop, T* o, long os, T* x, long xs, byte* m, long ms, long n) where T : unmanaged
        {
            if (ms == 0)
            {
                // One mask value for the whole call: one run or none.
                if (*m != 0)
                    Call(loop, o, os, x, xs, n);
                return;
            }

            // A loop whose result cannot depend on the run cuts folds the masked call in one piece (vectorized for the
            // integer and bool loops) instead of one call per run.
            if (os == 0 && loop.MaskedReduce != null)
            {
                *o = loop.MaskedReduce(*o, x, xs, m, ms, n);
                return;
            }

            if (os != 0 && loop.MaskedBinary != null)
            {
                loop.MaskedBinary(o, os, x, xs, m, ms, n);
                return;
            }

            long i = 0;
            if (ms == 1)
            {
                // Contiguous mask: the runs found with the vectorized span scans (npy_memchr's job). A run is ONE loop call
                // however long, so the scans search past int.MaxValue in chunks instead of cutting the call at a chunk end.
                while (i < n)
                {
                    long start = ScanMask(m, i, n, selected: true);
                    if (start == n)
                        return;
                    long end = ScanMask(m, start, n, selected: false);
                    Call(loop, o + start * os, os, x + start * xs, xs, end - start);
                    i = end;
                }

                return;
            }

            while (i < n)
            {
                while (i < n && m[i * ms] == 0)
                    i++;
                long start = i;
                while (i < n && m[i * ms] != 0)
                    i++;
                if (i > start)
                    Call(loop, o + start * os, os, x + start * xs, xs, i - start);
            }
        }

        /// <summary>
        ///     The first position at or after <paramref name="from"/> whose contiguous mask byte is selected (nonzero) or
        ///     unselected (zero), scanned with the vectorized span searches in chunks of at most <see cref="int.MaxValue"/>
        ///     bytes (a span's length limit).
        /// </summary>
        /// <param name="m">The call's first mask byte (stride 1).</param>
        /// <param name="from">The position the search starts at.</param>
        /// <param name="n">The call's element count.</param>
        /// <param name="selected">True to find a selected byte, false to find an unselected one.</param>
        /// <returns>The position found, or <paramref name="n"/> when none remains.</returns>
        private static long ScanMask(byte* m, long from, long n, bool selected)
        {
            for (long i = from; i < n;)
            {
                int len = (int)Math.Min(n - i, int.MaxValue);
                var span = new ReadOnlySpan<byte>(m + i, len);
                int k = selected ? span.IndexOfAnyExcept((byte)0) : span.IndexOf((byte)0);
                if (k >= 0)
                    return i + k;
                i += len;
            }

            return n;
        }

        /// <summary>
        ///     The buffer layout of an operand's fill: which iteration positions the buffer holds — the whole fill (inner 1,
        ///     outer the core), one core repeated (inner 1, outer 0), one element per core (inner 0, outer 1), or one element.
        /// </summary>
        /// <param name="op">The operand.</param>
        /// <param name="transfer">The fill's size.</param>
        /// <param name="count">Receives how many positions the buffer holds.</param>
        /// <param name="step">Receives the iteration-position distance between consecutive buffer elements (1, or the core
        ///     for one element per core; ignored for a single element).</param>
        private void BufferPositions(int op, long transfer, out long count, out long step)
        {
            long inner = InnerStride[op];
            long outer = ReduceOuterStride[op];
            if (inner == 1)
            {
                count = UsingReduce && outer == 0 ? CoreSize : transfer;
                step = 1;
            }
            else if (UsingReduce && outer == 1)
            {
                count = transfer / CoreSize;
                step = CoreSize;
            }
            else
            {
                count = 1;
                step = 1;
            }
        }

        /// <summary>
        ///     Copies an operand's buffered positions of one fill into its buffer, converting element by element
        ///     (<c>npyiter_copy_to_buffers</c>).
        /// </summary>
        /// <param name="op">The operand.</param>
        /// <param name="coords">The fill's multi-index.</param>
        /// <param name="transfer">The fill's size.</param>
        /// <param name="src">The operand's element at the fill start.</param>
        /// <param name="srcSize">The operand's element size in bytes.</param>
        /// <param name="dst">The buffer.</param>
        /// <param name="dstSize">The buffer's element size.</param>
        /// <param name="cast">The element conversion (operand dtype → buffer dtype).</param>
        private void Gather(int op, ReadOnlySpan<long> coords, long transfer, byte* src, int srcSize, byte* dst, int dstSize,
            Conversion cast)
        {
            BufferPositions(op, transfer, out long count, out long step);
            Walk(op, coords, count, step, src, srcSize, dst, dstSize, cast, toBuffer: true);
        }

        /// <summary>
        ///     Writes an operand's buffer back to its positions of one fill, converting element by element
        ///     (<c>npyiter_copy_from_buffers</c>).
        /// </summary>
        /// <param name="op">The operand (the out).</param>
        /// <param name="coords">The fill's multi-index.</param>
        /// <param name="transfer">The fill's size.</param>
        /// <param name="dstArr">The operand's element at the fill start.</param>
        /// <param name="dstSize">The operand's element size.</param>
        /// <param name="buf">The buffer.</param>
        /// <param name="bufSize">The buffer's element size.</param>
        /// <param name="cast">The element conversion (buffer dtype → operand dtype).</param>
        private void Scatter(int op, ReadOnlySpan<long> coords, long transfer, byte* dstArr, int dstSize, byte* buf, int bufSize,
            Conversion cast)
        {
            BufferPositions(op, transfer, out long count, out long step);
            Walk(op, coords, count, step, dstArr, dstSize, buf, bufSize, cast, toBuffer: false);
        }

        /// <summary>
        ///     The copy between an operand's array and its buffer over <paramref name="count"/> iteration positions spaced
        ///     <paramref name="step"/> apart from the fill's start: runs along the innermost axis go through the cast kernel
        ///     at once; the rest advance an odometer.
        /// </summary>
        /// <param name="op">The operand.</param>
        /// <param name="coords">The fill's multi-index.</param>
        /// <param name="count">Positions to copy.</param>
        /// <param name="step">Iteration distance between consecutive positions.</param>
        /// <param name="arr">The operand's element at the fill start.</param>
        /// <param name="arrSize">The operand's element size.</param>
        /// <param name="buf">The buffer.</param>
        /// <param name="bufSize">The buffer's element size.</param>
        /// <param name="cast">The conversion in the copy's direction.</param>
        /// <param name="toBuffer">True for array → buffer, false for buffer → array.</param>
        private void Walk(int op, ReadOnlySpan<long> coords, long count, long step, byte* arr, int arrSize, byte* buf, int bufSize,
            Conversion cast, bool toBuffer)
        {
            int nd = NDim;
            long[] st = Strides[op];

            // One position per core (an out that moves along the outer dimension, one value per row of a row
            // reduction): a fill starts on a core boundary and never crosses the outer dimension's block, so its positions
            // are consecutive indices of the outer dimension — one strided run of the array, one conversion call instead
            // of one per position.
            if (step == CoreSize && step > 1 && count > 1 && coords[OuterDim] + count <= Shape[OuterDim])
            {
                bool coreStart = true;
                for (int i = 0; i < OuterDim; i++)
                    coreStart &= coords[i] == 0;
                if (coreStart)
                {
                    long so = st[OuterDim] * arrSize;
                    if (toBuffer)
                        cast.Run(arr, so, buf, bufSize, count);
                    else
                        cast.Run(buf, bufSize, arr, so, count);
                    return;
                }
            }

            Span<long> c = nd <= 32 ? stackalloc long[32] : new long[nd];
            c = c[..nd];
            coords.CopyTo(c);
            long off = 0;   // relative to the fill start, in array elements
            long b = 0;
            while (b < count)
            {
                // A contiguous stretch of positions along axis 0 (step 1): one cast-kernel call.
                long run = step == 1 ? Math.Min(Shape[0] - c[0], count - b) : 1;
                byte* a = arr + off * arrSize;
                byte* p = buf + b * bufSize;
                if (toBuffer)
                    cast.Run(a, st[0] * arrSize, p, bufSize, run);
                else
                    cast.Run(p, bufSize, a, st[0] * arrSize, run);
                b += run;

                // Advance the odometer by `run * step` positions.
                long adv = run * step;
                for (int i = 0; i < nd && adv > 0; i++)
                {
                    long total = c[i] + adv;
                    long carry = total / Shape[i];
                    long next = total % Shape[i];
                    off += (next - c[i]) * st[i];
                    c[i] = next;
                    adv = carry;
                }
            }
        }

        /// <summary>
        ///     The element conversion of one direction of a buffer copy (an operand's dtype to the loop dtype, or back).
        ///     A run contiguous on both sides takes a plain copy (same dtype) or the house SIMD cast kernel
        ///     (<see cref="DirectILKernelGenerator.TryGetCastKernel"/>); any other run takes the strided IL inner cast
        ///     kernel, and the scalar converter where dynamic code is off. Every route is NumPy's unsafe conversion
        ///     (MSVC-pinned) — the cast tiers gate the SIMD kernels and the scalar ones against the same bytes.
        /// </summary>
        /// <remarks>
        ///     Why: the inner cast kernel converts one element per call to <c>Converts</c>, which made the buffer copies
        ///     the cost of a cast reduction (float32 → float64 over an 8192-element fill: 6.1 µs, NumPy 2.3 µs; float16 →
        ///     float32: 12.1 µs, NumPy 6.2 µs).
        /// </remarks>
        private sealed class Conversion
        {
            /// <summary>The strided conversion (any strides; the fallback).</summary>
            private readonly DirectILKernelGenerator.InnerCastLoop _strided;

            /// <summary>The contiguous SIMD conversion, or <c>null</c> when the pair has none.</summary>
            private readonly DirectILKernelGenerator.CastKernel _contig;

            /// <summary>True when both dtypes are the same: a contiguous run is a byte copy.</summary>
            private readonly bool _same;

            /// <summary>Source element size in bytes.</summary>
            private readonly int _srcSize;

            /// <summary>Destination element size in bytes.</summary>
            private readonly int _dstSize;

            /// <summary>Picks the kernels for one direction.</summary>
            /// <param name="from">Source dtype.</param>
            /// <param name="to">Destination dtype.</param>
            internal Conversion(NPTypeCode from, NPTypeCode to)
            {
                _same = from == to;
                _srcSize = from.SizeOf();
                _dstSize = to.SizeOf();
                _contig = _same ? null : DirectILKernelGenerator.TryGetCastKernel(from, to);
                _strided = DirectILKernelGenerator.TryGetInnerCastKernel(from, to) ?? ((s, ss, d, ds, n) =>
                {
                    byte* sp = (byte*)s, dp = (byte*)d;
                    for (long i = 0; i < n; i++, sp += ss, dp += ds)
                        NDIterCasting.ConvertValue(sp, dp, from, to);
                });
            }

            /// <summary>Converts <paramref name="n"/> elements.</summary>
            /// <param name="src">The first source element.</param>
            /// <param name="srcStrideBytes">The source stride in bytes.</param>
            /// <param name="dst">The first destination element.</param>
            /// <param name="dstStrideBytes">The destination stride in bytes.</param>
            /// <param name="n">The element count.</param>
            internal void Run(byte* src, long srcStrideBytes, byte* dst, long dstStrideBytes, long n)
            {
                if (srcStrideBytes == _srcSize && dstStrideBytes == _dstSize)
                {
                    if (_same)
                    {
                        Buffer.MemoryCopy(src, dst, n * _dstSize, n * _dstSize);
                        return;
                    }

                    if (_contig != null)
                    {
                        _contig(src, dst, n);
                        return;
                    }
                }

                _strided(src, srcStrideBytes, dst, dstStrideBytes, n);
            }
        }
    }
}
