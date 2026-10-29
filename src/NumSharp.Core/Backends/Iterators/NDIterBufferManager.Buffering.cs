using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;

// =============================================================================
// NDIterBufferManager.Buffering.cs — NumPy 2.4.2's buffered iteration, ported
// =============================================================================
//
// A buffered NDIter walks its operands the way NumPy's NpyIter does, statement for statement:
//
//   * npyiter_find_buffering_setup (nditer_constr.c)   -> FindBufferingSetup
//   * npyiter_allocate_buffers (nditer_api.c)           -> AllocateBuffers (NDIterBufferManager.cs)
//   * npyiter_goto_iterindex (nditer_api.c)             -> GotoIterIndex
//   * npyiter_copy_to_buffers (nditer_api.c)            -> CopyToBuffers
//   * npyiter_copy_from_buffers (nditer_api.c)          -> CopyFromBuffers
//   * npyiter_fill_buffercopy_params (nditer_api.c)     -> TransferBuffer
//   * PyArray_Transfer[Masked]NDimToStrided / ...StridedToNDim (lowlevel_strided_loops.c.src) -> TransferWalk
//   * npyiter_buffered_iternext (nditer_templ.c.src)    -> BufferedIternext
//   * npyiter_buffered_reduce_iternext (nditer_templ.c.src) -> BufferedReduceIternext
//
// THE SETUP. The iteration axes (NumPy numbers them from the INNERMOST, idim 0; NumSharp's axis NDim-1 is that axis)
// are cut at an OUTER dimension: the axes inside it form the CORE (CoreSize elements) and every buffer fill covers whole
// cores of one block of the outer dimension, never crossing into the next block of the axes beyond it. The outer
// dimension is chosen by NumPy's cost model — (1 + number of buffered operands) / (elements per fill) — while each
// operand that is not cast stays unbuffered (BUFNEVER) as long as one stride walks it across the core. When the chosen
// outer dimension is where a reduction operand's stride flips between zero and non-zero, the fill runs as NumPy's reduce
// DOUBLE LOOP (NDIterFlags.BUFREDUCE): ReduceOuterSize inner loops of one core, the pointers stepping by
// ReduceOuterStrides between them, so a reduce operand's element repeats along the core (or its core repeats per outer
// step) and its buffer holds only that. Under GROWINNER, an iteration whose operands all stay unbuffered keeps the whole
// outer block as one inner loop (np.nditer(arange(10), ['buffered', 'external_loop', 'grow_inner'], buffersize=3) is ONE
// chunk of 10).
//
// THE POINTERS during a buffer (see NDIterState): DataPtrs are NumPy's user pointers (a buffer, or the array of a
// BUFNEVER operand), ArrayWritebackPtrs are NumPy's NIT_DATAPTRS (the array positions at the buffer's start, where the
// write-back lands), and Coords stay at the buffer's start (a buffered multi_index reads the start's coordinates, as
// NumPy's does).
//
// THE BUFFER REUSE RULE (reduce loop only). The next fill of a reduce iteration compares every buffered operand's array
// position with the previous buffer's: an unchanged position whose last fill was complete (a whole buffer, or at least a
// core of a reduce operand whose outer stride is 0) keeps its buffer instead of reading it again. It is observable: a cast
// reduction operand round-trips through its array dtype at every fill it does NOT reuse. A StringDType buffer (NumPy's
// NPY_ITEM_REFCOUNT dtypes) is never reused.
//
// NUMPY'S OWN EDGES, KEPT. A fill that starts inside a core (after iterindex / iterrange moved the position there) covers
// only the rest of that core; in the reduce loop its outer size is then 0, so the write-back of a buffered operand
// transfers nothing — NumPy 2.4.2 loses that partial core's results, and so does this port. The same holds for the part
// of a core a range end cuts off.
//
// NUMPY'S EDGES NOT KEPT (memory safety). Before a delayed allocation is done, NumPy's iternext may write through a NULL
// buffer (it crashes for a cast); this port moves the position and skips the transfers instead. When a range end leaves
// a reduce fill holding one core and a part, NumPy's inner loop runs past the core (out of the reduce operand's array
// when its inner stride is not 0: a heap corruption); this port runs the core, and the part as the next fill
// (CopyToBuffers). And remove_multi_index keeps NumPy's setup only where it still walks the right elements (everything
// merged into one axis, merges above the outer position, or one core per fill); after any other partial merge NumPy
// walks wrong elements or reads its stale axis data, and this port runs the setup again (RebuildBufferingSetup).
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    public static unsafe partial class NDIterBufferManager
    {
        /// <summary>
        ///     Above this many axes or operands the per-call scratch of the buffering code lives on the managed heap instead of
        ///     the stack (NumSharp supports any number of both; a fixed-size stack block would overflow).
        /// </summary>
        private const int StackScratchLimit = 64;

        // =====================================================================
        // Axis access in NumPy's numbering (idim 0 = the innermost axis)
        // =====================================================================

        /// <summary>
        ///     The extent of NumPy axis <paramref name="idim"/> — NumSharp axis <c>NDim - 1 - idim</c>; 1 for the single axis
        ///     NumPy keeps for a 0-d iteration.
        /// </summary>
        /// <param name="state">The iterator state.</param>
        /// <param name="idim">The axis counted from the innermost (must be below <c>max(NDim, 1)</c>).</param>
        /// <returns>The axis extent.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long ShapeAt(ref NDIterState state, int idim)
            => state.NDim == 0 ? 1 : state.Shape[state.NDim - 1 - idim];

        /// <summary>
        ///     Operand <paramref name="op"/>'s stride along NumPy axis <paramref name="idim"/>, in ELEMENTS of the operand's
        ///     array dtype (NumPy keeps bytes; every comparison the setup makes is between one operand's own strides, where
        ///     the unit cancels); 0 for the single axis of a 0-d iteration.
        /// </summary>
        /// <param name="state">The iterator state.</param>
        /// <param name="idim">The axis counted from the innermost.</param>
        /// <param name="op">The operand.</param>
        /// <returns>The element stride.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long StrideAt(ref NDIterState state, int idim, int op)
            => state.NDim == 0 ? 0 : state.Strides[op * state.StridesNDim + state.NDim - 1 - idim];

        /// <summary>The coordinate along NumPy axis <paramref name="idim"/> (0 for the single axis of a 0-d iteration).</summary>
        /// <param name="state">The iterator state.</param>
        /// <param name="idim">The axis counted from the innermost.</param>
        /// <returns>The coordinate (the buffer start's, while a buffer is held).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long CoordAt(ref NDIterState state, int idim)
            => state.NDim == 0 ? 0 : state.Coords[state.NDim - 1 - idim];

        /// <summary>
        ///     NumPy's <c>outer_axisdata->shape - outer_axisdata->index</c>: how many blocks of the outer dimension remain from
        ///     the current position, which caps a fill (it never crosses into the next block of the axes beyond).
        /// </summary>
        /// <param name="state">The iterator state.</param>
        /// <returns>The remaining extent of the outer axis — or, once <c>remove_multi_index</c> merged that axis away, the
        ///     value NumPy reads from its stale storage (<see cref="NDIterState.StaleOuterRemaining"/>).</returns>
        private static long OuterRemaining(ref NDIterState state)
        {
            if (state.StaleOuterRemaining >= 0)
                return state.StaleOuterRemaining;
            int idim = state.OuterDim;
            return ShapeAt(ref state, idim) - CoordAt(ref state, idim);
        }

        // =====================================================================
        // npyiter_find_buffering_setup
        // =====================================================================

        /// <summary>
        ///     NumPy's <c>npyiter_find_buffering_setup</c>: chooses the outer dimension and core of the buffered iteration by
        ///     NumPy's cost model, decides whether the reduce double loop runs, and for every operand whether it is buffered,
        ///     which strides an inner loop sees (<see cref="NDIterState.BufStrides"/>) and which its reduce outer step takes
        ///     (<see cref="NDIterState.ReduceOuterStrides"/>). Ends with the elements one fill may cover
        ///     (<see cref="NDIterState.BufferSize"/>).
        /// </summary>
        /// <param name="state">The iterator state, its axes final (ordered, flipped, coalesced) and its operands' CAST, CONTIG,
        ///     WRITE and REDUCE flags set.</param>
        /// <param name="maximumSize">The buffer size the caller asked for, in elements (0 or less: NumPy's
        ///     <c>NPY_BUFSIZE</c>, 8192).</param>
        /// <remarks>
        ///     <para>
        ///     Sets <see cref="NDIterFlags.BUFREDUCE"/> when the double loop runs, and on each operand
        ///     <see cref="NDIterOpFlags.BUFNEVER"/> (never buffered), <see cref="NDIterOpFlags.BUF_SINGLESTRIDE"/> (one stride
        ///     walks it across what a fill copies) and <see cref="NDIterOpFlags.BUF_REDUCE"/> (iterated as a reduce operand) —
        ///     NumPy's <c>NPY_OP_ITFLAG_BUFNEVER</c>, <c>BUF_SINGLESTRIDE</c> and <c>REDUCE</c> as the setup leaves them. The
        ///     construction's <see cref="NDIterFlags.REDUCE"/> / <see cref="NDIterOpFlags.REDUCE"/> stay as they were.
        ///     </para>
        ///     <para>
        ///     A buffered (or no-GROWINNER) iteration's fill is the chosen size rounded down to whole cores, at most
        ///     <paramref name="maximumSize"/>; GROWINNER over operands that all stay unbuffered keeps the whole outer block,
        ///     which may exceed it — NumPy's rule, and why <c>grow_inner</c> hands out longer chunks than the buffer.
        ///     </para>
        /// </remarks>
        internal static void FindBufferingSetup(ref NDIterState state, long maximumSize)
        {
            int nop = state.NOp;
            int ndim = state.NDim;
            bool growInner = (state.ItFlags & (uint)NDIterFlags.GROWINNER) != 0;
            if (maximumSize <= 0)
                maximumSize = DefaultBufferSize;
            // Kept for a later rerun of the setup (RebuildBufferingSetup, after a partial remove_multi_index merge).
            state.MaxBufferSize = maximumSize;

            // How many axes (counted from the innermost) one stride walks each operand across, and the axis where a reduce
            // operand's stride flips between zero and non-zero (NumPy's op_single_stride_dims / op_reduce_outer_dim).
            Span<int> singleStrideDims = nop <= StackScratchLimit ? stackalloc int[nop] : new int[nop];
            Span<int> reduceOuterDims = nop <= StackScratchLimit ? stackalloc int[nop] : new int[nop];

            // The cost factor: 1 + the number of buffered operands. A cast operand is buffered whatever the split.
            int cost = 1;
            for (int op = 0; op < nop; op++)
            {
                singleStrideDims[op] = 1;
                reduceOuterDims[op] = 0;
                if ((state.GetOpFlags(op) & NDIterOpFlags.CAST) != 0)
                    cost++;
            }

            int outerReduceDim = 0;
            long size = ShapeAt(ref state, 0);
            int bestDim = 0;
            int bestCost = cost;
            long bestSize = size;
            long bestCore = 1;

            for (int idim = 1; idim < ndim; idim++)
            {
                // A reduction can only be split at its outer reduce axis: nothing beyond it is considered.
                if (outerReduceDim != 0)
                    break;
                // Past the buffer, only GROWINNER without any buffered operand could still grow the inner loop.
                if (size >= maximumSize && (cost > 1 || !growInner))
                    break;

                long prevShape = ShapeAt(ref state, idim - 1);
                for (int op = 0; op < nop; op++)
                {
                    long prevStride = StrideAt(ref state, idim - 1, op);
                    long stride = StrideAt(ref state, idim, op);
                    var flags = state.GetOpFlags(op);

                    if (singleStrideDims[op] == idim)
                    {
                        // The strides still collapse for this operand: one stride reaches across this axis too. NumPy's
                        // `continue` also skips the reduce-axis test below for it.
                        if (prevStride * prevShape == stride)
                        {
                            singleStrideDims[op]++;
                            continue;
                        }

                        // The operand needs a buffer from here on (unless it already counted as a cast).
                        if ((flags & NDIterOpFlags.CAST) == 0)
                            cost++;
                    }

                    // A reduction operand whose stride flips between zero and non-zero makes this the outer reduce axis.
                    if ((flags & NDIterOpFlags.REDUCE) != 0 && (stride == 0 || prevStride == 0))
                    {
                        reduceOuterDims[op] = idim;
                        outerReduceDim = idim;
                    }
                }

                long coreSize = size;
                size *= ShapeAt(ref state, idim);
                if (size == 0)
                    break;   // never a zero core

                // A buffered iteration is limited to the buffer in the comparison (doubles avoid an overflow).
                double bufsize = size;
                if (size > maximumSize && (cost > 1 || !growInner))
                    bufsize = maximumSize;

                if (cost * (double)bestSize <= bestCost * bufsize)
                {
                    bestCost = cost;
                    bestCore = coreSize;
                    bestSize = size;
                    bestDim = idim;
                }
            }

            bool usingReduce = outerReduceDim != 0 && bestDim == outerReduceDim;
            if (usingReduce)
                state.ItFlags |= (uint)NDIterFlags.BUFREDUCE;
            else
                state.ItFlags &= ~(uint)NDIterFlags.BUFREDUCE;

            bool mustBuffer = false;
            for (int op = 0; op < nop; op++)
            {
                var flags = state.GetOpFlags(op) &
                            ~(NDIterOpFlags.BUFNEVER | NDIterOpFlags.BUF_SINGLESTRIDE | NDIterOpFlags.BUF_REDUCE |
                              NDIterOpFlags.BUF_REUSABLE);
                bool buffered = (flags & NDIterOpFlags.CAST) != 0;
                // A requested contiguous operand that is only read must not end up with a zero stride.
                bool avoidZeroStrides = (flags & NDIterOpFlags.CONTIG) != 0 && (flags & NDIterOpFlags.WRITE) == 0;
                long reduceAxisStride = StrideAt(ref state, outerReduceDim, op);

                // Whether the double loop iterates this operand as a reduce operand (NumPy may choose so for an operand that
                // is not reduced at all, when that is the cheaper walk).
                bool isReduceOp;
                if (!usingReduce)
                    isReduceOp = false;
                else if (reduceOuterDims[op] == bestDim)
                    isReduceOp = true;                                   // this operand MUST use the reduce semantics
                else if (singleStrideDims[op] == bestDim && !buffered)
                    isReduceOp = true;                                   // unbuffered: iterate it as a reduce operand anyway
                else if (reduceAxisStride == 0 && singleStrideDims[op] <= bestDim && !avoidZeroStrides)
                    isReduceOp = true;                                   // buffer the core only, repeat it per outer step
                else
                    isReduceOp = false;

                // One stride across everything a fill copies (the reduce loop drops the outer axis), or a buffer.
                if (singleStrideDims[op] + (isReduceOp ? 1 : 0) > bestDim)
                    flags |= NDIterOpFlags.BUF_SINGLESTRIDE;
                else
                    buffered = true;

                long innerStride, reduceOuterStride;
                if (buffered)
                {
                    long itemsize = state.GetElementSize(op);
                    if (isReduceOp)
                    {
                        // A buffered reduce operand holds either the core (outer stride 0: it repeats per outer step) or
                        // the outer column (its element repeats along the core).
                        if (reduceAxisStride == 0)
                        {
                            innerStride = itemsize;
                            reduceOuterStride = 0;
                        }
                        else
                        {
                            innerStride = 0;
                            reduceOuterStride = itemsize;
                        }
                    }
                    else if ((flags & NDIterOpFlags.BUF_SINGLESTRIDE) != 0 && StrideAt(ref state, 0, op) == 0 &&
                             !avoidZeroStrides)
                    {
                        // Always the same element: even the buffer holds just that one.
                        innerStride = 0;
                        reduceOuterStride = 0;
                    }
                    else
                    {
                        innerStride = itemsize;
                        reduceOuterStride = itemsize * bestCore;
                    }
                }
                else
                {
                    // Unbuffered: the operand's own strides, in bytes of its array (no cast, so its iteration dtype too).
                    int arrSize = state.GetSrcElementSize(op);
                    innerStride = StrideAt(ref state, 0, op) * arrSize;
                    reduceOuterStride = reduceAxisStride * arrSize;
                }

                // NumPy invalidates the outer strides when the double loop does not run (NPY_MIN_INTP): nothing reads them.
                if (!usingReduce)
                    reduceOuterStride = long.MinValue;

                state.BufStrides[op] = innerStride;
                state.ReduceOuterStrides[op] = reduceOuterStride;
                if (isReduceOp)
                    flags |= NDIterOpFlags.BUF_REDUCE;
                if (!buffered)
                    flags |= NDIterOpFlags.BUFNEVER;
                else
                    mustBuffer = true;
                state.SetOpFlags(op, flags);
            }

            // A buffering (or non-GROWINNER) iteration keeps every fill within the requested size, in whole cores.
            if (mustBuffer || !growInner)
            {
                if (maximumSize < bestSize)
                    bestSize = bestCore * (maximumSize / bestCore);
            }

            state.BufferSize = bestSize;
            state.CoreSize = bestCore;
            state.OuterDim = bestDim;
            // The core starts at 0; every reposition updates the offset (GotoIterIndex).
            state.CoreOffset = 0;
            state.StaleOuterRemaining = -1;
            // NumPy initializes the reduce position here so a first-visit query before the first fill reads 0.
            state.ReducePos = 0;
            state.ReduceOuterSize = 0;
            state.BufTransferSize = 0;
            state.BufIterEnd = 0;

            // The outer axis's array strides in bytes (NumPy reads NAD_STRIDES(outer_axisdata) at every fill of the loop).
            for (int op = 0; op < nop; op++)
                state.OuterAxisStrides[op] = StrideAt(ref state, bestDim, op) * state.GetSrcElementSize(op);
        }

        // =====================================================================
        // npyiter_goto_iterindex (the buffering half)
        // =====================================================================

        /// <summary>
        ///     NumPy's <c>npyiter_goto_iterindex</c> on a buffered iterator: moves the coordinates and the ARRAY pointers to
        ///     <paramref name="iterindex"/> and records where inside its core that position is
        ///     (<see cref="NDIterState.CoreOffset"/>). The buffers are not touched: a fill (<see cref="CopyToBuffers"/>) follows.
        /// </summary>
        /// <param name="state">The buffered iterator's state.</param>
        /// <param name="iterindex">The target iteration index (not range-checked: the callers check).</param>
        internal static void GotoIterIndex(ref NDIterState state, long iterindex)
        {
            state.GotoIterIndex(iterindex);
            long core = state.CoreSize;
            state.CoreOffset = core > 0 ? iterindex - (iterindex / core) * core : 0;
        }

        // =====================================================================
        // npyiter_copy_to_buffers
        // =====================================================================

        /// <summary>
        ///     NumPy's <c>npyiter_copy_to_buffers</c>: sizes the next fill from the current position (whole cores within the
        ///     outer block, the rest of the core after a reposition inside one, never past the range end), records every
        ///     operand's array position as the write-back target, points the buffered operands at their buffers and reads the
        ///     readable ones into them — skipping a reduce iteration's buffer that still holds its data (the reuse rule).
        /// </summary>
        /// <param name="state">The buffered iterator's state, positioned (its DataPtrs on the ARRAYS) at the fill's start.</param>
        /// <param name="reusePrevious">True for the reduce loop's refill (NumPy passes the previous buffer's start pointers):
        ///     an operand whose array position did not move may keep its buffer. False everywhere else.</param>
        /// <remarks>
        ///     One deliberate difference from NumPy, in the reduce double loop: a fill that holds one core and a part of the
        ///     next (a range end inside the second core) runs as one core, and the part becomes the next fill. NumPy runs
        ///     the whole fill as one inner loop past the core, which adds the part into the core's reduce element or writes
        ///     past the reduce operand's array.
        /// </remarks>
        /// <exception cref="ValueError">A text read into a number does not parse (a string operand's cast).</exception>
        /// <exception cref="OverflowException">A text read into an integer holds a number outside it.</exception>
        /// <exception cref="UnicodeEncodeError">A text read into <c>S</c> is not ASCII.</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> text read into <c>U</c> is not ASCII.</exception>
        /// <exception cref="OutOfMemoryException">A <c>StringDType</c> fill cannot allocate a string.</exception>
        internal static void CopyToBuffers(ref NDIterState state, bool reusePrevious)
        {
            long transfersize = state.BufferSize;
            long coreSize = state.CoreSize;
            bool mayReuse = reusePrevious;

            if (state.CoreOffset != 0)
            {
                // Started inside a core: only its rest, and no buffer can be reused safely.
                mayReuse = false;
                transfersize = coreSize - state.CoreOffset;
            }
            else
            {
                // Never beyond the outer block (for a non-reduction NumPy notes this may not be necessary; it is its rule).
                long remainingOuter = OuterRemaining(ref state);
                if (transfersize > coreSize * remainingOuter)
                    transfersize = coreSize * remainingOuter;
            }

            // Never beyond the range end.
            if (transfersize > state.IterEnd - state.IterIndex)
                transfersize = state.IterEnd - state.IterIndex;

            state.BufTransferSize = transfersize;
            state.BufIterEnd = state.IterIndex + transfersize;
            if (transfersize <= 0)
                return;

            bool reduce = (state.ItFlags & (uint)NDIterFlags.BUFREDUCE) != 0;
            if (reduce)
            {
                // The double loop: one core per inner loop when the fill holds at least one (the inner loop's size then
                // excludes the outer steps; whole cores only, a part left over is the next fill). A fill shorter than a
                // core has outer size 0 (NumPy's integer division).
                //
                // NOT NumPy's `> 1`: NumPy keeps the inner loop at the whole fill when it holds one core and a part (only a
                // range end that is not a core boundary produces that), so its inner loop runs past the core — a reduce
                // operand then adds the next core's elements into this core's element, or, with a non-zero inner stride,
                // walks past the end of its array (NumPy 2.4.2 corrupts its heap there, probed: 0xC0000374). This port's
                // inner loop never spans more than one core, the rule NumPy itself applies to every larger fill.
                state.ReduceOuterSize = transfersize / coreSize;
                if (state.ReduceOuterSize >= 1)
                {
                    state.BufTransferSize = coreSize;
                    state.BufIterEnd = state.IterIndex + coreSize;
                }
                state.ReducePos = 0;
            }

            int nop = state.NOp;
            for (int op = 0; op < nop; op++)
            {
                // The array position NumPy keeps in NIT_DATAPTRS while the buffer is held; the previous buffer's is still in
                // ArrayWritebackPtrs until this line replaces it (the reduce loop's reuse test reads it first).
                long arrayPtr = state.DataPtrs[op];
                long previous = state.ArrayWritebackPtrs[op];
                state.ArrayWritebackPtrs[op] = arrayPtr;
                var flags = state.GetOpFlags(op);

                if ((flags & NDIterOpFlags.BUFNEVER) != 0)
                {
                    // Unbuffered: the user pointer is the array pointer.
                    state.ReduceOuterPtrs[op] = arrayPtr;
                    continue;
                }

                long buffer = state.Buffers[op];
                if (buffer == 0)
                {
                    // Allocation still delayed: NumPy points the user at its NULL buffer and (for a readable operand) copies
                    // into it — a crash or worse. The position moves, the operand stays on its array, and the element views
                    // refuse until the reset that allocates the buffers.
                    state.ReduceOuterPtrs[op] = arrayPtr;
                    continue;
                }

                bool unchangedPtr = mayReuse && previous == arrayPtr;
                state.DataPtrs[op] = buffer;
                state.ReduceOuterPtrs[op] = buffer;

                if ((flags & NDIterOpFlags.READ) == 0)
                    continue;   // a write-only operand's buffer is produced by the loop, never read

                if (unchangedPtr && (flags & NDIterOpFlags.BUF_REUSABLE) != 0)
                    continue;   // the buffer still holds exactly this data

                // A complete fill — a whole buffer, or at least a core of a reduce operand whose core repeats per outer step —
                // is reusable by the next fill at the same position; anything less is not (a fill started inside a core never is).
                if (transfersize == state.BufferSize ||
                    (transfersize >= coreSize && (flags & NDIterOpFlags.BUF_REDUCE) != 0 && state.OuterAxisStrides[op] == 0))
                    flags |= NDIterOpFlags.BUF_REUSABLE;
                else
                    flags &= ~NDIterOpFlags.BUF_REUSABLE;
                state.SetOpFlags(op, flags);

                TransferBuffer(ref state, op, transfersize, toBuffer: true);
            }
        }

        // =====================================================================
        // npyiter_copy_from_buffers
        // =====================================================================

        /// <summary>
        ///     NumPy's <c>npyiter_copy_from_buffers</c>: writes every buffered WRITE operand's buffer back to its array at the
        ///     current buffer's start (masked by the ARRAYMASK operand for a WRITEMASKED one). A no-op when no buffer is held
        ///     (<see cref="NDIterState.BufTransferSize"/> 0: before the first fill, or after the iteration ended).
        /// </summary>
        /// <param name="state">The buffered iterator's state; its coordinates and ArrayWritebackPtrs still at the buffer's start.</param>
        /// <remarks>
        ///     <para>
        ///     In the reduce double loop the fill held <c>BufTransferSize * ReduceOuterSize</c> elements; a reduce operand writes
        ///     back its core or its outer column only. A <c>StringDType</c> buffer loses its reuse mark here (NumPy trashes a
        ///     buffer of a dtype that holds references), so the next fill always reads it again.
        ///     </para>
        ///     <para>
        ///     NumPy writes back the same buffer again if asked twice; the callers never ask twice without a fill in between,
        ///     because reaching the end sets <see cref="NDIterState.BufTransferSize"/> to 0 (<see cref="BufferedIternext"/>,
        ///     <see cref="FinishBufferedLoop"/>), as NumPy's iternext does.
        ///     </para>
        /// </remarks>
        /// <exception cref="ValueError">A number written into a text, or a text into a number, does not convert.</exception>
        /// <exception cref="OverflowException">A text written into an integer holds a number outside it.</exception>
        /// <exception cref="UnicodeEncodeError">A text written into <c>S</c> is not ASCII.</exception>
        /// <exception cref="OutOfMemoryException">A <c>StringDType</c> write-back cannot allocate a string.</exception>
        internal static void CopyFromBuffers(ref NDIterState state)
        {
            long transfersize = state.BufTransferSize;
            if (transfersize == 0)
                return;
            if ((state.ItFlags & (uint)NDIterFlags.BUFREDUCE) != 0)
                transfersize *= state.ReduceOuterSize;

            var vstrings = NDIterVStringContext.Of(ref state);
            int nop = state.NOp;
            for (int op = 0; op < nop; op++)
            {
                var flags = state.GetOpFlags(op);
                if ((flags & NDIterOpFlags.BUFNEVER) != 0)
                    continue;

                // NumPy always trashes a buffer whose dtype holds references (a StringDType cell is a reference into the
                // buffer's allocator): it is never reused.
                if (vstrings != null && vstrings.IsBufferVString(op) && (flags & NDIterOpFlags.BUF_REUSABLE) != 0)
                {
                    flags &= ~NDIterOpFlags.BUF_REUSABLE;
                    state.SetOpFlags(op, flags);
                }

                if (state.Buffers[op] == 0 || (flags & NDIterOpFlags.WRITE) == 0)
                    continue;   // nothing allocated yet, or a read-only operand (nothing to write back)

                TransferBuffer(ref state, op, transfersize, toBuffer: false);
            }
        }

        /// <summary>
        ///     NumPy's <c>npyiter_clear_buffers</c>: drops the held buffers of an iteration that failed, so nothing of
        ///     the failed fill is written back. A <c>StringDType</c> buffer loses its reuse mark (its cells reference
        ///     strings of the failed fill; <see cref="FreeBuffers"/> releases them with the buffer), and the transfer
        ///     size becomes 0, NumPy's "the buffers are empty", so a later <see cref="CopyFromBuffers"/> writes
        ///     nothing.
        /// </summary>
        /// <param name="state">The buffered iterator's state.</param>
        /// <remarks>A no-op when no buffer is held (<see cref="NDIterState.BufTransferSize"/> already 0). The fills
        ///     that completed before the failure were written back when the iteration moved past them, as in
        ///     NumPy.</remarks>
        internal static void ClearBuffers(ref NDIterState state)
        {
            if (state.BufTransferSize == 0)
                return;

            // NumPy trashes a buffer whose dtype holds references: it is never reused (the next fill reads it again).
            var vstrings = NDIterVStringContext.Of(ref state);
            if (vstrings != null)
            {
                int nop = state.NOp;
                for (int op = 0; op < nop; op++)
                {
                    var flags = state.GetOpFlags(op);
                    if ((flags & NDIterOpFlags.BUFNEVER) == 0 && vstrings.IsBufferVString(op)
                        && (flags & NDIterOpFlags.BUF_REUSABLE) != 0)
                        state.SetOpFlags(op, flags & ~NDIterOpFlags.BUF_REUSABLE);
                }
            }

            // Signal that the buffers are empty.
            state.BufTransferSize = 0;
        }

        // =====================================================================
        // npyiter_fill_buffercopy_params + the N-D transfers
        // =====================================================================

        /// <summary>
        ///     One operand's fill or write-back — NumPy's <c>npyiter_fill_buffercopy_params</c> followed by
        ///     <c>PyArray_TransferNDimToStrided</c> (fill), <c>PyArray_TransferStridedToNDim</c> or
        ///     <c>PyArray_TransferMaskedStridedToNDim</c> (write-back): what to copy (the outer column of a reduce operand whose
        ///     element repeats along the core, one core of a reduce operand whose core repeats, a single repeated element, one
        ///     strided run, or an N-D walk from the buffer start's coordinates), then the copy.
        /// </summary>
        /// <param name="state">The buffered iterator's state.</param>
        /// <param name="op">The buffered operand.</param>
        /// <param name="transfersize">The fill's element count (for the write-back of the reduce loop, all its outer steps).</param>
        /// <param name="toBuffer">True for the fill (array to buffer), false for the write-back.</param>
        /// <exception cref="ValueError">A string operand's conversion fails (see <see cref="CopyStringRun"/>).</exception>
        /// <exception cref="OverflowException">A text converted into an integer holds a number outside it.</exception>
        /// <exception cref="OutOfMemoryException">A <c>StringDType</c> transfer cannot allocate a string.</exception>
        private static void TransferBuffer(ref NDIterState state, int op, long transfersize, bool toBuffer)
        {
            var flags = state.GetOpFlags(op);
            byte* arr = (byte*)state.ArrayWritebackPtrs[op];
            byte* buf = (byte*)state.Buffers[op];
            int arrSize = state.GetSrcElementSize(op);
            int bufSize = state.GetElementSize(op);
            int nd = Math.Max(state.NDim, 1);

            // The write-back of a WRITEMASKED operand is masked by the ARRAYMASK operand, read where it lives: its buffer, or
            // its array at the buffer's start when it is unbuffered (it then advances by its own single stride).
            byte* mask = null;
            long maskStride = 0;
            if (!toBuffer && (flags & NDIterOpFlags.WRITEMASKED) != 0 && state.MaskOp >= 0)
            {
                int maskOp = state.MaskOp;
                mask = (state.GetOpFlags(maskOp) & NDIterOpFlags.BUFNEVER) != 0
                    ? (byte*)state.ArrayWritebackPtrs[maskOp]
                    : (byte*)state.Buffers[maskOp];
                maskStride = state.BufStrides[maskOp];
            }

            if ((flags & NDIterOpFlags.BUF_REDUCE) != 0 && state.OuterAxisStrides[op] != 0)
            {
                // A reduce operand whose element repeats along the core: its buffer holds the outer column, one element per
                // outer step (ReduceOuterSize of them — none for a fill shorter than a core, NumPy's integer division).
                long elemStride = state.OuterAxisStrides[op] / arrSize;
                Transfer1D(ref state, op, arr, elemStride, buf, state.ReduceOuterStrides[op] / bufSize,
                    state.ReduceOuterSize, toBuffer, mask, maskStride);
                return;
            }

            long bufElemStride = state.BufStrides[op] / bufSize;
            long count = transfersize;
            int ndimTransfer = nd;
            if ((flags & NDIterOpFlags.BUF_REDUCE) != 0)
            {
                // A reduce operand whose core repeats per outer step: one core, the outer axis omitted.
                ndimTransfer--;
                if (count > state.CoreSize)
                    count = state.CoreSize;
            }

            if ((flags & NDIterOpFlags.BUF_SINGLESTRIDE) != 0)
            {
                long innerStride = StrideAt(ref state, 0, op);
                if (innerStride == 0 &&
                    ((flags & NDIterOpFlags.CONTIG) == 0 || (flags & NDIterOpFlags.WRITE) != 0))
                {
                    // Always the same element (unless a read-only operand asked for contiguous data): one element.
                    count = 1;
                    bufElemStride = 0;
                }

                Transfer1D(ref state, op, arr, innerStride, buf, bufElemStride, count, toBuffer, mask, maskStride);
                return;
            }

            // The general case: an N-D walk over the innermost ndimTransfer axes, starting at the buffer start's coordinates.
            if (ndimTransfer <= StackScratchLimit)
            {
                long* shape = stackalloc long[ndimTransfer];
                long* coords = stackalloc long[ndimTransfer];
                long* strides = stackalloc long[ndimTransfer];
                TransferNDim(ref state, op, ndimTransfer, shape, coords, strides, arr, buf, bufElemStride, count, toBuffer,
                    mask, maskStride);
            }
            else
            {
                // NumSharp has no axis limit: a very high-rank walk keeps its scratch on the (pinned) managed heap.
                var scratch = new long[3 * ndimTransfer];
                fixed (long* s = scratch)
                    TransferNDim(ref state, op, ndimTransfer, s, s + ndimTransfer, s + 2 * ndimTransfer, arr, buf,
                        bufElemStride, count, toBuffer, mask, maskStride);
            }
        }

        /// <summary>
        ///     A single-run transfer (an outer column, a single-stride walk, or one repeated element) — the 1-D form of
        ///     <see cref="TransferWalk"/>.
        /// </summary>
        /// <param name="state">The buffered iterator's state.</param>
        /// <param name="op">The operand.</param>
        /// <param name="arr">The run's first array element.</param>
        /// <param name="arrElemStride">The array stride in elements of the array dtype.</param>
        /// <param name="buf">The buffer's first element.</param>
        /// <param name="bufElemStride">The buffer stride in elements (1, or 0 for a single element).</param>
        /// <param name="count">The element count.</param>
        /// <param name="toBuffer">True for the fill, false for the write-back.</param>
        /// <param name="mask">The write-back's mask, or null.</param>
        /// <param name="maskStride">The mask's byte stride.</param>
        private static void Transfer1D(ref NDIterState state, int op, byte* arr, long arrElemStride, byte* buf,
                                       long bufElemStride, long count, bool toBuffer, byte* mask, long maskStride)
        {
            long shape = count, coord = 0, stride = arrElemStride;
            TransferWalk(ref state, op, 1, &shape, &coord, &stride, arr, buf, bufElemStride, count, toBuffer, mask, maskStride);
        }

        /// <summary>
        ///     The N-D transfer: gathers the walk's extents, the buffer start's coordinates and the operand's element strides
        ///     along the innermost <paramref name="ndimTransfer"/> axes into the scratch, then walks them.
        /// </summary>
        /// <param name="state">The buffered iterator's state.</param>
        /// <param name="op">The operand.</param>
        /// <param name="ndimTransfer">The axes the walk spans (counted from the innermost).</param>
        /// <param name="shape">Scratch for the extents (ndimTransfer entries).</param>
        /// <param name="coords">Scratch for the start coordinates.</param>
        /// <param name="strides">Scratch for the element strides.</param>
        /// <param name="arr">The array position at the buffer's start.</param>
        /// <param name="buf">The buffer's first element.</param>
        /// <param name="bufElemStride">The buffer stride in elements.</param>
        /// <param name="count">The element count.</param>
        /// <param name="toBuffer">True for the fill, false for the write-back.</param>
        /// <param name="mask">The write-back's mask, or null.</param>
        /// <param name="maskStride">The mask's byte stride.</param>
        private static void TransferNDim(ref NDIterState state, int op, int ndimTransfer, long* shape, long* coords,
                                         long* strides, byte* arr, byte* buf, long bufElemStride, long count, bool toBuffer,
                                         byte* mask, long maskStride)
        {
            for (int idim = 0; idim < ndimTransfer; idim++)
            {
                shape[idim] = ShapeAt(ref state, idim);
                coords[idim] = CoordAt(ref state, idim);
                strides[idim] = StrideAt(ref state, idim, op);
            }

            TransferWalk(ref state, op, ndimTransfer, shape, coords, strides, arr, buf, bufElemStride, count, toBuffer,
                mask, maskStride);
        }

        /// <summary>
        ///     NumPy's <c>PyArray_TransferNDimToStrided</c> / <c>PyArray_TransferStridedToNDim</c> /
        ///     <c>PyArray_TransferMaskedStridedToNDim</c>: moves <paramref name="count"/> elements between the array, walked
        ///     over <paramref name="ndimTransfer"/> axes (innermost first) from the given coordinates, and the buffer, walked
        ///     linearly — one inner-axis run at a time, each through the run transfers (the SIMD cast kernels, the string
        ///     transfers, the masked runs). The mask advances with the buffer.
        /// </summary>
        /// <param name="state">The buffered iterator's state.</param>
        /// <param name="op">The operand.</param>
        /// <param name="ndimTransfer">The axes the walk spans (at least 1).</param>
        /// <param name="shape">The extents, innermost first.</param>
        /// <param name="coords">The start coordinates (read only: the walk keeps its own).</param>
        /// <param name="strides">The array's element strides, innermost first.</param>
        /// <param name="arr">The array element at the start coordinates.</param>
        /// <param name="buf">The buffer's first element.</param>
        /// <param name="bufElemStride">The buffer stride in elements (1, or 0 with a count of 1).</param>
        /// <param name="count">The element count.</param>
        /// <param name="toBuffer">True for the fill, false for the write-back.</param>
        /// <param name="mask">The write-back's mask, or null.</param>
        /// <param name="maskStride">The mask's byte stride.</param>
        private static void TransferWalk(ref NDIterState state, int op, int ndimTransfer, long* shape, long* coords,
                                         long* strides, byte* arr, byte* buf, long bufElemStride, long count, bool toBuffer,
                                         byte* mask, long maskStride)
        {
            if (count <= 0)
                return;

            Debug.Assert(bufElemStride == 1 || count == 1, "a buffer is contiguous unless it holds one element");
            int arrSize = state.GetSrcElementSize(op);
            int bufSize = state.GetElementSize(op);

            // The walk's own coordinates of the axes beyond the innermost (the innermost restarts at 0 after the first run).
            Span<long> c = ndimTransfer <= StackScratchLimit ? stackalloc long[ndimTransfer] : new long[ndimTransfer];
            for (int d = 0; d < ndimTransfer; d++)
                c[d] = coords[d];

            long c0 = c[0];
            for (;;)
            {
                long run = Math.Min(count, shape[0] - c0);
                TransferRun(ref state, op, arr, strides[0], buf, bufElemStride, run, toBuffer, mask, maskStride, arrSize, bufSize);
                count -= run;
                if (count <= 0)
                    return;

                buf += run * bufElemStride * bufSize;
                if (mask != null)
                    mask += run * maskStride;

                // Back to the innermost axis's start, then carry into the outer axes (NumPy's odometer).
                arr -= c0 * strides[0] * arrSize;
                c0 = 0;
                int d = 1;
                for (; d < ndimTransfer; d++)
                {
                    arr += strides[d] * arrSize;
                    if (++c[d] < shape[d])
                        break;
                    arr -= strides[d] * shape[d] * arrSize;
                    c[d] = 0;
                }

                // The walk left its axes: nothing more lies in what the fill covers (the transfer sizes never ask for this).
                if (d == ndimTransfer)
                    return;
            }
        }

        /// <summary>
        ///     One run between the array and the buffer: the string transfer for a fixed-width or <c>StringDType</c> side,
        ///     otherwise the numeric run copies with the SIMD cast kernels; a masked write-back writes only the elements whose
        ///     mask byte is set.
        /// </summary>
        /// <param name="state">The buffered iterator's state.</param>
        /// <param name="op">The operand.</param>
        /// <param name="arr">The run's first array element.</param>
        /// <param name="arrElemStride">The array stride in elements.</param>
        /// <param name="buf">The run's first buffer element.</param>
        /// <param name="bufElemStride">The buffer stride in elements (1, or 0 with a count of 1).</param>
        /// <param name="count">The run length.</param>
        /// <param name="toBuffer">True for the fill, false for the write-back.</param>
        /// <param name="mask">The write-back's mask at this run, or null.</param>
        /// <param name="maskStride">The mask's byte stride.</param>
        /// <param name="arrSize">The array element size.</param>
        /// <param name="bufSize">The buffer element size.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void TransferRun(ref NDIterState state, int op, byte* arr, long arrElemStride, byte* buf,
                                        long bufElemStride, long count, bool toBuffer, byte* mask, long maskStride,
                                        int arrSize, int bufSize)
        {
            if (count <= 0)
                return;

            if (state.IsStringOp(op))
            {
                if (toBuffer)
                    CopyStringRun(ref state, op, arr, arrElemStride, buf, bufElemStride, count, toBuffer: true);
                else if (mask != null)
                    CopyStringRunFromBufferMasked(ref state, op, buf, arr, arrElemStride, count, mask, maskStride);
                else
                    CopyStringRun(ref state, op, buf, bufElemStride, arr, arrElemStride, count, toBuffer: false);
                return;
            }

            var arrType = state.GetOpSrcDType(op);
            var bufType = state.GetOpDType(op);
            if (toBuffer)
                CopyRunToBuffer(arr, arrElemStride, arrType, arrSize, buf, bufType, count);
            else if (mask != null)
                CopyRunFromBufferMasked(buf, bufType, bufSize, arr, arrElemStride, arrType, arrSize, count, mask, maskStride);
            else
                CopyRunFromBuffer(buf, bufType, bufSize, arr, arrElemStride, arrType, count);
        }

        // =====================================================================
        // The iteration steps
        // =====================================================================

        /// <summary>
        ///     NumPy's <c>npyiter_buffered_iternext</c>: one element inside the current buffer (or, with an external loop, the
        ///     whole inner loop); at the buffer's end the buffered operands are written back, the position moves to the next
        ///     fill and the buffers are filled there. Returns false at the end of the range, the last buffer written back.
        /// </summary>
        /// <param name="state">The buffered iterator's state (not in the reduce double loop).</param>
        /// <returns>True while the iteration continues.</returns>
        /// <exception cref="ValueError">A string operand's conversion fails during a write-back or fill.</exception>
        /// <exception cref="OverflowException">A text converted into an integer holds a number outside it.</exception>
        /// <exception cref="OutOfMemoryException">A <c>StringDType</c> transfer cannot allocate a string.</exception>
        internal static bool BufferedIternext(ref NDIterState state)
        {
            if ((state.ItFlags & (uint)NDIterFlags.EXLOOP) == 0)
            {
                // Inside the buffer: step every user pointer by its inner stride.
                if (++state.IterIndex < state.BufIterEnd)
                {
                    int nop = state.NOp;
                    for (int op = 0; op < nop; op++)
                        state.DataPtrs[op] += state.BufStrides[op];
                    return true;
                }
            }
            else
            {
                state.IterIndex += state.BufTransferSize;
            }

            CopyFromBuffers(ref state);

            if (state.IterIndex >= state.IterEnd)
            {
                // The end: no buffer is held any more, so a later write-back (Dispose, reset) writes nothing again.
                state.BufTransferSize = 0;
                return false;
            }

            GotoIterIndex(ref state, state.IterIndex);
            CopyToBuffers(ref state, reusePrevious: false);
            return true;
        }

        /// <summary>
        ///     NumPy's <c>npyiter_buffered_reduce_iternext</c>: inside the current core one element (or, with an external
        ///     loop, the whole core), then the reduce double loop's next outer step — every pointer stepping by its reduce
        ///     outer stride from the step's start, nothing copied — and only after the last outer step the write-back and the
        ///     next fill, which may keep a buffer whose array position did not move (the reuse rule).
        /// </summary>
        /// <param name="state">The buffered iterator's state, in the reduce double loop (<see cref="NDIterFlags.BUFREDUCE"/>).</param>
        /// <returns>True while the iteration continues.</returns>
        /// <exception cref="ValueError">A string operand's conversion fails during a write-back or fill.</exception>
        /// <exception cref="OverflowException">A text converted into an integer holds a number outside it.</exception>
        /// <exception cref="OutOfMemoryException">A <c>StringDType</c> transfer cannot allocate a string.</exception>
        internal static bool BufferedReduceIternext(ref NDIterState state)
        {
            int nop = state.NOp;
            if ((state.ItFlags & (uint)NDIterFlags.EXLOOP) == 0)
            {
                if (++state.IterIndex < state.BufIterEnd)
                {
                    for (int op = 0; op < nop; op++)
                        state.DataPtrs[op] += state.BufStrides[op];
                    return true;
                }
            }
            else
            {
                state.IterIndex += state.BufTransferSize;
            }

            if (ReduceOuterStep(ref state))
                return true;

            CopyFromBuffers(ref state);

            if (state.IterIndex >= state.IterEnd)
            {
                state.BufTransferSize = 0;
                return false;
            }

            GotoIterIndex(ref state, state.IterIndex);
            CopyToBuffers(ref state, reusePrevious: true);
            return true;
        }

        /// <summary>
        ///     The reduce double loop's outer increment: when the fill holds another outer step, every user pointer moves to
        ///     its step start plus its reduce outer stride and the next inner loop is one more core.
        /// </summary>
        /// <param name="state">The buffered iterator's state, in the reduce double loop.</param>
        /// <returns>True when an outer step was taken (nothing to write back yet).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool ReduceOuterStep(ref NDIterState state)
        {
            if (++state.ReducePos >= state.ReduceOuterSize)
                return false;

            int nop = state.NOp;
            for (int op = 0; op < nop; op++)
            {
                long ptr = state.ReduceOuterPtrs[op] + state.ReduceOuterStrides[op];
                state.DataPtrs[op] = ptr;
                state.ReduceOuterPtrs[op] = ptr;
            }

            state.BufIterEnd = state.IterIndex + state.BufTransferSize;
            return true;
        }

        /// <summary>
        ///     The kernel drivers' step past the inner loop a kernel just consumed — whatever the external-loop flag says (a
        ///     kernel always takes a whole inner loop): the reduce double loop's next outer step, or the write-back and the
        ///     next fill (with the reuse rule in the reduce loop). The same transitions as NumPy's external-loop iternext.
        /// </summary>
        /// <param name="state">The buffered iterator's state, positioned at the start of the inner loop just consumed (or
        ///     anywhere inside it).</param>
        /// <returns>True while the iteration continues.</returns>
        /// <exception cref="ValueError">A string operand's conversion fails during a write-back or fill.</exception>
        /// <exception cref="OverflowException">A text converted into an integer holds a number outside it.</exception>
        /// <exception cref="OutOfMemoryException">A <c>StringDType</c> transfer cannot allocate a string.</exception>
        internal static bool BufferedNextInnerLoop(ref NDIterState state)
        {
            // Past the inner loop (BufIterEnd marks its end in both loops).
            state.IterIndex = state.BufIterEnd;

            bool reduce = (state.ItFlags & (uint)NDIterFlags.BUFREDUCE) != 0;
            if (reduce && ReduceOuterStep(ref state))
                return true;

            CopyFromBuffers(ref state);

            if (state.IterIndex >= state.IterEnd)
            {
                state.BufTransferSize = 0;
                return false;
            }

            GotoIterIndex(ref state, state.IterIndex);
            CopyToBuffers(ref state, reusePrevious: reduce);
            return true;
        }

        /// <summary>
        ///     Ends a buffered iteration that a kernel consumed in ONE inner loop (the whole range in the first fill): the
        ///     position moves past it, the buffers are written back, and no buffer is held any more — the state NumPy's last
        ///     iternext leaves, so a later Dispose or reset does not write the same buffer again.
        /// </summary>
        /// <param name="state">The buffered iterator's state.</param>
        /// <exception cref="ValueError">A string operand's conversion fails during the write-back.</exception>
        /// <exception cref="OverflowException">A text converted into an integer holds a number outside it.</exception>
        /// <exception cref="OutOfMemoryException">A <c>StringDType</c> write-back cannot allocate a string.</exception>
        internal static void FinishBufferedLoop(ref NDIterState state)
        {
            if (state.BufTransferSize == 0)
                return;

            state.IterIndex = state.BufIterEnd;
            CopyFromBuffers(ref state);
            state.BufTransferSize = 0;
        }

        // =====================================================================
        // remove_multi_index over a buffered iteration
        // =====================================================================

        /// <summary>
        ///     Whether the buffering setup still describes the iteration's axes after <c>remove_multi_index</c> merged some of
        ///     them: the setup's outer position still names an axis, that axis has, for every operand, the stride the setup
        ///     recorded for its outer axis (<see cref="NDIterState.OuterAxisStrides"/>), and the axes below it still multiply
        ///     to the core size. True when the merges happened only ABOVE the outer position (a merged axis keeps its lower
        ///     axis's strides), where NumPy's stale setup stays consistent with the merged axes.
        /// </summary>
        /// <param name="state">The buffered iterator's state, its axes merged.</param>
        /// <returns>True when the setup can be kept; false when NumPy's setup would walk wrong elements.</returns>
        internal static bool SetupStillDescribesTheAxes(ref NDIterState state)
        {
            int outer = state.OuterDim;
            if (outer >= Math.Max(state.NDim, 1))
                return false;

            long core = 1;
            for (int idim = 0; idim < outer; idim++)
                core *= ShapeAt(ref state, idim);
            if (core != state.CoreSize)
                return false;

            for (int op = 0; op < state.NOp; op++)
                if (StrideAt(ref state, outer, op) * state.GetSrcElementSize(op) != state.OuterAxisStrides[op])
                    return false;
            return true;
        }

        /// <summary>
        ///     Whether the buffering setup, kept after a partial <c>remove_multi_index</c> merge, still fills ONE core at a time
        ///     over the merged axes: no reduce double loop, the setup's outer position still names an axis (NumPy reads its
        ///     live extent, at least 1, so a fill of <c>min(BufferSize, core * remaining)</c> is one core), and a fill is one
        ///     core (<see cref="NDIterState.BufferSize"/> equals <see cref="NDIterState.CoreSize"/>). Such fills start at core
        ///     boundaries (after the rest of a core a reposition started in) and never leave a core, which every unbuffered
        ///     operand walks with one stride, so NumPy's kept setup walks the right elements: chunks of the old core.
        /// </summary>
        /// <param name="state">The buffered iterator's state, its axes merged.</param>
        /// <returns>True when the kept setup fills core by core.</returns>
        internal static bool SetupFillsCoreByCore(ref NDIterState state)
            => (state.ItFlags & (uint)NDIterFlags.BUFREDUCE) == 0 &&
               state.OuterDim < Math.Max(state.NDim, 1) &&
               state.BufferSize == state.CoreSize;

        /// <summary>
        ///     Runs the buffering setup again over the current (merged) axes, with fresh buffers, and fills them at the range
        ///     start — what <c>RemoveMultiIndex</c> does when the old setup no longer describes the axes. NOT NumPy: NumPy keeps
        ///     the old setup, whose outer position then names another axis (or a stale slot), so its fills walk wrong elements
        ///     and its write-back of a reduce operand can land outside the operand (NumPy 2.4.2 writes one element past a (2, 4,
        ///     5) output after <c>remove_multi_index</c> on <c>np.nditer([x, y], ['reduce_ok', 'buffered', 'multi_index'],
        ///     op_axes=[None, [0, -1, 1, 2]], op_dtypes=['f8', 'f8'], casting='unsafe')</c>).
        /// </summary>
        /// <param name="state">The buffered iterator's state, its buffers allocated, its fill the one the reset at the start of
        ///     <c>RemoveMultiIndex</c> made.</param>
        /// <exception cref="OutOfMemoryException">The new buffers cannot be allocated.</exception>
        /// <exception cref="NotSupportedException">A WRITEMASKED operand the new setup buffers has a mask that is not bool or
        ///     uint8 (see <see cref="ValidateWriteMaskedTransfers"/>).</exception>
        /// <exception cref="ValueError">A string operand's conversion fails during the fill.</exception>
        /// <exception cref="OverflowException">A text converted into an integer holds a number outside it.</exception>
        /// <remarks>
        ///     The held fill is dropped, not written back: <c>RemoveMultiIndex</c> resets first and no element was visited since,
        ///     so a read operand's buffer holds only what its array holds and a write-only operand's holds nothing of it —
        ///     writing that back would put stale data over the range's first elements, and the old setup's outer strides would
        ///     direct a reduce operand's write-back to the wrong elements. The buffers are freed while the old setup still
        ///     names their size and who has one, then allocated for the new setup.
        /// </remarks>
        internal static void RebuildBufferingSetup(ref NDIterState state)
        {
            state.BufTransferSize = 0;
            FreeBuffers(ref state);
            FindBufferingSetup(ref state, state.MaxBufferSize);
            ValidateWriteMaskedTransfers(ref state);
            if (!AllocateBuffers(ref state, state.BufferSize))
                throw new OutOfMemoryException("Could not allocate the iterator's buffers");
            GotoIterIndex(ref state, state.IterStart);
            CopyToBuffers(ref state, reusePrevious: false);
        }

        // =====================================================================
        // Construction-time pieces
        // =====================================================================

        /// <summary>
        ///     NumPy's <c>NPY_ITER_CONTIG</c> rule (<c>npyiter_allocate_arrays</c>): an operand that asked for contiguous data and
        ///     whose innermost stride is not its iteration element size becomes a CAST operand — it is copied through a buffer
        ///     whatever the cost model says.
        /// </summary>
        /// <param name="state">The iterator state, its axes ordered.</param>
        /// <param name="innermostAxis">The NumSharp axis that is innermost in the current representation (0 for the ascending
        ///     representation the coalescing pass works in, <c>NDim - 1</c> otherwise); ignored for a 0-d iteration, whose
        ///     single NumPy axis has stride 0.</param>
        /// <remarks>
        ///     NumPy checks this before it coalesces the axes, so an innermost axis of extent 1 (stride 0) makes a contiguous
        ///     array a CAST operand too — its own note calls the place of the check "the wrong place in case of axes reorder".
        ///     The caller decides whether buffering is enabled (NumPy refuses otherwise with "Iterator operand required
        ///     buffering, to be contiguous as requested, but buffering is not enabled").
        /// </remarks>
        /// <returns>True when an operand turned into a CAST operand (the caller then refuses an unbuffered iterator).</returns>
        internal static bool ApplyContigRule(ref NDIterState state, int innermostAxis)
        {
            bool any = false;
            for (int op = 0; op < state.NOp; op++)
            {
                if (!ViolatesContig(ref state, op, innermostAxis))
                    continue;

                var flags = state.GetOpFlags(op);
                if ((flags & NDIterOpFlags.CAST) == 0)
                {
                    state.SetOpFlags(op, flags | NDIterOpFlags.CAST);
                    any = true;
                }
            }

            return any;
        }

        /// <summary>
        ///     The test of NumPy's <c>NPY_ITER_CONTIG</c> rule for one operand: it asked for contiguous data (<c>contig</c>) and
        ///     its innermost stride is not its iteration element size, so it must go through a buffer. Both construction paths
        ///     ask it — <see cref="ApplyContigRule"/> for the engine's own iterators, the descriptor path's per-operand check for
        ///     <c>np.nditer</c> — so the two cannot drift apart.
        /// </summary>
        /// <param name="state">The iterator state, its axes ordered (not yet coalesced: NumPy checks before it coalesces).</param>
        /// <param name="op">The operand.</param>
        /// <param name="innermostAxis">The NumSharp axis that is innermost in the current representation; ignored for a 0-d
        ///     iteration, whose single NumPy axis has stride 0.</param>
        /// <returns>True when the operand asked for contiguous data and its innermost stride is not its element size.</returns>
        internal static bool ViolatesContig(ref NDIterState state, int op, int innermostAxis)
        {
            if ((state.GetOpFlags(op) & NDIterOpFlags.CONTIG) == 0)
                return false;

            long strideBytes = state.NDim == 0
                ? 0
                : state.Strides[op * state.StridesNDim + innermostAxis] * state.GetSrcElementSize(op);
            return strideBytes != state.GetElementSize(op);
        }

        /// <summary>
        ///     NumPy's masked write-back transfer check (<c>npyiter_allocate_transfer_functions</c> builds a masked transfer for
        ///     every buffered WRITEMASKED operand at construction, and the masked transfer accepts only bool and uint8 masks).
        /// </summary>
        /// <param name="state">The iterator state after <see cref="FindBufferingSetup"/> (which decided who is buffered).</param>
        /// <exception cref="NotSupportedException">A buffered WRITEMASKED operand's mask (its array or its iteration dtype) is
        ///     not a 1-byte bool / uint8 type: "Only bool and uint8 masks are supported." (NumPy's TypeError text; the
        ///     masked write-back reads the mask byte by byte, so any other width would read garbage).</exception>
        internal static void ValidateWriteMaskedTransfers(ref NDIterState state)
        {
            int maskOp = state.MaskOp;
            if (maskOp < 0)
                return;

            for (int op = 0; op < state.NOp; op++)
            {
                var flags = state.GetOpFlags(op);
                if ((flags & NDIterOpFlags.BUFNEVER) != 0 ||
                    (flags & (NDIterOpFlags.WRITEMASKED | NDIterOpFlags.WRITE)) != (NDIterOpFlags.WRITEMASKED | NDIterOpFlags.WRITE))
                    continue;

                var maskSrcType = state.GetOpSrcDType(maskOp);
                var maskBufType = state.GetOpDType(maskOp);
                if ((maskSrcType != NPTypeCode.Boolean && maskSrcType != NPTypeCode.Byte) ||
                    (maskBufType != NPTypeCode.Boolean && maskBufType != NPTypeCode.Byte))
                    throw new NotSupportedException("Only bool and uint8 masks are supported.");
            }
        }
    }
}
