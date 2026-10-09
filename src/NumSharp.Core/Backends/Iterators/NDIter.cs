using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;
using NumSharp.Utilities;

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// Function to advance iterator to next position.
    /// Returns true if more iterations remain.
    /// </summary>
    public unsafe delegate bool NDIterNextFunc(ref NDIterState state);

    /// <summary>
    /// Function to get multi-index at current position.
    /// </summary>
    public unsafe delegate void NDIterGetMultiIndexFunc(ref NDIterState state, long* outCoords);

    /// <summary>
    /// Inner loop kernel called by iterator.
    /// </summary>
    public unsafe delegate void NDIterInnerLoopFunc(
        void** dataptrs,
        long* strides,
        long count,
        void* auxdata);

    /// <summary>
    /// High-performance multi-operand iterator matching NumPy's nditer API.
    /// </summary>
    public unsafe ref partial struct NDIterRef
    {
        private NDIterState* _state;
        private bool _ownsState;
        // Borrowed: an iterator walks the caller's operands and never owns them (NumPy's nditer
        // contract). The one array it substitutes itself — a COPY_IF_OVERLAP temporary — is written
        // back and released by ResolveWritebacks, not by any owner-of-the-slot rule. (An ALLOCATE
        // output is yielded to the caller through the operands array.)
        [NDBorrowed] private NDArray[]? _operands;
        private NDIterNextFunc? _cachedIterNext;

        /// <summary>
        /// Write-back registrations created by COPY_IF_OVERLAP: when entry i is
        /// non-null, <see cref="_operands"/>[i] is a temporary copy and the
        /// stored array is the user's original operand. <see cref="Dispose"/>
        /// copies the temporary back (NumPy's WRITEBACKIFCOPY resolved at
        /// NDIter_Deallocate). Not duplicated by <see cref="Copy"/> and not
        /// carried by <c>TransferStateOwnership</c> — the write-back belongs to
        /// the constructing NDIterRef. Borrowed: the entries ARE the user's operands.
        /// </summary>
        [NDBorrowed] private NDArray?[]? _writebackOriginals;

        /// <summary>
        ///     The READ-ONLY copy routes of the descriptor construction (<c>npyiter_allocate_arrays</c>' 0-d auto copy and
        ///     a read-only <c>copy</c> operand): when entry i is non-null, <see cref="_operands"/>[i] is a temporary in the
        ///     iteration dtype and the stored array is the user's original. Teardown puts the original back in the slot and
        ///     releases the temporary WITHOUT copying anything back — the temporary was only read, and writing a lossy cast
        ///     back would change the original. Borrowed: the entries ARE the user's operands.
        /// </summary>
        [NDBorrowed] private NDArray?[]? _releaseOnlyOriginals;

        /// <summary>
        ///     During the descriptor construction only: the operands the iterator ALLOCATED, which do not vote in the axis
        ///     ordering. NumPy allocates them after the order is settled (<c>npyiter_allocate_arrays</c>), so while it orders
        ///     the axes their strides are 0 and every comparison skips them; NumSharp allocates them first, so their strides
        ///     are zeroed and the contiguity / broadcast tests skip them until <c>MaterializeDescriptorRoutes</c> lays them
        ///     out in the settled order. Null outside that window (and on the internal, type-code path).
        /// </summary>
        [NDBorrowed] private bool[]? _abstainOps;

        /// <summary>
        ///     During construction only: the operand slots the iterator ALLOCATED (the caller passed null with
        ///     <c>allocate</c> / <c>virtual</c>). The caller receives those arrays only from a SUCCESSFUL construction
        ///     (through the operand array), so a construction that fails after allocating — a reduction check, a casting
        ///     check, the first buffer fill — disposes them and puts the caller's null back
        ///     (<see cref="DiscardConstructionTemps"/>); otherwise each one would hold a pooled buffer until its finalizer
        ///     ran. Null when nothing was allocated, and cleared once the construction succeeded.
        /// </summary>
        [NDBorrowed] private bool[]? _allocatedSlots;

        // =========================================================================
        // Factory Methods
        // =========================================================================

        /// <summary>
        /// Create single-operand iterator.
        /// Equivalent to NumPy's NDIter_New.
        /// </summary>
        public static NDIterRef New(
            NDArray op,
            NDIterGlobalFlags flags = NDIterGlobalFlags.None,
            NPY_ORDER order = NPY_ORDER.NPY_KEEPORDER,
            NPY_CASTING casting = NPY_CASTING.NPY_SAFE_CASTING,
            NPTypeCode? dtype = null)
        {
            var opFlags = new[] { NDIterPerOpFlags.READONLY };
            var dtypes = dtype.HasValue ? new[] { dtype.Value } : null;
            return MultiNew(1, new[] { op }, flags, order, casting, opFlags, dtypes);
        }

        /// <summary>
        /// Create multi-operand iterator.
        /// Equivalent to NumPy's NDIter_MultiNew.
        /// </summary>
        public static NDIterRef MultiNew(
            int nop,
            NDArray[] op,
            NDIterGlobalFlags flags,
            NPY_ORDER order,
            NPY_CASTING casting,
            NDIterPerOpFlags[] opFlags,
            NPTypeCode[]? opDtypes = null)
        {
            return AdvancedNew(nop, op, flags, order, casting, opFlags, opDtypes);
        }

        /// <summary>
        /// Create iterator with full control over all parameters.
        /// Equivalent to NumPy's NDIter_AdvancedNew.
        /// </summary>
        /// <param name="nop">The number of operands.</param>
        /// <param name="op">The operands; a null entry is allocated (ALLOCATE / VIRTUAL).</param>
        /// <param name="flags">The global flags.</param>
        /// <param name="order">The iteration order.</param>
        /// <param name="casting">The casting rule every operand's cast must satisfy.</param>
        /// <param name="opFlags">The per-operand flags.</param>
        /// <param name="opDtypes">The per-operand iteration type codes (<see cref="NPTypeCode.Empty"/> or null = the operand's).</param>
        /// <param name="opAxesNDim">The op_axes rank, or -1 when <paramref name="opAxes"/> is unused.</param>
        /// <param name="opAxes">The per-operand axis maps.</param>
        /// <param name="iterShape">An explicit iteration shape, or null to broadcast the operands.</param>
        /// <param name="bufferSize">The buffer size in elements (0 = <c>np.getbufsize()</c>).</param>
        /// <param name="opDescrs">
        ///     The per-operand iteration DESCRIPTORS — NumPy's resolved <c>op_dtype</c> array (a null entry means the operand's
        ///     own descriptor; for an operand the iterator allocates, the type-code rules of <paramref name="opDtypes"/>
        ///     decide its dtype instead). When given, the iterator runs NumPy's descriptor-level
        ///     construction: a fixed-width string operand is sized by its descriptor (<c>&lt;U5</c> = 20 bytes), an allocated
        ///     output takes its descriptor, and the cast decisions are NumPy's — <c>npyiter_check_casting</c>'s read and
        ///     write-back checks with their <see cref="TypeError"/> texts, then <c>npyiter_allocate_arrays</c>' copy routes (a
        ///     read-only 0-d operand copied into its iteration dtype, a <c>copy</c> / <c>updateifcopy</c> operand copied and
        ///     written back at close) and its "required copying or buffering" <see cref="TypeError"/>. Null (every internal
        ///     caller) keeps the type-code construction and its legacy cast errors.
        /// </param>
        /// <returns>The iterator (owns its state; dispose it).</returns>
        /// <exception cref="TypeError">With <paramref name="opDescrs"/>: an operand's cast violates <paramref name="casting"/>
        ///     (read or write-back direction), or a cast operand is neither buffered nor copied.</exception>
        /// <exception cref="ArgumentException">Inconsistent arguments, a broadcast mismatch, a refused reduction, or (without
        ///     <paramref name="opDescrs"/>) a cast without <see cref="NDIterGlobalFlags.BUFFERED"/>.</exception>
        public static NDIterRef AdvancedNew(
            int nop,
            NDArray[] op,
            NDIterGlobalFlags flags,
            NPY_ORDER order,
            NPY_CASTING casting,
            NDIterPerOpFlags[] opFlags,
            NPTypeCode[]? opDtypes = null,
            int opAxesNDim = -1,
            int[][]? opAxes = null,
            long[]? iterShape = null,
            long bufferSize = 0,
            DType[]? opDescrs = null)
            => AdvancedNew(nop, op, flags, order, casting, opFlags, opDtypes, opAxesNDim, opAxes, iterShape, bufferSize, opDescrs,
                           shapeValidated: false);

        /// <summary>
        ///     <see cref="AdvancedNew(int, NDArray[], NDIterGlobalFlags, NPY_ORDER, NPY_CASTING, NDIterPerOpFlags[],
        ///     NPTypeCode[], int, int[][], long[], long, DType[])"/> for a caller that has ALREADY run NumPy's shape and stride
        ///     passes over these operands (np.nditer's wrapper) and hands in the broadcast shape they produced.
        /// </summary>
        /// <param name="nop">The number of operands.</param>
        /// <param name="op">The operands; a null entry is allocated (ALLOCATE / VIRTUAL).</param>
        /// <param name="flags">The global flags.</param>
        /// <param name="order">The iteration order.</param>
        /// <param name="casting">The casting rule every operand's cast must satisfy.</param>
        /// <param name="opFlags">The per-operand flags.</param>
        /// <param name="opDtypes">The per-operand iteration type codes (<see cref="NPTypeCode.Empty"/> or null = the operand's).</param>
        /// <param name="opAxesNDim">The op_axes rank, or -1 when <paramref name="opAxes"/> is unused.</param>
        /// <param name="opAxes">The per-operand axis maps.</param>
        /// <param name="iterShape">The iteration shape; with <paramref name="shapeValidated"/> the broadcast shape the caller's
        ///     passes produced, which the iterator takes over (the caller must not reuse the array).</param>
        /// <param name="bufferSize">The buffer size in elements (0 = <c>np.getbufsize()</c>).</param>
        /// <param name="opDescrs">The per-operand iteration descriptors (see the public overload).</param>
        /// <param name="shapeValidated">True when the caller ran NumPy's shape pass (which computed
        ///     <paramref name="iterShape"/>) and stride-pass refusals: the engine's own broadcast computation and its
        ///     compatibility / stride-pass checks are then skipped. They would repeat the same work — np.nditer paid for both
        ///     (+30–170 ns a construction) — and can no longer fail.</param>
        /// <returns>The iterator (owns its state; dispose it).</returns>
        /// <exception cref="TypeError">With <paramref name="opDescrs"/>: an operand's cast violates <paramref name="casting"/>,
        ///     or a cast operand is neither buffered nor copied.</exception>
        /// <exception cref="ArgumentException">Inconsistent arguments, a broadcast mismatch or a refused reduction (only when
        ///     <paramref name="shapeValidated"/> is false), or (without <paramref name="opDescrs"/>) a cast without
        ///     <see cref="NDIterGlobalFlags.BUFFERED"/>.</exception>
        internal static NDIterRef AdvancedNew(
            int nop,
            NDArray[] op,
            NDIterGlobalFlags flags,
            NPY_ORDER order,
            NPY_CASTING casting,
            NDIterPerOpFlags[] opFlags,
            NPTypeCode[]? opDtypes,
            int opAxesNDim,
            int[][]? opAxes,
            long[]? iterShape,
            long bufferSize,
            DType[]? opDescrs,
            bool shapeValidated)
        {
            if (nop < 1)
                throw new ArgumentOutOfRangeException(nameof(nop), "At least one operand is required");

            if (op == null || op.Length < nop)
                throw new ArgumentException("Operand array must contain at least nop elements", nameof(op));

            if (opFlags == null || opFlags.Length < nop)
                throw new ArgumentException("OpFlags array must contain at least nop elements", nameof(opFlags));

            // Allocate state on heap for ref struct lifetime — one block carrying the header and
            // the arena the dimension/operand arrays are carved from (recycled per thread).
            var statePtr = AllocateStateBlock();

            // Declared outside the try so a failed construction can still release the copy-route temporaries it made.
            var iter = new NDIterRef
            {
                _state = statePtr,
                _ownsState = true,
                _operands = op,
            };
            try
            {
                iter.Initialize(nop, op, flags, order, casting, opFlags, opDtypes, opAxesNDim, opAxes, iterShape, bufferSize, opDescrs,
                                shapeValidated);
                // The allocated operands now belong to the caller (it reads them from the operand array).
                iter._allocatedSlots = null;
                return iter;
            }
            catch
            {
                // A copy-route or overlap temporary made before the failure (the first buffer fill can still raise) is
                // DISCARDED, never written back — NumPy's NpyIter_Deallocate under a pending error. Then tear down whatever
                // Initialize managed to build (buffers, external dimension arrays) and hand the block back.
                iter.DiscardConstructionTemps();
                FreeState(statePtr);
                throw;
            }
        }

        // =========================================================================
        // State block allocation — single block + per-thread recycling
        // =========================================================================
        //
        // Every heap NDIterState is ONE native allocation: the header followed by
        // StateArenaBytes of arena that NDIterState.AllocateDimArrays carves its dimension
        // and per-operand arrays from (see the arena notes in NDIter.State.cs). Freed blocks
        // are parked in a small per-thread cache and handed to the next construction after
        // ResetForRecycle re-zeroed only what the previous iterator used. Measured on the
        // three-operand 1-D construction: 161 ns → ~90 ns; the three calloc/free pairs it
        // replaces were 78 ns of that (one 616-byte calloc/free is 29 ns). The cache is
        // bounded (StateBlockCacheSlots blocks per thread, ~1.9 KB each) and blocks are
        // uniform, so there is no size-class logic; a block freed on another thread simply
        // joins that thread's cache. States allocated elsewhere (no arena) are freed
        // directly, so ReleaseStateBlock is safe for any pointer FreeState accepts.

        /// <summary>Bytes reserved after the state header for the dimension + per-operand arrays.</summary>
        internal const int StateArenaBytes = 1536;

        /// <summary>Blocks parked per thread by <see cref="ReleaseStateBlock"/>.</summary>
        internal const int StateBlockCacheSlots = 4;

        private sealed class StateBlockCache
        {
            public readonly nint[] Slots = new nint[StateBlockCacheSlots];
            public int Count;
        }

        [ThreadStatic]
        private static StateBlockCache? t_stateBlocks;

        /// <summary>
        /// Obtain a zeroed state header with an attached, zeroed inline arena — from the
        /// per-thread cache when one is parked there, else a fresh single allocation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static NDIterState* AllocateStateBlock()
        {
            var cache = t_stateBlocks;
            if (cache is not null && cache.Count > 0)
                return (NDIterState*)cache.Slots[--cache.Count];

            var block = (NDIterState*)NativeMemory.AllocZeroed((nuint)(sizeof(NDIterState) + StateArenaBytes));
            block->AttachInlineArena((byte*)block + sizeof(NDIterState), StateArenaBytes);
            return block;
        }

        /// <summary>
        /// Release a state header whose buffers and external dimension arrays have already
        /// been freed: single-block states are re-zeroed and parked in the per-thread cache
        /// (or freed when it is full); states without an inline arena are freed directly.
        /// </summary>
        internal static void ReleaseStateBlock(NDIterState* state)
        {
            if (state == null)
                return;

            if (!state->HasInlineArena)
            {
                NativeMemory.Free(state);
                return;
            }

            var cache = t_stateBlocks ??= new StateBlockCache();
            if (cache.Count < StateBlockCacheSlots)
            {
                state->ResetForRecycle();
                cache.Slots[cache.Count++] = (nint)state;
                return;
            }

            NativeMemory.Free(state);
        }

        /// <summary>Blocks currently parked in the calling thread's cache (diagnostics/tests).</summary>
        internal static int CachedStateBlockCount => t_stateBlocks?.Count ?? 0;

        /// <summary>
        ///     The construction body of <see cref="AdvancedNew(int, NDArray[], NDIterGlobalFlags, NPY_ORDER, NPY_CASTING,
        ///     NDIterPerOpFlags[], NPTypeCode[], int, int[][], long[], long, DType[], bool)"/>: the shapes, the operands, the
        ///     casts and copy routes, the axis order, the allocation and the buffers.
        /// </summary>
        /// <param name="nop">The number of operands.</param>
        /// <param name="op">The operands (null entries are allocated).</param>
        /// <param name="flags">The global flags.</param>
        /// <param name="order">The iteration order.</param>
        /// <param name="casting">The casting rule.</param>
        /// <param name="opFlags">The per-operand flags.</param>
        /// <param name="opDtypes">The per-operand iteration type codes, or null.</param>
        /// <param name="opAxesNDim">The op_axes rank, or -1.</param>
        /// <param name="opAxes">The per-operand axis maps, or null.</param>
        /// <param name="iterShape">An explicit iteration shape, or null to broadcast the operands.</param>
        /// <param name="bufferSize">The buffer size in elements (0 = <c>np.getbufsize()</c>).</param>
        /// <param name="opDescrs">The per-operand iteration descriptors, or null for the type-code construction.</param>
        /// <param name="shapeValidated">The caller ran NumPy's shape and stride passes and <paramref name="iterShape"/> is
        ///     their broadcast shape: it is taken over as is and <see cref="ValidateIterShape"/> is skipped.</param>
        /// <exception cref="ValueError">A refused construction (shapes, reductions, flags; NumPy's texts where ported).</exception>
        /// <exception cref="TypeError">A refused cast (descriptor construction).</exception>
        /// <exception cref="IncorrectShapeException">A shape the operands cannot take (house texts).</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void Initialize(
            int nop,
            NDArray[] op,
            NDIterGlobalFlags flags,
            NPY_ORDER order,
            NPY_CASTING casting,
            NDIterPerOpFlags[] opFlags,
            NPTypeCode[]? opDtypes,
            int opAxesNDim,
            int[][]? opAxes,
            long[]? iterShape,
            long bufferSize,
            DType[]? opDescrs = null,
            bool shapeValidated = false)
        {
            _state->MaskOp = -1;
            _state->IterStart = 0;

            // Pre-check per-operand flags BEFORE allocation (nop arg, not state):
            // WRITEMASKED/ARRAYMASK pairing, VIRTUAL/null-operand rules.
            // The actual MaskOp assignment happens after AllocateDimArrays when NOp is set.
            if (opFlags != null)
                PreCheckMaskOpPairing(nop, op, opFlags);

            // Calculate broadcast shape, optionally overridden by iterShape.
            // Dimensions are long (NumPy npy_intp parity) so large axes (> int.MaxValue
            // elements) survive iterator construction without narrowing.
            long[] broadcastShape;
            if (shapeValidated && iterShape != null && iterShape.Length > 0)
            {
                // np.nditer's wrapper ran NumPy's shape pass (which produced this shape, an array of its own) and the stride
                // pass's refusals: the broadcast and the validation below would repeat them and cannot fail.
                broadcastShape = iterShape;
            }
            else if (iterShape != null && iterShape.Length > 0)
            {
                // Use explicit iterShape - allows specifying iteration shape different from broadcast
                // NumPy's NDIter_AdvancedNew() uses this for reductions and custom iteration patterns
                broadcastShape = (long[])iterShape.Clone();
                // Validate that operands are compatible with the specified shape
                // Pass opAxes so validation accounts for -1 entries (broadcast/reduce axes)
                ValidateIterShape(nop, op, opFlags, flags, broadcastShape, opAxesNDim, opAxes);
            }
            else
            {
                broadcastShape = CalculateBroadcastShape(nop, op, opFlags, opAxesNDim, opAxes);
                // Validate NO_BROADCAST operands match without stretching
                ValidateIterShape(nop, op, opFlags, flags, broadcastShape, opAxesNDim, opAxes);
            }

            // COPY_IF_OVERLAP: if any write operand may share memory with any
            // read operand, replace the WRITE operand with a temporary copy and
            // register a write-back to the original on Dispose. This is what
            // makes ufunc-style calls with overlapping in/out well-defined
            // (e.g. add(a[:-1], a[:-1], out=a[1:])). Runs before the ALLOCATE
            // loop so iterator-allocated outputs (null here) are skipped — a
            // fresh allocation can never overlap an input.
            // NumPy: nditer_constr.c:3083-3168 (npyiter_allocate_arrays).
            // With descriptors the forced copy is one of npyiter_allocate_arrays' copy routes instead: decided here, made
            // after the cast check at the operand's ITERATION descriptor and laid out in the iterator's order
            // (MaterializeDescriptorRoutes) — one temporary that is both the overlap copy and the cast.
            bool[]? forceCopy = null;
            if ((flags & NDIterGlobalFlags.COPY_IF_OVERLAP) != 0)
            {
                forceCopy = ComputeCopyIfOverlap(nop, op, opFlags);
                if (forceCopy != null && opDescrs == null)
                    MaterializeForcedCopies(nop, op, opFlags, forceCopy);
            }

            // Allocate null operands that have ALLOCATE or VIRTUAL set.
            // NumPy: npyiter_allocate_arrays in nditer_constr.c allocates a temp
            // array for EVERY null operand — including VIRTUAL ones, whose
            // NPY_OP_ITFLAG_VIRTUAL is consumed nowhere but DebugPrint (the
            // NEP-12 buffer-only semantics never landed). Verified NumPy 2.4.2:
            // a virtual operand gets a real backing array; writes through the
            // iterator land in it; it.operands[i] exposes it.
            // Allocated output has shape = broadcastShape (accounting for op_axes).
            // Dtype: ALLOCATE takes opDtypes[opIdx] (NumSharp requires it);
            // VIRTUAL-only DISCARDS any requested dtype (npyiter_prepare_one_operand
            // c:1037-1039 nulls op_dtype for the non-ALLOCATE branch) and resolves
            // the common dtype of the real operands instead — except an ARRAYMASK
            // operand, which defaults to bool (c:1041-1049). When both flags are
            // set, ALLOCATE wins (c:1009 checks it first) and the request holds.
            // With descriptors (np.nditer), the operands the iterator allocates are remembered: NumPy's casting check and copy
            // routes skip them ("new arrays need no cast"), because the array is created in its iteration dtype. Internal
            // callers pass no descriptors and pay for no tracking array.
            bool[]? allocatedOps = opDescrs != null ? new bool[nop] : null;

            // Two facts NumPy reads off the operands BEFORE it allocates the null ones (npyiter_apply_forced_iteration_order
            // and the any_allocate guard of the stride flip): whether any operand is allocated, and whether every operand the
            // caller passed is F-contiguous (order='A' iterates in Fortran order exactly then).
            bool anyAllocate = false;
            bool givenAllFContiguous = true;
            for (int opIdx = 0; opIdx < nop; opIdx++)
            {
                if (op[opIdx] is null)
                    anyAllocate = true;
                else if (!op[opIdx].Shape.IsFContiguous)
                    givenAllFContiguous = false;
            }

            for (int opIdx = 0; opIdx < nop; opIdx++)
            {
                if (op[opIdx] is null &&
                    (opFlags[opIdx] & (NDIterPerOpFlags.ALLOCATE | NDIterPerOpFlags.VIRTUAL)) != 0)
                {
                    // np.nditer resolves every null operand's descriptor with NumPy's own rules before it gets here (the
                    // request of an ALLOCATE operand, bool for an ARRAYMASK one, else the common dtype of the readable
                    // operands — npyiter_prepare_one_operand + npyiter_get_common_dtype), so that descriptor is the dtype.
                    // A fixed-width string output is allocated from it (the type code names no width), an unsized request
                    // (`'U'`) growing to one character as every string allocation does (PyArray_NewFromDescr).
                    DType? resolvedDescr = opDescrs != null && opIdx < opDescrs.Length ? opDescrs[opIdx] : null;
                    DType? stringAllocDescr = NDStringStorage.IsString(resolvedDescr)
                        ? NDStringStorage.AllocationDescr(resolvedDescr)
                        : null;

                    NPTypeCode allocDtype;
                    bool isArrayMaskOp = (opFlags[opIdx] & NDIterPerOpFlags.ARRAYMASK) != 0;
                    if (resolvedDescr is not null)
                    {
                        allocDtype = resolvedDescr.GetTypeCode();
                    }
                    else if ((opFlags[opIdx] & NDIterPerOpFlags.ALLOCATE) != 0)
                    {
                        bool hasRequest = opDtypes != null && opIdx < opDtypes.Length &&
                                          opDtypes[opIdx] != NPTypeCode.Empty;
                        if (hasRequest)
                            allocDtype = opDtypes[opIdx];
                        else if (isArrayMaskOp)
                            allocDtype = NPTypeCode.Boolean;  // mask default (c:1041-1049)
                        else
                            throw new ArgumentException(
                                $"Operand {opIdx} is null with ALLOCATE flag but opDtypes is not provided", nameof(opDtypes));
                    }
                    else if (isArrayMaskOp)
                    {
                        allocDtype = NPTypeCode.Boolean;  // virtual mask default (c:1041-1049)
                    }
                    else
                    {
                        // VIRTUAL: common dtype over the real operands (the
                        // requested dtype was discarded, NumPy parity).
                        allocDtype = NPTypeCode.Empty;
                        for (int j = 0; j < nop; j++)
                        {
                            if (op[j] is null)
                                continue;
                            allocDtype = allocDtype == NPTypeCode.Empty
                                ? op[j].typecode
                                : NDIterCasting.PromoteTypes(allocDtype, op[j].typecode);
                        }
                        if (allocDtype == NPTypeCode.Empty)
                            throw new ArgumentException(
                                "at least one array or dtype is required");
                    }

                    // Determine output shape: for op_axes, filter out -1 entries
                    long[] outputShape;
                    if (opAxes != null && opIdx < opAxes.Length && opAxes[opIdx] != null)
                    {
                        var axisMap = opAxes[opIdx];
                        // Count non-negative entries
                        int realNDim = 0;
                        for (int i = 0; i < axisMap.Length; i++)
                            if (axisMap[i] >= 0) realNDim++;

                        outputShape = new long[realNDim];
                        int outIdx = 0;
                        for (int iterAxis = 0; iterAxis < axisMap.Length && iterAxis < broadcastShape.Length; iterAxis++)
                        {
                            // Non-negative entries map iterShape axes to output axes
                            if (axisMap[iterAxis] >= 0)
                                outputShape[axisMap[iterAxis]] = broadcastShape[iterAxis];
                            // -1 entries are "reduced" dimensions - not in output shape
                        }
                    }
                    else
                    {
                        // No op_axes: output has full broadcast shape
                        outputShape = (long[])broadcastShape.Clone();
                    }

                    // Allocate the NDArray with the resolved dtype and shape
                    var shape = outputShape.Length == 0 ? new Shape() : NumSharp.Shape.Adopt(outputShape);
                    op[opIdx] = stringAllocDescr is not null ? np.zeros(shape, stringAllocDescr) : np.zeros(shape, allocDtype);
                    if (allocatedOps is not null)
                        allocatedOps[opIdx] = true;
                    // Remembered so that a construction failing after this point releases the array (see _allocatedSlots).
                    (_allocatedSlots ??= new bool[nop])[opIdx] = true;
                }
            }
            // Update _operands so it reflects the allocated arrays
            _operands = op;

            // The descriptor path decides its casts and copy routes now, from descriptors and flags alone: the casting
            // error is raised only after the axis data is filled (NumPy's fill_axisdata errors win), and the copies are
            // made once the iteration order is final (MaterializeDescriptorRoutes, at the reorder/coalesce seam).
            DescriptorPlan? plan = opDescrs != null
                ? PlanDescriptorRoutes(nop, op, opFlags, opDescrs, allocatedOps!, forceCopy, casting)
                : null;

            // =========================================================================
            // NUMSHARP DIVERGENCE: Allocate dimension arrays dynamically
            // Unlike NumPy's fixed NPY_MAXDIMS=64, NumSharp supports unlimited dimensions.
            // Arrays are allocated based on actual ndim for memory efficiency.
            // =========================================================================
            _state->AllocateDimArrays(broadcastShape.Length, nop);

            // Set IDENTPERM on construction. Perm starts as identity (set by AllocateDimArrays);
            // reordering (ReorderAxesForCoalescing) and flipping (FlipNegativeStrides) clear
            // this flag when they mutate perm. Matches NumPy nditer_constr.c:262-264.
            _state->ItFlags |= (uint)NDIterFlags.IDENTPERM;

            // Set MaskOp for ARRAYMASK operand (if any). Requires NOp to be set by
            // AllocateDimArrays above. NumPy nditer_constr.c:1184-1196.
            if (opFlags != null)
                SetMaskOpFromFlags(opFlags);

            _state->IterSize = 1;

            for (int d = 0; d < _state->NDim; d++)
            {
                _state->Shape[d] = broadcastShape[d];
                _state->IterSize *= broadcastShape[d];
            }

            _state->IterEnd = _state->IterSize;

            // Handle zero-size iteration
            if (_state->IterSize == 0 && (flags & NDIterGlobalFlags.ZEROSIZE_OK) == 0)
            {
                // Just allow it anyway for now
            }

            // Determine common dtype if COMMON_DTYPE flag is set
            NPTypeCode? commonDtype = null;
            // (The descriptor path resolved its common dtype from descriptors already - np.nditer applied it.)
            if ((flags & NDIterGlobalFlags.COMMON_DTYPE) != 0 && plan == null)
            {
                commonDtype = NDIterCasting.FindCommonDtype(op, nop);
            }

            // Set up operands
            bool anyNeedsCast = false;
            for (int i = 0; i < nop; i++)
            {
                var arr = op[i];
                var arrShape = arr.Shape;

                // Determine buffer/target dtype. virtual-without-allocate is a
                // single masked equality so the common path pays one AND.
                NPTypeCode bufferDtype;
                bool virtualAllocated =
                    (opFlags[i] & (NDIterPerOpFlags.VIRTUAL | NDIterPerOpFlags.ALLOCATE))
                        == NDIterPerOpFlags.VIRTUAL;
                if (plan != null)
                {
                    // The descriptor path: both sides from DESCRIPTORS, which is what sizes a fixed-width string element
                    // (<U5 is 20 bytes; the String lane names no width) and records its class. A copy-route operand is
                    // re-pointed at its temporary later; until then it describes the original.
                    var iterDescr = plan.IterDescrs[i];
                    _state->SetOpSrcDType(i, arr.dtype);
                    _state->SetOpDType(i, iterDescr);
                    bufferDtype = iterDescr.GetTypeCode();
                }
                else if (virtualAllocated)
                {
                    // VIRTUAL discarded any requested dtype at allocation
                    // (NumPy parity — see the allocation loop above): the
                    // freshly allocated array IS the iteration dtype, no cast.
                    bufferDtype = arr.typecode;
                }
                else if (opDtypes != null && i < opDtypes.Length && opDtypes[i] != NPTypeCode.Empty)
                {
                    bufferDtype = opDtypes[i];
                }
                else if (commonDtype.HasValue)
                {
                    bufferDtype = commonDtype.Value;
                }
                else
                {
                    bufferDtype = arr.typecode;
                }
                if (plan == null)
                {
                    // Store source dtype (actual array dtype) and the buffer dtype resolved above.
                    _state->SetOpSrcDType(i, arr.typecode);
                    _state->SetOpDType(i, bufferDtype);
                }

                // Track if any operand needs casting. With descriptors the plan decided it (two strings of one lane, or
                // two byte orders of one type, still cast); a copy-route operand needs none once its temporary is in.
                bool needsCast = plan != null
                    ? plan.Routes[i] == DescriptorRoute.Cast
                    : arr.typecode != bufferDtype;
                if (needsCast)
                {
                    anyNeedsCast = true;
                }

                // Set operand flags
                var opFlag = TranslateOpFlags(opFlags[i]);

                // If operand needs casting, add CAST flag
                if (needsCast)
                {
                    opFlag |= NDIterOpFlags.CAST;
                }

                // COPY_IF_OVERLAP replaced this operand with a temporary that
                // is written back on Dispose (NumPy: FORCECOPY + WRITEBACKIFCOPY).
                if (_writebackOriginals?[i] is not null)
                {
                    opFlag |= NDIterOpFlags.FORCECOPY | NDIterOpFlags.HAS_WRITEBACK;
                }

                _state->SetOpFlags(i, opFlag);

                // Calculate strides for this operand
                var stridePtr = _state->GetStridesPointer(i);
                byte* basePtr;

                // Check if op_axes is provided for this operand
                if (opAxes != null && i < opAxes.Length && opAxes[i] != null)
                {
                    // Use op_axes mapping to set up strides directly.
                    // NumPy fill_axisdata parity (nditer_constr.c:1618-1679):
                    // the stride is 0 when the iterator dim is 1, when the entry
                    // is a broadcast/reduction axis (-1 or REDUCTION_AXIS-encoded),
                    // or when the operand's mapped axis has length 1 (broadcast
                    // stretch) — using the raw array stride for a stretched size-1
                    // axis walked out of bounds.
                    var opAxisMap = opAxes[i];
                    var arrStrides = arrShape.strides;
                    var arrDims = arrShape.dimensions;

                    // Operand base must point to the view's LOGICAL origin: add
                    // Shape.offset (in elements). The standard-broadcast branch below
                    // already does this (lines ~473/481); the op_axes branch did not,
                    // so any op_axes reduction over a view whose offset lives in
                    // Shape.offset (sliced a[1:3,1:3], negative-stride a[::-1] / a[:,::-1])
                    // read from the buffer base instead — wrong cells, and after
                    // FlipNegativeStrides moved the pointer, out-of-bounds reads/writes
                    // (garbage / NaN). Views with offset==0 (C-/F-contig, transpose,
                    // positive strided) were unaffected, which is why it hid so long.
                    // Use NPTypeCode.SizeOf() (1 byte for bool) NOT arr.dtypesize.
                    // The element size is the one just stored for the operand's array side: the storage lane's, or a
                    // fixed-width string's item size (the String lane names no width).
                    basePtr = (byte*)arr.Address + (arrShape.offset * _state->SrcElementSizes[i]);

                    for (int d = 0; d < _state->NDim; d++)
                    {
                        if (_state->Shape[d] == 1)
                        {
                            // Iterator dim of size 1: never advanced; NumPy forces
                            // stride 0 (the coalesce trivial-branch and merge rule
                            // rely on this invariant).
                            stridePtr[d] = 0;
                        }
                        else if (d < opAxisMap.Length)
                        {
                            int opAxis = NDIterUtils.GetOpAxis(opAxisMap[d], out _);
                            if (opAxis < 0)
                            {
                                // -1 means broadcast/reduce this dimension
                                stridePtr[d] = 0;
                            }
                            else if (opAxis < arrStrides.Length)
                            {
                                // Operand axis of length 1 is broadcast-stretched
                                // to the iter dim: stride 0, not the raw stride.
                                stridePtr[d] = arrDims[opAxis] == 1 ? 0 : arrStrides[opAxis];
                            }
                            else
                            {
                                stridePtr[d] = 0;
                            }
                        }
                        else
                        {
                            stridePtr[d] = 0;
                        }
                    }
                }
                else
                {
                    // Standard broadcasting.
                    //
                    // NOTE: must use NPTypeCode.SizeOf() (1 byte for bool) and
                    // NOT arr.dtypesize, which is implemented via
                    // Marshal.SizeOf and returns 4 for bool because bool is
                    // marshaled as win32 BOOL. In-memory layout uses 1 byte
                    // per bool element, so Marshal-based sizing produces
                    // pointer offsets 4x too large.
                    // The element size stored for the operand's array side (a string's item size included).
                    int elemBytes = _state->SrcElementSizes[i];

                    // Fast path (per-call construction cost): when the operand's
                    // shape already equals the iteration shape, np.broadcast_to is
                    // an identity transform — same offset, same strides, no stride-0
                    // insertion. Detect that and copy the operand's own offset/strides
                    // directly, skipping the broadcast_to call and its Shape allocation.
                    // This is the dominant elementwise case (a OP b, both same shape)
                    // and was ~25% of the iterator's setup overhead.
                    bool sameAsIter = arrShape.NDim == _state->NDim;
                    if (sameAsIter)
                    {
                        var adims = arrShape.dimensions;
                        for (int d = 0; d < _state->NDim; d++)
                            if (adims[d] != _state->Shape[d]) { sameAsIter = false; break; }
                    }

                    // NumPy fill_axisdata parity (nditer_constr.c:1594-1615): any
                    // iterator axis of size 1 gets stride 0 for every operand. The
                    // raw stride of a size-1 axis is arbitrary (transposed/sliced
                    // views carry nonzero values there) and must never survive
                    // into the iterator — CoalesceAxes' trivial-branch and merge
                    // rule both rely on the size-1 ⇒ stride-0 invariant.
                    if (sameAsIter)
                    {
                        basePtr = (byte*)arr.Address + (arrShape.offset * elemBytes);
                        var astrides = arrShape.strides;
                        for (int d = 0; d < _state->NDim; d++)
                            stridePtr[d] = _state->Shape[d] == 1 ? 0 : astrides[d];
                    }
                    else
                    {
                        var broadcastArr = np.broadcast_to(arrShape, NumSharp.Shape.Adopt(broadcastShape));
                        basePtr = (byte*)arr.Address + (broadcastArr.offset * elemBytes);

                        for (int d = 0; d < _state->NDim; d++)
                        {
                            stridePtr[d] = _state->Shape[d] == 1 ? 0 : broadcastArr.strides[d];
                        }
                    }
                }

                _state->SetDataPtr(i, basePtr);
                _state->SetResetDataPtr(i, basePtr);

                // Scan stretched dimensions (stride 0 with iter dim > 1).
                // Read operands: record SourceBroadcast (legacy layout flag).
                // Write operands: a stretched dim means several iterations land
                // on the same element — that is a reduction. NumPy fill_axisdata
                // (nditer_constr.c:1601-1611, npyiter_check_reduce_ok_and_set_flags)
                // refuses it unless REDUCE_OK is set and the operand is readable;
                // NumSharp previously filled stride 0 silently and let the
                // colliding writes race. op_axes-mapped operands get the same
                // treatment in ApplyOpAxes, which owns reduction semantics for
                // mapped axes (including REDUCTION_AXIS-encoded entries).
                bool opAxesMapped = opAxes != null && i < opAxes.Length && opAxes[i] != null;
                for (int d = 0; d < _state->NDim; d++)
                {
                    if (_state->Shape[d] > 1 && stridePtr[d] == 0)
                    {
                        if (!opAxesMapped && (opFlag & NDIterOpFlags.WRITE) != 0)
                        {
                            // NumPy marks a written operand's axis as a REDUCTION only where the OPERAND's own dimension
                            // is 1 or missing under the longer iteration axis (npyiter_fill_axisdata's broadcast branches),
                            // not wherever a stride is 0: a zero-size array has zero strides without being broadcast
                            // (NumPy zeroes them all; NumSharp's C strides are 0 on the axes before the zero-length one),
                            // and NumPy iterates np.zeros((3, 0)) read-write (visiting nothing). A writeable array with an
                            // explicit 0 stride (np.ndarray(..., strides=(0,))) is not refused by NumPy either.
                            if (!IsBroadcastAxis(arrShape, d, _state->NDim))
                                continue;

                            if ((flags & NDIterGlobalFlags.REDUCE_OK) == 0)
                            {
                                // NumPy's ValueError (house ValueError: an ArgumentException). Its loop walks the
                                // iterator axes from the INNERMOST out and numbers them that way, so the axis it reports is
                                // this operand's innermost broadcast axis, counted from the inside: an out of shape
                                // (1, 3, 4) against (2, 3, 4) is "dimension 2", (1, 1, 4) is "dimension 1".
                                int reported = d;
                                for (int e = _state->NDim - 1; e > d; e--)
                                {
                                    if (_state->Shape[e] > 1 && stridePtr[e] == 0 && IsBroadcastAxis(arrShape, e, _state->NDim))
                                    {
                                        reported = e;
                                        break;
                                    }
                                }
                                throw new ValueError(
                                    $"output operand requires a reduction along dimension {_state->NDim - 1 - reported}, " +
                                    "but the reduction is not enabled. The dimension size of 1 " +
                                    "does not match the expected output shape.");
                            }

                            if ((opFlag & NDIterOpFlags.READ) == 0)
                            {
                                // NumPy's ValueError (npyiter_check_reduce_ok_and_set_flags).
                                throw new ValueError(
                                    "output operand requires a reduction, but is flagged as " +
                                    "write-only, not read-write");
                            }

                            // The ARRAYMASK operand may never be the result of a
                            // reduction: a slot could be written while the mask
                            // reads True, then the reduction overwrite the mask
                            // to False — retroactively invalidating the write and
                            // violating the strict masking semantics. NumPy
                            // npyiter_check_reduce_ok_and_set_flags
                            // (nditer_constr.c:1404-1421).
                            if (i == _state->MaskOp)
                            {
                                // NumPy's ValueError, as the two above.
                                throw new ValueError(
                                    "output operand requires a " +
                                    "reduction, but is flagged as " +
                                    "the ARRAYMASK operand which " +
                                    "is not permitted to be the " +
                                    "result of a reduction");
                            }

                            opFlag |= NDIterOpFlags.REDUCE;
                            _state->SetOpFlags(i, opFlag);
                            _state->ItFlags |= (uint)NDIterFlags.REDUCE;
                        }
                        else
                        {
                            _state->ItFlags |= (uint)NDIterFlags.SourceBroadcast;
                        }
                    }
                }
            }

            if (plan == null)
            {
                // Validate that casting requires BUFFERED flag
                if (anyNeedsCast && (flags & NDIterGlobalFlags.BUFFERED) == 0)
                {
                    throw new ArgumentException(
                        "Casting between different dtypes requires the BUFFERED flag. " +
                        "Add NDIterGlobalFlags.BUFFERED to enable type conversion.");
                }

                // Validate casting rules
                NDIterCasting.ValidateCasts(ref *_state, casting);
            }
            else
            {
                // The descriptor path ran NumPy's own checks in the plan (raised below, after the axis data is complete) and
                // raises the "required copying or buffering" error at materialization; only the transfer flags are packed.
                NDIterCasting.PackTransferFlags(ref *_state);
            }

            // Apply op_axes remapping if provided
            if (opAxes != null && opAxesNDim >= 0)
            {
                ApplyOpAxes(opAxesNDim, opAxes, flags);
            }

            // npyiter_check_casting's verdict, now that every fill_axisdata error (broadcast, reduction) has had its turn —
            // and before npyiter_allocate_arrays' tail check below, which NumPy reaches only after the casts pass.
            if (plan?.CastError != null)
                throw plan.CastError;

            // Deferred WRITEMASKED + REDUCE validation: every operand's strides
            // are final here (standard fill AND op_axes both done), so the
            // per-dim "one mask value per WRITEMASKED element" invariant can be
            // checked against the mask's strides regardless of operand order.
            // NumPy runs the identical loop at the tail of
            // npyiter_allocate_arrays (nditer_constr.c:3351-3370).
            if (_state->MaskOp >= 0)
            {
                for (int iop = 0; iop < nop; iop++)
                {
                    if ((_state->GetOpFlags(iop) &
                         (NDIterOpFlags.WRITEMASKED | NDIterOpFlags.REDUCE)) ==
                        (NDIterOpFlags.WRITEMASKED | NDIterOpFlags.REDUCE))
                    {
                        CheckMaskForWriteMaskedReduction(iop);
                    }
                }
            }

            // An allocated operand does not vote in the axis ordering (NumPy has no array for it yet — its strides are 0 and
            // npyiter_find_best_axis_ordering skips zero strides): its strides are zeroed here, the contiguity and broadcast
            // tests below skip it, and MaterializeDescriptorRoutes lays it out in the settled order. Without this, a K-order
            // iteration over an F-contiguous input with an allocated output followed the output's C layout instead of the
            // input's memory order. Descriptor path only (np.nditer, np.nested_iters).
            if (plan != null && anyAllocate)
            {
                for (int iop = 0; iop < nop; iop++)
                {
                    if (!allocatedOps![iop])
                        continue;
                    for (int d = 0; d < _state->NDim; d++)
                        _state->SetStride(d, iop, 0);
                }

                _abstainOps = allocatedOps;
            }

            // Apply axis reordering based on iteration order.
            // NumPy reorders axes based on the order parameter, then coalesces if MULTI_INDEX is not set.
            //
            // Order semantics:
            // - C-order: last axis innermost (row-major logical iteration)
            // - F-order: first axis innermost (column-major logical iteration)
            // - K-order: smallest stride innermost (memory-order iteration)
            //
            // When MULTI_INDEX is set:
            // - Axes are reordered for the specified iteration order
            // - No coalescing (would invalidate multi-index tracking)
            // - GetMultiIndex/GotoMultiIndex use Perm to map between internal and original coords
            //
            // When MULTI_INDEX is NOT set:
            // - Axes are reordered AND coalesced for maximum efficiency
            bool hasMultiIndex = (flags & NDIterGlobalFlags.MULTI_INDEX) != 0;
            // HASINDEX (C_INDEX or F_INDEX): need original axis structure preserved
            // to compute the flat index correctly. Coalescing loses this info.
            bool hasFlatIndex = (flags & (NDIterGlobalFlags.C_INDEX | NDIterGlobalFlags.F_INDEX)) != 0;

            // Step 0: Flip negative strides for memory-order iteration
            // NumPy's npyiter_flip_negative_strides() (nditer_constr.c:297-307):
            //   if (!(itflags & NPY_ITFLAG_FORCEDORDER)) {
            //       if (!any_allocate && !(flags & NPY_ITER_DONT_NEGATE_STRIDES)) {
            //           npyiter_flip_negative_strides(iter);
            //       }
            //   }
            //
            // Only K-order does NOT set FORCEDORDER. C, F, and A orders all set FORCEDORDER
            // (see npyiter_apply_forced_iteration_order in nditer_constr.c:2490).
            // So negative strides should only be flipped for K-order.
            //
            // User-visible behavior:
            // - K-order on reversed array: iterate in memory order (faster)
            // - C/F/A order on reversed array: iterate in logical order (user asked for it)
            // NumPy's npyiter_apply_forced_iteration_order: 'A' is a FORCED order — Fortran order when every operand the
            // caller passed is F-contiguous and there is more than one axis, C order otherwise. It never takes K's stride
            // sort (a transposed 3-D view walks in C order under 'A', its memory order only under 'K').
            if (order == NPY_ORDER.NPY_ANYORDER)
                order = _state->NDim > 1 && givenAllFContiguous ? NPY_ORDER.NPY_FORTRANORDER : NPY_ORDER.NPY_CORDER;

            bool isForcedOrder = order == NPY_ORDER.NPY_CORDER
                              || order == NPY_ORDER.NPY_FORTRANORDER
                              || order == NPY_ORDER.NPY_ANYORDER;
            // NumPy: "If there's an output being allocated, we must not negate any strides" — an allocated operand is laid
            // out by the iteration order, so a K-order iteration over a reversed input with an allocated output walks the
            // input in its LOGICAL order.
            if (!isForcedOrder && !anyAllocate && (flags & NDIterGlobalFlags.DONT_NEGATE_STRIDES) == 0)
            {
                NDIterCoalescing.FlipNegativeStrides(ref *_state);
            }

            if (_state->NDim > 1)
            {
                // NumPy's coalescing strategy depends on the order parameter:
                //
                // Key insight: Coalescing produces MEMORY-order iteration. This is correct for:
                // - K-order: Memory order is exactly what we want
                // - C-order on C-contiguous: Memory order == C-order
                // - F-order on F-contiguous: Memory order == F-order
                //
                // But for F-order on C-contiguous (or C-order on F-contiguous), coalescing
                // would produce the WRONG iteration order, so we must not coalesce.
                //
                // NumPy's behavior:
                // - K-order on contiguous: Sort by stride, coalesce → memory order
                // - K-order on non-contiguous: Fall back to C-order (no stride sorting)
                // - C-order on C-contiguous: Sort by stride, coalesce → memory order (== C-order)
                // - C-order on non-C-contiguous: Keep C-order, no coalescing
                // - F-order on F-contiguous: Sort by stride, coalesce → memory order (== F-order)
                // - F-order on C-contiguous: NO coalescing, reverse axes, iterate F-order

                // Check contiguity once, use for all order decisions.
                // allowFlip=true (absolute strides) only when FlipNegativeStrides will run,
                // which is only for K-order (non-forced order).
                // For C/F/A forced orders, negative strides are not contiguous since we
                // preserve logical iteration order instead of memory order.
                bool allowFlip = !isForcedOrder && (flags & NDIterGlobalFlags.DONT_NEGATE_STRIDES) == 0;
                bool isCContiguous = CheckAllOperandsContiguous(true, allowFlip);
                bool isFContiguous = CheckAllOperandsContiguous(false, allowFlip);
                bool hasBroadcast = HasBroadcastStrides();

                // For coalescing to work correctly:
                // 1. All operands must be contiguous (either C or F order)
                //    - This includes reversed arrays (negative strides become positive after flip)
                // 2. No broadcast dimensions (stride=0) - breaks stride-based sorting
                bool isContiguous = (isCContiguous || isFContiguous) && !hasBroadcast;

                // Determine effective order for non-contiguous arrays.
                //
                // NumPy K-order reorders axes by |stride| to match memory traversal even for
                // non-contiguous views (e.g., transposed arrays). The only case where the
                // stride-based sort produces wrong results is with BROADCAST axes (stride=0),
                // because stride=0 breaks the ordering signal — we can't tell which broadcast
                // axis should be innermost.
                //
                // So: fall back to C-order only when broadcast is present. For merely
                // non-contiguous (transposed, strided views, negative strides), K-order does
                // a proper descending-stride sort to match NumPy memory-order iteration.
                NPY_ORDER effectiveOrder = order;
                // The descriptor path (np.nditer / np.nested_iters) orders K-order axes with NumPy's own sort
                // (npyiter_find_best_axis_ordering), which handles zero strides itself — a stride-0 axis is ambiguous for
                // that operand and C order wins a conflict — so it needs no broadcast fallback: a reduction output or a
                // broadcast operand next to an F-ordered input walks the input's memory order there, as in NumPy. The
                // internal callers keep the min-stride sort and its fallback.
                bool numpyOrdering = plan != null;
                if ((order == NPY_ORDER.NPY_KEEPORDER || order == NPY_ORDER.NPY_ANYORDER) && hasBroadcast && !numpyOrdering)
                {
                    effectiveOrder = NPY_ORDER.NPY_CORDER;
                }

                if (!hasMultiIndex && !hasFlatIndex)
                {
                    // Coalescing is possible when:
                    // - Arrays are contiguous in the REQUESTED order
                    // - No broadcast dimensions that would break stride-based sorting
                    // - No index tracking (C_INDEX/F_INDEX need original axis structure)
                    // Example: F-order on C-contiguous array should NOT coalesce
                    //          (coalescing produces memory-order which is C-order, wrong for F-order)
                    bool canCoalesce;

                    if (order == NPY_ORDER.NPY_KEEPORDER || order == NPY_ORDER.NPY_ANYORDER)
                    {
                        // K-order: coalesce if contiguous in either C or F order
                        canCoalesce = isContiguous;
                    }
                    else if (order == NPY_ORDER.NPY_CORDER)
                    {
                        // C-order: coalesce only if C-contiguous (no broadcast)
                        canCoalesce = isCContiguous && !hasBroadcast;
                    }
                    else // NPY_FORTRANORDER
                    {
                        // F-order: coalesce only if F-contiguous (no broadcast)
                        canCoalesce = isFContiguous && !hasBroadcast;
                    }

                    if (canCoalesce)
                    {
                        // Sort axes by stride, then coalesce
                        NDIterCoalescing.ReorderAxesForCoalescing(ref *_state, NPY_ORDER.NPY_KEEPORDER, forCoalescing: true,
                                                                  numpyOrdering: numpyOrdering);
                        // The copy routes run at NumPy's npyiter_allocate_arrays seam: the order is final, the axes not
                        // yet merged (this ascending representation has axis 0 innermost).
                        if (plan != null)
                            MaterializeDescriptorRoutes(plan, opFlags, flags, opAxes, innermostFirst: true);

                        if (plan != null && anyAllocate)
                        {
                            // An allocated operand did not vote in the contiguity test that chose this branch (it had no
                            // layout yet), and laid out now it can still stop the full collapse CoalesceAxes assumes: a
                            // reduction output has stride 0 along its reduced axes (op_axes -1). CoalesceAxes cannot
                            // leave such an iterator partly merged (its ascending layout would then walk the outer axis
                            // innermost), so the axes are turned into the iteration order and merged there, as NumPy's
                            // npyiter_coalesce_axes merges whatever its operands allow. It still collapses to one axis
                            // whenever every operand is contiguous.
                            NDIterCoalescing.ReverseAxisOrder(ref *_state);
                            NDIterCoalescing.CoalesceAxesIterationOrder(ref *_state);
                        }
                        else
                        {
                            NDIterCoalescing.CoalesceAxes(ref *_state);
                        }
                    }
                    else
                    {
                        // Can't coalesce - reorder for the requested iteration order
                        NDIterCoalescing.ReorderAxesForCoalescing(ref *_state, effectiveOrder, forCoalescing: false,
                                                                  numpyOrdering: numpyOrdering);
                        if (plan != null)
                            MaterializeDescriptorRoutes(plan, opFlags, flags, opAxes, innermostFirst: false);

                        // NumPy coalesces UNCONDITIONALLY after order resolution
                        // (npyiter_coalesce_axes has no contiguity guard): every
                        // size-1 axis is absorbed via the strict trivial branch AND
                        // every adjacent pair that all operands walk contiguously
                        // is merged — a C-contiguous (R, w) operand next to a 0-d
                        // scalar (strides 0, 0) collapses to ONE 1-D chunk, a
                        // trailing-narrow x[:, :, :w] to (R, w), a uniformly strided
                        // a[:, ::2] to one strided 1-D chunk. CoalesceAxes cannot
                        // do this here (it assumes the ascending pre-coalesce
                        // layout its own sort produces), so this branch used to
                        // only strip the size-1 axes and left every such case
                        // paying one kernel call per row. CoalesceAxesIterationOrder
                        // is the iteration-order merge (innermost axis last) and
                        // subsumes RemoveUnitAxes. Safe on this branch only: no
                        // multi-index/flat-index is tracked.
                        NDIterCoalescing.CoalesceAxesIterationOrder(ref *_state);
                    }
                }
                else
                {
                    // With MULTI_INDEX or HASINDEX (C_INDEX/F_INDEX), just reorder axes
                    // without coalescing. Use effectiveOrder which applies K-order → C-order
                    // fallback for non-contiguous arrays.
                    NDIterCoalescing.ReorderAxesForCoalescing(ref *_state, effectiveOrder, forCoalescing: false,
                                                              numpyOrdering: numpyOrdering);
                    if (plan != null)
                        MaterializeDescriptorRoutes(plan, opFlags, flags, opAxes, innermostFirst: false);
                }
            }
            else if (plan != null)
            {
                // 0-d and 1-d iterations have no axis order to settle: the copy routes run right after the flip.
                MaterializeDescriptorRoutes(plan, opFlags, flags, opAxes, innermostFirst: false);
            }

            // Set external loop flag separately (after coalescing)
            if ((flags & NDIterGlobalFlags.EXTERNAL_LOOP) != 0)
            {
                _state->ItFlags |= (uint)NDIterFlags.EXLOOP;
            }

            // Set GROWINNER flag to maximize inner loop size during buffering
            if ((flags & NDIterGlobalFlags.GROWINNER) != 0)
            {
                _state->ItFlags |= (uint)NDIterFlags.GROWINNER;
            }

            // NumPy's npyiter_check_global_flags turns NPY_ITER_RANGED into NPY_ITFLAG_RANGE at construction: it is what
            // lets the iterator's range be reset later (NDIterator.iterrange refuses an iterator built without it, as
            // NpyIter_ResetToIterIndexRange does). The whole range [0, IterSize) is the initial range either way.
            if ((flags & NDIterGlobalFlags.RANGED) != 0)
            {
                _state->ItFlags |= (uint)NDIterFlags.RANGE;
            }

            // Track multi-index if requested
            if ((flags & NDIterGlobalFlags.MULTI_INDEX) != 0)
            {
                _state->ItFlags |= (uint)NDIterFlags.HASMULTIINDEX;
            }

            // Track flat index if requested (C_INDEX or F_INDEX)
            if ((flags & NDIterGlobalFlags.C_INDEX) != 0)
            {
                _state->ItFlags |= (uint)NDIterFlags.HASINDEX;
                _state->IsCIndex = true;
            }
            else if ((flags & NDIterGlobalFlags.F_INDEX) != 0)
            {
                _state->ItFlags |= (uint)NDIterFlags.HASINDEX;
                _state->IsCIndex = false;
            }

            // A tracked flat index rides its own per-axis strides (NumPy's index "operand"), laid out over the axes as the
            // ordering and the flips left them. NumPy then coalesces the iterator with the index as one more operand
            // whenever no multi-index is tracked, so an index-tracking iterator merges the axes the index steps across as
            // one run (np.nditer(c_array, flags=["c_index"]) is one axis of 12; an f_index over the same array keeps its
            // 2 axes). The coalescing passes above skipped this iterator because the strides did not exist yet.
            if ((_state->ItFlags & (uint)NDIterFlags.HASINDEX) != 0)
            {
                _state->ComputeIndexStrides();
                if ((_state->ItFlags & (uint)NDIterFlags.HASMULTIINDEX) == 0 && _state->NDim > 1)
                    NDIterCoalescing.CoalesceAxesIterationOrder(ref *_state);
            }

            // Compute initial FlatIndex based on current coordinates (handles NEGPERM)
            // Must be called after HASINDEX is set and negative strides are flipped
            _state->InitializeFlatIndex();

            // Update inner strides cache
            // Note: CoalesceAxes calls this internally, but we need to ensure it's
            // called even when coalescing is skipped (NDim <= 1 or MULTI_INDEX set)
            if (_state->NDim <= 1 || (flags & NDIterGlobalFlags.MULTI_INDEX) != 0)
            {
                _state->UpdateInnerStrides();
            }

            // Update contiguity flags
            UpdateContiguityFlags();

            // Set up buffering if requested
            if ((flags & NDIterGlobalFlags.BUFFERED) != 0)
            {
                _state->ItFlags |= (uint)NDIterFlags.BUFFER;
                // An explicit buffersize wins; otherwise use the per-thread ufunc default
                // (np.getbufsize()/np.setbufsize()), which is 8192 unless the caller changed it.
                _state->BufferSize = bufferSize > 0 ? bufferSize : NDIterBufferManager.CurrentBufferSize;

                bool isReduce = (_state->ItFlags & (uint)NDIterFlags.REDUCE) != 0;

                if (!isReduce && (flags & NDIterGlobalFlags.DELAY_BUFALLOC) != 0)
                {
                    // DELAY_BUFALLOC: defer allocation + first fill until first
                    // use. NumPy demands an explicit Reset before iterating
                    // (ValueError otherwise); NumSharp auto-ensures at the
                    // execution entry points (EnsureBuffersReady) instead, so
                    // construction stays allocation-free for the common
                    // construct-then-execute pattern. Ignored for buffered
                    // REDUCE, whose double-loop setup is construction-bound.
                    _state->ItFlags |= (uint)NDIterFlags.DELAYBUF;
                }
                else if (!isReduce)
                {
                    // Windowed buffered iteration (non-reduce): allocate and
                    // prime the first window. FillBufferWindow swaps buffered
                    // operands' DataPtrs to their buffers and records the
                    // array write-back positions.
                    NDIterBufferManager.AllocateBuffers(ref *_state, _state->BufferSize);
                    NDIterBufferManager.FillBufferWindow(ref *_state,
                        NDIterBufferManager.ComputeTransferSize(ref *_state));
                }
                else
                {
                    // Buffered REDUCE keeps its dedicated double-loop machinery
                    // (BufferedReduceIternext) and legacy initial fill. NOTE:
                    // that machinery predates masked copy-back; construction
                    // with WRITEMASKED succeeds (NumPy parity — 2.4.2 accepts
                    // the aligned-mask pattern) but the legacy write-back
                    // refuses loudly in CopyReduceBuffersToArrays instead of
                    // silently writing unmasked slots. Proper masked support
                    // lands with reductions-through-core (roadmap Wave 5).
                    NDIterBufferManager.AllocateBuffers(ref *_state, _state->BufferSize);

                    // Copy initial data to buffers (with casting if needed)
                    long copyCount = Math.Min(_state->IterSize, _state->BufferSize);
                    for (int op1 = 0; op1 < nop; op1++)
                    {
                        var opFlag = _state->GetOpFlags(op1);
                        if ((opFlag & NDIterOpFlags.READ) != 0 || (opFlag & NDIterOpFlags.READWRITE) != 0)
                        {
                            NDIterBufferManager.CopyToBuffer(ref *_state, op1, copyCount);
                        }
                    }

                    _state->BufIterEnd = copyCount;
                    SetupBufferedReduction(copyCount);
                }
            }

            // Handle single iteration optimization. NumPy applies it to UNBUFFERED iterators only ("when there is just one
            // iteration and buffering is disabled"): a buffered iterator's iternext must still write its window back when
            // the iteration ends, or a one-element buffered readwrite operand would only be written back at close.
            if (_state->IterSize <= 1 && (_state->ItFlags & (uint)NDIterFlags.BUFFER) == 0)
            {
                _state->ItFlags |= (uint)NDIterFlags.ONEITERATION;
            }

            // PARALLEL_SAFE (NumSharp extension): the iteration range can be
            // split across workers (RANGED + Copy() per worker) without write
            // hazards when:
            //  - there is no REDUCE operand (cross-iteration accumulation on a
            //    shared slot; stretched write dims were flagged REDUCE above), and
            //  - there are no WRITE operands at all, or exactly one WRITE operand
            //    whose potential overlap with the inputs was already resolved by
            //    COPY_IF_OVERLAP processing (forced copy + write-back on Dispose).
            // Costs nothing extra at construction — it only reuses decisions the
            // overlap machinery already made. Consumed by the parallel ForEach
            // work (roadmap Wave 6.2).
            if ((_state->ItFlags & (uint)NDIterFlags.REDUCE) == 0)
            {
                int writeCount = 0;
                for (int iop = 0; iop < nop; iop++)
                {
                    if ((_state->GetOpFlags(iop) & NDIterOpFlags.WRITE) != 0)
                        writeCount++;
                }

                if (writeCount == 0 ||
                    (writeCount == 1 && (flags & NDIterGlobalFlags.COPY_IF_OVERLAP) != 0))
                {
                    _state->ItFlags |= (uint)NDIterFlags.PARALLEL_SAFE;
                }
            }
        }

        /// <summary>
        /// Compute iteration (broadcast) shape from operands.
        /// Uses production NumSharp.Shape.ResolveReturnShape for standard broadcasting.
        /// For op_axes, constructs a virtual shape per operand reflecting the mapping,
        /// then broadcasts those virtual shapes together.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static long[] CalculateBroadcastShape(int nop, NDArray[] op, NDIterPerOpFlags[] opFlags,
            int opAxesNDim = -1, int[][]? opAxes = null)
        {
            // Validate null operands have ALLOCATE or VIRTUAL (NumPy
            // npyiter_prepare_one_operand, nditer_constr.c:999-1007).
            for (int i = 0; i < nop; i++)
            {
                if (op[i] is null &&
                    (opFlags[i] & (NDIterPerOpFlags.ALLOCATE | NDIterPerOpFlags.VIRTUAL)) == 0)
                    throw new ArgumentException(
                        "Iterator operand was NULL, but neither the " +
                        "ALLOCATE nor the VIRTUAL flag was specified");
            }

            // With op_axes, iteration ndim is set by opAxesNDim. Each operand's virtual
            // shape per-iter-axis = opShape[op_axis] if op_axis >= 0, else 1.
            if (opAxes != null && opAxesNDim >= 0)
            {
                // opAxesNDim == 0: every iterated axis was dropped (the IterAllButAxis shape on a
                // 1-D operand, or any drop-all-axes request). NumPy's NDIter_RemoveAxis leaves a
                // 0-dimensional iterator with GetIterSize() == 1 — ONE iteration whose data pointer
                // is the operand base (the dropped axis is the caller's responsibility, walked via
                // its saved stride). Return an empty iteration shape so IterSize == 1. The previous
                // `opAxesNDim > 0` gate fell through to natural broadcasting and returned the
                // operand's own shape [N], so the kernel was driven N times — O(N^2) for the sort
                // driver, whose line kernel re-sorts the whole line on every call.
                if (opAxesNDim == 0)
                    return Array.Empty<long>();

                var virtualShapes = new System.Collections.Generic.List<NumSharp.Shape>(nop);
                for (int opIdx = 0; opIdx < nop; opIdx++)
                {
                    if (op[opIdx] is null)
                        continue;  // ALLOCATE operand adopts broadcast result

                    var virtualDims = new long[opAxesNDim];
                    if (opIdx < opAxes.Length && opAxes[opIdx] != null)
                    {
                        var axisMap = opAxes[opIdx];
                        var opShape = op[opIdx].Shape.dimensions;
                        for (int iterAxis = 0; iterAxis < opAxesNDim; iterAxis++)
                        {
                            int rawOpAxis = iterAxis < axisMap.Length ? axisMap[iterAxis] : -1;
                            // Decode NPY_ITER_REDUCTION_AXIS encoding (common.h:347).
                            int opAxis = NDIterUtils.GetOpAxis(rawOpAxis, out bool isReduction);

                            if (isReduction)
                            {
                                // Explicit reduction axis: operand's axis length must be exactly 1.
                                // If opAxis == -1, treat as broadcast (virtual dim = 1).
                                if (opAxis < 0)
                                {
                                    virtualDims[iterAxis] = 1;
                                }
                                else if (opAxis >= opShape.Length)
                                {
                                    throw new IncorrectShapeException(
                                        $"Operand {opIdx} op_axes refers to non-existent axis {opAxis}");
                                }
                                else
                                {
                                    long len = opShape[opAxis];
                                    if (len != 1)
                                    {
                                        // NumPy's ValueError (npyiter_fill_axisdata's explicit reduction branch).
                                        throw new ValueError(
                                            $"operand was set up as a reduction along axis {opAxis}, but the length of the " +
                                            $"axis is {len} (it has to be 1)");
                                    }
                                    virtualDims[iterAxis] = 1;
                                }
                            }
                            else if (opAxis < 0)
                            {
                                virtualDims[iterAxis] = 1;  // broadcast this dim
                            }
                            else if (opAxis >= opShape.Length)
                            {
                                throw new IncorrectShapeException(
                                    $"Operand {opIdx} op_axes refers to non-existent axis {opAxis}");
                            }
                            else
                            {
                                virtualDims[iterAxis] = opShape[opAxis];
                            }
                        }
                    }
                    else
                    {
                        // No op_axes for this operand: right-align shape to opAxesNDim
                        var opShape = op[opIdx].Shape.dimensions;
                        int offset = opAxesNDim - opShape.Length;
                        if (offset < 0)
                            throw new IncorrectShapeException(
                                $"Operand {opIdx} has {opShape.Length} dims but opAxesNDim={opAxesNDim}");
                        for (int d = 0; d < opAxesNDim; d++)
                            virtualDims[d] = d < offset ? 1 : opShape[d - offset];
                    }
                    virtualShapes.Add(NumSharp.Shape.Adopt(virtualDims));
                }

                if (virtualShapes.Count == 0)
                    return Array.Empty<long>();

                var resolved = NumSharp.Shape.ResolveReturnShape(virtualShapes.ToArray());
                var dims = resolved.dimensions;
                var result = new long[dims.Length];
                for (int i = 0; i < dims.Length; i++)
                    result[i] = dims[i];
                return result;
            }

            // Fast path (per-call construction cost): when every non-null operand
            // shares identical dimensions — the dominant elementwise case
            // (a OP b, out all same shape) — broadcasting is an identity, so the
            // result shape is just those shared dims. Skip the List + ToArray +
            // ResolveReturnShape + Shape allocations and return the dims directly.
            // Dimensions stay long (NumPy npy_intp parity) — no int narrowing.
            {
                long[] commonDims = null;
                bool allSame = true;
                for (int i = 0; i < nop; i++)
                {
                    if (op[i] is null) continue;
                    var d = op[i].Shape.dimensions;
                    if (commonDims is null) { commonDims = d; continue; }
                    if (d.Length != commonDims.Length) { allSame = false; break; }
                    for (int k = 0; k < d.Length; k++)
                        if (d[k] != commonDims[k]) { allSame = false; break; }
                    if (!allSame) break;
                }
                if (allSame && commonDims != null)
                {
                    // commonDims aliases the first non-null operand's immutable dimensions
                    // array. The returned broadcastShape is consumed strictly read-only by
                    // Initialize — copied element-wise into _state->Shape and (only for
                    // ALLOCATE outputs) cloned, never mutated in place — so hand back the
                    // shared reference and skip the defensive new long[] + Array.Copy.
                    return commonDims;
                }
            }

            // Broadcasting shapes that differ (a 0-d scalar operand, a row or column bound): one right-aligned pass
            // straight into the result dims — NumPy's npyiter_fill_axisdata rule, an extent-1 axis adopting any
            // extent. The ResolveReturnShape route below allocates a List, a Shape[] copy, a scratch Shape and a
            // cleaned one per call (~0.6 KB at three operands), all garbage, on every ufunc call whose operands
            // differ in shape. A mismatch is NOT raised here: it falls through to that route, which raises the
            // established exception and message.
            {
                int nd = 0;
                bool anyOperand = false;
                for (int i = 0; i < nop; i++)
                {
                    if (op[i] is null)
                        continue;
                    anyOperand = true;
                    int ond = op[i].Shape.dimensions.Length;
                    if (ond > nd)
                        nd = ond;
                }

                if (anyOperand)
                {
                    var broadcast = new long[nd];
                    for (int d = 0; d < nd; d++)
                        broadcast[d] = 1;

                    bool compatible = true;
                    for (int i = 0; i < nop && compatible; i++)
                    {
                        if (op[i] is null)
                            continue;
                        var od = op[i].Shape.dimensions;
                        int shift = nd - od.Length;
                        for (int k = 0; k < od.Length; k++)
                        {
                            long extent = od[k];
                            if (extent == 1)
                                continue;
                            ref long slot = ref broadcast[shift + k];
                            if (slot == 1)
                                slot = extent;
                            else if (slot != extent)
                            {
                                compatible = false;
                                break;
                            }
                        }
                    }

                    if (compatible)
                        return broadcast;
                }
            }

            // Standard broadcasting: use production NumSharp.Shape.ResolveReturnShape
            var shapes = new System.Collections.Generic.List<NumSharp.Shape>(nop);
            for (int i = 0; i < nop; i++)
            {
                if (op[i] is null)
                    continue;  // ALLOCATE operand adopts broadcast result
                shapes.Add(op[i].Shape);
            }

            if (shapes.Count == 0)
                return Array.Empty<long>();

            var resolvedShape = NumSharp.Shape.ResolveReturnShape(shapes.ToArray());
            var resultDims = resolvedShape.dimensions;
            var finalResult = new long[resultDims.Length];
            for (int i = 0; i < resultDims.Length; i++)
                finalResult[i] = resultDims[i];
            return finalResult;
        }

        /// <summary>
        ///     Validates the operands against the iteration shape in two passes: first the compatibility checks the
        ///     construction needs (an <c>op_axes</c> entry naming an axis the operand has, extents that broadcast), then the
        ///     refusals of NumPy's stride pass (<see cref="ThrowStridePassRefusals"/>: <c>no_broadcast</c>, an explicit
        ///     reduction axis of length other than 1, the reduction rule), in NumPy's order and with NumPy's texts.
        /// </summary>
        /// <param name="nop">The operand count.</param>
        /// <param name="op">The operands; a null entry is an output the iterator allocates and is not shape-checked.</param>
        /// <param name="opFlags">The per-operand flags (<c>NO_BROADCAST</c>, <c>ARRAYMASK</c> and the read/write mode are
        ///     read).</param>
        /// <param name="globalFlags">The construction flags (<c>REDUCE_OK</c> is read).</param>
        /// <param name="iterShape">The iteration shape (the broadcast shape, or the caller's <c>itershape</c>).</param>
        /// <param name="opAxesNDim">The <c>op_axes</c> rank (unused: each map's own length bounds its walk).</param>
        /// <param name="opAxes">The per-operand axis maps, or null for natural broadcasting.</param>
        /// <exception cref="IncorrectShapeException">An <c>op_axes</c> entry naming an axis the operand does not have, an
        ///     operand extent that cannot broadcast to the iteration's, or an operand with more axes than the iteration (the
        ///     house texts: np.nditer raises NumPy's shape-pass errors before it calls the engine).</exception>
        /// <exception cref="ValueError">A stride-pass refusal (<see cref="ThrowStridePassRefusals"/>).</exception>
        /// <remarks>
        ///     The stride-pass refusals run only after every compatibility check passed, as NumPy's stride pass follows its
        ///     shape pass. The <c>no_broadcast</c> check used to be interleaved with the compatibility checks and raised
        ///     <see cref="InvalidOperationException"/> with the shape printed as <c>(3)</c> (NumPy: a ValueError naming
        ///     <c>(3,)</c>), and the explicit reduction length was checked here, before the extents of later operands.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void ValidateIterShape(int nop, NDArray[] op, NDIterPerOpFlags[] opFlags, NDIterGlobalFlags globalFlags,
            long[] iterShape, int opAxesNDim, int[][]? opAxes)
        {
            for (int opIdx = 0; opIdx < nop; opIdx++)
            {
                // Skip null (ALLOCATE) operands - they will adopt the iterShape
                if (op[opIdx] is null)
                    continue;

                var opShape = op[opIdx].Shape.dimensions;

                // When opAxes is provided for this operand, use it for validation
                if (opAxes != null && opIdx < opAxes.Length && opAxes[opIdx] != null)
                {
                    var opAxisMap = opAxes[opIdx];
                    int mapLength = Math.Min(opAxisMap.Length, iterShape.Length);

                    for (int iterAxis = 0; iterAxis < mapLength; iterAxis++)
                    {
                        // Decode NPY_ITER_REDUCTION_AXIS encoding (common.h:347)
                        int opAxis = NDIterUtils.GetOpAxis(opAxisMap[iterAxis], out bool isReduction);

                        // Broadcast or reduction-broadcast: no further shape validation needed
                        if (opAxis < 0)
                            continue;

                        // Validate that the operand axis exists and is compatible
                        if (opAxis >= opShape.Length)
                            throw new IncorrectShapeException($"Operand {opIdx} op_axes refers to non-existent axis {opAxis}");

                        // An explicit reduction axis must have length 1 on the operand; that is a STRIDE-pass refusal in NumPy,
                        // raised after every shape check, so it is ThrowStridePassRefusals' (in NumPy's axis order), not this
                        // loop's. A reduction axis whose extent breaks the broadcast is still the shape error below, as in NumPy.
                        _ = isReduction;

                        long opDim = opShape[opAxis];
                        long iterDim = iterShape[iterAxis];

                        // opDim must equal iterDim or be 1 (broadcastable)
                        if (opDim != iterDim && opDim != 1)
                            throw new IncorrectShapeException($"Operand {opIdx} shape incompatible with iterShape at axis {iterAxis}");
                    }
                }
                else
                {
                    // No opAxes for this operand, use standard broadcasting validation
                    int offset = iterShape.Length - opShape.Length;

                    // Operand must have fewer or equal dimensions
                    if (offset < 0)
                        throw new IncorrectShapeException($"Operand {opIdx} has more dimensions than iterShape");

                    for (int d = 0; d < opShape.Length; d++)
                    {
                        long opDim = opShape[d];
                        long iterDim = iterShape[offset + d];

                        // opDim must equal iterDim or be 1 (broadcastable)
                        if (opDim != iterDim && opDim != 1)
                            throw new IncorrectShapeException($"Operand {opIdx} shape incompatible with iterShape at axis {d}");
                    }
                }
            }

            ThrowStridePassRefusals(nop, op, opFlags, globalFlags, iterShape, opAxes);
        }

        /// <summary>
        ///     The refusals of NumPy's stride pass (<c>npyiter_fill_axisdata</c> after its shape pass), in NumPy's ONE loop:
        ///     a <c>no_broadcast</c> operand the iteration stretches, an explicit reduction axis whose operand extent is not 1,
        ///     and a written operand stretched without <c>reduce_ok</c>, written but not read, or the ARRAYMASK
        ///     (<c>npyiter_check_reduce_ok_and_set_flags</c>).
        /// </summary>
        /// <param name="nop">The operand count.</param>
        /// <param name="op">The operands; a null entry is an output to allocate (checked only through a newaxis map entry,
        ///     as NumPy checks it).</param>
        /// <param name="opFlags">The per-operand flags (the read/write mode, <c>NO_BROADCAST</c>, <c>ARRAYMASK</c>).</param>
        /// <param name="globalFlags">The construction flags (<c>REDUCE_OK</c>).</param>
        /// <param name="iterShape">The broadcast iteration shape, outermost axis first.</param>
        /// <param name="opAxes">The per-operand axis maps, or null for natural broadcasting.</param>
        /// <exception cref="ValueError">The FIRST refusal in NumPy's order, with NumPy's text:
        ///     <c>operand_different_than_broadcast</c>'s "non-broadcastable [output] operand with shape … [remapped to …]
        ///     doesn't match the broadcast shape …"; "operand was set up as a reduction along axis {i}, but the length of the
        ///     axis is {n} (it has to be 1)"; "output operand requires a reduction along dimension {d}, but the reduction is
        ///     not enabled. …"; "output operand requires a reduction, but is flagged as write-only, not read-write";
        ///     "output operand requires a reduction, but is flagged as the ARRAYMASK operand which is not permitted to be the
        ///     result of a reduction".</exception>
        /// <remarks>
        ///     <para>
        ///     <b>Why one pass.</b> NumPy raises all of these from one loop — iteration axes INNERMOST first, every operand in
        ///     order within an axis — so which refusal it reports depends on the axis and the operand, not on the kind. Two
        ///     stretched operands report the one stretched along the inner axis, whatever its position, and a written
        ///     operand stretched along the inner axis is reported before a <c>no_broadcast</c> operand stretched along an
        ///     outer one. NumSharp used to check <c>no_broadcast</c> in one pass, the explicit reduction length in the
        ///     compatibility pass, and the reduction rule per operand during the operand setup and again in
        ///     <see cref="ApplyOpAxes"/>, which reported other operands than NumPy when several were refused. Those later
        ///     checks still set the REDUCE flags and keep their throws; this pass refuses everything they would, first.
        ///     </para>
        ///     <para>
        ///     <b>What counts as stretched</b> (NumPy's branches). Without an axis map: an operand that LACKS the axis
        ///     (<c>no_broadcast</c> refuses it even where the iteration's extent is 1), or whose extent is 1 under an
        ///     iteration extent other than 1 — a ZERO-length iteration axis stretches too, so a written <c>(1, 3)</c> operand
        ///     next to <c>(0, 3)</c> is a refused reduction (the setup's own check tests <c>extent &gt; 1</c> and let it pass).
        ///     With an axis map: an explicit reduction entry is only length-checked; under an iteration extent of 1 nothing
        ///     is checked; a real operand axis of extent 1 is stretched; a newaxis (any negative) entry is checked for a
        ///     reduction even on an operand still to be allocated, and is named <c>dimension -1</c>.
        ///     </para>
        ///     <para>
        ///     <b>Dimensions named.</b> The natural branch names <c>idim</c>, the axis counted from the INNERMOST (an out of
        ///     shape <c>(1, 3)</c> against <c>(2, 3)</c> is "dimension 1"); the mapped branch names the OPERAND axis.
        ///     </para>
        ///     <para>
        ///     <b>How it is walked.</b> Each (axis, operand) check depends only on that operand, the axis and the global flags
        ///     (NumPy's loop sets the REDUCE flags but never reads them back), so the first refusal of NumPy's axis-major loop
        ///     is the lexicographically smallest (axis, operand) among the operands' own first refusals. The walk is therefore
        ///     operand-major — each operand's shape read once, and an operand that can never be refused (read-only, without
        ///     <c>no_broadcast</c>, without an explicit reduction entry) skipped outright, which is every operand of a
        ///     plain read-only iteration — and a later operand replaces the best refusal only with a strictly inner axis. The
        ///     axis-major form read every operand's shape at every axis, twice per np.nditer call (the wrapper and the engine):
        ///     +30–170 ns a construction.
        ///     </para>
        /// </remarks>
        internal static void ThrowStridePassRefusals(int nop, NDArray[] op, NDIterPerOpFlags[] opFlags,
                                                     NDIterGlobalFlags globalFlags, ReadOnlySpan<long> iterShape, int[][]? opAxes)
        {
            int ndim = iterShape.Length;
            if (ndim == 0)
                return;

            bool reduceOk = (globalFlags & NDIterGlobalFlags.REDUCE_OK) != 0;
            // npyiter_prepare_operands settled the single ARRAYMASK operand before the stride pass; the flags name it.
            int maskOp = -1;
            for (int i = 0; i < nop; i++)
            {
                if ((opFlags[i] & NDIterPerOpFlags.ARRAYMASK) != 0)
                {
                    maskOp = i;
                    break;
                }
            }

            ValueError? first = null;
            int firstIdim = ndim;
            for (int opIdx = 0; opIdx < nop; opIdx++)
            {
                var flags = opFlags[opIdx];
                bool noBroadcast = (flags & NDIterPerOpFlags.NO_BROADCAST) != 0;
                bool written = (TranslateOpFlags(flags) & NDIterOpFlags.WRITE) != 0;
                int[]? map = opAxes != null && opIdx < opAxes.Length ? opAxes[opIdx] : null;
                if (!noBroadcast && !written && (map == null || !HasExplicitReductionEntry(map)))
                    continue;

                var operand = op[opIdx];
                long[]? dims = operand?.Shape.dimensions;
                // Only a strictly inner axis can beat an earlier operand's refusal: on the same axis NumPy reaches the earlier
                // operand first.
                for (int idim = 0; idim < firstIdim; idim++)
                {
                    var refusal = StridePassRefusal(operand, dims, flags, noBroadcast, opIdx == maskOp, reduceOk, map, iterShape,
                                                    idim);
                    if (refusal is not null)
                    {
                        first = refusal;
                        firstIdim = idim;
                        break;
                    }
                }
            }

            if (first is not null)
                throw first;
        }

        /// <summary>Whether an <c>op_axes</c> map holds an explicit reduction entry naming an operand axis.</summary>
        /// <param name="map">The operand's axis map.</param>
        /// <returns>True when some entry decodes (<see cref="NDIterUtils.GetOpAxis"/>) to a reduction of a real axis — the
        ///     only entry a read-only operand without <c>no_broadcast</c> can be refused for (its length).</returns>
        private static bool HasExplicitReductionEntry(int[] map)
        {
            foreach (int entry in map)
            {
                if (NDIterUtils.GetOpAxis(entry, out bool explicitReduction) >= 0 && explicitReduction)
                    return true;
            }

            return false;
        }

        /// <summary>
        ///     The refusal NumPy's stride pass raises for one operand at one iteration axis, if any — one iteration of the
        ///     loop body of <c>npyiter_fill_axisdata</c>'s stride pass, both branches (unmapped and mapped).
        /// </summary>
        /// <param name="operand">The operand, or null for one the iterator allocates.</param>
        /// <param name="dims">The operand's dimensions (null with <paramref name="operand"/>).</param>
        /// <param name="flags">The operand's per-operand flags.</param>
        /// <param name="noBroadcast">Whether the operand carries <c>NO_BROADCAST</c>.</param>
        /// <param name="isMask">Whether the operand is the ARRAYMASK.</param>
        /// <param name="reduceOk">Whether the construction set <c>reduce_ok</c>.</param>
        /// <param name="map">The operand's axis map, or null for natural broadcasting.</param>
        /// <param name="iterShape">The iteration shape, outermost axis first.</param>
        /// <param name="idim">The iteration axis, counted from the INNERMOST as NumPy counts it.</param>
        /// <returns>The error NumPy raises there, or null when the operand passes this axis.</returns>
        private static ValueError? StridePassRefusal(NDArray? operand, long[]? dims, NDIterPerOpFlags flags, bool noBroadcast,
                                                     bool isMask, bool reduceOk, int[]? map, ReadOnlySpan<long> iterShape,
                                                     int idim)
        {
            int ndim = iterShape.Length;
            int iterAxis = ndim - 1 - idim;
            long bshape = iterShape[iterAxis];
            if (map == null)
            {
                if (operand is null || dims is null)
                    return null;

                // Right-aligned: idim indexes the operand's own axes from its last one.
                int ondim = dims.Length;
                if (bshape == 1)
                    return idim >= ondim && noBroadcast ? NonBroadcastableOperand(operand, flags, map, iterShape) : null;
                if (idim >= ondim || dims[ondim - 1 - idim] == 1)
                    return noBroadcast
                        ? NonBroadcastableOperand(operand, flags, map, iterShape)
                        : ReductionRefusal(flags, isMask, reduceOk, idim);
                return null;
            }

            // An engine caller may pass a map shorter than the iteration (ApplyOpAxes skips the rest too); NumPy's maps
            // always span it.
            if (iterAxis >= map.Length)
                return null;

            int axis = NDIterUtils.GetOpAxis(map[iterAxis], out bool explicitReduction);
            if (explicitReduction)
            {
                // Marked a reduction without a reduce_ok check (ApplyOpAxes); only its length is refused.
                return axis >= 0 && dims is not null && axis < dims.Length && dims[axis] != 1
                    ? new ValueError(
                        $"operand was set up as a reduction along axis {axis}, but the length of the axis is {dims[axis]} " +
                        "(it has to be 1)")
                    : null;
            }

            // An iteration extent of 1 broadcasts anything, a missing axis included.
            if (bshape == 1)
                return null;

            if (axis >= 0)
            {
                // An allocated operand takes the iteration's extent; a given one is stretched along an axis of 1. The
                // compatibility pass already refused an axis the operand does not have; the bound keeps a malformed map from
                // indexing past the shape.
                if (operand is null || dims is null || axis >= dims.Length || dims[axis] != 1)
                    return null;
                return noBroadcast
                    ? NonBroadcastableOperand(operand, flags, map, iterShape)
                    : ReductionRefusal(flags, isMask, reduceOk, axis);
            }

            // A newaxis entry under a longer iteration axis: a written operand there is a reduction, even an operand still
            // to be allocated (NumPy calls the check whatever op_cur is).
            return ReductionRefusal(flags, isMask, reduceOk, axis);
        }

        /// <summary>
        ///     The refusals of NumPy's <c>npyiter_check_reduce_ok_and_set_flags</c> for an operand the iteration stretches: a
        ///     WRITTEN operand there is a reduction, which needs <c>reduce_ok</c>, a readable operand and not the ARRAYMASK; a
        ///     read-only operand is a broadcast and passes.
        /// </summary>
        /// <param name="flags">The operand's per-operand flags (its read/write mode, through <see cref="TranslateOpFlags"/>).</param>
        /// <param name="isMask">Whether the operand is the ARRAYMASK.</param>
        /// <param name="reduceOk">Whether the construction set <c>reduce_ok</c>.</param>
        /// <param name="dim">The dimension NumPy names: the innermost-counted iteration axis for an unmapped operand, the
        ///     operand axis (negative for a newaxis entry) for a mapped one.</param>
        /// <returns>NumPy's error for the first of its three refusals that applies, or null (a read-only operand, or a
        ///     reduction NumPy accepts).</returns>
        private static ValueError? ReductionRefusal(NDIterPerOpFlags flags, bool isMask, bool reduceOk, int dim)
        {
            var opFlag = TranslateOpFlags(flags);
            if ((opFlag & NDIterOpFlags.WRITE) == 0)
                return null;
            if (!reduceOk)
                return new ValueError(
                    $"output operand requires a reduction along dimension {dim}, " +
                    "but the reduction is not enabled. The dimension size of 1 " +
                    "does not match the expected output shape.");
            if ((opFlag & NDIterOpFlags.READ) == 0)
                return new ValueError(
                    "output operand requires a reduction, but is flagged as " +
                    "write-only, not read-write");
            // A slot could be written while the mask reads True, then the reduction turn the mask False — retroactively
            // invalidating the write — so the ARRAYMASK may never be the result of a reduction.
            if (isMask)
                return new ValueError(
                    "output operand requires a " +
                    "reduction, but is flagged as " +
                    "the ARRAYMASK operand which " +
                    "is not permitted to be the " +
                    "result of a reduction");
            return null;
        }

        /// <summary>
        ///     NumPy's <c>operand_different_than_broadcast</c> ValueError: "non-broadcastable operand" for a read-only operand,
        ///     "non-broadcastable output operand" otherwise, and for an operand with an axis map the remapped shape too.
        /// </summary>
        /// <param name="operand">The stretched operand.</param>
        /// <param name="flags">Its flags (<c>READONLY</c> picks the wording).</param>
        /// <param name="map">Its axis map, or null.</param>
        /// <param name="iterShape">The iteration shape.</param>
        /// <returns>The error to throw (the shapes printed by <see cref="NumSharp.Shape.ConvertShapeToString"/>, NumPy's
        ///     <c>convert_shape_to_string</c>: <c>(3,)</c>, <c>(2,3)</c>, <c>()</c>).</returns>
        /// <remarks>
        ///     NumPy fills the remapped shape as <c>remdims[idim] = shape[axes[ndim - idim - 1]]</c> and prints it in that
        ///     order, so it comes out with the iteration axes REVERSED: a <c>(1, 3)</c> operand mapped <c>[0, 1]</c> reads
        ///     "[remapped to (3,1)]". The text is reproduced as NumPy prints it, not corrected. An entry that names no operand
        ///     axis (a newaxis, an encoded reduction axis) prints as NumPy's unknown dimension: dropped when leading,
        ///     <c>newaxis</c> after (<c>(1,newaxis)</c>).
        /// </remarks>
        private static ValueError NonBroadcastableOperand(NDArray operand, NDIterPerOpFlags flags, int[]? map, ReadOnlySpan<long> iterShape)
        {
            var dims = operand.Shape.dimensions;
            string shape1 = NumSharp.Shape.ConvertShapeToString(dims);
            string shape2 = NumSharp.Shape.ConvertShapeToString(iterShape.ToArray());
            string kind = (flags & NDIterPerOpFlags.READONLY) != 0 ? "operand" : "output operand";
            if (map == null)
                return new ValueError($"non-broadcastable {kind} with shape {shape1} doesn't match the broadcast shape {shape2}");

            int ndim = iterShape.Length;
            var remapped = new long[ndim];
            for (int idim = 0; idim < ndim; idim++)
            {
                int entry = ndim - idim - 1;
                // The RAW entry, as NumPy reads it: an encoded reduction axis is a large positive value, not an axis.
                int raw = entry < map.Length ? map[entry] : -1;
                remapped[idim] = raw >= 0 && raw < dims.Length ? dims[raw] : -1;
            }

            return new ValueError($"non-broadcastable {kind} with shape {shape1} [remapped to " +
                                  $"{NumSharp.Shape.ConvertShapeToString(remapped)}] doesn't match the broadcast shape {shape2}");
        }

        /// <summary>
        ///     Whether iteration axis <paramref name="d"/> BROADCASTS an operand of shape <paramref name="arrShape"/> under
        ///     the standard (right-aligned) broadcasting: the operand has no axis there, or its axis has length 1. This is
        ///     NumPy's test for a reduction axis of a written operand (<c>npyiter_fill_axisdata</c>'s <c>shape == 1</c> and
        ///     absent-axis branches), and it is a SHAPE test: a stride of 0 on an axis the operand really has (a zero-size
        ///     array's strides, an explicit 0 stride of <c>np.ndarray(..., strides=...)</c>) is not a broadcast.
        /// </summary>
        /// <param name="arrShape">The operand's own shape (not broadcast).</param>
        /// <param name="d">The iteration axis.</param>
        /// <param name="iterNDim">The iteration's rank (the operand is right-aligned against it).</param>
        /// <returns>True when the operand is stretched along the axis.</returns>
        private static bool IsBroadcastAxis(in NumSharp.Shape arrShape, int d, int iterNDim)
        {
            int off = iterNDim - arrShape.NDim;
            return d < off || arrShape.dimensions[d - off] == 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static NDIterOpFlags TranslateOpFlags(NDIterPerOpFlags flags)
        {
            var result = NDIterOpFlags.None;

            if ((flags & NDIterPerOpFlags.READONLY) != 0)
                result |= NDIterOpFlags.READ;
            if ((flags & NDIterPerOpFlags.WRITEONLY) != 0)
                result |= NDIterOpFlags.WRITE;
            if ((flags & NDIterPerOpFlags.READWRITE) != 0)
                result |= NDIterOpFlags.READWRITE;
            if ((flags & NDIterPerOpFlags.COPY) != 0)
                result |= NDIterOpFlags.FORCECOPY;
            if ((flags & NDIterPerOpFlags.CONTIG) != 0)
                result |= NDIterOpFlags.CONTIG;
            // WRITEMASKED: the operand is written only where the mask (ARRAYMASK) is true.
            // Requires a corresponding ARRAYMASK operand. NumPy nditer_constr.c:950-965.
            if ((flags & NDIterPerOpFlags.WRITEMASKED) != 0)
                result |= NDIterOpFlags.WRITEMASKED;
            // VIRTUAL: the operand had no backing array — the iterator allocated
            // one (NumPy 2.x allocate-equivalent semantics; the flag survives
            // only as state, mirroring NPY_OP_ITFLAG_VIRTUAL whose sole consumer
            // is DebugPrint). NumPy nditer_constr.c:967-975.
            if ((flags & NDIterPerOpFlags.VIRTUAL) != 0)
                result |= NDIterOpFlags.VIRTUAL;

            return result;
        }

        /// <summary>
        /// Pre-construction per-operand flag validation. Matches NumPy's
        /// npyiter_prepare_operands loop (nditer_constr.c:1159-1242): per-op
        /// flag consistency (npyiter_check_per_op_flags, c:951-975), ARRAYMASK
        /// uniqueness, null/VIRTUAL operand rules (npyiter_prepare_one_operand,
        /// c:999-1062), then the WRITEMASKED/ARRAYMASK pairing checks — in
        /// NumPy's exact order, with NumPy's exact error texts.
        /// Runs before state allocation (uses the raw <paramref name="nop"/> arg).
        /// </summary>
        private static void PreCheckMaskOpPairing(int nop, NDArray[] op, NDIterPerOpFlags[] opFlags)
        {
            // Fast path (small-N construction cost): one OR-sweep decides
            // whether any operand carries a mask/virtual flag or is null;
            // the typical elementwise construction (plain READ/WRITE flags,
            // non-null operands) pays 1 OR + 1 null-test per op and skips
            // the full validation loop entirely.
            NDIterPerOpFlags combined = NDIterPerOpFlags.None;
            bool anyNull = false;
            for (int iop = 0; iop < nop && iop < opFlags.Length; iop++)
            {
                combined |= opFlags[iop];
                anyNull |= iop < op.Length && op[iop] is null;
            }

            if (!anyNull &&
                (combined & (NDIterPerOpFlags.WRITEMASKED |
                             NDIterPerOpFlags.ARRAYMASK |
                             NDIterPerOpFlags.VIRTUAL)) == 0)
                return;

            int maskOp = -1;
            bool anyWriteMasked = false;

            for (int iop = 0; iop < nop && iop < opFlags.Length; iop++)
            {
                var flags = opFlags[iop];
                bool isArrayMask = (flags & NDIterPerOpFlags.ARRAYMASK) != 0;
                bool isWriteMasked = (flags & NDIterPerOpFlags.WRITEMASKED) != 0;

                // WRITEMASKED needs a writable operand (nditer_constr.c:951-957).
                if (isWriteMasked &&
                    (flags & (NDIterPerOpFlags.WRITEONLY | NDIterPerOpFlags.READWRITE)) == 0)
                    throw new ArgumentException(
                        "The iterator flag WRITEMASKED may only " +
                        "be used with READWRITE or WRITEONLY");

                // The mask cannot itself be write-masked (nditer_constr.c:958-963).
                if (isWriteMasked && isArrayMask)
                    throw new ArgumentException(
                        "The iterator flag WRITEMASKED may not " +
                        "be used together with ARRAYMASK");

                // VIRTUAL requires READWRITE (nditer_constr.c:967-973). NumPy's
                // message carries a doubled "be" — matched verbatim.
                if ((flags & NDIterPerOpFlags.VIRTUAL) != 0 &&
                    (flags & NDIterPerOpFlags.READWRITE) == 0)
                    throw new ArgumentException(
                        "The iterator flag VIRTUAL should be " +
                        "be used together with READWRITE");

                if (isArrayMask)
                {
                    if (maskOp >= 0)
                        throw new ArgumentException(
                            "Only one iterator operand may receive an " +
                            "ARRAYMASK flag");
                    maskOp = iop;
                }

                if (isWriteMasked) anyWriteMasked = true;

                // npyiter_prepare_one_operand: null operands must be ALLOCATE
                // or VIRTUAL; VIRTUAL operands must be null (c:999-1062).
                if (iop < op.Length && op[iop] is null)
                {
                    if ((flags & (NDIterPerOpFlags.ALLOCATE | NDIterPerOpFlags.VIRTUAL)) == 0)
                        throw new ArgumentException(
                            "Iterator operand was NULL, but neither the " +
                            "ALLOCATE nor the VIRTUAL flag was specified");
                }
                else if ((flags & NDIterPerOpFlags.VIRTUAL) != 0)
                {
                    throw new ArgumentException(
                        "Iterator operand flag VIRTUAL was specified, " +
                        "but the operand was not NULL");
                }
            }

            if (anyWriteMasked && maskOp < 0)
                throw new ArgumentException(
                    "An iterator operand was flagged as WRITEMASKED, " +
                    "but no ARRAYMASK operand was given to supply " +
                    "the mask");
            if (!anyWriteMasked && maskOp >= 0)
                throw new ArgumentException(
                    "An iterator operand was flagged as the ARRAYMASK, " +
                    "but no WRITEMASKED operands were given to use " +
                    "the mask");
        }

        // =========================================================================
        // COPY_IF_OVERLAP (NumPy nditer_constr.c:3083-3168)
        // =========================================================================

        /// <summary>
        /// Decide which WRITE operands must be force-copied because they may
        /// share memory with a READ operand. Returns null when no copies are
        /// needed (the common case — fresh outputs never overlap inputs, and
        /// the memory-extent fast path rejects disjoint buffers cheaply).
        ///
        /// Mirrors NumPy's check loop exactly: only write operands are copied
        /// ("a more sophisticated heuristic could be substituted here later"),
        /// read operands already force-copied are skipped, and the
        /// OVERLAP_ASSUME_ELEMENTWISE pair short-circuit allows exact aliasing
        /// (same data, same layout, no internal overlap) without a copy because
        /// the caller's inner loop reads each element before writing it.
        /// </summary>
        private static bool[]? ComputeCopyIfOverlap(int nop, NDArray[] op, NDIterPerOpFlags[] opFlags)
        {
            bool[]? force = null;

            for (int iop = 0; iop < nop; iop++)
            {
                if (op[iop] is null)
                    continue; // iterator will allocate — cannot overlap

                if ((opFlags[iop] & (NDIterPerOpFlags.WRITEONLY | NDIterPerOpFlags.READWRITE)) == 0)
                    continue; // copy output operands only, not inputs

                bool mayShare = false;
                for (int iother = 0; iother < nop; iother++)
                {
                    if (iother == iop || op[iother] is null)
                        continue;

                    if ((opFlags[iother] & (NDIterPerOpFlags.READONLY | NDIterPerOpFlags.READWRITE)) == 0)
                        continue; // no data dependence for arrays not read from

                    if (force != null && force[iother])
                        continue; // already copied

                    if (AssumeElementwiseExactAlias(op[iop], op[iother], opFlags[iop], opFlags[iother]))
                        continue;

                    // Use max work = 1, as NumPy does for this check.
                    if (NDMemOverlap.SolveMayShareMemory(op[iop], op[iother], maxWork: 1) != MemOverlap.No)
                    {
                        mayShare = true;
                        break;
                    }
                }

                if (mayShare)
                {
                    force ??= new bool[nop];
                    force[iop] = true;
                }
            }

            return force;
        }

        /// <summary>
        /// The OVERLAP_ASSUME_ELEMENTWISE short-circuit: views of exactly the
        /// same data with identical layout need no copy when the caller
        /// accesses data strictly element-by-element in iterator order — but
        /// only if the array has no internal overlap (e.g. a stride-0
        /// broadcast dimension). NumPy nditer_constr.c:3130-3152.
        /// </summary>
        private static bool AssumeElementwiseExactAlias(NDArray a, NDArray b, NDIterPerOpFlags aFlags, NDIterPerOpFlags bFlags)
        {
            if ((aFlags & NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP) == 0 ||
                (bFlags & NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP) == 0)
                return false;

            // NumPy compares the descriptor objects (PyArray_DESCR(a) == PyArray_DESCR(b)); structural equality is that
            // test for the views it can matter for, and it tells two string widths of the one String lane apart.
            if (a.GetTypeCode != b.GetTypeCode || (a.typecode == NPTypeCode.String && a.dtype != b.dtype))
                return false;

            var sa = a.Shape;
            var sb = b.Shape;
            if (sa.NDim != sb.NDim)
                return false;

            int itemsize = ElementBytes(a);
            byte* baseA = (byte*)a.Address + (long)sa.offset * itemsize;
            byte* baseB = (byte*)b.Address + (long)sb.offset * itemsize;
            if (baseA != baseB)
                return false;

            var dimsA = sa.dimensions;
            var dimsB = sb.dimensions;
            var stridesA = sa.strides;
            var stridesB = sb.strides;
            for (int d = 0; d < sa.NDim; d++)
            {
                if (dimsA[d] != dimsB[d] || stridesA[d] != stridesB[d])
                    return false;
            }

            return NDMemOverlap.SolveMayHaveInternalOverlap(a, maxWork: 1) == MemOverlap.No;
        }

        /// <summary>
        /// Replace each force-copied WRITE operand with a fresh C-order
        /// temporary: copy the original in when the operand is also READ, and
        /// register the original for write-back on <see cref="Dispose"/>.
        /// NumPy: the FORCECOPY branch of npyiter_allocate_arrays
        /// (nditer_constr.c:3259-3311, WRITEBACKIFCOPY).
        /// </summary>
        private void MaterializeForcedCopies(int nop, NDArray[] op, NDIterPerOpFlags[] opFlags, bool[] forceCopy)
        {
            for (int iop = 0; iop < nop; iop++)
            {
                if (!forceCopy[iop])
                    continue;

                var original = op[iop];
                // The operand's own DESCRIPTOR: a fixed-width string temp needs its width (the String lane names none).
                var temp = new NDArray(original.dtype, original.Shape.Clean(), fillZeros: false);

                // If the data will be read, copy it into the temporary.
                if ((opFlags[iop] & (NDIterPerOpFlags.READONLY | NDIterPerOpFlags.READWRITE)) != 0)
                    np.copyto(temp, original);

                op[iop] = temp;

                _writebackOriginals ??= new NDArray?[nop];
                _writebackOriginals[iop] = original;
            }
        }

        /// <summary>
        /// Resolve pending COPY_IF_OVERLAP write-backs: copy each temporary
        /// operand back into the user's original array. Idempotent. NumPy
        /// resolves WRITEBACKIFCOPY in NDIter_Deallocate.
        /// </summary>
        private void ResolveWritebacks()
        {
            // The read-only copy routes first: their temporaries go back to the pool and the slots to the originals,
            // with nothing copied back (they were only read).
            var releaseOnly = _releaseOnlyOriginals;
            _releaseOnlyOriginals = null;
            ReleaseRouteTemps(_operands, releaseOnly);

            var originals = _writebackOriginals;
            if (originals is null)
                return;
            _writebackOriginals = null;

            var ops = _operands;
            if (ops is null)
                return;

            for (int iop = 0; iop < originals.Length && iop < ops.Length; iop++)
            {
                var original = originals[iop];
                if (original is null)
                    continue;
                var temp = ops[iop];
                // NumPy's PyArray_ResolveWritebackIfCopy copies with UNSAFE casting (PyArray_CopyAnyInto): a temporary in
                // its iteration dtype (an updateifcopy cast) writes back whatever the construction-time rule was.
                np.copyto(original, temp, casting: "unsafe");

                // The forced-copy temporary has served its purpose: put the user's original
                // back in the operand slot (NumPy's operands revert to the originals once
                // WRITEBACKIFCOPY resolves) and release the temp NOW. It used to be dropped
                // here and reclaimed only by a later GC + finalizer pass — one leaked pooled
                // buffer per in-place ufunc call (np.add(a, b, out=a) is COPY_IF_OVERLAP's
                // bread and butter), which is exactly the finalizer buffer-lifetime tax the
                // small-N benchmarks keep surfacing. Views taken over the temp (GetIterView)
                // keep their own reference on the storage block, so this only drops ours.
                ops[iop] = original;
                temp.Dispose();
            }
        }

        /// <summary>
        ///     Puts each original back in its operand slot and releases the temporary that stood in for it, copying nothing
        ///     back — the teardown of a read-only copy route, and of every temporary when a construction fails.
        /// </summary>
        /// <param name="ops">The operand slots (the temporaries are in them).</param>
        /// <param name="originals">Per slot, the original the temporary replaced, or null where nothing was replaced.</param>
        internal static void ReleaseRouteTemps(NDArray[]? ops, NDArray?[]? originals)
        {
            if (ops is null || originals is null)
                return;
            for (int iop = 0; iop < originals.Length && iop < ops.Length; iop++)
            {
                var original = originals[iop];
                if (original is null)
                    continue;
                var temp = ops[iop];
                ops[iop] = original;
                if (!ReferenceEquals(temp, original))
                    temp?.Dispose();
            }
        }

        /// <summary>
        ///     The failed-construction teardown of the temporaries a construction made: every overlap or copy-route
        ///     temporary is released and its slot reverted, and NO write-back runs — NumPy's
        ///     <c>PyArray_DiscardWritebackIfCopy</c> when <c>NpyIter_Deallocate</c> runs under a pending error. The operands
        ///     the iterator allocated are disposed too (NumPy decrefs them with the iterator), their slots set back to null.
        /// </summary>
        private void DiscardConstructionTemps()
        {
            var releaseOnly = _releaseOnlyOriginals;
            var writeback = _writebackOriginals;
            _releaseOnlyOriginals = null;
            _writebackOriginals = null;
            ReleaseRouteTemps(_operands, releaseOnly);
            ReleaseRouteTemps(_operands, writeback);

            // The arrays the iterator allocated never reached the caller: released, and the caller's null restored in its
            // operand array (the construction allocated into that array in place).
            var allocated = _allocatedSlots;
            _allocatedSlots = null;
            if (allocated != null && _operands != null)
            {
                for (int i = 0; i < allocated.Length && i < _operands.Length; i++)
                {
                    if (!allocated[i])
                        continue;
                    _operands[i]?.Dispose();
                    _operands[i] = null;
                }
            }
        }

        /// <summary>
        /// Sets <see cref="NDIterState.MaskOp"/> from the ARRAYMASK operand (if any).
        /// Pre-validated by <see cref="PreCheckMaskOpPairing"/>.
        /// </summary>
        private void SetMaskOpFromFlags(NDIterPerOpFlags[] opFlags)
        {
            for (int iop = 0; iop < _state->NOp && iop < opFlags.Length; iop++)
            {
                if ((opFlags[iop] & NDIterPerOpFlags.ARRAYMASK) != 0)
                {
                    _state->MaskOp = iop;
                    return;
                }
            }
        }

        /// <summary>
        /// Validates that a WRITEMASKED + REDUCE operand has exactly one mask value per
        /// reduction element. Matches NumPy's check_mask_for_writemasked_reduction
        /// (nditer_constr.c:1328-1377).
        ///
        /// The pathological case: maskstride != 0 && operand_stride == 0 on any axis
        /// means the operand is being broadcast but the mask is not — producing
        /// multiple mask values per reduction element, which is invalid.
        /// </summary>
        private void CheckMaskForWriteMaskedReduction(int iop)
        {
            int maskOp = _state->MaskOp;
            if (maskOp < 0) return;

            int stridesNDim = _state->StridesNDim;
            for (int idim = 0; idim < _state->NDim; idim++)
            {
                long iStride = _state->Strides[iop * stridesNDim + idim];
                long maskStride = _state->Strides[maskOp * stridesNDim + idim];

                if (maskStride != 0 && iStride == 0)
                {
                    throw new InvalidOperationException(
                        "Iterator reduction operand is WRITEMASKED, but also broadcasts " +
                        "to multiple mask values. There can be only one mask value per " +
                        "WRITEMASKED element.");
                }
            }
        }

        private void UpdateContiguityFlags()
        {
            if (_state->IterSize <= 1)
            {
                _state->ItFlags |= (uint)(NDIterFlags.SourceContiguous | NDIterFlags.DestinationContiguous | NDIterFlags.CONTIGUOUS);
                return;
            }

            bool allContiguous = true;

            for (int op = 0; op < _state->NOp; op++)
            {
                var stridePtr = _state->GetStridesPointer(op);
                if (!CheckContiguous(_state->GetShapePointer(), stridePtr, _state->NDim))
                {
                    allContiguous = false;
                    break;
                }
            }

            if (allContiguous)
            {
                _state->ItFlags |= (uint)NDIterFlags.CONTIGUOUS;
            }
            else
            {
                // GATHER_ELIGIBLE: every operand is a gather-capable element
                // width (32/64-bit) and every inner-axis byte stride fits the
                // int32 gather-index budget — the Tier-3B kernels' AVX2
                // hardware-gather strided path can service the inner loop.
                // Informational (kernels re-check at runtime); used by
                // DebugPrint and future per-layout kernel-key selection.
                bool gatherEligible = true;
                int innerDim = _state->NDim - 1;
                for (int op = 0; op < _state->NOp && gatherEligible; op++)
                {
                    // The stored element size: a fixed-width string's item size is not a function of its lane.
                    int elemSize = _state->ElementSizes[op];
                    if (elemSize != 4 && elemSize != 8)
                    {
                        gatherEligible = false;
                        break;
                    }
                    long byteStride = _state->GetStridesPointer(op)[innerDim] * elemSize;
                    if (byteStride > DirectILKernelGenerator.GatherStrideLimit ||
                        byteStride < -DirectILKernelGenerator.GatherStrideLimit)
                    {
                        gatherEligible = false;
                    }
                }

                if (gatherEligible)
                    _state->ItFlags |= (uint)NDIterFlags.GATHER_ELIGIBLE;
            }

            // Set legacy flags for first two operands
            if (_state->NOp >= 1)
            {
                var stridePtr = _state->GetStridesPointer(0);
                if (CheckContiguous(_state->GetShapePointer(), stridePtr, _state->NDim))
                    _state->ItFlags |= (uint)NDIterFlags.SourceContiguous;
            }

            if (_state->NOp >= 2)
            {
                var stridePtr = _state->GetStridesPointer(1);
                if (CheckContiguous(_state->GetShapePointer(), stridePtr, _state->NDim))
                    _state->ItFlags |= (uint)NDIterFlags.DestinationContiguous;
            }
        }

        private static bool CheckContiguous(long* shape, long* strides, int ndim)
        {
            if (ndim == 0)
                return true;

            long expected = 1;
            for (int axis = ndim - 1; axis >= 0; axis--)
            {
                long dim = shape[axis];
                if (dim == 0)
                    return true;
                if (dim != 1)
                {
                    if (strides[axis] != expected)
                        return false;
                    expected *= dim;
                }
            }

            return true;
        }

        /// <summary>
        /// Check if all operands are contiguous in the specified order.
        /// Uses the ORIGINAL operand arrays (before any axis reordering).
        /// </summary>
        /// <param name="cOrder">True for C-order (row-major), false for F-order (column-major)</param>
        /// <param name="allowFlip">True if negative strides will be flipped (K-order).
        /// When true, uses absolute values for stride comparison. When false (C/F/A forced
        /// orders), requires actual positive strides for contiguity.</param>
        private bool CheckAllOperandsContiguous(bool cOrder, bool allowFlip = true)
        {
            // Contiguity must be judged on the ITERATOR's effective layout, not the raw operand's.
            // With op_axes the iterator spans a SUBSET of the operand's axes: axes [0,2] of a
            // C-contiguous (2,3,2) give iterator strides (6,1) over shape (2,2), which is NOT
            // contiguous — yet the full operand still IS C-contiguous. Reading arr.shape/arr.strides
            // reported that operand contiguity and wrongly enabled coalescing for the op_axes subset,
            // so the ascending for-coalescing sort ran, CoalesceAxes could not merge (1*2 != 6), and
            // the iterator was left in F-order (np.nested_iters' bare value stream diverged from
            // NumPy's memory order). NumPy itself judges contiguity from the axisdata (the iterator's
            // own strides), never the base operand. The iterator state already carries the op_axes
            // remapping plus any broadcast fill / negative-stride flip that ran just above, so for a
            // PLAIN operand this is exactly equivalent to the old operand-based check (iterator strides
            // == operand strides there); it only changes the op_axes and other virtual-layout cases.
            if (_state->NDim <= 1)
                return true;  // 0-D / 1-D iteration is trivially contiguous

            int stridesNDim = _state->StridesNDim;
            var shape = _state->Shape;
            var strides = _state->Strides;

            for (int op = 0; op < _state->NOp; op++)
            {
                // An allocated operand is laid out by the order these tests decide: it does not vote (see _abstainOps).
                if (_abstainOps != null && _abstainOps[op])
                    continue;

                int baseIdx = op * stridesNDim;

                // Check contiguity using the iterator's per-operand strides.
                // Negative strides are only treated as "contiguous" when FlipNegativeStrides
                // will run (K-order / A-order without FORCEDORDER); it has already run by this
                // point, so the state strides are positive there — Math.Abs keeps a plain
                // reversed operand contiguous. For forced C/F order, negative strides break
                // contiguity because the iterator traverses logical order, not memory order.
                long expected = 1;
                if (cOrder)
                {
                    // C-order: last axis fastest, check from end to start
                    for (int axis = _state->NDim - 1; axis >= 0; axis--)
                    {
                        long dim = shape[axis];
                        if (dim == 1)
                            continue;  // Size-1 dimensions are always contiguous
                        long stride = allowFlip ? Math.Abs(strides[baseIdx + axis]) : strides[baseIdx + axis];
                        if (stride != expected)
                            return false;
                        expected *= dim;
                    }
                }
                else
                {
                    // F-order: first axis fastest, check from start to end
                    for (int axis = 0; axis < _state->NDim; axis++)
                    {
                        long dim = shape[axis];
                        if (dim == 1)
                            continue;  // Size-1 dimensions are always contiguous
                        long stride = allowFlip ? Math.Abs(strides[baseIdx + axis]) : strides[baseIdx + axis];
                        if (stride != expected)
                            return false;
                        expected *= dim;
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Check if any operand has broadcast strides (stride=0) in the iterator state.
        /// Broadcasting breaks stride-based sorting for K-order iteration.
        /// </summary>
        private bool HasBroadcastStrides()
        {
            if (_state->NDim <= 1)
                return false;

            int stridesNDim = _state->StridesNDim;
            var strides = _state->Strides;

            for (int op = 0; op < _state->NOp; op++)
            {
                // An allocated operand's strides are placeholders until it is laid out (see _abstainOps).
                if (_abstainOps != null && _abstainOps[op])
                    continue;

                int baseIdx = op * stridesNDim;
                for (int d = 0; d < _state->NDim; d++)
                {
                    // stride=0 with shape > 1 indicates a broadcast dimension
                    if (strides[baseIdx + d] == 0 && _state->Shape[d] > 1)
                        return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Set up buffered reduction double-loop parameters.
        /// Implements NumPy's pattern from nditer_api.c lines 2142-2149.
        ///
        /// The double-loop separates iteration into:
        /// - Inner loop: CoreSize elements (non-reduce dimensions)
        /// - Outer loop: ReduceOuterSize iterations (reduce dimension)
        ///
        /// Key insight: reduce operands have ReduceOuterStride=0, so their pointer
        /// stays fixed while input advances, accumulating values without re-buffering.
        /// </summary>
        private void SetupBufferedReduction(long transferSize)
        {
            // Find the outermost reduce dimension (dimension with stride=0 for a reduce operand)
            // For simplicity, we use the outermost dimension with any reduce operand having stride=0
            int outerDim = -1;
            for (int d = 0; d < _state->NDim; d++)
            {
                for (int op = 0; op < _state->NOp; op++)
                {
                    var opFlags = _state->GetOpFlags(op);
                    if ((opFlags & NDIterOpFlags.REDUCE) != 0)
                    {
                        long stride = _state->GetStride(d, op);
                        if (stride == 0 && _state->Shape[d] > 1)
                        {
                            outerDim = d;
                            break;  // Found reduce dimension
                        }
                    }
                }
                if (outerDim >= 0)
                    break;
            }

            if (outerDim < 0)
            {
                // No actual reduce dimension found, treat as normal buffering
                _state->CoreSize = transferSize;
                _state->ReduceOuterSize = 1;
                _state->ReducePos = 0;
                _state->OuterDim = 0;
                return;
            }

            _state->OuterDim = outerDim;

            // Count non-reduce axes. The double-loop (inner reduce-axis, outer non-reduce-axis)
            // only supports ONE non-reduce axis. For multiple non-reduce axes, the outer
            // advance needs multi-axis carry which single-stride double-loop can't express.
            int nonReduceAxisCount = 0;
            int firstNonReduceAxis = -1;
            for (int d = 0; d < _state->NDim; d++)
            {
                if (d != outerDim && _state->Shape[d] > 1)
                {
                    nonReduceAxisCount++;
                    if (firstNonReduceAxis < 0) firstNonReduceAxis = d;
                }
            }

            // When the iteration fits entirely in the buffer AND has >1 non-reduce axis,
            // defer to the regular N-D Advance() path (which correctly carries multiple
            // non-reduce axes via Coords + per-axis strides). Setting CoreSize = 0
            // short-circuits the BUFFER+REDUCE fast path in Iternext().
            if (nonReduceAxisCount > 1)
            {
                _state->CoreSize = 0;
                _state->ReduceOuterSize = 1;
                _state->ReducePos = 0;
                _state->CorePos = 0;
                return;
            }

            // CoreSize = size of the REDUCE dimension (how many inputs accumulate per output).
            // Inner loop iterates CoreSize times along the reduce axis, with the reduce
            // operand fixed (stride=0) and non-reduce operands advancing along that axis.
            long coreSize = _state->Shape[outerDim];
            if (coreSize < 1)
                coreSize = 1;

            _state->CoreSize = coreSize;

            // ReduceOuterSize = number of output slots = total iterations / inputs per output
            _state->ReduceOuterSize = transferSize / coreSize;
            if (_state->ReduceOuterSize < 1)
                _state->ReduceOuterSize = 1;

            // Reset reduce position and core position
            _state->ReducePos = 0;
            _state->CorePos = 0;

            // Identify a non-reduce axis: any axis with Shape > 1 that is not the reduce axis.
            // For 2D single-reduce cases this is unambiguous. For higher-dim cases, NumPy
            // splits across multiple levels; we pick the first non-reduce axis found (limited
            // support for >2D reduce — caller should broadcast into 2D when possible).
            int nonReduceAxis = -1;
            for (int d = 0; d < _state->NDim; d++)
            {
                if (d != outerDim && _state->Shape[d] > 1)
                {
                    nonReduceAxis = d;
                    break;
                }
            }

            int stridesNDim = _state->StridesNDim;

            // Set up per-operand strides for the double-loop.
            //
            // Inner loop (BufStride): advances along the REDUCE axis (outerDim).
            //   - Reduce operand: stride 0 on reduce axis → BufStride = 0 (stays on same output)
            //   - Non-reduce operand: array stride along reduce axis (in bytes)
            //
            // Outer loop (ReduceOuterStride): advances along the NON-reduce axis.
            //   - Reduce operand: stride along non-reduce axis (in bytes) — moves to next output
            //   - Non-reduce operand: stride along non-reduce axis (in bytes) — moves to next input column
            //
            // Matches NumPy nditer_api.c:npyiter_copy_to_buffers buffered-reduce path.
            for (int op = 0; op < _state->NOp; op++)
            {
                int elemSize = _state->GetElementSize(op);

                long innerElemStride = _state->Strides[op * stridesNDim + outerDim];
                long outerElemStride = nonReduceAxis >= 0
                    ? _state->Strides[op * stridesNDim + nonReduceAxis]
                    : 0;

                _state->SetBufStride(op, innerElemStride * elemSize);
                _state->SetReduceOuterStride(op, outerElemStride * elemSize);
            }

            // Set buffer iteration end
            // When bufferSize < coreSize, we can't fit a full core in one buffer
            // In this case, use bufferSize as the inner loop size, not coreSize
            long effectiveInnerSize = Math.Min(_state->BufferSize, coreSize);
            _state->BufIterEnd = effectiveInnerSize;

            // Recalculate ReduceOuterSize based on what fits in buffer
            // This represents how many complete output positions we can process per buffer
            // When buffer is smaller than core, ReduceOuterSize = 1 (one partial core at a time)
            if (_state->BufferSize < coreSize)
            {
                _state->ReduceOuterSize = 1;  // Process one (partial) output at a time
            }

            // Save current array positions for writeback BEFORE overwriting with buffer pointers
            // DataPtrs currently point to array positions (from initialization)
            for (int op = 0; op < _state->NOp; op++)
            {
                _state->SetArrayWritebackPtr(op, _state->GetDataPtr(op));
            }

            // For buffered reduce, DataPtrs need to point into buffers, not original arrays
            // BufferedReduceAdvance will update these using BufStrides
            for (int op = 0; op < _state->NOp; op++)
            {
                var buffer = _state->GetBuffer(op);
                if (buffer != null)
                {
                    _state->SetDataPtr(op, buffer);
                }
            }

            // Initialize reduce outer pointers from current data pointers (now pointing to buffers)
            _state->InitReduceOuterPtrs();
        }

        /// <summary>
        ///     The reduction semantics of the op_axes-mapped operands — the op_axes branch of NumPy's
        ///     <c>npyiter_fill_axisdata</c>. The strides are already set by the operand setup; this marks the reduction
        ///     axes and raises NumPy's errors, in NumPy's order.
        /// </summary>
        /// <param name="opAxesNDim">The iteration rank op_axes describes.</param>
        /// <param name="opAxes">Per operand, the operand axis of each iterator axis (-1 for an axis it lacks; an explicit
        ///     reduction axis encoded by <see cref="NDIterUtils.ReductionAxis"/>), or null for the operand's own axes.</param>
        /// <param name="globalFlags">The construction flags (<c>REDUCE_OK</c>).</param>
        /// <exception cref="ValueError">A written operand needs a reduction along an iterator axis (an op_axes -1 entry, or
        ///     an operand axis of length 1 under a longer iterator axis) without <c>reduce_ok</c> ("output operand requires
        ///     a reduction along dimension {i}, but the reduction is not enabled. ..."), is write-only ("... flagged as
        ///     write-only, not read-write") or is the ARRAYMASK operand.</exception>
        /// <remarks>
        ///     NumPy walks the iterator axes from the INNERMOST out and, per axis, every operand in order, so the first
        ///     failing (axis, operand) pair is the one it reports, and the dimension it names is the OPERAND axis
        ///     (<c>-1</c> for an axis the operand does not have). An iterator axis of length 1 broadcasts anything, an
        ///     operand the iterator allocates has no length to compare yet, and an explicit reduction axis is marked as one
        ///     without a <c>reduce_ok</c> check — NumPy's three exemptions. A read-only operand stretched along an axis is a
        ///     broadcast (NumSharp's <c>SourceBroadcast</c> layout flag).
        /// </remarks>
        private void ApplyOpAxes(int opAxesNDim, int[][] opAxes, NDIterGlobalFlags globalFlags)
        {
            if (opAxes == null || opAxesNDim <= 0)
                return;

            int iterNDim = Math.Min(opAxesNDim, _state->NDim);
            bool reduceOkSet = (globalFlags & NDIterGlobalFlags.REDUCE_OK) != 0;

            for (int iterAxis = iterNDim - 1; iterAxis >= 0; iterAxis--)
            {
                for (int op = 0; op < _state->NOp; op++)
                {
                    if (op >= opAxes.Length || opAxes[op] == null || iterAxis >= opAxes[op].Length)
                        continue;

                    int opAxis = NDIterUtils.GetOpAxis(opAxes[op][iterAxis], out bool explicitReduction);
                    var opFlags = _state->GetOpFlags(op);

                    if (explicitReduction)
                    {
                        // "This is explicitly a reduction axis": marked without a REDUCE_OK check (the axis-length-1
                        // validation happens with the broadcast shape).
                        _state->ItFlags |= (uint)NDIterFlags.REDUCE;
                        _state->SetOpFlags(op, opFlags | NDIterOpFlags.REDUCE);
                        continue;
                    }

                    if (_state->Shape[iterAxis] == 1)
                        continue;

                    if (opAxis >= 0)
                    {
                        // An allocated operand gets the iterator's extent (its stride is filled at allocation); a given
                        // one is stretched only along an axis of length 1.
                        if (_operands?[op] is not { } arr)
                            continue;
                        if (opAxis >= arr.Shape.dimensions.Length || arr.Shape.dimensions[opAxis] != 1)
                            continue;
                    }

                    CheckReduceOkAndSetFlags(op, opFlags, opAxis, reduceOkSet);
                }
            }
        }

        /// <summary>
        ///     NumPy's <c>npyiter_check_reduce_ok_and_set_flags</c> for one operand stretched along an iterator axis: a
        ///     written operand there is a reduction, which needs <c>reduce_ok</c>, a readable operand and not the ARRAYMASK;
        ///     a read-only one is a broadcast.
        /// </summary>
        /// <param name="op">The operand index.</param>
        /// <param name="opFlags">The operand's current flags.</param>
        /// <param name="dim">The axis NumPy names in its error: the operand axis, or -1 for one the operand lacks.</param>
        /// <param name="reduceOk">Whether the construction set <c>reduce_ok</c>.</param>
        /// <exception cref="ValueError">NumPy's three refusals, with its texts.</exception>
        private void CheckReduceOkAndSetFlags(int op, NDIterOpFlags opFlags, int dim, bool reduceOk)
        {
            if ((opFlags & NDIterOpFlags.WRITE) == 0)
            {
                _state->ItFlags |= (uint)NDIterFlags.SourceBroadcast;
                return;
            }

            if (!reduceOk)
                throw new ValueError(
                    $"output operand requires a reduction along dimension {dim}, " +
                    "but the reduction is not enabled. The dimension size of 1 " +
                    "does not match the expected output shape.");

            if ((opFlags & NDIterOpFlags.READ) == 0)
                throw new ValueError(
                    "output operand requires a reduction, but is flagged as " +
                    "write-only, not read-write");

            // The ARRAYMASK operand may never be the result of a reduction: a slot could be written while the mask reads
            // True and the reduction then turn the mask False, retroactively invalidating the write.
            if (op == _state->MaskOp)
                throw new ValueError(
                    "output operand requires a " +
                    "reduction, but is flagged as " +
                    "the ARRAYMASK operand which " +
                    "is not permitted to be the " +
                    "result of a reduction");

            // WRITEMASKED + REDUCE mask-broadcast validation is deferred to the unified post-ApplyOpAxes loop in
            // Initialize (NumPy defers it to the tail of npyiter_allocate_arrays the same way).
            _state->ItFlags |= (uint)NDIterFlags.REDUCE;
            _state->SetOpFlags(op, opFlags | NDIterOpFlags.REDUCE);
        }

        // =========================================================================
        // Properties
        // =========================================================================

        /// <summary>Number of operands.</summary>
        public int NOp => _state->NOp;

        /// <summary>
        /// Index of the ARRAYMASK operand (used by WRITEMASKED operands), or -1 if none.
        /// Matches NumPy's NIT_MASKOP(iter).
        /// </summary>
        public int MaskOp => _state->MaskOp;

        /// <summary>
        /// True if any operand is flagged WRITEMASKED (and a corresponding ARRAYMASK exists).
        /// </summary>
        public bool HasWriteMaskedOperand
        {
            get
            {
                if (_state->MaskOp < 0) return false;
                for (int iop = 0; iop < _state->NOp; iop++)
                {
                    if ((_state->GetOpFlags(iop) & NDIterOpFlags.WRITEMASKED) != 0)
                        return true;
                }
                return false;
            }
        }

        /// <summary>Number of dimensions after coalescing.</summary>
        public int NDim => _state->NDim;

        /// <summary>Total iteration count.</summary>
        public long IterSize => _state->IterSize;

        /// <summary>Current iteration index.</summary>
        public long IterIndex => _state->IterIndex;

        /// <summary>Whether iterator requires buffering.</summary>
        public bool RequiresBuffering => (_state->ItFlags & (uint)NDIterFlags.BUFFER) != 0;

        /// <summary>Whether all operands are contiguous.</summary>
        public bool IsContiguous => (_state->ItFlags & (uint)NDIterFlags.CONTIGUOUS) != 0;

        /// <summary>
        /// Whether the iteration range can be split across parallel workers
        /// without write hazards (see <see cref="NDIterFlags.PARALLEL_SAFE"/>):
        /// no REDUCE operand, and either no WRITE operands or exactly one whose
        /// potential overlap with inputs was resolved by COPY_IF_OVERLAP.
        /// </summary>
        public bool IsParallelSafe => (_state->ItFlags & (uint)NDIterFlags.PARALLEL_SAFE) != 0;

        /// <summary>Whether iterator has external loop.</summary>
        public bool HasExternalLoop => (_state->ItFlags & (uint)NDIterFlags.EXLOOP) != 0;

        /// <summary>Whether iterator uses GROWINNER optimization for buffering.</summary>
        public bool HasGrowInner => (_state->ItFlags & (uint)NDIterFlags.GROWINNER) != 0;

        // =========================================================================
        // Iteration Methods
        // =========================================================================

        /// <summary>
        /// Get the iteration-advance function.
        /// </summary>
        public NDIterNextFunc GetIterNext()
        {
            if (_cachedIterNext != null)
                return _cachedIterNext;

            var itflags = (NDIterFlags)_state->ItFlags;

            if ((itflags & NDIterFlags.ONEITERATION) != 0)
                _cachedIterNext = SingleIterationNext;
            else if ((itflags & NDIterFlags.BUFFER) != 0 && (itflags & NDIterFlags.REDUCE) == 0)
            {
                // NumPy protocol split (nditer_templ.c.src): with EXTERNAL_LOOP
                // the user consumes one buffer fill per iternext (window jump);
                // without it, iternext presents ONE ELEMENT at a time, walking
                // the buffer by BufStrides and refilling at the window edge.
                _cachedIterNext = (itflags & NDIterFlags.EXLOOP) != 0
                    ? BufferedNext
                    : BufferedElementNext;
            }
            else if ((itflags & NDIterFlags.EXLOOP) != 0)
                _cachedIterNext = ExternalLoopNext;
            else
                _cachedIterNext = StandardNext;

            return _cachedIterNext;
        }

        private static bool SingleIterationNext(ref NDIterState state)
        {
            if (state.IterIndex >= state.IterEnd)
                return false;
            state.IterIndex = state.IterEnd;
            return false;
        }

        /// <summary>
        /// iternext for windowed buffered (non-reduce) iteration. Matches
        /// NumPy's npyiter_buffered_iternext (nditer_templ.c.src): flush the
        /// current window's WRITE operands back to their arrays, advance to the
        /// window end, reposition into the source arrays, and fill the next
        /// window. Returns false when iteration is complete — by which point
        /// the final window has been flushed.
        /// </summary>
        private static bool BufferedNext(ref NDIterState state)
        {
            NDIterBufferManager.FlushBufferWindow(ref state);

            state.IterIndex = state.BufIterEnd;
            if (state.IterIndex >= state.IterEnd)
                return false;

            // Reposition DataPtrs into the source arrays at the new index
            // (GotoIterIndex computes from ResetDataPtrs — the array bases).
            state.GotoIterIndex(state.IterIndex);

            long count = NDIterBufferManager.ComputeTransferSize(ref state);
            NDIterBufferManager.FillBufferWindow(ref state, count);
            return true;
        }

        /// <summary>
        /// Per-element iternext for buffered (non-reduce) iteration WITHOUT
        /// EXTERNAL_LOOP — NumPy's user-facing nditer protocol: each call
        /// advances one element, walking the current buffer window via
        /// BufStrides; at the window edge the WRITE operands are flushed and
        /// the next window is filled. (NumPy nditer_templ.c.src buffered
        /// specialization + npyiter_buffered_iternext slow path.)
        /// </summary>
        private static bool BufferedElementNext(ref NDIterState state)
        {
            long next = state.IterIndex + 1;
            if (next < state.BufIterEnd)
            {
                state.IterIndex = next;
                for (int op = 0; op < state.NOp; op++)
                    state.DataPtrs[op] += state.BufStrides[op];
                return true;
            }

            // Window exhausted — flush, then refill at the next position.
            // NOTE: FlushBufferWindow relies on Coords still describing the
            // fill start; per-element stepping above only moved DataPtrs.
            NDIterBufferManager.FlushBufferWindow(ref state);

            // NumPy increments the position first (++NIT_ITERINDEX) and refills there. That is the window end whenever a
            // window was filled; while buffer allocation is still DELAYED no window exists (BufIterEnd stays 0), and the
            // position must still advance — np.nditer(..., delay_bufalloc).iternext() moves iterindex to 1, as NumPy's.
            state.IterIndex = next;
            if (state.IterIndex >= state.IterEnd)
                return false;

            state.GotoIterIndex(state.IterIndex);
            long count = NDIterBufferManager.ComputeTransferSize(ref state);
            NDIterBufferManager.FillBufferWindow(ref state, count);
            return true;
        }

        /// <summary>
        /// Advance an EXTERNAL_LOOP iterator past the inner loop the kernel just consumed:
        /// one outer-odometer step (ripple carry over axes NDim-2..0). This is the per-chunk
        /// cost every strided-row op pays, so the state's array pointers are hoisted into
        /// locals — through <c>ref state</c> the JIT must assume each <c>DataPtrs[op]</c>
        /// store may alias the pointer FIELDS and reload them per operand per axis — and the
        /// kernel drivers (<see cref="ForEach"/>, <c>ExecuteGenericMulti</c>,
        /// <see cref="ExecuteReducing{TKernel, TAccum}"/>) call it DIRECTLY rather than
        /// through the cached <see cref="NDIterNextFunc"/> delegate, letting it inline into
        /// the driver loop. It stays a valid delegate target for <see cref="GetIterNext"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static bool ExternalLoopNext(ref NDIterState state)
        {
            // For external loop, we advance outer dimensions
            // Inner dimension is handled by caller
            if (state.IterIndex >= state.IterEnd)
                return false;

            int ndim = state.NDim;
            long* shape = state.Shape;

            state.IterIndex += shape[ndim - 1];

            if (state.IterIndex >= state.IterEnd)
                return false;

            int nop = state.NOp;
            int stridesNDim = state.StridesNDim;
            long* strides = state.Strides;
            long* dataPtrs = state.DataPtrs;
            int* srcElementSizes = state.SrcElementSizes;
            long* coords = state.Coords;

            // Advance outer coordinates. DataPtrs traverse SOURCE-array memory,
            // so multiply element strides by the SOURCE element size (the buffer
            // dtype's ElementSizes diverges under a buffered cast — bug (b)).
            for (int axis = ndim - 2; axis >= 0; axis--)
            {
                long coord = ++coords[axis];
                long dim = shape[axis];

                if (coord < dim)
                {
                    // Update data pointers
                    for (int op = 0; op < nop; op++)
                    {
                        long stride = strides[op * stridesNDim + axis];
                        dataPtrs[op] += stride * srcElementSizes[op];
                    }
                    return true;
                }

                // Carry
                coords[axis] = 0;
                long back = dim - 1;
                for (int op = 0; op < nop; op++)
                {
                    long stride = strides[op * stridesNDim + axis];
                    dataPtrs[op] -= stride * back * srcElementSizes[op];
                }
            }

            return true;
        }

        /// <summary>
        /// True when the iterator's advance function would resolve to
        /// <see cref="ExternalLoopNext"/> (see <see cref="GetIterNext"/>): EXTERNAL_LOOP, not
        /// a single iteration, and not windowed-buffered. The kernel drivers use it to call
        /// the advancer directly instead of through the delegate.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool IsPlainExternalLoopAdvance()
        {
            uint f = _state->ItFlags;
            if ((f & (uint)NDIterFlags.EXLOOP) == 0)
                return false;
            if ((f & (uint)NDIterFlags.ONEITERATION) != 0)
                return false;
            if ((f & (uint)NDIterFlags.BUFFER) != 0 && (f & (uint)NDIterFlags.REDUCE) == 0)
                return false;
            return true;
        }

        private static bool StandardNext(ref NDIterState state)
        {
            if (state.IterIndex >= state.IterEnd)
                return false;

            state.Advance();
            return state.IterIndex < state.IterEnd;
        }

        /// <summary>
        /// Get array of current data pointers.
        /// </summary>
        public void** GetDataPtrArray()
        {
            return (void**)Unsafe.AsPointer(ref _state->DataPtrs[0]);
        }

        /// <summary>
        /// Get inner loop stride array.
        /// </summary>
        public long* GetInnerStrideArray()
        {
            // For each operand, return the stride for the innermost dimension
            // These are stored at offset [op * StridesNDim + (NDim - 1)]
            return _state->GetInnerStrideArray();
        }

        /// <summary>
        /// Get pointer to inner loop size.
        /// </summary>
        public long* GetInnerLoopSizePtr()
        {
            if ((_state->ItFlags & (uint)NDIterFlags.BUFFER) != 0)
            {
                // Buffered REDUCE keeps its legacy double-loop semantics where
                // BufIterEnd doubles as the inner count; the windowed non-reduce
                // path exposes the fill size (NumPy's NBF_SIZE).
                if ((_state->ItFlags & (uint)NDIterFlags.REDUCE) != 0)
                    return &_state->BufIterEnd;
                return &_state->BufTransferSize;
            }

            // Return pointer to innermost shape dimension
            return &_state->Shape[_state->NDim - 1];
        }

        /// <summary>
        /// Reset iterator to the beginning. For windowed buffered iterators this
        /// flushes the pending window (NumPy flushes in NDIter_Reset too),
        /// materializes DELAY_BUFALLOC buffers, and primes the first window.
        /// </summary>
        public bool Reset()
        {
            uint f = _state->ItFlags;
            if ((f & (uint)NDIterFlags.BUFFER) != 0 && (f & (uint)NDIterFlags.REDUCE) == 0)
            {
                if ((f & (uint)NDIterFlags.DELAYBUF) != 0)
                {
                    if (!NDIterBufferManager.AllocateBuffers(ref *_state, _state->BufferSize))
                        return false;
                    _state->ItFlags &= ~(uint)NDIterFlags.DELAYBUF;
                }
                else
                {
                    // NpyIter_Reset: a window that already starts at IterStart and lies inside the range is left alone.
                    // Flushing and refilling it would not be a no-op: a write-only operand's buffer is never read from
                    // its array, so after a first reset it still holds the PREVIOUS window's data, and a second flush
                    // would write that data over the elements at the start of the range.
                    if (_state->IterIndex == _state->IterStart &&
                        _state->BufIterEnd <= _state->IterEnd &&
                        _state->BufTransferSize > 0)
                        return true;
                    NDIterBufferManager.FlushBufferWindow(ref *_state);
                }

                _state->Reset();
                NDIterBufferManager.FillBufferWindow(ref *_state,
                    NDIterBufferManager.ComputeTransferSize(ref *_state));
                return true;
            }

            _state->Reset();
            return true;
        }

        /// <summary>
        /// Materialize DELAY_BUFALLOC buffers and prime the first window. No-op
        /// unless BUFFER+DELAYBUF are pending. NumPy requires an explicit Reset
        /// before iterating a delay-allocated iterator; NumSharp auto-ensures at
        /// the execution entry points instead.
        /// </summary>
        public void EnsureBuffersReady()
        {
            uint f = _state->ItFlags;
            if ((f & (uint)NDIterFlags.BUFFER) == 0 || (f & (uint)NDIterFlags.DELAYBUF) == 0)
                return;

            NDIterBufferManager.AllocateBuffers(ref *_state, _state->BufferSize);
            _state->ItFlags &= ~(uint)NDIterFlags.DELAYBUF;
            NDIterBufferManager.FillBufferWindow(ref *_state,
                NDIterBufferManager.ComputeTransferSize(ref *_state));
        }

        /// <summary>
        /// Fetch the NDArrayMethodFlags (runtime) flags for all transfer functions
        /// (i.e. copy to buffer/casts). Matches NumPy's NDIter_GetTransferFlags
        /// (nditer_api.c:903). Decoded from the top 8 bits of ItFlags.
        ///
        /// In .NET context, REQUIRES_PYAPI is never set — included for API parity only.
        /// </summary>
        public NDArrayMethodFlags GetTransferFlags()
        {
            return (NDArrayMethodFlags)(_state->ItFlags >> NDIterConstants.TRANSFERFLAGS_SHIFT);
        }

        /// <summary>
        /// Dumps a verbose textual representation of the iterator's internal state to
        /// the specified TextWriter. Matches NumPy's NDIter_DebugPrint (nditer_api.c:1402)
        /// format as closely as possible.
        ///
        /// Output includes: ItFlags (decoded), NDim, NOp, IterSize/Start/End/Index,
        /// Perm, DTypes, DataPtrs, BaseOffsets, OpItFlags, BufferData, and per-axis data.
        /// </summary>
        public void DebugPrint(System.IO.TextWriter writer)
        {
            if (writer == null) throw new ArgumentNullException(nameof(writer));

            uint itf = _state->ItFlags;
            int ndim = _state->NDim;
            int nop = _state->NOp;

            writer.WriteLine();
            writer.WriteLine("------ BEGIN ITERATOR DUMP ------");
            writer.WriteLine($"| Iterator Address: 0x{(nuint)_state:X}");

            // Decode ItFlags
            writer.Write("| ItFlags: ");
            if ((itf & (uint)NDIterFlags.IDENTPERM) != 0) writer.Write("IDENTPERM ");
            if ((itf & (uint)NDIterFlags.NEGPERM) != 0) writer.Write("NEGPERM ");
            if ((itf & (uint)NDIterFlags.HASINDEX) != 0) writer.Write("HASINDEX ");
            if ((itf & (uint)NDIterFlags.HASMULTIINDEX) != 0) writer.Write("HASMULTIINDEX ");
            if ((itf & (uint)NDIterFlags.FORCEDORDER) != 0) writer.Write("FORCEDORDER ");
            if ((itf & (uint)NDIterFlags.EXLOOP) != 0) writer.Write("EXLOOP ");
            if ((itf & (uint)NDIterFlags.RANGE) != 0) writer.Write("RANGE ");
            if ((itf & (uint)NDIterFlags.BUFFER) != 0) writer.Write("BUFFER ");
            if ((itf & (uint)NDIterFlags.GROWINNER) != 0) writer.Write("GROWINNER ");
            if ((itf & (uint)NDIterFlags.ONEITERATION) != 0) writer.Write("ONEITERATION ");
            if ((itf & (uint)NDIterFlags.DELAYBUF) != 0) writer.Write("DELAYBUF ");
            if ((itf & (uint)NDIterFlags.REDUCE) != 0) writer.Write("REDUCE ");
            if ((itf & (uint)NDIterFlags.REUSE_REDUCE_LOOPS) != 0) writer.Write("REUSE_REDUCE_LOOPS ");
            writer.WriteLine();

            writer.WriteLine($"| NDim: {ndim}");
            writer.WriteLine($"| NOp: {nop}");
            if (_state->MaskOp >= 0) writer.WriteLine($"| MaskOp: {_state->MaskOp}");
            writer.WriteLine($"| IterSize: {_state->IterSize}");
            writer.WriteLine($"| IterStart: {_state->IterStart}");
            writer.WriteLine($"| IterEnd: {_state->IterEnd}");
            writer.WriteLine($"| IterIndex: {_state->IterIndex}");
            writer.WriteLine("|");

            // Perm array
            writer.Write("| Perm: ");
            for (int idim = 0; idim < ndim; idim++)
                writer.Write($"{_state->Perm[idim]} ");
            writer.WriteLine();

            // DTypes (per operand, NPTypeCode names since we don't have PyArray_Descr)
            writer.Write("| DTypes: ");
            for (int iop = 0; iop < nop; iop++)
            {
                var dt = _state->GetOpDType(iop);
                // A fixed-width string's lane names no dtype: print its descriptor (<U3, |S5) instead.
                writer.Write(dt == NPTypeCode.String ? $"{_state->GetOpDescr(iop).str} " : $"{dt.AsNumpyDtypeName()} ");
            }
            writer.WriteLine();

            // Initial data ptrs (reset ptrs)
            writer.Write("| InitDataPtrs: ");
            for (int iop = 0; iop < nop; iop++)
                writer.Write($"0x{_state->ResetDataPtrs[iop]:X} ");
            writer.WriteLine();

            // Base offsets
            writer.Write("| BaseOffsets: ");
            for (int iop = 0; iop < nop; iop++)
                writer.Write($"{_state->BaseOffsets[iop]} ");
            writer.WriteLine();

            // Current data pointers
            writer.Write("| Ptrs: ");
            for (int iop = 0; iop < nop; iop++)
                writer.Write($"0x{_state->DataPtrs[iop]:X} ");
            writer.WriteLine();

            if ((itf & (uint)NDIterFlags.HASINDEX) != 0)
                writer.WriteLine($"| FlatIndex: {_state->FlatIndex}");

            // OpItFlags
            writer.WriteLine("| OpItFlags:");
            for (int iop = 0; iop < nop; iop++)
            {
                writer.Write($"|   Flags[{iop}]: ");
                var of = _state->GetOpFlags(iop);
                if ((of & NDIterOpFlags.READ) != 0) writer.Write("READ ");
                if ((of & NDIterOpFlags.WRITE) != 0) writer.Write("WRITE ");
                if ((of & NDIterOpFlags.CAST) != 0) writer.Write("CAST ");
                if ((of & NDIterOpFlags.BUFNEVER) != 0) writer.Write("BUFNEVER ");
                if ((of & NDIterOpFlags.REDUCE) != 0) writer.Write("REDUCE ");
                if ((of & NDIterOpFlags.VIRTUAL) != 0) writer.Write("VIRTUAL ");
                if ((of & NDIterOpFlags.WRITEMASKED) != 0) writer.Write("WRITEMASKED ");
                if ((of & NDIterOpFlags.BUF_SINGLESTRIDE) != 0) writer.Write("BUF_SINGLESTRIDE ");
                if ((of & NDIterOpFlags.CONTIG) != 0) writer.Write("CONTIG ");
                if ((of & NDIterOpFlags.BUF_REUSABLE) != 0) writer.Write("BUF_REUSABLE ");
                writer.WriteLine();
            }
            writer.WriteLine("|");

            // Buffer data
            if ((itf & (uint)NDIterFlags.BUFFER) != 0)
            {
                writer.WriteLine("| BufferData:");
                writer.WriteLine($"|   BufferSize: {_state->BufferSize}");
                writer.WriteLine($"|   BufIterEnd: {_state->BufIterEnd}");
                writer.WriteLine($"|   CoreSize: {_state->CoreSize}");
                if ((itf & (uint)NDIterFlags.REDUCE) != 0)
                {
                    writer.WriteLine($"|   REDUCE Pos: {_state->ReducePos}");
                    writer.WriteLine($"|   REDUCE OuterSize: {_state->ReduceOuterSize}");
                    writer.WriteLine($"|   REDUCE OuterDim: {_state->OuterDim}");
                }
                writer.Write("|   BufStrides: ");
                for (int iop = 0; iop < nop; iop++)
                    writer.Write($"{_state->BufStrides[iop]} ");
                writer.WriteLine();
                if ((itf & (uint)NDIterFlags.REDUCE) != 0)
                {
                    writer.Write("|   REDUCE Outer Strides: ");
                    for (int iop = 0; iop < nop; iop++)
                        writer.Write($"{_state->ReduceOuterStrides[iop]} ");
                    writer.WriteLine();
                    writer.Write("|   REDUCE Outer Ptrs: ");
                    for (int iop = 0; iop < nop; iop++)
                        writer.Write($"0x{_state->ReduceOuterPtrs[iop]:X} ");
                    writer.WriteLine();
                }
                writer.Write("|   Buffers: ");
                for (int iop = 0; iop < nop; iop++)
                    writer.Write($"0x{_state->Buffers[iop]:X} ");
                writer.WriteLine();
                writer.WriteLine("|");
            }

            // Per-axis data
            for (int idim = 0; idim < ndim; idim++)
            {
                writer.WriteLine($"| AxisData[{idim}]:");
                writer.WriteLine($"|   Shape: {_state->Shape[idim]}");
                writer.WriteLine($"|   Index: {_state->Coords[idim]}");
                writer.Write("|   Strides: ");
                int stridesNDim = _state->StridesNDim;
                for (int iop = 0; iop < nop; iop++)
                    writer.Write($"{_state->Strides[iop * stridesNDim + idim]} ");
                writer.WriteLine();
            }

            writer.WriteLine("------- END ITERATOR DUMP -------");
            writer.Flush();
        }

        /// <summary>
        /// Dumps iterator state to standard output. See <see cref="DebugPrint(System.IO.TextWriter)"/>.
        /// </summary>
        public void DebugPrint()
        {
            DebugPrint(Console.Out);
        }

        /// <summary>
        /// Returns the debug dump as a string.
        /// </summary>
        public string DebugPrintToString()
        {
            using var sw = new System.IO.StringWriter();
            DebugPrint(sw);
            return sw.ToString();
        }

        /// <summary>
        /// Builds a set of strides that match the iterator's axis ordering for a
        /// hypothetical contiguous array (like the result of NPY_ITER_ALLOCATE).
        /// Matches NumPy's NDIter_CreateCompatibleStrides (nditer_api.c:1058).
        ///
        /// Use case: match the shape/layout of an iterator while tacking on extra
        /// dimensions (e.g., gradient vector per element, Hessian matrix).
        /// If an array is created with these strides, adding <paramref name="itemsize"/>
        /// each iteration traverses the array matching the iterator.
        ///
        /// Requirements:
        /// - Iterator must be tracking a multi-index (HASMULTIINDEX flag).
        /// - No axis may be flipped (NPY_ITER_DONT_NEGATE_STRIDES must have been used,
        ///   or the iterator must have no negative-stride axes to flip).
        /// </summary>
        /// <param name="itemsize">Base stride (typically element size in bytes).</param>
        /// <param name="outStrides">Output span of length ≥ NDim, one stride per axis
        ///                          in original array order (C-order).</param>
        public bool CreateCompatibleStrides(long itemsize, scoped Span<long> outStrides)
        {
            if ((_state->ItFlags & (uint)NDIterFlags.HASMULTIINDEX) == 0)
            {
                throw new InvalidOperationException(
                    "Iterator CreateCompatibleStrides may only be called if a multi-index is being tracked.");
            }

            if (outStrides.Length < _state->NDim)
                throw new ArgumentException(
                    $"outStrides must have at least {_state->NDim} elements.", nameof(outStrides));

            // Walk from innermost axis outward, accumulating itemsize.
            // NumSharp's innermost is at NDim-1 (opposite of NumPy's reversed storage
            // where idim=0 is innermost). So we iterate NDim-1 down to 0.
            for (int idim = _state->NDim - 1; idim >= 0; idim--)
            {
                int p = _state->Perm[idim];
                bool flipped = p < 0;
                int originalAxis;

                if (flipped)
                {
                    throw new InvalidOperationException(
                        "Iterator CreateCompatibleStrides may only be called if " +
                        "DONT_NEGATE_STRIDES was used to prevent reverse iteration of an axis.");
                }
                else
                {
                    originalAxis = p;
                }

                outStrides[originalAxis] = itemsize;
                itemsize *= _state->Shape[idim];
            }

            return true;
        }

        /// <summary>
        /// Gets the array of strides for the specified axis, one stride per operand.
        /// Matches NumPy's NDIter_GetAxisStrideArray (nditer_api.c:1309).
        ///
        /// If the iterator is tracking a multi-index, returns strides for the user-supplied
        /// axis in original-array coordinates (perm is walked to locate the internal axis).
        /// Otherwise returns strides for iteration axis <paramref name="axis"/> in Fortran
        /// order (fastest-changing axis first).
        ///
        /// Strides are returned in BYTES (multiplying NumSharp's internal element-count
        /// strides by the operand's element size) to match NumPy's byte-stride convention.
        /// </summary>
        /// <param name="axis">Axis index (0-based). With HASMULTIINDEX: original-array axis.
        ///                    Without: fastest-changing-first (Fortran) ordering.</param>
        /// <param name="outStrides">Output span of length ≥ NOp; filled with byte strides.</param>
        public void GetAxisStrideArray(int axis, scoped Span<long> outStrides)
        {
            if (axis < 0 || axis >= _state->NDim)
                throw new ArgumentOutOfRangeException(nameof(axis),
                    $"axis {axis} out of bounds for iterator with NDim={_state->NDim}");

            if (outStrides.Length < _state->NOp)
                throw new ArgumentException(
                    $"outStrides must have at least {_state->NOp} elements.", nameof(outStrides));

            int nop = _state->NOp;
            int stridesNDim = _state->StridesNDim;
            int internalIdim;

            if ((_state->ItFlags & (uint)NDIterFlags.HASMULTIINDEX) != 0)
            {
                // Walk perm to find the internal axis corresponding to the user's axis.
                // NumSharp's perm[idim] = original_axis (or -1-original if flipped).
                // (Unlike NumPy, NumSharp does NOT reverse axis storage, so no axis reversal
                // is needed on the input.)
                internalIdim = -1;
                for (int idim = 0; idim < _state->NDim; idim++)
                {
                    int p = _state->Perm[idim];
                    if (p == axis || -1 - p == axis)
                    {
                        internalIdim = idim;
                        break;
                    }
                }
                if (internalIdim < 0)
                    throw new InvalidOperationException("internal error in iterator perm");
            }
            else
            {
                // Non-MULTI_INDEX: axis is in Fortran order (fastest-first).
                // NumSharp's innermost axis is at NDim-1, so internal idim = NDim-1-axis.
                internalIdim = _state->NDim - 1 - axis;
            }

            // Return byte strides (NumPy convention). Internal strides are element
            // counts of the SOURCE arrays (NumPy's NAD_STRIDES are array byte
            // strides), so the multiplier is the SOURCE element size — ElementSizes
            // is the buffer dtype size and diverges under a buffered cast (bug (b)
            // family).
            for (int op = 0; op < nop; op++)
            {
                long elemStride = _state->Strides[op * stridesNDim + internalIdim];
                outStrides[op] = elemStride * _state->SrcElementSizes[op];
            }
        }

        /// <summary>
        /// Copies the array of strides that are fixed during iteration into <paramref name="outStrides"/>.
        /// Matches NumPy's NDIter_GetInnerFixedStrideArray (nditer_api.c:1357).
        ///
        /// Buffered iterators copy <see cref="NDIterState.BufStrides"/>. Non-buffered iterators
        /// copy the innermost-axis stride from <see cref="NDIterState.Strides"/> and convert it
        /// to bytes.
        /// </summary>
        /// <param name="outStrides">Output span of length at least NOp.</param>
        public void GetInnerFixedStrideArray(scoped Span<long> outStrides)
        {
            if (outStrides.Length < _state->NOp)
                throw new ArgumentException(
                    $"outStrides must have at least {_state->NOp} elements.", nameof(outStrides));

            int nop = _state->NOp;

            if ((_state->ItFlags & (uint)NDIterFlags.BUFFER) != 0)
            {
                // Buffered: BufStrides already stored in bytes (NDIterBufferManager assigns
                // BufStrides[op] = GetElementSize(op)).
                for (int op = 0; op < nop; op++)
                    outStrides[op] = _state->BufStrides[op];
            }
            else
            {
                // Non-buffered: innermost-axis stride for each operand, converted to BYTE units
                // to match NumPy (NumSharp internally stores element-count strides).
                if (_state->NDim == 0)
                {
                    for (int op = 0; op < nop; op++)
                        outStrides[op] = 0;
                }
                else
                {
                    int innermost = _state->NDim - 1;
                    int stridesNDim = _state->StridesNDim;
                    for (int op = 0; op < nop; op++)
                    {
                        long elemStride = _state->Strides[op * stridesNDim + innermost];
                        // Strides traverse source-array memory → source element
                        // size. (Equal to ElementSizes on this branch: casts
                        // require BUFFER, which took the BufStrides branch above.)
                        outStrides[op] = elemStride * _state->SrcElementSizes[op];
                    }
                }
            }
        }

        /// <summary>
        /// Resets the iterator to its initial state with new base data pointers.
        /// Matches NumPy's NDIter_ResetBasePointers (nditer_api.c:314).
        ///
        /// For each operand, sets resetdataptr[iop] = baseptrs[iop] + baseoffsets[iop],
        /// where baseoffsets is the cumulative byte offset recorded by FlipNegativeStrides.
        /// Then repositions the iterator to IterStart.
        ///
        /// The new arrays pointed to by baseptrs MUST have the exact same shape, dtype,
        /// and memory layout as the original operands. This is typically used in nested
        /// iteration (ufunc-style) where one iterator feeds data pointers to another.
        ///
        /// Throws ArgumentException if baseptrs.Length != NOp.
        /// </summary>
        /// <param name="baseptrs">Array of new base data pointers, one per operand.</param>
        /// <returns>True on success.</returns>
        public bool ResetBasePointers(scoped ReadOnlySpan<IntPtr> baseptrs)
        {
            if (baseptrs.Length != _state->NOp)
            {
                throw new ArgumentException(
                    $"baseptrs length {baseptrs.Length} does not match operand count {_state->NOp}.",
                    nameof(baseptrs));
            }

            uint itFlags = _state->ItFlags;

            // If buffering, handle pending buffer state first
            if ((itFlags & (uint)NDIterFlags.BUFFER) != 0)
            {
                if ((itFlags & (uint)NDIterFlags.DELAYBUF) != 0)
                {
                    // Delayed buffer allocation: allocate now
                    if (!NDIterBufferManager.AllocateBuffers(ref *_state, _state->BufferSize))
                    {
                        return false;
                    }
                    _state->ItFlags &= ~(uint)NDIterFlags.DELAYBUF;
                }
                else if ((itFlags & (uint)NDIterFlags.REDUCE) != 0)
                {
                    // Flush any pending writes before replacing pointers
                    CopyReduceBuffersToArrays();
                }
                else
                {
                    // Windowed path: flush the pending window before replacing pointers
                    NDIterBufferManager.FlushBufferWindow(ref *_state);
                }
            }

            // Install new reset pointers: resetdataptr[iop] = baseptrs[iop] + baseoffsets[iop].
            // NumPy nditer_api.c:343-345.
            for (int iop = 0; iop < _state->NOp; iop++)
            {
                _state->ResetDataPtrs[iop] = (long)baseptrs[iop] + _state->BaseOffsets[iop];
            }

            // Reposition to IterStart using the new base pointers.
            _state->GotoIterIndex(_state->IterStart);

            // Re-prime buffers if buffered
            if ((itFlags & (uint)NDIterFlags.BUFFER) != 0)
            {
                if ((itFlags & (uint)NDIterFlags.REDUCE) == 0)
                {
                    // Windowed path: prime a fresh window at the new position.
                    NDIterBufferManager.FillBufferWindow(ref *_state,
                        NDIterBufferManager.ComputeTransferSize(ref *_state));
                }
                else
                {
                    long remaining = _state->IterEnd - _state->IterIndex;
                    long copyCount = Math.Min(remaining, _state->BufferSize);
                    if (copyCount > 0)
                    {
                        for (int iop = 0; iop < _state->NOp; iop++)
                        {
                            var opFlags = _state->GetOpFlags(iop);
                            if ((opFlags & NDIterOpFlags.READ) != 0)
                            {
                                NDIterBufferManager.CopyToBuffer(ref *_state, iop, copyCount);
                            }
                        }
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Convenience overload: resets base pointers using the data pointers of new NDArray operands.
        /// The new arrays must have the same shape, dtype, and layout as the original operands.
        /// </summary>
        public unsafe bool ResetBasePointers(NDArray[] newOperands)
        {
            if (newOperands == null)
                throw new ArgumentNullException(nameof(newOperands));
            if (newOperands.Length != _state->NOp)
            {
                throw new ArgumentException(
                    $"newOperands length {newOperands.Length} does not match operand count {_state->NOp}.",
                    nameof(newOperands));
            }

            Span<IntPtr> baseptrs = stackalloc IntPtr[newOperands.Length];
            for (int i = 0; i < newOperands.Length; i++)
            {
                var arr = newOperands[i];
                if (arr is null)
                    throw new ArgumentException($"newOperands[{i}] is null.");
                // ElementBytes — not arr.dtypesize — because the latter uses
                // Marshal.SizeOf(bool) == 4 while in-memory bool storage is 1
                // byte per element; it also sizes a fixed-width string element.
                int elemBytes = ElementBytes(arr);
                byte* basePtr = (byte*)arr.Address + (arr.Shape.offset * elemBytes);
                baseptrs[i] = (IntPtr)basePtr;
            }

            return ResetBasePointers(baseptrs);
        }

        /// <summary>
        /// Advance to next position and return whether more iterations remain.
        /// Matches NumPy's iternext() behavior.
        /// Returns true if more elements exist, false when iteration is complete.
        ///
        /// When BUFFERED + REDUCE flags are set, uses the double-loop pattern
        /// from NumPy's npyiter_buffered_reduce_iternext (nditer_templ.c.src lines 131-210).
        /// </summary>
        public bool Iternext()
        {
            if (_state->IterIndex >= _state->IterEnd)
                return false;

            // Check for buffered reduce path
            // Use double-loop for any buffered reduction (even when ReduceOuterSize = 1)
            // because we need to use BufStrides which has 0 for reduce operands
            uint itFlags = _state->ItFlags;
            if ((itFlags & (uint)NDIterFlags.BUFFER) != 0 &&
                (itFlags & (uint)NDIterFlags.REDUCE) != 0 &&
                _state->CoreSize > 0)
            {
                return BufferedReduceIternext();
            }

            // Respect EXLOOP / ONEITERATION. The kernel processes the innermost
            // loop, so a plain one-element Advance() over-steps an external-loop
            // iterator by NDim-1 positions per call (reading past the inner-loop
            // buffer). GetIterNext() returns the correct advancer — ExternalLoopNext
            // for EXLOOP, SingleIterationNext for ONEITERATION, StandardNext
            // otherwise. StandardNext is byte-for-byte the old Advance() path for
            // the common (non-EXLOOP, non-ONEITERATION) case, so this is a strict
            // correction with no behavior change there.
            return GetIterNext()(ref *_state);
        }

        /// <summary>
        /// Buffered reduce iteration using NumPy's double-loop pattern.
        /// Avoids re-buffering during reduction by separating iteration into:
        /// - Inner loop: CoreSize elements
        /// - Outer loop: ReduceOuterSize iterations
        /// </summary>
        private bool BufferedReduceIternext()
        {
            int result = _state->BufferedReduceAdvance();

            if (result == 1)
            {
                // More elements in current buffer
                return true;
            }

            if (result == -1)
            {
                // Iteration complete - write back remaining buffer contents
                CopyReduceBuffersToArrays();
                return false;
            }

            // result == 0: Buffer exhausted, need to refill

            // Write back reduce buffers to arrays
            CopyReduceBuffersToArrays();

            // Check if we're past the end
            if (_state->IterIndex >= _state->IterEnd)
            {
                return false;
            }

            // Move to next buffer position - this updates DataPtrs to current array positions
            _state->GotoIterIndex(_state->IterIndex);

            // Calculate how much to copy for next buffer
            long remaining = _state->IterEnd - _state->IterIndex;
            long copyCount = Math.Min(remaining, _state->BufferSize);

            // Copy to buffers
            // For reduce operands, check if we're at a NEW output position
            // (i.e., the reduce operand's array pointer changed from the previous writeback position)
            for (int op = 0; op < _state->NOp; op++)
            {
                var opFlags = _state->GetOpFlags(op);

                // For reduce operands, only reload if at a new output position
                if ((opFlags & NDIterOpFlags.REDUCE) != 0)
                {
                    void* currentArrayPos = _state->GetDataPtr(op);
                    void* previousWritebackPos = _state->GetArrayWritebackPtr(op);

                    // If pointer changed, we're at a new output position - reload
                    // If same, we're continuing the same output - skip reload
                    if (currentArrayPos == previousWritebackPos)
                    {
                        continue;  // Same output position, keep accumulating
                    }
                }

                if ((opFlags & NDIterOpFlags.READ) != 0 || (opFlags & NDIterOpFlags.READWRITE) != 0)
                {
                    NDIterBufferManager.CopyToBuffer(ref *_state, op, copyCount);
                }
            }

            // Save current array positions for writeback (after checking but before buffer overwrite)
            // These are the positions where CopyReduceBuffersToArrays will write
            for (int op = 0; op < _state->NOp; op++)
            {
                _state->SetArrayWritebackPtr(op, _state->GetDataPtr(op));
            }

            // Reset DataPtrs to point to buffer start (BufferedReduceAdvance uses these)
            for (int op = 0; op < _state->NOp; op++)
            {
                var buffer = _state->GetBuffer(op);
                if (buffer != null)
                {
                    _state->SetDataPtr(op, buffer);
                }
            }

            // For small buffer handling, set ReduceOuterSize based on buffer capacity
            _state->ReduceOuterSize = 1;
            _state->ReducePos = 0;
            _state->CorePos = 0;

            // Set buffer iteration end
            _state->BufIterEnd = _state->IterIndex + copyCount;

            // Initialize reduce outer pointers (pointing to buffer start)
            _state->InitReduceOuterPtrs();

            return true;
        }

        /// <summary>
        /// Copy reduce buffers back to original arrays.
        /// For reduce operands, only copies CoreSize elements (the accumulated results).
        /// For non-reduce operands, copies CoreSize * ReduceOuterSize elements.
        /// Uses ArrayWritebackPtrs (saved during buffer fill) as destination.
        /// </summary>
        private void CopyReduceBuffersToArrays()
        {
            for (int op = 0; op < _state->NOp; op++)
            {
                var opFlags = _state->GetOpFlags(op);

                // Only copy WRITE or READWRITE operands
                if ((opFlags & NDIterOpFlags.WRITE) == 0 && (opFlags & NDIterOpFlags.READWRITE) == 0)
                    continue;

                var buffer = _state->GetBuffer(op);
                if (buffer == null)
                    continue;

                // The legacy reduce write-back predates masked copy-back: it
                // would write the buffer's unmasked slots too, silently
                // violating WRITEMASKED. Refuse at the exact corruption point
                // (construction succeeds, NumPy parity). Masked reduce
                // write-back lands with reductions-through-core (roadmap Wave 5);
                // the windowed non-reduce path already enforces masks
                // (NDIterBufferManager.FlushBufferWindow).
                if ((opFlags & NDIterOpFlags.WRITEMASKED) != 0)
                {
                    throw new NotSupportedException(
                        "WRITEMASKED write-back is not supported on a buffered REDUCE " +
                        "iterator yet (legacy buffered-reduce machinery; roadmap Wave 5). " +
                        "Use an unbuffered iterator (mask enforcement is then the " +
                        "kernel's responsibility, matching NumPy) or drop REDUCE.");
                }

                // Get array writeback pointer (saved at buffer start)
                // Falls back to ResetDataPtr if ArrayWritebackPtr not set
                void* dst = _state->GetArrayWritebackPtr(op);
                if (dst == null)
                    dst = _state->GetResetDataPtr(op);
                if (dst == null)
                    continue;

                int elemSize = _state->GetElementSize(op);

                // For reduce operands, buffer has ReduceOuterSize unique output positions
                // For non-reduce operands, buffer has full CoreSize * ReduceOuterSize elements
                long copyCount;
                if ((opFlags & NDIterOpFlags.REDUCE) != 0)
                {
                    // Reduce operand: ReduceOuterSize unique output positions
                    // (each position accumulated CoreSize inputs)
                    copyCount = _state->ReduceOuterSize;
                }
                else
                {
                    // Non-reduce operand: full buffer contents
                    copyCount = _state->CoreSize * _state->ReduceOuterSize;
                }

                // For reduce operands, we have stride=0 in the reduce dimension
                // which means all output goes to the same position(s)
                // Just copy CoreSize elements from buffer to array
                if ((opFlags & NDIterOpFlags.REDUCE) != 0)
                {
                    // Simple copy - buffer[0:CoreSize] to dst[0:CoreSize]
                    Buffer.MemoryCopy(buffer, dst, copyCount * elemSize, copyCount * elemSize);
                }
                else
                {
                    // Non-reduce: need strided copy (handled by existing logic)
                    // Temporarily set DataPtr to array position for CopyFromBuffer
                    void* savedDataPtr = _state->GetDataPtr(op);
                    _state->SetDataPtr(op, dst);
                    NDIterBufferManager.CopyFromBuffer(ref *_state, op, copyCount);
                    _state->SetDataPtr(op, savedDataPtr);
                }
            }
        }

        /// <summary>
        /// Reset iterator to a specific iteration range.
        /// Enables ranged iteration for parallel chunking.
        /// </summary>
        /// <param name="start">Start index (inclusive)</param>
        /// <param name="end">End index (exclusive)</param>
        /// <returns>True if range is valid (and the reset succeeded), false otherwise</returns>
        /// <remarks>
        ///     Ends with <see cref="Reset"/>, as NumPy's <c>NpyIter_ResetToIterIndexRange</c> ends with <c>NpyIter_Reset</c>:
        ///     the position moves to <paramref name="start"/>, a buffered window is written back and primed for the new
        ///     range, and delayed buffers are allocated (false when that allocation fails).
        /// </remarks>
        public bool ResetToIterIndexRange(long start, long end)
        {
            if (start < 0 || end > _state->IterSize || start > end)
                return false;

            _state->IterStart = start;
            _state->IterEnd = end;
            _state->ItFlags |= (uint)NDIterFlags.RANGE;

            // NpyIter_ResetToIterIndexRange ends with NpyIter_Reset: a buffered window is written back and primed again
            // for the NEW range (a seek within the old window would keep a window that runs past the new end), and
            // delayed buffers are allocated.
            return Reset();
        }

        /// <summary>
        /// Get the current iteration range start.
        /// </summary>
        public long IterStart => _state->IterStart;

        /// <summary>
        /// Get the current iteration range end.
        /// </summary>
        public long IterEnd => _state->IterEnd;

        /// <summary>
        /// Check if iterator is using ranged iteration.
        /// </summary>
        public bool IsRanged => (_state->ItFlags & (uint)NDIterFlags.RANGE) != 0;

        /// <summary>
        /// Jump to a specific iteration index.
        /// </summary>
        public void GotoIterIndex(long iterindex)
        {
            // A windowed buffered iterator holds the current window in its buffers (NumPy's NpyIter_GotoIterIndex):
            //  - a position INSIDE the current window only moves the pointers by the buffer strides — the coordinates go
            //    on describing the window's start, which is why a buffered multi_index reads the window start's
            //    coordinates after such a seek, as NumPy's does;
            //  - any other position flushes the window (written elements reach the array) and primes a new one at the
            //    destination. Moving only the state's pointers there would leave them on ARRAY memory that the views read
            //    as buffer elements (a buffered cast, or a string widened in its buffer, then read garbage).
            // Delayed buffers have no window yet: the state moves alone.
            uint f = _state->ItFlags;
            bool windowed = (f & (uint)NDIterFlags.BUFFER) != 0 &&
                            (f & (uint)(NDIterFlags.REDUCE | NDIterFlags.DELAYBUF)) == 0;
            if (windowed)
            {
                long size = _state->BufTransferSize;
                long bufIterEnd = _state->BufIterEnd;
                if (iterindex < bufIterEnd && iterindex >= bufIterEnd - size)
                {
                    long delta = iterindex - _state->IterIndex;
                    for (int op = 0; op < _state->NOp; op++)
                        _state->DataPtrs[op] += delta * _state->BufStrides[op];
                    _state->IterIndex = iterindex;
                    return;
                }

                NDIterBufferManager.FlushBufferWindow(ref *_state);
            }

            _state->GotoIterIndex(iterindex);
            if (windowed)
                NDIterBufferManager.FillBufferWindow(ref *_state, NDIterBufferManager.ComputeTransferSize(ref *_state));
        }

        /// <summary>
        /// Returns a specialized delegate for computing multi-index based on iterator flags.
        /// Matches NumPy's NDIter_GetGetMultiIndex (nditer_templ.c.src:481).
        ///
        /// NumPy generates 12 specializations on (HASINDEX × IDENTPERM × NEGPERM × BUFFER).
        /// NumSharp dispatches to 3 variants (BUFFER and HASINDEX don't affect coords):
        ///   1. IDENTPERM — direct copy of internal coords
        ///   2. Positive perm — apply perm[] mapping
        ///   3. NEGPERM — apply perm[] with flip decoding
        ///
        /// The returned delegate takes raw NDIterState and a pointer to output coords.
        /// </summary>
        /// <param name="errmsg">Set on failure; null on success.</param>
        /// <returns>Delegate, or null if iterator is not tracking multi-index.</returns>
        public NDIterGetMultiIndexFunc? GetMultiIndexFunc(out string? errmsg)
        {
            errmsg = null;
            if ((_state->ItFlags & (uint)NDIterFlags.HASMULTIINDEX) == 0)
            {
                errmsg = "Iterator not tracking multi-index. Use NDIterGlobalFlags.MULTI_INDEX during construction.";
                return null;
            }

            uint itf = _state->ItFlags;
            if ((itf & (uint)NDIterFlags.IDENTPERM) != 0)
                return GetMultiIndex_Identity;
            if ((itf & (uint)NDIterFlags.NEGPERM) != 0)
                return GetMultiIndex_NegPerm;
            return GetMultiIndex_PosPerm;
        }

        /// <summary>
        /// Returns a specialized delegate for computing multi-index.
        /// Matches NumPy's NDIter_GetGetMultiIndex. Throws on failure instead of
        /// returning null (thin wrapper over the out-errmsg overload).
        /// </summary>
        public NDIterGetMultiIndexFunc GetMultiIndexFunc()
        {
            var fn = GetMultiIndexFunc(out string? errmsg);
            if (fn == null) throw new InvalidOperationException(errmsg ?? "GetMultiIndexFunc unavailable");
            return fn;
        }

        /// <summary>
        /// Invokes the specialized multi-index delegate with this iterator's internal state.
        /// This mirrors NumPy's pattern: <c>fn(iter, outcoords)</c>, where NumSharp's iterator
        /// handle is a ref struct and the state is held internally.
        /// </summary>
        public void InvokeMultiIndex(NDIterGetMultiIndexFunc fn, long* outCoords)
        {
            if (fn == null) throw new ArgumentNullException(nameof(fn));
            fn(ref *_state, outCoords);
        }

        /// <summary>
        /// Span overload of <see cref="InvokeMultiIndex"/>.
        /// </summary>
        public void InvokeMultiIndex(NDIterGetMultiIndexFunc fn, scoped Span<long> outCoords)
        {
            if (fn == null) throw new ArgumentNullException(nameof(fn));
            if (outCoords.Length < _state->NDim)
                throw new ArgumentException($"outCoords must have at least {_state->NDim} elements.", nameof(outCoords));
            fixed (long* ptr = outCoords)
            {
                fn(ref *_state, ptr);
            }
        }

        // Specialized implementations — matches NumPy's 3 structural patterns
        // (HASINDEX and BUFFER don't affect coord output so they're not specialized).

        private static void GetMultiIndex_Identity(ref NDIterState state, long* outCoords)
        {
            for (int d = 0; d < state.NDim; d++)
                outCoords[d] = state.Coords[d];
        }

        private static void GetMultiIndex_PosPerm(ref NDIterState state, long* outCoords)
        {
            for (int d = 0; d < state.NDim; d++)
            {
                int p = state.Perm[d];
                outCoords[p] = state.Coords[d];
            }
        }

        private static void GetMultiIndex_NegPerm(ref NDIterState state, long* outCoords)
        {
            for (int d = 0; d < state.NDim; d++)
            {
                int p = state.Perm[d];
                if (p < 0)
                {
                    int originalAxis = -1 - p;
                    outCoords[originalAxis] = state.Shape[d] - state.Coords[d] - 1;
                }
                else
                {
                    outCoords[p] = state.Coords[d];
                }
            }
        }

        /// <summary>
        /// Get the current multi-index (coordinates) in original axis order.
        /// Uses the Perm array to map internal coordinates to original array coordinates.
        /// When NEGPERM is set, flipped axes have negative perm entries and their
        /// coordinates are reversed (shape - coord - 1).
        /// Requires MULTI_INDEX flag to be set during construction.
        /// </summary>
        public void GetMultiIndex(scoped Span<long> outCoords)
        {
            if ((_state->ItFlags & (uint)NDIterFlags.HASMULTIINDEX) == 0)
                throw new InvalidOperationException("Iterator not tracking multi-index. Use NDIterGlobalFlags.MULTI_INDEX during construction.");

            if (outCoords.Length < _state->NDim)
                throw new ArgumentException($"Output span must have at least {_state->NDim} elements", nameof(outCoords));

            // Fast path: IDENTPERM means perm is identity (no reordering or flipping)
            if ((_state->ItFlags & (uint)NDIterFlags.IDENTPERM) != 0)
            {
                for (int d = 0; d < _state->NDim; d++)
                    outCoords[d] = _state->Coords[d];
                return;
            }

            // Apply permutation: Perm[internal_axis] = original_axis (or -1-original if flipped)
            // When perm[d] >= 0: outCoords[perm[d]] = Coords[d]
            // When perm[d] < 0: original = -1 - perm[d], and coordinate is flipped
            bool hasNegPerm = (_state->ItFlags & (uint)NDIterFlags.NEGPERM) != 0;

            for (int d = 0; d < _state->NDim; d++)
            {
                int p = _state->Perm[d];
                if (hasNegPerm && p < 0)
                {
                    // Flipped axis: original = -1 - p, coordinate = shape - coord - 1
                    int originalAxis = -1 - p;
                    outCoords[originalAxis] = _state->Shape[d] - _state->Coords[d] - 1;
                }
                else
                {
                    outCoords[p] = _state->Coords[d];
                }
            }
        }

        /// <summary>
        /// Jump to a specific multi-index (coordinates) given in original axis order.
        /// Uses the Perm array to map original coordinates to internal iteration order.
        /// When NEGPERM is set, flipped axes have negative perm entries and their
        /// coordinates are reversed when mapping to internal coordinates.
        /// Requires MULTI_INDEX flag to be set during construction.
        /// </summary>
        public void GotoMultiIndex(scoped ReadOnlySpan<long> coords)
        {
            if ((_state->ItFlags & (uint)NDIterFlags.HASMULTIINDEX) == 0)
                throw new InvalidOperationException("Iterator not tracking multi-index. Use NDIterGlobalFlags.MULTI_INDEX during construction.");

            if (coords.Length < _state->NDim)
                throw new ArgumentException($"Coordinates must have at least {_state->NDim} elements", nameof(coords));

            bool hasNegPerm = (_state->ItFlags & (uint)NDIterFlags.NEGPERM) != 0;

            // Apply permutation: Perm[internal_axis] = original_axis (or -1-original if flipped)
            // When perm[d] >= 0: Coords[d] = coords[perm[d]]
            // When perm[d] < 0: original = -1 - perm[d], Coords[d] = shape[d] - coords[original] - 1
            long iterIndex = 0;
            long multiplier = 1;

            for (int d = _state->NDim - 1; d >= 0; d--)
            {
                int p = _state->Perm[d];
                int originalAxis;
                long coord;

                if (hasNegPerm && p < 0)
                {
                    // Flipped axis: map original coord to internal (flipped)
                    originalAxis = -1 - p;
                    coord = _state->Shape[d] - coords[originalAxis] - 1;
                }
                else
                {
                    originalAxis = p;
                    coord = coords[originalAxis];
                }

                if (coord < 0 || coord >= _state->Shape[d])
                    throw new IndexOutOfRangeException($"Coordinate {coords[originalAxis]} out of range for original axis {originalAxis} (size {_state->Shape[d]})");

                _state->Coords[d] = coord;
                iterIndex += coord * multiplier;
                multiplier *= _state->Shape[d];
            }

            _state->IterIndex = iterIndex;

            // A tracked flat index follows the new coordinates through its index strides (the closed form NumPy's
            // npyiter_goto_iterindex evaluates for the index "operand").
            if ((_state->ItFlags & (uint)NDIterFlags.HASINDEX) != 0)
                _state->FlatIndex = _state->ComputeFlatIndex();

            // Update data pointers using internal coordinates. ResetDataPtrs are
            // source-array bases — multiply by the SOURCE element size (bug (b):
            // ElementSizes is the buffer dtype size under a buffered cast).
            for (int op = 0; op < _state->NOp; op++)
            {
                long offset = 0;
                for (int d = 0; d < _state->NDim; d++)
                    offset += _state->Coords[d] * _state->GetStride(d, op);

                _state->DataPtrs[op] = _state->ResetDataPtrs[op] + offset * _state->SrcElementSizes[op];
            }
        }

        /// <summary>
        /// Check if iterator is tracking multi-index.
        /// </summary>
        public bool HasMultiIndex => (_state->ItFlags & (uint)NDIterFlags.HASMULTIINDEX) != 0;

        /// <summary>
        /// Check if iterator is tracking a flat index.
        /// </summary>
        public bool HasIndex => (_state->ItFlags & (uint)NDIterFlags.HASINDEX) != 0;

        /// <summary>
        /// Check if any axes have negative permutation entries (flipped for memory-order iteration).
        /// When NEGPERM is set, GetMultiIndex reverses indices for those axes.
        /// </summary>
        public bool HasNegPerm => (_state->ItFlags & (uint)NDIterFlags.NEGPERM) != 0;

        /// <summary>
        /// Check if the axis permutation is identity (no reordering).
        /// Mutually exclusive with NEGPERM - if NEGPERM is set, IDENTPERM is cleared.
        /// </summary>
        public bool HasIdentPerm => (_state->ItFlags & (uint)NDIterFlags.IDENTPERM) != 0;

        /// <summary>
        /// Check if iteration is finished.
        /// </summary>
        public bool Finished => _state->IterIndex >= _state->IterEnd;

        /// <summary>
        /// Get the current iterator shape in original axis order.
        /// When MULTI_INDEX is set, returns shape in original axis order.
        /// When NEGPERM is set, handles flipped axes correctly.
        /// Otherwise returns internal (possibly coalesced) shape.
        /// </summary>
        public long[] Shape
        {
            get
            {
                var result = new long[_state->NDim];

                if ((_state->ItFlags & (uint)NDIterFlags.HASMULTIINDEX) != 0)
                {
                    bool hasNegPerm = (_state->ItFlags & (uint)NDIterFlags.NEGPERM) != 0;

                    // Return shape in original axis order
                    for (int d = 0; d < _state->NDim; d++)
                    {
                        int p = _state->Perm[d];
                        int origAxis = (hasNegPerm && p < 0) ? (-1 - p) : p;
                        result[origAxis] = _state->Shape[d];
                    }
                }
                else
                {
                    // Return internal (coalesced) shape
                    for (int d = 0; d < _state->NDim; d++)
                        result[d] = _state->Shape[d];
                }
                return result;
            }
        }

        /// <summary>
        /// Get the current iteration range as (start, end) tuple.
        /// </summary>
        public (long Start, long End) IterRange => (_state->IterStart, _state->IterEnd);

        /// <summary>
        /// Get the current flat index.
        /// Requires C_INDEX or F_INDEX flag to be set during construction.
        /// </summary>
        public long GetIndex()
        {
            if ((_state->ItFlags & (uint)NDIterFlags.HASINDEX) == 0)
                throw new InvalidOperationException("Iterator not tracking index. Use NDIterGlobalFlags.C_INDEX or F_INDEX during construction.");

            return _state->FlatIndex;
        }

        /// <summary>
        ///     The iteration index the ORIGINAL coordinates <paramref name="coords"/> name, without moving — the position
        ///     <see cref="GotoMultiIndex"/> would go to. NumPy's <c>NpyIter_GotoMultiIndex</c> computes it first, so it can
        ///     refuse an out-of-bounds coordinate or a position outside the iteration range before anything changes.
        /// </summary>
        /// <param name="coords">One coordinate per original axis (at least <see cref="NDim"/>).</param>
        /// <returns>The iteration index, or -1 when a coordinate is outside its axis.</returns>
        internal long IterIndexOfMultiIndex(scoped ReadOnlySpan<long> coords)
        {
            bool hasNegPerm = (_state->ItFlags & (uint)NDIterFlags.NEGPERM) != 0;
            long iterIndex = 0;
            long multiplier = 1;
            for (int d = _state->NDim - 1; d >= 0; d--)
            {
                int p = _state->Perm[d];
                // A flipped axis walks its coordinates backward (Perm holds -1 - original for it).
                long coord = hasNegPerm && p < 0 ? _state->Shape[d] - coords[-1 - p] - 1 : coords[p];
                if (coord < 0 || coord >= _state->Shape[d])
                    return -1;
                iterIndex += coord * multiplier;
                multiplier *= _state->Shape[d];
            }

            return iterIndex;
        }

        /// <summary>
        ///     The iteration index of flat index <paramref name="flatIndex"/> (C or F order of the operands' own shape, per
        ///     the construction's <c>c_index</c> / <c>f_index</c>), without moving — the position <see cref="GotoIndex"/>
        ///     would go to, which NumPy's <c>NpyIter_GotoIndex</c> checks against the iteration range first.
        /// </summary>
        /// <param name="flatIndex">A flat index in <c>[0, IterSize)</c>.</param>
        /// <returns>The iteration index.</returns>
        /// <remarks>
        ///     Read off the index strides (<see cref="NDIterState.IterIndexOfIndex"/>), so it stays right after the axes
        ///     coalesced; decomposing the index into coordinates of the operands' shape needs every original axis to be its
        ///     own iteration axis.
        /// </remarks>
        internal long IterIndexOfIndex(long flatIndex) => _state->IterIndexOfIndex(flatIndex);

        /// <summary>
        /// Jump to a specific flat index position (C or F order based on construction flags).
        /// Requires C_INDEX or F_INDEX flag to be set during construction.
        /// Matches NumPy's NDIter_GotoIndex behavior: the position is read off the index strides, so flipped and
        /// coalesced axes need no special case.
        /// </summary>
        /// <param name="flatIndex">The flat index in C or F order (depending on flags)</param>
        /// <exception cref="InvalidOperationException">No flat index is tracked.</exception>
        /// <exception cref="IndexOutOfRangeException"><paramref name="flatIndex"/> is outside <c>[0, IterSize)</c>.</exception>
        public void GotoIndex(long flatIndex)
        {
            if ((_state->ItFlags & (uint)NDIterFlags.HASINDEX) == 0)
                throw new InvalidOperationException("Iterator not tracking index. Use NDIterGlobalFlags.C_INDEX or F_INDEX during construction.");

            if (flatIndex < 0 || flatIndex >= _state->IterSize)
                throw new IndexOutOfRangeException($"Flat index {flatIndex} out of range [0, {_state->IterSize})");

            // NpyIter_GotoIndex: the iteration index the flat index names, read off the index strides (correct on coalesced
            // and flipped axes alike), then the coordinates of that position.
            long iterIndex = _state->IterIndexOfIndex(flatIndex);
            long remaining = iterIndex;
            for (int d = _state->NDim - 1; d >= 0; d--)
            {
                long extent = _state->Shape[d];
                _state->Coords[d] = remaining % extent;
                remaining /= extent;
            }

            _state->IterIndex = iterIndex;
            _state->FlatIndex = flatIndex;

            // Update data pointers (source element size — see GotoIterIndex / bug (b))
            for (int op = 0; op < _state->NOp; op++)
            {
                long offset = 0;
                for (int d = 0; d < _state->NDim; d++)
                    offset += _state->Coords[d] * _state->GetStride(d, op);

                _state->DataPtrs[op] = _state->ResetDataPtrs[op] + offset * _state->SrcElementSizes[op];
            }
        }

        /// <summary>
        /// Get operand arrays.
        /// </summary>
        public NDArray[]? GetOperandArray() => _operands;

        /// <summary>
        /// Returns a view of the i-th operand with the iterator's internal axes ordering.
        /// A C-order iteration of this view is equivalent to the iterator's iteration order.
        ///
        /// For example, if a 3D array was coalesced to 1D, this returns a 1D view.
        /// If axes were reordered for memory efficiency, this reflects that reordering.
        ///
        /// Not available when buffering is enabled.
        /// Matches NumPy's NDIter_GetIterView behavior.
        /// </summary>
        /// <param name="operand">The operand index (0 to NOp-1)</param>
        /// <returns>An NDArray view with the iterator's internal shape and strides</returns>
        public NDArray GetIterView(int operand)
        {
            if ((uint)operand >= (uint)_state->NOp)
                throw new ArgumentOutOfRangeException(nameof(operand), $"Operand index {operand} out of range [0, {_state->NOp})");

            // Cannot provide views when buffering is enabled (data may be in temporary buffers)
            // NumPy's ValueError (npyiter_itviews_get), whether or not any operand is actually buffered.
            if ((_state->ItFlags & (uint)NDIterFlags.BUFFER) != 0)
                throw new ValueError("cannot provide an iterator view when buffering is enabled");

            if (_operands == null || _operands.Length <= operand)
                throw new InvalidOperationException("Operand array not available");

            var original = _operands[operand];
            int ndim = _state->NDim;

            if (ndim == 0)
            {
                // Scalar case - return a scalar view
                return original.flat[0];
            }

            // Build shape and strides from the iterator's internal state
            // NumSharp's internal Shape[0] is already the outermost axis, matching standard convention
            // (NumPy reverses because their axisdata iteration starts from innermost, but we don't need to)
            var viewShape = new long[ndim];
            var viewStrides = new long[ndim];

            for (int d = 0; d < ndim; d++)
            {
                viewShape[d] = _state->Shape[d];
                viewStrides[d] = _state->GetStride(d, operand);
            }

            // Get the reset data pointer (base pointer for this operand)
            void* dataPtr = _state->GetResetDataPtr(operand);

            // Create a view that shares storage with the original
            // We need to create an NDArray that points to the same underlying storage
            // but with the iterator's shape and strides
            var storage = original.Storage;

            // Calculate the offset from storage base to the reset data pointer. An unbuffered view walks the operand's
            // own memory, so the element size is the array side's (a string's item size included).
            int elementSize = _state->GetSrcElementSize(operand);
            long offsetBytes = (long)dataPtr - (long)storage.Address;
            long offsetElements = offsetBytes / elementSize;

            // Calculate total buffer size (from original storage)
            long bufferSize = storage.Count;

            // Create a new shape with the offset using internal constructor
            var viewShapeWithOffset = new Shape(viewShape, viewStrides, offsetElements, bufferSize);

            // Create a view NDArray that shares the same storage
            return new NDArray(storage, viewShapeWithOffset);
        }

        /// <summary>
        /// Get operand dtypes.
        /// </summary>
        public NPTypeCode[] GetDescrArray()
        {
            var result = new NPTypeCode[_state->NOp];
            for (int i = 0; i < _state->NOp; i++)
                result[i] = _state->GetOpDType(i);
            return result;
        }

        /// <summary>
        /// Get pointer to current data for operand.
        /// When buffering is enabled, returns pointer to buffer position.
        /// Otherwise returns pointer to source array position.
        /// Matches NumPy's dataptrs[i] access.
        /// </summary>
        public void* GetDataPtr(int operand)
        {
            if ((uint)operand >= (uint)_state->NOp)
                throw new ArgumentOutOfRangeException(nameof(operand));

            uint itFlags = _state->ItFlags;

            // If buffering is enabled and we have a buffer, use it
            if ((itFlags & (uint)NDIterFlags.BUFFER) != 0)
            {
                var buffer = _state->GetBuffer(operand);
                if (buffer != null)
                {
                    // REDUCE mode: DataPtrs track the current array/buffer position.
                    // - With CoreSize > 0 (double-loop active): BufferedReduceAdvance maintains DataPtrs.
                    // - With CoreSize == 0 (fallback to regular Advance): DataPtrs maintained by
                    //   Advance() using per-axis strides (stride=0 on reduce axis keeps pointer fixed).
                    // In both cases, DataPtrs is correct; don't override via IterIndex-indexed buffer.
                    if ((itFlags & (uint)NDIterFlags.REDUCE) != 0)
                    {
                        return _state->GetDataPtr(operand);
                    }

                    // Windowed buffered iteration keeps DataPtrs correct at all
                    // times: FillBufferWindow swaps buffered operands' pointers to
                    // their buffers and BufferedElementNext walks them by
                    // BufStrides. The legacy recomputation that used to live here
                    // (IterIndex - (BufIterEnd - BufferSize)) silently broke on
                    // the PARTIAL final window — e.g. 20005 elements with the
                    // default 8192 buffer put the third window's start at
                    // 20005-8192=11813 instead of 16384, reading stale data from
                    // the previous fill. DataPtrs is the single source of truth.
                    return _state->GetDataPtr(operand);
                }
            }

            return _state->GetDataPtr(operand);
        }

        /// <summary>
        /// Get current value for operand as T.
        /// When buffering with casting is enabled, reads from buffer (which has target dtype).
        /// </summary>
        public T GetValue<T>(int operand = 0) where T : unmanaged
        {
            return *(T*)GetDataPtr(operand);
        }

        /// <summary>
        /// Set current value for operand.
        /// When buffering with casting is enabled, writes to buffer (which has target dtype).
        /// </summary>
        public void SetValue<T>(T value, int operand = 0) where T : unmanaged
        {
            *(T*)GetDataPtr(operand) = value;
        }

        // =========================================================================
        // Configuration Methods
        // =========================================================================

        /// <summary>
        ///     Removes ORIGINAL axis <paramref name="axis"/> from the iteration — NumPy's <c>NpyIter_RemoveAxis</c>: the
        ///     iterator then walks every other axis with that axis fixed at logical index 0, so a caller can walk it itself
        ///     (the IterAllButAxis pattern of NumPy's reductions and scans), and the iterator is reset to its start.
        /// </summary>
        /// <param name="axis">The axis in the operands' own (original) numbering, not the iterator's internal order.</param>
        /// <returns>True (failures raise NumPy's errors).</returns>
        /// <exception cref="RuntimeError">The iterator does not track a multi-index, tracks a flat index, or is buffered
        ///     (NumPy's three texts).</exception>
        /// <exception cref="ValueError"><paramref name="axis"/> is outside <c>[0, ndim)</c>: "axis out of bounds in
        ///     iterator RemoveAxis".</exception>
        /// <remarks>
        ///     The axis is located through the permutation, so it is the right one whatever order the iterator settled on (an
        ///     F-ordered or transposed operand under K order, a reversed axis). A REVERSED axis (one the construction flipped
        ///     to walk memory forward) first moves every operand's base back to its logical index 0. Removing the last axis
        ///     leaves a 0-d iteration of one element.
        /// </remarks>
        public bool RemoveAxis(int axis)
        {
            uint itflags = _state->ItFlags;
            int ndim = _state->NDim;
            int nop = _state->NOp;

            // NumPy's checks, in its order.
            if ((itflags & (uint)NDIterFlags.HASMULTIINDEX) == 0)
                throw new RuntimeError("Iterator RemoveAxis may only be called if a multi-index is being tracked");
            if ((itflags & (uint)NDIterFlags.HASINDEX) != 0)
                throw new RuntimeError("Iterator RemoveAxis may not be called on an index is being tracked");
            if ((itflags & (uint)NDIterFlags.BUFFER) != 0)
                throw new RuntimeError("Iterator RemoveAxis may not be called on a buffered iterator");
            if (axis < 0 || axis >= ndim)
                throw new ValueError("axis out of bounds in iterator RemoveAxis");

            // Find the internal axis that holds the original one: Perm[d] = original axis, or -1 - original when the
            // construction flipped the axis to walk memory forward.
            int xdim = -1;
            for (int d = 0; d < ndim; d++)
            {
                int p = _state->Perm[d];
                if (p == axis)
                {
                    xdim = d;
                    break;
                }

                if (-1 - p == axis)
                {
                    // A flipped axis starts at its LAST logical element: move the base (and the reset pointer every
                    // GotoIterIndex starts from) back to logical index 0, the position the removed axis keeps. The stride
                    // is the flipped (positive) one and counts source elements, as everywhere in the state.
                    long last = _state->Shape[d] - 1;
                    for (int op = 0; op < nop; op++)
                    {
                        long offset = last * _state->GetStride(d, op) * _state->SrcElementSizes[op];
                        _state->BaseOffsets[op] += offset;
                        _state->ResetDataPtrs[op] += offset;
                    }

                    xdim = d;
                    break;
                }
            }

            if (xdim < 0)
                throw new RuntimeError("internal error in iterator perm");

            // The permutation of the remaining axes: entries past the removed original axis move down by one.
            for (int d = 0; d < ndim - 1; d++)
            {
                int p = d < xdim ? _state->Perm[d] : _state->Perm[d + 1];
                if (p >= 0)
                {
                    if (p > axis)
                        p--;
                }
                else if (p < -1 - axis)
                {
                    p++;
                }

                _state->Perm[d] = (sbyte)p;
            }

            // Shift the axis data of the later internal axes down over the removed one.
            for (int d = xdim; d < ndim - 1; d++)
            {
                _state->Shape[d] = _state->Shape[d + 1];
                _state->Coords[d] = _state->Coords[d + 1];

                for (int op = 0; op < nop; op++)
                {
                    _state->SetStride(d, op, _state->GetStride(d + 1, op));
                }
            }

            _state->NDim = ndim - 1;

            // The iteration size of the remaining axes; the range end follows it (NumPy leaves the start alone).
            long size = 1;
            for (int d = 0; d < ndim - 1; d++)
                size *= _state->Shape[d];
            _state->IterSize = size;
            _state->IterEnd = size;

            // A 0-d iteration keeps NumPy's singleton axis data (extent 1, strides 0) for anything that still reads it.
            if (ndim == 1)
            {
                _state->Shape[0] = 1;
                for (int op = 0; op < nop; op++)
                    _state->SetStride(0, op, 0);
            }

            // The inner strides and the specialized iternext both depend on the axes.
            _state->UpdateInnerStrides();
            _cachedIterNext = null;

            // NpyIter_RemoveAxis ends with NpyIter_Reset.
            return Reset();
        }

        /// <summary>
        /// Remove multi-index tracking and enable coalescing.
        /// Matches NumPy's NDIter_RemoveMultiIndex behavior.
        /// Note: Resets iterator position to the beginning.
        /// </summary>
        public bool RemoveMultiIndex()
        {
            // NpyIter_RemoveMultiIndex resets the iterator FIRST, whether or not a multi-index is tracked: a buffered
            // iterator's pending window is flushed and it restarts at IterStart (delayed buffers are allocated too).
            if (!Reset())
                return false;

            if ((_state->ItFlags & (uint)NDIterFlags.HASMULTIINDEX) == 0)
                return false;

            // Clear the multi-index flag
            _state->ItFlags &= ~(uint)NDIterFlags.HASMULTIINDEX;

            // NpyIter_RemoveMultiIndex only COALESCES: the axes were ordered (and flipped) at construction, with or without
            // a multi-index, and that order is the iteration order the caller asked for (a forced C or F order included).
            // Sorting them again by stride walked a forced-order iterator in memory order instead (np.nditer(c_array,
            // ["multi_index"], order="F") then remove_multi_index walked C order), and the stride sort's ascending layout
            // reversed the walk of any iterator that did not coalesce to one axis. The merge is NumPy's, in iteration
            // order, with a tracked flat index merged as one more operand, so an f_index keeps counting in F order.
            if (_state->NDim > 1)
                NDIterCoalescing.CoalesceAxesIterationOrder(ref *_state);

            // Reset iterator to beginning (NumPy behavior). A windowed buffered iterator was flushed and refilled by the
            // reset above; the coalesced axes walk the same elements in the same order, but the window bookkeeping is laid
            // out by the axes, so it is primed again under the new ones (nothing was written since the first fill, so
            // there is nothing to flush). Rewinding the state alone would point the data pointers back at ARRAY memory
            // that the views then read as buffer elements.
            uint f = _state->ItFlags;
            bool windowed = (f & (uint)NDIterFlags.BUFFER) != 0 &&
                            (f & (uint)(NDIterFlags.REDUCE | NDIterFlags.DELAYBUF)) == 0;
            _state->Reset();
            if (windowed)
                NDIterBufferManager.FillBufferWindow(ref *_state, NDIterBufferManager.ComputeTransferSize(ref *_state));

            // Clear cached iteration function
            _cachedIterNext = null;

            return true;
        }

        /// <summary>
        ///     Switches the iterator to external-loop iteration and resets it — NumPy's <c>NpyIter_EnableExternalLoop</c>:
        ///     each step then hands out a whole inner loop, starting again from the first one.
        /// </summary>
        /// <returns>True when the iterator was reset (false only when delayed buffers cannot be allocated).</returns>
        /// <exception cref="ValueError">A flat index or a multi-index is tracked ("Iterator flag EXTERNAL_LOOP cannot be
        ///     used if an index or multi-index is being tracked"), or NumPy's ranged-without-buffering refusal.</exception>
        /// <remarks>
        ///     The reset is NumPy's, and it matters: an external loop starts at the beginning of an inner run, so keeping a
        ///     position in the middle of one would hand out a chunk that crosses into the next row.
        /// </remarks>
        public bool EnableExternalLoop()
        {
            uint itflags = _state->ItFlags;

            // NumPy's checks: an external loop hands out runs, which no index can describe. The second check can only
            // trigger on an iterator that is already external-loop, ranged and unbuffered, which construction refuses.
            if ((itflags & (uint)(NDIterFlags.HASINDEX | NDIterFlags.HASMULTIINDEX)) != 0)
                throw new ValueError(
                    "Iterator flag EXTERNAL_LOOP cannot be used if an index or multi-index is being tracked");
            if ((itflags & (uint)(NDIterFlags.BUFFER | NDIterFlags.RANGE | NDIterFlags.EXLOOP)) ==
                (uint)(NDIterFlags.RANGE | NDIterFlags.EXLOOP))
                throw new ValueError(
                    "Iterator flag EXTERNAL_LOOP cannot be used with ranged iteration unless buffering is also enabled");

            if ((itflags & (uint)NDIterFlags.EXLOOP) == 0)
            {
                _state->ItFlags |= (uint)NDIterFlags.EXLOOP;
                _cachedIterNext = null;
            }

            // NpyIter_EnableExternalLoop ends with NpyIter_Reset.
            return Reset();
        }

        // =========================================================================
        // Reduction Support
        // =========================================================================

        /// <summary>
        /// Check if iteration includes reduction operands.
        /// </summary>
        public bool IsReduction => (_state->ItFlags & (uint)NDIterFlags.REDUCE) != 0;

        /// <summary>
        /// Check if a specific operand is a reduction operand (has stride=0 for READWRITE).
        /// </summary>
        public bool IsOperandReduction(int operand)
        {
            if ((uint)operand >= (uint)_state->NOp)
                throw new ArgumentOutOfRangeException(nameof(operand));

            return (_state->GetOpFlags(operand) & NDIterOpFlags.REDUCE) != 0;
        }

        /// <summary>
        /// Check if this is the first visit to the current element of a reduction operand.
        /// This is used for initialization (e.g., set to 0 before summing).
        ///
        /// For reduction operands (stride=0 on some axes), returns true when all
        /// coordinates on reduction axes are 0. Returns false when any coordinate
        /// on a reduction axis is non-zero (meaning we've already visited this
        /// output element from another input element).
        ///
        /// For non-reduction operands, always returns true (every visit is "first").
        ///
        /// Matches NumPy's NDIter_IsFirstVisit behavior.
        /// </summary>
        public bool IsFirstVisit(int operand)
        {
            if ((uint)operand >= (uint)_state->NOp)
                throw new ArgumentOutOfRangeException(nameof(operand));

            // If this operand is not a reduction, every visit is "first"
            if ((_state->GetOpFlags(operand) & NDIterOpFlags.REDUCE) == 0)
                return true;

            // Part 1: Check coordinates (unbuffered reduction check)
            // For reduction operands, check if any reduction axis coordinate is non-zero
            // A reduction axis is one where stride = 0 (but shape > 1)
            for (int d = 0; d < _state->NDim; d++)
            {
                long stride = _state->GetStride(d, operand);
                long coord = _state->Coords[d];

                // If this is a reduction dimension (stride=0) and coordinate is not 0,
                // we've already visited this output element
                if (stride == 0 && coord != 0)
                    return false;
            }

            // Part 2: Check buffer positions (buffered reduction check)
            // When BUFFERED flag is set, use CorePos to determine first visit
            // CorePos = 0 means we're at the start of a new output element
            if ((_state->ItFlags & (uint)NDIterFlags.BUFFER) != 0 && _state->CoreSize > 0)
            {
                // For buffered reduce, first visit is only when CorePos = 0
                // (at the start of accumulation for each output element)
                if (_state->CorePos != 0)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Create an independent copy of the iterator at its current position.
        /// Matches NumPy's NDIter_Copy behavior.
        /// The copy has its own state and can be advanced independently.
        /// </summary>
        public NDIterRef Copy()
        {
            // Allocate new state on heap (single block + arena, like AdvancedNew)
            var newStatePtr = AllocateStateBlock();

            try
            {
                // Copy scalar fields (excludes pointers since they will be re-allocated)
                newStatePtr->ItFlags = _state->ItFlags;
                newStatePtr->NDim = _state->NDim;
                newStatePtr->NOp = _state->NOp;
                newStatePtr->MaskOp = _state->MaskOp;
                newStatePtr->IterSize = _state->IterSize;
                newStatePtr->IterIndex = _state->IterIndex;
                newStatePtr->IterStart = _state->IterStart;
                newStatePtr->IterEnd = _state->IterEnd;
                newStatePtr->FlatIndex = _state->FlatIndex;
                newStatePtr->IsCIndex = _state->IsCIndex;
                newStatePtr->IndexReset = _state->IndexReset;
                newStatePtr->DType = _state->DType;
                newStatePtr->StridesNDim = _state->StridesNDim;
                newStatePtr->BufferSize = _state->BufferSize;
                newStatePtr->BufIterEnd = _state->BufIterEnd;
                newStatePtr->BufTransferSize = _state->BufTransferSize;
                newStatePtr->BufFlushed = _state->BufFlushed;
                newStatePtr->ReducePos = _state->ReducePos;
                newStatePtr->ReduceOuterSize = _state->ReduceOuterSize;
                newStatePtr->CoreSize = _state->CoreSize;
                newStatePtr->CorePos = _state->CorePos;
                newStatePtr->OuterDim = _state->OuterDim;
                newStatePtr->CoreOffset = _state->CoreOffset;

                // ALWAYS allocate new arrays (both dimension and operand arrays are dynamic now)
                newStatePtr->AllocateDimArrays(_state->NDim, _state->NOp, _state->StridesNDim);

                // Copy dimension arrays (if NDim > 0)
                if (_state->NDim > 0)
                {
                    // Copy Shape
                    for (int d = 0; d < _state->NDim; d++)
                        newStatePtr->Shape[d] = _state->Shape[d];

                    // Copy Coords
                    for (int d = 0; d < _state->NDim; d++)
                        newStatePtr->Coords[d] = _state->Coords[d];

                    // Copy Perm
                    for (int d = 0; d < _state->NDim; d++)
                        newStatePtr->Perm[d] = _state->Perm[d];

                    // Copy Strides
                    int strideCount = _state->StridesNDim * _state->NOp;
                    for (int i = 0; i < strideCount; i++)
                        newStatePtr->Strides[i] = _state->Strides[i];

                    // Copy the tracked flat index's strides (the copy's index advances with its own axes).
                    for (int d = 0; d < _state->StridesNDim; d++)
                        newStatePtr->IndexStrides[d] = _state->IndexStrides[d];
                }

                // Copy per-operand arrays
                int nop = _state->NOp;
                for (int op = 0; op < nop; op++)
                {
                    newStatePtr->DataPtrs[op] = _state->DataPtrs[op];
                    newStatePtr->ResetDataPtrs[op] = _state->ResetDataPtrs[op];
                    newStatePtr->BaseOffsets[op] = _state->BaseOffsets[op];
                    newStatePtr->OpItFlags[op] = _state->OpItFlags[op];
                    newStatePtr->OpDTypes[op] = _state->OpDTypes[op];
                    newStatePtr->OpSrcDTypes[op] = _state->OpSrcDTypes[op];
                    newStatePtr->OpStringKinds[op] = _state->OpStringKinds[op];
                    newStatePtr->ElementSizes[op] = _state->ElementSizes[op];
                    newStatePtr->SrcElementSizes[op] = _state->SrcElementSizes[op];
                    newStatePtr->InnerStrides[op] = _state->InnerStrides[op];
                    newStatePtr->BufStrides[op] = _state->BufStrides[op];
                    newStatePtr->ReduceOuterStrides[op] = _state->ReduceOuterStrides[op];
                    newStatePtr->ReduceOuterPtrs[op] = _state->ReduceOuterPtrs[op];
                    newStatePtr->ArrayWritebackPtrs[op] = _state->ArrayWritebackPtrs[op];

                    var sourceBuffer = (void*)_state->Buffers[op];
                    if (sourceBuffer != null)
                    {
                        // Sized by the stored element size, which a fixed-width string's buffer needs (the String
                        // lane names no width).
                        var buffer = NDIterBufferManager.AllocateAligned(_state->BufferSize, _state->ElementSizes[op]);
                        if (buffer == null)
                            throw new OutOfMemoryException("Failed to allocate iterator copy buffer.");

                        var bytes = _state->BufferSize * _state->ElementSizes[op];
                        Buffer.MemoryCopy(sourceBuffer, buffer, bytes, bytes);
                        newStatePtr->Buffers[op] = (long)buffer;

                        var dataPtr = _state->DataPtrs[op];
                        var bufferStart = _state->Buffers[op];
                        var bufferEnd = bufferStart + bytes;
                        if (dataPtr >= bufferStart && dataPtr < bufferEnd)
                            newStatePtr->DataPtrs[op] = (long)buffer + (dataPtr - bufferStart);
                    }
                }

                // Create new iterator owning the state
                return new NDIterRef
                {
                    _state = newStatePtr,
                    _ownsState = true,
                    _operands = _operands,  // Share operand references (they're not modified)
                    _cachedIterNext = null  // Don't copy cached delegate
                };
            }
            catch
            {
                if ((newStatePtr->ItFlags & (uint)NDIterFlags.BUFFER) != 0)
                    NDIterBufferManager.FreeBuffers(ref *newStatePtr);
                newStatePtr->FreeDimArrays();
                ReleaseStateBlock(newStatePtr);
                throw;
            }
        }

        // =========================================================================
        // Lifecycle
        // =========================================================================

        /// <summary>
        /// Deallocate iterator resources.
        /// </summary>
        public void Dispose()
        {
            // Windowed buffered iteration: flush the pending window so kernels
            // that consumed the final fill without a trailing iternext (e.g. the
            // single-inner-loop fast path) still land their writes (NumPy
            // resolves pending buffer write-backs in NDIter_Deallocate too).
            // Must run BEFORE ResolveWritebacks so the buffer contents reach the
            // forced-copy temp before that temp is copied to the user's array.
            if (_state != null &&
                (_state->ItFlags & (uint)NDIterFlags.BUFFER) != 0 &&
                (_state->ItFlags & (uint)NDIterFlags.REDUCE) == 0 &&
                (_state->ItFlags & (uint)NDIterFlags.DELAYBUF) == 0)
            {
                NDIterBufferManager.FlushBufferWindow(ref *_state);
            }

            // COPY_IF_OVERLAP write-backs must resolve regardless of state
            // ownership (NumPy resolves WRITEBACKIFCOPY in NDIter_Deallocate).
            ResolveWritebacks();

            if (_ownsState && _state != null)
            {
                // Free any buffers using NDIterBufferManager.FreeBuffers
                // NOTE: Buffers are allocated with AlignedAlloc, must be freed with AlignedFree
                if ((_state->ItFlags & (uint)NDIterFlags.BUFFER) != 0)
                {
                    NDIterBufferManager.FreeBuffers(ref *_state);
                }

                // Free dynamically allocated dimension arrays
                // NUMSHARP DIVERGENCE: Unlike NumPy's fixed arrays, we allocate dynamically
                _state->FreeDimArrays();

                // Park the block for the next construction on this thread (or free it).
                ReleaseStateBlock(_state);
                _state = null;
                _ownsState = false;
            }
        }

        /// <summary>
        /// Transfer ownership of the underlying <see cref="NDIterState"/>
        /// pointer out of this <see cref="NDIterRef"/>. After the call, this
        /// instance's <see cref="Dispose"/> is a no-op and the returned
        /// pointer becomes the caller's responsibility to free via
        /// <see cref="FreeState"/> (or equivalent manual teardown:
        /// <see cref="NDIterBufferManager.FreeBuffers"/> when BUFFER is set,
        /// <see cref="NDIterState.FreeDimArrays"/>, and
        /// <see cref="NativeMemory.Free"/>).
        ///
        /// Intended for callers that need to hold the iterator state across a
        /// non-ref-struct boundary (class fields, long-lived objects) where a
        /// ref struct can't live.
        /// </summary>
        public NDIterState* ReleaseState()
        {
            if (!_ownsState)
                throw new InvalidOperationException("Iterator does not own its state; cannot release.");

            var released = _state;
            _state = null;
            _ownsState = false;
            return released;
        }

        /// <summary>
        /// Tear down a state pointer previously obtained from
        /// <see cref="ReleaseState"/>. Mirrors <see cref="Dispose"/>'s cleanup
        /// path but operates on a bare pointer so long-lived owners can free
        /// the state without reconstructing an NDIterRef.
        /// </summary>
        public static void FreeState(NDIterState* state)
        {
            if (state == null) return;
            if ((state->ItFlags & (uint)NDIterFlags.BUFFER) != 0)
                NDIterBufferManager.FreeBuffers(ref *state);
            state->FreeDimArrays();
            ReleaseStateBlock(state);
        }
    }

    // =========================================================================
    // Static NDIter Class (backward compatible API)
    // =========================================================================

    /// <summary>
    /// Static iterator helper methods (backward compatible API).
    ///
    /// NUMSHARP DIVERGENCE: These methods support unlimited dimensions via dynamic allocation.
    /// Dimension arrays are allocated on demand and freed after use.
    /// </summary>
    public static unsafe class NDIter
    {
        /// <inheritdoc cref="ReduceBool{T, TKernel}(UnmanagedStorage)"/>
        public static bool ReduceBool<T, TKernel>(NDArray src)
            where T : unmanaged
            where TKernel : struct, INDBooleanReductionKernel<T>
            => ReduceBool<T, TKernel>(src.Storage);

        public static bool ReduceBool<T, TKernel>(UnmanagedStorage src)
            where T : unmanaged
            where TKernel : struct, INDBooleanReductionKernel<T>
        {
            var state = CreateReductionState(src);
            try
            {
                if (state.Size == 0)
                    return TKernel.Identity;

                if ((state.Flags & NDIterFlags.SourceContiguous) != 0)
                {
                    var input = (void*)state.GetDataPointer(0);
                    return TKernel.Identity
                        ? DirectILKernelGenerator.AllSimdHelper<T>(input, state.Size)
                        : DirectILKernelGenerator.AnySimdHelper<T>(input, state.Size);
                }

                return ReduceBoolGeneral<T, TKernel>(ref state);
            }
            finally
            {
                // Free dynamically allocated dimension arrays
                state.FreeDimArrays();
            }
        }

        // True when src and dst occupy a single gap-free buffer with the SAME layout: same shape,
        // identical strides, and contiguous in C OR F order. Then every logical index maps to the
        // same byte offset in both, so the whole copy is one flat cpblk regardless of C/F order.
        // (Differing orders — e.g. F-contig transpose src into a C-contig dst — have non-identical
        // strides and are correctly excluded, keeping the transpose copy on its strided path.)
        private static bool IsSameFlatLayout(Shape a, Shape b)
        {
            int nd = a.NDim;
            if (nd != b.NDim) return false;
            if (!(a.IsContiguous || a.IsFContiguous)) return false;
            var ad = a.dimensions; var bd = b.dimensions;
            var asr = a.strides; var bsr = b.strides;
            for (int i = 0; i < nd; i++)
                if (ad[i] != bd[i] || asr[i] != bsr[i]) return false;
            return true;
        }

        /// <inheritdoc cref="TryCopySameType(UnmanagedStorage, UnmanagedStorage)"/>
        public static bool TryCopySameType(NDArray dst, NDArray src)
            => TryCopySameType(dst.Storage, src.Storage);

        public static bool TryCopySameType(UnmanagedStorage dst, UnmanagedStorage src)
        {
            if (dst.TypeCode != src.TypeCode)
                return false;

            NumSharpException.ThrowIfNotWriteable(dst.Shape);

            // Scalar-broadcast fast path: src has ALL strides 0, so every logical element is the
            // SAME single value. When dst is a gap-free, whole-buffer C- OR F-contiguous target, fill
            // it with one typed fill (1-byte InitBlock/memset, wider unrolled/SIMD) instead of the
            // per-element strided walk the general path takes here (IsContiguousCopy is false for a
            // broadcast src, so the cpblk tier below is skipped). Mirrors the IsScalarBroadcast tier
            // in UnmanagedStorage.CloneData and is bit-identical: the same GetValue(0) value lands in
            // every slot (read once, before the write, so a src/dst alias is safe). 6-8x for 1-byte,
            // ~2x wider. The fill is order-independent — every slot gets the same value — so an F-
            // contiguous whole buffer (offset 0, size == Count, no gaps) is equally fillable as C.
            // GetValue(0) is offset-aware (GetOffset honours the broadcast strides) and boxes as the
            // storage dtype, which IArraySlice.Fill unboxes via the same-type cast.
            if (src.Shape.IsScalarBroadcast
                && (dst.Shape.IsContiguous || dst.Shape.IsFContiguous)
                && dst.Shape.offset == 0
                && dst.Shape.size > 0
                && dst.Shape.size == dst.InternalArray.Count)
            {
                dst.InternalArray.Fill(src.GetValue(0));
                return true;
            }

            // Trivial flat copy — SKIP the iterator state build. When src and dst share an identical
            // gap-free layout (same dims AND strides, C- or F-contiguous), the i-th logical element
            // sits at the same buffer offset on both sides, so the whole copy is a single
            // Buffer.MemoryCopy on the offset-adjusted raw pointers. This is the SAME block copy the
            // post-state Contiguous CopyKernel does (IsSameFlatLayout is already trusted for that
            // tier below), but it bypasses CreateCopyState — the dimension-array allocation + broadcast/
            // stride analysis that otherwise runs even here, measured at ~330ns and ~97% of a small
            // (n<=1000) copyto/astype (that fixed cost is what put copy/cast at 0.32-0.80x vs NumPy
            // below 10K elements). Overlap is already resolved by the caller — Copy(UnmanagedStorage,…)
            // clones an aliasing src that needs a temp before reaching here — and Buffer.MemoryCopy is
            // the identical primitive the cpblk kernel emits, so this is byte- and overlap-identical.
            if (IsSameFlatLayout(src.Shape, dst.Shape) && dst.Shape.size > 0)
            {
                int itemLen = dst.InternalArray.ItemLength;
                byte* sp = src.Address + src.Shape.offset * itemLen;
                byte* dp = dst.Address + dst.Shape.offset * itemLen;
                long bytes = dst.Shape.size * itemLen;
                Buffer.MemoryCopy(sp, dp, bytes, bytes);
                return true;
            }

            var state = CreateCopyState(src, dst);
            try
            {
                if (state.Size == 0)
                    return true;

                // Contiguous fast path: existing IL CopyKernel uses cpblk — single block copy,
                // minimal overhead. Cross-platform memcpy is the cheapest possible.
                //
                // IsContiguousCopy keys off C-contiguity only. A same-type copy whose src and dst
                // share an identical-stride GAP-FREE buffer (C- OR F-contiguous) is equally a single
                // flat cpblk — the i-th logical element sits at the same byte offset in both. astype
                // preserves order (F-src -> F-dst), so this is the common F-contiguous case; without
                // it the Vector-less dtypes (Boolean/Char/Half/Complex) the strided cast kernel
                // rejects fall to the slow per-element CopyGeneralSameType (the bool|F 0.18x cliff).
                if (state.IsContiguousCopy || IsSameFlatLayout(src.Shape, dst.Shape))
                {
                    var copyKernel = DirectILKernelGenerator.TryGetCopyKernel(new CopyKernelKey(dst.TypeCode, CopyExecutionPath.Contiguous));
                    if (copyKernel != null)
                    {
                        copyKernel(
                            (void*)state.GetDataPointer(0),
                            (void*)state.GetDataPointer(1),
                            state.GetStridesPointer(0),
                            state.GetStridesPointer(1),
                            state.GetShapePointer(),
                            state.NDim,
                            state.Size);
                        return true;
                    }
                }

                // General path (strided / broadcast): route through the IL StridedCastKernel —
                // detects unit-stride innermost axis and uses Buffer.MemoryCopy per row; falls
                // back to scalar incremental-coord inner loop when inner is also strided. Returns
                // null for the Vector-less dtypes (Char/Half/Decimal/Complex/Boolean), which then
                // take the CopyGeneralSameType fallback below — same row-memcpy + incremental-coord
                // shape, just without the SIMD cast body they can't use anyway for a same-type copy.
                var stridedKernel = DirectILKernelGenerator.TryGetStridedCastKernel(dst.TypeCode, dst.TypeCode);
                if (stridedKernel != null)
                {
                    stridedKernel(
                        (void*)state.GetDataPointer(0),
                        (void*)state.GetDataPointer(1),
                        state.GetStridesPointer(0),
                        state.GetStridesPointer(1),
                        state.GetShapePointer(),
                        state.NDim);
                    return true;
                }

                // Fallback for the dtypes the StridedCastKernel rejects (Char/Half/Decimal/
                // Complex/Boolean): CopyGeneralSameType — row-memcpy + incremental-coord, dtype-
                // agnostic byte movement that matches the StridedCastKernel's wall time.
                var fallbackKernel = DirectILKernelGenerator.TryGetCopyKernel(new CopyKernelKey(dst.TypeCode, CopyExecutionPath.General));
                if (fallbackKernel == null)
                    return false;

                fallbackKernel(
                    (void*)state.GetDataPointer(0),
                    (void*)state.GetDataPointer(1),
                    state.GetStridesPointer(0),
                    state.GetStridesPointer(1),
                    state.GetShapePointer(),
                    state.NDim,
                    state.Size);

                return true;
            }
            finally
            {
                state.FreeDimArrays();
            }
        }

        /// <summary>
        /// Copy <paramref name="src"/> into <paramref name="dst"/> with full
        /// support for broadcast, stride, and cross-dtype conversion.
        ///
        /// <list type="bullet">
        ///   <item>Same dtype (the common case) routes through the SIMD-accelerated
        ///   <see cref="TryCopySameType(UnmanagedStorage, UnmanagedStorage)"/>
        ///   IL copy kernel — broadcast and arbitrary strides are absorbed by the
        ///   coalesced iteration state.</item>
        ///   <item>Cross dtype falls through to a per-element cast loop
        ///   (<see cref="NDIterCasting.CopyStridedToStridedWithCast"/>) reusing
        ///   the same broadcast/coalescing state.</item>
        /// </list>
        ///
        /// Drop-in replacement for the legacy <c>MultiIterator.Assign(dst, src)</c>:
        /// matches its broadcast-src-to-dst-shape semantics and its cast-on-write
        /// behavior (read src as src.TypeCode, convert, write dst.TypeCode).
        /// </summary>
        /// <param name="dst">The destination array (written in place).</param>
        /// <param name="src">The source array; broadcast to <paramref name="dst"/>'s shape.</param>
        /// <exception cref="ArgumentNullException"><paramref name="dst"/> or <paramref name="src"/> is null.</exception>
        /// <exception cref="ValueError">If <paramref name="dst"/> is not writeable (e.g., broadcast view) — NumPy's "… is read-only".</exception>
        /// <remarks>
        ///     <see cref="NDAnchoredAttribute"/>: the copy runs storage to storage, and a storage does not hold its buffer
        ///     (the NDArray's ARC reference does), so once <c>src.Storage</c> is read nothing in this frame keeps a
        ///     caller's temporary source alive — the lifetime oracle's gap LT09 (every cross-dtype <c>TensorEngine.Cast</c>
        ///     and <c>np.copyto</c> came through here). The weaver roots both arrays until the copy returns.
        /// </remarks>
        [NDAnchored]
        public static void Copy(NDArray dst, NDArray src)
        {
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            if (src is null) throw new ArgumentNullException(nameof(src));
            Copy(dst.Storage, src.Storage);
        }

        /// <summary>
        ///     The unified allocate-and-fill core for copying, retyping, and casting. Allocates a
        ///     fresh array of <paramref name="dstType"/> whose physical layout follows
        ///     <paramref name="order"/> (NumPy C/F/A/K resolved via <see cref="OrderResolver"/>),
        ///     then fills it from <paramref name="src"/> through the in-place <see cref="Copy(NDArray, NDArray)"/>
        ///     primitive. Every elementwise concern — broadcast, arbitrary strides, transpose, and
        ///     cross-dtype conversion — is absorbed by the NDIter copy core, so a single call covers
        ///     <c>ndarray.copy()</c>, <c>astype()</c>, and <c>Clone()</c> alike.
        /// </summary>
        /// <param name="dstType">dtype DESCRIPTOR of the result (equal to <paramref name="src"/>'s for a plain copy/retype). A
        ///     descriptor, not a type code, because a fixed-width string's width lives on it (<c>&lt;U5</c>); an
        ///     <see cref="NPTypeCode"/> converts implicitly for the numeric lanes.</param>
        /// <param name="src">source array of any layout (contiguous, sliced, strided, broadcast, transposed, scalar, empty).</param>
        /// <param name="order">
        ///     'C' (row-major), 'F' (column-major), 'A' ('F' iff src is F-contiguous and not C), or
        ///     'K' (default; KEEPORDER — mirror src contiguity, matching NumPy astype order='K').
        /// </param>
        /// <param name="engine">tensor engine assigned to the result; defaults to <paramref name="src"/>'s engine.</param>
        /// <returns>A freshly allocated, owning array of <paramref name="dstType"/> holding src's values.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="src"/> is null.</exception>
        /// <remarks>
        ///     <see cref="NDAnchoredAttribute"/>: the fill reads <paramref name="src"/>'s storage, so the weaver roots it until
        ///     the copy returns — a caller that casts a temporary (<c>engine.Cast(np.arange(n), ...)</c>) cannot lose it
        ///     mid-copy (see <see cref="Copy(NDArray, NDArray)"/>).
        /// </remarks>
        [NDAnchored]
        public static NDArray CopyAs(DType dstType, NDArray src, char order = 'K', TensorEngine engine = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));

            engine = engine ?? src.TensorEngine;
            Shape srcShape = src.Shape;

            // 0-D scalar has no axes to order; everything else allocates with the resolved layout
            // (KEEPORDER lets an F-contiguous / transposed source cast as a single flat pass).
            Shape outShape;
            if (srcShape.NDim == 0)
                outShape = Shape.NewScalar();
            else if ((order == 'K' || order == 'k') && !srcShape.IsContiguous && !srcShape.IsFContiguous)
            {
                // True KEEPORDER (PyArray_NewLikeArray NPY_KEEPORDER): a neither-contiguous source
                // allocates along its SORTED STRIDE PERM rather than collapsing to C — a 3-D
                // transpose keeps its exact stride order (a neither-contiguous owned result) and a
                // broadcast's stride-0 axes sort slowest (F-ish layout). Probed NumPy 2.4.2 via
                // copy/astype/np.sort order='K'. Sources that ARE C- or F-contiguous fall through
                // to the binary resolve below, which is exact for them.
                outShape = srcShape.KeepOrder();
            }
            else
            {
                char physical = OrderResolver.Resolve(order, srcShape);
                outShape = Shape.Adopt((long[])srcShape.dimensions.Clone(), physical);
            }

            // Every byte of dst is written by the copy, so a U result skips NumPy's NEEDS_INIT zero fill (ForOverwrite);
            // a failing copy (an S <-> U conversion error) disposes it rather than hand out its undefined bytes.
            var dst = NDArray.ForOverwrite(dstType, outShape);
            dst.TensorEngine = engine;
            try
            {
                Copy(dst, src);
            }
            catch
            {
                dst.Dispose();
                throw;
            }
            return dst;
        }

        /// <inheritdoc cref="Copy(NDArray, NDArray)"/>
        public static void Copy(UnmanagedStorage dst, UnmanagedStorage src)
        {
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            if (src is null) throw new ArgumentNullException(nameof(src));

            // Fixed-width strings have no element kernel: the string route copies their bytes through uint8 / uint32
            // UNIT VIEWS (one extra trailing axis of characters) by calling back into this method, so broadcasting,
            // layouts, the overlap guard and the SIMD kernels below are the numeric ones.
            if (dst.TypeCode == NPTypeCode.String || src.TypeCode == NPTypeCode.String)
            {
                NumSharp.Backends.Strings.NDStringCopy.Copy(dst, src);
                return;
            }

            // OVERLAP GUARD — NumPy PyArray_AssignArray (array_assign_array.c:371-399).
            // A forward element/row copy reads source lanes after destination writes have
            // already clobbered them when src and dst alias the same buffer. A same-direction
            // 1-D run coalesces to a memmove-safe block copy and is fine; but 1-D opposite-
            // direction strides (e.g. a[:] = a[::-1]) and ANY multi-dimensional copy must
            // first materialize src into a temporary. Detection reuses the shared solver
            // (NDMemOverlap = NumPy solve_may_share_memory) directly on the storages — NOT via
            // throwaway NDArray wrappers, which under ARC would retain the buffers. A cross-dtype
            // copy targets a distinct buffer, so the bounds test short-circuits to No for free.
            //
            // A 1-D destination with both runs in the same direction is NOT copied first in NumPy:
            // raw_array_assign_array walks it in its own order (a negative destination stride
            // flipped, then backward when the source starts below the destination inside it),
            // and with unequal strides that order reads elements it has already written —
            // a[3:19:2] = a[6:14] gives NumPy's "smeared" values, not the copy-first ones. Such a
            // run takes AssignRaw1DLikeNumPy; a unit-stride forward run of one dtype stays on the
            // default path, whose memmove is NumPy's result. An overlap between different dtypes
            // (two reinterpreting views of one buffer) is copied first: NumPy's element order inside
            // a cast loop is not defined (its contiguous casts are vectorized).
            UnmanagedStorage overlapTemp = null;
            bool needsTemp = NeedsAssignmentTemp(dst.Shape, src.Shape);
            bool rawOrder = !needsTemp && RawAssignOrderMatters(dst, src);
            if ((needsTemp || rawOrder) && NDMemOverlap.StoragesMayShareMemory(src, dst))
            {
                if (!needsTemp && src.TypeCode == dst.TypeCode)
                {
                    AssignRaw1DLikeNumPy(dst, src);
                    return;
                }

                // Reassigning the parameter here drops the clone on method exit, leaking its pooled
                // buffer to a future GC + finalizer pass. The clone is a bare UnmanagedStorage — NOT
                // an NDArray — so NO [NDScoped] ambient scope reclaims it (NDScope.Track fires only
                // from NDArray.InitializeArc); capture it so the finally returns its buffer to the
                // pool. The buffer is a fresh, private copy this method owns outright.
                src = src.Clone();
                overlapTemp = src;
            }

            try
            {
                // Same-dtype fast path: SIMD copy kernel, broadcast + stride aware.
                if (TryCopySameType(dst, src))
                    return;

                // Cross-dtype: per-element cast via NDIterCasting.ConvertValue,
                // driven by the same coalesced broadcast state used by TryCopySameType.
                NumSharpException.ThrowIfNotWriteable(dst.Shape);

                // Scalar-broadcast cross-dtype fast path: the source is a single value (ALL strides 0),
                // so convert it ONCE with the same NumPy-faithful NDIterCasting.ConvertValue the scalar
                // fallback below uses, then typed-fill the whole contiguous dst — instead of casting the
                // identical value per element. Whole-buffer C/F-contiguous dst only (offset 0, size ==
                // Count); the fill is order-independent. dst[0] is written first (offset 0 ⇒ buffer[0]),
                // then read back boxed as dst's dtype and replicated across every slot via IArraySlice.Fill.
                if (src.Shape.IsScalarBroadcast
                    && (dst.Shape.IsContiguous || dst.Shape.IsFContiguous)
                    && dst.Shape.offset == 0
                    && dst.Shape.size > 0
                    && dst.Shape.size == dst.InternalArray.Count)
                {
                    byte* sp = src.Address + src.Shape.offset * src.InternalArray.ItemLength;
                    NDIterCasting.ConvertValue(sp, (void*)dst.Address, src.TypeCode, dst.TypeCode);
                    dst.InternalArray.Fill(dst.GetValue(0));
                    return;
                }

                var state = CreateCopyState(src, dst);
                try
                {
                    if (state.Size == 0)
                        return;

                    // SIMD fast path 1: both src and dst contiguous, no broadcast.
                    // IL-generated contig cast kernel — minimal overhead for the common case.
                    if (state.IsContiguousCopy && state.Size > 0)
                    {
                        var castKernel = NumSharp.Backends.Kernels.DirectILKernelGenerator
                            .TryGetCastKernel(src.TypeCode, dst.TypeCode);
                        if (castKernel != null)
                        {
                            castKernel((void*)state.GetDataPointer(0), (void*)state.GetDataPointer(1), state.Size);
                            return;
                        }
                    }

                    // SIMD fast path 2: strided/broadcast cast. IL kernel walks outer dims via incremental
                    // coord advance and uses the same SIMD body as the contig kernel for any inner axis
                    // with stride==1 for both src and dst. Falls back internally to scalar strided inner
                    // loop when the innermost axis has non-unit stride for either side.
                    if (state.Size > 0)
                    {
                        var stridedKernel = NumSharp.Backends.Kernels.DirectILKernelGenerator
                            .TryGetStridedCastKernel(src.TypeCode, dst.TypeCode);
                        if (stridedKernel != null)
                        {
                            stridedKernel(
                                (void*)state.GetDataPointer(0),
                                (void*)state.GetDataPointer(1),
                                state.GetStridesPointer(0),
                                state.GetStridesPointer(1),
                                state.GetShapePointer(),
                                state.NDim);
                            return;
                        }
                    }

                    // Scalar dispatch: Decimal/Complex/Half/Char/Boolean involved.
                    NDIterCasting.CopyStridedToStridedWithCast(
                        (void*)state.GetDataPointer(0),
                        state.GetStridesPointer(0),
                        src.TypeCode,
                        (void*)state.GetDataPointer(1),
                        state.GetStridesPointer(1),
                        dst.TypeCode,
                        state.GetShapePointer(),
                        state.NDim,
                        state.Size);
                }
                finally
                {
                    state.FreeDimArrays();
                }
            }
            finally
            {
                // Return the overlap-copy clone's pooled buffer (see the capture above). The clone was
                // never InitializeArc'd — a bare UnmanagedStorage carries no counted reference — so its
                // block sits at refcount 0 and a lone Release() is a no-op stray. Take the one owning
                // reference the clone implicitly holds, then release it: the 0 → 1 → 0 transition frees
                // the buffer straight back to SizeBucketedBufferPool (the finalizer would eventually do
                // the same via Abandon, but eagerly here reclaims the pool slot for warm reuse).
                var overlapBlock = overlapTemp?.InternalArray;
                if (overlapBlock is not null)
                {
                    overlapBlock.TryAddRef();
                    overlapBlock.Release();
                }
            }
        }

        /// <summary>
        /// NumPy's array_assign_array.c gate for copying an overlapping source first: a destination
        /// of more than one axis, or a 1-D destination whose run points the OTHER way than the
        /// source's last axis (<c>dst.strides[0] * src.strides[-1] &lt; 0</c>, e.g. <c>a[:] = a[::-1]</c>).
        /// Every other overlapping 1-D copy is walked in NumPy's raw order instead
        /// (<see cref="RawAssignOrderMatters"/>, <see cref="AssignRaw1DLikeNumPy"/>), which with
        /// equal strides equals copying first and with unequal strides does not
        /// (<c>a[3:19:2] = a[6:14]</c> reads its own writes in NumPy). Allocation-free; the overlap
        /// test runs only when this or <see cref="RawAssignOrderMatters"/> is true.
        /// </summary>
        /// <param name="dst">The destination's shape.</param>
        /// <param name="src">The source's shape (its ORIGINAL strides: NumPy tests them before broadcasting).</param>
        /// <returns><c>true</c> when an overlapping source must be copied to a temporary first.</returns>
        private static bool NeedsAssignmentTemp(Shape dst, Shape src)
        {
            int dn = (int)dst.NDim;
            if (dn > 1)
                return true;

            int sn = (int)src.NDim;
            return dn == 1 && sn >= 1 && dst.strides[0] * src.strides[sn - 1] < 0;
        }

        /// <summary>
        /// Whether a 1-D copy that NumPy does NOT copy first can still differ from NumSharp's default
        /// path when the operands overlap — so that an overlap must take NumPy's raw order (or, for
        /// two dtypes, a temporary). False for a one-element or broadcast-scalar copy (no order to
        /// observe) and for a unit-stride forward run of one dtype (memmove, which NumPy's contiguous
        /// copy also is); true otherwise.
        /// </summary>
        /// <param name="dst">The destination storage (1-D for a true result).</param>
        /// <param name="src">The source storage.</param>
        /// <returns><c>true</c> when an overlap needs <see cref="AssignRaw1DLikeNumPy"/> or a temporary.</returns>
        private static bool RawAssignOrderMatters(UnmanagedStorage dst, UnmanagedStorage src)
        {
            if (dst.Shape.NDim != 1 || dst.Shape.dimensions[0] <= 1)
                return false;
            if (!TryEffectiveSourceStride1D(dst.Shape.dimensions[0], src.Shape, out long ss) || ss == 0)
                return false;
            if (dst.TypeCode != src.TypeCode)
                return true;
            return !(ss == 1 && dst.Shape.strides[0] == 1);
        }

        /// <summary>
        /// The element stride a source walks a 1-D destination of <paramref name="n"/> elements with:
        /// 0 when it broadcasts one element (0-d, or a last axis of extent 1), its last stride when that
        /// axis has extent <paramref name="n"/>. Leading axes must all have extent 1.
        /// </summary>
        /// <param name="n">The destination's extent.</param>
        /// <param name="src">The source's shape.</param>
        /// <param name="stride">The effective element stride.</param>
        /// <returns><c>false</c> when the source does not broadcast to <paramref name="n"/> elements (the
        /// default path then reports the error).</returns>
        private static bool TryEffectiveSourceStride1D(long n, Shape src, out long stride)
        {
            stride = 0;
            int sn = (int)src.NDim;
            if (sn == 0)
                return true;
            for (int d = 0; d < sn - 1; d++)
                if (src.dimensions[d] != 1)
                    return false;
            long last = src.dimensions[sn - 1];
            if (last == 1)
                return true;
            if (last != n)
                return false;
            stride = src.strides[sn - 1];
            return true;
        }

        /// <summary>
        /// NumPy's 1-D <c>raw_array_assign_array</c> for an overlapping same-dtype copy: the destination's
        /// negative stride flipped, then a backward walk when the source starts below the destination
        /// inside it, element by element (<see cref="DirectILKernelGenerator.AssignRaw1D"/>).
        /// </summary>
        /// <param name="dst">The 1-D destination storage.</param>
        /// <param name="src">The same-dtype source storage (checked by <see cref="RawAssignOrderMatters"/>).</param>
        /// <exception cref="ValueError">The destination is read-only ("assignment destination is read-only").</exception>
        private static void AssignRaw1DLikeNumPy(UnmanagedStorage dst, UnmanagedStorage src)
        {
            NumSharpException.ThrowIfNotWriteable(dst.Shape);
            long n = dst.Shape.dimensions[0];
            TryEffectiveSourceStride1D(n, src.Shape, out long ss);
            int itemsize = dst.TypeCode.SizeOf();
            byte* d = dst.Address + dst.Shape.offset * itemsize;
            byte* s = src.Address + src.Shape.offset * itemsize;
            DirectILKernelGenerator.AssignRaw1D(d, dst.Shape.strides[0] * itemsize, s, ss * itemsize, n, itemsize);
        }

        private static bool ReduceBoolGeneral<T, TKernel>(ref NDIterState state)
            where T : unmanaged
            where TKernel : struct, INDBooleanReductionKernel<T>
        {
            var shape = state.GetShapePointer();
            var strides = state.GetStridesPointer(0);
            var coords = state.GetCoordsPointer();
            var data = (T*)state.GetDataPointer(0);

            long offset = 0;
            bool accumulator = TKernel.Identity;

            for (long linearIndex = 0; linearIndex < state.Size; linearIndex++)
            {
                accumulator = TKernel.Accumulate(accumulator, data[offset]);
                if (TKernel.ShouldExit(accumulator))
                    break;

                Advance(shape, strides, coords, state.NDim, ref offset);
            }

            return accumulator;
        }

        /// <summary>
        /// Create state for copy operation.
        /// IMPORTANT: Caller must call state.FreeDimArrays() when done!
        /// </summary>
        public static NDIterState CreateCopyState(UnmanagedStorage src, UnmanagedStorage dst)
        {
            // Fast path: when src and dst have identical dimensions, no broadcast
            // is needed — the broadcast result is just src.Shape (same dims, same
            // strides, same offset). Skip np.broadcast_to entirely to avoid the
            // ValidateBroadcastTo loop and function-dispatch overhead per call.
            //
            // Strides are intentionally NOT compared: the iterator uses src.strides
            // for reads and dst.strides for writes, regardless of how each side's
            // memory is laid out. As long as the dimensions match position-by-position,
            // no broadcast stretching is required.
            //
            // Falls through to np.broadcast_to for:
            //   - Different NDim (e.g. src=(N,), dst=(M,N) requires dim prepend)
            //   - Any axis where src=1 but dst>1 (broadcast stretch → stride 0)
            //   - Any axis where src!=dst (validation throws)
            Shape broadcastSrcShape;
            if (ShapesMatchExactly(src.Shape, dst.Shape))
                broadcastSrcShape = src.Shape;
            else
                broadcastSrcShape = np.broadcast_to(src.Shape, dst.Shape);

            int ndim = checked((int)dst.Shape.NDim);

            // NUMSHARP DIVERGENCE: No MaxDims limit - supports unlimited dimensions
            var state = new NDIterState
            {
                Size = dst.Shape.size,
                DType = dst.TypeCode,
                Flags = NDIterFlags.None,
            };

            // Allocate dimension arrays dynamically
            state.AllocateDimArrays(ndim, 2);

            state.SetOpDType(0, src.TypeCode);
            state.SetOpDType(1, dst.TypeCode);

            state.SetDataPointer(0, (IntPtr)((byte*)src.Address + (broadcastSrcShape.offset * src.InternalArray.ItemLength)));
            state.SetDataPointer(1, (IntPtr)((byte*)dst.Address + (dst.Shape.offset * dst.InternalArray.ItemLength)));

            var shape = state.GetShapePointer();
            var srcStridePtr = state.GetStridesPointer(0);
            var dstStridePtr = state.GetStridesPointer(1);

            for (int axis = 0; axis < ndim; axis++)
            {
                shape[axis] = dst.Shape.dimensions[axis];
                srcStridePtr[axis] = broadcastSrcShape.strides[axis];
                dstStridePtr[axis] = dst.Shape.strides[axis];

                if (shape[axis] > 1 && srcStridePtr[axis] == 0)
                    state.Flags |= NDIterFlags.SourceBroadcast;
            }

            CoalesceAxes(ref state, shape, srcStridePtr, dstStridePtr);

            // K-order axis permutation (descending by absolute stride).
            //
            // Without this, F-contiguous and transposed inputs whose sliced
            // dst is non-contig end up with the unit-stride axis at position
            // 0 (outermost) — leaving the strided cast kernel's "inner-contig"
            // fast path unused because the innermost axis has large stride.
            //
            // Example: F-contig dst (2000, 1000) sliced as dst[0:1000, :]
            //   shape=(1000, 1000), src=(1, 1000), dst=(1, 2000)
            // After CoalesceAxes (no coalescing possible), the innermost axis
            // has src=1000, dst=2000 — strided. Descending sort puts unit
            // stride at the innermost axis:
            //   shape=(1000, 1000), src=(1000, 1), dst=(2000, 1)
            // The IL kernel then emits 1000 inner memcpys of 1000 elements
            // each, ~17x faster than the scalar strided inner loop.
            //
            // K-order matches NumPy's `iter._kind == "K"` behavior for plain
            // copies. Skipped when only one axis remains (no-op).
            if (state.NDim > 1)
                NDIterCoalescing.ReorderAxesForCoalescing(
                    ref state, NPY_ORDER.NPY_KEEPORDER, forCoalescing: false);

            UpdateLayoutFlags(ref state, shape, srcStridePtr, dstStridePtr);

            return state;
        }

        /// <summary>
        ///     Returns true iff <paramref name="src"/> and <paramref name="dst"/>
        ///     have the same number of dimensions and every dimension matches
        ///     position-by-position. When this is true, no broadcast stretching
        ///     is required and the copy iterator can reuse <c>src</c>'s shape
        ///     directly (its strides and offset are preserved as-is).
        /// </summary>
        /// <remarks>
        ///     Strides are intentionally NOT compared — different layouts
        ///     (e.g. C-contig src vs F-contig dst) still iterate correctly
        ///     because the iterator walks each side using its own strides.
        ///     Only dimension equality matters for "no broadcast needed".
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static bool ShapesMatchExactly(in Shape src, in Shape dst)
        {
            if (src.NDim != dst.NDim) return false;

            // Scalar fast path: both 0-D = both scalar = match.
            if (src.NDim == 0) return true;

            var srcDims = src.dimensions;
            var dstDims = dst.dimensions;
            for (int i = 0; i < srcDims.Length; i++)
                if (srcDims[i] != dstDims[i]) return false;

            return true;
        }

        /// <summary>
        /// Create state for reduction operation.
        /// IMPORTANT: Caller must call state.FreeDimArrays() when done!
        /// </summary>
        public static NDIterState CreateReductionState(UnmanagedStorage src)
        {
            int ndim = checked((int)src.Shape.NDim);

            // NUMSHARP DIVERGENCE: No MaxDims limit - supports unlimited dimensions
            var state = new NDIterState
            {
                Size = src.Shape.size,
                DType = src.TypeCode,
                Flags = src.Shape.IsContiguous ? NDIterFlags.SourceContiguous : NDIterFlags.None,
            };

            // Allocate dimension arrays dynamically
            state.AllocateDimArrays(ndim, 1);

            state.SetOpDType(0, src.TypeCode);
            state.SetDataPointer(0, (IntPtr)((byte*)src.Address + (src.Shape.offset * src.InternalArray.ItemLength)));

            var shape = state.GetShapePointer();
            var srcStridePtr = state.GetStridesPointer(0);

            for (int axis = 0; axis < ndim; axis++)
            {
                shape[axis] = src.Shape.dimensions[axis];
                srcStridePtr[axis] = src.Shape.strides[axis];
            }

            return state;
        }

        public static void CoalesceAxes(ref NDIterState state, long* shape, long* srcStrides, long* dstStrides)
        {
            if (state.NDim <= 1)
                return;

            int writeAxis = 0;
            int newNDim = 1;

            for (int axis = 0; axis < state.NDim - 1; axis++)
            {
                int nextAxis = axis + 1;
                long shape0 = shape[writeAxis];
                long shape1 = shape[nextAxis];

                bool srcCanCoalesce =
                    ((shape0 == 1 && srcStrides[writeAxis] == 0) ||
                     (shape1 == 1 && srcStrides[nextAxis] == 0) ||
                     (srcStrides[writeAxis] * shape0 == srcStrides[nextAxis]));

                bool dstCanCoalesce =
                    ((shape0 == 1 && dstStrides[writeAxis] == 0) ||
                     (shape1 == 1 && dstStrides[nextAxis] == 0) ||
                     (dstStrides[writeAxis] * shape0 == dstStrides[nextAxis]));

                if (srcCanCoalesce && dstCanCoalesce)
                {
                    shape[writeAxis] *= shape1;
                    if (srcStrides[writeAxis] == 0)
                        srcStrides[writeAxis] = srcStrides[nextAxis];
                    if (dstStrides[writeAxis] == 0)
                        dstStrides[writeAxis] = dstStrides[nextAxis];
                }
                else
                {
                    writeAxis++;
                    if (writeAxis != nextAxis)
                    {
                        shape[writeAxis] = shape[nextAxis];
                        srcStrides[writeAxis] = srcStrides[nextAxis];
                        dstStrides[writeAxis] = dstStrides[nextAxis];
                    }
                    newNDim++;
                }
            }

            state.NDim = newNDim;
        }

        public static void UpdateLayoutFlags(ref NDIterState state, long* shape, long* srcStrides, long* dstStrides)
        {
            if (state.Size <= 1)
            {
                state.Flags |= NDIterFlags.SourceContiguous | NDIterFlags.DestinationContiguous;
                return;
            }

            if (IsContiguous(shape, srcStrides, state.NDim))
                state.Flags |= NDIterFlags.SourceContiguous;
            if (IsContiguous(shape, dstStrides, state.NDim))
                state.Flags |= NDIterFlags.DestinationContiguous;
        }

        public static bool IsContiguous(long* shape, long* strides, int ndim)
        {
            if (ndim == 0)
                return true;

            long expected = 1;
            for (int axis = ndim - 1; axis >= 0; axis--)
            {
                long dim = shape[axis];
                if (dim == 0)
                    return true;
                if (dim != 1)
                {
                    if (strides[axis] != expected)
                        return false;
                    expected *= dim;
                }
            }

            return true;
        }

        public static void Advance(long* shape, long* strides, long* coords, int ndim, ref long offset)
        {
            for (int axis = ndim - 1; axis >= 0; axis--)
            {
                long next = coords[axis] + 1;
                if (next < shape[axis])
                {
                    coords[axis] = next;
                    offset += strides[axis];
                    return;
                }

                coords[axis] = 0;
                offset -= strides[axis] * (shape[axis] - 1);
            }
        }
    }
}
