using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// Function to advance iterator to next position.
    /// Returns true if more iterations remain.
    /// </summary>
    internal unsafe delegate bool NpyIterNextFunc(ref NpyIterState state);

    /// <summary>
    /// Function to get multi-index at current position.
    /// </summary>
    internal unsafe delegate void NpyIterGetMultiIndexFunc(ref NpyIterState state, long* outCoords);

    /// <summary>
    /// Inner loop kernel called by iterator.
    /// </summary>
    internal unsafe delegate void NpyIterInnerLoopFunc(
        void** dataptrs,
        long* strides,
        long count,
        void* auxdata);

    /// <summary>
    /// High-performance multi-operand iterator matching NumPy's nditer API.
    /// </summary>
    internal unsafe ref partial struct NpyIterRef
    {
        private NpyIterState* _state;
        private bool _ownsState;
        private NDArray[]? _operands;
        private NpyIterNextFunc? _cachedIterNext;

        // =========================================================================
        // Factory Methods
        // =========================================================================

        /// <summary>
        /// Create single-operand iterator.
        /// Equivalent to NumPy's NpyIter_New.
        /// </summary>
        public static NpyIterRef New(
            NDArray op,
            NpyIterGlobalFlags flags = NpyIterGlobalFlags.None,
            NPY_ORDER order = NPY_ORDER.NPY_KEEPORDER,
            NPY_CASTING casting = NPY_CASTING.NPY_SAFE_CASTING,
            NPTypeCode? dtype = null)
        {
            var opFlags = new[] { NpyIterPerOpFlags.READONLY };
            var dtypes = dtype.HasValue ? new[] { dtype.Value } : null;
            return MultiNew(1, new[] { op }, flags, order, casting, opFlags, dtypes);
        }

        /// <summary>
        /// Create multi-operand iterator.
        /// Equivalent to NumPy's NpyIter_MultiNew.
        /// </summary>
        public static NpyIterRef MultiNew(
            int nop,
            NDArray[] op,
            NpyIterGlobalFlags flags,
            NPY_ORDER order,
            NPY_CASTING casting,
            NpyIterPerOpFlags[] opFlags,
            NPTypeCode[]? opDtypes = null)
        {
            return AdvancedNew(nop, op, flags, order, casting, opFlags, opDtypes);
        }

        /// <summary>
        /// Create iterator with full control over all parameters.
        /// Equivalent to NumPy's NpyIter_AdvancedNew.
        /// </summary>
        public static NpyIterRef AdvancedNew(
            int nop,
            NDArray[] op,
            NpyIterGlobalFlags flags,
            NPY_ORDER order,
            NPY_CASTING casting,
            NpyIterPerOpFlags[] opFlags,
            NPTypeCode[]? opDtypes = null,
            int opAxesNDim = -1,
            int[][]? opAxes = null,
            long[]? iterShape = null,
            long bufferSize = 0)
        {
            if (nop < 1)
                throw new ArgumentOutOfRangeException(nameof(nop), "At least one operand is required");

            if (op == null || op.Length < nop)
                throw new ArgumentException("Operand array must contain at least nop elements", nameof(op));

            if (opFlags == null || opFlags.Length < nop)
                throw new ArgumentException("OpFlags array must contain at least nop elements", nameof(opFlags));

            // Allocate state on heap for ref struct lifetime
            var statePtr = (NpyIterState*)NativeMemory.AllocZeroed((nuint)sizeof(NpyIterState));

            try
            {
                var iter = new NpyIterRef
                {
                    _state = statePtr,
                    _ownsState = true,
                    _operands = op,
                };

                iter.Initialize(nop, op, flags, order, casting, opFlags, opDtypes, opAxesNDim, opAxes, iterShape, bufferSize);
                return iter;
            }
            catch
            {
                // Free dimension arrays if they were allocated
                statePtr->FreeDimArrays();
                NativeMemory.Free(statePtr);
                throw;
            }
        }

        private void Initialize(
            int nop,
            NDArray[] op,
            NpyIterGlobalFlags flags,
            NPY_ORDER order,
            NPY_CASTING casting,
            NpyIterPerOpFlags[] opFlags,
            NPTypeCode[]? opDtypes,
            int opAxesNDim,
            int[][]? opAxes,
            long[]? iterShape,
            long bufferSize)
        {
            _state->MaskOp = -1;
            _state->IterStart = 0;

            // Pre-check WRITEMASKED/ARRAYMASK pairing BEFORE allocation (nop arg, not state).
            // The actual MaskOp assignment happens after AllocateDimArrays when NOp is set.
            if (opFlags != null)
                PreCheckMaskOpPairing(nop, opFlags);

            // Calculate broadcast shape, optionally overridden by iterShape
            int[] broadcastShape;
            if (iterShape != null && iterShape.Length > 0)
            {
                // Use explicit iterShape - allows specifying iteration shape different from broadcast
                // NumPy's NpyIter_AdvancedNew() uses this for reductions and custom iteration patterns
                broadcastShape = new int[iterShape.Length];
                for (int i = 0; i < iterShape.Length; i++)
                {
                    broadcastShape[i] = checked((int)iterShape[i]);
                }
                // Validate that operands are compatible with the specified shape
                // Pass opAxes so validation accounts for -1 entries (broadcast/reduce axes)
                ValidateIterShape(nop, op, opFlags, broadcastShape, opAxesNDim, opAxes);
            }
            else
            {
                broadcastShape = CalculateBroadcastShape(nop, op, opFlags, opAxesNDim, opAxes);
                // Validate NO_BROADCAST operands match without stretching
                ValidateIterShape(nop, op, opFlags, broadcastShape, opAxesNDim, opAxes);
            }

            // Allocate null operands that have ALLOCATE flag set.
            // NumPy: npyiter_allocate_arrays in nditer_constr.c
            // Allocated output has shape = broadcastShape (accounting for op_axes)
            // and dtype = opDtypes[opIdx] (required when ALLOCATE is set)
            for (int opIdx = 0; opIdx < nop; opIdx++)
            {
                if (op[opIdx] is null && (opFlags[opIdx] & NpyIterPerOpFlags.ALLOCATE) != 0)
                {
                    if (opDtypes is null || opIdx >= opDtypes.Length)
                        throw new ArgumentException(
                            $"Operand {opIdx} is null with ALLOCATE flag but opDtypes is not provided", nameof(opDtypes));

                    // Determine output shape: for op_axes, filter out -1 entries
                    int[] outputShape;
                    if (opAxes != null && opIdx < opAxes.Length && opAxes[opIdx] != null)
                    {
                        var axisMap = opAxes[opIdx];
                        // Count non-negative entries
                        int realNDim = 0;
                        for (int i = 0; i < axisMap.Length; i++)
                            if (axisMap[i] >= 0) realNDim++;

                        outputShape = new int[realNDim];
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
                        outputShape = (int[])broadcastShape.Clone();
                    }

                    // Allocate the NDArray with specified dtype and shape
                    var shape = outputShape.Length == 0 ? new Shape() : new Shape(outputShape);
                    op[opIdx] = np.zeros(shape, opDtypes[opIdx]);
                }
            }
            // Update _operands so it reflects the allocated arrays
            _operands = op;

            // =========================================================================
            // NUMSHARP DIVERGENCE: Allocate dimension arrays dynamically
            // Unlike NumPy's fixed NPY_MAXDIMS=64, NumSharp supports unlimited dimensions.
            // Arrays are allocated based on actual ndim for memory efficiency.
            // =========================================================================
            _state->AllocateDimArrays(broadcastShape.Length, nop);

            // Set IDENTPERM on construction. Perm starts as identity (set by AllocateDimArrays);
            // reordering (ReorderAxesForCoalescing) and flipping (FlipNegativeStrides) clear
            // this flag when they mutate perm. Matches NumPy nditer_constr.c:262-264.
            _state->ItFlags |= (uint)NpyIterFlags.IDENTPERM;

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
            if (_state->IterSize == 0 && (flags & NpyIterGlobalFlags.ZEROSIZE_OK) == 0)
            {
                // Just allow it anyway for now
            }

            // Determine common dtype if COMMON_DTYPE flag is set
            NPTypeCode? commonDtype = null;
            if ((flags & NpyIterGlobalFlags.COMMON_DTYPE) != 0)
            {
                commonDtype = NpyIterCasting.FindCommonDtype(op, nop);
            }

            // Set up operands
            bool anyNeedsCast = false;
            for (int i = 0; i < nop; i++)
            {
                var arr = op[i];
                var arrShape = arr.Shape;

                // Store source dtype (actual array dtype)
                _state->SetOpSrcDType(i, arr.typecode);

                // Determine buffer/target dtype
                NPTypeCode bufferDtype;
                if (opDtypes != null && i < opDtypes.Length && opDtypes[i] != NPTypeCode.Empty)
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
                _state->SetOpDType(i, bufferDtype);

                // Track if any operand needs casting
                if (arr.typecode != bufferDtype)
                {
                    anyNeedsCast = true;
                }

                // Set operand flags
                var opFlag = TranslateOpFlags(opFlags[i]);

                // If operand needs casting, add CAST flag
                if (arr.typecode != bufferDtype)
                {
                    opFlag |= NpyIterOpFlags.CAST;
                }

                _state->SetOpFlags(i, opFlag);

                // Calculate strides for this operand
                var stridePtr = _state->GetStridesPointer(i);
                byte* basePtr;

                // Check if op_axes is provided for this operand
                if (opAxes != null && i < opAxes.Length && opAxes[i] != null)
                {
                    // Use op_axes mapping to set up strides directly
                    var opAxisMap = opAxes[i];
                    var arrStrides = arrShape.strides;

                    basePtr = (byte*)arr.Address;

                    for (int d = 0; d < _state->NDim; d++)
                    {
                        if (d < opAxisMap.Length)
                        {
                            int opAxis = opAxisMap[d];
                            if (opAxis < 0)
                            {
                                // -1 means broadcast/reduce this dimension
                                stridePtr[d] = 0;
                            }
                            else if (opAxis < arrStrides.Length)
                            {
                                // Use stride from the mapped axis
                                stridePtr[d] = arrStrides[opAxis];
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
                    // Standard broadcasting
                    var broadcastArr = np.broadcast_to(arrShape, new Shape(broadcastShape));
                    basePtr = (byte*)arr.Address + (broadcastArr.offset * arr.dtypesize);

                    for (int d = 0; d < _state->NDim; d++)
                    {
                        stridePtr[d] = broadcastArr.strides[d];
                    }
                }

                _state->SetDataPtr(i, basePtr);
                _state->SetResetDataPtr(i, basePtr);

                // Check for broadcast
                for (int d = 0; d < _state->NDim; d++)
                {
                    if (_state->Shape[d] > 1 && stridePtr[d] == 0)
                    {
                        _state->ItFlags |= (uint)NpyIterFlags.SourceBroadcast;
                        break;
                    }
                }
            }

            // Validate that casting requires BUFFERED flag
            if (anyNeedsCast && (flags & NpyIterGlobalFlags.BUFFERED) == 0)
            {
                throw new ArgumentException(
                    "Casting between different dtypes requires the BUFFERED flag. " +
                    "Add NpyIterGlobalFlags.BUFFERED to enable type conversion.");
            }

            // Validate casting rules
            NpyIterCasting.ValidateCasts(ref *_state, casting);

            // Apply op_axes remapping if provided
            if (opAxes != null && opAxesNDim >= 0)
            {
                ApplyOpAxes(opAxesNDim, opAxes, flags);
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
            bool hasMultiIndex = (flags & NpyIterGlobalFlags.MULTI_INDEX) != 0;
            // HASINDEX (C_INDEX or F_INDEX): need original axis structure preserved
            // to compute the flat index correctly. Coalescing loses this info.
            bool hasFlatIndex = (flags & (NpyIterGlobalFlags.C_INDEX | NpyIterGlobalFlags.F_INDEX)) != 0;

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
            bool isForcedOrder = order == NPY_ORDER.NPY_CORDER
                              || order == NPY_ORDER.NPY_FORTRANORDER
                              || order == NPY_ORDER.NPY_ANYORDER;
            if (!isForcedOrder && (flags & NpyIterGlobalFlags.DONT_NEGATE_STRIDES) == 0)
            {
                NpyIterCoalescing.FlipNegativeStrides(ref *_state);
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
                bool allowFlip = !isForcedOrder && (flags & NpyIterGlobalFlags.DONT_NEGATE_STRIDES) == 0;
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
                if ((order == NPY_ORDER.NPY_KEEPORDER || order == NPY_ORDER.NPY_ANYORDER) && hasBroadcast)
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
                        NpyIterCoalescing.ReorderAxesForCoalescing(ref *_state, NPY_ORDER.NPY_KEEPORDER, forCoalescing: true);
                        NpyIterCoalescing.CoalesceAxes(ref *_state);
                    }
                    else
                    {
                        // Can't coalesce - reorder for the requested iteration order
                        NpyIterCoalescing.ReorderAxesForCoalescing(ref *_state, effectiveOrder, forCoalescing: false);
                    }
                }
                else
                {
                    // With MULTI_INDEX or HASINDEX (C_INDEX/F_INDEX), just reorder axes
                    // without coalescing. Use effectiveOrder which applies K-order → C-order
                    // fallback for non-contiguous arrays.
                    NpyIterCoalescing.ReorderAxesForCoalescing(ref *_state, effectiveOrder, forCoalescing: false);
                }
            }

            // Set external loop flag separately (after coalescing)
            if ((flags & NpyIterGlobalFlags.EXTERNAL_LOOP) != 0)
            {
                _state->ItFlags |= (uint)NpyIterFlags.EXLOOP;
            }

            // Set GROWINNER flag to maximize inner loop size during buffering
            if ((flags & NpyIterGlobalFlags.GROWINNER) != 0)
            {
                _state->ItFlags |= (uint)NpyIterFlags.GROWINNER;
            }

            // Track multi-index if requested
            if ((flags & NpyIterGlobalFlags.MULTI_INDEX) != 0)
            {
                _state->ItFlags |= (uint)NpyIterFlags.HASMULTIINDEX;
            }

            // Track flat index if requested (C_INDEX or F_INDEX)
            if ((flags & NpyIterGlobalFlags.C_INDEX) != 0)
            {
                _state->ItFlags |= (uint)NpyIterFlags.HASINDEX;
                _state->IsCIndex = true;
            }
            else if ((flags & NpyIterGlobalFlags.F_INDEX) != 0)
            {
                _state->ItFlags |= (uint)NpyIterFlags.HASINDEX;
                _state->IsCIndex = false;
            }

            // Compute initial FlatIndex based on current coordinates (handles NEGPERM)
            // Must be called after HASINDEX is set and negative strides are flipped
            _state->InitializeFlatIndex();

            // Update inner strides cache
            // Note: CoalesceAxes calls this internally, but we need to ensure it's
            // called even when coalescing is skipped (NDim <= 1 or MULTI_INDEX set)
            if (_state->NDim <= 1 || (flags & NpyIterGlobalFlags.MULTI_INDEX) != 0)
            {
                _state->UpdateInnerStrides();
            }

            // Update contiguity flags
            UpdateContiguityFlags();

            // Set up buffering if requested
            if ((flags & NpyIterGlobalFlags.BUFFERED) != 0)
            {
                _state->ItFlags |= (uint)NpyIterFlags.BUFFER;
                _state->BufferSize = bufferSize > 0 ? bufferSize : NpyIterBufferManager.DefaultBufferSize;

                // Allocate buffers for each operand
                NpyIterBufferManager.AllocateBuffers(ref *_state, _state->BufferSize);

                // Copy initial data to buffers (with casting if needed)
                long copyCount = Math.Min(_state->IterSize, _state->BufferSize);
                for (int op1 = 0; op1 < nop; op1++)
                {
                    var opFlag = _state->GetOpFlags(op1);
                    if ((opFlag & NpyIterOpFlags.READ) != 0 || (opFlag & NpyIterOpFlags.READWRITE) != 0)
                    {
                        NpyIterBufferManager.CopyToBuffer(ref *_state, op1, copyCount);
                    }
                }

                _state->BufIterEnd = copyCount;

                // Set up buffered reduction if REDUCE flag is also set
                if ((_state->ItFlags & (uint)NpyIterFlags.REDUCE) != 0)
                {
                    SetupBufferedReduction(copyCount);
                }
            }

            // Handle single iteration optimization
            if (_state->IterSize <= 1)
            {
                _state->ItFlags |= (uint)NpyIterFlags.ONEITERATION;
            }
        }

        /// <summary>
        /// Compute iteration (broadcast) shape from operands.
        /// Uses production NumSharp.Shape.ResolveReturnShape for standard broadcasting.
        /// For op_axes, constructs a virtual shape per operand reflecting the mapping,
        /// then broadcasts those virtual shapes together.
        /// </summary>
        private static int[] CalculateBroadcastShape(int nop, NDArray[] op, NpyIterPerOpFlags[] opFlags,
            int opAxesNDim = -1, int[][]? opAxes = null)
        {
            // Validate null operands have ALLOCATE flag
            for (int i = 0; i < nop; i++)
            {
                if (op[i] is null && (opFlags[i] & NpyIterPerOpFlags.ALLOCATE) == 0)
                    throw new ArgumentException($"Operand {i} is null but ALLOCATE flag is not set", nameof(op));
            }

            // With op_axes, iteration ndim is set by opAxesNDim. Each operand's virtual
            // shape per-iter-axis = opShape[op_axis] if op_axis >= 0, else 1.
            if (opAxes != null && opAxesNDim > 0)
            {
                var virtualShapes = new System.Collections.Generic.List<NumSharp.Shape>(nop);
                for (int opIdx = 0; opIdx < nop; opIdx++)
                {
                    if (op[opIdx] is null)
                        continue;  // ALLOCATE operand adopts broadcast result

                    var virtualDims = new long[opAxesNDim];
                    if (opIdx < opAxes.Length && opAxes[opIdx] != null)
                    {
                        var axisMap = opAxes[opIdx];
                        var opShape = op[opIdx].shape;
                        for (int iterAxis = 0; iterAxis < opAxesNDim; iterAxis++)
                        {
                            int rawOpAxis = iterAxis < axisMap.Length ? axisMap[iterAxis] : -1;
                            // Decode NPY_ITER_REDUCTION_AXIS encoding (common.h:347).
                            int opAxis = NpyIterUtils.GetOpAxis(rawOpAxis, out bool isReduction);

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
                                        throw new IncorrectShapeException(
                                            $"Operand {opIdx} reduction axis {opAxis} has length {len}, must be 1.");
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
                        var opShape = op[opIdx].shape;
                        int offset = opAxesNDim - opShape.Length;
                        if (offset < 0)
                            throw new IncorrectShapeException(
                                $"Operand {opIdx} has {opShape.Length} dims but opAxesNDim={opAxesNDim}");
                        for (int d = 0; d < opAxesNDim; d++)
                            virtualDims[d] = d < offset ? 1 : opShape[d - offset];
                    }
                    virtualShapes.Add(new NumSharp.Shape(virtualDims));
                }

                if (virtualShapes.Count == 0)
                    return Array.Empty<int>();

                var resolved = NumSharp.Shape.ResolveReturnShape(virtualShapes.ToArray());
                var dims = resolved.dimensions;
                var result = new int[dims.Length];
                for (int i = 0; i < dims.Length; i++)
                    result[i] = checked((int)dims[i]);
                return result;
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
                return Array.Empty<int>();

            var resolvedShape = NumSharp.Shape.ResolveReturnShape(shapes.ToArray());
            var resultDims = resolvedShape.dimensions;
            var finalResult = new int[resultDims.Length];
            for (int i = 0; i < resultDims.Length; i++)
                finalResult[i] = checked((int)resultDims[i]);
            return finalResult;
        }

        /// <summary>
        /// Validate that operands are compatible with the specified iterShape.
        /// Each operand dimension must either equal the iterShape or be 1 (broadcastable).
        /// When opAxes is provided, -1 entries indicate dimensions that don't need validation.
        /// </summary>
        private static void ValidateIterShape(int nop, NDArray[] op, NpyIterPerOpFlags[] opFlags,
            int[] iterShape, int opAxesNDim, int[][]? opAxes)
        {
            for (int opIdx = 0; opIdx < nop; opIdx++)
            {
                // Skip null (ALLOCATE) operands - they will adopt the iterShape
                if (op[opIdx] is null)
                    continue;

                bool noBroadcast = (opFlags[opIdx] & NpyIterPerOpFlags.NO_BROADCAST) != 0;
                var opShape = op[opIdx].shape;

                // When opAxes is provided for this operand, use it for validation
                if (opAxes != null && opIdx < opAxes.Length && opAxes[opIdx] != null)
                {
                    var opAxisMap = opAxes[opIdx];
                    int mapLength = Math.Min(opAxisMap.Length, iterShape.Length);

                    for (int iterAxis = 0; iterAxis < mapLength; iterAxis++)
                    {
                        // Decode NPY_ITER_REDUCTION_AXIS encoding (common.h:347)
                        int opAxis = NpyIterUtils.GetOpAxis(opAxisMap[iterAxis], out bool isReduction);

                        // Broadcast or reduction-broadcast: no further shape validation needed
                        if (opAxis < 0)
                            continue;

                        // Validate that the operand axis exists and is compatible
                        if (opAxis >= opShape.Length)
                            throw new IncorrectShapeException($"Operand {opIdx} op_axes refers to non-existent axis {opAxis}");

                        // Explicit reduction axis must have length 1 on the operand
                        if (isReduction && opShape[opAxis] != 1)
                            throw new IncorrectShapeException(
                                $"Operand {opIdx} explicit reduction axis {opAxis} has length {opShape[opAxis]}, must be 1.");

                        int opDim = (int)opShape[opAxis];
                        int iterDim = iterShape[iterAxis];

                        // opDim must equal iterDim or be 1 (broadcastable)
                        if (opDim != iterDim && opDim != 1)
                            throw new IncorrectShapeException($"Operand {opIdx} shape incompatible with iterShape at axis {iterAxis}");

                        // NO_BROADCAST: dim of 1 that needs stretching is forbidden
                        if (noBroadcast && opDim == 1 && iterDim != 1)
                            throw new InvalidOperationException(
                                $"non-broadcastable operand with shape ({string.Join(",", opShape)}) " +
                                $"doesn't match the broadcast shape ({string.Join(",", iterShape)})");
                    }
                }
                else
                {
                    // No opAxes for this operand, use standard broadcasting validation
                    int offset = iterShape.Length - opShape.Length;

                    // Operand must have fewer or equal dimensions
                    if (offset < 0)
                        throw new IncorrectShapeException($"Operand {opIdx} has more dimensions than iterShape");

                    // NO_BROADCAST: operand must match iterShape ndim (no prepending of size-1)
                    if (noBroadcast && offset > 0)
                        throw new InvalidOperationException(
                            $"non-broadcastable operand with shape ({string.Join(",", opShape)}) " +
                            $"doesn't match the broadcast shape ({string.Join(",", iterShape)})");

                    for (int d = 0; d < opShape.Length; d++)
                    {
                        int opDim = (int)opShape[d];
                        int iterDim = iterShape[offset + d];

                        // opDim must equal iterDim or be 1 (broadcastable)
                        if (opDim != iterDim && opDim != 1)
                            throw new IncorrectShapeException($"Operand {opIdx} shape incompatible with iterShape at axis {d}");

                        // NO_BROADCAST: dim of 1 that needs stretching is forbidden
                        if (noBroadcast && opDim == 1 && iterDim != 1)
                            throw new InvalidOperationException(
                                $"non-broadcastable operand with shape ({string.Join(",", opShape)}) " +
                                $"doesn't match the broadcast shape ({string.Join(",", iterShape)})");
                    }
                }
            }
        }

        private static NpyIterOpFlags TranslateOpFlags(NpyIterPerOpFlags flags)
        {
            var result = NpyIterOpFlags.None;

            if ((flags & NpyIterPerOpFlags.READONLY) != 0)
                result |= NpyIterOpFlags.READ;
            if ((flags & NpyIterPerOpFlags.WRITEONLY) != 0)
                result |= NpyIterOpFlags.WRITE;
            if ((flags & NpyIterPerOpFlags.READWRITE) != 0)
                result |= NpyIterOpFlags.READWRITE;
            if ((flags & NpyIterPerOpFlags.COPY) != 0)
                result |= NpyIterOpFlags.FORCECOPY;
            if ((flags & NpyIterPerOpFlags.CONTIG) != 0)
                result |= NpyIterOpFlags.CONTIG;
            // WRITEMASKED: the operand is written only where the mask (ARRAYMASK) is true.
            // Requires a corresponding ARRAYMASK operand. NumPy nditer_constr.c:950-965.
            if ((flags & NpyIterPerOpFlags.WRITEMASKED) != 0)
                result |= NpyIterOpFlags.WRITEMASKED;

            return result;
        }

        /// <summary>
        /// Pre-construction check for WRITEMASKED/ARRAYMASK pairing.
        /// Matches NumPy's prepare_operands checks (nditer_constr.c:1176-1230).
        /// Runs before state allocation (uses the raw <paramref name="nop"/> arg).
        /// </summary>
        private static void PreCheckMaskOpPairing(int nop, NpyIterPerOpFlags[] opFlags)
        {
            int maskOp = -1;
            bool anyWriteMasked = false;

            for (int iop = 0; iop < nop && iop < opFlags.Length; iop++)
            {
                bool isArrayMask = (opFlags[iop] & NpyIterPerOpFlags.ARRAYMASK) != 0;
                bool isWriteMasked = (opFlags[iop] & NpyIterPerOpFlags.WRITEMASKED) != 0;

                if (isArrayMask && isWriteMasked)
                    throw new ArgumentException(
                        $"Operand {iop} cannot be both ARRAYMASK and WRITEMASKED.");

                if (isArrayMask)
                {
                    if (maskOp >= 0)
                        throw new ArgumentException(
                            $"At most one operand may be flagged ARRAYMASK " +
                            $"(currently {maskOp} and {iop}).");
                    maskOp = iop;
                }

                if (isWriteMasked) anyWriteMasked = true;
            }

            if (anyWriteMasked && maskOp < 0)
                throw new ArgumentException(
                    "Iterator operand has WRITEMASKED but no operand has ARRAYMASK.");
            if (!anyWriteMasked && maskOp >= 0)
                throw new ArgumentException(
                    $"Operand {maskOp} has ARRAYMASK but no operand has WRITEMASKED.");
        }

        /// <summary>
        /// Sets <see cref="NpyIterState.MaskOp"/> from the ARRAYMASK operand (if any).
        /// Pre-validated by <see cref="PreCheckMaskOpPairing"/>.
        /// </summary>
        private void SetMaskOpFromFlags(NpyIterPerOpFlags[] opFlags)
        {
            for (int iop = 0; iop < _state->NOp && iop < opFlags.Length; iop++)
            {
                if ((opFlags[iop] & NpyIterPerOpFlags.ARRAYMASK) != 0)
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
                _state->ItFlags |= (uint)(NpyIterFlags.SourceContiguous | NpyIterFlags.DestinationContiguous | NpyIterFlags.CONTIGUOUS);
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
                _state->ItFlags |= (uint)NpyIterFlags.CONTIGUOUS;

            // Set legacy flags for first two operands
            if (_state->NOp >= 1)
            {
                var stridePtr = _state->GetStridesPointer(0);
                if (CheckContiguous(_state->GetShapePointer(), stridePtr, _state->NDim))
                    _state->ItFlags |= (uint)NpyIterFlags.SourceContiguous;
            }

            if (_state->NOp >= 2)
            {
                var stridePtr = _state->GetStridesPointer(1);
                if (CheckContiguous(_state->GetShapePointer(), stridePtr, _state->NDim))
                    _state->ItFlags |= (uint)NpyIterFlags.DestinationContiguous;
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
            if (_operands is null)
                return false;

            for (int op = 0; op < _state->NOp; op++)
            {
                var arr = _operands[op];
                if (arr is null)
                    continue;

                // Check if operand is contiguous in the requested order
                var arrShape = arr.shape;
                if (arr.ndim == 0 || arr.size <= 1)
                    continue;  // Trivially contiguous

                // Get strides from the original array
                var strides = arr.strides;

                // Check contiguity using actual strides.
                // Negative strides are only treated as "contiguous" when FlipNegativeStrides
                // will run (K-order / A-order without FORCEDORDER). For forced C/F order,
                // negative strides break contiguity because the iterator will traverse
                // logical order, not memory order.
                long expected = 1;
                if (cOrder)
                {
                    // C-order: last axis fastest, check from end to start
                    for (int axis = arr.ndim - 1; axis >= 0; axis--)
                    {
                        long dim = arrShape[axis];
                        if (dim == 1)
                            continue;  // Size-1 dimensions are always contiguous
                        // Check stride (abs if flipping, actual if not)
                        long stride = allowFlip ? Math.Abs(strides[axis]) : strides[axis];
                        if (stride != expected)
                            return false;
                        expected *= dim;
                    }
                }
                else
                {
                    // F-order: first axis fastest, check from start to end
                    for (int axis = 0; axis < arr.ndim; axis++)
                    {
                        long dim = arrShape[axis];
                        if (dim == 1)
                            continue;  // Size-1 dimensions are always contiguous
                        // Check stride (abs if flipping, actual if not)
                        long stride = allowFlip ? Math.Abs(strides[axis]) : strides[axis];
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
                    if ((opFlags & NpyIterOpFlags.REDUCE) != 0)
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
        /// Validate op_axes mappings and set reduction flags where applicable.
        /// Strides are already correctly set in the main operand setup loop - this method
        /// only handles the reduction semantics (detecting reduce axes, validating REDUCE_OK).
        /// </summary>
        private void ApplyOpAxes(int opAxesNDim, int[][] opAxes, NpyIterGlobalFlags globalFlags)
        {
            if (opAxes == null || opAxesNDim <= 0)
                return;

            int iterNDim = Math.Min(opAxesNDim, _state->NDim);
            bool reduceOkSet = (globalFlags & NpyIterGlobalFlags.REDUCE_OK) != 0;

            for (int op = 0; op < _state->NOp; op++)
            {
                // Skip if no mapping for this operand
                if (op >= opAxes.Length || opAxes[op] == null)
                    continue;

                var opAxisMap = opAxes[op];
                var opFlags = _state->GetOpFlags(op);
                // Check if WRITE flag is set (includes both WRITE-only and READWRITE)
                bool isWriteable = (opFlags & NpyIterOpFlags.WRITE) != 0;
                bool hasReductionAxis = false;

                // Scan for reduction axes (op_axis=-1 on a writeable operand,
                // OR explicit encoding via NpyIterUtils.ReductionAxis).
                for (int iterAxis = 0; iterAxis < iterNDim && iterAxis < opAxisMap.Length; iterAxis++)
                {
                    int rawOpAxis = opAxisMap[iterAxis];
                    int opAxis = NpyIterUtils.GetOpAxis(rawOpAxis, out bool explicitReduction);

                    if (explicitReduction)
                    {
                        // Explicit reduction axis: must be READWRITE and REDUCE_OK set.
                        // NumPy nditer_constr.c:1621-1638 additionally validates operand's
                        // axis length is exactly 1; that check is handled during broadcast
                        // shape resolution via CalculateBroadcastShape.
                        if (!reduceOkSet)
                        {
                            throw new ArgumentException(
                                $"Operand {op} uses an explicit reduction axis at iter dim {iterAxis}, " +
                                "but REDUCE_OK is not set. Add NpyIterGlobalFlags.REDUCE_OK.");
                        }

                        hasReductionAxis = true;
                    }
                    else if (opAxis < 0)
                    {
                        // Implicit reduction or broadcast: op_axis = -1
                        if (isWriteable && _state->Shape[iterAxis] > 1)
                        {
                            hasReductionAxis = true;

                            if (!reduceOkSet)
                            {
                                throw new ArgumentException(
                                    $"Output operand {op} requires a reduction along dimension {iterAxis}, " +
                                    "but the reduction is not enabled. " +
                                    "Add NpyIterGlobalFlags.REDUCE_OK to allow reduction.");
                            }
                        }
                        else
                        {
                            // Read-only operand with stride=0 is a broadcast
                            _state->ItFlags |= (uint)NpyIterFlags.SourceBroadcast;
                        }
                    }
                }

                // Set reduction flags if this operand has reduction axes
                if (hasReductionAxis)
                {
                    // NumPy requires READWRITE, not WRITEONLY for reduction operands
                    // because reduction must read existing value before accumulating
                    if ((opFlags & NpyIterOpFlags.READ) == 0)
                    {
                        throw new ArgumentException(
                            $"Output operand {op} requires a reduction, but is flagged as " +
                            "write-only, not read-write. Use READWRITE instead of WRITEONLY.");
                    }

                    _state->ItFlags |= (uint)NpyIterFlags.REDUCE;
                    _state->SetOpFlags(op, opFlags | NpyIterOpFlags.REDUCE);

                    // If this reduction operand is also WRITEMASKED, enforce the
                    // "one mask value per reduction element" constraint.
                    // NumPy: check_mask_for_writemasked_reduction (nditer_constr.c:1328).
                    if ((opFlags & NpyIterOpFlags.WRITEMASKED) != 0)
                    {
                        CheckMaskForWriteMaskedReduction(op);
                    }
                }
            }
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
                    if ((_state->GetOpFlags(iop) & NpyIterOpFlags.WRITEMASKED) != 0)
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
        public bool RequiresBuffering => (_state->ItFlags & (uint)NpyIterFlags.BUFFER) != 0;

        /// <summary>Whether all operands are contiguous.</summary>
        public bool IsContiguous => (_state->ItFlags & (uint)NpyIterFlags.CONTIGUOUS) != 0;

        /// <summary>Whether iterator has external loop.</summary>
        public bool HasExternalLoop => (_state->ItFlags & (uint)NpyIterFlags.EXLOOP) != 0;

        /// <summary>Whether iterator uses GROWINNER optimization for buffering.</summary>
        public bool HasGrowInner => (_state->ItFlags & (uint)NpyIterFlags.GROWINNER) != 0;

        // =========================================================================
        // Iteration Methods
        // =========================================================================

        /// <summary>
        /// Get the iteration-advance function.
        /// </summary>
        public NpyIterNextFunc GetIterNext()
        {
            if (_cachedIterNext != null)
                return _cachedIterNext;

            var itflags = (NpyIterFlags)_state->ItFlags;

            if ((itflags & NpyIterFlags.ONEITERATION) != 0)
                _cachedIterNext = SingleIterationNext;
            else if ((itflags & NpyIterFlags.EXLOOP) != 0)
                _cachedIterNext = ExternalLoopNext;
            else
                _cachedIterNext = StandardNext;

            return _cachedIterNext;
        }

        private static bool SingleIterationNext(ref NpyIterState state)
        {
            if (state.IterIndex >= state.IterEnd)
                return false;
            state.IterIndex = state.IterEnd;
            return false;
        }

        private static bool ExternalLoopNext(ref NpyIterState state)
        {
            // For external loop, we advance outer dimensions
            // Inner dimension is handled by caller
            if (state.IterIndex >= state.IterEnd)
                return false;

            state.IterIndex += state.Shape[state.NDim - 1];

            if (state.IterIndex >= state.IterEnd)
                return false;

            // Advance outer coordinates
            for (int axis = state.NDim - 2; axis >= 0; axis--)
            {
                state.Coords[axis]++;

                if (state.Coords[axis] < state.Shape[axis])
                {
                    // Update data pointers
                    for (int op = 0; op < state.NOp; op++)
                    {
                        long stride = state.GetStride(axis, op);
                        state.DataPtrs[op] += stride * state.ElementSizes[op];
                    }
                    return true;
                }

                // Carry
                state.Coords[axis] = 0;
                for (int op = 0; op < state.NOp; op++)
                {
                    long stride = state.GetStride(axis, op);
                    state.DataPtrs[op] -= stride * (state.Shape[axis] - 1) * state.ElementSizes[op];
                }
            }

            return true;
        }

        private static bool StandardNext(ref NpyIterState state)
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
            if ((_state->ItFlags & (uint)NpyIterFlags.BUFFER) != 0)
                return &_state->BufIterEnd;

            // Return pointer to innermost shape dimension
            return &_state->Shape[_state->NDim - 1];
        }

        /// <summary>
        /// Reset iterator to the beginning.
        /// </summary>
        public bool Reset()
        {
            _state->Reset();
            return true;
        }

        /// <summary>
        /// Fetch the NpyArrayMethodFlags (runtime) flags for all transfer functions
        /// (i.e. copy to buffer/casts). Matches NumPy's NpyIter_GetTransferFlags
        /// (nditer_api.c:903). Decoded from the top 8 bits of ItFlags.
        ///
        /// In .NET context, REQUIRES_PYAPI is never set — included for API parity only.
        /// </summary>
        public NpyArrayMethodFlags GetTransferFlags()
        {
            return (NpyArrayMethodFlags)(_state->ItFlags >> NpyIterConstants.TRANSFERFLAGS_SHIFT);
        }

        /// <summary>
        /// Copies the array of strides that are fixed during iteration into <paramref name="outStrides"/>.
        /// Matches NumPy's NpyIter_GetInnerFixedStrideArray (nditer_api.c:1357).
        ///
        /// - Buffered: copies <see cref="NpyIterState.BufStrides"/> (one entry per operand).
        /// - Non-buffered: copies the innermost-axis stride from <see cref="NpyIterState.Strides"/>
        ///   (equivalent to NumPy's NAD_STRIDES(axisdata[0]) in its reverse-C ordering).
        ///
        /// Once the iterator is ready to iterate, call this to obtain strides guaranteed
        /// not to change between inner-loop iterations — enabling the caller to choose an
        /// optimized inner loop function.
        ///
        /// GIL-safe (no allocation, no exceptions under valid inputs).
        /// </summary>
        /// <param name="outStrides">Output span of length ≥ NOp.</param>
        /// <summary>
        /// Dumps a verbose textual representation of the iterator's internal state to
        /// the specified TextWriter. Matches NumPy's NpyIter_DebugPrint (nditer_api.c:1402)
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
            if ((itf & (uint)NpyIterFlags.IDENTPERM) != 0) writer.Write("IDENTPERM ");
            if ((itf & (uint)NpyIterFlags.NEGPERM) != 0) writer.Write("NEGPERM ");
            if ((itf & (uint)NpyIterFlags.HASINDEX) != 0) writer.Write("HASINDEX ");
            if ((itf & (uint)NpyIterFlags.HASMULTIINDEX) != 0) writer.Write("HASMULTIINDEX ");
            if ((itf & (uint)NpyIterFlags.FORCEDORDER) != 0) writer.Write("FORCEDORDER ");
            if ((itf & (uint)NpyIterFlags.EXLOOP) != 0) writer.Write("EXLOOP ");
            if ((itf & (uint)NpyIterFlags.RANGE) != 0) writer.Write("RANGE ");
            if ((itf & (uint)NpyIterFlags.BUFFER) != 0) writer.Write("BUFFER ");
            if ((itf & (uint)NpyIterFlags.GROWINNER) != 0) writer.Write("GROWINNER ");
            if ((itf & (uint)NpyIterFlags.ONEITERATION) != 0) writer.Write("ONEITERATION ");
            if ((itf & (uint)NpyIterFlags.DELAYBUF) != 0) writer.Write("DELAYBUF ");
            if ((itf & (uint)NpyIterFlags.REDUCE) != 0) writer.Write("REDUCE ");
            if ((itf & (uint)NpyIterFlags.REUSE_REDUCE_LOOPS) != 0) writer.Write("REUSE_REDUCE_LOOPS ");
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
                writer.Write($"{dt.AsNumpyDtypeName()} ");
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

            if ((itf & (uint)NpyIterFlags.HASINDEX) != 0)
                writer.WriteLine($"| FlatIndex: {_state->FlatIndex}");

            // OpItFlags
            writer.WriteLine("| OpItFlags:");
            for (int iop = 0; iop < nop; iop++)
            {
                writer.Write($"|   Flags[{iop}]: ");
                var of = _state->GetOpFlags(iop);
                if ((of & NpyIterOpFlags.READ) != 0) writer.Write("READ ");
                if ((of & NpyIterOpFlags.WRITE) != 0) writer.Write("WRITE ");
                if ((of & NpyIterOpFlags.CAST) != 0) writer.Write("CAST ");
                if ((of & NpyIterOpFlags.BUFNEVER) != 0) writer.Write("BUFNEVER ");
                if ((of & NpyIterOpFlags.REDUCE) != 0) writer.Write("REDUCE ");
                if ((of & NpyIterOpFlags.VIRTUAL) != 0) writer.Write("VIRTUAL ");
                if ((of & NpyIterOpFlags.WRITEMASKED) != 0) writer.Write("WRITEMASKED ");
                if ((of & NpyIterOpFlags.BUF_SINGLESTRIDE) != 0) writer.Write("BUF_SINGLESTRIDE ");
                if ((of & NpyIterOpFlags.CONTIG) != 0) writer.Write("CONTIG ");
                if ((of & NpyIterOpFlags.BUF_REUSABLE) != 0) writer.Write("BUF_REUSABLE ");
                writer.WriteLine();
            }
            writer.WriteLine("|");

            // Buffer data
            if ((itf & (uint)NpyIterFlags.BUFFER) != 0)
            {
                writer.WriteLine("| BufferData:");
                writer.WriteLine($"|   BufferSize: {_state->BufferSize}");
                writer.WriteLine($"|   BufIterEnd: {_state->BufIterEnd}");
                writer.WriteLine($"|   CoreSize: {_state->CoreSize}");
                if ((itf & (uint)NpyIterFlags.REDUCE) != 0)
                {
                    writer.WriteLine($"|   REDUCE Pos: {_state->ReducePos}");
                    writer.WriteLine($"|   REDUCE OuterSize: {_state->ReduceOuterSize}");
                    writer.WriteLine($"|   REDUCE OuterDim: {_state->OuterDim}");
                }
                writer.Write("|   BufStrides: ");
                for (int iop = 0; iop < nop; iop++)
                    writer.Write($"{_state->BufStrides[iop]} ");
                writer.WriteLine();
                if ((itf & (uint)NpyIterFlags.REDUCE) != 0)
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
        /// Matches NumPy's NpyIter_CreateCompatibleStrides (nditer_api.c:1058).
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
            if ((_state->ItFlags & (uint)NpyIterFlags.HASMULTIINDEX) == 0)
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
        /// Matches NumPy's NpyIter_GetAxisStrideArray (nditer_api.c:1309).
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

            if ((_state->ItFlags & (uint)NpyIterFlags.HASMULTIINDEX) != 0)
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

            // Return byte strides (NumPy convention); internal strides are element counts.
            for (int op = 0; op < nop; op++)
            {
                long elemStride = _state->Strides[op * stridesNDim + internalIdim];
                outStrides[op] = elemStride * _state->ElementSizes[op];
            }
        }

        public void GetInnerFixedStrideArray(scoped Span<long> outStrides)
        {
            if (outStrides.Length < _state->NOp)
                throw new ArgumentException(
                    $"outStrides must have at least {_state->NOp} elements.", nameof(outStrides));

            int nop = _state->NOp;

            if ((_state->ItFlags & (uint)NpyIterFlags.BUFFER) != 0)
            {
                // Buffered: BufStrides already stored in bytes (NpyIterBufferManager assigns
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
                        outStrides[op] = elemStride * _state->ElementSizes[op];
                    }
                }
            }
        }

        /// <summary>
        /// Resets the iterator to its initial state with new base data pointers.
        /// Matches NumPy's NpyIter_ResetBasePointers (nditer_api.c:314).
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
            if ((itFlags & (uint)NpyIterFlags.BUFFER) != 0)
            {
                if ((itFlags & (uint)NpyIterFlags.DELAYBUF) != 0)
                {
                    // Delayed buffer allocation: allocate now
                    if (!NpyIterBufferManager.AllocateBuffers(ref *_state, _state->BufferSize))
                    {
                        return false;
                    }
                    _state->ItFlags &= ~(uint)NpyIterFlags.DELAYBUF;
                }
                else
                {
                    // Flush any pending writes before replacing pointers
                    CopyReduceBuffersToArrays();
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
            if ((itFlags & (uint)NpyIterFlags.BUFFER) != 0)
            {
                long remaining = _state->IterEnd - _state->IterIndex;
                long copyCount = Math.Min(remaining, _state->BufferSize);
                if (copyCount > 0)
                {
                    for (int iop = 0; iop < _state->NOp; iop++)
                    {
                        var opFlags = _state->GetOpFlags(iop);
                        if ((opFlags & NpyIterOpFlags.READ) != 0)
                        {
                            NpyIterBufferManager.CopyToBuffer(ref *_state, iop, copyCount);
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
                byte* basePtr = (byte*)arr.Address + (arr.Shape.offset * arr.dtypesize);
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
            if ((itFlags & (uint)NpyIterFlags.BUFFER) != 0 &&
                (itFlags & (uint)NpyIterFlags.REDUCE) != 0 &&
                _state->CoreSize > 0)
            {
                return BufferedReduceIternext();
            }

            _state->Advance();
            return _state->IterIndex < _state->IterEnd;
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
                if ((opFlags & NpyIterOpFlags.REDUCE) != 0)
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

                if ((opFlags & NpyIterOpFlags.READ) != 0 || (opFlags & NpyIterOpFlags.READWRITE) != 0)
                {
                    NpyIterBufferManager.CopyToBuffer(ref *_state, op, copyCount);
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
                if ((opFlags & NpyIterOpFlags.WRITE) == 0 && (opFlags & NpyIterOpFlags.READWRITE) == 0)
                    continue;

                var buffer = _state->GetBuffer(op);
                if (buffer == null)
                    continue;

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
                if ((opFlags & NpyIterOpFlags.REDUCE) != 0)
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
                if ((opFlags & NpyIterOpFlags.REDUCE) != 0)
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
                    NpyIterBufferManager.CopyFromBuffer(ref *_state, op, copyCount);
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
        /// <returns>True if range is valid, false otherwise</returns>
        public bool ResetToIterIndexRange(long start, long end)
        {
            if (start < 0 || end > _state->IterSize || start > end)
                return false;

            _state->IterStart = start;
            _state->IterEnd = end;
            _state->ItFlags |= (uint)NpyIterFlags.RANGE;

            GotoIterIndex(start);
            return true;
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
        public bool IsRanged => (_state->ItFlags & (uint)NpyIterFlags.RANGE) != 0;

        /// <summary>
        /// Jump to a specific iteration index.
        /// </summary>
        public void GotoIterIndex(long iterindex)
        {
            _state->GotoIterIndex(iterindex);
        }

        /// <summary>
        /// Returns a specialized delegate for computing multi-index based on iterator flags.
        /// Matches NumPy's NpyIter_GetGetMultiIndex (nditer_templ.c.src:481).
        ///
        /// NumPy generates 12 specializations on (HASINDEX × IDENTPERM × NEGPERM × BUFFER).
        /// NumSharp dispatches to 3 variants (BUFFER and HASINDEX don't affect coords):
        ///   1. IDENTPERM — direct copy of internal coords
        ///   2. Positive perm — apply perm[] mapping
        ///   3. NEGPERM — apply perm[] with flip decoding
        ///
        /// The returned delegate takes raw NpyIterState and a pointer to output coords.
        /// </summary>
        /// <param name="errmsg">Set on failure; null on success.</param>
        /// <returns>Delegate, or null if iterator is not tracking multi-index.</returns>
        public NpyIterGetMultiIndexFunc? GetMultiIndexFunc(out string? errmsg)
        {
            errmsg = null;
            if ((_state->ItFlags & (uint)NpyIterFlags.HASMULTIINDEX) == 0)
            {
                errmsg = "Iterator not tracking multi-index. Use NpyIterGlobalFlags.MULTI_INDEX during construction.";
                return null;
            }

            uint itf = _state->ItFlags;
            if ((itf & (uint)NpyIterFlags.IDENTPERM) != 0)
                return GetMultiIndex_Identity;
            if ((itf & (uint)NpyIterFlags.NEGPERM) != 0)
                return GetMultiIndex_NegPerm;
            return GetMultiIndex_PosPerm;
        }

        /// <summary>
        /// Returns a specialized delegate for computing multi-index.
        /// Matches NumPy's NpyIter_GetGetMultiIndex. Throws on failure instead of
        /// returning null (thin wrapper over the out-errmsg overload).
        /// </summary>
        public NpyIterGetMultiIndexFunc GetMultiIndexFunc()
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
        public void InvokeMultiIndex(NpyIterGetMultiIndexFunc fn, long* outCoords)
        {
            if (fn == null) throw new ArgumentNullException(nameof(fn));
            fn(ref *_state, outCoords);
        }

        /// <summary>
        /// Span overload of <see cref="InvokeMultiIndex"/>.
        /// </summary>
        public void InvokeMultiIndex(NpyIterGetMultiIndexFunc fn, scoped Span<long> outCoords)
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

        private static void GetMultiIndex_Identity(ref NpyIterState state, long* outCoords)
        {
            for (int d = 0; d < state.NDim; d++)
                outCoords[d] = state.Coords[d];
        }

        private static void GetMultiIndex_PosPerm(ref NpyIterState state, long* outCoords)
        {
            for (int d = 0; d < state.NDim; d++)
            {
                int p = state.Perm[d];
                outCoords[p] = state.Coords[d];
            }
        }

        private static void GetMultiIndex_NegPerm(ref NpyIterState state, long* outCoords)
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
            if ((_state->ItFlags & (uint)NpyIterFlags.HASMULTIINDEX) == 0)
                throw new InvalidOperationException("Iterator not tracking multi-index. Use NpyIterGlobalFlags.MULTI_INDEX during construction.");

            if (outCoords.Length < _state->NDim)
                throw new ArgumentException($"Output span must have at least {_state->NDim} elements", nameof(outCoords));

            // Fast path: IDENTPERM means perm is identity (no reordering or flipping)
            if ((_state->ItFlags & (uint)NpyIterFlags.IDENTPERM) != 0)
            {
                for (int d = 0; d < _state->NDim; d++)
                    outCoords[d] = _state->Coords[d];
                return;
            }

            // Apply permutation: Perm[internal_axis] = original_axis (or -1-original if flipped)
            // When perm[d] >= 0: outCoords[perm[d]] = Coords[d]
            // When perm[d] < 0: original = -1 - perm[d], and coordinate is flipped
            bool hasNegPerm = (_state->ItFlags & (uint)NpyIterFlags.NEGPERM) != 0;

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
            if ((_state->ItFlags & (uint)NpyIterFlags.HASMULTIINDEX) == 0)
                throw new InvalidOperationException("Iterator not tracking multi-index. Use NpyIterGlobalFlags.MULTI_INDEX during construction.");

            if (coords.Length < _state->NDim)
                throw new ArgumentException($"Coordinates must have at least {_state->NDim} elements", nameof(coords));

            bool hasNegPerm = (_state->ItFlags & (uint)NpyIterFlags.NEGPERM) != 0;

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

            // Update flat index if tracking (C_INDEX or F_INDEX)
            // Note: C_INDEX/F_INDEX are computed in ORIGINAL array order, not iteration order
            // The coords provided by the user are in original order, so use them directly
            if ((_state->ItFlags & (uint)NpyIterFlags.HASINDEX) != 0)
            {
                // Build original shape for index computation (handle NEGPERM)
                var origShape = stackalloc long[_state->NDim];
                for (int d = 0; d < _state->NDim; d++)
                {
                    int p = _state->Perm[d];
                    int origAxis = (hasNegPerm && p < 0) ? (-1 - p) : p;
                    origShape[origAxis] = _state->Shape[d];
                }

                if (_state->IsCIndex)
                {
                    // C-order flat index in original array
                    long cIndex = 0;
                    multiplier = 1;
                    for (int d = _state->NDim - 1; d >= 0; d--)
                    {
                        cIndex += coords[d] * multiplier;
                        multiplier *= origShape[d];
                    }
                    _state->FlatIndex = cIndex;
                }
                else
                {
                    // F-order flat index in original array
                    long fIndex = 0;
                    multiplier = 1;
                    for (int d = 0; d < _state->NDim; d++)
                    {
                        fIndex += coords[d] * multiplier;
                        multiplier *= origShape[d];
                    }
                    _state->FlatIndex = fIndex;
                }
            }

            // Update data pointers using internal coordinates
            for (int op = 0; op < _state->NOp; op++)
            {
                long offset = 0;
                for (int d = 0; d < _state->NDim; d++)
                    offset += _state->Coords[d] * _state->GetStride(d, op);

                _state->DataPtrs[op] = _state->ResetDataPtrs[op] + offset * _state->ElementSizes[op];
            }
        }

        /// <summary>
        /// Check if iterator is tracking multi-index.
        /// </summary>
        public bool HasMultiIndex => (_state->ItFlags & (uint)NpyIterFlags.HASMULTIINDEX) != 0;

        /// <summary>
        /// Check if iterator is tracking a flat index.
        /// </summary>
        public bool HasIndex => (_state->ItFlags & (uint)NpyIterFlags.HASINDEX) != 0;

        /// <summary>
        /// Check if any axes have negative permutation entries (flipped for memory-order iteration).
        /// When NEGPERM is set, GetMultiIndex reverses indices for those axes.
        /// </summary>
        public bool HasNegPerm => (_state->ItFlags & (uint)NpyIterFlags.NEGPERM) != 0;

        /// <summary>
        /// Check if the axis permutation is identity (no reordering).
        /// Mutually exclusive with NEGPERM - if NEGPERM is set, IDENTPERM is cleared.
        /// </summary>
        public bool HasIdentPerm => (_state->ItFlags & (uint)NpyIterFlags.IDENTPERM) != 0;

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

                if ((_state->ItFlags & (uint)NpyIterFlags.HASMULTIINDEX) != 0)
                {
                    bool hasNegPerm = (_state->ItFlags & (uint)NpyIterFlags.NEGPERM) != 0;

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
            if ((_state->ItFlags & (uint)NpyIterFlags.HASINDEX) == 0)
                throw new InvalidOperationException("Iterator not tracking index. Use NpyIterGlobalFlags.C_INDEX or F_INDEX during construction.");

            return _state->FlatIndex;
        }

        /// <summary>
        /// Jump to a specific flat index position (C or F order based on construction flags).
        /// Requires C_INDEX or F_INDEX flag to be set during construction.
        /// Matches NumPy's NpyIter_GotoIndex behavior.
        /// When NEGPERM is set, handles flipped axes correctly.
        /// </summary>
        /// <param name="flatIndex">The flat index in C or F order (depending on flags)</param>
        public void GotoIndex(long flatIndex)
        {
            if ((_state->ItFlags & (uint)NpyIterFlags.HASINDEX) == 0)
                throw new InvalidOperationException("Iterator not tracking index. Use NpyIterGlobalFlags.C_INDEX or F_INDEX during construction.");

            if (flatIndex < 0 || flatIndex >= _state->IterSize)
                throw new IndexOutOfRangeException($"Flat index {flatIndex} out of range [0, {_state->IterSize})");

            bool hasNegPerm = (_state->ItFlags & (uint)NpyIterFlags.NEGPERM) != 0;

            // Get original shape (using Perm to map internal to original)
            // Handle NEGPERM: when perm[d] < 0, originalAxis = -1 - perm[d]
            var origShape = stackalloc long[_state->NDim];
            for (int d = 0; d < _state->NDim; d++)
            {
                int p = _state->Perm[d];
                int origAxis = (hasNegPerm && p < 0) ? (-1 - p) : p;
                origShape[origAxis] = _state->Shape[d];
            }

            // Convert flat index to original coordinates
            var coords = stackalloc long[_state->NDim];
            long remaining = flatIndex;

            if (_state->IsCIndex)
            {
                // C-order: last axis changes fastest
                for (int d = _state->NDim - 1; d >= 0; d--)
                {
                    coords[d] = remaining % origShape[d];
                    remaining /= origShape[d];
                }
            }
            else
            {
                // F-order: first axis changes fastest
                for (int d = 0; d < _state->NDim; d++)
                {
                    coords[d] = remaining % origShape[d];
                    remaining /= origShape[d];
                }
            }

            // Update state
            _state->FlatIndex = flatIndex;

            // Convert original coords to internal coords and update position
            // Handle NEGPERM: flipped axes need reversed coordinates
            long iterIndex = 0;
            long multiplier = 1;

            for (int d = _state->NDim - 1; d >= 0; d--)
            {
                int p = _state->Perm[d];
                int origAxis;
                long coord;

                if (hasNegPerm && p < 0)
                {
                    // Flipped axis: map original coord to internal (flipped)
                    origAxis = -1 - p;
                    coord = _state->Shape[d] - coords[origAxis] - 1;
                }
                else
                {
                    origAxis = p;
                    coord = coords[origAxis];
                }

                _state->Coords[d] = coord;
                iterIndex += coord * multiplier;
                multiplier *= _state->Shape[d];
            }

            _state->IterIndex = iterIndex;

            // Update data pointers
            for (int op = 0; op < _state->NOp; op++)
            {
                long offset = 0;
                for (int d = 0; d < _state->NDim; d++)
                    offset += _state->Coords[d] * _state->GetStride(d, op);

                _state->DataPtrs[op] = _state->ResetDataPtrs[op] + offset * _state->ElementSizes[op];
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
        /// Matches NumPy's NpyIter_GetIterView behavior.
        /// </summary>
        /// <param name="operand">The operand index (0 to NOp-1)</param>
        /// <returns>An NDArray view with the iterator's internal shape and strides</returns>
        public NDArray GetIterView(int operand)
        {
            if ((uint)operand >= (uint)_state->NOp)
                throw new ArgumentOutOfRangeException(nameof(operand), $"Operand index {operand} out of range [0, {_state->NOp})");

            // Cannot provide views when buffering is enabled (data may be in temporary buffers)
            if ((_state->ItFlags & (uint)NpyIterFlags.BUFFER) != 0)
                throw new InvalidOperationException("Cannot provide an iterator view when buffering is enabled");

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

            // Calculate the offset from storage base to the reset data pointer
            int elementSize = _state->GetElementSize(operand);
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
            if ((itFlags & (uint)NpyIterFlags.BUFFER) != 0)
            {
                var buffer = _state->GetBuffer(operand);
                if (buffer != null)
                {
                    // REDUCE mode: DataPtrs track the current array/buffer position.
                    // - With CoreSize > 0 (double-loop active): BufferedReduceAdvance maintains DataPtrs.
                    // - With CoreSize == 0 (fallback to regular Advance): DataPtrs maintained by
                    //   Advance() using per-axis strides (stride=0 on reduce axis keeps pointer fixed).
                    // In both cases, DataPtrs is correct; don't override via IterIndex-indexed buffer.
                    if ((itFlags & (uint)NpyIterFlags.REDUCE) != 0)
                    {
                        return _state->GetDataPtr(operand);
                    }

                    // For simple buffered iteration (non-reduce), compute from IterIndex
                    // (IterIndex directly maps to buffer position within current buffer)
                    int elemSize = _state->GetElementSize(operand);
                    long bufferPos = _state->IterIndex - (_state->BufIterEnd - Math.Min(_state->BufferSize, _state->IterSize - _state->IterStart));
                    if (bufferPos < 0) bufferPos = _state->IterIndex;
                    return (byte*)buffer + bufferPos * elemSize;
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
        /// Remove axis from iteration (enables external loop for that axis).
        /// Matches NumPy's NpyIter_RemoveAxis behavior.
        /// </summary>
        public bool RemoveAxis(int axis)
        {
            if (axis < 0 || axis >= _state->NDim)
                return false;

            // Shift dimensions down
            for (int d = axis; d < _state->NDim - 1; d++)
            {
                _state->Shape[d] = _state->Shape[d + 1];
                _state->Coords[d] = _state->Coords[d + 1];

                for (int op = 0; op < _state->NOp; op++)
                {
                    _state->SetStride(d, op, _state->GetStride(d + 1, op));
                }
            }

            _state->NDim--;

            // Recalculate itersize based on remaining shape
            _state->IterSize = 1;
            for (int d = 0; d < _state->NDim; d++)
                _state->IterSize *= _state->Shape[d];
            _state->IterEnd = _state->IterSize;

            // Update inner strides cache after dimension change
            _state->UpdateInnerStrides();

            return true;
        }

        /// <summary>
        /// Remove multi-index tracking and enable coalescing.
        /// Matches NumPy's NpyIter_RemoveMultiIndex behavior.
        /// Note: Resets iterator position to the beginning.
        /// </summary>
        public bool RemoveMultiIndex()
        {
            if ((_state->ItFlags & (uint)NpyIterFlags.HASMULTIINDEX) == 0)
                return false;

            // Clear the multi-index flag
            _state->ItFlags &= ~(uint)NpyIterFlags.HASMULTIINDEX;

            // Perform axis reordering and coalescing now that multi-index is disabled
            // This matches NumPy behavior: when MULTI_INDEX is set during construction,
            // axis reordering is skipped. RemoveMultiIndex enables both reordering and coalescing.
            if (_state->NDim > 1)
            {
                // Step 1: Reorder axes by stride (smallest first = innermost in memory)
                NpyIterCoalescing.ReorderAxesForCoalescing(ref *_state, NPY_ORDER.NPY_KEEPORDER);

                // Step 2: Coalesce adjacent axes that have compatible strides
                NpyIterCoalescing.CoalesceAxes(ref *_state);
            }

            // Reset iterator to beginning (NumPy behavior)
            _state->Reset();

            // Clear cached iteration function
            _cachedIterNext = null;

            return true;
        }

        /// <summary>
        /// Enable external loop handling.
        /// </summary>
        public bool EnableExternalLoop()
        {
            _state->ItFlags |= (uint)NpyIterFlags.EXLOOP;
            _cachedIterNext = null;
            return true;
        }

        // =========================================================================
        // Reduction Support
        // =========================================================================

        /// <summary>
        /// Check if iteration includes reduction operands.
        /// </summary>
        public bool IsReduction => (_state->ItFlags & (uint)NpyIterFlags.REDUCE) != 0;

        /// <summary>
        /// Check if a specific operand is a reduction operand (has stride=0 for READWRITE).
        /// </summary>
        public bool IsOperandReduction(int operand)
        {
            if ((uint)operand >= (uint)_state->NOp)
                throw new ArgumentOutOfRangeException(nameof(operand));

            return (_state->GetOpFlags(operand) & NpyIterOpFlags.REDUCE) != 0;
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
        /// Matches NumPy's NpyIter_IsFirstVisit behavior.
        /// </summary>
        public bool IsFirstVisit(int operand)
        {
            if ((uint)operand >= (uint)_state->NOp)
                throw new ArgumentOutOfRangeException(nameof(operand));

            // If this operand is not a reduction, every visit is "first"
            if ((_state->GetOpFlags(operand) & NpyIterOpFlags.REDUCE) == 0)
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
            if ((_state->ItFlags & (uint)NpyIterFlags.BUFFER) != 0 && _state->CoreSize > 0)
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
        /// Matches NumPy's NpyIter_Copy behavior.
        /// The copy has its own state and can be advanced independently.
        /// </summary>
        public NpyIterRef Copy()
        {
            // Allocate new state on heap
            var newStatePtr = (NpyIterState*)NativeMemory.AllocZeroed((nuint)sizeof(NpyIterState));

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
                newStatePtr->DType = _state->DType;
                newStatePtr->StridesNDim = _state->StridesNDim;
                newStatePtr->BufferSize = _state->BufferSize;
                newStatePtr->BufIterEnd = _state->BufIterEnd;
                newStatePtr->ReducePos = _state->ReducePos;
                newStatePtr->ReduceOuterSize = _state->ReduceOuterSize;
                newStatePtr->CoreSize = _state->CoreSize;
                newStatePtr->CorePos = _state->CorePos;
                newStatePtr->OuterDim = _state->OuterDim;
                newStatePtr->CoreOffset = _state->CoreOffset;

                // ALWAYS allocate new arrays (both dimension and operand arrays are dynamic now)
                newStatePtr->AllocateDimArrays(_state->NDim, _state->NOp);

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
                    newStatePtr->ElementSizes[op] = _state->ElementSizes[op];
                    newStatePtr->SrcElementSizes[op] = _state->SrcElementSizes[op];
                    newStatePtr->InnerStrides[op] = _state->InnerStrides[op];
                    newStatePtr->Buffers[op] = _state->Buffers[op];
                    newStatePtr->BufStrides[op] = _state->BufStrides[op];
                    newStatePtr->ReduceOuterStrides[op] = _state->ReduceOuterStrides[op];
                    newStatePtr->ReduceOuterPtrs[op] = _state->ReduceOuterPtrs[op];
                    newStatePtr->ArrayWritebackPtrs[op] = _state->ArrayWritebackPtrs[op];
                }

                // Create new iterator owning the state
                return new NpyIterRef
                {
                    _state = newStatePtr,
                    _ownsState = true,
                    _operands = _operands,  // Share operand references (they're not modified)
                    _cachedIterNext = null  // Don't copy cached delegate
                };
            }
            catch
            {
                newStatePtr->FreeDimArrays();
                NativeMemory.Free(newStatePtr);
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
            if (_ownsState && _state != null)
            {
                // Free any buffers using NpyIterBufferManager.FreeBuffers
                // NOTE: Buffers are allocated with AlignedAlloc, must be freed with AlignedFree
                if ((_state->ItFlags & (uint)NpyIterFlags.BUFFER) != 0)
                {
                    NpyIterBufferManager.FreeBuffers(ref *_state);
                }

                // Free dynamically allocated dimension arrays
                // NUMSHARP DIVERGENCE: Unlike NumPy's fixed arrays, we allocate dynamically
                _state->FreeDimArrays();

                NativeMemory.Free(_state);
                _state = null;
                _ownsState = false;
            }
        }
    }

    // =========================================================================
    // Static NpyIter Class (backward compatible API)
    // =========================================================================

    /// <summary>
    /// Static iterator helper methods (backward compatible API).
    ///
    /// NUMSHARP DIVERGENCE: These methods support unlimited dimensions via dynamic allocation.
    /// Dimension arrays are allocated on demand and freed after use.
    /// </summary>
    internal static unsafe class NpyIter
    {
        internal static bool ReduceBool<T, TKernel>(UnmanagedStorage src)
            where T : unmanaged
            where TKernel : struct, INpyBooleanReductionKernel<T>
        {
            var state = CreateReductionState(src);
            try
            {
                if (state.Size == 0)
                    return TKernel.Identity;

                if ((state.Flags & NpyIterFlags.SourceContiguous) != 0)
                {
                    var input = (void*)state.GetDataPointer(0);
                    return TKernel.Identity
                        ? ILKernelGenerator.AllSimdHelper<T>(input, state.Size)
                        : ILKernelGenerator.AnySimdHelper<T>(input, state.Size);
                }

                return ReduceBoolGeneral<T, TKernel>(ref state);
            }
            finally
            {
                // Free dynamically allocated dimension arrays
                state.FreeDimArrays();
            }
        }

        internal static bool TryCopySameType(UnmanagedStorage dst, UnmanagedStorage src)
        {
            if (dst.TypeCode != src.TypeCode)
                return false;

            NumSharpException.ThrowIfNotWriteable(dst.Shape);

            var state = CreateCopyState(src, dst);
            try
            {
                if (state.Size == 0)
                    return true;

                var path = state.IsContiguousCopy ? CopyExecutionPath.Contiguous : CopyExecutionPath.General;
                var kernel = ILKernelGenerator.TryGetCopyKernel(new CopyKernelKey(dst.TypeCode, path));
                if (kernel == null)
                    return false;

                var shape = state.GetShapePointer();
                var srcStrides = state.GetStridesPointer(0);
                var dstStrides = state.GetStridesPointer(1);

                kernel(
                    (void*)state.GetDataPointer(0),
                    (void*)state.GetDataPointer(1),
                    srcStrides,
                    dstStrides,
                    shape,
                    state.NDim,
                    state.Size);

                return true;
            }
            finally
            {
                // Free dynamically allocated dimension arrays
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
        ///   (<see cref="NpyIterCasting.CopyStridedToStridedWithCast"/>) reusing
        ///   the same broadcast/coalescing state.</item>
        /// </list>
        ///
        /// Drop-in replacement for the legacy <c>MultiIterator.Assign(dst, src)</c>:
        /// matches its broadcast-src-to-dst-shape semantics and its cast-on-write
        /// behavior (read src as src.TypeCode, convert, write dst.TypeCode).
        /// </summary>
        /// <exception cref="NumSharpException">If <paramref name="dst"/> is not writeable (e.g., broadcast view).</exception>
        internal static void Copy(UnmanagedStorage dst, UnmanagedStorage src)
        {
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            if (src is null) throw new ArgumentNullException(nameof(src));

            // Same-dtype fast path: SIMD copy kernel, broadcast + stride aware.
            if (TryCopySameType(dst, src))
                return;

            // Cross-dtype: per-element cast via NpyIterCasting.ConvertValue,
            // driven by the same coalesced broadcast state used by TryCopySameType.
            NumSharpException.ThrowIfNotWriteable(dst.Shape);

            var state = CreateCopyState(src, dst);
            try
            {
                if (state.Size == 0)
                    return;

                NpyIterCasting.CopyStridedToStridedWithCast(
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

        private static bool ReduceBoolGeneral<T, TKernel>(ref NpyIterState state)
            where T : unmanaged
            where TKernel : struct, INpyBooleanReductionKernel<T>
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
        internal static NpyIterState CreateCopyState(UnmanagedStorage src, UnmanagedStorage dst)
        {
            var broadcastSrcShape = np.broadcast_to(src.Shape, dst.Shape);
            int ndim = checked((int)dst.Shape.NDim);

            // NUMSHARP DIVERGENCE: No MaxDims limit - supports unlimited dimensions
            var state = new NpyIterState
            {
                Size = dst.Shape.size,
                DType = dst.TypeCode,
                Flags = NpyIterFlags.None,
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
                    state.Flags |= NpyIterFlags.SourceBroadcast;
            }

            CoalesceAxes(ref state, shape, srcStridePtr, dstStridePtr);
            UpdateLayoutFlags(ref state, shape, srcStridePtr, dstStridePtr);

            return state;
        }

        /// <summary>
        /// Create state for reduction operation.
        /// IMPORTANT: Caller must call state.FreeDimArrays() when done!
        /// </summary>
        internal static NpyIterState CreateReductionState(UnmanagedStorage src)
        {
            int ndim = checked((int)src.Shape.NDim);

            // NUMSHARP DIVERGENCE: No MaxDims limit - supports unlimited dimensions
            var state = new NpyIterState
            {
                Size = src.Shape.size,
                DType = src.TypeCode,
                Flags = src.Shape.IsContiguous ? NpyIterFlags.SourceContiguous : NpyIterFlags.None,
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

        internal static void CoalesceAxes(ref NpyIterState state, long* shape, long* srcStrides, long* dstStrides)
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

        internal static void UpdateLayoutFlags(ref NpyIterState state, long* shape, long* srcStrides, long* dstStrides)
        {
            if (state.Size <= 1)
            {
                state.Flags |= NpyIterFlags.SourceContiguous | NpyIterFlags.DestinationContiguous;
                return;
            }

            if (IsContiguous(shape, srcStrides, state.NDim))
                state.Flags |= NpyIterFlags.SourceContiguous;
            if (IsContiguous(shape, dstStrides, state.NDim))
                state.Flags |= NpyIterFlags.DestinationContiguous;
        }

        internal static bool IsContiguous(long* shape, long* strides, int ndim)
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

        internal static void Advance(long* shape, long* strides, long* coords, int ndim, ref long offset)
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
