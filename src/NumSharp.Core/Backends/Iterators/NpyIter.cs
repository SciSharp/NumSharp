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
    internal unsafe ref struct NpyIterRef
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
            if (nop < 1 || nop > NpyIterState.MaxOperands)
                throw new ArgumentOutOfRangeException(nameof(nop), $"Number of operands must be between 1 and {NpyIterState.MaxOperands}");

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
                ValidateIterShape(nop, op, opFlags, broadcastShape);
            }
            else
            {
                broadcastShape = CalculateBroadcastShape(nop, op, opFlags);
            }

            // =========================================================================
            // NUMSHARP DIVERGENCE: Allocate dimension arrays dynamically
            // Unlike NumPy's fixed NPY_MAXDIMS=64, NumSharp supports unlimited dimensions.
            // Arrays are allocated based on actual ndim for memory efficiency.
            // =========================================================================
            _state->AllocateDimArrays(broadcastShape.Length, nop);

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

            // Set up operands
            for (int i = 0; i < nop; i++)
            {
                var arr = op[i];
                var arrShape = arr.Shape;

                // Set dtype
                var dtype = opDtypes != null && i < opDtypes.Length ? opDtypes[i] : arr.typecode;
                _state->SetOpDType(i, dtype);

                // Set operand flags
                var opFlag = TranslateOpFlags(opFlags[i]);
                _state->SetOpFlags(i, opFlag);

                // Calculate broadcast strides for this operand
                var broadcastArr = np.broadcast_to(arrShape, new Shape(broadcastShape));
                var basePtr = (byte*)arr.Address + (broadcastArr.offset * arr.dtypesize);

                _state->SetDataPtr(i, basePtr);
                _state->SetResetDataPtr(i, basePtr);

                // Set strides
                var stridePtr = _state->GetStridesPointer(i);
                for (int d = 0; d < _state->NDim; d++)
                {
                    stridePtr[d] = broadcastArr.strides[d];
                }

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

            // Apply op_axes remapping if provided
            if (opAxes != null && opAxesNDim >= 0)
            {
                ApplyOpAxes(opAxesNDim, opAxes);
            }

            // Apply axis reordering and coalescing unless multi-index tracking is requested
            // NumPy always coalesces after construction: nditer_constr.c line 395-396
            // if (ndim > 1 && !(itflags & NPY_ITFLAG_HASMULTIINDEX)) { npyiter_coalesce_axes(iter); }
            //
            // IMPORTANT: NumPy reorders axes BEFORE coalescing so that axes are sorted by
            // stride magnitude. This allows contiguous arrays to fully coalesce to 1D.
            // Without reordering, a C-contiguous (2,3,4) array with strides [12,4,1] cannot
            // coalesce because stride[0]*shape[0]=24 != stride[1]=4.
            // After reordering to [4,3,2] with strides [1,4,12]:
            // - stride[0]*shape[0]=1*4=4 == stride[1]=4 ✓ → coalesce to [12,2], strides [1,12]
            // - stride[0]*shape[0]=1*12=12 == stride[1]=12 ✓ → coalesce to [24], strides [1]
            if (_state->NDim > 1 && (flags & NpyIterGlobalFlags.MULTI_INDEX) == 0)
            {
                // Step 1: Reorder axes by stride (smallest first = innermost in memory)
                // This matches NumPy's npyiter_apply_order() behavior
                NpyIterCoalescing.ReorderAxesForCoalescing(ref *_state, order);

                // Step 2: Now coalesce adjacent axes that have compatible strides
                NpyIterCoalescing.CoalesceAxes(ref *_state);
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
                _state->FlatIndex = 0;
            }
            else if ((flags & NpyIterGlobalFlags.F_INDEX) != 0)
            {
                _state->ItFlags |= (uint)NpyIterFlags.HASINDEX;
                _state->IsCIndex = false;
                _state->FlatIndex = 0;
            }

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
            }

            // Handle single iteration optimization
            if (_state->IterSize <= 1)
            {
                _state->ItFlags |= (uint)NpyIterFlags.ONEITERATION;
            }
        }

        private static int[] CalculateBroadcastShape(int nop, NDArray[] op, NpyIterPerOpFlags[] opFlags)
        {
            int maxNdim = 0;
            for (int i = 0; i < nop; i++)
            {
                if (op[i].ndim > maxNdim)
                    maxNdim = op[i].ndim;
            }

            if (maxNdim == 0)
                return Array.Empty<int>();

            var result = new int[maxNdim];
            for (int i = 0; i < maxNdim; i++)
                result[i] = 1;

            for (int opIdx = 0; opIdx < nop; opIdx++)
            {
                if ((opFlags[opIdx] & NpyIterPerOpFlags.NO_BROADCAST) != 0)
                    continue;

                var opShape = op[opIdx].shape;
                int offset = maxNdim - opShape.Length;

                for (int d = 0; d < opShape.Length; d++)
                {
                    int dim = (int)opShape[d];
                    int rd = offset + d;

                    if (result[rd] == 1)
                        result[rd] = dim;
                    else if (dim != 1 && dim != result[rd])
                        throw new IncorrectShapeException($"Operands could not be broadcast together");
                }
            }

            return result;
        }

        /// <summary>
        /// Validate that operands are compatible with the specified iterShape.
        /// Each operand dimension must either equal the iterShape or be 1 (broadcastable).
        /// </summary>
        private static void ValidateIterShape(int nop, NDArray[] op, NpyIterPerOpFlags[] opFlags, int[] iterShape)
        {
            for (int opIdx = 0; opIdx < nop; opIdx++)
            {
                if ((opFlags[opIdx] & NpyIterPerOpFlags.NO_BROADCAST) != 0)
                    continue;

                var opShape = op[opIdx].shape;
                int offset = iterShape.Length - opShape.Length;

                // Operand must have fewer or equal dimensions
                if (offset < 0)
                    throw new IncorrectShapeException($"Operand {opIdx} has more dimensions than iterShape");

                for (int d = 0; d < opShape.Length; d++)
                {
                    int opDim = (int)opShape[d];
                    int iterDim = iterShape[offset + d];

                    // opDim must equal iterDim or be 1 (broadcastable)
                    if (opDim != iterDim && opDim != 1)
                        throw new IncorrectShapeException($"Operand {opIdx} shape incompatible with iterShape at axis {d}");
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

            return result;
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
        /// Apply op_axes remapping to operand strides.
        /// op_axes allows custom mapping of operand dimensions to iterator dimensions.
        /// A value of -1 indicates the dimension should be broadcast (stride = 0).
        /// </summary>
        private void ApplyOpAxes(int opAxesNDim, int[][] opAxes)
        {
            if (opAxes == null || opAxesNDim <= 0)
                return;

            // Ensure we don't exceed iterator dimensions
            int iterNDim = Math.Min(opAxesNDim, _state->NDim);

            for (int op = 0; op < _state->NOp; op++)
            {
                // Skip if no mapping for this operand
                if (op >= opAxes.Length || opAxes[op] == null)
                    continue;

                var opAxisMap = opAxes[op];
                var stridePtr = _state->GetStridesPointer(op);

                // Gather original strides before remapping
                // NUMSHARP DIVERGENCE: Use actual ndim, not fixed MaxDims
                var originalStrides = stackalloc long[iterNDim];
                for (int d = 0; d < iterNDim; d++)
                    originalStrides[d] = stridePtr[d];

                // Apply remapping
                for (int iterAxis = 0; iterAxis < iterNDim && iterAxis < opAxisMap.Length; iterAxis++)
                {
                    int opAxis = opAxisMap[iterAxis];

                    if (opAxis < 0)
                    {
                        // -1 means broadcast this dimension (reduction axis)
                        stridePtr[iterAxis] = 0;
                        // Mark as broadcast
                        _state->ItFlags |= (uint)NpyIterFlags.SourceBroadcast;
                    }
                    else if (opAxis < iterNDim)
                    {
                        // Remap: use stride from the specified axis
                        stridePtr[iterAxis] = originalStrides[opAxis];
                    }
                    // else: invalid axis, keep original
                }
            }
        }

        // =========================================================================
        // Properties
        // =========================================================================

        /// <summary>Number of operands.</summary>
        public int NOp => _state->NOp;

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
        /// Advance to next position and return whether more iterations remain.
        /// Matches NumPy's iternext() behavior.
        /// Returns true if more elements exist, false when iteration is complete.
        /// </summary>
        public bool Iternext()
        {
            if (_state->IterIndex >= _state->IterEnd)
                return false;

            _state->Advance();
            return _state->IterIndex < _state->IterEnd;
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
        /// Get the current multi-index (coordinates).
        /// Requires MULTI_INDEX flag to be set during construction.
        /// </summary>
        public void GetMultiIndex(Span<long> outCoords)
        {
            if ((_state->ItFlags & (uint)NpyIterFlags.HASMULTIINDEX) == 0)
                throw new InvalidOperationException("Iterator not tracking multi-index. Use NpyIterGlobalFlags.MULTI_INDEX during construction.");

            if (outCoords.Length < _state->NDim)
                throw new ArgumentException($"Output span must have at least {_state->NDim} elements", nameof(outCoords));

            for (int d = 0; d < _state->NDim; d++)
                outCoords[d] = _state->Coords[d];
        }

        /// <summary>
        /// Jump to a specific multi-index (coordinates).
        /// Requires MULTI_INDEX flag to be set during construction.
        /// </summary>
        public void GotoMultiIndex(ReadOnlySpan<long> coords)
        {
            if ((_state->ItFlags & (uint)NpyIterFlags.HASMULTIINDEX) == 0)
                throw new InvalidOperationException("Iterator not tracking multi-index. Use NpyIterGlobalFlags.MULTI_INDEX during construction.");

            if (coords.Length < _state->NDim)
                throw new ArgumentException($"Coordinates must have at least {_state->NDim} elements", nameof(coords));

            // Validate coordinates and compute linear index (C-order)
            long iterIndex = 0;
            long multiplier = 1;

            for (int d = _state->NDim - 1; d >= 0; d--)
            {
                if (coords[d] < 0 || coords[d] >= _state->Shape[d])
                    throw new IndexOutOfRangeException($"Coordinate {coords[d]} out of range for axis {d} (size {_state->Shape[d]})");

                _state->Coords[d] = coords[d];
                iterIndex += coords[d] * multiplier;
                multiplier *= _state->Shape[d];
            }

            _state->IterIndex = iterIndex;

            // Update flat index if tracking (C_INDEX or F_INDEX)
            if ((_state->ItFlags & (uint)NpyIterFlags.HASINDEX) != 0)
            {
                if (_state->IsCIndex)
                {
                    // C-order: iterIndex is already the C-order flat index
                    _state->FlatIndex = iterIndex;
                }
                else
                {
                    // F-order: compute column-major index
                    long fIndex = 0;
                    multiplier = 1;
                    for (int d = 0; d < _state->NDim; d++)
                    {
                        fIndex += coords[d] * multiplier;
                        multiplier *= _state->Shape[d];
                    }
                    _state->FlatIndex = fIndex;
                }
            }

            // Update data pointers
            for (int op = 0; op < _state->NOp; op++)
            {
                long offset = 0;
                for (int d = 0; d < _state->NDim; d++)
                    offset += coords[d] * _state->GetStride(d, op);

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
        /// Check if iteration is finished.
        /// </summary>
        public bool Finished => _state->IterIndex >= _state->IterEnd;

        /// <summary>
        /// Get the current iterator shape.
        /// This reflects the shape after coalescing (if any).
        /// </summary>
        public long[] Shape
        {
            get
            {
                var result = new long[_state->NDim];
                for (int d = 0; d < _state->NDim; d++)
                    result[d] = _state->Shape[d];
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
        /// Get operand arrays.
        /// </summary>
        public NDArray[]? GetOperandArray() => _operands;

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
        /// Matches NumPy's dataptrs[i] access.
        /// </summary>
        public void* GetDataPtr(int operand)
        {
            if ((uint)operand >= (uint)_state->NOp)
                throw new ArgumentOutOfRangeException(nameof(operand));
            return _state->GetDataPtr(operand);
        }

        /// <summary>
        /// Get current value for operand as T.
        /// </summary>
        public T GetValue<T>(int operand = 0) where T : unmanaged
        {
            return *(T*)GetDataPtr(operand);
        }

        /// <summary>
        /// Set current value for operand.
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
        // Lifecycle
        // =========================================================================

        /// <summary>
        /// Deallocate iterator resources.
        /// </summary>
        public void Dispose()
        {
            if (_ownsState && _state != null)
            {
                // Free any buffers
                if ((_state->ItFlags & (uint)NpyIterFlags.BUFFER) != 0)
                {
                    for (int op = 0; op < _state->NOp; op++)
                    {
                        var buf = _state->GetBuffer(op);
                        if (buf != null)
                        {
                            NativeMemory.Free(buf);
                            _state->SetBuffer(op, null);
                        }
                    }
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
