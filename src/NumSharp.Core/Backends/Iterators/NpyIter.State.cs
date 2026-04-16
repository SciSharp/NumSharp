using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Utilities;

namespace NumSharp.Backends.Iteration
{
    // =====================================================================================
    // NumSharp Divergence from NumPy: Unlimited Dimensions
    // =====================================================================================
    //
    // NumPy uses a fixed NPY_MAXDIMS=64 limit for array dimensions. This is a hard-coded
    // constant that limits all NumPy operations to 64 dimensions maximum.
    //
    // NumSharp takes a different approach: UNLIMITED DIMENSIONS.
    //
    // NumSharp's Shape struct uses regular managed arrays (int[] dimensions, int[] strides)
    // which can be any size. The practical limit is around 300,000 dimensions, soft-limited
    // by stackalloc buffer sizes used in coordinate iteration. However, for typical use
    // cases (even extreme ones like deep learning with thousands of dimensions), there is
    // effectively no limit.
    //
    // To maintain consistency with NumSharp's unlimited dimension philosophy, NpyIterState
    // uses dynamically allocated arrays instead of fixed-size buffers. This means:
    //
    // 1. Dimension-dependent arrays (Shape, Coords, Perm, Strides) are allocated based on
    //    actual NDim at construction time
    // 2. Per-operand arrays still use a fixed MaxOperands=8 limit (this is reasonable as
    //    very few operations need more than 8 operands)
    // 3. Memory is allocated via NativeMemory and must be explicitly freed
    //
    // Trade-offs:
    // - Pro: No artificial dimension limit, matches NumSharp's core philosophy
    // - Pro: Memory usage scales with actual dimensions, not worst case
    // - Con: Slightly more complex allocation/deallocation
    // - Con: Cannot use simple fixed() statements, need explicit pointer management
    //
    // =====================================================================================

    /// <summary>
    /// Core iterator state with dynamically allocated dimension arrays.
    ///
    /// NUMSHARP DIVERGENCE: Unlike NumPy's fixed NPY_MAXDIMS=64, NumSharp supports
    /// unlimited dimensions. Dimension-dependent arrays are allocated dynamically
    /// based on actual NDim. See class-level comments for rationale.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct NpyIterState
    {
        // =========================================================================
        // Constants
        // =========================================================================

        /// <summary>
        /// Maximum supported operands. This remains fixed as very few operations
        /// need more than 8 operands, and keeping this fixed simplifies the struct.
        /// </summary>
        internal const int MaxOperands = 8;

        /// <summary>
        /// Threshold for using stackalloc vs heap allocation for temporary buffers.
        /// Arrays with more dimensions than this will use heap allocation.
        /// </summary>
        internal const int StackAllocThreshold = 64;

        // =========================================================================
        // Core Scalar Fields
        // =========================================================================

        /// <summary>Iterator flags (NpyIterFlags bitmask).</summary>
        public uint ItFlags;

        /// <summary>Number of dimensions after coalescing.</summary>
        public int NDim;

        /// <summary>Number of operands.</summary>
        public int NOp;

        /// <summary>Mask operand index (-1 if none).</summary>
        public int MaskOp;

        /// <summary>Total number of iterations.</summary>
        public long IterSize;

        /// <summary>Current iteration index.</summary>
        public long IterIndex;

        /// <summary>Range start for ranged iteration.</summary>
        public long IterStart;

        /// <summary>Range end for ranged iteration.</summary>
        public long IterEnd;

        /// <summary>
        /// Flat index for C_INDEX or F_INDEX tracking.
        /// Updated by Advance() when HASINDEX flag is set.
        /// </summary>
        public long FlatIndex;

        /// <summary>
        /// True if tracking C-order index, false for F-order.
        /// Only meaningful when HASINDEX flag is set.
        /// </summary>
        public bool IsCIndex;

        // =========================================================================
        // Legacy compatibility fields
        // =========================================================================

        /// <summary>Legacy: total size (alias for IterSize).</summary>
        public long Size
        {
            readonly get => IterSize;
            set => IterSize = value;
        }

        /// <summary>Legacy: flags (lower bits of ItFlags).</summary>
        public NpyIterFlags Flags
        {
            readonly get => (NpyIterFlags)(ItFlags & 0xFFFF);
            set => ItFlags = (ItFlags & 0xFFFF0000) | (uint)value;
        }

        /// <summary>Legacy: primary dtype.</summary>
        public NPTypeCode DType;

        // =========================================================================
        // Dynamically Allocated Dimension Arrays (NUMSHARP DIVERGENCE)
        // =========================================================================
        // These arrays are allocated based on actual NDim, not a fixed maximum.
        // This enables unlimited dimension support matching NumSharp's core design.
        // =========================================================================

        /// <summary>
        /// Axis permutation (maps iterator axis to original axis).
        /// Dynamically allocated: size = NDim.
        /// </summary>
        public sbyte* Perm;

        /// <summary>
        /// Shape after coalescing.
        /// Dynamically allocated: size = NDim.
        /// </summary>
        public long* Shape;

        /// <summary>
        /// Current coordinates.
        /// Dynamically allocated: size = NDim.
        /// </summary>
        public long* Coords;

        /// <summary>
        /// Strides for each operand along each axis.
        /// Dynamically allocated: size = NDim * NOp.
        /// Layout: [op0_axis0, op0_axis1, ..., op1_axis0, op1_axis1, ...]
        /// Access: Strides[operand * NDim + axis]
        ///
        /// Note: Unlike fixed layout which uses MaxDims spacing, dynamic layout
        /// packs strides contiguously based on actual NDim.
        /// </summary>
        public long* Strides;

        /// <summary>
        /// Allocated NDim for the Strides array. Used to compute correct offsets
        /// when NDim changes (e.g., after coalescing). Strides array maintains
        /// its original allocation size for safety.
        /// </summary>
        public int StridesNDim;

        // =========================================================================
        // Fixed Per-Operand Arrays (MaxOperands is reasonable limit)
        // =========================================================================

        /// <summary>Current data pointers for each operand.</summary>
        public fixed long DataPtrs[MaxOperands];

        /// <summary>Reset data pointers (base + offset).</summary>
        public fixed long ResetDataPtrs[MaxOperands];

        /// <summary>Base offsets for each operand.</summary>
        public fixed long BaseOffsets[MaxOperands];

        /// <summary>Per-operand flags.</summary>
        public fixed ushort OpItFlags[MaxOperands];

        /// <summary>Buffer/target dtypes for each operand.</summary>
        public fixed byte OpDTypes[MaxOperands];

        /// <summary>Source array dtypes for each operand (used for casting).</summary>
        public fixed byte OpSrcDTypes[MaxOperands];

        /// <summary>Element sizes for each operand (based on buffer dtype).</summary>
        public fixed int ElementSizes[MaxOperands];

        /// <summary>Source element sizes for each operand (based on source dtype).</summary>
        public fixed int SrcElementSizes[MaxOperands];

        /// <summary>
        /// Inner strides for each operand (gathered from main Strides array for fast access).
        /// Layout: [op0_inner_stride, op1_inner_stride, ...]
        /// </summary>
        public fixed long InnerStrides[MaxOperands];

        // =========================================================================
        // Buffer Data (when BUFFERED flag is set)
        // =========================================================================

        /// <summary>Buffer size (elements per buffer).</summary>
        public long BufferSize;

        /// <summary>Current buffer iteration end.</summary>
        public long BufIterEnd;

        /// <summary>Buffer pointers for each operand.</summary>
        public fixed long Buffers[MaxOperands];

        /// <summary>Buffer strides (always element size for contiguous buffers).</summary>
        public fixed long BufStrides[MaxOperands];

        // =========================================================================
        // Buffered Reduction Data (when BUFFERED + REDUCE flags are set)
        // =========================================================================
        // NumPy uses a double-loop pattern for buffered reduction:
        // - Outer loop: iterates over non-reduce axes
        // - Inner loop: iterates over reduce axis within buffer
        // =========================================================================

        /// <summary>
        /// Current position in reduce outer loop.
        /// Used by IsFirstVisit for buffered reduction.
        /// </summary>
        public long ReducePos;

        /// <summary>
        /// Size of reduce outer loop (number of reduction iterations).
        /// </summary>
        public long ReduceOuterSize;

        /// <summary>
        /// Outer strides for reduction (stride per reduce outer iteration).
        /// Layout: [op0_reduce_stride, op1_reduce_stride, ...]
        /// When stride is 0, the operand is a reduction target for that axis.
        /// </summary>
        public fixed long ReduceOuterStrides[MaxOperands];

        // =========================================================================
        // Allocation and Deallocation
        // =========================================================================

        /// <summary>
        /// Allocate dimension-dependent arrays for given ndim and nop.
        /// Must be called before using Shape, Coords, Perm, or Strides.
        /// Initializes Perm to identity permutation [0, 1, 2, ...].
        /// </summary>
        public void AllocateDimArrays(int ndim, int nop)
        {
            if (ndim < 0) throw new ArgumentOutOfRangeException(nameof(ndim));
            if (nop < 1 || nop > MaxOperands) throw new ArgumentOutOfRangeException(nameof(nop));

            NDim = ndim;
            NOp = nop;
            StridesNDim = ndim;

            if (ndim == 0)
            {
                // Scalar case - no dimension arrays needed
                Shape = null;
                Coords = null;
                Perm = null;
                Strides = null;
                return;
            }

            // Allocate all dimension arrays in one contiguous block for cache efficiency
            // Layout: [Shape: ndim longs][Coords: ndim longs][Strides: ndim*nop longs][Perm: ndim sbytes]
            long shapeBytes = ndim * sizeof(long);
            long coordsBytes = ndim * sizeof(long);
            long stridesBytes = ndim * nop * sizeof(long);
            long permBytes = ndim * sizeof(sbyte);

            // Align perm to 8 bytes for cleaner memory layout
            long permBytesAligned = (permBytes + 7) & ~7L;

            long totalBytes = shapeBytes + coordsBytes + stridesBytes + permBytesAligned;

            byte* block = (byte*)NativeMemory.AllocZeroed((nuint)totalBytes);

            Shape = (long*)block;
            Coords = (long*)(block + shapeBytes);
            Strides = (long*)(block + shapeBytes + coordsBytes);
            Perm = (sbyte*)(block + shapeBytes + coordsBytes + stridesBytes);

            // Initialize Perm to identity permutation
            // Perm[internal_axis] = original_axis
            for (int d = 0; d < ndim; d++)
                Perm[d] = (sbyte)d;
        }

        /// <summary>
        /// Free dimension-dependent arrays. Must be called before freeing the state itself.
        /// </summary>
        public void FreeDimArrays()
        {
            // All arrays are in one contiguous block starting at Shape
            if (Shape != null)
            {
                NativeMemory.Free(Shape);
                Shape = null;
                Coords = null;
                Strides = null;
                Perm = null;
            }
        }

        // =========================================================================
        // Accessor Methods
        // =========================================================================

        /// <summary>Get pointer to Shape array.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long* GetShapePointer() => Shape;

        /// <summary>Get pointer to Coords array.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long* GetCoordsPointer() => Coords;

        /// <summary>
        /// Get pointer to strides for a specific operand.
        /// Uses actual NDim (or StridesNDim if NDim changed after allocation).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long* GetStridesPointer(int operand)
        {
            if ((uint)operand >= (uint)NOp)
                throw new ArgumentOutOfRangeException(nameof(operand));

            return Strides + (operand * StridesNDim);
        }

        /// <summary>Get stride for operand at axis.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long GetStride(int axis, int op)
        {
            return Strides[op * StridesNDim + axis];
        }

        /// <summary>Set stride for operand at axis.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetStride(int axis, int op, long value)
        {
            Strides[op * StridesNDim + axis] = value;
        }

        /// <summary>Get current data pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void* GetDataPtr(int op)
        {
            fixed (long* p = DataPtrs)
                return (void*)p[op];
        }

        /// <summary>Set current data pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetDataPtr(int op, void* ptr)
        {
            fixed (long* p = DataPtrs)
                p[op] = (long)ptr;
        }

        /// <summary>Get data pointer (legacy interface).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly IntPtr GetDataPointer(int operand)
        {
            fixed (long* p = DataPtrs)
                return (IntPtr)p[operand];
        }

        /// <summary>Set data pointer (legacy interface).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetDataPointer(int operand, IntPtr pointer)
        {
            fixed (long* p = DataPtrs)
                p[operand] = (long)pointer;
        }

        /// <summary>Get reset data pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void* GetResetDataPtr(int op)
        {
            fixed (long* p = ResetDataPtrs)
                return (void*)p[op];
        }

        /// <summary>Set reset data pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetResetDataPtr(int op, void* ptr)
        {
            fixed (long* p = ResetDataPtrs)
                p[op] = (long)ptr;
        }

        /// <summary>Get operand dtype.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public NPTypeCode GetOpDType(int op)
        {
            fixed (byte* p = OpDTypes)
                return (NPTypeCode)p[op];
        }

        /// <summary>Set operand dtype (buffer/target dtype).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetOpDType(int op, NPTypeCode dtype)
        {
            fixed (byte* p = OpDTypes)
                p[op] = (byte)dtype;

            fixed (int* s = ElementSizes)
                s[op] = InfoOf.GetSize(dtype);
        }

        /// <summary>Get source array dtype for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public NPTypeCode GetOpSrcDType(int op)
        {
            fixed (byte* p = OpSrcDTypes)
                return (NPTypeCode)p[op];
        }

        /// <summary>Set source array dtype for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetOpSrcDType(int op, NPTypeCode dtype)
        {
            fixed (byte* p = OpSrcDTypes)
                p[op] = (byte)dtype;

            fixed (int* s = SrcElementSizes)
                s[op] = InfoOf.GetSize(dtype);
        }

        /// <summary>Get source element size for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetSrcElementSize(int op)
        {
            fixed (int* p = SrcElementSizes)
                return p[op];
        }

        /// <summary>Check if operand needs casting (source dtype != buffer dtype).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool NeedsCast(int op)
        {
            return GetOpSrcDType(op) != GetOpDType(op);
        }

        /// <summary>Get operand flags.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public NpyIterOpFlags GetOpFlags(int op)
        {
            fixed (ushort* p = OpItFlags)
                return (NpyIterOpFlags)p[op];
        }

        /// <summary>Set operand flags.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetOpFlags(int op, NpyIterOpFlags flags)
        {
            fixed (ushort* p = OpItFlags)
                p[op] = (ushort)flags;
        }

        /// <summary>Get element size for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetElementSize(int op)
        {
            fixed (int* p = ElementSizes)
                return p[op];
        }

        /// <summary>Get buffer pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void* GetBuffer(int op)
        {
            fixed (long* p = Buffers)
                return (void*)p[op];
        }

        /// <summary>Set buffer pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetBuffer(int op, void* ptr)
        {
            fixed (long* p = Buffers)
                p[op] = (long)ptr;
        }

        /// <summary>Get reduce outer stride for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long GetReduceOuterStride(int op)
        {
            fixed (long* p = ReduceOuterStrides)
                return p[op];
        }

        /// <summary>Set reduce outer stride for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetReduceOuterStride(int op, long stride)
        {
            fixed (long* p = ReduceOuterStrides)
                p[op] = stride;
        }

        /// <summary>
        /// Get inner stride array pointer - returns contiguous array of inner strides for all operands.
        /// Layout: [op0_inner_stride, op1_inner_stride, ...]
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long* GetInnerStrideArray()
        {
            fixed (long* p = InnerStrides)
                return p;
        }

        /// <summary>
        /// Update the InnerStrides array from the main Strides array.
        /// Must be called after coalescing or axis removal.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void UpdateInnerStrides()
        {
            fixed (long* inner = InnerStrides)
            {
                if (NDim == 0)
                {
                    // Scalar - all inner strides are 0
                    for (int op = 0; op < NOp; op++)
                        inner[op] = 0;
                    return;
                }

                int innerAxis = NDim - 1;
                for (int op = 0; op < NOp; op++)
                    inner[op] = Strides[op * StridesNDim + innerAxis];
            }
        }

        /// <summary>Check if this is a contiguous copy operation (legacy).</summary>
        public readonly bool IsContiguousCopy =>
            ((NpyIterFlags)ItFlags & (NpyIterFlags.SourceContiguous | NpyIterFlags.DestinationContiguous)) ==
            (NpyIterFlags.SourceContiguous | NpyIterFlags.DestinationContiguous) &&
            ((NpyIterFlags)ItFlags & NpyIterFlags.SourceBroadcast) == 0;

        // =========================================================================
        // Iteration Methods
        // =========================================================================

        /// <summary>
        /// Advance iterator by one position using ripple carry.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Advance()
        {
            IterIndex++;

            // Track whether we need to compute FlatIndex (deferred until after coord update)
            bool needsFlatIndex = (ItFlags & (uint)NpyIterFlags.HASINDEX) != 0;
            bool usesFastPath = needsFlatIndex && IsCIndex && (ItFlags & (uint)NpyIterFlags.IDENTPERM) != 0;

            fixed (long* dataPtrs = DataPtrs)
            fixed (int* elemSizes = ElementSizes)
            {
                for (int axis = NDim - 1; axis >= 0; axis--)
                {
                    Coords[axis]++;

                    if (Coords[axis] < Shape[axis])
                    {
                        // Advance data pointers along this axis
                        for (int op = 0; op < NOp; op++)
                        {
                            long stride = Strides[op * StridesNDim + axis];
                            dataPtrs[op] += stride * elemSizes[op];
                        }

                        // Update flat index AFTER coords are updated
                        if (needsFlatIndex)
                        {
                            if (usesFastPath)
                                FlatIndex++;
                            else
                                FlatIndex = ComputeFlatIndex();
                        }
                        return;
                    }

                    // Carry: reset this axis, continue to next
                    Coords[axis] = 0;

                    // Reset data pointers for this axis
                    for (int op = 0; op < NOp; op++)
                    {
                        long stride = Strides[op * StridesNDim + axis];
                        long axisShape = Shape[axis];
                        dataPtrs[op] -= stride * (axisShape - 1) * elemSizes[op];
                    }
                }
            }

            // If we reach here, all coords wrapped (end of iteration)
            // Update flat index for completeness
            if (needsFlatIndex)
            {
                if (usesFastPath)
                    FlatIndex++;
                else
                    FlatIndex = ComputeFlatIndex();
            }
        }

        /// <summary>
        /// Reset iterator to the beginning.
        /// </summary>
        public void Reset()
        {
            IterIndex = IterStart;
            FlatIndex = 0;

            for (int d = 0; d < NDim; d++)
                Coords[d] = 0;

            fixed (long* dataPtrs = DataPtrs)
            fixed (long* resetPtrs = ResetDataPtrs)
            {
                for (int op = 0; op < NOp; op++)
                    dataPtrs[op] = resetPtrs[op];
            }

            // Invalidate all buffer reuse flags since position changed
            InvalidateAllBufferReuse();
        }

        /// <summary>
        /// Invalidate buffer reuse flags for all operands.
        /// Called when iterator position changes (Reset, GotoIterIndex).
        /// </summary>
        private void InvalidateAllBufferReuse()
        {
            fixed (ushort* flags = OpItFlags)
            {
                for (int op = 0; op < NOp; op++)
                {
                    flags[op] = (ushort)(flags[op] & ~(ushort)NpyIterOpFlags.BUF_REUSABLE);
                }
            }
        }

        /// <summary>
        /// Jump to a specific iteration index.
        /// </summary>
        public void GotoIterIndex(long iterindex)
        {
            IterIndex = iterindex;

            // Calculate coordinates from linear index
            long remaining = iterindex;
            for (int d = NDim - 1; d >= 0; d--)
            {
                long dimSize = Shape[d];
                Coords[d] = remaining % dimSize;
                remaining /= dimSize;
            }

            // Update flat index if tracking
            if ((ItFlags & (uint)NpyIterFlags.HASINDEX) != 0)
            {
                FlatIndex = ComputeFlatIndex();
            }

            // Update data pointers
            fixed (long* dataPtrs = DataPtrs)
            fixed (long* resetPtrs = ResetDataPtrs)
            fixed (int* elemSizes = ElementSizes)
            {
                for (int op = 0; op < NOp; op++)
                {
                    long offset = 0;
                    for (int d = 0; d < NDim; d++)
                    {
                        offset += Coords[d] * Strides[op * StridesNDim + d];
                    }
                    dataPtrs[op] = resetPtrs[op] + offset * elemSizes[op];
                }
            }

            // Invalidate all buffer reuse flags since position changed
            InvalidateAllBufferReuse();
        }

        /// <summary>
        /// Initialize FlatIndex based on current coordinates.
        /// Should be called after HASINDEX flag is set and all axis setup is complete.
        /// </summary>
        public void InitializeFlatIndex()
        {
            if ((ItFlags & (uint)NpyIterFlags.HASINDEX) != 0)
            {
                FlatIndex = ComputeFlatIndex();
            }
        }

        /// <summary>
        /// Compute the flat index from current coordinates based on C or F order.
        /// Uses original (pre-reordering) coordinate order via Perm array.
        /// When NEGPERM is set, flipped axes have negative perm entries and their
        /// coordinates are reversed when computing the original index.
        /// </summary>
        private long ComputeFlatIndex()
        {
            if (NDim == 0)
                return 0;

            bool hasNegPerm = (ItFlags & (uint)NpyIterFlags.NEGPERM) != 0;

            // Build original coords and shape from internal using Perm
            // Perm[internal_axis] = original_axis (or -1-original if flipped)
            var origCoords = stackalloc long[NDim];
            var origShape = stackalloc long[NDim];

            for (int d = 0; d < NDim; d++)
            {
                int p = Perm[d];
                int origAxis;
                long origCoord;

                if (hasNegPerm && p < 0)
                {
                    // Flipped axis: original = -1 - p, coord is reversed
                    origAxis = -1 - p;
                    origCoord = Shape[d] - Coords[d] - 1;
                }
                else
                {
                    origAxis = p;
                    origCoord = Coords[d];
                }

                origCoords[origAxis] = origCoord;
                origShape[origAxis] = Shape[d];
            }

            long index = 0;
            if (IsCIndex)
            {
                // C-order: row-major, last dimension varies fastest
                long multiplier = 1;
                for (int d = NDim - 1; d >= 0; d--)
                {
                    index += origCoords[d] * multiplier;
                    multiplier *= origShape[d];
                }
            }
            else
            {
                // F-order: column-major, first dimension varies fastest
                long multiplier = 1;
                for (int d = 0; d < NDim; d++)
                {
                    index += origCoords[d] * multiplier;
                    multiplier *= origShape[d];
                }
            }
            return index;
        }
    }
}
