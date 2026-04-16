using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Utilities;

namespace NumSharp.Backends.Iteration
{
    // =====================================================================================
    // NumSharp Divergence from NumPy: Unlimited Dimensions AND Unlimited Operands
    // =====================================================================================
    //
    // NumPy uses fixed limits:
    // - NPY_MAXDIMS = 64 (maximum array dimensions)
    // - NPY_MAXARGS = 64 (maximum operands in NumPy 2.x, was 32 in 1.x)
    //
    // NumSharp takes a different approach: UNLIMITED for both.
    //
    // NumSharp's Shape struct uses regular managed arrays (int[] dimensions, int[] strides)
    // which can be any size. The practical limit is around 300,000 dimensions, soft-limited
    // by stackalloc buffer sizes used in coordinate iteration.
    //
    // For operands, while NumPy caps at 64, NumSharp supports unlimited operands. This is
    // achieved by dynamically allocating all per-operand arrays based on actual NOp count.
    //
    // Trade-offs:
    // - Pro: No artificial limits, matches NumSharp's core philosophy
    // - Pro: Memory usage scales with actual usage, not worst case
    // - Pro: Enables complex multi-operand operations without artificial constraints
    // - Con: Slightly more complex allocation/deallocation
    // - Con: Cannot use simple fixed() statements, need explicit pointer management
    //
    // =====================================================================================

    /// <summary>
    /// Core iterator state with dynamically allocated arrays for both dimensions and operands.
    ///
    /// NUMSHARP DIVERGENCE: Unlike NumPy's fixed NPY_MAXDIMS=64 and NPY_MAXARGS=64,
    /// NumSharp supports unlimited dimensions AND unlimited operands. All arrays are
    /// allocated dynamically based on actual NDim and NOp values.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct NpyIterState
    {
        // =========================================================================
        // Constants
        // =========================================================================

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
        // Dynamically Allocated Dimension Arrays
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
        // Dynamically Allocated Per-Operand Arrays (NUMSHARP DIVERGENCE)
        // =========================================================================
        // Unlike NumPy's fixed NPY_MAXARGS=64, NumSharp supports unlimited operands.
        // All per-operand arrays are allocated based on actual NOp count.
        // =========================================================================

        /// <summary>Current data pointers for each operand. Size = NOp.</summary>
        public long* DataPtrs;

        /// <summary>Reset data pointers (base + offset). Size = NOp.</summary>
        public long* ResetDataPtrs;

        /// <summary>Base offsets for each operand. Size = NOp.</summary>
        public long* BaseOffsets;

        /// <summary>Per-operand flags. Size = NOp.</summary>
        public ushort* OpItFlags;

        /// <summary>Buffer/target dtypes for each operand. Size = NOp.</summary>
        public byte* OpDTypes;

        /// <summary>Source array dtypes for each operand (used for casting). Size = NOp.</summary>
        public byte* OpSrcDTypes;

        /// <summary>Element sizes for each operand (based on buffer dtype). Size = NOp.</summary>
        public int* ElementSizes;

        /// <summary>Source element sizes for each operand (based on source dtype). Size = NOp.</summary>
        public int* SrcElementSizes;

        /// <summary>
        /// Inner strides for each operand (gathered from main Strides array for fast access).
        /// Layout: [op0_inner_stride, op1_inner_stride, ...]
        /// Size = NOp.
        /// </summary>
        public long* InnerStrides;

        // =========================================================================
        // Buffer Data (when BUFFERED flag is set)
        // =========================================================================

        /// <summary>Buffer size (elements per buffer).</summary>
        public long BufferSize;

        /// <summary>Current buffer iteration end.</summary>
        public long BufIterEnd;

        /// <summary>Buffer pointers for each operand. Size = NOp.</summary>
        public long* Buffers;

        /// <summary>Buffer strides (always element size for contiguous buffers). Size = NOp.</summary>
        public long* BufStrides;

        // =========================================================================
        // Buffered Reduction Data (when BUFFERED + REDUCE flags are set)
        // =========================================================================
        // NumPy uses a double-loop pattern for buffered reduction:
        // - Inner loop: iterates through CoreSize elements (non-reduce dimensions)
        // - Outer loop: iterates ReduceOuterSize times (reduce dimension)
        //
        // The key insight: reduce operands have ReduceOuterStride=0, so their
        // pointer stays fixed while input advances, accumulating values.
        //
        // Reference: numpy/_core/src/multiarray/nditer_templ.c.src lines 131-210
        // =========================================================================

        /// <summary>
        /// Current position in reduce outer loop [0, ReduceOuterSize).
        /// Used by IsFirstVisit for buffered reduction.
        /// </summary>
        public long ReducePos;

        /// <summary>
        /// Size of reduce outer loop (transfersize / CoreSize).
        /// Number of times to iterate the reduce dimension within buffer.
        /// </summary>
        public long ReduceOuterSize;

        /// <summary>
        /// Inner loop size (number of inputs per output element).
        /// When reducing, Size is set to CoreSize and we iterate ReduceOuterSize times.
        /// </summary>
        public long CoreSize;

        /// <summary>
        /// Current position within core [0, CoreSize).
        /// Reset to 0 when advancing to next outer iteration.
        /// Used by IsFirstVisit - returns true only when CorePos = 0.
        /// </summary>
        public long CorePos;

        /// <summary>
        /// Which dimension is the reduce outer dimension.
        /// Used for stride calculation.
        /// </summary>
        public int OuterDim;

        /// <summary>
        /// Offset into core (for partial buffer fills).
        /// </summary>
        public long CoreOffset;

        /// <summary>
        /// Outer strides for reduction (stride per reduce outer iteration).
        /// Layout: [op0_reduce_stride, op1_reduce_stride, ...]
        /// When stride is 0, the operand is a reduction target for that axis.
        /// Size = NOp.
        /// </summary>
        public long* ReduceOuterStrides;

        /// <summary>
        /// Reset pointers for outer loop iteration.
        /// After completing inner loop, we advance these by ReduceOuterStrides.
        /// Layout: [op0_ptr, op1_ptr, ...]
        /// Size = NOp.
        /// </summary>
        public long* ReduceOuterPtrs;

        /// <summary>
        /// Array positions at buffer start, used for writeback.
        /// Stored separately from ResetDataPtrs which is the base for GotoIterIndex.
        /// Layout: [op0_ptr, op1_ptr, ...]
        /// Size = NOp.
        /// </summary>
        public long* ArrayWritebackPtrs;

        // =========================================================================
        // Private allocation tracking
        // =========================================================================

        /// <summary>Pointer to dimension arrays block (for freeing).</summary>
        private void* _dimArraysBlock;

        /// <summary>Pointer to operand arrays block (for freeing).</summary>
        private void* _opArraysBlock;

        // =========================================================================
        // Allocation and Deallocation
        // =========================================================================

        /// <summary>
        /// Allocate all dynamic arrays for given ndim and nop.
        /// Must be called before using any pointer fields.
        /// Initializes Perm to identity permutation [0, 1, 2, ...].
        /// </summary>
        public void AllocateDimArrays(int ndim, int nop)
        {
            if (ndim < 0) throw new ArgumentOutOfRangeException(nameof(ndim));
            if (nop < 1) throw new ArgumentOutOfRangeException(nameof(nop), "At least one operand is required");

            NDim = ndim;
            NOp = nop;
            StridesNDim = ndim;

            // =========================================================================
            // Allocate dimension-dependent arrays
            // =========================================================================
            if (ndim == 0)
            {
                // Scalar case - no dimension arrays needed
                Shape = null;
                Coords = null;
                Perm = null;
                Strides = null;
                _dimArraysBlock = null;
            }
            else
            {
                // Allocate all dimension arrays in one contiguous block for cache efficiency
                // Layout: [Shape: ndim longs][Coords: ndim longs][Strides: ndim*nop longs][Perm: ndim sbytes]
                long shapeBytes = ndim * sizeof(long);
                long coordsBytes = ndim * sizeof(long);
                long stridesBytes = ndim * nop * sizeof(long);
                long permBytes = ndim * sizeof(sbyte);

                // Align perm to 8 bytes for cleaner memory layout
                long permBytesAligned = (permBytes + 7) & ~7L;

                long totalDimBytes = shapeBytes + coordsBytes + stridesBytes + permBytesAligned;

                byte* dimBlock = (byte*)NativeMemory.AllocZeroed((nuint)totalDimBytes);
                _dimArraysBlock = dimBlock;

                Shape = (long*)dimBlock;
                Coords = (long*)(dimBlock + shapeBytes);
                Strides = (long*)(dimBlock + shapeBytes + coordsBytes);
                Perm = (sbyte*)(dimBlock + shapeBytes + coordsBytes + stridesBytes);

                // Initialize Perm to identity permutation
                // Perm[internal_axis] = original_axis
                for (int d = 0; d < ndim; d++)
                    Perm[d] = (sbyte)d;
            }

            // =========================================================================
            // Allocate per-operand arrays (NUMSHARP DIVERGENCE: unlimited operands)
            // =========================================================================
            // Layout: All long* arrays first (8-byte aligned), then int* arrays, then smaller types
            // This ensures proper alignment for all array types.
            //
            // long arrays (8 bytes each element):
            //   DataPtrs, ResetDataPtrs, BaseOffsets, InnerStrides, Buffers, BufStrides,
            //   ReduceOuterStrides, ReduceOuterPtrs, ArrayWritebackPtrs = 9 arrays
            // int arrays (4 bytes each element):
            //   ElementSizes, SrcElementSizes = 2 arrays
            // ushort arrays (2 bytes each element):
            //   OpItFlags = 1 array
            // byte arrays (1 byte each element):
            //   OpDTypes, OpSrcDTypes = 2 arrays

            long longArraysBytes = 9L * nop * sizeof(long);
            long intArraysBytes = 2L * nop * sizeof(int);
            long ushortArraysBytes = 1L * nop * sizeof(ushort);
            long byteArraysBytes = 2L * nop * sizeof(byte);

            // Align sections to 8 bytes
            long intArraysStart = longArraysBytes;
            long ushortArraysStart = intArraysStart + intArraysBytes;
            ushortArraysStart = (ushortArraysStart + 7) & ~7L; // Align to 8
            long byteArraysStart = ushortArraysStart + ushortArraysBytes;
            byteArraysStart = (byteArraysStart + 7) & ~7L; // Align to 8

            long totalOpBytes = byteArraysStart + byteArraysBytes;

            byte* opBlock = (byte*)NativeMemory.AllocZeroed((nuint)totalOpBytes);
            _opArraysBlock = opBlock;

            // Assign long* arrays (9 arrays, each nop elements)
            long* longPtr = (long*)opBlock;
            DataPtrs = longPtr; longPtr += nop;
            ResetDataPtrs = longPtr; longPtr += nop;
            BaseOffsets = longPtr; longPtr += nop;
            InnerStrides = longPtr; longPtr += nop;
            Buffers = longPtr; longPtr += nop;
            BufStrides = longPtr; longPtr += nop;
            ReduceOuterStrides = longPtr; longPtr += nop;
            ReduceOuterPtrs = longPtr; longPtr += nop;
            ArrayWritebackPtrs = longPtr;

            // Assign int* arrays (2 arrays, each nop elements)
            int* intPtr = (int*)(opBlock + intArraysStart);
            ElementSizes = intPtr; intPtr += nop;
            SrcElementSizes = intPtr;

            // Assign ushort* array (1 array, nop elements)
            OpItFlags = (ushort*)(opBlock + ushortArraysStart);

            // Assign byte* arrays (2 arrays, each nop elements)
            byte* bytePtr = (byte*)(opBlock + byteArraysStart);
            OpDTypes = bytePtr; bytePtr += nop;
            OpSrcDTypes = bytePtr;
        }

        /// <summary>
        /// Free all dynamically allocated arrays. Must be called before freeing the state itself.
        /// </summary>
        public void FreeDimArrays()
        {
            // Free dimension arrays block
            if (_dimArraysBlock != null)
            {
                NativeMemory.Free(_dimArraysBlock);
                _dimArraysBlock = null;
                Shape = null;
                Coords = null;
                Strides = null;
                Perm = null;
            }

            // Free operand arrays block
            if (_opArraysBlock != null)
            {
                NativeMemory.Free(_opArraysBlock);
                _opArraysBlock = null;
                DataPtrs = null;
                ResetDataPtrs = null;
                BaseOffsets = null;
                OpItFlags = null;
                OpDTypes = null;
                OpSrcDTypes = null;
                ElementSizes = null;
                SrcElementSizes = null;
                InnerStrides = null;
                Buffers = null;
                BufStrides = null;
                ReduceOuterStrides = null;
                ReduceOuterPtrs = null;
                ArrayWritebackPtrs = null;
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
            return (void*)DataPtrs[op];
        }

        /// <summary>Set current data pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetDataPtr(int op, void* ptr)
        {
            DataPtrs[op] = (long)ptr;
        }

        /// <summary>Get data pointer (legacy interface).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly IntPtr GetDataPointer(int operand)
        {
            return (IntPtr)DataPtrs[operand];
        }

        /// <summary>Set data pointer (legacy interface).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetDataPointer(int operand, IntPtr pointer)
        {
            DataPtrs[operand] = (long)pointer;
        }

        /// <summary>Get reset data pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void* GetResetDataPtr(int op)
        {
            return (void*)ResetDataPtrs[op];
        }

        /// <summary>Set reset data pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetResetDataPtr(int op, void* ptr)
        {
            ResetDataPtrs[op] = (long)ptr;
        }

        /// <summary>Get operand dtype.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public NPTypeCode GetOpDType(int op)
        {
            return (NPTypeCode)OpDTypes[op];
        }

        /// <summary>Set operand dtype (buffer/target dtype).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetOpDType(int op, NPTypeCode dtype)
        {
            OpDTypes[op] = (byte)dtype;
            ElementSizes[op] = InfoOf.GetSize(dtype);
        }

        /// <summary>Get source array dtype for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public NPTypeCode GetOpSrcDType(int op)
        {
            return (NPTypeCode)OpSrcDTypes[op];
        }

        /// <summary>Set source array dtype for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetOpSrcDType(int op, NPTypeCode dtype)
        {
            OpSrcDTypes[op] = (byte)dtype;
            SrcElementSizes[op] = InfoOf.GetSize(dtype);
        }

        /// <summary>Get source element size for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetSrcElementSize(int op)
        {
            return SrcElementSizes[op];
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
            return (NpyIterOpFlags)OpItFlags[op];
        }

        /// <summary>Set operand flags.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetOpFlags(int op, NpyIterOpFlags flags)
        {
            OpItFlags[op] = (ushort)flags;
        }

        /// <summary>Get element size for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetElementSize(int op)
        {
            return ElementSizes[op];
        }

        /// <summary>Get buffer pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void* GetBuffer(int op)
        {
            return (void*)Buffers[op];
        }

        /// <summary>Set buffer pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetBuffer(int op, void* ptr)
        {
            Buffers[op] = (long)ptr;
        }

        /// <summary>Get buffer stride for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long GetBufStride(int op)
        {
            return BufStrides[op];
        }

        /// <summary>Set buffer stride for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetBufStride(int op, long stride)
        {
            BufStrides[op] = stride;
        }

        /// <summary>Get reduce outer stride for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long GetReduceOuterStride(int op)
        {
            return ReduceOuterStrides[op];
        }

        /// <summary>Set reduce outer stride for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetReduceOuterStride(int op, long stride)
        {
            ReduceOuterStrides[op] = stride;
        }

        /// <summary>Get reduce outer pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void* GetReduceOuterPtr(int op)
        {
            return (void*)ReduceOuterPtrs[op];
        }

        /// <summary>Set reduce outer pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetReduceOuterPtr(int op, void* ptr)
        {
            ReduceOuterPtrs[op] = (long)ptr;
        }

        /// <summary>Get array writeback pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void* GetArrayWritebackPtr(int op)
        {
            return (void*)ArrayWritebackPtrs[op];
        }

        /// <summary>Set array writeback pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetArrayWritebackPtr(int op, void* ptr)
        {
            ArrayWritebackPtrs[op] = (long)ptr;
        }

        /// <summary>
        /// Get inner stride array pointer - returns contiguous array of inner strides for all operands.
        /// Layout: [op0_inner_stride, op1_inner_stride, ...]
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long* GetInnerStrideArray()
        {
            return InnerStrides;
        }

        /// <summary>
        /// Update the InnerStrides array from the main Strides array.
        /// Must be called after coalescing or axis removal.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void UpdateInnerStrides()
        {
            if (NDim == 0)
            {
                // Scalar - all inner strides are 0
                for (int op = 0; op < NOp; op++)
                    InnerStrides[op] = 0;
                return;
            }

            int innerAxis = NDim - 1;
            for (int op = 0; op < NOp; op++)
                InnerStrides[op] = Strides[op * StridesNDim + innerAxis];
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

            for (int axis = NDim - 1; axis >= 0; axis--)
            {
                Coords[axis]++;

                if (Coords[axis] < Shape[axis])
                {
                    // Advance data pointers along this axis
                    for (int op = 0; op < NOp; op++)
                    {
                        long stride = Strides[op * StridesNDim + axis];
                        DataPtrs[op] += stride * ElementSizes[op];
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
                    DataPtrs[op] -= stride * (axisShape - 1) * ElementSizes[op];
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
        /// Buffered reduce iteration advance.
        /// Implements NumPy's double-loop pattern for efficient buffered reduction.
        ///
        /// Returns:
        /// - 1: More elements in current buffer (inner or outer loop)
        /// - 0: Buffer exhausted, need to refill
        /// - -1: Iteration complete
        ///
        /// Reference: numpy/_core/src/multiarray/nditer_templ.c.src lines 131-210
        /// </summary>
        public int BufferedReduceAdvance()
        {
            // === INNER LOOP INCREMENT ===
            // Check if we can advance within the current core (inner loop)
            if (++IterIndex < BufIterEnd)
            {
                // Still within core - advance pointers by buffer strides
                // Also track position within core for IsFirstVisit
                CorePos++;

                for (int op = 0; op < NOp; op++)
                {
                    DataPtrs[op] += BufStrides[op];
                }
                return 1;  // More elements
            }

            // === OUTER LOOP INCREMENT (the double-loop magic) ===
            // Inner loop exhausted, try advancing the reduce outer loop
            if (++ReducePos < ReduceOuterSize)
            {
                // Reset core position for new outer iteration
                CorePos = 0;

                // Advance to next reduce position without re-buffering
                for (int op = 0; op < NOp; op++)
                {
                    // Advance outer pointer by reduce outer stride
                    long ptr = ReduceOuterPtrs[op] + ReduceOuterStrides[op];
                    DataPtrs[op] = ptr;       // Current pointer
                    ReduceOuterPtrs[op] = ptr;      // Save for next outer iteration
                }

                // Reset inner loop bounds
                // Note: Size holds CoreSize when reducing
                BufIterEnd = IterIndex + CoreSize;
                return 1;  // More elements (restart inner loop)
            }

            // === BUFFER EXHAUSTED ===
            // Both inner and outer loops exhausted
            // Check if we're past the end
            if (IterIndex >= IterEnd)
            {
                return -1;  // Iteration complete
            }

            // Need to refill buffers - return 0 to signal caller
            return 0;
        }

        /// <summary>
        /// Initialize reduce outer pointers from current data pointers.
        /// Called after buffer fill to set up the outer loop start positions.
        /// </summary>
        public void InitReduceOuterPtrs()
        {
            for (int op = 0; op < NOp; op++)
            {
                ReduceOuterPtrs[op] = DataPtrs[op];
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

            for (int op = 0; op < NOp; op++)
                DataPtrs[op] = ResetDataPtrs[op];

            // Invalidate all buffer reuse flags since position changed
            InvalidateAllBufferReuse();
        }

        /// <summary>
        /// Invalidate buffer reuse flags for all operands.
        /// Called when iterator position changes (Reset, GotoIterIndex).
        /// </summary>
        private void InvalidateAllBufferReuse()
        {
            for (int op = 0; op < NOp; op++)
            {
                OpItFlags[op] = (ushort)(OpItFlags[op] & ~(ushort)NpyIterOpFlags.BUF_REUSABLE);
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
            for (int op = 0; op < NOp; op++)
            {
                long offset = 0;
                for (int d = 0; d < NDim; d++)
                {
                    offset += Coords[d] * Strides[op * StridesNDim + d];
                }
                DataPtrs[op] = ResetDataPtrs[op] + offset * ElementSizes[op];
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
