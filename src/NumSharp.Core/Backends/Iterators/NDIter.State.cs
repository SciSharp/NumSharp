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
    public unsafe struct NDIterState
    {
        // =========================================================================
        // Constants
        // =========================================================================

        /// <summary>
        /// Threshold for using stackalloc vs heap allocation for temporary buffers.
        /// Arrays with more dimensions than this will use heap allocation.
        /// </summary>
        public const int StackAllocThreshold = 64;

        // =========================================================================
        // Core Scalar Fields
        // =========================================================================

        /// <summary>Iterator flags (NDIterFlags bitmask).</summary>
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
        ///     The tracked flat index at the iteration's coordinate origin (every <see cref="Coords"/> entry 0): NumPy's
        ///     reset value of the index "operand", which starts at 0 and moves to <c>(shape - 1) * stride</c> along every
        ///     axis the construction flipped. <see cref="FlatIndex"/> is <c>IndexReset + Σ Coords[d] * IndexStrides[d]</c>.
        ///     Meaningful only when HASINDEX is set (set by <see cref="ComputeIndexStrides"/>).
        /// </summary>
        public long IndexReset;

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
        public NDIterFlags Flags
        {
            readonly get => (NDIterFlags)(ItFlags & 0xFFFF);
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
        ///     Per iteration axis, the step of the tracked C or F flat index (NumPy keeps the index as one more "operand"
        ///     whose strides count index units). Size = <see cref="StridesNDim"/> (the axes can only shrink). Computed by
        ///     <see cref="ComputeIndexStrides"/> once the axes are ordered and flipped, then permuted and merged by
        ///     coalescing exactly like an operand's strides — which is what lets an index-tracking iterator coalesce when
        ///     the index allows it, as NumPy's does, without losing the index. Meaningful only when HASINDEX is set.
        /// </summary>
        public long* IndexStrides;

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

        /// <summary>
        ///     The fixed-width string CLASS of each operand, one byte per operand: bit <see cref="StringBufferUnicode"/> is set
        ///     when the operand's ITERATION (buffer) descriptor is <c>U</c>, bit <see cref="StringSourceUnicode"/> when its ARRAY
        ///     descriptor is <c>U</c>; zero for a numeric operand and for <c>S</c>. A string type code names no dtype (both
        ///     classes share <see cref="NPTypeCode.String"/>, and the width lives on the descriptor), so this byte and the
        ///     element sizes are what a buffered string transfer reads: class and width on both sides. Size = NOp.
        /// </summary>
        public byte* OpStringKinds;

        /// <summary><see cref="OpStringKinds"/> bit: the operand's iteration (buffer) descriptor is <c>U</c> (UCS-4 units).</summary>
        public const byte StringBufferUnicode = 1;

        /// <summary><see cref="OpStringKinds"/> bit: the operand array's own descriptor is <c>U</c> (UCS-4 units).</summary>
        public const byte StringSourceUnicode = 2;

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

        /// <summary>
        /// Number of elements in the current buffer fill (the inner-loop count
        /// the kernel must process). Matches NumPy's NBF_SIZE. For the
        /// non-reduce buffered path, <see cref="BufIterEnd"/> = fill-start
        /// IterIndex + BufTransferSize.
        /// </summary>
        public long BufTransferSize;

        /// <summary>
        /// 1 when the current fill's WRITE operands have been flushed back to
        /// their arrays (or no fill is pending). Lets flush be idempotent so
        /// iternext, the single-inner-loop fast path, and Dispose can all call
        /// it safely. Matches the role of NumPy's "buffers written" tracking
        /// in npyiter_copy_from_buffers.
        /// </summary>
        public byte BufFlushed;

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
        // Inline arena (single-block state allocation)
        // =========================================================================
        //
        // NDIterRef.AllocateStateBlock allocates the state header and a trailing arena in ONE
        // native block, and AllocateDimArrays carves the dimension + per-operand arrays out of
        // that arena instead of calling the allocator twice more. Measured: the three
        // calloc/free pairs were 78 ns of a 161 ns three-operand construction (one 616-byte
        // calloc/free is 29 ns), i.e. the allocator was ~half of the iterator's fixed setup
        // cost. The block is recycled through a small per-thread cache
        // (NDIterRef.ReleaseStateBlock) so the construct→execute→dispose pattern pays no
        // allocator call at all — NumPy's NpyIter is likewise one malloc'd blob.
        //
        // A state whose arrays exceed the arena (very high ndim × nop) falls back to the
        // separate blocks exactly as before; a state built on the stack
        // (CreateCopyState/CreateReductionState) has no arena and keeps the original path.

        /// <summary>Start of the inline arena (null when this state has none).</summary>
        private byte* _arenaBase;

        /// <summary>Capacity of the inline arena in bytes.</summary>
        private int _arenaCapacity;

        /// <summary>Bytes of the inline arena carved by <see cref="AllocateDimArrays"/> (0 = untouched).</summary>
        private int _arenaUsed;

        /// <summary>True when this state owns an inline arena (allocated by <c>NDIterRef.AllocateStateBlock</c>).</summary>
        internal readonly bool HasInlineArena => _arenaBase != null;

        /// <summary>
        /// Attach the trailing arena of a single-block allocation. Must be called on a
        /// zeroed header before <see cref="AllocateDimArrays"/>; the arena itself must be
        /// zeroed (AllocateDimArrays relies on AllocZeroed semantics for every array it
        /// carves: Coords, BaseOffsets, Buffers, ... start at zero).
        /// </summary>
        internal void AttachInlineArena(byte* arena, int capacity)
        {
            _arenaBase = arena;
            _arenaCapacity = capacity;
            _arenaUsed = 0;
        }

        /// <summary>
        /// Return the block to a pristine state so it can host a fresh iterator: the header
        /// and the CARVED part of the arena are re-zeroed (never the whole arena — only what
        /// the last iterator touched), the arena descriptor survives. Any external blocks
        /// and buffers must already have been released (FreeBuffers / FreeDimArrays).
        /// </summary>
        internal void ResetForRecycle()
        {
            byte* arena = _arenaBase;
            int capacity = _arenaCapacity;
            int used = _arenaUsed;
            if (used > 0)
                NativeMemory.Clear(arena, (nuint)used);
            this = default;
            _arenaBase = arena;
            _arenaCapacity = capacity;
        }

        // =========================================================================
        // Allocation and Deallocation
        // =========================================================================

        /// <summary>
        /// Allocate all dynamic arrays for given ndim and nop.
        /// Must be called before using any pointer fields.
        /// Initializes Perm to identity permutation [0, 1, 2, ...].
        /// </summary>
        public void AllocateDimArrays(int ndim, int nop, int stridesNDim = -1)
        {
            if (ndim < 0) throw new ArgumentOutOfRangeException(nameof(ndim));
            if (nop < 1) throw new ArgumentOutOfRangeException(nameof(nop), "At least one operand is required");
            if (stridesNDim < 0) stridesNDim = ndim;
            if (stridesNDim < ndim) throw new ArgumentOutOfRangeException(nameof(stridesNDim));

            NDim = ndim;
            NOp = nop;
            StridesNDim = stridesNDim;

            // =========================================================================
            // Size the dimension-dependent arrays
            // =========================================================================
            // Dimension arrays live in one contiguous block for cache efficiency.
            // Layout: [Shape: ndim longs][Coords: ndim longs][Strides: stridesNDim*nop longs]
            //         [IndexStrides: stridesNDim longs][Perm: ndim sbytes]
            bool hasDimArrays = !(ndim == 0 && stridesNDim == 0);
            long shapeBytes = ndim * sizeof(long);
            long coordsBytes = ndim * sizeof(long);
            long stridesBytes = stridesNDim * nop * sizeof(long);
            long indexStridesBytes = stridesNDim * sizeof(long);
            long permBytes = ndim * sizeof(sbyte);

            // Align perm to 8 bytes for cleaner memory layout (and so the operand block that
            // follows it inside an inline arena stays 8-byte aligned).
            long permBytesAligned = (permBytes + 7) & ~7L;

            long totalDimBytes = hasDimArrays ? shapeBytes + coordsBytes + stridesBytes + indexStridesBytes + permBytesAligned : 0;

            // =========================================================================
            // Size the per-operand arrays (NUMSHARP DIVERGENCE: unlimited operands)
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
            //   OpDTypes, OpSrcDTypes, OpStringKinds = 3 arrays

            long longArraysBytes = 9L * nop * sizeof(long);
            long intArraysBytes = 2L * nop * sizeof(int);
            long ushortArraysBytes = 1L * nop * sizeof(ushort);
            long byteArraysBytes = 3L * nop * sizeof(byte);

            // Align sections to 8 bytes
            long intArraysStart = longArraysBytes;
            long ushortArraysStart = intArraysStart + intArraysBytes;
            ushortArraysStart = (ushortArraysStart + 7) & ~7L; // Align to 8
            long byteArraysStart = ushortArraysStart + ushortArraysBytes;
            byteArraysStart = (byteArraysStart + 7) & ~7L; // Align to 8

            long totalOpBytes = byteArraysStart + byteArraysBytes;

            // =========================================================================
            // Obtain the memory: carve the inline arena when it fits, else allocate
            // =========================================================================
            // The arena is zeroed (AllocZeroed at block creation, ResetForRecycle on reuse),
            // so carved arrays carry the same all-zero initial contents as the AllocZeroed
            // blocks they replace. Both blocks are carved from ONE arena, dim block first:
            // totalDimBytes is a multiple of 8, so the operand block keeps its alignment.
            byte* dimBlock;
            byte* opBlock;
            if (_arenaBase != null && _arenaUsed == 0 && totalDimBytes + totalOpBytes <= _arenaCapacity)
            {
                dimBlock = hasDimArrays ? _arenaBase : null;
                opBlock = _arenaBase + totalDimBytes;
                _arenaUsed = (int)(totalDimBytes + totalOpBytes);
                _dimArraysBlock = null;   // not separately owned — freed with the block
                _opArraysBlock = null;
            }
            else
            {
                dimBlock = hasDimArrays ? (byte*)NativeMemory.AllocZeroed((nuint)totalDimBytes) : null;
                _dimArraysBlock = dimBlock;
                opBlock = (byte*)NativeMemory.AllocZeroed((nuint)totalOpBytes);
                _opArraysBlock = opBlock;
            }

            // =========================================================================
            // Lay out the dimension-dependent arrays
            // =========================================================================
            if (!hasDimArrays)
            {
                // Scalar case - no dimension arrays needed
                Shape = null;
                Coords = null;
                Perm = null;
                Strides = null;
                IndexStrides = null;
            }
            else
            {
                Shape = (long*)dimBlock;
                Coords = (long*)(dimBlock + shapeBytes);
                Strides = (long*)(dimBlock + shapeBytes + coordsBytes);
                IndexStrides = (long*)(dimBlock + shapeBytes + coordsBytes + stridesBytes);
                Perm = (sbyte*)(dimBlock + shapeBytes + coordsBytes + stridesBytes + indexStridesBytes);

                // Initialize Perm to identity permutation
                // Perm[internal_axis] = original_axis
                for (int d = 0; d < ndim; d++)
                    Perm[d] = (sbyte)d;
            }

            // =========================================================================
            // Lay out the per-operand arrays
            // =========================================================================

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

            // Assign byte* arrays (3 arrays, each nop elements)
            byte* bytePtr = (byte*)(opBlock + byteArraysStart);
            OpDTypes = bytePtr; bytePtr += nop;
            OpSrcDTypes = bytePtr; bytePtr += nop;
            OpStringKinds = bytePtr;
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
                IndexStrides = null;
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
                OpStringKinds = null;
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public long* GetShapePointer() => Shape;

        /// <summary>Get pointer to Coords array.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public long* GetCoordsPointer() => Coords;

        /// <summary>
        /// Get pointer to strides for a specific operand.
        /// Uses actual NDim (or StridesNDim if NDim changed after allocation).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public long* GetStridesPointer(int operand)
        {
            if ((uint)operand >= (uint)NOp)
                throw new ArgumentOutOfRangeException(nameof(operand));

            return Strides + (operand * StridesNDim);
        }

        /// <summary>Get stride for operand at axis.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public long GetStride(int axis, int op)
        {
            return Strides[op * StridesNDim + axis];
        }

        /// <summary>Set stride for operand at axis.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetStride(int axis, int op, long value)
        {
            Strides[op * StridesNDim + axis] = value;
        }

        /// <summary>Get current data pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void* GetDataPtr(int op)
        {
            return (void*)DataPtrs[op];
        }

        /// <summary>Set current data pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetDataPtr(int op, void* ptr)
        {
            DataPtrs[op] = (long)ptr;
        }

        /// <summary>Get data pointer (legacy interface).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public readonly IntPtr GetDataPointer(int operand)
        {
            return (IntPtr)DataPtrs[operand];
        }

        /// <summary>Set data pointer (legacy interface).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetDataPointer(int operand, IntPtr pointer)
        {
            DataPtrs[operand] = (long)pointer;
        }

        /// <summary>Get reset data pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void* GetResetDataPtr(int op)
        {
            return (void*)ResetDataPtrs[op];
        }

        /// <summary>Set reset data pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetResetDataPtr(int op, void* ptr)
        {
            ResetDataPtrs[op] = (long)ptr;
        }

        /// <summary>Get operand dtype.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public NPTypeCode GetOpDType(int op)
        {
            return (NPTypeCode)OpDTypes[op];
        }

        /// <summary>Set operand dtype (buffer/target dtype).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetOpDType(int op, NPTypeCode dtype)
        {
            OpDTypes[op] = (byte)dtype;
            ElementSizes[op] = InfoOf.GetSize(dtype);
        }

        /// <summary>
        ///     Sets an operand's iteration (buffer) dtype from its DESCRIPTOR: the type code, the element size in bytes, and
        ///     for a fixed-width string the class bit of <see cref="OpStringKinds"/>. The only way to give a string operand
        ///     its element size: <see cref="NPTypeCode.String"/> has none of its own (<c>&lt;U5</c> is 20 bytes, <c>|S5</c> 5).
        /// </summary>
        /// <param name="op">The operand index.</param>
        /// <param name="descr">The iteration descriptor (sized; a string one native-endian).</param>
        /// <exception cref="NotSupportedException"><paramref name="descr"/> has no storage lane (a datetime descriptor
        ///     before Stage C).</exception>
        public void SetOpDType(int op, DType descr)
        {
            var code = descr.GetTypeCode();
            OpDTypes[op] = (byte)code;
            if (code == NPTypeCode.String)
            {
                ElementSizes[op] = descr.itemsize;
                OpStringKinds[op] = (byte)((OpStringKinds[op] & ~StringBufferUnicode) |
                                           (descr.kind == 'U' ? StringBufferUnicode : 0));
            }
            else
            {
                ElementSizes[op] = InfoOf.GetSize(code);
                OpStringKinds[op] = (byte)(OpStringKinds[op] & ~StringBufferUnicode);
            }
        }

        /// <summary>
        ///     Sets an operand's SOURCE (array) dtype from its descriptor — the counterpart of
        ///     <see cref="SetOpDType(int, DType)"/> for <see cref="OpSrcDTypes"/>, <see cref="SrcElementSizes"/> and the source
        ///     class bit of <see cref="OpStringKinds"/>.
        /// </summary>
        /// <param name="op">The operand index.</param>
        /// <param name="descr">The operand array's descriptor.</param>
        /// <exception cref="NotSupportedException"><paramref name="descr"/> has no storage lane.</exception>
        public void SetOpSrcDType(int op, DType descr)
        {
            var code = descr.GetTypeCode();
            OpSrcDTypes[op] = (byte)code;
            if (code == NPTypeCode.String)
            {
                SrcElementSizes[op] = descr.itemsize;
                OpStringKinds[op] = (byte)((OpStringKinds[op] & ~StringSourceUnicode) |
                                           (descr.kind == 'U' ? StringSourceUnicode : 0));
            }
            else
            {
                SrcElementSizes[op] = InfoOf.GetSize(code);
                OpStringKinds[op] = (byte)(OpStringKinds[op] & ~StringSourceUnicode);
            }
        }

        /// <summary>Whether either side of operand <paramref name="op"/> (its array or its iteration dtype) is a fixed-width string.</summary>
        /// <param name="op">The operand index.</param>
        /// <returns>True when a buffer transfer of this operand must take the string route.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public bool IsStringOp(int op)
            => (NPTypeCode)OpDTypes[op] == NPTypeCode.String || (NPTypeCode)OpSrcDTypes[op] == NPTypeCode.String;

        /// <summary>
        ///     The descriptor of operand <paramref name="op"/>'s ITERATION (buffer) side, rebuilt from its type code, element
        ///     size and string class — the state stores no descriptor objects, and a string's width and class are exactly the
        ///     two facts a type code drops. Byte order is not stored (buffers are native), so a non-native request reads back
        ///     native.
        /// </summary>
        /// <param name="op">The operand index.</param>
        /// <returns>A sized descriptor (for messages and debug output; not cached).</returns>
        public DType GetOpDescr(int op)
            => DescrOf((NPTypeCode)OpDTypes[op], ElementSizes[op], (OpStringKinds[op] & StringBufferUnicode) != 0);

        /// <summary>The descriptor of operand <paramref name="op"/>'s ARRAY side — the counterpart of <see cref="GetOpDescr"/>.</summary>
        /// <param name="op">The operand index.</param>
        /// <returns>A sized descriptor (for messages and debug output; not cached).</returns>
        public DType GetOpSrcDescr(int op)
            => DescrOf((NPTypeCode)OpSrcDTypes[op], SrcElementSizes[op], (OpStringKinds[op] & StringSourceUnicode) != 0);

        /// <summary>A descriptor from a stored (type code, element size, unicode bit) triple.</summary>
        /// <param name="code">The storage lane.</param>
        /// <param name="size">The element size in bytes.</param>
        /// <param name="unicode">For the string lane: <c>U</c> rather than <c>S</c>.</param>
        /// <returns>The descriptor (a fresh one for a string, the class singleton otherwise).</returns>
        private static DType DescrOf(NPTypeCode code, int size, bool unicode)
            => code != NPTypeCode.String
                ? NumSharp.DType.From(code)
                : unicode ? DTypeRegistry.Str.Descr(size / 4) : DTypeRegistry.Bytes.Descr(size);

        /// <summary>Get source array dtype for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public NPTypeCode GetOpSrcDType(int op)
        {
            return (NPTypeCode)OpSrcDTypes[op];
        }

        /// <summary>Set source array dtype for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetOpSrcDType(int op, NPTypeCode dtype)
        {
            OpSrcDTypes[op] = (byte)dtype;
            SrcElementSizes[op] = InfoOf.GetSize(dtype);
        }

        /// <summary>Get source element size for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public int GetSrcElementSize(int op)
        {
            return SrcElementSizes[op];
        }

        /// <summary>
        ///     Check if operand needs casting (source dtype != buffer dtype). Two string sides share one type code, so for
        ///     them the width and the class decide: <c>&lt;U3</c> into a <c>&lt;U5</c> buffer, or <c>|S3</c> into <c>&lt;U3</c>, is a
        ///     cast (NumPy's <c>string_to_string</c> zero-pad copy / its <c>S ↔ U</c> casts).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public bool NeedsCast(int op)
        {
            var src = GetOpSrcDType(op);
            if (src != GetOpDType(op))
                return true;
            if (src != NPTypeCode.String)
                return false;
            int kinds = OpStringKinds[op];
            bool srcUnicode = (kinds & StringSourceUnicode) != 0;
            bool bufUnicode = (kinds & StringBufferUnicode) != 0;
            return SrcElementSizes[op] != ElementSizes[op] || srcUnicode != bufUnicode;
        }

        /// <summary>Get operand flags.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public NDIterOpFlags GetOpFlags(int op)
        {
            return (NDIterOpFlags)OpItFlags[op];
        }

        /// <summary>Set operand flags.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetOpFlags(int op, NDIterOpFlags flags)
        {
            OpItFlags[op] = (ushort)flags;
        }

        /// <summary>Get element size for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public int GetElementSize(int op)
        {
            return ElementSizes[op];
        }

        /// <summary>Get buffer pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void* GetBuffer(int op)
        {
            return (void*)Buffers[op];
        }

        /// <summary>Set buffer pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetBuffer(int op, void* ptr)
        {
            Buffers[op] = (long)ptr;
        }

        /// <summary>Get buffer stride for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public long GetBufStride(int op)
        {
            return BufStrides[op];
        }

        /// <summary>Set buffer stride for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetBufStride(int op, long stride)
        {
            BufStrides[op] = stride;
        }

        /// <summary>Get reduce outer stride for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public long GetReduceOuterStride(int op)
        {
            return ReduceOuterStrides[op];
        }

        /// <summary>Set reduce outer stride for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetReduceOuterStride(int op, long stride)
        {
            ReduceOuterStrides[op] = stride;
        }

        /// <summary>Get reduce outer pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void* GetReduceOuterPtr(int op)
        {
            return (void*)ReduceOuterPtrs[op];
        }

        /// <summary>Set reduce outer pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetReduceOuterPtr(int op, void* ptr)
        {
            ReduceOuterPtrs[op] = (long)ptr;
        }

        /// <summary>Get array writeback pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void* GetArrayWritebackPtr(int op)
        {
            return (void*)ArrayWritebackPtrs[op];
        }

        /// <summary>Set array writeback pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetArrayWritebackPtr(int op, void* ptr)
        {
            ArrayWritebackPtrs[op] = (long)ptr;
        }

        /// <summary>
        /// Get inner stride array pointer - returns contiguous array of inner strides for all operands.
        /// Layout: [op0_inner_stride, op1_inner_stride, ...]
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public long* GetInnerStrideArray()
        {
            return InnerStrides;
        }

        /// <summary>
        /// Update the InnerStrides array from the main Strides array.
        /// Must be called after coalescing or axis removal.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
            ((NDIterFlags)ItFlags & (NDIterFlags.SourceContiguous | NDIterFlags.DestinationContiguous)) ==
            (NDIterFlags.SourceContiguous | NDIterFlags.DestinationContiguous) &&
            ((NDIterFlags)ItFlags & NDIterFlags.SourceBroadcast) == 0;

        // =========================================================================
        // Iteration Methods
        // =========================================================================

        /// <summary>
        /// Advance iterator by one position using ripple carry.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Advance()
        {
            IterIndex++;

            // A tracked flat index advances by its own per-axis strides, exactly like a data pointer (NumPy keeps it as
            // the index "operand" next to the data pointers): O(1) per step, and correct after coalescing, where the
            // coordinates no longer name the operands' axes.
            bool needsFlatIndex = (ItFlags & (uint)NDIterFlags.HASINDEX) != 0;

            for (int axis = NDim - 1; axis >= 0; axis--)
            {
                Coords[axis]++;

                if (Coords[axis] < Shape[axis])
                {
                    // Advance data pointers along this axis.
                    // Strides are SOURCE-array element strides and DataPtrs point
                    // into source-array memory, so the byte multiplier must be the
                    // SOURCE element size. ElementSizes is the buffer dtype size,
                    // which diverges when a buffered cast is active (bug (b):
                    // int32 source buffered as float64 made every delta 2x).
                    for (int op = 0; op < NOp; op++)
                    {
                        long stride = Strides[op * StridesNDim + axis];
                        DataPtrs[op] += stride * SrcElementSizes[op];
                    }

                    if (needsFlatIndex)
                        FlatIndex += IndexStrides[axis];
                    return;
                }

                // Carry: reset this axis, continue to next
                Coords[axis] = 0;
                if (needsFlatIndex)
                    FlatIndex -= IndexStrides[axis] * (Shape[axis] - 1);

                // Reset data pointers for this axis (source element size — see above)
                for (int op = 0; op < NOp; op++)
                {
                    long stride = Strides[op * StridesNDim + axis];
                    long axisShape = Shape[axis];
                    DataPtrs[op] -= stride * (axisShape - 1) * SrcElementSizes[op];
                }
            }

            // If we reach here, all coords wrapped (end of iteration): every carry above moved the index back with its
            // axis, so it reads the origin's index again, as the coordinates do. Nothing reads it past the end.
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
        /// Reset iterator to IterStart (which may be >0 for ranged iterators).
        /// Matches NumPy's NDIter_Reset + npyiter_goto_iterindex(ITERSTART) semantics.
        /// </summary>
        public void Reset()
        {
            // Delegate to GotoIterIndex so Coords, FlatIndex, and DataPtrs all
            // agree with IterStart (critical for ranged iterators where IterStart > 0).
            FlatIndex = 0;
            GotoIterIndex(IterStart);
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
                OpItFlags[op] = (ushort)(OpItFlags[op] & ~(ushort)NDIterOpFlags.BUF_REUSABLE);
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
                // An empty iteration (an extent-0 axis that was not coalesced away — multi_index / an index keeps every
                // axis) has no position to decompose: its coordinates stay at the start. Dividing by the 0 extent threw
                // DivideByZeroException from reset() of np.nditer(empty, ["zerosize_ok", "multi_index"]).
                if (dimSize == 0)
                {
                    Coords[d] = 0;
                    continue;
                }

                Coords[d] = remaining % dimSize;
                remaining /= dimSize;
            }

            // Update flat index if tracking
            if ((ItFlags & (uint)NDIterFlags.HASINDEX) != 0)
            {
                FlatIndex = ComputeFlatIndex();
            }

            // Update data pointers. ResetDataPtrs are source-array bases and
            // Strides are source element strides, so the byte multiplier is the
            // SOURCE element size (ElementSizes is the buffer dtype size and
            // diverges under a buffered cast — bug (b)).
            for (int op = 0; op < NOp; op++)
            {
                long offset = 0;
                for (int d = 0; d < NDim; d++)
                {
                    offset += Coords[d] * Strides[op * StridesNDim + d];
                }
                DataPtrs[op] = ResetDataPtrs[op] + offset * SrcElementSizes[op];
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
            if ((ItFlags & (uint)NDIterFlags.HASINDEX) != 0)
            {
                FlatIndex = ComputeFlatIndex();
            }
        }

        /// <summary>
        ///     The tracked flat index at the current coordinates: <see cref="IndexReset"/> plus each coordinate times its
        ///     axis's <see cref="IndexStrides"/> entry — NumPy's index "pointer" computed the way
        ///     <c>npyiter_goto_iterindex</c> computes a data pointer.
        /// </summary>
        /// <returns>The C or F flat index of the current position (in the operands' own shape).</returns>
        /// <remarks>
        ///     Valid for any axis layout the construction or a later coalescing produced, because the strides were
        ///     permuted, flipped and merged with the axes. Recomputing the index from the coordinates through <see cref="Perm"/>
        ///     instead — what this did before — is only correct while every original axis is still its own iteration axis:
        ///     after <c>remove_multi_index</c> coalesced an <c>f_index</c> iterator, it counted in C order.
        /// </remarks>
        internal long ComputeFlatIndex()
        {
            long index = IndexReset;
            for (int d = 0; d < NDim; d++)
                index += Coords[d] * IndexStrides[d];
            return index;
        }

        /// <summary>
        ///     Lays out the tracked flat index as per-axis strides over the CURRENT axis order — NumPy's
        ///     <c>npyiter_compute_index_strides</c> followed by what its axis ordering and stride flips do to the index
        ///     "operand". Call once the axes are ordered and flipped and before any coalescing.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     Internal axis <c>d</c> holds original axis <c>a</c> (<see cref="Perm"/>; <c>-1 - a</c> when the construction
        ///     flipped it). Its index stride is the C stride of axis <c>a</c> in the operands' own shape (the product of the
        ///     extents after it) for <c>c_index</c>, the F stride (the product before it) for <c>f_index</c> — and 0 for an
        ///     axis of extent 1, as NumPy sets it, so such an axis coalesces away. A flipped axis walks its elements
        ///     backward: its stride is negated and the origin's index moves to <c>(shape - 1) * stride</c>
        ///     (<see cref="IndexReset"/>), exactly as <c>npyiter_flip_negative_strides</c> adjusts the index's base offset.
        ///     </para>
        ///     <para>
        ///     Not valid after a coalescing has merged axes (the original axes are gone): coalescing keeps the strides it
        ///     merges instead.
        ///     </para>
        /// </remarks>
        internal void ComputeIndexStrides()
        {
            IndexReset = 0;
            if (NDim == 0)
                return;

            bool hasNegPerm = (ItFlags & (uint)NDIterFlags.NEGPERM) != 0;

            // The operands' own shape, through the permutation.
            var origShape = stackalloc long[NDim];
            for (int d = 0; d < NDim; d++)
            {
                int p = Perm[d];
                origShape[hasNegPerm && p < 0 ? -1 - p : p] = Shape[d];
            }

            for (int d = 0; d < NDim; d++)
            {
                int p = Perm[d];
                bool flipped = hasNegPerm && p < 0;
                int axis = flipped ? -1 - p : p;

                long stride = 1;
                if (IsCIndex)
                {
                    for (int b = axis + 1; b < NDim; b++)
                        stride *= origShape[b];
                }
                else
                {
                    for (int b = 0; b < axis; b++)
                        stride *= origShape[b];
                }

                if (Shape[d] == 1)
                    stride = 0;

                if (flipped)
                {
                    IndexReset += (Shape[d] - 1) * stride;
                    stride = -stride;
                }

                IndexStrides[d] = stride;
            }
        }

        /// <summary>
        ///     The iteration index whose position holds flat index <paramref name="flatIndex"/>, without moving — NumPy's
        ///     <c>NpyIter_GotoIndex</c> loop over the index strides.
        /// </summary>
        /// <param name="flatIndex">A flat index in <c>[0, IterSize)</c>.</param>
        /// <returns>The iteration index (in <c>[0, IterSize)</c>).</returns>
        /// <remarks>
        ///     Each axis takes the coordinate its index stride names: <c>(index / stride) % shape</c>, counted from the end
        ///     for a negative (flipped) stride, and 0 for a stride of 0 (an extent-1 axis). The coordinates then combine as
        ///     the iteration index, innermost axis fastest. Correct on coalesced axes too, which is why it reads the strides
        ///     rather than recomputing C/F coordinates of the operands' shape.
        /// </remarks>
        internal long IterIndexOfIndex(long flatIndex)
        {
            long iterIndex = 0;
            long factor = 1;
            for (int d = NDim - 1; d >= 0; d--)
            {
                long stride = IndexStrides[d];
                long shape = Shape[d];
                long coord;
                if (stride == 0)
                    coord = 0;
                else if (stride < 0)
                    coord = shape - (flatIndex / -stride) % shape - 1;
                else
                    coord = (flatIndex / stride) % shape;

                iterIndex += factor * coord;
                factor *= shape;
            }

            return iterIndex;
        }
    }
}
