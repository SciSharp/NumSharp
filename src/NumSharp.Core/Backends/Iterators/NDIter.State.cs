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

        /// <summary>
        ///     <see cref="OpStringKinds"/> bit: the operand's iteration (buffer) descriptor is a NEP 55 <c>StringDType</c>. Its
        ///     16-byte cells look like an <c>|S16</c> element to the type code and the element size, so this bit is what keeps a
        ///     buffer transfer from moving them as bytes (a cell means nothing without its buffer's allocator).
        /// </summary>
        public const byte StringBufferVString = 4;

        /// <summary><see cref="OpStringKinds"/> bit: the operand array's own descriptor is a NEP 55 <c>StringDType</c>.</summary>
        public const byte StringSourceVString = 8;

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
        // Buffer Data (when BUFFERED flag is set) — NumPy's NpyIter_BufferData
        // =========================================================================
        // The buffered iteration is a port of NumPy 2.4.2's (nditer_constr.c npyiter_find_buffering_setup,
        // nditer_api.c npyiter_copy_to_buffers / npyiter_copy_from_buffers / npyiter_goto_iterindex, nditer_templ.c.src
        // npyiter_buffered_iternext / npyiter_buffered_reduce_iternext), carried out by NDIterBufferManager. The roles of
        // NumPy's pointer arrays during a buffer:
        //   * DataPtrs           = NIT_USERPTRS: what the user and the kernels read — a buffer for a buffered operand,
        //                          the array for a BUFNEVER one — advanced by BufStrides inside an inner loop;
        //   * ArrayWritebackPtrs = NIT_DATAPTRS: each operand's ARRAY position at the start of the current buffer, where
        //                          the write-back lands (Coords stay at that start for the whole buffer, as NAD_INDEX does);
        //   * ResetDataPtrs      = NIT_RESETDATAPTR: the base every reposition (GotoIterIndex) starts from.
        // =========================================================================

        /// <summary>
        ///     Elements one buffer fill may cover — NumPy's <c>NBF_BUFFERSIZE</c>. Holds the requested maximum until
        ///     <c>NDIterBufferManager.FindBufferingSetup</c> runs, then the setup's <c>best_size</c>: a whole number of
        ///     <see cref="CoreSize"/> cores at most the requested maximum, except that GROWINNER over operands that need no
        ///     buffer keeps the whole outer block (the inner loop may then exceed the requested size). Every buffer holds
        ///     this many elements.
        /// </summary>
        public long BufferSize;

        /// <summary>
        ///     The buffer size the iterator was built with (its <c>buffersize</c> argument, 0 resolved to NumPy's 8192), as
        ///     <c>NDIterBufferManager.FindBufferingSetup</c> received it. NumPy keeps no such field (it runs the setup once);
        ///     NumSharp needs it for the one place it runs the setup again — <c>RemoveMultiIndex</c> over a partly merged
        ///     buffered iteration, where NumPy's stale setup would walk wrong elements and can write outside an operand.
        /// </summary>
        public long MaxBufferSize;

        /// <summary>
        ///     The iteration index one past the current inner loop — NumPy's <c>NBF_BUFITEREND</c>. In the reduce double loop
        ///     it marks the end of the current CORE (not of the whole fill), and moves by <see cref="BufTransferSize"/> per
        ///     outer step.
        /// </summary>
        public long BufIterEnd;

        /// <summary>
        ///     The current inner loop's element count — NumPy's <c>NBF_SIZE</c>, what <c>GetInnerLoopSizePtr</c> exposes.
        ///     The whole fill outside the reduce double loop; one core inside it when the fill holds more than one core (the
        ///     fill is then <c>BufTransferSize * ReduceOuterSize</c> elements). Zero before the first fill and after the
        ///     iteration ended, which is what makes a second write-back of the same buffer a no-op.
        /// </summary>
        public long BufTransferSize;

        /// <summary>Buffer pointers for each operand (null for a BUFNEVER operand, and while allocation is delayed). Size = NOp.</summary>
        public long* Buffers;

        /// <summary>
        ///     Each operand's inner-loop stride in BYTES — NumPy's <c>NBF_STRIDES</c>: the buffer element size for a buffered
        ///     operand (0 for a buffer holding a single repeated element, and for a reduce operand whose element repeats along
        ///     the core), the operand's own innermost array stride for a BUFNEVER one. Size = NOp.
        /// </summary>
        public long* BufStrides;

        /// <summary>
        ///     A <see cref="System.Runtime.InteropServices.GCHandle"/> to the iterator's <see cref="NDIterVStringContext"/> (the
        ///     descriptors and allocators of its NEP 55 <c>StringDType</c> operands), or zero when no operand has a
        ///     <c>StringDType</c> side. Released by <see cref="FreeDimArrays"/>, the step every teardown of a state runs.
        /// </summary>
        public IntPtr VStringContext;

        // =========================================================================
        // The core / outer split and the reduce double loop (when BUFFERED is set)
        // =========================================================================
        // npyiter_find_buffering_setup picks an OUTER dimension: the axes inside it form the CORE (CoreSize elements), and
        // every buffer fill covers whole cores of one outer block (never crossing into the next block of the axes beyond
        // it). When that outer dimension is where a reduction operand's stride flips between zero and non-zero, the fill
        // is walked as a double loop (NDIterFlags.BUFREDUCE): ReduceOuterSize inner loops of one core each, every operand
        // stepping by its ReduceOuterStrides between them — a reduce operand's element (or its core) repeating while the
        // inputs move on.
        //
        // Reference: numpy/_core/src/multiarray/nditer_constr.c (npyiter_find_buffering_setup) and
        // nditer_templ.c.src (npyiter_buffered_reduce_iternext).
        // =========================================================================

        /// <summary>
        ///     The current outer step of the reduce double loop, in <c>[0, ReduceOuterSize)</c> — NumPy's
        ///     <c>NBF_REDUCE_POS</c>. Zero outside the reduce loop. <c>IsFirstVisit</c> reads it: a reduce operand whose
        ///     outer stride is 0 repeats its core at every step after the first.
        /// </summary>
        public long ReducePos;

        /// <summary>
        ///     The number of outer steps of the current fill — NumPy's <c>NBF_REDUCE_OUTERSIZE</c>: the fill's size divided by
        ///     <see cref="CoreSize"/>, rounded down. Zero for a fill shorter than one core (a fill that starts inside a core),
        ///     which NumPy's write-back then multiplies into a zero-element transfer.
        /// </summary>
        public long ReduceOuterSize;

        /// <summary>
        ///     The elements of one core — NumPy's <c>NBF_CORESIZE</c>: the product of the extents of the axes inside the outer
        ///     dimension. At least 1.
        /// </summary>
        public long CoreSize;

        /// <summary>
        ///     The buffering setup's outer dimension in NumPy's numbering — the axis index counted from the INNERMOST axis
        ///     (NumPy's <c>NBF_OUTERDIM</c>), not a NumSharp axis index: NumSharp axis <c>NDim - 1 - OuterDim</c>. Kept in NumPy's
        ///     numbering because <c>remove_multi_index</c> coalesces the axes after the setup and NumPy keeps using that index
        ///     on the merged axes (see <see cref="StaleOuterRemaining"/>).
        /// </summary>
        public int OuterDim;

        /// <summary>
        ///     The position inside its core of the iteration index the iterator was last moved to — NumPy's
        ///     <c>coreoffset</c>, set by every reposition (<c>iterindex % CoreSize</c>). A fill starting inside a core covers
        ///     only the rest of that core and never reuses a buffer.
        /// </summary>
        public long CoreOffset;

        /// <summary>
        ///     Each operand's pointer step between the outer steps of the reduce double loop, in BYTES — NumPy's
        ///     <c>NBF_REDUCE_OUTERSTRIDES</c>: 0 for a reduce operand whose core repeats, the element size for a buffered reduce
        ///     operand whose element repeats along the core, a whole core for any other buffered operand, the array stride along
        ///     the outer axis for a BUFNEVER one. <see cref="long.MinValue"/> (NumPy's <c>NPY_MIN_INTP</c>, "invalid") when the
        ///     iteration does not use the reduce loop. Size = NOp.
        /// </summary>
        public long* ReduceOuterStrides;

        /// <summary>
        ///     Each operand's pointer at the start of the current outer step of the reduce double loop — NumPy's
        ///     <c>NBF_REDUCE_OUTERPTRS</c>: the buffer start (or the array position for a BUFNEVER operand) after a fill, then
        ///     advanced by <see cref="ReduceOuterStrides"/> at every outer step. Size = NOp.
        /// </summary>
        public long* ReduceOuterPtrs;

        /// <summary>
        ///     Each operand's ARRAY position at the start of the current buffer — NumPy's <c>NIT_DATAPTRS</c> while buffering:
        ///     the source of the fill and the destination of the write-back, and what the next fill of the reduce double loop
        ///     compares with to decide whether a buffer still holds its operand's data (NumPy's reuse rule). Size = NOp.
        /// </summary>
        public long* ArrayWritebackPtrs;

        /// <summary>
        ///     Each operand's array stride along the buffering setup's outer axis, in BYTES — NumPy's
        ///     <c>NAD_STRIDES(outer_axisdata)</c>, which the fills, the write-backs and the reuse rule of the reduce double loop
        ///     read. Filled by the setup and kept across a <c>remove_multi_index</c> merge that leaves the setup in place (a
        ///     merge into one axis, or above the outer position, where the outer axis keeps these strides); a partial merge
        ///     below the outer position runs the setup again (<c>NDIterBufferManager.RebuildBufferingSetup</c>). Size = NOp.
        /// </summary>
        public long* OuterAxisStrides;

        /// <summary>
        ///     <c>-1</c>, or — after <c>remove_multi_index</c> merged every axis into one, the setup's outer axis with them, so
        ///     that <see cref="OuterDim"/> no longer names an axis — the remaining extent of the vanished outer axis at the merge
        ///     (its extent minus its coordinate), which NumPy then reads from that axis's stale storage at every fill: no
        ///     reposition updates it.
        /// </summary>
        public long StaleOuterRemaining;

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
            //   ReduceOuterStrides, ReduceOuterPtrs, ArrayWritebackPtrs, OuterAxisStrides = 10 arrays
            // int arrays (4 bytes each element):
            //   ElementSizes, SrcElementSizes = 2 arrays
            // ushort arrays (2 bytes each element):
            //   OpItFlags = 1 array
            // byte arrays (1 byte each element):
            //   OpDTypes, OpSrcDTypes, OpStringKinds = 3 arrays

            long longArraysBytes = 10L * nop * sizeof(long);
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

            // Assign long* arrays (10 arrays, each nop elements)
            long* longPtr = (long*)opBlock;
            DataPtrs = longPtr; longPtr += nop;
            ResetDataPtrs = longPtr; longPtr += nop;
            BaseOffsets = longPtr; longPtr += nop;
            InnerStrides = longPtr; longPtr += nop;
            Buffers = longPtr; longPtr += nop;
            BufStrides = longPtr; longPtr += nop;
            ReduceOuterStrides = longPtr; longPtr += nop;
            ReduceOuterPtrs = longPtr; longPtr += nop;
            ArrayWritebackPtrs = longPtr; longPtr += nop;
            OuterAxisStrides = longPtr;

            // No vanished outer axis until remove_multi_index merges one away (see StaleOuterRemaining).
            StaleOuterRemaining = -1;

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
            // Every teardown of a state runs this (Dispose, FreeState, a failed construction or copy), after the buffers were
            // freed: the StringDType context's handle goes with it, before ResetForRecycle zeroes the field.
            NDIterVStringContext.Detach(ref this);

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
                OuterAxisStrides = null;
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
                // A StringDType iteration side is a 16-byte cell (its itemsize) and carries its own bit: the class and the
                // element size alone would read it as |S16.
                ElementSizes[op] = descr.itemsize;
                OpStringKinds[op] = (byte)((OpStringKinds[op] & ~(StringBufferUnicode | StringBufferVString)) |
                                           (descr.kind == 'U' ? StringBufferUnicode : 0) |
                                           (descr.Meta is VStringDTypeMeta ? StringBufferVString : 0));
            }
            else
            {
                ElementSizes[op] = InfoOf.GetSize(code);
                OpStringKinds[op] = (byte)(OpStringKinds[op] & ~(StringBufferUnicode | StringBufferVString));
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
                OpStringKinds[op] = (byte)((OpStringKinds[op] & ~(StringSourceUnicode | StringSourceVString)) |
                                           (descr.kind == 'U' ? StringSourceUnicode : 0) |
                                           (descr.Meta is VStringDTypeMeta ? StringSourceVString : 0));
            }
            else
            {
                SrcElementSizes[op] = InfoOf.GetSize(code);
                OpStringKinds[op] = (byte)(OpStringKinds[op] & ~(StringSourceUnicode | StringSourceVString));
            }
        }

        /// <summary>Whether either side of operand <paramref name="op"/> (its array or its iteration dtype) is a fixed-width string.</summary>
        /// <param name="op">The operand index.</param>
        /// <returns>True when a buffer transfer of this operand must take the string route.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public bool IsStringOp(int op)
            => (NPTypeCode)OpDTypes[op] == NPTypeCode.String || (NPTypeCode)OpSrcDTypes[op] == NPTypeCode.String;

        /// <summary>
        ///     Whether either side of operand <paramref name="op"/> is a NEP 55 <c>StringDType</c> — its transfers then go through
        ///     the iterator's <see cref="NDIterVStringContext"/> (each string duplicated into the destination's allocator), never
        ///     as bytes.
        /// </summary>
        /// <param name="op">The operand index.</param>
        /// <returns>True for a <c>StringDType</c> array or iteration side.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public bool IsVStringOp(int op)
            => (OpStringKinds[op] & (StringBufferVString | StringSourceVString)) != 0;

        /// <summary>
        ///     The descriptor of operand <paramref name="op"/>'s ITERATION (buffer) side, rebuilt from its type code, element
        ///     size and string class — the state stores no descriptor objects, and a string's width and class are exactly the
        ///     two facts a type code drops. Byte order is not stored (buffers are native), so a non-native request reads back
        ///     native. A <c>StringDType</c> side is the descriptor instance the iterator's <see cref="NDIterVStringContext"/>
        ///     keeps (its <c>na_object</c> and <c>coerce</c> are not facts of the state).
        /// </summary>
        /// <param name="op">The operand index.</param>
        /// <returns>A sized descriptor (for messages and debug output; not cached).</returns>
        public DType GetOpDescr(int op)
        {
            if ((OpStringKinds[op] & StringBufferVString) != 0 && NDIterVStringContext.Of(ref this) is { } ctx && ctx.BufferDescrs[op] != null)
                return ctx.BufferDescrs[op];
            return DescrOf((NPTypeCode)OpDTypes[op], ElementSizes[op], (OpStringKinds[op] & StringBufferUnicode) != 0);
        }

        /// <summary>The descriptor of operand <paramref name="op"/>'s ARRAY side — the counterpart of <see cref="GetOpDescr"/>.</summary>
        /// <param name="op">The operand index.</param>
        /// <returns>A sized descriptor (for messages and debug output; not cached).</returns>
        public DType GetOpSrcDescr(int op)
        {
            if ((OpStringKinds[op] & StringSourceVString) != 0 && NDIterVStringContext.Of(ref this) is { } ctx && ctx.ArrayDescrs[op] != null)
                return ctx.ArrayDescrs[op];
            return DescrOf((NPTypeCode)OpSrcDTypes[op], SrcElementSizes[op], (OpStringKinds[op] & StringSourceUnicode) != 0);
        }

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
            // A StringDType side against a fixed-width one is a cast whatever the sizes (an |S16 element is 16 bytes too).
            // Two StringDType sides are not one here: the transfer duplicates each string either way, and the NA rule between
            // two descriptors is the transfer's own (NDVStringCasting.TransferRun).
            bool srcVString = (kinds & StringSourceVString) != 0;
            bool bufVString = (kinds & StringBufferVString) != 0;
            return SrcElementSizes[op] != ElementSizes[op] || srcUnicode != bufUnicode || srcVString != bufVString;
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
        ///     Moves the position to <see cref="IterStart"/> (which may be above 0 for a ranged iterator): the coordinates,
        ///     the tracked flat index and the ARRAY pointers — the axis half of NumPy's <c>NpyIter_Reset</c>
        ///     (<c>npyiter_goto_iterindex(ITERSTART)</c>).
        /// </summary>
        /// <remarks>
        ///     A buffered iterator's buffers are not touched: <c>NDIterRef.Reset</c> writes the current buffer back first and
        ///     fills the next one after, as NumPy's reset does. The buffer-reuse marks (<see cref="NDIterOpFlags.BUF_REUSABLE"/>)
        ///     are left alone too: the next fill decides reuse itself, from the previous buffer's start positions.
        /// </remarks>
        public void Reset()
        {
            // Delegate to GotoIterIndex so Coords, FlatIndex, and DataPtrs all
            // agree with IterStart (critical for ranged iterators where IterStart > 0).
            FlatIndex = 0;
            GotoIterIndex(IterStart);
        }

        /// <summary>
        ///     Moves the position to <paramref name="iterindex"/>: the coordinates (decomposed innermost axis first), the
        ///     tracked flat index, and every operand's ARRAY pointer from its reset pointer — NumPy's axis half of
        ///     <c>npyiter_goto_iterindex</c>.
        /// </summary>
        /// <param name="iterindex">The target iteration index (not range-checked: the callers check).</param>
        /// <remarks>
        ///     The data pointers then point into the operands' arrays, never into buffers: a buffered iterator follows this with
        ///     <c>NDIterBufferManager.CopyToBuffers</c> (which records these array positions as the buffer's write-back targets
        ///     and switches the buffered operands' pointers to their buffers), and sets <see cref="CoreOffset"/> through
        ///     <c>NDIterBufferManager.GotoIterIndex</c>. The buffer-reuse marks are NOT cleared here: NumPy's reduce loop moves
        ///     the position between a write-back and the next fill, and the fill's reuse rule needs the marks of the previous one.
        /// </remarks>
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
        }

        /// <summary>
        ///     Recomputes the coordinates of the current <see cref="IterIndex"/> over the current axes WITHOUT moving any pointer —
        ///     for <c>remove_multi_index</c> on a buffered iterator, which merges the axes while a buffer is held: the buffer and
        ///     the pointers into it stay (NumPy keeps them), but the coordinates the next write-back and fill walk from must
        ///     describe the merged axes.
        /// </summary>
        /// <remarks>
        ///     An extent-0 axis keeps coordinate 0 (an empty iteration has no position to decompose), as
        ///     <see cref="GotoIterIndex"/> does.
        /// </remarks>
        internal void RecomputeCoords()
        {
            long remaining = IterIndex;
            for (int d = NDim - 1; d >= 0; d--)
            {
                long dimSize = Shape[d];
                if (dimSize == 0)
                {
                    Coords[d] = 0;
                    continue;
                }

                Coords[d] = remaining % dimSize;
                remaining /= dimSize;
            }
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
