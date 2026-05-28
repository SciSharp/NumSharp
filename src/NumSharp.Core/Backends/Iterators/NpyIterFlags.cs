using System;

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// Iterator-level flags. Conceptually matches NumPy's NPY_ITFLAG_* constants.
    ///
    /// NOTE: Bit positions differ from NumPy's implementation:
    /// - NumPy uses bits 0-7 for IDENTPERM, NEGPERM, HASINDEX, etc.
    /// - NumSharp reserves bits 0-7 for legacy compatibility flags (SourceBroadcast, SourceContiguous, DestinationContiguous)
    /// - NumPy-equivalent flags are shifted to bits 8-15
    ///
    /// This layout maintains backward compatibility with existing NumSharp code while
    /// adding NumPy parity flags. The semantic meaning of each flag matches NumPy,
    /// only the bit positions differ.
    /// </summary>
    [Flags]
    public enum NpyIterFlags : uint
    {
        None = 0,

        // =========================================================================
        // Legacy flags (bits 0-7, backward compatibility with existing NpyIter)
        // These do not have NumPy equivalents at these positions.
        // =========================================================================

        /// <summary>Source operand has broadcast dimensions (stride=0).</summary>
        SourceBroadcast = 1 << 0,

        /// <summary>Source operand is contiguous after coalescing.</summary>
        SourceContiguous = 1 << 1,

        /// <summary>Destination operand is contiguous after coalescing.</summary>
        DestinationContiguous = 1 << 2,

        // =========================================================================
        // Permutation Flags (bits 8-15, NumPy parity - shifted from NumPy's bits 0-7)
        // NumPy: NPY_ITFLAG_IDENTPERM = 1<<0, NPY_ITFLAG_NEGPERM = 1<<1, etc.
        // NumSharp: These are at 1<<8, 1<<9, etc. to avoid collision with legacy flags.
        // =========================================================================

        /// <summary>The axis permutation is identity.</summary>
        IDENTPERM = 0x0001 << 8,

        /// <summary>The permutation has negative entries (flipped axes).</summary>
        NEGPERM = 0x0002 << 8,

        // =========================================================================
        // Index Tracking Flags
        // =========================================================================

        /// <summary>Iterator is tracking a flat index.</summary>
        HASINDEX = 0x0004 << 8,

        /// <summary>Iterator is tracking a multi-index.</summary>
        HASMULTIINDEX = 0x0008 << 8,

        // =========================================================================
        // Order and Loop Flags
        // =========================================================================

        /// <summary>Iteration order was forced on construction.</summary>
        FORCEDORDER = 0x0010 << 8,

        /// <summary>Inner loop is handled outside the iterator.</summary>
        EXLOOP = 0x0020 << 8,

        /// <summary>Iterator is ranged (subset iteration).</summary>
        RANGE = 0x0040 << 8,

        // =========================================================================
        // Buffering Flags
        // =========================================================================

        /// <summary>Iterator uses buffering.</summary>
        BUFFER = 0x0080 << 8,

        /// <summary>Grow the buffered inner loop when possible.</summary>
        GROWINNER = 0x0100 << 8,

        /// <summary>Single iteration, can specialize iternext.</summary>
        ONEITERATION = 0x0200 << 8,

        /// <summary>Delay buffer allocation until first Reset.</summary>
        DELAYBUF = 0x0400 << 8,

        // =========================================================================
        // Reduction Flags
        // =========================================================================

        /// <summary>Iteration includes reduction operands.</summary>
        REDUCE = 0x0800 << 8,

        /// <summary>Reduce loops don't need recalculation.</summary>
        REUSE_REDUCE_LOOPS = 0x1000 << 8,

        // =========================================================================
        // NumSharp Extensions
        // =========================================================================

        /// <summary>All operands are contiguous (SIMD eligible).</summary>
        CONTIGUOUS = 0x00010000,

        /// <summary>Can use AVX2 gather for strided access.</summary>
        GATHER_ELIGIBLE = 0x00020000,

        /// <summary>Operation supports early exit (boolean ops).</summary>
        EARLY_EXIT = 0x00040000,

        /// <summary>Parallel outer loop is safe.</summary>
        PARALLEL_SAFE = 0x00080000,
    }

    /// <summary>
    /// Per-operand flags during iteration. Matches NumPy's NPY_OP_ITFLAG_* constants.
    /// </summary>
    [Flags]
    public enum NpyIterOpFlags : ushort
    {
        None = 0,

        // =========================================================================
        // Read/Write Flags
        // =========================================================================

        /// <summary>Operand will be written to.</summary>
        WRITE = 0x0001,

        /// <summary>Operand will be read from.</summary>
        READ = 0x0002,

        /// <summary>Operand is read-write.</summary>
        READWRITE = READ | WRITE,

        // =========================================================================
        // Buffering Flags
        // =========================================================================

        /// <summary>Operand needs type conversion/byte swapping/alignment.</summary>
        CAST = 0x0004,

        /// <summary>Operand never needs buffering.</summary>
        BUFNEVER = 0x0008,

        /// <summary>Buffer filling can use single stride.</summary>
        BUF_SINGLESTRIDE = 0x0010,

        // =========================================================================
        // Reduction Flags
        // =========================================================================

        /// <summary>Operand is being reduced.</summary>
        REDUCE = 0x0020,

        /// <summary>Operand is virtual (no backing array).</summary>
        VIRTUAL = 0x0040,

        /// <summary>Operand requires masking when copying buffer to array.</summary>
        WRITEMASKED = 0x0080,

        // =========================================================================
        // Buffer State Flags
        // =========================================================================

        /// <summary>Buffer is fully filled and ready for reuse.</summary>
        BUF_REUSABLE = 0x0100,

        /// <summary>Operand must be copied.</summary>
        FORCECOPY = 0x0200,

        /// <summary>Operand has temporary data, write back at dealloc.</summary>
        HAS_WRITEBACK = 0x0400,

        /// <summary>User requested contiguous operand.</summary>
        CONTIG = 0x0800,
    }

    /// <summary>
    /// Global flags passed to iterator construction.
    /// Bit values match NumPy's NPY_ITER_* constants exactly
    /// (see numpy/_core/include/numpy/ndarraytypes.h).
    /// </summary>
    [Flags]
    public enum NpyIterGlobalFlags : uint
    {
        None = 0,

        // =========================================================================
        // Index Tracking (NPY_ITER_C_INDEX .. NPY_ITER_MULTI_INDEX)
        // =========================================================================

        /// <summary>Track a C-order flat index. (NPY_ITER_C_INDEX)</summary>
        C_INDEX = 0x00000001,

        /// <summary>Track an F-order flat index. (NPY_ITER_F_INDEX)</summary>
        F_INDEX = 0x00000002,

        /// <summary>Track a multi-index. (NPY_ITER_MULTI_INDEX)</summary>
        MULTI_INDEX = 0x00000004,

        // =========================================================================
        // Loop Control
        // =========================================================================

        /// <summary>Expose inner loop to external code. (NPY_ITER_EXTERNAL_LOOP)</summary>
        EXTERNAL_LOOP = 0x00000008,

        // =========================================================================
        // Type Handling
        // =========================================================================

        /// <summary>Find common dtype for all operands. (NPY_ITER_COMMON_DTYPE)</summary>
        COMMON_DTYPE = 0x00000010,

        // =========================================================================
        // Safety and Compatibility
        // =========================================================================

        /// <summary>Allow object dtype arrays (not supported in NumSharp). (NPY_ITER_REFS_OK)</summary>
        REFS_OK = 0x00000020,

        /// <summary>Allow zero-size arrays. (NPY_ITER_ZEROSIZE_OK)</summary>
        ZEROSIZE_OK = 0x00000040,

        /// <summary>Allow reduction operands. (NPY_ITER_REDUCE_OK)</summary>
        REDUCE_OK = 0x00000080,

        /// <summary>Enable ranged iteration. (NPY_ITER_RANGED)</summary>
        RANGED = 0x00000100,

        // =========================================================================
        // Buffering
        // =========================================================================

        /// <summary>Enable buffering. (NPY_ITER_BUFFERED)</summary>
        BUFFERED = 0x00000200,

        /// <summary>Grow inner loop when possible. (NPY_ITER_GROWINNER)</summary>
        GROWINNER = 0x00000400,

        /// <summary>Delay buffer allocation until Reset. (NPY_ITER_DELAY_BUFALLOC)</summary>
        DELAY_BUFALLOC = 0x00000800,

        // =========================================================================
        // Stride & Overlap Control
        // =========================================================================

        /// <summary>Don't negate strides for axes iterated in reverse. (NPY_ITER_DONT_NEGATE_STRIDES)</summary>
        DONT_NEGATE_STRIDES = 0x00001000,

        /// <summary>Copy operands if they overlap in memory. (NPY_ITER_COPY_IF_OVERLAP)</summary>
        COPY_IF_OVERLAP = 0x00002000,

        /// <summary>
        /// Assume elementwise access for overlap detection. (NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE)
        /// Note: NumPy places this in the per-operand bit range (0x40000000), but it is passed
        /// alongside global flags. Kept here for API compatibility with earlier NumSharp releases.
        /// </summary>
        OVERLAP_ASSUME_ELEMENTWISE = 0x40000000,
    }

    /// <summary>
    /// Per-operand flags passed to iterator construction.
    /// Bit values match NumPy's NPY_ITER_* per-operand constants exactly
    /// (see numpy/_core/include/numpy/ndarraytypes.h). All values occupy the
    /// high 16 bits per NumPy's NPY_ITER_PER_OP_FLAGS mask (0xffff0000).
    /// </summary>
    [Flags]
    public enum NpyIterPerOpFlags : uint
    {
        None = 0,

        // =========================================================================
        // Read/Write Mode
        // =========================================================================

        /// <summary>Operand is read-write. (NPY_ITER_READWRITE)</summary>
        READWRITE = 0x00010000,

        /// <summary>Operand is read-only. (NPY_ITER_READONLY)</summary>
        READONLY = 0x00020000,

        /// <summary>Operand is write-only. (NPY_ITER_WRITEONLY)</summary>
        WRITEONLY = 0x00040000,

        // =========================================================================
        // Memory Layout
        // =========================================================================

        /// <summary>Require native byte order. (NPY_ITER_NBO)</summary>
        NBO = 0x00080000,

        /// <summary>Require aligned data. (NPY_ITER_ALIGNED)</summary>
        ALIGNED = 0x00100000,

        /// <summary>Require contiguous data. (NPY_ITER_CONTIG)</summary>
        CONTIG = 0x00200000,

        // =========================================================================
        // Allocation and Copying
        // =========================================================================

        /// <summary>Copy operand data. (NPY_ITER_COPY)</summary>
        COPY = 0x00400000,

        /// <summary>Update original if copy is made. (NPY_ITER_UPDATEIFCOPY)</summary>
        UPDATEIFCOPY = 0x00800000,

        /// <summary>Allocate output array if null. (NPY_ITER_ALLOCATE)</summary>
        ALLOCATE = 0x01000000,

        /// <summary>Don't allocate with subtype. (NPY_ITER_NO_SUBTYPE)</summary>
        NO_SUBTYPE = 0x02000000,

        /// <summary>Virtual operand slot (no backing array, temporary data only). (NPY_ITER_VIRTUAL)</summary>
        VIRTUAL = 0x04000000,

        // =========================================================================
        // Broadcasting Control
        // =========================================================================

        /// <summary>Don't broadcast this operand. (NPY_ITER_NO_BROADCAST)</summary>
        NO_BROADCAST = 0x08000000,

        // =========================================================================
        // Masking
        // =========================================================================

        /// <summary>Write only where mask is true. (NPY_ITER_WRITEMASKED)</summary>
        WRITEMASKED = 0x10000000,

        /// <summary>This operand is an array mask. (NPY_ITER_ARRAYMASK)</summary>
        ARRAYMASK = 0x20000000,

        // =========================================================================
        // Overlap Handling
        // =========================================================================

        /// <summary>
        /// Assume iterator-order access for COPY_IF_OVERLAP. (NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE)
        ///
        /// When COPY_IF_OVERLAP is set and this operand has this flag, the overlap check
        /// can short-circuit: if both operands point to the same buffer with identical
        /// memory layout and no internal overlap, no copy is needed (because the caller's
        /// inner loop accesses data strictly element-by-element in iterator order).
        /// NumPy nditer_constr.c:3130-3137 (same-data overlap short-circuit).
        /// </summary>
        OVERLAP_ASSUME_ELEMENTWISE_PER_OP = 0x40000000u,
    }

    /// <summary>
    /// Flags characterizing the transfer (cast/copy) functions set up by an iterator.
    /// Matches NumPy's NPY_ARRAYMETHOD_FLAGS (dtype_api.h:66).
    ///
    /// Packed into the top 8 bits of <see cref="NpyIterState.ItFlags"/> at offset
    /// <see cref="NpyIterConstants.TRANSFERFLAGS_SHIFT"/> (=24). Retrieved via
    /// <see cref="NpyIterRef.GetTransferFlags"/> — the preferred way to check whether
    /// the iteration can run without the GIL (in NumPy) or might set FP errors.
    /// </summary>
    [Flags]
    public enum NpyArrayMethodFlags : uint
    {
        /// <summary>No special transfer characteristics.</summary>
        None = 0,

        /// <summary>Flag for whether the GIL is required. Never set in NumSharp (no Python). (NPY_METH_REQUIRES_PYAPI)</summary>
        REQUIRES_PYAPI = 1 << 0,

        /// <summary>
        /// Function cannot set floating point error flags. Can skip FP error setup.
        /// Always set in NumSharp (.NET casts never raise FPE). (NPY_METH_NO_FLOATINGPOINT_ERRORS)
        /// </summary>
        NO_FLOATINGPOINT_ERRORS = 1 << 1,

        /// <summary>Method supports unaligned access. Always set in NumSharp (raw byte pointer loops). (NPY_METH_SUPPORTS_UNALIGNED)</summary>
        SUPPORTS_UNALIGNED = 1 << 2,

        /// <summary>Used for reductions to allow reordering. Applies to normal ops too. (NPY_METH_IS_REORDERABLE)</summary>
        IS_REORDERABLE = 1 << 3,

        /// <summary>Mask of flags that can change at runtime. (NPY_METH_RUNTIME_FLAGS)</summary>
        RUNTIME_FLAGS = REQUIRES_PYAPI | NO_FLOATINGPOINT_ERRORS,
    }

    /// <summary>
    /// NpyIter-related bit-packing constants that don't belong on the flag enums.
    /// </summary>
    public static class NpyIterConstants
    {
        /// <summary>
        /// Shift amount into <see cref="NpyIterState.ItFlags"/> where transfer flags are packed.
        /// Matches NumPy's NPY_ITFLAG_TRANSFERFLAGS_SHIFT (nditer_impl.h:111).
        /// </summary>
        public const int TRANSFERFLAGS_SHIFT = 24;

        /// <summary>Mask covering the packed transfer-flag bits (top 8 bits).</summary>
        public const uint TRANSFERFLAGS_MASK = 0xFFu << TRANSFERFLAGS_SHIFT;

        /// <summary>
        /// Additive offset for encoding reduction axes in op_axes entries.
        /// Matches NumPy's NPY_ITER_REDUCTION_AXIS (common.h:347):
        ///   <c>axis + (1 &lt;&lt; (NPY_BITSOF_INT - 2))</c> = <c>axis + 0x40000000</c>.
        ///
        /// To mark an op_axes entry as an explicit reduction axis, use
        /// <see cref="NpyIterUtils.ReductionAxis(int)"/>.
        /// </summary>
        public const int REDUCTION_AXIS_OFFSET = 1 << 30;
    }

    /// <summary>
    /// Helper utilities for NpyIter op_axes encoding/decoding.
    /// </summary>
    public static class NpyIterUtils
    {
        /// <summary>
        /// Encodes an op_axes entry as an explicit reduction axis.
        /// Matches NumPy's NPY_ITER_REDUCTION_AXIS macro (common.h:347).
        ///
        /// Use in the <c>opAxes</c> parameter of <see cref="NpyIterRef.AdvancedNew"/>
        /// to mark an axis as a reduction target (must have length 1 on the operand,
        /// and the operand must be READWRITE with REDUCE_OK set).
        /// </summary>
        /// <param name="axis">The axis index (may be -1 to mean broadcast+reduce).</param>
        /// <returns>The encoded value for op_axes[iop][idim].</returns>
        public static int ReductionAxis(int axis)
        {
            return axis + NpyIterConstants.REDUCTION_AXIS_OFFSET;
        }

        /// <summary>
        /// Decodes an op_axes entry. Matches NumPy's npyiter_get_op_axis
        /// (nditer_constr.c:1439).
        /// </summary>
        /// <param name="axis">The raw value from op_axes[iop][idim].</param>
        /// <param name="isReduction">True if the entry was flagged as a reduction axis.</param>
        /// <returns>The axis index (with reduction flag stripped if present).</returns>
        public static int GetOpAxis(int axis, out bool isReduction)
        {
            isReduction = axis >= NpyIterConstants.REDUCTION_AXIS_OFFSET - 1;
            if (isReduction)
                return axis - NpyIterConstants.REDUCTION_AXIS_OFFSET;
            return axis;
        }
    }

    /// <summary>
    /// Execution path for NpyIter operations.
    /// </summary>
    public enum NpyIterExecutionPath
    {
        /// <summary>All operands contiguous, use direct SIMD.</summary>
        Contiguous,

        /// <summary>Strided but gather-compatible, use AVX2 gather.</summary>
        Strided,

        /// <summary>Copy to contiguous buffers, SIMD on buffers.</summary>
        Buffered,

        /// <summary>Coordinate-based iteration, scalar operations.</summary>
        General,
    }

    /// <summary>
    /// Iteration order enumeration matching NumPy's NPY_ORDER.
    /// </summary>
    public enum NPY_ORDER
    {
        /// <summary>Keep existing order.</summary>
        NPY_KEEPORDER = 0,

        /// <summary>Force C (row-major) order.</summary>
        NPY_CORDER = 1,

        /// <summary>Force Fortran (column-major) order.</summary>
        NPY_FORTRANORDER = 2,

        /// <summary>Any order that allows contiguous access.</summary>
        NPY_ANYORDER = 3,
    }

    /// <summary>
    /// Casting rules enumeration matching NumPy's NPY_CASTING.
    /// </summary>
    public enum NPY_CASTING
    {
        /// <summary>No casting allowed.</summary>
        NPY_NO_CASTING = 0,

        /// <summary>Only casting that preserves values.</summary>
        NPY_EQUIV_CASTING = 1,

        /// <summary>Safe casting (no loss of precision).</summary>
        NPY_SAFE_CASTING = 2,

        /// <summary>Same-kind casting allowed.</summary>
        NPY_SAME_KIND_CASTING = 3,

        /// <summary>Any casting allowed.</summary>
        NPY_UNSAFE_CASTING = 4,
    }

    /// <summary>
    /// Bit masks that partition the NpyIter flag space into global (bits 0-15)
    /// and per-operand (bits 16-31) regions. Matches NumPy's NPY_ITER_GLOBAL_FLAGS
    /// and NPY_ITER_PER_OP_FLAGS macros.
    /// </summary>
    public static class NpyIterFlagMasks
    {
        /// <summary>Mask covering NpyIterGlobalFlags bits. (NPY_ITER_GLOBAL_FLAGS)</summary>
        public const uint NPY_ITER_GLOBAL_FLAGS = 0x0000ffff;

        /// <summary>Mask covering NpyIterPerOpFlags bits. (NPY_ITER_PER_OP_FLAGS)</summary>
        public const uint NPY_ITER_PER_OP_FLAGS = 0xffff0000;
    }
}
