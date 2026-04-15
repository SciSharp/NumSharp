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
    /// Matches NumPy's NPY_ITER_* constants.
    /// </summary>
    [Flags]
    public enum NpyIterGlobalFlags : uint
    {
        None = 0,

        // =========================================================================
        // Index Tracking
        // =========================================================================

        /// <summary>Track a C-order flat index.</summary>
        C_INDEX = 0x0001,

        /// <summary>Track an F-order flat index.</summary>
        F_INDEX = 0x0002,

        /// <summary>Track a multi-index.</summary>
        MULTI_INDEX = 0x0004,

        // =========================================================================
        // Loop Control
        // =========================================================================

        /// <summary>Expose inner loop to external code.</summary>
        EXTERNAL_LOOP = 0x0008,

        /// <summary>Don't negate strides for axes iterated in reverse.</summary>
        DONT_NEGATE_STRIDES = 0x0010,

        // =========================================================================
        // Buffering
        // =========================================================================

        /// <summary>Enable buffering.</summary>
        BUFFERED = 0x0020,

        /// <summary>Grow inner loop when possible.</summary>
        GROWINNER = 0x0040,

        /// <summary>Delay buffer allocation until Reset.</summary>
        DELAY_BUFALLOC = 0x0080,

        // =========================================================================
        // Safety and Compatibility
        // =========================================================================

        /// <summary>Allow zero-size arrays.</summary>
        ZEROSIZE_OK = 0x0100,

        /// <summary>Allow object dtype arrays (not supported in NumSharp).</summary>
        REFS_OK = 0x0200,

        /// <summary>Allow reduction operands.</summary>
        REDUCE_OK = 0x0400,

        /// <summary>Enable ranged iteration.</summary>
        RANGED = 0x0800,

        // =========================================================================
        // Type Handling
        // =========================================================================

        /// <summary>Find common dtype for all operands.</summary>
        COMMON_DTYPE = 0x1000,

        /// <summary>Copy operands if they overlap in memory.</summary>
        COPY_IF_OVERLAP = 0x2000,

        /// <summary>Assume elementwise access for overlap detection.</summary>
        OVERLAP_ASSUME_ELEMENTWISE = 0x4000,
    }

    /// <summary>
    /// Per-operand flags passed to iterator construction.
    /// Matches NumPy's NPY_ITER_* per-operand constants.
    /// </summary>
    [Flags]
    public enum NpyIterPerOpFlags : uint
    {
        None = 0,

        // =========================================================================
        // Read/Write Mode
        // =========================================================================

        /// <summary>Operand is read-only.</summary>
        READONLY = 0x0001,

        /// <summary>Operand is write-only.</summary>
        WRITEONLY = 0x0002,

        /// <summary>Operand is read-write.</summary>
        READWRITE = 0x0004,

        // =========================================================================
        // Allocation and Copying
        // =========================================================================

        /// <summary>Copy operand data.</summary>
        COPY = 0x0008,

        /// <summary>Update original if copy is made.</summary>
        UPDATEIFCOPY = 0x0010,

        /// <summary>Allocate output array if null.</summary>
        ALLOCATE = 0x0020,

        /// <summary>Don't allocate with subtype.</summary>
        NO_SUBTYPE = 0x0040,

        // =========================================================================
        // Broadcasting Control
        // =========================================================================

        /// <summary>Don't broadcast this operand.</summary>
        NO_BROADCAST = 0x0080,

        // =========================================================================
        // Memory Layout
        // =========================================================================

        /// <summary>Require contiguous data.</summary>
        CONTIG = 0x0100,

        /// <summary>Require aligned data.</summary>
        ALIGNED = 0x0200,

        /// <summary>Require native byte order.</summary>
        NBO = 0x0400,

        // =========================================================================
        // Masking
        // =========================================================================

        /// <summary>This operand is an array mask.</summary>
        ARRAYMASK = 0x0800,

        /// <summary>Write only where mask is true.</summary>
        WRITEMASKED = 0x1000,
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
}
