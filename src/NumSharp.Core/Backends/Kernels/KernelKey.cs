namespace NumSharp.Backends.Kernels
{
    // =============================================================================
    // KernelKey.cs - Centralized cache key structs for kernel system
    // =============================================================================
    //
    // This file consolidates cache key definitions used across kernel providers.
    // Keys are used for O(1) kernel lookup in ConcurrentDictionary caches.
    //
    // EXISTING KEYS (defined in BinaryKernel.cs, ReductionKernel.cs):
    //   - MixedTypeKernelKey - mixed-type binary operations
    //   - UnaryKernelKey - unary operations
    //   - ComparisonKernelKey - comparison operations
    //   - ElementReductionKernelKey - element-wise reductions
    //   - AxisReductionKernelKey - axis reductions
    //   - CumulativeKernelKey - cumulative reductions
    //
    // NEW KEYS (defined here):
    //   - ContiguousKernelKey - same-type contiguous binary operations
    //   - UnaryScalarKey - scalar unary operations
    //   - BinaryScalarKey - scalar binary operations
    //   - ComparisonScalarKey - scalar comparison operations
    // =============================================================================

    /// <summary>
    /// Cache key for same-type contiguous binary operations.
    /// Used for fast-path SIMD kernels when both operands are contiguous with identical type.
    /// </summary>
    public readonly record struct ContiguousKernelKey(
        NPTypeCode Type,
        BinaryOp Op
    )
    {
        public override string ToString() => $"Contig_{Op}_{Type}";
    }

    /// <summary>
    /// Cache key for scalar unary operations.
    /// Used for element-by-element operations in general/broadcast paths.
    /// </summary>
    public readonly record struct UnaryScalarKey(
        NPTypeCode InputType,
        NPTypeCode OutputType,
        UnaryOp Op
    )
    {
        public bool IsSameType => InputType == OutputType;
        public override string ToString() => $"UnaryScalar_{Op}_{InputType}_{OutputType}";
    }

    /// <summary>
    /// Cache key for scalar binary operations.
    /// Used for element-by-element operations in general/broadcast paths.
    /// </summary>
    public readonly record struct BinaryScalarKey(
        NPTypeCode LhsType,
        NPTypeCode RhsType,
        NPTypeCode ResultType,
        BinaryOp Op
    )
    {
        public bool IsSameType => LhsType == RhsType && RhsType == ResultType;
        public override string ToString() => $"BinaryScalar_{Op}_{LhsType}_{RhsType}_{ResultType}";
    }

    /// <summary>
    /// Cache key for scalar comparison operations.
    /// Used for element-by-element comparisons in general/broadcast paths.
    /// </summary>
    public readonly record struct ComparisonScalarKey(
        NPTypeCode LhsType,
        NPTypeCode RhsType,
        ComparisonOp Op
    )
    {
        /// <summary>
        /// Get the common type for comparison (promote both operands to this type).
        /// </summary>
        public NPTypeCode ComparisonType => np._FindCommonScalarType(LhsType, RhsType);

        public override string ToString() => $"CmpScalar_{Op}_{LhsType}_{RhsType}";
    }
}
