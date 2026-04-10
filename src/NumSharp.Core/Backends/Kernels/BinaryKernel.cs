using System;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Cache key for mixed-type binary operation kernels.
    /// Identifies a unique kernel by LHS type, RHS type, result type, operation, and execution path.
    /// </summary>
    /// <remarks>
    /// Supports up to 3,600 unique kernels: 12 × 12 × 5 × 5 = 3,600
    /// (12 LHS types × 12 RHS types × 5 operations × 5 paths, result type determined by promotion rules)
    /// </remarks>
    public readonly record struct MixedTypeKernelKey(
        NPTypeCode LhsType,
        NPTypeCode RhsType,
        NPTypeCode ResultType,
        BinaryOp Op,
        ExecutionPath Path
    )
    {
        /// <summary>
        /// Returns true if all three types are the same (no conversion needed).
        /// </summary>
        public bool IsSameType => LhsType == RhsType && RhsType == ResultType;

        /// <summary>
        /// Returns true if the LHS needs conversion to result type.
        /// </summary>
        public bool NeedsLhsConversion => LhsType != ResultType;

        /// <summary>
        /// Returns true if the RHS needs conversion to result type.
        /// </summary>
        public bool NeedsRhsConversion => RhsType != ResultType;

        public override string ToString() => $"{Op}_{LhsType}_{RhsType}_{ResultType}_{Path}";
    }

    /// <summary>
    /// Mixed-type binary operation kernel signature using void pointers.
    /// Handles operations where LHS, RHS, and result may have different types.
    /// Type conversion is handled internally by the generated IL.
    /// </summary>
    /// <param name="lhs">Pointer to left operand data (typed as LhsType)</param>
    /// <param name="rhs">Pointer to right operand data (typed as RhsType)</param>
    /// <param name="result">Pointer to output data (typed as ResultType)</param>
    /// <param name="lhsStrides">Left operand strides (element units)</param>
    /// <param name="rhsStrides">Right operand strides (element units)</param>
    /// <param name="shape">Output shape dimensions</param>
    /// <param name="ndim">Number of dimensions</param>
    /// <param name="totalSize">Total number of output elements</param>
    public unsafe delegate void MixedTypeKernel(
        void* lhs,
        void* rhs,
        void* result,
        long* lhsStrides,
        long* rhsStrides,
        long* shape,
        int ndim,
        long totalSize
    );

    #region Unary Operations

    /// <summary>
    /// Cache key for unary operation kernels.
    /// Identifies a unique kernel by input type, output type, operation, and whether contiguous.
    /// </summary>
    public readonly record struct UnaryKernelKey(
        NPTypeCode InputType,
        NPTypeCode OutputType,
        UnaryOp Op,
        bool IsContiguous
    )
    {
        /// <summary>
        /// Returns true if input and output types are the same (no conversion needed).
        /// </summary>
        public bool IsSameType => InputType == OutputType;

        public override string ToString() => $"{Op}_{InputType}_{OutputType}_{(IsContiguous ? "Contig" : "Strided")}";
    }

    /// <summary>
    /// Unary operation kernel signature using void pointers.
    /// Handles operations where input and output may have different types.
    /// Type conversion is handled internally by the generated IL.
    /// </summary>
    /// <param name="input">Pointer to input data</param>
    /// <param name="output">Pointer to output data</param>
    /// <param name="strides">Input strides (element units, not bytes)</param>
    /// <param name="shape">Shape dimensions</param>
    /// <param name="ndim">Number of dimensions</param>
    /// <param name="totalSize">Total number of elements</param>
    public unsafe delegate void UnaryKernel(
        void* input,
        void* output,
        long* strides,
        long* shape,
        int ndim,
        long totalSize
    );

    #endregion

    #region Comparison Operations

    /// <summary>
    /// Cache key for comparison operation kernels.
    /// Identifies a unique kernel by LHS type, RHS type, operation, and execution path.
    /// Result type is always bool (NPTypeCode.Boolean).
    /// </summary>
    /// <remarks>
    /// Supports up to 4,320 unique kernels: 12 × 12 × 6 × 5 = 4,320
    /// (12 LHS types × 12 RHS types × 6 comparison ops × 5 paths)
    /// </remarks>
    public readonly record struct ComparisonKernelKey(
        NPTypeCode LhsType,
        NPTypeCode RhsType,
        ComparisonOp Op,
        ExecutionPath Path
    )
    {
        /// <summary>
        /// Returns true if both input types are the same (no conversion needed for comparison).
        /// </summary>
        public bool IsSameType => LhsType == RhsType;

        /// <summary>
        /// Get the common type for comparison (promote both operands to this type).
        /// </summary>
        public NPTypeCode ComparisonType => GetComparisonType(LhsType, RhsType);

        /// <summary>
        /// Result type is always bool for comparisons.
        /// </summary>
        public NPTypeCode ResultType => NPTypeCode.Boolean;

        public override string ToString() => $"{Op}_{LhsType}_{RhsType}_{Path}";

        /// <summary>
        /// Determine the common type to use for comparison between two types.
        /// Both operands should be promoted to this type before comparison.
        /// </summary>
        private static NPTypeCode GetComparisonType(NPTypeCode lhs, NPTypeCode rhs)
        {
            if (lhs == rhs) return lhs;

            // Use the same type promotion rules as binary operations
            // Prefer wider types and floating point over integer
            return np._FindCommonScalarType(lhs, rhs);
        }
    }

    /// <summary>
    /// Comparison operation kernel signature using void pointers.
    /// LHS and RHS may have different types, but result is always bool.
    /// Type conversion is handled internally by the generated IL.
    /// </summary>
    /// <param name="lhs">Pointer to left operand data</param>
    /// <param name="rhs">Pointer to right operand data</param>
    /// <param name="result">Pointer to output data (always bool*)</param>
    /// <param name="lhsStrides">Left operand strides (element units)</param>
    /// <param name="rhsStrides">Right operand strides (element units)</param>
    /// <param name="shape">Output shape dimensions</param>
    /// <param name="ndim">Number of dimensions</param>
    /// <param name="totalSize">Total number of output elements</param>
    public unsafe delegate void ComparisonKernel(
        void* lhs,
        void* rhs,
        bool* result,
        long* lhsStrides,
        long* rhsStrides,
        long* shape,
        int ndim,
        long totalSize
    );

    #endregion
}
