namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Cache key for unary scalar operation kernels.
    /// Identifies a unique kernel by input type, output type, and operation.
    /// </summary>
    /// <remarks>
    /// Used to cache IL-generated Func delegates that eliminate dynamic dispatch overhead.
    /// The delegate type varies based on input/output types, so we store as System.Delegate.
    /// </remarks>
    public readonly record struct UnaryScalarKernelKey(
        NPTypeCode InputType,
        NPTypeCode OutputType,
        UnaryOp Op
    )
    {
        /// <summary>
        /// Returns true if input and output types are the same.
        /// </summary>
        public bool IsSameType => InputType == OutputType;

        public override string ToString() => $"Scalar_{Op}_{InputType}_{OutputType}";
    }

    /// <summary>
    /// Cache key for binary scalar operation kernels.
    /// Identifies a unique kernel by LHS type, RHS type, result type, and operation.
    /// </summary>
    /// <remarks>
    /// Used to cache IL-generated Func delegates that eliminate dynamic dispatch overhead.
    /// The delegate type varies based on operand/result types, so we store as System.Delegate.
    /// </remarks>
    public readonly record struct BinaryScalarKernelKey(
        NPTypeCode LhsType,
        NPTypeCode RhsType,
        NPTypeCode ResultType,
        BinaryOp Op
    )
    {
        /// <summary>
        /// Returns true if all three types are the same.
        /// </summary>
        public bool IsSameType => LhsType == RhsType && RhsType == ResultType;

        public override string ToString() => $"Scalar_{Op}_{LhsType}_{RhsType}_{ResultType}";
    }
}
