namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Binary operations supported by kernel providers.
    /// </summary>
    public enum BinaryOp
    {
        // Arithmetic
        Add,
        Subtract,
        Multiply,
        Divide,
        Mod,
        // Bitwise
        BitwiseAnd,
        BitwiseOr,
        BitwiseXor,
        // Future
        Power,
        FloorDivide,
        LeftShift,
        RightShift
    }

    /// <summary>
    /// Unary operations supported by kernel providers.
    /// </summary>
    public enum UnaryOp
    {
        // Core operations (Phase 1)
        Negate,
        Abs,
        Sqrt,
        Exp,
        Log,
        Sin,
        Cos,

        // Extended operations (Phase 2 - future)
        Tan,
        Exp2,
        Expm1,
        Log2,
        Log10,
        Log1p,
        Sinh,
        Cosh,
        Tanh,
        ASin,
        ACos,
        ATan,
        Sign,
        Ceil,
        Floor,
        Round,

        // Future
        Truncate,
        Reciprocal,
        Square,
        Cbrt,
        Deg2Rad,
        Rad2Deg,
        BitwiseNot
    }

    /// <summary>
    /// Reduction operations supported by kernel providers.
    /// </summary>
    public enum ReductionOp
    {
        /// <summary>Sum of elements (add reduction)</summary>
        Sum,
        /// <summary>Product of elements (multiply reduction)</summary>
        Prod,
        /// <summary>Maximum element</summary>
        Max,
        /// <summary>Minimum element</summary>
        Min,
        /// <summary>Index of maximum element (returns int)</summary>
        ArgMax,
        /// <summary>Index of minimum element (returns int)</summary>
        ArgMin,
        /// <summary>Mean = Sum / count</summary>
        Mean,
        /// <summary>Cumulative sum (running total)</summary>
        CumSum,
        /// <summary>All elements non-zero (logical AND reduction, returns bool)</summary>
        All,
        /// <summary>Any element non-zero (logical OR reduction, returns bool)</summary>
        Any,
        // Future
        Std,
        Var,
        NanSum,
        NanProd,
        NanMin,
        NanMax
    }

    /// <summary>
    /// Comparison operations supported by kernel providers.
    /// All comparison operations return bool (NPTypeCode.Boolean).
    /// </summary>
    public enum ComparisonOp
    {
        Equal,
        NotEqual,
        Less,
        LessEqual,
        Greater,
        GreaterEqual
    }

    /// <summary>
    /// Execution paths for binary operations, selected based on stride analysis.
    /// </summary>
    public enum ExecutionPath
    {
        /// <summary>Both operands are fully C-contiguous with identical shapes.</summary>
        SimdFull,
        /// <summary>Right operand is a scalar (all strides = 0).</summary>
        SimdScalarRight,
        /// <summary>Left operand is a scalar (all strides = 0).</summary>
        SimdScalarLeft,
        /// <summary>Inner dimension is contiguous/broadcast for both operands.</summary>
        SimdChunk,
        /// <summary>Arbitrary strides, requires coordinate-based iteration.</summary>
        General
    }
}
