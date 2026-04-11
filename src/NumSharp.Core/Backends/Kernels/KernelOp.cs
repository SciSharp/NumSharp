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
        RightShift,
        // Transcendental binary
        /// <summary>Element-wise arc tangent of y/x choosing the quadrant correctly (np.arctan2)</summary>
        ATan2
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
        BitwiseNot,
        /// <summary>Logical NOT for boolean arrays (! operator, not ~ bitwise)</summary>
        LogicalNot,

        // Floating-point classification (returns bool)
        /// <summary>Test element-wise for finiteness (not infinity and not NaN)</summary>
        IsFinite,
        /// <summary>Test element-wise for NaN</summary>
        IsNan,
        /// <summary>Test element-wise for positive or negative infinity</summary>
        IsInf
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
        /// <summary>Cumulative product (running product)</summary>
        CumProd,
        /// <summary>All elements non-zero (logical AND reduction, returns bool)</summary>
        All,
        /// <summary>Any element non-zero (logical OR reduction, returns bool)</summary>
        Any,
        /// <summary>Standard deviation</summary>
        Std,
        /// <summary>Variance</summary>
        Var,
        /// <summary>Sum ignoring NaN values (treats NaN as 0)</summary>
        NanSum,
        /// <summary>Product ignoring NaN values (treats NaN as 1)</summary>
        NanProd,
        /// <summary>Minimum ignoring NaN values (all-NaN returns NaN)</summary>
        NanMin,
        /// <summary>Maximum ignoring NaN values (all-NaN returns NaN)</summary>
        NanMax,
        /// <summary>Mean ignoring NaN values (all-NaN returns NaN)</summary>
        NanMean,
        /// <summary>Variance ignoring NaN values (all-NaN returns NaN)</summary>
        NanVar,
        /// <summary>Standard deviation ignoring NaN values (all-NaN returns NaN)</summary>
        NanStd
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
