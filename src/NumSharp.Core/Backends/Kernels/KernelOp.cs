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
        /// <summary>C-style floating remainder (np.fmod): result takes the sign of the DIVIDEND,
        /// unlike <see cref="Mod"/> (floored, sign of divisor). Scalar-only, all int+float dtypes.</summary>
        Fmod,
        LeftShift,
        RightShift,
        // Transcendental binary
        /// <summary>Element-wise arc tangent of y/x choosing the quadrant correctly (np.arctan2)</summary>
        ATan2,
        // Element-wise min/max (np.maximum/minimum/fmax/fmin)
        /// <summary>Element-wise maximum, NaN-propagating (np.maximum): a NaN operand wins.</summary>
        Maximum,
        /// <summary>Element-wise minimum, NaN-propagating (np.minimum): a NaN operand wins.</summary>
        Minimum,
        /// <summary>Element-wise maximum, NaN-ignoring (np.fmax): returns the non-NaN operand.</summary>
        FMax,
        /// <summary>Element-wise minimum, NaN-ignoring (np.fmin): returns the non-NaN operand.</summary>
        FMin,
        // Log-sum-exp + IEEE step (np.logaddexp/logaddexp2/nextafter) — float-tier binary ufuncs,
        // same promotion as ATan2. Scalar-only (no Vector op); routed to NDLogAddExpMath helpers.
        /// <summary>log(exp(x1)+exp(x2)) computed stably (np.logaddexp).</summary>
        LogAddExp,
        /// <summary>log2(2**x1 + 2**x2) computed stably (np.logaddexp2).</summary>
        LogAddExp2,
        /// <summary>Next representable value after x1 toward x2 (np.nextafter).</summary>
        NextAfter,
        /// <summary>Magnitude of x1 with the sign of x2 (np.copysign).</summary>
        CopySign,
        /// <summary>sqrt(x1**2 + x2**2) without spurious overflow/underflow (np.hypot). Correctly-rounded
        /// (Borges FMA); routed to NDHypotMath, scalar-only like the rest of this family.</summary>
        Hypot,
        /// <summary>The Heaviside step function (np.heaviside): 0 if x1&lt;0, x2 if x1==0, 1 if x1&gt;0, and
        /// the positive canonical NaN if x1 is NaN. NOT commutative — x1 selects the branch, x2 is only the
        /// x1==0 fill (its exact bits, NaN sign included, pass through). Float-tier promotion like the rest
        /// of this family (ATan2), but unlike them it has a branchless SIMD fast path (compare + select,
        /// no libm) because every output is an exact value; routed to NDHeavisideMath.</summary>
        Heaviside,
        // Number-theoretic binary — INTEGER-ONLY (no bool/float/complex/decimal loop; those raise the
        // no-loop error at the np.* boundary). Data-dependent scalar Euclidean loop per element, so — like
        // NumPy itself ("It may be nice to vectorize these, OTOH…") — there is NO SIMD path: absent from
        // CanUseSimdForOp, they route through the scalar per-element kernel (NDGcdLcm helpers). Uniform
        // NEP50 promotion (both operands + output share one integer dtype); uint64+signed → float64 → no loop.
        /// <summary>Greatest common divisor of |x1| and |x2| (np.gcd). Result is non-negative except where
        /// the magnitude wraps the signed range (gcd(int8 -128,-128) == -128). gcd(0,0) == 0.</summary>
        Gcd,
        /// <summary>Lowest common multiple of |x1| and |x2| (np.lcm): 0 if either is 0, else |x1|/gcd*|x2|
        /// (divide-before-multiply; the product WRAPS the dtype on overflow, matching NumPy).</summary>
        Lcm
    }

    /// <summary>
    /// Unary operations supported by kernel providers.
    /// </summary>
    public enum UnaryOp
    {
        // Core operations (Phase 1)
        Negate,
        Abs,
        /// <summary>
        /// np.fabs — the float-only absolute value. The KERNEL is identical to <see cref="Abs"/> on
        /// the float loops (clear the IEEE sign bit), so every emit site aliases Fabs to Abs; it is a
        /// DISTINCT op only so it reports the ufunc name "fabs" (not "absolute") in errors and carries
        /// its own float-tier dtype resolution (see <c>DefaultEngine.Fabs</c>). Never has complex/int
        /// input at the kernel — <c>Fabs</c> rejects complex and promotes int→float before dispatch.
        /// </summary>
        Fabs,
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
        // Inverse hyperbolic (np.arcsinh/arccosh/arctanh; Array-API aliases asinh/acosh/atanh)
        Asinh,
        Acosh,
        Atanh,
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
        /// <summary>Identity at every dtype (NumPy 'positive'; also the masked-copy vehicle for out=/where= compositions)</summary>
        Positive,

        // Floating-point classification (returns bool)
        /// <summary>Test element-wise for finiteness (not infinity and not NaN)</summary>
        IsFinite,
        /// <summary>Test element-wise for NaN</summary>
        IsNan,
        /// <summary>Test element-wise for positive or negative infinity</summary>
        IsInf,
        /// <summary>Test element-wise for positive infinity (np.isposinf). x == +inf.</summary>
        IsPosInf,
        /// <summary>Test element-wise for negative infinity (np.isneginf). x == -inf.</summary>
        IsNegInf,
        /// <summary>
        /// Test element-wise whether the IEEE sign bit is set (np.signbit). Returns bool.
        /// This is NOT <c>x &lt; 0</c>: it is defined on the raw bit pattern, so <c>-0.0</c> and a
        /// negative NaN are True while <c>+0.0</c>, <c>+inf</c> and a positive NaN are False. For
        /// signed integers it coincides with <c>x &lt; 0</c> (the two's-complement MSB); for unsigned
        /// integers / bool it is always False; complex has no loop (rejected at the np.* layer).
        /// </summary>
        SignBit,

        /// <summary>
        /// Complex conjugate (np.conjugate / np.conj). Identity at every real dtype
        /// (bool/int/char/half/single/double/decimal) — the loaded value IS the result — and
        /// flips the sign of the imaginary part for Complex (via <c>System.Numerics.Complex.Conjugate</c>).
        /// Dtype is preserved (never promoted), matching NumPy's per-dtype conjugate loops.
        /// </summary>
        Conjugate,

        /// <summary>
        /// Distance to the adjacent representable value away from zero — one ULP (np.spacing).
        /// Float-only ufunc (ee/ff/dd loops + NumSharp's decimal extension; complex has NO loop).
        /// float32/float64 are SIGNED (carry the sign of x, +minsubnormal at ±0) via the raw
        /// bit-increment <c>reinterpret(bits(x)+1) - x</c>; float16 is NumPy's separate always-positive
        /// <c>npy_half_spacing</c>. See <see cref="NumSharp.Utilities.NDSpacingMath"/>.
        /// </summary>
        Spacing,

        /// <summary>
        /// Population count of the absolute value — the number of set bits in <c>|x|</c> (np.bitwise_count),
        /// NumPy's <c>npy_popcount(a &lt; 0 ? -a : a)</c>. INTEGER/BOOL ONLY: every supported dtype
        /// (bool/byte/sbyte/int16/uint16/int32/uint32/int64/uint64/char) maps to a <b>uint8</b> result,
        /// float/complex/decimal/half raise the no-loop TypeError at the np.* boundary. Two consequences
        /// that make it behave like the float classification predicates (<see cref="IsNan"/> etc.) rather
        /// than an ordinary math op: it CONSUMES the input dtype and the emitter itself yields the (byte)
        /// result — the scalar/strided loops must NOT convert input→output first (which would truncate a
        /// wide value to 8 bits before counting) — and the output dtype is fixed regardless of input width.
        /// Signed negatives count the magnitude (<c>bitwise_count(-1)==1</c>, not 8; the min value's
        /// two's-complement negation wraps to itself, so <c>bitwise_count(int8 -128)==1</c>), computed via
        /// a branchless abs (<c>(x^(x&gt;&gt;w-1))-(x&gt;&gt;w-1)</c>) before <c>BitOperations.PopCount</c>.
        /// </summary>
        BitwiseCount
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
