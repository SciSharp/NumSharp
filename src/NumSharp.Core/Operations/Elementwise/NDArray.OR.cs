namespace NumSharp
{
    /// <summary>
    ///     Bitwise OR operator for NDArray.
    ///     Uses the object pattern matching NumPy's PyArray_FromAny behavior.
    /// </summary>
    public partial class NDArray
    {
        /// <summary>
        /// Element-wise bitwise OR operation.
        /// For boolean arrays: logical OR.
        /// For integer arrays: bitwise OR.
        /// Supports broadcasting.
        /// </summary>
        public static NDArray operator |(NDArray lhs, NDArray rhs)
        {
            return lhs.TensorEngine.BitwiseOr(lhs, rhs);
        }

        /// <summary>
        /// Element-wise bitwise OR with any scalar or array-like.
        /// </summary>
        // Scope: np.asanyarray(rhs) mints a temp for a scalar/array-like operand; [NDScoped]
        // reclaims it while leaving an NDArray-input passthrough untouched (see NDArray.AND.cs).
        [NDScoped]
        public static NDArray operator |(NDArray lhs, object rhs)
        {
            return lhs | np.asanyarray(rhs);
        }

        /// <summary>
        /// Element-wise bitwise OR with any scalar or array-like on left.
        /// </summary>
        [NDScoped]
        public static NDArray operator |(object lhs, NDArray rhs)
        {
            return np.asanyarray(lhs) | rhs;
        }
    }
}
