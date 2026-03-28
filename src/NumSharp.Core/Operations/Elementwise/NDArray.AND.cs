namespace NumSharp
{
    /// <summary>
    ///     Bitwise AND operator for NDArray.
    ///     Uses the object pattern matching NumPy's PyArray_FromAny behavior.
    /// </summary>
    public partial class NDArray
    {
        /// <summary>
        /// Element-wise bitwise AND operation.
        /// For boolean arrays: logical AND.
        /// For integer arrays: bitwise AND.
        /// Supports broadcasting.
        /// </summary>
        public static NDArray operator &(NDArray lhs, NDArray rhs)
        {
            return lhs.TensorEngine.BitwiseAnd(lhs, rhs);
        }

        /// <summary>
        /// Element-wise bitwise AND with any scalar or array-like.
        /// </summary>
        public static NDArray operator &(NDArray lhs, object rhs)
        {
            return lhs & np.asanyarray(rhs);
        }

        /// <summary>
        /// Element-wise bitwise AND with any scalar or array-like on left.
        /// </summary>
        public static NDArray operator &(object lhs, NDArray rhs)
        {
            return np.asanyarray(lhs) & rhs;
        }
    }
}
