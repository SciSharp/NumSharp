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
        // Scope: OperatorOperand(rhs) MINTS a fresh 0-d/array temp when rhs is a scalar or
        // array-like — a leftover reclaimable only by the finalizer. A plain
        // `using var t = OperatorOperand(rhs)` would be a BUG: when rhs is already an NDArray,
        // OperatorOperand returns THAT input, so `using` would dispose the caller's array (rule R2).
        // [NDScoped] tracks only arrays CONSTRUCTED inside the scope, so the minted temp is
        // reclaimed while an input passthrough is left untouched; the result is yielded.
        [NDScoped]
        public static NDArray operator &(NDArray lhs, object rhs)
        {
            return lhs & OperatorOperand(rhs);
        }

        /// <summary>
        /// Element-wise bitwise AND with any scalar or array-like on left.
        /// </summary>
        [NDScoped]
        public static NDArray operator &(object lhs, NDArray rhs)
        {
            return OperatorOperand(lhs) & rhs;
        }
    }
}
