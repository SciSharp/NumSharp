namespace NumSharp.Generic
{
    public partial class NDArray<TDType>
    {
        /// <summary>
        /// Element-wise bitwise AND for typed arrays.
        /// Resolves ambiguity when using NDArray&lt;bool&gt; operands.
        /// </summary>
        public static NDArray<TDType> operator &(NDArray<TDType> lhs, NDArray<TDType> rhs)
        {
            // Scope: the engine result's untyped wrapper is reclaimed once the typed alias
            // (which holds its own ARC ref) is yielded — otherwise every typed & strands it.
            using var scope = NDScope.Open();
            return scope.Returns(((NDArray)lhs).TensorEngine.BitwiseAnd(lhs, rhs).MakeGeneric<TDType>());
        }

        /// <summary>
        /// Element-wise bitwise OR for typed arrays.
        /// Resolves ambiguity when using NDArray&lt;bool&gt; operands.
        /// </summary>
        public static NDArray<TDType> operator |(NDArray<TDType> lhs, NDArray<TDType> rhs)
        {
            using var scope = NDScope.Open();
            return scope.Returns(((NDArray)lhs).TensorEngine.BitwiseOr(lhs, rhs).MakeGeneric<TDType>());
        }

        /// <summary>
        /// Element-wise bitwise XOR for typed arrays.
        /// </summary>
        public static NDArray<TDType> operator ^(NDArray<TDType> lhs, NDArray<TDType> rhs)
        {
            using var scope = NDScope.Open();
            return scope.Returns(((NDArray)lhs).TensorEngine.BitwiseXor(lhs, rhs).MakeGeneric<TDType>());
        }
    }
}
