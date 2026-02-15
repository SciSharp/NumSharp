namespace NumSharp
{
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
        /// Element-wise bitwise AND with scalar.
        /// </summary>
        public static NDArray operator &(NDArray lhs, byte rhs)
        {
            return lhs.TensorEngine.BitwiseAnd(lhs, Scalar(rhs));
        }

        /// <summary>
        /// Element-wise bitwise AND with scalar.
        /// </summary>
        public static NDArray operator &(byte lhs, NDArray rhs)
        {
            return rhs.TensorEngine.BitwiseAnd(Scalar(lhs), rhs);
        }

        /// <summary>
        /// Element-wise bitwise AND with scalar.
        /// </summary>
        public static NDArray operator &(NDArray lhs, int rhs)
        {
            return lhs.TensorEngine.BitwiseAnd(lhs, Scalar(rhs));
        }

        /// <summary>
        /// Element-wise bitwise AND with scalar.
        /// </summary>
        public static NDArray operator &(int lhs, NDArray rhs)
        {
            return rhs.TensorEngine.BitwiseAnd(Scalar(lhs), rhs);
        }

        /// <summary>
        /// Element-wise bitwise AND with scalar.
        /// </summary>
        public static NDArray operator &(NDArray lhs, long rhs)
        {
            return lhs.TensorEngine.BitwiseAnd(lhs, Scalar(rhs));
        }

        /// <summary>
        /// Element-wise bitwise AND with scalar.
        /// </summary>
        public static NDArray operator &(long lhs, NDArray rhs)
        {
            return rhs.TensorEngine.BitwiseAnd(Scalar(lhs), rhs);
        }

        /// <summary>
        /// Element-wise bitwise AND with boolean scalar.
        /// </summary>
        public static NDArray operator &(NDArray lhs, bool rhs)
        {
            return lhs.TensorEngine.BitwiseAnd(lhs, Scalar(rhs));
        }

        /// <summary>
        /// Element-wise bitwise AND with boolean scalar.
        /// </summary>
        public static NDArray operator &(bool lhs, NDArray rhs)
        {
            return rhs.TensorEngine.BitwiseAnd(Scalar(lhs), rhs);
        }
    }
}
