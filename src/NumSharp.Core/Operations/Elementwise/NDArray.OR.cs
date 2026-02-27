namespace NumSharp
{
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
        /// Element-wise bitwise OR with scalar.
        /// </summary>
        public static NDArray operator |(NDArray lhs, byte rhs)
        {
            return lhs.TensorEngine.BitwiseOr(lhs, Scalar(rhs));
        }

        /// <summary>
        /// Element-wise bitwise OR with scalar.
        /// </summary>
        public static NDArray operator |(byte lhs, NDArray rhs)
        {
            return rhs.TensorEngine.BitwiseOr(Scalar(lhs), rhs);
        }

        /// <summary>
        /// Element-wise bitwise OR with scalar.
        /// </summary>
        public static NDArray operator |(NDArray lhs, int rhs)
        {
            return lhs.TensorEngine.BitwiseOr(lhs, Scalar(rhs));
        }

        /// <summary>
        /// Element-wise bitwise OR with scalar.
        /// </summary>
        public static NDArray operator |(int lhs, NDArray rhs)
        {
            return rhs.TensorEngine.BitwiseOr(Scalar(lhs), rhs);
        }

        /// <summary>
        /// Element-wise bitwise OR with scalar.
        /// </summary>
        public static NDArray operator |(NDArray lhs, long rhs)
        {
            return lhs.TensorEngine.BitwiseOr(lhs, Scalar(rhs));
        }

        /// <summary>
        /// Element-wise bitwise OR with scalar.
        /// </summary>
        public static NDArray operator |(long lhs, NDArray rhs)
        {
            return rhs.TensorEngine.BitwiseOr(Scalar(lhs), rhs);
        }

        /// <summary>
        /// Element-wise bitwise OR with boolean scalar.
        /// </summary>
        public static NDArray operator |(NDArray lhs, bool rhs)
        {
            return lhs.TensorEngine.BitwiseOr(lhs, Scalar(rhs));
        }

        /// <summary>
        /// Element-wise bitwise OR with boolean scalar.
        /// </summary>
        public static NDArray operator |(bool lhs, NDArray rhs)
        {
            return rhs.TensorEngine.BitwiseOr(Scalar(lhs), rhs);
        }
    }
}
