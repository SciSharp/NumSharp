namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Matrix product of two arrays.
        /// </summary>
        /// <param name="x1">Lhs Input array, scalars not allowed.</param>
        /// <param name="x2">Rhs Input array, scalars not allowed.</param>
        /// <returns>The matrix product of the inputs. This is a scalar only when both x1, x2 are 1-d vectors.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html</remarks>
        public static NDArray matmul(in NDArray x1, in NDArray x2)
            => x1.TensorEngine.MatMul(x1, x2);
    }
}
