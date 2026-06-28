using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Element-wise maximum of array elements.
        ///     Compare two arrays and returns a new array containing the element-wise maxima. If one of the elements being compared is a NaN, then that element is returned. If both elements are NaNs then the first is returned. The latter distinction is important for complex NaNs, which are defined as at least one of the real or imaginary parts being a NaN. The net effect is that NaNs are propagated.
        /// </summary>
        /// <param name="x1">The arrays holding the elements to be compared. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).</param>
        /// <param name="x2">The arrays holding the elements to be compared. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).</param>
        /// <param name="dtype">Loop dtype (NumPy ufunc dtype=): the comparison runs at this precision.</param>
        /// <returns>The maximum of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars.</returns>
        public static NDArray maximum(NDArray x1, NDArray x2, NPTypeCode? dtype = null)
            => x1.TensorEngine.Maximum(x1, x2, dtype);

        /// <summary>
        ///     Element-wise maximum of array elements (NaN-propagating). See <see cref="maximum(NDArray,NDArray,NPTypeCode?)"/>.
        /// </summary>
        public static NDArray maximum(NDArray x1, NDArray x2, Type dtype)
            => x1.TensorEngine.Maximum(x1, x2, dtype?.GetTypeCode());

        /// <summary>
        ///     Element-wise maximum of array elements (NaN-propagating), writing into <paramref name="@out"/>.
        /// </summary>
        /// <param name="@out">A location into which the result is stored. Must broadcast-match the inputs.</param>
        public static NDArray maximum(NDArray x1, NDArray x2, NDArray @out)
            => x1.TensorEngine.Maximum(x1, x2, null, @out);

        /// <summary>
        ///     Element-wise maximum of array elements, ignoring NaNs.
        ///     Compare two arrays and returns a new array containing the element-wise maxima. If one of the elements being compared is a NaN, then the non-nan element is returned. If both elements are NaNs then the first is returned. The net effect is that NaNs are ignored when possible.
        /// </summary>
        /// <param name="x1">The arrays holding the elements to be compared. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).</param>
        /// <param name="x2">The arrays holding the elements to be compared. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).</param>
        /// <param name="dtype">Loop dtype (NumPy ufunc dtype=): the comparison runs at this precision.</param>
        /// <returns>The maximum of x1 and x2, element-wise, ignoring NaNs.</returns>
        public static NDArray fmax(NDArray x1, NDArray x2, NPTypeCode? dtype = null)
            => x1.TensorEngine.FMax(x1, x2, dtype);

        /// <summary>
        ///     Element-wise maximum of array elements, ignoring NaNs. See <see cref="fmax(NDArray,NDArray,NPTypeCode?)"/>.
        /// </summary>
        public static NDArray fmax(NDArray x1, NDArray x2, Type dtype)
            => x1.TensorEngine.FMax(x1, x2, dtype?.GetTypeCode());

        /// <summary>
        ///     Element-wise maximum of array elements, ignoring NaNs, writing into <paramref name="@out"/>.
        /// </summary>
        /// <param name="@out">A location into which the result is stored. Must broadcast-match the inputs.</param>
        public static NDArray fmax(NDArray x1, NDArray x2, NDArray @out)
            => x1.TensorEngine.FMax(x1, x2, null, @out);
    }
}
