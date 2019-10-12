using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Element-wise minimum of array elements.
        ///     Compare two arrays and returns a new array containing the element-wise minima.If one of the elements being compared is a NaN, then that element is returned.If both elements are NaNs then the first is returned.The latter distinction is important for complex NaNs, which are defined as at least one of the real or imaginary parts being a NaN. The net effect is that NaNs are propagated.
        /// </summary>
        /// <param name="x1">The arrays holding the elements to be compared. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).</param>
        /// <param name="x2">The arrays holding the elements to be compared. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).</param>
        /// <param name="outType"></param>
        /// <returns>The maximum of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars.</returns>
        public static NDArray minimum(in NDArray x1, in NDArray x2, NPTypeCode? outType = null)
        {
            var (_x1, _x2) = np.broadcast_arrays(x1, x2);
            return np.clip(_x1, a_min: null, a_max: _x2, outType: outType);
        }

        /// <summary>
        ///     Element-wise minimum of array elements.
        ///     Compare two arrays and returns a new array containing the element-wise minima.If one of the elements being compared is a NaN, then that element is returned.If both elements are NaNs then the first is returned.The latter distinction is important for complex NaNs, which are defined as at least one of the real or imaginary parts being a NaN. The net effect is that NaNs are propagated.
        /// </summary>
        /// <param name="x1">The arrays holding the elements to be compared. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).</param>
        /// <param name="x2">The arrays holding the elements to be compared. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).</param>
        /// <param name="outType"></param>
        /// <returns>The maximum of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars.</returns>
        public static NDArray minimum(in NDArray x1, in NDArray x2, Type outType)
        {
            var (_x1, _x2) = np.broadcast_arrays(x1, x2);
            return np.clip(_x1, a_min: null, a_max: _x2, outType: outType);
        }

        /// <summary>
        ///     Element-wise minimum of array elements.
        ///     Compare two arrays and returns a new array containing the element-wise minima.If one of the elements being compared is a NaN, then that element is returned.If both elements are NaNs then the first is returned.The latter distinction is important for complex NaNs, which are defined as at least one of the real or imaginary parts being a NaN. The net effect is that NaNs are propagated.
        /// </summary>
        /// <param name="x1">The arrays holding the elements to be compared. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).</param>
        /// <param name="x2">The arrays holding the elements to be compared. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).</param>
        /// <param name="@out">A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to. If not provided or None, a freshly-allocated array is returned. A tuple (possible only as a keyword argument) must have length equal to the number of outputs.</param>
        /// <returns>The maximum of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars.</returns>
        public static NDArray minimum(in NDArray x1, in NDArray x2, NDArray @out)
        {
            var (_x1, _x2) = np.broadcast_arrays(x1, x2);
            return np.clip(_x1, a_min: null, a_max: _x2, @out: @out);
        }

        /// <summary>
        ///     Element-wise maximum of array elements.
        ///     Compare two arrays and returns a new array containing the element-wise maxima.If one of the elements being compared is a NaN, then that element is returned.If both elements are NaNs then the first is returned.The latter distinction is important for complex NaNs, which are defined as at least one of the real or imaginary parts being a NaN. The net effect is that NaNs are propagated.
        /// </summary>
        /// <param name="x1">The arrays holding the elements to be compared. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).</param>
        /// <param name="x2">The arrays holding the elements to be compared. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).</param>
        /// <param name="outType"></param>
        /// <returns>The maximum of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars.</returns>
        public static NDArray fmin(in NDArray x1, in NDArray x2, NPTypeCode? outType = null)
        {
            var (_x1, _x2) = np.broadcast_arrays(x1, x2);
            return np.clip(_x1, a_min: null, a_max: _x2, outType: outType);
        }

        /// <summary>
        ///     Element-wise maximum of array elements.
        ///     Compare two arrays and returns a new array containing the element-wise maxima.If one of the elements being compared is a NaN, then that element is returned.If both elements are NaNs then the first is returned.The latter distinction is important for complex NaNs, which are defined as at least one of the real or imaginary parts being a NaN. The net effect is that NaNs are propagated.
        /// </summary>
        /// <param name="x1">The arrays holding the elements to be compared. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).</param>
        /// <param name="x2">The arrays holding the elements to be compared. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).</param>
        /// <param name="outType"></param>
        /// <returns>The maximum of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars.</returns>
        public static NDArray fmin(in NDArray x1, in NDArray x2, Type outType)
        {
            var (_x1, _x2) = np.broadcast_arrays(x1, x2);
            return np.clip(_x1, a_min: null, a_max: _x2, outType: outType);
        }

        /// <summary>
        ///     Element-wise maximum of array elements.
        ///     Compare two arrays and returns a new array containing the element-wise maxima.If one of the elements being compared is a NaN, then that element is returned.If both elements are NaNs then the first is returned.The latter distinction is important for complex NaNs, which are defined as at least one of the real or imaginary parts being a NaN. The net effect is that NaNs are propagated.
        /// </summary>
        /// <param name="x1">The arrays holding the elements to be compared. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).</param>
        /// <param name="x2">The arrays holding the elements to be compared. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).</param>
        /// <param name="@out">A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to. If not provided or None, a freshly-allocated array is returned. A tuple (possible only as a keyword argument) must have length equal to the number of outputs.</param>
        /// <returns>The maximum of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars.</returns>
        public static NDArray fmin(in NDArray x1, in NDArray x2, NDArray @out)
        {
            var (_x1, _x2) = np.broadcast_arrays(x1, x2);
            return np.clip(_x1, a_min: null, a_max: _x2, @out: @out);
        }
    }
}
