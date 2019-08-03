using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the non-negative square-root of an array, element-wise.
        /// </summary>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <param name="x">The values whose square-roots are required.</param>
        /// <returns>An array of the same shape as x, containing the positive square-root of each element in x. If any element in x is complex, a complex array is returned (and the square-roots of negative reals are calculated). If all of the elements in x are real, so is y, with negative elements returning nan. If out was provided, y is a reference to it. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.sqrt.html</remarks>
        public static NDArray sqrt(in NDArray x, NPTypeCode? outType = null)
            => x.TensorEngine.Sqrt(x, outType);

        /// <summary>
        ///     Return the non-negative square-root of an array, element-wise.
        /// </summary>
        /// <param name="x">The values whose square-roots are required.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array of the same shape as x, containing the positive square-root of each element in x. If any element in x is complex, a complex array is returned (and the square-roots of negative reals are calculated). If all of the elements in x are real, so is y, with negative elements returning nan. If out was provided, y is a reference to it. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.sqrt.html</remarks>
        public static NDArray sqrt(in NDArray x, Type outType)
            => x.TensorEngine.Sqrt(x);
    }
}
