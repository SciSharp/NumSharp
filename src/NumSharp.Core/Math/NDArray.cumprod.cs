using System;
using System.Diagnostics.CodeAnalysis;
using NumSharp.Backends;

namespace NumSharp
{
    [SuppressMessage("ReSharper", "ParameterHidesMember")]
    public partial class NDArray
    {
        /// <summary>
        ///     Return the cumulative product of the elements along a given axis.
        /// </summary>
        /// <param name="axis">Axis along which the cumulative product is computed. The default (null) is to compute the cumprod over the flattened array.</param>
        /// <param name="dtype">Type of the returned array and of the accumulator in which the elements are multiplied. If dtype is not specified, it defaults to the dtype of a, unless a has an integer dtype with a precision less than that of the default platform integer. In that case, the default platform integer is used.</param>
        /// <param name="out">Alternate output array in which to place the result. It must have the same shape as the expected output. A reference to <paramref name="out"/> is returned.</param>
        /// <returns>A new array holding the result is returned unless out is specified, in which case a reference to out is returned. The result has the same size as a, and the same shape as a if axis is not None or a is a 1-d array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.cumprod.html</remarks>
        public NDArray cumprod(int? axis = null, Type dtype = null, NDArray @out = null)
        {
            return np.cumprod(this, axis, dtype?.GetTypeCode(), @out);
        }
    }
}
