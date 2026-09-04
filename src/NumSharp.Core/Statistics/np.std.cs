using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the standard deviation of the flattened array.
        ///     Returns the standard deviation, a measure of the spread of a distribution, of the array elements.
        /// </summary>
        /// <param name="a">Calculate the standard deviation of these values.</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as size-one dimensions.</param>
        /// <param name="ddof">Delta Degrees of Freedom. The divisor used is N - ddof (default 0).</param>
        /// <param name="dtype">The <see cref="DType"/> the computation/result should use (a C# <see cref="System.Type"/>, an <see cref="NPTypeCode"/> or a NumPy dtype string all convert implicitly).</param>
        /// <returns>A new array containing the std values, or a reference to the output array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.std.html</remarks>
        public static NDArray std(NDArray a, bool keepdims = false, int? ddof = null, DType dtype = null)
            => a.TensorEngine.ReduceStd(a, null, keepdims, ddof, dtype);

        /// <summary>
        ///     Compute the standard deviation along the specified axis.
        /// </summary>
        /// <param name="a">Calculate the standard deviation of these values.</param>
        /// <param name="axis">Axis along which the standard deviation is computed.</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as size-one dimensions.</param>
        /// <param name="ddof">Delta Degrees of Freedom. The divisor used is N - ddof (default 0).</param>
        /// <param name="dtype">The <see cref="DType"/> the computation/result should use (implicit from <see cref="System.Type"/> / <see cref="NPTypeCode"/> / NumPy dtype string).</param>
        /// <returns>A new array containing the std values, or a reference to the output array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.std.html</remarks>
        public static NDArray std(NDArray a, int axis, bool keepdims = false, int? ddof = null, DType dtype = null)
            => a.TensorEngine.ReduceStd(a, axis, keepdims, ddof, dtype);
    }
}
