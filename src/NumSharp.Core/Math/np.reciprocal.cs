using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return the reciprocal of the argument, element-wise.
        /// Calculates 1/x.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="outType">The dtype the returned ndarray should be of.</param>
        /// <returns>Return array containing 1/x for each element in x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reciprocal.html</remarks>
        public static NDArray reciprocal(in NDArray x, NPTypeCode? outType = null)
            => x.TensorEngine.Reciprocal(x, outType);

        /// <summary>
        /// Return the reciprocal of the argument, element-wise.
        /// Calculates 1/x.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="outType">The dtype the returned ndarray should be of.</param>
        /// <returns>Return array containing 1/x for each element in x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reciprocal.html</remarks>
        public static NDArray reciprocal(in NDArray x, Type outType)
            => x.TensorEngine.Reciprocal(x, outType);
    }
}
