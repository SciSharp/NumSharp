using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return the truncated value of the input, element-wise.
        /// The truncated value of the scalar x is the nearest integer i which is closer to zero than x is.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="outType">The dtype the returned ndarray should be of.</param>
        /// <returns>The truncated value of each element in x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.trunc.html</remarks>
        public static NDArray trunc(in NDArray x, NPTypeCode? outType = null)
            => x.TensorEngine.Truncate(x, outType);

        /// <summary>
        /// Return the truncated value of the input, element-wise.
        /// The truncated value of the scalar x is the nearest integer i which is closer to zero than x is.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="outType">The dtype the returned ndarray should be of.</param>
        /// <returns>The truncated value of each element in x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.trunc.html</remarks>
        public static NDArray trunc(in NDArray x, Type outType)
            => x.TensorEngine.Truncate(x, outType);
    }
}
