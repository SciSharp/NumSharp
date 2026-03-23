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
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>The truncated value of each element in x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.trunc.html</remarks>
        public static NDArray trunc(NDArray x, NPTypeCode? dtype = null)
            => x.TensorEngine.Truncate(x, dtype);

        /// <summary>
        /// Return the truncated value of the input, element-wise.
        /// The truncated value of the scalar x is the nearest integer i which is closer to zero than x is.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of.</param>
        /// <returns>The truncated value of each element in x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.trunc.html</remarks>
        public static NDArray trunc(NDArray x, Type dtype)
            => x.TensorEngine.Truncate(x, dtype);
    }
}
