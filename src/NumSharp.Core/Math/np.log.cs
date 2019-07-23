using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        ///     Natural logarithm, element-wise.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <param name="dtype">The dtype of the returned NDArray</param>
        /// <returns>The natural logarithm of x, element-wise. This is a scalar NDArray.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html</remarks>
        public static NDArray log(in NDArray a, Type dtype) => a.log(dtype);

        /// <summary>
        ///     Natural logarithm, element-wise.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <param name="typeCode">The dtype of the returned NDArray</param>
        /// <returns>The natural logarithm of x, element-wise. This is a scalar NDArray.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html</remarks>
        public static NDArray log(in NDArray a, NPTypeCode typeCode) => a.log(typeCode);

        /// <summary>
        ///     Natural logarithm, element-wise.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <param name="dtype">The dtype of the returned NDArray</param>
        /// <returns>The natural logarithm of x, element-wise. This is a scalar NDArray.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html</remarks>
        public static NDArray log(in NDArray a) => a.log();
    }
}
