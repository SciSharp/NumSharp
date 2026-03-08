using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the fractional and integral parts of an array, element-wise.
        ///     The fractional and integral parts are negative if the given number is negative.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>Fractional part of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.modf.html</remarks>
        public static (NDArray Fractional, NDArray Intergral) modf(in NDArray x, NPTypeCode? dtype = null)
            => x.TensorEngine.ModF(x, dtype);

        /// <summary>
        ///     Return the fractional and integral parts of an array, element-wise.
        ///     The fractional and integral parts are negative if the given number is negative.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>Fractional part of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.modf.html</remarks>
        public static (NDArray Fractional, NDArray Intergral) modf(in NDArray x, Type dtype) 
            => x.TensorEngine.ModF(x, dtype);
    }
}
