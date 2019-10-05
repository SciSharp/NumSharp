using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Evenly round to the given number of decimals.
        /// </summary>
        /// <param name="x">Input array</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array of the same type as a, containing the rounded values. Unless out was specified, a new array is created. A reference to the result is returned.
        ///  The real and imaginary parts of complex numbers are rounded separately.The result of rounding a float is a float.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.around.html</remarks>
        public static NDArray round_(in NDArray x, NPTypeCode? outType = null)
            => x.TensorEngine.Round(x, outType);

        /// <summary>
        ///     Evenly round to the given number of decimals.
        /// </summary>
        /// <param name="x">Input array</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array of the same type as a, containing the rounded values. Unless out was specified, a new array is created. A reference to the result is returned.
        ///  The real and imaginary parts of complex numbers are rounded separately.The result of rounding a float is a float.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.around.html</remarks>
        public static NDArray round_(in NDArray x, Type outType) 
            => x.TensorEngine.Round(x, outType);

        /// <summary>
        ///     Evenly round to the given number of decimals.
        /// </summary>
        /// <param name="x">Input array</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array of the same type as a, containing the rounded values. Unless out was specified, a new array is created. A reference to the result is returned.
        ///  The real and imaginary parts of complex numbers are rounded separately.The result of rounding a float is a float.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.around.html</remarks>
        public static NDArray around(in NDArray x, NPTypeCode? outType = null)
            => x.TensorEngine.Round(x, outType);

        /// <summary>
        ///     Evenly round to the given number of decimals.
        /// </summary>
        /// <param name="x">Input array</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array of the same type as a, containing the rounded values. Unless out was specified, a new array is created. A reference to the result is returned.
        ///  The real and imaginary parts of complex numbers are rounded separately.The result of rounding a float is a float.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.around.html</remarks>
        public static NDArray around(in NDArray x, Type outType)
            => x.TensorEngine.Round(x, outType);
    }
}
