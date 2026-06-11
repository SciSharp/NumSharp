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
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array of the same type as a, containing the rounded values. Unless out was specified, a new array is created. A reference to the result is returned.
        ///  The real and imaginary parts of complex numbers are rounded separately.The result of rounding a float is a float.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.around.html</remarks>
        public static NDArray round_(NDArray x, NDArray @out = null)
            => x.TensorEngine.Round(x, null, @out, null);

        /// <summary>
        ///     Computed in <paramref name="dtype"/> — positional-dtype convenience
        ///     overload. NumPy's np.round/np.around accept <c>out=</c> only (they are
        ///     functions, not ufuncs — no where=/dtype= kwargs; probed 2.4.2).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.around.html</remarks>
        public static NDArray round_(NDArray x, NPTypeCode dtype)
            => x.TensorEngine.Round(x, dtype);

        /// <summary>
        ///     Evenly round to the given number of decimals.
        /// </summary>
        /// <param name="x">Input array</param>
        /// <param name="decimals">Number of decimal places to round to</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array of the same type as a, containing the rounded values. Unless out was specified, a new array is created. A reference to the result is returned.
        ///  The real and imaginary parts of complex numbers are rounded separately.The result of rounding a float is a float.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.around.html</remarks>
        public static NDArray round_(NDArray x, int decimals, NDArray @out = null)
            => x.TensorEngine.Round(x, decimals, null, @out);

        /// <summary>
        ///     Computed in <paramref name="dtype"/> — positional-dtype convenience
        ///     overload (out= only at the NumPy-shaped surface; see above).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.around.html</remarks>
        public static NDArray round_(NDArray x, int decimals, NPTypeCode dtype)
            => x.TensorEngine.Round(x, decimals, dtype);

        /// <summary>
        ///     Evenly round to the given number of decimals.
        /// </summary>
        /// <param name="x">Input array</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array of the same type as a, containing the rounded values. Unless out was specified, a new array is created. A reference to the result is returned.
        ///  The real and imaginary parts of complex numbers are rounded separately.The result of rounding a float is a float.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.around.html</remarks>
        public static NDArray round_(NDArray x, Type dtype) 
            => x.TensorEngine.Round(x, dtype);

        /// <summary>
        ///     Evenly round to the given number of decimals.
        /// </summary>
        /// <param name="x">Input array</param>
        /// <param name="decimals">Number of decimal places to round to</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array of the same type as a, containing the rounded values. Unless out was specified, a new array is created. A reference to the result is returned.
        ///  The real and imaginary parts of complex numbers are rounded separately.The result of rounding a float is a float.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.around.html</remarks>
        public static NDArray round_(NDArray x, int decimals, Type dtype)
            => x.TensorEngine.Round(x, decimals, dtype);

        /// <summary>
        ///     Evenly round to the given number of decimals.
        /// </summary>
        /// <param name="x">Input array</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array of the same type as a, containing the rounded values. Unless out was specified, a new array is created. A reference to the result is returned.
        ///  The real and imaginary parts of complex numbers are rounded separately.The result of rounding a float is a float.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.around.html</remarks>
        public static NDArray around(NDArray x, NDArray @out = null)
            => x.TensorEngine.Round(x, null, @out, null);

        /// <summary>
        ///     Computed in <paramref name="dtype"/> — positional-dtype convenience
        ///     overload. NumPy's np.round/np.around accept <c>out=</c> only (they are
        ///     functions, not ufuncs — no where=/dtype= kwargs; probed 2.4.2).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.around.html</remarks>
        public static NDArray around(NDArray x, NPTypeCode dtype)
            => x.TensorEngine.Round(x, dtype);

        /// <summary>
        ///     Evenly round to the given number of decimals.
        /// </summary>
        /// <param name="x">Input array</param>
        /// <param name="decimals">Number of decimal places to round to</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array of the same type as a, containing the rounded values. Unless out was specified, a new array is created. A reference to the result is returned.
        ///  The real and imaginary parts of complex numbers are rounded separately.The result of rounding a float is a float.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.around.html</remarks>
        public static NDArray around(NDArray x, int decimals, NDArray @out = null)
            => x.TensorEngine.Round(x, decimals, null, @out);

        /// <summary>
        ///     Computed in <paramref name="dtype"/> — positional-dtype convenience
        ///     overload (out= only at the NumPy-shaped surface; see above).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.around.html</remarks>
        public static NDArray around(NDArray x, int decimals, NPTypeCode dtype)
            => x.TensorEngine.Round(x, decimals, dtype);

        /// <summary>
        ///     Evenly round to the given number of decimals.
        /// </summary>
        /// <param name="x">Input array</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array of the same type as a, containing the rounded values. Unless out was specified, a new array is created. A reference to the result is returned.
        ///  The real and imaginary parts of complex numbers are rounded separately.The result of rounding a float is a float.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.around.html</remarks>
        public static NDArray around(NDArray x, Type dtype)
            => x.TensorEngine.Round(x, dtype);

        /// <summary>
        ///     Evenly round to the given number of decimals.
        /// </summary>
        /// <param name="x">Input array</param>
        /// <param name="decimals">Number of decimal places to round to</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>An array of the same type as a, containing the rounded values. Unless out was specified, a new array is created. A reference to the result is returned.
        ///  The real and imaginary parts of complex numbers are rounded separately.The result of rounding a float is a float.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.around.html</remarks>
        public static NDArray around(NDArray x, int decimals, Type dtype)
            => x.TensorEngine.Round(x, decimals, dtype);
    }
}
