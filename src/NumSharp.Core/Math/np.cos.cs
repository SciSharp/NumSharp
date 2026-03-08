using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Cosine element-wise.
        /// </summary>
        /// <param name="x">Input array in radians.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The sine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.cos.html</remarks>
        public static NDArray cos(in NDArray x, NPTypeCode? dtype = null) 
            => x.TensorEngine.Cos(x, dtype);

        /// <summary>
        ///     Cosine element-wise.
        /// </summary>
        /// <param name="x">Input array in radians.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The sine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.cos.html</remarks>
        public static NDArray cos(in NDArray x, Type dtype) 
            => x.TensorEngine.Cos(x, dtype);

        /// <summary>
        ///     Hyperbolic cosine, element-wise. <br></br>
        ///     Equivalent to 1/2 * (np.exp(x) + np.exp(-x)) and np.cos(1j* x).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>Output array of same shape as x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.cosh.html</remarks>
        public static NDArray cosh(in NDArray x, NPTypeCode? dtype = null) 
            => x.TensorEngine.Cosh(x, dtype);

        /// <summary>
        ///     Hyperbolic cosine, element-wise. <br></br>
        ///     Equivalent to 1/2 * (np.exp(x) + np.exp(-x)) and np.cos(1j* x).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>Output array of same shape as x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.cosh.html</remarks>
        public static NDArray cosh(in NDArray x, Type dtype) 
            => x.TensorEngine.Cosh(x, dtype);

        /// <summary>
        ///     Trigonometric inverse cosine, element-wise. <br></br>
        ///     The inverse of cos so that, if y = cos(x), then x = arccos(y).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The angle of the ray intersecting the unit circle at the given x-coordinate in radians [0, pi]. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.arccos.html</remarks>
        public static NDArray arccos(in NDArray x, NPTypeCode? dtype = null)
            => x.TensorEngine.ACos(x, dtype);

        /// <summary>
        ///     Trigonometric inverse cosine, element-wise. <br></br>
        ///     The inverse of cos so that, if y = cos(x), then x = arccos(y).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The angle of the ray intersecting the unit circle at the given x-coordinate in radians [0, pi]. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.arccos.html</remarks>
        public static NDArray arccos(in NDArray x, Type dtype)
            => x.TensorEngine.ACos(x, dtype);
    }
}
