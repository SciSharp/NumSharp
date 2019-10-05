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
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The sine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.cos.html</remarks>
        public static NDArray cos(in NDArray x, NPTypeCode? outType = null) 
            => x.TensorEngine.Cos(x, outType);

        /// <summary>
        ///     Cosine element-wise.
        /// </summary>
        /// <param name="x">Input array in radians.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The sine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.cos.html</remarks>
        public static NDArray cos(in NDArray x, Type outType) 
            => x.TensorEngine.Cos(x, outType);

        /// <summary>
        ///     Hyperbolic cosine, element-wise. <br></br>
        ///     Equivalent to 1/2 * (np.exp(x) + np.exp(-x)) and np.cos(1j* x).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>Output array of same shape as x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.cosh.html</remarks>
        public static NDArray cosh(in NDArray x, NPTypeCode? outType = null) 
            => x.TensorEngine.Cosh(x, outType);

        /// <summary>
        ///     Hyperbolic cosine, element-wise. <br></br>
        ///     Equivalent to 1/2 * (np.exp(x) + np.exp(-x)) and np.cos(1j* x).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>Output array of same shape as x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.cosh.html</remarks>
        public static NDArray cosh(in NDArray x, Type outType) 
            => x.TensorEngine.Cosh(x, outType);

        /// <summary>
        ///     Trigonometric inverse cosine, element-wise. <br></br>
        ///     The inverse of cos so that, if y = cos(x), then x = arccos(y).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The angle of the ray intersecting the unit circle at the given x-coordinate in radians [0, pi]. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.arccos.html</remarks>
        public static NDArray arccos(in NDArray x, NPTypeCode? outType = null)
            => x.TensorEngine.ACos(x, outType);

        /// <summary>
        ///     Trigonometric inverse cosine, element-wise. <br></br>
        ///     The inverse of cos so that, if y = cos(x), then x = arccos(y).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The angle of the ray intersecting the unit circle at the given x-coordinate in radians [0, pi]. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.arccos.html</remarks>
        public static NDArray arccos(in NDArray x, Type outType)
            => x.TensorEngine.ACos(x, outType);
    }
}
