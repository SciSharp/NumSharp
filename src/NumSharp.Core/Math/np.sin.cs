using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Trigonometric sine, element-wise.
        /// </summary>
        /// <param name="x">Angle, in radians (2 \pi rad equals 360 degrees).</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The sine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.sin.html</remarks>
        public static NDArray sin(in NDArray x, NPTypeCode? outType = null) 
            => x.TensorEngine.Sin(x, outType);

        /// <summary>
        ///     Trigonometric sine, element-wise.
        /// </summary>
        /// <param name="x">Angle, in radians (2 \pi rad equals 360 degrees).</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The sine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.sin.html</remarks>
        public static NDArray sin(in NDArray x, Type outType) 
            => x.TensorEngine.Sin(x, outType);

        /// <summary>
        ///     Hyperbolic sine, element-wise. <br></br>
        ///     Equivalent to 1/2 * (np.exp(x) - np.exp(-x)) or -1j * np.sin(1j*x).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The sine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.sinh.html</remarks>
        public static NDArray sinh(in NDArray x, NPTypeCode? outType = null) 
            => x.TensorEngine.Sinh(x, outType);

        /// <summary>
        ///     Hyperbolic sine, element-wise. <br></br>
        ///     Equivalent to 1/2 * (np.exp(x) - np.exp(-x)) or -1j * np.sin(1j*x).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The sine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.sinh.html</remarks>
        public static NDArray sinh(in NDArray x, Type outType) 
            => x.TensorEngine.Sinh(x, outType);

        /// <summary>
        ///     Inverse sine, element-wise. <br></br>
        ///     The convention is to return the angle z whose real part lies in [-pi/2, pi/2].
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The inverse sine of each element in x, in radians and in the closed interval [-pi/2, pi/2]. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.arcsin.html</remarks>
        public static NDArray arcsin(in NDArray x, NPTypeCode? outType = null)
            => x.TensorEngine.ASin(x, outType);

        /// <summary>
        ///     Inverse sine, element-wise. <br></br>
        ///     The convention is to return the angle z whose real part lies in [-pi/2, pi/2].
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The inverse sine of each element in x, in radians and in the closed interval [-pi/2, pi/2]. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.arcsin.html</remarks>
        public static NDArray arcsin(in NDArray x, Type outType)
            => x.TensorEngine.ASin(x, outType);
    }
}
