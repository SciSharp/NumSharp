using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute tangent element-wise. <br></br>
        ///     Equivalent to np.sin(x)/np.cos(x) element-wise.
        /// </summary>
        /// <param name="x">Angle, in radians (2 \pi rad equals 360 degrees).</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The sine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.tan.html</remarks>
        public static NDArray tan(in NDArray x, NPTypeCode? outType = null) 
            => x.TensorEngine.Tan(x, outType);

        /// <summary>
        ///     Trigonometric sine, element-wise.
        /// </summary>
        /// <param name="x">Angle, in radians (2 \pi rad equals 360 degrees).</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The sine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.tan.html</remarks>
        public static NDArray tan(in NDArray x, Type outType) 
            => x.TensorEngine.Tan(x, outType);

        /// <summary>
        ///     Compute hyperbolic tangent element-wise. <br></br>
        ///     Equivalent to np.sinh(x)/np.cosh(x) or -1j * np.tan(1j*x).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The sine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.tanh.html</remarks>
        public static NDArray tanh(in NDArray x, NPTypeCode? outType = null) 
            => x.TensorEngine.Tanh(x, outType);

        /// <summary>
        ///     Compute hyperbolic tangent element-wise. <br></br>
        ///     Equivalent to np.sinh(x)/np.cosh(x) or -1j * np.tan(1j*x).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The sine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.tanh.html</remarks>
        public static NDArray tanh(in NDArray x, Type outType) 
            => x.TensorEngine.Tanh(x, outType);

        /// <summary>
        ///     Compute trigonometric inverse tangent, element-wise. <br></br>
        ///     The inverse of tan, so that if y = tan(x) then x = arctan(y).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>Return has the same shape as x. Its real part is in [-pi/2, pi/2] (arctan(+/-inf) returns +/-pi/2). This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan.html</remarks>
        public static NDArray arctan(in NDArray x, NPTypeCode? outType = null)
            => x.TensorEngine.ATan(x, outType);

        /// <summary>
        ///     Compute trigonometric inverse tangent, element-wise. <br></br>
        ///     The inverse of tan, so that if y = tan(x) then x = arctan(y).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>Return has the same shape as x. Its real part is in [-pi/2, pi/2] (arctan(+/-inf) returns +/-pi/2). This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan.html</remarks>
        public static NDArray arctan(in NDArray x, Type outType)
            => x.TensorEngine.ATan(x, outType);

        /// <summary>
        ///     Compute Element-wise arc tangent of x1/x2 choosing the quadrant correctly. <br></br>
        ///     By IEEE convention, this function is defined for x2 = +/-0 and for either or both of x1 and x2 = +/-inf
        /// </summary>
        /// <param name="x">Input array y-coordinates.</param>
        /// <param name="x">x-coordinates. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The Array of angles in radians, in the range [-pi, pi]. This is a scalar if both x1 and x2 are scalars.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan2.html</remarks>
        public static NDArray arctan2(in NDArray y, in NDArray x, NPTypeCode? outType = null)
            => x.TensorEngine.ATan2(y, x, outType);

        /// <summary>
        ///     Compute Element-wise arc tangent of x1/x2 choosing the quadrant correctly. <br></br>
        ///     By IEEE convention, this function is defined for x2 = +/-0 and for either or both of x1 and x2 = +/-inf
        /// </summary>
        /// <param name="x">Input array y-coordinates.</param>
        /// <param name="x">x-coordinates. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The Array of angles in radians, in the range [-pi, pi]. This is a scalar if both x1 and x2 are scalars.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan2.html</remarks>
        public static NDArray arctan2(in NDArray y, in NDArray x, Type outType)
            => x.TensorEngine.ATan2(y, x, outType);
    }
}
