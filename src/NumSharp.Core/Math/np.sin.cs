using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Trigonometric sine, element-wise.
        ///     Mirrors NumPy's ufunc signature: <c>sin(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x">Angle, in radians (2 \pi rad equals 360 degrees).</param>
        /// <param name="@out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the computation runs at this precision; integer/bool requests raise NumPy's "No loop matching" error.</param>
        /// <returns>The sine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.sin.html</remarks>
        public static NDArray sin(NDArray x, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x.TensorEngine.Sin(x, dtype, @out, where);

        /// <summary>
        ///     Trigonometric sine, element-wise, computed in <paramref name="dtype"/>.
        ///     Positional-dtype convenience overload (NumPy accepts dtype only as a keyword).
        /// </summary>
        /// <param name="x">Angle, in radians (2 \pi rad equals 360 degrees).</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.sin.html</remarks>
        public static NDArray sin(NDArray x, NPTypeCode dtype)
            => x.TensorEngine.Sin(x, dtype);

        /// <summary>
        ///     Trigonometric sine, element-wise.
        /// </summary>
        /// <param name="x">Angle, in radians (2 \pi rad equals 360 degrees).</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The sine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.sin.html</remarks>
        public static NDArray sin(NDArray x, Type dtype) 
            => x.TensorEngine.Sin(x, dtype);

        /// <summary>
        ///     Hyperbolic sine, element-wise. <br></br>
        ///     Equivalent to 1/2 * (np.exp(x) - np.exp(-x)) or -1j * np.sin(1j*x).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The sine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.sinh.html</remarks>
        public static NDArray sinh(NDArray x, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x.TensorEngine.Sinh(x, dtype, @out, where);

        /// <summary>
        ///     Computed in <paramref name="dtype"/> — positional-dtype convenience
        ///     overload (NumPy accepts dtype only as a keyword).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The loop dtype the computation should run in.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.sinh.html</remarks>
        public static NDArray sinh(NDArray x, NPTypeCode dtype)
            => x.TensorEngine.Sinh(x, dtype);

        /// <summary>
        ///     Hyperbolic sine, element-wise. <br></br>
        ///     Equivalent to 1/2 * (np.exp(x) - np.exp(-x)) or -1j * np.sin(1j*x).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The sine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.sinh.html</remarks>
        public static NDArray sinh(NDArray x, Type dtype) 
            => x.TensorEngine.Sinh(x, dtype);

        /// <summary>
        ///     Inverse sine, element-wise. <br></br>
        ///     The convention is to return the angle z whose real part lies in [-pi/2, pi/2].
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The inverse sine of each element in x, in radians and in the closed interval [-pi/2, pi/2]. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html</remarks>
        public static NDArray arcsin(NDArray x, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x.TensorEngine.ASin(x, dtype, @out, where);

        /// <summary>
        ///     Computed in <paramref name="dtype"/> — positional-dtype convenience
        ///     overload (NumPy accepts dtype only as a keyword).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The loop dtype the computation should run in.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html</remarks>
        public static NDArray arcsin(NDArray x, NPTypeCode dtype)
            => x.TensorEngine.ASin(x, dtype);

        /// <summary>
        ///     Inverse sine, element-wise. <br></br>
        ///     The convention is to return the angle z whose real part lies in [-pi/2, pi/2].
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The inverse sine of each element in x, in radians and in the closed interval [-pi/2, pi/2]. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html</remarks>
        public static NDArray arcsin(NDArray x, Type dtype)
            => x.TensorEngine.ASin(x, dtype);
    }
}
