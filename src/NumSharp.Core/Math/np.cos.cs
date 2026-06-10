using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Cosine element-wise.
        ///     Mirrors NumPy's ufunc signature: <c>cos(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x">Input array in radians.</param>
        /// <param name="@out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the computation runs at this precision; integer/bool requests raise NumPy's "No loop matching" error.</param>
        /// <returns>The cosine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.cos.html</remarks>
        public static NDArray cos(NDArray x, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x.TensorEngine.Cos(x, dtype, @out, where);

        /// <summary>
        ///     Cosine element-wise, computed in <paramref name="dtype"/>.
        ///     Positional-dtype convenience overload (NumPy accepts dtype only as a keyword).
        /// </summary>
        /// <param name="x">Input array in radians.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.cos.html</remarks>
        public static NDArray cos(NDArray x, NPTypeCode dtype)
            => x.TensorEngine.Cos(x, dtype);

        /// <summary>
        ///     Cosine element-wise.
        /// </summary>
        /// <param name="x">Input array in radians.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The sine of each element of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.cos.html</remarks>
        public static NDArray cos(NDArray x, Type dtype) 
            => x.TensorEngine.Cos(x, dtype);

        /// <summary>
        ///     Hyperbolic cosine, element-wise. <br></br>
        ///     Equivalent to 1/2 * (np.exp(x) + np.exp(-x)) and np.cos(1j* x).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>Output array of same shape as x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.cosh.html</remarks>
        public static NDArray cosh(NDArray x, NPTypeCode? dtype = null) 
            => x.TensorEngine.Cosh(x, dtype);

        /// <summary>
        ///     Hyperbolic cosine, element-wise. <br></br>
        ///     Equivalent to 1/2 * (np.exp(x) + np.exp(-x)) and np.cos(1j* x).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>Output array of same shape as x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.cosh.html</remarks>
        public static NDArray cosh(NDArray x, Type dtype) 
            => x.TensorEngine.Cosh(x, dtype);

        /// <summary>
        ///     Trigonometric inverse cosine, element-wise. <br></br>
        ///     The inverse of cos so that, if y = cos(x), then x = arccos(y).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The angle of the ray intersecting the unit circle at the given x-coordinate in radians [0, pi]. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arccos.html</remarks>
        public static NDArray arccos(NDArray x, NPTypeCode? dtype = null)
            => x.TensorEngine.ACos(x, dtype);

        /// <summary>
        ///     Trigonometric inverse cosine, element-wise. <br></br>
        ///     The inverse of cos so that, if y = cos(x), then x = arccos(y).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The angle of the ray intersecting the unit circle at the given x-coordinate in radians [0, pi]. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arccos.html</remarks>
        public static NDArray arccos(NDArray x, Type dtype)
            => x.TensorEngine.ACos(x, dtype);
    }
}
