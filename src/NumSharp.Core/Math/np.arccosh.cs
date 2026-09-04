using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Inverse hyperbolic cosine, element-wise. <br></br>
        ///     The inverse of <c>cosh</c> so that, if <c>y = cosh(x)</c>, then <c>x = arccosh(y)</c>.
        ///     Mirrors NumPy's ufunc signature: <c>arccosh(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="@out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the computation runs at this precision; integer/bool requests raise NumPy's "No loop matching" error.</param>
        /// <returns>Array of the same shape as x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arccosh.html</remarks>
        public static NDArray arccosh(NDArray x, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x.TensorEngine.ACosh(x, dtype.AsType(), @out, where);

        /// <summary>
        ///     Inverse hyperbolic cosine, element-wise, computed in <paramref name="dtype"/>.
        ///     Positional-dtype convenience overload (NumPy accepts dtype only as a keyword).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arccosh.html</remarks>
        public static NDArray arccosh(NDArray x, NPTypeCode dtype)
            => x.TensorEngine.ACosh(x, dtype.AsType());

        /// <summary>
        ///     Inverse hyperbolic cosine, element-wise.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>Array of the same shape as x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arccosh.html</remarks>
        public static NDArray arccosh(NDArray x, Type dtype)
            => x.TensorEngine.ACosh(x, dtype);

        /// <summary>
        ///     Inverse hyperbolic cosine, element-wise (Array-API alias of <see cref="arccosh(NDArray, NDArray, NDArray, NPTypeCode?)"/>, added in NumPy 2.0).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.acosh.html</remarks>
        public static NDArray acosh(NDArray x, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x.TensorEngine.ACosh(x, dtype.AsType(), @out, where);

        /// <summary>
        ///     Inverse hyperbolic cosine, element-wise (Array-API alias), computed in <paramref name="dtype"/>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.acosh.html</remarks>
        public static NDArray acosh(NDArray x, NPTypeCode dtype)
            => x.TensorEngine.ACosh(x, dtype.AsType());

        /// <summary>
        ///     Inverse hyperbolic cosine, element-wise (Array-API alias).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.acosh.html</remarks>
        public static NDArray acosh(NDArray x, Type dtype)
            => x.TensorEngine.ACosh(x, dtype);
    }
}
