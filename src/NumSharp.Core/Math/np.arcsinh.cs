using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Inverse hyperbolic sine, element-wise. <br></br>
        ///     The inverse of <c>sinh</c> so that, if <c>y = sinh(x)</c>, then <c>x = arcsinh(y)</c>.
        ///     Mirrors NumPy's ufunc signature: <c>arcsinh(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="@out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the computation runs at this precision; integer/bool requests raise NumPy's "No loop matching" error.</param>
        /// <returns>Array of the same shape as x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arcsinh.html</remarks>
        public static NDArray arcsinh(NDArray x, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x.TensorEngine.ASinh(x, dtype, @out, where);

        /// <summary>
        ///     Inverse hyperbolic sine, element-wise, computed in <paramref name="dtype"/>.
        ///     Positional-dtype convenience overload (NumPy accepts dtype only as a keyword).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arcsinh.html</remarks>
        public static NDArray arcsinh(NDArray x, NPTypeCode dtype)
            => x.TensorEngine.ASinh(x, dtype);

        /// <summary>
        ///     Inverse hyperbolic sine, element-wise.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>Array of the same shape as x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arcsinh.html</remarks>
        public static NDArray arcsinh(NDArray x, Type dtype)
            => x.TensorEngine.ASinh(x, dtype);

        /// <summary>
        ///     Inverse hyperbolic sine, element-wise (Array-API alias of <see cref="arcsinh(NDArray, NDArray, NDArray, NPTypeCode?)"/>, added in NumPy 2.0).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asinh.html</remarks>
        public static NDArray asinh(NDArray x, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x.TensorEngine.ASinh(x, dtype, @out, where);

        /// <summary>
        ///     Inverse hyperbolic sine, element-wise (Array-API alias), computed in <paramref name="dtype"/>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asinh.html</remarks>
        public static NDArray asinh(NDArray x, NPTypeCode dtype)
            => x.TensorEngine.ASinh(x, dtype);

        /// <summary>
        ///     Inverse hyperbolic sine, element-wise (Array-API alias).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asinh.html</remarks>
        public static NDArray asinh(NDArray x, Type dtype)
            => x.TensorEngine.ASinh(x, dtype);
    }
}
