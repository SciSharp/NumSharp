using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class np
    {

        /// <summary>
        ///     Natural logarithm, element-wise.
        ///     The natural logarithm log is the inverse of the exponential function, so that log(exp(x)) = x.
        ///     The natural logarithm is logarithm in base e.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>The natural logarithm of x, element-wise. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.log.html</remarks>
        public static NDArray log(NDArray x) => x.TensorEngine.Log(x);

        /// <summary>
        ///     Natural logarithm, element-wise.
        ///     Mirrors NumPy's ufunc signature: <c>log(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <param name="@out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the computation runs at this precision; integer/bool requests raise NumPy's "No loop matching" error.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.log.html</remarks>
        public static NDArray log(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x.TensorEngine.Log(x, dtype, @out, where);

        /// <summary>
        ///     Base-2 logarithm of x.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>Base-2 logarithm of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.log2.html</remarks>
        public static NDArray log2(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null) => x.TensorEngine.Log2(x, dtype, @out, where);

        /// <summary>
        ///     Base-2 logarithm of x.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>Base-2 logarithm of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.log2.html</remarks>
        public static NDArray log2(NDArray x) => x.TensorEngine.Log2(x);

        /// <summary>
        ///     Return the base 10 logarithm of the input array, element-wise.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>The logarithm to the base 10 of x, element-wise. NaNs are returned where x is negative. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.log10.html</remarks>
        public static NDArray log10(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null) => x.TensorEngine.Log10(x, dtype, @out, where);

        /// <summary>
        ///     Return the base 10 logarithm of the input array, element-wise.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>The logarithm to the base 10 of x, element-wise. NaNs are returned where x is negative. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.log10.html</remarks>
        public static NDArray log10(NDArray x) => x.TensorEngine.Log10(x);

        /// <summary>
        ///     Return the natural logarithm of one plus the input array, element-wise.<br></br>
        ///     Calculates log(1 + x).
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>Natural logarithm of 1 + x, element-wise. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.log1p.html</remarks>
        public static NDArray log1p(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null) => x.TensorEngine.Log1p(x, dtype, @out, where);

        /// <summary>
        ///     Return the natural logarithm of one plus the input array, element-wise.<br></br>
        ///     Calculates log(1 + x).
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>Natural logarithm of 1 + x, element-wise. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.log1p.html</remarks>
        public static NDArray log1p(NDArray x) => x.TensorEngine.Log1p(x);
    }
}
