using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        ///     First array elements raised to powers from second array, element-wise.
        /// </summary>
        /// <param name="x1">The bases.</param>
        /// <param name="x2">The exponents (scalar or array-like).</param>
        /// <param name="dtype">The dtype of the returned NDArray</param>
        /// <returns>The bases in x1 raised to the exponents in x2. This is a scalar NDArray if both x1 and x2 are scalars.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.power.html</remarks>
        public static NDArray power(NDArray x1, object x2, Type dtype) => x1.TensorEngine.Power(x1, np.asanyarray(x2), dtype);

        /// <summary>
        ///     First array elements raised to powers from second array, element-wise.
        /// </summary>
        /// <param name="x1">The bases.</param>
        /// <param name="x2">The exponents (scalar or array-like).</param>
        /// <param name="typeCode">The dtype of the returned NDArray</param>
        /// <returns>The bases in x1 raised to the exponents in x2. This is a scalar NDArray if both x1 and x2 are scalars.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.power.html</remarks>
        public static NDArray power(NDArray x1, object x2, NPTypeCode typeCode) => x1.TensorEngine.Power(x1, np.asanyarray(x2), typeCode);

        /// <summary>
        ///     First array elements raised to powers from second array, element-wise.
        /// </summary>
        /// <param name="x1">The bases.</param>
        /// <param name="x2">The exponents (scalar or array-like).</param>
        /// <returns>The bases in x1 raised to the exponents in x2. This is a scalar NDArray if both x1 and x2 are scalars.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.power.html</remarks>
        public static NDArray power(NDArray x1, object x2) => x1.TensorEngine.Power(x1, np.asanyarray(x2), (NPTypeCode?)null);

        /// <summary>
        ///     First array elements raised to powers from second array, element-wise.
        ///     Supports broadcasting between x1 and x2.
        /// </summary>
        /// <param name="x1">The bases.</param>
        /// <param name="x2">The exponents (array).</param>
        /// <param name="dtype">The dtype of the returned NDArray.</param>
        /// <returns>The bases in x1 raised to the exponents in x2.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.power.html</remarks>
        public static NDArray power(NDArray x1, NDArray x2, Type dtype) => x1.TensorEngine.Power(x1, x2, dtype);

        /// <summary>
        ///     First array elements raised to powers from second array, element-wise.
        ///     Supports broadcasting between x1 and x2.
        /// </summary>
        /// <param name="x1">The bases.</param>
        /// <param name="x2">The exponents (array).</param>
        /// <param name="typeCode">The dtype of the returned NDArray.</param>
        /// <returns>The bases in x1 raised to the exponents in x2.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.power.html</remarks>
        public static NDArray power(NDArray x1, NDArray x2, NPTypeCode typeCode) => x1.TensorEngine.Power(x1, x2, typeCode);

        /// <summary>
        ///     First array elements raised to powers from second array, element-wise.
        ///     Supports broadcasting between x1 and x2.
        /// </summary>
        /// <param name="x1">The bases.</param>
        /// <param name="x2">The exponents (array).</param>
        /// <returns>The bases in x1 raised to the exponents in x2.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.power.html</remarks>
        public static NDArray power(NDArray x1, NDArray x2) => x1.TensorEngine.Power(x1, x2, (NPTypeCode?)null);

        /// <summary>
        ///     First array elements raised to powers from second array, element-wise.
        ///     Mirrors NumPy's ufunc signature: <c>power(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x1">The bases.</param>
        /// <param name="x2">The exponents (array).</param>
        /// <param name="@out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the computation runs in this dtype (power(2, -1, dtype: float64) = 0.5; inputs must be same_kind-castable to it).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.power.html</remarks>
        public static NDArray power(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x1.TensorEngine.Power(x1, x2, dtype, @out, where);

        /// <summary>
        ///     First array elements raised to powers from second array, element-wise.
        ///     Supports broadcasting between x1 and x2.
        /// </summary>
        /// <param name="x1">The bases.</param>
        /// <param name="x2">The exponents (array).</param>
        /// <param name="dtype">The dtype of the returned NDArray.</param>
        /// <returns>The bases in x1 raised to the exponents in x2.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.power.html</remarks>
        public static NDArray power(in NDArray x1, in NDArray x2, Type dtype) => x1.TensorEngine.Power(x1, x2, dtype);

        /// <summary>
        ///     First array elements raised to powers from second array, element-wise.
        ///     Supports broadcasting between x1 and x2.
        /// </summary>
        /// <param name="x1">The bases.</param>
        /// <param name="x2">The exponents (array).</param>
        /// <param name="typeCode">The dtype of the returned NDArray.</param>
        /// <returns>The bases in x1 raised to the exponents in x2.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.power.html</remarks>
        public static NDArray power(in NDArray x1, in NDArray x2, NPTypeCode typeCode) => x1.TensorEngine.Power(x1, x2, typeCode);

        /// <summary>
        ///     First array elements raised to powers from second array, element-wise.
        ///     Supports broadcasting between x1 and x2.
        /// </summary>
        /// <param name="x1">The bases.</param>
        /// <param name="x2">The exponents (array).</param>
        /// <returns>The bases in x1 raised to the exponents in x2.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.power.html</remarks>
        public static NDArray power(in NDArray x1, in NDArray x2) => x1.TensorEngine.Power(x1, x2, (NPTypeCode?)null);

        /// <summary>
        ///     Return the element-wise square of the input.
        /// </summary>
        /// <param name="x">Input data.</param>
        /// <returns>Element-wise x*x, of the same shape and dtype as x. Returns scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.power.html</remarks>
        public static NDArray square(NDArray x) => x.TensorEngine.Square(x, (NPTypeCode?)null);

        /// <summary>
        ///     Return the element-wise square of the input.
        ///     Mirrors NumPy's ufunc signature: <c>square(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x">Input data.</param>
        /// <param name="@out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the input must be same_kind-castable to it.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.square.html</remarks>
        public static NDArray square(NDArray x, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x.TensorEngine.Square(x, dtype, @out, where);
    }
}
