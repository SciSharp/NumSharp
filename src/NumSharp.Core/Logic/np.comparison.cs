using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        #region equal

        /// <summary>
        /// Return (x1 == x2) element-wise.
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <returns>Output array of bools.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.equal.html</remarks>
        public static NDArray<bool> equal(NDArray x1, NDArray x2) => x1 == x2;

        /// <summary>
        /// Return (equal) element-wise into <paramref name="@out"/> (NumPy ufunc out=/where=).
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <param name="@out">A location into which the result is stored; any numeric dtype (bool casts same_kind to all of them, True→1); returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written; masked-off out slots keep prior contents.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.equal.html</remarks>
        public static NDArray equal(NDArray x1, NDArray x2, NDArray @out, NDArray where = null)
            => x1.TensorEngine.Compare(x1, x2, @out, where);

        /// <summary>
        /// Return (x1 == x2) element-wise with scalar.
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Scalar or array-like value.</param>
        /// <returns>Output array of bools.</returns>
        public static NDArray<bool> equal(NDArray x1, object x2) => x1 == x2;

        /// <summary>
        /// Return (x1 == x2) element-wise with scalar on left.
        /// </summary>
        /// <param name="x1">Scalar or array-like value.</param>
        /// <param name="x2">Input array.</param>
        /// <returns>Output array of bools.</returns>
        public static NDArray<bool> equal(object x1, NDArray x2) => x1 == x2;

        #endregion

        #region not_equal

        /// <summary>
        /// Return (x1 != x2) element-wise.
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <returns>Output array of bools.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.not_equal.html</remarks>
        public static NDArray<bool> not_equal(NDArray x1, NDArray x2) => x1 != x2;

        /// <summary>
        /// Return (not_equal) element-wise into <paramref name="@out"/> (NumPy ufunc out=/where=).
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <param name="@out">A location into which the result is stored; any numeric dtype (bool casts same_kind to all of them, True→1); returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written; masked-off out slots keep prior contents.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.not_equal.html</remarks>
        public static NDArray not_equal(NDArray x1, NDArray x2, NDArray @out, NDArray where = null)
            => x1.TensorEngine.NotEqual(x1, x2, @out, where);

        /// <summary>
        /// Return (x1 != x2) element-wise with scalar.
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Scalar or array-like value.</param>
        /// <returns>Output array of bools.</returns>
        public static NDArray<bool> not_equal(NDArray x1, object x2) => x1 != x2;

        /// <summary>
        /// Return (x1 != x2) element-wise with scalar on left.
        /// </summary>
        /// <param name="x1">Scalar or array-like value.</param>
        /// <param name="x2">Input array.</param>
        /// <returns>Output array of bools.</returns>
        public static NDArray<bool> not_equal(object x1, NDArray x2) => x1 != x2;

        #endregion

        #region less

        /// <summary>
        /// Return (x1 &lt; x2) element-wise.
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <returns>Output array of bools.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.less.html</remarks>
        public static NDArray<bool> less(NDArray x1, NDArray x2) => x1 < x2;

        /// <summary>
        /// Return (less) element-wise into <paramref name="@out"/> (NumPy ufunc out=/where=).
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <param name="@out">A location into which the result is stored; any numeric dtype (bool casts same_kind to all of them, True→1); returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written; masked-off out slots keep prior contents.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.less.html</remarks>
        public static NDArray less(NDArray x1, NDArray x2, NDArray @out, NDArray where = null)
            => x1.TensorEngine.Less(x1, x2, @out, where);

        /// <summary>
        /// Return (x1 &lt; x2) element-wise with scalar.
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Scalar or array-like value.</param>
        /// <returns>Output array of bools.</returns>
        public static NDArray<bool> less(NDArray x1, object x2) => x1 < x2;

        /// <summary>
        /// Return (x1 &lt; x2) element-wise with scalar on left.
        /// </summary>
        /// <param name="x1">Scalar or array-like value.</param>
        /// <param name="x2">Input array.</param>
        /// <returns>Output array of bools.</returns>
        public static NDArray<bool> less(object x1, NDArray x2) => x1 < x2;

        #endregion

        #region greater

        /// <summary>
        /// Return (x1 &gt; x2) element-wise.
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <returns>Output array of bools.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.greater.html</remarks>
        public static NDArray<bool> greater(NDArray x1, NDArray x2) => x1 > x2;

        /// <summary>
        /// Return (greater) element-wise into <paramref name="@out"/> (NumPy ufunc out=/where=).
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <param name="@out">A location into which the result is stored; any numeric dtype (bool casts same_kind to all of them, True→1); returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written; masked-off out slots keep prior contents.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.greater.html</remarks>
        public static NDArray greater(NDArray x1, NDArray x2, NDArray @out, NDArray where = null)
            => x1.TensorEngine.Greater(x1, x2, @out, where);

        /// <summary>
        /// Return (x1 &gt; x2) element-wise with scalar.
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Scalar or array-like value.</param>
        /// <returns>Output array of bools.</returns>
        public static NDArray<bool> greater(NDArray x1, object x2) => x1 > x2;

        /// <summary>
        /// Return (x1 &gt; x2) element-wise with scalar on left.
        /// </summary>
        /// <param name="x1">Scalar or array-like value.</param>
        /// <param name="x2">Input array.</param>
        /// <returns>Output array of bools.</returns>
        public static NDArray<bool> greater(object x1, NDArray x2) => x1 > x2;

        #endregion

        #region less_equal

        /// <summary>
        /// Return (x1 &lt;= x2) element-wise.
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <returns>Output array of bools.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.less_equal.html</remarks>
        public static NDArray<bool> less_equal(NDArray x1, NDArray x2) => x1 <= x2;

        /// <summary>
        /// Return (less_equal) element-wise into <paramref name="@out"/> (NumPy ufunc out=/where=).
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <param name="@out">A location into which the result is stored; any numeric dtype (bool casts same_kind to all of them, True→1); returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written; masked-off out slots keep prior contents.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.less_equal.html</remarks>
        public static NDArray less_equal(NDArray x1, NDArray x2, NDArray @out, NDArray where = null)
            => x1.TensorEngine.LessEqual(x1, x2, @out, where);

        /// <summary>
        /// Return (x1 &lt;= x2) element-wise with scalar.
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Scalar or array-like value.</param>
        /// <returns>Output array of bools.</returns>
        public static NDArray<bool> less_equal(NDArray x1, object x2) => x1 <= x2;

        /// <summary>
        /// Return (x1 &lt;= x2) element-wise with scalar on left.
        /// </summary>
        /// <param name="x1">Scalar or array-like value.</param>
        /// <param name="x2">Input array.</param>
        /// <returns>Output array of bools.</returns>
        public static NDArray<bool> less_equal(object x1, NDArray x2) => x1 <= x2;

        #endregion

        #region greater_equal

        /// <summary>
        /// Return (x1 &gt;= x2) element-wise.
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <returns>Output array of bools.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.greater_equal.html</remarks>
        public static NDArray<bool> greater_equal(NDArray x1, NDArray x2) => x1 >= x2;

        /// <summary>
        /// Return (greater_equal) element-wise into <paramref name="@out"/> (NumPy ufunc out=/where=).
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <param name="@out">A location into which the result is stored; any numeric dtype (bool casts same_kind to all of them, True→1); returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written; masked-off out slots keep prior contents.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.greater_equal.html</remarks>
        public static NDArray greater_equal(NDArray x1, NDArray x2, NDArray @out, NDArray where = null)
            => x1.TensorEngine.GreaterEqual(x1, x2, @out, where);

        /// <summary>
        /// Return (x1 &gt;= x2) element-wise with scalar.
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Scalar or array-like value.</param>
        /// <returns>Output array of bools.</returns>
        public static NDArray<bool> greater_equal(NDArray x1, object x2) => x1 >= x2;

        /// <summary>
        /// Return (x1 &gt;= x2) element-wise with scalar on left.
        /// </summary>
        /// <param name="x1">Scalar or array-like value.</param>
        /// <param name="x2">Input array.</param>
        /// <returns>Output array of bools.</returns>
        public static NDArray<bool> greater_equal(object x1, NDArray x2) => x1 >= x2;

        #endregion
    }
}
