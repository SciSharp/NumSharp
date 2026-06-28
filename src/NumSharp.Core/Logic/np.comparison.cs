using NumSharp.Backends;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        #region equal

        /// <summary>
        /// Return (x1 == x2) element-wise.
        /// Mirrors NumPy's ufunc signature: <c>equal(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// A plain call returns a bool-dtype array (the instance is an <see cref="NDArray{T}"/> of bool —
        /// cast or use the <c>==</c> operator for the typed wrapper).
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <param name="@out">A location into which the result is stored; any numeric dtype (bool casts same_kind to all of them, True→1); returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written; masked-off out slots keep prior contents.</param>
        /// <param name="dtype">Validate-only (NumPy parity): comparisons have bool loops only — any non-bool request raises the no-loop TypeError.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.equal.html</remarks>
        public static NDArray equal(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x1.TensorEngine.Compare(x1, x2, dtype, @out, where);

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
        /// Mirrors NumPy's ufunc signature: <c>not_equal(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// A plain call returns a bool-dtype array (the instance is an <see cref="NDArray{T}"/> of bool —
        /// cast or use the <c>!=</c> operator for the typed wrapper).
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <param name="@out">A location into which the result is stored; any numeric dtype (bool casts same_kind to all of them, True→1); returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written; masked-off out slots keep prior contents.</param>
        /// <param name="dtype">Validate-only (NumPy parity): comparisons have bool loops only — any non-bool request raises the no-loop TypeError.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.not_equal.html</remarks>
        public static NDArray not_equal(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x1.TensorEngine.NotEqual(x1, x2, dtype, @out, where);

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
        /// Mirrors NumPy's ufunc signature: <c>less(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// A plain call returns a bool-dtype array (the instance is an <see cref="NDArray{T}"/> of bool —
        /// cast or use the <c>&lt;</c> operator for the typed wrapper).
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <param name="@out">A location into which the result is stored; any numeric dtype (bool casts same_kind to all of them, True→1); returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written; masked-off out slots keep prior contents.</param>
        /// <param name="dtype">Validate-only (NumPy parity): comparisons have bool loops only — any non-bool request raises the no-loop TypeError.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.less.html</remarks>
        public static NDArray less(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x1.TensorEngine.Less(x1, x2, dtype, @out, where);

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
        /// Mirrors NumPy's ufunc signature: <c>greater(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// A plain call returns a bool-dtype array (the instance is an <see cref="NDArray{T}"/> of bool —
        /// cast or use the <c>&gt;</c> operator for the typed wrapper).
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <param name="@out">A location into which the result is stored; any numeric dtype (bool casts same_kind to all of them, True→1); returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written; masked-off out slots keep prior contents.</param>
        /// <param name="dtype">Validate-only (NumPy parity): comparisons have bool loops only — any non-bool request raises the no-loop TypeError.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.greater.html</remarks>
        public static NDArray greater(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x1.TensorEngine.Greater(x1, x2, dtype, @out, where);

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
        /// Mirrors NumPy's ufunc signature: <c>less_equal(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// A plain call returns a bool-dtype array (the instance is an <see cref="NDArray{T}"/> of bool —
        /// cast or use the <c>&lt;=</c> operator for the typed wrapper).
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <param name="@out">A location into which the result is stored; any numeric dtype (bool casts same_kind to all of them, True→1); returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written; masked-off out slots keep prior contents.</param>
        /// <param name="dtype">Validate-only (NumPy parity): comparisons have bool loops only — any non-bool request raises the no-loop TypeError.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.less_equal.html</remarks>
        public static NDArray less_equal(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x1.TensorEngine.LessEqual(x1, x2, dtype, @out, where);

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
        /// Mirrors NumPy's ufunc signature: <c>greater_equal(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// A plain call returns a bool-dtype array (the instance is an <see cref="NDArray{T}"/> of bool —
        /// cast or use the <c>&gt;=</c> operator for the typed wrapper).
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="x2">Input array.</param>
        /// <param name="@out">A location into which the result is stored; any numeric dtype (bool casts same_kind to all of them, True→1); returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written; masked-off out slots keep prior contents.</param>
        /// <param name="dtype">Validate-only (NumPy parity): comparisons have bool loops only — any non-bool request raises the no-loop TypeError.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.greater_equal.html</remarks>
        public static NDArray greater_equal(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x1.TensorEngine.GreaterEqual(x1, x2, dtype, @out, where);

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
