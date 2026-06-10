using System;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// Return the largest integer smaller or equal to the division of the inputs.
        /// It is equivalent to the Python // operator.
        /// Mirrors NumPy's ufunc signature: <c>floor_divide(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x1">Dividend array.</param>
        /// <param name="x2">Divisor array.</param>
        /// <param name="@out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the computation runs in this dtype; inputs must be same_kind-castable to it.</param>
        /// <returns>y = floor(x1/x2). This is a scalar if both x1 and x2 are scalars.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.floor_divide.html</remarks>
        public static NDArray floor_divide(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x1.TensorEngine.FloorDivide(x1, x2, dtype, @out, where);

        /// <summary>
        /// Return the largest integer smaller or equal to the division of the inputs.
        /// It is equivalent to the Python // operator.
        /// </summary>
        /// <param name="x1">Dividend array.</param>
        /// <param name="x2">Divisor array.</param>
        /// <param name="dtype">The dtype of the returned NDArray.</param>
        /// <returns>y = floor(x1/x2). This is a scalar if both x1 and x2 are scalars.</returns>
        public static NDArray floor_divide(NDArray x1, NDArray x2, Type dtype)
            => x1.TensorEngine.FloorDivide(x1, x2, dtype);

        /// <summary>
        /// Return the largest integer smaller or equal to the division of the inputs.
        /// It is equivalent to the Python // operator.
        /// </summary>
        /// <param name="x1">Dividend array.</param>
        /// <param name="x2">Divisor array.</param>
        /// <param name="typeCode">The dtype of the returned NDArray.</param>
        /// <returns>y = floor(x1/x2). This is a scalar if both x1 and x2 are scalars.</returns>
        public static NDArray floor_divide(NDArray x1, NDArray x2, NPTypeCode typeCode)
            => x1.TensorEngine.FloorDivide(x1, x2, typeCode);

        /// <summary>
        /// Return the largest integer smaller or equal to the division of the inputs.
        /// Scalar or array-like divisor version.
        /// </summary>
        /// <param name="x1">Dividend array.</param>
        /// <param name="x2">Scalar or array-like divisor.</param>
        /// <returns>y = floor(x1/x2).</returns>
        public static NDArray floor_divide(NDArray x1, object x2)
            => x1.TensorEngine.FloorDivide(x1, np.asanyarray(x2));

        /// <summary>
        /// Return the largest integer smaller or equal to the division of the inputs.
        /// Scalar or array-like divisor version.
        /// </summary>
        /// <param name="x1">Dividend array.</param>
        /// <param name="x2">Scalar or array-like divisor.</param>
        /// <param name="dtype">The dtype of the returned NDArray.</param>
        /// <returns>y = floor(x1/x2).</returns>
        public static NDArray floor_divide(NDArray x1, object x2, Type dtype)
            => x1.TensorEngine.FloorDivide(x1, np.asanyarray(x2), dtype);

        /// <summary>
        /// Return the largest integer smaller or equal to the division of the inputs.
        /// Scalar or array-like divisor version.
        /// </summary>
        /// <param name="x1">Dividend array.</param>
        /// <param name="x2">Scalar or array-like divisor.</param>
        /// <param name="typeCode">The dtype of the returned NDArray.</param>
        /// <returns>y = floor(x1/x2).</returns>
        public static NDArray floor_divide(NDArray x1, object x2, NPTypeCode typeCode)
            => x1.TensorEngine.FloorDivide(x1, np.asanyarray(x2), typeCode);
    }
}

