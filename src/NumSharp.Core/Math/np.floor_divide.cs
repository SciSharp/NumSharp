using System;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// Return the largest integer smaller or equal to the division of the inputs.
        /// It is equivalent to the Python // operator.
        /// </summary>
        /// <param name="x1">Dividend array.</param>
        /// <param name="x2">Divisor array.</param>
        /// <returns>y = floor(x1/x2). This is a scalar if both x1 and x2 are scalars.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.floor_divide.html</remarks>
        public static NDArray floor_divide(NDArray x1, NDArray x2)
            => x1.TensorEngine.FloorDivide(x1, x2);

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
        /// Scalar divisor version.
        /// </summary>
        /// <param name="x1">Dividend array.</param>
        /// <param name="x2">Scalar divisor.</param>
        /// <returns>y = floor(x1/x2).</returns>
        public static NDArray floor_divide(NDArray x1, ValueType x2)
            => x1.TensorEngine.FloorDivide(x1, x2);

        /// <summary>
        /// Return the largest integer smaller or equal to the division of the inputs.
        /// Scalar divisor version.
        /// </summary>
        /// <param name="x1">Dividend array.</param>
        /// <param name="x2">Scalar divisor.</param>
        /// <param name="dtype">The dtype of the returned NDArray.</param>
        /// <returns>y = floor(x1/x2).</returns>
        public static NDArray floor_divide(NDArray x1, ValueType x2, Type dtype)
            => x1.TensorEngine.FloorDivide(x1, x2, dtype);

        /// <summary>
        /// Return the largest integer smaller or equal to the division of the inputs.
        /// Scalar divisor version.
        /// </summary>
        /// <param name="x1">Dividend array.</param>
        /// <param name="x2">Scalar divisor.</param>
        /// <param name="typeCode">The dtype of the returned NDArray.</param>
        /// <returns>y = floor(x1/x2).</returns>
        public static NDArray floor_divide(NDArray x1, ValueType x2, NPTypeCode typeCode)
            => x1.TensorEngine.FloorDivide(x1, x2, typeCode);
    }
}

