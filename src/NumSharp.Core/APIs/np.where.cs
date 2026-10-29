using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Equivalent to <see cref="nonzero(NDArray)"/>: returns the indices where
        ///     <paramref name="condition"/> is non-zero.
        /// </summary>
        /// <param name="condition">Input array. Non-zero entries yield their indices.</param>
        /// <returns>Tuple of arrays with indices where condition is non-zero, one per dimension.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.where.html</remarks>
        public static NDArray<long>[] where(NDArray condition)
        {
            using var plain = NDMatrixBoundary.PlainFor(condition);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return nonzero(condition);
        }

        /// <summary>
        ///     Return elements chosen from `x` or `y` depending on `condition`.
        /// </summary>
        /// <param name="condition">Where True, yield `x`, otherwise yield `y`.</param>
        /// <param name="x">Values from which to choose where condition is True.</param>
        /// <param name="y">Values from which to choose where condition is False.</param>
        /// <returns>An array with elements from `x` where `condition` is True, and elements from `y` elsewhere.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.where.html</remarks>
        public static NDArray where(NDArray condition, NDArray x, NDArray y)
        {
            using var plain = NDMatrixBoundary.PlainFor(condition, x, y);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return NDArray.Where(condition, x, y);
        }

        /// <summary>
        ///     Return elements chosen from `x` or `y` depending on `condition`.
        ///     Scalar overload for x.
        /// </summary>
        [NDScoped]
        public static NDArray where(NDArray condition, object x, NDArray y)
        {
            using var plain = NDMatrixBoundary.PlainFor(condition, y);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return NDArray.Where(condition, NDArray.FromPythonOperand(x), y);
        }

        /// <summary>
        ///     Return elements chosen from `x` or `y` depending on `condition`.
        ///     Scalar overload for y.
        /// </summary>
        [NDScoped]
        public static NDArray where(NDArray condition, NDArray x, object y)
        {
            using var plain = NDMatrixBoundary.PlainFor(condition, x);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return NDArray.Where(condition, x, NDArray.FromPythonOperand(y));
        }

        /// <summary>
        ///     Return elements chosen from `x` or `y` depending on `condition`.
        ///     Scalar overload for both x and y.
        /// </summary>
        [NDScoped]
        public static NDArray where(NDArray condition, object x, object y)
        {
            using var plain = NDMatrixBoundary.PlainFor(condition);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return NDArray.Where(condition, NDArray.FromPythonOperand(x), NDArray.FromPythonOperand(y));
        }
    }
}
