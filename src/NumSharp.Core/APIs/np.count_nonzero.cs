using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Counts the number of non-zero values in the array.
        /// </summary>
        /// <param name="a">The array for which to count non-zeros.</param>
        /// <returns>Number of non-zero values in the array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html</remarks>
        public static long count_nonzero(NDArray a)
        {
            using var plain = NDMatrixBoundary.PlainFor(a);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (a.size == 0)
                return 0;

            return a.TensorEngine.CountNonZero(a);
        }

        /// <summary>
        /// Counts the number of non-zero values in the array along the given axis.
        /// </summary>
        /// <param name="a">The array for which to count non-zeros.</param>
        /// <param name="axis">Axis along which to count non-zeros.</param>
        /// <param name="keepdims">If True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns>Number of non-zero values along the specified axis.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html</remarks>
        public static NDArray count_nonzero(NDArray a, int axis, bool keepdims = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(a);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // Text: NumPy compares every element with '' (the next phase) and then sums with the axis, so an empty or
            // non-empty string array reports a bad axis first (AxisError, probed).
            Backends.Strings.NDStringUfuncs.ThrowIfReduce("count_nonzero", a, axis);
            if (a.size == 0)
            {
                // Handle empty arrays - return zeros with reduced shape. The axis is validated as NumPy's sum validates
                // it: an AxisError naming the axis as given (a wrapping loop used to turn -3 on a 1-D array into axis 0).
                var shape = a.Shape;
                if (axis < -a.ndim || axis >= a.ndim)
                    throw new AxisError(axis, a.ndim);
                if (axis < 0)
                    axis += a.ndim;

                var resultShape = Shape.GetAxis(shape, axis);
                var result = np.zeros(Shape.Adopt(resultShape), NPTypeCode.Int64);
                if (keepdims)
                {
                    var ks = new long[a.ndim];
                    for (int d = 0, sd = 0; d < a.ndim; d++)
                        ks[d] = (d == axis) ? 1 : resultShape[sd++];
                    result.Storage.Reshape(Shape.Adopt(ks));
                }
                return result;
            }

            return a.TensorEngine.CountNonZero(a, axis, keepdims);
        }
    }
}
