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
        public static int count_nonzero(in NDArray a)
        {
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
        public static NDArray count_nonzero(in NDArray a, int axis, bool keepdims = false)
        {
            if (a.size == 0)
            {
                // Handle empty arrays - return zeros with reduced shape
                var shape = a.Shape;
                while (axis < 0)
                    axis = a.ndim + axis;
                if (axis >= a.ndim)
                    throw new ArgumentOutOfRangeException(nameof(axis));

                var resultShape = Shape.GetAxis(shape, axis);
                var result = np.zeros(new Shape(resultShape), NPTypeCode.Int64);
                if (keepdims)
                {
                    var ks = new int[a.ndim];
                    for (int d = 0, sd = 0; d < a.ndim; d++)
                        ks[d] = (d == axis) ? 1 : resultShape[sd++];
                    result.Storage.Reshape(new Shape(ks));
                }
                return result;
            }

            return a.TensorEngine.CountNonZero(a, axis, keepdims);
        }
    }
}
