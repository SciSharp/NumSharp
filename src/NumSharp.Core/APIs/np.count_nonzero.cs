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
        /// <returns>Number of non-zero values in the array (a text element counts when non-empty); 0 for an empty
        ///     array.</returns>
        /// <exception cref="NullReferenceException"><paramref name="a"/> is null.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html — the count is the engine's
        ///     (<c>TensorEngine.CountNonZero</c>).</remarks>
        public static long count_nonzero(NDArray a)
        {
            using var plain = NDMatrixBoundary.PlainFor(a);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // NumPy's count_nonzero over every element (multiarray.count_nonzero), text and empty arrays included, is the
            // engine's: DefaultEngine.CountNonZero.
            return a.TensorEngine.CountNonZero(a);
        }

        /// <summary>
        /// Counts the number of non-zero values in the array along the given axis.
        /// </summary>
        /// <param name="a">The array for which to count non-zeros.</param>
        /// <param name="axis">Axis along which to count non-zeros.</param>
        /// <param name="keepdims">If True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns>Number of non-zero values along the specified axis.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds (a 0-d string array accepts 0 and -1).</exception>
        /// <remarks>A string array is counted by its truth values: NumPy's <c>a != ''</c> then the sum (a non-empty string
        ///     counts; never raises).</remarks>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html</remarks>
        public static NDArray count_nonzero(NDArray a, int axis, bool keepdims = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(a);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // NumPy's count_nonzero(a, axis, keepdims=) is `a_bool.sum(axis=axis, dtype=intp, keepdims=keepdims)` over the
            // truth values (text through a != ''); the whole call, the empty array included, is the engine's.
            return a.TensorEngine.CountNonZero(a, axis, keepdims);
        }
    }
}
