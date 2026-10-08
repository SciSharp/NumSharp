using System;
using System.Collections.Generic;
using System.Linq;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     NumPy's <c>np.expand_dims(a, axis)</c>: a view of <paramref name="a"/> with a length-one axis inserted at
        ///     position <paramref name="axis"/> of the result.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="axis">The position of the new axis in the RESULT, which has rank <c>a.ndim + 1</c>; negative values
        ///     count from the end of the result.</param>
        /// <returns>A view of <paramref name="a"/> with the new axis.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <c>[-(a.ndim + 1), a.ndim]</c>: "axis … is out of
        ///     bounds for array of dimension {a.ndim + 1}", the axis as given.</exception>
        public static NDArray expand_dims(NDArray a, int axis)
        {
            using var plain = NDMatrixBoundary.PlainFor(a);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // Only the uninitialized-shape sentinel is passed through. A genuine empty array
            // (size == 0 but with real dimensions, e.g. (0, 3)) MUST still gain the new axis —
            // NumPy: expand_dims((0, 3), 0) -> (1, 0, 3). Guarding on a.size == 0 here made
            // expand_dims a no-op for every empty array (and, transitively, broke np.stack and
            // np.atleast_2d / np.vstack, which build on it).
            if (a.Shape.IsEmpty)
                return a;

            // NumPy normalizes the axis against the OUTPUT ndim (a.ndim + 1) and reports the axis
            // AS GIVEN. Shape.ExpandDimension guards only the negative side, and does it against
            // the already-normalized value, so all three of these were wrong:
            //   expand_dims(arange(6), 5)  -> ArgumentOutOfRangeException "... (Parameter 'index')"
            //                                 leaking out of Arrays.Insert, three frames down
            //   expand_dims(arange(6), -3) -> "Effective axis -1 is less than 0" (the NORMALIZED
            //                                 axis, not the -3 the caller passed)
            //   expand_dims(scalar, 5)     -> silently returned shape (1); the 0-d branch never
            //                                 looked at the axis at all
            // NumPy answers all three with: axis N is out of bounds for array of dimension {outNdim}.
            //
            // The guard lives here rather than in Shape.ExpandDimension because that method has
            // internal callers (keepdims reductions, fancy-index shaping, asmatrix) that are not
            // bound by NumPy's axis contract. The sibling int[] overload already validates this way,
            // and this reproduces its text exactly so both overloads agree.
            // normalize_axis_tuple((axis,), out_ndim) raises NumPy's AxisError; the range test runs on the axis as given.
            int outNdim = a.ndim + 1;
            Backends.DefaultEngine.check_and_adjust_axis(outNdim, axis);

            return new NDArray(a.Storage.Alias(a.Shape.ExpandDimension(axis))) { TensorEngine = a.TensorEngine };
        }

        /// <summary>
        ///     Expand the shape of an array. Insert new axes that will appear at
        ///     the <paramref name="axis"/> positions in the expanded output.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="axis">The positions of the new axes in the RESULT, which has rank <c>a.ndim + axis.Length</c>;
        ///     negative values count from the end of the result. An empty array inserts nothing.</param>
        /// <returns>A view of <paramref name="a"/> with the new axes (a fresh view object, the empty tuple included —
        ///     NumPy's <c>a.reshape(shape)</c>).</returns>
        /// <remarks>
        ///     Matches NumPy 2.x: each axis in the tuple is normalized against
        ///     the FINAL output ndim (<c>a.ndim + axis.Length</c>) by
        ///     <c>normalize_axis_tuple</c>, every range check before the repeat check.
        /// </remarks>
        /// <exception cref="AxisError">An axis is out of range for the result's rank.</exception>
        /// <exception cref="ValueError">Two axes name the same position ("repeated axis").</exception>
        public static NDArray expand_dims(NDArray a, int[] axis)
        {
            using var plain = NDMatrixBoundary.PlainFor(a);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // An empty tuple reshapes to the same shape: a NEW view object in NumPy, never the input itself (a caller
            // that disposes or resizes the result must not touch the input).
            if (axis == null || axis.Length == 0)
                return new NDArray(a.Storage.Alias()) { TensorEngine = a.TensorEngine };

            // See the single-axis overload: a size == 0 array must still gain the new axes.
            if (a.Shape.IsEmpty)
                return a;

            return new NDArray(a.Storage.Alias(a.Shape.ExpandDimensions(axis))) { TensorEngine = a.TensorEngine };
        }

        /// <summary>
        ///     Sequence overload — accepts any <see cref="IEnumerable{Int32}"/>,
        ///     materializes to an array, and delegates to the tuple-axis path.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="axis">The positions of the new axes in the result, or null for none.</param>
        /// <returns>A view of <paramref name="a"/> with the new axes (<paramref name="a"/> itself for a null sequence).</returns>
        /// <exception cref="AxisError">An axis is out of range for the result's rank.</exception>
        /// <exception cref="ValueError">Two axes name the same position ("repeated axis").</exception>
        public static NDArray expand_dims(NDArray a, IEnumerable<int> axis)
        {
            using var plain = NDMatrixBoundary.PlainFor(a);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (axis == null)
                return a;

            return expand_dims(a, axis as int[] ?? axis.ToArray());
        }
    }
}
