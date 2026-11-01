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
        /// <returns>A view of <paramref name="a"/> with the new axis.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <c>[-(a.ndim + 1), a.ndim]</c>: "axis … is out of
        ///     bounds for array of dimension {a.ndim + 1}", the axis as given.</exception>
        public static NDArray expand_dims(NDArray a, int axis)
        {
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.Kept((a, axis), static s => expand_dims(s.a, s.axis));
            using var plain = NDMatrixBoundary.PlainFor(a);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return NDArray.expand_dims(a, axis);
        }

        /// <summary>
        ///     Expand the shape of an array. Insert new axes that will appear at
        ///     the <paramref name="axis"/> positions in the expanded output.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="axis">The positions of the new axes in the RESULT, which has rank <c>a.ndim + axis.Length</c>;
        ///     negative values count from the end of the result. An empty array inserts nothing.</param>
        /// <returns>A view of <paramref name="a"/> with the new axes (a fresh view object, the empty tuple included —
        ///     NumPy's <c>a.reshape(shape)</c>).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <remarks>
        ///     Matches NumPy 2.x: each axis in the tuple is normalized against
        ///     the FINAL output ndim (<c>a.ndim + axis.Length</c>) by
        ///     <c>normalize_axis_tuple</c>, every range check before the repeat check.
        /// </remarks>
        /// <exception cref="AxisError">An axis is out of range for the result's rank.</exception>
        /// <exception cref="ValueError">Two axes name the same position ("repeated axis").</exception>
        public static NDArray expand_dims(NDArray a, int[] axis)
        {
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.Kept((a, axis), static s => expand_dims(s.a, s.axis));
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
        /// <returns>A view of <paramref name="a"/> with the new axes (<paramref name="a"/> itself for a null sequence).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <exception cref="AxisError">An axis is out of range for the result's rank.</exception>
        /// <exception cref="ValueError">Two axes name the same position ("repeated axis").</exception>
        public static NDArray expand_dims(NDArray a, IEnumerable<int> axis)
        {
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.Kept((a, axis), static s => expand_dims(s.a, s.axis));
            using var plain = NDMatrixBoundary.PlainFor(a);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (axis == null)
                return a;

            return expand_dims(a, axis as int[] ?? axis.ToArray());
        }
    }
}
