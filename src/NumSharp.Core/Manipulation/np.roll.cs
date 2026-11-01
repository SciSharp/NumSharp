using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Roll array elements along a given axis.
        ///
        /// Elements that roll beyond the last position are re-introduced at the first.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="shift">The number of places by which elements are shifted.</param>
        /// <param name="axis">Axis along which elements are shifted. By default, the array
        /// is flattened before shifting, after which the original shape is restored.</param>
        /// <returns>Output array, with the same shape as <paramref name="a"/>.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <remarks>
        /// Matches NumPy's algorithm: empty_like + slice-copy pairs (the array object's <c>NDArray.roll</c>, which releases
        /// its intermediates).
        /// https://numpy.org/doc/stable/reference/generated/numpy.roll.html
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for <paramref name="a"/>'s rank.</exception>
        /// <exception cref="NullReferenceException"><paramref name="a"/> is null.</exception>
        public static NDArray roll(NDArray a, long shift, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, shift, axis), static s => roll(s.a, s.shift, s.axis));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.Kept((a, shift, axis), static s => roll(s.a, s.shift, s.axis));

            // NumPy's statements (the flat roll, or empty_like and the two slice copies along the axis) are the array
            // object's.
            return NDArray.roll(a, shift, axis);
        }

        /// <summary>
        /// Roll array elements along a given axis; the <see cref="int"/> shift of <see cref="roll(NDArray, long, int?)"/>.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="shift">The number of places by which elements are shifted.</param>
        /// <param name="axis">Axis along which elements are shifted, or null to roll the flattened array.</param>
        /// <returns>Output array, with the same shape as <paramref name="a"/> (see <see cref="roll(NDArray, long, int?)"/>).</returns>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for <paramref name="a"/>'s rank.</exception>
        /// <exception cref="NullReferenceException"><paramref name="a"/> is null.</exception>
        public static NDArray roll(NDArray a, int shift, int? axis = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((a, shift, axis), static s => roll(s.a, s.shift, s.axis))
            : NDCharArrayBoundary.Sees(a)   // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            ? NDCharArrayBoundary.Kept((a, shift, axis), static s => roll(s.a, s.shift, s.axis))
            : NDArray.roll(a, (long)shift, axis);   // the array object's statements, the shift widened
    }
}
