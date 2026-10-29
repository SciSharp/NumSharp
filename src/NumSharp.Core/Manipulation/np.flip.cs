using System;
using System.Collections.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>fliplr's axis tuple — m[:, ::-1] == np.flip(m, axis: 1).</summary>
        private static readonly int[] FlipAxis1 = {1};

        /// <summary>flipud's axis tuple — m[::-1, ...] == np.flip(m, axis: 0).</summary>
        private static readonly int[] FlipAxis0 = {0};

        /// <summary>
        ///     Reverse the order of elements in an array along the given axis.
        ///     The shape of the array is preserved, but the elements are reordered.
        /// </summary>
        /// <param name="m">Input array.</param>
        /// <param name="axis">
        ///     Axis along which to flip over. The default, null, will flip over all of the axes of the
        ///     input array. If axis is negative it counts from the last to the first axis.
        /// </param>
        /// <returns>
        ///     A view of <paramref name="m"/> with the entries of axis reversed. Since a view is returned,
        ///     this operation is done in constant time. A <c>numpy.char.chararray</c> operand gives a chararray view, and a
        ///     0-d one its element: the read-only stand-in of NumPy's Python <c>str</c> / <c>bytes</c>, stripped.
        /// </returns>
        /// <exception cref="AxisError">When <paramref name="axis"/> is out of bounds for the array's dimensions.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.flip.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray flip(NDArray m, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(m))
                return NDMatrixBoundary.Wrapped((m, axis), static s => flip(s.m, s.axis));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary): flip ends with
            // m[indexer], chararray.__getitem__, so a 0-d operand's result is its element, stripped
            if (NDCharArrayBoundary.Sees(m))
                return NDCharArrayBoundary.KeptIndexed((m, axis), static s => flip(s.m, s.axis));
            if (m is null)
                throw new ArgumentNullException(nameof(m));

            if (axis == null)
                return NDArray.FlipAxes(m, null);

            // normalize_axis_index: AxisError reports the ORIGINAL axis value.
            int ndim = m.ndim;
            int ax = axis.Value;
            int adjusted = ax >= 0 ? ax : ndim + ax;
            if (adjusted < 0 || adjusted >= ndim)
                throw new AxisError(ax, ndim);

            return NDArray.FlipAxes(m, new[] {adjusted});
        }

        /// <summary>
        ///     Reverse the order of elements in an array along the given axes.
        ///     The shape of the array is preserved, but the elements are reordered.
        /// </summary>
        /// <param name="m">Input array.</param>
        /// <param name="axis">
        ///     Axes along which to flip over — NumPy's tuple-of-ints form; flipping is performed on all of
        ///     the specified axes. Negative axes count from the last to the first axis. An empty array
        ///     flips no axis (returns an unreversed view of the whole array); null flips all axes.
        /// </param>
        /// <returns>
        ///     A view of <paramref name="m"/> with the entries of the given axes reversed. Since a view is
        ///     returned, this operation is done in constant time. A <c>numpy.char.chararray</c> operand gives a chararray
        ///     view, and a 0-d one its element: the read-only stand-in of NumPy's Python <c>str</c> / <c>bytes</c>, stripped.
        /// </returns>
        /// <exception cref="AxisError">When any axis is out of bounds for the array's dimensions.</exception>
        /// <exception cref="ValueError">When an axis is repeated ("repeated axis").</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.flip.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray flip(NDArray m, int[] axis)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(m))
                return NDMatrixBoundary.Wrapped((m, axis), static s => flip(s.m, s.axis));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary): flip ends with
            // m[indexer], chararray.__getitem__, so a 0-d operand's result is its element, stripped
            if (NDCharArrayBoundary.Sees(m))
                return NDCharArrayBoundary.KeptIndexed((m, axis), static s => flip(s.m, s.axis));
            if (m is null)
                throw new ArgumentNullException(nameof(m));

            if (axis == null)
                return NDArray.FlipAxes(m, null);

            int ndim = m.ndim;

            // normalize_axis_tuple: normalize the whole tuple first (AxisError reports the ORIGINAL
            // axis value), only then reject duplicates — NumPy raises AxisError for (0, 0, 5).
            var normalized = new int[axis.Length];
            for (int i = 0; i < axis.Length; i++)
            {
                int ax = axis[i];
                int adjusted = ax >= 0 ? ax : ndim + ax;
                if (adjusted < 0 || adjusted >= ndim)
                    throw new AxisError(ax, ndim);
                normalized[i] = adjusted;
            }

            var seen = new HashSet<int>();
            for (int i = 0; i < normalized.Length; i++)
                if (!seen.Add(normalized[i]))
                    throw new ValueError("repeated axis");

            return NDArray.FlipAxes(m, normalized);
        }

        /// <summary>
        ///     Reverse the order of elements along axis 1 (left/right).
        ///     For a 2-D array, this flips the entries in each row in the left/right direction.
        ///     Columns are preserved, but appear in a different order than before.
        /// </summary>
        /// <param name="m">Input array, must be at least 2-D.</param>
        /// <returns>
        ///     A view of <paramref name="m"/> with the columns reversed — equivalent to m[:, ::-1]
        ///     or np.flip(m, axis: 1). Since a view is returned, this operation is done in constant time.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <exception cref="ValueError">When <paramref name="m"/> is less than 2-d ("Input must be &gt;= 2-d.").</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fliplr.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray fliplr(NDArray m)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(m))
                return NDMatrixBoundary.Wrapped(m, static s => fliplr(s));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(m))
                return NDCharArrayBoundary.Kept(m, static s => fliplr(s));
            if (m is null)
                throw new ArgumentNullException(nameof(m));

            if (m.ndim < 2)
                throw new ValueError("Input must be >= 2-d.");

            return NDArray.FlipAxes(m, FlipAxis1);
        }

        /// <summary>
        ///     Reverse the order of elements along axis 0 (up/down).
        ///     For a 2-D array, this flips the entries in each column in the up/down direction.
        ///     Rows are preserved, but appear in a different order than before.
        /// </summary>
        /// <param name="m">Input array, must be at least 1-D.</param>
        /// <returns>
        ///     A view of <paramref name="m"/> with the rows reversed — equivalent to m[::-1, ...]
        ///     or np.flip(m, axis: 0). Since a view is returned, this operation is done in constant time.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <exception cref="ValueError">When <paramref name="m"/> is less than 1-d ("Input must be &gt;= 1-d.").</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.flipud.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray flipud(NDArray m)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(m))
                return NDMatrixBoundary.Wrapped(m, static s => flipud(s));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(m))
                return NDCharArrayBoundary.Kept(m, static s => flipud(s));
            if (m is null)
                throw new ArgumentNullException(nameof(m));

            if (m.ndim < 1)
                throw new ValueError("Input must be >= 1-d.");

            return NDArray.FlipAxes(m, FlipAxis0);
        }
    }
}
