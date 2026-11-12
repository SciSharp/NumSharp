using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>fliplr's axis tuple — m[:, ::-1] == np.flip(m, axis: 1).</summary>
        private static readonly int[] FlipAxis1 = {1};

        /// <summary>flipud's axis tuple — m[::-1, ...] == np.flip(m, axis: 0).</summary>
        private static readonly int[] FlipAxis0 = {0};

        /// <summary>
        ///     NumPy's <c>np.flip(m, axis=None)</c>: a view of <paramref name="m"/> with the order of the elements reversed
        ///     along the given axes (every axis by default). The shape is preserved; only the strides and the start of the
        ///     view change, so the call is constant time whatever the size.
        /// </summary>
        /// <param name="m">Input array.</param>
        /// <param name="axis">
        ///     NumPy's <c>axis</c>: <c>null</c> (None, the default) flips every axis; one axis (<c>0</c>, <c>-1</c>) flips
        ///     that axis; a tuple (<c>(0, 2)</c> or <c>new[] { 0, 2 }</c>) flips each of its axes, and the empty tuple flips
        ///     none (an unreversed view). Negative axes count from the end.
        /// </param>
        /// <returns>
        ///     A view of <paramref name="m"/> with the entries of the selected axes reversed: it shares memory with
        ///     <paramref name="m"/> (a write through it changes <paramref name="m"/>) and is read-only when
        ///     <paramref name="m"/> is. A <c>numpy.char.chararray</c> operand gives a chararray view, and a 0-d one its
        ///     element: the read-only stand-in of NumPy's Python <c>str</c> / <c>bytes</c>, stripped.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="m"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of bounds for <paramref name="m"/> (the first one, reported as
        ///     given; every axis is range-checked before the repeat check, so <c>(0, 0, 5)</c> reports axis 5). A 0-d
        ///     <paramref name="m"/> has no axis to name.</exception>
        /// <exception cref="ValueError">An axis is named twice (<c>repeated axis</c>); a <c>numpy.matrix</c> operand
        ///     whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a
        ///     matrix.</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.flip.html. NumPy's statements:
        ///     <c>axis = normalize_axis_tuple(axis, m.ndim)</c>, then <c>m[indexer]</c> with <c>::-1</c> on those axes.
        ///     <para>One signature since API-parity batch 4 (2026-11-12): the <c>int?</c> and <c>int[]</c> overloads it
        ///     replaces made <c>np.flip(m, null)</c> ambiguous (CS0121) and had no tuple literal; both spellings convert to
        ///     <see cref="NDAxis"/>.</para>
        /// </remarks>
        public static NDArray flip(NDArray m, NDAxis? axis = null)
        {
            if (m is null)
                throw new ArgumentNullException(nameof(m));

            // axis=None: every axis reversed (the view builder reverses all axes for a null list).
            if (axis is null)
                return NDArray.FlipAxes(m, null);

            // normalize_axis_tuple(axis, m.ndim): every axis' range first (the AxisError names the axis as given, so
            // (0, 0, 5) reports 5), then the repeat check; an int axis is the one-element tuple.
            return NDArray.FlipAxes(m, axis.Value.Normalize(m.ndim));
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
            if (m is null)
                throw new ArgumentNullException(nameof(m));

            if (m.ndim < 1)
                throw new ValueError("Input must be >= 1-d.");

            return NDArray.FlipAxes(m, FlipAxis0);
        }
    }
}
