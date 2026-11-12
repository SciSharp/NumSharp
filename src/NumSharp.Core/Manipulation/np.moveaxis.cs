using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     NumPy's <c>np.moveaxis(a, source, destination)</c>: a view of <paramref name="a"/> with the axes of
        ///     <paramref name="source"/> moved to the positions of <paramref name="destination"/>; the other axes keep their
        ///     order.
        /// </summary>
        /// <param name="a">The array whose axes are reordered.</param>
        /// <param name="source">The original positions of the axes to move: one axis or a tuple (<c>(0, 1)</c> or
        ///     <c>new[] { 0, 1 }</c>), each distinct; negative positions count from the end.</param>
        /// <param name="destination">The new positions, one per <paramref name="source"/> axis, each distinct.</param>
        /// <returns>A view of <paramref name="a"/> with the moved axes (NumPy's <c>a.transpose(order)</c>): it shares memory
        ///     with <paramref name="a"/>. A <c>numpy.matrix</c> or <c>numpy.char.chararray</c> operand gives its class over
        ///     the result (NumPy creates it as the subclass), owning its data exactly when the plain result would.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null, or an axis argument was converted from a
        ///     null <c>int[]</c> (a non-nullable <see cref="NDAxis"/> has no None).</exception>
        /// <exception cref="AxisError">An axis is out of range: <c>source: axis 3 is out of bounds for array of dimension
        ///     3</c> (NumPy names the argument; <paramref name="source"/> is checked whole before
        ///     <paramref name="destination"/>).</exception>
        /// <exception cref="ValueError">An argument repeats an axis (<c>repeated axis in `source` argument</c>), the two
        ///     have different lengths (<c>`source` and `destination` arguments must have the same number of
        ///     elements</c>, checked after both are normalized), or a <c>numpy.matrix</c> operand's result cannot be a matrix
        ///     (more than two axes longer than 1: <c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html. NumPy's statements (each argument
        ///     through <c>normalize_axis_tuple(..., 'source')</c> / <c>'destination'</c>, the length check, the order, the
        ///     transpose) are the engine's <see cref="TensorEngine.MoveAxis"/>.
        ///     <para>One signature since API-parity batch 4 (2026-11-12): it replaces the four <c>int</c> / <c>int[]</c>
        ///     overloads, which had no tuple literal.</para>
        /// </remarks>
        public static NDArray moveaxis(NDArray a, NDAxis source, NDAxis destination)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // A single axis is NumPy's one-element tuple here (normalize_axis_tuple wraps an int), so both arguments reach
            // the engine as arrays, unnormalized: it reports each with NumPy's argument name.
            return a.TensorEngine.MoveAxis(a, source.ToArray(), destination.ToArray());
        }
    }
}
