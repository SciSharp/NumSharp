using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Rotate an array by 90 degrees in the plane specified by axes.
        ///
        ///     Rotation direction is from the first towards the second axis. This means that
        ///     for a 2-D array with the default <paramref name="k"/> and <paramref name="axes"/>
        ///     the rotation will be counterclockwise.
        /// </summary>
        /// <param name="m">Array of two or more dimensions.</param>
        /// <param name="k">Number of times the array is rotated by 90 degrees.</param>
        /// <param name="axes">
        ///     The array is rotated in the plane defined by the axes. Axes must be different.
        ///     Defaults to <c>(0, 1)</c>.
        /// </param>
        /// <returns>A rotated view of <paramref name="m"/>.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <remarks>
        ///     Port of NumPy's <c>numpy.rot90</c> (<c>numpy/lib/_function_base_impl.py</c>): a pure
        ///     composition of axis flips and a transpose, so the result is always a view that shares
        ///     memory with <paramref name="m"/> (read-only when the source is, e.g. a broadcast view).
        ///     <para><c>rot90(m, k=1, axes=(1, 0))</c> is the reverse of <c>rot90(m, k=1, axes=(0, 1))</c>.</para>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.rot90.html
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ValueError"><paramref name="axes"/> does not hold two axes ("len(axes) must be 2."), names one
        ///     axis twice ("Axes must be different.", tested before the range), or is out of range ("Axes=(a, b) out of range
        ///     for array of ndim=n.") — NumPy's own ValueErrors, not normalize_axis_index's AxisError.</exception>
        /// <exception cref="NullReferenceException"><paramref name="m"/> is null (its rank is read after the axes'
        ///     length).</exception>
        public static NDArray rot90(NDArray m, int k = 1, int[] axes = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(m))
                return NDMatrixBoundary.Wrapped((m, k, axes), static s => rot90(s.m, s.k, s.axes));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(m))
                return NDCharArrayBoundary.Kept((m, k, axes), static s => rot90(s.m, s.k, s.axes));

            // NumPy's statements (the axes' checks, k modulo 4, the flips and the transpose) are the array object's.
            return NDArray.rot90(m, k, axes);
        }
    }
}
