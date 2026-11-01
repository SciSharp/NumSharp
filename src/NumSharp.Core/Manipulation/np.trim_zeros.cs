using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Remove values along a dimension which are zero along all other dimensions. Mirrors NumPy's
        ///     <c>numpy.trim_zeros(filt, trim='fb', axis=None)</c> exactly: <paramref name="axis"/> is a single
        ///     axis, or <c>null</c> for the whole-array bounding box. Pass an <c>int[]</c> to trim several axes.
        /// </summary>
        /// <param name="filt">Input array.</param>
        /// <param name="trim">'f' trims from the front, 'b' from the back; "fb" (default) trims both. Case-insensitive.</param>
        /// <param name="axis">The single dimension to trim; <c>null</c> trims the whole-array bounding box.</param>
        /// <returns>A view of <paramref name="filt"/> (see the <c>int[]</c> overload).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <exception cref="ValueError"><paramref name="trim"/> contains unexpected characters.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="filt"/> ("axis: axis … is
        ///     out of bounds for array of dimension …", NumPy names the argument).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.trim_zeros.html. A string array is trimmed by its
        ///     truth values (an empty string is a zero), as NumPy's <c>argwhere</c> reads them.</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray trim_zeros(NDArray filt, string trim = "fb", int? axis = null)
            => NDMatrixBoundary.Sees(filt)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((filt, trim, axis), static s => trim_zeros(s.filt, s.trim, s.axis))
            : NDCharArrayBoundary.Sees(filt)   // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            ? NDCharArrayBoundary.Kept((filt, trim, axis), static s => trim_zeros(s.filt, s.trim, s.axis))
            : trim_zeros(filt, trim, axis.HasValue ? new[] {axis.Value} : null);

        /// <summary>
        ///     Remove values which are zero along all other dimensions, trimming the given sequence of axes.
        ///     <paramref name="trim"/> is required on this overload so it does not collide with the single-axis
        ///     primary above on calls like <c>trim_zeros(filt)</c> / <c>trim_zeros(filt, "f")</c>.
        /// </summary>
        /// <param name="filt">Input array.</param>
        /// <param name="trim">
        ///     A string with 'f' representing trim from front and 'b' to trim from back. By default,
        ///     zeros are trimmed on both sides ("fb"). Case-insensitive; "bf" is accepted as an alias of "fb".
        /// </param>
        /// <param name="axis">
        ///     If <c>null</c>, <paramref name="filt"/> is cropped to the smallest bounding box that still
        ///     contains all non-zero values. If axes are specified, <paramref name="filt"/> is sliced in
        ///     those dimensions only, on the sides selected by <paramref name="trim"/>. An empty array of
        ///     axes leaves the input unmodified.
        /// </param>
        /// <returns>
        ///     A view of <paramref name="filt"/> with leading/trailing all-zero hyperplanes removed. The number
        ///     of dimensions and the input dtype are preserved.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <exception cref="ValueError">If <paramref name="trim"/> contains unexpected characters, or an axis is repeated
        ///     (NumPy's ValueError texts).</exception>
        /// <exception cref="AxisError">If an axis is out of bounds for <paramref name="filt"/> (checked for every axis before
        ///     the repeat check).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.trim_zeros.html. A string array is trimmed by its
        ///     truth values (an empty string — all NULs — is a zero): NumPy's <c>_arg_trim_zeros</c> runs <c>np.argwhere</c>,
        ///     the dtype's nonzero route, which never raises; the result is a view of the text itself.</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="filt"/> is null.</exception>
        [NDScoped]
        public static NDArray trim_zeros(NDArray filt, string trim, int[] axis)
        {
            if (filt is null) throw new ArgumentNullException(nameof(filt));
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(filt))
                return NDMatrixBoundary.Wrapped((filt, trim, axis), static s => trim_zeros(s.filt, s.trim, s.axis));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(filt))
                return NDCharArrayBoundary.Kept((filt, trim, axis), static s => trim_zeros(s.filt, s.trim, s.axis));
            return filt.TensorEngine.TrimZeros(filt, trim, axis);
        }
    }
}
