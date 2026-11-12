using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     NumPy's <c>np.trim_zeros(filt, trim='fb', axis=None)</c>: remove the leading and/or trailing hyperplanes that
        ///     are zero along all other dimensions - over every axis (the smallest bounding box that still holds every non-zero
        ///     value) by default, or only along the given axes.
        /// </summary>
        /// <param name="filt">Input array.</param>
        /// <param name="trim">
        ///     A string with 'f' representing trim from front and 'b' to trim from back. By default,
        ///     zeros are trimmed on both sides ("fb"). Case-insensitive; "bf" is accepted as an alias of "fb".
        /// </param>
        /// <param name="axis">
        ///     NumPy's <c>axis</c>: <c>null</c> (None, the default) crops <paramref name="filt"/> to the smallest bounding box
        ///     that still contains all non-zero values; one axis (<c>0</c>, <c>-1</c>) or a tuple (<c>(0, 1)</c>,
        ///     <c>new[] { 0, 1 }</c>) slices only those dimensions, on the sides <paramref name="trim"/> selects. The empty
        ///     tuple leaves the input unmodified.
        /// </param>
        /// <returns>
        ///     A view of <paramref name="filt"/> with leading/trailing all-zero hyperplanes removed. The number
        ///     of dimensions and the input dtype are preserved.
        ///     A <c>numpy.char.chararray</c> or <c>numpy.matrix</c> operand gives its class over the result (NumPy creates it
        ///     as the subclass), owning its data exactly when the plain result would.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="filt"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="trim"/> contains unexpected characters, an axis is repeated
        ///     (<c>repeated axis in `axis` argument</c>), or a <c>numpy.matrix</c> operand's result cannot be a matrix (more
        ///     than two axes longer than 1: <c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="AxisError">An axis is out of bounds for <paramref name="filt"/> (<c>axis: axis 2 is out of
        ///     bounds for array of dimension 2</c>: NumPy names the argument; every axis is checked before the repeat
        ///     check).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.trim_zeros.html. A string array is trimmed by its
        ///     truth values (an empty string — all NULs — is a zero): NumPy's <c>_arg_trim_zeros</c> runs <c>np.argwhere</c>,
        ///     the dtype's nonzero route, which never raises; the result is a view of the text itself.
        ///     <para>One signature since API-parity batch 4 (2026-11-12): it replaces <c>(filt, trim = "fb", int? axis)</c>
        ///     and <c>(filt, string trim, int[] axis)</c>, which made <c>np.trim_zeros(filt, "fb", null)</c> ambiguous
        ///     (CS0121) and had no tuple literal.</para></remarks>
        [NDScoped]
        public static NDArray trim_zeros(NDArray filt, string trim = "fb", NDAxis? axis = null)
        {
            if (filt is null) throw new ArgumentNullException(nameof(filt));
            // NumPy's statements (trim's check, normalize_axis_tuple(axis, filt.ndim, "axis"), the slices) are the engine's;
            // an integer axis is the one-element tuple there, as NumPy's normalize_axis_tuple makes it.
            return filt.TensorEngine.TrimZeros(filt, trim, axis?.ToArray());
        }
    }
}
