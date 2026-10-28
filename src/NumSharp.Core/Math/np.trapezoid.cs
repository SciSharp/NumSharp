using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Integrate <paramref name="y"/> along the given axis using the
        ///     composite trapezoidal rule.
        /// </summary>
        /// <param name="y">Input array to integrate (at least one dimensional).</param>
        /// <param name="x">
        ///     The sample points corresponding to the <paramref name="y"/> values.
        ///     When <c>null</c> (the default) the samples are assumed evenly spaced
        ///     <paramref name="dx"/> apart. A 1-D <paramref name="x"/> supplies the
        ///     spacing along <paramref name="axis"/> (broadcast across the other
        ///     axes); an N-D <paramref name="x"/> is differenced along its own
        ///     <paramref name="axis"/> (counted against <c>x.ndim</c>, as NumPy's
        ///     <c>diff(x, axis=axis)</c> does) and broadcast against <paramref name="y"/>.
        ///     Integration follows <paramref name="x"/>'s order — the points are NOT
        ///     sorted, so a decreasing <paramref name="x"/> integrates in reverse
        ///     (negative area).
        /// </param>
        /// <param name="dx">
        ///     The spacing between sample points when <paramref name="x"/> is
        ///     <c>null</c>. Default 1. Ignored when <paramref name="x"/> is supplied.
        /// </param>
        /// <param name="axis">
        ///     The axis along which to integrate; default is the last axis.
        ///     Negative axes count from the end. It is read against each array of the
        ///     composition the way NumPy reads it: <paramref name="y"/>'s slices use
        ///     <c>y.ndim</c>, the N-D <paramref name="x"/>'s difference uses <c>x.ndim</c>, and
        ///     the final sum uses the rank of the broadcast product (which an N-D
        ///     <paramref name="x"/> with more dimensions than <paramref name="y"/> raises).
        /// </param>
        /// <returns>
        ///     Definite integral approximated along <paramref name="axis"/>. If
        ///     <paramref name="y"/> is 1-D (and no higher-rank <paramref name="x"/> broadcasts
        ///     it) the result is a 0-D array (a scalar); otherwise it is the broadcast product
        ///     <c>d * (y[1:] + y[:-1])</c> with the integration axis removed, laid out as NumPy's
        ///     reduction lays it out. The dtype follows NEP50 promotion of the composition and is
        ///     always in the float family (float16/float32/float64/complex128, or the
        ///     NumSharp-only Decimal); integer and boolean inputs yield float64. The values are
        ///     bit-identical to NumPy's for every memory layout of <paramref name="y"/> and
        ///     <paramref name="x"/>, a single interval included: its final sum runs over a length-1 axis,
        ///     which starts from NumPy's <c>+0.0</c> identity, so a <c>-0.0</c> area comes back <c>+0.0</c>
        ///     (<c>np_sum_identity_tests.Trapezoid_OneInterval_NegativeZeroArea_IsPositiveZero</c>).
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="y"/> is <c>null</c>.</exception>
        /// <exception cref="IndexError">
        ///     <paramref name="axis"/> is outside <c>[-y.ndim, y.ndim-1]</c>, which every axis of a 0-D
        ///     <paramref name="y"/> is: NumPy's "list assignment index out of range", verbatim. NumPy's trapezoid
        ///     never checks the axis; Python's assignment into its list of length <c>y.ndim</c> raises. An error
        ///     in <paramref name="x"/> (the two below) is reported first, as in NumPy, which resolves the spacing
        ///     before it slices <paramref name="y"/>.
        /// </exception>
        /// <exception cref="ArgumentOutOfRangeException">
        ///     An N-D <paramref name="x"/> has no axis <paramref name="axis"/>:
        ///     <see cref="diff(NDArray, int, int, object, object)"/>'s "axis N is out of bounds for array of
        ///     dimension M" (NumPy's <c>AxisError</c> from <c>diff(x, axis=axis)</c>).
        /// </exception>
        /// <exception cref="ArgumentException">
        ///     A 0-D <paramref name="x"/> ("diff requires input that is at least one dimensional",
        ///     <see cref="diff(NDArray, int, int, object, object)"/>'s error).
        /// </exception>
        /// <exception cref="IncorrectShapeException">
        ///     The spacing does not broadcast against <c>y[1:] + y[:-1]</c> — NumPy's
        ///     <c>operands could not be broadcast together with shapes … </c> from the
        ///     <c>d * (…)</c> multiply, verbatim.
        /// </exception>
        /// <exception cref="TypeError">
        ///     A text <paramref name="y"/> or <paramref name="x"/> (U, S or StringDType): NumPy's statements run as they are,
        ///     and the first one without a loop for its operands refuses with NumPy's text — np.diff's subtract for a text
        ///     <paramref name="x"/>, the multiply of the spacing and the concatenated <c>y[1:] + y[:-1]</c> (for a fixed-width
        ///     <paramref name="y"/> and an integer spacing, the string multiply's "The 'out' kwarg is necessary" error), or the
        ///     division by <c>2.0</c> of a StringDType product an integer spacing repeated. A StringDType concatenation that
        ///     meets an NA it cannot add raises NumPy's <see cref="ValueError"/> instead.
        /// </exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html</remarks>
        [NDScoped]
        public static NDArray trapezoid(NDArray y, NDArray x = null, double dx = 1.0, int axis = -1)
        {
            // A numpy.matrix operand. NumPy ends in `(d * (y[1:] + y[:-1]) / 2.0).sum(axis)`: for a matrix y that
            // is the matrix method's sum, a row or a column. For a matrix x the product of the two matrices
            // raises inside NumPy, which then repeats the statement on plain arrays, so the result is plain.
            if (NDMatrixBoundary.Sees(y, x))
                return NDMatrixBoundary.IsMatrix(x)
                    ? NDMatrixBoundary.Plain((y, x, dx, axis), static s => trapezoid(s.y, s.x, s.dx, s.axis))
                    : NDMatrixBoundary.Reduced((y, x, dx, axis), static s => trapezoid(s.y, s.x, s.dx, s.axis), axis, false, null);
            if (y is null) throw new ArgumentNullException(nameof(y));
            return y.TensorEngine.Trapezoid(y, x, dx, axis);
        }
    }
}
