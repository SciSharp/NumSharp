namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Gives a new shape to an array without changing its data — NumPy's one signature,
        ///     <c>np.reshape(a, /, shape, order='C', *, copy=None)</c>, which (as NumPy's <c>_wrapfunc</c> does) calls the
        ///     array's own <see cref="NDArray.reshape(Shape, char, bool?)"/>.
        /// </summary>
        /// <param name="a">Array to be reshaped.</param>
        /// <param name="shape">
        ///     The new shape. One parameter takes every spelling: an <c>int</c> (a 1-D result of that length), a tuple
        ///     (<c>(2, 3)</c>), an <c>int[]</c> / <c>long[]</c> or a <see cref="Shape"/> — <see cref="Shape"/>'s implicit
        ///     conversions. One dimension may be -1: it is inferred from the size and the other dimensions.
        /// </param>
        /// <param name="order">
        ///     The index order the elements are read in and placed in: <c>'C'</c> (the default, row-major), <c>'F'</c>
        ///     (column-major), or <c>'A'</c> (<c>'F'</c> when <paramref name="a"/> is Fortran-contiguous and not
        ///     C-contiguous, else <c>'C'</c>). <c>'K'</c> is NumPy's ValueError; either case is accepted.
        /// </param>
        /// <param name="copy">
        ///     <see langword="null"/> (the default): a view when the reshape can be one, otherwise a copy;
        ///     <see langword="true"/>: always a copy, sharing no memory with <paramref name="a"/> (even for the same shape);
        ///     <see langword="false"/>: never a copy — NumPy's <c>ValueError("Unable to avoid creating a copy while
        ///     reshaping.")</c> when no view can express the reshape (a transposed array to 1-D in C order).
        /// </param>
        /// <returns>The reshaped array: a view of <paramref name="a"/> whenever possible (and always for
        ///     <c>copy: false</c>), otherwise a view over an internal copy (owndata False either way, as in NumPy). A
        ///     <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the subclass),
        ///     owning its data exactly when the plain result would; a <c>numpy.matrix</c> operand gives a matrix.</returns>
        /// <exception cref="System.NullReferenceException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError"><c>order 'K' is not permitted for reshaping</c>; two -1 dimensions; the
        ///     <c>copy: false</c> refusal; a <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes
        ///     longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="IncorrectShapeException">The size does not match, with NumPy's text (<c>cannot reshape array of
        ///     size 6 into shape (4,2)</c>; NumPy raises it as a ValueError) — checked before the copy mode.</exception>
        /// <exception cref="System.ArgumentException">An order letter outside NumPy's vocabulary.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
        ///     <para>API-parity batch 7 made this NumPy's ONE signature: it replaced four overloads (<c>int[]</c>,
        ///     <c>long[]</c>, <c>Shape</c> and <c>ref Shape</c>, every one without <c>order</c> or <c>copy</c>) whose
        ///     shapes all reach <paramref name="shape"/> through its conversions, and the first parameter took NumPy's name
        ///     (<c>a</c>; it was <c>nd</c>).</para>
        /// </remarks>
        public static NDArray reshape(NDArray a, Shape shape, char order = 'C', bool? copy = null)
            => a.reshape(shape, order, copy); // NumPy's `_wrapfunc(a, 'reshape', shape, order=order, copy=copy)`
    }
}
