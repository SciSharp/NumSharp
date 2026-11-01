namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the imaginary part of the complex argument, element-wise.
        ///     One of the four basic complex-number accessors (with <see cref="real"/>, <see cref="angle"/>
        ///     and <see cref="conjugate"/>) — the standard post-FFT spectrum component extractors: for
        ///     <c>A = np.fft.fft(a)</c>, <c>A.imag</c> / <c>np.imag(A)</c> is the imaginary component.
        /// </summary>
        /// <param name="val">Input array.</param>
        /// <returns>
        ///     For a COMPLEX input: a float64 VIEW onto the imaginary lane — it SHARES memory with
        ///     <paramref name="val"/> and is writeable, so <c>np.imag(z)[i] = x</c> writes through to
        ///     <c>z[i]</c>'s imaginary part (reproducing NumPy's <c>z.imag</c>; complex128 -&gt; float64).
        ///     For a REAL / integer / boolean input: a fresh, READ-ONLY all-zeros array of the same shape
        ///     and dtype (the imaginary part of a real number is zero), matching NumPy exactly.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.imag.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray imag(NDArray val)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(val))
                return NDMatrixBoundary.Wrapped(val, static s => NDArray.imag_(s));
            // NumPy's `val.imag`: the statements are the array object's (NDArray.TypeCheck.cs)
            return NDArray.imag_(val);
        }
    }
}
