namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the real part of the complex argument, element-wise.
        ///     One of the four basic complex-number accessors (with <see cref="imag"/>, <see cref="angle"/>
        ///     and <see cref="conjugate"/>) — the standard post-FFT spectrum component extractors: for
        ///     <c>A = np.fft.fft(a)</c>, <c>A.real</c> / <c>np.real(A)</c> is the real component.
        /// </summary>
        /// <param name="val">Input array.</param>
        /// <returns>
        ///     For a COMPLEX input: a float64 VIEW onto the real lane — it SHARES memory with
        ///     <paramref name="val"/> and is writeable, so <c>np.real(z)[i] = x</c> writes through to
        ///     <c>z[i]</c>'s real part (reproducing NumPy's <c>z.real</c>; complex128 -&gt; float64).
        ///     For a REAL / integer / boolean input: NumPy returns the array itself (the real part of a
        ///     real number is the number), so the input is returned unchanged with its dtype preserved.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.real.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray real(NDArray val)
        {
            // a numpy.matrix operand: NumPy's val.real is a matrix over the real lane for a complex matrix, and a real
            // matrix itself (NDArray.RealPart hands the matrix object back, which Wrapped returns unchanged)
            if (NDMatrixBoundary.Sees(val))
                return NDMatrixBoundary.Wrapped(val, static s => NDArray.RealPart(s));
            // NumPy's `val.real`: the statements are the array object's (NDArray.TypeCheck.cs)
            return NDArray.RealPart(val);
        }
    }
}
