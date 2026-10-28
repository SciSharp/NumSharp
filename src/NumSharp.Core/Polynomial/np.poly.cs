using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Find the coefficients of a polynomial with the given sequence of roots (leading coefficient 1).
        ///     A square 2-D array is treated as a matrix, returning the coefficients of its characteristic
        ///     polynomial (via <see cref="linalg.eigvals"/>).
        /// </summary>
        /// <param name="seq_of_zeros">A 1-D sequence of roots, or a square 2-D array.</param>
        /// <returns>1-D coefficients highest degree first, <c>c[0] == 1</c>.</returns>
        /// <exception cref="ValueError">An input that is neither 1-D nor a non-empty square 2-D array (<c>input must be 1d or
        ///     non-empty square 2d array.</c>).</exception>
        /// <exception cref="TypeError">A square text matrix: its eigenvalues' finiteness check (<c>isfinite</c> has no text
        ///     loop).</exception>
        /// <exception cref="ValueError">A 1-D text root that does not parse as a float (NumPy casts the roots to their
        ///     mintypecode, 'd' for text: <c>could not convert string to float: np.str_('x')</c>).</exception>
        /// <exception cref="NotSupportedException">A square matrix and no LAPACK backend installed (the eigenvalues).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.poly.html
        ///     <para>An empty input returns the 0-d scalar <c>1.0</c> (NumPy's Python float <c>1.0</c>).</para>
        /// </remarks>
        [NDScoped]
        public static NDArray poly(NDArray seq_of_zeros)
        {
            using var plain = NDMatrixBoundary.PlainFor(seq_of_zeros);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return seq_of_zeros.TensorEngine.Poly(seq_of_zeros);
        }
    }
}
