// The functions of numpy/lib/_polynomial_impl.py that compute (the legacy np.poly* API): poly, polyval, polyint,
// polyadd, polysub, polymul and polydiv. NumPy writes them in Python; here each is an engine member that DefaultEngine
// implements with NumPy's statements. The numpy.polynomial package (Polynomial/Package) is its own port and is not on
// the engine.

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        /// <summary>
        ///     NumPy's <c>np.poly(seq_of_zeros)</c>: the coefficients (highest degree first, leading 1) of the
        ///     polynomial with the given roots, or of a square matrix's characteristic polynomial (its eigenvalues are
        ///     the roots).
        /// </summary>
        /// <param name="seq_of_zeros">A 1-D sequence of roots, or a non-empty square 2-D array.</param>
        /// <returns>The coefficients, 1-D: float64 for real roots of any dtype but float32 (kept) and complex (kept;
        ///     the real part when the roots come in conjugate pairs); the 0-d float64 <c>1.0</c> for no
        ///     roots.</returns>
        /// <exception cref="ValueError">Neither 1-D nor a non-empty square 2-D array (<c>input must be 1d or non-empty
        ///     square 2d array.</c>); a 1-D text root that does not parse as a float (NumPy's mintypecode cast to
        ///     <c>'d'</c>).</exception>
        /// <exception cref="TypeError">A square text matrix: its eigenvalues' finiteness check has no text
        ///     loop.</exception>
        /// <exception cref="System.NotSupportedException">A square matrix with no LAPACK backend (the
        ///     eigenvalues).</exception>
        public abstract NDArray Poly(NDArray seq_of_zeros);

        /// <summary>
        ///     NumPy's <c>np.polyval(p, x)</c>: Horner's scheme, <c>y = y * x + p[k]</c> from <c>y = zeros_like(x)</c>.
        /// </summary>
        /// <param name="p">The coefficients, highest degree first (1-D; NumPy iterates it).</param>
        /// <param name="x">The points, any shape.</param>
        /// <returns>The values, of dtype <c>result_type(p, x)</c> and <paramref name="x"/>'s shape; with no
        ///     coefficient, <c>zeros_like(x)</c>.</returns>
        /// <exception cref="TypeError">A 0-d <paramref name="p"/> (<c>iteration over a 0-d array</c>), or text, which
        ///     has no multiply (in <paramref name="x"/>) or add (in <paramref name="p"/>) loop.</exception>
        public abstract NDArray Polyval(NDArray p, NDArray x);

        /// <summary>
        ///     NumPy's <c>np.polyint(p, m=1, k=None)</c>: the <paramref name="m"/>-th antiderivative, each integration
        ///     dividing the coefficients by their exponents and appending a constant.
        /// </summary>
        /// <param name="p">The coefficients, highest degree first.</param>
        /// <param name="m">The order of the antiderivative.</param>
        /// <param name="k">The integration constants, highest order first: null for zeros, one value for every order,
        ///     or at least <paramref name="m"/> values.</param>
        /// <returns>The antiderivative's coefficients (floating or complex: the division by the exponents); for
        ///     <paramref name="m"/> 0, <paramref name="p"/> as an array.</returns>
        /// <exception cref="ValueError">A negative <paramref name="m"/>, or fewer constants than
        ///     <paramref name="m"/>.</exception>
        /// <exception cref="TypeError">With <paramref name="m"/> above 0: a 0-d <paramref name="p"/> (<c>len() of
        ///     unsized object</c>), or text, whose division by the exponents has no loop.</exception>
        public abstract NDArray Polyint(NDArray p, int m = 1, NDArray k = null);

        /// <summary>
        ///     NumPy's <c>np.polyadd(a1, a2)</c>: the shorter coefficient array zero-extended on the high-degree side,
        ///     then added.
        /// </summary>
        /// <param name="a1">The first polynomial's coefficients (a scalar is a constant).</param>
        /// <param name="a2">The second polynomial's coefficients.</param>
        /// <returns>The sum's coefficients, in the addition's promoted dtype.</returns>
        /// <exception cref="TypeError">Coefficients the addition has no loop for (text with numbers).</exception>
        public abstract NDArray Polyadd(NDArray a1, NDArray a2);

        /// <summary>
        ///     NumPy's <c>np.polysub(a1, a2)</c>: the shorter coefficient array zero-extended on the high-degree side,
        ///     then <c>a1 - a2</c>.
        /// </summary>
        /// <param name="a1">The minuend's coefficients.</param>
        /// <param name="a2">The subtrahend's coefficients.</param>
        /// <returns>The difference's coefficients, in the subtraction's promoted dtype.</returns>
        /// <exception cref="TypeError">Coefficients the subtraction has no loop for (text, bool).</exception>
        public abstract NDArray Polysub(NDArray a1, NDArray a2);

        /// <summary>
        ///     NumPy's <c>np.polymul(a1, a2)</c>: each input's leading zeros dropped (NumPy wraps it in <c>poly1d</c>),
        ///     then the full convolution of the coefficients.
        /// </summary>
        /// <param name="a1">The first polynomial's coefficients.</param>
        /// <param name="a2">The second polynomial's coefficients.</param>
        /// <returns>The product's coefficients.</returns>
        /// <exception cref="ValueError">An input of rank above 1 (<c>Polynomial must be 1d only.</c>, the first input's
        ///     first); fixed-width text otherwise (<c>data type must provide an itemsize</c>).</exception>
        public abstract NDArray Polymul(NDArray a1, NDArray a2);

        /// <summary>
        ///     NumPy's <c>np.polydiv(u, v)</c>: long division of the floated coefficients (<c>u + 0.0</c>, <c>v +
        ///     0.0</c>), the scalar quotient coefficients computed with NumPy's scalar arithmetic, the remainder's
        ///     leading near-zero coefficients dropped (<c>rtol=1e-14</c>, at least one kept).
        /// </summary>
        /// <param name="u">The dividend's coefficients.</param>
        /// <param name="v">The divisor's coefficients.</param>
        /// <returns>The quotient and the remainder, floating or complex.</returns>
        /// <exception cref="TypeError">Coefficients the float conversion has no loop for (text).</exception>
        public abstract (NDArray q, NDArray r) Polydiv(NDArray u, NDArray v);
    }
}
