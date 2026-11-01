// The functions of numpy/lib/_polynomial_impl.py that compute (the legacy np.poly* API): poly, polyval, polyint,
// polyadd, polysub, polymul, polydiv and roots. NumPy writes them in Python; here each is an engine member that DefaultEngine
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
        /// <param name="seq_of_zeros">A 1-D sequence of roots, or a non-empty square 2-D array. A 0-d input is one root
        ///     (atleast_1d): this is poly's body, which NumPy's np.poly reaches only through its dispatcher, and that
        ///     refuses a 0-d input first (the np.poly facade raises its TypeError).</param>
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
        ///     NumPy's <c>np.polyval(p, x)</c>: Horner's scheme, <c>y = y * x + pv</c> for each <c>pv</c> in
        ///     <paramref name="p"/>, from <c>y = zeros_like(x)</c>.
        /// </summary>
        /// <param name="p">The coefficients, highest degree first. NumPy iterates its first axis (<c>len(p)</c>
        ///     coefficients): an N-D p's coefficients are its rows, each broadcast against x.</param>
        /// <param name="x">The points, any shape.</param>
        /// <returns>The values, of dtype <c>result_type(p, x)</c>; of <paramref name="x"/>'s shape for a 1-D p, of the
        ///     broadcast of x's shape and a row's for an N-D p; with no coefficient, <c>zeros_like(x)</c>.</returns>
        /// <exception cref="TypeError">A 0-d <paramref name="p"/> (<c>iteration over a 0-d array</c>), or text, which
        ///     has no multiply (in <paramref name="x"/>) or add (in <paramref name="p"/>) loop.</exception>
        /// <exception cref="IncorrectShapeException">An N-D p whose rows do not broadcast with x (the add's
        ///     <c>operands could not be broadcast together</c> text, NumPy's ValueError).</exception>
        public abstract NDArray Polyval(NDArray p, NDArray x);

        /// <summary>
        ///     NumPy's <c>np.polyint(p, m=1, k=None)</c>: the <paramref name="m"/>-th antiderivative, each integration
        ///     dividing the coefficients by their exponents and appending a constant.
        /// </summary>
        /// <param name="p">The coefficients, highest degree first, measured by <c>len()</c> (its first axis: an N-D p
        ///     meets the division's broadcast or the concatenation's rank error, as in NumPy).</param>
        /// <param name="m">The order of the antiderivative.</param>
        /// <param name="k">The integration constants, highest order first, measured by <c>len()</c>: null for zeros,
        ///     one value for every order, or at least <paramref name="m"/> values.</param>
        /// <returns>The antiderivative's coefficients (floating or complex: the division by the exponents); for
        ///     <paramref name="m"/> 0, <paramref name="p"/> as an array.</returns>
        /// <exception cref="ValueError">A negative <paramref name="m"/>, or fewer constants than
        ///     <paramref name="m"/>.</exception>
        /// <exception cref="TypeError">With <paramref name="m"/> above 0: a 0-d <paramref name="p"/> (<c>len() of
        ///     unsized object</c>), or text, whose division by the exponents has no loop.</exception>
        /// <exception cref="IncorrectShapeException">An N-D p or k whose shapes the division, the constants' broadcast or
        ///     the concatenation refuses (NumPy's ValueError texts).</exception>
        public abstract NDArray Polyint(NDArray p, int m = 1, NDArray k = null);

        /// <summary>
        ///     NumPy's <c>np.polyder(p, m=1)</c>: the <paramref name="m"/>-th derivative, each order
        ///     <c>p[:-1] * arange(n, 0, -1)</c> with <c>n = len(p) - 1</c>.
        /// </summary>
        /// <param name="p">The coefficients, highest degree first (null is NumPy's None: <c>asarray(None)</c>'s refusal).
        ///     Each order slices the FIRST axis and multiplies by an int64 ramp of the array's size, so an N-D p meets
        ///     the multiply's broadcast error, as in NumPy.</param>
        /// <param name="m">The order of differentiation: 0 returns <c>asarray(p)</c> itself.</param>
        /// <returns>The derivative's coefficients (int32 coefficients become int64 through the int64 ramp); for
        ///     <paramref name="m"/> past the degree, an empty array.</returns>
        /// <exception cref="ValueError">A negative <paramref name="m"/> (<c>Order of derivative must be positive (see
        ///     polyint)</c>), checked before <paramref name="p"/> is read.</exception>
        /// <exception cref="System.NotSupportedException"><paramref name="p"/> is null (NumPy's object array of None).</exception>
        /// <exception cref="IncorrectShapeException">An N-D p whose rows do not broadcast with the ramp (the multiply's
        ///     <c>operands could not be broadcast together</c>, NumPy's ValueError).</exception>
        /// <exception cref="IndexError">A 0-d p sliced with <c>[:-1]</c> (<c>too many indices for array</c>).</exception>
        /// <exception cref="TypeError">Text coefficients, which the multiply has no loop for.</exception>
        /// <remarks>
        ///     NumPy writes polyder in Python (<c>numpy/lib/_polynomial_impl.py</c>, which also handles a poly1d - NumSharp's
        ///     poly1d calls its own derivative); DefaultEngine runs its loop over the array object's slice and multiply.
        ///     <c>np.polyder</c> calls it inside the matrix boundary's plain region, on <paramref name="p"/>'s engine (the
        ///     default one for null); an override replaces it for every array whose engine it is.
        /// </remarks>
        public abstract NDArray Polyder(NDArray p, int m = 1);

        /// <summary>
        ///     NumPy's <c>np.polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)</c>: the least-squares polynomial
        ///     of degree <paramref name="deg"/> through the points, from the Vandermonde matrix of <paramref name="x"/> with
        ///     its columns scaled to unit norm, one <c>np.linalg.lstsq</c> solve, and the scale taken back out of the
        ///     coefficients.
        /// </summary>
        /// <param name="x">The sample points, 1-D (null is NumPy's None: <c>asarray(None)</c>'s refusal); <c>x + 0.0</c>
        ///     makes them float (float32 stays float32).</param>
        /// <param name="y">The sample values: 1-D, or 2-D with one data set per column, as long as
        ///     <paramref name="x"/>.</param>
        /// <param name="deg">The degree of the polynomial.</param>
        /// <param name="rcond">The relative condition number of the fit, or null for <c>len(x) * eps</c> of x's float
        ///     dtype.</param>
        /// <param name="full">Also return lstsq's residuals, rank and singular values (NumPy's five-tuple).</param>
        /// <param name="w">Weights applied to the unsquared residuals, one per sample point, or null.</param>
        /// <param name="cov">NumPy's <c>cov=</c>: null / <c>false</c> (no covariance), <c>true</c> (scaled by the residuals),
        ///     <c>"unscaled"</c>, or any other object, read as Python's truth value (the empty string is false). Ignored with
        ///     <paramref name="full"/>.</param>
        /// <returns>
        ///     The <see cref="PolyfitResult"/> carrying exactly the arrays NumPy returns: the coefficients (highest degree
        ///     first, one column per data set); with <paramref name="full"/>, the residuals, rank and singular values; with a
        ///     truthy <paramref name="cov"/>, the covariance. Every other slot is null.
        /// </returns>
        /// <exception cref="ValueError">A negative <paramref name="deg"/> (<c>expected deg &gt;= 0</c>); a scaled covariance
        ///     with no more points than coefficients (<c>the number of data points must exceed order to scale the covariance
        ///     matrix</c>).</exception>
        /// <exception cref="TypeError">NumPy's shape refusals in its order (<c>expected 1D vector for x</c>, <c>expected
        ///     non-empty vector for x</c>, <c>expected 1D or 2D array for y</c>, <c>expected x and y to have same length</c>,
        ///     <c>expected a 1-d array for weights</c>, <c>expected w and y to have the same length</c>); text the float
        ///     conversion <c>+ 0.0</c> has no loop for; a float16 or decimal x that numpy.linalg refuses (<c>array type float16
        ///     is unsupported in linalg</c>).</exception>
        /// <exception cref="System.NotSupportedException"><paramref name="x"/> or <paramref name="y"/> is null (NumPy's object array
        ///     of None).</exception>
        /// <exception cref="OpenBlasMissingBackendException">No LAPACK backend is installed: lstsq needs one.</exception>
        /// <exception cref="LinAlgError">lstsq's SVD does not converge (a NaN or an infinity in the data), or the covariance's
        ///     inverse meets a singular matrix.</exception>
        /// <exception cref="IncorrectShapeException">Weights that do not broadcast against a 2-D y's columns.</exception>
        /// <remarks>
        ///     NumPy writes polyfit in Python; DefaultEngine runs its statements over the operand engines' <see cref="Vander"/>,
        ///     <see cref="Sum(NDArray, int?, DType, bool)"/>, <see cref="Sqrt"/>, <see cref="Dot"/>, <see cref="Outer"/> and the
        ///     <see cref="Linalg"/> sub-engine's lstsq and inv. NumPy's RankWarning for a rank-deficient fit is not modelled.
        ///     <c>np.polyfit</c> calls it inside the matrix boundary's plain region, on <paramref name="x"/>'s engine (the
        ///     default one for null); an override replaces it for every array whose engine it is.
        /// </remarks>
        public abstract PolyfitResult Polyfit(NDArray x, NDArray y, int deg, double? rcond = null, bool full = false,
            NDArray w = null, object cov = null);

        /// <summary>
        ///     NumPy's <c>np.polyadd(a1, a2)</c>: the shorter coefficient array (by <c>len()</c>, the first axis)
        ///     zero-extended on the high-degree side with zeros of its own dtype, then added.
        /// </summary>
        /// <param name="a1">The first polynomial's coefficients (a scalar is a constant).</param>
        /// <param name="a2">The second polynomial's coefficients.</param>
        /// <returns>The sum's coefficients, in the addition's promoted dtype.</returns>
        /// <exception cref="TypeError">Coefficients the addition has no loop for (text with numbers).</exception>
        /// <exception cref="IncorrectShapeException">An N-D operand: the 1-D pad meets np.concatenate's rank error, and
        ///     equal first axes the addition's broadcast error (NumPy's ValueError texts).</exception>
        public abstract NDArray Polyadd(NDArray a1, NDArray a2);

        /// <summary>
        ///     NumPy's <c>np.polysub(a1, a2)</c>: the shorter coefficient array (by <c>len()</c>, the first axis)
        ///     zero-extended on the high-degree side with zeros of its own dtype, then <c>a1 - a2</c>.
        /// </summary>
        /// <param name="a1">The minuend's coefficients.</param>
        /// <param name="a2">The subtrahend's coefficients.</param>
        /// <returns>The difference's coefficients, in the subtraction's promoted dtype.</returns>
        /// <exception cref="TypeError">Coefficients the subtraction has no loop for (text, bool).</exception>
        /// <exception cref="IncorrectShapeException">An N-D operand: the 1-D pad meets np.concatenate's rank error, and
        ///     equal first axes the subtraction's broadcast error (NumPy's ValueError texts).</exception>
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
        /// <param name="u">The dividend's coefficients, measured by <c>len()</c> (its first axis).</param>
        /// <param name="v">The divisor's coefficients, measured by <c>len()</c>.</param>
        /// <returns>The quotient and the remainder, floating or complex; for an N-D u shorter than v, NumPy's
        ///     <c>(zeros(1), u)</c> with u's leading near-zero rows dropped.</returns>
        /// <exception cref="TypeError">Coefficients the float conversion has no loop for (text); or a complex quotient
        ///     coefficient with dimensions (u or v N-D), which NumPy's <c>q[k] = d</c> refuses through complex()
        ///     (<c>only 0-dimensional arrays can be converted to Python scalars</c>).</exception>
        /// <exception cref="ValueError">A real quotient coefficient with dimensions (<c>setting an array element with a
        ///     sequence.</c>).</exception>
        /// <exception cref="IncorrectShapeException">Rows of an N-D u and v that do not broadcast: NumPy's first
        ///     statement <c>w = u[0] + v[0]</c> raises the add's error, u's row first.</exception>
        /// <exception cref="IndexError">An empty u or v (<c>u[0]</c> / <c>v[0]</c>).</exception>
        public abstract (NDArray q, NDArray r) Polydiv(NDArray u, NDArray v);

        /// <summary>
        ///     NumPy's <c>np.roots(p)</c>: the roots of the polynomial with coefficients <paramref name="p"/> (highest
        ///     degree first), the eigenvalues of its companion matrix after the leading and trailing zeros are stripped
        ///     (each trailing zero a root at 0).
        /// </summary>
        /// <param name="p">The coefficients, 1-D after <c>atleast_1d</c>. This is roots' body: a 0-d input is one
        ///     coefficient here, while the np.roots facade refuses it first with NumPy's dispatcher TypeError.</param>
        /// <returns>The roots: complex128 or float64 (LAPACK's eigenvalues), with the zero roots appended; an empty
        ///     float64 array for no non-zero coefficient or a single one.</returns>
        /// <exception cref="ValueError">p is not rank 1 (<c>Input must be a rank-1 array.</c>), or a text coefficient
        ///     between the first and last non-empty one does not parse as a float.</exception>
        /// <exception cref="System.NotSupportedException">Two or more non-zero coefficients and no LAPACK backend.</exception>
        public abstract NDArray Roots(NDArray p);
    }
}
