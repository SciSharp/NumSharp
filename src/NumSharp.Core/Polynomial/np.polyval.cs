using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Evaluate a polynomial at specific values.
        ///     If <paramref name="p"/> has length <c>N</c> the value returned is
        ///     <c>p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]</c>, computed with Horner's scheme.
        /// </summary>
        /// <param name="p">1-D array of polynomial coefficients, highest degree first.</param>
        /// <param name="x">A number, or an array of numbers, at which to evaluate <paramref name="p"/>.</param>
        /// <returns>The evaluated values, of dtype <c>result_type(p, x)</c> and the shape of <paramref name="x"/>.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polyval.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray polyval(NDArray p, NDArray x)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(x))
                return NDMatrixBoundary.Wrapped((p, x), static s => polyval(s.p, s.x));
            p = np.asarray(p);

            // A poly1d `x` composes the two polynomials — handled by the poly1d overload below.
            x = np.asanyarray(x);

            // NumPy's `for pv in p` cannot iterate a 0-d coefficient array (a scalar is not a
            // length-1 polynomial — that would be `[c]`). A 1-D length-1 array is a valid constant.
            if (p.ndim == 0)
                throw new TypeError("iteration over a 0-d array");

            long n = p.size;
            // No coefficients — NumPy's loop never runs, so the result is zeros_like(x) (a string x included).
            if (n == 0)
                return np.zeros_like(x);

            // NumPy's Horner loop starts with y = zeros_like(x) and y * x: text in x has no multiply loop. Text in p meets
            // the first add, y * x + pv, whose left side keeps x's dtype; iterating a 1-D p yields NumPy SCALARS, whose
            // descriptor is their own text length (np.str_('a') is <U1, '' too), while an N-D p yields rows of p's dtype.
            if (Backends.Strings.NDStringUfuncs.IsString(x))
                throw Backends.Strings.NDStringUfuncs.NoLoop("multiply", x.dtype, x.dtype);
            if (Backends.Strings.NDStringUfuncs.IsString(p))
                throw Backends.Strings.NDStringUfuncs.NoLoop("add", x.dtype,
                    p.ndim == 1 ? Backends.Strings.NDStringArrays.ScalarDescr(p.Storage.GetAtIndex(0)) : p.dtype);

            // Horner accumulates `y = y*x + p[k]`. NumPy lets `y` (starting at zeros_like(x)) grow in
            // dtype as wider coefficients arrive; the fixed common dtype below is bit-identical because
            // every widening cast (int->wider, float32->float64, real->complex) is exact and `y` starts
            // at zero. Computing in `result_type` up front lets the whole loop reuse two buffers.
            NPTypeCode t = np.result_type(p, x);
            NDArray xc = x.astype(t);
            NDArray y = np.zeros(x.Shape, t);

            // Iterate from k = 0 so the very first `0 * x` reproduces NumPy's `0*inf -> nan` edge.
            for (long k = 0; k < n; k++)
            {
                np.multiply(y, xc, @out: y);
                np.add(y, p[k.ToString()], @out: y);
            }

            return y;
        }
    }
}
