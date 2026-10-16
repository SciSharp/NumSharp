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
        /// <returns>The evaluated values, of dtype <c>result_type(p, x)</c> and the shape of <paramref name="x"/>. With no
        ///     coefficient, <c>zeros_like(x)</c> — a chararray for a <c>numpy.char.chararray</c> <paramref name="x"/>.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polyval.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes
        ///     longer than 1 (<c>shape too large to be a matrix.</c>); a <c>numpy.char.chararray</c> <paramref name="x"/> with
        ///     a coefficient (<c>Can only multiply by integers</c>: NumPy's first <c>y * x</c> is the chararray's
        ///     repetition operator).</exception>
        /// <exception cref="TypeError">A 0-d <paramref name="p"/> (<c>iteration over a 0-d array</c>), or text that has no
        ///     multiply / add loop.</exception>
        [NDScoped]
        public static NDArray polyval(NDArray p, NDArray x)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(x))
                return NDMatrixBoundary.Wrapped((p, x), static s => polyval(s.p, s.x));
            // a numpy.char.chararray x: NumPy's y = zeros_like(x) is a chararray, so with no coefficient the result is that
            // chararray, and otherwise the first y * x is chararray.__mul__ — np.char.multiply(y, x), whose count is not an
            // integer: its ValueError, after p's own check (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(x))
                return PolyvalOfCharArray(p, x);
            // a numpy.char.chararray p alone: NumPy's asarray(p) drops the class
            if (NDCharArrayBoundary.Sees(p))
                return NDCharArrayBoundary.Plain((p, x), static s => polyval(s.p, s.x));
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

        /// <summary>
        ///     <c>np.polyval(p, x)</c> for a <c>numpy.char.chararray</c> <paramref name="x"/>, in NumPy's statement order:
        ///     <c>p = asarray(p)</c>; <c>y = zeros_like(x)</c>, which creates a chararray; then the Horner loop, whose first
        ///     <c>y * x</c> is <c>chararray.__mul__</c> = <c>np.char.multiply(y, x)</c> — the count <paramref name="x"/> is
        ///     never an integer array there, so NumPy raises <c>np.char.multiply</c>'s ValueError.
        /// </summary>
        /// <param name="p">The coefficients.</param>
        /// <param name="x">The chararray of points.</param>
        /// <returns>With no coefficient, NumPy's <c>zeros_like(x)</c>: an owning chararray of empty strings (zeros for an
        ///     <c>int8</c> chararray).</returns>
        /// <exception cref="TypeError">A 0-d <paramref name="p"/> (<c>iteration over a 0-d array</c>, NumPy's
        ///     <c>for pv in p</c>).</exception>
        /// <exception cref="ValueError">At least one coefficient: <c>Can only multiply by integers</c>.</exception>
        /// <remarks>Only the rank and size of <c>asarray(p)</c> are read, which the conversion does not change, so
        ///     <paramref name="p"/> is read as it is. The zeros are <c>np.zeros_like</c>'s plain result kept as a chararray by
        ///     the boundary's <see cref="NDCharArrayBoundary.Kept{TState}"/> rule, so no public function sees the chararray
        ///     from library code.</remarks>
        private static NDArray PolyvalOfCharArray(NDArray p, NDArray x)
        {
            if (p.ndim == 0)
                throw new TypeError("iteration over a 0-d array");
            if (p.size == 0)
                return NDCharArrayBoundary.Kept(x, static s => np.zeros_like(s));
            throw new ValueError("Can only multiply by integers");
        }
    }
}
