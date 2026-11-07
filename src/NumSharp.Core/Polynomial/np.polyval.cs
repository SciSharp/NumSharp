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
            // NumPy's statements (asarray(p), y = zeros_like(x), the Horner loop) are the engine's; the result is created as
            // x's class (zeros_like's subok), and a chararray x's own errors come from its operators there.
            => p.TensorEngine.Polyval(p, x);
    }
}
