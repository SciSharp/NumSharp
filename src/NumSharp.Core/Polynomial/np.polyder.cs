using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the derivative of the specified order of a polynomial.
        /// </summary>
        /// <param name="p">Polynomial coefficients, highest degree first.</param>
        /// <param name="m">Order of differentiation (default 1).</param>
        /// <returns>A new polynomial representing the derivative.</returns>
        /// <exception cref="ValueError">A negative <paramref name="m"/>, checked first.</exception>
        /// <exception cref="NotSupportedException"><paramref name="p"/> is null (NumPy's object array of None).</exception>
        /// <exception cref="IncorrectShapeException">An N-D <paramref name="p"/> whose rows do not broadcast with the
        ///     ramp.</exception>
        /// <exception cref="IndexError">A 0-d <paramref name="p"/>.</exception>
        /// <exception cref="TypeError">Text coefficients.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="p"/> is null.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.polyder.html
        ///     <para>The statements are <see cref="TensorEngine.Polyder"/>, on <paramref name="p"/>'s engine (the default one
        ///     for null, so the order check still comes first), inside the matrix boundary's plain region.</para>
        /// </remarks>
        public static NDArray polyder(NDArray p, int m = 1)
        {
            if (p is null) throw new ArgumentNullException(nameof(p));
            using var plain = NDMatrixBoundary.PlainFor(p);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return p.TensorEngine.Polyder(p, m);
        }
    }
}
