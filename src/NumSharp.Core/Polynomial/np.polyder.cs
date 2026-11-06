using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the derivative of the specified order of a polynomial.
        /// </summary>
        /// <param name="p">Polynomial coefficients, highest degree first. NumPy measures them with <c>len()</c>: a 2-D p is
        ///     differentiated along its first axis, each row a coefficient (<c>np.polyder(np.ones((3, 2)))</c> is
        ///     <c>[[2, 1], [2, 1]]</c>).</param>
        /// <param name="m">Order of differentiation (default 1); 0 returns <paramref name="p"/> as an array.</param>
        /// <returns>A new polynomial representing the derivative.</returns>
        /// <exception cref="ValueError">A negative <paramref name="m"/>, checked first.</exception>
        /// <exception cref="NotSupportedException"><paramref name="p"/> is null (NumPy's object array of None).</exception>
        /// <exception cref="IncorrectShapeException">An N-D <paramref name="p"/> whose rows do not broadcast with the ramp
        ///     at some order, the discarded <c>m == 0</c> product included, as NumPy's recursion computes it.</exception>
        /// <exception cref="TypeError">A 0-d <paramref name="p"/> at every order (<c>len() of unsized object</c>); text
        ///     coefficients the multiply has no loop for.</exception>
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
