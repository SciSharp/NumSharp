using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return an antiderivative (indefinite integral) of a polynomial.
        /// </summary>
        /// <param name="p">Polynomial to integrate (coefficients, highest degree first).</param>
        /// <param name="m">Order of the antiderivative (default 1).</param>
        /// <param name="k">
        ///     Integration constants, highest-order term first. <c>null</c> (default) means all zero.
        ///     For <c>m == 1</c> a single scalar may be given.
        /// </param>
        /// <returns>The antiderivative — always floating (or complex) because of the term-wise division.</returns>
        /// <exception cref="ValueError">A negative <paramref name="m"/>, or fewer integration constants than
        ///     <paramref name="m"/>.</exception>
        /// <exception cref="TypeError">With <paramref name="m"/> above 0: a 0-d <paramref name="p"/> (<c>len() of unsized
        ///     object</c>), or text, whose division by the exponents has no loop.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polyint.html</remarks>
        [NDScoped]
        public static NDArray polyint(NDArray p, int m = 1, NDArray k = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(p, k);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return p.TensorEngine.Polyint(p, m, k);
        }

        /// <summary>Convenience overload — <c>polyint</c> with a single scalar integration constant.</summary>
        [NDScoped] // the 0-d k wrapper is built in THIS frame, before the main overload's scope opens
        public static NDArray polyint(NDArray p, int m, double k)
        {
            using var plain = NDMatrixBoundary.PlainFor(p);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return polyint(p, m, NDArray.Scalar(k));
        }
    }
}
