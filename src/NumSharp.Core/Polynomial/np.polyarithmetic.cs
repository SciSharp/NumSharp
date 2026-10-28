using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Find the sum of two polynomials. The shorter input is zero-extended on the high-degree side.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polyadd.html</remarks>
        public static NDArray polyadd(NDArray a1, NDArray a2)
        {
            using var plain = NDMatrixBoundary.PlainFor(a1, a2);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return a1.TensorEngine.Polyadd(a1, a2);
        }

        /// <summary>
        ///     Difference (subtraction) of two polynomials, <c>a1 - a2</c>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polysub.html</remarks>
        public static NDArray polysub(NDArray a1, NDArray a2)
        {
            using var plain = NDMatrixBoundary.PlainFor(a1, a2);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return a1.TensorEngine.Polysub(a1, a2);
        }

        /// <summary>
        ///     Find the product of two polynomials — the (full-mode) convolution of their coefficients.
        ///     Leading zeros of each input are dropped first (NumPy wraps each in <c>poly1d</c>).
        /// </summary>
        /// <param name="a1">The first polynomial's coefficients, highest degree first.</param>
        /// <param name="a2">The second polynomial's coefficients.</param>
        /// <returns>The product's coefficients.</returns>
        /// <exception cref="ValueError">An input of rank above 1 (<c>Polynomial must be 1d only.</c>, the first input's
        ///     first); with a text input otherwise, <c>data type must provide an itemsize</c>.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polymul.html</remarks>
        [NDScoped]
        public static NDArray polymul(NDArray a1, NDArray a2)
        {
            using var plain = NDMatrixBoundary.PlainFor(a1, a2);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return a1.TensorEngine.Polymul(a1, a2);
        }

        /// <summary>
        ///     Returns the quotient and remainder of polynomial division <c>u / v</c>.
        /// </summary>
        /// <param name="u">Dividend polynomial coefficients.</param>
        /// <param name="v">Divisor polynomial coefficients.</param>
        /// <returns><c>(q, r)</c> — quotient and remainder coefficients, floating (or complex) dtype.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.polydiv.html
        ///     <para>
        ///     A <c>poly1d</c> input does NOT force a <c>poly1d</c> return here (a C# tuple slot is a fixed
        ///     type); the <c>poly1d</c> <c>/</c> operator performs that wrapping instead.
        ///     </para>
        /// </remarks>
        // NDScoped: the floated inputs, per-iteration scaled quotient/remainder products and the
        // leading-zero-strip view chain are reclaimed; both tuple components (q, r) are yielded.
        [NDScoped]
        public static (NDArray q, NDArray r) polydiv(NDArray u, NDArray v)
        {
            using var plain = NDMatrixBoundary.PlainFor(u, v);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return u.TensorEngine.Polydiv(u, v);
        }
    }
}
