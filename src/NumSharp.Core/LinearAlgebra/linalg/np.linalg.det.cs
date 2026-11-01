using System;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            ///     Determinant of a matrix, or of each matrix in a stack.
            /// </summary>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html
            ///     <para>
            ///     Computed from an LU factorisation (<c>getrf</c>), not by expansion — so for a
            ///     large or ill-conditioned matrix the product of the pivots can overflow or
            ///     underflow where <see cref="slogdet"/> stays finite.
            ///     </para>
            /// </remarks>
            /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
            /// <param name="a">The square matrix, or a stack of them.</param>
            /// <returns>The determinant: a 0-d array for one matrix, one value per matrix of a stack.</returns>
            /// <exception cref="LinAlgError">Fewer than two axes, or a non-square operand.</exception>
            /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
            /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal).</exception>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            // Scope: the facade keeps its scope over the call (the Linalg member reclaims the ToCommon dtype-cast temp).
            [NDScoped]
            public static NDArray det(NDArray a)
            {
                if (a is null) throw new ArgumentNullException(nameof(a));
                using var plain = NDMatrixBoundary.PlainFor(a);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
                return a.TensorEngine.Linalg.Det(a);
            }

            /// <summary>
            ///     Sign and natural log of the absolute determinant — the overflow-safe
            ///     <see cref="det"/>.
            /// </summary>
            /// <returns>
            ///     <c>(sign, logabsdet)</c>, such that <c>sign * exp(logabsdet)</c> is the
            ///     determinant. A singular matrix gives <c>(0, -inf)</c>; for a complex operand the
            ///     sign is a unit-modulus complex number rather than ±1.
            /// </returns>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.slogdet.html</remarks>
            /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
            /// <param name="a">The square matrix, or a stack of them.</param>
            /// <exception cref="LinAlgError">Fewer than two axes, or a non-square operand.</exception>
            /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
            /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal).</exception>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            // Scope: the facade keeps its scope over the call; the (sign, logabsdet) tuple result is yielded through the
            // weaver's Returns<T1,T2> tuple egress (the Linalg member reclaims the ToCommon dtype-cast temp).
            [NDScoped]
            public static (NDArray sign, NDArray logabsdet) slogdet(NDArray a)
            {
                if (a is null) throw new ArgumentNullException(nameof(a));
                using var plain = NDMatrixBoundary.PlainFor(a);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
                return a.TensorEngine.Linalg.Slogdet(a);
            }
        }
    }
}
