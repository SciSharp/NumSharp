using System;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            ///     Cholesky decomposition of a Hermitian positive-definite matrix.
            /// </summary>
            /// <param name="upper">
            ///     False (the default) returns the LOWER-triangular factor <c>L</c> with
            ///     <c>a = L L*</c>; true returns the upper-triangular <c>U</c> with
            ///     <c>a = U* U</c>.
            /// </param>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html
            ///     <para>
            ///     Only the lower triangle of <paramref name="a"/> is read, so a non-Hermitian
            ///     operand is interpreted as its own reflection rather than rejected. A matrix that
            ///     is not positive definite raises
            ///     <c>LinAlgError("Matrix is not positive definite")</c> from the factorisation.
            ///     </para>
            /// </remarks>
            /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves (an empty operand or a zero-size batch needs none).</exception>
            /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
            [NDScoped] // reclaims the ToCommon cast temp
            public static NDArray cholesky(NDArray a, bool upper = false)
            {
                // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
                if (NDMatrixBoundary.Sees(a))
                    return NDMatrixBoundary.Wrapped((a, upper), static s => cholesky(s.a, s.upper));
                AssertStackedSquare(a);
                var common = CommonType(a);
                // NumPy calls _umath_linalg.cholesky_lo / cholesky_up: its input cast refuses text.
                ThrowIfTextGufunc(upper ? "cholesky_up" : "cholesky_lo", a);
                return a.TensorEngine.Cholesky(ToCommon(a, common), upper);
            }
        }
    }
}
