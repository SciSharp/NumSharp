using System;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            ///     Eigenvalues and right eigenvectors of a general (not necessarily symmetric) matrix.
            /// </summary>
            /// <returns><c>(eigenvalues, eigenvectors)</c>, the columns of the second being the vectors.</returns>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
            ///     <para>
            ///     The result dtype is DATA-dependent: a real matrix with a complex-conjugate pair of
            ///     eigenvalues yields complex output, so the dtype cannot be predicted from the input
            ///     dtype alone. Use <see cref="eigh"/> where the operand is known symmetric or
            ///     Hermitian — it is both faster and guaranteed to give real eigenvalues.
            ///     </para>
            /// </remarks>
            /// <exception cref="NotSupportedException">No LAPACK backend serves these operands.</exception>
            public static (NDArray Eigenvalues, NDArray Eigenvectors) eig(NDArray a)
            {
                AssertStackedSquare(a);
                var common = CommonType(a);
                return a.TensorEngine.Eig(ToCommon(a, common), computeVectors: true);
            }

            /// <summary>
            ///     Eigenvalues of a general matrix, without the eigenvectors.
            /// </summary>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvals.html</remarks>
            /// <exception cref="NotSupportedException">No LAPACK backend serves these operands.</exception>
            public static NDArray eigvals(NDArray a)
            {
                AssertStackedSquare(a);
                var common = CommonType(a);
                return a.TensorEngine.Eig(ToCommon(a, common), computeVectors: false).Eigenvalues;
            }

            /// <summary>
            ///     Eigenvalues and eigenvectors of a real symmetric or complex Hermitian matrix.
            /// </summary>
            /// <param name="UPLO">
            ///     Which triangle holds the data — <c>'L'</c> (default) or <c>'U'</c>. Case
            ///     insensitive. The other triangle is NOT read, so a non-symmetric operand is
            ///     silently interpreted as its own reflection rather than rejected.
            /// </param>
            /// <returns>
            ///     <c>(eigenvalues, eigenvectors)</c> with the eigenvalues in ASCENDING order and
            ///     always REAL — even for a complex operand, where they come back as the real
            ///     counterpart dtype.
            /// </returns>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html</remarks>
            /// <exception cref="NotSupportedException">No LAPACK backend serves these operands.</exception>
            public static (NDArray Eigenvalues, NDArray Eigenvectors) eigh(NDArray a, char UPLO = 'L')
            {
                char uplo = RequireUplo(UPLO);
                AssertStackedSquare(a);
                var common = CommonType(a);
                return a.TensorEngine.Eigh(ToCommon(a, common), uplo, computeVectors: true);
            }

            /// <summary>
            ///     Eigenvalues of a real symmetric or complex Hermitian matrix, ascending.
            /// </summary>
            /// <inheritdoc cref="eigh"/>
            public static NDArray eigvalsh(NDArray a, char UPLO = 'L')
            {
                char uplo = RequireUplo(UPLO);
                AssertStackedSquare(a);
                var common = CommonType(a);
                return a.TensorEngine.Eigh(ToCommon(a, common), uplo, computeVectors: false).Eigenvalues;
            }

            /// <summary>
            ///     NumPy validates UPLO before touching the array, and accepts either case.
            /// </summary>
            private static char RequireUplo(char uplo)
            {
                char upper = char.ToUpperInvariant(uplo);
                if (upper != 'L' && upper != 'U')
                    throw new ValueError("UPLO argument must be 'L' or 'U'");
                return upper;
            }
        }
    }
}
