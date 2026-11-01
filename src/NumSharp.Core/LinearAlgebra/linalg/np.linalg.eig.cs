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
            /// <param name="a">The square matrix, or a stack of them.</param>
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
            /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves (an empty operand or a zero-size batch needs none).</exception>
            /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
            /// <exception cref="LinAlgError">Fewer than two axes, a non-square operand, or an operand holding inf / NaN.</exception>
            /// <exception cref="TypeError">A text operand, or a float16, decimal or char operand.</exception>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            [NDScoped] // the class boundary's temporaries; the Linalg member scopes the computation's own
            public static (NDArray eigenvalues, NDArray eigenvectors) eig(NDArray a)
            {
                if (a is null) throw new ArgumentNullException(nameof(a));
                // a numpy.matrix operand: NumPy returns the plain eigenvalues and wrap(vt), the eigenvectors as a matrix
                if (NDMatrixBoundary.Sees(a))
                    return NDMatrixBoundary.WrappedPair(a, static s => eig(s), wrapFirst: false, wrapSecond: true);
                return a.TensorEngine.Linalg.Eig(a);
            }

            /// <summary>
            ///     Eigenvalues of a general matrix, without the eigenvectors.
            /// </summary>
            /// <param name="a">The square matrix, or a stack of them.</param>
            /// <returns>The eigenvalues: real for a real operand whose eigenvalues are all real, else complex.</returns>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvals.html</remarks>
            /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves (an empty operand or a zero-size batch needs none).</exception>
            /// <exception cref="LinAlgError">Fewer than two axes, a non-square operand, or an operand holding inf / NaN.</exception>
            /// <exception cref="TypeError">A text operand, or a float16, decimal or char operand.</exception>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            [NDScoped] // the facade keeps its scope over the call; the Linalg member scopes the computation's own
            public static NDArray eigvals(NDArray a)
            {
                if (a is null) throw new ArgumentNullException(nameof(a));
                using var plain = NDMatrixBoundary.PlainFor(a);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
                return a.TensorEngine.Linalg.Eigvals(a);
            }

            /// <summary>
            ///     Eigenvalues and eigenvectors of a real symmetric or complex Hermitian matrix.
            /// </summary>
            /// <param name="a">The real symmetric / complex Hermitian matrix, or a stack of them.</param>
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
            /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves (an empty operand or a zero-size batch needs none).</exception>
            /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>);
            ///     a <paramref name="UPLO"/> that is neither 'L' nor 'U'.</exception>
            /// <exception cref="LinAlgError">Fewer than two axes, or a non-square operand.</exception>
            /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
            /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal).</exception>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            [NDScoped] // the class boundary's temporaries; the Linalg member scopes the computation's own
            public static (NDArray eigenvalues, NDArray eigenvectors) eigh(NDArray a, char UPLO = 'L')
            {
                if (a is null) throw new ArgumentNullException(nameof(a));
                // a numpy.matrix operand: NumPy returns the plain eigenvalues and wrap(vt), the eigenvectors as a matrix
                if (NDMatrixBoundary.Sees(a))
                    return NDMatrixBoundary.WrappedPair((a, UPLO), static s => eigh(s.a, s.UPLO), wrapFirst: false, wrapSecond: true);
                return a.TensorEngine.Linalg.Eigh(a, UPLO);
            }

            /// <summary>
            ///     Eigenvalues of a real symmetric or complex Hermitian matrix, ascending.
            /// </summary>
            /// <inheritdoc cref="eigh"/>
            /// <returns>The eigenvalues, ascending and real.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            [NDScoped] // the facade keeps its scope over the call; the Linalg member scopes the computation's own
            public static NDArray eigvalsh(NDArray a, char UPLO = 'L')
            {
                if (a is null) throw new ArgumentNullException(nameof(a));
                using var plain = NDMatrixBoundary.PlainFor(a);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
                return a.TensorEngine.Linalg.Eigvalsh(a, UPLO);
            }
        }
    }
}
