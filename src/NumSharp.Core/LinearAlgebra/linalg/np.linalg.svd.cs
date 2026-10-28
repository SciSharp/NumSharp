using System;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            ///     Singular value decomposition — <c>a = U diag(S) Vh</c>.
            /// </summary>
            /// <param name="a">The matrix, or a stack of them.</param>
            /// <param name="full_matrices">
            ///     True (the default) makes U and Vh full square factors; false truncates them to
            ///     <c>K = min(M, N)</c> columns/rows.
            /// </param>
            /// <param name="compute_uv">
            ///     False returns the singular values alone — <c>U</c> and <c>Vh</c> are then null.
            /// </param>
            /// <param name="hermitian">
            ///     Assume the operand is Hermitian, allowing the cheaper symmetric eigensolver.
            /// </param>
            /// <returns><c>(U, S, Vh)</c>, the singular values descending.</returns>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
            ///     <para>
            ///     Singular values come back in DESCENDING order. This is the routine behind
            ///     <see cref="pinv"/>, <see cref="matrix_rank"/>, <see cref="cond"/> and the
            ///     spectral and nuclear matrix norms — which is why all of those stop here too.
            ///     </para>
            /// </remarks>
            /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves (an empty operand or a zero-size batch needs none).</exception>
            /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
            /// <exception cref="LinAlgError">Fewer than two axes, or a non-square operand with <paramref name="hermitian"/>.</exception>
            /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
            /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal).</exception>
            [NDScoped] // the class boundary's temporaries; the Linalg member scopes the computation's own
            public static (NDArray U, NDArray S, NDArray Vh) svd(NDArray a, bool full_matrices = true,
                bool compute_uv = true, bool hermitian = false)
            {
                // a numpy.matrix operand: NumPy returns wrap(u), the plain singular values and wrap(vh)
                if (NDMatrixBoundary.Sees(a))
                    return NDMatrixBoundary.WrappedTriple((a, full_matrices, compute_uv, hermitian),
                        static s => svd(s.a, s.full_matrices, s.compute_uv, s.hermitian), wrap0: true, wrap1: false, wrap2: true);
                return EngineOf(a).Linalg.Svd(a, full_matrices, compute_uv, hermitian);
            }

            /// <summary>
            ///     Singular values of a matrix, descending — the Array-API spelling of
            ///     <c>svd(x, compute_uv: false)</c>.
            /// </summary>
            /// <param name="x">The matrix, or a stack of them.</param>
            /// <returns>The singular values, descending.</returns>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.svdvals.html</remarks>
            /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves (an empty operand or a zero-size batch needs none).</exception>
            /// <exception cref="LinAlgError">Fewer than two axes.</exception>
            /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
            /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal).</exception>
            [NDScoped] // the facade keeps its scope over the call; the Linalg member scopes the computation's own
            public static NDArray svdvals(NDArray x)
            {
                using var plain = NDMatrixBoundary.PlainFor(x);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
                return EngineOf(x).Linalg.Svdvals(x);
            }

            /// <summary>
            ///     Rank of a matrix — the number of singular values greater than the tolerance.
            /// </summary>
            /// <param name="A">The matrix, or a stack of them.</param>
            /// <param name="tol">
            ///     Absolute threshold. Defaults to <c>S.max() * max(M, N) * eps</c>.
            /// </param>
            /// <param name="hermitian">Assume the operand is Hermitian.</param>
            /// <param name="rtol">Relative threshold; supplying it together with <paramref name="tol"/> is an error.</param>
            /// <returns>The rank: NumPy's <c>int(not all(A == 0))</c> below two axes, else the count per matrix.</returns>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html</remarks>
            /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves (an empty operand or a zero-size batch needs none).</exception>
            /// <exception cref="ValueError">Both <paramref name="tol"/> and <paramref name="rtol"/> set.</exception>
            /// <exception cref="LinAlgError">A non-square operand with <paramref name="hermitian"/>.</exception>
            /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
            /// <exception cref="ArgumentException">A text operand of two or more axes (NumPy's gufunc input-cast refusal).</exception>
            [NDScoped] // the facade keeps its scope over the call; the Linalg member scopes the computation's own
            public static NDArray matrix_rank(NDArray A, double? tol = null, bool hermitian = false,
                double? rtol = null)
            {
                using var plain = NDMatrixBoundary.PlainFor(A);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
                return EngineOf(A).Linalg.MatrixRank(A, tol, hermitian, rtol);
            }

            /// <summary>
            ///     Condition number of a matrix in the given norm.
            /// </summary>
            /// <param name="x">The matrix, or a stack of them.</param>
            /// <param name="p">
            ///     The order: <c>null</c> (default) and <c>±2</c> use singular values;
            ///     <c>±1</c>, <c>±inf</c> and <c>"fro"</c> are defined as
            ///     <c>norm(x, p) * norm(inv(x), p)</c>.
            /// </param>
            /// <returns>The condition number, real: a 0-d array for one matrix (NumPy's scalar), one per matrix of a stack;
            ///     inf for a singular matrix.</returns>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html
            ///     <para>
            ///     Every order needs a factorisation: the singular-value ones (null, ±2) and <c>"nuc"</c> the SVD, which needs
            ///     a matrix backend for a non-empty operand; ±1, ±inf and <c>"fro"</c> the inverse, which the managed LU
            ///     computes without one. As in NumPy the inverse runs in float64 (complex128 for a complex operand) whatever
            ///     the result dtype, and a singular matrix's inverse is NaN rather than an error, so its condition number is
            ///     inf, inside a stack too.
            ///     </para>
            /// </remarks>
            /// <exception cref="OpenBlasMissingBackendException">An SVD-based order (null, ±2, <c>"nuc"</c>) over a non-empty
            ///     operand that no matrix backend serves (an empty operand or a zero-size batch needs none).</exception>
            /// <exception cref="TypeError">An array or list <paramref name="p"/>: NumPy's <c>p in {2, -2}</c> hashes it
            ///     (<c>unhashable type: 'numpy.ndarray'</c> / <c>'list'</c>), after the empty-operand check; a float16,
            ///     decimal or char operand.</exception>
            /// <exception cref="LinAlgError">An empty operand, a non-square one for an inverse-based order, or
            ///     <c>"nuc"</c> of a singular matrix (the SVD of its NaN inverse: <c>SVD did not converge</c>).</exception>
            /// <exception cref="ValueError">An order the norm refuses.</exception>
            /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal).</exception>
            [NDScoped] // the facade keeps its scope over the call; the Linalg member scopes the computation's own
            public static NDArray cond(NDArray x, object p = null)
            {
                using var plain = NDMatrixBoundary.PlainFor(x);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
                return EngineOf(x).Linalg.Cond(x, p);
            }
        }
    }
}
