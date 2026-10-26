using System;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            ///     Multiplicative inverse of a matrix, or of each matrix in a stack.
            /// </summary>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
            ///     <para>
            ///     NumPy does not call an explicit inversion routine: it solves <c>a x = I</c> with
            ///     <c>gesv</c>, which is why a singular operand surfaces as
            ///     <c>LinAlgError("Singular matrix")</c> from the factorisation rather than from a
            ///     determinant test.
            ///     </para>
            /// </remarks>
            /// <param name="a">The square matrix, or a stack of them.</param>
            /// <returns>The inverse, a fresh array (float64 for bool / integer operands).</returns>
            /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
            /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
            /// <exception cref="LinAlgError">Fewer than two axes, a non-square operand, or a singular matrix.</exception>
            /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
            /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal).</exception>
            // Scope: the class boundary's temporaries; LinalgEngine.Inv reclaims the ToCommon dtype-cast temp. Matches
            // the scoped siblings pinv/tensorinv.
            [NDScoped]
            public static NDArray inv(NDArray a)
            {
                // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
                if (NDMatrixBoundary.Sees(a))
                    return NDMatrixBoundary.Wrapped(a, static s => inv(s));
                return LinalgEngine.Inv(a);
            }

            /// <summary>
            ///     Moore-Penrose pseudo-inverse, computed from the singular value decomposition.
            /// </summary>
            /// <param name="a">The matrix, or a stack of them.</param>
            /// <param name="rcond">
            ///     Singular values below <c>rcond * largest</c> are treated as zero. NumPy's default
            ///     is <c>1e-15</c>.
            /// </param>
            /// <param name="hermitian">
            ///     Assume the operand is Hermitian, allowing the cheaper symmetric eigensolver.
            /// </param>
            /// <param name="rtol">
            ///     The Array-API spelling of <paramref name="rcond"/>. Supplying both is an error, as
            ///     upstream.
            /// </param>
            /// <returns>The pseudo-inverse, shape <c>(..., N, M)</c>.</returns>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html</remarks>
            /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves (an empty operand or a zero-size batch needs none).</exception>
            /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>);
            ///     both <paramref name="rcond"/> and <paramref name="rtol"/> set.</exception>
            /// <exception cref="TypeError">A text operand, or a float16, decimal or char operand.</exception>
            /// <exception cref="LinAlgError">Fewer than two axes, or a non-square operand with <paramref name="hermitian"/>.</exception>
            [NDScoped] // the class boundary's temporaries; LinalgEngine.Pinv scopes the computation's own
            public static NDArray pinv(NDArray a, double? rcond = null, bool hermitian = false, double? rtol = null)
            {
                // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
                if (NDMatrixBoundary.Sees(a))
                    return NDMatrixBoundary.Wrapped((a, rcond, hermitian, rtol), static s => pinv(s.a, s.rcond, s.hermitian, s.rtol));
                return LinalgEngine.Pinv(a, rcond, hermitian, rtol);
            }

            /// <summary>
            ///     Inverse of an N-dimensional array with respect to a <see cref="tensordot(NDArray, NDArray, int)"/>
            ///     contraction over its last <c>a.ndim - ind</c> axes.
            /// </summary>
            /// <param name="a">The tensor to invert.</param>
            /// <param name="ind">
            ///     How many LEADING axes form the "row" side. Must be positive.
            /// </param>
            /// <returns>The inverse, shape <c>a.shape[ind:] + a.shape[:ind]</c>.</returns>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensorinv.html</remarks>
            /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
            /// <exception cref="ValueError"><paramref name="ind"/> is not positive, or the reshape to a square matrix is
            ///     impossible.</exception>
            /// <exception cref="LinAlgError">The reshaped matrix is not square, or is singular.</exception>
            [NDScoped] // the facade keeps its scope over the call; LinalgEngine.Tensorinv scopes the computation's own
            public static NDArray tensorinv(NDArray a, int ind = 2)
            {
                using var plain = NDMatrixBoundary.PlainFor(a);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
                return LinalgEngine.Tensorinv(a, ind);
            }
        }
    }
}
