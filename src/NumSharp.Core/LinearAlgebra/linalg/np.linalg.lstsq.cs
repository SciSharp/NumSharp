using System;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            ///     Least-squares solution to <c>a x = b</c>.
            /// </summary>
            /// <param name="a">Coefficient matrix, <c>(M, N)</c>. Exactly 2-D — lstsq does NOT stack.</param>
            /// <param name="b">Ordinate values, <c>(M,)</c> or <c>(M, K)</c>.</param>
            /// <param name="rcond">
            ///     Singular values smaller than <c>rcond</c> times the largest are treated as zero.
            /// </param>
            /// <returns>
            ///     <c>(solution, residuals, rank, singularValues)</c> — NumPy's four-tuple. The
            ///     residuals are EMPTY unless the system is overdetermined and full-rank.
            /// </returns>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
            ///     <para>
            ///     Unlike most of this module, <c>lstsq</c> takes a single matrix rather than a stack
            ///     — a 3-D operand is rejected with the "must be two-dimensional" message rather than
            ///     the "at least two-dimensional" one.
            ///     </para>
            /// </remarks>
            /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves (an empty operand or a zero-size batch needs none).</exception>
            /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
            /// <exception cref="LinAlgError">An operand that is not two-dimensional, or <paramref name="b"/>'s rows differ
            ///     from <paramref name="a"/>'s.</exception>
            /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
            /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal).</exception>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            [NDScoped] // the class boundary's temporaries; the Linalg member scopes the computation's own
            public static (NDArray Solution, NDArray Residuals, NDArray Rank, NDArray SingularValues) lstsq(
                NDArray a, NDArray b, double? rcond = null)
            {
                if (a is null) throw new ArgumentNullException(nameof(a));
                // NumPy takes its wrap from b (`a, _ = _makearray(a); b, wrap = _makearray(b)`) and returns
                // wrap(x), wrap(resids), rank, s: a numpy.matrix b gives the solution and the residuals as matrices
                // (probed 2.4.2: a 2 x 2 matrix and a 1 x 0 matrix), a matrix a with a plain b plain arrays.
                if (NDMatrixBoundary.IsMatrix(b) && NDMatrixBoundary.Sees(b))
                    return NDMatrixBoundary.WrappedFirstTwoOfFour((a, b, rcond), static s => lstsq(s.a, s.b, s.rcond));
                using var plain = NDMatrixBoundary.PlainFor(a, b);   // a matrix a with a plain b: computed as plain arrays
                return a.TensorEngine.Linalg.Lstsq(a, b, rcond);
            }
        }
    }
}
