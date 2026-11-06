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
            /// <exception cref="TypeError">A text operand (NumPy's gufunc input-cast refusal).</exception>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            [NDScoped] // what the call leaves (the Linalg member and the woven subclass path scope their own)
            public static (NDArray Solution, NDArray Residuals, NDArray Rank, NDArray SingularValues) lstsq(
                NDArray a, NDArray b, double? rcond = null)
            {
                if (a is null) throw new ArgumentNullException(nameof(a));
                return a.TensorEngine.Linalg.Lstsq(a, b, rcond);
            }
        }
    }
}
