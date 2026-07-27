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
            /// <exception cref="NotSupportedException">No LAPACK backend serves these operands.</exception>
            public static (NDArray Solution, NDArray Residuals, NDArray Rank, NDArray SingularValues) lstsq(
                NDArray a, NDArray b, double? rcond = null)
            {
                Assert2d(a);

                if (b.ndim != 1 && b.ndim != 2)
                    throw new LinAlgError($"{b.ndim}-dimensional array given. Array must be two-dimensional");

                long rows = a.Shape.dimensions[0];
                if (b.Shape.dimensions[0] != rows)
                    throw new LinAlgError("Incompatible dimensions");

                var common = CommonType(a, b);

                // NumPy's default rcond is machine epsilon times the larger side; the explicit value
                // is passed through untouched so a backend reproduces the same cutoff.
                double cutoff = rcond ?? -1d;

                return a.TensorEngine.Lstsq(ToCommon(a, common), ToCommon(b, common), cutoff);
            }
        }
    }
}
