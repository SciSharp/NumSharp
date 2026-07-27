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
            /// <exception cref="NotSupportedException">No LAPACK backend serves these operands.</exception>
            public static NDArray cholesky(NDArray a, bool upper = false)
            {
                AssertStackedSquare(a);
                var common = CommonType(a);
                return a.TensorEngine.Cholesky(ToCommon(a, common), upper);
            }
        }
    }
}
