using System;

namespace NumSharp
{
    internal static partial class LinalgEngine
    {
        /// <summary>
        ///     The body of <c>np.linalg.cholesky</c>: <c>_assert_stacked_square</c>, <c>_commonType</c>, the gufunc's text
        ///     refusal, then <see cref="TensorEngine.Cholesky"/> at the loop dtype.
        /// </summary>
        /// <param name="a">The Hermitian positive-definite matrix, or a stack of them (plain: the facade applied the
        ///     class boundary).</param>
        /// <param name="upper">True for the upper factor <c>U</c> (<c>a = U* U</c>), false for the lower <c>L</c>.</param>
        /// <returns>The triangular factor, a fresh array of the loop dtype.</returns>
        /// <exception cref="LinAlgError">Fewer than two axes, a non-square operand, or a matrix that is not positive
        ///     definite (<c>Matrix is not positive definite</c>, from the factorisation).</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand (<c>_commonType</c>).</exception>
        /// <exception cref="ArgumentException">A text operand (the gufunc's input cast).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        [NDScoped] // reclaims the ToCommon cast temp
        internal static NDArray Cholesky(NDArray a, bool upper)
        {
            AssertStackedSquare(a);
            var common = CommonType(a);
            // NumPy calls _umath_linalg.cholesky_lo / cholesky_up: its input cast refuses text.
            ThrowIfTextGufunc(upper ? "cholesky_up" : "cholesky_lo", a);
            return a.TensorEngine.Cholesky(ToCommon(a, common), upper);
        }
    }
}
