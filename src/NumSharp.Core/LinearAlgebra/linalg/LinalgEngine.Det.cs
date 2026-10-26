using System;

namespace NumSharp
{
    internal static partial class LinalgEngine
    {
        /// <summary>
        ///     The body of <c>np.linalg.det</c>: <c>_assert_stacked_square</c>, <c>_commonType</c>, the gufunc's text
        ///     refusal, then <see cref="TensorEngine.Det"/> at the loop dtype.
        /// </summary>
        /// <param name="a">The square matrix, or a stack of them.</param>
        /// <returns>The determinant (a 0-d array for one matrix, one per matrix of a stack) in the loop dtype.</returns>
        /// <exception cref="LinAlgError">Fewer than two axes, or a non-square operand.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand (<c>_commonType</c>).</exception>
        /// <exception cref="ArgumentException">A text operand (the gufunc's input cast).</exception>
        /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
        // Scope: reclaims the ToCommon dtype-cast temp (minted only when `a` needs casting;
        // a passthrough input is never tracked), yielding the engine's fresh determinant scalar.
        [NDScoped]
        internal static NDArray Det(NDArray a)
        {
            AssertStackedSquare(a);
            var common = CommonType(a);
            // NumPy calls _umath_linalg.det: its input cast refuses text, an empty operand included.
            ThrowIfTextGufunc("det", a);
            return a.TensorEngine.Det(ToCommon(a, common));
        }

        /// <summary>
        ///     The body of <c>np.linalg.slogdet</c>: <c>_assert_stacked_square</c>, <c>_commonType</c>, the gufunc's
        ///     text refusal, then <see cref="TensorEngine.Slogdet"/> at the loop dtype.
        /// </summary>
        /// <param name="a">The square matrix, or a stack of them.</param>
        /// <returns><c>(sign, logabsdet)</c>, such that <c>sign * exp(logabsdet)</c> is the determinant; a singular
        ///     matrix gives <c>(0, -inf)</c>, a complex operand a unit-modulus complex sign.</returns>
        /// <exception cref="LinAlgError">Fewer than two axes, or a non-square operand.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand (<c>_commonType</c>).</exception>
        /// <exception cref="ArgumentException">A text operand (the gufunc's input cast).</exception>
        /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
        // Scope: reclaims the ToCommon dtype-cast temp; the (sign, logabsdet) tuple result is
        // yielded through the weaver's Returns<T1,T2> tuple egress (both engine-fresh).
        [NDScoped]
        internal static (NDArray sign, NDArray logabsdet) Slogdet(NDArray a)
        {
            AssertStackedSquare(a);
            var common = CommonType(a);
            // NumPy calls _umath_linalg.slogdet: its input cast refuses text, an empty operand included.
            ThrowIfTextGufunc("slogdet", a);
            return a.TensorEngine.Slogdet(ToCommon(a, common));
        }
    }
}
