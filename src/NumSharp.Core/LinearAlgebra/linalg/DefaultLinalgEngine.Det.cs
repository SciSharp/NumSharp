using System;

namespace NumSharp.Backends
{
    public partial class DefaultLinalgEngine
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: <c>_assert_stacked_square</c>, <c>_commonType</c>, the text refusal of the <c>det</c>
        ///     gufunc's input cast, then <see cref="DetKernel"/> at the loop dtype.
        /// </remarks>
        // Scope: reclaims the ToCommon dtype-cast temp (minted only when `a` needs casting;
        // a passthrough input is never tracked), yielding the engine's fresh determinant scalar.
        [NDScoped]
        public override NDArray Det(NDArray a)
        {
            AssertStackedSquare(a);
            var common = CommonType(a);
            // NumPy calls _umath_linalg.det: its input cast refuses text, an empty operand included.
            ThrowIfTextGufunc("det", a);
            return DetKernel(ToCommon(a, common));
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: <c>_assert_stacked_square</c>, <c>_commonType</c>, the text refusal of the <c>slogdet</c>
        ///     gufunc's input cast, then <see cref="SlogdetKernel"/> at the loop dtype.
        /// </remarks>
        // Scope: reclaims the ToCommon dtype-cast temp; the (sign, logabsdet) tuple result is
        // yielded through the weaver's Returns<T1,T2> tuple egress (both engine-fresh).
        [NDScoped]
        public override (NDArray sign, NDArray logabsdet) Slogdet(NDArray a)
        {
            AssertStackedSquare(a);
            var common = CommonType(a);
            // NumPy calls _umath_linalg.slogdet: its input cast refuses text, an empty operand included.
            ThrowIfTextGufunc("slogdet", a);
            return SlogdetKernel(ToCommon(a, common));
        }
    }
}
