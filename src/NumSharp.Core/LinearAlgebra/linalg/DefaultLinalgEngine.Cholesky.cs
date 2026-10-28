using System;

namespace NumSharp.Backends
{
    public partial class DefaultLinalgEngine
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: <c>_assert_stacked_square</c>, <c>_commonType</c>, the text refusal of the
        ///     <c>cholesky_lo</c> / <c>cholesky_up</c> gufunc's input cast, then <see cref="CholeskyKernel"/> at the loop dtype.
        /// </remarks>
        [NDScoped] // reclaims the ToCommon cast temp
        public override NDArray Cholesky(NDArray a, bool upper = false)
        {
            AssertStackedSquare(a);
            var common = CommonType(a);
            // NumPy calls _umath_linalg.cholesky_lo / cholesky_up: its input cast refuses text.
            ThrowIfTextGufunc(upper ? "cholesky_up" : "cholesky_lo", a);
            return CholeskyKernel(ToCommon(a, common), upper);
        }
    }
}
