using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return Pearson product-moment correlation coefficients.
        ///
        ///     The relationship between the correlation coefficient matrix <c>R</c> and the
        ///     covariance matrix <c>C</c> (see <see cref="cov"/>) is
        ///     <c>R_ij = C_ij / sqrt(C_ii * C_jj)</c>. The values of <c>R</c> are between
        ///     <c>-1</c> and <c>1</c>, inclusive.
        /// </summary>
        /// <param name="x">
        ///     A 1-D or 2-D array containing multiple variables and observations. Each row of
        ///     <paramref name="x"/> represents a variable, each column a single observation of all
        ///     those variables (see <paramref name="rowvar"/>).
        /// </param>
        /// <param name="y">An additional set of variables and observations, same shape as <paramref name="x"/>.</param>
        /// <param name="rowvar">
        ///     If true (default) each row is a variable with observations in the columns. Otherwise
        ///     the relationship is transposed: each column is a variable, rows are observations.
        /// </param>
        /// <param name="dtype">Data-type of the result. By default the result has at least float64 precision.</param>
        /// <returns>The correlation coefficient matrix of the variables.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <exception cref="ValueError">More than 2 dimensions, and the other refusals of <see cref="cov"/> that its
        ///     <c>m</c> / <c>y</c> / <c>rowvar</c> / <c>dtype</c> can reach (<see cref="TensorEngine.Corrcoef"/> lists
        ///     them).</exception>
        /// <exception cref="IncorrectShapeException">Numeric variables with different observation counts.</exception>
        /// <exception cref="TypeError">Text that cov's mean refuses.</exception>
        /// <exception cref="DTypePromotionError"><c>StringDType</c> variables without <paramref name="dtype"/>.</exception>
        /// <exception cref="ArgumentException">An integer or bool <paramref name="dtype"/>.</exception>
        /// <remarks>
        ///     Port of NumPy 2.4.2's <c>numpy/lib/_function_base_impl.corrcoef</c> — a thin wrapper
        ///     over <see cref="cov"/>. Like NumPy 2.x this exposes ONLY <c>x</c>/<c>y</c>/<c>rowvar</c>/
        ///     <c>dtype</c>; the long-deprecated <c>bias</c>/<c>ddof</c> parameters were removed and are
        ///     not offered. All dtype/promotion/edge behaviour is inherited from <see cref="cov"/>, so
        ///     the result is bit-identical to NumPy for float64/complex128 (within the managed GEMM's
        ///     rounding) and within that GEMM's ULP tolerance for float32/float16 — the same parity
        ///     profile as <see cref="cov"/> itself. Due to floating-point rounding the diagonal may not
        ///     be exactly 1 and off-diagonal magnitudes may nudge past 1; the real (and, for complex
        ///     input, imaginary) parts are therefore clipped to <c>[-1, 1]</c>, exactly as NumPy does.
        ///     The statements are <see cref="TensorEngine.Corrcoef"/>, on <paramref name="x"/>'s engine.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
        /// </remarks>
        public static NDArray corrcoef(NDArray x, NDArray y = null, bool rowvar = true, DType dtype = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(x, y);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (x is null) throw new ArgumentNullException(nameof(x));
            return x.TensorEngine.Corrcoef(x, y, rowvar, dtype);
        }
    }
}
