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
        ///     https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
        /// </remarks>
        public static NDArray corrcoef(NDArray x, NDArray y = null, bool rowvar = true, NPTypeCode? dtype = null)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));

            // c = cov(x, y, rowvar, dtype=dtype). Delegating here also inherits cov's validation and
            // its verbatim error taxonomy (e.g. ValueError "m has more than 2 dimensions").
            NDArray c = np.cov(x, y, rowvar, dtype: dtype);

            // d = diag(c). cov returns a 0-d SCALAR for a single variable, and np.diag raises on a 0-d
            // input (NumPy's ValueError, NumSharp's ArgumentException) — NumPy's "scalar covariance"
            // branch, `return c / c`: 1 for a good value, nan for a nan/inf/0 covariance.
            NDArray d;
            try
            {
                d = np.diag(c);
            }
            catch (ArgumentException)
            {
                return c / c;
            }

            // stddev = sqrt(d.real). The diagonal holds the variances, which are real even for complex
            // input, so `.real` (a no-op for a real array) keeps stddev real.
            NDArray stddev = np.sqrt(np.real(d));

            // c /= stddev[:, None]; c /= stddev[None, :] — NumPy performs BOTH divisions in
            // place, so preserve that ufunc/out structure. The underlying complex division still
            // inherits the tightly bounded npy_cdivide-vs-System.Numerics 1-ULP difference recorded
            // by the oracle; changing allocation alone cannot remove that algorithmic delta.
            np.divide(c, stddev.reshape(stddev.size, 1), @out: c);
            np.divide(c, stddev.reshape(1, stddev.size), @out: c);

            // Clip real (and, for complex, imaginary) lanes to [-1, 1] IN PLACE, exactly as NumPy does:
            // np.real(c) is c itself for a real array and a write-through float64 lane view for a
            // complex array, so the clipped values land back in c either way.
            ClipToUnit(np.real(c));
            if (np.iscomplexobj(c))
                ClipToUnit(np.imag(c));

            return c;

            // Clip a float lane to [-1, 1] in place. The bounds are cast to the lane's dtype so clipping
            // stays in that precision: a C# `-1`/`1` literal is a STRONG int32 scalar, which under NEP50
            // would promote a float32 lane to float64 and then fail clip's same_kind out-cast. NumPy's
            // Python `-1`/`1` are WEAK ints that adopt the operand dtype — which this reproduces, and
            // ±1 is exactly representable in every float type, so the values are unchanged.
            static void ClipToUnit(NDArray lane)
            {
                NDArray lo = NDArray.Scalar(-1).astype(lane.typecode);
                NDArray hi = NDArray.Scalar(1).astype(lane.typecode);
                np.clip(lane, lo, hi, @out: lane);
            }
        }
    }
}
