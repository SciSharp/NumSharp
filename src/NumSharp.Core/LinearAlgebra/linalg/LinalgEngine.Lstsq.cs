using System;

namespace NumSharp
{
    internal static partial class LinalgEngine
    {
        /// <summary>
        ///     The body of <c>np.linalg.lstsq</c>: the 1-D <paramref name="b"/> promotion, <c>_assert_2d</c>, the row
        ///     check, <c>_commonType</c>, the gufunc's text refusal, the default cutoff, the <c>nrhs == 0</c> padding,
        ///     <see cref="TensorEngine.Lstsq"/>, and NumPy's tail (the squeeze, the discarded residuals, the casts).
        /// </summary>
        /// <param name="a">Coefficient matrix, <c>(M, N)</c> (plain: the facade applied the class boundary).</param>
        /// <param name="b">Ordinate values, <c>(M,)</c> or <c>(M, K)</c>.</param>
        /// <param name="rcond">The cutoff ratio; null for NumPy's <c>eps * max(M, N)</c>.</param>
        /// <returns><c>(solution, residuals, rank, singularValues)</c>: the residuals EMPTY unless the system is
        ///     overdetermined and full-rank.</returns>
        /// <exception cref="LinAlgError">An operand that is not two-dimensional (after the promotion), or <c>b</c>'s
        ///     rows differ from <c>a</c>'s (<c>Incompatible dimensions</c>).</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand (<c>_commonType</c>).</exception>
        /// <exception cref="ArgumentException">A text operand (the gufunc's input cast).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        [NDScoped] // reclaims the column-promoted b, the ToCommon casts and the nrhs==0 padding temps; the 4-tuple is yielded
        internal static (NDArray Solution, NDArray Residuals, NDArray Rank, NDArray SingularValues) Lstsq(
            NDArray a, NDArray b, double? rcond)
        {
            // A 1-D b is promoted to a single column, exactly as NumPy's `b = b[:, newaxis]`; the
            // solution is squeezed back at the end. _assert_2d then rejects any higher rank.
            bool is1d = b.ndim == 1;
            var b2 = is1d ? np.expand_dims(b, -1) : b;

            Assert2d(a, b2);

            long m = a.Shape.dimensions[0];
            long n = a.Shape.dimensions[1];
            long nRhs = b2.Shape.dimensions[1];
            if (b2.Shape.dimensions[0] != m)
                throw new LinAlgError("Incompatible dimensions");

            var common = CommonType(a, b2);
            // NumPy calls _umath_linalg.lstsq(a, b, rcond): its input casts refuse text (input 0, then input 1).
            ThrowIfTextGufunc("lstsq", a, b2);
            // The condition number and singular values are always real (NumPy's result_real_t).
            var realCommon = common == NPTypeCode.Complex ? NPTypeCode.Double : common;

            // NumPy 2.x default: `finfo(t).eps * max(n, m)` where t is the COMPUTE type — and linalg
            // always computes in double, so eps is float64's (2.220446049250313e-16) regardless of
            // input width. (Passing -1 uses machine precision alone; any explicit value is honoured.)
            double cutoff = rcond ?? (2.220446049250313e-16 * Math.Max(n, m));

            var aC = ToCommon(a, common);
            var bC = ToCommon(b2, common);

            // LAPACK can't take nrhs == 0 — NumPy pads b with one column, then trims x/resids back.
            bool padded = nRhs == 0;
            if (padded)
                bC = np.zeros(new Shape(m, 1), common);

            var (x, resids, rank, s) = aC.TensorEngine.Lstsq(aC, bC, cutoff);

            if (padded)
            {
                x = np.zeros(Shape.Adopt(new long[] {n, 0}), common);            // x[..., :0]
                resids = np.zeros(Shape.Adopt(new long[] {0}), NPTypeCode.Double); // resids[..., :0]
            }

            if (is1d)
                x = np.reshape(x, new long[] {n});                             // squeeze(axis=-1)

            // Residuals are documented meaningful ONLY for a full-rank over-determined system; NumPy
            // discards them otherwise, replacing them with an empty array of the real result type.
            long rankVal = Convert.ToInt64(rank.GetAtIndex(0));
            if (rankVal != n || m <= n)
                resids = np.zeros(Shape.Adopt(new long[] {0}), realCommon);
            else
                resids = resids.astype(realCommon);

            s = s.astype(realCommon);
            x = x.astype(common);

            return (x, resids, rank, s);
        }
    }
}
