using System;

namespace NumSharp.Backends
{
    public partial class DefaultLinalgEngine
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: the 1-D <c>b</c> promotion, <c>_assert_2d</c>, the row check, <c>_commonType</c>, the text
        ///     refusal of the <c>lstsq</c> gufunc's input casts, the default cutoff, the <c>nrhs == 0</c> padding,
        ///     <see cref="LstsqKernel"/>, and NumPy's tail (the squeeze, the discarded residuals, the casts). The
        ///     promotion, the allocations and the squeeze are the array object's members (np.expand_dims', np.zeros' and
        ///     np.reshape's), not their facades.
        /// </remarks>
        [NDScoped] // reclaims the column-promoted b, the ToCommon casts and the nrhs==0 padding temps; the 4-tuple is yielded
        public override (NDArray Solution, NDArray Residuals, NDArray Rank, NDArray SingularValues) Lstsq(
            NDArray a, NDArray b, double? rcond = null)
        {
            // A 1-D b is promoted to a single column, exactly as NumPy's `b = b[:, newaxis]`; the
            // solution is squeezed back at the end. _assert_2d then rejects any higher rank. np.expand_dims' member:
            // np.linalg.lstsq's plain region (or its matrix rule's) already kept the facade's class rules from firing.
            bool is1d = b.ndim == 1;
            var b2 = is1d ? NDArray.ExpandDims(b, -1) : b;

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

            // LAPACK can't take nrhs == 0 — NumPy pads b with one column, then trims x/resids back. Each np.zeros below
            // is its own allocation (a zero-filled owner), without the call up into np.*.
            bool padded = nRhs == 0;
            if (padded)
                bC = new NDArray(common, new Shape(m, 1), true);

            var (x, resids, rank, s) = LstsqKernel(aC, bC, cutoff);

            if (padded)
            {
                x = new NDArray(common, Shape.Adopt(new long[] {n, 0}), true);            // x[..., :0]
                resids = new NDArray(NPTypeCode.Double, Shape.Adopt(new long[] {0}), true); // resids[..., :0]
            }

            if (is1d)
                x = x.reshape(new long[] {n});                             // squeeze(axis=-1), np.reshape's member

            // Residuals are documented meaningful ONLY for a full-rank over-determined system; NumPy
            // discards them otherwise, replacing them with an empty array of the real result type.
            long rankVal = Convert.ToInt64(rank.GetAtIndex(0));
            if (rankVal != n || m <= n)
                resids = new NDArray(realCommon, Shape.Adopt(new long[] {0}), true);
            else
                resids = resids.astype(realCommon);

            s = s.astype(realCommon);
            x = x.astype(common);

            return (x, resids, rank, s);
        }
    }
}
