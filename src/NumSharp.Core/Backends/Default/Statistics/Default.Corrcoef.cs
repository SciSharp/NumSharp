// NumPy's np.corrcoef (numpy/lib/_function_base_impl.py) as DefaultEngine's member: NumPy's statements over cov, in
// NumPy's order and with its in-place divisions and clips. The facade is Statistics/np.corrcoef.cs.

using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        [NDScoped /* cov's matrix is the result; the diagonal, the deviations and the clip bounds are reclaimed */]
        public override NDArray Corrcoef(NDArray x, NDArray y = null, bool rowvar = true, DType dtype = null)
        {
            // c = cov(x, y, rowvar, dtype=dtype): np.cov's member on x's engine, bias / ddof / the weights at NumPy's
            // defaults. It also brings cov's validation and its verbatim errors (ValueError "m has more than 2 dimensions",
            // ...), text included: cov follows NumPy's statements over text up to its first reduction, which refuses it, or
            // returns the empty float64 (0, 0) result for no variables, which corrcoef then divides like NumPy.
            NDArray c = x.TensorEngine.Cov(x, y, rowvar, false, null, null, null, dtype);

            // d = diag(c) (np.diag's body, the array object's). cov returns a 0-d SCALAR for a single variable, and the
            // diagonal of a 0-d array is refused (NumPy's ValueError, NumSharp's ArgumentException): NumPy's "scalar
            // covariance" branch, `return c / c`: 1 for a good value, NaN for a NaN, infinite or zero covariance.
            NDArray d;
            try
            {
                d = NDArray.diag(c, 0);
            }
            catch (ArgumentException)
            {
                return c / c;
            }

            // stddev = sqrt(d.real). The diagonal holds the variances, which are real even for a complex input, so the real
            // lane (the array itself for a real one) keeps stddev real.
            NDArray realD = NDArray.real_(d);
            NDArray stddev = realD.TensorEngine.Sqrt(realD, null, null, null);

            // c /= stddev[:, None]; c /= stddev[None, :] - NumPy divides in place both times, so the ufunc's out= structure
            // is kept. The complex division is bit-exact with NumPy (ComplexDivideNumPy ports CDOUBLE_divide), so a residual
            // complex corrcoef divergence comes from cov's managed complex product, not from here.
            c.TensorEngine.Divide(c, stddev.reshape(stddev.size, 1), null, c, null);
            c.TensorEngine.Divide(c, stddev.reshape(1, stddev.size), null, c, null);

            // Clip the real (and, for complex, the imaginary) lane to [-1, 1] IN PLACE, as NumPy does: the real lane is c
            // itself for a real array and a write-through float64 view for a complex one, so the clipped values land in c
            // either way. np.iscomplexobj's test is the storage type code.
            ClipToUnit(NDArray.real_(c));
            if (c.GetTypeCode == NPTypeCode.Complex)
                ClipToUnit(NDArray.imag_(c));

            return c;

            // Clips a float lane to [-1, 1] in place: np.clip(lane, -1, 1, out=lane)'s ufunc call (its facade's keyword rule
            // and read-only check have nothing to refuse: the lane is a view of cov's fresh, writeable result). The bounds are
            // cast to the lane's dtype so the clip stays in that precision: a C# -1 / 1 is a STRONG int32 scalar, which under
            // NEP 50 would promote a float32 lane to float64 and then fail the same_kind out cast, where NumPy's Python -1 / 1
            // are WEAK ints that take the operand's dtype - which this reproduces; +-1 is exact in every float type.
            static void ClipToUnit(NDArray lane)
            {
                NDArray lo = NDArray.Scalar(-1).astype(lane.typecode);
                NDArray hi = NDArray.Scalar(1).astype(lane.typecode);
                lane.TensorEngine.ClipNDArray(lane, lo, hi, null, lane, null);
            }
        }
    }
}
