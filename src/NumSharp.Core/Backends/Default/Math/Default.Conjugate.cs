using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     Element-wise complex conjugate (np.conjugate / np.conj) using the IL-generated
        ///     unary kernel. Identity at every real dtype (bool/int/char/half/single/double/decimal —
        ///     the value is copied unchanged, dtype preserved); for Complex it flips the sign of the
        ///     imaginary part. Handles every memory layout (contiguous / strided / transposed /
        ///     broadcast / sliced) through the shared unary dispatch, and honours out=/where=.
        /// </summary>
        public override NDArray Conjugate(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            // Conjugate has a loop at EVERY dtype (unlike negative, which rejects bool), so there is
            // no dtype rejection here. dtype= (when given) selects the loop and preserves that dtype.
            return ExecuteUnaryOp(nd, UnaryOp.Conjugate, typeCode, @out, where);
        }
    }
}
