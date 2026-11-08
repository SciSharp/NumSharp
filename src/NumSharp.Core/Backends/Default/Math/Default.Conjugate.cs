using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     Element-wise complex conjugate (np.conjugate / np.conj) using the IL-generated
        ///     unary kernel. Identity at every real dtype (int/char/half/single/double/decimal — the
        ///     value is copied unchanged, dtype preserved; bool, which has no loop, resolves to int8
        ///     without dtype=, as NumPy's ufunc does); for Complex it flips the sign of the imaginary
        ///     part. Handles every memory layout (contiguous / strided / transposed /
        ///     broadcast / sliced) through the shared unary dispatch, and honours out=/where=.
        /// </summary>
        public override NDArray Conjugate(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            // NumPy's `conjugate` ufunc has NO bool loop: a plain bool input resolves to the smallest
            // matching loop — int8 — so the result is int8 (values 0/1), NOT bool (probed against
            // NumPy 2.4.2). Promote here to match. An explicit dtype= selects the loop instead, exactly
            // as NumPy does (e.g. conjugate(bool, dtype: Double) -> float64).
            if (dtype is null && nd.typecode == NPTypeCode.Boolean)
                nd = nd.astype(NPTypeCode.SByte);
            NPTypeCode? typeCode = UfuncLoopCode(dtype, "conjugate", nd, null, @out, where);
            // Every other dtype has a loop (unlike negative, which rejects bool), so there is no dtype
            // rejection here. dtype= (when given) selects the loop and preserves that dtype.
            return ExecuteUnaryOp(nd, UnaryOp.Conjugate, typeCode, @out, where);
        }
    }
}
