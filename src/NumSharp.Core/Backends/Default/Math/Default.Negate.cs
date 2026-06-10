using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise negation using IL-generated kernels.
        /// </summary>
        public override NDArray Negate(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            // NumPy rejects boolean negative (unary `-` / np.negative) at LOOP
            // resolution: there is no negative ufunc loop for the bool dtype.
            // An explicit dtype= picks the loop, so negative(bool, dtype=f64)
            // is legal ([-1., -0.]) while negative(f64, dtype=bool) raises the
            // same TypeError as plain negative(bool) — both probed on 2.4.2.
            // Callers that want a boolean flip use `~` (np.invert) or
            // np.logical_not.
            var loopType = typeCode ?? nd.GetTypeCode;
            if (loopType == NPTypeCode.Boolean)
                throw new System.NotSupportedException(
                    "The numpy boolean negative, the `-` operator, is not supported, " +
                    "use the `~` operator or the logical_not function instead.");

            // dtype= runs the loop in that dtype: the input must reach it via
            // a same_kind cast (negative(f64, dtype=i32) raises, probed).
            if (typeCode.HasValue)
                ValidateUnaryInputCast(nd.GetTypeCode, typeCode.Value, "negative");

            return ExecuteUnaryOp(nd, UnaryOp.Negate, typeCode, @out, where);
        }
    }
}
