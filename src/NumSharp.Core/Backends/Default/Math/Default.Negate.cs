using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise negation using IL-generated kernels.
        /// </summary>
        public override NDArray Negate(NDArray nd)
        {
            // NumPy rejects boolean negative (unary `-` / np.negative): there is
            // no negative ufunc loop for the bool dtype. Callers that want a
            // boolean flip use the `~` operator (np.invert) or np.logical_not.
            if (nd.GetTypeCode == NPTypeCode.Boolean)
                throw new System.NotSupportedException(
                    "The numpy boolean negative, the `-` operator, is not supported, " +
                    "use the `~` operator or the logical_not function instead.");

            return ExecuteUnaryOp(nd, UnaryOp.Negate);
        }
    }
}
