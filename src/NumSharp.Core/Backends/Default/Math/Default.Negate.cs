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
            // Boolean negation is logical NOT, not arithmetic negation
            // Use LogicalNot which properly handles non-contiguous arrays
            if (nd.GetTypeCode == NPTypeCode.Boolean)
            {
                return ExecuteUnaryOp(nd, UnaryOp.LogicalNot);
            }

            return ExecuteUnaryOp(nd, UnaryOp.Negate);
        }
    }
}
