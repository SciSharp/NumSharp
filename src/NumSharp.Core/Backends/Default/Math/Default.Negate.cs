using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise negation using IL-generated kernels.
        /// </summary>
        public override NDArray Negate(in NDArray nd)
        {
            // Boolean negation is logical NOT, not arithmetic negation
            // Use LogicalNot which properly handles non-contiguous arrays
            if (nd.GetTypeCode == NPTypeCode.Boolean)
            {
                return ExecuteUnaryOp(in nd, UnaryOp.LogicalNot);
            }

            return ExecuteUnaryOp(in nd, UnaryOp.Negate);
        }
    }
}
