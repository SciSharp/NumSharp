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
            if (nd.GetTypeCode == NPTypeCode.Boolean)
            {
                return NegateBoolean(nd);
            }

            return ExecuteUnaryOp(in nd, UnaryOp.Negate);
        }

        /// <summary>
        /// Boolean negation (logical NOT).
        /// </summary>
        private unsafe NDArray NegateBoolean(in NDArray nd)
        {
            if (nd.size == 0)
                return nd.Clone();

            var result = new NDArray(nd.dtype, nd.Shape.Clean(), false);
            var outAddr = (bool*)result.Address;
            var inAddr = (bool*)nd.Address;
            var len = nd.size;

            for (int i = 0; i < len; i++)
                outAddr[i] = !inAddr[i];

            return result;
        }
    }
}
