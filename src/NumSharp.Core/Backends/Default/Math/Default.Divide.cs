using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise division using IL-generated kernels.
        /// Supports all 144 type combinations with automatic type promotion.
        /// </summary>
        public override NDArray Divide(in NDArray lhs, in NDArray rhs)
        {
            return ExecuteBinaryOp(in lhs, in rhs, BinaryOp.Divide);
        }
    }
}
