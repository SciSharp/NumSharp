using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise modulo using IL-generated kernels.
        /// Supports all 144 type combinations with automatic type promotion.
        /// </summary>
        public override NDArray Mod(NDArray lhs, NDArray rhs)
        {
            return ExecuteBinaryOp(lhs, rhs, BinaryOp.Mod);
        }
    }
}
