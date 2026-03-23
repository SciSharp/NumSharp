using System.Runtime.CompilerServices;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    /// <summary>
    /// Bitwise operation dispatch using IL-generated kernels.
    /// </summary>
    public partial class DefaultEngine
    {
        /// <summary>
        /// Execute bitwise AND operation.
        /// </summary>
        public override NDArray BitwiseAnd(NDArray lhs, NDArray rhs)
        {
            return ExecuteBinaryOp(lhs, rhs, BinaryOp.BitwiseAnd);
        }

        /// <summary>
        /// Execute bitwise OR operation.
        /// </summary>
        public override NDArray BitwiseOr(NDArray lhs, NDArray rhs)
        {
            return ExecuteBinaryOp(lhs, rhs, BinaryOp.BitwiseOr);
        }

        /// <summary>
        /// Execute bitwise XOR operation.
        /// </summary>
        public override NDArray BitwiseXor(NDArray lhs, NDArray rhs)
        {
            return ExecuteBinaryOp(lhs, rhs, BinaryOp.BitwiseXor);
        }
    }
}
