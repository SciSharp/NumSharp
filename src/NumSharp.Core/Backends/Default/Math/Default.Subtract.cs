using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise subtraction using IL-generated kernels.
        /// Supports all 144 type combinations with automatic type promotion.
        /// </summary>
        public override NDArray Subtract(NDArray lhs, NDArray rhs)
        {
            // NumPy rejects boolean subtraction: there is no subtract ufunc loop
            // for the bool dtype, so `bool - bool` (both operands boolean) raises.
            // Mixed bool + numeric promotes to the numeric type and subtracts fine,
            // so the guard is specifically "both operands boolean".
            if (lhs.GetTypeCode == NPTypeCode.Boolean && rhs.GetTypeCode == NPTypeCode.Boolean)
                throw new NotSupportedException(
                    "numpy boolean subtract, the `-` operator, is not supported, " +
                    "use the bitwise_xor, the `^` operator, or the logical_xor function instead.");

            return ExecuteBinaryOp(lhs, rhs, BinaryOp.Subtract);
        }
    }
}
