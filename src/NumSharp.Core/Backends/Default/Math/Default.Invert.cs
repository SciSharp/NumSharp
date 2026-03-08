using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Invert(in NDArray nd, Type dtype) => Invert(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise bitwise NOT using IL-generated kernels.
        /// Computes ~x (ones complement).
        /// </summary>
        public override NDArray Invert(in NDArray nd, NPTypeCode? typeCode = null)
        {
            // For invert, we keep the same type (it's a bitwise operation)
            return ExecuteUnaryOp(in nd, UnaryOp.BitwiseNot, typeCode ?? nd.typecode);
        }
    }
}
