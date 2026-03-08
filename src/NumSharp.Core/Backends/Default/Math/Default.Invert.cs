using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Invert(in NDArray nd, Type dtype) => Invert(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise bitwise NOT using IL-generated kernels.
        /// For integers: computes ~x (ones complement).
        /// For booleans: computes logical NOT (NumPy behavior).
        /// </summary>
        public override NDArray Invert(in NDArray nd, NPTypeCode? typeCode = null)
        {
            // NumPy treats boolean invert as logical NOT, not bitwise NOT
            // ~True = False, ~False = True (not ~1 = 0xFE)
            if (nd.typecode == NPTypeCode.Boolean)
            {
                return ExecuteUnaryOp(in nd, UnaryOp.LogicalNot, NPTypeCode.Boolean);
            }

            // For integer types: use bitwise NOT (~x)
            return ExecuteUnaryOp(in nd, UnaryOp.BitwiseNot, typeCode ?? nd.typecode);
        }
    }
}
