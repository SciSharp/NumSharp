using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise power with array exponents: x1 ** x2
        /// </summary>
        public override NDArray Power(NDArray lhs, NDArray rhs, Type dtype)
            => Power(lhs, rhs, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise power with array exponents: x1 ** x2
        /// Uses ExecuteBinaryOp with BinaryOp.Power for broadcasting support.
        /// </summary>
        public override NDArray Power(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null)
        {
            var result = ExecuteBinaryOp(lhs, rhs, BinaryOp.Power);
            if (typeCode.HasValue && result.typecode != typeCode.Value)
                return Cast(result, typeCode.Value, copy: false);
            return result;
        }
    }
}
