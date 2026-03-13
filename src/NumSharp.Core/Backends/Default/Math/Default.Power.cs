using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise power with array exponents: x1 ** x2
        /// </summary>
        public override NDArray Power(in NDArray lhs, in NDArray rhs, Type dtype)
            => Power(lhs, rhs, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise power with array exponents: x1 ** x2
        /// Uses ExecuteBinaryOp with BinaryOp.Power for broadcasting support.
        /// </summary>
        public override NDArray Power(in NDArray lhs, in NDArray rhs, NPTypeCode? typeCode = null)
        {
            var result = ExecuteBinaryOp(in lhs, in rhs, BinaryOp.Power);
            if (typeCode.HasValue && result.typecode != typeCode.Value)
                return Cast(result, typeCode.Value, copy: false);
            return result;
        }

        public override NDArray Power(in NDArray lhs, in ValueType rhs, Type dtype) => Power(lhs, rhs, dtype?.GetTypeCode());

        public override NDArray Power(in NDArray lhs, in ValueType rhs, NPTypeCode? typeCode = null)
        {
            if (lhs.size == 0)
                return lhs.Clone();

            // Convert scalar exponent to NDArray and use ExecuteBinaryOp
            // Type promotion is handled in ExecuteBinaryOp for Power operation
            // The scalar is created with appropriate type to trigger correct promotion:
            // - C# int/long → preserve integer type for int^int
            // - C# float/double → promote to float64 for int^float
            var rhsArray = NDArray.Scalar(rhs);
            var result = ExecuteBinaryOp(in lhs, in rhsArray, BinaryOp.Power);

            if (typeCode.HasValue && result.typecode != typeCode.Value)
                return Cast(result, typeCode.Value, copy: false);
            return result;
        }
    }
}
