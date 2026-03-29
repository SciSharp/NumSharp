using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray FloorDivide(NDArray lhs, NDArray rhs, System.Type dtype)
            => FloorDivide(lhs, rhs, dtype?.GetTypeCode());

        public override NDArray FloorDivide(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null)
        {
            // If typeCode specified, cast result after operation
            var result = ExecuteBinaryOp(lhs, rhs, BinaryOp.FloorDivide);
            if (typeCode.HasValue && result.typecode != typeCode.Value)
                return Cast(result, typeCode.Value, copy: false);
            return result;
        }
    }
}

