using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceAMin(NDArray arr, int? axis_, bool keepdims = false, NPTypeCode? typeCode = null)
        {
            var shape = arr.Shape;

            // Handle empty arrays - need to check axis-specific behavior
            if (shape.IsEmpty || shape.size == 0)
            {
                return HandleEmptyArrayMinMaxReduction(arr, axis_, keepdims, typeCode, "minimum");
            }

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
                return HandleScalarReduction(arr, keepdims, typeCode, null);

            if (axis_ == null)
            {
                var result = min_elementwise_il(arr, typeCode);
                var r = NDArray.Scalar(result);
                if (keepdims) { var ks = new int[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(new Shape(ks)); }
                else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1) r.Storage.Reshape(Shape.Scalar);
                return r;
            }

            var axis = NormalizeAxis(axis_.Value, arr.ndim);
            var outputType = typeCode ?? arr.GetTypeCode;

            if (shape[axis] == 1)
                return HandleTrivialAxisReduction(arr, axis, keepdims, outputType, null);

            return ExecuteAxisReduction(arr, axis, keepdims, outputType, null, ReductionOp.Min);
        }
    }
}
