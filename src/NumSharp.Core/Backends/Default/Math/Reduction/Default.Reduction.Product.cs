using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceProduct(NDArray arr, int? axis_, bool keepdims = false, NPTypeCode? typeCode = null)
        {
            var shape = arr.Shape;

            if (shape.IsEmpty)
                return NDArray.Scalar((typeCode ?? arr.typecode).GetOneValue());

            if (shape.size == 0)
            {
                if (axis_ == null)
                {
                    var r = NDArray.Scalar((typeCode ?? arr.typecode).GetOneValue());
                    if (keepdims) { var ks = new int[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(new Shape(ks)); }
                    return r;
                }
                var axis = NormalizeAxis(axis_.Value, arr.ndim);
                var resultShape = Shape.GetAxis(shape, axis);
                var result = np.ones(new Shape(resultShape), typeCode ?? arr.GetTypeCode.GetAccumulatingType());
                if (keepdims)
                {
                    var ks = new int[arr.ndim];
                    for (int d = 0, sd = 0; d < arr.ndim; d++) ks[d] = (d == axis) ? 1 : resultShape[sd++];
                    result.Storage.Reshape(new Shape(ks));
                }
                return result;
            }

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
                return HandleScalarReduction(arr, keepdims, typeCode, null);

            if (axis_ == null)
            {
                var result = prod_elementwise_il(arr, typeCode);
                var r = NDArray.Scalar(result);
                if (keepdims) { var ks = new int[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(new Shape(ks)); }
                else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1) r.Storage.Reshape(Shape.Scalar);
                return r;
            }

            var axis2 = NormalizeAxis(axis_.Value, arr.ndim);
            var outputType = typeCode ?? arr.GetTypeCode.GetAccumulatingType();

            if (shape[axis2] == 1)
                return HandleTrivialAxisReduction(arr, axis2, keepdims, outputType, null);

            return ExecuteAxisReduction(arr, axis2, keepdims, outputType, null, ReductionOp.Prod);
        }
    }
}
