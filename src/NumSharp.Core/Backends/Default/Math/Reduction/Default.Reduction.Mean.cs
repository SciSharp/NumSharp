using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceMean(NDArray arr, int? axis_, bool keepdims = false, NPTypeCode? typeCode = null)
        {
            var shape = arr.Shape;

            if (shape.IsEmpty)
                return NDArray.Scalar(double.NaN);

            if (shape.size == 0)
            {
                if (axis_ == null)
                {
                    var r = NDArray.Scalar(double.NaN);
                    if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(new Shape(ks)); }
                    return r;
                }
                var axis = NormalizeAxis(axis_.Value, arr.ndim);
                var resultShape = Shape.GetAxis(shape, axis);
                var outputType = typeCode ?? NPTypeCode.Double;
                NDArray result;
                if (shape[axis] == 0)
                {
                    result = np.empty(new Shape(resultShape), outputType);
                    for (long i = 0; i < result.size; i++) result.SetAtIndex(double.NaN, i);
                }
                else result = np.empty(new Shape(resultShape), outputType);
                if (keepdims)
                {
                    var ks = new long[arr.ndim];
                    for (int d = 0, sd = 0; d < arr.ndim; d++) ks[d] = (d == axis) ? 1 : resultShape[sd++];
                    result.Storage.Reshape(new Shape(ks));
                }
                return result;
            }

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
            {
                var val = arr.GetAtIndex(0);
                var outputType = typeCode ?? NPTypeCode.Double;
                var r = NDArray.Scalar(Converts.ChangeType(val, outputType));
                if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(new Shape(ks)); }
                return r;
            }

            if (axis_ == null)
            {
                var result = mean_elementwise_il(arr, typeCode);
                var r = NDArray.Scalar(result);
                if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(new Shape(ks)); }
                else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1) r.Storage.Reshape(Shape.Scalar);
                return r;
            }

            var axis2 = NormalizeAxis(axis_.Value, arr.ndim);
            var outputType2 = typeCode ?? NPTypeCode.Double;

            if (shape[axis2] == 1)
                return HandleTrivialAxisReduction(arr, axis2, keepdims, outputType2, null);

            return ExecuteAxisReduction(arr, axis2, keepdims, outputType2, null, ReductionOp.Mean);
        }

        /// <summary>
        /// Element-wise mean for typed result. Compatibility method for Std/Var.
        /// </summary>
        public T MeanElementwise<T>(NDArray arr, NPTypeCode? typeCode) where T : unmanaged
        {
            var result = mean_elementwise_il(arr, typeCode);
            return (T)Converts.ChangeType(result, typeof(T).GetTypeCode());
        }
    }
}
