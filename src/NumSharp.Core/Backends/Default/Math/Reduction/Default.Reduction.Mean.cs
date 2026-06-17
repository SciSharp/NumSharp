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
                // B2/B16: NumPy mean preserves float/complex input dtype (half→half, complex→complex).
                // Only integer inputs promote to float64. GetComputingType() enforces this rule.
                var outputType = typeCode ?? arr.GetTypeCode.GetComputingType();
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
            var inputTc = arr.GetTypeCode;

            // B2: Complex mean axis is handled by the NpyIter REDUCE path (ExecuteAxisReduction
            // → ExecuteAxisReductionNpyIter), which runs a one-pass complex Sum kernel then
            // divides by the axis length — both components preserved (NumPy parity). This
            // replaced the per-output-row-allocating MeanAxisComplex (15–45× slower).

            // B16: Half mean axis computes in Double then casts back to preserve Half dtype.
            bool needsCast = !typeCode.HasValue && inputTc == NPTypeCode.Half;
            // NumPy parity: mean preserves float input dtype (float32→float32, float64→float64);
            // integer inputs promote to float64. GetComputingType() encodes exactly this rule and
            // also keeps InputType == AccumulatorType for floats, which is what unlocks the
            // SIMD same-type axis-reduction kernel. Forcing Double here was a 2576× regression
            // on mean(float32, axis=0) because it dropped into the scalar promoted helper.
            var outputType2 = needsCast ? NPTypeCode.Double : (typeCode ?? inputTc.GetComputingType());

            NDArray result2;
            if (shape[axis2] == 1)
                result2 = HandleTrivialAxisReduction(arr, axis2, keepdims, outputType2, null);
            else
                result2 = ExecuteAxisReduction(arr, axis2, keepdims, outputType2, null, ReductionOp.Mean);

            if (needsCast)
                result2 = Cast(result2, inputTc, copy: true);
            return result2;
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
