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

            // B2: Complex mean axis needs a dedicated path — the Double-based kernel drops imag.
            if (!typeCode.HasValue && inputTc == NPTypeCode.Complex)
                return MeanAxisComplex(arr, axis2, keepdims);

            // B16: Half mean axis computes in Double then casts back to preserve Half dtype.
            bool needsCast = !typeCode.HasValue && inputTc == NPTypeCode.Half;
            var outputType2 = needsCast ? NPTypeCode.Double : (typeCode ?? NPTypeCode.Double);

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
        /// B2: NumPy-parity Complex mean along an axis. Iterator-based since the IL kernel path
        /// routes through Double accumulators and drops the imaginary component.
        /// </summary>
        private NDArray MeanAxisComplex(NDArray arr, int axis, bool keepdims)
        {
            var shape = arr.Shape;
            Shape axisedShape = Shape.GetAxis(shape, axis);
            var ret = new NDArray(NPTypeCode.Complex, axisedShape, false);
            var iterAxis = new NDCoordinatesAxisIncrementor(ref shape, axis);
            var iterRet = new ValueCoordinatesIncrementor(ref axisedShape);
            var iterIndex = iterRet.Index;
            var slices = iterAxis.Slices;

            do
            {
                var slice = arr[slices];
                var sum = System.Numerics.Complex.Zero;
                var it = slice.AsIterator<System.Numerics.Complex>();
                long n = 0;
                while (it.HasNext()) { sum += it.MoveNext(); n++; }
                var mean = n > 0 ? sum / (double)n : new System.Numerics.Complex(double.NaN, double.NaN);
                ret.SetAtIndex(mean, iterIndex[0]);
            } while (iterAxis.Next() != null && iterRet.Next() != null);

            if (keepdims) ret.Storage.ExpandDimension(axis);
            return ret;
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
