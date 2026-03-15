using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceStd(NDArray arr, int? axis_, bool keepdims = false, int? ddof = null, NPTypeCode? typeCode = null)
        {
            var shape = arr.Shape;
            if (shape.IsEmpty)
                return arr;

            // Handle empty arrays (size == 0) with axis reduction
            // NumPy: np.std(np.zeros((0,3)), axis=0) returns array([nan, nan, nan]) (reducing along zero-size axis)
            // NumPy: np.std(np.zeros((0,3)), axis=1) returns array([]) with shape (0,) (reducing along non-zero axis)
            if (arr.size == 0)
            {
                if (axis_ == null)
                {
                    // No axis specified - return NaN scalar
                    var r = NDArray.Scalar(double.NaN);
                    if (keepdims)
                    {
                        var keepdimsShape = new long[arr.ndim];
                        for (int i = 0; i < arr.ndim; i++)
                            keepdimsShape[i] = 1;
                        r.Storage.Reshape(new Shape(keepdimsShape));
                    }
                    return r;
                }

                // Axis specified - check if reducing along zero-size axis
                var emptyAxis = axis_.Value;
                while (emptyAxis < 0)
                    emptyAxis = arr.ndim + emptyAxis;
                if (emptyAxis >= arr.ndim)
                    throw new ArgumentOutOfRangeException(nameof(axis_));

                var resultShape = Shape.GetAxis(shape, emptyAxis);
                var emptyOutputType = typeCode ?? NPTypeCode.Double;

                NDArray result;
                if (shape[emptyAxis] == 0)
                {
                    // Reducing along a zero-size axis - return NaN filled array
                    result = np.empty(new Shape(resultShape), emptyOutputType);
                    for (long i = 0; i < result.size; i++)
                        result.SetAtIndex(double.NaN, i);
                }
                else
                {
                    // Reducing along non-zero axis - return empty array with reduced shape
                    result = np.empty(new Shape(resultShape), emptyOutputType);
                }

                if (keepdims)
                {
                    var keepdimsShape = new long[arr.ndim];
                    for (int d = 0, sd = 0; d < arr.ndim; d++)
                    {
                        if (d == emptyAxis)
                            keepdimsShape[d] = 1;
                        else
                            keepdimsShape[d] = resultShape[sd++];
                    }
                    result.Storage.Reshape(new Shape(keepdimsShape));
                }

                return result;
            }

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
            {
                // NumPy: std of single element with ddof=0 is 0.0
                // With ddof >= size, the divisor is <= 0, which produces NaN
                int _ddof = ddof ?? 0;
                double value = (arr.size - _ddof) <= 0 ? double.NaN : 0.0;
                var r = NDArray.Scalar(value);
                if (keepdims)
                {
                    // NumPy: keepdims preserves the number of dimensions, all set to 1
                    var keepdimsShape = new long[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++)
                        keepdimsShape[i] = 1;
                    r.Storage.Reshape(new Shape(keepdimsShape));
                }
                else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1)
                    r.Storage.Reshape(Shape.Scalar);
                return r;
            }

            if (axis_ == null)
            {
                var r = NDArray.Scalar(std_elementwise(arr, typeCode, ddof));
                if (keepdims)
                {
                    // NumPy: keepdims preserves the number of dimensions, all set to 1
                    var keepdimsShape = new long[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++)
                        keepdimsShape[i] = 1;
                    r.Storage.Reshape(new Shape(keepdimsShape));
                }
                else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1)
                    r.Storage.Reshape(Shape.Scalar);
                return r;
            }
            var axis = axis_.Value;
            while (axis < 0)
                axis = arr.ndim + axis; //handle negative axis

            if (axis >= arr.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis));

            if (shape[axis] == 1)
            {
                //if the given div axis is 1 - std of a single element is 0
                //Return zeros with the appropriate shape (NumPy behavior)
                if (keepdims)
                {
                    var keepdimsShapeDims = new long[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++)
                        keepdimsShapeDims[i] = (i == axis) ? 1 : shape[i];
                    return np.zeros(keepdimsShapeDims, typeCode ?? arr.GetTypeCode.GetComputingType());
                }
                return np.zeros(Shape.GetAxis(shape, axis), typeCode ?? arr.GetTypeCode.GetComputingType());
            }

            // IL-generated axis reduction fast path - handles all numeric types
            if (ILKernelGenerator.Enabled)
            {
                var ilResult = ExecuteAxisStdReductionIL(arr, axis, keepdims, typeCode ?? NPTypeCode.Double, ddof ?? 0);
                if (ilResult != null)
                    return ilResult;
            }

            // Fallback: iterator-based axis reduction (handles non-contiguous, broadcast, edge cases)
            return ExecuteAxisStdReductionFallback(arr, axis, keepdims, typeCode, ddof);
        }

        /// <summary>
        /// Fallback axis std reduction using iterators. Used when IL kernel not available.
        /// </summary>
        private NDArray ExecuteAxisStdReductionFallback(NDArray arr, int axis, bool keepdims, NPTypeCode? typeCode, int? ddof)
        {
            var shape = arr.Shape;
            Shape axisedShape = Shape.GetAxis(shape, axis);
            var retType = typeCode ?? arr.GetTypeCode.GetComputingType();

            var ret = new NDArray(retType, axisedShape, false);
            var iterAxis = new NDCoordinatesAxisIncrementor(ref shape, axis);
            var iterRet = new ValueCoordinatesIncrementor(ref axisedShape);
            var iterIndex = iterRet.Index;
            var slices = iterAxis.Slices;

            int _ddof = ddof ?? 0;

            // Use double accumulator for all types (sufficient precision)
            do
            {
                var slice = arr[slices];
                var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                double sum = 0;
                var iter = slice.AsIterator<double>();
                var moveNext = iter.MoveNext;
                var hasNext = iter.HasNext;

                while (hasNext())
                {
                    var a = moveNext() - xmean;
                    sum += a * a;
                }

                var std = Math.Sqrt(sum / (slice.size - _ddof));
                ret.SetDouble(Converts.ToDouble(std), iterIndex);
            } while (iterAxis.Next() != null && iterRet.Next() != null);

            if (keepdims)
                ret.Storage.ExpandDimension(axis);

            return ret;
        }

        public T StdElementwise<T>(NDArray arr, NPTypeCode? typeCode, int? ddof) where T : unmanaged
        {
            return Converts.ChangeType<T>(std_elementwise(arr, typeCode, ddof));
        }

        protected object std_elementwise(NDArray arr, NPTypeCode? typeCode, int? ddof)
        {
            if (arr.Shape.IsScalar || (arr.Shape.size == 1 && arr.Shape.NDim == 1))
            {
                // With ddof >= size, divisor is <= 0, which produces NaN
                int _ddof = ddof ?? 0;
                return (arr.size - _ddof) <= 0 ? double.NaN : 0.0;
            }

            var retType = typeCode ?? (arr.GetTypeCode).GetComputingType();

            // SIMD fast-path for contiguous arrays
            if (ILKernelGenerator.Enabled && arr.Shape.IsContiguous)
            {
                int _ddof = ddof ?? 0;
                double std;

                unsafe
                {
                    switch (arr.GetTypeCode)
                    {
                        case NPTypeCode.Single:
                            std = ILKernelGenerator.StdSimdHelper((float*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.Double:
                            std = ILKernelGenerator.StdSimdHelper((double*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.Byte:
                            std = ILKernelGenerator.StdSimdHelper((byte*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.Int16:
                            std = ILKernelGenerator.StdSimdHelper((short*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.UInt16:
                            std = ILKernelGenerator.StdSimdHelper((ushort*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.Int32:
                            std = ILKernelGenerator.StdSimdHelper((int*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.UInt32:
                            std = ILKernelGenerator.StdSimdHelper((uint*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.Int64:
                            std = ILKernelGenerator.StdSimdHelper((long*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.UInt64:
                            std = ILKernelGenerator.StdSimdHelper((ulong*)arr.Address, arr.size, _ddof);
                            break;
                        default:
                            goto fallback;
                    }
                }

                // Convert to requested return type
                return Converts.ChangeType(std, retType);

                fallback:;
            }

            // Fallback: iterator-based (handles non-contiguous, decimal, char, bool)
            return std_elementwise_fallback(arr, retType, ddof);
        }

        /// <summary>
        /// Fallback element-wise std using iterators.
        /// </summary>
        private object std_elementwise_fallback(NDArray arr, NPTypeCode retType, int? ddof)
        {
            int _ddof = ddof ?? 0;

            // Handle Decimal separately for precision
            if (arr.GetTypeCode == NPTypeCode.Decimal)
            {
                var iter = arr.AsIterator<decimal>();
                var moveNext = iter.MoveNext;
                var hasNext = iter.HasNext;
                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Decimal);

                decimal sum = 0;
                while (hasNext())
                {
                    var a = moveNext() - xmean;
                    sum += a * a;
                }

                var std = Utilities.DecimalMath.Sqrt(sum / ((decimal)arr.size - _ddof));
                return Converts.ChangeType(std, retType);
            }

            // All other types: iterate as double
            {
                var iter = arr.AsIterator<double>();
                var moveNext = iter.MoveNext;
                var hasNext = iter.HasNext;
                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                double sum = 0;
                while (hasNext())
                {
                    var a = moveNext() - xmean;
                    sum += a * a;
                }

                var std = Math.Sqrt(sum / (arr.size - _ddof));
                return Converts.ChangeType(std, retType);
            }
        }

        /// <summary>
        /// IL-generated axis standard deviation reduction. Returns null if kernel not available.
        /// </summary>
        private unsafe NDArray ExecuteAxisStdReductionIL(NDArray arr, int axis, bool keepdims, NPTypeCode outputType, int ddof)
        {
            var shape = arr.Shape;
            var inputType = arr.GetTypeCode;

            // Std axis reduction always outputs double for accuracy
            var key = new AxisReductionKernelKey(inputType, NPTypeCode.Double, ReductionOp.Std, shape.IsContiguous && axis == arr.ndim - 1);
            var kernel = ILKernelGenerator.TryGetAxisReductionKernel(key);

            if (kernel == null)
                return null;

            var outputDims = new long[arr.ndim - 1];
            for (int d = 0, od = 0; d < arr.ndim; d++)
                if (d != axis) outputDims[od++] = shape.dimensions[d];

            var outputShape = outputDims.Length > 0 ? new Shape(outputDims) : Shape.Scalar;
            var result = new NDArray(NPTypeCode.Double, outputShape, false);

            long axisSize = shape.dimensions[axis];
            long outputSize = result.size > 0 ? result.size : 1;
            byte* inputAddr = (byte*)arr.Address + shape.offset * arr.dtypesize;

            fixed (long* inputStrides = shape.strides)
            fixed (long* inputDims = shape.dimensions)
            fixed (long* outputStrides = result.Shape.strides)
            {
                // The kernel computes std with ddof=0 by default
                kernel((void*)inputAddr, (void*)result.Address, inputStrides, inputDims, outputStrides, axis, axisSize, arr.ndim, outputSize);

                // For ddof != 0, adjust: std_ddof = std_0 * sqrt(n / (n - ddof))
                if (ddof != 0)
                {
                    double* resultPtr = (double*)result.Address;
                    double adjustment = Math.Sqrt((double)axisSize / (axisSize - ddof));
                    for (long i = 0; i < outputSize; i++)
                        resultPtr[i] *= adjustment;
                }
            }

            // Convert to requested output type if different from double
            if (outputType != NPTypeCode.Double)
            {
                result = Cast(result, outputType, copy: true);
            }

            if (keepdims)
            {
                var ks = new long[arr.ndim];
                for (int d = 0, sd = 0; d < arr.ndim; d++)
                    ks[d] = (d == axis) ? 1 : (sd < outputDims.Length ? outputDims[sd++] : 1);
                result.Storage.Reshape(new Shape(ks));
            }

            return result;
        }
    }
}
