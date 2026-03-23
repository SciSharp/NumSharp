using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceAdd(NDArray arr, int? axis_, bool keepdims = false, NPTypeCode? typeCode = null, NDArray @out = null)
        {
            var shape = arr.Shape;

            if (shape.IsEmpty)
            {
                var defaultVal = (typeCode ?? arr.typecode).GetDefaultValue();
                if (@out != null) { @out.SetAtIndex(defaultVal, 0); return @out; }
                return NDArray.Scalar(defaultVal);
            }

            if (shape.size == 0)
                return HandleEmptyArrayReduction(arr, axis_, keepdims, typeCode, @out, ReductionOp.Sum);

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
                return HandleScalarReduction(arr, keepdims, typeCode, @out);

            if (axis_ == null)
                return HandleElementWiseSum(arr, keepdims, typeCode, @out);

            var axis = NormalizeAxis(axis_.Value, arr.ndim);
            var outputType = typeCode ?? arr.GetTypeCode.GetAccumulatingType();

            if (shape[axis] == 1)
                return HandleTrivialAxisReduction(arr, axis, keepdims, outputType, @out);

            return ExecuteAxisReduction(arr, axis, keepdims, outputType, @out, ReductionOp.Sum);
        }

        private NDArray HandleElementWiseSum(NDArray arr, bool keepdims, NPTypeCode? typeCode, NDArray @out)
        {
            var result = sum_elementwise_il(arr, typeCode);
            if (@out != null) { @out.SetAtIndex(result, 0); return @out; }
            var r = NDArray.Scalar(result);
            if (keepdims) { var ks = new int[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(new Shape(ks)); }
            else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1) r.Storage.Reshape(Shape.Scalar);
            return r;
        }

        private unsafe NDArray ExecuteAxisReduction(NDArray arr, int axis, bool keepdims, NPTypeCode outputType, NDArray @out, ReductionOp op)
        {
            var shape = arr.Shape;
            var inputType = arr.GetTypeCode;
            var key = new AxisReductionKernelKey(inputType, outputType, op, shape.IsContiguous && axis == arr.ndim - 1);
            var kernel = ILKernelGenerator.TryGetAxisReductionKernel(key);
            if (kernel == null)
                throw new NotSupportedException($"Axis reduction kernel not available for {op}({inputType}) -> {outputType}.");

            var outputDims = new int[arr.ndim - 1];
            for (int d = 0, od = 0; d < arr.ndim; d++) if (d != axis) outputDims[od++] = shape.dimensions[d];

            var outputShape = outputDims.Length > 0 ? new Shape(outputDims) : Shape.Scalar;
            NDArray result;
            if (@out != null) { if (@out.Shape != outputShape) throw new IncorrectShapeException($"Output shape mismatch"); result = @out; }
            else result = new NDArray(outputType, outputShape, false);

            int axisSize = shape.dimensions[axis];
            int outputSize = result.size > 0 ? result.size : 1;
            byte* inputAddr = (byte*)arr.Address + shape.offset * arr.dtypesize;

            fixed (int* inputStrides = shape.strides)
            fixed (int* inputDims = shape.dimensions)
            fixed (int* outputStrides = result.Shape.strides)
            {
                kernel((void*)inputAddr, (void*)result.Address, inputStrides, inputDims, outputStrides, axis, axisSize, arr.ndim, outputSize);
            }

            if (keepdims)
            {
                var ks = new int[arr.ndim];
                for (int d = 0, sd = 0; d < arr.ndim; d++) ks[d] = (d == axis) ? 1 : result.shape[sd++];
                result.Storage.Reshape(new Shape(ks));
            }
            return result;
        }

        /// <summary>
        /// Handle empty array min/max reductions.
        /// NumPy behavior:
        /// - np.min([]) raises ValueError (no identity for min/max)
        /// - np.min(zeros((0,3)), axis=0) raises ValueError (reducing along empty dimension)
        /// - np.min(zeros((0,3)), axis=1) returns [] with shape (0,) (output is also empty)
        /// </summary>
        private NDArray HandleEmptyArrayMinMaxReduction(NDArray arr, int? axis_, bool keepdims, NPTypeCode? typeCode, string opName)
        {
            var shape = arr.Shape;

            // No axis specified - always throw for empty arrays (no identity element for min/max)
            if (axis_ == null)
                throw new ArgumentException($"zero-size array to reduction operation {opName} which has no identity");

            var axis = NormalizeAxis(axis_.Value, arr.ndim);

            // If the axis being reduced has size 0, we're reducing over an empty dimension
            // which results in an error (no values to take min/max of)
            if (shape.dimensions[axis] == 0)
                throw new ArgumentException($"zero-size array to reduction operation {opName} which has no identity");

            // If the axis being reduced has size > 0, but the result would be empty,
            // return an empty array of the correct shape
            var resultShape = Shape.GetAxis(shape, axis);
            var outputType = typeCode ?? arr.GetTypeCode;
            var result = new NDArray(outputType, new Shape(resultShape), false);

            if (keepdims)
            {
                var ks = new int[arr.ndim];
                for (int d = 0, sd = 0; d < arr.ndim; d++)
                    ks[d] = (d == axis) ? 1 : resultShape[sd++];
                result.Storage.Reshape(new Shape(ks));
            }
            return result;
        }

        private NDArray HandleEmptyArrayReduction(NDArray arr, int? axis_, bool keepdims, NPTypeCode? typeCode, NDArray @out, ReductionOp op)
        {
            var shape = arr.Shape;
            if (axis_ == null)
            {
                var defaultVal = (typeCode ?? arr.typecode).GetDefaultValue();
                if (@out != null) { @out.SetAtIndex(defaultVal, 0); return @out; }
                var r = NDArray.Scalar(defaultVal);
                if (keepdims) { var ks = new int[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(new Shape(ks)); }
                return r;
            }
            var axis = NormalizeAxis(axis_.Value, arr.ndim);
            var resultShape = Shape.GetAxis(shape, axis);
            var outputType = typeCode ?? arr.GetTypeCode.GetAccumulatingType();
            var result = np.zeros(new Shape(resultShape), outputType);
            if (keepdims)
            {
                var ks = new int[arr.ndim];
                for (int d = 0, sd = 0; d < arr.ndim; d++) ks[d] = (d == axis) ? 1 : resultShape[sd++];
                result.Storage.Reshape(new Shape(ks));
            }
            if (@out != null) { np.copyto(@out, result); return @out; }
            return result;
        }

        private NDArray HandleScalarReduction(NDArray arr, bool keepdims, NPTypeCode? typeCode, NDArray @out)
        {
            var r = typeCode.HasValue ? Cast(arr, typeCode.Value, true) : arr.Clone();
            if (@out != null) { @out.SetAtIndex(r.GetAtIndex(0), 0); return @out; }
            if (keepdims) { var ks = new int[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(new Shape(ks)); }
            else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1) r.Storage.Reshape(Shape.Scalar);
            return r;
        }

        private NDArray HandleTrivialAxisReduction(NDArray arr, int axis, bool keepdims, NPTypeCode outputType, NDArray @out)
        {
            if (@out != null) return null;
            var shape = arr.Shape;
            int[] resultDims;
            if (keepdims) { resultDims = (int[])shape.dimensions.Clone(); resultDims[axis] = 1; }
            else { resultDims = new int[arr.ndim - 1]; for (int d = 0, rd = 0; d < arr.ndim; d++) if (d != axis) resultDims[rd++] = shape.dimensions[d]; }
            if (resultDims.Length == 0)
            {
                var v = arr.GetAtIndex(0);
                if (outputType != arr.GetTypeCode) v = (ValueType)Converts.ChangeType(v, outputType);
                return NDArray.Scalar(v);
            }
            var result = new NDArray(outputType, new Shape(resultDims), false);
            if (outputType == arr.GetTypeCode) for (int i = 0; i < result.size; i++) result.SetAtIndex(arr.GetAtIndex(i), i);
            else for (int i = 0; i < result.size; i++) result.SetAtIndex(Converts.ChangeType(arr.GetAtIndex(i), outputType), i);
            return result;
        }

        private static int NormalizeAxis(int axis, int ndim)
        {
            while (axis < 0) axis = ndim + axis;
            if (axis >= ndim) throw new ArgumentOutOfRangeException(nameof(axis));
            return axis;
        }
    }
}
