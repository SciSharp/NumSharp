using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceArgMax(NDArray arr, int? axis_, bool keepdims = false)
        {
            //in order to iterate an axis:
            //consider arange shaped (1,2,3,4) when we want to summarize axis 1 (2nd dimension which its value is 2)
            //the size of the array is [1, 2, n, m] all shapes after 2nd multiplied gives size
            //the size of what we need to reduce is the size of the shape of the given axis (shape[axis])
            var shape = arr.Shape;

            if (shape.IsEmpty)
                throw new ArgumentException("attempt to get argmax of an empty sequence");

            // Handle empty arrays (size == 0) with axis reduction
            // NumPy: np.argmax(np.zeros((0,3)), axis=0) raises ValueError (reducing along zero-size axis)
            // NumPy: np.argmax(np.zeros((0,3)), axis=1) returns array([], dtype=int64) with shape (0,)
            if (shape.size == 0)
            {
                if (axis_ == null)
                {
                    // No axis specified - raise error
                    throw new ArgumentException("attempt to get argmax of an empty sequence");
                }

                // Axis specified - check if reducing along zero-size axis
                var emptyAxis = axis_.Value;
                while (emptyAxis < 0)
                    emptyAxis = arr.ndim + emptyAxis;
                if (emptyAxis >= arr.ndim)
                    throw new ArgumentOutOfRangeException(nameof(axis_));

                if (shape[emptyAxis] == 0)
                {
                    // Reducing along a zero-size axis - raise error
                    throw new ArgumentException("attempt to get argmax of an empty sequence");
                }

                // Reducing along non-zero axis - return empty Int64 array with reduced shape
                var resultShape = Shape.GetAxis(shape, emptyAxis);
                var result = np.empty(new Shape(resultShape), NPTypeCode.Int64);

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
                var r = NDArray.Scalar(0L);  // Int64 for NumPy 2.x alignment
                if (keepdims)
                {
                    var keepdimsShape = new long[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++)
                        keepdimsShape[i] = 1;
                    r.Storage.Reshape(new Shape(keepdimsShape));
                }
                return r;
            }

            if (axis_ == null)
            {
                // Use IL-generated kernels for element-wise reduction
                var r = NDArray.Scalar(argmax_elementwise_il(arr));
                if (keepdims)
                {
                    // NumPy: keepdims preserves the number of dimensions, all set to 1
                    var keepdimsShape = new long[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++)
                        keepdimsShape[i] = 1;
                    r.Storage.Reshape(new Shape(keepdimsShape));
                }
                return r;
            }

            var axis = axis_.Value;
            while (axis < 0)
                axis = arr.ndim + axis; //handle negative axis

            if (axis >= arr.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis));

            if (shape[axis] == 1)
            {
                //if the given div axis is 1, result is all zeros
                if (keepdims)
                {
                    // Keep the axis but reduce to size 1 (it's already 1)
                    return np.zeros(shape.dimensions, NPTypeCode.Int64);
                }
                return np.squeeze_fast(np.zeros(shape.dimensions, NPTypeCode.Int64), axis);
            }

            //handle keepdims - prepare output shape
            Shape axisedShape = Shape.GetAxis(shape, axis);
            Shape outputShape = axisedShape;
            if (keepdims)
            {
                // Insert a 1 at the axis position
                var keepdimsShapeDims = new long[arr.ndim];
                int srcIdx = 0;
                for (int i = 0; i < arr.ndim; i++)
                {
                    if (i == axis)
                        keepdimsShapeDims[i] = 1;
                    else
                        keepdimsShapeDims[i] = axisedShape.dimensions[srcIdx++];
                }
                outputShape = new Shape(keepdimsShapeDims);
            }

            // Use IL kernel for axis reduction
            return ExecuteAxisArgReduction(arr, axis, keepdims, outputShape, axisedShape, ReductionOp.ArgMax);
        }

        /// <summary>
        /// Execute axis ArgMax/ArgMin reduction using IL kernels.
        /// </summary>
        private unsafe NDArray ExecuteAxisArgReduction(NDArray arr, int axis, bool keepdims, Shape outputShape, Shape axisedShape, ReductionOp op)
        {
            var shape = arr.Shape;
            var inputType = arr.GetTypeCode;

            // ArgMax/ArgMin always output Int64
            var key = new AxisReductionKernelKey(inputType, NPTypeCode.Int64, op, shape.IsContiguous && axis == arr.ndim - 1);
            var kernel = ILKernelGenerator.TryGetAxisReductionKernel(key);

            if (kernel == null)
                throw new InvalidOperationException($"IL kernel not available for ArgMax/ArgMin axis reduction. Ensure ILKernelGenerator.Enabled is true. Type: {inputType}");

            // Use IL kernel path
            var ret = new NDArray(NPTypeCode.Int64, axisedShape, false);
            long axisSize = shape.dimensions[axis];
            long outputSize = ret.size > 0 ? ret.size : 1;
            byte* inputAddr = (byte*)arr.Address + shape.offset * arr.dtypesize;

            fixed (long* inputStrides = shape.strides)
            fixed (long* inputDims = shape.dimensions)
            fixed (long* outputStrides = ret.Shape.strides)
            {
                kernel((void*)inputAddr, (void*)ret.Address, inputStrides, inputDims, outputStrides, axis, axisSize, arr.ndim, outputSize);
            }

            if (keepdims)
                ret.Storage.Reshape(outputShape);

            return ret;
        }

    }
}
