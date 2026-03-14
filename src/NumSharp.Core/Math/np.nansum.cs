using System;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
        /// </summary>
        /// <param name="a">Array containing numbers whose sum is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the sum is computed. The default is to compute the sum of the flattened array.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns>A new array containing the sum, with NaN values treated as zero.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.nansum.html
        /// Only applicable to float and double arrays.
        /// For integer arrays, this is equivalent to np.sum (no NaN values possible).
        /// </remarks>
        public static NDArray nansum(in NDArray a, int? axis = null, bool keepdims = false)
        {
            var arr = a;
            var shape = arr.Shape;

            // Non-float types: fall back to regular sum (no NaN possible)
            if (arr.GetTypeCode != NPTypeCode.Single && arr.GetTypeCode != NPTypeCode.Double)
                return sum(arr, axis: axis, keepdims: keepdims);

            if (shape.IsEmpty)
                return arr;

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
            {
                // Scalar case: check for NaN and return identity (0) if so
                var val = arr.GetAtIndex(0);
                switch (arr.GetTypeCode)
                {
                    case NPTypeCode.Single:
                        if (float.IsNaN((float)val))
                            return NDArray.Scalar(0f);
                        break;
                    case NPTypeCode.Double:
                        if (double.IsNaN((double)val))
                            return NDArray.Scalar(0.0);
                        break;
                }
                return a.Clone();
            }

            // Element-wise (axis=None): use SIMD-optimized path
            if (axis == null)
            {
                if (ILKernelGenerator.Enabled && arr.Shape.IsContiguous)
                {
                    object result;
                    unsafe
                    {
                        switch (arr.GetTypeCode)
                        {
                            case NPTypeCode.Single:
                                result = ILKernelGenerator.NanSumSimdHelperFloat((float*)arr.Address, arr.size);
                                break;
                            case NPTypeCode.Double:
                                result = ILKernelGenerator.NanSumSimdHelperDouble((double*)arr.Address, arr.size);
                                break;
                            default:
                                return sum(arr, axis: null, keepdims: keepdims);
                        }
                    }

                    var r = NDArray.Scalar(result);
                    if (keepdims)
                    {
                        var keepdimsShape = new int[arr.ndim];
                        for (int i = 0; i < arr.ndim; i++)
                            keepdimsShape[i] = 1;
                        r.Storage.Reshape(new Shape(keepdimsShape));
                    }
                    return r;
                }
                else
                {
                    // Non-contiguous: use scalar fallback
                    return nansum_scalar(arr, keepdims);
                }
            }
            else
            {
                // Axis reduction: use NaN-aware axis reduction kernel
                return ExecuteNanAxisReduction(arr, axis.Value, keepdims, ReductionOp.NanSum);
            }
        }

        /// <summary>
        /// Scalar fallback for nansum when SIMD is not available.
        /// </summary>
        private static NDArray nansum_scalar(NDArray arr, bool keepdims)
        {
            object result;

            switch (arr.GetTypeCode)
            {
                case NPTypeCode.Single:
                {
                    var iter = arr.AsIterator<float>();
                    float sum = 0f;
                    while (iter.HasNext())
                    {
                        float val = iter.MoveNext();
                        if (!float.IsNaN(val))
                            sum += val;
                    }
                    result = sum;
                    break;
                }
                case NPTypeCode.Double:
                {
                    var iter = arr.AsIterator<double>();
                    double sum = 0.0;
                    while (iter.HasNext())
                    {
                        double val = iter.MoveNext();
                        if (!double.IsNaN(val))
                            sum += val;
                    }
                    result = sum;
                    break;
                }
                default:
                    // Non-float types: regular sum
                    return sum(arr);
            }

            var r = NDArray.Scalar(result);
            if (keepdims)
            {
                var keepdimsShape = new int[arr.ndim];
                for (int i = 0; i < arr.ndim; i++)
                    keepdimsShape[i] = 1;
                r.Storage.Reshape(new Shape(keepdimsShape));
            }
            return r;
        }

        /// <summary>
        /// Execute a NaN-aware axis reduction.
        /// </summary>
        private static unsafe NDArray ExecuteNanAxisReduction(in NDArray arr, int axis, bool keepdims, ReductionOp op)
        {
            var shape = arr.Shape;

            // Normalize axis
            while (axis < 0) axis = arr.ndim + axis;
            if (axis >= arr.ndim) throw new ArgumentOutOfRangeException(nameof(axis));

            // Get kernel
            var inputType = arr.GetTypeCode;
            var key = new AxisReductionKernelKey(inputType, inputType, op, shape.IsContiguous && axis == arr.ndim - 1);
            var kernel = ILKernelGenerator.TryGetNanAxisReductionKernel(key);

            if (kernel == null)
            {
                // Fallback to scalar implementation
                return ExecuteNanAxisReductionScalar(arr, axis, keepdims, op);
            }

            // Create output array
            var outputDims = new int[arr.ndim - 1];
            for (int d = 0, od = 0; d < arr.ndim; d++)
                if (d != axis) outputDims[od++] = shape.dimensions[d];

            var outputShape = outputDims.Length > 0 ? new Shape(outputDims) : Shape.Scalar;
            var result = new NDArray(inputType, outputShape, false);

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
                for (int d = 0, sd = 0; d < arr.ndim; d++)
                    ks[d] = (d == axis) ? 1 : result.shape[sd++];
                result.Storage.Reshape(new Shape(ks));
            }
            return result;
        }

        /// <summary>
        /// Scalar fallback for NaN axis reduction.
        /// </summary>
        private static NDArray ExecuteNanAxisReductionScalar(NDArray arr, int axis, bool keepdims, ReductionOp op)
        {
            var shape = arr.Shape;
            var inputType = arr.GetTypeCode;

            // Create output shape
            var outputDims = new int[arr.ndim - 1];
            for (int d = 0, od = 0; d < arr.ndim; d++)
                if (d != axis) outputDims[od++] = shape.dimensions[d];

            var outputShape = outputDims.Length > 0 ? new Shape(outputDims) : Shape.Scalar;
            var result = new NDArray(inputType, outputShape, false);

            int axisSize = shape.dimensions[axis];
            int outputSize = result.size > 0 ? result.size : 1;

            // Compute output strides for iteration
            int[] outputDimStrides = new int[arr.ndim - 1 > 0 ? arr.ndim - 1 : 1];
            if (arr.ndim > 1)
            {
                outputDimStrides[arr.ndim - 2] = 1;
                for (int d = arr.ndim - 3; d >= 0; d--)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    int nextInputDim = (d + 1) >= axis ? d + 2 : d + 1;
                    outputDimStrides[d] = outputDimStrides[d + 1] * shape.dimensions[nextInputDim];
                }
            }

            // Iterate over output positions
            for (int outIdx = 0; outIdx < outputSize; outIdx++)
            {
                int remaining = outIdx;
                int inputBaseOffset = 0;

                for (int d = 0; d < arr.ndim - 1; d++)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    int coord = remaining / outputDimStrides[d];
                    remaining = remaining % outputDimStrides[d];
                    inputBaseOffset += coord * shape.strides[inputDim];
                }

                // Reduce along axis
                object reduced;
                switch (inputType)
                {
                    case NPTypeCode.Single:
                        reduced = ReduceNanAxisScalarFloat(arr, inputBaseOffset, axisSize, shape.strides[axis], op);
                        break;
                    case NPTypeCode.Double:
                        reduced = ReduceNanAxisScalarDouble(arr, inputBaseOffset, axisSize, shape.strides[axis], op);
                        break;
                    default:
                        reduced = 0;
                        break;
                }

                result.SetAtIndex(reduced, outIdx);
            }

            if (keepdims)
            {
                var ks = new int[arr.ndim];
                for (int d = 0, sd = 0; d < arr.ndim; d++)
                    ks[d] = (d == axis) ? 1 : result.shape[sd++];
                result.Storage.Reshape(new Shape(ks));
            }
            return result;
        }

        private static float ReduceNanAxisScalarFloat(NDArray arr, int baseOffset, int axisSize, int axisStride, ReductionOp op)
        {
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    float sum = 0f;
                    for (int i = 0; i < axisSize; i++)
                    {
                        float val = (float)arr.GetAtIndex(baseOffset + i * axisStride);
                        if (!float.IsNaN(val)) sum += val;
                    }
                    return sum;
                }
                case ReductionOp.NanProd:
                {
                    float prod = 1f;
                    for (int i = 0; i < axisSize; i++)
                    {
                        float val = (float)arr.GetAtIndex(baseOffset + i * axisStride);
                        if (!float.IsNaN(val)) prod *= val;
                    }
                    return prod;
                }
                case ReductionOp.NanMin:
                {
                    float minVal = float.PositiveInfinity;
                    bool foundNonNaN = false;
                    for (int i = 0; i < axisSize; i++)
                    {
                        float val = (float)arr.GetAtIndex(baseOffset + i * axisStride);
                        if (!float.IsNaN(val)) { if (val < minVal) minVal = val; foundNonNaN = true; }
                    }
                    return foundNonNaN ? minVal : float.NaN;
                }
                case ReductionOp.NanMax:
                {
                    float maxVal = float.NegativeInfinity;
                    bool foundNonNaN = false;
                    for (int i = 0; i < axisSize; i++)
                    {
                        float val = (float)arr.GetAtIndex(baseOffset + i * axisStride);
                        if (!float.IsNaN(val)) { if (val > maxVal) maxVal = val; foundNonNaN = true; }
                    }
                    return foundNonNaN ? maxVal : float.NaN;
                }
                default:
                    return 0f;
            }
        }

        private static double ReduceNanAxisScalarDouble(NDArray arr, int baseOffset, int axisSize, int axisStride, ReductionOp op)
        {
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    double sum = 0.0;
                    for (int i = 0; i < axisSize; i++)
                    {
                        double val = (double)arr.GetAtIndex(baseOffset + i * axisStride);
                        if (!double.IsNaN(val)) sum += val;
                    }
                    return sum;
                }
                case ReductionOp.NanProd:
                {
                    double prod = 1.0;
                    for (int i = 0; i < axisSize; i++)
                    {
                        double val = (double)arr.GetAtIndex(baseOffset + i * axisStride);
                        if (!double.IsNaN(val)) prod *= val;
                    }
                    return prod;
                }
                case ReductionOp.NanMin:
                {
                    double minVal = double.PositiveInfinity;
                    bool foundNonNaN = false;
                    for (int i = 0; i < axisSize; i++)
                    {
                        double val = (double)arr.GetAtIndex(baseOffset + i * axisStride);
                        if (!double.IsNaN(val)) { if (val < minVal) minVal = val; foundNonNaN = true; }
                    }
                    return foundNonNaN ? minVal : double.NaN;
                }
                case ReductionOp.NanMax:
                {
                    double maxVal = double.NegativeInfinity;
                    bool foundNonNaN = false;
                    for (int i = 0; i < axisSize; i++)
                    {
                        double val = (double)arr.GetAtIndex(baseOffset + i * axisStride);
                        if (!double.IsNaN(val)) { if (val > maxVal) maxVal = val; foundNonNaN = true; }
                    }
                    return foundNonNaN ? maxVal : double.NaN;
                }
                default:
                    return 0.0;
            }
        }
    }
}
