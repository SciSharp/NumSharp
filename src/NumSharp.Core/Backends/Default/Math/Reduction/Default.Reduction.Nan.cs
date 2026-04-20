using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Return the sum of array elements over a given axis treating NaNs as zero.
        /// </summary>
        public override NDArray NanSum(NDArray a, int? axis = null, bool keepdims = false)
        {
            var arr = a;
            var shape = arr.Shape;

            // B15: Complex nansum — treat any entry with NaN in real OR imag as zero.
            if (arr.GetTypeCode == NPTypeCode.Complex)
                return NanSumComplex(arr, axis, keepdims);

            // Non-float types: fall back to regular sum (no NaN possible)
            if (arr.GetTypeCode != NPTypeCode.Single && arr.GetTypeCode != NPTypeCode.Double && arr.GetTypeCode != NPTypeCode.Half)
                return Sum(arr, axis: axis, keepdims: keepdims);

            if (shape.IsEmpty)
                return arr;

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
            {
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
                    case NPTypeCode.Half:
                        if (Half.IsNaN((Half)val))
                            return NDArray.Scalar(Half.Zero);
                        break;
                }
                return a.Clone();
            }

            if (axis == null)
            {
                return NanReductionElementWise(arr, ReductionOp.NanSum, keepdims);
            }
            else
            {
                return ExecuteNanAxisReduction(arr, axis.Value, keepdims, ReductionOp.NanSum);
            }
        }

        /// <summary>
        /// Return the product of array elements over a given axis treating NaNs as ones.
        /// </summary>
        public override NDArray NanProd(NDArray a, int? axis = null, bool keepdims = false)
        {
            var arr = a;
            var shape = arr.Shape;

            // Non-float types: fall back to regular prod (no NaN possible)
            if (arr.GetTypeCode != NPTypeCode.Single && arr.GetTypeCode != NPTypeCode.Double && arr.GetTypeCode != NPTypeCode.Half)
                return ReduceProduct(arr, axis, keepdims: keepdims);

            if (shape.IsEmpty)
                return arr;

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
            {
                var val = arr.GetAtIndex(0);
                switch (arr.GetTypeCode)
                {
                    case NPTypeCode.Single:
                        if (float.IsNaN((float)val))
                            return NDArray.Scalar(1f);
                        break;
                    case NPTypeCode.Double:
                        if (double.IsNaN((double)val))
                            return NDArray.Scalar(1.0);
                        break;
                    case NPTypeCode.Half:
                        if (Half.IsNaN((Half)val))
                            return NDArray.Scalar((Half)1.0);
                        break;
                }
                return a.Clone();
            }

            if (axis == null)
            {
                return NanReductionElementWise(arr, ReductionOp.NanProd, keepdims);
            }
            else
            {
                return ExecuteNanAxisReduction(arr, axis.Value, keepdims, ReductionOp.NanProd);
            }
        }

        /// <summary>
        /// Return minimum of an array or minimum along an axis, ignoring NaNs.
        /// </summary>
        public override NDArray NanMin(NDArray a, int? axis = null, bool keepdims = false)
        {
            var arr = a;
            var shape = arr.Shape;

            // Non-float types: fall back to regular amin (no NaN possible)
            if (arr.GetTypeCode != NPTypeCode.Single && arr.GetTypeCode != NPTypeCode.Double && arr.GetTypeCode != NPTypeCode.Half)
                return ReduceAMin(arr, axis, keepdims: keepdims);

            if (shape.IsEmpty)
                return arr;

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
            {
                return a.Clone();
            }

            if (axis == null)
            {
                return NanReductionElementWise(arr, ReductionOp.NanMin, keepdims);
            }
            else
            {
                return ExecuteNanAxisReduction(arr, axis.Value, keepdims, ReductionOp.NanMin);
            }
        }

        /// <summary>
        /// Return maximum of an array or maximum along an axis, ignoring NaNs.
        /// </summary>
        public override NDArray NanMax(NDArray a, int? axis = null, bool keepdims = false)
        {
            var arr = a;
            var shape = arr.Shape;

            // Non-float types: fall back to regular amax (no NaN possible)
            if (arr.GetTypeCode != NPTypeCode.Single && arr.GetTypeCode != NPTypeCode.Double && arr.GetTypeCode != NPTypeCode.Half)
                return ReduceAMax(arr, axis, keepdims: keepdims);

            if (shape.IsEmpty)
                return arr;

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
            {
                return a.Clone();
            }

            if (axis == null)
            {
                return NanReductionElementWise(arr, ReductionOp.NanMax, keepdims);
            }
            else
            {
                return ExecuteNanAxisReduction(arr, axis.Value, keepdims, ReductionOp.NanMax);
            }
        }

        /// <summary>
        /// Element-wise NaN reduction (axis=null).
        /// </summary>
        private NDArray NanReductionElementWise(NDArray arr, ReductionOp op, bool keepdims)
        {
            if (ILKernelGenerator.Enabled && arr.Shape.IsContiguous)
            {
                object result;
                unsafe
                {
                    switch (arr.GetTypeCode)
                    {
                        case NPTypeCode.Single:
                            result = op switch
                            {
                                ReductionOp.NanSum => ILKernelGenerator.NanSumSimdHelperFloat((float*)arr.Address, arr.size),
                                ReductionOp.NanProd => ILKernelGenerator.NanProdSimdHelperFloat((float*)arr.Address, arr.size),
                                ReductionOp.NanMin => ILKernelGenerator.NanMinSimdHelperFloat((float*)arr.Address, arr.size),
                                ReductionOp.NanMax => ILKernelGenerator.NanMaxSimdHelperFloat((float*)arr.Address, arr.size),
                                _ => throw new NotSupportedException($"Unsupported NaN reduction: {op}")
                            };
                            break;
                        case NPTypeCode.Double:
                            result = op switch
                            {
                                ReductionOp.NanSum => ILKernelGenerator.NanSumSimdHelperDouble((double*)arr.Address, arr.size),
                                ReductionOp.NanProd => ILKernelGenerator.NanProdSimdHelperDouble((double*)arr.Address, arr.size),
                                ReductionOp.NanMin => ILKernelGenerator.NanMinSimdHelperDouble((double*)arr.Address, arr.size),
                                ReductionOp.NanMax => ILKernelGenerator.NanMaxSimdHelperDouble((double*)arr.Address, arr.size),
                                _ => throw new NotSupportedException($"Unsupported NaN reduction: {op}")
                            };
                            break;
                        case NPTypeCode.Half:
                            result = op switch
                            {
                                ReductionOp.NanSum => ILKernelGenerator.NanSumHalfHelper((Half*)arr.Address, arr.size),
                                ReductionOp.NanProd => ILKernelGenerator.NanProdHalfHelper((Half*)arr.Address, arr.size),
                                ReductionOp.NanMin => ILKernelGenerator.NanMinHalfHelper((Half*)arr.Address, arr.size),
                                ReductionOp.NanMax => ILKernelGenerator.NanMaxHalfHelper((Half*)arr.Address, arr.size),
                                _ => throw new NotSupportedException($"Unsupported NaN reduction: {op}")
                            };
                            break;
                        default:
                            throw new NotSupportedException($"NaN reductions only support float/double/half, got {arr.GetTypeCode}");
                    }
                }

                var r = NDArray.Scalar(result);
                if (keepdims)
                {
                    var keepdimsShape = new long[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++)
                        keepdimsShape[i] = 1;
                    r.Storage.Reshape(new Shape(keepdimsShape));
                }
                return r;
            }
            else
            {
                return NanReductionScalar(arr, op, keepdims);
            }
        }

        /// <summary>
        /// Scalar fallback for NaN reduction when SIMD is not available.
        /// </summary>
        private NDArray NanReductionScalar(NDArray arr, ReductionOp op, bool keepdims)
        {
            object result;

            switch (arr.GetTypeCode)
            {
                case NPTypeCode.Single:
                    result = NanReduceScalarFloat(arr, op);
                    break;
                case NPTypeCode.Double:
                    result = NanReduceScalarDouble(arr, op);
                    break;
                case NPTypeCode.Half:
                    result = NanReduceScalarHalf(arr, op);
                    break;
                default:
                    throw new NotSupportedException($"NaN reductions only support float/double/half, got {arr.GetTypeCode}");
            }

            var r = NDArray.Scalar(result);
            if (keepdims)
            {
                var keepdimsShape = new long[arr.ndim];
                for (int i = 0; i < arr.ndim; i++)
                    keepdimsShape[i] = 1;
                r.Storage.Reshape(new Shape(keepdimsShape));
            }
            return r;
        }

        private static float NanReduceScalarFloat(NDArray arr, ReductionOp op)
        {
            var iter = arr.AsIterator<float>();
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    float sum = 0f;
                    while (iter.HasNext())
                    {
                        float val = iter.MoveNext();
                        if (!float.IsNaN(val))
                            sum += val;
                    }
                    return sum;
                }
                case ReductionOp.NanProd:
                {
                    float prod = 1f;
                    while (iter.HasNext())
                    {
                        float val = iter.MoveNext();
                        if (!float.IsNaN(val))
                            prod *= val;
                    }
                    return prod;
                }
                case ReductionOp.NanMin:
                {
                    float minVal = float.PositiveInfinity;
                    bool foundNonNaN = false;
                    while (iter.HasNext())
                    {
                        float val = iter.MoveNext();
                        if (!float.IsNaN(val))
                        {
                            if (val < minVal) minVal = val;
                            foundNonNaN = true;
                        }
                    }
                    return foundNonNaN ? minVal : float.NaN;
                }
                case ReductionOp.NanMax:
                {
                    float maxVal = float.NegativeInfinity;
                    bool foundNonNaN = false;
                    while (iter.HasNext())
                    {
                        float val = iter.MoveNext();
                        if (!float.IsNaN(val))
                        {
                            if (val > maxVal) maxVal = val;
                            foundNonNaN = true;
                        }
                    }
                    return foundNonNaN ? maxVal : float.NaN;
                }
                default:
                    throw new NotSupportedException($"Unsupported NaN reduction: {op}");
            }
        }

        private static double NanReduceScalarDouble(NDArray arr, ReductionOp op)
        {
            var iter = arr.AsIterator<double>();
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    double sum = 0.0;
                    while (iter.HasNext())
                    {
                        double val = iter.MoveNext();
                        if (!double.IsNaN(val))
                            sum += val;
                    }
                    return sum;
                }
                case ReductionOp.NanProd:
                {
                    double prod = 1.0;
                    while (iter.HasNext())
                    {
                        double val = iter.MoveNext();
                        if (!double.IsNaN(val))
                            prod *= val;
                    }
                    return prod;
                }
                case ReductionOp.NanMin:
                {
                    double minVal = double.PositiveInfinity;
                    bool foundNonNaN = false;
                    while (iter.HasNext())
                    {
                        double val = iter.MoveNext();
                        if (!double.IsNaN(val))
                        {
                            if (val < minVal) minVal = val;
                            foundNonNaN = true;
                        }
                    }
                    return foundNonNaN ? minVal : double.NaN;
                }
                case ReductionOp.NanMax:
                {
                    double maxVal = double.NegativeInfinity;
                    bool foundNonNaN = false;
                    while (iter.HasNext())
                    {
                        double val = iter.MoveNext();
                        if (!double.IsNaN(val))
                        {
                            if (val > maxVal) maxVal = val;
                            foundNonNaN = true;
                        }
                    }
                    return foundNonNaN ? maxVal : double.NaN;
                }
                default:
                    throw new NotSupportedException($"Unsupported NaN reduction: {op}");
            }
        }

        private static Half NanReduceScalarHalf(NDArray arr, ReductionOp op)
        {
            var iter = arr.AsIterator<Half>();
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    double sum = 0.0; // Use double for precision
                    while (iter.HasNext())
                    {
                        Half val = iter.MoveNext();
                        if (!Half.IsNaN(val))
                            sum += (double)val;
                    }
                    return (Half)sum;
                }
                case ReductionOp.NanProd:
                {
                    double prod = 1.0; // Use double for precision
                    while (iter.HasNext())
                    {
                        Half val = iter.MoveNext();
                        if (!Half.IsNaN(val))
                            prod *= (double)val;
                    }
                    return (Half)prod;
                }
                case ReductionOp.NanMin:
                {
                    Half minVal = Half.PositiveInfinity;
                    bool foundNonNaN = false;
                    while (iter.HasNext())
                    {
                        Half val = iter.MoveNext();
                        if (!Half.IsNaN(val))
                        {
                            if (val < minVal) minVal = val;
                            foundNonNaN = true;
                        }
                    }
                    return foundNonNaN ? minVal : Half.NaN;
                }
                case ReductionOp.NanMax:
                {
                    Half maxVal = Half.NegativeInfinity;
                    bool foundNonNaN = false;
                    while (iter.HasNext())
                    {
                        Half val = iter.MoveNext();
                        if (!Half.IsNaN(val))
                        {
                            if (val > maxVal) maxVal = val;
                            foundNonNaN = true;
                        }
                    }
                    return foundNonNaN ? maxVal : Half.NaN;
                }
                default:
                    throw new NotSupportedException($"Unsupported NaN reduction: {op}");
            }
        }

        /// <summary>
        /// Execute a NaN-aware axis reduction.
        /// </summary>
        private unsafe NDArray ExecuteNanAxisReduction(NDArray arr, int axis, bool keepdims, ReductionOp op)
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
                return ExecuteNanAxisReductionScalar(arr, axis, keepdims, op);
            }

            // Create output array
            var outputDims = new long[arr.ndim - 1];
            for (int d = 0, od = 0; d < arr.ndim; d++)
                if (d != axis) outputDims[od++] = shape.dimensions[d];

            var outputShape = outputDims.Length > 0 ? new Shape(outputDims) : Shape.Scalar;
            var result = new NDArray(inputType, outputShape, false);

            long axisSize = shape.dimensions[axis];
            long outputSize = result.size > 0 ? result.size : 1;
            byte* inputAddr = (byte*)arr.Address + shape.offset * arr.dtypesize;

            fixed (long* inputStrides = shape.strides)
            fixed (long* inputDims = shape.dimensions)
            fixed (long* outputStrides = result.Shape.strides)
            {
                kernel((void*)inputAddr, (void*)result.Address, inputStrides, inputDims, outputStrides, axis, axisSize, arr.ndim, outputSize);
            }

            if (keepdims)
            {
                var ks = new long[arr.ndim];
                for (int d = 0, sd = 0; d < arr.ndim; d++)
                    ks[d] = (d == axis) ? 1 : result.shape[sd++];
                result.Storage.Reshape(new Shape(ks));
            }
            return result;
        }

        /// <summary>
        /// Scalar fallback for NaN axis reduction.
        /// </summary>
        private NDArray ExecuteNanAxisReductionScalar(NDArray arr, int axis, bool keepdims, ReductionOp op)
        {
            var shape = arr.Shape;
            var inputType = arr.GetTypeCode;

            var outputDims = new long[arr.ndim - 1];
            for (int d = 0, od = 0; d < arr.ndim; d++)
                if (d != axis) outputDims[od++] = shape.dimensions[d];

            var outputShape = outputDims.Length > 0 ? new Shape(outputDims) : Shape.Scalar;
            var result = new NDArray(inputType, outputShape, false);

            long axisSize = shape.dimensions[axis];
            long outputSize = result.size > 0 ? result.size : 1;

            long[] outputDimStrides = new long[arr.ndim - 1 > 0 ? arr.ndim - 1 : 1];
            if (arr.ndim > 1)
            {
                outputDimStrides[arr.ndim - 2] = 1;
                for (int d = arr.ndim - 3; d >= 0; d--)
                {
                    int nextInputDim = (d + 1) >= axis ? d + 2 : d + 1;
                    outputDimStrides[d] = outputDimStrides[d + 1] * shape.dimensions[nextInputDim];
                }
            }

            for (long outIdx = 0; outIdx < outputSize; outIdx++)
            {
                long remaining = outIdx;
                long inputBaseOffset = 0;

                for (int d = 0; d < arr.ndim - 1; d++)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    long coord = remaining / outputDimStrides[d];
                    remaining = remaining % outputDimStrides[d];
                    inputBaseOffset += coord * shape.strides[inputDim];
                }

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
                var ks = new long[arr.ndim];
                for (int d = 0, sd = 0; d < arr.ndim; d++)
                    ks[d] = (d == axis) ? 1 : result.shape[sd++];
                result.Storage.Reshape(new Shape(ks));
            }
            return result;
        }

        private static float ReduceNanAxisScalarFloat(NDArray arr, long baseOffset, long axisSize, long axisStride, ReductionOp op)
        {
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    float sum = 0f;
                    for (long i = 0; i < axisSize; i++)
                    {
                        float val = (float)arr.GetAtIndex(baseOffset + i * axisStride);
                        if (!float.IsNaN(val)) sum += val;
                    }
                    return sum;
                }
                case ReductionOp.NanProd:
                {
                    float prod = 1f;
                    for (long i = 0; i < axisSize; i++)
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
                    for (long i = 0; i < axisSize; i++)
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
                    for (long i = 0; i < axisSize; i++)
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

        private static double ReduceNanAxisScalarDouble(NDArray arr, long baseOffset, long axisSize, long axisStride, ReductionOp op)
        {
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    double sum = 0.0;
                    for (long i = 0; i < axisSize; i++)
                    {
                        double val = (double)arr.GetAtIndex(baseOffset + i * axisStride);
                        if (!double.IsNaN(val)) sum += val;
                    }
                    return sum;
                }
                case ReductionOp.NanProd:
                {
                    double prod = 1.0;
                    for (long i = 0; i < axisSize; i++)
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
                    for (long i = 0; i < axisSize; i++)
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
                    for (long i = 0; i < axisSize; i++)
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

        /// <summary>
        /// B15: NumPy-parity Complex nansum. Treats any element with NaN in real OR imag
        /// as zero (skipped). Sum type is Complex.
        /// </summary>
        private NDArray NanSumComplex(NDArray arr, int? axis, bool keepdims)
        {
            var shape = arr.Shape;
            if (shape.IsEmpty) return arr;

            if (axis == null)
            {
                var sum = System.Numerics.Complex.Zero;
                var iter = arr.AsIterator<System.Numerics.Complex>();
                while (iter.HasNext())
                {
                    var v = iter.MoveNext();
                    if (double.IsNaN(v.Real) || double.IsNaN(v.Imaginary)) continue;
                    sum += v;
                }
                var r = NDArray.Scalar(sum);
                if (keepdims)
                {
                    var ks = new long[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++) ks[i] = 1;
                    r.Storage.Reshape(new Shape(ks));
                }
                return r;
            }

            // Axis reduction via iterator: iterate per slice and sum with NaN-skip.
            var ax = axis.Value;
            while (ax < 0) ax = arr.ndim + ax;
            Shape axisedShape = Shape.GetAxis(shape, ax);
            var ret = new NDArray(NPTypeCode.Complex, axisedShape, false);
            var iterAxis = new NDCoordinatesAxisIncrementor(ref shape, ax);
            var iterRet = new ValueCoordinatesIncrementor(ref axisedShape);
            var iterIndex = iterRet.Index;
            var slices = iterAxis.Slices;

            do
            {
                var slice = arr[slices];
                var sum = System.Numerics.Complex.Zero;
                var it = slice.AsIterator<System.Numerics.Complex>();
                while (it.HasNext())
                {
                    var v = it.MoveNext();
                    if (double.IsNaN(v.Real) || double.IsNaN(v.Imaginary)) continue;
                    sum += v;
                }
                ret.SetAtIndex(sum, iterIndex[0]);
            } while (iterAxis.Next() != null && iterRet.Next() != null);

            if (keepdims) ret.Storage.ExpandDimension(ax);
            return ret;
        }
    }
}
