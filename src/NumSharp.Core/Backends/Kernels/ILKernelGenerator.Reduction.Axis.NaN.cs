using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Reduction.Axis.NaN.cs - NaN-aware Axis Reduction Kernels
// =============================================================================
//
// RESPONSIBILITY:
//   - NaN-aware axis reduction kernels (NanSum, NanProd, NanMin, NanMax)
//   - SIMD implementations for contiguous axis reduction
//   - Scalar fallback for strided cases
//
// BEHAVIOR (matching NumPy 2.4.2):
//   - NanSum: Treat NaN as 0 (identity for addition). Empty/all-NaN returns 0.
//   - NanProd: Treat NaN as 1 (identity for multiplication). Empty/all-NaN returns 1.
//   - NanMin: Ignore NaN values. All-NaN slice returns NaN.
//   - NanMax: Ignore NaN values. All-NaN slice returns NaN.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public sealed partial class ILKernelGenerator
    {
        #region NaN-aware Axis Reduction

        /// <summary>
        /// Cache for NaN axis reduction kernels.
        /// </summary>
        private static readonly ConcurrentDictionary<AxisReductionKernelKey, AxisReductionKernel> _nanAxisReductionCache = new();

        /// <summary>
        /// Number of NaN axis reduction kernels in cache.
        /// </summary>
        public static int NanAxisReductionCachedCount => _nanAxisReductionCache.Count;

        /// <summary>
        /// Try to get a NaN-aware axis reduction kernel.
        /// Only supports float and double types (NaN is only defined for floating-point).
        /// </summary>
        public static AxisReductionKernel? TryGetNanAxisReductionKernel(AxisReductionKernelKey key)
        {
            if (!Enabled)
                return null;

            // Only support NaN operations
            if (key.Op != ReductionOp.NanSum && key.Op != ReductionOp.NanProd &&
                key.Op != ReductionOp.NanMin && key.Op != ReductionOp.NanMax)
            {
                return null;
            }

            // NaN is only defined for float and double
            if (key.InputType != NPTypeCode.Single && key.InputType != NPTypeCode.Double)
            {
                return null;
            }

            return _nanAxisReductionCache.GetOrAdd(key, CreateNanAxisReductionKernel);
        }

        /// <summary>
        /// Create a NaN-aware axis reduction kernel.
        /// </summary>
        private static AxisReductionKernel CreateNanAxisReductionKernel(AxisReductionKernelKey key)
        {
            return key.InputType switch
            {
                NPTypeCode.Single => CreateNanAxisReductionKernelTyped<float>(key),
                NPTypeCode.Double => CreateNanAxisReductionKernelTyped<double>(key),
                _ => throw new NotSupportedException($"NaN operations only support float and double, not {key.InputType}")
            };
        }

        /// <summary>
        /// Create a typed NaN-aware axis reduction kernel.
        /// </summary>
        private static unsafe AxisReductionKernel CreateNanAxisReductionKernelTyped<T>(AxisReductionKernelKey key)
            where T : unmanaged, IFloatingPoint<T>
        {
            return (void* input, void* output, int* inputStrides, int* inputShape,
                    int* outputStrides, int axis, int axisSize, int ndim, int outputSize) =>
            {
                NanAxisReductionSimdHelper<T>(
                    (T*)input, (T*)output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize,
                    key.Op);
            };
        }

        /// <summary>
        /// SIMD helper for NaN-aware axis reduction operations.
        /// Reduces along a specific axis, writing results to output array.
        /// </summary>
        internal static unsafe void NanAxisReductionSimdHelper<T>(
            T* input, T* output,
            int* inputStrides, int* inputShape, int* outputStrides,
            int axis, int axisSize, int ndim, int outputSize,
            ReductionOp op)
            where T : unmanaged, IFloatingPoint<T>
        {
            int axisStride = inputStrides[axis];

            // Check if the reduction axis is contiguous (stride == 1)
            bool axisContiguous = axisStride == 1;

            // Compute output shape strides for coordinate calculation
            int outputNdim = ndim - 1;
            Span<int> outputDimStrides = stackalloc int[outputNdim > 0 ? outputNdim : 1];
            if (outputNdim > 0)
            {
                outputDimStrides[outputNdim - 1] = 1;
                for (int d = outputNdim - 2; d >= 0; d--)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    int nextInputDim = (d + 1) >= axis ? d + 2 : d + 1;
                    outputDimStrides[d] = outputDimStrides[d + 1] * inputShape[nextInputDim];
                }
            }

            // Iterate over all output elements
            for (int outIdx = 0; outIdx < outputSize; outIdx++)
            {
                // Convert linear output index to coordinates and compute input base offset
                int remaining = outIdx;
                int inputBaseOffset = 0;
                int outputOffset = 0;

                for (int d = 0; d < outputNdim; d++)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    int coord = remaining / outputDimStrides[d];
                    remaining = remaining % outputDimStrides[d];
                    inputBaseOffset += coord * inputStrides[inputDim];
                    outputOffset += coord * outputStrides[d];
                }

                T* axisStart = input + inputBaseOffset;

                T result;
                if (axisContiguous)
                {
                    // Fast path: axis is contiguous, use SIMD
                    result = NanReduceContiguousAxis(axisStart, axisSize, op);
                }
                else
                {
                    // Strided path: axis is not contiguous
                    result = NanReduceStridedAxis(axisStart, axisSize, axisStride, op);
                }

                output[outputOffset] = result;
            }
        }

        /// <summary>
        /// Reduce a contiguous axis with NaN handling using SIMD.
        /// </summary>
        private static unsafe T NanReduceContiguousAxis<T>(T* data, int size, ReductionOp op)
            where T : unmanaged, IFloatingPoint<T>
        {
            if (size == 0)
            {
                return GetNanIdentityValue<T>(op);
            }

            if (size == 1)
            {
                T val = data[0];
                if (T.IsNaN(val))
                {
                    return op switch
                    {
                        ReductionOp.NanSum => T.Zero,
                        ReductionOp.NanProd => T.One,
                        ReductionOp.NanMin or ReductionOp.NanMax => val, // Return NaN
                        _ => val
                    };
                }
                return val;
            }

            // Dispatch to type-specific SIMD implementations
            if (typeof(T) == typeof(float))
            {
                return (T)(object)NanReduceContiguousAxisFloat((float*)(void*)data, size, op);
            }
            if (typeof(T) == typeof(double))
            {
                return (T)(object)NanReduceContiguousAxisDouble((double*)(void*)data, size, op);
            }

            // Fallback to scalar
            return NanReduceContiguousAxisScalar(data, size, op);
        }

        /// <summary>
        /// SIMD implementation for float NaN reduction.
        /// </summary>
        private static unsafe float NanReduceContiguousAxisFloat(float* data, int size, ReductionOp op)
        {
            if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
            {
                return NanReduceContiguousAxisFloat256(data, size, op);
            }
            if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
            {
                return NanReduceContiguousAxisFloat128(data, size, op);
            }
            return NanReduceContiguousAxisScalarFloat(data, size, op);
        }

        /// <summary>
        /// SIMD implementation for double NaN reduction.
        /// </summary>
        private static unsafe double NanReduceContiguousAxisDouble(double* data, int size, ReductionOp op)
        {
            if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
            {
                return NanReduceContiguousAxisDouble256(data, size, op);
            }
            if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
            {
                return NanReduceContiguousAxisDouble128(data, size, op);
            }
            return NanReduceContiguousAxisScalarDouble(data, size, op);
        }

        #region Float SIMD Implementations

        private static unsafe float NanReduceContiguousAxisFloat256(float* data, int size, ReductionOp op)
        {
            int vectorCount = Vector256<float>.Count;
            int vectorEnd = size - vectorCount;
            int i = 0;

            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    var sumVec = Vector256<float>.Zero;
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector256.Load(data + i);
                        var nanMask = Vector256.Equals(vec, vec); // true for non-NaN
                        var cleaned = Vector256.BitwiseAnd(vec, nanMask.AsSingle());
                        sumVec = Vector256.Add(sumVec, cleaned);
                    }
                    float sum = Vector256.Sum(sumVec);
                    for (; i < size; i++)
                    {
                        if (!float.IsNaN(data[i]))
                            sum += data[i];
                    }
                    return sum;
                }
                case ReductionOp.NanProd:
                {
                    var prodVec = Vector256.Create(1f);
                    var oneVec = Vector256.Create(1f);
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector256.Load(data + i);
                        var nanMask = Vector256.Equals(vec, vec);
                        var cleaned = Vector256.ConditionalSelect(nanMask, vec, oneVec);
                        prodVec = Vector256.Multiply(prodVec, cleaned);
                    }
                    float prod = prodVec.GetElement(0);
                    for (int j = 1; j < vectorCount; j++)
                        prod *= prodVec.GetElement(j);
                    for (; i < size; i++)
                    {
                        if (!float.IsNaN(data[i]))
                            prod *= data[i];
                    }
                    return prod;
                }
                case ReductionOp.NanMin:
                {
                    var minVec = Vector256.Create(float.PositiveInfinity);
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector256.Load(data + i);
                        var nanMask = Vector256.Equals(vec, vec);
                        var cleaned = Vector256.ConditionalSelect(nanMask, vec, Vector256.Create(float.PositiveInfinity));
                        minVec = Vector256.Min(minVec, cleaned);
                    }
                    float minVal = minVec.GetElement(0);
                    for (int j = 1; j < vectorCount; j++)
                    {
                        float elem = minVec.GetElement(j);
                        if (elem < minVal) minVal = elem;
                    }
                    for (; i < size; i++)
                    {
                        if (!float.IsNaN(data[i]) && data[i] < minVal)
                            minVal = data[i];
                    }
                    return float.IsPositiveInfinity(minVal) ? float.NaN : minVal;
                }
                case ReductionOp.NanMax:
                {
                    var maxVec = Vector256.Create(float.NegativeInfinity);
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector256.Load(data + i);
                        var nanMask = Vector256.Equals(vec, vec);
                        var cleaned = Vector256.ConditionalSelect(nanMask, vec, Vector256.Create(float.NegativeInfinity));
                        maxVec = Vector256.Max(maxVec, cleaned);
                    }
                    float maxVal = maxVec.GetElement(0);
                    for (int j = 1; j < vectorCount; j++)
                    {
                        float elem = maxVec.GetElement(j);
                        if (elem > maxVal) maxVal = elem;
                    }
                    for (; i < size; i++)
                    {
                        if (!float.IsNaN(data[i]) && data[i] > maxVal)
                            maxVal = data[i];
                    }
                    return float.IsNegativeInfinity(maxVal) ? float.NaN : maxVal;
                }
                default:
                    return NanReduceContiguousAxisScalarFloat(data, size, op);
            }
        }

        private static unsafe float NanReduceContiguousAxisFloat128(float* data, int size, ReductionOp op)
        {
            int vectorCount = Vector128<float>.Count;
            int vectorEnd = size - vectorCount;
            int i = 0;

            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    var sumVec = Vector128<float>.Zero;
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector128.Load(data + i);
                        var nanMask = Vector128.Equals(vec, vec);
                        var cleaned = Vector128.BitwiseAnd(vec, nanMask.AsSingle());
                        sumVec = Vector128.Add(sumVec, cleaned);
                    }
                    float sum = Vector128.Sum(sumVec);
                    for (; i < size; i++)
                    {
                        if (!float.IsNaN(data[i]))
                            sum += data[i];
                    }
                    return sum;
                }
                case ReductionOp.NanProd:
                {
                    var prodVec = Vector128.Create(1f);
                    var oneVec = Vector128.Create(1f);
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector128.Load(data + i);
                        var nanMask = Vector128.Equals(vec, vec);
                        var cleaned = Vector128.ConditionalSelect(nanMask, vec, oneVec);
                        prodVec = Vector128.Multiply(prodVec, cleaned);
                    }
                    float prod = prodVec.GetElement(0);
                    for (int j = 1; j < vectorCount; j++)
                        prod *= prodVec.GetElement(j);
                    for (; i < size; i++)
                    {
                        if (!float.IsNaN(data[i]))
                            prod *= data[i];
                    }
                    return prod;
                }
                case ReductionOp.NanMin:
                {
                    var minVec = Vector128.Create(float.PositiveInfinity);
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector128.Load(data + i);
                        var nanMask = Vector128.Equals(vec, vec);
                        var cleaned = Vector128.ConditionalSelect(nanMask, vec, Vector128.Create(float.PositiveInfinity));
                        minVec = Vector128.Min(minVec, cleaned);
                    }
                    float minVal = minVec.GetElement(0);
                    for (int j = 1; j < vectorCount; j++)
                    {
                        float elem = minVec.GetElement(j);
                        if (elem < minVal) minVal = elem;
                    }
                    for (; i < size; i++)
                    {
                        if (!float.IsNaN(data[i]) && data[i] < minVal)
                            minVal = data[i];
                    }
                    return float.IsPositiveInfinity(minVal) ? float.NaN : minVal;
                }
                case ReductionOp.NanMax:
                {
                    var maxVec = Vector128.Create(float.NegativeInfinity);
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector128.Load(data + i);
                        var nanMask = Vector128.Equals(vec, vec);
                        var cleaned = Vector128.ConditionalSelect(nanMask, vec, Vector128.Create(float.NegativeInfinity));
                        maxVec = Vector128.Max(maxVec, cleaned);
                    }
                    float maxVal = maxVec.GetElement(0);
                    for (int j = 1; j < vectorCount; j++)
                    {
                        float elem = maxVec.GetElement(j);
                        if (elem > maxVal) maxVal = elem;
                    }
                    for (; i < size; i++)
                    {
                        if (!float.IsNaN(data[i]) && data[i] > maxVal)
                            maxVal = data[i];
                    }
                    return float.IsNegativeInfinity(maxVal) ? float.NaN : maxVal;
                }
                default:
                    return NanReduceContiguousAxisScalarFloat(data, size, op);
            }
        }

        private static unsafe float NanReduceContiguousAxisScalarFloat(float* data, int size, ReductionOp op)
        {
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    float sum = 0f;
                    for (int i = 0; i < size; i++)
                    {
                        if (!float.IsNaN(data[i]))
                            sum += data[i];
                    }
                    return sum;
                }
                case ReductionOp.NanProd:
                {
                    float prod = 1f;
                    for (int i = 0; i < size; i++)
                    {
                        if (!float.IsNaN(data[i]))
                            prod *= data[i];
                    }
                    return prod;
                }
                case ReductionOp.NanMin:
                {
                    float minVal = float.PositiveInfinity;
                    bool foundNonNaN = false;
                    for (int i = 0; i < size; i++)
                    {
                        if (!float.IsNaN(data[i]))
                        {
                            if (data[i] < minVal)
                                minVal = data[i];
                            foundNonNaN = true;
                        }
                    }
                    return foundNonNaN ? minVal : float.NaN;
                }
                case ReductionOp.NanMax:
                {
                    float maxVal = float.NegativeInfinity;
                    bool foundNonNaN = false;
                    for (int i = 0; i < size; i++)
                    {
                        if (!float.IsNaN(data[i]))
                        {
                            if (data[i] > maxVal)
                                maxVal = data[i];
                            foundNonNaN = true;
                        }
                    }
                    return foundNonNaN ? maxVal : float.NaN;
                }
                default:
                    return 0f;
            }
        }

        #endregion

        #region Double SIMD Implementations

        private static unsafe double NanReduceContiguousAxisDouble256(double* data, int size, ReductionOp op)
        {
            int vectorCount = Vector256<double>.Count;
            int vectorEnd = size - vectorCount;
            int i = 0;

            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    var sumVec = Vector256<double>.Zero;
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector256.Load(data + i);
                        var nanMask = Vector256.Equals(vec, vec);
                        var cleaned = Vector256.BitwiseAnd(vec, nanMask.AsDouble());
                        sumVec = Vector256.Add(sumVec, cleaned);
                    }
                    double sum = Vector256.Sum(sumVec);
                    for (; i < size; i++)
                    {
                        if (!double.IsNaN(data[i]))
                            sum += data[i];
                    }
                    return sum;
                }
                case ReductionOp.NanProd:
                {
                    var prodVec = Vector256.Create(1.0);
                    var oneVec = Vector256.Create(1.0);
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector256.Load(data + i);
                        var nanMask = Vector256.Equals(vec, vec);
                        var cleaned = Vector256.ConditionalSelect(nanMask, vec, oneVec);
                        prodVec = Vector256.Multiply(prodVec, cleaned);
                    }
                    double prod = prodVec.GetElement(0);
                    for (int j = 1; j < vectorCount; j++)
                        prod *= prodVec.GetElement(j);
                    for (; i < size; i++)
                    {
                        if (!double.IsNaN(data[i]))
                            prod *= data[i];
                    }
                    return prod;
                }
                case ReductionOp.NanMin:
                {
                    var minVec = Vector256.Create(double.PositiveInfinity);
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector256.Load(data + i);
                        var nanMask = Vector256.Equals(vec, vec);
                        var cleaned = Vector256.ConditionalSelect(nanMask, vec, Vector256.Create(double.PositiveInfinity));
                        minVec = Vector256.Min(minVec, cleaned);
                    }
                    double minVal = minVec.GetElement(0);
                    for (int j = 1; j < vectorCount; j++)
                    {
                        double elem = minVec.GetElement(j);
                        if (elem < minVal) minVal = elem;
                    }
                    for (; i < size; i++)
                    {
                        if (!double.IsNaN(data[i]) && data[i] < minVal)
                            minVal = data[i];
                    }
                    return double.IsPositiveInfinity(minVal) ? double.NaN : minVal;
                }
                case ReductionOp.NanMax:
                {
                    var maxVec = Vector256.Create(double.NegativeInfinity);
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector256.Load(data + i);
                        var nanMask = Vector256.Equals(vec, vec);
                        var cleaned = Vector256.ConditionalSelect(nanMask, vec, Vector256.Create(double.NegativeInfinity));
                        maxVec = Vector256.Max(maxVec, cleaned);
                    }
                    double maxVal = maxVec.GetElement(0);
                    for (int j = 1; j < vectorCount; j++)
                    {
                        double elem = maxVec.GetElement(j);
                        if (elem > maxVal) maxVal = elem;
                    }
                    for (; i < size; i++)
                    {
                        if (!double.IsNaN(data[i]) && data[i] > maxVal)
                            maxVal = data[i];
                    }
                    return double.IsNegativeInfinity(maxVal) ? double.NaN : maxVal;
                }
                default:
                    return NanReduceContiguousAxisScalarDouble(data, size, op);
            }
        }

        private static unsafe double NanReduceContiguousAxisDouble128(double* data, int size, ReductionOp op)
        {
            int vectorCount = Vector128<double>.Count;
            int vectorEnd = size - vectorCount;
            int i = 0;

            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    var sumVec = Vector128<double>.Zero;
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector128.Load(data + i);
                        var nanMask = Vector128.Equals(vec, vec);
                        var cleaned = Vector128.BitwiseAnd(vec, nanMask.AsDouble());
                        sumVec = Vector128.Add(sumVec, cleaned);
                    }
                    double sum = Vector128.Sum(sumVec);
                    for (; i < size; i++)
                    {
                        if (!double.IsNaN(data[i]))
                            sum += data[i];
                    }
                    return sum;
                }
                case ReductionOp.NanProd:
                {
                    var prodVec = Vector128.Create(1.0);
                    var oneVec = Vector128.Create(1.0);
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector128.Load(data + i);
                        var nanMask = Vector128.Equals(vec, vec);
                        var cleaned = Vector128.ConditionalSelect(nanMask, vec, oneVec);
                        prodVec = Vector128.Multiply(prodVec, cleaned);
                    }
                    double prod = prodVec.GetElement(0);
                    for (int j = 1; j < vectorCount; j++)
                        prod *= prodVec.GetElement(j);
                    for (; i < size; i++)
                    {
                        if (!double.IsNaN(data[i]))
                            prod *= data[i];
                    }
                    return prod;
                }
                case ReductionOp.NanMin:
                {
                    var minVec = Vector128.Create(double.PositiveInfinity);
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector128.Load(data + i);
                        var nanMask = Vector128.Equals(vec, vec);
                        var cleaned = Vector128.ConditionalSelect(nanMask, vec, Vector128.Create(double.PositiveInfinity));
                        minVec = Vector128.Min(minVec, cleaned);
                    }
                    double minVal = minVec.GetElement(0);
                    for (int j = 1; j < vectorCount; j++)
                    {
                        double elem = minVec.GetElement(j);
                        if (elem < minVal) minVal = elem;
                    }
                    for (; i < size; i++)
                    {
                        if (!double.IsNaN(data[i]) && data[i] < minVal)
                            minVal = data[i];
                    }
                    return double.IsPositiveInfinity(minVal) ? double.NaN : minVal;
                }
                case ReductionOp.NanMax:
                {
                    var maxVec = Vector128.Create(double.NegativeInfinity);
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector128.Load(data + i);
                        var nanMask = Vector128.Equals(vec, vec);
                        var cleaned = Vector128.ConditionalSelect(nanMask, vec, Vector128.Create(double.NegativeInfinity));
                        maxVec = Vector128.Max(maxVec, cleaned);
                    }
                    double maxVal = maxVec.GetElement(0);
                    for (int j = 1; j < vectorCount; j++)
                    {
                        double elem = maxVec.GetElement(j);
                        if (elem > maxVal) maxVal = elem;
                    }
                    for (; i < size; i++)
                    {
                        if (!double.IsNaN(data[i]) && data[i] > maxVal)
                            maxVal = data[i];
                    }
                    return double.IsNegativeInfinity(maxVal) ? double.NaN : maxVal;
                }
                default:
                    return NanReduceContiguousAxisScalarDouble(data, size, op);
            }
        }

        private static unsafe double NanReduceContiguousAxisScalarDouble(double* data, int size, ReductionOp op)
        {
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    double sum = 0.0;
                    for (int i = 0; i < size; i++)
                    {
                        if (!double.IsNaN(data[i]))
                            sum += data[i];
                    }
                    return sum;
                }
                case ReductionOp.NanProd:
                {
                    double prod = 1.0;
                    for (int i = 0; i < size; i++)
                    {
                        if (!double.IsNaN(data[i]))
                            prod *= data[i];
                    }
                    return prod;
                }
                case ReductionOp.NanMin:
                {
                    double minVal = double.PositiveInfinity;
                    bool foundNonNaN = false;
                    for (int i = 0; i < size; i++)
                    {
                        if (!double.IsNaN(data[i]))
                        {
                            if (data[i] < minVal)
                                minVal = data[i];
                            foundNonNaN = true;
                        }
                    }
                    return foundNonNaN ? minVal : double.NaN;
                }
                case ReductionOp.NanMax:
                {
                    double maxVal = double.NegativeInfinity;
                    bool foundNonNaN = false;
                    for (int i = 0; i < size; i++)
                    {
                        if (!double.IsNaN(data[i]))
                        {
                            if (data[i] > maxVal)
                                maxVal = data[i];
                            foundNonNaN = true;
                        }
                    }
                    return foundNonNaN ? maxVal : double.NaN;
                }
                default:
                    return 0.0;
            }
        }

        #endregion

        /// <summary>
        /// Reduce a contiguous axis with NaN handling using scalar loop.
        /// Generic fallback for types without specialized SIMD.
        /// </summary>
        private static unsafe T NanReduceContiguousAxisScalar<T>(T* data, int size, ReductionOp op)
            where T : unmanaged, IFloatingPoint<T>
        {
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    T sum = T.Zero;
                    for (int i = 0; i < size; i++)
                    {
                        if (!T.IsNaN(data[i]))
                            sum += data[i];
                    }
                    return sum;
                }
                case ReductionOp.NanProd:
                {
                    T prod = T.One;
                    for (int i = 0; i < size; i++)
                    {
                        if (!T.IsNaN(data[i]))
                            prod *= data[i];
                    }
                    return prod;
                }
                case ReductionOp.NanMin:
                {
                    T minVal = T.CreateTruncating(double.PositiveInfinity);
                    bool foundNonNaN = false;
                    for (int i = 0; i < size; i++)
                    {
                        if (!T.IsNaN(data[i]))
                        {
                            if (data[i] < minVal)
                                minVal = data[i];
                            foundNonNaN = true;
                        }
                    }
                    return foundNonNaN ? minVal : T.CreateTruncating(double.NaN);
                }
                case ReductionOp.NanMax:
                {
                    T maxVal = T.CreateTruncating(double.NegativeInfinity);
                    bool foundNonNaN = false;
                    for (int i = 0; i < size; i++)
                    {
                        if (!T.IsNaN(data[i]))
                        {
                            if (data[i] > maxVal)
                                maxVal = data[i];
                            foundNonNaN = true;
                        }
                    }
                    return foundNonNaN ? maxVal : T.CreateTruncating(double.NaN);
                }
                default:
                    return T.Zero;
            }
        }

        /// <summary>
        /// Reduce a strided axis with NaN handling.
        /// </summary>
        private static unsafe T NanReduceStridedAxis<T>(T* data, int size, int stride, ReductionOp op)
            where T : unmanaged, IFloatingPoint<T>
        {
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    T sum = T.Zero;
                    for (int i = 0; i < size; i++)
                    {
                        T val = data[i * stride];
                        if (!T.IsNaN(val))
                            sum += val;
                    }
                    return sum;
                }
                case ReductionOp.NanProd:
                {
                    T prod = T.One;
                    for (int i = 0; i < size; i++)
                    {
                        T val = data[i * stride];
                        if (!T.IsNaN(val))
                            prod *= val;
                    }
                    return prod;
                }
                case ReductionOp.NanMin:
                {
                    T minVal = T.CreateTruncating(double.PositiveInfinity);
                    bool foundNonNaN = false;
                    for (int i = 0; i < size; i++)
                    {
                        T val = data[i * stride];
                        if (!T.IsNaN(val))
                        {
                            if (val < minVal)
                                minVal = val;
                            foundNonNaN = true;
                        }
                    }
                    return foundNonNaN ? minVal : T.CreateTruncating(double.NaN);
                }
                case ReductionOp.NanMax:
                {
                    T maxVal = T.CreateTruncating(double.NegativeInfinity);
                    bool foundNonNaN = false;
                    for (int i = 0; i < size; i++)
                    {
                        T val = data[i * stride];
                        if (!T.IsNaN(val))
                        {
                            if (val > maxVal)
                                maxVal = val;
                            foundNonNaN = true;
                        }
                    }
                    return foundNonNaN ? maxVal : T.CreateTruncating(double.NaN);
                }
                default:
                    return T.Zero;
            }
        }

        /// <summary>
        /// Get the identity value for NaN reduction operations.
        /// </summary>
        private static T GetNanIdentityValue<T>(ReductionOp op)
            where T : unmanaged, IFloatingPoint<T>
        {
            return op switch
            {
                ReductionOp.NanSum => T.Zero,
                ReductionOp.NanProd => T.One,
                ReductionOp.NanMin or ReductionOp.NanMax => T.CreateTruncating(double.NaN),
                _ => T.Zero
            };
        }

        #endregion
    }
}
