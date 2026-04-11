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
    public static partial class ILKernelGenerator
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
                key.Op != ReductionOp.NanMin && key.Op != ReductionOp.NanMax &&
                key.Op != ReductionOp.NanMean && key.Op != ReductionOp.NanVar &&
                key.Op != ReductionOp.NanStd)
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
            // NanMean, NanVar, NanStd use two-pass algorithm (need count tracking)
            if (key.Op == ReductionOp.NanMean || key.Op == ReductionOp.NanVar || key.Op == ReductionOp.NanStd)
            {
                return key.InputType switch
                {
                    NPTypeCode.Single => CreateNanStatAxisKernelTyped<float>(key),
                    NPTypeCode.Double => CreateNanStatAxisKernelTyped<double>(key),
                    _ => throw new NotSupportedException($"NaN operations only support float and double, not {key.InputType}")
                };
            }

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
            return (void* input, void* output, long* inputStrides, long* inputShape,
                    long* outputStrides, int axis, long axisSize, int ndim, long outputSize) =>
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
            long* inputStrides, long* inputShape, long* outputStrides,
            int axis, long axisSize, int ndim, long outputSize,
            ReductionOp op)
            where T : unmanaged, IFloatingPoint<T>
        {
            long axisStride = inputStrides[axis];

            // Check if the reduction axis is contiguous (stride == 1)
            bool axisContiguous = axisStride == 1;

            // Compute output shape strides for coordinate calculation
            int outputNdim = ndim - 1;
            Span<long> outputDimStrides = stackalloc long[outputNdim > 0 ? outputNdim : 1];
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
            for (long outIdx = 0; outIdx < outputSize; outIdx++)
            {
                // Convert linear output index to coordinates and compute input base offset
                long remaining = outIdx;
                long inputBaseOffset = 0;
                long outputOffset = 0;

                for (int d = 0; d < outputNdim; d++)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    long coord = remaining / outputDimStrides[d];
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
        private static unsafe T NanReduceContiguousAxis<T>(T* data, long size, ReductionOp op)
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
        private static unsafe float NanReduceContiguousAxisFloat(float* data, long size, ReductionOp op)
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
        private static unsafe double NanReduceContiguousAxisDouble(double* data, long size, ReductionOp op)
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

        private static unsafe float NanReduceContiguousAxisFloat256(float* data, long size, ReductionOp op)
        {
            int vectorCount = Vector256<float>.Count;
            long vectorEnd = size - vectorCount;
            long i = 0;

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

        private static unsafe float NanReduceContiguousAxisFloat128(float* data, long size, ReductionOp op)
        {
            int vectorCount = Vector128<float>.Count;
            long vectorEnd = size - vectorCount;
            long i = 0;

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

        private static unsafe float NanReduceContiguousAxisScalarFloat(float* data, long size, ReductionOp op)
        {
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    float sum = 0f;
                    for (long i = 0; i < size; i++)
                    {
                        if (!float.IsNaN(data[i]))
                            sum += data[i];
                    }
                    return sum;
                }
                case ReductionOp.NanProd:
                {
                    float prod = 1f;
                    for (long i = 0; i < size; i++)
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
                    for (long i = 0; i < size; i++)
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
                    for (long i = 0; i < size; i++)
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

        private static unsafe double NanReduceContiguousAxisDouble256(double* data, long size, ReductionOp op)
        {
            int vectorCount = Vector256<double>.Count;
            long vectorEnd = size - vectorCount;
            long i = 0;

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

        private static unsafe double NanReduceContiguousAxisDouble128(double* data, long size, ReductionOp op)
        {
            int vectorCount = Vector128<double>.Count;
            long vectorEnd = size - vectorCount;
            long i = 0;

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

        private static unsafe double NanReduceContiguousAxisScalarDouble(double* data, long size, ReductionOp op)
        {
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    double sum = 0.0;
                    for (long i = 0; i < size; i++)
                    {
                        if (!double.IsNaN(data[i]))
                            sum += data[i];
                    }
                    return sum;
                }
                case ReductionOp.NanProd:
                {
                    double prod = 1.0;
                    for (long i = 0; i < size; i++)
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
                    for (long i = 0; i < size; i++)
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
                    for (long i = 0; i < size; i++)
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
        private static unsafe T NanReduceContiguousAxisScalar<T>(T* data, long size, ReductionOp op)
            where T : unmanaged, IFloatingPoint<T>
        {
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    T sum = T.Zero;
                    for (long i = 0; i < size; i++)
                    {
                        if (!T.IsNaN(data[i]))
                            sum += data[i];
                    }
                    return sum;
                }
                case ReductionOp.NanProd:
                {
                    T prod = T.One;
                    for (long i = 0; i < size; i++)
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
                    for (long i = 0; i < size; i++)
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
                    for (long i = 0; i < size; i++)
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
        private static unsafe T NanReduceStridedAxis<T>(T* data, long size, long stride, ReductionOp op)
            where T : unmanaged, IFloatingPoint<T>
        {
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    T sum = T.Zero;
                    for (long i = 0; i < size; i++)
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
                    for (long i = 0; i < size; i++)
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
                    for (long i = 0; i < size; i++)
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
                    for (long i = 0; i < size; i++)
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

        #region NaN Statistics (NanMean, NanVar, NanStd)

        /// <summary>
        /// Create a typed NaN statistics axis reduction kernel (NanMean, NanVar, NanStd).
        /// These require two-pass algorithm with count tracking.
        /// </summary>
        private static unsafe AxisReductionKernel CreateNanStatAxisKernelTyped<T>(AxisReductionKernelKey key)
            where T : unmanaged, IFloatingPoint<T>
        {
            return (void* input, void* output, long* inputStrides, long* inputShape,
                    long* outputStrides, int axis, long axisSize, int ndim, long outputSize) =>
            {
                NanStatAxisReductionHelper<T>(
                    (T*)input, (T*)output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize,
                    key.Op);
            };
        }

        /// <summary>
        /// Helper for NaN-aware statistics axis reduction (NanMean, NanVar, NanStd).
        /// Two-pass algorithm: first compute mean (with count), then variance if needed.
        /// </summary>
        internal static unsafe void NanStatAxisReductionHelper<T>(
            T* input, T* output,
            long* inputStrides, long* inputShape, long* outputStrides,
            int axis, long axisSize, int ndim, long outputSize,
            ReductionOp op)
            where T : unmanaged, IFloatingPoint<T>
        {
            long axisStride = inputStrides[axis];
            bool axisContiguous = axisStride == 1;

            // Compute output shape strides for coordinate calculation
            int outputNdim = ndim - 1;
            Span<long> outputDimStrides = stackalloc long[outputNdim > 0 ? outputNdim : 1];
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
            for (long outIdx = 0; outIdx < outputSize; outIdx++)
            {
                // Convert linear output index to coordinates and compute input base offset
                long remaining = outIdx;
                long inputBaseOffset = 0;
                long outputOffset = 0;

                for (int d = 0; d < outputNdim; d++)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    long coord = remaining / outputDimStrides[d];
                    remaining = remaining % outputDimStrides[d];
                    inputBaseOffset += coord * inputStrides[inputDim];
                    outputOffset += coord * outputStrides[d];
                }

                T* axisStart = input + inputBaseOffset;

                T result;
                if (axisContiguous)
                {
                    result = NanStatReduceContiguousAxis(axisStart, axisSize, op);
                }
                else
                {
                    result = NanStatReduceStridedAxis(axisStart, axisSize, axisStride, op);
                }

                output[outputOffset] = result;
            }
        }

        /// <summary>
        /// Reduce a contiguous axis for NaN statistics (NanMean, NanVar, NanStd).
        /// </summary>
        private static unsafe T NanStatReduceContiguousAxis<T>(T* data, long size, ReductionOp op)
            where T : unmanaged, IFloatingPoint<T>
        {
            if (size == 0)
                return T.CreateTruncating(double.NaN);

            // Pass 1: Compute sum and count
            double sum = 0.0;
            long count = 0;

            if (typeof(T) == typeof(float))
            {
                float* p = (float*)(void*)data;
                for (long i = 0; i < size; i++)
                {
                    if (!float.IsNaN(p[i]))
                    {
                        sum += p[i];
                        count++;
                    }
                }
            }
            else if (typeof(T) == typeof(double))
            {
                double* p = (double*)(void*)data;
                for (long i = 0; i < size; i++)
                {
                    if (!double.IsNaN(p[i]))
                    {
                        sum += p[i];
                        count++;
                    }
                }
            }

            if (count == 0)
                return T.CreateTruncating(double.NaN);

            double mean = sum / count;

            // For NanMean, we're done
            if (op == ReductionOp.NanMean)
                return T.CreateTruncating(mean);

            // Pass 2: Compute sum of squared differences (for NanVar/NanStd)
            double sqDiffSum = 0.0;

            if (typeof(T) == typeof(float))
            {
                float* p = (float*)(void*)data;
                for (long i = 0; i < size; i++)
                {
                    if (!float.IsNaN(p[i]))
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
            }
            else if (typeof(T) == typeof(double))
            {
                double* p = (double*)(void*)data;
                for (long i = 0; i < size; i++)
                {
                    if (!double.IsNaN(p[i]))
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
            }

            double variance = sqDiffSum / count;  // ddof=0 for now
            return op == ReductionOp.NanStd
                ? T.CreateTruncating(Math.Sqrt(variance))
                : T.CreateTruncating(variance);
        }

        /// <summary>
        /// Reduce a strided axis for NaN statistics (NanMean, NanVar, NanStd).
        /// </summary>
        private static unsafe T NanStatReduceStridedAxis<T>(T* data, long size, long stride, ReductionOp op)
            where T : unmanaged, IFloatingPoint<T>
        {
            if (size == 0)
                return T.CreateTruncating(double.NaN);

            // Pass 1: Compute sum and count
            double sum = 0.0;
            long count = 0;

            for (long i = 0; i < size; i++)
            {
                T val = data[i * stride];
                if (!T.IsNaN(val))
                {
                    sum += double.CreateTruncating(val);
                    count++;
                }
            }

            if (count == 0)
                return T.CreateTruncating(double.NaN);

            double mean = sum / count;

            // For NanMean, we're done
            if (op == ReductionOp.NanMean)
                return T.CreateTruncating(mean);

            // Pass 2: Compute sum of squared differences (for NanVar/NanStd)
            double sqDiffSum = 0.0;

            for (long i = 0; i < size; i++)
            {
                T val = data[i * stride];
                if (!T.IsNaN(val))
                {
                    double diff = double.CreateTruncating(val) - mean;
                    sqDiffSum += diff * diff;
                }
            }

            double variance = sqDiffSum / count;  // ddof=0 for now
            return op == ReductionOp.NanStd
                ? T.CreateTruncating(Math.Sqrt(variance))
                : T.CreateTruncating(variance);
        }

        #endregion
    }
}
