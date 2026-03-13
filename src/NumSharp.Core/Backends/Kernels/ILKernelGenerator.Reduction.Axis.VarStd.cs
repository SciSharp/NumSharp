using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Reduction.Axis.VarStd.cs - Variance/StdDev Axis Reductions
// =============================================================================
//
// RESPONSIBILITY:
//   - CreateAxisVarStdReductionKernel - Var/Std dispatcher
//   - AxisVarStdSimdHelper<T> - two-pass SIMD algorithm
//   - SumContiguousAxisDouble<T> - first pass (compute mean)
//   - SumSquaredDiffContiguous<T> - second pass (squared differences)
//   - Decimal and general helpers
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public sealed partial class ILKernelGenerator
    {
        #region Var/Std Axis Reduction

        /// <summary>
        /// Create an axis Var/Std reduction kernel.
        /// Uses two-pass algorithm: first compute mean along axis, then sum of squared differences.
        /// </summary>
        private static AxisReductionKernel CreateAxisVarStdReductionKernel(AxisReductionKernelKey key)
        {
            // Dispatch based on input type - output is always double for accuracy
            return key.InputType switch
            {
                NPTypeCode.Byte => CreateAxisVarStdKernelTyped<byte>(key),
                NPTypeCode.Int16 => CreateAxisVarStdKernelTyped<short>(key),
                NPTypeCode.UInt16 => CreateAxisVarStdKernelTyped<ushort>(key),
                NPTypeCode.Int32 => CreateAxisVarStdKernelTyped<int>(key),
                NPTypeCode.UInt32 => CreateAxisVarStdKernelTyped<uint>(key),
                NPTypeCode.Int64 => CreateAxisVarStdKernelTyped<long>(key),
                NPTypeCode.UInt64 => CreateAxisVarStdKernelTyped<ulong>(key),
                NPTypeCode.Single => CreateAxisVarStdKernelTyped<float>(key),
                NPTypeCode.Double => CreateAxisVarStdKernelTyped<double>(key),
                NPTypeCode.Decimal => CreateAxisVarStdKernelTypedDecimal(key),
                _ => CreateAxisVarStdKernelGeneral(key) // Fallback for Boolean, Char
            };
        }

        /// <summary>
        /// Create a typed axis Var/Std kernel.
        /// </summary>
        private static unsafe AxisReductionKernel CreateAxisVarStdKernelTyped<TInput>(AxisReductionKernelKey key)
            where TInput : unmanaged
        {
            bool isStd = key.Op == ReductionOp.Std;
            // ddof is not part of kernel key - will be passed as 0 for now
            // For proper ddof support, we'd need an extended kernel signature
            // The DefaultEngine handles ddof at a higher level

            return (void* input, void* output, int* inputStrides, int* inputShape,
                    int* outputStrides, int axis, int axisSize, int ndim, int outputSize) =>
            {
                AxisVarStdSimdHelper<TInput>(
                    (TInput*)input, (double*)output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize,
                    isStd, ddof: 0);
            };
        }

        /// <summary>
        /// Create a decimal axis Var/Std kernel.
        /// </summary>
        private static unsafe AxisReductionKernel CreateAxisVarStdKernelTypedDecimal(AxisReductionKernelKey key)
        {
            bool isStd = key.Op == ReductionOp.Std;

            return (void* input, void* output, int* inputStrides, int* inputShape,
                    int* outputStrides, int axis, int axisSize, int ndim, int outputSize) =>
            {
                AxisVarStdDecimalHelper(
                    (decimal*)input, (double*)output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize,
                    isStd, ddof: 0);
            };
        }

        /// <summary>
        /// Create a general (fallback) axis Var/Std kernel.
        /// </summary>
        private static unsafe AxisReductionKernel CreateAxisVarStdKernelGeneral(AxisReductionKernelKey key)
        {
            bool isStd = key.Op == ReductionOp.Std;

            return (void* input, void* output, int* inputStrides, int* inputShape,
                    int* outputStrides, int axis, int axisSize, int ndim, int outputSize) =>
            {
                AxisVarStdGeneralHelper(
                    input, (double*)output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize,
                    key.InputType, isStd, ddof: 0);
            };
        }

        /// <summary>
        /// SIMD helper for axis Var/Std reduction.
        /// Uses two-pass algorithm: first compute mean, then sum of squared differences.
        /// </summary>
        internal static unsafe void AxisVarStdSimdHelper<TInput>(
            TInput* input, double* output,
            int* inputStrides, int* inputShape, int* outputStrides,
            int axis, int axisSize, int ndim, int outputSize,
            bool computeStd, int ddof)
            where TInput : unmanaged
        {
            int axisStride = inputStrides[axis];
            bool axisContiguous = axisStride == 1;

            // Compute output dimension strides for coordinate calculation
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

            // Divisor for variance (size - ddof)
            double divisor = axisSize - ddof;
            if (divisor <= 0)
                divisor = 1; // Prevent division by zero; will produce NaN behavior

            // Iterate over all output elements
            for (int outIdx = 0; outIdx < outputSize; outIdx++)
            {
                // Convert linear output index to coordinates and compute offsets
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

                TInput* axisStart = input + inputBaseOffset;

                // Pass 1: Compute mean along axis
                double sum = 0;
                if (axisContiguous)
                {
                    sum = SumContiguousAxisDouble(axisStart, axisSize);
                }
                else
                {
                    for (int i = 0; i < axisSize; i++)
                        sum += ConvertToDouble(axisStart[i * axisStride]);
                }
                double mean = sum / axisSize;

                // Pass 2: Compute sum of squared differences
                double sqDiffSum = 0;
                if (axisContiguous)
                {
                    sqDiffSum = SumSquaredDiffContiguous(axisStart, axisSize, mean);
                }
                else
                {
                    for (int i = 0; i < axisSize; i++)
                    {
                        double val = ConvertToDouble(axisStart[i * axisStride]);
                        double diff = val - mean;
                        sqDiffSum += diff * diff;
                    }
                }

                double variance = sqDiffSum / divisor;
                output[outputOffset] = computeStd ? Math.Sqrt(variance) : variance;
            }
        }

        /// <summary>
        /// Sum contiguous axis as double (for mean computation in Var/Std).
        /// </summary>
        private static unsafe double SumContiguousAxisDouble<T>(T* data, int size)
            where T : unmanaged
        {
            double sum = 0;

            if (typeof(T) == typeof(double))
            {
                double* p = (double*)(void*)data;
                if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
                {
                    int vectorCount = Vector256<double>.Count;
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector256<double>.Zero;
                    int i = 0;
                    for (; i <= vectorEnd; i += vectorCount)
                        sumVec = Vector256.Add(sumVec, Vector256.Load(p + i));
                    sum = Vector256.Sum(sumVec);
                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    for (int i = 0; i < size; i++)
                        sum += p[i];
                }
            }
            else if (typeof(T) == typeof(float))
            {
                float* p = (float*)(void*)data;
                if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
                {
                    int vectorCount = Vector256<float>.Count;
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector256<float>.Zero;
                    int i = 0;
                    for (; i <= vectorEnd; i += vectorCount)
                        sumVec = Vector256.Add(sumVec, Vector256.Load(p + i));
                    sum = Vector256.Sum(sumVec);
                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    for (int i = 0; i < size; i++)
                        sum += p[i];
                }
            }
            else if (typeof(T) == typeof(int))
            {
                int* p = (int*)(void*)data;
                if (Vector256.IsHardwareAccelerated && Vector256<int>.IsSupported && size >= Vector256<int>.Count)
                {
                    // Process int vectors, widening to long for accumulation to avoid overflow
                    int vectorCount = Vector256<int>.Count;  // 8 ints per vector
                    int vectorEnd = size - vectorCount;
                    long totalSum = 0;
                    int i = 0;
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var intVec = Vector256.Load(p + i);
                        // Sum all 8 ints - Vector256.Sum returns int, then add to long
                        totalSum += Vector256.Sum(intVec);
                    }
                    sum = totalSum;
                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    long totalSum = 0;
                    for (int i = 0; i < size; i++)
                        totalSum += p[i];
                    sum = totalSum;
                }
            }
            else if (typeof(T) == typeof(long))
            {
                long* p = (long*)(void*)data;
                if (Vector256.IsHardwareAccelerated && Vector256<long>.IsSupported && size >= Vector256<long>.Count)
                {
                    int vectorCount = Vector256<long>.Count;  // 4 longs per vector
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector256<long>.Zero;
                    int i = 0;
                    for (; i <= vectorEnd; i += vectorCount)
                        sumVec = Vector256.Add(sumVec, Vector256.Load(p + i));
                    sum = Vector256.Sum(sumVec);
                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    for (int i = 0; i < size; i++)
                        sum += p[i];
                }
            }
            else if (typeof(T) == typeof(short))
            {
                short* p = (short*)(void*)data;
                if (Vector256.IsHardwareAccelerated && Vector256<short>.IsSupported && size >= Vector256<short>.Count)
                {
                    // Process short vectors - 16 shorts per vector
                    int vectorCount = Vector256<short>.Count;
                    int vectorEnd = size - vectorCount;
                    long totalSum = 0;
                    int i = 0;
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var shortVec = Vector256.Load(p + i);
                        // Convert to ints, then sum (avoids overflow)
                        var (lower, upper) = Vector256.Widen(shortVec);
                        totalSum += Vector256.Sum(lower) + Vector256.Sum(upper);
                    }
                    sum = totalSum;
                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    long totalSum = 0;
                    for (int i = 0; i < size; i++)
                        totalSum += p[i];
                    sum = totalSum;
                }
            }
            else if (typeof(T) == typeof(byte))
            {
                byte* p = (byte*)(void*)data;
                if (Vector256.IsHardwareAccelerated && Vector256<byte>.IsSupported && size >= Vector256<byte>.Count)
                {
                    // Process byte vectors - 32 bytes per vector
                    int vectorCount = Vector256<byte>.Count;
                    int vectorEnd = size - vectorCount;
                    long totalSum = 0;
                    int i = 0;
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var byteVec = Vector256.Load(p + i);
                        // Widen byte->ushort->uint and sum
                        var (lower16, upper16) = Vector256.Widen(byteVec);
                        var (ll32, lu32) = Vector256.Widen(lower16);
                        var (ul32, uu32) = Vector256.Widen(upper16);
                        totalSum += Vector256.Sum(ll32) + Vector256.Sum(lu32) +
                                    Vector256.Sum(ul32) + Vector256.Sum(uu32);
                    }
                    sum = totalSum;
                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    long totalSum = 0;
                    for (int i = 0; i < size; i++)
                        totalSum += p[i];
                    sum = totalSum;
                }
            }
            else
            {
                // For other types (ushort, uint, ulong, char), use scalar loop
                for (int i = 0; i < size; i++)
                    sum += ConvertToDouble(data[i]);
            }

            return sum;
        }

        /// <summary>
        /// Sum squared differences from mean for contiguous axis.
        /// </summary>
        private static unsafe double SumSquaredDiffContiguous<T>(T* data, int size, double mean)
            where T : unmanaged
        {
            double sqDiffSum = 0;

            if (typeof(T) == typeof(double))
            {
                double* p = (double*)(void*)data;
                if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
                {
                    int vectorCount = Vector256<double>.Count;
                    int vectorEnd = size - vectorCount;
                    var meanVec = Vector256.Create(mean);
                    var sqDiffVec = Vector256<double>.Zero;
                    int i = 0;
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector256.Load(p + i);
                        var diff = Vector256.Subtract(vec, meanVec);
                        sqDiffVec = Vector256.Add(sqDiffVec, Vector256.Multiply(diff, diff));
                    }
                    sqDiffSum = Vector256.Sum(sqDiffVec);
                    for (; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
                else
                {
                    for (int i = 0; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
            }
            else if (typeof(T) == typeof(float))
            {
                float* p = (float*)(void*)data;
                float fMean = (float)mean;
                if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
                {
                    int vectorCount = Vector256<float>.Count;
                    int vectorEnd = size - vectorCount;
                    var meanVec = Vector256.Create(fMean);
                    var sqDiffVec = Vector256<float>.Zero;
                    int i = 0;
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector256.Load(p + i);
                        var diff = Vector256.Subtract(vec, meanVec);
                        sqDiffVec = Vector256.Add(sqDiffVec, Vector256.Multiply(diff, diff));
                    }
                    sqDiffSum = Vector256.Sum(sqDiffVec);
                    for (; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
                else
                {
                    for (int i = 0; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
            }
            else if (typeof(T) == typeof(int))
            {
                int* p = (int*)(void*)data;
                // 4x loop unrolling for better performance
                int unrollEnd = size - 4;
                int i = 0;
                for (; i <= unrollEnd; i += 4)
                {
                    double d0 = p[i] - mean;
                    double d1 = p[i + 1] - mean;
                    double d2 = p[i + 2] - mean;
                    double d3 = p[i + 3] - mean;
                    sqDiffSum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                }
                for (; i < size; i++)
                {
                    double diff = p[i] - mean;
                    sqDiffSum += diff * diff;
                }
            }
            else if (typeof(T) == typeof(long))
            {
                long* p = (long*)(void*)data;
                int unrollEnd = size - 4;
                int i = 0;
                for (; i <= unrollEnd; i += 4)
                {
                    double d0 = p[i] - mean;
                    double d1 = p[i + 1] - mean;
                    double d2 = p[i + 2] - mean;
                    double d3 = p[i + 3] - mean;
                    sqDiffSum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                }
                for (; i < size; i++)
                {
                    double diff = p[i] - mean;
                    sqDiffSum += diff * diff;
                }
            }
            else if (typeof(T) == typeof(short))
            {
                short* p = (short*)(void*)data;
                int unrollEnd = size - 4;
                int i = 0;
                for (; i <= unrollEnd; i += 4)
                {
                    double d0 = p[i] - mean;
                    double d1 = p[i + 1] - mean;
                    double d2 = p[i + 2] - mean;
                    double d3 = p[i + 3] - mean;
                    sqDiffSum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                }
                for (; i < size; i++)
                {
                    double diff = p[i] - mean;
                    sqDiffSum += diff * diff;
                }
            }
            else if (typeof(T) == typeof(byte))
            {
                byte* p = (byte*)(void*)data;
                int unrollEnd = size - 4;
                int i = 0;
                for (; i <= unrollEnd; i += 4)
                {
                    double d0 = p[i] - mean;
                    double d1 = p[i + 1] - mean;
                    double d2 = p[i + 2] - mean;
                    double d3 = p[i + 3] - mean;
                    sqDiffSum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                }
                for (; i < size; i++)
                {
                    double diff = p[i] - mean;
                    sqDiffSum += diff * diff;
                }
            }
            else
            {
                // For other types (ushort, uint, ulong, char)
                for (int i = 0; i < size; i++)
                {
                    double diff = ConvertToDouble(data[i]) - mean;
                    sqDiffSum += diff * diff;
                }
            }

            return sqDiffSum;
        }

        /// <summary>
        /// Helper for axis Var/Std with decimal input.
        /// </summary>
        internal static unsafe void AxisVarStdDecimalHelper(
            decimal* input, double* output,
            int* inputStrides, int* inputShape, int* outputStrides,
            int axis, int axisSize, int ndim, int outputSize,
            bool computeStd, int ddof)
        {
            int axisStride = inputStrides[axis];

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

            double divisor = axisSize - ddof;
            if (divisor <= 0) divisor = 1;

            for (int outIdx = 0; outIdx < outputSize; outIdx++)
            {
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

                decimal* axisStart = input + inputBaseOffset;

                // Pass 1: Compute mean
                decimal sum = 0;
                for (int i = 0; i < axisSize; i++)
                    sum += axisStart[i * axisStride];
                decimal mean = sum / axisSize;

                // Pass 2: Sum of squared differences
                decimal sqDiffSum = 0;
                for (int i = 0; i < axisSize; i++)
                {
                    decimal diff = axisStart[i * axisStride] - mean;
                    sqDiffSum += diff * diff;
                }

                double variance = (double)(sqDiffSum / (decimal)divisor);
                output[outputOffset] = computeStd ? Math.Sqrt(variance) : variance;
            }
        }

        /// <summary>
        /// General helper for axis Var/Std with runtime type dispatch.
        /// </summary>
        internal static unsafe void AxisVarStdGeneralHelper(
            void* input, double* output,
            int* inputStrides, int* inputShape, int* outputStrides,
            int axis, int axisSize, int ndim, int outputSize,
            NPTypeCode inputType, bool computeStd, int ddof)
        {
            int axisStride = inputStrides[axis];
            int inputElemSize = inputType.SizeOf();

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

            byte* inputBytes = (byte*)input;
            double divisor = axisSize - ddof;
            if (divisor <= 0) divisor = 1;

            for (int outIdx = 0; outIdx < outputSize; outIdx++)
            {
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

                // Pass 1: Compute mean
                double sum = 0;
                for (int i = 0; i < axisSize; i++)
                {
                    int inputOffset = inputBaseOffset + i * axisStride;
                    sum += ReadAsDouble(inputBytes + inputOffset * inputElemSize, inputType);
                }
                double mean = sum / axisSize;

                // Pass 2: Sum of squared differences
                double sqDiffSum = 0;
                for (int i = 0; i < axisSize; i++)
                {
                    int inputOffset = inputBaseOffset + i * axisStride;
                    double val = ReadAsDouble(inputBytes + inputOffset * inputElemSize, inputType);
                    double diff = val - mean;
                    sqDiffSum += diff * diff;
                }

                double variance = sqDiffSum / divisor;
                output[outputOffset] = computeStd ? Math.Sqrt(variance) : variance;
            }
        }

        #endregion
    }
}
