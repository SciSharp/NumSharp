using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
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
    public static partial class ILKernelGenerator
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
                // Complex variance = E[|z - mean(z)|²], returns double. Can't use the typed
                // helper path (uses double intermediate, dropping imaginary) — dedicated helper
                // following the Decimal convention.
                NPTypeCode.Complex => CreateAxisVarStdKernelTypedComplex(key),
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

            return (void* input, void* output, long* inputStrides, long* inputShape,
                    long* outputStrides, int axis, long axisSize, int ndim, long outputSize) =>
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

            return (void* input, void* output, long* inputStrides, long* inputShape,
                    long* outputStrides, int axis, long axisSize, int ndim, long outputSize) =>
            {
                AxisVarStdDecimalHelper(
                    (decimal*)input, (double*)output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize,
                    isStd, ddof: 0);
            };
        }

        /// <summary>
        /// Create a Complex axis Var/Std kernel.
        /// </summary>
        private static unsafe AxisReductionKernel CreateAxisVarStdKernelTypedComplex(AxisReductionKernelKey key)
        {
            bool isStd = key.Op == ReductionOp.Std;

            return (void* input, void* output, long* inputStrides, long* inputShape,
                    long* outputStrides, int axis, long axisSize, int ndim, long outputSize) =>
            {
                AxisVarStdComplexHelper(
                    (System.Numerics.Complex*)input, (double*)output,
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

            return (void* input, void* output, long* inputStrides, long* inputShape,
                    long* outputStrides, int axis, long axisSize, int ndim, long outputSize) =>
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
            long* inputStrides, long* inputShape, long* outputStrides,
            int axis, long axisSize, int ndim, long outputSize,
            bool computeStd, int ddof)
            where TInput : unmanaged
        {
            long axisStride = inputStrides[axis];
            bool axisContiguous = axisStride == 1;

            // ─────────────────────────────────────────────────────────────────
            // FAST PATH — leading-axis on C-contiguous input (axis < ndim-1).
            //
            // Mirror of AxisReductionSimdHelper's leading-axis path: walk axis
            // rows sequentially and column-tile the accumulators into the inner
            // slab. Eliminates the 85× regression on std(axis=0) where the
            // generic per-output loop was walking stride=innerSize columns one
            // by one (cache-cold) AND repeating the walk in pass 2.
            //
            // Two-pass algorithm:
            //   Pass 1 — column-tiled sum → per-column double mean.
            //   Pass 2 — column-tiled (val - mean[col])² accumulation.
            // The shared temp buffer holds N doubles per outer slab; rented
            // from ArrayPool to avoid pressuring the LOH for large inner dims.
            // ─────────────────────────────────────────────────────────────────
            if (axis < ndim - 1 && IsCContig(inputStrides, inputShape, ndim))
            {
                long innerSize = 1;
                for (int d = axis + 1; d < ndim; d++) innerSize *= inputShape[d];
                long outerSize = 1;
                for (int d = 0; d < axis; d++) outerSize *= inputShape[d];

                AxisVarStdLeadingTyped<TInput>(input, output, outerSize, axisSize, innerSize, computeStd, ddof);
                return;
            }

            // Compute output dimension strides for coordinate calculation
            int outputNdim = ndim - 1;

            // Store output dimension strides in a fixed array for parallel access
            long[] outputDimStridesArray = new long[outputNdim > 0 ? outputNdim : 1];
            if (outputNdim > 0)
            {
                outputDimStridesArray[outputNdim - 1] = 1;
                for (int d = outputNdim - 2; d >= 0; d--)
                {
                    int nextInputDim = (d + 1) >= axis ? d + 2 : d + 1;
                    outputDimStridesArray[d] = outputDimStridesArray[d + 1] * inputShape[nextInputDim];
                }
            }

            // Divisor for variance (size - ddof)
            double divisor = axisSize - ddof;
            if (divisor <= 0)
                divisor = 1; // Prevent division by zero; will produce NaN behavior

            // Sequential loop over output elements
            for (long outIdx = 0; outIdx < outputSize; outIdx++)
                {
                    // Convert linear output index to coordinates and compute offsets
                    long remaining = outIdx;
                    long inputBaseOffset = 0;
                    long outputOffset = 0;

                    for (int d = 0; d < outputNdim; d++)
                    {
                        int inputDim = d >= axis ? d + 1 : d;
                        long coord = remaining / outputDimStridesArray[d];
                        remaining = remaining % outputDimStridesArray[d];
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
                        for (long i = 0; i < axisSize; i++)
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
                        for (long i = 0; i < axisSize; i++)
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
        /// Column-tiled Var/Std for the C-contig leading-axis case (axis &lt; ndim-1).
        ///
        /// Output shape is (outerSize × innerSize) flattened; for each outer slab we
        /// hold per-column sums and squared-diffs in a rented double buffer, walking
        /// axis rows sequentially so the input streams contig from memory and per-
        /// column accumulators stay hot in L1.
        ///
        /// Two passes over the input. Float widening (Vector256&lt;float&gt; → 2× Vector256&lt;double&gt;)
        /// keeps the SIMD body engaged for the dominant TInput=float case.
        /// </summary>
        private static unsafe void AxisVarStdLeadingTyped<TInput>(
            TInput* input, double* output,
            long outerSize, long axisSize, long innerSize,
            bool computeStd, int ddof)
            where TInput : unmanaged
        {
            // ddof semantics match the per-output path: clamp to 1 to produce NaN
            // through (sqDiff/divisor) when ddof >= axisSize. Matches existing parity.
            double divisor = axisSize - ddof;
            if (divisor <= 0) divisor = 1;

            // Two scratch buffers per outer slab: sums (Pass 1 → means after divide)
            // and sqdiffs (Pass 2 → variance after divide). Rented to avoid LOH
            // pressure for very wide innerSize.
            var pool = System.Buffers.ArrayPool<double>.Shared;
            double[] scratch = pool.Rent((int)Math.Min(innerSize * 2, int.MaxValue));
            try
            {
                fixed (double* scratchPin = scratch)
                {
                    double* sums = scratchPin;
                    double* sqdiffs = scratchPin + innerSize;

                    for (long o = 0; o < outerSize; o++)
                    {
                        TInput* inBase = input + o * axisSize * innerSize;
                        double* outSlab = output + o * innerSize;

                        // ─── Pass 1: per-column sum ───────────────────────────
                        // Initialize from row 0 (widening cast TInput → double).
                        // Subsequent rows accumulate in place.
                        AxisVarStdSeedRowAsDouble<TInput>(inBase, sums, innerSize);
                        for (long a = 1; a < axisSize; a++)
                            AxisVarStdAddRowAsDouble<TInput>(inBase + a * innerSize, sums, innerSize);

                        // Compute means in place (sums[] becomes means[])
                        double invN = 1.0 / axisSize;
                        for (long i = 0; i < innerSize; i++)
                            sums[i] *= invN;

                        // ─── Pass 2: per-column sum of squared diffs ─────────
                        // Zero the sqdiffs buffer.
                        new Span<double>(sqdiffs, (int)innerSize).Clear();
                        for (long a = 0; a < axisSize; a++)
                            AxisVarStdAccumulateSqDiff<TInput>(inBase + a * innerSize, sums, sqdiffs, innerSize);

                        // Finalize: variance = sqdiff/divisor, std = sqrt(variance).
                        double invDiv = 1.0 / divisor;
                        if (computeStd)
                        {
                            for (long i = 0; i < innerSize; i++)
                                outSlab[i] = Math.Sqrt(sqdiffs[i] * invDiv);
                        }
                        else
                        {
                            for (long i = 0; i < innerSize; i++)
                                outSlab[i] = sqdiffs[i] * invDiv;
                        }
                    }
                }
            }
            finally
            {
                pool.Return(scratch);
            }
        }

        /// <summary>
        /// Seed the per-column accumulator from a single input row, widening TInput → double.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void AxisVarStdSeedRowAsDouble<TInput>(TInput* row, double* sums, long n)
            where TInput : unmanaged
        {
            if (typeof(TInput) == typeof(double))
            {
                Buffer.MemoryCopy(row, sums, n * sizeof(double), n * sizeof(double));
                return;
            }
            if (typeof(TInput) == typeof(float))
            {
                float* p = (float*)(void*)row;
                long i = 0;
                if (Vector256.IsHardwareAccelerated && n >= Vector256<float>.Count)
                {
                    int vc = Vector256<float>.Count;            // 8 floats
                    long end = n - vc;
                    for (; i <= end; i += vc)
                    {
                        var (lo, hi) = Vector256.Widen(Vector256.Load(p + i));
                        Vector256.Store(lo, sums + i);
                        Vector256.Store(hi, sums + i + Vector256<double>.Count);
                    }
                }
                for (; i < n; i++) sums[i] = p[i];
                return;
            }
            // Scalar fallback for other integral types (int16/int32/etc.) — JIT folds typeof.
            for (long i = 0; i < n; i++) sums[i] = ConvertToDouble(row[i]);
        }

        /// <summary>
        /// Add one input row to the per-column accumulator, widening TInput → double.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void AxisVarStdAddRowAsDouble<TInput>(TInput* row, double* sums, long n)
            where TInput : unmanaged
        {
            if (typeof(TInput) == typeof(double))
            {
                double* p = (double*)(void*)row;
                long i = 0;
                if (Vector256.IsHardwareAccelerated && n >= Vector256<double>.Count)
                {
                    int vc = Vector256<double>.Count;
                    long end = n - vc;
                    for (; i <= end; i += vc)
                        Vector256.Store(Vector256.Add(Vector256.Load(sums + i), Vector256.Load(p + i)), sums + i);
                }
                for (; i < n; i++) sums[i] += p[i];
                return;
            }
            if (typeof(TInput) == typeof(float))
            {
                float* p = (float*)(void*)row;
                long i = 0;
                if (Vector256.IsHardwareAccelerated && n >= Vector256<float>.Count)
                {
                    int vc = Vector256<float>.Count;
                    int vcD = Vector256<double>.Count;
                    long end = n - vc;
                    for (; i <= end; i += vc)
                    {
                        var (lo, hi) = Vector256.Widen(Vector256.Load(p + i));
                        Vector256.Store(Vector256.Add(Vector256.Load(sums + i), lo), sums + i);
                        Vector256.Store(Vector256.Add(Vector256.Load(sums + i + vcD), hi), sums + i + vcD);
                    }
                }
                for (; i < n; i++) sums[i] += p[i];
                return;
            }
            for (long i = 0; i < n; i++) sums[i] += ConvertToDouble(row[i]);
        }

        /// <summary>
        /// Accumulate (val - mean[col])² per column from one input row.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void AxisVarStdAccumulateSqDiff<TInput>(TInput* row, double* means, double* sqdiffs, long n)
            where TInput : unmanaged
        {
            if (typeof(TInput) == typeof(double))
            {
                double* p = (double*)(void*)row;
                long i = 0;
                if (Vector256.IsHardwareAccelerated && n >= Vector256<double>.Count)
                {
                    int vc = Vector256<double>.Count;
                    long end = n - vc;
                    for (; i <= end; i += vc)
                    {
                        var diff = Vector256.Subtract(Vector256.Load(p + i), Vector256.Load(means + i));
                        Vector256.Store(Vector256.Add(Vector256.Load(sqdiffs + i), Vector256.Multiply(diff, diff)), sqdiffs + i);
                    }
                }
                for (; i < n; i++) { double d = p[i] - means[i]; sqdiffs[i] += d * d; }
                return;
            }
            if (typeof(TInput) == typeof(float))
            {
                float* p = (float*)(void*)row;
                long i = 0;
                if (Vector256.IsHardwareAccelerated && n >= Vector256<float>.Count)
                {
                    int vc = Vector256<float>.Count;
                    int vcD = Vector256<double>.Count;
                    long end = n - vc;
                    for (; i <= end; i += vc)
                    {
                        var (lo, hi) = Vector256.Widen(Vector256.Load(p + i));
                        var diffLo = Vector256.Subtract(lo, Vector256.Load(means + i));
                        var diffHi = Vector256.Subtract(hi, Vector256.Load(means + i + vcD));
                        Vector256.Store(Vector256.Add(Vector256.Load(sqdiffs + i), Vector256.Multiply(diffLo, diffLo)), sqdiffs + i);
                        Vector256.Store(Vector256.Add(Vector256.Load(sqdiffs + i + vcD), Vector256.Multiply(diffHi, diffHi)), sqdiffs + i + vcD);
                    }
                }
                for (; i < n; i++) { double d = p[i] - means[i]; sqdiffs[i] += d * d; }
                return;
            }
            for (long i = 0; i < n; i++)
            {
                double d = ConvertToDouble(row[i]) - means[i];
                sqdiffs[i] += d * d;
            }
        }

        /// <summary>
        /// Sum contiguous axis as double (for mean computation in Var/Std).
        /// </summary>
        private static unsafe double SumContiguousAxisDouble<T>(T* data, long size)
            where T : unmanaged
        {
            double sum = 0;

            if (typeof(T) == typeof(double))
            {
                double* p = (double*)(void*)data;
                if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
                {
                    int vectorCount = Vector256<double>.Count;
                    long vectorEnd = size - vectorCount;
                    var sumVec = Vector256<double>.Zero;
                    long i = 0;
                    for (; i <= vectorEnd; i += vectorCount)
                        sumVec = Vector256.Add(sumVec, Vector256.Load(p + i));
                    sum = Vector256.Sum(sumVec);
                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    for (long i = 0; i < size; i++)
                        sum += p[i];
                }
            }
            else if (typeof(T) == typeof(float))
            {
                float* p = (float*)(void*)data;
                if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
                {
                    int vectorCount = Vector256<float>.Count;
                    long vectorEnd = size - vectorCount;
                    var sumVec = Vector256<float>.Zero;
                    long i = 0;
                    for (; i <= vectorEnd; i += vectorCount)
                        sumVec = Vector256.Add(sumVec, Vector256.Load(p + i));
                    sum = Vector256.Sum(sumVec);
                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    for (long i = 0; i < size; i++)
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
                    long vectorEnd = size - vectorCount;
                    long totalSum = 0;
                    long i = 0;
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
                    for (long i = 0; i < size; i++)
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
                    long vectorEnd = size - vectorCount;
                    var sumVec = Vector256<long>.Zero;
                    long i = 0;
                    for (; i <= vectorEnd; i += vectorCount)
                        sumVec = Vector256.Add(sumVec, Vector256.Load(p + i));
                    sum = Vector256.Sum(sumVec);
                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    for (long i = 0; i < size; i++)
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
                    long vectorEnd = size - vectorCount;
                    long totalSum = 0;
                    long i = 0;
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
                    for (long i = 0; i < size; i++)
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
                    long vectorEnd = size - vectorCount;
                    long totalSum = 0;
                    long i = 0;
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
                    for (long i = 0; i < size; i++)
                        totalSum += p[i];
                    sum = totalSum;
                }
            }
            else
            {
                // For other types (ushort, uint, ulong, char), use scalar loop
                for (long i = 0; i < size; i++)
                    sum += ConvertToDouble(data[i]);
            }

            return sum;
        }

        /// <summary>
        /// Sum squared differences from mean for contiguous axis.
        /// </summary>
        private static unsafe double SumSquaredDiffContiguous<T>(T* data, long size, double mean)
            where T : unmanaged
        {
            double sqDiffSum = 0;

            if (typeof(T) == typeof(double))
            {
                double* p = (double*)(void*)data;
                if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
                {
                    int vectorCount = Vector256<double>.Count;
                    long vectorEnd = size - vectorCount;
                    var meanVec = Vector256.Create(mean);
                    var sqDiffVec = Vector256<double>.Zero;
                    long i = 0;
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
                    for (long i = 0; i < size; i++)
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
                    long vectorEnd = size - vectorCount;
                    var meanVec = Vector256.Create(fMean);
                    var sqDiffVec = Vector256<float>.Zero;
                    long i = 0;
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
                    for (long i = 0; i < size; i++)
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
                long unrollEnd = size - 4;
                long i = 0;
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
                long unrollEnd = size - 4;
                long i = 0;
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
                long unrollEnd = size - 4;
                long i = 0;
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
                long unrollEnd = size - 4;
                long i = 0;
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
                for (long i = 0; i < size; i++)
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
            long* inputStrides, long* inputShape, long* outputStrides,
            int axis, long axisSize, int ndim, long outputSize,
            bool computeStd, int ddof)
        {
            long axisStride = inputStrides[axis];

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

            double divisor = axisSize - ddof;
            if (divisor <= 0) divisor = 1;

            for (long outIdx = 0; outIdx < outputSize; outIdx++)
            {
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

                decimal* axisStart = input + inputBaseOffset;

                // Pass 1: Compute mean
                decimal sum = 0;
                for (long i = 0; i < axisSize; i++)
                    sum += axisStart[i * axisStride];
                decimal mean = sum / axisSize;

                // Pass 2: Sum of squared differences
                decimal sqDiffSum = 0;
                for (long i = 0; i < axisSize; i++)
                {
                    decimal diff = axisStart[i * axisStride] - mean;
                    sqDiffSum += diff * diff;
                }

                double variance = (double)(sqDiffSum / (decimal)divisor);
                output[outputOffset] = computeStd ? Math.Sqrt(variance) : variance;
            }
        }

        /// <summary>
        /// Complex helper for axis Var/Std. NumPy parity:
        ///   variance = E[|z - mean(z)|²] = (sum((Re - muR)² + (Im - muI)²)) / (N - ddof)
        /// where mean is Complex, but the variance itself is a real (double).
        /// </summary>
        internal static unsafe void AxisVarStdComplexHelper(
            System.Numerics.Complex* input, double* output,
            long* inputStrides, long* inputShape, long* outputStrides,
            int axis, long axisSize, int ndim, long outputSize,
            bool computeStd, int ddof)
        {
            long axisStride = inputStrides[axis];

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

            double divisor = axisSize - ddof;
            if (divisor <= 0) divisor = 1;

            for (long outIdx = 0; outIdx < outputSize; outIdx++)
            {
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

                System.Numerics.Complex* axisStart = input + inputBaseOffset;

                // Pass 1: Compute Complex mean along axis
                double sumR = 0.0, sumI = 0.0;
                for (long i = 0; i < axisSize; i++)
                {
                    var z = axisStart[i * axisStride];
                    sumR += z.Real;
                    sumI += z.Imaginary;
                }
                double muR = sumR / axisSize;
                double muI = sumI / axisSize;

                // Pass 2: Sum of |z - mean|² (= dR² + dI²)
                double sqDiffSum = 0.0;
                for (long i = 0; i < axisSize; i++)
                {
                    var z = axisStart[i * axisStride];
                    double dR = z.Real - muR;
                    double dI = z.Imaginary - muI;
                    sqDiffSum += dR * dR + dI * dI;
                }

                double variance = sqDiffSum / divisor;
                output[outputOffset] = computeStd ? Math.Sqrt(variance) : variance;
            }
        }

        /// <summary>
        /// General helper for axis Var/Std with runtime type dispatch.
        /// </summary>
        internal static unsafe void AxisVarStdGeneralHelper(
            void* input, double* output,
            long* inputStrides, long* inputShape, long* outputStrides,
            int axis, long axisSize, int ndim, long outputSize,
            NPTypeCode inputType, bool computeStd, int ddof)
        {
            long axisStride = inputStrides[axis];
            int inputElemSize = inputType.SizeOf();

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

            byte* inputBytes = (byte*)input;
            double divisor = axisSize - ddof;
            if (divisor <= 0) divisor = 1;

            for (long outIdx = 0; outIdx < outputSize; outIdx++)
            {
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

                // Pass 1: Compute mean
                double sum = 0;
                for (long i = 0; i < axisSize; i++)
                {
                    long inputOffset = inputBaseOffset + i * axisStride;
                    sum += ReadAsDouble(inputBytes + inputOffset * inputElemSize, inputType);
                }
                double mean = sum / axisSize;

                // Pass 2: Sum of squared differences
                double sqDiffSum = 0;
                for (long i = 0; i < axisSize; i++)
                {
                    long inputOffset = inputBaseOffset + i * axisStride;
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
