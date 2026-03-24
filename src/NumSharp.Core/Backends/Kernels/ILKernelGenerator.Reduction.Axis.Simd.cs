using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// ILKernelGenerator.Reduction.Axis.Simd.cs - SIMD Axis Reduction Kernels
// =============================================================================
//
// RESPONSIBILITY:
//   - CreateAxisReductionKernelTyped<T> - typed SIMD kernel
//   - AxisReductionSimdHelper<T> - main SIMD helper
//   - ReduceContiguousAxis variants (SIMD256, SIMD128, scalar)
//   - ReduceStridedAxis with AVX2 gather for float/double
//   - Vector identity/combine/horizontal helpers
//   - SIMD helper methods for DefaultEngine
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        #region Typed SIMD Axis Reduction
        private static unsafe AxisReductionKernel CreateAxisReductionKernelTyped<T>(AxisReductionKernelKey key)
            where T : unmanaged
        {
            return (void* input, void* output, int* inputStrides, int* inputShape,
                    int* outputStrides, int axis, int axisSize, int ndim, int outputSize) =>
            {
                AxisReductionSimdHelper<T>(
                    (T*)input, (T*)output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize,
                    key.Op);
            };
        }

        /// <summary>
        /// SIMD helper for axis reduction operations.
        /// Reduces along a specific axis, writing results to output array.
        /// </summary>
        /// <typeparam name="T">Element type</typeparam>
        /// <param name="input">Input data pointer</param>
        /// <param name="output">Output data pointer</param>
        /// <param name="inputStrides">Input strides (element units)</param>
        /// <param name="inputShape">Input shape</param>
        /// <param name="outputStrides">Output strides (element units)</param>
        /// <param name="axis">Axis to reduce along</param>
        /// <param name="axisSize">Size of the axis being reduced</param>
        /// <param name="ndim">Number of input dimensions</param>
        /// <param name="outputSize">Total number of output elements</param>
        /// <param name="op">Reduction operation</param>
        internal static unsafe void AxisReductionSimdHelper<T>(
            T* input, T* output,
            int* inputStrides, int* inputShape, int* outputStrides,
            int axis, int axisSize, int ndim, int outputSize,
            ReductionOp op)
            where T : unmanaged
        {
            int axisStride = inputStrides[axis];

            // Check if the reduction axis is contiguous (stride == 1)
            bool axisContiguous = axisStride == 1;

            // Compute output shape strides for coordinate calculation
            // Output has ndim-1 dimensions (axis removed)
            int outputNdim = ndim - 1;

            // Store output dimension strides in a fixed array for parallel access
            long[] outputDimStridesArray = new long[outputNdim > 0 ? outputNdim : 1];
            if (outputNdim > 0)
            {
                outputDimStridesArray[outputNdim - 1] = 1;
                for (int d = outputNdim - 2; d >= 0; d--)
                {
                    // Map output dimension d to input dimension (d if d < axis, d+1 if d >= axis)
                    int nextInputDim = (d + 1) >= axis ? d + 2 : d + 1;
                    outputDimStridesArray[d] = outputDimStridesArray[d + 1] * inputShape[nextInputDim];
                }
            }

            // For Mean, use Sum operation then divide
            ReductionOp actualOp = op == ReductionOp.Mean ? ReductionOp.Sum : op;
            bool isMean = op == ReductionOp.Mean;

            // Sequential loop over output elements
            for (int outIdx = 0; outIdx < outputSize; outIdx++)
            {
                // Convert linear output index to coordinates and compute input base offset
                int remaining = outIdx;
                int inputBaseOffset = 0;
                int outputOffset = 0;

                for (int d = 0; d < outputNdim; d++)
                {
                    // Map output dimension d to input dimension
                    int inputDim = d >= axis ? d + 1 : d;

                    int coord = remaining / outputDimStridesArray[d];
                    remaining = remaining % outputDimStridesArray[d];

                    inputBaseOffset += coord * inputStrides[inputDim];
                    outputOffset += coord * outputStrides[d];
                }

                // Now reduce along the axis
                T* axisStart = input + inputBaseOffset;

                T result;
                if (axisContiguous)
                {
                    // Fast path: axis is contiguous, use SIMD
                    result = ReduceContiguousAxis(axisStart, axisSize, actualOp);
                }
                else
                {
                    // Strided path: axis is not contiguous, use SIMD gather if beneficial
                    result = ReduceStridedAxis(axisStart, axisSize, axisStride, actualOp);
                }

                // For Mean, divide by count
                if (isMean)
                    result = DivideByCountTyped(result, axisSize);

                output[outputOffset] = result;
            }
        }

        /// <summary>
        /// Divide a typed value by count (for Mean operation in SIMD path).
        /// </summary>
        private static T DivideByCountTyped<T>(T value, long count) where T : unmanaged
        {
            if (typeof(T) == typeof(float))
            {
                float result = (float)(object)value / count;
                return (T)(object)result;
            }
            if (typeof(T) == typeof(double))
            {
                double result = (double)(object)value / count;
                return (T)(object)result;
            }
            if (typeof(T) == typeof(int))
            {
                // Integer division
                int result = (int)(object)value / count;
                return (T)(object)result;
            }
            if (typeof(T) == typeof(long))
            {
                long result = (long)(object)value / count;
                return (T)(object)result;
            }
            if (typeof(T) == typeof(byte))
            {
                byte result = (byte)((byte)(object)value / count);
                return (T)(object)result;
            }
            if (typeof(T) == typeof(short))
            {
                short result = (short)((short)(object)value / count);
                return (T)(object)result;
            }
            if (typeof(T) == typeof(ushort))
            {
                ushort result = (ushort)((ushort)(object)value / count);
                return (T)(object)result;
            }
            if (typeof(T) == typeof(uint))
            {
                uint result = (uint)(object)value / (uint)count;
                return (T)(object)result;
            }
            if (typeof(T) == typeof(ulong))
            {
                ulong result = (ulong)(object)value / (ulong)count;
                return (T)(object)result;
            }
            // Fallback via double
            double dval = ConvertToDouble(value);
            return ConvertFromDouble<T>(dval / count);
        }

        /// <summary>
        /// Reduce a contiguous axis using SIMD.
        /// </summary>
        private static unsafe T ReduceContiguousAxis<T>(T* data, long size, ReductionOp op)
            where T : unmanaged
        {
            if (size == 0)
            {
                return GetIdentityValue<T>(op);
            }

            if (size == 1)
            {
                return data[0];
            }

            // Use SIMD for Sum, Prod, Min, Max
            if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && size >= Vector256<T>.Count)
            {
                return ReduceContiguousAxisSimd256(data, size, op);
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && size >= Vector128<T>.Count)
            {
                return ReduceContiguousAxisSimd128(data, size, op);
            }
            else
            {
                return ReduceContiguousAxisScalar(data, size, op);
            }
        }

        /// <summary>
        /// Reduce contiguous axis using Vector256 SIMD with 4x unrolling.
        /// Uses 4 independent accumulators to break dependency chains.
        /// </summary>
        private static unsafe T ReduceContiguousAxisSimd256<T>(T* data, long size, ReductionOp op)
            where T : unmanaged
        {
            int vectorCount = Vector256<T>.Count;
            long vectorEnd = size - vectorCount;

            // Initialize 4 independent accumulators for loop unrolling
            var acc0 = CreateIdentityVector256<T>(op);
            var acc1 = CreateIdentityVector256<T>(op);
            var acc2 = CreateIdentityVector256<T>(op);
            var acc3 = CreateIdentityVector256<T>(op);

            long unrollStep = vectorCount * 4;
            long unrollEnd = size - unrollStep;

            long i = 0;

            // 4x unrolled loop - process 4 vectors per iteration
            for (; i <= unrollEnd; i += unrollStep)
            {
                var v0 = Vector256.Load(data + i);
                var v1 = Vector256.Load(data + i + vectorCount);
                var v2 = Vector256.Load(data + i + vectorCount * 2);
                var v3 = Vector256.Load(data + i + vectorCount * 3);
                acc0 = CombineVectors256(acc0, v0, op);
                acc1 = CombineVectors256(acc1, v1, op);
                acc2 = CombineVectors256(acc2, v2, op);
                acc3 = CombineVectors256(acc3, v3, op);
            }

            // Tree reduction: 4 -> 2 -> 1
            var acc01 = CombineVectors256(acc0, acc1, op);
            var acc23 = CombineVectors256(acc2, acc3, op);
            var accumVec = CombineVectors256(acc01, acc23, op);

            // Remainder loop (0-3 vectors)
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector256.Load(data + i);
                accumVec = CombineVectors256(accumVec, vec, op);
            }

            // Horizontal reduce the vector
            T result = HorizontalReduce256(accumVec, op);

            // Process scalar tail
            for (; i < size; i++)
            {
                result = CombineScalars(result, data[i], op);
            }

            return result;
        }

        /// <summary>
        /// Reduce contiguous axis using Vector128 SIMD with 4x unrolling.
        /// Uses 4 independent accumulators to break dependency chains.
        /// </summary>
        private static unsafe T ReduceContiguousAxisSimd128<T>(T* data, long size, ReductionOp op)
            where T : unmanaged
        {
            int vectorCount = Vector128<T>.Count;
            long vectorEnd = size - vectorCount;

            // Initialize 4 independent accumulators for loop unrolling
            var acc0 = CreateIdentityVector128<T>(op);
            var acc1 = CreateIdentityVector128<T>(op);
            var acc2 = CreateIdentityVector128<T>(op);
            var acc3 = CreateIdentityVector128<T>(op);

            long unrollStep = vectorCount * 4;
            long unrollEnd = size - unrollStep;

            long i = 0;

            // 4x unrolled loop - process 4 vectors per iteration
            for (; i <= unrollEnd; i += unrollStep)
            {
                var v0 = Vector128.Load(data + i);
                var v1 = Vector128.Load(data + i + vectorCount);
                var v2 = Vector128.Load(data + i + vectorCount * 2);
                var v3 = Vector128.Load(data + i + vectorCount * 3);
                acc0 = CombineVectors128(acc0, v0, op);
                acc1 = CombineVectors128(acc1, v1, op);
                acc2 = CombineVectors128(acc2, v2, op);
                acc3 = CombineVectors128(acc3, v3, op);
            }

            // Tree reduction: 4 -> 2 -> 1
            var acc01 = CombineVectors128(acc0, acc1, op);
            var acc23 = CombineVectors128(acc2, acc3, op);
            var accumVec = CombineVectors128(acc01, acc23, op);

            // Remainder loop (0-3 vectors)
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector128.Load(data + i);
                accumVec = CombineVectors128(accumVec, vec, op);
            }

            // Horizontal reduce the vector
            T result = HorizontalReduce128(accumVec, op);

            // Process scalar tail
            for (; i < size; i++)
            {
                result = CombineScalars(result, data[i], op);
            }

            return result;
        }

        /// <summary>
        /// Reduce contiguous axis using scalar loop.
        /// </summary>
        private static unsafe T ReduceContiguousAxisScalar<T>(T* data, long size, ReductionOp op)
            where T : unmanaged
        {
            T result = GetIdentityValue<T>(op);

            for (long i = 0; i < size; i++)
            {
                result = CombineScalars(result, data[i], op);
            }

            return result;
        }

        /// <summary>
        /// Reduce a strided axis (non-contiguous).
        /// Uses AVX2 gather instructions for float/double when beneficial (stride fits in int32).
        /// </summary>
        private static unsafe T ReduceStridedAxis<T>(T* data, long size, long stride, ReductionOp op)
            where T : unmanaged
        {
            if (size == 0)
                return GetIdentityValue<T>(op);

            if (size == 1)
                return data[0];

            // Try AVX2 gather for float/double - provides ~2-3x speedup for strided access
            // Only beneficial when we have enough elements to amortize gather overhead
            if (Avx2.IsSupported && size >= 8)
            {
                if (typeof(T) == typeof(float))
                {
                    return (T)(object)ReduceStridedAxisGatherFloat((float*)data, size, stride, op);
                }
                if (typeof(T) == typeof(double))
                {
                    return (T)(object)ReduceStridedAxisGatherDouble((double*)data, size, stride, op);
                }
            }

            // Scalar fallback with 4x loop unrolling for better ILP
            return ReduceStridedAxisScalar(data, size, stride, op);
        }

        /// <summary>
        /// Strided reduction using AVX2 gather for float.
        /// Uses Vector256 gather to load 8 floats at once from strided positions.
        /// </summary>
        private static unsafe float ReduceStridedAxisGatherFloat(float* data, long size, long stride, ReductionOp op)
        {
            // Create index vector: [0, stride, 2*stride, ..., 7*stride]
            // Note: AVX2 gather requires int32 indices, so stride must fit in int32
            int strideInt = (int)stride;
            var indices = Vector256.Create(
                0, strideInt, strideInt * 2, strideInt * 3,
                strideInt * 4, strideInt * 5, strideInt * 6, strideInt * 7);

            int vectorCount = 8; // Vector256<float>.Count
            long vectorEnd = size - vectorCount;

            var accum = CreateIdentityVector256<float>(op);

            long i = 0;

            // Main gather loop - process 8 elements per iteration
            for (; i <= vectorEnd; i += vectorCount)
            {
                // GatherVector256: load data[indices[j]] for j in 0..7
                // Scale = 4 because float is 4 bytes
                var gathered = Avx2.GatherVector256(data + i * stride, indices, 4);
                accum = CombineVectors256(accum, gathered, op);
            }

            // Horizontal reduce the vector
            float result = HorizontalReduce256(accum, op);

            // Process remaining elements with scalar loop
            for (; i < size; i++)
            {
                result = CombineScalars(result, data[i * stride], op);
            }

            return result;
        }

        /// <summary>
        /// Strided reduction using AVX2 gather for double.
        /// Uses Vector256 gather to load 4 doubles at once from strided positions.
        /// </summary>
        private static unsafe double ReduceStridedAxisGatherDouble(double* data, long size, long stride, ReductionOp op)
        {
            // Create index vector: [0, stride, 2*stride, 3*stride]
            // Note: AVX2 gather requires int32 indices, so stride must fit in int32
            int strideInt = (int)stride;
            var indices = Vector128.Create(0, strideInt, strideInt * 2, strideInt * 3);

            int vectorCount = 4; // Vector256<double>.Count
            long vectorEnd = size - vectorCount;

            var accum = CreateIdentityVector256<double>(op);

            long i = 0;

            // Main gather loop - process 4 elements per iteration
            for (; i <= vectorEnd; i += vectorCount)
            {
                // GatherVector256: load data[indices[j]] for j in 0..3
                // Scale = 8 because double is 8 bytes
                var gathered = Avx2.GatherVector256(data + i * stride, indices, 8);
                accum = CombineVectors256(accum, gathered, op);
            }

            // Horizontal reduce the vector
            double result = HorizontalReduce256(accum, op);

            // Process remaining elements with scalar loop
            for (; i < size; i++)
            {
                result = CombineScalars(result, data[i * stride], op);
            }

            return result;
        }

        /// <summary>
        /// Scalar strided reduction with 4x loop unrolling.
        /// </summary>
        private static unsafe T ReduceStridedAxisScalar<T>(T* data, long size, long stride, ReductionOp op)
            where T : unmanaged
        {
            T result = GetIdentityValue<T>(op);

            // 4x unrolled loop for better instruction-level parallelism
            long unrollEnd = size - 4;
            long i = 0;

            for (; i <= unrollEnd; i += 4)
            {
                T v0 = data[i * stride];
                T v1 = data[(i + 1) * stride];
                T v2 = data[(i + 2) * stride];
                T v3 = data[(i + 3) * stride];

                result = CombineScalars(result, v0, op);
                result = CombineScalars(result, v1, op);
                result = CombineScalars(result, v2, op);
                result = CombineScalars(result, v3, op);
            }

            // Handle remaining elements
            for (; i < size; i++)
            {
                result = CombineScalars(result, data[i * stride], op);
            }

            return result;
        }

        /// <summary>
        /// Get the identity value for a reduction operation.
        /// </summary>
        private static T GetIdentityValue<T>(ReductionOp op) where T : unmanaged
        {
            if (typeof(T) == typeof(float))
            {
                float val = op switch
                {
                    ReductionOp.Sum => 0f,
                    ReductionOp.Prod => 1f,
                    ReductionOp.Min => float.PositiveInfinity,
                    ReductionOp.Max => float.NegativeInfinity,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(double))
            {
                double val = op switch
                {
                    ReductionOp.Sum => 0.0,
                    ReductionOp.Prod => 1.0,
                    ReductionOp.Min => double.PositiveInfinity,
                    ReductionOp.Max => double.NegativeInfinity,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(int))
            {
                int val = op switch
                {
                    ReductionOp.Sum => 0,
                    ReductionOp.Prod => 1,
                    ReductionOp.Min => int.MaxValue,
                    ReductionOp.Max => int.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(long))
            {
                long val = op switch
                {
                    ReductionOp.Sum => 0L,
                    ReductionOp.Prod => 1L,
                    ReductionOp.Min => long.MaxValue,
                    ReductionOp.Max => long.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(byte))
            {
                byte val = op switch
                {
                    ReductionOp.Sum => 0,
                    ReductionOp.Prod => 1,
                    ReductionOp.Min => byte.MaxValue,
                    ReductionOp.Max => byte.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(short))
            {
                short val = op switch
                {
                    ReductionOp.Sum => 0,
                    ReductionOp.Prod => 1,
                    ReductionOp.Min => short.MaxValue,
                    ReductionOp.Max => short.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(ushort))
            {
                ushort val = op switch
                {
                    ReductionOp.Sum => 0,
                    ReductionOp.Prod => 1,
                    ReductionOp.Min => ushort.MaxValue,
                    ReductionOp.Max => ushort.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(uint))
            {
                uint val = op switch
                {
                    ReductionOp.Sum => 0u,
                    ReductionOp.Prod => 1u,
                    ReductionOp.Min => uint.MaxValue,
                    ReductionOp.Max => uint.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(ulong))
            {
                ulong val = op switch
                {
                    ReductionOp.Sum => 0UL,
                    ReductionOp.Prod => 1UL,
                    ReductionOp.Min => ulong.MaxValue,
                    ReductionOp.Max => ulong.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }

            throw new NotSupportedException($"Type {typeof(T)} not supported for axis reduction");
        }

        /// <summary>
        /// Create identity Vector256 for reduction operation.
        /// </summary>
        private static Vector256<T> CreateIdentityVector256<T>(ReductionOp op) where T : unmanaged
        {
            T identity = GetIdentityValue<T>(op);
            return Vector256.Create(identity);
        }

        /// <summary>
        /// Create identity Vector128 for reduction operation.
        /// </summary>
        private static Vector128<T> CreateIdentityVector128<T>(ReductionOp op) where T : unmanaged
        {
            T identity = GetIdentityValue<T>(op);
            return Vector128.Create(identity);
        }

        /// <summary>
        /// Combine two Vector256 values using reduction operation.
        /// </summary>
        private static Vector256<T> CombineVectors256<T>(Vector256<T> a, Vector256<T> b, ReductionOp op)
            where T : unmanaged
        {
            return op switch
            {
                ReductionOp.Sum => Vector256.Add(a, b),
                ReductionOp.Prod => Vector256.Multiply(a, b),
                ReductionOp.Min => Vector256.Min(a, b),
                ReductionOp.Max => Vector256.Max(a, b),
                _ => throw new NotSupportedException()
            };
        }

        /// <summary>
        /// Combine two Vector128 values using reduction operation.
        /// </summary>
        private static Vector128<T> CombineVectors128<T>(Vector128<T> a, Vector128<T> b, ReductionOp op)
            where T : unmanaged
        {
            return op switch
            {
                ReductionOp.Sum => Vector128.Add(a, b),
                ReductionOp.Prod => Vector128.Multiply(a, b),
                ReductionOp.Min => Vector128.Min(a, b),
                ReductionOp.Max => Vector128.Max(a, b),
                _ => throw new NotSupportedException()
            };
        }

        /// <summary>
        /// Horizontal reduce Vector256 to scalar.
        /// </summary>
        private static T HorizontalReduce256<T>(Vector256<T> vec, ReductionOp op) where T : unmanaged
        {
            // First reduce to Vector128
            var lower = vec.GetLower();
            var upper = vec.GetUpper();
            var combined = CombineVectors128(lower, upper, op);

            return HorizontalReduce128(combined, op);
        }

        /// <summary>
        /// Horizontal reduce Vector128 to scalar.
        /// </summary>
        private static T HorizontalReduce128<T>(Vector128<T> vec, ReductionOp op) where T : unmanaged
        {
            int count = Vector128<T>.Count;
            T result = vec.GetElement(0);

            for (int i = 1; i < count; i++)
            {
                result = CombineScalars(result, vec.GetElement(i), op);
            }

            return result;
        }

        /// <summary>
        /// Combine two scalar values using reduction operation.
        /// </summary>
        private static T CombineScalars<T>(T a, T b, ReductionOp op) where T : unmanaged
        {
            if (typeof(T) == typeof(float))
            {
                float fa = (float)(object)a;
                float fb = (float)(object)b;
                float result = op switch
                {
                    ReductionOp.Sum => fa + fb,
                    ReductionOp.Prod => fa * fb,
                    ReductionOp.Min => Math.Min(fa, fb),
                    ReductionOp.Max => Math.Max(fa, fb),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(double))
            {
                double da = (double)(object)a;
                double db = (double)(object)b;
                double result = op switch
                {
                    ReductionOp.Sum => da + db,
                    ReductionOp.Prod => da * db,
                    ReductionOp.Min => Math.Min(da, db),
                    ReductionOp.Max => Math.Max(da, db),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(int))
            {
                int ia = (int)(object)a;
                int ib = (int)(object)b;
                int result = op switch
                {
                    ReductionOp.Sum => ia + ib,
                    ReductionOp.Prod => ia * ib,
                    ReductionOp.Min => Math.Min(ia, ib),
                    ReductionOp.Max => Math.Max(ia, ib),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(long))
            {
                long la = (long)(object)a;
                long lb = (long)(object)b;
                long result = op switch
                {
                    ReductionOp.Sum => la + lb,
                    ReductionOp.Prod => la * lb,
                    ReductionOp.Min => Math.Min(la, lb),
                    ReductionOp.Max => Math.Max(la, lb),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(byte))
            {
                int ba = (byte)(object)a;
                int bb = (byte)(object)b;
                byte result = op switch
                {
                    ReductionOp.Sum => (byte)(ba + bb),
                    ReductionOp.Prod => (byte)(ba * bb),
                    ReductionOp.Min => (byte)Math.Min(ba, bb),
                    ReductionOp.Max => (byte)Math.Max(ba, bb),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(short))
            {
                int sa = (short)(object)a;
                int sb = (short)(object)b;
                short result = op switch
                {
                    ReductionOp.Sum => (short)(sa + sb),
                    ReductionOp.Prod => (short)(sa * sb),
                    ReductionOp.Min => (short)Math.Min(sa, sb),
                    ReductionOp.Max => (short)Math.Max(sa, sb),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(ushort))
            {
                int usa = (ushort)(object)a;
                int usb = (ushort)(object)b;
                ushort result = op switch
                {
                    ReductionOp.Sum => (ushort)(usa + usb),
                    ReductionOp.Prod => (ushort)(usa * usb),
                    ReductionOp.Min => (ushort)Math.Min(usa, usb),
                    ReductionOp.Max => (ushort)Math.Max(usa, usb),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(uint))
            {
                uint ua = (uint)(object)a;
                uint ub = (uint)(object)b;
                uint result = op switch
                {
                    ReductionOp.Sum => ua + ub,
                    ReductionOp.Prod => ua * ub,
                    ReductionOp.Min => Math.Min(ua, ub),
                    ReductionOp.Max => Math.Max(ua, ub),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(ulong))
            {
                ulong ula = (ulong)(object)a;
                ulong ulb = (ulong)(object)b;
                ulong result = op switch
                {
                    ReductionOp.Sum => ula + ulb,
                    ReductionOp.Prod => ula * ulb,
                    ReductionOp.Min => Math.Min(ula, ulb),
                    ReductionOp.Max => Math.Max(ula, ulb),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }

            throw new NotSupportedException($"Type {typeof(T)} not supported");
        }

        #endregion
    }
}
