using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
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
            return (void* input, void* output, long* inputStrides, long* inputShape,
                    long* outputStrides, int axis, long axisSize, int ndim, long outputSize) =>
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
            long* inputStrides, long* inputShape, long* outputStrides,
            int axis, long axisSize, int ndim, long outputSize,
            ReductionOp op)
            where T : unmanaged
        {
            long axisStride = inputStrides[axis];

            // ─────────────────────────────────────────────────────────────────
            // FAST PATH — leading-axis on C-contiguous input.
            //
            // For C-contig input with axis < ndim-1, the inner slab
            // (dims axis+1..ndim-1) is contiguous, and the axis stride equals
            // innerSize. We walk axis rows sequentially and SIMD-elementwise-
            // reduce each row into the output slab — output stays hot in cache,
            // input streams sequentially. This is the NumPy reduction pattern
            // and turns axis=0 from O(out × stridedGather) into O(in_bytes).
            // For axis = ndim-1 (innermost) the existing per-output contig
            // SIMD reduce is optimal; we don't take this path.
            // ─────────────────────────────────────────────────────────────────
            if (axis < ndim - 1 && IsCContig(inputStrides, inputShape, ndim))
            {
                long innerSize = 1;
                for (int d = axis + 1; d < ndim; d++) innerSize *= inputShape[d];
                long outerSize = 1;
                for (int d = 0; d < axis; d++) outerSize *= inputShape[d];

                ReductionOp innerOp = op == ReductionOp.Mean ? ReductionOp.Sum : op;
                DispatchLeading<T>(input, output, outerSize, axisSize, innerSize, innerOp);

                if (op == ReductionOp.Mean)
                    DivideArrayByCount<T>(output, outerSize * innerSize, axisSize);
                return;
            }

            // ─────────────────────────────────────────────────────────────────
            // FAST PATH — innermost-axis on C-contiguous input.
            //
            // axis == ndim-1 and the array is C-contig: each output reduces a
            // single contiguous run of axisSize elements. Walk outputs sequentially
            // and call the typed per-row reducer (struct-generic op tag → JIT
            // inlines the SIMD intrinsic with no per-output runtime switch).
            // ─────────────────────────────────────────────────────────────────
            if (axis == ndim - 1 && IsCContig(inputStrides, inputShape, ndim))
            {
                ReductionOp innerOp = op == ReductionOp.Mean ? ReductionOp.Sum : op;
                DispatchInnermost<T>(input, output, outputSize, axisSize, innerOp);
                if (op == ReductionOp.Mean)
                    DivideArrayByCount<T>(output, outputSize, axisSize);
                return;
            }

            // ─────────────────────────────────────────────────────────────────
            // FAST PATH — axis=0 with INNER SLAB C-contig (covers sliced inputs
            // like a[::2,:], a[::-1,:], a[100:900, 100:900]). The slab traversal
            // is identical to the C-contig leading-axis case, but axis-row spacing
            // uses the natural axis stride (could be != innerSize for sliced inputs,
            // could be negative for reversed views). Output is shape (inner...,)
            // which is freshly allocated C-contig — matches the slab layout.
            // ─────────────────────────────────────────────────────────────────
            if (axis == 0 && ndim >= 2 && IsInnerSlabCContig(inputStrides, inputShape, 0, ndim))
            {
                long innerSize = 1;
                for (int d = 1; d < ndim; d++) innerSize *= inputShape[d];
                long axisStrideEl = inputStrides[0];
                ReductionOp innerOp = op == ReductionOp.Mean ? ReductionOp.Sum : op;
                DispatchLeadingStrided<T>(input, output, axisSize, innerSize, axisStrideEl, innerOp);
                if (op == ReductionOp.Mean)
                    DivideArrayByCount<T>(output, innerSize, axisSize);
                return;
            }

            // ─────────────────────────────────────────────────────────────────
            // FAST PATH — F-contig leading-axis (axis == ndim-1 on F-contig).
            //
            // For F-contig input, axis=ndim-1 has the LARGEST stride (analogous
            // to axis=0 on C-contig). The slab below it (axes 0..ndim-2) is
            // contiguous (it's just F-contig of size prod(shape[0..ndim-2])).
            // Same memory access pattern as the C-contig leading-axis case:
            // walk the axis row-by-row, SIMD-fold each into the output buffer.
            //
            // Output is 1D (or higher with F-contig layout — the output buffer
            // happens to be C-contig but for a 1D output that's identical).
            // For higher-rank F-contig with non-innermost axis the slab/output
            // layouts mismatch, so we restrict to the common case.
            // ─────────────────────────────────────────────────────────────────
            if (axis == ndim - 1 && ndim == 2 && IsFContig(inputStrides, inputShape, ndim))
            {
                long innerSize = inputShape[0]; // the contig slab (axis 0)
                long outerSize = 1;             // no outer dims
                ReductionOp innerOp = op == ReductionOp.Mean ? ReductionOp.Sum : op;
                DispatchLeading<T>(input, output, outerSize, axisSize, innerSize, innerOp);
                if (op == ReductionOp.Mean)
                    DivideArrayByCount<T>(output, innerSize, axisSize);
                return;
            }

            // ─────────────────────────────────────────────────────────────────
            // FAST PATH — F-contig innermost-axis (axis == 0 on F-contig).
            //
            // For F-contig axis=0, each output reduces a contiguous run of
            // axisSize elements (stride 1). Same pattern as C-contig innermost
            // — route through the same typed kernel. Output is 1D (for 2D
            // input) or F-contig higher-rank; for 2D input the output is 1D
            // so layout doesn't matter.
            // ─────────────────────────────────────────────────────────────────
            if (axis == 0 && ndim == 2 && IsFContig(inputStrides, inputShape, ndim))
            {
                long inner = inputShape[1]; // number of contig rows along axis=1
                ReductionOp innerOp = op == ReductionOp.Mean ? ReductionOp.Sum : op;
                DispatchInnermost<T>(input, output, inner, axisSize, innerOp);
                if (op == ReductionOp.Mean)
                    DivideArrayByCount<T>(output, inner, axisSize);
                return;
            }

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
            for (long outIdx = 0; outIdx < outputSize; outIdx++)
            {
                // Convert linear output index to coordinates and compute input base offset
                long remaining = outIdx;
                long inputBaseOffset = 0;
                long outputOffset = 0;

                for (int d = 0; d < outputNdim; d++)
                {
                    // Map output dimension d to input dimension
                    int inputDim = d >= axis ? d + 1 : d;

                    long coord = remaining / outputDimStridesArray[d];
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
                int result = (int)((int)(object)value / count);
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
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
            // AVX2 gather requires int32 indices, so stride must fit in int32
            if (Avx2.IsSupported && size >= 8 && stride <= int.MaxValue)
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
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
            // B5: Add SByte identity values for axis reductions.
            if (typeof(T) == typeof(sbyte))
            {
                sbyte val = op switch
                {
                    ReductionOp.Sum => (sbyte)0,
                    ReductionOp.Prod => (sbyte)1,
                    ReductionOp.Min => sbyte.MaxValue,
                    ReductionOp.Max => sbyte.MinValue,
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<T> CreateIdentityVector256<T>(ReductionOp op) where T : unmanaged
        {
            T identity = GetIdentityValue<T>(op);
            return Vector256.Create(identity);
        }

        /// <summary>
        /// Create identity Vector128 for reduction operation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<T> CreateIdentityVector128<T>(ReductionOp op) where T : unmanaged
        {
            T identity = GetIdentityValue<T>(op);
            return Vector128.Create(identity);
        }

        /// <summary>
        /// Combine two Vector256 values using reduction operation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
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
            // B5: SByte axis reduction support (pair-combine).
            if (typeof(T) == typeof(sbyte))
            {
                int sba = (sbyte)(object)a;
                int sbb = (sbyte)(object)b;
                sbyte result = op switch
                {
                    ReductionOp.Sum => (sbyte)(sba + sbb),
                    ReductionOp.Prod => (sbyte)(sba * sbb),
                    ReductionOp.Min => (sbyte)Math.Min(sba, sbb),
                    ReductionOp.Max => (sbyte)Math.Max(sba, sbb),
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

        #region Leading-axis fast path (axis < ndim-1 on C-contiguous input)

        // Detect C-contiguous from strides+shape: stride[ndim-1] == 1 and
        // stride[i] == stride[i+1] * shape[i+1] for all i in [0, ndim-2].
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool IsCContig(long* strides, long* shape, int ndim)
        {
            if (ndim == 0) return true;
            if (strides[ndim - 1] != 1) return false;
            for (int d = ndim - 2; d >= 0; d--)
                if (strides[d] != strides[d + 1] * shape[d + 1]) return false;
            return true;
        }

        // Detect F-contiguous: stride[0] == 1 and stride[i] == stride[i-1] * shape[i-1]
        // for all i in [1, ndim-1]. Common case: `.T` on a C-contig 2D array.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool IsFContig(long* strides, long* shape, int ndim)
        {
            if (ndim == 0) return true;
            if (strides[0] != 1) return false;
            for (int d = 1; d < ndim; d++)
                if (strides[d] != strides[d - 1] * shape[d - 1]) return false;
            return true;
        }

        // Detect whether the INNER slab (dims after axis) is C-contiguous as a flat
        // run of memory. This is looser than full C-contig — it does NOT require the
        // outer strides to match the C-contig formula, so it covers sliced inputs
        // like a[::2,:], a[::-1,:], or a[100:900, 100:900], whose inner dim is still
        // stride-1 but whose outer stride differs from a fresh C-contig array.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool IsInnerSlabCContig(long* strides, long* shape, int axis, int ndim)
        {
            if (axis >= ndim - 1) return false;
            if (strides[ndim - 1] != 1) return false;
            for (int d = ndim - 2; d > axis; d--)
                if (strides[d] != strides[d + 1] * shape[d + 1]) return false;
            return true;
        }

        // Per-op dispatch into the typed kernel. The runtime branch on `op` happens
        // ONCE here (not in the hot loop), and routes to a kernel with the op
        // hard-coded via struct generic — JIT inlines the SIMD intrinsic.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void DispatchLeading<T>(
            T* input, T* output, long outerSize, long axisSize, long innerSize, ReductionOp op)
            where T : unmanaged
        {
            switch (op)
            {
                case ReductionOp.Sum:  AxisReductionLeadingTyped<T, AddOp<T>>(input, output, outerSize, axisSize, innerSize); return;
                case ReductionOp.Prod: AxisReductionLeadingTyped<T, MulOp<T>>(input, output, outerSize, axisSize, innerSize); return;
                case ReductionOp.Min:  AxisReductionLeadingTyped<T, MinOp<T>>(input, output, outerSize, axisSize, innerSize); return;
                case ReductionOp.Max:  AxisReductionLeadingTyped<T, MaxOp<T>>(input, output, outerSize, axisSize, innerSize); return;
                default: throw new NotSupportedException($"DispatchLeading: {op}");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void DispatchInnermost<T>(
            T* input, T* output, long outputSize, long axisSize, ReductionOp op)
            where T : unmanaged
        {
            switch (op)
            {
                case ReductionOp.Sum:  AxisReductionInnermostTyped<T, AddOp<T>>(input, output, outputSize, axisSize); return;
                case ReductionOp.Prod: AxisReductionInnermostTyped<T, MulOp<T>>(input, output, outputSize, axisSize); return;
                case ReductionOp.Min:  AxisReductionInnermostTyped<T, MinOp<T>>(input, output, outputSize, axisSize); return;
                case ReductionOp.Max:  AxisReductionInnermostTyped<T, MaxOp<T>>(input, output, outputSize, axisSize); return;
                default: throw new NotSupportedException($"DispatchInnermost: {op}");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void DispatchLeadingStrided<T>(
            T* input, T* output, long axisSize, long innerSize, long axisStride, ReductionOp op)
            where T : unmanaged
        {
            switch (op)
            {
                case ReductionOp.Sum:  AxisReductionLeadingStridedTyped<T, AddOp<T>>(input, output, axisSize, innerSize, axisStride); return;
                case ReductionOp.Prod: AxisReductionLeadingStridedTyped<T, MulOp<T>>(input, output, axisSize, innerSize, axisStride); return;
                case ReductionOp.Min:  AxisReductionLeadingStridedTyped<T, MinOp<T>>(input, output, axisSize, innerSize, axisStride); return;
                case ReductionOp.Max:  AxisReductionLeadingStridedTyped<T, MaxOp<T>>(input, output, axisSize, innerSize, axisStride); return;
                default: throw new NotSupportedException($"DispatchLeadingStrided: {op}");
            }
        }

        // axis=0 leading-axis where the inner slab is C-contig but the axis
        // stride may differ from innerSize (sliced/reversed inputs). The slab
        // (output buffer) is C-contig of innerSize elements; we copy the first
        // axis row in as the accumulator seed, then SIMD-fold each subsequent
        // axis row using the per-row pointer `input + a*axisStride`.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void AxisReductionLeadingStridedTyped<T, TOp>(
            T* input, T* output, long axisSize, long innerSize, long axisStride)
            where T : unmanaged
            where TOp : struct, ITypedReductionOp<T>
        {
            long bytesPerSlab = innerSize * sizeof(T);
            TOp opAgent = default;

            // Seed: first axis row (a=0) copied to output. Axis row starts at input + 0*axisStride.
            Buffer.MemoryCopy(input, output, bytesPerSlab, bytesPerSlab);

            for (long a = 1; a < axisSize; a++)
            {
                T* row = input + a * axisStride;
                long i = 0;
                if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && innerSize >= Vector256<T>.Count)
                {
                    int vc = Vector256<T>.Count;
                    long unrollEnd = innerSize - vc * 4;
                    for (; i <= unrollEnd; i += vc * 4)
                    {
                        Vector256.Store(opAgent.Combine256(Vector256.Load(output + i),         Vector256.Load(row + i)),         output + i);
                        Vector256.Store(opAgent.Combine256(Vector256.Load(output + i + vc),    Vector256.Load(row + i + vc)),    output + i + vc);
                        Vector256.Store(opAgent.Combine256(Vector256.Load(output + i + vc*2),  Vector256.Load(row + i + vc*2)),  output + i + vc*2);
                        Vector256.Store(opAgent.Combine256(Vector256.Load(output + i + vc*3),  Vector256.Load(row + i + vc*3)),  output + i + vc*3);
                    }
                    for (; i + vc <= innerSize; i += vc)
                        Vector256.Store(opAgent.Combine256(Vector256.Load(output + i), Vector256.Load(row + i)), output + i);
                }
                else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && innerSize >= Vector128<T>.Count)
                {
                    int vc = Vector128<T>.Count;
                    long unrollEnd = innerSize - vc * 4;
                    for (; i <= unrollEnd; i += vc * 4)
                    {
                        Vector128.Store(opAgent.Combine128(Vector128.Load(output + i),         Vector128.Load(row + i)),         output + i);
                        Vector128.Store(opAgent.Combine128(Vector128.Load(output + i + vc),    Vector128.Load(row + i + vc)),    output + i + vc);
                        Vector128.Store(opAgent.Combine128(Vector128.Load(output + i + vc*2),  Vector128.Load(row + i + vc*2)),  output + i + vc*2);
                        Vector128.Store(opAgent.Combine128(Vector128.Load(output + i + vc*3),  Vector128.Load(row + i + vc*3)),  output + i + vc*3);
                    }
                    for (; i + vc <= innerSize; i += vc)
                        Vector128.Store(opAgent.Combine128(Vector128.Load(output + i), Vector128.Load(row + i)), output + i);
                }
                for (; i < innerSize; i++)
                    output[i] = opAgent.CombineScalar(output[i], row[i]);
            }
        }

        // Innermost-axis reduction (axis == ndim-1, C-contiguous). Each output
        // reduces a contiguous run of axisSize elements. Per output: 4× unrolled
        // SIMD with the op-struct (JIT inlines Vector.Add/Min/Max/Multiply) and
        // tree-reduces to a scalar. The identity vectors are hoisted out of the
        // per-output loop; the op-tag struct ensures no runtime switch.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void AxisReductionInnermostTyped<T, TOp>(
            T* input, T* output, long outputSize, long axisSize)
            where T : unmanaged
            where TOp : struct, ITypedReductionOp<T>
        {
            TOp opAgent = default;
            bool useV256 = Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && axisSize >= Vector256<T>.Count;
            bool useV128 = !useV256 && Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && axisSize >= Vector128<T>.Count;

            // Hoist: identity vector / scalar identity created once.
            Vector256<T> identV256 = default;
            Vector128<T> identV128 = default;
            T identity = opAgent.Identity();
            if (useV256) identV256 = Vector256.Create(identity);
            else if (useV128) identV128 = Vector128.Create(identity);

            if (useV256)
            {
                int vc = Vector256<T>.Count;
                long unrollStep = vc * 4;
                long unrollEnd = axisSize - unrollStep;
                long vectorEnd = axisSize - vc;

                for (long o = 0; o < outputSize; o++)
                {
                    T* row = input + o * axisSize;
                    long i = 0;
                    var acc0 = identV256;
                    var acc1 = identV256;
                    var acc2 = identV256;
                    var acc3 = identV256;

                    for (; i <= unrollEnd; i += unrollStep)
                    {
                        acc0 = opAgent.Combine256(acc0, Vector256.Load(row + i));
                        acc1 = opAgent.Combine256(acc1, Vector256.Load(row + i + vc));
                        acc2 = opAgent.Combine256(acc2, Vector256.Load(row + i + vc * 2));
                        acc3 = opAgent.Combine256(acc3, Vector256.Load(row + i + vc * 3));
                    }
                    var acc = opAgent.Combine256(opAgent.Combine256(acc0, acc1), opAgent.Combine256(acc2, acc3));
                    for (; i <= vectorEnd; i += vc)
                        acc = opAgent.Combine256(acc, Vector256.Load(row + i));

                    var acc128 = opAgent.Combine128(acc.GetLower(), acc.GetUpper());
                    T scalarAcc = HorizontalReduceTyped<T, TOp>(acc128);
                    for (; i < axisSize; i++)
                        scalarAcc = opAgent.CombineScalar(scalarAcc, row[i]);
                    output[o] = scalarAcc;
                }
            }
            else if (useV128)
            {
                int vc = Vector128<T>.Count;
                long unrollStep = vc * 4;
                long unrollEnd = axisSize - unrollStep;
                long vectorEnd = axisSize - vc;

                for (long o = 0; o < outputSize; o++)
                {
                    T* row = input + o * axisSize;
                    long i = 0;
                    var acc0 = identV128;
                    var acc1 = identV128;
                    var acc2 = identV128;
                    var acc3 = identV128;

                    for (; i <= unrollEnd; i += unrollStep)
                    {
                        acc0 = opAgent.Combine128(acc0, Vector128.Load(row + i));
                        acc1 = opAgent.Combine128(acc1, Vector128.Load(row + i + vc));
                        acc2 = opAgent.Combine128(acc2, Vector128.Load(row + i + vc * 2));
                        acc3 = opAgent.Combine128(acc3, Vector128.Load(row + i + vc * 3));
                    }
                    var acc = opAgent.Combine128(opAgent.Combine128(acc0, acc1), opAgent.Combine128(acc2, acc3));
                    for (; i <= vectorEnd; i += vc)
                        acc = opAgent.Combine128(acc, Vector128.Load(row + i));

                    T scalarAcc = HorizontalReduceTyped<T, TOp>(acc);
                    for (; i < axisSize; i++)
                        scalarAcc = opAgent.CombineScalar(scalarAcc, row[i]);
                    output[o] = scalarAcc;
                }
            }
            else
            {
                // Fully scalar
                for (long o = 0; o < outputSize; o++)
                {
                    T* row = input + o * axisSize;
                    T scalarAcc = identity;
                    for (long i = 0; i < axisSize; i++)
                        scalarAcc = opAgent.CombineScalar(scalarAcc, row[i]);
                    output[o] = scalarAcc;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T HorizontalReduceTyped<T, TOp>(Vector128<T> v)
            where T : unmanaged
            where TOp : struct, ITypedReductionOp<T>
        {
            TOp op = default;
            int count = Vector128<T>.Count;
            T r = v.GetElement(0);
            for (int j = 1; j < count; j++) r = op.CombineScalar(r, v.GetElement(j));
            return r;
        }

        // Leading-axis reduction (struct-generic op tag → JIT specializes per op).
        // For each outer slab: copy the first axis row to the output slab as the
        // accumulator seed, then SIMD-elementwise-reduce every remaining axis row
        // into it. Output stays L1-hot; input streams sequentially. The op is
        // passed via a zero-sized struct that implements ITypedReductionOp<T>,
        // so the JIT inlines the actual SIMD instruction (Vector256.Max etc.)
        // with no runtime switch in the hot loop.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void AxisReductionLeadingTyped<T, TOp>(
            T* input, T* output,
            long outerSize, long axisSize, long innerSize)
            where T : unmanaged
            where TOp : struct, ITypedReductionOp<T>
        {
            long inputSlab = axisSize * innerSize;
            long bytesPerSlab = innerSize * sizeof(T);
            TOp opAgent = default;

            for (long o = 0; o < outerSize; o++)
            {
                T* outSlab = output + o * innerSize;
                T* inBase = input + o * inputSlab;

                Buffer.MemoryCopy(inBase, outSlab, bytesPerSlab, bytesPerSlab);

                for (long a = 1; a < axisSize; a++)
                {
                    T* row = inBase + a * innerSize;
                    long i = 0;
                    if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && innerSize >= Vector256<T>.Count)
                    {
                        int vc = Vector256<T>.Count;
                        long unrollEnd = innerSize - vc * 4;
                        for (; i <= unrollEnd; i += vc * 4)
                        {
                            Vector256.Store(opAgent.Combine256(Vector256.Load(outSlab + i),         Vector256.Load(row + i)),         outSlab + i);
                            Vector256.Store(opAgent.Combine256(Vector256.Load(outSlab + i + vc),    Vector256.Load(row + i + vc)),    outSlab + i + vc);
                            Vector256.Store(opAgent.Combine256(Vector256.Load(outSlab + i + vc*2),  Vector256.Load(row + i + vc*2)),  outSlab + i + vc*2);
                            Vector256.Store(opAgent.Combine256(Vector256.Load(outSlab + i + vc*3),  Vector256.Load(row + i + vc*3)),  outSlab + i + vc*3);
                        }
                        for (; i + vc <= innerSize; i += vc)
                            Vector256.Store(opAgent.Combine256(Vector256.Load(outSlab + i), Vector256.Load(row + i)), outSlab + i);
                    }
                    else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && innerSize >= Vector128<T>.Count)
                    {
                        int vc = Vector128<T>.Count;
                        long unrollEnd = innerSize - vc * 4;
                        for (; i <= unrollEnd; i += vc * 4)
                        {
                            Vector128.Store(opAgent.Combine128(Vector128.Load(outSlab + i),         Vector128.Load(row + i)),         outSlab + i);
                            Vector128.Store(opAgent.Combine128(Vector128.Load(outSlab + i + vc),    Vector128.Load(row + i + vc)),    outSlab + i + vc);
                            Vector128.Store(opAgent.Combine128(Vector128.Load(outSlab + i + vc*2),  Vector128.Load(row + i + vc*2)),  outSlab + i + vc*2);
                            Vector128.Store(opAgent.Combine128(Vector128.Load(outSlab + i + vc*3),  Vector128.Load(row + i + vc*3)),  outSlab + i + vc*3);
                        }
                        for (; i + vc <= innerSize; i += vc)
                            Vector128.Store(opAgent.Combine128(Vector128.Load(outSlab + i), Vector128.Load(row + i)), outSlab + i);
                    }
                    for (; i < innerSize; i++)
                        outSlab[i] = opAgent.CombineScalar(outSlab[i], row[i]);
                }
            }
        }

        // Op-tag interface. The JIT specializes the generic method per implementing
        // struct, so opAgent.Combine256(a, b) compiles to a single SIMD instruction
        // (no switch, no virtual call).
        internal interface ITypedReductionOp<T> where T : unmanaged
        {
            Vector256<T> Combine256(Vector256<T> a, Vector256<T> b);
            Vector128<T> Combine128(Vector128<T> a, Vector128<T> b);
            T CombineScalar(T a, T b);
            T Identity();
        }

        // Per-T identity helpers (one/min/max) — JIT-folded per specialization.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T OneOf<T>() where T : unmanaged
        {
            if (typeof(T) == typeof(float))  return (T)(object)1f;
            if (typeof(T) == typeof(double)) return (T)(object)1.0;
            if (typeof(T) == typeof(int))    return (T)(object)1;
            if (typeof(T) == typeof(long))   return (T)(object)1L;
            if (typeof(T) == typeof(byte))   return (T)(object)(byte)1;
            if (typeof(T) == typeof(sbyte))  return (T)(object)(sbyte)1;
            if (typeof(T) == typeof(short))  return (T)(object)(short)1;
            if (typeof(T) == typeof(ushort)) return (T)(object)(ushort)1;
            if (typeof(T) == typeof(uint))   return (T)(object)1u;
            if (typeof(T) == typeof(ulong))  return (T)(object)1UL;
            throw new NotSupportedException();
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T MaxValueOf<T>() where T : unmanaged
        {
            if (typeof(T) == typeof(float))  return (T)(object)float.PositiveInfinity;
            if (typeof(T) == typeof(double)) return (T)(object)double.PositiveInfinity;
            if (typeof(T) == typeof(int))    return (T)(object)int.MaxValue;
            if (typeof(T) == typeof(long))   return (T)(object)long.MaxValue;
            if (typeof(T) == typeof(byte))   return (T)(object)byte.MaxValue;
            if (typeof(T) == typeof(sbyte))  return (T)(object)sbyte.MaxValue;
            if (typeof(T) == typeof(short))  return (T)(object)short.MaxValue;
            if (typeof(T) == typeof(ushort)) return (T)(object)ushort.MaxValue;
            if (typeof(T) == typeof(uint))   return (T)(object)uint.MaxValue;
            if (typeof(T) == typeof(ulong))  return (T)(object)ulong.MaxValue;
            throw new NotSupportedException();
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T MinValueOf<T>() where T : unmanaged
        {
            if (typeof(T) == typeof(float))  return (T)(object)float.NegativeInfinity;
            if (typeof(T) == typeof(double)) return (T)(object)double.NegativeInfinity;
            if (typeof(T) == typeof(int))    return (T)(object)int.MinValue;
            if (typeof(T) == typeof(long))   return (T)(object)long.MinValue;
            if (typeof(T) == typeof(byte))   return (T)(object)byte.MinValue;
            if (typeof(T) == typeof(sbyte))  return (T)(object)sbyte.MinValue;
            if (typeof(T) == typeof(short))  return (T)(object)short.MinValue;
            if (typeof(T) == typeof(ushort)) return (T)(object)ushort.MinValue;
            if (typeof(T) == typeof(uint))   return (T)(object)uint.MinValue;
            if (typeof(T) == typeof(ulong))  return (T)(object)ulong.MinValue;
            throw new NotSupportedException();
        }

        internal readonly struct AddOp<T> : ITypedReductionOp<T> where T : unmanaged
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)] public Vector256<T> Combine256(Vector256<T> a, Vector256<T> b) => Vector256.Add(a, b);
            [MethodImpl(MethodImplOptions.AggressiveInlining)] public Vector128<T> Combine128(Vector128<T> a, Vector128<T> b) => Vector128.Add(a, b);
            [MethodImpl(MethodImplOptions.AggressiveInlining)] public T CombineScalar(T a, T b) => AddScalar(a, b);
            [MethodImpl(MethodImplOptions.AggressiveInlining)] public T Identity() => default;
        }
        internal readonly struct MulOp<T> : ITypedReductionOp<T> where T : unmanaged
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)] public Vector256<T> Combine256(Vector256<T> a, Vector256<T> b) => Vector256.Multiply(a, b);
            [MethodImpl(MethodImplOptions.AggressiveInlining)] public Vector128<T> Combine128(Vector128<T> a, Vector128<T> b) => Vector128.Multiply(a, b);
            [MethodImpl(MethodImplOptions.AggressiveInlining)] public T CombineScalar(T a, T b) => MulScalar(a, b);
            [MethodImpl(MethodImplOptions.AggressiveInlining)] public T Identity() => OneOf<T>();
        }
        internal readonly struct MinOp<T> : ITypedReductionOp<T> where T : unmanaged
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)] public Vector256<T> Combine256(Vector256<T> a, Vector256<T> b) => Vector256.Min(a, b);
            [MethodImpl(MethodImplOptions.AggressiveInlining)] public Vector128<T> Combine128(Vector128<T> a, Vector128<T> b) => Vector128.Min(a, b);
            [MethodImpl(MethodImplOptions.AggressiveInlining)] public T CombineScalar(T a, T b) => MinScalar(a, b);
            [MethodImpl(MethodImplOptions.AggressiveInlining)] public T Identity() => MaxValueOf<T>();
        }
        internal readonly struct MaxOp<T> : ITypedReductionOp<T> where T : unmanaged
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)] public Vector256<T> Combine256(Vector256<T> a, Vector256<T> b) => Vector256.Max(a, b);
            [MethodImpl(MethodImplOptions.AggressiveInlining)] public Vector128<T> Combine128(Vector128<T> a, Vector128<T> b) => Vector128.Max(a, b);
            [MethodImpl(MethodImplOptions.AggressiveInlining)] public T CombineScalar(T a, T b) => MaxScalar(a, b);
            [MethodImpl(MethodImplOptions.AggressiveInlining)] public T Identity() => MinValueOf<T>();
        }

        // Per-T scalar helpers. The typeof(T)==typeof(X) chain JIT-folds to a single
        // branch per specialization.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T AddScalar<T>(T a, T b) where T : unmanaged
        {
            if (typeof(T) == typeof(float))  { float  x = (float) (object)a, y = (float) (object)b; return (T)(object)(x + y); }
            if (typeof(T) == typeof(double)) { double x = (double)(object)a, y = (double)(object)b; return (T)(object)(x + y); }
            if (typeof(T) == typeof(int))    { int    x = (int)   (object)a, y = (int)   (object)b; return (T)(object)(x + y); }
            if (typeof(T) == typeof(long))   { long   x = (long)  (object)a, y = (long)  (object)b; return (T)(object)(x + y); }
            if (typeof(T) == typeof(byte))   { byte   x = (byte)  (object)a, y = (byte)  (object)b; return (T)(object)((byte)(x + y)); }
            if (typeof(T) == typeof(sbyte))  { sbyte  x = (sbyte) (object)a, y = (sbyte) (object)b; return (T)(object)((sbyte)(x + y)); }
            if (typeof(T) == typeof(short))  { short  x = (short) (object)a, y = (short) (object)b; return (T)(object)((short)(x + y)); }
            if (typeof(T) == typeof(ushort)) { ushort x = (ushort)(object)a, y = (ushort)(object)b; return (T)(object)((ushort)(x + y)); }
            if (typeof(T) == typeof(uint))   { uint   x = (uint)  (object)a, y = (uint)  (object)b; return (T)(object)(x + y); }
            if (typeof(T) == typeof(ulong))  { ulong  x = (ulong) (object)a, y = (ulong) (object)b; return (T)(object)(x + y); }
            throw new NotSupportedException();
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T MulScalar<T>(T a, T b) where T : unmanaged
        {
            if (typeof(T) == typeof(float))  { float  x = (float) (object)a, y = (float) (object)b; return (T)(object)(x * y); }
            if (typeof(T) == typeof(double)) { double x = (double)(object)a, y = (double)(object)b; return (T)(object)(x * y); }
            if (typeof(T) == typeof(int))    { int    x = (int)   (object)a, y = (int)   (object)b; return (T)(object)(x * y); }
            if (typeof(T) == typeof(long))   { long   x = (long)  (object)a, y = (long)  (object)b; return (T)(object)(x * y); }
            if (typeof(T) == typeof(byte))   { byte   x = (byte)  (object)a, y = (byte)  (object)b; return (T)(object)((byte)(x * y)); }
            if (typeof(T) == typeof(sbyte))  { sbyte  x = (sbyte) (object)a, y = (sbyte) (object)b; return (T)(object)((sbyte)(x * y)); }
            if (typeof(T) == typeof(short))  { short  x = (short) (object)a, y = (short) (object)b; return (T)(object)((short)(x * y)); }
            if (typeof(T) == typeof(ushort)) { ushort x = (ushort)(object)a, y = (ushort)(object)b; return (T)(object)((ushort)(x * y)); }
            if (typeof(T) == typeof(uint))   { uint   x = (uint)  (object)a, y = (uint)  (object)b; return (T)(object)(x * y); }
            if (typeof(T) == typeof(ulong))  { ulong  x = (ulong) (object)a, y = (ulong) (object)b; return (T)(object)(x * y); }
            throw new NotSupportedException();
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T MinScalar<T>(T a, T b) where T : unmanaged
        {
            if (typeof(T) == typeof(float))  { float  x = (float) (object)a, y = (float) (object)b; return (T)(object)Math.Min(x, y); }
            if (typeof(T) == typeof(double)) { double x = (double)(object)a, y = (double)(object)b; return (T)(object)Math.Min(x, y); }
            if (typeof(T) == typeof(int))    { int    x = (int)   (object)a, y = (int)   (object)b; return (T)(object)Math.Min(x, y); }
            if (typeof(T) == typeof(long))   { long   x = (long)  (object)a, y = (long)  (object)b; return (T)(object)Math.Min(x, y); }
            if (typeof(T) == typeof(byte))   { byte   x = (byte)  (object)a, y = (byte)  (object)b; return (T)(object)Math.Min(x, y); }
            if (typeof(T) == typeof(sbyte))  { sbyte  x = (sbyte) (object)a, y = (sbyte) (object)b; return (T)(object)Math.Min(x, y); }
            if (typeof(T) == typeof(short))  { short  x = (short) (object)a, y = (short) (object)b; return (T)(object)Math.Min(x, y); }
            if (typeof(T) == typeof(ushort)) { ushort x = (ushort)(object)a, y = (ushort)(object)b; return (T)(object)Math.Min(x, y); }
            if (typeof(T) == typeof(uint))   { uint   x = (uint)  (object)a, y = (uint)  (object)b; return (T)(object)Math.Min(x, y); }
            if (typeof(T) == typeof(ulong))  { ulong  x = (ulong) (object)a, y = (ulong) (object)b; return (T)(object)Math.Min(x, y); }
            throw new NotSupportedException();
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T MaxScalar<T>(T a, T b) where T : unmanaged
        {
            if (typeof(T) == typeof(float))  { float  x = (float) (object)a, y = (float) (object)b; return (T)(object)Math.Max(x, y); }
            if (typeof(T) == typeof(double)) { double x = (double)(object)a, y = (double)(object)b; return (T)(object)Math.Max(x, y); }
            if (typeof(T) == typeof(int))    { int    x = (int)   (object)a, y = (int)   (object)b; return (T)(object)Math.Max(x, y); }
            if (typeof(T) == typeof(long))   { long   x = (long)  (object)a, y = (long)  (object)b; return (T)(object)Math.Max(x, y); }
            if (typeof(T) == typeof(byte))   { byte   x = (byte)  (object)a, y = (byte)  (object)b; return (T)(object)Math.Max(x, y); }
            if (typeof(T) == typeof(sbyte))  { sbyte  x = (sbyte) (object)a, y = (sbyte) (object)b; return (T)(object)Math.Max(x, y); }
            if (typeof(T) == typeof(short))  { short  x = (short) (object)a, y = (short) (object)b; return (T)(object)Math.Max(x, y); }
            if (typeof(T) == typeof(ushort)) { ushort x = (ushort)(object)a, y = (ushort)(object)b; return (T)(object)Math.Max(x, y); }
            if (typeof(T) == typeof(uint))   { uint   x = (uint)  (object)a, y = (uint)  (object)b; return (T)(object)Math.Max(x, y); }
            if (typeof(T) == typeof(ulong))  { ulong  x = (ulong) (object)a, y = (ulong) (object)b; return (T)(object)Math.Max(x, y); }
            throw new NotSupportedException();
        }

        // Divide an array in place by an integer count. Used by Mean axis
        // reductions after the leading-axis Sum accumulation finishes.
        private static unsafe void DivideArrayByCount<T>(T* arr, long n, long count) where T : unmanaged
        {
            for (long i = 0; i < n; i++)
                arr[i] = DivideByCountTyped(arr[i], count);
        }

        #endregion
    }
}
