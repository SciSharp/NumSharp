using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// DirectILKernelGenerator.Reduction.Axis.Boolean.cs - Boolean Axis Reduction Kernels
// =============================================================================
//
// RESPONSIBILITY:
//   - All / Any reductions along a specific axis.
//   - Output is always Boolean regardless of input dtype.
//   - SIMD per-row for the inner axis (stride == 1).
//   - AVX2 gather for strided rows when input is float / double.
//   - Scalar with early-exit otherwise.
//
// BEHAVIOR (matching NumPy 2.4.2):
//   - All: identity = True. Element-wise non-zero check, AND-reduced.
//   - Any: identity = False. Element-wise non-zero check, OR-reduced.
//   - NaN is non-zero (NaN == 0 is false in IEEE 754) → counts as truthy.
//   - Empty axis (axisSize == 0) returns identity per output cell.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        #region Boolean Axis Reduction (All / Any)

        internal static readonly ConcurrentDictionary<AxisReductionKernelKey, AxisReductionKernel> _boolAxisReductionCache = new();

        /// <summary>
        ///     Try to get a boolean axis reduction kernel (All / Any).
        ///     Returns null for non-SIMD-capable dtypes (Half, Complex, Decimal, Char) so the
        ///     caller can fall back to the NpyAxisIter scalar path.
        /// </summary>
        public static AxisReductionKernel? TryGetBooleanAxisReductionKernel(AxisReductionKernelKey key)
        {
            if (!Enabled)
                return null;

            if (key.Op != ReductionOp.All && key.Op != ReductionOp.Any)
                return null;

            if (!IsBoolAxisSimdCapable(key.InputType))
                return null;

            return _boolAxisReductionCache.GetOrAdd(key, CreateBooleanAxisReductionKernel);
        }

        private static bool IsBoolAxisSimdCapable(NPTypeCode t)
            => t == NPTypeCode.Byte
            || t == NPTypeCode.SByte
            || t == NPTypeCode.Int16
            || t == NPTypeCode.UInt16
            || t == NPTypeCode.Int32
            || t == NPTypeCode.UInt32
            || t == NPTypeCode.Int64
            || t == NPTypeCode.UInt64
            || t == NPTypeCode.Single
            || t == NPTypeCode.Double
            || t == NPTypeCode.Boolean;

        private static AxisReductionKernel CreateBooleanAxisReductionKernel(AxisReductionKernelKey key)
        {
            bool isAll = key.Op == ReductionOp.All;
            return key.InputType switch
            {
                NPTypeCode.Byte    => CreateBoolAxisKernel<byte>(isAll),
                NPTypeCode.SByte   => CreateBoolAxisKernel<sbyte>(isAll),
                NPTypeCode.Int16   => CreateBoolAxisKernel<short>(isAll),
                NPTypeCode.UInt16  => CreateBoolAxisKernel<ushort>(isAll),
                NPTypeCode.Int32   => CreateBoolAxisKernel<int>(isAll),
                NPTypeCode.UInt32  => CreateBoolAxisKernel<uint>(isAll),
                NPTypeCode.Int64   => CreateBoolAxisKernel<long>(isAll),
                NPTypeCode.UInt64  => CreateBoolAxisKernel<ulong>(isAll),
                NPTypeCode.Single  => CreateBoolAxisKernel<float>(isAll),
                NPTypeCode.Double  => CreateBoolAxisKernel<double>(isAll),
                NPTypeCode.Boolean => CreateBoolAxisKernel<byte>(isAll), // bool ≡ byte at the SIMD level
                _ => throw new NotSupportedException($"Boolean axis reduction not supported for {key.InputType}")
            };
        }

        private static unsafe AxisReductionKernel CreateBoolAxisKernel<T>(bool isAll) where T : unmanaged
        {
            return (void* input, void* output, long* inputStrides, long* inputShape,
                    long* outputStrides, int axis, long axisSize, int ndim, long outputSize) =>
            {
                BoolAxisReductionHelper<T>(
                    (T*)input, (bool*)output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize, isAll);
            };
        }

        /// <summary>
        ///     Per-output-cell loop: walks the non-reduced (outer) dimensions, computing the
        ///     base offset of the reduction row, then delegates to the contig (stride==1) or
        ///     strided SIMD/scalar inner reducer.
        /// </summary>
        internal static unsafe void BoolAxisReductionHelper<T>(
            T* input, bool* output,
            long* inputStrides, long* inputShape, long* outputStrides,
            int axis, long axisSize, int ndim, long outputSize,
            bool isAll)
            where T : unmanaged
        {
            long axisStride = inputStrides[axis];
            bool axisContiguous = axisStride == 1;

            int outputNdim = ndim - 1;
            // outputDimStrides[d] = product of inputShape over output dims to the right of d.
            // Used to convert a linear output index into per-dim coordinates without div/mod
            // per element; mirrors the layout used by AxisReductionSimdHelper.
            Span<long> outputDimStrides = stackalloc long[outputNdim > 0 ? outputNdim : 1];
            if (outputNdim > 0)
            {
                outputDimStrides[outputNdim - 1] = 1;
                for (int d = outputNdim - 2; d >= 0; d--)
                {
                    int nextInputDim = (d + 1) >= axis ? d + 2 : d + 1;
                    outputDimStrides[d] = outputDimStrides[d + 1] * inputShape[nextInputDim];
                }
            }

            if (outputSize == 0)
                return;

            // Empty reduction axis: every output cell gets the identity value.
            if (axisSize == 0)
            {
                bool identity = isAll;
                for (long o = 0; o < outputSize; o++)
                {
                    long outOff = ComputeOutputOffset(o, outputDimStrides, outputStrides, outputNdim);
                    output[outOff] = identity;
                }
                return;
            }

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

                T* axisStart = input + inputBaseOffset;

                bool result = axisContiguous
                    ? (isAll ? AllSimdHelper<T>(axisStart, axisSize)
                             : AnySimdHelper<T>(axisStart, axisSize))
                    : ReduceStridedAxisBool<T>(axisStart, axisSize, axisStride, isAll);

                output[outputOffset] = result;
            }
        }

        private static unsafe long ComputeOutputOffset(
            long outIdx,
            Span<long> outputDimStrides,
            long* outputStrides,
            int outputNdim)
        {
            long offset = 0;
            long remaining = outIdx;
            for (int d = 0; d < outputNdim; d++)
            {
                long coord = remaining / outputDimStrides[d];
                remaining = remaining % outputDimStrides[d];
                offset += coord * outputStrides[d];
            }
            return offset;
        }

        /// <summary>
        ///     Strided boolean reduction. AVX2 gather covers float/double when stride fits
        ///     in int32; integer types use AVX2 32-bit / 64-bit gather; everything else uses
        ///     a scalar early-exit loop. Same lane semantics as <see cref="AllSimdHelper{T}"/>
        ///     / <see cref="AnySimdHelper{T}"/>: a lane comparing equal to zero contributes a
        ///     kill bit for All, an empty bit for Any.
        /// </summary>
        private static unsafe bool ReduceStridedAxisBool<T>(T* data, long size, long stride, bool isAll)
            where T : unmanaged
        {
            if (size == 0)
                return isAll;

            // AVX2 gather: index vector requires int32 lanes.
            if (Avx2.IsSupported && size >= 8 && stride <= int.MaxValue)
            {
                if (typeof(T) == typeof(float))
                    return ReduceStridedAxisBoolGatherFloat((float*)data, size, stride, isAll);
                if (typeof(T) == typeof(double))
                    return ReduceStridedAxisBoolGatherDouble((double*)data, size, stride, isAll);
                if (typeof(T) == typeof(int))
                    return ReduceStridedAxisBoolGatherInt32((int*)data, size, stride, isAll);
                if (typeof(T) == typeof(uint))
                    return ReduceStridedAxisBoolGatherInt32((int*)data, size, stride, isAll);
                if (typeof(T) == typeof(long))
                    return ReduceStridedAxisBoolGatherInt64((long*)data, size, stride, isAll);
                if (typeof(T) == typeof(ulong))
                    return ReduceStridedAxisBoolGatherInt64((long*)data, size, stride, isAll);
            }

            // Specialized scalar paths for the remaining primitive types — direct comparison
            // is significantly faster than EqualityComparer<T>.Default.Equals because the JIT
            // can vectorize/auto-unroll the loop and skip the static-readonly lookup.
            if (typeof(T) == typeof(byte))   return ReduceStridedAxisBoolByte((byte*)data, size, stride, isAll);
            if (typeof(T) == typeof(sbyte))  return ReduceStridedAxisBoolSByte((sbyte*)data, size, stride, isAll);
            if (typeof(T) == typeof(short))  return ReduceStridedAxisBoolInt16((short*)data, size, stride, isAll);
            if (typeof(T) == typeof(ushort)) return ReduceStridedAxisBoolUInt16((ushort*)data, size, stride, isAll);

            // Generic fallback (only hit when no specialized path applies — e.g. AVX2 missing).
            T zero = default;
            if (isAll)
            {
                for (long i = 0; i < size; i++)
                {
                    if (EqualityComparer<T>.Default.Equals(data[i * stride], zero))
                        return false;
                }
                return true;
            }

            for (long i = 0; i < size; i++)
            {
                if (!EqualityComparer<T>.Default.Equals(data[i * stride], zero))
                    return true;
            }
            return false;
        }

        // AVX2 32-bit integer gather: 8 ints per pass, 4-byte scale.
        // Works for int / uint (the comparison `== 0` is bitwise-identical for both).
        private static unsafe bool ReduceStridedAxisBoolGatherInt32(int* data, long size, long stride, bool isAll)
        {
            int strideInt = (int)stride;
            var indices = Vector256.Create(
                0, strideInt, strideInt * 2, strideInt * 3,
                strideInt * 4, strideInt * 5, strideInt * 6, strideInt * 7);
            int vCount = 8;
            long vEnd = size - vCount;
            long i = 0;

            if (isAll)
            {
                for (; i <= vEnd; i += vCount)
                {
                    var gathered = Avx2.GatherVector256(data + i * stride, indices, 4);
                    var mask = Vector256.Equals(gathered, Vector256<int>.Zero);
                    if (Vector256.ExtractMostSignificantBits(mask) != 0)
                        return false;
                }
                for (; i < size; i++)
                {
                    if (data[i * stride] == 0)
                        return false;
                }
                return true;
            }

            const uint allOnes = 0xFFu;
            for (; i <= vEnd; i += vCount)
            {
                var gathered = Avx2.GatherVector256(data + i * stride, indices, 4);
                var mask = Vector256.Equals(gathered, Vector256<int>.Zero);
                if (Vector256.ExtractMostSignificantBits(mask) != allOnes)
                    return true;
            }
            for (; i < size; i++)
            {
                if (data[i * stride] != 0)
                    return true;
            }
            return false;
        }

        // AVX2 64-bit integer gather: 4 longs per pass, 8-byte scale.
        // Works for long / ulong (the comparison `== 0` is bitwise-identical for both).
        private static unsafe bool ReduceStridedAxisBoolGatherInt64(long* data, long size, long stride, bool isAll)
        {
            int strideInt = (int)stride;
            var indices = Vector128.Create(0, strideInt, strideInt * 2, strideInt * 3);
            int vCount = 4;
            long vEnd = size - vCount;
            long i = 0;

            if (isAll)
            {
                for (; i <= vEnd; i += vCount)
                {
                    var gathered = Avx2.GatherVector256(data + i * stride, indices, 8);
                    var mask = Vector256.Equals(gathered, Vector256<long>.Zero);
                    if (Vector256.ExtractMostSignificantBits(mask) != 0)
                        return false;
                }
                for (; i < size; i++)
                {
                    if (data[i * stride] == 0)
                        return false;
                }
                return true;
            }

            const uint allOnes = 0xFu;
            for (; i <= vEnd; i += vCount)
            {
                var gathered = Avx2.GatherVector256(data + i * stride, indices, 8);
                var mask = Vector256.Equals(gathered, Vector256<long>.Zero);
                if (Vector256.ExtractMostSignificantBits(mask) != allOnes)
                    return true;
            }
            for (; i < size; i++)
            {
                if (data[i * stride] != 0)
                    return true;
            }
            return false;
        }

        // Specialized strided scalar paths for narrow integer types (no AVX2 gather for them).
        // Direct comparison instead of EqualityComparer; the JIT can ILP/unroll a tight loop.
        private static unsafe bool ReduceStridedAxisBoolByte(byte* data, long size, long stride, bool isAll)
        {
            if (isAll) { for (long i = 0; i < size; i++) if (data[i * stride] == 0) return false; return true; }
            for (long i = 0; i < size; i++) if (data[i * stride] != 0) return true; return false;
        }

        private static unsafe bool ReduceStridedAxisBoolSByte(sbyte* data, long size, long stride, bool isAll)
        {
            if (isAll) { for (long i = 0; i < size; i++) if (data[i * stride] == 0) return false; return true; }
            for (long i = 0; i < size; i++) if (data[i * stride] != 0) return true; return false;
        }

        private static unsafe bool ReduceStridedAxisBoolInt16(short* data, long size, long stride, bool isAll)
        {
            if (isAll) { for (long i = 0; i < size; i++) if (data[i * stride] == 0) return false; return true; }
            for (long i = 0; i < size; i++) if (data[i * stride] != 0) return true; return false;
        }

        private static unsafe bool ReduceStridedAxisBoolUInt16(ushort* data, long size, long stride, bool isAll)
        {
            if (isAll) { for (long i = 0; i < size; i++) if (data[i * stride] == 0) return false; return true; }
            for (long i = 0; i < size; i++) if (data[i * stride] != 0) return true; return false;
        }

        // AVX2 gather: 8 floats per pass, 4-byte scale.
        private static unsafe bool ReduceStridedAxisBoolGatherFloat(float* data, long size, long stride, bool isAll)
        {
            int strideInt = (int)stride;
            var indices = Vector256.Create(
                0, strideInt, strideInt * 2, strideInt * 3,
                strideInt * 4, strideInt * 5, strideInt * 6, strideInt * 7);
            int vCount = 8;
            long vEnd = size - vCount;
            long i = 0;

            // For All: any lane equal-to-zero kills the result. mask bits != 0 → return false.
            // For Any: any lane not-equal-to-zero saves the result. mask bits != allOnes → return true.
            //   NaN handles correctly: NaN == 0 is false, so NaN contributes a 0 bit (truthy lane).
            if (isAll)
            {
                for (; i <= vEnd; i += vCount)
                {
                    var gathered = Avx2.GatherVector256(data + i * stride, indices, 4);
                    var mask = Vector256.Equals(gathered, Vector256<float>.Zero);
                    if (Vector256.ExtractMostSignificantBits(mask) != 0)
                        return false;
                }
                for (; i < size; i++)
                {
                    if (data[i * stride] == 0f)
                        return false;
                }
                return true;
            }

            const uint allOnes = 0xFFu;
            for (; i <= vEnd; i += vCount)
            {
                var gathered = Avx2.GatherVector256(data + i * stride, indices, 4);
                var mask = Vector256.Equals(gathered, Vector256<float>.Zero);
                if (Vector256.ExtractMostSignificantBits(mask) != allOnes)
                    return true;
            }
            for (; i < size; i++)
            {
                if (data[i * stride] != 0f)
                    return true;
            }
            return false;
        }

        // AVX2 gather: 4 doubles per pass, 8-byte scale.
        private static unsafe bool ReduceStridedAxisBoolGatherDouble(double* data, long size, long stride, bool isAll)
        {
            int strideInt = (int)stride;
            var indices = Vector128.Create(0, strideInt, strideInt * 2, strideInt * 3);
            int vCount = 4;
            long vEnd = size - vCount;
            long i = 0;

            if (isAll)
            {
                for (; i <= vEnd; i += vCount)
                {
                    var gathered = Avx2.GatherVector256(data + i * stride, indices, 8);
                    var mask = Vector256.Equals(gathered, Vector256<double>.Zero);
                    if (Vector256.ExtractMostSignificantBits(mask) != 0)
                        return false;
                }
                for (; i < size; i++)
                {
                    if (data[i * stride] == 0.0)
                        return false;
                }
                return true;
            }

            const uint allOnes = 0xFu; // 4 lanes
            for (; i <= vEnd; i += vCount)
            {
                var gathered = Avx2.GatherVector256(data + i * stride, indices, 8);
                var mask = Vector256.Equals(gathered, Vector256<double>.Zero);
                if (Vector256.ExtractMostSignificantBits(mask) != allOnes)
                    return true;
            }
            for (; i < size; i++)
            {
                if (data[i * stride] != 0.0)
                    return true;
            }
            return false;
        }

        #endregion
    }
}
