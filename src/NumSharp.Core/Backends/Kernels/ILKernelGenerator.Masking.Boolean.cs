using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Masking.Boolean.cs - Boolean Masking SIMD Helpers
// =============================================================================
//
// RESPONSIBILITY:
//   - CountTrueSimdHelper - count true values in bool array
//   - CopyMaskedElementsHelper<T> - copy elements where mask is true
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        #region Boolean Masking SIMD Helpers

        /// <summary>
        /// SIMD helper to count true values in a boolean array.
        /// </summary>
        internal static unsafe long CountTrueSimdHelper(bool* mask, long size)
        {
            if (size == 0)
                return 0;

            long count = 0;

            if (Vector256.IsHardwareAccelerated && Vector256<byte>.IsSupported && size >= Vector256<byte>.Count)
            {
                int vectorCount = Vector256<byte>.Count;
                long vectorEnd = size - vectorCount;
                var zero = Vector256<byte>.Zero;
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load((byte*)(mask + i));
                    var cmp = Vector256.Equals(vec, zero);
                    uint bits = Vector256.ExtractMostSignificantBits(cmp);

                    // Count non-zero (true) values: invert mask, popcount
                    uint nonZeroBits = ~bits;
                    count += System.Numerics.BitOperations.PopCount(nonZeroBits);
                }

                // Scalar tail
                for (; i < size; i++)
                {
                    if (mask[i])
                        count++;
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<byte>.IsSupported && size >= Vector128<byte>.Count)
            {
                int vectorCount = Vector128<byte>.Count;
                long vectorEnd = size - vectorCount;
                var zero = Vector128<byte>.Zero;
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load((byte*)(mask + i));
                    var cmp = Vector128.Equals(vec, zero);
                    uint bits = Vector128.ExtractMostSignificantBits(cmp);

                    uint nonZeroBits = ~bits & 0xFFFFu;
                    count += System.Numerics.BitOperations.PopCount(nonZeroBits);
                }

                for (; i < size; i++)
                {
                    if (mask[i])
                        count++;
                }
            }
            else
            {
                // Scalar fallback
                for (long i = 0; i < size; i++)
                {
                    if (mask[i])
                        count++;
                }
            }

            return count;
        }

        /// <summary>
        /// SIMD helper to copy elements where mask is true.
        /// Copies from src to dst where mask[i] is true.
        /// </summary>
        /// <returns>Number of elements copied</returns>
        internal static unsafe long CopyMaskedElementsHelper<T>(T* src, bool* mask, T* dst, long size)
            where T : unmanaged
        {
            long dstIdx = 0;

            // For masking, we can't easily vectorize the gather/scatter
            // But we can vectorize the mask scanning to find true indices faster
            if (Vector256.IsHardwareAccelerated && Vector256<byte>.IsSupported && size >= Vector256<byte>.Count)
            {
                int vectorCount = Vector256<byte>.Count;
                long vectorEnd = size - vectorCount;
                var zero = Vector256<byte>.Zero;
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var maskVec = Vector256.Load((byte*)(mask + i));
                    var cmp = Vector256.Equals(maskVec, zero);
                    uint bits = Vector256.ExtractMostSignificantBits(cmp);
                    uint nonZeroBits = ~bits;

                    // Copy elements where mask is true
                    while (nonZeroBits != 0)
                    {
                        int bitPos = System.Numerics.BitOperations.TrailingZeroCount(nonZeroBits);
                        dst[dstIdx++] = src[i + bitPos];
                        nonZeroBits &= nonZeroBits - 1;
                    }
                }

                // Scalar tail
                for (; i < size; i++)
                {
                    if (mask[i])
                        dst[dstIdx++] = src[i];
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<byte>.IsSupported && size >= Vector128<byte>.Count)
            {
                int vectorCount = Vector128<byte>.Count;
                long vectorEnd = size - vectorCount;
                var zero = Vector128<byte>.Zero;
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var maskVec = Vector128.Load((byte*)(mask + i));
                    var cmp = Vector128.Equals(maskVec, zero);
                    uint bits = Vector128.ExtractMostSignificantBits(cmp);
                    uint nonZeroBits = ~bits & 0xFFFFu;

                    while (nonZeroBits != 0)
                    {
                        int bitPos = System.Numerics.BitOperations.TrailingZeroCount(nonZeroBits);
                        dst[dstIdx++] = src[i + bitPos];
                        nonZeroBits &= nonZeroBits - 1;
                    }
                }

                for (; i < size; i++)
                {
                    if (mask[i])
                        dst[dstIdx++] = src[i];
                }
            }
            else
            {
                // Scalar fallback
                for (long i = 0; i < size; i++)
                {
                    if (mask[i])
                        dst[dstIdx++] = src[i];
                }
            }

            return dstIdx;
        }

        #endregion
    }
}
