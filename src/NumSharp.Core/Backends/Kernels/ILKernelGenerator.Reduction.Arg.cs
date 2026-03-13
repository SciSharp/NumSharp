using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Reduction.Arg.cs - ArgMax/ArgMin Reductions
// =============================================================================
//
// RESPONSIBILITY:
//   - ArgMax/ArgMin reduction with SIMD index tracking
//   - Two-pass algorithm: find extreme value with SIMD, then find index
//   - EmitArgMaxMinSimdLoop() - IL emission
//   - ArgMaxSimdHelper<T>(), ArgMinSimdHelper<T>() - SIMD helpers
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public sealed partial class ILKernelGenerator
    {
        #region ArgMax/ArgMin Reduction Helpers
        /// <summary>
        /// Emit ArgMax/ArgMin SIMD loop.
        /// Uses helper methods for clean implementation with SIMD index tracking.
        /// </summary>
        private static void EmitArgMaxMinSimdLoop(ILGenerator il, ElementReductionKernelKey key, int inputSize)
        {
            var helperMethod = typeof(ILKernelGenerator).GetMethod(
                key.Op == ReductionOp.ArgMax ? nameof(ArgMaxSimdHelper) : nameof(ArgMinSimdHelper),
                BindingFlags.NonPublic | BindingFlags.Static);

            var genericHelper = helperMethod!.MakeGenericMethod(GetClrType(key.InputType));

            // Call helper: ArgMaxSimdHelper<T>(input, totalSize) or ArgMinSimdHelper<T>(input, totalSize)
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldarg_S, (byte)4); // totalSize
            il.EmitCall(OpCodes.Call, genericHelper, null);

            // Result (int) is already on stack
        }

        /// <summary>
        /// SIMD helper for ArgMax reduction.
        /// Returns the index of the maximum element.
        /// Uses SIMD to find candidates then scalar to resolve exact index.
        /// </summary>
        internal static unsafe int ArgMaxSimdHelper<T>(void* input, int totalSize) where T : unmanaged, IComparable<T>
        {
            if (totalSize == 0)
                return -1;

            if (totalSize == 1)
                return 0;

            T* src = (T*)input;
            T bestValue = src[0];
            int bestIndex = 0;

            if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && totalSize >= Vector256<T>.Count * 2)
            {
                int vectorCount = Vector256<T>.Count;
                int vectorEnd = totalSize - vectorCount;

                // First pass: find the maximum value using SIMD
                var maxVec = Vector256.Load(src);
                int i = vectorCount;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    maxVec = Vector256.Max(maxVec, vec);
                }

                // Horizontal reduce the max vector to find the scalar max
                T maxValue = maxVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    T elem = maxVec.GetElement(j);
                    if (elem.CompareTo(maxValue) > 0)
                        maxValue = elem;
                }

                // Process scalar tail for max value
                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(maxValue) > 0)
                        maxValue = src[i];
                }

                // Second pass: find the first index with the max value
                // Use SIMD to quickly scan for the max value
                var targetVec = Vector256.Create(maxValue);
                for (i = 0; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var mask = Vector256.Equals(vec, targetVec);
                    uint bits = Vector256.ExtractMostSignificantBits(mask);
                    if (bits != 0)
                    {
                        // Found it! Return index of first match
                        return i + System.Numerics.BitOperations.TrailingZeroCount(bits);
                    }
                }

                // Check scalar tail
                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(maxValue) == 0)
                        return i;
                }

                return 0; // Should never reach here
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && totalSize >= Vector128<T>.Count * 2)
            {
                int vectorCount = Vector128<T>.Count;
                int vectorEnd = totalSize - vectorCount;

                var maxVec = Vector128.Load(src);
                int i = vectorCount;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    maxVec = Vector128.Max(maxVec, vec);
                }

                T maxValue = maxVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    T elem = maxVec.GetElement(j);
                    if (elem.CompareTo(maxValue) > 0)
                        maxValue = elem;
                }

                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(maxValue) > 0)
                        maxValue = src[i];
                }

                var targetVec = Vector128.Create(maxValue);
                for (i = 0; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var mask = Vector128.Equals(vec, targetVec);
                    uint bits = Vector128.ExtractMostSignificantBits(mask);
                    if (bits != 0)
                    {
                        return i + System.Numerics.BitOperations.TrailingZeroCount(bits);
                    }
                }

                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(maxValue) == 0)
                        return i;
                }

                return 0;
            }
            else
            {
                // Scalar fallback
                for (int i = 1; i < totalSize; i++)
                {
                    if (src[i].CompareTo(bestValue) > 0)
                    {
                        bestValue = src[i];
                        bestIndex = i;
                    }
                }
                return bestIndex;
            }
        }

        /// <summary>
        /// SIMD helper for ArgMin reduction.
        /// Returns the index of the minimum element.
        /// Uses SIMD to find candidates then scalar to resolve exact index.
        /// </summary>
        internal static unsafe int ArgMinSimdHelper<T>(void* input, int totalSize) where T : unmanaged, IComparable<T>
        {
            if (totalSize == 0)
                return -1;

            if (totalSize == 1)
                return 0;

            T* src = (T*)input;
            T bestValue = src[0];
            int bestIndex = 0;

            if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && totalSize >= Vector256<T>.Count * 2)
            {
                int vectorCount = Vector256<T>.Count;
                int vectorEnd = totalSize - vectorCount;

                // First pass: find the minimum value using SIMD
                var minVec = Vector256.Load(src);
                int i = vectorCount;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    minVec = Vector256.Min(minVec, vec);
                }

                // Horizontal reduce the min vector to find the scalar min
                T minValue = minVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    T elem = minVec.GetElement(j);
                    if (elem.CompareTo(minValue) < 0)
                        minValue = elem;
                }

                // Process scalar tail for min value
                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(minValue) < 0)
                        minValue = src[i];
                }

                // Second pass: find the first index with the min value
                var targetVec = Vector256.Create(minValue);
                for (i = 0; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var mask = Vector256.Equals(vec, targetVec);
                    uint bits = Vector256.ExtractMostSignificantBits(mask);
                    if (bits != 0)
                    {
                        return i + System.Numerics.BitOperations.TrailingZeroCount(bits);
                    }
                }

                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(minValue) == 0)
                        return i;
                }

                return 0;
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && totalSize >= Vector128<T>.Count * 2)
            {
                int vectorCount = Vector128<T>.Count;
                int vectorEnd = totalSize - vectorCount;

                var minVec = Vector128.Load(src);
                int i = vectorCount;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    minVec = Vector128.Min(minVec, vec);
                }

                T minValue = minVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    T elem = minVec.GetElement(j);
                    if (elem.CompareTo(minValue) < 0)
                        minValue = elem;
                }

                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(minValue) < 0)
                        minValue = src[i];
                }

                var targetVec = Vector128.Create(minValue);
                for (i = 0; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var mask = Vector128.Equals(vec, targetVec);
                    uint bits = Vector128.ExtractMostSignificantBits(mask);
                    if (bits != 0)
                    {
                        return i + System.Numerics.BitOperations.TrailingZeroCount(bits);
                    }
                }

                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(minValue) == 0)
                        return i;
                }

                return 0;
            }
            else
            {
                // Scalar fallback
                for (int i = 1; i < totalSize; i++)
                {
                    if (src[i].CompareTo(bestValue) < 0)
                    {
                        bestValue = src[i];
                        bestIndex = i;
                    }
                }
                return bestIndex;
            }
        }

        #endregion
    }
}
