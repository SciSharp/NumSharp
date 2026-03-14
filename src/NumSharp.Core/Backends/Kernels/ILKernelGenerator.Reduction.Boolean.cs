using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Reduction.Boolean.cs - All/Any Boolean Reductions
// =============================================================================
//
// RESPONSIBILITY:
//   - All/Any reduction with early-exit SIMD optimization
//   - EmitAllAnySimdLoop() - IL emission for All/Any
//   - AllSimdHelper<T>() - returns false immediately on first zero
//   - AnySimdHelper<T>() - returns true immediately on first non-zero
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        #region Boolean Reduction Helpers (All/Any)
        /// <summary>
        /// Emit All/Any SIMD loop with early-exit.
        /// All: returns false immediately when any zero is found
        /// Any: returns true immediately when any non-zero is found
        /// </summary>
        private static void EmitAllAnySimdLoop(ILGenerator il, ElementReductionKernelKey key, int inputSize)
        {
            // For All/Any, we use a helper method approach because:
            // 1. Early-exit logic is complex to emit in IL
            // 2. The helper method can be JIT-optimized effectively
            // 3. This matches the pattern used elsewhere in the codebase

            var helperMethod = typeof(ILKernelGenerator).GetMethod(
                key.Op == ReductionOp.All ? nameof(AllSimdHelper) : nameof(AnySimdHelper),
                BindingFlags.NonPublic | BindingFlags.Static);

            var genericHelper = helperMethod!.MakeGenericMethod(GetClrType(key.InputType));

            // Call helper: AllSimdHelper<T>(input, totalSize) or AnySimdHelper<T>(input, totalSize)
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldarg_S, (byte)4); // totalSize
            il.EmitCall(OpCodes.Call, genericHelper, null);

            // Result (bool) is already on stack, but we need to convert to TResult (also bool for All/Any)
            // Stack has: bool result - convert to byte (0 or 1) for bool return
        }

        /// <summary>
        /// SIMD helper for All reduction with early-exit.
        /// Returns true if ALL elements are non-zero.
        /// </summary>
        internal static unsafe bool AllSimdHelper<T>(void* input, long totalSize) where T : unmanaged
        {
            if (totalSize == 0)
                return true; // NumPy: all([]) == True (vacuous truth)

            T* src = (T*)input;

            if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && totalSize >= Vector256<T>.Count)
            {
                int vectorCount = Vector256<T>.Count;
                long vectorEnd = totalSize - vectorCount;
                var zero = Vector256<T>.Zero;
                long i = 0;

                // SIMD loop with early exit
                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var mask = Vector256.Equals(vec, zero);

                    // If ANY element equals zero, return false
                    if (Vector256.ExtractMostSignificantBits(mask) != 0)
                        return false;
                }

                // Scalar tail
                for (; i < totalSize; i++)
                {
                    if (System.Collections.Generic.EqualityComparer<T>.Default.Equals(src[i], default))
                        return false;
                }

                return true;
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && totalSize >= Vector128<T>.Count)
            {
                int vectorCount = Vector128<T>.Count;
                long vectorEnd = totalSize - vectorCount;
                var zero = Vector128<T>.Zero;
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var mask = Vector128.Equals(vec, zero);

                    if (Vector128.ExtractMostSignificantBits(mask) != 0)
                        return false;
                }

                for (; i < totalSize; i++)
                {
                    if (System.Collections.Generic.EqualityComparer<T>.Default.Equals(src[i], default))
                        return false;
                }

                return true;
            }
            else
            {
                // Scalar fallback
                for (long i = 0; i < totalSize; i++)
                {
                    if (System.Collections.Generic.EqualityComparer<T>.Default.Equals(src[i], default))
                        return false;
                }
                return true;
            }
        }

        /// <summary>
        /// SIMD helper for Any reduction with early-exit.
        /// Returns true if ANY element is non-zero.
        /// </summary>
        internal static unsafe bool AnySimdHelper<T>(void* input, int totalSize) where T : unmanaged
        {
            if (totalSize == 0)
                return false; // NumPy: any([]) == False

            T* src = (T*)input;

            if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && totalSize >= Vector256<T>.Count)
            {
                int vectorCount = Vector256<T>.Count;
                int vectorEnd = totalSize - vectorCount;
                var zero = Vector256<T>.Zero;
                uint allZeroMask = (1u << vectorCount) - 1;
                int i = 0;

                // SIMD loop with early exit
                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var mask = Vector256.Equals(vec, zero);
                    uint bits = Vector256.ExtractMostSignificantBits(mask);

                    // If NOT all elements are zero, we found a non-zero
                    if (bits != allZeroMask)
                        return true;
                }

                // Scalar tail
                for (; i < totalSize; i++)
                {
                    if (!System.Collections.Generic.EqualityComparer<T>.Default.Equals(src[i], default))
                        return true;
                }

                return false;
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && totalSize >= Vector128<T>.Count)
            {
                int vectorCount = Vector128<T>.Count;
                int vectorEnd = totalSize - vectorCount;
                var zero = Vector128<T>.Zero;
                uint allZeroMask = (1u << vectorCount) - 1;
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var mask = Vector128.Equals(vec, zero);
                    uint bits = Vector128.ExtractMostSignificantBits(mask);

                    if (bits != allZeroMask)
                        return true;
                }

                for (; i < totalSize; i++)
                {
                    if (!System.Collections.Generic.EqualityComparer<T>.Default.Equals(src[i], default))
                        return true;
                }

                return false;
            }
            else
            {
                // Scalar fallback
                for (int i = 0; i < totalSize; i++)
                {
                    if (!System.Collections.Generic.EqualityComparer<T>.Default.Equals(src[i], default))
                        return true;
                }
                return false;
            }
        }

        #endregion
    }
}
