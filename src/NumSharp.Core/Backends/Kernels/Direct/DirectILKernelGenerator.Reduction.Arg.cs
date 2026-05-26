using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// DirectILKernelGenerator.Reduction.Arg.cs - ArgMax/ArgMin Reductions
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
    public static partial class DirectILKernelGenerator
    {
        #region ArgMax/ArgMin Reduction Helpers
        /// <summary>
        /// Emit ArgMax/ArgMin SIMD loop.
        /// Uses helper methods for clean implementation with SIMD index tracking.
        /// Dispatches to type-specific helpers for NaN-awareness (float/double) and Boolean.
        /// </summary>
        private static void EmitArgMaxMinSimdLoop(ILGenerator il, ElementReductionKernelKey key, int inputSize)
        {
            // Dispatch to specialized helpers for types needing special handling.
            bool isMax = key.Op == ReductionOp.ArgMax;
            MethodInfo helperMethod;
            bool isGeneric = true;

            if (key.InputType == NPTypeCode.Single)
            {
                helperMethod = GetHelper(isMax ? nameof(ArgMaxFloatNaNHelper) : nameof(ArgMinFloatNaNHelper));
                isGeneric = false;
            }
            else if (key.InputType == NPTypeCode.Double)
            {
                helperMethod = GetHelper(isMax ? nameof(ArgMaxDoubleNaNHelper) : nameof(ArgMinDoubleNaNHelper));
                isGeneric = false;
            }
            else if (key.InputType == NPTypeCode.Half)
            {
                helperMethod = GetHelper(isMax ? nameof(ArgMaxHalfNaNHelper) : nameof(ArgMinHalfNaNHelper));
                isGeneric = false;
            }
            else if (key.InputType == NPTypeCode.Boolean)
            {
                helperMethod = GetHelper(isMax ? nameof(ArgMaxBoolHelper) : nameof(ArgMinBoolHelper));
                isGeneric = false;
            }
            else if (key.InputType == NPTypeCode.Complex)
            {
                // Complex uses magnitude comparison.
                helperMethod = GetHelper(isMax ? nameof(ArgMaxComplexHelper) : nameof(ArgMinComplexHelper));
                isGeneric = false;
            }
            else
            {
                // Generic SIMD path for integer types.
                helperMethod = GetHelper(isMax ? nameof(ArgMaxSimdHelper) : nameof(ArgMinSimdHelper));
            }

            if (isGeneric)
                helperMethod = helperMethod.MakeGenericMethod(GetClrType(key.InputType));

            // Call helper: *Helper(input, totalSize)
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldarg_S, (byte)4); // totalSize
            il.EmitCall(OpCodes.Call, helperMethod, null);

            // Result (long) is already on stack
        }

        /// <summary>
        /// SIMD helper for ArgMax reduction.
        /// Returns the index of the maximum element.
        /// Uses SIMD to find candidates then scalar to resolve exact index.
        /// </summary>
        internal static unsafe long ArgMaxSimdHelper<T>(void* input, long totalSize) where T : unmanaged, IComparable<T>
        {
            if (totalSize == 0)
                return -1;

            if (totalSize == 1)
                return 0;

            T* src = (T*)input;
            T bestValue = src[0];
            long bestIndex = 0;

            int vectorCount = Vector256<T>.Count;
            if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && totalSize >= vectorCount * 2)
            {
                long vectorEnd = totalSize - vectorCount;

                // First pass: find the maximum value using SIMD
                var maxVec = Vector256.Load(src);
                long i = vectorCount;

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
            vectorCount = Vector128<T>.Count;
            if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && totalSize >= vectorCount * 2)
            {
                long vectorEnd = totalSize - vectorCount;

                var maxVec = Vector128.Load(src);
                long i = vectorCount;

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
            // Scalar fallback
            for (long i = 1; i < totalSize; i++)
            {
                if (src[i].CompareTo(bestValue) > 0)
                {
                    bestValue = src[i];
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        /// <summary>
        /// SIMD helper for ArgMin reduction.
        /// Returns the index of the minimum element.
        /// Uses SIMD to find candidates then scalar to resolve exact index.
        /// </summary>
        internal static unsafe long ArgMinSimdHelper<T>(void* input, long totalSize) where T : unmanaged, IComparable<T>
        {
            if (totalSize == 0)
                return -1;

            if (totalSize == 1)
                return 0;

            T* src = (T*)input;
            T bestValue = src[0];
            long bestIndex = 0;

            int vectorCount = Vector256<T>.Count;
            if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && totalSize >= vectorCount * 2)
            {
                long vectorEnd = totalSize - vectorCount;

                // First pass: find the minimum value using SIMD
                var minVec = Vector256.Load(src);
                long i = vectorCount;

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
            vectorCount = Vector128<T>.Count;
            if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && totalSize >= vectorCount * 2)
            {
                long vectorEnd = totalSize - vectorCount;

                var minVec = Vector128.Load(src);
                long i = vectorCount;

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
            // Scalar fallback
            for (long i = 1; i < totalSize; i++)
            {
                if (src[i].CompareTo(bestValue) < 0)
                {
                    bestValue = src[i];
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        #endregion

        #region NaN-Aware ArgMax/ArgMin Helpers (Float/Double)

        /// <summary>
        /// ArgMax helper for float with NaN awareness.
        /// NumPy behavior: first NaN always wins (considered "maximum").
        ///
        /// Known gap: NumSharp ~12× behind NumPy on 1M float (1322 µs vs 106 µs).
        /// The gap is platform-specific. NumPy uses x86 vmaxps + a separate
        /// VPTEST for NaN detection in their C kernel; .NET's Vector256.Max
        /// for float falls through a lane-by-lane NaN-propagating path that's
        /// ~57% slower than vmaxps (measured 887 µs vs 565 µs on 1M float).
        /// Switching to System.Runtime.Intrinsics.X86.Avx.Max would close
        /// half the gap but introduces a correctness bug: vmaxps returns the
        /// SECOND operand when either input is NaN, so NaN from the first
        /// operand silently disappears — breaks the (data=[NaN, ...]) case.
        /// Proper fix requires IL-emitted single-pass argmax with vectorized
        /// index tracking; tracked separately.
        /// </summary>
        internal static unsafe long ArgMaxFloatNaNHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;
            if (totalSize == 1) return 0;

            float* src = (float*)input;
            float bestValue = src[0];
            long bestIndex = 0;

            for (long i = 1; i < totalSize; i++)
            {
                float val = src[i];
                // NumPy: first NaN always wins
                if (val > bestValue || (float.IsNaN(val) && !float.IsNaN(bestValue)))
                {
                    bestValue = val;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        /// <summary>
        /// ArgMin helper for float with NaN awareness.
        /// NumPy behavior: first NaN always wins (considered "minimum").
        /// </summary>
        internal static unsafe long ArgMinFloatNaNHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;
            if (totalSize == 1) return 0;

            float* src = (float*)input;
            float bestValue = src[0];
            long bestIndex = 0;

            for (long i = 1; i < totalSize; i++)
            {
                float val = src[i];
                // NumPy: first NaN always wins
                if (val < bestValue || (float.IsNaN(val) && !float.IsNaN(bestValue)))
                {
                    bestValue = val;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        /// <summary>
        /// ArgMax helper for double with NaN awareness.
        /// NumPy behavior: first NaN always wins (considered "maximum").
        /// </summary>
        internal static unsafe long ArgMaxDoubleNaNHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;
            if (totalSize == 1) return 0;

            double* src = (double*)input;
            double bestValue = src[0];
            long bestIndex = 0;

            for (long i = 1; i < totalSize; i++)
            {
                double val = src[i];
                // NumPy: first NaN always wins
                if (val > bestValue || (double.IsNaN(val) && !double.IsNaN(bestValue)))
                {
                    bestValue = val;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        /// <summary>
        /// ArgMin helper for double with NaN awareness.
        /// NumPy behavior: first NaN always wins (considered "minimum").
        /// </summary>
        internal static unsafe long ArgMinDoubleNaNHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;
            if (totalSize == 1) return 0;

            double* src = (double*)input;
            double bestValue = src[0];
            long bestIndex = 0;

            for (long i = 1; i < totalSize; i++)
            {
                double val = src[i];
                // NumPy: first NaN always wins
                if (val < bestValue || (double.IsNaN(val) && !double.IsNaN(bestValue)))
                {
                    bestValue = val;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        /// <summary>
        /// ArgMax helper for Half with NaN awareness.
        /// NumPy behavior: first NaN always wins (considered "maximum").
        /// </summary>
        internal static unsafe long ArgMaxHalfNaNHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;
            if (totalSize == 1) return 0;

            Half* src = (Half*)input;
            Half bestValue = src[0];
            long bestIndex = 0;

            for (long i = 1; i < totalSize; i++)
            {
                Half val = src[i];
                // NumPy: first NaN always wins
                if (val > bestValue || (Half.IsNaN(val) && !Half.IsNaN(bestValue)))
                {
                    bestValue = val;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        /// <summary>
        /// ArgMin helper for Half with NaN awareness.
        /// NumPy behavior: first NaN always wins (considered "minimum").
        /// </summary>
        internal static unsafe long ArgMinHalfNaNHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;
            if (totalSize == 1) return 0;

            Half* src = (Half*)input;
            Half bestValue = src[0];
            long bestIndex = 0;

            for (long i = 1; i < totalSize; i++)
            {
                Half val = src[i];
                // NumPy: first NaN always wins
                if (val < bestValue || (Half.IsNaN(val) && !Half.IsNaN(bestValue)))
                {
                    bestValue = val;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        #endregion

        #region Boolean ArgMax/ArgMin Helpers

        /// <summary>
        /// ArgMax helper for boolean arrays.
        /// Boolean: True=1, False=0, so argmax finds first True.
        /// </summary>
        internal static unsafe long ArgMaxBoolHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;
            if (totalSize == 1) return 0;

            bool* src = (bool*)input;
            bool bestValue = src[0];
            long bestIndex = 0;

            // If first is already True, we can't do better
            if (bestValue) return 0;

            for (long i = 1; i < totalSize; i++)
            {
                if (src[i]) // True > False
                {
                    return i; // First True found
                }
            }
            return bestIndex; // All False, return 0
        }

        /// <summary>
        /// ArgMin helper for boolean arrays.
        /// Boolean: True=1, False=0, so argmin finds first False.
        /// </summary>
        internal static unsafe long ArgMinBoolHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;
            if (totalSize == 1) return 0;

            bool* src = (bool*)input;
            bool bestValue = src[0];
            long bestIndex = 0;

            // If first is already False, we can't do better
            if (!bestValue) return 0;

            for (long i = 1; i < totalSize; i++)
            {
                if (!src[i]) // False < True
                {
                    return i; // First False found
                }
            }
            return bestIndex; // All True, return 0
        }

        #endregion

        #region Complex ArgMax/ArgMin Helpers

        /// <summary>
        /// ArgMax helper for Complex arrays.
        /// NumPy: argmax uses magnitude |z| = sqrt(real² + imag²) for comparison.
        /// On tie (equal magnitudes), returns first occurrence.
        /// </summary>
        internal static unsafe long ArgMaxComplexHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;
            if (totalSize == 1) return 0;

            Complex* src = (Complex*)input;
            double bestMagnitude = Complex.Abs(src[0]);
            long bestIndex = 0;

            for (long i = 1; i < totalSize; i++)
            {
                double mag = Complex.Abs(src[i]);
                if (mag > bestMagnitude)
                {
                    bestMagnitude = mag;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        /// <summary>
        /// ArgMin helper for Complex arrays.
        /// NumPy: argmin uses magnitude |z| = sqrt(real² + imag²) for comparison.
        /// On tie (equal magnitudes), returns first occurrence.
        /// </summary>
        internal static unsafe long ArgMinComplexHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;
            if (totalSize == 1) return 0;

            Complex* src = (Complex*)input;
            double bestMagnitude = Complex.Abs(src[0]);
            long bestIndex = 0;

            for (long i = 1; i < totalSize; i++)
            {
                double mag = Complex.Abs(src[i]);
                if (mag < bestMagnitude)
                {
                    bestMagnitude = mag;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        #endregion
    }
}
