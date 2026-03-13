using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Clip - SIMD-optimized Clip operations
// =============================================================================
//
// This partial class provides high-performance Clip operations using SIMD.
// Clip(x, min, max) = Min(Max(x, min), max)
//
// SIMD approach:
// - Create broadcast vectors for min and max values
// - Apply Vector.Max(x, minVec) then Vector.Min(result, maxVec)
// - Process remainder with scalar loop
//
// Three operation modes:
// - Both min and max: Clip to range [min, max]
// - Min only: Clip to [min, +inf)
// - Max only: Clip to (-inf, max]
//
// Two execution paths:
// - Contiguous: Direct SIMD vectorization (ClipHelper)
// - Strided: Coordinate-based iteration with scalar clip (ClipStrided)
//
// NaN handling (NumPy behavior):
// - For floating-point, NaN in data propagates: clip(NaN, min, max) = NaN
// - For NaN in min/max: entire output becomes NaN (not implemented here - caller handles)
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public sealed partial class ILKernelGenerator
    {
        #region Clip Helpers (Contiguous)

        /// <summary>
        /// SIMD-optimized Clip operation for contiguous arrays (min and max).
        /// Modifies the array in-place: data[i] = Min(Max(data[i], minVal), maxVal)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ClipHelper<T>(T* data, int size, T minVal, T maxVal) where T : unmanaged, IComparable<T>
        {
            if (size == 0) return;

            // Try SIMD path for supported types
            if (VectorBits >= 256 && size >= 32)
            {
                if (typeof(T) == typeof(float))
                {
                    ClipSimd256((float*)data, size, Unsafe.As<T, float>(ref minVal), Unsafe.As<T, float>(ref maxVal));
                    return;
                }
                if (typeof(T) == typeof(double))
                {
                    ClipSimd256((double*)data, size, Unsafe.As<T, double>(ref minVal), Unsafe.As<T, double>(ref maxVal));
                    return;
                }
                if (typeof(T) == typeof(int))
                {
                    ClipSimd256((int*)data, size, Unsafe.As<T, int>(ref minVal), Unsafe.As<T, int>(ref maxVal));
                    return;
                }
                if (typeof(T) == typeof(long))
                {
                    ClipSimd256((long*)data, size, Unsafe.As<T, long>(ref minVal), Unsafe.As<T, long>(ref maxVal));
                    return;
                }
                if (typeof(T) == typeof(short))
                {
                    ClipSimd256((short*)data, size, Unsafe.As<T, short>(ref minVal), Unsafe.As<T, short>(ref maxVal));
                    return;
                }
                if (typeof(T) == typeof(byte))
                {
                    ClipSimd256((byte*)data, size, Unsafe.As<T, byte>(ref minVal), Unsafe.As<T, byte>(ref maxVal));
                    return;
                }
            }
            else if (VectorBits >= 128 && size >= 16)
            {
                if (typeof(T) == typeof(float))
                {
                    ClipSimd128((float*)data, size, Unsafe.As<T, float>(ref minVal), Unsafe.As<T, float>(ref maxVal));
                    return;
                }
                if (typeof(T) == typeof(double))
                {
                    ClipSimd128((double*)data, size, Unsafe.As<T, double>(ref minVal), Unsafe.As<T, double>(ref maxVal));
                    return;
                }
                if (typeof(T) == typeof(int))
                {
                    ClipSimd128((int*)data, size, Unsafe.As<T, int>(ref minVal), Unsafe.As<T, int>(ref maxVal));
                    return;
                }
                if (typeof(T) == typeof(long))
                {
                    ClipSimd128((long*)data, size, Unsafe.As<T, long>(ref minVal), Unsafe.As<T, long>(ref maxVal));
                    return;
                }
                if (typeof(T) == typeof(short))
                {
                    ClipSimd128((short*)data, size, Unsafe.As<T, short>(ref minVal), Unsafe.As<T, short>(ref maxVal));
                    return;
                }
                if (typeof(T) == typeof(byte))
                {
                    ClipSimd128((byte*)data, size, Unsafe.As<T, byte>(ref minVal), Unsafe.As<T, byte>(ref maxVal));
                    return;
                }
            }

            // Scalar fallback
            ClipScalar(data, size, minVal, maxVal);
        }

        /// <summary>
        /// SIMD-optimized Min-only Clip operation (no upper bound).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ClipMinHelper<T>(T* data, int size, T minVal) where T : unmanaged, IComparable<T>
        {
            if (size == 0) return;

            // Try SIMD path
            if (VectorBits >= 256 && size >= 32)
            {
                if (typeof(T) == typeof(float))
                {
                    ClipMinSimd256((float*)data, size, Unsafe.As<T, float>(ref minVal));
                    return;
                }
                if (typeof(T) == typeof(double))
                {
                    ClipMinSimd256((double*)data, size, Unsafe.As<T, double>(ref minVal));
                    return;
                }
                if (typeof(T) == typeof(int))
                {
                    ClipMinSimd256((int*)data, size, Unsafe.As<T, int>(ref minVal));
                    return;
                }
                if (typeof(T) == typeof(long))
                {
                    ClipMinSimd256((long*)data, size, Unsafe.As<T, long>(ref minVal));
                    return;
                }
            }
            else if (VectorBits >= 128 && size >= 16)
            {
                if (typeof(T) == typeof(float))
                {
                    ClipMinSimd128((float*)data, size, Unsafe.As<T, float>(ref minVal));
                    return;
                }
                if (typeof(T) == typeof(double))
                {
                    ClipMinSimd128((double*)data, size, Unsafe.As<T, double>(ref minVal));
                    return;
                }
                if (typeof(T) == typeof(int))
                {
                    ClipMinSimd128((int*)data, size, Unsafe.As<T, int>(ref minVal));
                    return;
                }
                if (typeof(T) == typeof(long))
                {
                    ClipMinSimd128((long*)data, size, Unsafe.As<T, long>(ref minVal));
                    return;
                }
            }

            // Scalar fallback
            ClipMinScalar(data, size, minVal);
        }

        /// <summary>
        /// SIMD-optimized Max-only Clip operation (no lower bound).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ClipMaxHelper<T>(T* data, int size, T maxVal) where T : unmanaged, IComparable<T>
        {
            if (size == 0) return;

            // Try SIMD path
            if (VectorBits >= 256 && size >= 32)
            {
                if (typeof(T) == typeof(float))
                {
                    ClipMaxSimd256((float*)data, size, Unsafe.As<T, float>(ref maxVal));
                    return;
                }
                if (typeof(T) == typeof(double))
                {
                    ClipMaxSimd256((double*)data, size, Unsafe.As<T, double>(ref maxVal));
                    return;
                }
                if (typeof(T) == typeof(int))
                {
                    ClipMaxSimd256((int*)data, size, Unsafe.As<T, int>(ref maxVal));
                    return;
                }
                if (typeof(T) == typeof(long))
                {
                    ClipMaxSimd256((long*)data, size, Unsafe.As<T, long>(ref maxVal));
                    return;
                }
            }
            else if (VectorBits >= 128 && size >= 16)
            {
                if (typeof(T) == typeof(float))
                {
                    ClipMaxSimd128((float*)data, size, Unsafe.As<T, float>(ref maxVal));
                    return;
                }
                if (typeof(T) == typeof(double))
                {
                    ClipMaxSimd128((double*)data, size, Unsafe.As<T, double>(ref maxVal));
                    return;
                }
                if (typeof(T) == typeof(int))
                {
                    ClipMaxSimd128((int*)data, size, Unsafe.As<T, int>(ref maxVal));
                    return;
                }
                if (typeof(T) == typeof(long))
                {
                    ClipMaxSimd128((long*)data, size, Unsafe.As<T, long>(ref maxVal));
                    return;
                }
            }

            // Scalar fallback
            ClipMaxScalar(data, size, maxVal);
        }

        #endregion

        #region Scalar Fallback Implementations

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipScalar<T>(T* data, int size, T minVal, T maxVal) where T : unmanaged, IComparable<T>
        {
            for (int i = 0; i < size; i++)
            {
                var val = data[i];
                if (val.CompareTo(maxVal) > 0)
                    val = maxVal;
                else if (val.CompareTo(minVal) < 0)
                    val = minVal;
                data[i] = val;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipMinScalar<T>(T* data, int size, T minVal) where T : unmanaged, IComparable<T>
        {
            for (int i = 0; i < size; i++)
            {
                if (data[i].CompareTo(minVal) < 0)
                    data[i] = minVal;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipMaxScalar<T>(T* data, int size, T maxVal) where T : unmanaged, IComparable<T>
        {
            for (int i = 0; i < size; i++)
            {
                if (data[i].CompareTo(maxVal) > 0)
                    data[i] = maxVal;
            }
        }

        #endregion

        #region Vector256 SIMD Implementations

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipSimd256<T>(T* data, int size, T minVal, T maxVal) where T : unmanaged
        {
            int vectorCount = Vector256<T>.Count;
            int vectorEnd = size - vectorCount;
            var minVec = Vector256.Create(minVal);
            var maxVec = Vector256.Create(maxVal);

            int i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector256.Load(data + i);
                vec = Vector256.Max(vec, minVec);
                vec = Vector256.Min(vec, maxVec);
                vec.Store(data + i);
            }

            // Scalar tail
            for (; i < size; i++)
            {
                var val = data[i];
                if (Comparer<T>.Default.Compare(val, maxVal) > 0)
                    val = maxVal;
                else if (Comparer<T>.Default.Compare(val, minVal) < 0)
                    val = minVal;
                data[i] = val;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipMinSimd256<T>(T* data, int size, T minVal) where T : unmanaged
        {
            int vectorCount = Vector256<T>.Count;
            int vectorEnd = size - vectorCount;
            var minVec = Vector256.Create(minVal);

            int i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector256.Load(data + i);
                vec = Vector256.Max(vec, minVec);
                vec.Store(data + i);
            }

            // Scalar tail
            for (; i < size; i++)
            {
                if (Comparer<T>.Default.Compare(data[i], minVal) < 0)
                    data[i] = minVal;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipMaxSimd256<T>(T* data, int size, T maxVal) where T : unmanaged
        {
            int vectorCount = Vector256<T>.Count;
            int vectorEnd = size - vectorCount;
            var maxVec = Vector256.Create(maxVal);

            int i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector256.Load(data + i);
                vec = Vector256.Min(vec, maxVec);
                vec.Store(data + i);
            }

            // Scalar tail
            for (; i < size; i++)
            {
                if (Comparer<T>.Default.Compare(data[i], maxVal) > 0)
                    data[i] = maxVal;
            }
        }

        #endregion

        #region Vector128 SIMD Implementations

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipSimd128<T>(T* data, int size, T minVal, T maxVal) where T : unmanaged
        {
            int vectorCount = Vector128<T>.Count;
            int vectorEnd = size - vectorCount;
            var minVec = Vector128.Create(minVal);
            var maxVec = Vector128.Create(maxVal);

            int i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector128.Load(data + i);
                vec = Vector128.Max(vec, minVec);
                vec = Vector128.Min(vec, maxVec);
                vec.Store(data + i);
            }

            // Scalar tail
            for (; i < size; i++)
            {
                var val = data[i];
                if (Comparer<T>.Default.Compare(val, maxVal) > 0)
                    val = maxVal;
                else if (Comparer<T>.Default.Compare(val, minVal) < 0)
                    val = minVal;
                data[i] = val;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipMinSimd128<T>(T* data, int size, T minVal) where T : unmanaged
        {
            int vectorCount = Vector128<T>.Count;
            int vectorEnd = size - vectorCount;
            var minVec = Vector128.Create(minVal);

            int i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector128.Load(data + i);
                vec = Vector128.Max(vec, minVec);
                vec.Store(data + i);
            }

            // Scalar tail
            for (; i < size; i++)
            {
                if (Comparer<T>.Default.Compare(data[i], minVal) < 0)
                    data[i] = minVal;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipMaxSimd128<T>(T* data, int size, T maxVal) where T : unmanaged
        {
            int vectorCount = Vector128<T>.Count;
            int vectorEnd = size - vectorCount;
            var maxVec = Vector128.Create(maxVal);

            int i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector128.Load(data + i);
                vec = Vector128.Min(vec, maxVec);
                vec.Store(data + i);
            }

            // Scalar tail
            for (; i < size; i++)
            {
                if (Comparer<T>.Default.Compare(data[i], maxVal) > 0)
                    data[i] = maxVal;
            }
        }

        #endregion

        #region Clip Strided (Non-Contiguous)

        /// <summary>
        /// Clip operation for strided (non-contiguous) arrays.
        /// Uses coordinate-based iteration via Shape.TransformOffset.
        /// </summary>
        /// <remarks>
        /// This handles arrays that are:
        /// - Transposed (stride order differs from dimension order)
        /// - Sliced with step (e.g., arr[::2])
        /// - Views with non-standard memory layout
        ///
        /// Performance is O(n) with coordinate overhead per element.
        /// For contiguous arrays, use ClipHelper instead.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ClipStrided<T>(T* data, int size, T minVal, T maxVal, Shape shape) where T : unmanaged, IComparable<T>
        {
            if (size == 0) return;

            // Special case: if actually contiguous, use the fast path
            if (shape.IsContiguous)
            {
                ClipHelper(data + shape.Offset, size, minVal, maxVal);
                return;
            }

            // Strided iteration using coordinate transformation
            for (int i = 0; i < size; i++)
            {
                int offset = shape.TransformOffset(i);
                var val = data[offset];
                if (val.CompareTo(maxVal) > 0)
                    val = maxVal;
                else if (val.CompareTo(minVal) < 0)
                    val = minVal;
                data[offset] = val;
            }
        }

        /// <summary>
        /// Min-only Clip operation for strided arrays.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ClipMinStrided<T>(T* data, int size, T minVal, Shape shape) where T : unmanaged, IComparable<T>
        {
            if (size == 0) return;

            if (shape.IsContiguous)
            {
                ClipMinHelper(data + shape.Offset, size, minVal);
                return;
            }

            for (int i = 0; i < size; i++)
            {
                int offset = shape.TransformOffset(i);
                if (data[offset].CompareTo(minVal) < 0)
                    data[offset] = minVal;
            }
        }

        /// <summary>
        /// Max-only Clip operation for strided arrays.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ClipMaxStrided<T>(T* data, int size, T maxVal, Shape shape) where T : unmanaged, IComparable<T>
        {
            if (size == 0) return;

            if (shape.IsContiguous)
            {
                ClipMaxHelper(data + shape.Offset, size, maxVal);
                return;
            }

            for (int i = 0; i < size; i++)
            {
                int offset = shape.TransformOffset(i);
                if (data[offset].CompareTo(maxVal) > 0)
                    data[offset] = maxVal;
            }
        }

        #endregion

        #region Unified Clip Entry Points

        /// <summary>
        /// Unified Clip operation that handles both contiguous and strided arrays.
        /// Automatically selects the optimal path based on array contiguity.
        /// </summary>
        /// <param name="data">Pointer to the data buffer (at offset 0, not adjusted for shape.offset)</param>
        /// <param name="size">Number of elements to process</param>
        /// <param name="minVal">Minimum value to clip to</param>
        /// <param name="maxVal">Maximum value to clip to</param>
        /// <param name="shape">Shape describing the memory layout</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ClipUnified<T>(T* data, int size, T minVal, T maxVal, Shape shape) where T : unmanaged, IComparable<T>
        {
            if (shape.IsContiguous)
                ClipHelper(data + shape.Offset, size, minVal, maxVal);
            else
                ClipStrided(data, size, minVal, maxVal, shape);
        }

        /// <summary>
        /// Unified Min-only Clip operation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ClipMinUnified<T>(T* data, int size, T minVal, Shape shape) where T : unmanaged, IComparable<T>
        {
            if (shape.IsContiguous)
                ClipMinHelper(data + shape.Offset, size, minVal);
            else
                ClipMinStrided(data, size, minVal, shape);
        }

        /// <summary>
        /// Unified Max-only Clip operation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ClipMaxUnified<T>(T* data, int size, T maxVal, Shape shape) where T : unmanaged, IComparable<T>
        {
            if (shape.IsContiguous)
                ClipMaxHelper(data + shape.Offset, size, maxVal);
            else
                ClipMaxStrided(data, size, maxVal, shape);
        }

        #endregion
    }
}
