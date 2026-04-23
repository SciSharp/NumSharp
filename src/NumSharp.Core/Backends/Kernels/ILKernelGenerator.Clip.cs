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
    public static partial class ILKernelGenerator
    {
        #region Clip Helpers (Contiguous)

        /// <summary>
        /// SIMD-optimized Clip operation for contiguous arrays (min and max).
        /// Modifies the array in-place: data[i] = Min(Max(data[i], minVal), maxVal)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ClipHelper<T>(T* data, long size, T minVal, T maxVal) where T : unmanaged, IComparable<T>
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
        public static unsafe void ClipMinHelper<T>(T* data, long size, T minVal) where T : unmanaged, IComparable<T>
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
        public static unsafe void ClipMaxHelper<T>(T* data, long size, T maxVal) where T : unmanaged, IComparable<T>
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
        private static unsafe void ClipScalar<T>(T* data, long size, T minVal, T maxVal) where T : unmanaged, IComparable<T>
        {
            // Use specialized implementations for float/double to handle NaN correctly
            if (typeof(T) == typeof(float))
            {
                ClipScalarFloat((float*)data, size, Unsafe.As<T, float>(ref minVal), Unsafe.As<T, float>(ref maxVal));
                return;
            }
            if (typeof(T) == typeof(double))
            {
                ClipScalarDouble((double*)data, size, Unsafe.As<T, double>(ref minVal), Unsafe.As<T, double>(ref maxVal));
                return;
            }

            for (long i = 0; i < size; i++)
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
        private static unsafe void ClipMinScalar<T>(T* data, long size, T minVal) where T : unmanaged, IComparable<T>
        {
            // Use specialized implementations for float/double to handle NaN correctly
            if (typeof(T) == typeof(float))
            {
                ClipMinScalarFloat((float*)data, size, Unsafe.As<T, float>(ref minVal));
                return;
            }
            if (typeof(T) == typeof(double))
            {
                ClipMinScalarDouble((double*)data, size, Unsafe.As<T, double>(ref minVal));
                return;
            }

            for (long i = 0; i < size; i++)
            {
                if (data[i].CompareTo(minVal) < 0)
                    data[i] = minVal;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipMaxScalar<T>(T* data, long size, T maxVal) where T : unmanaged, IComparable<T>
        {
            // Use specialized implementations for float/double to handle NaN correctly
            if (typeof(T) == typeof(float))
            {
                ClipMaxScalarFloat((float*)data, size, Unsafe.As<T, float>(ref maxVal));
                return;
            }
            if (typeof(T) == typeof(double))
            {
                ClipMaxScalarDouble((double*)data, size, Unsafe.As<T, double>(ref maxVal));
                return;
            }

            for (long i = 0; i < size; i++)
            {
                if (data[i].CompareTo(maxVal) > 0)
                    data[i] = maxVal;
            }
        }

        #region Floating-Point Scalar Implementations (NaN-aware)

        // These use Math.Max/Min which properly propagate NaN per IEEE semantics

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipScalarFloat(float* data, long size, float minVal, float maxVal)
        {
            for (long i = 0; i < size; i++)
            {
                // Math.Max/Min propagate NaN: if either operand is NaN, result is NaN
                data[i] = Math.Min(Math.Max(data[i], minVal), maxVal);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipScalarDouble(double* data, long size, double minVal, double maxVal)
        {
            for (long i = 0; i < size; i++)
            {
                data[i] = Math.Min(Math.Max(data[i], minVal), maxVal);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipMinScalarFloat(float* data, long size, float minVal)
        {
            for (long i = 0; i < size; i++)
            {
                data[i] = Math.Max(data[i], minVal);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipMinScalarDouble(double* data, long size, double minVal)
        {
            for (long i = 0; i < size; i++)
            {
                data[i] = Math.Max(data[i], minVal);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipMaxScalarFloat(float* data, long size, float maxVal)
        {
            for (long i = 0; i < size; i++)
            {
                data[i] = Math.Min(data[i], maxVal);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipMaxScalarDouble(double* data, long size, double maxVal)
        {
            for (long i = 0; i < size; i++)
            {
                data[i] = Math.Min(data[i], maxVal);
            }
        }

        #endregion

        #endregion

        #region Vector256 SIMD Implementations

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipSimd256<T>(T* data, long size, T minVal, T maxVal) where T : unmanaged
        {
            int vectorCount = Vector256<T>.Count;
            long vectorEnd = size - vectorCount;
            var minVec = Vector256.Create(minVal);
            var maxVec = Vector256.Create(maxVal);

            long i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector256.Load(data + i);
                vec = Vector256.Max(vec, minVec);
                vec = Vector256.Min(vec, maxVec);
                vec.Store(data + i);
            }

            // Scalar tail - use NaN-aware helpers for float/double
            ClipScalarTail(data + i, size - i, minVal, maxVal);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipMinSimd256<T>(T* data, long size, T minVal) where T : unmanaged
        {
            int vectorCount = Vector256<T>.Count;
            long vectorEnd = size - vectorCount;
            var minVec = Vector256.Create(minVal);

            long i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector256.Load(data + i);
                vec = Vector256.Max(vec, minVec);
                vec.Store(data + i);
            }

            // Scalar tail - use NaN-aware helpers for float/double
            ClipMinScalarTail(data + i, size - i, minVal);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipMaxSimd256<T>(T* data, long size, T maxVal) where T : unmanaged
        {
            int vectorCount = Vector256<T>.Count;
            long vectorEnd = size - vectorCount;
            var maxVec = Vector256.Create(maxVal);

            long i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector256.Load(data + i);
                vec = Vector256.Min(vec, maxVec);
                vec.Store(data + i);
            }

            // Scalar tail - use NaN-aware helpers for float/double
            ClipMaxScalarTail(data + i, size - i, maxVal);
        }

        #endregion

        #region Scalar Tail Helpers (NaN-aware for float/double)

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipScalarTail<T>(T* data, long size, T minVal, T maxVal) where T : unmanaged
        {
            if (size <= 0) return;

            if (typeof(T) == typeof(float))
            {
                var fMin = Unsafe.As<T, float>(ref minVal);
                var fMax = Unsafe.As<T, float>(ref maxVal);
                var fData = (float*)data;
                for (long i = 0; i < size; i++)
                    fData[i] = Math.Min(Math.Max(fData[i], fMin), fMax);
            }
            else if (typeof(T) == typeof(double))
            {
                var dMin = Unsafe.As<T, double>(ref minVal);
                var dMax = Unsafe.As<T, double>(ref maxVal);
                var dData = (double*)data;
                for (long i = 0; i < size; i++)
                    dData[i] = Math.Min(Math.Max(dData[i], dMin), dMax);
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    var val = data[i];
                    if (Comparer<T>.Default.Compare(val, maxVal) > 0)
                        val = maxVal;
                    else if (Comparer<T>.Default.Compare(val, minVal) < 0)
                        val = minVal;
                    data[i] = val;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipMinScalarTail<T>(T* data, long size, T minVal) where T : unmanaged
        {
            if (size <= 0) return;

            if (typeof(T) == typeof(float))
            {
                var fMin = Unsafe.As<T, float>(ref minVal);
                var fData = (float*)data;
                for (long i = 0; i < size; i++)
                    fData[i] = Math.Max(fData[i], fMin);
            }
            else if (typeof(T) == typeof(double))
            {
                var dMin = Unsafe.As<T, double>(ref minVal);
                var dData = (double*)data;
                for (long i = 0; i < size; i++)
                    dData[i] = Math.Max(dData[i], dMin);
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (Comparer<T>.Default.Compare(data[i], minVal) < 0)
                        data[i] = minVal;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipMaxScalarTail<T>(T* data, long size, T maxVal) where T : unmanaged
        {
            if (size <= 0) return;

            if (typeof(T) == typeof(float))
            {
                var fMax = Unsafe.As<T, float>(ref maxVal);
                var fData = (float*)data;
                for (long i = 0; i < size; i++)
                    fData[i] = Math.Min(fData[i], fMax);
            }
            else if (typeof(T) == typeof(double))
            {
                var dMax = Unsafe.As<T, double>(ref maxVal);
                var dData = (double*)data;
                for (long i = 0; i < size; i++)
                    dData[i] = Math.Min(dData[i], dMax);
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (Comparer<T>.Default.Compare(data[i], maxVal) > 0)
                        data[i] = maxVal;
                }
            }
        }

        #endregion

        #region Vector128 SIMD Implementations

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipSimd128<T>(T* data, long size, T minVal, T maxVal) where T : unmanaged
        {
            int vectorCount = Vector128<T>.Count;
            long vectorEnd = size - vectorCount;
            var minVec = Vector128.Create(minVal);
            var maxVec = Vector128.Create(maxVal);

            long i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector128.Load(data + i);
                vec = Vector128.Max(vec, minVec);
                vec = Vector128.Min(vec, maxVec);
                vec.Store(data + i);
            }

            // Scalar tail - use NaN-aware helpers for float/double
            ClipScalarTail(data + i, size - i, minVal, maxVal);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipMinSimd128<T>(T* data, long size, T minVal) where T : unmanaged
        {
            int vectorCount = Vector128<T>.Count;
            long vectorEnd = size - vectorCount;
            var minVec = Vector128.Create(minVal);

            long i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector128.Load(data + i);
                vec = Vector128.Max(vec, minVec);
                vec.Store(data + i);
            }

            // Scalar tail - use NaN-aware helpers for float/double
            ClipMinScalarTail(data + i, size - i, minVal);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipMaxSimd128<T>(T* data, long size, T maxVal) where T : unmanaged
        {
            int vectorCount = Vector128<T>.Count;
            long vectorEnd = size - vectorCount;
            var maxVec = Vector128.Create(maxVal);

            long i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector128.Load(data + i);
                vec = Vector128.Min(vec, maxVec);
                vec.Store(data + i);
            }

            // Scalar tail - use NaN-aware helpers for float/double
            ClipMaxScalarTail(data + i, size - i, maxVal);
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
        public static unsafe void ClipStrided<T>(T* data, long size, T minVal, T maxVal, Shape shape) where T : unmanaged, IComparable<T>
        {
            if (size == 0) return;

            // Special case: if actually contiguous, use the fast path
            if (shape.IsContiguous)
            {
                ClipHelper(data + shape.Offset, size, minVal, maxVal);
                return;
            }

            // Strided iteration using coordinate transformation
            for (long i = 0; i < size; i++)
            {
                long offset = shape.TransformOffset(i);
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
        public static unsafe void ClipMinStrided<T>(T* data, long size, T minVal, Shape shape) where T : unmanaged, IComparable<T>
        {
            if (size == 0) return;

            if (shape.IsContiguous)
            {
                ClipMinHelper(data + shape.Offset, size, minVal);
                return;
            }

            for (long i = 0; i < size; i++)
            {
                long offset = shape.TransformOffset(i);
                if (data[offset].CompareTo(minVal) < 0)
                    data[offset] = minVal;
            }
        }

        /// <summary>
        /// Max-only Clip operation for strided arrays.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ClipMaxStrided<T>(T* data, long size, T maxVal, Shape shape) where T : unmanaged, IComparable<T>
        {
            if (size == 0) return;

            if (shape.IsContiguous)
            {
                ClipMaxHelper(data + shape.Offset, size, maxVal);
                return;
            }

            for (long i = 0; i < size; i++)
            {
                long offset = shape.TransformOffset(i);
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
        public static unsafe void ClipUnified<T>(T* data, long size, T minVal, T maxVal, Shape shape) where T : unmanaged, IComparable<T>
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
        public static unsafe void ClipMinUnified<T>(T* data, long size, T minVal, Shape shape) where T : unmanaged, IComparable<T>
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
        public static unsafe void ClipMaxUnified<T>(T* data, long size, T maxVal, Shape shape) where T : unmanaged, IComparable<T>
        {
            if (shape.IsContiguous)
                ClipMaxHelper(data + shape.Offset, size, maxVal);
            else
                ClipMaxStrided(data, size, maxVal, shape);
        }

        #endregion

        #region Array Bounds Clip (Element-wise min/max from arrays)

        // =============================================================================
        // Array Bounds Clip - when min and/or max are arrays instead of scalars
        // =============================================================================
        //
        // This section handles np.clip(a, min_array, max_array) where min/max are arrays
        // that may be broadcast to match the input shape.
        //
        // Unlike scalar clip which can use SIMD with broadcast vectors, array bounds
        // require element-wise reading from min/max arrays. We still use SIMD where
        // all three arrays are contiguous and aligned.
        //
        // NumPy behavior:
        // - min > max at any position: result = max (per NumPy documentation)
        // - NaN in bounds: result = NaN (IEEE semantics via comparison)
        // - Broadcasting handled by caller (np.broadcast_to)
        //
        // =============================================================================

        /// <summary>
        /// Clip with element-wise array bounds (both min and max arrays).
        /// All three arrays must be broadcast to the same shape by the caller.
        /// For contiguous arrays of SIMD-supported types, uses Vector operations.
        /// </summary>
        /// <remarks>
        /// NumPy clip semantics: result[i] = min(max(a[i], min[i]), max[i])
        /// When min[i] > max[i], result is max[i] (per NumPy behavior).
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ClipArrayBounds<T>(T* output, T* minArr, T* maxArr, long size)
            where T : unmanaged, IComparable<T>
        {
            if (size == 0) return;

            // Try SIMD path for supported types with sufficient size (V512 -> V256 -> V128 -> scalar)
            if (VectorBits >= 512 && size >= 64)
            {
                if (typeof(T) == typeof(float))
                {
                    ClipArrayBoundsSimd512((float*)output, (float*)minArr, (float*)maxArr, size);
                    return;
                }
                if (typeof(T) == typeof(double))
                {
                    ClipArrayBoundsSimd512((double*)output, (double*)minArr, (double*)maxArr, size);
                    return;
                }
                if (typeof(T) == typeof(int))
                {
                    ClipArrayBoundsSimd512((int*)output, (int*)minArr, (int*)maxArr, size);
                    return;
                }
                if (typeof(T) == typeof(long))
                {
                    ClipArrayBoundsSimd512((long*)output, (long*)minArr, (long*)maxArr, size);
                    return;
                }
            }
            else if (VectorBits >= 256 && size >= 32)
            {
                if (typeof(T) == typeof(float))
                {
                    ClipArrayBoundsSimd256((float*)output, (float*)minArr, (float*)maxArr, size);
                    return;
                }
                if (typeof(T) == typeof(double))
                {
                    ClipArrayBoundsSimd256((double*)output, (double*)minArr, (double*)maxArr, size);
                    return;
                }
                if (typeof(T) == typeof(int))
                {
                    ClipArrayBoundsSimd256((int*)output, (int*)minArr, (int*)maxArr, size);
                    return;
                }
                if (typeof(T) == typeof(long))
                {
                    ClipArrayBoundsSimd256((long*)output, (long*)minArr, (long*)maxArr, size);
                    return;
                }
            }
            else if (VectorBits >= 128 && size >= 16)
            {
                if (typeof(T) == typeof(float))
                {
                    ClipArrayBoundsSimd128((float*)output, (float*)minArr, (float*)maxArr, size);
                    return;
                }
                if (typeof(T) == typeof(double))
                {
                    ClipArrayBoundsSimd128((double*)output, (double*)minArr, (double*)maxArr, size);
                    return;
                }
                if (typeof(T) == typeof(int))
                {
                    ClipArrayBoundsSimd128((int*)output, (int*)minArr, (int*)maxArr, size);
                    return;
                }
                if (typeof(T) == typeof(long))
                {
                    ClipArrayBoundsSimd128((long*)output, (long*)minArr, (long*)maxArr, size);
                    return;
                }
            }

            // Scalar fallback for all types
            ClipArrayBoundsScalar(output, minArr, maxArr, size);
        }

        /// <summary>
        /// Clip with element-wise min array bounds only (no max).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ClipArrayMin<T>(T* output, T* minArr, long size)
            where T : unmanaged, IComparable<T>
        {
            if (size == 0) return;

            // Try SIMD path (V512 -> V256 -> V128 -> scalar)
            if (VectorBits >= 512 && size >= 64)
            {
                if (typeof(T) == typeof(float))
                {
                    ClipArrayMinSimd512((float*)output, (float*)minArr, size);
                    return;
                }
                if (typeof(T) == typeof(double))
                {
                    ClipArrayMinSimd512((double*)output, (double*)minArr, size);
                    return;
                }
                if (typeof(T) == typeof(int))
                {
                    ClipArrayMinSimd512((int*)output, (int*)minArr, size);
                    return;
                }
                if (typeof(T) == typeof(long))
                {
                    ClipArrayMinSimd512((long*)output, (long*)minArr, size);
                    return;
                }
            }
            else if (VectorBits >= 256 && size >= 32)
            {
                if (typeof(T) == typeof(float))
                {
                    ClipArrayMinSimd256((float*)output, (float*)minArr, size);
                    return;
                }
                if (typeof(T) == typeof(double))
                {
                    ClipArrayMinSimd256((double*)output, (double*)minArr, size);
                    return;
                }
                if (typeof(T) == typeof(int))
                {
                    ClipArrayMinSimd256((int*)output, (int*)minArr, size);
                    return;
                }
                if (typeof(T) == typeof(long))
                {
                    ClipArrayMinSimd256((long*)output, (long*)minArr, size);
                    return;
                }
            }
            else if (VectorBits >= 128 && size >= 16)
            {
                if (typeof(T) == typeof(float))
                {
                    ClipArrayMinSimd128((float*)output, (float*)minArr, size);
                    return;
                }
                if (typeof(T) == typeof(double))
                {
                    ClipArrayMinSimd128((double*)output, (double*)minArr, size);
                    return;
                }
                if (typeof(T) == typeof(int))
                {
                    ClipArrayMinSimd128((int*)output, (int*)minArr, size);
                    return;
                }
                if (typeof(T) == typeof(long))
                {
                    ClipArrayMinSimd128((long*)output, (long*)minArr, size);
                    return;
                }
            }

            ClipArrayMinScalar(output, minArr, size);
        }

        /// <summary>
        /// Clip with element-wise max array bounds only (no min).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ClipArrayMax<T>(T* output, T* maxArr, long size)
            where T : unmanaged, IComparable<T>
        {
            if (size == 0) return;

            // Try SIMD path (V512 -> V256 -> V128 -> scalar)
            if (VectorBits >= 512 && size >= 64)
            {
                if (typeof(T) == typeof(float))
                {
                    ClipArrayMaxSimd512((float*)output, (float*)maxArr, size);
                    return;
                }
                if (typeof(T) == typeof(double))
                {
                    ClipArrayMaxSimd512((double*)output, (double*)maxArr, size);
                    return;
                }
                if (typeof(T) == typeof(int))
                {
                    ClipArrayMaxSimd512((int*)output, (int*)maxArr, size);
                    return;
                }
                if (typeof(T) == typeof(long))
                {
                    ClipArrayMaxSimd512((long*)output, (long*)maxArr, size);
                    return;
                }
            }
            else if (VectorBits >= 256 && size >= 32)
            {
                if (typeof(T) == typeof(float))
                {
                    ClipArrayMaxSimd256((float*)output, (float*)maxArr, size);
                    return;
                }
                if (typeof(T) == typeof(double))
                {
                    ClipArrayMaxSimd256((double*)output, (double*)maxArr, size);
                    return;
                }
                if (typeof(T) == typeof(int))
                {
                    ClipArrayMaxSimd256((int*)output, (int*)maxArr, size);
                    return;
                }
                if (typeof(T) == typeof(long))
                {
                    ClipArrayMaxSimd256((long*)output, (long*)maxArr, size);
                    return;
                }
            }
            else if (VectorBits >= 128 && size >= 16)
            {
                if (typeof(T) == typeof(float))
                {
                    ClipArrayMaxSimd128((float*)output, (float*)maxArr, size);
                    return;
                }
                if (typeof(T) == typeof(double))
                {
                    ClipArrayMaxSimd128((double*)output, (double*)maxArr, size);
                    return;
                }
                if (typeof(T) == typeof(int))
                {
                    ClipArrayMaxSimd128((int*)output, (int*)maxArr, size);
                    return;
                }
                if (typeof(T) == typeof(long))
                {
                    ClipArrayMaxSimd128((long*)output, (long*)maxArr, size);
                    return;
                }
            }

            ClipArrayMaxScalar(output, maxArr, size);
        }

        #region Array Bounds - Scalar Implementations

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayBoundsScalar<T>(T* output, T* minArr, T* maxArr, long size)
            where T : unmanaged, IComparable<T>
        {
            if (typeof(T) == typeof(float))
            {
                ClipArrayBoundsScalarFloat((float*)output, (float*)minArr, (float*)maxArr, size);
                return;
            }
            if (typeof(T) == typeof(double))
            {
                ClipArrayBoundsScalarDouble((double*)output, (double*)minArr, (double*)maxArr, size);
                return;
            }
            if (typeof(T) == typeof(Half))
            {
                ClipArrayBoundsScalarHalf((Half*)output, (Half*)minArr, (Half*)maxArr, size);
                return;
            }

            for (long i = 0; i < size; i++)
            {
                var val = output[i];
                var minVal = minArr[i];
                var maxVal = maxArr[i];
                // NumPy semantics: min(max(val, minVal), maxVal)
                if (val.CompareTo(minVal) < 0)
                    val = minVal;
                if (val.CompareTo(maxVal) > 0)
                    val = maxVal;
                output[i] = val;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayMinScalar<T>(T* output, T* minArr, long size)
            where T : unmanaged, IComparable<T>
        {
            if (typeof(T) == typeof(float))
            {
                ClipArrayMinScalarFloat((float*)output, (float*)minArr, size);
                return;
            }
            if (typeof(T) == typeof(double))
            {
                ClipArrayMinScalarDouble((double*)output, (double*)minArr, size);
                return;
            }
            if (typeof(T) == typeof(Half))
            {
                ClipArrayMinScalarHalf((Half*)output, (Half*)minArr, size);
                return;
            }

            for (long i = 0; i < size; i++)
            {
                if (output[i].CompareTo(minArr[i]) < 0)
                    output[i] = minArr[i];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayMaxScalar<T>(T* output, T* maxArr, long size)
            where T : unmanaged, IComparable<T>
        {
            if (typeof(T) == typeof(float))
            {
                ClipArrayMaxScalarFloat((float*)output, (float*)maxArr, size);
                return;
            }
            if (typeof(T) == typeof(double))
            {
                ClipArrayMaxScalarDouble((double*)output, (double*)maxArr, size);
                return;
            }
            if (typeof(T) == typeof(Half))
            {
                ClipArrayMaxScalarHalf((Half*)output, (Half*)maxArr, size);
                return;
            }

            for (long i = 0; i < size; i++)
            {
                if (output[i].CompareTo(maxArr[i]) > 0)
                    output[i] = maxArr[i];
            }
        }

        #region Array Bounds - Float/Double Scalar (NaN-aware)

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayBoundsScalarFloat(float* output, float* minArr, float* maxArr, long size)
        {
            for (long i = 0; i < size; i++)
                output[i] = Math.Min(Math.Max(output[i], minArr[i]), maxArr[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayBoundsScalarDouble(double* output, double* minArr, double* maxArr, long size)
        {
            for (long i = 0; i < size; i++)
                output[i] = Math.Min(Math.Max(output[i], minArr[i]), maxArr[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayMinScalarFloat(float* output, float* minArr, long size)
        {
            for (long i = 0; i < size; i++)
                output[i] = Math.Max(output[i], minArr[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayMinScalarDouble(double* output, double* minArr, long size)
        {
            for (long i = 0; i < size; i++)
                output[i] = Math.Max(output[i], minArr[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayMaxScalarFloat(float* output, float* maxArr, long size)
        {
            for (long i = 0; i < size; i++)
                output[i] = Math.Min(output[i], maxArr[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayMaxScalarDouble(double* output, double* maxArr, long size)
        {
            for (long i = 0; i < size; i++)
                output[i] = Math.Min(output[i], maxArr[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayBoundsScalarHalf(Half* output, Half* minArr, Half* maxArr, long size)
        {
            for (long i = 0; i < size; i++)
            {
                var val = output[i];
                var lo = minArr[i];
                var hi = maxArr[i];
                if (Half.IsNaN(val) || Half.IsNaN(lo) || Half.IsNaN(hi)) { output[i] = Half.NaN; continue; }
                if (val < lo) val = lo;
                if (val > hi) val = hi;
                output[i] = val;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayMinScalarHalf(Half* output, Half* minArr, long size)
        {
            for (long i = 0; i < size; i++)
            {
                var val = output[i];
                var lo = minArr[i];
                if (Half.IsNaN(val) || Half.IsNaN(lo)) { output[i] = Half.NaN; continue; }
                if (val < lo) output[i] = lo;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayMaxScalarHalf(Half* output, Half* maxArr, long size)
        {
            for (long i = 0; i < size; i++)
            {
                var val = output[i];
                var hi = maxArr[i];
                if (Half.IsNaN(val) || Half.IsNaN(hi)) { output[i] = Half.NaN; continue; }
                if (val > hi) output[i] = hi;
            }
        }

        #endregion

        #region Array Bounds - Scalar Tail Helpers (NaN-aware)

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayBoundsScalarTail<T>(T* output, T* minArr, T* maxArr, long size)
            where T : unmanaged
        {
            if (size <= 0) return;

            if (typeof(T) == typeof(float))
            {
                ClipArrayBoundsScalarFloat((float*)output, (float*)minArr, (float*)maxArr, size);
            }
            else if (typeof(T) == typeof(double))
            {
                ClipArrayBoundsScalarDouble((double*)output, (double*)minArr, (double*)maxArr, size);
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    var val = output[i];
                    var minVal = minArr[i];
                    var maxVal = maxArr[i];
                    if (Comparer<T>.Default.Compare(val, minVal) < 0)
                        val = minVal;
                    if (Comparer<T>.Default.Compare(val, maxVal) > 0)
                        val = maxVal;
                    output[i] = val;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayMinScalarTail<T>(T* output, T* minArr, long size)
            where T : unmanaged
        {
            if (size <= 0) return;

            if (typeof(T) == typeof(float))
            {
                ClipArrayMinScalarFloat((float*)output, (float*)minArr, size);
            }
            else if (typeof(T) == typeof(double))
            {
                ClipArrayMinScalarDouble((double*)output, (double*)minArr, size);
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (Comparer<T>.Default.Compare(output[i], minArr[i]) < 0)
                        output[i] = minArr[i];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayMaxScalarTail<T>(T* output, T* maxArr, long size)
            where T : unmanaged
        {
            if (size <= 0) return;

            if (typeof(T) == typeof(float))
            {
                ClipArrayMaxScalarFloat((float*)output, (float*)maxArr, size);
            }
            else if (typeof(T) == typeof(double))
            {
                ClipArrayMaxScalarDouble((double*)output, (double*)maxArr, size);
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (Comparer<T>.Default.Compare(output[i], maxArr[i]) > 0)
                        output[i] = maxArr[i];
                }
            }
        }

        #endregion

        #endregion

        #region Array Bounds - Vector512 SIMD Implementations

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayBoundsSimd512<T>(T* output, T* minArr, T* maxArr, long size)
            where T : unmanaged
        {
            int vectorCount = Vector512<T>.Count;
            long vectorEnd = size - vectorCount;

            long i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector512.Load(output + i);
                var minVec = Vector512.Load(minArr + i);
                var maxVec = Vector512.Load(maxArr + i);
                vec = Vector512.Max(vec, minVec);
                vec = Vector512.Min(vec, maxVec);
                vec.Store(output + i);
            }

            // Scalar tail - use NaN-aware helper
            ClipArrayBoundsScalarTail(output + i, minArr + i, maxArr + i, size - i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayMinSimd512<T>(T* output, T* minArr, long size)
            where T : unmanaged
        {
            int vectorCount = Vector512<T>.Count;
            long vectorEnd = size - vectorCount;

            long i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector512.Load(output + i);
                var minVec = Vector512.Load(minArr + i);
                vec = Vector512.Max(vec, minVec);
                vec.Store(output + i);
            }

            // Scalar tail - use NaN-aware helper
            ClipArrayMinScalarTail(output + i, minArr + i, size - i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayMaxSimd512<T>(T* output, T* maxArr, long size)
            where T : unmanaged
        {
            int vectorCount = Vector512<T>.Count;
            long vectorEnd = size - vectorCount;

            long i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector512.Load(output + i);
                var maxVec = Vector512.Load(maxArr + i);
                vec = Vector512.Min(vec, maxVec);
                vec.Store(output + i);
            }

            // Scalar tail - use NaN-aware helper
            ClipArrayMaxScalarTail(output + i, maxArr + i, size - i);
        }

        #endregion

        #region Array Bounds - Vector256 SIMD Implementations

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayBoundsSimd256<T>(T* output, T* minArr, T* maxArr, long size)
            where T : unmanaged
        {
            int vectorCount = Vector256<T>.Count;
            long vectorEnd = size - vectorCount;

            long i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector256.Load(output + i);
                var minVec = Vector256.Load(minArr + i);
                var maxVec = Vector256.Load(maxArr + i);
                vec = Vector256.Max(vec, minVec);
                vec = Vector256.Min(vec, maxVec);
                vec.Store(output + i);
            }

            // Scalar tail - use NaN-aware helper
            ClipArrayBoundsScalarTail(output + i, minArr + i, maxArr + i, size - i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayMinSimd256<T>(T* output, T* minArr, long size)
            where T : unmanaged
        {
            int vectorCount = Vector256<T>.Count;
            long vectorEnd = size - vectorCount;

            long i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector256.Load(output + i);
                var minVec = Vector256.Load(minArr + i);
                vec = Vector256.Max(vec, minVec);
                vec.Store(output + i);
            }

            // Scalar tail - use NaN-aware helper
            ClipArrayMinScalarTail(output + i, minArr + i, size - i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayMaxSimd256<T>(T* output, T* maxArr, long size)
            where T : unmanaged
        {
            int vectorCount = Vector256<T>.Count;
            long vectorEnd = size - vectorCount;

            long i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector256.Load(output + i);
                var maxVec = Vector256.Load(maxArr + i);
                vec = Vector256.Min(vec, maxVec);
                vec.Store(output + i);
            }

            // Scalar tail - use NaN-aware helper
            ClipArrayMaxScalarTail(output + i, maxArr + i, size - i);
        }

        #endregion

        #region Array Bounds - Vector128 SIMD Implementations

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayBoundsSimd128<T>(T* output, T* minArr, T* maxArr, long size)
            where T : unmanaged
        {
            int vectorCount = Vector128<T>.Count;
            long vectorEnd = size - vectorCount;

            long i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector128.Load(output + i);
                var minVec = Vector128.Load(minArr + i);
                var maxVec = Vector128.Load(maxArr + i);
                vec = Vector128.Max(vec, minVec);
                vec = Vector128.Min(vec, maxVec);
                vec.Store(output + i);
            }

            // Scalar tail - use NaN-aware helper
            ClipArrayBoundsScalarTail(output + i, minArr + i, maxArr + i, size - i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayMinSimd128<T>(T* output, T* minArr, long size)
            where T : unmanaged
        {
            int vectorCount = Vector128<T>.Count;
            long vectorEnd = size - vectorCount;

            long i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector128.Load(output + i);
                var minVec = Vector128.Load(minArr + i);
                vec = Vector128.Max(vec, minVec);
                vec.Store(output + i);
            }

            // Scalar tail - use NaN-aware helper
            ClipArrayMinScalarTail(output + i, minArr + i, size - i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ClipArrayMaxSimd128<T>(T* output, T* maxArr, long size)
            where T : unmanaged
        {
            int vectorCount = Vector128<T>.Count;
            long vectorEnd = size - vectorCount;

            long i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector128.Load(output + i);
                var maxVec = Vector128.Load(maxArr + i);
                vec = Vector128.Min(vec, maxVec);
                vec.Store(output + i);
            }

            // Scalar tail - use NaN-aware helper
            ClipArrayMaxScalarTail(output + i, maxArr + i, size - i);
        }

        #endregion

        #endregion
    }
}
