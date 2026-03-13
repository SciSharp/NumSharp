using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Masking.cs - NonZero and Boolean Masking Operations
// =============================================================================
//
// RESPONSIBILITY:
//   - NonZero SIMD helpers for np.nonzero
//   - Boolean masking SIMD helpers for fancy indexing
//   - CountTrueSimdHelper, CopyMaskedElementsHelper
//   - NaN-aware helpers (NanSum, NanProd, NanMin, NanMax)
//
// NOTE: These are selection/masking operations, NOT reductions.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public sealed partial class ILKernelGenerator
    {
        #region NonZero SIMD Helpers

        /// <summary>
        /// SIMD helper for NonZero operation.
        /// Finds all indices where elements are non-zero.
        /// </summary>
        /// <param name="src">Source array pointer</param>
        /// <param name="size">Number of elements</param>
        /// <param name="indices">Output list to populate with non-zero indices</param>
        internal static unsafe void NonZeroSimdHelper<T>(T* src, int size, System.Collections.Generic.List<int> indices)
            where T : unmanaged
        {
            if (size == 0)
                return;

            if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && size >= Vector256<T>.Count)
            {
                int vectorCount = Vector256<T>.Count;
                int vectorEnd = size - vectorCount;
                var zero = Vector256<T>.Zero;
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var mask = Vector256.Equals(vec, zero);
                    uint bits = Vector256.ExtractMostSignificantBits(mask);

                    // Invert: we want non-zero elements
                    uint nonZeroBits = ~bits & ((1u << vectorCount) - 1);

                    // Extract indices where bits are set
                    while (nonZeroBits != 0)
                    {
                        int bitPos = System.Numerics.BitOperations.TrailingZeroCount(nonZeroBits);
                        indices.Add(i + bitPos);
                        nonZeroBits &= nonZeroBits - 1; // Clear lowest bit
                    }
                }

                // Scalar tail
                for (; i < size; i++)
                {
                    if (!System.Collections.Generic.EqualityComparer<T>.Default.Equals(src[i], default))
                        indices.Add(i);
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && size >= Vector128<T>.Count)
            {
                int vectorCount = Vector128<T>.Count;
                int vectorEnd = size - vectorCount;
                var zero = Vector128<T>.Zero;
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var mask = Vector128.Equals(vec, zero);
                    uint bits = Vector128.ExtractMostSignificantBits(mask);

                    uint nonZeroBits = ~bits & ((1u << vectorCount) - 1);

                    while (nonZeroBits != 0)
                    {
                        int bitPos = System.Numerics.BitOperations.TrailingZeroCount(nonZeroBits);
                        indices.Add(i + bitPos);
                        nonZeroBits &= nonZeroBits - 1;
                    }
                }

                for (; i < size; i++)
                {
                    if (!System.Collections.Generic.EqualityComparer<T>.Default.Equals(src[i], default))
                        indices.Add(i);
                }
            }
            else
            {
                // Scalar fallback
                for (int i = 0; i < size; i++)
                {
                    if (!System.Collections.Generic.EqualityComparer<T>.Default.Equals(src[i], default))
                        indices.Add(i);
                }
            }
        }

        /// <summary>
        /// Convert flat indices to per-dimension coordinate arrays.
        /// </summary>
        /// <param name="flatIndices">List of flat (linear) indices</param>
        /// <param name="shape">Shape of the array</param>
        /// <returns>Array of NDArray&lt;int&gt;, one per dimension</returns>
        internal static unsafe NumSharp.Generic.NDArray<int>[] ConvertFlatIndicesToCoordinates(
            System.Collections.Generic.List<int> flatIndices, int[] shape)
        {
            int ndim = shape.Length;
            int len = flatIndices.Count;

            // Create result arrays
            var result = new NumSharp.Generic.NDArray<int>[ndim];
            for (int d = 0; d < ndim; d++)
                result[d] = new NumSharp.Generic.NDArray<int>(len);

            // Get addresses for direct writing
            var addresses = new int*[ndim];
            for (int d = 0; d < ndim; d++)
                addresses[d] = (int*)result[d].Address;

            // Pre-compute strides for index conversion
            var strides = new int[ndim];
            strides[ndim - 1] = 1;
            for (int d = ndim - 2; d >= 0; d--)
                strides[d] = strides[d + 1] * shape[d + 1];

            // Convert each flat index to coordinates
            for (int i = 0; i < len; i++)
            {
                int flatIdx = flatIndices[i];
                for (int d = 0; d < ndim; d++)
                {
                    addresses[d][i] = flatIdx / strides[d];
                    flatIdx %= strides[d];
                }
            }

            return result;
        }

        /// <summary>
        /// Find non-zero elements in a strided (non-contiguous) array.
        /// Uses coordinate-based iteration to handle arbitrary strides (transposed, sliced, etc.).
        /// Returns per-dimension index arrays matching NumPy's nonzero() output.
        /// </summary>
        /// <typeparam name="T">Element type</typeparam>
        /// <param name="data">Pointer to array data (base address)</param>
        /// <param name="shape">Array dimensions</param>
        /// <param name="strides">Array strides (in elements, not bytes)</param>
        /// <param name="offset">Base offset into storage</param>
        /// <returns>Array of NDArray&lt;int&gt;, one per dimension</returns>
        internal static unsafe NumSharp.Generic.NDArray<int>[] FindNonZeroStridedHelper<T>(
            T* data, int[] shape, int[] strides, int offset) where T : unmanaged
        {
            int ndim = shape.Length;

            // Handle empty array
            int size = 1;
            for (int d = 0; d < ndim; d++)
                size *= shape[d];

            if (size == 0)
            {
                var emptyResult = new NumSharp.Generic.NDArray<int>[ndim];
                for (int d = 0; d < ndim; d++)
                    emptyResult[d] = new NumSharp.Generic.NDArray<int>(0);
                return emptyResult;
            }

            // Collect coordinates of non-zero elements
            // Pre-allocate with estimated capacity (assume ~25% non-zero for efficiency)
            var nonzeroCoords = new System.Collections.Generic.List<int[]>(Math.Max(16, size / 4));

            // Initialize coordinate array
            var coords = new int[ndim];

            // Iterate through all elements using coordinate-based iteration
            // This handles arbitrary strides including negative strides
            while (true)
            {
                // Calculate offset for current coordinates: offset + sum(coords[i] * strides[i])
                int elemOffset = offset;
                for (int d = 0; d < ndim; d++)
                    elemOffset += coords[d] * strides[d];

                // Check if element is non-zero
                if (!System.Collections.Generic.EqualityComparer<T>.Default.Equals(data[elemOffset], default))
                {
                    // Clone coordinates and add to result
                    var coordsCopy = new int[ndim];
                    Array.Copy(coords, coordsCopy, ndim);
                    nonzeroCoords.Add(coordsCopy);
                }

                // Increment coordinates (rightmost dimension first, like C-order iteration)
                int dim = ndim - 1;
                while (dim >= 0)
                {
                    coords[dim]++;
                    if (coords[dim] < shape[dim])
                        break;
                    coords[dim] = 0;
                    dim--;
                }

                // If we've wrapped past the first dimension, we're done
                if (dim < 0)
                    break;
            }

            // Convert collected coordinates to per-dimension arrays
            int len = nonzeroCoords.Count;
            var result = new NumSharp.Generic.NDArray<int>[ndim];
            for (int d = 0; d < ndim; d++)
                result[d] = new NumSharp.Generic.NDArray<int>(len);

            // Get addresses for direct writing
            var addresses = new int*[ndim];
            for (int d = 0; d < ndim; d++)
                addresses[d] = (int*)result[d].Address;

            // Extract coordinates into per-dimension arrays
            for (int i = 0; i < len; i++)
            {
                var coord = nonzeroCoords[i];
                for (int d = 0; d < ndim; d++)
                    addresses[d][i] = coord[d];
            }

            return result;
        }

        #endregion

        #region Boolean Masking SIMD Helpers

        /// <summary>
        /// SIMD helper to count true values in a boolean array.
        /// </summary>
        internal static unsafe int CountTrueSimdHelper(bool* mask, int size)
        {
            if (size == 0)
                return 0;

            int count = 0;

            if (Vector256.IsHardwareAccelerated && Vector256<byte>.IsSupported && size >= Vector256<byte>.Count)
            {
                int vectorCount = Vector256<byte>.Count;
                int vectorEnd = size - vectorCount;
                var zero = Vector256<byte>.Zero;
                int i = 0;

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
                int vectorEnd = size - vectorCount;
                var zero = Vector128<byte>.Zero;
                int i = 0;

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
                for (int i = 0; i < size; i++)
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
        internal static unsafe int CopyMaskedElementsHelper<T>(T* src, bool* mask, T* dst, int size)
            where T : unmanaged
        {
            int dstIdx = 0;

            // For masking, we can't easily vectorize the gather/scatter
            // But we can vectorize the mask scanning to find true indices faster
            if (Vector256.IsHardwareAccelerated && Vector256<byte>.IsSupported && size >= Vector256<byte>.Count)
            {
                int vectorCount = Vector256<byte>.Count;
                int vectorEnd = size - vectorCount;
                var zero = Vector256<byte>.Zero;
                int i = 0;

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
                int vectorEnd = size - vectorCount;
                var zero = Vector128<byte>.Zero;
                int i = 0;

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
                for (int i = 0; i < size; i++)
                {
                    if (mask[i])
                        dst[dstIdx++] = src[i];
                }
            }

            return dstIdx;
        }

        /// <summary>
        /// SIMD helper for computing variance of a contiguous array.
        /// Uses two-pass algorithm: compute mean, then sum of squared differences.
        /// </summary>
        /// <typeparam name="T">Element type (float or double)</typeparam>
        /// <param name="src">Pointer to contiguous data</param>
        /// <param name="size">Number of elements</param>
        /// <param name="ddof">Delta degrees of freedom (0 for population variance, 1 for sample variance)</param>
        /// <returns>The variance as double</returns>
        internal static unsafe double VarSimdHelper<T>(T* src, int size, int ddof = 0)
            where T : unmanaged
        {
            if (size == 0)
                return double.NaN;

            if (size <= ddof)
                return double.NaN; // Division by zero or negative

            // Pass 1: Compute mean
            double sum = 0;

            if (typeof(T) == typeof(double))
            {
                double* p = (double*)(void*)src;

                if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
                {
                    int vectorCount = Vector256<double>.Count;
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector256<double>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        sumVec = Vector256.Add(sumVec, Vector256.Load(p + i));
                    }

                    // Horizontal sum
                    sum = Vector256.Sum(sumVec);

                    // Scalar tail
                    for (; i < size; i++)
                        sum += p[i];
                }
                else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
                {
                    int vectorCount = Vector128<double>.Count;
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector128<double>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        sumVec = Vector128.Add(sumVec, Vector128.Load(p + i));
                    }

                    sum = Vector128.Sum(sumVec);

                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    for (int i = 0; i < size; i++)
                        sum += p[i];
                }

                double mean = sum / size;

                // Pass 2: Sum of squared differences
                double sqDiffSum = 0;

                if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
                {
                    int vectorCount = Vector256<double>.Count;
                    int vectorEnd = size - vectorCount;
                    var meanVec = Vector256.Create(mean);
                    var sqDiffVec = Vector256<double>.Zero;
                    int i = 0;

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
                else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
                {
                    int vectorCount = Vector128<double>.Count;
                    int vectorEnd = size - vectorCount;
                    var meanVec = Vector128.Create(mean);
                    var sqDiffVec = Vector128<double>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector128.Load(p + i);
                        var diff = Vector128.Subtract(vec, meanVec);
                        sqDiffVec = Vector128.Add(sqDiffVec, Vector128.Multiply(diff, diff));
                    }

                    sqDiffSum = Vector128.Sum(sqDiffVec);

                    for (; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
                else
                {
                    for (int i = 0; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }

                return sqDiffSum / (size - ddof);
            }
            else if (typeof(T) == typeof(float))
            {
                float* p = (float*)(void*)src;

                if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
                {
                    int vectorCount = Vector256<float>.Count;
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector256<float>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        sumVec = Vector256.Add(sumVec, Vector256.Load(p + i));
                    }

                    sum = Vector256.Sum(sumVec);

                    for (; i < size; i++)
                        sum += p[i];
                }
                else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
                {
                    int vectorCount = Vector128<float>.Count;
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector128<float>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        sumVec = Vector128.Add(sumVec, Vector128.Load(p + i));
                    }

                    sum = Vector128.Sum(sumVec);

                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    for (int i = 0; i < size; i++)
                        sum += p[i];
                }

                double mean = sum / size;

                // Pass 2: Sum of squared differences (compute in double for precision)
                double sqDiffSum = 0;

                if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
                {
                    int vectorCount = Vector256<float>.Count;
                    int vectorEnd = size - vectorCount;
                    var meanVec = Vector256.Create((float)mean);
                    var sqDiffVec = Vector256<float>.Zero;
                    int i = 0;

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
                else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
                {
                    int vectorCount = Vector128<float>.Count;
                    int vectorEnd = size - vectorCount;
                    var meanVec = Vector128.Create((float)mean);
                    var sqDiffVec = Vector128<float>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector128.Load(p + i);
                        var diff = Vector128.Subtract(vec, meanVec);
                        sqDiffVec = Vector128.Add(sqDiffVec, Vector128.Multiply(diff, diff));
                    }

                    sqDiffSum = Vector128.Sum(sqDiffVec);

                    for (; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
                else
                {
                    for (int i = 0; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }

                return sqDiffSum / (size - ddof);
            }
            else
            {
                // For integer types, convert to double and compute
                double doubleSum = 0;
                for (int i = 0; i < size; i++)
                {
                    doubleSum += Convert.ToDouble(src[i]);
                }
                double mean = doubleSum / size;

                double sqDiffSum = 0;
                for (int i = 0; i < size; i++)
                {
                    double diff = Convert.ToDouble(src[i]) - mean;
                    sqDiffSum += diff * diff;
                }

                return sqDiffSum / (size - ddof);
            }
        }

        /// <summary>
        /// SIMD helper for computing standard deviation of a contiguous array.
        /// Returns sqrt(variance).
        /// </summary>
        /// <typeparam name="T">Element type (float or double)</typeparam>
        /// <param name="src">Pointer to contiguous data</param>
        /// <param name="size">Number of elements</param>
        /// <param name="ddof">Delta degrees of freedom (0 for population std, 1 for sample std)</param>
        /// <returns>The standard deviation as double</returns>
        internal static unsafe double StdSimdHelper<T>(T* src, int size, int ddof = 0)
            where T : unmanaged
        {
            double variance = VarSimdHelper(src, size, ddof);
            return Math.Sqrt(variance);
        }

        /// <summary>
        /// SIMD helper for NaN-aware sum of a contiguous array.
        /// NaN values are treated as 0 (ignored in the sum).
        /// </summary>
        /// <param name="src">Pointer to contiguous float data</param>
        /// <param name="size">Number of elements</param>
        /// <returns>Sum of non-NaN elements</returns>
        internal static unsafe float NanSumSimdHelperFloat(float* src, int size)
        {
            if (size == 0)
                return 0f;

            float sum = 0f;

            if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
            {
                int vectorCount = Vector256<float>.Count;
                int vectorEnd = size - vectorCount;
                var sumVec = Vector256<float>.Zero;
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    // Create mask where NaN becomes 0, valid values stay
                    // NaN comparison: x != x is true for NaN
                    var nanMask = Vector256.Equals(vec, vec); // true for non-NaN, false for NaN
                    var cleaned = Vector256.BitwiseAnd(vec, nanMask.AsSingle());
                    sumVec = Vector256.Add(sumVec, cleaned);
                }

                sum = Vector256.Sum(sumVec);

                // Scalar tail
                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        sum += src[i];
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
            {
                int vectorCount = Vector128<float>.Count;
                int vectorEnd = size - vectorCount;
                var sumVec = Vector128<float>.Zero;
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.BitwiseAnd(vec, nanMask.AsSingle());
                    sumVec = Vector128.Add(sumVec, cleaned);
                }

                sum = Vector128.Sum(sumVec);

                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        sum += src[i];
                }
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        sum += src[i];
                }
            }

            return sum;
        }

        /// <summary>
        /// SIMD helper for NaN-aware sum of a contiguous double array.
        /// NaN values are treated as 0 (ignored in the sum).
        /// </summary>
        internal static unsafe double NanSumSimdHelperDouble(double* src, int size)
        {
            if (size == 0)
                return 0.0;

            double sum = 0.0;

            if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
            {
                int vectorCount = Vector256<double>.Count;
                int vectorEnd = size - vectorCount;
                var sumVec = Vector256<double>.Zero;
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var nanMask = Vector256.Equals(vec, vec);
                    var cleaned = Vector256.BitwiseAnd(vec, nanMask.AsDouble());
                    sumVec = Vector256.Add(sumVec, cleaned);
                }

                sum = Vector256.Sum(sumVec);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        sum += src[i];
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
            {
                int vectorCount = Vector128<double>.Count;
                int vectorEnd = size - vectorCount;
                var sumVec = Vector128<double>.Zero;
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.BitwiseAnd(vec, nanMask.AsDouble());
                    sumVec = Vector128.Add(sumVec, cleaned);
                }

                sum = Vector128.Sum(sumVec);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        sum += src[i];
                }
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        sum += src[i];
                }
            }

            return sum;
        }

        /// <summary>
        /// SIMD helper for NaN-aware product of a contiguous float array.
        /// NaN values are treated as 1 (ignored in the product).
        /// </summary>
        internal static unsafe float NanProdSimdHelperFloat(float* src, int size)
        {
            if (size == 0)
                return 1f;

            float prod = 1f;

            if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
            {
                int vectorCount = Vector256<float>.Count;
                int vectorEnd = size - vectorCount;
                var prodVec = Vector256.Create(1f);
                var oneVec = Vector256.Create(1f);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    // Replace NaN with 1: if NaN, use 1; otherwise use original
                    var nanMask = Vector256.Equals(vec, vec); // true for non-NaN
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, oneVec);
                    prodVec = Vector256.Multiply(prodVec, cleaned);
                }

                // Horizontal product (no built-in, do manually)
                prod = prodVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                    prod *= prodVec.GetElement(j);

                // Scalar tail
                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        prod *= src[i];
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
            {
                int vectorCount = Vector128<float>.Count;
                int vectorEnd = size - vectorCount;
                var prodVec = Vector128.Create(1f);
                var oneVec = Vector128.Create(1f);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, oneVec);
                    prodVec = Vector128.Multiply(prodVec, cleaned);
                }

                prod = prodVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                    prod *= prodVec.GetElement(j);

                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        prod *= src[i];
                }
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        prod *= src[i];
                }
            }

            return prod;
        }

        /// <summary>
        /// SIMD helper for NaN-aware product of a contiguous double array.
        /// NaN values are treated as 1 (ignored in the product).
        /// </summary>
        internal static unsafe double NanProdSimdHelperDouble(double* src, int size)
        {
            if (size == 0)
                return 1.0;

            double prod = 1.0;

            if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
            {
                int vectorCount = Vector256<double>.Count;
                int vectorEnd = size - vectorCount;
                var prodVec = Vector256.Create(1.0);
                var oneVec = Vector256.Create(1.0);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var nanMask = Vector256.Equals(vec, vec);
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, oneVec);
                    prodVec = Vector256.Multiply(prodVec, cleaned);
                }

                prod = prodVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                    prod *= prodVec.GetElement(j);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        prod *= src[i];
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
            {
                int vectorCount = Vector128<double>.Count;
                int vectorEnd = size - vectorCount;
                var prodVec = Vector128.Create(1.0);
                var oneVec = Vector128.Create(1.0);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, oneVec);
                    prodVec = Vector128.Multiply(prodVec, cleaned);
                }

                prod = prodVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                    prod *= prodVec.GetElement(j);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        prod *= src[i];
                }
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        prod *= src[i];
                }
            }

            return prod;
        }

        /// <summary>
        /// SIMD helper for NaN-aware minimum of a contiguous float array.
        /// NaN values are ignored; returns NaN if all values are NaN.
        /// </summary>
        internal static unsafe float NanMinSimdHelperFloat(float* src, int size)
        {
            if (size == 0)
                return float.NaN;

            float minVal = float.PositiveInfinity;
            bool foundNonNaN = false;

            if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
            {
                int vectorCount = Vector256<float>.Count;
                int vectorEnd = size - vectorCount;
                var minVec = Vector256.Create(float.PositiveInfinity);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    // Replace NaN with +Inf so they don't affect minimum
                    var nanMask = Vector256.Equals(vec, vec); // true for non-NaN
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, Vector256.Create(float.PositiveInfinity));
                    minVec = Vector256.Min(minVec, cleaned);
                }

                // Horizontal min
                minVal = minVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    float elem = minVec.GetElement(j);
                    if (elem < minVal)
                        minVal = elem;
                }

                // Scalar tail
                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]) && src[i] < minVal)
                        minVal = src[i];
                }

                foundNonNaN = !float.IsPositiveInfinity(minVal);
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
            {
                int vectorCount = Vector128<float>.Count;
                int vectorEnd = size - vectorCount;
                var minVec = Vector128.Create(float.PositiveInfinity);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, Vector128.Create(float.PositiveInfinity));
                    minVec = Vector128.Min(minVec, cleaned);
                }

                minVal = minVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    float elem = minVec.GetElement(j);
                    if (elem < minVal)
                        minVal = elem;
                }

                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]) && src[i] < minVal)
                        minVal = src[i];
                }

                foundNonNaN = !float.IsPositiveInfinity(minVal);
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                    {
                        if (src[i] < minVal)
                            minVal = src[i];
                        foundNonNaN = true;
                    }
                }
            }

            return foundNonNaN ? minVal : float.NaN;
        }

        /// <summary>
        /// SIMD helper for NaN-aware minimum of a contiguous double array.
        /// NaN values are ignored; returns NaN if all values are NaN.
        /// </summary>
        internal static unsafe double NanMinSimdHelperDouble(double* src, int size)
        {
            if (size == 0)
                return double.NaN;

            double minVal = double.PositiveInfinity;
            bool foundNonNaN = false;

            if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
            {
                int vectorCount = Vector256<double>.Count;
                int vectorEnd = size - vectorCount;
                var minVec = Vector256.Create(double.PositiveInfinity);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var nanMask = Vector256.Equals(vec, vec);
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, Vector256.Create(double.PositiveInfinity));
                    minVec = Vector256.Min(minVec, cleaned);
                }

                minVal = minVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    double elem = minVec.GetElement(j);
                    if (elem < minVal)
                        minVal = elem;
                }

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]) && src[i] < minVal)
                        minVal = src[i];
                }

                foundNonNaN = !double.IsPositiveInfinity(minVal);
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
            {
                int vectorCount = Vector128<double>.Count;
                int vectorEnd = size - vectorCount;
                var minVec = Vector128.Create(double.PositiveInfinity);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, Vector128.Create(double.PositiveInfinity));
                    minVec = Vector128.Min(minVec, cleaned);
                }

                minVal = minVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    double elem = minVec.GetElement(j);
                    if (elem < minVal)
                        minVal = elem;
                }

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]) && src[i] < minVal)
                        minVal = src[i];
                }

                foundNonNaN = !double.IsPositiveInfinity(minVal);
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                    {
                        if (src[i] < minVal)
                            minVal = src[i];
                        foundNonNaN = true;
                    }
                }
            }

            return foundNonNaN ? minVal : double.NaN;
        }

        /// <summary>
        /// SIMD helper for NaN-aware maximum of a contiguous float array.
        /// NaN values are ignored; returns NaN if all values are NaN.
        /// </summary>
        internal static unsafe float NanMaxSimdHelperFloat(float* src, int size)
        {
            if (size == 0)
                return float.NaN;

            float maxVal = float.NegativeInfinity;
            bool foundNonNaN = false;

            if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
            {
                int vectorCount = Vector256<float>.Count;
                int vectorEnd = size - vectorCount;
                var maxVec = Vector256.Create(float.NegativeInfinity);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    // Replace NaN with -Inf so they don't affect maximum
                    var nanMask = Vector256.Equals(vec, vec);
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, Vector256.Create(float.NegativeInfinity));
                    maxVec = Vector256.Max(maxVec, cleaned);
                }

                // Horizontal max
                maxVal = maxVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    float elem = maxVec.GetElement(j);
                    if (elem > maxVal)
                        maxVal = elem;
                }

                // Scalar tail
                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]) && src[i] > maxVal)
                        maxVal = src[i];
                }

                foundNonNaN = !float.IsNegativeInfinity(maxVal);
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
            {
                int vectorCount = Vector128<float>.Count;
                int vectorEnd = size - vectorCount;
                var maxVec = Vector128.Create(float.NegativeInfinity);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, Vector128.Create(float.NegativeInfinity));
                    maxVec = Vector128.Max(maxVec, cleaned);
                }

                maxVal = maxVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    float elem = maxVec.GetElement(j);
                    if (elem > maxVal)
                        maxVal = elem;
                }

                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]) && src[i] > maxVal)
                        maxVal = src[i];
                }

                foundNonNaN = !float.IsNegativeInfinity(maxVal);
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                    {
                        if (src[i] > maxVal)
                            maxVal = src[i];
                        foundNonNaN = true;
                    }
                }
            }

            return foundNonNaN ? maxVal : float.NaN;
        }

        /// <summary>
        /// SIMD helper for NaN-aware maximum of a contiguous double array.
        /// NaN values are ignored; returns NaN if all values are NaN.
        /// </summary>
        internal static unsafe double NanMaxSimdHelperDouble(double* src, int size)
        {
            if (size == 0)
                return double.NaN;

            double maxVal = double.NegativeInfinity;
            bool foundNonNaN = false;

            if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
            {
                int vectorCount = Vector256<double>.Count;
                int vectorEnd = size - vectorCount;
                var maxVec = Vector256.Create(double.NegativeInfinity);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var nanMask = Vector256.Equals(vec, vec);
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, Vector256.Create(double.NegativeInfinity));
                    maxVec = Vector256.Max(maxVec, cleaned);
                }

                maxVal = maxVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    double elem = maxVec.GetElement(j);
                    if (elem > maxVal)
                        maxVal = elem;
                }

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]) && src[i] > maxVal)
                        maxVal = src[i];
                }

                foundNonNaN = !double.IsNegativeInfinity(maxVal);
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
            {
                int vectorCount = Vector128<double>.Count;
                int vectorEnd = size - vectorCount;
                var maxVec = Vector128.Create(double.NegativeInfinity);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, Vector128.Create(double.NegativeInfinity));
                    maxVec = Vector128.Max(maxVec, cleaned);
                }

                maxVal = maxVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    double elem = maxVec.GetElement(j);
                    if (elem > maxVal)
                        maxVal = elem;
                }

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]) && src[i] > maxVal)
                        maxVal = src[i];
                }

                foundNonNaN = !double.IsNegativeInfinity(maxVal);
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                    {
                        if (src[i] > maxVal)
                            maxVal = src[i];
                        foundNonNaN = true;
                    }
                }
            }

            return foundNonNaN ? maxVal : double.NaN;
        }

        #endregion
    }
}
