using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Masking.cs - NonZero SIMD Helpers
// =============================================================================
//
// RESPONSIBILITY:
//   - NonZeroSimdHelper<T> - finds indices of non-zero elements
//   - ConvertFlatIndicesToCoordinates - flat indices to per-dimension arrays
//   - FindNonZeroStridedHelper<T> - strided array support
//
// RELATED FILES:
//   - ILKernelGenerator.Masking.Boolean.cs - CountTrue, CopyMasked
//   - ILKernelGenerator.Masking.VarStd.cs - Var/Std SIMD helpers
//   - ILKernelGenerator.Masking.NaN.cs - NaN-aware helpers
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        #region NonZero SIMD Helpers

        /// <summary>
        /// SIMD helper for NonZero operation.
        /// Finds all indices where elements are non-zero.
        /// </summary>
        /// <param name="src">Source array pointer</param>
        /// <param name="size">Number of elements</param>
        /// <param name="indices">Output list to populate with non-zero indices</param>
        internal static unsafe void NonZeroSimdHelper<T>(T* src, long size, System.Collections.Generic.List<long> indices)
            where T : unmanaged
        {
            if (size == 0)
                return;

            if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && size >= Vector256<T>.Count)
            {
                int vectorCount = Vector256<T>.Count;
                long vectorEnd = size - vectorCount;
                var zero = Vector256<T>.Zero;
                long i = 0;

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
                long vectorEnd = size - vectorCount;
                var zero = Vector128<T>.Zero;
                long i = 0;

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
                for (long i = 0; i < size; i++)
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
        /// <returns>Array of NDArray&lt;long&gt;, one per dimension</returns>
        internal static unsafe NumSharp.Generic.NDArray<long>[] ConvertFlatIndicesToCoordinates(
            System.Collections.Generic.List<long> flatIndices, int[] shape)
        {
            int ndim = shape.Length;
            int len = flatIndices.Count;

            // Create result arrays
            var result = new NumSharp.Generic.NDArray<long>[ndim];
            for (int d = 0; d < ndim; d++)
                result[d] = new NumSharp.Generic.NDArray<long>(len);

            // Get addresses for direct writing
            var addresses = new long*[ndim];
            for (int d = 0; d < ndim; d++)
                addresses[d] = (long*)result[d].Address;

            // Pre-compute strides for index conversion
            var strides = new long[ndim];
            strides[ndim - 1] = 1;
            for (int d = ndim - 2; d >= 0; d--)
                strides[d] = strides[d + 1] * shape[d + 1];

            // Convert each flat index to coordinates
            for (int i = 0; i < len; i++)
            {
                long flatIdx = flatIndices[i];
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
        /// <param name="strides">Array strides (elements, not bytes)</param>
        /// <param name="offset">Base offset into storage</param>
        /// <returns>Array of NDArray&lt;long&gt;, one per dimension</returns>
        internal static unsafe NumSharp.Generic.NDArray<long>[] FindNonZeroStridedHelper<T>(
            T* data, int[] shape, long[] strides, long offset) where T : unmanaged
        {
            int ndim = shape.Length;

            // Handle empty array
            long size = 1;
            for (int d = 0; d < ndim; d++)
                size *= shape[d];

            if (size == 0)
            {
                var emptyResult = new NumSharp.Generic.NDArray<long>[ndim];
                for (int d = 0; d < ndim; d++)
                    emptyResult[d] = new NumSharp.Generic.NDArray<long>(0);
                return emptyResult;
            }

            // Collect coordinates of non-zero elements
            // Pre-allocate with estimated capacity (assume ~25% non-zero for efficiency)
            // List<T> capacity is int-limited by .NET design.
            // For very large arrays, start with a reasonable capacity and let it grow.
            int initialCapacity = size <= int.MaxValue
                ? Math.Max(16, (int)(size / 4))
                : 1 << 20; // 1M for very large arrays
            var nonzeroCoords = new System.Collections.Generic.List<long[]>(initialCapacity);

            // Initialize coordinate array
            var coords = new long[ndim];

            // Iterate through all elements using coordinate-based iteration
            // This handles arbitrary strides including negative strides
            while (true)
            {
                // Calculate offset for current coordinates: offset + sum(coords[i] * strides[i])
                long elemOffset = offset;
                for (int d = 0; d < ndim; d++)
                    elemOffset += coords[d] * strides[d];

                // Check if element is non-zero
                if (!System.Collections.Generic.EqualityComparer<T>.Default.Equals(data[elemOffset], default))
                {
                    // Clone coordinates and add to result
                    var coordsCopy = new long[ndim];
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
            var result = new NumSharp.Generic.NDArray<long>[ndim];
            for (int d = 0; d < ndim; d++)
                result[d] = new NumSharp.Generic.NDArray<long>(len);

            // Get addresses for direct writing
            var addresses = new long*[ndim];
            for (int d = 0; d < ndim; d++)
                addresses[d] = (long*)result[d].Address;

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
    }
}
