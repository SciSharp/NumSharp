using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Modf - SIMD-optimized Modf operations
// =============================================================================
//
// This partial class provides high-performance Modf operations using SIMD.
// Modf(x) returns (fractional_part, integral_part) where:
//   integral_part = Truncate(x)
//   fractional_part = x - Truncate(x)
//
// SIMD approach (.NET 9+):
// - Use Vector.Truncate to get integral parts
// - Subtract from original to get fractional parts
// - Store both results in parallel
//
// .NET 8 fallback: Scalar loop only (Vector.Truncate not available)
//
// Only float/double supported (modf is a floating-point operation).
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public sealed partial class ILKernelGenerator
    {
        #region Modf Helpers

        /// <summary>
        /// SIMD-optimized Modf operation for contiguous float arrays.
        /// Computes fractional and integral parts in-place.
        /// </summary>
        /// <param name="data">Input array (will contain fractional parts after)</param>
        /// <param name="integral">Output array for integral parts</param>
        /// <param name="size">Number of elements</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ModfHelper(float* data, float* integral, int size)
        {
            if (size == 0) return;

            int i = 0;

#if NET9_0_OR_GREATER
            // Vector256 path (.NET 9+ has Vector.Truncate)
            if (VectorBits >= 256 && size >= Vector256<float>.Count)
            {
                int vectorCount = Vector256<float>.Count;
                int vectorEnd = size - vectorCount;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(data + i);
                    var truncVec = Vector256.Truncate(vec);
                    var fracVec = vec - truncVec;
                    fracVec.Store(data + i);
                    truncVec.Store(integral + i);
                }
            }
            // Vector128 path
            else if (VectorBits >= 128 && size >= Vector128<float>.Count)
            {
                int vectorCount = Vector128<float>.Count;
                int vectorEnd = size - vectorCount;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(data + i);
                    var truncVec = Vector128.Truncate(vec);
                    var fracVec = vec - truncVec;
                    fracVec.Store(data + i);
                    truncVec.Store(integral + i);
                }
            }
#endif

            // Scalar tail (or full loop on .NET 8)
            for (; i < size; i++)
            {
                var trunc = MathF.Truncate(data[i]);
                integral[i] = trunc;
                data[i] = data[i] - trunc;
            }
        }

        /// <summary>
        /// SIMD-optimized Modf operation for contiguous double arrays.
        /// Computes fractional and integral parts in-place.
        /// </summary>
        /// <param name="data">Input array (will contain fractional parts after)</param>
        /// <param name="integral">Output array for integral parts</param>
        /// <param name="size">Number of elements</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ModfHelper(double* data, double* integral, int size)
        {
            if (size == 0) return;

            int i = 0;

#if NET9_0_OR_GREATER
            // Vector256 path (.NET 9+ has Vector.Truncate)
            if (VectorBits >= 256 && size >= Vector256<double>.Count)
            {
                int vectorCount = Vector256<double>.Count;
                int vectorEnd = size - vectorCount;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(data + i);
                    var truncVec = Vector256.Truncate(vec);
                    var fracVec = vec - truncVec;
                    fracVec.Store(data + i);
                    truncVec.Store(integral + i);
                }
            }
            // Vector128 path
            else if (VectorBits >= 128 && size >= Vector128<double>.Count)
            {
                int vectorCount = Vector128<double>.Count;
                int vectorEnd = size - vectorCount;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(data + i);
                    var truncVec = Vector128.Truncate(vec);
                    var fracVec = vec - truncVec;
                    fracVec.Store(data + i);
                    truncVec.Store(integral + i);
                }
            }
#endif

            // Scalar tail (or full loop on .NET 8)
            for (; i < size; i++)
            {
                var trunc = Math.Truncate(data[i]);
                integral[i] = trunc;
                data[i] = data[i] - trunc;
            }
        }

        #endregion
    }
}
