using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Modf - SIMD-optimized Modf operations
// =============================================================================
//
// This partial class provides high-performance Modf operations using SIMD.
// Modf(x) returns (fractional_part, integral_part) following C standard modf:
//
//   Normal values:
//     integral_part = Truncate(x)
//     fractional_part = x - Truncate(x)
//
//   Special values (C standard / NumPy behavior):
//     modf(+inf) = (+0.0, +inf)
//     modf(-inf) = (-0.0, -inf)
//     modf(nan)  = (nan, nan)
//
// SIMD approach (.NET 9+):
// - Use Vector.Truncate to get integral parts
// - Subtract from original to get fractional parts
// - Special value handling via scalar fixup
//
// .NET 8 fallback: Scalar loop only (Vector.Truncate not available)
//
// Only float/double supported (modf is a floating-point operation).
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        #region Modf Helpers

        /// <summary>
        /// Scalar modf for float following C standard / NumPy behavior.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ModfScalar(float value, out float fractional, out float integral)
        {
            if (float.IsNaN(value))
            {
                fractional = value;
                integral = value;
            }
            else if (float.IsInfinity(value))
            {
                // C standard: modf(inf) = (copysign(0.0, inf), inf)
                integral = value;
                fractional = float.CopySign(0f, value);
            }
            else
            {
                integral = MathF.Truncate(value);
                // Preserve sign of zero: -0.0 - (-0.0) = +0.0, but NumPy returns -0.0
                fractional = value - integral;
                // Ensure fractional has same sign as input for zero results
                if (fractional == 0f)
                    fractional = float.CopySign(0f, value);
            }
        }

        /// <summary>
        /// Scalar modf for double following C standard / NumPy behavior.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ModfScalar(double value, out double fractional, out double integral)
        {
            if (double.IsNaN(value))
            {
                fractional = value;
                integral = value;
            }
            else if (double.IsInfinity(value))
            {
                // C standard: modf(inf) = (copysign(0.0, inf), inf)
                integral = value;
                fractional = double.CopySign(0d, value);
            }
            else
            {
                integral = Math.Truncate(value);
                // Preserve sign of zero: -0.0 - (-0.0) = +0.0, but NumPy returns -0.0
                fractional = value - integral;
                // Ensure fractional has same sign as input for zero results
                if (fractional == 0d)
                    fractional = double.CopySign(0d, value);
            }
        }

        /// <summary>
        /// SIMD-optimized Modf operation for contiguous float arrays.
        /// Computes fractional and integral parts in-place.
        /// Handles special values (NaN, Inf) according to C standard modf.
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
            // Vector512 path (.NET 9+ has Vector.Truncate)
            // Note: SIMD path produces NaN for inf-inf, we fixup afterwards
            if (VectorBits >= 512 && size >= Vector512<float>.Count)
            {
                int vectorCount = Vector512<float>.Count;
                int vectorEnd = size - vectorCount;
                var signBitMask = Vector512.Create(-0f); // Sign bit mask: 0x80000000
                var zero = Vector512<float>.Zero;
                var posInf = Vector512.Create(float.PositiveInfinity);

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector512.Load(data + i);
                    var truncVec = Vector512.Truncate(vec);
                    var fracVec = vec - truncVec;

                    // Extract sign bits from input
                    var signedZero = vec & signBitMask;

                    // Fixup 1: where input is inf, frac should be copysign(0, input)
                    var infMask = Vector512.Equals(Vector512.Abs(vec), posInf);
                    fracVec = Vector512.ConditionalSelect(infMask, signedZero, fracVec);

                    // Fixup 2: where frac is zero, preserve sign from input
                    var zeroMask = Vector512.Equals(fracVec, zero);
                    fracVec = Vector512.ConditionalSelect(zeroMask, signedZero | fracVec, fracVec);

                    fracVec.Store(data + i);
                    truncVec.Store(integral + i);
                }
            }
            // Vector256 path
            else if (VectorBits >= 256 && size >= Vector256<float>.Count)
            {
                int vectorCount = Vector256<float>.Count;
                int vectorEnd = size - vectorCount;
                var signBitMask = Vector256.Create(-0f); // Sign bit mask: 0x80000000
                var zero = Vector256<float>.Zero;
                var posInf = Vector256.Create(float.PositiveInfinity);

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(data + i);
                    var truncVec = Vector256.Truncate(vec);
                    var fracVec = vec - truncVec;

                    // Extract sign bits from input
                    var signedZero = vec & signBitMask;

                    // Fixup 1: where input is inf, frac should be copysign(0, input)
                    var infMask = Vector256.Equals(Vector256.Abs(vec), posInf);
                    fracVec = Vector256.ConditionalSelect(infMask, signedZero, fracVec);

                    // Fixup 2: where frac is zero, preserve sign from input
                    var zeroMask = Vector256.Equals(fracVec, zero);
                    fracVec = Vector256.ConditionalSelect(zeroMask, signedZero | fracVec, fracVec);

                    fracVec.Store(data + i);
                    truncVec.Store(integral + i);
                }
            }
            // Vector128 path
            else if (VectorBits >= 128 && size >= Vector128<float>.Count)
            {
                int vectorCount = Vector128<float>.Count;
                int vectorEnd = size - vectorCount;
                var signBitMask = Vector128.Create(-0f); // Sign bit mask: 0x80000000
                var zero = Vector128<float>.Zero;
                var posInf = Vector128.Create(float.PositiveInfinity);

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(data + i);
                    var truncVec = Vector128.Truncate(vec);
                    var fracVec = vec - truncVec;

                    // Extract sign bits from input
                    var signedZero = vec & signBitMask;

                    // Fixup 1: where input is inf, frac should be copysign(0, input)
                    var infMask = Vector128.Equals(Vector128.Abs(vec), posInf);
                    fracVec = Vector128.ConditionalSelect(infMask, signedZero, fracVec);

                    // Fixup 2: where frac is zero, preserve sign from input
                    var zeroMask = Vector128.Equals(fracVec, zero);
                    fracVec = Vector128.ConditionalSelect(zeroMask, signedZero | fracVec, fracVec);

                    fracVec.Store(data + i);
                    truncVec.Store(integral + i);
                }
            }
#endif

            // Scalar tail (or full loop on .NET 8) - handles special values correctly
            for (; i < size; i++)
            {
                ModfScalar(data[i], out data[i], out integral[i]);
            }
        }

        /// <summary>
        /// SIMD-optimized Modf operation for contiguous double arrays.
        /// Computes fractional and integral parts in-place.
        /// Handles special values (NaN, Inf) according to C standard modf.
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
            // Vector512 path (.NET 9+ has Vector.Truncate)
            // Note: SIMD path produces NaN for inf-inf, we fixup afterwards
            if (VectorBits >= 512 && size >= Vector512<double>.Count)
            {
                int vectorCount = Vector512<double>.Count;
                int vectorEnd = size - vectorCount;
                var signBitMask = Vector512.Create(-0d); // Sign bit mask: 0x8000000000000000
                var zero = Vector512<double>.Zero;
                var posInf = Vector512.Create(double.PositiveInfinity);

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector512.Load(data + i);
                    var truncVec = Vector512.Truncate(vec);
                    var fracVec = vec - truncVec;

                    // Extract sign bits from input
                    var signedZero = vec & signBitMask;

                    // Fixup 1: where input is inf, frac should be copysign(0, input)
                    var infMask = Vector512.Equals(Vector512.Abs(vec), posInf);
                    fracVec = Vector512.ConditionalSelect(infMask, signedZero, fracVec);

                    // Fixup 2: where frac is zero, preserve sign from input
                    var zeroMask = Vector512.Equals(fracVec, zero);
                    fracVec = Vector512.ConditionalSelect(zeroMask, signedZero | fracVec, fracVec);

                    fracVec.Store(data + i);
                    truncVec.Store(integral + i);
                }
            }
            // Vector256 path
            else if (VectorBits >= 256 && size >= Vector256<double>.Count)
            {
                int vectorCount = Vector256<double>.Count;
                int vectorEnd = size - vectorCount;
                var signBitMask = Vector256.Create(-0d); // Sign bit mask: 0x8000000000000000
                var zero = Vector256<double>.Zero;
                var posInf = Vector256.Create(double.PositiveInfinity);

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(data + i);
                    var truncVec = Vector256.Truncate(vec);
                    var fracVec = vec - truncVec;

                    // Extract sign bits from input
                    var signedZero = vec & signBitMask;

                    // Fixup 1: where input is inf, frac should be copysign(0, input)
                    var infMask = Vector256.Equals(Vector256.Abs(vec), posInf);
                    fracVec = Vector256.ConditionalSelect(infMask, signedZero, fracVec);

                    // Fixup 2: where frac is zero, preserve sign from input
                    var zeroMask = Vector256.Equals(fracVec, zero);
                    fracVec = Vector256.ConditionalSelect(zeroMask, signedZero | fracVec, fracVec);

                    fracVec.Store(data + i);
                    truncVec.Store(integral + i);
                }
            }
            // Vector128 path
            else if (VectorBits >= 128 && size >= Vector128<double>.Count)
            {
                int vectorCount = Vector128<double>.Count;
                int vectorEnd = size - vectorCount;
                var signBitMask = Vector128.Create(-0d); // Sign bit mask: 0x8000000000000000
                var zero = Vector128<double>.Zero;
                var posInf = Vector128.Create(double.PositiveInfinity);

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(data + i);
                    var truncVec = Vector128.Truncate(vec);
                    var fracVec = vec - truncVec;

                    // Extract sign bits from input
                    var signedZero = vec & signBitMask;

                    // Fixup 1: where input is inf, frac should be copysign(0, input)
                    var infMask = Vector128.Equals(Vector128.Abs(vec), posInf);
                    fracVec = Vector128.ConditionalSelect(infMask, signedZero, fracVec);

                    // Fixup 2: where frac is zero, preserve sign from input
                    var zeroMask = Vector128.Equals(fracVec, zero);
                    fracVec = Vector128.ConditionalSelect(zeroMask, signedZero | fracVec, fracVec);

                    fracVec.Store(data + i);
                    truncVec.Store(integral + i);
                }
            }
#endif

            // Scalar tail (or full loop on .NET 8) - handles special values correctly
            for (; i < size; i++)
            {
                ModfScalar(data[i], out data[i], out integral[i]);
            }
        }

        #endregion
    }
}
