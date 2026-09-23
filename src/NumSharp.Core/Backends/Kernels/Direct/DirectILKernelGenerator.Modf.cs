using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

// =============================================================================
// DirectILKernelGenerator.Modf - SIMD-optimized Modf operations
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
    public static partial class DirectILKernelGenerator
    {
        #region Modf Helpers

        /// <summary>
        /// Scalar modf for float following C standard / NumPy behavior.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
        /// Scalar modf for Half — a byte-exact port of NumPy 2.4.2's HALF_modf loop
        /// (loops.c.src:1908): widen to float, run the float modf (<see cref="ModfScalar(float, out float, out float)"/>
        /// == C <c>modff</c>), then narrow BOTH results to Half. There is no vector f16 truncate in the
        /// BCL and NumPy's own half loop is scalar, so this stays scalar.
        /// </summary>
        /// <param name="value">The input value.</param>
        /// <param name="fractional">The signed fractional part (Half).</param>
        /// <param name="integral">The signed integral part (Half).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static void ModfScalar(Half value, out Half fractional, out Half integral)
        {
            // NumPy: half -> float -> modff -> narrow both to half. The float ModfScalar already carries
            // the inf/nan/signed-zero fixups of C modff, so the widen/narrow inherits them exactly
            // (modf(inf)=(+0,inf), modf(-inf)=(-0,-inf), modf(nan)=(nan,nan), modf(-0)=(-0,-0)).
            ModfScalar((float)value, out float f, out float ig);
            fractional = (Half)f;
            integral = (Half)ig;
        }

        /// <summary>
        /// Modf over a contiguous Half array (scalar loop; see <see cref="ModfScalar(Half, out Half, out Half)"/>).
        /// Byte-exact with NumPy's HALF_modf. Writes fractional parts back into <paramref name="data"/>.
        /// </summary>
        /// <param name="data">Input array (holds the fractional parts on return).</param>
        /// <param name="integral">Output array for the integral parts.</param>
        /// <param name="size">Number of elements.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ModfHelper(Half* data, Half* integral, long size)
            => ModfHelper(data, data, integral, size);

        /// <summary>
        /// Out-of-place Half modf: reads <paramref name="input"/> once, writes fractional/integral to their
        /// own buffers (see the float overload; NumPy's HALF_modf is scalar so this is a scalar loop).
        /// </summary>
        /// <param name="input">Source array (read-only unless it aliases an output).</param>
        /// <param name="frac">Destination for the fractional parts.</param>
        /// <param name="integral">Destination for the integral parts.</param>
        /// <param name="size">Number of elements.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ModfHelper(Half* input, Half* frac, Half* integral, long size)
        {
            for (long i = 0; i < size; i++)
                ModfScalar(input[i], out frac[i], out integral[i]);
        }

        /// <summary>
        /// In-place SIMD Modf wrapper for contiguous float arrays: <paramref name="data"/> holds the input
        /// on entry and the fractional parts on return. Kept for callers that mutate the input copy in
        /// place; forwards to the out-of-place kernel with input==frac (which is a safe per-element alias).
        /// </summary>
        /// <param name="data">Input array (holds the fractional parts on return).</param>
        /// <param name="integral">Output array for the integral parts.</param>
        /// <param name="size">Number of elements.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ModfHelper(float* data, float* integral, long size)
            => ModfHelper(data, data, integral, size);

        /// <summary>
        /// Out-of-place SIMD Modf for contiguous float arrays: reads <paramref name="input"/> ONCE and
        /// writes the fractional and integral parts to their own buffers — NumPy's 3-touch pattern (1 read,
        /// 2 writes), avoiding the extra input-copy + write-back pass an in-place kernel needs when the
        /// caller supplies output arrays. Any two of the three pointers may alias per-element (input==frac
        /// gives the in-place case; a provided out that equals the input is handled the same way). Handles
        /// the special values (NaN, ±inf, signed zero) per C standard modf.
        /// </summary>
        /// <param name="input">Source array (read-only unless it aliases an output).</param>
        /// <param name="frac">Destination for the fractional parts.</param>
        /// <param name="integral">Destination for the integral parts.</param>
        /// <param name="size">Number of elements.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ModfHelper(float* input, float* frac, float* integral, long size)
        {
            if (size == 0) return;

            long i = 0;

#if NET9_0_OR_GREATER
            // Vector512 path (.NET 9+ has Vector.Truncate)
            // Note: SIMD path produces NaN for inf-inf, we fixup afterwards
            if (VectorBits >= 512 && size >= Vector512<float>.Count)
            {
                int vectorCount = Vector512<float>.Count;
                long vectorEnd = size - vectorCount;
                var signBitMask = Vector512.Create(-0f); // Sign bit mask: 0x80000000
                var zero = Vector512<float>.Zero;
                var posInf = Vector512.Create(float.PositiveInfinity);

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector512.Load(input + i);
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

                    fracVec.Store(frac + i);
                    truncVec.Store(integral + i);
                }
            }
            // Vector256 path
            else if (VectorBits >= 256 && size >= Vector256<float>.Count)
            {
                int vectorCount = Vector256<float>.Count;
                long vectorEnd = size - vectorCount;
                var signBitMask = Vector256.Create(-0f); // Sign bit mask: 0x80000000
                var zero = Vector256<float>.Zero;
                var posInf = Vector256.Create(float.PositiveInfinity);

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(input + i);
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

                    fracVec.Store(frac + i);
                    truncVec.Store(integral + i);
                }
            }
            // Vector128 path
            else if (VectorBits >= 128 && size >= Vector128<float>.Count)
            {
                int vectorCount = Vector128<float>.Count;
                long vectorEnd = size - vectorCount;
                var signBitMask = Vector128.Create(-0f); // Sign bit mask: 0x80000000
                var zero = Vector128<float>.Zero;
                var posInf = Vector128.Create(float.PositiveInfinity);

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(input + i);
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

                    fracVec.Store(frac + i);
                    truncVec.Store(integral + i);
                }
            }
#endif

            // Scalar tail (or full loop on .NET 8) - handles special values correctly. Read input[i] into a
            // local FIRST so an input==frac alias is safe (the store below may overwrite input[i]).
            for (; i < size; i++)
            {
                ModfScalar(input[i], out frac[i], out integral[i]);
            }
        }

        /// <summary>
        /// In-place SIMD Modf wrapper for contiguous double arrays (see the float overload). Forwards to
        /// the out-of-place kernel with input==frac.
        /// </summary>
        /// <param name="data">Input array (holds the fractional parts on return).</param>
        /// <param name="integral">Output array for the integral parts.</param>
        /// <param name="size">Number of elements.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ModfHelper(double* data, double* integral, long size)
            => ModfHelper(data, data, integral, size);

        /// <summary>
        /// Out-of-place SIMD Modf for contiguous double arrays: reads <paramref name="input"/> once and
        /// writes fractional/integral to their own buffers (NumPy's 3-touch pattern). See the float
        /// overload for the aliasing and special-value contract.
        /// </summary>
        /// <param name="input">Source array (read-only unless it aliases an output).</param>
        /// <param name="frac">Destination for the fractional parts.</param>
        /// <param name="integral">Destination for the integral parts.</param>
        /// <param name="size">Number of elements.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void ModfHelper(double* input, double* frac, double* integral, long size)
        {
            if (size == 0) return;

            long i = 0;

#if NET9_0_OR_GREATER
            // Vector512 path (.NET 9+ has Vector.Truncate)
            // Note: SIMD path produces NaN for inf-inf, we fixup afterwards
            if (VectorBits >= 512 && size >= Vector512<double>.Count)
            {
                int vectorCount = Vector512<double>.Count;
                long vectorEnd = size - vectorCount;
                var signBitMask = Vector512.Create(-0d); // Sign bit mask: 0x8000000000000000
                var zero = Vector512<double>.Zero;
                var posInf = Vector512.Create(double.PositiveInfinity);

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector512.Load(input + i);
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

                    fracVec.Store(frac + i);
                    truncVec.Store(integral + i);
                }
            }
            // Vector256 path
            else if (VectorBits >= 256 && size >= Vector256<double>.Count)
            {
                int vectorCount = Vector256<double>.Count;
                long vectorEnd = size - vectorCount;
                var signBitMask = Vector256.Create(-0d); // Sign bit mask: 0x8000000000000000
                var zero = Vector256<double>.Zero;
                var posInf = Vector256.Create(double.PositiveInfinity);

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(input + i);
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

                    fracVec.Store(frac + i);
                    truncVec.Store(integral + i);
                }
            }
            // Vector128 path
            else if (VectorBits >= 128 && size >= Vector128<double>.Count)
            {
                int vectorCount = Vector128<double>.Count;
                long vectorEnd = size - vectorCount;
                var signBitMask = Vector128.Create(-0d); // Sign bit mask: 0x8000000000000000
                var zero = Vector128<double>.Zero;
                var posInf = Vector128.Create(double.PositiveInfinity);

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(input + i);
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

                    fracVec.Store(frac + i);
                    truncVec.Store(integral + i);
                }
            }
#endif

            // Scalar tail (or full loop on .NET 8) - handles special values correctly.
            for (; i < size; i++)
            {
                ModfScalar(input[i], out frac[i], out integral[i]);
            }
        }

        #endregion
    }
}
