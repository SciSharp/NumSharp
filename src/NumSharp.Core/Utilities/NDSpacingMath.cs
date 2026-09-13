using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    ///     Scalar kernels backing NumSharp's <c>np.spacing</c> unary ufunc — the distance from a
    ///     floating-point value to the adjacent representable value in the direction <b>away from
    ///     zero</b> (one ULP), the unary sibling of <see cref="NDLogAddExpMath.NextAfter(double,double)"/>.
    ///     Each entry point is a bit-for-bit port of the exact NumPy 2.4.2 routine, so the result is
    ///     byte-identical to NumPy on every finite input (verified: float16 exhaustively over all
    ///     65 536 patterns, float32/float64 over 1M random samples plus the ±0/±inf/NaN/subnormal/max
    ///     edge grid).
    ///
    ///     <para><b>float64 / float32 are <see cref="npy_spacing"/> = <c>_next(x, 1) - x</c></b>
    ///     (<c>ieee754.c.src</c>): NumPy steps the raw bit pattern one integer up (<c>bits + 1</c>),
    ///     which moves toward +inf for positive <c>x</c> and toward -inf for negative <c>x</c> — i.e.
    ///     AWAY from zero — then subtracts. So the result carries the SIGN of <c>x</c> (unlike float16,
    ///     see below): <c>spacing(1.0)</c> = <c>+eps</c>, <c>spacing(-1.0)</c> = <c>-eps</c>. The one
    ///     input the raw increment gets wrong is <c>-0.0</c> (its <c>bits+1</c> is <c>-minsubnormal</c>),
    ///     which NumPy special-cases to <c>+minsubnormal</c> — hence the explicit <c>x == 0</c> guard,
    ///     which also covers <c>+0.0</c> at the same value. <c>±inf</c> and <c>NaN</c> fall out of the
    ///     same arithmetic (<c>bits+1</c> of an infinity is a NaN pattern, and <c>NaN - x = NaN</c>), so
    ///     the SIMD kernel (<c>DirectILKernelGenerator.EmitVectorSpacing</c>) needs no branch for them;
    ///     these scalar entry points keep the explicit <c>inf → NaN</c> / <c>NaN → NaN</c> guards for
    ///     clarity. The result NaN's bit pattern is host/path-dependent (NumPy's canonical positive
    ///     <c>NPY_NAN</c> vs .NET's negative <c>NaN</c>), a difference the oracle tokenizes.</para>
    ///
    ///     <para><b>float16 is <see cref="Spacing(Half)"/> = <c>npy_half_spacing</c></b>
    ///     (<c>halffloat.cpp</c>), a SEPARATE hand-written routine on the raw 16-bit pattern that is
    ///     <b>always non-negative</b> and treats a negative power-of-2 boundary specially (it returns the
    ///     smaller, toward-zero ULP there). This is a genuine NumPy inconsistency with the signed
    ///     float32/float64 result and is reproduced verbatim rather than "fixed", so
    ///     <c>spacing(float16(-1.0))</c> = <c>+2^-11</c> where <c>spacing(float32(-1.0))</c> = <c>-2^-23</c>.</para>
    ///
    ///     <para><b>Decimal</b> (no NumPy analog) bridges through double — <c>(decimal)Spacing((double)x)</c>
    ///     — matching the house policy for every decimal transcendental. A decimal is always finite, so
    ///     no inf/NaN path is reachable.</para>
    /// </summary>
    internal static class NDSpacingMath
    {
        /// <summary>
        ///     <c>npy_spacing</c> for float64: the signed one-ULP step away from zero, <c>_next(x,1) - x</c>.
        /// </summary>
        /// <param name="x">The value whose spacing (ULP) is requested.</param>
        /// <returns>
        ///     For finite non-zero <paramref name="x"/>, the distance to the next representable value away
        ///     from zero, carrying the sign of <paramref name="x"/> (overflows to <c>+inf</c> at
        ///     <c>double.MaxValue</c>). For <c>±0</c>, <c>+double.Epsilon</c> (the smallest subnormal).
        ///     For <c>±inf</c>, <c>NaN</c>. For <c>NaN</c>, the input NaN unchanged.
        /// </returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static double Spacing(double x)
        {
            // NaN passes through (NumPy's _next returns the NaN, then NaN - NaN = NaN); the exact bit
            // pattern is tokenized by the oracle, so returning the input NaN is as valid as recomputing.
            if (double.IsNaN(x)) return x;
            // ±inf -> NaN (npy_spacing's explicit isinf guard).
            if (double.IsInfinity(x)) return double.NaN;
            // Both signs of zero map to +minsubnormal: NumPy's _next(0, 1) steps toward +inf regardless
            // of the zero's sign, and the raw increment of -0.0 alone would give the wrong -minsubnormal.
            if (x == 0.0) return double.Epsilon;
            // Raw bit increment = one ULP away from zero (toward +inf for x>0, toward -inf for x<0).
            long bits = BitConverter.DoubleToInt64Bits(x) + 1L;
            return BitConverter.Int64BitsToDouble(bits) - x;
        }

        /// <summary>
        ///     <c>npy_spacingf</c> for float32: the signed one-ULP step away from zero (see
        ///     <see cref="Spacing(double)"/> — identical algorithm at single precision).
        /// </summary>
        /// <param name="x">The value whose spacing (ULP) is requested.</param>
        /// <returns>
        ///     The signed one-ULP distance for finite non-zero <paramref name="x"/> (<c>+inf</c> at
        ///     <c>float.MaxValue</c>); <c>+float.Epsilon</c> for <c>±0</c>; <c>NaN</c> for <c>±inf</c>;
        ///     the input unchanged for <c>NaN</c>.
        /// </returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static float Spacing(float x)
        {
            if (float.IsNaN(x)) return x;
            if (float.IsInfinity(x)) return float.NaN;
            if (x == 0.0f) return float.Epsilon;
            int bits = BitConverter.SingleToInt32Bits(x) + 1;
            return BitConverter.Int32BitsToSingle(bits) - x;
        }

        /// <summary>
        ///     <c>npy_half_spacing</c> (<c>halffloat.cpp</c>): float16 spacing on the raw 16-bit pattern.
        ///     UNLIKE <see cref="Spacing(double)"/>/<see cref="Spacing(float)"/> this is
        ///     <b>always non-negative</b> and returns the smaller, toward-zero ULP at a negative power-of-2
        ///     boundary — a deliberate NumPy inconsistency reproduced verbatim for parity, not a bug.
        /// </summary>
        /// <param name="hv">The float16 value whose spacing (ULP) is requested.</param>
        /// <returns>
        ///     The (non-negative) ULP at <paramref name="hv"/>: <c>+inf</c> at the largest finite half
        ///     (<c>0x7bff</c>), <c>NaN</c> for <c>±inf</c>/<c>NaN</c>, the smallest subnormal half
        ///     (<c>0x0001</c>) near zero, and the boundary-specific smaller ULP for negative powers of two.
        /// </returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Half Spacing(Half hv)
        {
            // Work on the raw 16-bit pattern in int arithmetic (no float round-trip): npy_half_spacing.
            int h = BitConverter.HalfToUInt16Bits(hv);
            int h_exp = h & 0x7c00;   // biased exponent field
            int h_sig = h & 0x03ff;   // significand field
            int ret;
            if (h_exp == 0x7c00)                       ret = 0x7e00;         // inf/nan -> NPY_HALF_NAN
            else if (h == 0x7bff)                      ret = 0x7c00;         // largest +half -> +inf (overflow)
            else if ((h & 0x8000) != 0 && h_sig == 0)                        // negative power-of-2 boundary
            {
                // At a negative boundary the toward-zero ULP is smaller than the away-from-zero one;
                // NumPy returns that smaller (positive) value.
                if (h_exp > 0x2c00)      ret = h_exp - 0x2c00;               // result still normalized
                else if (h_exp > 0x0400) ret = 1 << ((h_exp >> 10) - 2);     // subnormal, not smallest
                else                     ret = 0x0001;                       // smallest subnormal half
            }
            else if (h_exp > 0x2800)  ret = h_exp - 0x2800;                  // normal
            else if (h_exp > 0x0400)  ret = 1 << ((h_exp >> 10) - 1);        // subnormal, not smallest
            else                      ret = 0x0001;                          // smallest subnormal half
            return BitConverter.UInt16BitsToHalf((ushort)ret);
        }

        /// <summary>
        ///     Decimal spacing via the double bridge (no NumPy analog). Matches the house convention for
        ///     decimal transcendentals; a decimal is always finite so no inf/NaN path is reachable.
        /// </summary>
        /// <param name="x">The decimal value whose spacing (ULP, at double precision) is requested.</param>
        /// <returns><c>(decimal)Spacing((double)x)</c> — the double-precision ULP cast back to decimal.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static decimal Spacing(decimal x) => (decimal)Spacing((double)x);
    }
}
