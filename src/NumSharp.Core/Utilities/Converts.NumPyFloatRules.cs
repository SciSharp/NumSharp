using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    public static partial class Converts
    {
        // =====================================================================
        // Converts.NumPyFloatRules.cs
        //   The two places where NumPy's scalar float conversions and the BCL's disagree, as bit-exact helpers the
        //   Converts table, the scalar converter (NDIterCasting.ConvertValue) and the IL kernels' conversion emitter
        //   (DirectILKernelGenerator.EmitConvertTo) all route through, so every conversion path gives NumPy's bits.
        //
        //   1. A NaN crossing float16. NumPy converts float16 <-> float32 / float64 bit by bit
        //      (npy_halfbits_to_floatbits, npy_halfbits_to_doublebits, npy_floatbits_to_halfbits,
        //      npy_doublebits_to_halfbits in numpy/_core/src/npymath/halffloat.cpp): the significand's top bits move
        //      across and a SIGNALLING NaN STAYS SIGNALLING. The BCL casts — (float)Half, (double)Half, (Half)float,
        //      (Half)double — set the quiet bit, and a narrowing whose surviving payload bits are all zero becomes the
        //      canonical 0x7E00 where NumPy keeps the NaN signalling as 0x7C01. Finite values and infinities are
        //      identical both ways (exact widening; round-to-nearest-even narrowing), so only the NaN arm differs.
        //   2. uint64 -> float32 / float64. NumPy's C casts round ONCE. .NET 8 converts ulong -> float through double
        //      (two roundings, wrong for many values past 2^53: 2^60 + 2^36 + 1 lands on 0x5D800000, not 0x5D800001;
        //      2^64 - 2^39 - 1 on 2^64 itself) and ulong -> double from 2^63 on as signed plus 2^64 (two roundings:
        //      2^63 + 1025 lands on 2^63, not 2^63 + 2048). .NET 9 and later round once, and so do these helpers, on
        //      every runtime.
        // =====================================================================

        /// <summary>
        ///     Whether a float16 bit pattern is a NaN (all-ones exponent, non-zero significand) — the one case where
        ///     NumPy's float16 conversions differ from the BCL casts.
        /// </summary>
        /// <param name="bits">The IEEE binary16 encoding.</param>
        /// <returns>True for every NaN, quiet or signalling, either sign.</returns>
        [MethodImpl(OptimizeAndInline)]
        internal static bool IsHalfNaNBits(ushort bits) => (bits & 0x7FFF) > 0x7C00;

        /// <summary>
        ///     NumPy's float16 NaN → float32 (<c>npy_halfbits_to_floatbits</c>): the sign, an all-ones exponent and the
        ///     10-bit significand shifted into the top of the 23-bit one — a signalling NaN stays signalling.
        /// </summary>
        /// <param name="bits">A float16 NaN's encoding (<see cref="IsHalfNaNBits"/> is true).</param>
        /// <returns>The float32 NaN NumPy produces.</returns>
        [MethodImpl(OptimizeAndInline)]
        internal static float HalfNaNBitsToSingle(ushort bits)
            => BitConverter.UInt32BitsToSingle(((uint)(bits & 0x8000) << 16) | 0x7F80_0000u | ((uint)(bits & 0x03FF) << 13));

        /// <summary>
        ///     NumPy's float16 NaN → float64 (<c>npy_halfbits_to_doublebits</c>): the sign, an all-ones exponent and the
        ///     10-bit significand shifted into the top of the 52-bit one — a signalling NaN stays signalling.
        /// </summary>
        /// <param name="bits">A float16 NaN's encoding (<see cref="IsHalfNaNBits"/> is true).</param>
        /// <returns>The float64 NaN NumPy produces.</returns>
        [MethodImpl(OptimizeAndInline)]
        internal static double HalfNaNBitsToDouble(ushort bits)
            => BitConverter.UInt64BitsToDouble(((ulong)(bits & 0x8000) << 48) | 0x7FF0_0000_0000_0000UL | ((ulong)(bits & 0x03FF) << 42));

        /// <summary>
        ///     NumPy's float32 NaN → float16 (<c>npy_floatbits_to_halfbits</c>): the top 10 of the 23 significand bits
        ///     are kept — a signalling NaN stays signalling — and a NaN whose kept bits are all zero becomes 0x7C01
        ///     (still a NaN, still signalling), never the BCL's canonical 0x7E00.
        /// </summary>
        /// <param name="bits">A float32 NaN's encoding.</param>
        /// <returns>The float16 NaN NumPy produces.</returns>
        [MethodImpl(OptimizeAndInline)]
        internal static Half SingleNaNBitsToHalf(uint bits)
        {
            uint ret = 0x7C00u + ((bits & 0x007F_FFFFu) >> 13);
            // A payload only in the 13 dropped bits: add one so the result is not +/-inf.
            if (ret == 0x7C00u)
                ret++;
            return BitConverter.UInt16BitsToHalf((ushort)(((bits >> 16) & 0x8000u) | ret));
        }

        /// <summary>
        ///     NumPy's float64 NaN → float16 (<c>npy_doublebits_to_halfbits</c>): the top 10 of the 52 significand bits
        ///     are kept — a signalling NaN stays signalling — and a NaN whose kept bits are all zero becomes 0x7C01.
        /// </summary>
        /// <param name="bits">A float64 NaN's encoding.</param>
        /// <returns>The float16 NaN NumPy produces.</returns>
        [MethodImpl(OptimizeAndInline)]
        internal static Half DoubleNaNBitsToHalf(ulong bits)
        {
            uint ret = 0x7C00u + (uint)((bits & 0x000F_FFFF_FFFF_FFFFUL) >> 42);
            // A payload only in the 42 dropped bits: add one so the result is not +/-inf.
            if (ret == 0x7C00u)
                ret++;
            return BitConverter.UInt16BitsToHalf((ushort)(((uint)(bits >> 48) & 0x8000u) | ret));
        }

        /// <summary>
        ///     uint64 → float64 with ONE rounding (round-to-nearest-even), as NumPy's C cast does, on every runtime.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The float64 nearest <paramref name="value"/> (ties to even).</returns>
        /// <remarks>
        ///     Below 2⁶³ the value is a non-negative int64, whose conversion rounds once everywhere (cvtsi2sd). From 2⁶³ on
        ///     it is halved with the dropped low bit OR-ed back in as a sticky bit: the half has 63 significant bits and
        ///     rounds to 53 exactly as the value would (the sticky bit lies below the rounding position, so it only
        ///     breaks ties the way the full value would), and doubling it back is exact. .NET 8's own conversion instead
        ///     converts as signed and adds 2⁶⁴ — two roundings.
        /// </remarks>
        [MethodImpl(OptimizeAndInline)]
        internal static double UInt64ToDoubleRoundOnce(ulong value)
        {
            if ((long)value >= 0)
                return (long)value;
            return (long)((value >> 1) | (value & 1)) * 2.0;
        }

        /// <summary>
        ///     uint64 → float32 with ONE rounding (round-to-nearest-even), as NumPy's C cast does, on every runtime.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The float32 nearest <paramref name="value"/> (ties to even).</returns>
        /// <remarks>
        ///     Below 2⁶³ the value is a non-negative int64, whose conversion rounds once everywhere (cvtsi2ss) — .NET 8's
        ///     own ulong conversion goes through float64 and rounds twice. From 2⁶³ on, the sticky-bit halving of
        ///     <see cref="UInt64ToDoubleRoundOnce"/>: the half rounds to 24 bits exactly as the value would, and doubling
        ///     is exact (the largest result, 2⁶⁴, is a float32).
        /// </remarks>
        [MethodImpl(OptimizeAndInline)]
        internal static float UInt64ToSingleRoundOnce(ulong value)
        {
            if ((long)value >= 0)
                return (long)value;
            return (long)((value >> 1) | (value & 1)) * 2f;
        }
    }
}
