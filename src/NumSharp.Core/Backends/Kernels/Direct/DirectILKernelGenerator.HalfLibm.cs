using System;
using System.Runtime.CompilerServices;

// =============================================================================
// DirectILKernelGenerator.HalfLibm — NumPy's HALF loops for arctan2, log1p and expm1
// =============================================================================
//
// NumPy runs these three float16 ufuncs through their float32 loops (generate_umath.py: astype={'e': 'f'}): each
// float16 operand is widened to float32 bit for bit (npy_half_to_float keeps a signalling NaN signalling), the C
// library's float function runs — on win-amd64 the UCRT's atan2f / log1pf / expm1f — and npy_float_to_half narrows the
// float32 result. Rounding a float64 result once to float16, as NumSharp did, lands on the other neighbour wherever the
// float32 result sits on a float16 rounding midpoint, and the double bridge quieted a signalling NaN that NumPy keeps
// (parity QA-24). Every helper below is verified against NumPy 2.4.2 over all 65,536 float16 inputs, raw bits, NaN
// payloads included.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        /// One element of NumPy's <c>HALF_arctan2</c>: both halves widened to float32 bit for bit, the CRT's <c>atan2f</c>
        /// (<see cref="MathF.Atan2(float, float)"/> calls the same UCRT function NumPy's float32 loop does), then NumPy's
        /// <c>npy_float_to_half</c> narrowing — so a float32 result on a float16 midpoint ties to even as NumPy's does
        /// (<c>np.arctan2(float16 26.17, float16 -0.00011706)</c> is <c>0x3E48</c>, where rounding the double result gave
        /// <c>0x3E49</c>).
        /// </summary>
        /// <param name="y">The ordinate.</param>
        /// <param name="x">The abscissa.</param>
        /// <returns>The float16 angle NumPy returns, NaN bits included.</returns>
        [MethodImpl(OptimizeAndInline)]
        internal static Half ATan2HalfNumPy(Half y, Half x)
            => BitConverter.UInt16BitsToHalf(SingleToHalfBits(MathF.Atan2(
                HalfToFloatScalarExact(BitConverter.HalfToUInt16Bits(y)),
                HalfToFloatScalarExact(BitConverter.HalfToUInt16Bits(x)))));

        /// <summary>
        /// One element of NumPy's <c>HALF_log1p</c>: <c>npy_float_to_half(log1pf(x))</c>. The UCRT's <c>log1pf</c> is
        /// modelled exactly over the float16 domain: the float64 <c>log(1 + x)</c> (exact <c>1 + x</c> for every float16
        /// <c>x</c>) rounded to float32 first and then to float16 reproduces it on all 65,536 inputs; a NaN input comes back
        /// untouched (a signalling NaN stays signalling, as <c>log1pf</c> returns it); <c>x &lt; -1</c> (and <c>-inf</c>) is
        /// the POSITIVE quiet NaN <c>0x7E00</c> that <c>log1pf</c> produces.
        /// </summary>
        /// <param name="h">The argument.</param>
        /// <returns>The float16 <c>log1p</c> NumPy returns, NaN bits included.</returns>
        /// <remarks>
        /// The sign of zero: <c>log(1 + -0)</c> is <c>+0</c>, so the result takes the input's sign; <c>log1p(x)</c> has the
        /// sign of <c>x</c> over its whole real domain <c>(-1, inf)</c>, so this is exact there.
        /// </remarks>
        [MethodImpl(OptimizeAndInline)]
        internal static Half Log1pHalfNumPy(Half h)
        {
            ushort bits = BitConverter.HalfToUInt16Bits(h);
            // A NaN passes through log1pf untouched; widening and narrowing it changes nothing, so return the bits.
            if ((bits & 0x7FFF) > 0x7C00)
                return h;
            float f = HalfToFloatScalarExact(bits);
            // log1pf's domain error is the positive quiet NaN (the double route's log of a negative gave a negative one).
            if (f < -1f)
                return BitConverter.UInt16BitsToHalf(0x7E00);
            double d = f;
            double r = Math.CopySign(Math.Log(1.0 + d), d);   // 1 + d is exact: a float16 spans at most 41 bits
            return BitConverter.UInt16BitsToHalf(SingleToHalfBits((float)r));
        }

        /// <summary>
        /// One element of NumPy's <c>HALF_expm1</c>: <c>npy_float_to_half(expm1f(x))</c>. The UCRT's <c>expm1f</c> is not
        /// correctly rounded below <c>|x| = 1</c> (up to 2 ULP) and has no published source, so it is modelled over the
        /// float16 domain: the float64 <c>exp(x) - 1</c> rounded ONCE to float16 equals NumPy on all 65,536 inputs except
        /// two, where <c>expm1f</c>'s float32 result lands exactly on a float16 midpoint and ties to even — <c>0x2864</c>
        /// (0.0343) gives <c>0x2878</c> and <c>0xB94E</c> (-0.663) gives <c>0xB7C2</c> — which are pinned here. A NaN input
        /// comes back untouched (a signalling NaN stays signalling, as <c>expm1f</c> returns it).
        /// </summary>
        /// <param name="h">The argument.</param>
        /// <returns>The float16 <c>expm1</c> NumPy returns, NaN bits included.</returns>
        /// <remarks>
        /// A platform pin, like the MSVC float-to-unsigned conversions: on Linux NumPy calls glibc's <c>expm1f</c>, which
        /// rounds those two inputs the other way. Rounding through float32 instead would follow <c>expm1f</c> less closely
        /// (3 differences, the double-rounding at <c>0x11A8</c> among them), which is why this route rounds once.
        /// </remarks>
        [MethodImpl(OptimizeAndInline)]
        internal static Half Expm1HalfNumPy(Half h)
        {
            ushort bits = BitConverter.HalfToUInt16Bits(h);
            if ((bits & 0x7FFF) > 0x7C00)
                return h;
            // The two float16 inputs whose UCRT expm1f result is a float16 midpoint (exhaustive probe, NumPy 2.4.2).
            if (bits == 0x2864)
                return BitConverter.UInt16BitsToHalf(0x2878);
            if (bits == 0xB94E)
                return BitConverter.UInt16BitsToHalf(0xB7C2);
            double d = HalfToFloatScalarExact(bits);
            // expm1 keeps the sign of x (and of a zero) over its whole domain; exp(-0) - 1 alone would give +0.
            return (Half)Math.CopySign(Math.Exp(d) - 1.0, d);
        }
    }
}
