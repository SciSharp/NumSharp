using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Vector-width overloads of the <see cref="NDFloatMath"/> float32 kernels, so the IL-emitted
    /// SIMD unary loop can run the same NumPy algorithm a register at a time.
    ///
    /// <para>Every step of <c>simd_exp_FLOAT</c> is elementwise, so lane <c>i</c> of
    /// <c>Exp(vector)</c> is bit-identical to <c>Exp(vector[i])</c> — and to NumPy, at any width.
    /// That is not an accident of this port: NumPy generates its own kernel from one template for
    /// both <c>__m256</c> and <c>__m512</c> for exactly the same reason. The one place NumPy uses a
    /// whole-vector branch (the denormal half of <c>fma_scalef_ps</c>) is kept, because it is not
    /// only structural: taking it unconditionally costs a second <c>vdivps</c> on every vector, and
    /// exp's output is subnormal only below about -87.3, so the branch is essentially never taken.
    /// Measured, folding it away cost ~0.17 ns/element — a fifth of the whole kernel.</para>
    ///
    /// <para><b>One deliberate re-spelling of NumPy's integer steps</b>, exactly equal over the
    /// reachable domain: NumPy forms the divisor as <c>cvtepi32_ps(1 &lt;&lt; quadDiff)</c>
    /// (a variable per-lane shift, <c>vpsllvd</c>, which is AVX2-only), while this builds the same
    /// power of two straight into the exponent field as <c>(quadDiff + 127) &lt;&lt; 23</c> — equal
    /// for all <c>quadDiff ∈ [0, 25]</c>, the range NumPy's own bounds allow, and expressible with
    /// the cross-platform constant-shift API.</para>
    ///
    /// <para><b>Hardware gate.</b> Only the FMA is width-specific; <see cref="IsExpVectorAccelerated"/>
    /// reports whether this host has it, and the kernel emitters consult that before choosing the
    /// vector loop (falling back to the scalar entry point, which is the same bits). The software
    /// FMA fallback below is not a shipping path — it exists so the wider overloads stay testable on
    /// a host that cannot execute them natively.</para>
    /// </summary>
    public static partial class NDFloatMath
    {
        /// <summary>
        /// Whether <c>Exp(Vector{<paramref name="vectorBits"/>}&lt;float&gt;)</c> lowers to hardware
        /// FMA on this machine. NumPy gates its own vector exp the same way (its kernel is compiled
        /// only for AVX2+FMA3 / AVX-512F targets, and hosts without them run the scalar libm).
        /// </summary>
        public static bool IsExpVectorAccelerated(int vectorBits) => vectorBits switch
        {
            512 => Avx512F.IsSupported,
            256 => Fma.IsSupported,
            128 => Fma.IsSupported,
            _ => false
        };

        /// <summary>NumPy 2.4.2's float32 exponential, 4 lanes at a time. See <see cref="Exp(float)"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static Vector128<float> Exp(Vector128<float> x)
        {
            var zero = Vector128<float>.Zero;

            var nanMask = ~Vector128.Equals(x, x);
            x = Vector128.ConditionalSelect(nanMask, zero, x);

            var xmaxMask = Vector128.GreaterThanOrEqual(x, Vector128.Create(ExpXMax));
            var xminMask = Vector128.LessThanOrEqual(x, Vector128.Create(ExpXMin));
            x = Vector128.ConditionalSelect(nanMask | xminMask | xmaxMask, zero, x);

            var magic = Vector128.Create(RintCvtMagic);
            var quadrant = MulAdd(x, Vector128.Create(Log2E), magic) - magic;

            var y = MulAdd(quadrant, Vector128.Create(CodyWaiteLogE2High), x);
            y = MulAdd(quadrant, Vector128.Create(CodyWaiteLogE2Low), y);
            y = MulAdd(quadrant, zero, y);

            var num = MulAdd(Vector128.Create(ExpP5), y, Vector128.Create(ExpP4));
            num = MulAdd(num, y, Vector128.Create(ExpP3));
            num = MulAdd(num, y, Vector128.Create(ExpP2));
            num = MulAdd(num, y, Vector128.Create(ExpP1));
            num = MulAdd(num, y, Vector128.Create(ExpP0));
            var den = MulAdd(Vector128.Create(ExpQ2), y, Vector128.Create(ExpQ1));
            den = MulAdd(den, y, Vector128.Create(ExpQ0));
            var poly = num / den;

            var minQuad = Vector128.Create(-125f);
            var denormMask = Vector128.LessThanOrEqual(quadrant, minQuad);
            if (Vector128.ExtractMostSignificantBits(denormMask) != 0)
            {
                // At least one lane scales below the smallest normal, so take NumPy's split:
                // apply 2^-125 through the exponent field and let a division make the subnormal.
                var quadDiff = Vector128.ConditionalSelect(denormMask, zero - (quadrant - minQuad), zero);
                var scaled = (poly.AsInt32() + Vector128.ShiftLeft(
                    Vector128.ConvertToInt32(Vector128.Max(quadrant, minQuad)), 23)).AsSingle();
                var twoPow = Vector128.ShiftLeft(
                    Vector128.ConvertToInt32(quadDiff) + Vector128.Create(127), 23).AsSingle();
                poly = scaled / twoPow;   // non-denormal lanes divide by exactly 1.0 — an identity
            }
            else
            {
                poly = (poly.AsInt32() + Vector128.ShiftLeft(Vector128.ConvertToInt32(quadrant), 23)).AsSingle();
            }

            poly = Vector128.ConditionalSelect(nanMask, Vector128.Create(CanonicalNaN), poly);
            poly = Vector128.ConditionalSelect(xmaxMask, Vector128.Create(float.PositiveInfinity), poly);
            return Vector128.ConditionalSelect(xminMask, zero, poly);
        }

        /// <summary>NumPy 2.4.2's float32 exponential, 8 lanes at a time. See <see cref="Exp(float)"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Exp(Vector256<float> x)
        {
            var zero = Vector256<float>.Zero;

            var nanMask = ~Vector256.Equals(x, x);
            x = Vector256.ConditionalSelect(nanMask, zero, x);

            var xmaxMask = Vector256.GreaterThanOrEqual(x, Vector256.Create(ExpXMax));
            var xminMask = Vector256.LessThanOrEqual(x, Vector256.Create(ExpXMin));
            x = Vector256.ConditionalSelect(nanMask | xminMask | xmaxMask, zero, x);

            var magic = Vector256.Create(RintCvtMagic);
            var quadrant = MulAdd(x, Vector256.Create(Log2E), magic) - magic;

            var y = MulAdd(quadrant, Vector256.Create(CodyWaiteLogE2High), x);
            y = MulAdd(quadrant, Vector256.Create(CodyWaiteLogE2Low), y);
            y = MulAdd(quadrant, zero, y);

            var num = MulAdd(Vector256.Create(ExpP5), y, Vector256.Create(ExpP4));
            num = MulAdd(num, y, Vector256.Create(ExpP3));
            num = MulAdd(num, y, Vector256.Create(ExpP2));
            num = MulAdd(num, y, Vector256.Create(ExpP1));
            num = MulAdd(num, y, Vector256.Create(ExpP0));
            var den = MulAdd(Vector256.Create(ExpQ2), y, Vector256.Create(ExpQ1));
            den = MulAdd(den, y, Vector256.Create(ExpQ0));
            var poly = num / den;

            var minQuad = Vector256.Create(-125f);
            var denormMask = Vector256.LessThanOrEqual(quadrant, minQuad);
            if (Vector256.ExtractMostSignificantBits(denormMask) != 0)
            {
                // At least one lane scales below the smallest normal, so take NumPy's split:
                // apply 2^-125 through the exponent field and let a division make the subnormal.
                var quadDiff = Vector256.ConditionalSelect(denormMask, zero - (quadrant - minQuad), zero);
                var scaled = (poly.AsInt32() + Vector256.ShiftLeft(
                    Vector256.ConvertToInt32(Vector256.Max(quadrant, minQuad)), 23)).AsSingle();
                var twoPow = Vector256.ShiftLeft(
                    Vector256.ConvertToInt32(quadDiff) + Vector256.Create(127), 23).AsSingle();
                poly = scaled / twoPow;   // non-denormal lanes divide by exactly 1.0 — an identity
            }
            else
            {
                poly = (poly.AsInt32() + Vector256.ShiftLeft(Vector256.ConvertToInt32(quadrant), 23)).AsSingle();
            }

            poly = Vector256.ConditionalSelect(nanMask, Vector256.Create(CanonicalNaN), poly);
            poly = Vector256.ConditionalSelect(xmaxMask, Vector256.Create(float.PositiveInfinity), poly);
            return Vector256.ConditionalSelect(xminMask, zero, poly);
        }

        /// <summary>NumPy 2.4.2's float32 exponential, 16 lanes at a time. See <see cref="Exp(float)"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static Vector512<float> Exp(Vector512<float> x)
        {
            var zero = Vector512<float>.Zero;

            var nanMask = ~Vector512.Equals(x, x);
            x = Vector512.ConditionalSelect(nanMask, zero, x);

            var xmaxMask = Vector512.GreaterThanOrEqual(x, Vector512.Create(ExpXMax));
            var xminMask = Vector512.LessThanOrEqual(x, Vector512.Create(ExpXMin));
            x = Vector512.ConditionalSelect(nanMask | xminMask | xmaxMask, zero, x);

            var magic = Vector512.Create(RintCvtMagic);
            var quadrant = MulAdd(x, Vector512.Create(Log2E), magic) - magic;

            var y = MulAdd(quadrant, Vector512.Create(CodyWaiteLogE2High), x);
            y = MulAdd(quadrant, Vector512.Create(CodyWaiteLogE2Low), y);
            y = MulAdd(quadrant, zero, y);

            var num = MulAdd(Vector512.Create(ExpP5), y, Vector512.Create(ExpP4));
            num = MulAdd(num, y, Vector512.Create(ExpP3));
            num = MulAdd(num, y, Vector512.Create(ExpP2));
            num = MulAdd(num, y, Vector512.Create(ExpP1));
            num = MulAdd(num, y, Vector512.Create(ExpP0));
            var den = MulAdd(Vector512.Create(ExpQ2), y, Vector512.Create(ExpQ1));
            den = MulAdd(den, y, Vector512.Create(ExpQ0));
            var poly = num / den;

            var minQuad = Vector512.Create(-125f);
            var denormMask = Vector512.LessThanOrEqual(quadrant, minQuad);
            if (Vector512.ExtractMostSignificantBits(denormMask) != 0)
            {
                // At least one lane scales below the smallest normal, so take NumPy's split:
                // apply 2^-125 through the exponent field and let a division make the subnormal.
                var quadDiff = Vector512.ConditionalSelect(denormMask, zero - (quadrant - minQuad), zero);
                var scaled = (poly.AsInt32() + Vector512.ShiftLeft(
                    Vector512.ConvertToInt32(Vector512.Max(quadrant, minQuad)), 23)).AsSingle();
                var twoPow = Vector512.ShiftLeft(
                    Vector512.ConvertToInt32(quadDiff) + Vector512.Create(127), 23).AsSingle();
                poly = scaled / twoPow;   // non-denormal lanes divide by exactly 1.0 — an identity
            }
            else
            {
                poly = (poly.AsInt32() + Vector512.ShiftLeft(Vector512.ConvertToInt32(quadrant), 23)).AsSingle();
            }

            poly = Vector512.ConditionalSelect(nanMask, Vector512.Create(CanonicalNaN), poly);
            poly = Vector512.ConditionalSelect(xmaxMask, Vector512.Create(float.PositiveInfinity), poly);
            return Vector512.ConditionalSelect(xminMask, zero, poly);
        }


        /// <summary>NumPy 2.4.2's float32 natural logarithm, 4 lanes at a time. See <see cref="Log(float)"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static Vector128<float> Log(Vector128<float> x)
        {{
            var zero = Vector128<float>.Zero;
            var ones = Vector128.Create(1f);

            var negMask = Vector128.LessThan(x, zero);
            var zeroMask = Vector128.Equals(x, zero);
            var infMask = Vector128.Equals(x, Vector128.Create(float.PositiveInfinity));
            var nanMask = ~Vector128.Equals(x, x);
            x = Vector128.ConditionalSelect(negMask, zero, x);

            // fma_get_exponent / fma_get_mantissa. NumPy rescales subnormals by 2^100 and then
            // subtracts 100 from the exponent — computed unconditionally in its kernel, but the
            // whole thing is an identity for every normal lane, and a subnormal input to log is
            // vanishingly rare. Branching it out is the same win the exp kernel's denormal split
            // gets (~0.15 ns/element here) and is exactly equivalent lane for lane.
            var fltMin = Vector128.Create(FloatMin);
            var denormMask = Vector128.LessThan(x, fltMin);
            Vector128<float> xs, exponent;
            if (Vector128.ExtractMostSignificantBits(denormMask) != 0)
            {
                var normalMask = Vector128.GreaterThanOrEqual(x, fltMin);
                var scaled = Vector128.ConditionalSelect(normalMask, zero, x)
                           * Vector128.Create(BitConverter.UInt32BitsToSingle(TwoPower100Bits));
                xs = Vector128.ConditionalSelect(denormMask, scaled, x);
                var expD = Vector128.ConvertToSingle(
                    Vector128.ShiftRightLogical(xs.AsInt32(), 23) - Vector128.Create(0x7E));
                exponent = Vector128.ConditionalSelect(denormMask, expD - Vector128.Create(100f), expD);
            }
            else
            {
                xs = x;
                exponent = Vector128.ConvertToSingle(
                    Vector128.ShiftRightLogical(xs.AsInt32(), 23) - Vector128.Create(0x7E));
            }
            var m = ((xs.AsInt32() & Vector128.Create(0x7fffff)) | Vector128.Create(126 << 23)).AsSingle();

            // if (m <= 1/sqrt 2) {{ m *= 2; exponent -= 1 }}
            var sqrt2Mask = Vector128.LessThanOrEqual(m, Vector128.Create(Sqrt1_2));
            m = Vector128.ConditionalSelect(sqrt2Mask, m + m, m);
            exponent = Vector128.ConditionalSelect(sqrt2Mask, exponent - ones, exponent);
            m = m - ones;

            var num = MulAdd(Vector128.Create(LogP5), m, Vector128.Create(LogP4));
            num = MulAdd(num, m, Vector128.Create(LogP3));
            num = MulAdd(num, m, Vector128.Create(LogP2));
            num = MulAdd(num, m, Vector128.Create(LogP1));
            num = MulAdd(num, m, Vector128.Create(LogP0));
            var den = MulAdd(Vector128.Create(LogQ5), m, Vector128.Create(LogQ4));
            den = MulAdd(den, m, Vector128.Create(LogQ3));
            den = MulAdd(den, m, Vector128.Create(LogQ2));
            den = MulAdd(den, m, Vector128.Create(LogQ1));
            den = MulAdd(den, m, Vector128.Create(LogQ0));

            var poly = MulAdd(exponent, Vector128.Create(LogE2), num / den);

            poly = Vector128.ConditionalSelect(nanMask, Vector128.Create(CanonicalNaN), poly);
            poly = Vector128.ConditionalSelect(negMask, Vector128.Create(NegativeNaN), poly);
            poly = Vector128.ConditionalSelect(zeroMask, Vector128.Create(float.NegativeInfinity), poly);
            return Vector128.ConditionalSelect(infMask, Vector128.Create(float.PositiveInfinity), poly);
        }}

        /// <summary>NumPy 2.4.2's float32 natural logarithm, 8 lanes at a time. See <see cref="Log(float)"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Log(Vector256<float> x)
        {{
            var zero = Vector256<float>.Zero;
            var ones = Vector256.Create(1f);

            var negMask = Vector256.LessThan(x, zero);
            var zeroMask = Vector256.Equals(x, zero);
            var infMask = Vector256.Equals(x, Vector256.Create(float.PositiveInfinity));
            var nanMask = ~Vector256.Equals(x, x);
            x = Vector256.ConditionalSelect(negMask, zero, x);

            // fma_get_exponent / fma_get_mantissa. NumPy rescales subnormals by 2^100 and then
            // subtracts 100 from the exponent — computed unconditionally in its kernel, but the
            // whole thing is an identity for every normal lane, and a subnormal input to log is
            // vanishingly rare. Branching it out is the same win the exp kernel's denormal split
            // gets (~0.15 ns/element here) and is exactly equivalent lane for lane.
            var fltMin = Vector256.Create(FloatMin);
            var denormMask = Vector256.LessThan(x, fltMin);
            Vector256<float> xs, exponent;
            if (Vector256.ExtractMostSignificantBits(denormMask) != 0)
            {
                var normalMask = Vector256.GreaterThanOrEqual(x, fltMin);
                var scaled = Vector256.ConditionalSelect(normalMask, zero, x)
                           * Vector256.Create(BitConverter.UInt32BitsToSingle(TwoPower100Bits));
                xs = Vector256.ConditionalSelect(denormMask, scaled, x);
                var expD = Vector256.ConvertToSingle(
                    Vector256.ShiftRightLogical(xs.AsInt32(), 23) - Vector256.Create(0x7E));
                exponent = Vector256.ConditionalSelect(denormMask, expD - Vector256.Create(100f), expD);
            }
            else
            {
                xs = x;
                exponent = Vector256.ConvertToSingle(
                    Vector256.ShiftRightLogical(xs.AsInt32(), 23) - Vector256.Create(0x7E));
            }
            var m = ((xs.AsInt32() & Vector256.Create(0x7fffff)) | Vector256.Create(126 << 23)).AsSingle();

            // if (m <= 1/sqrt 2) {{ m *= 2; exponent -= 1 }}
            var sqrt2Mask = Vector256.LessThanOrEqual(m, Vector256.Create(Sqrt1_2));
            m = Vector256.ConditionalSelect(sqrt2Mask, m + m, m);
            exponent = Vector256.ConditionalSelect(sqrt2Mask, exponent - ones, exponent);
            m = m - ones;

            var num = MulAdd(Vector256.Create(LogP5), m, Vector256.Create(LogP4));
            num = MulAdd(num, m, Vector256.Create(LogP3));
            num = MulAdd(num, m, Vector256.Create(LogP2));
            num = MulAdd(num, m, Vector256.Create(LogP1));
            num = MulAdd(num, m, Vector256.Create(LogP0));
            var den = MulAdd(Vector256.Create(LogQ5), m, Vector256.Create(LogQ4));
            den = MulAdd(den, m, Vector256.Create(LogQ3));
            den = MulAdd(den, m, Vector256.Create(LogQ2));
            den = MulAdd(den, m, Vector256.Create(LogQ1));
            den = MulAdd(den, m, Vector256.Create(LogQ0));

            var poly = MulAdd(exponent, Vector256.Create(LogE2), num / den);

            poly = Vector256.ConditionalSelect(nanMask, Vector256.Create(CanonicalNaN), poly);
            poly = Vector256.ConditionalSelect(negMask, Vector256.Create(NegativeNaN), poly);
            poly = Vector256.ConditionalSelect(zeroMask, Vector256.Create(float.NegativeInfinity), poly);
            return Vector256.ConditionalSelect(infMask, Vector256.Create(float.PositiveInfinity), poly);
        }}

        /// <summary>NumPy 2.4.2's float32 natural logarithm, 16 lanes at a time. See <see cref="Log(float)"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static Vector512<float> Log(Vector512<float> x)
        {{
            var zero = Vector512<float>.Zero;
            var ones = Vector512.Create(1f);

            var negMask = Vector512.LessThan(x, zero);
            var zeroMask = Vector512.Equals(x, zero);
            var infMask = Vector512.Equals(x, Vector512.Create(float.PositiveInfinity));
            var nanMask = ~Vector512.Equals(x, x);
            x = Vector512.ConditionalSelect(negMask, zero, x);

            // fma_get_exponent / fma_get_mantissa. NumPy rescales subnormals by 2^100 and then
            // subtracts 100 from the exponent — computed unconditionally in its kernel, but the
            // whole thing is an identity for every normal lane, and a subnormal input to log is
            // vanishingly rare. Branching it out is the same win the exp kernel's denormal split
            // gets (~0.15 ns/element here) and is exactly equivalent lane for lane.
            var fltMin = Vector512.Create(FloatMin);
            var denormMask = Vector512.LessThan(x, fltMin);
            Vector512<float> xs, exponent;
            if (Vector512.ExtractMostSignificantBits(denormMask) != 0)
            {
                var normalMask = Vector512.GreaterThanOrEqual(x, fltMin);
                var scaled = Vector512.ConditionalSelect(normalMask, zero, x)
                           * Vector512.Create(BitConverter.UInt32BitsToSingle(TwoPower100Bits));
                xs = Vector512.ConditionalSelect(denormMask, scaled, x);
                var expD = Vector512.ConvertToSingle(
                    Vector512.ShiftRightLogical(xs.AsInt32(), 23) - Vector512.Create(0x7E));
                exponent = Vector512.ConditionalSelect(denormMask, expD - Vector512.Create(100f), expD);
            }
            else
            {
                xs = x;
                exponent = Vector512.ConvertToSingle(
                    Vector512.ShiftRightLogical(xs.AsInt32(), 23) - Vector512.Create(0x7E));
            }
            var m = ((xs.AsInt32() & Vector512.Create(0x7fffff)) | Vector512.Create(126 << 23)).AsSingle();

            // if (m <= 1/sqrt 2) {{ m *= 2; exponent -= 1 }}
            var sqrt2Mask = Vector512.LessThanOrEqual(m, Vector512.Create(Sqrt1_2));
            m = Vector512.ConditionalSelect(sqrt2Mask, m + m, m);
            exponent = Vector512.ConditionalSelect(sqrt2Mask, exponent - ones, exponent);
            m = m - ones;

            var num = MulAdd(Vector512.Create(LogP5), m, Vector512.Create(LogP4));
            num = MulAdd(num, m, Vector512.Create(LogP3));
            num = MulAdd(num, m, Vector512.Create(LogP2));
            num = MulAdd(num, m, Vector512.Create(LogP1));
            num = MulAdd(num, m, Vector512.Create(LogP0));
            var den = MulAdd(Vector512.Create(LogQ5), m, Vector512.Create(LogQ4));
            den = MulAdd(den, m, Vector512.Create(LogQ3));
            den = MulAdd(den, m, Vector512.Create(LogQ2));
            den = MulAdd(den, m, Vector512.Create(LogQ1));
            den = MulAdd(den, m, Vector512.Create(LogQ0));

            var poly = MulAdd(exponent, Vector512.Create(LogE2), num / den);

            poly = Vector512.ConditionalSelect(nanMask, Vector512.Create(CanonicalNaN), poly);
            poly = Vector512.ConditionalSelect(negMask, Vector512.Create(NegativeNaN), poly);
            poly = Vector512.ConditionalSelect(zeroMask, Vector512.Create(float.NegativeInfinity), poly);
            return Vector512.ConditionalSelect(infMask, Vector512.Create(float.PositiveInfinity), poly);
        }}


        /// <summary>NumPy 2.4.2's float32 sine, 4 lanes at a time. See <see cref="Sin(float)"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static Vector128<float> Sin(Vector128<float> x) => SinCosV(x, 0, MaxCodySin);

        /// <summary>NumPy 2.4.2's float32 cosine, 4 lanes at a time. See <see cref="Cos(float)"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static Vector128<float> Cos(Vector128<float> x) => SinCosV(x, 1, MaxCodyCos);

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static Vector128<float> SinCosV(Vector128<float> xIn, int quadrantBias, float maxCody)
        {{
            var zero = Vector128<float>.Zero;

            // NumPy blanks NaN lanes BEFORE the range test, so a NaN lane counts as "in range"
            // (its result is overwritten with NaN at the end) and never reaches the libm fallback.
            var nnan = Vector128.Equals(xIn, xIn);
            var x0 = Vector128.ConditionalSelect(nnan, xIn, zero);
            var inRange = Vector128.LessThanOrEqual(Vector128.Abs(x0), Vector128.Create(maxCody));
            var x = Vector128.ConditionalSelect(inRange & nnan, x0, zero);

            var magic = Vector128.Create(RintCvtMagic);
            var quadrant = MulAdd(x, Vector128.Create(TwoOverPi), magic) - magic;

            var r = MulAdd(quadrant, Vector128.Create(CodyWaitePio2High), x);
            r = MulAdd(quadrant, Vector128.Create(CodyWaitePio2Med), r);
            r = MulAdd(quadrant, Vector128.Create(CodyWaitePio2Low), r);
            var r2 = r * r;

            var cos = MulAdd(Vector128.Create(CosInv8), r2, Vector128.Create(CosInv6));
            cos = MulAdd(cos, r2, Vector128.Create(CosInv4));
            cos = MulAdd(cos, r2, Vector128.Create(CosInv2));
            cos = MulAdd(cos, r2, Vector128.Create(CosInv0));

            var sin = MulAdd(Vector128.Create(SinInv9), r2, Vector128.Create(SinInv7));
            sin = MulAdd(sin, r2, Vector128.Create(SinInv5));
            sin = MulAdd(sin, r2, Vector128.Create(SinInv3));
            sin = MulAdd(sin, r2, zero);
            sin = MulAdd(sin, r, r);

            var iq = Vector128.ConvertToInt32(quadrant) + Vector128.Create(quadrantBias);
            var sineMask = Vector128.Equals(iq & Vector128.Create(1), Vector128<int>.Zero).AsSingle();
            var res = Vector128.ConditionalSelect(sineMask, sin, cos);
            var negMask = Vector128.Equals(iq & Vector128.Create(2), Vector128.Create(2)).AsSingle();
            res = Vector128.ConditionalSelect(negMask, zero - res, res);
            res = Vector128.ConditionalSelect(nnan, res, Vector128.Create(CanonicalNaN));

            // Lanes past NumPy's Cody-Waite limit take the platform libm, exactly as NumPy's own
            // kernel does. Rare, so the whole-vector test keeps it off the hot path.
            var fallback = Vector128.AndNot(nnan, inRange);   // nnan AND NOT inRange
            if (Vector128.ExtractMostSignificantBits(fallback) != 0)
                res = SinCosFallback(xIn, res, fallback, quadrantBias);
            return res;
        }}

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Vector128<float> SinCosFallback(Vector128<float> xIn, Vector128<float> res, Vector128<float> mask, int quadrantBias)
        {{
            Span<float> rv = stackalloc float[Vector128<float>.Count];
            res.CopyTo(rv);
            uint bits = (uint)Vector128.ExtractMostSignificantBits(mask);
            for (int i = 0; i < rv.Length; i++)
                if ((bits & (1u << i)) != 0)
                    rv[i] = quadrantBias == 0 ? MathF.Sin(xIn[i]) : MathF.Cos(xIn[i]);
            return Vector128.Create<float>(rv);
        }}

        /// <summary>NumPy 2.4.2's float32 sine, 8 lanes at a time. See <see cref="Sin(float)"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Sin(Vector256<float> x) => SinCosV(x, 0, MaxCodySin);

        /// <summary>NumPy 2.4.2's float32 cosine, 8 lanes at a time. See <see cref="Cos(float)"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Cos(Vector256<float> x) => SinCosV(x, 1, MaxCodyCos);

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static Vector256<float> SinCosV(Vector256<float> xIn, int quadrantBias, float maxCody)
        {{
            var zero = Vector256<float>.Zero;

            // NumPy blanks NaN lanes BEFORE the range test, so a NaN lane counts as "in range"
            // (its result is overwritten with NaN at the end) and never reaches the libm fallback.
            var nnan = Vector256.Equals(xIn, xIn);
            var x0 = Vector256.ConditionalSelect(nnan, xIn, zero);
            var inRange = Vector256.LessThanOrEqual(Vector256.Abs(x0), Vector256.Create(maxCody));
            var x = Vector256.ConditionalSelect(inRange & nnan, x0, zero);

            var magic = Vector256.Create(RintCvtMagic);
            var quadrant = MulAdd(x, Vector256.Create(TwoOverPi), magic) - magic;

            var r = MulAdd(quadrant, Vector256.Create(CodyWaitePio2High), x);
            r = MulAdd(quadrant, Vector256.Create(CodyWaitePio2Med), r);
            r = MulAdd(quadrant, Vector256.Create(CodyWaitePio2Low), r);
            var r2 = r * r;

            var cos = MulAdd(Vector256.Create(CosInv8), r2, Vector256.Create(CosInv6));
            cos = MulAdd(cos, r2, Vector256.Create(CosInv4));
            cos = MulAdd(cos, r2, Vector256.Create(CosInv2));
            cos = MulAdd(cos, r2, Vector256.Create(CosInv0));

            var sin = MulAdd(Vector256.Create(SinInv9), r2, Vector256.Create(SinInv7));
            sin = MulAdd(sin, r2, Vector256.Create(SinInv5));
            sin = MulAdd(sin, r2, Vector256.Create(SinInv3));
            sin = MulAdd(sin, r2, zero);
            sin = MulAdd(sin, r, r);

            var iq = Vector256.ConvertToInt32(quadrant) + Vector256.Create(quadrantBias);
            var sineMask = Vector256.Equals(iq & Vector256.Create(1), Vector256<int>.Zero).AsSingle();
            var res = Vector256.ConditionalSelect(sineMask, sin, cos);
            var negMask = Vector256.Equals(iq & Vector256.Create(2), Vector256.Create(2)).AsSingle();
            res = Vector256.ConditionalSelect(negMask, zero - res, res);
            res = Vector256.ConditionalSelect(nnan, res, Vector256.Create(CanonicalNaN));

            // Lanes past NumPy's Cody-Waite limit take the platform libm, exactly as NumPy's own
            // kernel does. Rare, so the whole-vector test keeps it off the hot path.
            var fallback = Vector256.AndNot(nnan, inRange);   // nnan AND NOT inRange
            if (Vector256.ExtractMostSignificantBits(fallback) != 0)
                res = SinCosFallback(xIn, res, fallback, quadrantBias);
            return res;
        }}

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Vector256<float> SinCosFallback(Vector256<float> xIn, Vector256<float> res, Vector256<float> mask, int quadrantBias)
        {{
            Span<float> rv = stackalloc float[Vector256<float>.Count];
            res.CopyTo(rv);
            uint bits = (uint)Vector256.ExtractMostSignificantBits(mask);
            for (int i = 0; i < rv.Length; i++)
                if ((bits & (1u << i)) != 0)
                    rv[i] = quadrantBias == 0 ? MathF.Sin(xIn[i]) : MathF.Cos(xIn[i]);
            return Vector256.Create<float>(rv);
        }}

        /// <summary>NumPy 2.4.2's float32 sine, 16 lanes at a time. See <see cref="Sin(float)"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static Vector512<float> Sin(Vector512<float> x) => SinCosV(x, 0, MaxCodySin);

        /// <summary>NumPy 2.4.2's float32 cosine, 16 lanes at a time. See <see cref="Cos(float)"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static Vector512<float> Cos(Vector512<float> x) => SinCosV(x, 1, MaxCodyCos);

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static Vector512<float> SinCosV(Vector512<float> xIn, int quadrantBias, float maxCody)
        {{
            var zero = Vector512<float>.Zero;

            // NumPy blanks NaN lanes BEFORE the range test, so a NaN lane counts as "in range"
            // (its result is overwritten with NaN at the end) and never reaches the libm fallback.
            var nnan = Vector512.Equals(xIn, xIn);
            var x0 = Vector512.ConditionalSelect(nnan, xIn, zero);
            var inRange = Vector512.LessThanOrEqual(Vector512.Abs(x0), Vector512.Create(maxCody));
            var x = Vector512.ConditionalSelect(inRange & nnan, x0, zero);

            var magic = Vector512.Create(RintCvtMagic);
            var quadrant = MulAdd(x, Vector512.Create(TwoOverPi), magic) - magic;

            var r = MulAdd(quadrant, Vector512.Create(CodyWaitePio2High), x);
            r = MulAdd(quadrant, Vector512.Create(CodyWaitePio2Med), r);
            r = MulAdd(quadrant, Vector512.Create(CodyWaitePio2Low), r);
            var r2 = r * r;

            var cos = MulAdd(Vector512.Create(CosInv8), r2, Vector512.Create(CosInv6));
            cos = MulAdd(cos, r2, Vector512.Create(CosInv4));
            cos = MulAdd(cos, r2, Vector512.Create(CosInv2));
            cos = MulAdd(cos, r2, Vector512.Create(CosInv0));

            var sin = MulAdd(Vector512.Create(SinInv9), r2, Vector512.Create(SinInv7));
            sin = MulAdd(sin, r2, Vector512.Create(SinInv5));
            sin = MulAdd(sin, r2, Vector512.Create(SinInv3));
            sin = MulAdd(sin, r2, zero);
            sin = MulAdd(sin, r, r);

            var iq = Vector512.ConvertToInt32(quadrant) + Vector512.Create(quadrantBias);
            var sineMask = Vector512.Equals(iq & Vector512.Create(1), Vector512<int>.Zero).AsSingle();
            var res = Vector512.ConditionalSelect(sineMask, sin, cos);
            var negMask = Vector512.Equals(iq & Vector512.Create(2), Vector512.Create(2)).AsSingle();
            res = Vector512.ConditionalSelect(negMask, zero - res, res);
            res = Vector512.ConditionalSelect(nnan, res, Vector512.Create(CanonicalNaN));

            // Lanes past NumPy's Cody-Waite limit take the platform libm, exactly as NumPy's own
            // kernel does. Rare, so the whole-vector test keeps it off the hot path.
            var fallback = Vector512.AndNot(nnan, inRange);   // nnan AND NOT inRange
            if (Vector512.ExtractMostSignificantBits(fallback) != 0)
                res = SinCosFallback(xIn, res, fallback, quadrantBias);
            return res;
        }}

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Vector512<float> SinCosFallback(Vector512<float> xIn, Vector512<float> res, Vector512<float> mask, int quadrantBias)
        {{
            Span<float> rv = stackalloc float[Vector512<float>.Count];
            res.CopyTo(rv);
            uint bits = (uint)Vector512.ExtractMostSignificantBits(mask);
            for (int i = 0; i < rv.Length; i++)
                if ((bits & (1u << i)) != 0)
                    rv[i] = quadrantBias == 0 ? MathF.Sin(xIn[i]) : MathF.Cos(xIn[i]);
            return Vector512.Create<float>(rv);
        }}


        /// <summary>
        /// Whether <c>Tanh(Vector{<paramref name="vectorBits"/>}&lt;float&gt;)</c> can run its table
        /// lookup in hardware on this machine. Distinct from <see cref="IsExpVectorAccelerated"/>
        /// because tanh needs a capability the pure-arithmetic kernels do not: its coefficients are
        /// chosen PER LANE by the exponent of <c>|x|</c>, so they cannot be held in registers and the
        /// vector form must read the table — AVX2, on top of the FMA they all need. A host without it
        /// takes the scalar entry point, which computes the SAME bits one lane at a time; this gate
        /// is about speed, never results.
        ///
        /// <para><b>The 128 arm is NOT dead code — do not "simplify" it away.</b> It is tempting to
        /// argue that the emitter picks 128 only when <c>Vector256.IsHardwareAccelerated</c> is
        /// false, that this means AVX2 is absent, and therefore that <c>128 =&gt; Avx2.IsSupported</c>
        /// can never be true. That is WRONG, and the counter-example is a supported runtime knob
        /// rather than exotic hardware: with <c>DOTNET_PreferredVectorBitWidth=128</c> an ordinary
        /// AVX2 host reports <c>VectorBits = 128</c> while <c>Avx2.IsSupported</c> stays true
        /// (probed). Dropping the arm demotes tanh to the scalar loop for everyone who caps the
        /// width — a pure perf regression, invisible to any correctness gate because the bits are
        /// identical either way. A host that genuinely lacks AVX2 is still handled: the arm is false
        /// there.</para>
        /// </summary>
        public static bool IsTanhVectorAccelerated(int vectorBits) => vectorBits switch
        {
            512 => Avx2.IsSupported && Fma.IsSupported,
            256 => Avx2.IsSupported && Fma.IsSupported,
            128 => Avx2.IsSupported && Fma.IsSupported,
            _ => false
        };

        /// <summary>Cached base address of <see cref="TanhLutF32"/>. Safe to hold because the array
        /// lives on the pinned object heap (see its declaration) — it is allocated once at startup
        /// and never relocated, so no per-call <c>fixed</c> is needed in the hot loop.</summary>
        private static readonly unsafe float* TanhLutF32Ptr =
            (float*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(TanhLutF32));

        /// <summary>
        /// NumPy 2.4.2's float32 tanh, 4 lanes at a time. See <see cref="Tanh(float)"/>.
        ///
        /// <para><b>This width gathers where its 256-bit sibling transposes, and that is deliberate —
        /// the verdict flips with the lane count.</b> A subinterval is 8 consecutive floats, so the
        /// transpose reads whole rows and is unbeatable when 8 lanes each need one (see
        /// <see cref="Tanh(Vector256{float})"/>). At 4 lanes it is not: expressing this width as the
        /// low half of an 8-lane call — which would give the file a single table read — computes four
        /// lanes it throws away, and measured 0.171 ms / 17.56 ms against the gather's
        /// <b>0.139 ms / 13.58 ms</b> at 100K / 10M under <c>DOTNET_PreferredVectorBitWidth=128</c>,
        /// i.e. the gather is 1.23-1.29x faster and the composed form is barely ahead of the scalar
        /// loop (1.06x) where the gather is 1.31-1.39x. Two lookup strategies is the right answer
        /// here; both are verified bit-exact against the same table.</para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static unsafe Vector128<float> Tanh(Vector128<float> x)
        {
            if (!Avx2.IsSupported)
                return SoftwareTanh(x);

            var ndnan = x.AsInt32() & Vector128.Create(unchecked((int)TanhNdNanMaskF32));
            var saturated = Vector128.GreaterThan(ndnan, Vector128.Create(TanhSaturatedF32));

            var index = ndnan - Vector128.Create(TanhIndexBiasF32);
            index = Vector128.Max(index, Vector128<int>.Zero);
            index = Vector128.Min(index, Vector128.Create(TanhIndexClampF32));
            var row = Vector128.ShiftLeft(Vector128.ShiftRightLogical(index, 21), 3);   // subinterval * 8

            float* lut = TanhLutF32Ptr;
            var c6 = Avx2.GatherVector128(lut, row, 4);
            var c5 = Avx2.GatherVector128(lut, row + Vector128.Create(1), 4);
            var c4 = Avx2.GatherVector128(lut, row + Vector128.Create(2), 4);
            var c3 = Avx2.GatherVector128(lut, row + Vector128.Create(3), 4);
            var c2 = Avx2.GatherVector128(lut, row + Vector128.Create(4), 4);
            var c1 = Avx2.GatherVector128(lut, row + Vector128.Create(5), 4);
            var c0 = Avx2.GatherVector128(lut, row + Vector128.Create(6), 4);
            var b = Avx2.GatherVector128(lut, row + Vector128.Create(7), 4);

            var y = Vector128.Abs(x) - b;
            var r = MulAdd(c6, y, c5);
            r = MulAdd(r, y, c4);
            r = MulAdd(r, y, c3);
            r = MulAdd(r, y, c2);
            r = MulAdd(r, y, c1);
            r = MulAdd(r, y, c0);

            r = Vector128.ConditionalSelect(saturated.AsSingle(), Vector128.Create(1.0f), r);
            r = (r.AsInt32() | (x.AsInt32() & Vector128.Create(unchecked((int)0x80000000)))).AsSingle();
            return Vector128.ConditionalSelect(~Vector128.Equals(x, x), Vector128.Create(CanonicalNaN), r);
        }

        /// <summary>
        /// NumPy 2.4.2's float32 tanh, 8 lanes at a time. See <see cref="Tanh(float)"/>.
        ///
        /// <para><b>The coefficient fetch is a transpose, not a gather</b> — the one place this port
        /// deliberately departs from the shape of NumPy's own AVX2 code, and it is worth the words.
        /// A subinterval's 8 coefficients are 8 <em>consecutive</em> floats, i.e. exactly one
        /// <c>Vector256</c>, so the whole table read is 8 contiguous row loads followed by an 8x8
        /// transpose that lands each coefficient across the lanes — where a gather-per-coefficient
        /// would issue 8 <c>vgatherdps</c>, 64 separate element loads for the same 8 rows. Both were
        /// implemented and measured here, bit-identical and against the same table: transpose
        /// <b>0.071 ms</b> vs gather 0.124 ms at 100K, and <b>7.41 ms</b> vs 12.87 ms at 10M — 1.7x,
        /// and the difference between losing to NumPy and beating it. (NumPy reaches the same place
        /// differently, permuting a pre-transposed table; its layout is chosen to suit AVX-512's
        /// two-table lookup, which this host does not have.)</para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static unsafe Vector256<float> Tanh(Vector256<float> x)
        {
            if (!Avx2.IsSupported)
                return SoftwareTanh(x);

            var ndnan = x.AsInt32() & Vector256.Create(unchecked((int)TanhNdNanMaskF32));
            var saturated = Vector256.GreaterThan(ndnan, Vector256.Create(TanhSaturatedF32));

            var index = ndnan - Vector256.Create(TanhIndexBiasF32);
            index = Vector256.Max(index, Vector256<int>.Zero);
            index = Vector256.Min(index, Vector256.Create(TanhIndexClampF32));
            var row = Vector256.ShiftLeft(Vector256.ShiftRightLogical(index, 21), 3);

            float* lut = TanhLutF32Ptr;

            // Spilling the 8 row offsets through the stack looks like the obvious thing to remove —
            // it is the only stack traffic in the hot loop, and without [SkipLocalsInit] the JIT may
            // zero the buffer on every call. Both alternatives were measured THROUGH THE EMITTED
            // KERNEL (a script microbenchmark disagrees, because the JIT schedules this differently
            // when it is inlined into a C# loop instead of the IL-emitted one), 10M f32:
            //   this form                       7.39-7.50 ms
            //   + [SkipLocalsInit]              7.24-7.84 ms   -> inside run-to-run variance, no gain
            //   offsets via row.GetElement(i)   8.60 ms        -> ~16% SLOWER, clearly outside it
            // So the spill is free here and the readable form is also the fast one. Leave it.
            int* offsets = stackalloc int[Vector256<int>.Count];
            Avx.Store(offsets, row);

            // One whole subinterval per load: [c6 c5 c4 c3 c2 c1 c0 b] for that lane.
            var r0 = Avx.LoadVector256(lut + offsets[0]);
            var r1 = Avx.LoadVector256(lut + offsets[1]);
            var r2 = Avx.LoadVector256(lut + offsets[2]);
            var r3 = Avx.LoadVector256(lut + offsets[3]);
            var r4 = Avx.LoadVector256(lut + offsets[4]);
            var r5 = Avx.LoadVector256(lut + offsets[5]);
            var r6 = Avx.LoadVector256(lut + offsets[6]);
            var r7 = Avx.LoadVector256(lut + offsets[7]);

            // Standard 8x8 float transpose: pairwise interleave, then 64-bit halves, then the
            // 128-bit lane swap. After it, each vector holds one coefficient for all 8 lanes.
            var t0 = Avx.UnpackLow(r0, r1);  var t1 = Avx.UnpackHigh(r0, r1);
            var t2 = Avx.UnpackLow(r2, r3);  var t3 = Avx.UnpackHigh(r2, r3);
            var t4 = Avx.UnpackLow(r4, r5);  var t5 = Avx.UnpackHigh(r4, r5);
            var t6 = Avx.UnpackLow(r6, r7);  var t7 = Avx.UnpackHigh(r6, r7);

            var s0 = Avx.Shuffle(t0, t2, 0x44); var s1 = Avx.Shuffle(t0, t2, 0xEE);
            var s2 = Avx.Shuffle(t1, t3, 0x44); var s3 = Avx.Shuffle(t1, t3, 0xEE);
            var s4 = Avx.Shuffle(t4, t6, 0x44); var s5 = Avx.Shuffle(t4, t6, 0xEE);
            var s6 = Avx.Shuffle(t5, t7, 0x44); var s7 = Avx.Shuffle(t5, t7, 0xEE);

            var c6 = Avx.Permute2x128(s0, s4, 0x20);
            var c5 = Avx.Permute2x128(s1, s5, 0x20);
            var c4 = Avx.Permute2x128(s2, s6, 0x20);
            var c3 = Avx.Permute2x128(s3, s7, 0x20);
            var c2 = Avx.Permute2x128(s0, s4, 0x31);
            var c1 = Avx.Permute2x128(s1, s5, 0x31);
            var c0 = Avx.Permute2x128(s2, s6, 0x31);
            var b = Avx.Permute2x128(s3, s7, 0x31);

            var y = Vector256.Abs(x) - b;
            var r = MulAdd(c6, y, c5);
            r = MulAdd(r, y, c4);
            r = MulAdd(r, y, c3);
            r = MulAdd(r, y, c2);
            r = MulAdd(r, y, c1);
            r = MulAdd(r, y, c0);

            r = Vector256.ConditionalSelect(saturated.AsSingle(), Vector256.Create(1.0f), r);
            r = (r.AsInt32() | (x.AsInt32() & Vector256.Create(unchecked((int)0x80000000)))).AsSingle();
            return Vector256.ConditionalSelect(~Vector256.Equals(x, x), Vector256.Create(CanonicalNaN), r);
        }

        /// <summary>
        /// NumPy 2.4.2's float32 tanh, 16 lanes at a time — <b>composed from two 8-lane halves</b>
        /// rather than written against AVX-512 gathers. Every lane is independent here, so splitting
        /// the register is an identity, and it means the 512-bit width runs exactly the code path
        /// that was verified exhaustively rather than an untested transcription of it. The AVX-512
        /// gather would save one instruction per coefficient and is not worth an unverifiable kernel.
        /// See <see cref="Tanh(float)"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static Vector512<float> Tanh(Vector512<float> x)
            => Vector512.Create(Tanh(x.GetLower()), Tanh(x.GetUpper()));

        // Per-lane fallbacks so every width stays callable — and therefore testable — on a host
        // without the gather. Not a shipping path: IsTanhVectorAccelerated keeps them off the hot
        // loop, and they return the same bits the vector form does.

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Vector128<float> SoftwareTanh(Vector128<float> x)
        {
            Span<float> r = stackalloc float[Vector128<float>.Count];
            for (int i = 0; i < r.Length; i++)
                r[i] = Tanh(x[i]);
            return Vector128.Create<float>(r);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Vector256<float> SoftwareTanh(Vector256<float> x)
        {
            Span<float> r = stackalloc float[Vector256<float>.Count];
            for (int i = 0; i < r.Length; i++)
                r[i] = Tanh(x[i]);
            return Vector256.Create<float>(r);
        }

        // ---- fused multiply-add per width -------------------------------------------------
        // a*b + c with a SINGLE rounding. Vector{N}.FusedMultiplyAdd is .NET 9+, and NumSharp also
        // targets net8.0, so the hardware form goes through the x86 intrinsics that exist on both.
        // The scalar fallbacks keep every overload callable (and therefore testable) on a host whose
        // width is not accelerated; IsExpVectorAccelerated is what keeps them off the hot path.

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> MulAdd(Vector128<float> a, Vector128<float> b, Vector128<float> c)
            => Fma.IsSupported ? Fma.MultiplyAdd(a, b, c) : SoftwareMulAdd(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> MulAdd(Vector256<float> a, Vector256<float> b, Vector256<float> c)
            => Fma.IsSupported ? Fma.MultiplyAdd(a, b, c) : SoftwareMulAdd(a, b, c);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<float> MulAdd(Vector512<float> a, Vector512<float> b, Vector512<float> c)
            => Avx512F.IsSupported ? Avx512F.FusedMultiplyAdd(a, b, c) : SoftwareMulAdd(a, b, c);

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Vector128<float> SoftwareMulAdd(Vector128<float> a, Vector128<float> b, Vector128<float> c)
        {
            Span<float> r = stackalloc float[Vector128<float>.Count];
            for (int i = 0; i < r.Length; i++)
                r[i] = MathF.FusedMultiplyAdd(a[i], b[i], c[i]);
            return Vector128.Create<float>(r);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Vector256<float> SoftwareMulAdd(Vector256<float> a, Vector256<float> b, Vector256<float> c)
        {
            Span<float> r = stackalloc float[Vector256<float>.Count];
            for (int i = 0; i < r.Length; i++)
                r[i] = MathF.FusedMultiplyAdd(a[i], b[i], c[i]);
            return Vector256.Create<float>(r);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Vector512<float> SoftwareMulAdd(Vector512<float> a, Vector512<float> b, Vector512<float> c)
        {
            Span<float> r = stackalloc float[Vector512<float>.Count];
            for (int i = 0; i < r.Length; i++)
                r[i] = MathF.FusedMultiplyAdd(a[i], b[i], c[i]);
            return Vector512.Create<float>(r);
        }
    }
}
