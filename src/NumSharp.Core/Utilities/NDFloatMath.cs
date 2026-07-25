using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// float32 transcendental helpers backing NumSharp's <c>float32</c> unary ufuncs. Unlike the
    /// BCL's <see cref="MathF"/> (which routes to the platform libm and is only ~correctly rounded),
    /// each entry point here is a port of the kernel NumPy 2.4.2 actually runs, so the result is
    /// <b>bit-identical</b> to NumPy rather than merely within a couple of ULP.
    ///
    /// <para><b><see cref="Exp"/> = <c>simd_exp_FLOAT</c>, <see cref="Log"/> = <c>simd_log_FLOAT</c></b>
    /// (<c>numpy/_core/src/umath/loops_exponent_log.dispatch.c.src</c>, the <c>SIMD_AVX2_FMA3</c>
    /// instantiation). NumPy's own algorithm: clamp/flag the overflow and underflow ends, Cody-Waite
    /// range reduction <c>y = x - k·ln2</c> with <c>k = rint(x·log2(e))</c>, evaluate <c>exp(y)</c> as
    /// the ratio of a 5th-order over a 2nd-order Remez minimax polynomial, then scale by <c>2^k</c>.
    /// It is NOT correctly rounded — NumPy documents a max error of 2.52 ULP (at
    /// <c>x = 0xc2781e37</c>) — so reproducing NumPy means reproducing that error, operation for
    /// operation, in the same order.</para>
    ///
    /// <para><b>Why this is lane-width independent.</b> Every step is elementwise (no cross-lane
    /// reduction, shuffle or horizontal op), and the whole-vector <c>if</c> inside NumPy's
    /// <c>fma_scalef_ps</c> is followed by a per-lane blend whose non-denormal lanes are unchanged by
    /// the taken branch. This scalar entry point therefore yields the same bits the 8-lane
    /// <c>__m256</c> kernel yields — as do the <c>Vector{128,256,512}</c> overloads in
    /// <c>NDFloatMath.Simd.cs</c>, which the IL kernels prefer when the host has FMA.</para>
    ///
    /// <para><b>The FMA contraction is part of the answer, not an optimization.</b> NumPy writes the
    /// quadrant as <c>_mm256_mul_ps(x, log2e)</c> followed by <c>_mm256_add_ps(quadrant, magic)</c>,
    /// but the wheel's compiler (MSVC 19.44, the numpy==2.4.2 win-amd64 build) contracts that pair
    /// into a single <c>vfmadd</c>. The difference is observable: at <c>x = 0xc26d0e6c</c> the
    /// un-contracted product lands on the exact tie <c>-85.5</c> and the magic-constant rint takes it
    /// to <c>-86</c> (half-to-even), while the fused form keeps the extra bits of
    /// <c>-85.4999987495703</c> and rounds to <c>-85</c> — a 1-ULP difference in the result. Hence
    /// <see cref="MathF.FusedMultiplyAdd"/> here, matching the binary NumPy ships.</para>
    ///
    /// <para><b>Specials</b> (probed against 2.4.2, and a consequence of NumPy zeroing the NaN lanes
    /// <em>before</em> the range comparisons, which makes the three masks mutually exclusive): any
    /// NaN — quiet or signalling, either sign, any payload — returns the canonical
    /// <c>0x7fc00000</c>; <c>x ≥ 88.72283935546875</c> (incl. <c>+inf</c>) returns <c>+inf</c>;
    /// <c>x ≤ -103.97208404541015625</c> (incl. <c>-inf</c>) returns <c>+0</c>; <c>±0</c> returns
    /// <c>1</c>. Results between those ends denormalize gracefully through the
    /// <see cref="ScalefDenormal"/> path. NumPy additionally raises the FP overflow/underflow status
    /// flags (surfacing as a <c>RuntimeWarning</c>); NumSharp models no FP status word, so that
    /// signalling is absent — a pre-existing, engine-wide difference, not one this port introduces.
    /// </para>
    ///
    /// <para><b>Verified exhaustively:</b> all 2^32 float32 bit patterns agree with NumPy 2.4.2
    /// (chunked checksum sweep), not a sample. <b>Perf:</b> the finite path is
    /// <see cref="MethodImplOptions.AggressiveInlining"/> so the JIT folds it into the IL-emitted
    /// unary kernel with no per-element call frame; the NaN/overflow/underflow ends and the denormal
    /// scale-back live in <see cref="MethodImplOptions.NoInlining"/> cold helpers so the hot path
    /// stays inlineable.</para>
    /// </summary>
    public static partial class NDFloatMath
    {
        // ---- simd_exp_FLOAT constants (numpy/_core/src/umath/npy_simd_data.h + npy_math.h) ----

        /// <summary>Above this (inclusive) exp overflows to +inf — NumPy's <c>xmax</c>.</summary>
        private const float ExpXMax = 88.72283935546875f;

        /// <summary>At or below this exp underflows to +0 — NumPy's <c>xmin</c>.</summary>
        private const float ExpXMin = -103.97208404541015625f;

        // Cody-Waite split of ln(2): c1 is exact in float, c2 carries the residual so that
        // x - k*(c1+c2) keeps more bits than x - k*ln2 would. NumPy passes c3 = 0.
        private const float CodyWaiteLogE2High = -6.93145752e-1f;   // NPY_CODY_WAITE_LOGE_2_HIGHf
        private const float CodyWaiteLogE2Low = -1.42860677e-6f;    // NPY_CODY_WAITE_LOGE_2_LOWf

        // Remez minimax numerator (5th order) and denominator (2nd order) for exp on [0, ln2].
        private const float ExpP0 = 9.999999999980870924916e-01f;
        private const float ExpP1 = 7.257664613233124478488e-01f;
        private const float ExpP2 = 2.473615434895520810817e-01f;
        private const float ExpP3 = 5.114512081637298353406e-02f;
        private const float ExpP4 = 6.757896990527504603057e-03f;
        private const float ExpP5 = 5.082762527590693718096e-04f;
        private const float ExpQ0 = 1.000000000000000000000e+00f;
        private const float ExpQ1 = -2.742335390411667452936e-01f;
        private const float ExpQ2 = 2.159509375685829852307e-02f;

        /// <summary>
        /// NPY_RINT_CVT_MAGICf = <c>0x1.800000p+23</c>. Adding then subtracting it rounds a float
        /// whose magnitude is below 2^22 to the nearest integer (ties to even), because the addition
        /// pushes the value into the binade whose ULP is exactly 1.
        /// </summary>
        private const float RintCvtMagic = 12582912.0f;

        /// <summary>NPY_LOG2Ef — log2(e).</summary>
        private const float Log2E = 1.442695040888963407359924681001892137f;

        /// <summary>The canonical quiet NaN NumPy's <c>NPY_NANF</c> stores for every NaN input. Note
        /// this is NOT <see cref="float.NaN"/>, whose sign bit is set (<c>0xffc00000</c>).</summary>
        private const uint CanonicalNaNBits = 0x7fc00000u;

        // ---- simd_log_FLOAT constants (same NumPy sources) ----

        // Remez minimax numerator and denominator (both 5th order) for log(1+x) on the reduced range.
        private const float LogP0 = 0.000000000000000000000e+00f;
        private const float LogP1 = 9.999999999999998702752e-01f;
        private const float LogP2 = 2.112677543073053063722e+00f;
        private const float LogP3 = 1.480000633576506585156e+00f;
        private const float LogP4 = 3.808837741388407920751e-01f;
        private const float LogP5 = 2.589979117907922693523e-02f;
        private const float LogQ0 = 1.000000000000000000000e+00f;
        private const float LogQ1 = 2.612677543073109236779e+00f;
        private const float LogQ2 = 2.453006071784736363091e+00f;
        private const float LogQ3 = 9.864942958519418960339e-01f;
        private const float LogQ4 = 1.546476374983906719538e-01f;
        private const float LogQ5 = 5.875095403124574342950e-03f;

        /// <summary>NPY_LOGE2f — ln(2).</summary>
        private const float LogE2 = 0.693147180559945309417232121458176568f;

        /// <summary>NPY_SQRT1_2f — 1/sqrt(2), the mantissa split point.</summary>
        private const float Sqrt1_2 = 0.707106781186547524400844362104849039f;

        /// <summary>FLT_MIN — the smallest NORMAL float; below it NumPy rescales by 2^100.</summary>
        private const float FloatMin = 1.17549435082228750797e-38f;

        /// <summary>2^100 as a bit pattern (NumPy's <c>0x71800000</c>), the denormal rescale factor.</summary>
        private const uint TwoPower100Bits = 0x71800000u;

        /// <summary>NumPy returns a NEGATIVE NaN for a negative argument to log (<c>-NPY_NANF</c>).</summary>
        private const uint NegativeNaNBits = 0xffc00000u;

        // ---- simd_sincos_f32 constants (numpy/_core/src/umath/loops_trigonometric.dispatch.cpp,
        // the NPY_SIMD_FMA3 kernel). Hex-float literals in the original; the decimal spellings below
        // are the exact round-trips, with NumPy's own form and the bit pattern in the comment.

        private const float TwoOverPi = 0.6366197466850281f;            // 0x1.45f306p-1   0x3f22f983
        private const float CodyWaitePio2High = -1.570796012878418f;    // -0x1.921fb0p+00 0xbfc90fd8
        private const float CodyWaitePio2Med = -3.1391647326017846e-07f;// -0x1.5110b4p-22 0xb4a8885a
        private const float CodyWaitePio2Low = -5.390302529957765e-15f; // -0x1.846988p-48 0xa7c234c4

        // Cosine polynomial in x^2 (even terms).
        private const float CosInv8 = 2.4372266125283204e-05f;          // 0x1.98e616p-16  0x37cc730b
        private const float CosInv6 = -0.001388652017340064f;           // -0x1.6c06dcp-10 0xbab6036e
        private const float CosInv4 = 0.04166661947965622f;             // 0x1.55553cp-05  0x3d2aaa9e
        private const float CosInv2 = -0.5f;                            // -0x1.000000p-01 0xbf000000
        private const float CosInv0 = 1.0f;                             // 0x1.000000p+00  0x3f800000

        // Sine polynomial (T. Myklebust); NumPy documents max 0.647 ULP on [-pi/4, pi/4].
        private const float SinInv9 = 2.8404097065504175e-06f;          // 0x1.7d3bbcp-19  0x363e9dde
        private const float SinInv7 = -0.00019856491417158395f;         // -0x1.a06bbap-13 0xb95035dd
        private const float SinInv5 = 0.008333397097885609f;            // 0x1.11119ap-07  0x3c0888cd
        private const float SinInv3 = -0.1666666716337204f;             // -0x1.555556p-03 0xbe2aaaab

        /// <summary>Above this |x| Cody-Waite reduction loses accuracy and NumPy calls libc sinf.</summary>
        private const float MaxCodySin = 117435.9921875f;               // 117435.992      0x47e55dff

        /// <summary>The same threshold for cosine - NumPy uses a DIFFERENT (smaller) one.</summary>
        private const float MaxCodyCos = 71476.0625f;                   // 71476.0625      0x478b9a08


        /// <summary><see cref="CanonicalNaNBits"/> as a float, for the vector overloads' selects.</summary>
        private static readonly float CanonicalNaN = BitConverter.UInt32BitsToSingle(CanonicalNaNBits);

        /// <summary><see cref="NegativeNaNBits"/> as a float, for the vector overloads' selects.</summary>
        private static readonly float NegativeNaN = BitConverter.UInt32BitsToSingle(NegativeNaNBits);

        /// <summary>
        /// NumPy 2.4.2's <c>float32</c> exponential, bit-for-bit (port of <c>simd_exp_FLOAT</c>).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Exp(float x)
        {
            // Ordered comparisons, so a NaN fails both and joins the overflow/underflow ends in the
            // cold path. NumPy computes xmax_mask/xmin_mask AFTER blanking the NaN lanes to zero,
            // which is what makes the three masks mutually exclusive and this test equivalent.
            if (!(x > ExpXMin && x < ExpXMax))
                return ExpNonFinite(x);

            // quadrant = rint(x * log2(e)). The mul+add is FUSED (see the type remarks): that is
            // what the numpy wheel's compiler emitted, and near-tie inputs can see the difference.
            float quadrant = MathF.FusedMultiplyAdd(x, Log2E, RintCvtMagic) - RintCvtMagic;

            // Cody-Waite range reduction: y = x - quadrant*ln2, in three fused steps. The third
            // multiplier is NumPy's c3 = 0 (it passes zeros_f), kept so the operation sequence — and
            // therefore the rounding — is identical.
            float y = MathF.FusedMultiplyAdd(quadrant, CodyWaiteLogE2High, x);
            y = MathF.FusedMultiplyAdd(quadrant, CodyWaiteLogE2Low, y);
            y = MathF.FusedMultiplyAdd(quadrant, 0f, y);

            float num = MathF.FusedMultiplyAdd(ExpP5, y, ExpP4);
            num = MathF.FusedMultiplyAdd(num, y, ExpP3);
            num = MathF.FusedMultiplyAdd(num, y, ExpP2);
            num = MathF.FusedMultiplyAdd(num, y, ExpP1);
            num = MathF.FusedMultiplyAdd(num, y, ExpP0);
            float den = MathF.FusedMultiplyAdd(ExpQ2, y, ExpQ1);
            den = MathF.FusedMultiplyAdd(den, y, ExpQ0);
            float poly = num / den;

            // fma_scalef_ps: multiply by 2^quadrant by adding to the exponent field. quadrant is an
            // exact integer-valued float here (the magic-constant rint above), so the int conversion
            // is exact and truncation == the kernel's round-to-nearest cvtps_epi32.
            if (quadrant <= -125f)
                return ScalefDenormal(poly, quadrant);

            return BitConverter.Int32BitsToSingle(
                BitConverter.SingleToInt32Bits(poly) + (((int)quadrant) << 23));
        }

        /// <summary>
        /// NumPy 2.4.2's <c>float32</c> natural logarithm, bit-for-bit (port of <c>simd_log_FLOAT</c>,
        /// the sibling of <see cref="Exp(float)"/> in the same NumPy source file). Splits x into
        /// exponent and mantissa, folds the mantissa into <c>[1/sqrt 2, sqrt 2)</c>, evaluates
        /// <c>log(1+m)</c> as a 5th-over-5th-order Remez minimax ratio, and adds <c>exponent*ln2</c>
        /// with a final fused multiply-add. NumPy documents a max error of 3.83 ULP here (at
        /// <c>x = 0x3f486945</c>) — reproducing NumPy means reproducing that.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Log(float x)
        {
            // Ordered test: NaN, negatives, zero and +inf all fail it and take the cold path, which
            // is exactly NumPy's four masks. Subnormals stay on the fast path (rescaled below).
            if (!(x > 0f && x < float.PositiveInfinity))
                return LogNonFinite(x);

            float exponent = LogExponent(x);
            float m = LogMantissa(x);

            // if (m <= 1/sqrt 2) { m *= 2; exponent -= 1 } — centres the polynomial's argument.
            if (m <= Sqrt1_2)
            {
                m = m + m;
                exponent = exponent - 1f;
            }
            m = m - 1f;

            float num = MathF.FusedMultiplyAdd(LogP5, m, LogP4);
            num = MathF.FusedMultiplyAdd(num, m, LogP3);
            num = MathF.FusedMultiplyAdd(num, m, LogP2);
            num = MathF.FusedMultiplyAdd(num, m, LogP1);
            num = MathF.FusedMultiplyAdd(num, m, LogP0);
            float den = MathF.FusedMultiplyAdd(LogQ5, m, LogQ4);
            den = MathF.FusedMultiplyAdd(den, m, LogQ3);
            den = MathF.FusedMultiplyAdd(den, m, LogQ2);
            den = MathF.FusedMultiplyAdd(den, m, LogQ1);
            den = MathF.FusedMultiplyAdd(den, m, LogQ0);

            return MathF.FusedMultiplyAdd(exponent, LogE2, num / den);
        }

        /// <summary>
        /// <see cref="Log"/>'s four special values, in NumPy's precedence. Note the asymmetry a
        /// correctly-rounded libm does not have: a NaN argument yields the POSITIVE canonical NaN
        /// while a negative argument yields a NEGATIVE one (NumPy stores <c>-NPY_NANF</c> there).
        /// </summary>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static float LogNonFinite(float x)
        {
            if (float.IsNaN(x))
                return BitConverter.UInt32BitsToSingle(CanonicalNaNBits);
            if (x < 0f)
                return BitConverter.UInt32BitsToSingle(NegativeNaNBits);
            if (x == 0f)                       // both +0 and -0 (NumPy compares ordered-equal)
                return float.NegativeInfinity;
            return float.PositiveInfinity;     // +inf
        }

        /// <summary>
        /// NumPy's <c>fma_get_exponent</c>: the unbiased exponent as a float. Subnormals are first
        /// multiplied by 2^100 so their exponent field is meaningful, then 100 is subtracted back.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float LogExponent(float x)
        {
            if (x < FloatMin)
            {
                x = x * BitConverter.UInt32BitsToSingle(TwoPower100Bits);
                return (float)((BitConverter.SingleToInt32Bits(x) >>> 23) - 0x7E) - 100f;
            }
            return (float)((BitConverter.SingleToInt32Bits(x) >>> 23) - 0x7E);
        }

        /// <summary>
        /// NumPy's <c>fma_get_mantissa</c>: the mantissa re-exponented into <c>[0.5, 1)</c>. Same
        /// 2^100 rescale for subnormals — which does not disturb the mantissa bits themselves.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float LogMantissa(float x)
        {
            if (x < FloatMin)
                x = x * BitConverter.UInt32BitsToSingle(TwoPower100Bits);
            return BitConverter.Int32BitsToSingle(
                (BitConverter.SingleToInt32Bits(x) & 0x7fffff) | (126 << 23));
        }

        /// <summary>
        /// NumPy 2.4.2's <c>float32</c> sine, bit-for-bit (port of <c>simd_sincos_f32</c> with
        /// <c>SIMD_COMPUTE_SIN</c>). Cody-Waite reduction against pi/2 in three fused steps, then a
        /// sine/cosine polynomial pair selected by the quadrant.
        ///
        /// <para><b>Large arguments deliberately fall back to the platform libm</b>, exactly as
        /// NumPy does: beyond |x| = 117435.992 (sine) its own comment says Cody-Waite "becomes
        /// inaccurate and we will call libc". Those inputs therefore inherit whatever difference
        /// exists between .NET's <see cref="MathF.Sin"/> and the CRT's <c>sinf</c> - the port cannot
        /// close that, and extending the polynomial past NumPy's own limit would move us AWAY from
        /// NumPy rather than toward it.</para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sin(float x)
        {
            if (!(MathF.Abs(x) <= MaxCodySin))
                return SinOutOfRange(x);
            return SinCosCore(x, 0);
        }

        /// <summary>
        /// NumPy 2.4.2's <c>float32</c> cosine, bit-for-bit (<c>simd_sincos_f32</c> with
        /// <c>SIMD_COMPUTE_COS</c> - the same kernel, biasing the quadrant by one). See
        /// <see cref="Sin(float)"/>; note NumPy uses a SMALLER Cody-Waite limit for cosine.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Cos(float x)
        {
            if (!(MathF.Abs(x) <= MaxCodyCos))
                return CosOutOfRange(x);
            return SinCosCore(x, 1);
        }

        /// <summary>
        /// The shared reduce-and-evaluate body. <paramref name="quadrantBias"/> is 0 for sine and 1
        /// for cosine, which is how NumPy turns one kernel into both.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float SinCosCore(float x, int quadrantBias)
        {
            // quadrant = rint(x * 2/pi), through the magic constant. FUSED, like exp's quadrant -
            // the wheel's compiler contracts NumPy's separate multiply and add.
            float quadrant = MathF.FusedMultiplyAdd(x, TwoOverPi, RintCvtMagic) - RintCvtMagic;

            float r = MathF.FusedMultiplyAdd(quadrant, CodyWaitePio2High, x);
            r = MathF.FusedMultiplyAdd(quadrant, CodyWaitePio2Med, r);
            r = MathF.FusedMultiplyAdd(quadrant, CodyWaitePio2Low, r);
            float r2 = r * r;

            float cos = MathF.FusedMultiplyAdd(CosInv8, r2, CosInv6);
            cos = MathF.FusedMultiplyAdd(cos, r2, CosInv4);
            cos = MathF.FusedMultiplyAdd(cos, r2, CosInv2);
            cos = MathF.FusedMultiplyAdd(cos, r2, CosInv0);

            float sin = MathF.FusedMultiplyAdd(SinInv9, r2, SinInv7);
            sin = MathF.FusedMultiplyAdd(sin, r2, SinInv5);
            sin = MathF.FusedMultiplyAdd(sin, r2, SinInv3);
            sin = MathF.FusedMultiplyAdd(sin, r2, 0f);
            sin = MathF.FusedMultiplyAdd(sin, r, r);

            int iq = (int)quadrant + quadrantBias;      // quadrant is an exact integer float
            float res = (iq & 1) == 0 ? sin : cos;
            return (iq & 2) == 2 ? 0f - res : res;
        }

        /// <summary>NaN, infinities and the large-|x| libm fallback for <see cref="Sin(float)"/>.</summary>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static float SinOutOfRange(float x)
            => float.IsNaN(x) ? BitConverter.UInt32BitsToSingle(CanonicalNaNBits) : MathF.Sin(x);

        /// <summary>NaN, infinities and the large-|x| libm fallback for <see cref="Cos(float)"/>.</summary>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static float CosOutOfRange(float x)
            => float.IsNaN(x) ? BitConverter.UInt32BitsToSingle(CanonicalNaNBits) : MathF.Cos(x);

        /// <summary>
        /// The three ends of <see cref="Exp"/>'s domain, in NumPy's own precedence: NaN wins, then
        /// the overflow clamp, then the underflow clamp. Cold — a well-formed array of finite values
        /// never reaches it.
        /// </summary>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static float ExpNonFinite(float x)
        {
            if (float.IsNaN(x))
                return BitConverter.UInt32BitsToSingle(CanonicalNaNBits);
            if (x >= ExpXMax)       // includes +inf
                return float.PositiveInfinity;
            return 0f;              // x <= xmin, includes -inf
        }

        /// <summary>
        /// The denormal half of NumPy's <c>fma_scalef_ps</c>. Adding <c>quadrant</c> to the exponent
        /// field breaks down once the product is subnormal (<c>quadrant &lt;= -125</c>), so NumPy
        /// splits the scale: apply <c>2^-125</c> through the exponent field, then divide by
        /// <c>2^(-125 - quadrant)</c> and let the division generate the subnormal (and its rounding).
        /// </summary>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static float ScalefDenormal(float poly, float quadrant)
        {
            // 0 - (quadrant + 125) — the number of halvings the exponent field cannot express.
            // Reachable range is [0, 25] (quadrant bottoms out at -150 for x just above xmin), so
            // this never approaches the 32-bit shift wrap where C#'s masking would diverge from
            // AVX2's vpsllvd (which yields 0 for counts >= 32).
            float quadDiff = 0f - (quadrant - (-125f));
            int twoPowerDiff = 1 << (int)quadDiff;

            // NumPy's _mm256_max_ps(quadrant, -125) — on this branch quadrant <= -125, so the
            // clamped exponent is exactly -125.
            poly = BitConverter.Int32BitsToSingle(
                BitConverter.SingleToInt32Bits(poly) + (-125 << 23));
            return poly / (float)twoPowerDiff;
        }
    }
}
