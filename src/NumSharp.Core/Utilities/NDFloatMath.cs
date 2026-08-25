using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Transcendental helpers backing NumSharp's float unary ufuncs. Unlike the BCL's
    /// <see cref="MathF"/>/<see cref="Math"/> (which route to the platform libm and are only
    /// ~correctly rounded), each entry point here is a port of the kernel NumPy 2.4.2 actually runs,
    /// so the result is <b>bit-identical</b> to NumPy rather than merely within a couple of ULP.
    ///
    /// <para>Most of these are float32-only, because at float64 the platform libm already agrees
    /// with NumPy bit-for-bit on this host and there is nothing to port. <see cref="Tanh(double)"/>
    /// is the exception: NumPy ships its own kernel at <em>both</em> widths, so both diverged from
    /// the BCL and both are ported.</para>
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

        /// <summary>The float64 canonical quiet NaN — the double-precision counterpart of
        /// <see cref="CanonicalNaNBits"/>, used by <see cref="Tanh(double)"/>.</summary>
        private const ulong CanonicalNaNBitsF64 = 0x7ff8000000000000UL;

        // ---- exp2 (2^x) constants ----
        //
        // Unlike Exp/Log/Sin/Cos above, exp2 is NOT a port of a NumPy float32 SIMD kernel: NumPy's
        // vector exp2 (simd_exp2 in loops_umath_fp.dispatch) is SVML, gated on
        // NPY_HAVE_AVX512_SKX && NPY_CAN_LINK_SVML — Linux/AVX-512 only — so the numpy==2.4.2
        // win-amd64 wheel this repo targets runs the SCALAR fallback `npy_exp2f`, i.e. the MSVC CRT
        // `exp2f`, one element at a time. There is therefore no reproducible NumPy float kernel to
        // match bit-for-bit; the CRT exp2f is ~correctly-rounded and a pure-float polynomial cannot
        // reach it (float rounding of the Horner chain alone leaves ~1 ULP on a few percent of
        // inputs). So this kernel computes 2^r in DOUBLE — where the approximation error is far below
        // float precision — and rounds ONCE to float, which is correctly-rounded for every normal
        // result (scaling by the integer power 2^n is exact and cannot re-round). The residual
        // divergence is ≤ 1 ULP on ~0.2% of inputs, all in the subnormal-output band, where narrowing
        // then the exponent-field denormal split double-rounds — matching the accuracy of the old
        // `Math.Pow(2, (double)x)` path it replaces while running ~2.4x faster than NumPy's scalar
        // loop (vectorized: 8 floats -> two Vector256<double> halves -> narrow).

        /// <summary>At or above this (inclusive) exp2 overflows: 2^128 is not a finite float.</summary>
        private const float Exp2XMax = 128.0f;

        /// <summary>At or below this exp2 underflows to +0: 2^-150 rounds to zero (tie-to-even).</summary>
        private const float Exp2XMin = -150.0f;

        // Degree-8 minimax (Chebyshev) fit of 2^r on r ∈ [-0.5, 0.5]; max relative error 4.5e-12 ≈
        // 2^-37.7, ~13 bits below float32 precision, so `(float)(poly in double)` rounds as if from
        // the exact 2^r. Descending order for a Horner evaluation.
        private const double Exp2C0 = 1.32619074488454443e-06;
        private const double Exp2C1 = 1.53065969128547553e-05;
        private const double Exp2C2 = 1.54034253539828126e-04;
        private const double Exp2C3 = 1.33334640219601218e-03;
        private const double Exp2C4 = 9.61812920299949005e-03;
        private const double Exp2C5 = 5.55041092671512276e-02;
        private const double Exp2C6 = 2.40226506956126962e-01;
        private const double Exp2C7 = 6.93147180549698372e-01;
        private const double Exp2C8 = 1.00000000000094302e+00;

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
        [MethodImpl(OptimizeAndInline)]
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
        /// float32 base-2 exponential (<c>2^x</c>), the fast replacement for
        /// <c>(float)Math.Pow(2, (double)x)</c>. Unlike <see cref="Exp(float)"/> this is <b>not</b> a
        /// bit-for-bit port of a NumPy kernel — see the constants block for why (the numpy 2.4.2
        /// win-amd64 wheel runs a SCALAR <c>exp2f</c>, its SVML vector kernel being AVX-512/Linux
        /// only). It agrees with NumPy to <b>≤ 1 ULP</b> (≈0.2% of inputs differ, all producing
        /// subnormal outputs), the same accuracy class the <c>Math.Pow</c> path had, computed by
        /// evaluating <c>2^r</c> in double and rounding once.
        ///
        /// <para><b>Algorithm.</b> Split <c>x = n + r</c> with <c>n = rint(x)</c> and
        /// <c>r ∈ [-0.5, 0.5]</c>, so <c>2^x = 2^r · 2^n</c>. <c>2^r</c> is a degree-8 minimax
        /// polynomial evaluated in <b>double</b> (its 2^-37.7 approximation error is far below float
        /// precision, so the narrowing to float is correctly rounded), then scaled by the integer
        /// power <c>2^n</c> straight into the exponent field — exact, so it introduces no second
        /// rounding for a normal result. The <c>n ≤ -125</c> denormal branch reuses the exp kernel's
        /// split (<see cref="ScalefDenormal"/>): clamp the field to <c>2^-125</c> and divide.</para>
        ///
        /// <para><b>Lane-width independent, same as the exp/log ports:</b> every step is elementwise,
        /// so this scalar entry point yields the same bits as the <c>Vector{128,256,512}</c> overloads
        /// in <c>NDFloatMath.Simd.cs</c> — verified 0-diff against the vector form over the full
        /// ~7M-input accuracy corpus.</para>
        ///
        /// <para><b>Specials</b> (probed against 2.4.2): any NaN returns the canonical
        /// <c>0x7fc00000</c>; <c>x ≥ 128</c> (incl. <c>+inf</c>) returns <c>+inf</c>; <c>x ≤ -150</c>
        /// (incl. <c>-inf</c>) returns <c>+0</c>; <c>±0</c> returns <c>1</c>.</para>
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
        public static float Exp2(float x)
        {
            // Ordered test: NaN fails both compares and joins the overflow/underflow ends in the cold
            // path, exactly as the exp kernel treats its own NaN lanes.
            if (!(x > Exp2XMin && x < Exp2XMax))
                return Exp2NonFinite(x);

            // n = rint(x): an exact integer-valued float via the magic-constant round-to-even.
            float n = MathF.FusedMultiplyAdd(x, 1f, RintCvtMagic) - RintCvtMagic;
            // r = x - n, formed in double so it is exact (both operands are exactly representable and
            // |r| ≤ 0.5); this is what keeps the double poly correctly rounded when narrowed.
            double r = (double)x - (double)n;

            // 2^r — degree-8 Horner in double.
            double p = Math.FusedMultiplyAdd(Exp2C0, r, Exp2C1);
            p = Math.FusedMultiplyAdd(p, r, Exp2C2);
            p = Math.FusedMultiplyAdd(p, r, Exp2C3);
            p = Math.FusedMultiplyAdd(p, r, Exp2C4);
            p = Math.FusedMultiplyAdd(p, r, Exp2C5);
            p = Math.FusedMultiplyAdd(p, r, Exp2C6);
            p = Math.FusedMultiplyAdd(p, r, Exp2C7);
            p = Math.FusedMultiplyAdd(p, r, Exp2C8);

            float f = (float)p;                 // 2^r, correctly rounded to float (in [0.707, 1.414])
            int ni = (int)n;
            if (ni <= -125)
                return Exp2ScalefDenormal(f, ni);
            // Multiply by 2^ni by adding ni to the exponent field — exact for a normal result.
            return BitConverter.Int32BitsToSingle(BitConverter.SingleToInt32Bits(f) + (ni << 23));
        }

        /// <summary>
        /// The three ends of <see cref="Exp2(float)"/>'s domain, in NumPy's precedence: NaN, then the
        /// overflow clamp, then the underflow clamp. Cold — a well-formed array of finite magnitudes
        /// never reaches it.
        /// </summary>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static float Exp2NonFinite(float x)
        {
            if (float.IsNaN(x))
                return BitConverter.UInt32BitsToSingle(CanonicalNaNBits);
            if (x >= Exp2XMax)      // includes +inf
                return float.PositiveInfinity;
            return 0f;              // x <= xmin, includes -inf
        }

        /// <summary>
        /// <see cref="Exp2(float)"/>'s subnormal scale, the exact analogue of <see cref="ScalefDenormal"/>
        /// for the exp kernel: once <c>n ≤ -125</c> the exponent-field add would underflow, so clamp
        /// the applied exponent to <c>2^-125</c> and divide by <c>2^(-125 - n)</c>, letting the
        /// division generate the subnormal (and its rounding). Cold — only <c>x</c> below about -126
        /// reaches it. <paramref name="quadDiff"/> stays in <c>[0, 25]</c>, well clear of the 32-bit
        /// shift wrap.
        /// </summary>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static float Exp2ScalefDenormal(float f, int n)
        {
            int quadDiff = -125 - n;            // in [0, 25] for n in [-150, -125]
            f = BitConverter.Int32BitsToSingle(BitConverter.SingleToInt32Bits(f) + (-125 << 23));
            return f / (float)(1 << quadDiff);
        }

        /// <summary>
        /// NumPy 2.4.2's <c>float32</c> natural logarithm, bit-for-bit (port of <c>simd_log_FLOAT</c>,
        /// the sibling of <see cref="Exp(float)"/> in the same NumPy source file). Splits x into
        /// exponent and mantissa, folds the mantissa into <c>[1/sqrt 2, sqrt 2)</c>, evaluates
        /// <c>log(1+m)</c> as a 5th-over-5th-order Remez minimax ratio, and adds <c>exponent*ln2</c>
        /// with a final fused multiply-add. NumPy documents a max error of 3.83 ULP here (at
        /// <c>x = 0x3f486945</c>) — reproducing NumPy means reproducing that.
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
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
        [MethodImpl(OptimizeAndInline)]
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
        [MethodImpl(OptimizeAndInline)]
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
        [MethodImpl(OptimizeAndInline)]
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
        [MethodImpl(OptimizeAndInline)]
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
        [MethodImpl(OptimizeAndInline)]
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

        // ---- simd_tanh_f32 / simd_tanh_f64 (loops_hyperbolic.dispatch.cpp.src) ----

        /// <summary>
        /// NumPy's <c>lut8x32</c> verbatim: 32 subintervals of 8 coefficients each, in the order the
        /// kernel loads them — <c>[c6, c5, c4, c3, c2, c1, c0, b]</c>. Kept as raw bit patterns
        /// because that is how NumPy's source spells them, which is what makes them checkable against
        /// it line by line; <see cref="TanhLutF32"/> reinterprets them once at startup so the hot path
        /// indexes floats directly.
        /// </summary>
        private static readonly uint[] TanhLutF32Bits =
        {
            // [0] c6 c5 c4 c3 c2 c1 c0 b
            0xbc0e2f66U, 0x3e0910e9U, 0xb76dd6b9U, 0xbeaaaaa5U, 0xb0343c7bU, 0x3f800000U, 0x00000000U, 0x00000000U,
            // [1] c6 c5 c4 c3 c2 c1 c0 b
            0x460bda12U, 0x43761143U, 0xbe1c276dU, 0xbeab0612U, 0xbd6ee69dU, 0x3f7f1f84U, 0x3d6fb9c9U, 0x3d700000U,
            // [2] c6 c5 c4 c3 c2 c1 c0 b
            0x43d638efU, 0x4165ecdcU, 0x3c1dcf2fU, 0xbea7f01fU, 0xbd8f0da7U, 0x3f7ebd11U, 0x3d8fc35fU, 0x3d900000U,
            // [3] c6 c5 c4 c3 c2 c1 c0 b
            0xc3e11c3eU, 0xc190f756U, 0x3dc1a78dU, 0xbea4e120U, 0xbdae477dU, 0x3f7e1e5fU, 0x3daf9169U, 0x3db00000U,
            // [4] c6 c5 c4 c3 c2 c1 c0 b
            0xc2baa4e9U, 0xc08c097dU, 0x3d96f985U, 0xbea387b7U, 0xbdcd2a1fU, 0x3f7d609fU, 0x3dcf49abU, 0x3dd00000U,
            // [5] c6 c5 c4 c3 c2 c1 c0 b
            0xc249da2dU, 0xc02ba813U, 0x3da2b61bU, 0xbea15962U, 0xbdeba80dU, 0x3f7c842dU, 0x3deee849U, 0x3df00000U,
            // [6] c6 c5 c4 c3 c2 c1 c0 b
            0xc1859b82U, 0xbf7f6bdaU, 0x3dc13397U, 0xbe9d57f7U, 0xbe0c443bU, 0x3f7b00e5U, 0x3e0f0ee8U, 0x3e100000U,
            // [7] c6 c5 c4 c3 c2 c1 c0 b
            0x40dd5b57U, 0x3f2b1dc0U, 0x3dd2f670U, 0xbe976b5aU, 0xbe293cf3U, 0x3f789580U, 0x3e2e4984U, 0x3e300000U,
            // [8] c6 c5 c4 c3 c2 c1 c0 b
            0x40494640U, 0x3ece105dU, 0x3df48a0aU, 0xbe90230dU, 0xbe44f282U, 0x3f75b8adU, 0x3e4d2f8eU, 0x3e500000U,
            // [9] c6 c5 c4 c3 c2 c1 c0 b
            0x40c730a8U, 0x3f426a94U, 0x3e06c5a8U, 0xbe880dffU, 0xbe5f3651U, 0x3f726fd9U, 0x3e6bb32eU, 0x3e700000U,
            // [10] c6 c5 c4 c3 c2 c1 c0 b
            0xbf0f160eU, 0xbadb0dc4U, 0x3e1a3abaU, 0xbe7479b3U, 0xbe81c7c0U, 0x3f6cc59bU, 0x3e8c51cdU, 0x3e900000U,
            // [11] c6 c5 c4 c3 c2 c1 c0 b
            0x3e30e76fU, 0x3da43b17U, 0x3e27c405U, 0xbe4c3d88U, 0xbe96d7caU, 0x3f63fb92U, 0x3ea96163U, 0x3eb00000U,
            // [12] c6 c5 c4 c3 c2 c1 c0 b
            0xbea81387U, 0xbd51ab88U, 0x3e2e78d0U, 0xbe212482U, 0xbea7fb8eU, 0x3f59ff97U, 0x3ec543f1U, 0x3ed00000U,
            // [13] c6 c5 c4 c3 c2 c1 c0 b
            0xbdb26a1cU, 0xbcaea23dU, 0x3e2c3e44U, 0xbdeb8cbaU, 0xbeb50e9eU, 0x3f4f11d7U, 0x3edfd735U, 0x3ef00000U,
            // [14] c6 c5 c4 c3 c2 c1 c0 b
            0xbd351e57U, 0xbd3b6d8dU, 0x3e1d3097U, 0xbd5e78adU, 0xbec12efeU, 0x3f3d7573U, 0x3f028438U, 0x3f100000U,
            // [15] c6 c5 c4 c3 c2 c1 c0 b
            0xbb4c01a0U, 0xbd6caaadU, 0x3df4a8f4U, 0x3c6b5e6eU, 0xbec4be92U, 0x3f24f360U, 0x3f18abf0U, 0x3f300000U,
            // [16] c6 c5 c4 c3 c2 c1 c0 b
            0x3c1d7bfbU, 0xbd795bedU, 0x3da38508U, 0x3d839143U, 0xbebce070U, 0x3f0cbfe7U, 0x3f2bc480U, 0x3f500000U,
            // [17] c6 c5 c4 c3 c2 c1 c0 b
            0x3c722cd1U, 0xbd5fdddaU, 0x3d31416aU, 0x3dc21ee1U, 0xbead510eU, 0x3eec1a69U, 0x3f3bec1cU, 0x3f700000U,
            // [18] c6 c5 c4 c3 c2 c1 c0 b
            0x3c973f1cU, 0xbd038f3bU, 0x3b562657U, 0x3de347afU, 0xbe8ef7d6U, 0x3eb0a801U, 0x3f4f2e5bU, 0x3f900000U,
            // [19] c6 c5 c4 c3 c2 c1 c0 b
            0x3c33a31bU, 0xbc1cad63U, 0xbcaeeac9U, 0x3dcbec96U, 0xbe4b8704U, 0x3e6753a2U, 0x3f613c53U, 0x3fb00000U,
            // [20] c6 c5 c4 c3 c2 c1 c0 b
            0x3b862ef4U, 0x3abb4766U, 0xbcce9419U, 0x3d99ef2dU, 0xbe083237U, 0x3e132f1aU, 0x3f6ce37dU, 0x3fd00000U,
            // [21] c6 c5 c4 c3 c2 c1 c0 b
            0x3a27b3d0U, 0x3b95f10bU, 0xbcaaeac4U, 0x3d542ea1U, 0xbdaf7449U, 0x3db7e7d3U, 0x3f743c4fU, 0x3ff00000U,
            // [22] c6 c5 c4 c3 c2 c1 c0 b
            0xba3b5907U, 0x3b825873U, 0xbc49e7d0U, 0x3cdde701U, 0xbd2e1ec4U, 0x3d320845U, 0x3f7a5febU, 0x40100000U,
            // [23] c6 c5 c4 c3 c2 c1 c0 b
            0xba0efc22U, 0x3afaea66U, 0xbba71dddU, 0x3c2cca67U, 0xbc83bf06U, 0x3c84d3d4U, 0x3f7dea85U, 0x40300000U,
            // [24] c6 c5 c4 c3 c2 c1 c0 b
            0xb97f9f0fU, 0x3a49f878U, 0xbb003b0eU, 0x3b81cb27U, 0xbbc3e0b5U, 0x3bc477b7U, 0x3f7f3b3dU, 0x40500000U,
            // [25] c6 c5 c4 c3 c2 c1 c0 b
            0xb8c8af50U, 0x39996bf3U, 0xba3f9a05U, 0x3ac073a1U, 0xbb10aadcU, 0x3b10d3daU, 0x3f7fb78cU, 0x40700000U,
            // [26] c6 c5 c4 c3 c2 c1 c0 b
            0xb7bdddfbU, 0x388f3e6cU, 0xb92c08a7U, 0x39ac3032U, 0xba0157dbU, 0x3a01601eU, 0x3f7fefd4U, 0x40900000U,
            // [27] c6 c5 c4 c3 c2 c1 c0 b
            0xb64f2950U, 0x371bb0e3U, 0xb7ba9232U, 0x383a94d9U, 0xb88c18f2U, 0x388c1a3bU, 0x3f7ffdd0U, 0x40b00000U,
            // [28] c6 c5 c4 c3 c2 c1 c0 b
            0xb4e085b1U, 0x35a8a5e6U, 0xb64a0b0fU, 0x36ca081dU, 0xb717b096U, 0x3717b0daU, 0x3f7fffb4U, 0x40d00000U,
            // [29] c6 c5 c4 c3 c2 c1 c0 b
            0xb3731dfaU, 0x34369b17U, 0xb4dac169U, 0x355abd4cU, 0xb5a43baeU, 0x35a43bceU, 0x3f7ffff6U, 0x40f00000U,
            // [30] c6 c5 c4 c3 c2 c1 c0 b
            0xb15a1f04U, 0x322487b0U, 0xb2ab78acU, 0x332b3cb6U, 0xb383012cU, 0x338306c6U, 0x3f7fffffU, 0x41100000U,
            // [31] c6 c5 c4 c3 c2 c1 c0 b
            0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U, 0x3f800000U, 0x00000000U,
        };

        /// <summary>
        /// NumPy's <c>lut18x16</c> verbatim: 16 subintervals of 18 coefficients each, ordered
        /// <c>[b, c0, c1, ..., c16]</c>. See <see cref="TanhLutF32Bits"/> for why the bits are stored
        /// rather than decimal literals.
        /// </summary>
        private static readonly ulong[] TanhLutF64Bits =
        {
            // [0] b c0..c16
            0x0000000000000000UL, 0x0000000000000000UL, 0x3ff0000000000000UL, 0xbbf0b3ea3fdfaa19UL, 0xbfd5555555555555UL, 0xbce6863ee44ed636UL,
            0x3fc1111111112ab5UL, 0xbda1ea19ddddb3b4UL, 0xbfaba1ba1990520bUL, 0xbe351ca7f096011fUL, 0x3f9664f94e6ac14eUL, 0xbea8c4c1fd7852feUL,
            0xbf822404577aa9ddUL, 0xbefdd99a221ed573UL, 0x3f6e3be689423841UL, 0xbf2a1306713a4f3aUL, 0xbf55d7e76dc56871UL, 0x3f35e67ab76a26e7UL,
            // [1] b c0..c16
            0x3fcc000000000000UL, 0x3fcb8fd0416a7c92UL, 0x3fee842ca3f08532UL, 0xbfca48aaeb53bc21UL, 0xbfd183afc292ba11UL, 0x3fc04dcd0476c75eUL,
            0x3fb5c19efdfc08adUL, 0xbfb0b8df995ce4dfUL, 0xbf96e37bba52f6fcUL, 0x3f9eaaf3320c3851UL, 0xbf94d3343bae39ddUL, 0xbfccce16b1046f13UL,
            0x403d8b07f7a82aa3UL, 0x4070593a3735bab4UL, 0xc0d263511f5baac1UL, 0xc1045e509116b066UL, 0x41528c38809c90c7UL, 0x41848ee0627d8206UL,
            // [2] b c0..c16
            0x3fd4000000000000UL, 0x3fd35f98a0ea650eUL, 0x3fed11574af58f1bUL, 0xbfd19921f4329916UL, 0xbfcc1a4b039c9bfaUL, 0x3fc43d3449a80f08UL,
            0x3fa74c98dc34fbacUL, 0xbfb2955cf41e8164UL, 0x3ecff7df18455399UL, 0x3f9cf823fe761fc1UL, 0xbf7bc748e60df843UL, 0xbf81a16f224bb7b6UL,
            0xbf9f44ab92fbab0aUL, 0xbfccab654e44835eUL, 0x40169f73b15ebe5cUL, 0x4041fab9250984ceUL, 0xc076d57fb5190b02UL, 0xc0a216d618b489ecUL,
            // [3] b c0..c16
            0x3fdc000000000000UL, 0x3fda5729ee488037UL, 0x3fea945b9c24e4f9UL, 0xbfd5e0f09bef8011UL, 0xbfc16e1e6d8d0be6UL, 0x3fc5c26f3699b7e7UL,
            0xbf790d6a8eff0a77UL, 0xbfaf9d05c309f7c6UL, 0x3f97362834d33a4eUL, 0x3f9022271754ff1fUL, 0xbf8c89372b43ba85UL, 0xbf62cbf00406bc09UL,
            0x3fb2eac604473d6aUL, 0x3fd13ed80037dbacUL, 0xc025c1dd41cd6cb5UL, 0xc0458d090ec3de95UL, 0x4085f09f888f8adaUL, 0x40a5b89107c8af4fUL,
            // [4] b c0..c16
            0x3fe4000000000000UL, 0x3fe1bf47eabb8f95UL, 0x3fe6284c3374f815UL, 0xbfd893b59c35c882UL, 0xbf92426c751e48a2UL, 0x3fc1a686f6ab2533UL,
            0xbfac3c021789a786UL, 0xbf987d27ccff4291UL, 0x3f9e7f8380184b45UL, 0xbf731fe77c9c60afUL, 0xbf8129a092de747aUL, 0x3f75b29bb02cf69bUL,
            0x3f45f87d903aaac8UL, 0xbf6045b9076cc487UL, 0xbf58fd89fe05e0d1UL, 0xbf74949d60113d63UL, 0x3fa246332a2fcba5UL, 0x3fb69d8374520edaUL,
            // [5] b c0..c16
            0x3fec000000000000UL, 0x3fe686650b8c2015UL, 0x3fe02500a09f8d6eUL, 0xbfd6ba7cb7576538UL, 0x3fb4f152b2bad124UL, 0x3faf203c316ce730UL,
            0xbfae2196b7326859UL, 0x3f8b2ca62572b098UL, 0x3f869543e7c420d4UL, 0xbf84a6046865ec7dUL, 0x3f60c85b4d538746UL, 0x3f607df0f9f90c17UL,
            0xbf5e104671036300UL, 0x3f2085ee7e8ac170UL, 0x3f73f7af01d5af7aUL, 0x3f7c9fd6200d0adeUL, 0xbfb29d851a896fcdUL, 0xbfbded519f981716UL,
            // [6] b c0..c16
            0x3ff4000000000000UL, 0x3feb2523bb6b2deeUL, 0x3fd1f25131e3a8c0UL, 0xbfce7291743d7555UL, 0x3fbbba40cbef72beUL, 0xbf89c7a02788557cUL,
            0xbf93a7a011ff8c2aUL, 0x3f8f1cf6c7f5b00aUL, 0xbf7326bd4914222aUL, 0xbf4ca3f1f2b9192bUL, 0x3f5be9392199ec18UL, 0xbf4b852a6e0758d5UL,
            0x3f19bc98ddf0f340UL, 0x3f23524622610430UL, 0xbf1e40bdead17e6bUL, 0x3f02cd40e0ad0a9fUL, 0x3ed9065ae369b212UL, 0xbef02d288b5b3371UL,
            // [7] b c0..c16
            0x3ffc000000000000UL, 0x3fee1fbf97e33527UL, 0x3fbd22ca1c24a139UL, 0xbfbb6d85a01efb80UL, 0x3fb01ba038be6a3dUL, 0xbf98157e26e0d541UL,
            0x3f6e4709c7e8430eUL, 0x3f60379811e43dd5UL, 0xbf5fc15b0a9d98faUL, 0x3f4c77dee0afd227UL, 0xbf2a0c68a4489f10UL, 0xbf0078c63d1b8445UL,
            0x3f0d4304bc9246e8UL, 0xbeff12a6626911b4UL, 0x3ee224cd6c4513e5UL, 0xbe858ab8e019f311UL, 0xbeb8e1ba4c98a030UL, 0x3eb290981209c1a6UL,
            // [8] b c0..c16
            0x4004000000000000UL, 0x3fef9258260a71c2UL, 0x3f9b3afe1fba5c76UL, 0xbf9addae58c7141aUL, 0x3f916df44871efc8UL, 0xbf807b55c1c7d278UL,
            0x3f67682afa611151UL, 0xbf4793826f78537eUL, 0x3f14cffcfa69fbb6UL, 0x3f04055bce68597aUL, 0xbf00462601dc2faaUL, 0x3eec12eadd55be7aUL,
            0xbed13c415f7b9d41UL, 0x3eab9008bca408afUL, 0xbe24b645e68eeaa3UL, 0xbe792fa6323b7cf8UL, 0x3e6ffd0766ad4016UL, 0xbe567e924bf5ff6eUL,
            // [9] b c0..c16
            0x400c000000000000UL, 0x3feff112c63a9077UL, 0x3f6dd37d19b22b21UL, 0xbf6dc59376c7aa19UL, 0x3f63c6869dfc8870UL, 0xbf53a18d5843190fUL,
            0x3f3ef2ee77717cbfUL, 0xbf2405695e36240fUL, 0x3f057e48e5b79d10UL, 0xbee2bf0cb4a71647UL, 0x3eb7b6a219dea9f4UL, 0xbe6fa600f593181bUL,
            0xbe722b8d9720cdb0UL, 0x3e634df71865f620UL, 0xbe4abfebfb72bc83UL, 0x3e2df04d67876402UL, 0xbe0c63c29f505f5bUL, 0x3de3f7f7de6b0eb6UL,
            // [10] b c0..c16
            0x4014000000000000UL, 0x3fefff419668df11UL, 0x3f27ccec13a9ef96UL, 0xbf27cc5e74677410UL, 0x3f1fb9aef915d828UL, 0xbf0fb6bbc89b1a5bUL,
            0x3ef95a4482f180b7UL, 0xbee0e08de39ce756UL, 0x3ec33b66d7d77264UL, 0xbea31eaafe73efd5UL, 0x3e80cbcc8d4c5c8aUL, 0xbe5a3c935dce3f7dUL,
            0x3e322666d739bec0UL, 0xbe05bb1bcf83ca73UL, 0x3dd51c38f8695ed3UL, 0xbd95c72be95e4d2cUL, 0xbd7fab216b9e0e49UL, 0x3d69ed18bae3ebbcUL,
            // [11] b c0..c16
            0x401c000000000000UL, 0x3feffffc832750f2UL, 0x3ecbe6c3f33250aeUL, 0xbecbe6c0e8b4cc87UL, 0x3ec299d1e27c6e11UL, 0xbeb299c9c684a963UL,
            0x3e9dc2c27da3b603UL, 0xbe83d709ba5f714eUL, 0x3e66ac4e578b9b10UL, 0xbe46abb02c4368edUL, 0x3e2425bb231a5e29UL, 0xbe001c6d95e3ae96UL,
            0x3dd76a553d7e7918UL, 0xbdaf2ac143fb6762UL, 0x3d8313ac38c6832bUL, 0xbd55a89c30203106UL, 0x3d2826b62056aa27UL, 0xbcf7534c4f3dfa71UL,
            // [12] b c0..c16
            0x4024000000000000UL, 0x3feffffffdc96f35UL, 0x3e41b4865394f75fUL, 0xbe41b486526b0565UL, 0x3e379b5ddcca334cUL, 0xbe279b5dd4fb3d01UL,
            0x3e12e2afd9f7433eUL, 0xbdf92e3fc5ee63e0UL, 0x3ddcc74b8d3d5c42UL, 0xbdbcc749ca8079ddUL, 0x3d9992a4beac8662UL, 0xbd74755a00ea1fd3UL,
            0x3d4de0fa59416a39UL, 0xbd23eae52a3dbf57UL, 0x3cf7787935626685UL, 0xbccad6b3bb9eff65UL, 0x3ca313e31762f523UL, 0xbc730b73f1eaff20UL,
            // [13] b c0..c16
            0x402c000000000000UL, 0x3fefffffffffcf58UL, 0x3d8853f01bda5f28UL, 0xbd8853f01bef63a4UL, 0x3d8037f57bc62c9aUL, 0xbd7037f57ae72aa6UL,
            0x3d59f320348679baUL, 0xbd414cc030f2110eUL, 0x3d23c589137f92b4UL, 0xbd03c5883836b9d2UL, 0x3ce191ba5ed3fb67UL, 0xbcbc1c6c063bb7acUL,
            0x3c948716cf3681b4UL, 0xbc6b5e3e9ca0955eUL, 0x3c401ffc49c6bc29UL, 0xbc12705ccd3dd884UL, 0x3bea37aa21895319UL, 0xbbba2cff8135d462UL,
            // [14] b c0..c16
            0x4034000000000000UL, 0x3ff0000000000000UL, 0x3c73953c0197ef58UL, 0xbc73955be519be31UL, 0x3c6a2d4b50a2cff7UL, 0xbc5a2ca2bba78e86UL,
            0x3c44b61d9bbcc940UL, 0xbc2ba022e8d82a87UL, 0x3c107f8e2c8707a1UL, 0xbbf07a5416264aecUL, 0x3bc892450bad44c4UL, 0xbba3be9a4460fe00UL,
            0x3b873f9f2d2fda99UL, 0xbb5eca68e2c1ba2eUL, 0xbabf0b21acfa52abUL, 0xba8e0a4c47ae75f5UL, 0x3ae5c7f1fd871496UL, 0xbab5a71b5f7d9035UL,
            // [15] b c0..c16
            0x0000000000000000UL, 0x3ff0000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL,
            0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL,
            0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL,
        };

        /// <summary>The float view of <see cref="TanhLutF32Bits"/> (NumPy casts the same storage to
        /// <c>const float*</c>). Initialized after it — static field initializers run in textual
        /// order, so the declarations above must stay above this one.
        ///
        /// <para>Allocated on the <b>pinned</b> object heap so its address never moves: the vector
        /// overloads reach it through an AVX2 gather, which needs a raw <c>float*</c>, and pinning it
        /// once at startup is what lets them cache that pointer instead of re-pinning per call.</para>
        /// </summary>
        internal static readonly float[] TanhLutF32 = ReinterpretToSingle(TanhLutF32Bits, pinned: true);

        /// <summary>The double view of <see cref="TanhLutF64Bits"/>.</summary>
        private static readonly double[] TanhLutF64 = ReinterpretToDouble(TanhLutF64Bits);

        private static float[] ReinterpretToSingle(uint[] bits, bool pinned = false)
        {
            var result = pinned ? GC.AllocateArray<float>(bits.Length, pinned: true) : new float[bits.Length];
            for (int i = 0; i < bits.Length; i++)
                result[i] = BitConverter.UInt32BitsToSingle(bits[i]);
            return result;
        }

        private static double[] ReinterpretToDouble(ulong[] bits)
        {
            var result = new double[bits.Length];
            for (int i = 0; i < bits.Length; i++)
                result[i] = BitConverter.UInt64BitsToDouble(bits[i]);
            return result;
        }

        // The exponent-field mask, saturation threshold, index bias and index clamp, straight from
        // the kernel. `ndnan` keeps the exponent plus the top mantissa bits and drops the sign, so it
        // is always non-negative and one signed compare separates the polynomial's domain from the
        // saturated/inf/NaN end.
        private const uint TanhNdNanMaskF32 = 0x7fe00000u;
        private const int TanhSaturatedF32 = 0x7f000000;
        private const int TanhIndexBiasF32 = 0x3d400000;
        private const int TanhIndexClampF32 = 0x3e00000;

        private const ulong TanhNdNanMaskF64 = 0x7ff8000000000000UL;
        private const long TanhSaturatedF64 = 0x7fe0000000000000L;
        private const long TanhIndexBiasF64 = 0x3fc0000000000000L;
        private const int TanhIndexClampF64 = 0x780000;

        /// <summary>
        /// <c>simd_tanh_f32</c> — NumPy's table-driven tanh, converted by NumPy from Intel SVML's
        /// <c>svml_z0_tanh_s_la</c>. tanh is odd, so the kernel works on <c>|x|</c> and re-applies the
        /// sign bit at the end. <c>[0, SATURATION_THRESHOLD)</c> is split into 32 subintervals; the
        /// exponent field of <c>|x|</c> indexes straight into a table of per-subinterval minimax
        /// coefficients and a recentring offset <c>b</c>, and the result is a degree-6 Horner
        /// evaluation of <c>P(|x| - b)</c>. The top subintervals hold <c>1 + 0·y + 0·y² …</c>, which is
        /// how the saturated range returns exactly 1 through the same arithmetic.
        ///
        /// <para><b>Why this is lane-width independent.</b> Every step is elementwise; the only
        /// width-dependent code in NumPy is <em>how</em> the table is fetched (a 4×4 lane transpose on
        /// SSE4, <c>TwoTablesLookupLanes</c> on AVX2/AVX-512, a gather elsewhere), and all three fetch
        /// the same coefficients. So this scalar entry point, the
        /// <c>Vector{128,256,512}</c> overloads in <c>NDFloatMath.Simd.cs</c>, and NumPy's own kernel
        /// at any width all produce identical bits.</para>
        ///
        /// <para><b>Unlike the exp/log/sincos ports, the FMA here is not a host pin.</b> Those had to
        /// reproduce MSVC's contraction of a separate multiply and add; this kernel spells its Horner
        /// steps as <c>hn::MulAdd</c> — an explicit fused multiply-add — so
        /// <see cref="MathF.FusedMultiplyAdd"/> is a literal transcription rather than a guess about
        /// the compiler.</para>
        ///
        /// <para><b>Specials</b> (probed against 2.4.2): any NaN — quiet or signalling, either sign,
        /// any payload — returns the canonical <c>0x7fc00000</c>; <c>±inf</c> and every <c>|x|</c> past
        /// the saturation threshold return <c>±1</c>; <c>±0</c> returns <c>±0</c> (subinterval 0 has
        /// <c>b = c0 = 0</c> and <c>c1 = 1</c>, so the polynomial collapses to <c>|x|</c> and the sign
        /// bit is OR-ed back). NumPy also clears the FP status word around this loop; NumSharp models
        /// no FP status, a pre-existing engine-wide difference this port does not change.</para>
        ///
        /// <para><b>Verified exhaustively:</b> all 2^32 float32 bit patterns agree with NumPy 2.4.2,
        /// through both this entry point and the SIMD kernel.</para>
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
        public static float Tanh(float x)
        {
            uint bits = BitConverter.SingleToUInt32Bits(x);
            int ndnan = (int)(bits & TanhNdNanMaskF32);

            // NaN and inf both have an all-ones exponent, so they land here too and never reach the
            // polynomial — which is what lets the cold helper own every special case.
            if (ndnan > TanhSaturatedF32)
                return TanhSaturated(x, bits);

            int index = ndnan - TanhIndexBiasF32;
            index = Math.Max(index, 0);
            index = Math.Min(index, TanhIndexClampF32);
            int offset = (int)((uint)index >> 21) * 8;

            ref float c = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(TanhLutF32), offset);
            float y = MathF.Abs(x) - Unsafe.Add(ref c, 7);              // |x| - b
            float r = MathF.FusedMultiplyAdd(c, y, Unsafe.Add(ref c, 1));           // c6·y + c5
            r = MathF.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 2));                 // ·y + c4
            r = MathF.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 3));                 // ·y + c3
            r = MathF.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 4));                 // ·y + c2
            r = MathF.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 5));                 // ·y + c1
            r = MathF.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 6));                 // ·y + c0

            // tanh(|x|) is non-negative, so re-applying the sign is an OR, exactly as NumPy does it.
            return BitConverter.UInt32BitsToSingle(
                BitConverter.SingleToUInt32Bits(r) | (bits & 0x80000000u));
        }

        /// <summary>
        /// The saturated end of <see cref="Tanh(float)"/>: NaN first, then <c>±1</c> for <c>±inf</c>
        /// and for every finite <c>|x|</c> above the threshold. Cold — an ordinary array of small
        /// magnitudes never reaches it.
        /// </summary>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static float TanhSaturated(float x, uint bits)
        {
            if (float.IsNaN(x))
                return BitConverter.UInt32BitsToSingle(CanonicalNaNBits);
            return BitConverter.UInt32BitsToSingle(0x3f800000u | (bits & 0x80000000u));
        }

        /// <summary>
        /// <c>simd_tanh_f64</c> — the double-precision half of the same kernel: 16 subintervals and a
        /// degree-16 Horner evaluation. See <see cref="Tanh(float)"/> for the algorithm, the
        /// lane-width argument and the specials; the only structural difference is the index
        /// extraction. NumPy computes it with a 32-bit max/min over the 64-bit lanes, noting it
        /// "is fine … since we're not crossing 32-bit edge": the mask and the bias both have a zero
        /// low half, so the subtraction cannot borrow and the whole index lives in the high 32 bits.
        /// This port clamps that high half directly, which is the same arithmetic without the
        /// bitcast.
        ///
        /// <para><b>That re-spelling is the one line here a reader cannot diff against NumPy's, so
        /// it is PROVEN rather than argued</b> — and proven exhaustively, not sampled.
        /// <c>ndnan = bits &amp; 0x7ff8000000000000</c> keeps only 11 exponent bits plus mantissa bit
        /// 51 and clears everything else, and both formulas depend on nothing else, so the input
        /// domain is exactly 2^12 = 4096 values. <c>TanhFloat64IndexExtraction_MatchesNumPySpelling</c>
        /// (Math/np.Tanh.IndexExtraction.Test.cs) enumerates all 4096 and asserts NumPy's spelling and
        /// this one agree on every single one — and separately confirms that real double bit patterns
        /// actually reach all 4096, so the domain is exact rather than an over-approximation.</para>
        ///
        /// <para>Verified against NumPy 2.4.2 over a 300K-value structured sweep (specials,
        /// subnormals, both tails, interior) and the committed corpus tier; float64 has no exhaustive
        /// analogue.</para>
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
        public static double Tanh(double x)
        {
            ulong bits = BitConverter.DoubleToUInt64Bits(x);
            long ndnan = (long)(bits & TanhNdNanMaskF64);

            if (ndnan > TanhSaturatedF64)
                return TanhSaturated(x, bits);

            int index = (int)((ndnan - TanhIndexBiasF64) >> 32);
            index = Math.Max(index, 0);
            index = Math.Min(index, TanhIndexClampF64);
            int offset = (index >> 19) * 18;

            ref double c = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(TanhLutF64), offset);
            double y = Math.Abs(x) - c;                                 // |x| - b
            double r = Math.FusedMultiplyAdd(Unsafe.Add(ref c, 17), y, Unsafe.Add(ref c, 16));
            r = Math.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 15));
            r = Math.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 14));
            r = Math.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 13));
            r = Math.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 12));
            r = Math.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 11));
            r = Math.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 10));
            r = Math.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 9));
            r = Math.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 8));
            r = Math.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 7));
            r = Math.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 6));
            r = Math.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 5));
            r = Math.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 4));
            r = Math.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 3));
            r = Math.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 2));
            r = Math.FusedMultiplyAdd(r, y, Unsafe.Add(ref c, 1));

            return BitConverter.UInt64BitsToDouble(
                BitConverter.DoubleToUInt64Bits(r) | (bits & 0x8000000000000000UL));
        }

        /// <summary>The saturated end of <see cref="Tanh(double)"/>. See
        /// <see cref="TanhSaturated(float, uint)"/>.</summary>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static double TanhSaturated(double x, ulong bits)
        {
            if (double.IsNaN(x))
                return BitConverter.UInt64BitsToDouble(CanonicalNaNBitsF64);
            return BitConverter.UInt64BitsToDouble(0x3ff0000000000000UL | (bits & 0x8000000000000000UL));
        }
    }
}
