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
    /// <para><b><see cref="Exp"/> = <c>simd_exp_FLOAT</c></b>
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

        /// <summary><see cref="CanonicalNaNBits"/> as a float, for the vector overloads' selects.</summary>
        private static readonly float CanonicalNaN = BitConverter.UInt32BitsToSingle(CanonicalNaNBits);

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
