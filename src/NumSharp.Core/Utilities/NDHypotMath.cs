using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Scalar kernel backing NumSharp's <c>np.hypot</c> binary ufunc — <c>sqrt(x1**2 + x2**2)</c>
    /// computed without spurious overflow/underflow, the length of the hypotenuse of a right triangle.
    ///
    /// <para><b>Why a from-scratch algorithm instead of a library call.</b> NumPy's <c>hypot</c> ufunc is
    /// a scalar <c>BINARY_LOOP</c> that calls <c>npy_hypot</c> (<c>npy_math_internal.h.src</c>). On the
    /// win-amd64 NumPy 2.4.2 wheel <c>NPY_BLOCK_HYPOT</c> is <b>not</b> defined (it is set only for MinGW
    /// and 32-bit MSVC — <c>npy_config.h</c>), so <c>npy_hypot</c> forwards straight to the MSVC UCRT
    /// <c>hypot</c>/<c>hypotf</c>. The BCL has no <c>Math.Hypot</c>, and the one hypot in the framework —
    /// the private helper behind <see cref="System.Numerics.Complex.Abs"/> — is the naive
    /// <c>large·sqrt(1+(small/large)²)</c> "block" form, which diverges from UCRT's <c>hypot</c> on ~19%
    /// of inputs (1–2 ULP). So there is no managed call that reproduces the NumPy value, and the loop is
    /// scalar-only either way (no SIMD hypot exists in NumPy on this platform).</para>
    ///
    /// <para><b>float32 / float16 are BIT-EXACT with NumPy; float64 is ≤1 ULP.</b> UCRT's <c>hypot</c> is
    /// a <i>faithful</i> (≤1 ULP) but not correctly-rounded implementation — it disagrees with the exact
    /// correctly-rounded result on 8.7% of float64 inputs (measured on this host: <c>np.hypot</c> vs a
    /// 80-digit <c>Decimal</c> reference). <see cref="Hypot(double,double)"/> is the CORRECTLY-ROUNDED
    /// result (Borges' FMA algorithm, verified bit-identical to CPython's correctly-rounded
    /// <c>math.hypot</c> over 1.1M adversarial pairs including subnormals/overflow), so it is bit-exact
    /// with NumPy on 91.3% of float64 inputs and ≤1 ULP (never 2) — and MORE accurate — on the rest.
    /// That divergence is documented/excused in the oracle exactly like <c>logaddexp</c>'s ≤2 ULP libm
    /// gap. The <c>ff->f</c> and (via float32, <c>astype e->f</c>) <c>ee->e</c> loops round the
    /// correctly-rounded double down to float, which reproduces UCRT's <c>hypotf</c> EXACTLY (verified
    /// 0-diff over 200K float32 and 300K float16 pairs) — a narrower type leaves no room for UCRT's
    /// double-precision rounding quirk to show.</para>
    ///
    /// <para>Decimal (no NumPy analog) computes the highest-precision result natively via
    /// <see cref="DecimalMath.Sqrt(decimal)"/> with the overflow-safe scaled form.</para>
    /// </summary>
    public static partial class NDHypotMath
    {
        /// <summary>
        /// Borges' correctly-rounded core (2020, "An Improved Algorithm for hypot(a,b)"): assumes
        /// <paramref name="a"/> ≥ <paramref name="b"/> ≥ 0, both finite and pre-scaled so <c>a*a</c>
        /// neither overflows nor loses <c>b*b</c> to underflow. Forms the exact <c>a²+b²</c> with two
        /// FMA 2-products and a Fast2Sum, takes <c>sqrt</c>, then a single Newton step on the exactly
        /// computed residual <c>(a²+b²) − h²</c> lifts the faithful root to the correctly-rounded one.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double Core(double a, double b)
        {
            double xx = a * a, e_xx = Math.FusedMultiplyAdd(a, a, -xx);   // a² = xx + e_xx  (exact)
            double yy = b * b, e_yy = Math.FusedMultiplyAdd(b, b, -yy);   // b² = yy + e_yy  (exact)
            double s = xx + yy, e_s = yy - (s - xx);    // xx+yy = s + e_s  (Fast2Sum, xx>=yy)
            double sigma = e_xx + e_yy + e_s;           // low-order bits of a²+b²
            double h = Math.Sqrt(s);
            double hh = h * h, e_hh = Math.FusedMultiplyAdd(h, h, -hh);   // h² = hh + e_hh  (exact)
            double residual = (s - hh) + sigma - e_hh;  // (a²+b²) − h²  (s−hh exact by Sterbenz)
            return h + residual / (2.0 * h);            // one Newton step → correctly rounded
        }

        /// <summary>
        /// Correctly-rounded <c>sqrt(x²+y²)</c> (double). Matches NumPy's <c>hypot</c> value on ~91% of
        /// inputs and is within 1 ULP (more accurate) on the rest. Non-finite handling follows the C
        /// contract NumPy inherits from UCRT: an infinite operand yields <c>+inf</c> even when the other
        /// is NaN; otherwise a NaN operand yields NaN; the result is always non-negative.
        /// </summary>
        public static double Hypot(double x, double y)
        {
            x = Math.Abs(x);
            y = Math.Abs(y);
            // inf beats nan (C99 Annex F / npy_hypot): hypot(±inf, nan) = +inf.
            if (double.IsPositiveInfinity(x) || double.IsPositiveInfinity(y))
                return double.PositiveInfinity;
            if (double.IsNaN(x) || double.IsNaN(y))
                return double.NaN;                       // float NaN is tokenized by the oracle
            if (x < y) { double t = x; x = y; y = t; }   // x >= y >= 0
            if (y == 0.0) return x;                      // covers x==0 -> +0, and hypot(x,0)=|x|

            int ex = Math.ILogB(x);
            // Safe middle: |x| in [2^-500, 2^500] keeps x*x normal — square directly.
            if (ex > -500 && ex < 500)
                return Core(x, y);

            // Extremes: shift x into [1,2) by an exact power of 2, compute, shift the root back.
            // Math.ScaleB manipulates the exponent field, so no intermediate over/underflows even when
            // 2^-ex is itself unrepresentable (subnormal x, ex down to -1074).
            double xs = Math.ScaleB(x, -ex);
            double ys = Math.ScaleB(y, -ex);
            return Math.ScaleB(Core(xs, ys), ex);
        }

        /// <summary>
        /// <c>hypotf</c> (float). NumPy's <c>ff->f</c> loop; bit-exact with UCRT's <c>hypotf</c> because
        /// rounding the correctly-rounded double result down to float coincides with it.
        /// </summary>
        public static float HypotF(float x, float y) => (float)Hypot(x, y);

        /// <summary>
        /// Half computes in float32, matching NumPy's <c>ee->e</c> loop (<c>astype e->f</c>: half→float,
        /// <c>hypotf</c>, float→half). Bit-exact with NumPy.
        /// </summary>
        public static Half HypotHalf(Half x, Half y) => (Half)HypotF((float)x, (float)y);

        /// <summary>
        /// Decimal (no NumPy analog): the overflow-safe scaled form <c>|x|·sqrt(1+(|y|/|x|)²)</c> keeps
        /// the radicand in [1,2] so only a genuine out-of-range result overflows (decimal has no infinity).
        /// </summary>
        public static decimal HypotDecimal(decimal x, decimal y)
        {
            x = Math.Abs(x);
            y = Math.Abs(y);
            if (x < y) { decimal t = x; x = y; y = t; }
            if (y == 0m) return x;
            decimal r = y / x;
            return x * DecimalMath.Sqrt(1m + r * r);
        }
    }
}
