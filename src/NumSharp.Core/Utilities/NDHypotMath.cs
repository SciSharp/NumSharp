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
    /// <para><b>Every float dtype is BIT-EXACT with NumPy.</b> UCRT's <c>hypot</c> is only <i>faithfully</i>
    /// rounded (it disagrees with the correctly rounded result on 8.7% of float64 inputs), because it is
    /// the plain <c>sqrt(x*x + y*y)</c> — two products, a sum and a square root, each rounded — with an
    /// exact power-of-two rescale only where <c>x*x</c> would overflow or lose <c>y*y</c> to underflow.
    /// Probed through <c>ucrtbase</c> (parity QA-25): that model equals UCRT's <c>hypot</c> on every one
    /// of 1.7M pairs spanning every exponent, subnormals and the corners, and <c>np.hypot</c> with it.
    /// <see cref="Hypot(double,double)"/> is that model, so float64 <c>np.hypot</c> returns NumPy's bits —
    /// including where they are 1 ULP from the correctly rounded value (<c>np.hypot(0.1, 0.1)</c> is
    /// <c>0x3fc21a1851ff630b</c>). Until parity QA-25 this kernel was Borges' correctly rounded FMA
    /// algorithm, more accurate than NumPy and excused in the oracle. The <c>ff->f</c> and (via float32,
    /// <c>astype e->f</c>) <c>ee->e</c> loops round the double result down to float, which reproduces
    /// UCRT's <c>hypotf</c> exactly (0 of 300,000 float32 pairs differ; a float operand's square is exact
    /// in double, so the double sum leaves no room for a float32 rounding difference).</para>
    ///
    /// <para>Decimal (no NumPy analog) computes the highest-precision result natively via
    /// <see cref="DecimalMath.Sqrt(decimal)"/> with the overflow-safe scaled form.</para>
    /// </summary>
    public static partial class NDHypotMath
    {
        /// <summary>
        /// UCRT's <c>hypot</c> core: <c>sqrt(a*a + b*b)</c>, every operation rounded on its own (no fused
        /// multiply-add — RyuJIT never contracts one, and UCRT's results carry the separate roundings).
        /// Assumes both operands finite and pre-scaled so <c>a*a</c> neither overflows nor loses
        /// <c>b*b</c> to underflow (the caller's safe middle); within it the operand order does not matter,
        /// since IEEE addition is commutative.
        /// </summary>
        /// <param name="a">The larger magnitude (non-negative, finite, in the safe range).</param>
        /// <param name="b">The smaller magnitude (non-negative, finite).</param>
        /// <returns>The faithfully rounded <c>sqrt(a² + b²)</c>, UCRT's bits.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double Core(double a, double b) => Math.Sqrt(a * a + b * b);

        /// <summary>
        /// float64 <c>np.hypot</c> of one pair, NumPy's <c>npy_hypot</c> = UCRT's <c>hypot</c> on win-amd64:
        /// <c>sqrt(x² + y²)</c> faithfully rounded the way UCRT rounds it (see the class remarks), without
        /// spurious overflow or underflow. An infinite operand yields <c>+inf</c> even when the other is NaN;
        /// otherwise the FIRST NaN operand comes back as it is (payload and sign, as UCRT returns it); the
        /// result is otherwise non-negative.
        /// </summary>
        /// <param name="x">The first leg.</param>
        /// <param name="y">The second leg.</param>
        /// <returns>The hypotenuse, NumPy's bits.</returns>
        [MethodImpl(OptimizeAndInline)]
        public static double Hypot(double x, double y)
        {
            // inf beats nan (C99 Annex F / npy_hypot): hypot(±inf, nan) = +inf.
            if (double.IsInfinity(x) || double.IsInfinity(y))
                return double.PositiveInfinity;
            // UCRT returns its first NaN operand untouched (probed: hypot(1, -nan) is that -nan), so the NaN test
            // runs BEFORE the absolute values, which would clear the sign.
            if (double.IsNaN(x))
                return x;
            if (double.IsNaN(y))
                return y;
            x = Math.Abs(x);
            y = Math.Abs(y);
            if (x < y) { double t = x; x = y; y = t; }   // x >= y >= 0
            if (y == 0.0) return x;                      // covers x==0 -> +0, and hypot(x,0)=|x|

            int ex = Math.ILogB(x);
            // Safe middle: |x| in [2^-500, 2^500] keeps x*x normal — square directly.
            if (ex > -500 && ex < 500)
                return Core(x, y);

            // Extremes: shift x into [1,2) by an exact power of 2, compute, shift the root back.
            // Math.ScaleB manipulates the exponent field, so no intermediate over/underflows even when
            // 2^-ex is itself unrepresentable (subnormal x, ex down to -1074). A power-of-two rescale changes
            // no rounding of the products, the sum or the root (an even exponent shift passes through sqrt
            // exactly), so this is UCRT's own rescaled result bit for bit, its 2^±600 shift included.
            double xs = Math.ScaleB(x, -ex);
            double ys = Math.ScaleB(y, -ex);
            return Math.ScaleB(Core(xs, ys), ex);
        }

        /// <summary>
        /// <c>hypotf</c> (float). NumPy's <c>ff->f</c> loop; bit-exact with UCRT's <c>hypotf</c> because
        /// rounding the double result down to float coincides with it (a float operand's square is exact in
        /// double, so only the sum and the root round before the final narrowing: 0 of 300,000 probed pairs
        /// differ from <c>ucrtbase</c>'s <c>_hypotf</c>).
        /// </summary>
        /// <param name="x">The first leg.</param>
        /// <param name="y">The second leg.</param>
        /// <returns>The hypotenuse, NumPy's bits.</returns>
        [MethodImpl(OptimizeAndInline)]
        public static float HypotF(float x, float y) => (float)Hypot(x, y);

        /// <summary>
        /// Half computes in float32, matching NumPy's <c>ee->e</c> loop (<c>astype e->f</c>: half→float,
        /// <c>hypotf</c>, float→half). Bit-exact with NumPy.
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
        public static Half HypotHalf(Half x, Half y) => (Half)HypotF((float)x, (float)y);

        /// <summary>
        /// Decimal (no NumPy analog): the overflow-safe scaled form <c>|x|·sqrt(1+(|y|/|x|)²)</c> keeps
        /// the radicand in [1,2] so only a genuine out-of-range result overflows (decimal has no infinity).
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
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
