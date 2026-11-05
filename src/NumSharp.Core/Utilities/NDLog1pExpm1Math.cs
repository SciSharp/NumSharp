using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Scalar kernels backing NumSharp's float32 / float64 <c>np.log1p</c> and <c>np.expm1</c>, and the <c>log1p</c> inside
    /// <c>np.logaddexp</c> / <c>np.logaddexp2</c> (<see cref="NDLogAddExpMath"/>): models of the MSVC UCRT routines that
    /// NumPy 2.4.2's win-amd64 wheel calls (<c>npy_log1p</c> / <c>npy_expm1</c> forward to the C library; the AVX-512 SVML
    /// loops that would replace them are not built into that wheel).
    ///
    /// <para><b>log1p / log1pf are BIT-EXACT with NumPy.</b> Probed against <c>np.log1p</c> (parity QA-22, 2026-11-05): the
    /// UCRT's <c>log1p</c> is Goldberg's correction over its own <c>log</c>, <c>u = 1 + x; log(u) - ((u - 1) - x) / u</c>,
    /// applied for every <c>u</c> (also above 2), and <c>log1pf</c> is the same statements in float32 over <c>logf</c>.
    /// <see cref="Math.Log(double)"/> / <see cref="MathF.Log(float)"/> ARE the UCRT's <c>log</c> / <c>logf</c>, so the model
    /// equals NumPy on all 320,000 finite inputs per dtype of the probe (tiny and subnormal, small, (-1, 1), near -1, [1, 10],
    /// up to the largest finite value), and the specials are pinned by hand: a NaN comes back untouched (a signalling NaN
    /// stays signalling, sign and payload kept), <c>log1p(-1) = -inf</c>, <c>log1p(+inf) = +inf</c>, and every <c>x &lt; -1</c>
    /// (<c>-inf</c> included) gives the POSITIVE quiet NaN (.NET's <c>Math.Log</c> of a negative is the negative one).
    /// Faithfully rounded, not correctly rounded: about 6% of float64 results sit one ULP from the true value, exactly where
    /// NumPy's do.</para>
    ///
    /// <para><b>expm1 / expm1f are bit-exact outside a band and within 2-3 ULP inside it.</b> Outside
    /// <c>[-ln 2, ln 1.5]</c>, where <c>|expm1(x)| &gt;= 0.5</c> and the subtraction loses nothing, the UCRT returns exactly
    /// <c>exp(x) - 1</c> (<c>expf(x) - 1f</c> for float32, whose band ends at the float32 roundings of the same constants):
    /// those results are NumPy's bit for bit. Inside the band the UCRT runs an undocumented approximation no public formula
    /// reproduces (see <see cref="Expm1Sun"/>), so the band runs Sun's <c>s_expm1.c</c> — accurate to one ULP of the true
    /// value, equal to NumPy on about 58% of float64 and 57% of float32 in-band inputs, never more than 2 (float64) / 3
    /// (float32) ULP from it. A NaN comes back untouched, <c>-0.0</c> keeps its sign, <c>expm1(-inf) = -1</c>.</para>
    ///
    /// <para>The naive forms these replace, <c>Exp(x) - 1</c> and <c>Log(1 + x)</c> (also what .NET's <c>ExpM1</c> /
    /// <c>LogP1</c> compute), cancel catastrophically near zero: <c>expm1(1e-8)</c> kept 8 digits, <c>expm1(1e-300)</c>
    /// and <c>log1p(1e-300)</c> were 0, and <c>-0.0</c> became <c>+0.0</c>. A platform pin, like <see cref="NDHypotMath"/>:
    /// Linux NumPy calls glibc's routines, which round differently.</para>
    /// </summary>
    public static class NDLog1pExpm1Math
    {
        /// <summary><c>-ln 2</c> as a double: below it the UCRT's <c>expm1</c> is exactly <c>exp(x) - 1</c>.</summary>
        internal const double Expm1BandLow = -0.6931471805599453;

        /// <summary><c>ln 1.5</c> as a double: above it the UCRT's <c>expm1</c> is exactly <c>exp(x) - 1</c>.</summary>
        internal const double Expm1BandHigh = 0.4054651081081644;

        /// <summary>
        /// <c>-ln 2</c> rounded to float32 (<c>0xBF317218</c>): at and below it the UCRT's <c>expm1f</c> is exactly
        /// <c>expf(x) - 1f</c> (every float32 in <c>[-0.8, -0.6]</c> probed; the largest in-band input is <c>0xBF317217</c>).
        /// </summary>
        internal const float Expm1BandLowF = -0.6931472f;

        /// <summary>
        /// <c>ln 1.5</c> rounded to float32 (<c>0x3ECF991F</c>): at and above it the UCRT's <c>expm1f</c> is exactly
        /// <c>expf(x) - 1f</c> (every float32 in <c>[0.35, 0.5]</c> probed; the largest in-band input is <c>0x3ECF991E</c>).
        /// </summary>
        internal const float Expm1BandHighF = 0.4054651f;

        /// <summary>
        /// float64 <c>np.log1p</c> of one element — <c>ln(1 + x)</c> accurate near zero, NumPy's bits on win-amd64 (the UCRT
        /// model of the class remarks).
        /// </summary>
        /// <param name="x">The argument.</param>
        /// <returns><c>ln(1 + x)</c>; <c>x</c> itself while <c>1 + x</c> rounds to 1 (so <c>-0.0</c>, subnormals and tiny
        /// values return unchanged); <c>-inf</c> at -1; the positive quiet NaN below -1; a NaN argument untouched.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Log1p(double x)
        {
            // The specials first, in the order the UCRT resolves them: a NaN passes through bit for bit (u = 1 + x would
            // quiet a signalling one), the pole and the domain error have their own answers (the formula gives NaN at -1
            // through 0/0, and .NET's negative NaN below it), and +inf would meet inf - inf in the correction.
            if (double.IsNaN(x))
                return x;
            if (x <= -1.0)
                return x == -1.0 ? double.NegativeInfinity : BitConverter.Int64BitsToDouble(0x7FF8000000000000);
            if (double.IsPositiveInfinity(x))
                return x;
            double u = 1.0 + x;
            // 1 + x rounded to 1: the logarithm's leading term is the whole answer (keeps -0.0 and subnormals exact).
            if (u == 1.0)
                return x;
            // Goldberg: log(u) is exact for u, and ((u - 1) - x) is the rounding error of u = 1 + x, which the division by u
            // turns into the first-order correction log(1 + x) = log(u) - (u - (1 + x)) / u. Applied for every u, as the UCRT
            // does (skipping it above u = 2 differs from NumPy on ~2% of x in [1, 10]).
            return Math.Log(u) - ((u - 1.0) - x) / u;
        }

        /// <summary>
        /// float32 <c>np.log1p</c> of one element — the UCRT's <c>log1pf</c>: <see cref="Log1p(double)"/>'s statements in
        /// float32 over <c>logf</c>, NumPy's bits on win-amd64.
        /// </summary>
        /// <param name="x">The argument.</param>
        /// <returns><c>ln(1 + x)</c> in float32, with <see cref="Log1p(double)"/>'s special values (the positive quiet NaN is
        /// <c>0x7FC00000</c>).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Log1pF(float x)
        {
            // Same order and reasons as the float64 model.
            if (float.IsNaN(x))
                return x;
            if (x <= -1.0f)
                return x == -1.0f ? float.NegativeInfinity : BitConverter.Int32BitsToSingle(0x7FC00000);
            if (float.IsPositiveInfinity(x))
                return x;
            float u = 1.0f + x;
            if (u == 1.0f)
                return x;
            // Every operation in float32 (the UCRT's log1pf does not widen): computing in double and narrowing once is a
            // DIFFERENT function, 78% equal to NumPy on small arguments.
            return MathF.Log(u) - ((u - 1.0f) - x) / u;
        }

        /// <summary>
        /// float64 <c>np.expm1</c> of one element — <c>e^x - 1</c> accurate near zero: NumPy's bits outside
        /// <c>[-ln 2, ln 1.5]</c>, Sun's <c>s_expm1.c</c> (within 2 ULP of NumPy) inside it.
        /// </summary>
        /// <param name="x">The argument.</param>
        /// <returns><c>e^x - 1</c>; <c>-1</c> for <c>-inf</c> (and once <c>e^x</c> underflows), <c>+inf</c> on overflow, the
        /// sign of a zero kept, a NaN argument untouched.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Expm1(double x)
        {
            // A NaN passes through bit for bit, as the UCRT returns it (both branches below would quiet a signalling NaN).
            if (double.IsNaN(x))
                return x;
            // Outside the band |expm1(x)| >= 0.5, so exp(x) - 1 loses nothing — and it IS the UCRT's formula there.
            if (x < Expm1BandLow || x > Expm1BandHigh)
                return Math.Exp(x) - 1.0;
            return Expm1Sun(x);
        }

        /// <summary>
        /// float32 <c>np.expm1</c> of one element: the UCRT's <c>expf(x) - 1f</c> outside the float32 band (NumPy's bits),
        /// the float64 <see cref="Expm1Sun"/> rounded once inside it (within 3 ULP of NumPy's in-band polynomial, which is
        /// itself up to 4 ULP from the true value).
        /// </summary>
        /// <param name="x">The argument.</param>
        /// <returns><c>e^x - 1</c> in float32, with <see cref="Expm1(double)"/>'s special values.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Expm1F(float x)
        {
            if (float.IsNaN(x))
                return x;
            // float32 arithmetic outside the band (MathF.Exp is the UCRT's expf): (float)(exp(x) - 1) in double rounds
            // differently on ~1.5% of those inputs.
            if (x <= Expm1BandLowF || x >= Expm1BandHighF)
                return MathF.Exp(x) - 1f;
            // In the band the double result is about correctly rounded, the closest any known formula gets to expm1f's
            // undocumented polynomial (float32 Sun / Taylor forms match it less often).
            return (float)Expm1Sun(x);
        }

        /// <summary>Sun's (FreeBSD msun / glibc) <c>s_expm1.c</c>, transcribed statement by statement.</summary>
        /// <param name="x">The argument (called for the band <c>[-ln 2, ln 1.5]</c>, but valid everywhere).</param>
        /// <returns><c>e^x - 1</c>, within one ulp.</returns>
        /// <remarks>
        /// The band arm of <see cref="Expm1(double)"/>, <see cref="Expm1F(float)"/> and the random samplers'
        /// <c>Generator.Expm1</c> (moved here from <c>Generator.Kernels.cs</c> so the ufuncs and the samplers share one
        /// transcription). Inside the band the UCRT uses an undocumented approximation that no public formula reproduces
        /// (not correctly rounded: about a third of its results sit one ulp or more from the true value; Cephes,
        /// FreeBSD/Sun, Kahan's (with <c>log</c> and <c>log1p</c>), Newton-on-<c>log1p</c>, Taylor/Horner,
        /// <c>sinh</c>/<c>tanh</c>/<c>exp2</c> forms, x87 <c>F2XM1</c> roundings and AMD's libm all tested and rejected);
        /// this port agrees with it on about 58% of in-band inputs and is never more than 2 ulp away. A NaN argument comes
        /// back quiet (<c>x + x</c>): the public entry points answer NaN before calling it.
        /// </remarks>
        internal static double Expm1Sun(double x)
        {
            const double o_threshold = 7.09782712893383973096e+02;
            const double ln2_hi = 6.93147180369123816490e-01;
            const double ln2_lo = 1.90821492927058770002e-10;
            const double invln2 = 1.44269504088896338700e+00;
            const double Q1 = -3.33333333333331316428e-02;
            const double Q2 = 1.58730158725481460165e-03;
            const double Q3 = -7.93650757867487942473e-05;
            const double Q4 = 4.00821782732936239552e-06;
            const double Q5 = -2.01099218183624371326e-07;

            double y, hi, lo, c = 0, t, e, hxs, hfx, r1, twopk;
            int k;
            long bits = BitConverter.DoubleToInt64Bits(x);
            uint hx = (uint)(bits >> 32);
            bool negative = (hx & 0x80000000) != 0;
            hx &= 0x7fffffff;

            /* filter out huge and non-finite argument */
            if (hx >= 0x4043687A)
            {
                if (hx >= 0x40862E42)
                {
                    if (hx >= 0x7ff00000)
                    {
                        uint low = (uint)bits;
                        if (((hx & 0xfffff) | low) != 0)
                            return x + x; /* NaN */
                        return negative ? -1.0 : x; /* exp(+-inf) - 1 = {inf, -1} */
                    }
                    if (x > o_threshold)
                        return double.PositiveInfinity; /* overflow */
                }
                if (negative && x + 1e-300 < 0.0)
                    return 1e-300 - 1.0; /* x < -56*ln2: return -1.0 with inexact */
            }

            /* argument reduction */
            if (hx > 0x3fd62e42)
            {
                /* if |x| > 0.5 ln2 */
                if (hx < 0x3FF0A2B2)
                {
                    /* and |x| < 1.5 ln2 */
                    if (!negative)
                    {
                        hi = x - ln2_hi;
                        lo = ln2_lo;
                        k = 1;
                    }
                    else
                    {
                        hi = x + ln2_hi;
                        lo = -ln2_lo;
                        k = -1;
                    }
                }
                else
                {
                    k = (int)(invln2 * x + (!negative ? 0.5 : -0.5));
                    t = k;
                    hi = x - t * ln2_hi; /* t*ln2_hi is exact here */
                    lo = t * ln2_lo;
                }
                x = hi - lo;
                c = (hi - x) - lo;
            }
            else if (hx < 0x3c900000)
            {
                /* when |x| < 2**-54, return x */
                t = 1e300 + x; /* return x with inexact flags when x != 0 */
                return x - (t - (1e300 + x));
            }
            else
            {
                k = 0;
            }

            /* x is now in primary range */
            hfx = 0.5 * x;
            hxs = x * hfx;
            r1 = 1.0 + hxs * (Q1 + hxs * (Q2 + hxs * (Q3 + hxs * (Q4 + hxs * Q5))));
            t = 3.0 - r1 * hfx;
            e = hxs * ((r1 - t) / (6.0 - x * t));
            if (k == 0)
                return x - (x * e - hxs); /* c is 0 */
            twopk = BitConverter.Int64BitsToDouble((long)(0x3ff + k) << 52);
            e = (x * (e - c) - c);
            e -= hxs;
            if (k == -1)
                return 0.5 * (x - e) - 0.5;
            if (k == 1)
            {
                if (x < -0.25)
                    return -2.0 * (e - (x + 0.5));
                return 1.0 + 2.0 * (x - e);
            }
            if (k <= -2 || k > 56)
            {
                /* suffice to return exp(x) - 1 */
                y = 1.0 - (e - x);
                if (k == 1024)
                    y = y * 2.0 * BitConverter.Int64BitsToDouble(0x7fe0000000000000);
                else
                    y = y * twopk;
                return y - 1.0;
            }
            if (k < 20)
            {
                t = BitConverter.Int64BitsToDouble((long)(0x3ff - (0x200000 >> k)) << 32); /* t = 1 - 2^-k */
                y = t - (e - x);
                y = y * twopk;
            }
            else
            {
                t = BitConverter.Int64BitsToDouble((long)((0x3ff - k) << 20) << 32); /* 2^-k */
                y = x - (e + t);
                y += 1.0;
                y = y * twopk;
            }
            return y;
        }
    }
}
