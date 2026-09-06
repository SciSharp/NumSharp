using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Scalar kernels backing NumSharp's <c>np.logaddexp</c> / <c>np.logaddexp2</c> /
    /// <c>np.nextafter</c> binary ufuncs. Each entry point reproduces the exact NumPy 2.4.2
    /// algorithm (<c>npy_math_internal.h.src</c> / <c>ieee754</c> / <c>halffloat.cpp</c>), operation
    /// for operation, so the result matches NumPy on this platform to within the accuracy of the one
    /// primitive NumSharp cannot reproduce bit-for-bit — the CRT's <c>log1p</c>.
    ///
    /// <para><b>nextafter is BIT-EXACT.</b> <see cref="Math.BitIncrement(double)"/> /
    /// <see cref="Math.BitDecrement(double)"/> (and the <see cref="MathF"/> twins) compose to exactly
    /// the ucrtbase <c>nextafter</c>/<c>nextafterf</c> NumPy calls (verified 0-diff over 6M random
    /// pairs). Half is a straight port of <c>npy_half_nextafter</c> on the raw 16-bit pattern.</para>
    ///
    /// <para><b>logaddexp / logaddexp2 are ≤2 ULP.</b> <see cref="Math.Exp(double)"/>,
    /// <see cref="double.Exp2(double)"/> and <see cref="MathF.Exp(float)"/> are already bit-identical
    /// to the ucrtbase <c>exp</c>/<c>exp2</c>/<c>expf</c> NumPy calls; the only divergence is
    /// <see cref="Log1p(double)"/> — an fdlibm port that agrees with the (closed) ucrtbase
    /// <c>log1p</c> to ≤1 ULP. So float64 <c>logaddexp</c> is ≤1 ULP, <c>logaddexp2</c> ≤2 ULP (the
    /// <c>LOG2E·log1p</c> product), float32 <c>logaddexp</c> is bit-exact and <c>logaddexp2</c> ≤1 ULP.
    /// The catastrophic small-value case that a naive <c>log(1+exp(-tmp))</c> loses
    /// (<c>logaddexp(0,-50)</c> = 1.93e-22, not 0) is fully recovered.</para>
    ///
    /// <para>Half is computed in <b>float32</b> (<c>(Half)F((float)x,(float)y)</c>), matching NumPy's
    /// <c>ee->e</c> half loops which promote to float. Decimal (no NumPy analog) bridges through
    /// double.</para>
    /// </summary>
    public static class NDLogAddExpMath
    {
        // NumPy NPY_LOGE2 / NPY_LOG2E (npy_math_internal.h.src LOGE2 / LOG2E), rounded to each width
        // exactly as NumPy's NPY__FP_SFX macro does.
        internal const double LOGE2 = 0.693147180559945309417232121458176568;
        internal const double LOG2E = 1.442695040888963407359924681001892137;
        internal const float LOGE2F = 0.693147180559945309417232121458176568f;
        internal const float LOG2EF = 1.442695040888963407359924681001892137f;

        // =====================================================================
        // log1p — fdlibm s_log1p.c / s_log1pf.c ports. NumPy uses the CRT log1p
        // (bit-exact with ucrtbase, but closed); this agrees to <=1 ULP.
        // =====================================================================

        // fdlibm double coefficients
        private const double ln2_hi = 6.93147180369123816490e-01, ln2_lo = 1.90821492927058770002e-10,
            two54 = 1.80143985094819840000e+16,
            Lp1 = 6.666666666666735130e-01, Lp2 = 3.999999999940941908e-01, Lp3 = 2.857142874366239149e-01,
            Lp4 = 2.222219843214978396e-01, Lp5 = 1.818357216161805012e-01, Lp6 = 1.531383769920937332e-01,
            Lp7 = 1.479819860511658591e-01;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int HighWord(double x) => (int)(BitConverter.DoubleToInt64Bits(x) >> 32);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double SetHighWord(double x, int hi)
        {
            long b = BitConverter.DoubleToInt64Bits(x);
            return BitConverter.Int64BitsToDouble((b & 0xffffffffL) | ((long)hi << 32));
        }

        /// <summary>log(1+x), fdlibm port (≤1 ULP vs ucrtbase log1p).</summary>
        public static double Log1p(double x)
        {
            double hfsq, f = 0, c = 0, s, z, R, u;
            int k = 1;
            int hx = HighWord(x);
            int ax = hx & 0x7fffffff;
            if (hx < 0x3FDA827A)                       // 1+x < sqrt(2)+
            {
                if (ax >= 0x3ff00000)                  // x <= -1.0
                {
                    if (x == -1.0) return double.NegativeInfinity;
                    return (x - x) / (x - x);          // NaN
                }
                if (ax < 0x3e200000)                   // |x| < 2**-29
                {
                    if (two54 + x > 0.0 && ax < 0x3c900000) return x; // |x| < 2**-54
                    return x - x * x * 0.5;
                }
                if (hx > 0 || hx <= unchecked((int)0xbfd2bec4)) { k = 0; f = x; }
            }
            if (hx >= 0x7ff00000) return x + x;        // +inf / NaN
            int hu = 0;
            if (k != 0)
            {
                if (hx < 0x43400000)
                {
                    u = 1.0 + x;
                    hu = HighWord(u);
                    k = (hu >> 20) - 1023;
                    c = (k > 0) ? 1.0 - (u - x) : x - (u - 1.0);
                    c /= u;
                }
                else
                {
                    u = x;
                    hu = HighWord(u);
                    k = (hu >> 20) - 1023;
                    c = 0;
                }
                hu &= 0x000fffff;
                if (hu < 0x6a09e) { u = SetHighWord(u, hu | 0x3ff00000); }
                else { k += 1; u = SetHighWord(u, hu | 0x3fe00000); hu = (0x00100000 - hu) >> 2; }
                f = u - 1.0;
            }
            hfsq = 0.5 * f * f;
            if (k == 0)
            {
                if (f == 0.0) return c == 0.0 ? 0.0 : c;
            }
            s = f / (2.0 + f);
            z = s * s;
            R = z * (Lp1 + z * (Lp2 + z * (Lp3 + z * (Lp4 + z * (Lp5 + z * (Lp6 + z * Lp7))))));
            if (k == 0) return f - (hfsq - s * (hfsq + R));
            return k * ln2_hi - ((hfsq - (s * (hfsq + R) + (k * ln2_lo + c))) - f);
        }

        // fdlibm float coefficients (s_log1pf.c)
        private const float ln2_hiF = 6.9313812256e-01f, ln2_loF = 9.0580006145e-06f, two25F = 3.355443200e+07f,
            Lg1 = 0.66666662693f, Lg2 = 0.40000972152f, Lg3 = 0.28498786688f, Lg4 = 0.24279078841f;

        /// <summary>log(1+x) at single precision, fdlibm port.</summary>
        public static float Log1pf(float x)
        {
            float hfsq, f = 0, c = 0, s, z, R, u;
            int k = 1;
            int hx = BitConverter.SingleToInt32Bits(x);
            int ax = hx & 0x7fffffff;
            if (hx < 0x3ed413d0)
            {
                if (ax >= 0x3f800000)
                {
                    if (x == -1.0f) return float.NegativeInfinity;
                    return (x - x) / (x - x);
                }
                if (ax < 0x38000000)
                {
                    if (two25F + x > 0.0f && ax < 0x33800000) return x;
                    return x - x * x * 0.5f;
                }
                if (hx > 0 || hx <= unchecked((int)0xbe95f619)) { k = 0; f = x; }
            }
            if (hx >= 0x7f800000) return x + x;
            if (k != 0)
            {
                int hu;
                if (hx < 0x5a000000)
                {
                    u = 1.0f + x;
                    hu = BitConverter.SingleToInt32Bits(u);
                    k = (hu >> 23) - 127;
                    c = (k > 0) ? 1.0f - (u - x) : x - (u - 1.0f);
                    c /= u;
                }
                else
                {
                    u = x;
                    hu = BitConverter.SingleToInt32Bits(u);
                    k = (hu >> 23) - 127;
                    c = 0;
                }
                hu &= 0x007fffff;
                if (hu < 0x3504f7) { u = BitConverter.Int32BitsToSingle(hu | 0x3f800000); }
                else { k += 1; u = BitConverter.Int32BitsToSingle(hu | 0x3f000000); hu = (0x00800000 - hu) >> 2; }
                f = u - 1.0f;
            }
            hfsq = 0.5f * f * f;
            if (k == 0)
            {
                if (f == 0.0f) return c == 0.0f ? 0.0f : c;
            }
            s = f / (2.0f + f);
            z = s * s;
            R = z * (Lg1 + z * (Lg2 + z * (Lg3 + z * Lg4)));
            if (k == 0) return f - (hfsq - s * (hfsq + R));
            return k * ln2_hiF - ((hfsq - (s * (hfsq + R) + (k * ln2_loF + c))) - f);
        }

        // =====================================================================
        // logaddexp = log(exp(x)+exp(y)) — npy_logaddexp
        // =====================================================================

        public static double LogAddExp(double x, double y)
        {
            if (x == y) return x + LOGE2;              // handles same-sign infinities without warnings
            double tmp = x - y;
            if (tmp > 0) return x + Log1p(Math.Exp(-tmp));
            if (tmp <= 0) return y + Log1p(Math.Exp(tmp));
            return tmp;                                // NaN
        }

        public static float LogAddExpF(float x, float y)
        {
            if (x == y) return x + LOGE2F;
            float tmp = x - y;
            if (tmp > 0) return x + Log1pf(MathF.Exp(-tmp));
            if (tmp <= 0) return y + Log1pf(MathF.Exp(tmp));
            return tmp;
        }

        // Half computes in float32, matching NumPy's ee->e half loop (npy_half_to_float, compute, back).
        public static Half LogAddExpHalf(Half x, Half y) => (Half)LogAddExpF((float)x, (float)y);

        // Decimal (no NumPy analog) bridges through double.
        public static decimal LogAddExpDecimal(decimal x, decimal y) => (decimal)LogAddExp((double)x, (double)y);

        // =====================================================================
        // logaddexp2 = log2(2**x + 2**y) — npy_logaddexp2 (npy_log2_1p = LOG2E*log1p)
        // =====================================================================

        public static double LogAddExp2(double x, double y)
        {
            if (x == y) return x + 1.0;
            double tmp = x - y;
            if (tmp > 0) return x + LOG2E * Log1p(double.Exp2(-tmp));
            if (tmp <= 0) return y + LOG2E * Log1p(double.Exp2(tmp));
            return tmp;
        }

        public static float LogAddExp2F(float x, float y)
        {
            if (x == y) return x + 1.0f;
            float tmp = x - y;
            if (tmp > 0) return x + LOG2EF * Log1pf(float.Exp2(-tmp));
            if (tmp <= 0) return y + LOG2EF * Log1pf(float.Exp2(tmp));
            return tmp;
        }

        public static Half LogAddExp2Half(Half x, Half y) => (Half)LogAddExp2F((float)x, (float)y);

        public static decimal LogAddExp2Decimal(decimal x, decimal y) => (decimal)LogAddExp2((double)x, (double)y);

        // =====================================================================
        // nextafter — C nextafter, bit-exact via BitIncrement/BitDecrement.
        // NaN: double/float propagate the operand NaN (x+y), matching ucrtbase.
        // =====================================================================

        public static double NextAfter(double x, double y)
        {
            if (double.IsNaN(x) || double.IsNaN(y)) return x + y;
            if (x == y) return y;                      // C99: nextafter(x,x)=y (carries y's sign at zero)
            return x < y ? Math.BitIncrement(x) : Math.BitDecrement(x);
        }

        public static float NextAfterF(float x, float y)
        {
            if (float.IsNaN(x) || float.IsNaN(y)) return x + y;
            if (x == y) return y;
            return x < y ? MathF.BitIncrement(x) : MathF.BitDecrement(x);
        }

        // NPY_HALF_NAN — canonical quiet NaN pattern npy_half_nextafter returns.
        private const ushort NPY_HALF_NAN = 0x7e00;

        /// <summary>Port of <c>npy_half_nextafter</c> (halffloat.cpp): raw 16-bit stepping.
        /// Note NumPy's half returns <b>x</b> when x==y (incl. signed-zero-equal), unlike the C99
        /// double/float path which returns y.</summary>
        public static Half NextAfterHalf(Half x, Half y)
        {
            ushort hx = BitConverter.HalfToUInt16Bits(x);
            ushort hy = BitConverter.HalfToUInt16Bits(y);
            ushort ret;
            if (Half.IsNaN(x) || Half.IsNaN(y))
                ret = NPY_HALF_NAN;
            else if (x == y)                           // eq_nonan: +0 == -0 -> returns x
                ret = hx;
            else if ((hx & 0x7fffu) == 0)              // x == 0 -> smallest subnormal, sign of y
                ret = (ushort)((hy & 0x8000u) + 1);
            else if ((hx & 0x8000u) == 0)              // x > 0
                ret = (short)hx > (short)hy ? (ushort)(hx - 1) : (ushort)(hx + 1);
            else                                       // x < 0
                ret = ((hy & 0x8000u) == 0 || (hx & 0x7fffu) > (hy & 0x7fffu))
                    ? (ushort)(hx - 1) : (ushort)(hx + 1);
            return BitConverter.UInt16BitsToHalf(ret);
        }

        public static decimal NextAfterDecimal(decimal x, decimal y) => (decimal)NextAfter((double)x, (double)y);

        // =====================================================================
        // copysign — magnitude of x with the sign of y. BIT-EXACT (Math.CopySign
        // / MathF.CopySign are the IEEE bit op NumPy's C copysign performs).
        // =====================================================================

        public static double CopySign(double x, double y) => Math.CopySign(x, y);

        public static float CopySignF(float x, float y) => MathF.CopySign(x, y);

        /// <summary>Half copysign on raw bits: |x|'s magnitude bits ORed with y's sign bit
        /// (NaN payload preserved, sign taken from y — matches NumPy).</summary>
        public static Half CopySignHalf(Half x, Half y)
        {
            ushort hx = BitConverter.HalfToUInt16Bits(x);
            ushort hy = BitConverter.HalfToUInt16Bits(y);
            return BitConverter.UInt16BitsToHalf((ushort)((hx & 0x7fffu) | (hy & 0x8000u)));
        }

        // Decimal (no NumPy analog, no signed zero): magnitude of x, sign of y (y<0 -> negative).
        public static decimal CopySignDecimal(decimal x, decimal y) => y < 0m ? -Math.Abs(x) : Math.Abs(x);
    }
}
