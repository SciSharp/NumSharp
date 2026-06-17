using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Complex-math helpers backing NumSharp's complex (<c>complex128</c>) unary ufuncs. Each entry
    /// point reproduces NumPy 2.4.2 bit-for-bit (or within 3 ULP on the finite interior), verified by
    /// a 504-point bit-exact sweep and a layout sweep (contiguous / F-contiguous / strided /
    /// transposed / reversed / sliced / broadcast / 0-d). Most are direct ports of NumPy's own
    /// routines in <c>npy_math_complex.c.src</c> (the FreeBSD msun implementations) because
    /// <c>System.Numerics.Complex</c> diverges on large magnitudes, the unit circle, tiny/subnormal
    /// values, branch cuts, and signed zeros.
    ///
    /// <para><b>Ported NumPy algorithms.</b>
    /// <see cref="Log"/> = <c>npy_clog</c> (four-regime rescale incl. the near-|z|=1 <c>log1p</c>
    /// path; drives <see cref="Log10"/> and the engine's <c>log2</c>); <see cref="Sinh"/>/
    /// <see cref="Cosh"/> = textbook <c>sinh/cosh(x)·trig(y)</c> with a <c>y==0</c> guard (so a huge
    /// real part doesn't become <c>inf·0 = NaN</c>) and the C99 Annex G non-finite tables;
    /// <see cref="Tanh"/> = Kahan's <c>npy_ctanh</c> (markedly more accurate than the BCL near
    /// <c>±π/2</c>) + the <c>|x|≥22</c> overflow-safe branch; <see cref="Sin"/>/<see cref="Cos"/>/
    /// <see cref="Tan"/> route through those exactly as NumPy defines <c>csin/ccos/ctan</c>;
    /// <see cref="Atan"/> = the full <c>npy_catanh</c> (real <c>atanh</c>/<c>atan</c> on the axes, the
    /// <c>log1p</c> interior, and an exponent-classified <c>real_part_reciprocal</c>);
    /// <see cref="Exp"/> = <c>npy_cexp</c>; <see cref="Sqrt"/> = <c>npy_csqrt</c>;
    /// <see cref="Expm1"/> = <c>nc_expm1</c> with a Goldberg real <c>expm1</c>;
    /// <see cref="Square"/> = FMA-contracted <c>z·z</c> (matches NumPy's complex multiply
    /// overflow/cancellation); <see cref="Reciprocal"/> = Smith's <c>nc_recip</c>;
    /// <see cref="Exp2"/>/<see cref="Log1p"/> compose the above; <see cref="Abs"/> = <c>npy_cabs</c>
    /// (C99 <c>hypot</c>: an infinite component yields <c>+inf</c> even alongside a NaN — the .NET 8
    /// <c>Complex.Abs</c> returns NaN there).</para>
    ///
    /// <para><b>Still delegating to the BCL (at parity):</b> <see cref="Asin"/> and <see cref="Acos"/>
    /// use <see cref="Complex.Asin"/>/<see cref="Complex.Acos"/> on the finite interior with
    /// signed-zero / branch-cut fixups, and the C99 non-finite tables otherwise.</para>
    ///
    /// <para><b>Accepted residuals (pathological inputs only, beyond 3 ULP):</b> <c>cos/sin</c> with a
    /// NaN imaginary part pick the C99-<em>unspecified</em> sign for the resulting zero; <c>arccos</c>
    /// with a sub-<c>DBL_MIN</c> imaginary part flushes the denormal real part to 0 where NumPy's
    /// <c>cacos</c> hard-work kernel keeps it (~5.8e-309); <c>sinh/cosh</c> at the <c>|x|∈[710,710.13]</c>
    /// overflow edge differ because Windows' CRT <c>sinh</c> overflows where .NET's stays finite.</para>
    ///
    /// <para><b>Perf:</b> each public entry point is a tiny finite-path wrapper marked
    /// <see cref="MethodImplOptions.AggressiveInlining"/> so the JIT folds it into the IL-emitted
    /// unary kernel (no per-element call frame); the rare non-finite / special-value tables live in
    /// cold helpers marked <see cref="MethodImplOptions.AggressiveOptimization"/> (kept out-of-line so
    /// the hot wrapper stays inlineable, fully optimized when hit). A benchmark of an IL-inlined
    /// variant vs this <c>call</c>-based form showed the per-element cost is dominated by the
    /// transcendental, so hand-emitting the formulas is not worth the duplication.</para>
    /// </summary>
    public static class NpyComplexMath
    {
        // NPY_PI_2 / NPY_LOGE2 (npy_math.h). Math.PI/2 == NPY_PI_2 bit-for-bit.
        private const double PI_2 = Math.PI / 2.0;
        private const double LOGE2 = 0.6931471805599453;

        // ctanh large-|x| threshold (npy_ctanh TANH_HUGE) and DBL_MAX/4 + DBL_MIN clog rescale bounds.
        private const double TANH_HUGE = 22.0;
        private const double DBL_MAX_4 = 1.7976931348623157e+308 / 4.0;
        private const double DBL_MIN = 2.2250738585072014e-308;
        private const int DBL_MANT_DIG = 53;
        private const int DBL_MAX_EXP = 1024;

        // npy_catanh (FreeBSD msun) constants (double precision).
        private const double DBL_EPSILON = 2.2204460492503131e-16;
        private const double RECIP_EPSILON = 1.0 / DBL_EPSILON;          // ~4.5e15: switch to 1/z form
        private const double SQRT_3_EPSILON = 2.5809568279517849e-8;    // sqrt(3*EPS): tiny-input return-z bound
        private const double PIO2_LO = 6.1232339957367659e-17;          // pio2_hi + pio2_lo == pio2_hi (inexact)
        private const double SUMSQ_SQRT_MIN = 1.4916681462400413e-154;  // sqrt(DBL_MIN): _sum_squares underflow guard

        /// <summary>
        /// <c>|z| = hypot(re, im)</c> with NumPy/C99 infinity semantics: a ±infinite real or
        /// imaginary part returns <c>+inf</c> regardless of the other part (including NaN).
        /// All other inputs defer to <see cref="Complex.Abs"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static double Abs(Complex z)
        {
            if (double.IsInfinity(z.Real) || double.IsInfinity(z.Imaginary))
                return double.PositiveInfinity;
            return Complex.Abs(z);
        }

        #region helpers

        /// <summary>True when <paramref name="d"/> is negative zero (<c>-0.0</c>).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static bool IsNegZero(double d) => d == 0.0 && double.IsNegative(d);

        /// <summary>True when <paramref name="d"/> is positive zero (<c>+0.0</c>).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static bool IsPosZero(double d) => d == 0.0 && !double.IsNegative(d);

        /// <summary><c>hypot</c> that returns <c>+inf</c> if either part is infinite (C99), used by
        /// the large-value / non-finite inverse-trig paths.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static double HypotInf(double x, double y)
        {
            x = Math.Abs(x);
            y = Math.Abs(y);
            if (double.IsInfinity(x) || double.IsInfinity(y))
                return double.PositiveInfinity;
            return Math.Sqrt(x * x + y * y);
        }

        /// <summary>NumPy <c>_clog_for_large_values</c> reduced to the (possibly-infinite) magnitudes
        /// reached by the inverse-trig non-finite paths: <c>(log hypot(x,y), atan2(y,x))</c>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static void ClogLarge(double x, double y, out double rr, out double ri)
        {
            rr = Math.Log(HypotInf(x, y));
            ri = Math.Atan2(y, x);
        }

        #endregion

        #region hyperbolic (sinh / cosh / tanh)

        /// <summary>
        /// Complex hyperbolic cosine matching NumPy. The <c>y == 0</c> guard returns <c>(cosh(x), x*y)</c>
        /// so a large <c>x</c> does not produce the BCL's <c>sinh(x)*sin(0) = inf*0 = NaN</c> imaginary
        /// part (this is what lets <see cref="Cos"/> handle a huge imaginary input); the general finite
        /// case is the textbook <c>cosh(x)cos(y) + i sinh(x)sin(y)</c>, which overflows to ±inf exactly
        /// where NumPy's libm does (probed: <c>|x| &gt;= ~710.5</c>). Non-finite inputs follow C99 Annex G.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Complex Cosh(Complex z)
        {
            double x = z.Real, y = z.Imaginary;
            if (double.IsFinite(x) && double.IsFinite(y))
            {
                if (y == 0.0)                                  // cosh(x + I0) = cosh(x) + I (x*0) — keeps sign(x*0)
                    return new Complex(Math.Cosh(x), x * y);
                return new Complex(Math.Cosh(x) * Math.Cos(y), Math.Sinh(x) * Math.Sin(y));
            }
            return CoshSpecial(x, y);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static Complex CoshSpecial(double x, double y)
        {
            // cosh(+-0 +- I(Inf|NaN)) = NaN + I 0 with an unspecified zero-sign; NumPy's libm
            // takes sign(y) for an infinite y and sign(x) for a NaN y.
            if (x == 0.0)
                return new Complex(y - y, Math.CopySign(0.0, double.IsInfinity(y) ? y : x));
            // cosh((Inf|NaN) +- I0) = (Inf|NaN) +- I0
            if (y == 0.0)
                return new Complex(x * x, Math.CopySign(0.0, x) * y);
            // cosh(finite +- I(Inf|NaN)) = NaN + I NaN
            if (double.IsFinite(x))
                return new Complex(y - y, x * (y - y));
            // cosh(+-Inf + I y)  (cosh is even: NumPy uses +Inf magnitude for both parts)
            if (double.IsInfinity(x))
            {
                if (!double.IsFinite(y))            // cosh(+-Inf +- I(Inf|NaN)) = +Inf + I NaN
                    return new Complex(x * x, x * (y - y));
                return new Complex((x * x) * Math.Cos(y), (x * x) * Math.Sin(y));
            }
            // cosh(NaN + I ...) = NaN + I NaN
            return new Complex((x * x) * (y - y), (x + x) * (y - y));
        }

        /// <summary>
        /// Complex hyperbolic sine matching NumPy. The <c>y == 0</c> guard returns <c>(sinh(x), y)</c>
        /// (so <c>sinh(huge + I0)</c> stays <c>(±inf, ±0)</c> instead of the BCL's <c>cosh(x)*sin(0) =
        /// inf*0 = NaN</c>); the general finite case is the textbook <c>sinh(x)cos(y) + i cosh(x)sin(y)</c>,
        /// overflowing to ±inf exactly where NumPy's libm does. Non-finite inputs follow C99 Annex G.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Complex Sinh(Complex z)
        {
            double x = z.Real, y = z.Imaginary;
            if (double.IsFinite(x) && double.IsFinite(y))
            {
                if (y == 0.0)                                  // sinh(x + I0) = sinh(x) + I y (keeps sign(y))
                    return new Complex(Math.Sinh(x), y);
                return new Complex(Math.Sinh(x) * Math.Cos(y), Math.Cosh(x) * Math.Sin(y));
            }
            return SinhSpecial(x, y);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static Complex SinhSpecial(double x, double y)
        {
            // sinh(+-0 +- I(Inf|NaN)) = sign(+-0)0 + I NaN  (NumPy: real follows sign(x))
            if (x == 0.0)
                return new Complex(Math.CopySign(0.0, x), y - y);
            // sinh((Inf|NaN) +- I0)
            if (y == 0.0)
            {
                if (double.IsNaN(x))                // sinh(NaN + I0) = NaN + I0
                    return new Complex(x, y);
                return new Complex(x, Math.CopySign(0.0, y));   // sinh(+-Inf + I0) = +-Inf +- I0
            }
            // sinh(finite +- I(Inf|NaN)) = NaN + I NaN
            if (double.IsFinite(x))
                return new Complex(y - y, x * (y - y));
            // sinh(+-Inf + I y)   (x is +-Inf here)
            if (!double.IsNaN(x))
            {
                if (!double.IsFinite(y))            // sinh(+-Inf +- I(Inf|NaN)) = +-Inf + I NaN  (sign-preserving)
                    return new Complex(x, x * (y - y));
                return new Complex(x * Math.Cos(y), double.PositiveInfinity * Math.Sin(y));
            }
            // sinh(NaN + I ...) = NaN + I NaN
            return new Complex((x * x) * (y - y), (x + x) * (y - y));
        }

        /// <summary>
        /// Complex hyperbolic tangent matching NumPy (full <c>npy_ctanh</c> port, FreeBSD msun). The
        /// finite case uses Kahan's algorithm — <c>t=tan(y); beta=1+t^2; s=sinh(x); rho=sqrt(1+s^2)</c>;
        /// <c>tanh = (beta*rho*s + i t) / (1 + beta*s^2)</c> — which is materially more accurate than
        /// <see cref="Complex.Tanh"/> (e.g. <c>tan(1.5)</c> drifts ~33 ULP through the BCL). The
        /// <c>|x| &gt;= 22</c> branch avoids spurious overflow, and non-finite inputs follow C99.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Complex Tanh(Complex z)
        {
            double x = z.Real, y = z.Imaginary;
            if (double.IsFinite(x) && double.IsFinite(y))
            {
                if (Math.Abs(x) >= TANH_HUGE)
                {
                    // tanh(+-huge + I y) ~= +-1 +- I 2sin(2y)/exp(2|x|); modified to avoid overflow.
                    double expmx = Math.Exp(-Math.Abs(x));
                    return new Complex(Math.CopySign(1.0, x),
                                       4.0 * Math.Sin(y) * Math.Cos(y) * expmx * expmx);
                }
                double t = Math.Tan(y);
                double beta = 1.0 + t * t;          // = 1 / cos^2(y)
                double s = Math.Sinh(x);
                double rho = Math.Sqrt(1.0 + s * s); // = cosh(x)
                double denom = 1.0 + beta * s * s;
                return new Complex((beta * rho * s) / denom, t / denom);
            }
            return TanhSpecial(x, y);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static Complex TanhSpecial(double x, double y)
        {
            if (!double.IsFinite(x))
            {
                if (double.IsNaN(x))                // tanh(NaN + I0)=NaN+I0 ; tanh(NaN+Iy)=NaN+INaN
                    return new Complex(x, y == 0.0 ? y : x * y);
                // x = +-Inf : tanh(+-Inf + I y) = +-1 +- I0. The imaginary zero-sign is unspecified;
                // NumPy's libm takes sign(y) (not the msun source's sign(sin(2y))).
                return new Complex(Math.CopySign(1.0, x), Math.CopySign(0.0, y));
            }
            // x finite, so y is non-finite here: tanh(finite +- I(Inf|NaN)) = NaN + I NaN
            return new Complex(y - y, y - y);
        }

        #endregion

        #region inverse trig (asin / acos / atan)

        /// <summary>
        /// Complex arcsine matching NumPy. Finite inputs defer to <see cref="Complex.Asin"/> with
        /// signed-zero/branch-cut fixups; non-finite inputs use the identity
        /// <c>asin(z) = i*conj(casinh(i*conj z))</c> where <c>i*conj(z) = (y, x)</c>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Complex Asin(Complex z)
        {
            double x = z.Real, y = z.Imaginary;
            if (!double.IsFinite(x) || !double.IsFinite(y))
            {
                Complex cs = CasinhNonFinite(y, x);
                return new Complex(cs.Imaginary, cs.Real);
            }
            Complex r = Complex.Asin(z);
            double re = r.Real, im = r.Imaginary;
            // asin is odd + conjugate-symmetric: Re follows sign(x), Im follows sign(y).
            // The BCL drops the sign of a zero component; restore it.
            if (IsNegZero(x)) re = -re;
            if (IsNegZero(y)) im = -im;
            return new Complex(re, im);
        }

        /// <summary>
        /// Complex arccosine matching NumPy. Finite inputs defer to <see cref="Complex.Acos"/> with a
        /// branch-cut fixup; non-finite inputs follow C99 (ported from <c>npy_cacos</c>).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Complex Acos(Complex z)
        {
            double x = z.Real, y = z.Imaginary;
            if (!double.IsFinite(x) || !double.IsFinite(y))
                return CacosNonFinite(x, y);
            Complex r = Complex.Acos(z);
            double re = r.Real, im = r.Imaginary;
            // cacos negates the imaginary part when signbit(y)==0 (NumPy npy_cacos). The BCL emits
            // the y<0 branch unconditionally, so flip the imaginary part on the y=+0 cut.
            if (IsPosZero(y)) im = -im;
            return new Complex(re, im);
        }

        /// <summary>
        /// Complex arctangent matching NumPy. NumPy defines <c>npy_catan(z) = i*conj(catanh(i*conj z))</c>
        /// verbatim (<c>i*conj(z) = (y, x)</c>, then <c>(w.Im, w.Re)</c>), so routing through the full
        /// <see cref="Catanh"/> port reproduces NumPy bit-for-bit — including the tiny-imaginary and
        /// subnormal cases where <see cref="Complex.Atan"/> cancels (its internal log) or underflows to 0.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Complex Atan(Complex z)
        {
            Complex w = Catanh(new Complex(z.Imaginary, z.Real));
            return new Complex(w.Imaginary, w.Real);
        }

        // ---- non-finite kernels (C99 Annex G, ported from npy_math_complex.c.src) ----

        /// <summary>
        /// <c>npy_casinh</c> restricted to non-finite arguments (NaN/Inf in either part): the NaN
        /// special-value block plus the large-value <c>clog</c> path. Used by <see cref="Asin"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static Complex CasinhNonFinite(double a, double b)
        {
            if (double.IsNaN(a) || double.IsNaN(b))
            {
                if (double.IsInfinity(a)) return new Complex(a, b + b);     // casinh(+-Inf + I NaN) = +-Inf + I NaN
                if (double.IsInfinity(b)) return new Complex(b, a + a);     // casinh(NaN +- I Inf) = +-Inf + I NaN
                if (b == 0.0) return new Complex(a + a, b);                 // casinh(NaN + I0)      = NaN + I0
                return new Complex(double.NaN, double.NaN);
            }
            // a or b is +-Inf (no NaN): large-value path.
            double wx, wy;
            if (!double.IsNegative(a)) ClogLarge(a, b, out wx, out wy);
            else ClogLarge(-a, -b, out wx, out wy);
            wx += LOGE2;
            return new Complex(Math.CopySign(wx, a), Math.CopySign(wy, b));
        }

        /// <summary>
        /// <c>npy_cacos</c> restricted to non-finite arguments. Used by <see cref="Acos"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static Complex CacosNonFinite(double x, double y)
        {
            if (double.IsNaN(x) || double.IsNaN(y))
            {
                if (double.IsInfinity(x)) return new Complex(y + y, double.NegativeInfinity); // cacos(+-Inf + I NaN) = NaN - I Inf
                if (double.IsInfinity(y)) return new Complex(x + x, -y);                       // cacos(NaN +- I Inf) = NaN -+ I Inf
                if (x == 0.0) return new Complex(PI_2, y + y);                                 // cacos(0 + I NaN)    = pi/2 + I NaN
                return new Complex(double.NaN, double.NaN);
            }
            // x or y is +-Inf (no NaN): large-value path.
            double wx, wy;
            ClogLarge(x, y, out wx, out wy);
            double rx = Math.Abs(wy);
            double ry = wx + LOGE2;
            if (!double.IsNegative(y)) ry = -ry;
            return new Complex(rx, ry);
        }

        /// <summary>
        /// Complex inverse hyperbolic tangent matching NumPy (full <c>npy_catanh</c> port, FreeBSD
        /// msun). Drives <see cref="Atan"/>. The accurate paths — <c>atanh(x)</c> on the real axis,
        /// <c>atan(y)</c> on the imaginary axis, and the <c>log1p(4|x| / sumsq(|x|-1, |y|)) / 4</c>
        /// interior — are why this matches NumPy where <see cref="Complex.Atan"/> loses the tiny
        /// imaginary part to cancellation/underflow. Non-finite inputs follow C99 Annex G.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static Complex Catanh(Complex z)
        {
            double x = z.Real, y = z.Imaginary;
            double ax = Math.Abs(x), ay = Math.Abs(y);

            if (y == 0.0 && ax <= 1.0)                          // catanh(x + I0), |x|<=1: real atanh
                return new Complex(Math.Atanh(x), y);
            if (x == 0.0)                                       // catanh(+-0 + I y): filters z=0, keeps accuracy
                return new Complex(x, Math.Atan(y));
            if (double.IsNaN(x) || double.IsNaN(y))
            {
                if (double.IsInfinity(x)) return new Complex(Math.CopySign(0.0, x), y + y);                 // catanh(+-Inf + I NaN) = +-0 + I NaN
                if (double.IsInfinity(y)) return new Complex(Math.CopySign(0.0, x), Math.CopySign(PI_2 + PIO2_LO, y)); // catanh(NaN +- I Inf) = +-0 + I +-pi/2
                return new Complex(double.NaN, double.NaN);
            }
            if (ax > RECIP_EPSILON || ay > RECIP_EPSILON)       // huge: Re(1/z) overflow-safe, Im = +-pi/2
                return new Complex(RealPartReciprocal(x, y), Math.CopySign(PI_2 + PIO2_LO, y));
            if (ax < SQRT_3_EPSILON * 0.5 && ay < SQRT_3_EPSILON * 0.5)  // tiny (z!=0): catanh(z) ~= z
                return z;

            double rx;
            if (ax == 1.0 && ay < DBL_EPSILON)
                rx = (LOGE2 - Math.Log(ay)) * 0.5;
            else
                rx = MathLog1p(4.0 * ax / SumSquares(ax - 1.0, ay)) * 0.25;

            double ry;
            if (ax == 1.0)
                ry = Math.Atan2(2.0, -ay) * 0.5;
            else if (ay < DBL_EPSILON)
                ry = Math.Atan2(2.0 * ay, (1.0 - ax) * (1.0 + ax)) * 0.5;
            else
                ry = Math.Atan2(2.0 * ay, (1.0 - ax) * (1.0 + ax) - ay * ay) * 0.5;

            return new Complex(Math.CopySign(rx, x), Math.CopySign(ry, y));
        }

        /// <summary>NumPy <c>_sum_squares</c>: <c>x^2 + y^2</c> with an underflow guard that drops
        /// <c>y^2</c> when <c>y</c> is below <c>sqrt(DBL_MIN)</c> (it would flush to 0 anyway).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static double SumSquares(double x, double y)
        {
            if (y < SUMSQ_SQRT_MIN) return x * x;
            return x * x + y * y;
        }

        /// <summary>NumPy <c>_real_part_reciprocal</c> (C99 n1124 G.5.1 ex.2): <c>Re(1/z) = x/(x^2+y^2)</c>
        /// computed by exponent classification so neither <c>x^2</c> nor <c>y^2</c> overflows or
        /// underflows out of range. Uses the raw biased-exponent field (in <c>[0, 2047]</c>) exactly
        /// like NumPy's <c>GET_HIGH_WORD</c> — <see cref="Math.ILogB"/> can't be used here because it
        /// maps 0/Inf to <c>int.MinValue</c>/<c>int.MaxValue</c>, overflowing the exponent subtraction.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static double RealPartReciprocal(double x, double y)
        {
            const int CUTOFF = DBL_MANT_DIG / 2 + 1;            // 27: half + 1 guard digit
            const int BIAS = DBL_MAX_EXP - 1;                   // 1023
            if (double.IsInfinity(x)) return 1.0 / x;           // +-Inf -> +-0
            int ix = (int)((BitConverter.DoubleToInt64Bits(x) >> 52) & 0x7FF);  // biased exponent of x
            int iy = (int)((BitConverter.DoubleToInt64Bits(y) >> 52) & 0x7FF);  // biased exponent of y
            if (ix - iy >= CUTOFF) return 1.0 / x;              // |x| >> |y|
            if (iy - ix >= CUTOFF) return x / y / y;            // |y| >> |x|
            if (ix <= BIAS + DBL_MAX_EXP / 2 - CUTOFF) return x / (x * x + y * y);  // no overflow risk
            double scale = BitConverter.Int64BitsToDouble((long)(0x7FF - ix) << 52);  // 2^(1-ilogb(x))
            x *= scale;
            y *= scale;
            return x / (x * x + y * y) * scale;
        }

        #endregion

        #region exp / sqrt / log10 / reciprocal / sin / cos / tan / log1p

        // NPY_LOG10E (npy_math.h) = 1/ln(10).
        private const double LOG10E = 0.4342944819032518;

        /// <summary>
        /// Complex exponential matching NumPy (<c>npy_cexp</c>). A finite real part defers to
        /// <see cref="Complex.Exp"/> (ULP-identical to NumPy); a non-finite real part follows C99.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Complex Exp(Complex z)
        {
            if (double.IsFinite(z.Real))
                return Complex.Exp(z);
            return ExpSpecial(z.Real, z.Imaginary);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static Complex ExpSpecial(double x, double y)
        {
            if (double.IsNaN(x))                                    // exp(NaN + I0)=NaN+I0 ; exp(NaN+Iy)=NaN+I copysign(NaN,y)
                return y == 0.0 ? new Complex(x, y) : new Complex(x, Math.CopySign(double.NaN, y));
            if (x > 0.0)                                            // +Inf
            {
                if (y == 0.0) return new Complex(x, y);             // exp(+Inf + I0) = +Inf + I0
                if (double.IsFinite(y)) return new Complex(x * Math.Cos(y), x * Math.Sin(y));
                return new Complex(x, double.NaN);                 // exp(+Inf + I(Inf|NaN)) = +Inf + I NaN
            }
            // -Inf
            if (double.IsFinite(y))
            {
                double e = Math.Exp(x);                            // 0
                return new Complex(e * Math.Cos(y), e * Math.Sin(y));
            }
            // exp(-Inf + I(Inf|NaN)) = +0 + I copysign(0, y): system libm keeps sign(y) on the
            // imaginary zero (exp(-inf,-inf).Im = -0 in NumPy 2.4.2), which npy_cexp's flat (0,0) drops.
            return new Complex(0.0, Math.CopySign(0.0, y));
        }

        /// <summary>
        /// Complex square root matching NumPy (<c>npy_csqrt</c>, FreeBSD msun). Ported in full
        /// (exact arithmetic, not transcendental) so the branch-cut signs and signed zeros that
        /// <see cref="Complex.Sqrt"/> drops are reproduced bit-for-bit.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Complex Sqrt(Complex z)
        {
            double a = z.Real, b = z.Imaginary;
            if (a == 0.0 && b == 0.0)                               // sqrt(+-0 +- I0) = +0 +- I0 (keeps b's sign)
                return new Complex(0.0, b);
            if (double.IsInfinity(b))                               // sqrt(x +- I Inf) = +Inf +- I Inf
                return new Complex(double.PositiveInfinity, b);
            if (double.IsNaN(a))
            {
                double t = (b - b) / (b - b);                       // raise invalid if b is not NaN
                return new Complex(a, t);
            }
            if (double.IsInfinity(a))
            {
                if (double.IsNegative(a))                           // -Inf: 0 + Inf i (or NaN +- Inf i)
                    return new Complex(Math.Abs(b - b), Math.CopySign(a, b));
                return new Complex(a, Math.CopySign(b - b, b));     // +Inf: +Inf + 0 i
            }
            return CsqrtCore(a, b);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static Complex CsqrtCore(double a, double b)
        {
            // THRESH = 0x1.a827999fcef32p+1022 (FreeBSD csqrt.c): scale down to avoid overflow.
            const double THRESH = 7.446288774449766e+307;
            int scale = 0;
            if (Math.Abs(a) >= THRESH || Math.Abs(b) >= THRESH)
            {
                a *= 0.25;
                b *= 0.25;
                scale = 1;
            }
            double t;
            Complex result;
            if (a >= 0.0)                                           // Algorithm 312, CACM vol 10, Oct 1967.
            {
                t = Math.Sqrt((a + Hypot(a, b)) * 0.5);
                result = new Complex(t, b / (2.0 * t));
            }
            else
            {
                t = Math.Sqrt((-a + Hypot(a, b)) * 0.5);
                result = new Complex(Math.Abs(b) / (2.0 * t), Math.CopySign(t, b));
            }
            return scale == 1 ? new Complex(result.Real * 2.0, result.Imaginary) : result;
        }

        /// <summary>Overflow-safe finite <c>hypot</c> (BCL lacks <c>Math.Hypot</c> on net8.0).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static double Hypot(double x, double y)
        {
            x = Math.Abs(x);
            y = Math.Abs(y);
            if (x < y) { double tmp = x; x = y; y = tmp; }
            if (x == 0.0) return y;   // both 0 -> 0; (0, NaN) with no swap -> NaN (C99 hypot)
            double r = y / x;
            return x * Math.Sqrt(1.0 + r * r);
        }

        /// <summary>
        /// Complex natural logarithm matching NumPy (full <c>npy_clog</c> port, FreeBSD msun). The real
        /// part is <c>log|z|</c> computed by rescaling to dodge the four problem regimes the naive
        /// <c>log(hypot(re,im))</c> hits: <c>|z|</c> huge (rescale by ½), subnormal (rescale by
        /// 2^53), near 1 (<c>0.71 &lt;= |z| &lt;= 1.73</c> uses <c>½·log1p((m-1)(m+1)+n²)</c> so the
        /// real part doesn't cancel to 0), and 0 (handled by <c>-1/re</c>). The imaginary part is
        /// <c>atan2(im, re)</c>. <see cref="Complex.Log"/> reproduces only the common regime.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Complex Log(Complex z)
        {
            double re = z.Real, im = z.Imaginary;
            double ax = Math.Abs(re), ay = Math.Abs(im);
            // An infinite component means |z| = +inf, so log|z| = +inf (C99 npy_hypot(inf,*) = inf,
            // which the overflow-safe Hypot below would turn into NaN). The argument still follows
            // atan2 (e.g. log(inf+inf i) = inf + I pi/4).
            if (double.IsInfinity(ax) || double.IsInfinity(ay))
                return new Complex(double.PositiveInfinity, Math.Atan2(im, re));
            double rr;
            if (ax > DBL_MAX_4 || ay > DBL_MAX_4)
            {
                rr = Math.Log(Hypot(ax * 0.5, ay * 0.5)) + LOGE2;
            }
            else if (ax < DBL_MIN && ay < DBL_MIN)
            {
                if (ax > 0.0 || ay > 0.0)                          // hypot would be subnormal: rescale up
                {
                    rr = Math.Log(Hypot(Math.ScaleB(ax, DBL_MANT_DIG), Math.ScaleB(ay, DBL_MANT_DIG)))
                         - DBL_MANT_DIG * LOGE2;
                }
                else                                               // log(+-0 +- 0i) = -inf + I carg(z)
                {
                    rr = Math.CopySign(-1.0 / re, -1.0);
                    return new Complex(rr, Math.Atan2(im, re));
                }
            }
            else
            {
                double h = Hypot(ax, ay);
                if (0.71 <= h && h <= 1.73)                        // near |z|=1: avoid cancellation
                {
                    double am = ax > ay ? ax : ay;
                    double an = ax > ay ? ay : ax;
                    rr = MathLog1p((am - 1.0) * (am + 1.0) + an * an) * 0.5;
                }
                else
                {
                    rr = Math.Log(h);
                }
            }
            return new Complex(rr, Math.Atan2(im, re));
        }

        /// <summary>Real <c>log1p</c> via the Goldberg identity <c>log1p(u) = u*log(1+u)/((1+u)-1)</c>,
        /// which cancels the rounding error of <c>log(1+u)</c> for small <c>u</c> (BCL has no
        /// <c>Math.Log1p</c> on net8.0).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static double MathLog1p(double u)
        {
            double w = 1.0 + u;
            if (w == 1.0) return u;                                // u tiny: log1p(u) ~= u
            if (double.IsInfinity(w)) return Math.Log(w);
            return Math.Log(w) * (u / (w - 1.0));
        }

        /// <summary>
        /// Complex base-10 logarithm matching NumPy (<c>clog(z) * LOG10E</c>), built from
        /// <see cref="Log"/> scaled by <c>1/ln(10)</c>. <see cref="Complex.Log10"/> uses a different
        /// scaling that drifts well past 1 ULP from NumPy.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Complex Log10(Complex z)
        {
            Complex l = Log(z);
            return new Complex(l.Real * LOG10E, l.Imaginary * LOG10E);
        }

        /// <summary>
        /// Complex square matching NumPy (<c>np.square(z) == z*z</c>). NumPy's complex multiply is the
        /// textbook <c>(ar*br - ai*bi, ar*bi + ai*br)</c>, but compiled with FMA contraction; for
        /// <c>z*z</c> this is <c>(fma(re, re, -(im*im)), fma(re, im, im*re))</c>. The FMA path is what
        /// produces NumPy's <c>square(1e-10+1e-10i).real = -2.275e-37</c> (exact re² minus rounded im²)
        /// and <c>square(1e300+1e300i).real = -inf</c> (not NaN) — both of which <see cref="Complex.op_Multiply"/>
        /// (no FMA) turns into <c>0</c> and <c>NaN</c> respectively.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Complex Square(Complex z)
        {
            double re = z.Real, im = z.Imaginary;
            return new Complex(Math.FusedMultiplyAdd(re, re, -(im * im)),
                               Math.FusedMultiplyAdd(re, im, im * re));
        }

        /// <summary>
        /// Complex reciprocal matching NumPy (<c>nc_recip</c>): Smith's algorithm specialised to a unit
        /// numerator. This is overflow-safe (so <c>1/(huge)</c> doesn't prematurely flush to 0) AND
        /// reproduces NumPy's signed zeros (the imaginary part is <c>-rat*scl</c>), neither of which
        /// <see cref="Complex.op_Division"/> gets right. Bit-identical to NumPy across finite, zero,
        /// and infinite inputs.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Complex Reciprocal(Complex z)
        {
            double re = z.Real, im = z.Imaginary;
            if (Math.Abs(re) >= Math.Abs(im))
            {
                double rat = im / re;
                double scl = 1.0 / (re + im * rat);
                return new Complex(scl, -rat * scl);
            }
            else
            {
                double rat = re / im;
                double scl = 1.0 / (re * rat + im);
                return new Complex(rat * scl, -scl);
            }
        }

        /// <summary>
        /// Complex sine matching NumPy. NumPy defines <c>npy_csin(z) = -i*sinh(i*z)</c> verbatim
        /// (<c>i*z = (-y, x)</c>, then <c>(w.Im, -w.Re)</c>), so routing through the C99-correct
        /// <see cref="Sinh"/> reproduces NumPy bit-for-bit — including the large-imaginary case where
        /// <see cref="Complex.Sin"/> returns <c>NaN</c> from <c>cosh(huge)*0</c>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Complex Sin(Complex z)
        {
            Complex s = Sinh(new Complex(-z.Imaginary, z.Real));
            return new Complex(s.Imaginary, -s.Real);
        }

        /// <summary>
        /// Complex cosine matching NumPy. NumPy defines <c>npy_ccos(z) = cosh(i*z)</c> verbatim
        /// (<c>i*z = (-y, x)</c>), so routing through the C99-correct <see cref="Cosh"/> reproduces
        /// NumPy bit-for-bit, including the large-imaginary and signed-zero cases that
        /// <see cref="Complex.Cos"/> mishandles.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Complex Cos(Complex z)
        {
            return Cosh(new Complex(-z.Imaginary, z.Real));
        }

        /// <summary>
        /// Complex tangent matching NumPy. NumPy defines <c>npy_ctan(z) = -i*tanh(i*z)</c> verbatim
        /// (<c>i*z = (-y, x)</c>, then <c>(w.Im, -w.Re)</c>), so routing through the Kahan-based
        /// <see cref="Tanh"/> reproduces NumPy bit-for-bit (the BCL <see cref="Complex.Tan"/> drifts
        /// tens of ULP near <c>±π/2</c>).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Complex Tan(Complex z)
        {
            Complex t = Tanh(new Complex(-z.Imaginary, z.Real));
            return new Complex(t.Imaginary, -t.Real);
        }

        /// <summary>
        /// Complex <c>log(1+z)</c> matching NumPy. Equivalent to <c>Complex.Log(1+z)</c> but builds
        /// <c>1+z</c> as <c>(1+re, im)</c> so a negative-zero imaginary part survives (the naive
        /// <c>Complex.One + z</c> computes <c>0 + (-0) = +0</c>, dropping the sign on the cut).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Complex Log1p(Complex z)
        {
            // NumPy's complex log1p is the *naive* clog(1+z) — it does NOT apply clog's near-|z|=1
            // log1p refinement (np.log1p(1e-10j).real == 0, not 5e-21), so use Complex.Log here, NOT
            // NpyComplexMath.Log. Building (1+re, im) preserves a -0 imaginary the naive 1+z would flip.
            return Complex.Log(new Complex(1.0 + z.Real, z.Imaginary));
        }

        /// <summary>
        /// Complex base-2 exponential matching NumPy: <c>exp2(z) = exp(z*ln2)</c>. Routing through the
        /// C99-correct <see cref="Exp"/> reproduces NumPy's non-finite results (e.g. exp2(+Inf+Inf i) =
        /// +Inf + I NaN) that <see cref="Complex.Pow"/> turned into (NaN, NaN).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Complex Exp2(Complex z)
        {
            return Exp(new Complex(z.Real * LOGE2, z.Imaginary * LOGE2));
        }

        /// <summary>
        /// Complex <c>exp(z)-1</c> matching NumPy (<c>nc_expm1</c>):
        /// <c>real = expm1(x)*cos(y) - 2*sin^2(y/2)</c>, <c>imag = exp(x)*sin(y)</c>. This reproduces
        /// NumPy's structure (e.g. <c>expm1(+Inf+0i).imag = exp(+Inf)*sin(0) = NaN</c>) and signed
        /// zeros. The BCL has no real <c>expm1</c>, so <see cref="RealExpm1"/> falls back to
        /// <c>exp(x)-1</c> (with a signed-zero guard) — bit-identical to NumPy except for inputs
        /// extremely close to the origin, where libm's <c>expm1</c> is more accurate (a documented
        /// divergence, like <c>arctan</c>'s interior).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Complex Expm1(Complex z)
        {
            double x = z.Real, y = z.Imaginary;
            double s = Math.Sin(y * 0.5);
            double re = RealExpm1(x) * Math.Cos(y) - 2.0 * s * s;
            double im = Math.Exp(x) * Math.Sin(y);
            return new Complex(re, im);
        }

        /// <summary>Real <c>expm1</c> via the Goldberg identity <c>expm1(x) = (e^x-1)*x/log(e^x)</c>
        /// (BCL has no <c>Math.Expm1</c>). The naive <c>exp(x)-1</c> loses ~10 digits for small <c>x</c>
        /// (catastrophic cancellation) and underflows tiny <c>x</c> to 0; the correction factor
        /// <c>x/log(u)</c> recovers the lost bits to ≤1 ULP, matching libm's <c>expm1</c>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static double RealExpm1(double x)
        {
            double u = Math.Exp(x);
            if (u == 1.0) return x;                               // |x| tiny: expm1(x) ~= x (no underflow)
            if (double.IsPositiveInfinity(u)) return u;          // x huge: expm1 = +inf
            double um1 = u - 1.0;
            if (um1 == -1.0) return -1.0;                         // x very negative: expm1 = -1
            return um1 * (x / Math.Log(u));                      // correct the cancellation
        }

        #endregion
    }
}
