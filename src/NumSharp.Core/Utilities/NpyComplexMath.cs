using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Complex-math helpers matching NumPy's complex ufunc edge semantics where the
    /// .NET BCL diverges.
    ///
    /// <para><see cref="Abs"/> replicates NumPy's <c>npy_cabs</c>
    /// (<c>= npy_hypot(creal(z), cimag(z))</c>, <c>npy_math_complex.c.src</c>): under C99
    /// <c>hypot</c>, an infinite component yields <c>+inf</c> <em>even when the other component
    /// is NaN</em> (the infinity test precedes the NaN test). <see cref="Complex.Abs"/> routes
    /// through a private <c>Hypot</c> that orders its operands with a NaN-unaware comparison, so on
    /// <c>net8.0</c> it returns <c>NaN</c> for <c>abs(NaN + inf*i)</c> instead of <c>+inf</c>
    /// (fixed in the .NET 9+ BCL). Guarding the infinity case explicitly makes every target
    /// framework agree with NumPy while leaving the finite/NaN-only magnitudes — which already
    /// match NumPy bit-for-bit — to <see cref="Complex.Abs"/>.</para>
    ///
    /// <para><see cref="Sinh"/>/<see cref="Cosh"/>/<see cref="Tanh"/>/<see cref="Asin"/>/
    /// <see cref="Acos"/>/<see cref="Atan"/> match NumPy's complex hyperbolic and inverse-trig
    /// ufuncs. For finite inputs the .NET BCL agrees with NumPy to within a few ULP (verified by a
    /// bit-exact probe over a finite battery), so those delegate straight to <c>System.Numerics.Complex</c>.
    /// Where the BCL diverges — non-finite inputs (it returns <c>(NaN,NaN)</c> instead of the C99
    /// Annex G values), branch-cut sign, and dropped signed-zeros — these helpers apply the C99
    /// fixups. The non-finite paths and the special-value tables are ported faithfully from
    /// NumPy's <c>npy_math_complex.c.src</c> (the FreeBSD msun <c>ccosh</c>/<c>csinh</c>/<c>ctanh</c>/
    /// <c>cacos</c>/<c>casinh</c>/<c>catanh</c> implementations); <c>asin</c>/<c>atan</c> reuse
    /// NumPy's identities <c>asin(z)=i*conj(casinh(i*conj z))</c> and
    /// <c>atan(z)=i*conj(catanh(i*conj z))</c>. The lone documented divergence is <c>arctan</c>'s
    /// finite interior, where the BCL stays within 3 ULP of NumPy (and ~1e-7 relative for inputs
    /// extremely close to the origin) because .NET's <c>Atan</c> uses a less accurate kernel than
    /// NumPy's <c>log1p</c>-based one — this is the agreed cost of delegating the interior.</para>
    /// </summary>
    public static class NpyComplexMath
    {
        // NPY_PI_2 / NPY_LOGE2 (npy_math.h). Math.PI/2 == NPY_PI_2 bit-for-bit.
        private const double PI_2 = Math.PI / 2.0;
        private const double LOGE2 = 0.6931471805599453;

        /// <summary>
        /// <c>|z| = hypot(re, im)</c> with NumPy/C99 infinity semantics: a ±infinite real or
        /// imaginary part returns <c>+inf</c> regardless of the other part (including NaN).
        /// All other inputs defer to <see cref="Complex.Abs"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Abs(Complex z)
        {
            if (double.IsInfinity(z.Real) || double.IsInfinity(z.Imaginary))
                return double.PositiveInfinity;
            return Complex.Abs(z);
        }

        #region helpers

        /// <summary>True when <paramref name="d"/> is negative zero (<c>-0.0</c>).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsNegZero(double d) => d == 0.0 && double.IsNegative(d);

        /// <summary>True when <paramref name="d"/> is positive zero (<c>+0.0</c>).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsPosZero(double d) => d == 0.0 && !double.IsNegative(d);

        /// <summary><c>hypot</c> that returns <c>+inf</c> if either part is infinite (C99), used by
        /// the large-value / non-finite inverse-trig paths.</summary>
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
        private static void ClogLarge(double x, double y, out double rr, out double ri)
        {
            rr = Math.Log(HypotInf(x, y));
            ri = Math.Atan2(y, x);
        }

        #endregion

        #region hyperbolic (sinh / cosh / tanh)

        /// <summary>
        /// Complex hyperbolic cosine matching NumPy. Finite inputs defer to <see cref="Complex.Cosh"/>;
        /// non-finite inputs follow C99 Annex G (ported from <c>npy_ccosh</c>).
        /// </summary>
        public static Complex Cosh(Complex z)
        {
            double x = z.Real, y = z.Imaginary;
            if (double.IsFinite(x) && double.IsFinite(y))
                return Complex.Cosh(z);

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
        /// Complex hyperbolic sine matching NumPy. Finite inputs defer to <see cref="Complex.Sinh"/>;
        /// non-finite inputs follow C99 Annex G (ported from <c>npy_csinh</c>).
        /// </summary>
        public static Complex Sinh(Complex z)
        {
            double x = z.Real, y = z.Imaginary;
            if (double.IsFinite(x) && double.IsFinite(y))
                return Complex.Sinh(z);

            // sinh(+-0 +- I(Inf|NaN)) = sign(+-0)0 + I NaN  (NumPy: real follows sign(x))
            if (x == 0.0)
                return new Complex(Math.CopySign(0.0, x), y - y);
            // sinh((Inf|NaN) +- I0)
            if (y == 0.0)
            {
                if (double.IsNaN(x))                // sinh(NaN + I0) = NaN + I0
                    return z;
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
        /// Complex hyperbolic tangent matching NumPy. Finite inputs defer to <see cref="Complex.Tanh"/>;
        /// non-finite inputs follow C99 Annex G (ported from <c>npy_ctanh</c>).
        /// </summary>
        public static Complex Tanh(Complex z)
        {
            double x = z.Real, y = z.Imaginary;
            if (!double.IsFinite(x))
            {
                if (double.IsNaN(x))                // tanh(NaN + I0)=NaN+I0 ; tanh(NaN+Iy)=NaN+INaN
                    return new Complex(x, y == 0.0 ? y : x * y);
                // x = +-Inf : tanh(+-Inf + I y) = +-1 +- I0. The imaginary zero-sign is unspecified;
                // NumPy's libm takes sign(y) (not the msun source's sign(sin(2y))).
                return new Complex(Math.CopySign(1.0, x), Math.CopySign(0.0, y));
            }
            if (!double.IsFinite(y))                // tanh(finite +- I(Inf|NaN)) = NaN + I NaN
                return new Complex(y - y, y - y);
            return Complex.Tanh(z);
        }

        #endregion

        #region inverse trig (asin / acos / atan)

        /// <summary>
        /// Complex arcsine matching NumPy. Finite inputs defer to <see cref="Complex.Asin"/> with
        /// signed-zero/branch-cut fixups; non-finite inputs use the identity
        /// <c>asin(z) = i*conj(casinh(i*conj z))</c> where <c>i*conj(z) = (y, x)</c>.
        /// </summary>
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
        /// Complex arctangent matching NumPy. Finite inputs defer to <see cref="Complex.Atan"/> with
        /// imaginary-axis and signed-zero fixups; non-finite inputs use the identity
        /// <c>atan(z) = i*conj(catanh(i*conj z))</c> where <c>i*conj(z) = (y, x)</c>.
        /// </summary>
        public static Complex Atan(Complex z)
        {
            double x = z.Real, y = z.Imaginary;
            if (!double.IsFinite(x) || !double.IsFinite(y))
            {
                Complex ct = CatanhNonFinite(y, x);
                return new Complex(ct.Imaginary, ct.Real);
            }
            Complex r = Complex.Atan(z);
            double re = r.Real, im = r.Imaginary;
            if (x == 0.0)
            {
                // On the imaginary axis the BCL's real part is wrong (it keys off the sign of y, or
                // emits NaN at the +-i poles). NumPy: Re = copysign(|y|>1 ? pi/2 : 0, x).
                re = Math.CopySign(Math.Abs(y) > 1.0 ? PI_2 : 0.0, x);
                if (IsNegZero(y)) im = -im;
            }
            else if (IsNegZero(y))
            {
                im = -im;
            }
            return new Complex(re, im);
        }

        // ---- non-finite kernels (C99 Annex G, ported from npy_math_complex.c.src) ----

        /// <summary>
        /// <c>npy_casinh</c> restricted to non-finite arguments (NaN/Inf in either part): the NaN
        /// special-value block plus the large-value <c>clog</c> path. Used by <see cref="Asin"/>.
        /// </summary>
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
        /// <c>npy_catanh</c> for the arguments reached by <see cref="Atan"/>'s non-finite path: the
        /// early <c>x==0</c> / <c>y==0&amp;&amp;|x|&lt;=1</c> returns, the NaN block, and the
        /// large-value path.
        /// </summary>
        private static Complex CatanhNonFinite(double a, double b)
        {
            if (b == 0.0 && Math.Abs(a) <= 1.0)
                return new Complex(Math.Atanh(a), b);          // catanh(x + I0), |x|<=1
            if (a == 0.0)
                return new Complex(a, Math.Atan(b));           // catanh(+-0 + I y) -> filters z=0 and keeps accuracy
            if (double.IsNaN(a) || double.IsNaN(b))
            {
                if (double.IsInfinity(a)) return new Complex(Math.CopySign(0.0, a), b + b);              // catanh(+-Inf + I NaN) = +-0 + I NaN
                if (double.IsInfinity(b)) return new Complex(Math.CopySign(0.0, a), Math.CopySign(PI_2, b)); // catanh(NaN +- I Inf) = +-0 + I +-pi/2
                return new Complex(double.NaN, double.NaN);
            }
            // a or b is +-Inf (no NaN): Re(1/z) -> +-0 (sign of a), Im -> copysign(pi/2, b).
            return new Complex(Math.CopySign(0.0, a), Math.CopySign(PI_2, b));
        }

        #endregion
    }
}
