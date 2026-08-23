using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

// =============================================================================
// Managed port of pocketfft (numpy 2.4.2's vendored engine, pinned commit
// 33ae5dc9): src/pocketfft/pocketfft_hdronly.h.
//
// This file ports:
//   * cmplx<T0>            -> struct Cmplx           (lines 236-294)
//   * sincos_2pibyn<T>     -> class SinCos2PiByN     (lines 299-372)
//   * util subset          -> static class PocketFFTUtil (lines 373-431)
//
// numpy compiles pocketfft with MSVC (POCKETFFT_NO_VECTORS is left DEFINED for
// MSVC, so VLEN==1 and the whole engine is SCALAR on win-amd64) and long double
// == double, so a faithful scalar double port reproduces numpy's per-transform
// arithmetic. Operation ORDER is preserved verbatim to keep bit-parity.
// =============================================================================

namespace NumSharp.Fourier
{
    /// <summary>
    /// Port of pocketfft's <c>cmplx&lt;T0&gt;</c> for the double engine (T0 == double).
    /// The field layout (<c>r</c> then <c>i</c>) is identical to
    /// <see cref="System.Numerics.Complex"/> (real then imaginary), so the two are
    /// bit-reinterpretable at the driver boundary. All operators reproduce
    /// pocketfft's exact real/imag operation order.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Cmplx
    {
        public double r, i;

        [MethodImpl(OptimizeAndInline)]
        public Cmplx(double r_, double i_) { r = r_; i = i_; }

        // cmplx + cmplx  -> {r+o.r, i+o.i}
        [MethodImpl(OptimizeAndInline)]
        public static Cmplx operator +(in Cmplx a, in Cmplx b) => new Cmplx(a.r + b.r, a.i + b.i);

        // cmplx - cmplx  -> {r-o.r, i-o.i}
        [MethodImpl(OptimizeAndInline)]
        public static Cmplx operator -(in Cmplx a, in Cmplx b) => new Cmplx(a.r - b.r, a.i - b.i);

        // cmplx * scalar -> {r*s, i*s}
        [MethodImpl(OptimizeAndInline)]
        public static Cmplx operator *(in Cmplx a, double s) => new Cmplx(a.r * s, a.i * s);

        // cmplx * cmplx  -> {r*o.r - i*o.i, r*o.i + i*o.r}
        [MethodImpl(OptimizeAndInline)]
        public static Cmplx operator *(in Cmplx a, in Cmplx b)
            => new Cmplx(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r);

        /// <summary>
        /// pocketfft <c>special_mul&lt;fwd&gt;</c> (twiddle multiply with direction-dependent
        /// conjugation). fwd: <c>(r*o.r+i*o.i, i*o.r-r*o.i)</c>; bwd: <c>(r*o.r-i*o.i, r*o.i+i*o.r)</c>.
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
        public static Cmplx SpecialMul(bool fwd, in Cmplx v1, in Cmplx v2)
            => fwd
                ? new Cmplx(v1.r * v2.r + v1.i * v2.i, v1.i * v2.r - v1.r * v2.i)
                : new Cmplx(v1.r * v2.r - v1.i * v2.i, v1.r * v2.i + v1.i * v2.r);
    }

    /// <summary>
    /// Port of pocketfft's <c>sincos_2pibyn&lt;T&gt;</c> (lines 299-372) — accurate twiddle
    /// factors exp(-2πi·idx/N) via angle reduction into <c>[0,π/4]</c> and a two-level table so
    /// only ~2√n trig calls are made. <b>Parity-critical</b>: the recurrence and multiply order
    /// are kept exact. Thigh == double for the double engine.
    /// </summary>
    public sealed class SinCos2PiByN
    {
        private readonly long N, mask, shift;
        private readonly Cmplx[] v1, v2;

        // pocketfft calc(): reduce x (in units of n/8 by the <<3) into the correct octant and
        // emit cos/sin of the reduced angle. std::cos/std::sin on MSVC == ucrtbase cos/sin ==
        // .NET Math.Cos/Math.Sin on Windows (same CRT), so this is bit-identical to numpy here.
        private static Cmplx Calc(long x, long n, double ang)
        {
            x <<= 3;
            if (x < 4 * n) // first half
            {
                if (x < 2 * n) // first quadrant
                {
                    if (x < n) return new Cmplx(Math.Cos(x * ang), Math.Sin(x * ang));
                    return new Cmplx(Math.Sin((2 * n - x) * ang), Math.Cos((2 * n - x) * ang));
                }
                else // second quadrant
                {
                    x -= 2 * n;
                    if (x < n) return new Cmplx(-Math.Sin(x * ang), Math.Cos(x * ang));
                    return new Cmplx(-Math.Cos((2 * n - x) * ang), Math.Sin((2 * n - x) * ang));
                }
            }
            else
            {
                x = 8 * n - x;
                if (x < 2 * n) // third quadrant
                {
                    if (x < n) return new Cmplx(Math.Cos(x * ang), -Math.Sin(x * ang));
                    return new Cmplx(Math.Sin((2 * n - x) * ang), -Math.Cos((2 * n - x) * ang));
                }
                else // fourth quadrant
                {
                    x -= 2 * n;
                    if (x < n) return new Cmplx(-Math.Sin(x * ang), -Math.Cos(x * ang));
                    return new Cmplx(-Math.Cos((2 * n - x) * ang), -Math.Sin((2 * n - x) * ang));
                }
            }
        }

        public SinCos2PiByN(long n)
        {
            N = n;
            // MSVC long double == double, so 0.25L*pi/n is computed in double just as here.
            const double pi = 3.141592653589793238462643383279502884197;
            double ang = 0.25 * pi / n;
            long nval = (n + 2) / 2;
            long sh = 1;
            while (((long)1 << (int)sh) * ((long)1 << (int)sh) < nval) ++sh;
            shift = sh;
            mask = ((long)1 << (int)shift) - 1;
            v1 = new Cmplx[mask + 1];
            v1[0] = new Cmplx(1.0, 0.0);
            for (long i = 1; i < v1.Length; ++i)
                v1[i] = Calc(i, n, ang);
            v2 = new Cmplx[(nval + mask) / (mask + 1)];
            v2[0] = new Cmplx(1.0, 0.0);
            for (long i = 1; i < v2.Length; ++i)
                v2[i] = Calc(i * (mask + 1), n, ang);
        }

        public Cmplx this[long idx]
        {
            get
            {
                if (2 * idx <= N)
                {
                    Cmplx x1 = v1[idx & mask], x2 = v2[idx >> (int)shift];
                    return new Cmplx(x1.r * x2.r - x1.i * x2.i, x1.r * x2.i + x1.i * x2.r);
                }
                idx = N - idx;
                {
                    Cmplx x1 = v1[idx & mask], x2 = v2[idx >> (int)shift];
                    return new Cmplx(x1.r * x2.r - x1.i * x2.i, -(x1.r * x2.i + x1.i * x2.r));
                }
            }
        }
    }

    /// <summary>
    /// Port of the pocketfft <c>util</c> subset used by 1-D planning (lines 373-431):
    /// prime-factor test, cost model and the "good size" search for Bluestein padding.
    /// </summary>
    public static class PocketFFTUtil
    {
        public static long LargestPrimeFactor(long n)
        {
            long res = 1;
            while ((n & 1) == 0) { res = 2; n >>= 1; }
            for (long x = 3; x * x <= n; x += 2)
                while ((n % x) == 0) { res = x; n /= x; }
            if (n > 1) res = n;
            return res;
        }

        public static double CostGuess(long n)
        {
            const double lfp = 1.1; // penalty for non-hardcoded larger factors
            long ni = n;
            double result = 0.0;
            while ((n & 1) == 0) { result += 2; n >>= 1; }
            for (long x = 3; x * x <= n; x += 2)
                while ((n % x) == 0)
                {
                    result += (x <= 5) ? (double)x : lfp * (double)x;
                    n /= x;
                }
            if (n > 1) result += (n <= 5) ? (double)n : lfp * (double)n;
            return result * (double)ni;
        }

        /// <summary>smallest composite of 2,3,5,7,11 which is &gt;= n.</summary>
        public static long GoodSizeCmplx(long n)
        {
            if (n <= 12) return n;

            long bestfac = 2 * n;
            for (long f11 = 1; f11 < bestfac; f11 *= 11)
                for (long f117 = f11; f117 < bestfac; f117 *= 7)
                    for (long f1175 = f117; f1175 < bestfac; f1175 *= 5)
                    {
                        long x = f1175;
                        while (x < n) x *= 2;
                        for (; ; )
                        {
                            if (x < n)
                                x *= 3;
                            else if (x > n)
                            {
                                if (x < bestfac) bestfac = x;
                                if ((x & 1) != 0) break;
                                x >>= 1;
                            }
                            else
                                return n;
                        }
                    }
            return bestfac;
        }

        /// <summary>smallest composite of 2,3,5 which is &gt;= n.</summary>
        public static long GoodSizeReal(long n)
        {
            if (n <= 6) return n;

            long bestfac = 2 * n;
            for (long f5 = 1; f5 < bestfac; f5 *= 5)
            {
                long x = f5;
                while (x < n) x *= 2;
                for (; ; )
                {
                    if (x < n)
                        x *= 3;
                    else if (x > n)
                    {
                        if (x < bestfac) bestfac = x;
                        if ((x & 1) != 0) break;
                        x >>= 1;
                    }
                    else
                        return n;
                }
            }
            return bestfac;
        }
    }
}
