using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

// =============================================================================
// Port of pocketfft's cfftp<T0> — the complex mixed-radix FFTPACK transform
// (pocketfft_hdronly.h lines 878-1598). Codelets pass2/3/4/5/7/8/11 + the
// generic passg, factorize(), comp_twiddle() and pass_all (the two-buffer
// ping-pong with fct scaling). Double engine (T0 == double).
//
// Every butterfly is transcribed with pocketfft's exact operation order. The
// PARTSTEP/PREP/ROTX macros are expanded inline (they are inlined in C++ too).
// The "twiddle imaginary sign" token trick in the C macros (e.g. `twai*t4.r
// twbi*t3.r`) is reproduced by passing the already-signed twiddle doubles.
// =============================================================================

namespace NumSharp.Fourier
{
    internal sealed unsafe class Cfftp
    {
        private sealed class FctData
        {
            public long fct;
            public Cmplx[] tw;   // (ip-1)*(ido-1) twiddles; may be empty
            public Cmplx[] tws;  // ip extra twiddles for passg (ip>11); else null
        }

        private readonly long length;
        private readonly List<FctData> fact = new List<FctData>();

        public long Length => length;

        public Cfftp(long length_)
        {
            length = length_;
            if (length == 0) throw new ArgumentException("zero-length FFT requested");
            if (length == 1) return;
            Factorize();
            CompTwiddle();
        }

        private void AddFactor(long factor) => fact.Add(new FctData { fct = factor });

        // ---- pocketfft cfftp::factorize (8s, 4s, a 2 moved to the front, then odd factors) ----
        private void Factorize()
        {
            long len = length;
            while ((len & 7) == 0) { AddFactor(8); len >>= 3; }
            while ((len & 3) == 0) { AddFactor(4); len >>= 2; }
            if ((len & 1) == 0)
            {
                len >>= 1;
                // factor 2 should be at the front of the factor list
                AddFactor(2);
                long tmp = fact[0].fct; fact[0].fct = fact[fact.Count - 1].fct; fact[fact.Count - 1].fct = tmp;
            }
            for (long divisor = 3; divisor * divisor <= len; divisor += 2)
                while ((len % divisor) == 0) { AddFactor(divisor); len /= divisor; }
            if (len > 1) AddFactor(len);
        }

        private void CompTwiddle()
        {
            var twiddle = new SinCos2PiByN(length);
            long l1 = 1;
            for (int k = 0; k < fact.Count; ++k)
            {
                long ip = fact[k].fct, ido = length / (l1 * ip);
                var tw = new Cmplx[(ip - 1) * (ido - 1)];
                for (long j = 1; j < ip; ++j)
                    for (long i = 1; i < ido; ++i)
                        tw[(j - 1) * (ido - 1) + i - 1] = twiddle[j * l1 * i];
                fact[k].tw = tw;
                if (ip > 11)
                {
                    var tws = new Cmplx[ip];
                    for (long j = 0; j < ip; ++j)
                        tws[j] = twiddle[j * l1 * ido];
                    fact[k].tws = tws;
                }
                l1 *= ip;
            }
        }

        // ================= local rotation / butterfly helpers (operate on locals) =================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ROT90(ref Cmplx a) { double t = a.r; a.r = -a.i; a.i = t; }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ROTX90(bool fwd, ref Cmplx a)
        { double t = fwd ? -a.r : a.r; a.r = fwd ? a.i : -a.i; a.i = t; }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void PMINPLACE(ref Cmplx a, ref Cmplx b) { Cmplx t = a; a = a + b; b = t - b; }

        private const double HSQT2 = 0.707106781186547524400844362104849;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ROTX45(bool fwd, ref Cmplx a)
        {
            if (fwd) { double t = a.r; a.r = HSQT2 * (a.r + a.i); a.i = HSQT2 * (a.i - t); }
            else { double t = a.r; a.r = HSQT2 * (a.r - a.i); a.i = HSQT2 * (a.i + t); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ROTX135(bool fwd, ref Cmplx a)
        {
            if (fwd) { double t = a.r; a.r = HSQT2 * (a.i - a.r); a.i = HSQT2 * (-t - a.i); }
            else { double t = a.r; a.r = HSQT2 * (-a.r - a.i); a.i = HSQT2 * (t - a.i); }
        }

        // PARTSTEP helpers: return (ca, cb). The imaginary twiddle args (twai/twbi, y*) are
        // already-signed doubles reproducing the C macro's sign-token concatenation.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static (Cmplx ca, Cmplx cb) Part3(in Cmplx t0, in Cmplx t1, in Cmplx t2, double twr, double twi)
        {
            Cmplx ca = t0 + t1 * twr;
            Cmplx cb = new Cmplx(-t2.i * twi, t2.r * twi);
            return (ca, cb);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static (Cmplx ca, Cmplx cb) Part5(in Cmplx t0, in Cmplx t1, in Cmplx t2, in Cmplx t3, in Cmplx t4,
            double twar, double twbr, double twai, double twbi)
        {
            Cmplx ca, cb;
            ca.r = t0.r + twar * t1.r + twbr * t2.r;
            ca.i = t0.i + twar * t1.i + twbr * t2.i;
            cb.i = twai * t4.r + twbi * t3.r;
            cb.r = -(twai * t4.i + twbi * t3.i);
            return (ca, cb);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static (Cmplx ca, Cmplx cb) Part7(in Cmplx t1, in Cmplx t2, in Cmplx t3, in Cmplx t4,
            in Cmplx t5, in Cmplx t6, in Cmplx t7, double x1, double x2, double x3, double y1, double y2, double y3)
        {
            Cmplx ca, cb;
            ca.r = t1.r + x1 * t2.r + x2 * t3.r + x3 * t4.r;
            ca.i = t1.i + x1 * t2.i + x2 * t3.i + x3 * t4.i;
            cb.i = y1 * t7.r + y2 * t6.r + y3 * t5.r;
            cb.r = -(y1 * t7.i + y2 * t6.i + y3 * t5.i);
            return (ca, cb);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static (Cmplx ca, Cmplx cb) Part11(in Cmplx t1, in Cmplx t2, in Cmplx t3, in Cmplx t4, in Cmplx t5,
            in Cmplx t6, in Cmplx t7, in Cmplx t8, in Cmplx t9, in Cmplx t10, in Cmplx t11,
            double x1, double x2, double x3, double x4, double x5,
            double y1, double y2, double y3, double y4, double y5)
        {
            Cmplx ca = t1 + t2 * x1 + t3 * x2 + t4 * x3 + t5 * x4 + t6 * x5;
            Cmplx cb;
            cb.i = y1 * t11.r + y2 * t10.r + y3 * t9.r + y4 * t8.r + y5 * t7.r;
            cb.r = -(y1 * t11.i + y2 * t10.i + y3 * t9.i + y4 * t8.i + y5 * t7.i);
            return (ca, cb);
        }

        // ================================ codelets ================================

        private static void Pass2(long ido, long l1, Cmplx* cc, Cmplx* ch, Cmplx* wa, bool fwd)
        {
            long CH(long a, long b, long c) => a + ido * (b + l1 * c);
            long CC(long a, long b, long c) => a + ido * (b + 2 * c);
            long WA(long x, long i) => i - 1 + x * (ido - 1);

            if (ido == 1)
                for (long k = 0; k < l1; ++k)
                {
                    ch[CH(0, k, 0)] = cc[CC(0, 0, k)] + cc[CC(0, 1, k)];
                    ch[CH(0, k, 1)] = cc[CC(0, 0, k)] - cc[CC(0, 1, k)];
                }
            else
                for (long k = 0; k < l1; ++k)
                {
                    ch[CH(0, k, 0)] = cc[CC(0, 0, k)] + cc[CC(0, 1, k)];
                    ch[CH(0, k, 1)] = cc[CC(0, 0, k)] - cc[CC(0, 1, k)];
                    for (long i = 1; i < ido; ++i)
                    {
                        ch[CH(i, k, 0)] = cc[CC(i, 0, k)] + cc[CC(i, 1, k)];
                        ch[CH(i, k, 1)] = Cmplx.SpecialMul(fwd, cc[CC(i, 0, k)] - cc[CC(i, 1, k)], wa[WA(0, i)]);
                    }
                }
        }

        private static void Pass3(long ido, long l1, Cmplx* cc, Cmplx* ch, Cmplx* wa, bool fwd)
        {
            const double tw1r = -0.5;
            double tw1i = (fwd ? -1.0 : 1.0) * 0.8660254037844386467637231707529362;

            long CH(long a, long b, long c) => a + ido * (b + l1 * c);
            long CC(long a, long b, long c) => a + ido * (b + 3 * c);
            long WA(long x, long i) => i - 1 + x * (ido - 1);

            for (long k = 0; k < l1; ++k)
            {
                {
                    // PREP3(0)
                    Cmplx t0 = cc[CC(0, 0, k)];
                    Cmplx t1 = cc[CC(0, 1, k)] + cc[CC(0, 2, k)];
                    Cmplx t2 = cc[CC(0, 1, k)] - cc[CC(0, 2, k)];
                    ch[CH(0, k, 0)] = t0 + t1;
                    // PARTSTEP3a(1,2,tw1r,tw1i)
                    var (ca, cb) = Part3(t0, t1, t2, tw1r, tw1i);
                    ch[CH(0, k, 1)] = ca + cb;
                    ch[CH(0, k, 2)] = ca - cb;
                }
                for (long i = 1; i < ido; ++i)
                {
                    // PREP3(i)
                    Cmplx t0 = cc[CC(i, 0, k)];
                    Cmplx t1 = cc[CC(i, 1, k)] + cc[CC(i, 2, k)];
                    Cmplx t2 = cc[CC(i, 1, k)] - cc[CC(i, 2, k)];
                    ch[CH(i, k, 0)] = t0 + t1;
                    // PARTSTEP3b(1,2,tw1r,tw1i)
                    var (ca, cb) = Part3(t0, t1, t2, tw1r, tw1i);
                    ch[CH(i, k, 1)] = Cmplx.SpecialMul(fwd, ca + cb, wa[WA(0, i)]);
                    ch[CH(i, k, 2)] = Cmplx.SpecialMul(fwd, ca - cb, wa[WA(1, i)]);
                }
            }
        }

        private static void Pass4(long ido, long l1, Cmplx* cc, Cmplx* ch, Cmplx* wa, bool fwd)
        {
            long CH(long a, long b, long c) => a + ido * (b + l1 * c);
            long CC(long a, long b, long c) => a + ido * (b + 4 * c);
            long WA(long x, long i) => i - 1 + x * (ido - 1);

            if (ido == 1)
                for (long k = 0; k < l1; ++k)
                {
                    Cmplx t1, t2, t3, t4;
                    t2 = cc[CC(0, 0, k)] + cc[CC(0, 2, k)]; t1 = cc[CC(0, 0, k)] - cc[CC(0, 2, k)];
                    t3 = cc[CC(0, 1, k)] + cc[CC(0, 3, k)]; t4 = cc[CC(0, 1, k)] - cc[CC(0, 3, k)];
                    ROTX90(fwd, ref t4);
                    ch[CH(0, k, 0)] = t2 + t3; ch[CH(0, k, 2)] = t2 - t3;
                    ch[CH(0, k, 1)] = t1 + t4; ch[CH(0, k, 3)] = t1 - t4;
                }
            else
                for (long k = 0; k < l1; ++k)
                {
                    {
                        Cmplx t1, t2, t3, t4;
                        t2 = cc[CC(0, 0, k)] + cc[CC(0, 2, k)]; t1 = cc[CC(0, 0, k)] - cc[CC(0, 2, k)];
                        t3 = cc[CC(0, 1, k)] + cc[CC(0, 3, k)]; t4 = cc[CC(0, 1, k)] - cc[CC(0, 3, k)];
                        ROTX90(fwd, ref t4);
                        ch[CH(0, k, 0)] = t2 + t3; ch[CH(0, k, 2)] = t2 - t3;
                        ch[CH(0, k, 1)] = t1 + t4; ch[CH(0, k, 3)] = t1 - t4;
                    }
                    for (long i = 1; i < ido; ++i)
                    {
                        Cmplx t1, t2, t3, t4;
                        Cmplx cc0 = cc[CC(i, 0, k)], cc1 = cc[CC(i, 1, k)], cc2 = cc[CC(i, 2, k)], cc3 = cc[CC(i, 3, k)];
                        t2 = cc0 + cc2; t1 = cc0 - cc2;
                        t3 = cc1 + cc3; t4 = cc1 - cc3;
                        ROTX90(fwd, ref t4);
                        ch[CH(i, k, 0)] = t2 + t3;
                        ch[CH(i, k, 1)] = Cmplx.SpecialMul(fwd, t1 + t4, wa[WA(0, i)]);
                        ch[CH(i, k, 2)] = Cmplx.SpecialMul(fwd, t2 - t3, wa[WA(1, i)]);
                        ch[CH(i, k, 3)] = Cmplx.SpecialMul(fwd, t1 - t4, wa[WA(2, i)]);
                    }
                }
        }

        private static void Pass5(long ido, long l1, Cmplx* cc, Cmplx* ch, Cmplx* wa, bool fwd)
        {
            const double tw1r = 0.3090169943749474241022934171828191;
            double tw1i = (fwd ? -1.0 : 1.0) * 0.9510565162951535721164393333793821;
            const double tw2r = -0.8090169943749474241022934171828191;
            double tw2i = (fwd ? -1.0 : 1.0) * 0.5877852522924731291687059546390728;

            long CH(long a, long b, long c) => a + ido * (b + l1 * c);
            long CC(long a, long b, long c) => a + ido * (b + 5 * c);
            long WA(long x, long i) => i - 1 + x * (ido - 1);

            for (long k = 0; k < l1; ++k)
            {
                {
                    // PREP5(0)
                    Cmplx t0 = cc[CC(0, 0, k)], t1, t2, t3, t4;
                    t1 = cc[CC(0, 1, k)] + cc[CC(0, 4, k)]; t4 = cc[CC(0, 1, k)] - cc[CC(0, 4, k)];
                    t2 = cc[CC(0, 2, k)] + cc[CC(0, 3, k)]; t3 = cc[CC(0, 2, k)] - cc[CC(0, 3, k)];
                    ch[CH(0, k, 0)] = new Cmplx(t0.r + t1.r + t2.r, t0.i + t1.i + t2.i);
                    var (ca1, cb1) = Part5(t0, t1, t2, t3, t4, tw1r, tw2r, tw1i, tw2i);
                    ch[CH(0, k, 1)] = ca1 + cb1; ch[CH(0, k, 4)] = ca1 - cb1;
                    var (ca2, cb2) = Part5(t0, t1, t2, t3, t4, tw2r, tw1r, tw2i, -tw1i);
                    ch[CH(0, k, 2)] = ca2 + cb2; ch[CH(0, k, 3)] = ca2 - cb2;
                }
                for (long i = 1; i < ido; ++i)
                {
                    Cmplx t0 = cc[CC(i, 0, k)], t1, t2, t3, t4;
                    t1 = cc[CC(i, 1, k)] + cc[CC(i, 4, k)]; t4 = cc[CC(i, 1, k)] - cc[CC(i, 4, k)];
                    t2 = cc[CC(i, 2, k)] + cc[CC(i, 3, k)]; t3 = cc[CC(i, 2, k)] - cc[CC(i, 3, k)];
                    ch[CH(i, k, 0)] = new Cmplx(t0.r + t1.r + t2.r, t0.i + t1.i + t2.i);
                    var (ca1, cb1) = Part5(t0, t1, t2, t3, t4, tw1r, tw2r, tw1i, tw2i);
                    ch[CH(i, k, 1)] = Cmplx.SpecialMul(fwd, ca1 + cb1, wa[WA(0, i)]);
                    ch[CH(i, k, 4)] = Cmplx.SpecialMul(fwd, ca1 - cb1, wa[WA(3, i)]);
                    var (ca2, cb2) = Part5(t0, t1, t2, t3, t4, tw2r, tw1r, tw2i, -tw1i);
                    ch[CH(i, k, 2)] = Cmplx.SpecialMul(fwd, ca2 + cb2, wa[WA(1, i)]);
                    ch[CH(i, k, 3)] = Cmplx.SpecialMul(fwd, ca2 - cb2, wa[WA(2, i)]);
                }
            }
        }

        private static void Pass7(long ido, long l1, Cmplx* cc, Cmplx* ch, Cmplx* wa, bool fwd)
        {
            const double tw1r = 0.6234898018587335305250048840042398;
            double tw1i = (fwd ? -1.0 : 1.0) * 0.7818314824680298087084445266740578;
            const double tw2r = -0.2225209339563144042889025644967948;
            double tw2i = (fwd ? -1.0 : 1.0) * 0.9749279121818236070181316829939312;
            const double tw3r = -0.9009688679024191262361023195074451;
            double tw3i = (fwd ? -1.0 : 1.0) * 0.433883739117558120475768332848359;

            long CH(long a, long b, long c) => a + ido * (b + l1 * c);
            long CC(long a, long b, long c) => a + ido * (b + 7 * c);
            long WA(long x, long i) => i - 1 + x * (ido - 1);

            for (long k = 0; k < l1; ++k)
            {
                {
                    // PREP7(0)
                    Cmplx t1 = cc[CC(0, 0, k)], t2, t3, t4, t5, t6, t7;
                    t2 = cc[CC(0, 1, k)] + cc[CC(0, 6, k)]; t7 = cc[CC(0, 1, k)] - cc[CC(0, 6, k)];
                    t3 = cc[CC(0, 2, k)] + cc[CC(0, 5, k)]; t6 = cc[CC(0, 2, k)] - cc[CC(0, 5, k)];
                    t4 = cc[CC(0, 3, k)] + cc[CC(0, 4, k)]; t5 = cc[CC(0, 3, k)] - cc[CC(0, 4, k)];
                    ch[CH(0, k, 0)] = new Cmplx(t1.r + t2.r + t3.r + t4.r, t1.i + t2.i + t3.i + t4.i);
                    var (ca1, cb1) = Part7(t1, t2, t3, t4, t5, t6, t7, tw1r, tw2r, tw3r, tw1i, tw2i, tw3i);
                    ch[CH(0, k, 1)] = ca1 + cb1; ch[CH(0, k, 6)] = ca1 - cb1;
                    var (ca2, cb2) = Part7(t1, t2, t3, t4, t5, t6, t7, tw2r, tw3r, tw1r, tw2i, -tw3i, -tw1i);
                    ch[CH(0, k, 2)] = ca2 + cb2; ch[CH(0, k, 5)] = ca2 - cb2;
                    var (ca3, cb3) = Part7(t1, t2, t3, t4, t5, t6, t7, tw3r, tw1r, tw2r, tw3i, -tw1i, tw2i);
                    ch[CH(0, k, 3)] = ca3 + cb3; ch[CH(0, k, 4)] = ca3 - cb3;
                }
                for (long i = 1; i < ido; ++i)
                {
                    Cmplx t1 = cc[CC(i, 0, k)], t2, t3, t4, t5, t6, t7;
                    t2 = cc[CC(i, 1, k)] + cc[CC(i, 6, k)]; t7 = cc[CC(i, 1, k)] - cc[CC(i, 6, k)];
                    t3 = cc[CC(i, 2, k)] + cc[CC(i, 5, k)]; t6 = cc[CC(i, 2, k)] - cc[CC(i, 5, k)];
                    t4 = cc[CC(i, 3, k)] + cc[CC(i, 4, k)]; t5 = cc[CC(i, 3, k)] - cc[CC(i, 4, k)];
                    ch[CH(i, k, 0)] = new Cmplx(t1.r + t2.r + t3.r + t4.r, t1.i + t2.i + t3.i + t4.i);
                    var (da1, db1) = Part7(t1, t2, t3, t4, t5, t6, t7, tw1r, tw2r, tw3r, tw1i, tw2i, tw3i);
                    ch[CH(i, k, 1)] = Cmplx.SpecialMul(fwd, da1, wa[WA(0, i)]);
                    ch[CH(i, k, 6)] = Cmplx.SpecialMul(fwd, db1, wa[WA(5, i)]);
                    var (da2, db2) = Part7(t1, t2, t3, t4, t5, t6, t7, tw2r, tw3r, tw1r, tw2i, -tw3i, -tw1i);
                    ch[CH(i, k, 2)] = Cmplx.SpecialMul(fwd, da2, wa[WA(1, i)]);
                    ch[CH(i, k, 5)] = Cmplx.SpecialMul(fwd, db2, wa[WA(4, i)]);
                    var (da3, db3) = Part7(t1, t2, t3, t4, t5, t6, t7, tw3r, tw1r, tw2r, tw3i, -tw1i, tw2i);
                    ch[CH(i, k, 3)] = Cmplx.SpecialMul(fwd, da3, wa[WA(2, i)]);
                    ch[CH(i, k, 4)] = Cmplx.SpecialMul(fwd, db3, wa[WA(3, i)]);
                }
            }
        }

        private static void Pass8(long ido, long l1, Cmplx* cc, Cmplx* ch, Cmplx* wa, bool fwd)
        {
            long CH(long a, long b, long c) => a + ido * (b + l1 * c);
            long CC(long a, long b, long c) => a + ido * (b + 8 * c);
            long WA(long x, long i) => i - 1 + x * (ido - 1);

            if (ido == 1)
                for (long k = 0; k < l1; ++k)
                {
                    Cmplx a0, a1, a2, a3, a4, a5, a6, a7;
                    a1 = cc[CC(0, 1, k)] + cc[CC(0, 5, k)]; a5 = cc[CC(0, 1, k)] - cc[CC(0, 5, k)];
                    a3 = cc[CC(0, 3, k)] + cc[CC(0, 7, k)]; a7 = cc[CC(0, 3, k)] - cc[CC(0, 7, k)];
                    PMINPLACE(ref a1, ref a3);
                    ROTX90(fwd, ref a3);
                    ROTX90(fwd, ref a7);
                    PMINPLACE(ref a5, ref a7);
                    ROTX45(fwd, ref a5);
                    ROTX135(fwd, ref a7);
                    a0 = cc[CC(0, 0, k)] + cc[CC(0, 4, k)]; a4 = cc[CC(0, 0, k)] - cc[CC(0, 4, k)];
                    a2 = cc[CC(0, 2, k)] + cc[CC(0, 6, k)]; a6 = cc[CC(0, 2, k)] - cc[CC(0, 6, k)];
                    { Cmplx s = a0 + a2; ch[CH(0, k, 0)] = s + a1; ch[CH(0, k, 4)] = s - a1; }
                    { Cmplx s = a0 - a2; ch[CH(0, k, 2)] = s + a3; ch[CH(0, k, 6)] = s - a3; }
                    ROTX90(fwd, ref a6);
                    { Cmplx s = a4 + a6; ch[CH(0, k, 1)] = s + a5; ch[CH(0, k, 5)] = s - a5; }
                    { Cmplx s = a4 - a6; ch[CH(0, k, 3)] = s + a7; ch[CH(0, k, 7)] = s - a7; }
                }
            else
                for (long k = 0; k < l1; ++k)
                {
                    {
                        Cmplx a0, a1, a2, a3, a4, a5, a6, a7;
                        a1 = cc[CC(0, 1, k)] + cc[CC(0, 5, k)]; a5 = cc[CC(0, 1, k)] - cc[CC(0, 5, k)];
                        a3 = cc[CC(0, 3, k)] + cc[CC(0, 7, k)]; a7 = cc[CC(0, 3, k)] - cc[CC(0, 7, k)];
                        PMINPLACE(ref a1, ref a3);
                        ROTX90(fwd, ref a3);
                        ROTX90(fwd, ref a7);
                        PMINPLACE(ref a5, ref a7);
                        ROTX45(fwd, ref a5);
                        ROTX135(fwd, ref a7);
                        a0 = cc[CC(0, 0, k)] + cc[CC(0, 4, k)]; a4 = cc[CC(0, 0, k)] - cc[CC(0, 4, k)];
                        a2 = cc[CC(0, 2, k)] + cc[CC(0, 6, k)]; a6 = cc[CC(0, 2, k)] - cc[CC(0, 6, k)];
                        { Cmplx s = a0 + a2; ch[CH(0, k, 0)] = s + a1; ch[CH(0, k, 4)] = s - a1; }
                        { Cmplx s = a0 - a2; ch[CH(0, k, 2)] = s + a3; ch[CH(0, k, 6)] = s - a3; }
                        ROTX90(fwd, ref a6);
                        { Cmplx s = a4 + a6; ch[CH(0, k, 1)] = s + a5; ch[CH(0, k, 5)] = s - a5; }
                        { Cmplx s = a4 - a6; ch[CH(0, k, 3)] = s + a7; ch[CH(0, k, 7)] = s - a7; }
                    }
                    for (long i = 1; i < ido; ++i)
                    {
                        Cmplx a0, a1, a2, a3, a4, a5, a6, a7;
                        a1 = cc[CC(i, 1, k)] + cc[CC(i, 5, k)]; a5 = cc[CC(i, 1, k)] - cc[CC(i, 5, k)];
                        a3 = cc[CC(i, 3, k)] + cc[CC(i, 7, k)]; a7 = cc[CC(i, 3, k)] - cc[CC(i, 7, k)];
                        ROTX90(fwd, ref a7);
                        PMINPLACE(ref a1, ref a3);
                        ROTX90(fwd, ref a3);
                        PMINPLACE(ref a5, ref a7);
                        ROTX45(fwd, ref a5);
                        ROTX135(fwd, ref a7);
                        a0 = cc[CC(i, 0, k)] + cc[CC(i, 4, k)]; a4 = cc[CC(i, 0, k)] - cc[CC(i, 4, k)];
                        a2 = cc[CC(i, 2, k)] + cc[CC(i, 6, k)]; a6 = cc[CC(i, 2, k)] - cc[CC(i, 6, k)];
                        PMINPLACE(ref a0, ref a2);
                        ch[CH(i, k, 0)] = a0 + a1;
                        ch[CH(i, k, 4)] = Cmplx.SpecialMul(fwd, a0 - a1, wa[WA(3, i)]);
                        ch[CH(i, k, 2)] = Cmplx.SpecialMul(fwd, a2 + a3, wa[WA(1, i)]);
                        ch[CH(i, k, 6)] = Cmplx.SpecialMul(fwd, a2 - a3, wa[WA(5, i)]);
                        ROTX90(fwd, ref a6);
                        PMINPLACE(ref a4, ref a6);
                        ch[CH(i, k, 1)] = Cmplx.SpecialMul(fwd, a4 + a5, wa[WA(0, i)]);
                        ch[CH(i, k, 5)] = Cmplx.SpecialMul(fwd, a4 - a5, wa[WA(4, i)]);
                        ch[CH(i, k, 3)] = Cmplx.SpecialMul(fwd, a6 + a7, wa[WA(2, i)]);
                        ch[CH(i, k, 7)] = Cmplx.SpecialMul(fwd, a6 - a7, wa[WA(6, i)]);
                    }
                }
        }

        private static void Pass11(long ido, long l1, Cmplx* cc, Cmplx* ch, Cmplx* wa, bool fwd)
        {
            const double tw1r = 0.8412535328311811688618116489193677;
            double tw1i = (fwd ? -1.0 : 1.0) * 0.5406408174555975821076359543186917;
            const double tw2r = 0.4154150130018864255292741492296232;
            double tw2i = (fwd ? -1.0 : 1.0) * 0.9096319953545183714117153830790285;
            const double tw3r = -0.1423148382732851404437926686163697;
            double tw3i = (fwd ? -1.0 : 1.0) * 0.9898214418809327323760920377767188;
            const double tw4r = -0.6548607339452850640569250724662936;
            double tw4i = (fwd ? -1.0 : 1.0) * 0.7557495743542582837740358439723444;
            const double tw5r = -0.9594929736144973898903680570663277;
            double tw5i = (fwd ? -1.0 : 1.0) * 0.2817325568414296977114179153466169;

            long CH(long a, long b, long c) => a + ido * (b + l1 * c);
            long CC(long a, long b, long c) => a + ido * (b + 11 * c);
            long WA(long x, long i) => i - 1 + x * (ido - 1);

            for (long k = 0; k < l1; ++k)
            {
                {
                    // PREP11(0)
                    Cmplx t1 = cc[CC(0, 0, k)], t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;
                    t2 = cc[CC(0, 1, k)] + cc[CC(0, 10, k)]; t11 = cc[CC(0, 1, k)] - cc[CC(0, 10, k)];
                    t3 = cc[CC(0, 2, k)] + cc[CC(0, 9, k)]; t10 = cc[CC(0, 2, k)] - cc[CC(0, 9, k)];
                    t4 = cc[CC(0, 3, k)] + cc[CC(0, 8, k)]; t9 = cc[CC(0, 3, k)] - cc[CC(0, 8, k)];
                    t5 = cc[CC(0, 4, k)] + cc[CC(0, 7, k)]; t8 = cc[CC(0, 4, k)] - cc[CC(0, 7, k)];
                    t6 = cc[CC(0, 5, k)] + cc[CC(0, 6, k)]; t7 = cc[CC(0, 5, k)] - cc[CC(0, 6, k)];
                    ch[CH(0, k, 0)] = new Cmplx(t1.r + t2.r + t3.r + t4.r + t5.r + t6.r, t1.i + t2.i + t3.i + t4.i + t5.i + t6.i);
                    var (ca1, cb1) = Part11(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, tw1r, tw2r, tw3r, tw4r, tw5r, tw1i, tw2i, tw3i, tw4i, tw5i);
                    ch[CH(0, k, 1)] = ca1 + cb1; ch[CH(0, k, 10)] = ca1 - cb1;
                    var (ca2, cb2) = Part11(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, tw2r, tw4r, tw5r, tw3r, tw1r, tw2i, tw4i, -tw5i, -tw3i, -tw1i);
                    ch[CH(0, k, 2)] = ca2 + cb2; ch[CH(0, k, 9)] = ca2 - cb2;
                    var (ca3, cb3) = Part11(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, tw3r, tw5r, tw2r, tw1r, tw4r, tw3i, -tw5i, -tw2i, tw1i, tw4i);
                    ch[CH(0, k, 3)] = ca3 + cb3; ch[CH(0, k, 8)] = ca3 - cb3;
                    var (ca4, cb4) = Part11(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, tw4r, tw3r, tw1r, tw5r, tw2r, tw4i, -tw3i, tw1i, tw5i, -tw2i);
                    ch[CH(0, k, 4)] = ca4 + cb4; ch[CH(0, k, 7)] = ca4 - cb4;
                    var (ca5, cb5) = Part11(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, tw5r, tw1r, tw4r, tw2r, tw3r, tw5i, -tw1i, tw4i, -tw2i, tw3i);
                    ch[CH(0, k, 5)] = ca5 + cb5; ch[CH(0, k, 6)] = ca5 - cb5;
                }
                for (long i = 1; i < ido; ++i)
                {
                    Cmplx t1 = cc[CC(i, 0, k)], t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;
                    t2 = cc[CC(i, 1, k)] + cc[CC(i, 10, k)]; t11 = cc[CC(i, 1, k)] - cc[CC(i, 10, k)];
                    t3 = cc[CC(i, 2, k)] + cc[CC(i, 9, k)]; t10 = cc[CC(i, 2, k)] - cc[CC(i, 9, k)];
                    t4 = cc[CC(i, 3, k)] + cc[CC(i, 8, k)]; t9 = cc[CC(i, 3, k)] - cc[CC(i, 8, k)];
                    t5 = cc[CC(i, 4, k)] + cc[CC(i, 7, k)]; t8 = cc[CC(i, 4, k)] - cc[CC(i, 7, k)];
                    t6 = cc[CC(i, 5, k)] + cc[CC(i, 6, k)]; t7 = cc[CC(i, 5, k)] - cc[CC(i, 6, k)];
                    ch[CH(i, k, 0)] = new Cmplx(t1.r + t2.r + t3.r + t4.r + t5.r + t6.r, t1.i + t2.i + t3.i + t4.i + t5.i + t6.i);
                    var (da1, db1) = Part11(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, tw1r, tw2r, tw3r, tw4r, tw5r, tw1i, tw2i, tw3i, tw4i, tw5i);
                    ch[CH(i, k, 1)] = Cmplx.SpecialMul(fwd, da1, wa[WA(0, i)]);
                    ch[CH(i, k, 10)] = Cmplx.SpecialMul(fwd, db1, wa[WA(9, i)]);
                    var (da2, db2) = Part11(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, tw2r, tw4r, tw5r, tw3r, tw1r, tw2i, tw4i, -tw5i, -tw3i, -tw1i);
                    ch[CH(i, k, 2)] = Cmplx.SpecialMul(fwd, da2, wa[WA(1, i)]);
                    ch[CH(i, k, 9)] = Cmplx.SpecialMul(fwd, db2, wa[WA(8, i)]);
                    var (da3, db3) = Part11(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, tw3r, tw5r, tw2r, tw1r, tw4r, tw3i, -tw5i, -tw2i, tw1i, tw4i);
                    ch[CH(i, k, 3)] = Cmplx.SpecialMul(fwd, da3, wa[WA(2, i)]);
                    ch[CH(i, k, 8)] = Cmplx.SpecialMul(fwd, db3, wa[WA(7, i)]);
                    var (da4, db4) = Part11(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, tw4r, tw3r, tw1r, tw5r, tw2r, tw4i, -tw3i, tw1i, tw5i, -tw2i);
                    ch[CH(i, k, 4)] = Cmplx.SpecialMul(fwd, da4, wa[WA(3, i)]);
                    ch[CH(i, k, 7)] = Cmplx.SpecialMul(fwd, db4, wa[WA(6, i)]);
                    var (da5, db5) = Part11(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, tw5r, tw1r, tw4r, tw2r, tw3r, tw5i, -tw1i, tw4i, -tw2i, tw3i);
                    ch[CH(i, k, 5)] = Cmplx.SpecialMul(fwd, da5, wa[WA(4, i)]);
                    ch[CH(i, k, 6)] = Cmplx.SpecialMul(fwd, db5, wa[WA(5, i)]);
                }
            }
        }

        private static void Passg(long ido, long ip, long l1, Cmplx* cc, Cmplx* ch, Cmplx* wa, Cmplx* csarr, bool fwd)
        {
            long cdim = ip;
            long ipph = (ip + 1) / 2;
            long idl1 = ido * l1;

            long CH(long a, long b, long c) => a + ido * (b + l1 * c);
            long CC(long a, long b, long c) => a + ido * (b + cdim * c);
            long CX(long a, long b, long c) => a + ido * (b + l1 * c);
            long CX2(long a, long b) => a + idl1 * b;
            long CH2(long a, long b) => a + idl1 * b;

            var wal = new Cmplx[ip];
            wal[0] = new Cmplx(1.0, 0.0);
            for (long i = 1; i < ip; ++i)
                wal[i] = new Cmplx(csarr[i].r, fwd ? -csarr[i].i : csarr[i].i);

            for (long k = 0; k < l1; ++k)
                for (long i = 0; i < ido; ++i)
                    ch[CH(i, k, 0)] = cc[CC(i, 0, k)];
            for (long j = 1, jc = ip - 1; j < ipph; ++j, --jc)
                for (long k = 0; k < l1; ++k)
                    for (long i = 0; i < ido; ++i)
                    {
                        Cmplx x = cc[CC(i, j, k)], y = cc[CC(i, jc, k)];
                        ch[CH(i, k, j)] = x + y;
                        ch[CH(i, k, jc)] = x - y;
                    }
            for (long k = 0; k < l1; ++k)
                for (long i = 0; i < ido; ++i)
                {
                    Cmplx tmp = ch[CH(i, k, 0)];
                    for (long j = 1; j < ipph; ++j)
                        tmp = tmp + ch[CH(i, k, j)];
                    cc[CX(i, k, 0)] = tmp;
                }
            for (long l = 1, lc = ip - 1; l < ipph; ++l, --lc)
            {
                // j=0
                for (long ik = 0; ik < idl1; ++ik)
                {
                    long il = CX2(ik, l), ilc = CX2(ik, lc);
                    Cmplx h0 = ch[CH2(ik, 0)], h1 = ch[CH2(ik, 1)], h2 = ch[CH2(ik, 2)];
                    Cmplx hm1 = ch[CH2(ik, ip - 1)], hm2 = ch[CH2(ik, ip - 2)];
                    cc[il].r = h0.r + wal[l].r * h1.r + wal[2 * l].r * h2.r;
                    cc[il].i = h0.i + wal[l].r * h1.i + wal[2 * l].r * h2.i;
                    cc[ilc].r = -wal[l].i * hm1.i - wal[2 * l].i * hm2.i;
                    cc[ilc].i = wal[l].i * hm1.r + wal[2 * l].i * hm2.r;
                }

                long iwal = 2 * l;
                long jj = 3, jjc = ip - 3;
                for (; jj < ipph - 1; jj += 2, jjc -= 2)
                {
                    iwal += l; if (iwal > ip) iwal -= ip;
                    Cmplx xwal = wal[iwal];
                    iwal += l; if (iwal > ip) iwal -= ip;
                    Cmplx xwal2 = wal[iwal];
                    for (long ik = 0; ik < idl1; ++ik)
                    {
                        long il = CX2(ik, l), ilc = CX2(ik, lc);
                        Cmplx hj = ch[CH2(ik, jj)], hj1 = ch[CH2(ik, jj + 1)];
                        Cmplx hjc = ch[CH2(ik, jjc)], hjc1 = ch[CH2(ik, jjc - 1)];
                        cc[il].r += hj.r * xwal.r + hj1.r * xwal2.r;
                        cc[il].i += hj.i * xwal.r + hj1.i * xwal2.r;
                        cc[ilc].r -= hjc.i * xwal.i + hjc1.i * xwal2.i;
                        cc[ilc].i += hjc.r * xwal.i + hjc1.r * xwal2.i;
                    }
                }
                for (; jj < ipph; ++jj, --jjc)
                {
                    iwal += l; if (iwal > ip) iwal -= ip;
                    Cmplx xwal = wal[iwal];
                    for (long ik = 0; ik < idl1; ++ik)
                    {
                        long il = CX2(ik, l), ilc = CX2(ik, lc);
                        Cmplx hj = ch[CH2(ik, jj)], hjc = ch[CH2(ik, jjc)];
                        cc[il].r += hj.r * xwal.r;
                        cc[il].i += hj.i * xwal.r;
                        cc[ilc].r -= hjc.i * xwal.i;
                        cc[ilc].i += hjc.r * xwal.i;
                    }
                }
            }

            // shuffling and twiddling
            if (ido == 1)
                for (long j = 1, jc = ip - 1; j < ipph; ++j, --jc)
                    for (long ik = 0; ik < idl1; ++ik)
                    {
                        Cmplx t1 = cc[CX2(ik, j)], t2 = cc[CX2(ik, jc)];
                        cc[CX2(ik, j)] = t1 + t2;
                        cc[CX2(ik, jc)] = t1 - t2;
                    }
            else
            {
                for (long j = 1, jc = ip - 1; j < ipph; ++j, --jc)
                    for (long k = 0; k < l1; ++k)
                    {
                        Cmplx t1 = cc[CX(0, k, j)], t2 = cc[CX(0, k, jc)];
                        cc[CX(0, k, j)] = t1 + t2;
                        cc[CX(0, k, jc)] = t1 - t2;
                        for (long i = 1; i < ido; ++i)
                        {
                            Cmplx x1 = cc[CX(i, k, j)] + cc[CX(i, k, jc)];
                            Cmplx x2 = cc[CX(i, k, j)] - cc[CX(i, k, jc)];
                            long idij = (j - 1) * (ido - 1) + i - 1;
                            cc[CX(i, k, j)] = Cmplx.SpecialMul(fwd, x1, wa[idij]);
                            idij = (jc - 1) * (ido - 1) + i - 1;
                            cc[CX(i, k, jc)] = Cmplx.SpecialMul(fwd, x2, wa[idij]);
                        }
                    }
            }
        }

        // ================================ pass_all ================================

        public void Exec(Cmplx* c, double fct, bool fwd)
        {
            if (length == 1) { c[0] = c[0] * fct; return; }
            long l1 = 1;
            var chArr = ArrayPool<Cmplx>.Shared.Rent((int)length);
            try
            {
                fixed (Cmplx* chp = chArr)
                {
                    Cmplx* p1 = c, p2 = chp;
                    for (int k1 = 0; k1 < fact.Count; ++k1)
                    {
                        long ip = fact[k1].fct;
                        long l2 = ip * l1;
                        long ido = length / l2;
                        fixed (Cmplx* tw = fact[k1].tw, tws = fact[k1].tws)
                        {
                            if (ip == 4) Pass4(ido, l1, p1, p2, tw, fwd);
                            else if (ip == 8) Pass8(ido, l1, p1, p2, tw, fwd);
                            else if (ip == 2) Pass2(ido, l1, p1, p2, tw, fwd);
                            else if (ip == 3) Pass3(ido, l1, p1, p2, tw, fwd);
                            else if (ip == 5) Pass5(ido, l1, p1, p2, tw, fwd);
                            else if (ip == 7) Pass7(ido, l1, p1, p2, tw, fwd);
                            else if (ip == 11) Pass11(ido, l1, p1, p2, tw, fwd);
                            else
                            {
                                Passg(ido, ip, l1, p1, p2, tw, tws, fwd);
                                { Cmplx* t = p1; p1 = p2; p2 = t; }
                            }
                        }
                        { Cmplx* t = p1; p1 = p2; p2 = t; }
                        l1 = l2;
                    }
                    if (p1 != c)
                    {
                        if (fct != 1.0)
                            for (long i = 0; i < length; ++i) c[i] = p1[i] * fct;
                        else
                            for (long i = 0; i < length; ++i) c[i] = p1[i];
                    }
                    else if (fct != 1.0)
                        for (long i = 0; i < length; ++i) c[i] = c[i] * fct;
                }
            }
            finally
            {
                ArrayPool<Cmplx>.Shared.Return(chArr);
            }
        }
    }
}
