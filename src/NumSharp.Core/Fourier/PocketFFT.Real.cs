using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

// =============================================================================
// Port of pocketfft's rfftp<T0> — the real mixed-radix FFTPACK transform
// (pocketfft_hdronly.h lines 1604-2407). Real codelets radf2/3/4/5 + radfg
// (r2hc, forward) and radb2/3/4/5 + radbg (hc2r, backward), factorize(),
// comp_twiddle() and exec (with FFTPACK half-complex order R0,R1,I1,...).
// Double engine (T0 == double). Operation ORDER kept verbatim for bit-parity.
//
// The half-complex <-> complex packing is applied by PocketFFTDriver, exactly
// as numpy's rfft_impl/irfft_loop do (see _pocketfft_umath.cpp).
// =============================================================================

namespace NumSharp.Fourier
{
    public sealed unsafe class Rfftp
    {
        private sealed class FctData
        {
            public long fct;
            public double[] tw;   // (ip-1)*(ido-1) interleaved r,i twiddles; null for the last factor
            public double[] tws;  // 2*ip twiddles for radfg/radbg (ip>5); else null
        }

        private readonly long length;
        private readonly List<FctData> fact = new List<FctData>();

        public long Length => length;

        public Rfftp(long length_)
        {
            length = length_;
            if (length == 0) throw new ArgumentException("zero-length FFT requested");
            if (length == 1) return;
            Factorize();
            CompTwiddle();
        }

        private void AddFactor(long factor) => fact.Add(new FctData { fct = factor });

        private void Factorize()
        {
            long len = length;
            while ((len % 4) == 0) { AddFactor(4); len >>= 2; }
            if ((len % 2) == 0)
            {
                len >>= 1;
                AddFactor(2);
                long tmp = fact[0].fct; fact[0].fct = fact[fact.Count - 1].fct; fact[fact.Count - 1].fct = tmp;
            }
            for (long divisor = 3; divisor * divisor <= len; divisor += 2)
                while ((len % divisor) == 0) { AddFactor(divisor); len /= divisor; }
            if (len > 1) AddFactor(len);
        }

        private void CompTwiddle()
        {
            var twid = new SinCos2PiByN(length);
            long l1 = 1;
            for (int k = 0; k < fact.Count; ++k)
            {
                long ip = fact[k].fct, ido = length / (l1 * ip);
                if (k < fact.Count - 1) // last factor doesn't need twiddles
                {
                    var tw = new double[(ip - 1) * (ido - 1)];
                    for (long j = 1; j < ip; ++j)
                        for (long i = 1; i <= (ido - 1) / 2; ++i)
                        {
                            Cmplx t = twid[j * l1 * i];
                            tw[(j - 1) * (ido - 1) + 2 * i - 2] = t.r;
                            tw[(j - 1) * (ido - 1) + 2 * i - 1] = t.i;
                        }
                    fact[k].tw = tw;
                }
                if (ip > 5) // special factors required by *g functions
                {
                    var tws = new double[2 * ip];
                    tws[0] = 1.0;
                    tws[1] = 0.0;
                    for (long i = 2, ic = 2 * ip - 2; i <= ic; i += 2, ic -= 2)
                    {
                        Cmplx t = twid[i / 2 * (length / ip)];
                        tws[i] = t.r;
                        tws[i + 1] = t.i;
                        tws[ic] = t.r;
                        tws[ic + 1] = -t.i;
                    }
                    fact[k].tws = tws;
                }
                l1 *= ip;
            }
        }

        // (a+ib) = conj(c+id) * (e+if):  a = c*e + d*f;  b = c*f - d*e
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void MULPM(out double a, out double b, double c, double d, double e, double f)
        { a = c * e + d * f; b = c * f - d * e; }

        // a2=a+b; b2=i*(b-a) — pocketfft REARRANGE macro
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Rearrange(ref double rx, ref double ix, ref double ry, ref double iy)
        {
            double t1 = rx + ry, t2 = ry - rx, t3 = ix + iy, t4 = ix - iy;
            rx = t1; ix = t3; ry = t4; iy = t2;
        }

        // ============================ forward (r2hc) codelets ============================

        private static void Radf2(long ido, long l1, double* cc, double* ch, double* wa)
        {
            long WA(long x, long i) => i + x * (ido - 1);
            long CC(long a, long b, long c) => a + ido * (b + l1 * c);
            long CH(long a, long b, long c) => a + ido * (b + 2 * c);

            for (long k = 0; k < l1; k++)
            {
                ch[CH(0, 0, k)] = cc[CC(0, k, 0)] + cc[CC(0, k, 1)];
                ch[CH(ido - 1, 1, k)] = cc[CC(0, k, 0)] - cc[CC(0, k, 1)];
            }
            if ((ido & 1) == 0)
                for (long k = 0; k < l1; k++)
                {
                    ch[CH(0, 1, k)] = -cc[CC(ido - 1, k, 1)];
                    ch[CH(ido - 1, 0, k)] = cc[CC(ido - 1, k, 0)];
                }
            if (ido <= 2) return;
            for (long k = 0; k < l1; k++)
                for (long i = 2; i < ido; i += 2)
                {
                    long ic = ido - i;
                    MULPM(out double tr2, out double ti2, wa[WA(0, i - 2)], wa[WA(0, i - 1)], cc[CC(i - 1, k, 1)], cc[CC(i, k, 1)]);
                    ch[CH(i - 1, 0, k)] = cc[CC(i - 1, k, 0)] + tr2;
                    ch[CH(ic - 1, 1, k)] = cc[CC(i - 1, k, 0)] - tr2;
                    ch[CH(i, 0, k)] = ti2 + cc[CC(i, k, 0)];
                    ch[CH(ic, 1, k)] = ti2 - cc[CC(i, k, 0)];
                }
        }

        private static void Radf3(long ido, long l1, double* cc, double* ch, double* wa)
        {
            const double taur = -0.5, taui = 0.8660254037844386467637231707529362;
            long WA(long x, long i) => i + x * (ido - 1);
            long CC(long a, long b, long c) => a + ido * (b + l1 * c);
            long CH(long a, long b, long c) => a + ido * (b + 3 * c);

            for (long k = 0; k < l1; k++)
            {
                double cr2 = cc[CC(0, k, 1)] + cc[CC(0, k, 2)];
                ch[CH(0, 0, k)] = cc[CC(0, k, 0)] + cr2;
                ch[CH(0, 2, k)] = taui * (cc[CC(0, k, 2)] - cc[CC(0, k, 1)]);
                ch[CH(ido - 1, 1, k)] = cc[CC(0, k, 0)] + taur * cr2;
            }
            if (ido == 1) return;
            for (long k = 0; k < l1; k++)
                for (long i = 2; i < ido; i += 2)
                {
                    long ic = ido - i;
                    MULPM(out double dr2, out double di2, wa[WA(0, i - 2)], wa[WA(0, i - 1)], cc[CC(i - 1, k, 1)], cc[CC(i, k, 1)]);
                    MULPM(out double dr3, out double di3, wa[WA(1, i - 2)], wa[WA(1, i - 1)], cc[CC(i - 1, k, 2)], cc[CC(i, k, 2)]);
                    Rearrange(ref dr2, ref di2, ref dr3, ref di3);
                    ch[CH(i - 1, 0, k)] = cc[CC(i - 1, k, 0)] + dr2;
                    ch[CH(i, 0, k)] = cc[CC(i, k, 0)] + di2;
                    double tr2 = cc[CC(i - 1, k, 0)] + taur * dr2;
                    double ti2 = cc[CC(i, k, 0)] + taur * di2;
                    double tr3 = taui * dr3;
                    double ti3 = taui * di3;
                    ch[CH(i - 1, 2, k)] = tr2 + tr3;
                    ch[CH(ic - 1, 1, k)] = tr2 - tr3;
                    ch[CH(i, 2, k)] = ti3 + ti2;
                    ch[CH(ic, 1, k)] = ti3 - ti2;
                }
        }

        private static void Radf4(long ido, long l1, double* cc, double* ch, double* wa)
        {
            const double hsqt2 = 0.707106781186547524400844362104849;
            long WA(long x, long i) => i + x * (ido - 1);
            long CC(long a, long b, long c) => a + ido * (b + l1 * c);
            long CH(long a, long b, long c) => a + ido * (b + 4 * c);

            for (long k = 0; k < l1; k++)
            {
                double tr1 = cc[CC(0, k, 3)] + cc[CC(0, k, 1)];
                ch[CH(0, 2, k)] = cc[CC(0, k, 3)] - cc[CC(0, k, 1)];
                double tr2 = cc[CC(0, k, 0)] + cc[CC(0, k, 2)];
                ch[CH(ido - 1, 1, k)] = cc[CC(0, k, 0)] - cc[CC(0, k, 2)];
                ch[CH(0, 0, k)] = tr2 + tr1;
                ch[CH(ido - 1, 3, k)] = tr2 - tr1;
            }
            if ((ido & 1) == 0)
                for (long k = 0; k < l1; k++)
                {
                    double ti1 = -hsqt2 * (cc[CC(ido - 1, k, 1)] + cc[CC(ido - 1, k, 3)]);
                    double tr1 = hsqt2 * (cc[CC(ido - 1, k, 1)] - cc[CC(ido - 1, k, 3)]);
                    ch[CH(ido - 1, 0, k)] = cc[CC(ido - 1, k, 0)] + tr1;
                    ch[CH(ido - 1, 2, k)] = cc[CC(ido - 1, k, 0)] - tr1;
                    ch[CH(0, 3, k)] = ti1 + cc[CC(ido - 1, k, 2)];
                    ch[CH(0, 1, k)] = ti1 - cc[CC(ido - 1, k, 2)];
                }
            if (ido <= 2) return;
            for (long k = 0; k < l1; k++)
                for (long i = 2; i < ido; i += 2)
                {
                    long ic = ido - i;
                    MULPM(out double cr2, out double ci2, wa[WA(0, i - 2)], wa[WA(0, i - 1)], cc[CC(i - 1, k, 1)], cc[CC(i, k, 1)]);
                    MULPM(out double cr3, out double ci3, wa[WA(1, i - 2)], wa[WA(1, i - 1)], cc[CC(i - 1, k, 2)], cc[CC(i, k, 2)]);
                    MULPM(out double cr4, out double ci4, wa[WA(2, i - 2)], wa[WA(2, i - 1)], cc[CC(i - 1, k, 3)], cc[CC(i, k, 3)]);
                    double tr1 = cr4 + cr2, tr4 = cr4 - cr2;
                    double ti1 = ci2 + ci4, ti4 = ci2 - ci4;
                    double tr2 = cc[CC(i - 1, k, 0)] + cr3, tr3 = cc[CC(i - 1, k, 0)] - cr3;
                    double ti2 = cc[CC(i, k, 0)] + ci3, ti3 = cc[CC(i, k, 0)] - ci3;
                    ch[CH(i - 1, 0, k)] = tr2 + tr1;
                    ch[CH(ic - 1, 3, k)] = tr2 - tr1;
                    ch[CH(i, 0, k)] = ti1 + ti2;
                    ch[CH(ic, 3, k)] = ti1 - ti2;
                    ch[CH(i - 1, 2, k)] = tr3 + ti4;
                    ch[CH(ic - 1, 1, k)] = tr3 - ti4;
                    ch[CH(i, 2, k)] = tr4 + ti3;
                    ch[CH(ic, 1, k)] = tr4 - ti3;
                }
        }

        private static void Radf5(long ido, long l1, double* cc, double* ch, double* wa)
        {
            const double tr11 = 0.3090169943749474241022934171828191, ti11 = 0.9510565162951535721164393333793821,
                         tr12 = -0.8090169943749474241022934171828191, ti12 = 0.5877852522924731291687059546390728;
            long WA(long x, long i) => i + x * (ido - 1);
            long CC(long a, long b, long c) => a + ido * (b + l1 * c);
            long CH(long a, long b, long c) => a + ido * (b + 5 * c);

            for (long k = 0; k < l1; k++)
            {
                double cr2 = cc[CC(0, k, 4)] + cc[CC(0, k, 1)], ci5 = cc[CC(0, k, 4)] - cc[CC(0, k, 1)];
                double cr3 = cc[CC(0, k, 3)] + cc[CC(0, k, 2)], ci4 = cc[CC(0, k, 3)] - cc[CC(0, k, 2)];
                ch[CH(0, 0, k)] = cc[CC(0, k, 0)] + cr2 + cr3;
                ch[CH(ido - 1, 1, k)] = cc[CC(0, k, 0)] + tr11 * cr2 + tr12 * cr3;
                ch[CH(0, 2, k)] = ti11 * ci5 + ti12 * ci4;
                ch[CH(ido - 1, 3, k)] = cc[CC(0, k, 0)] + tr12 * cr2 + tr11 * cr3;
                ch[CH(0, 4, k)] = ti12 * ci5 - ti11 * ci4;
            }
            if (ido == 1) return;
            for (long k = 0; k < l1; ++k)
                for (long i = 2, ic = ido - 2; i < ido; i += 2, ic -= 2)
                {
                    MULPM(out double dr2, out double di2, wa[WA(0, i - 2)], wa[WA(0, i - 1)], cc[CC(i - 1, k, 1)], cc[CC(i, k, 1)]);
                    MULPM(out double dr3, out double di3, wa[WA(1, i - 2)], wa[WA(1, i - 1)], cc[CC(i - 1, k, 2)], cc[CC(i, k, 2)]);
                    MULPM(out double dr4, out double di4, wa[WA(2, i - 2)], wa[WA(2, i - 1)], cc[CC(i - 1, k, 3)], cc[CC(i, k, 3)]);
                    MULPM(out double dr5, out double di5, wa[WA(3, i - 2)], wa[WA(3, i - 1)], cc[CC(i - 1, k, 4)], cc[CC(i, k, 4)]);
                    Rearrange(ref dr2, ref di2, ref dr5, ref di5);
                    Rearrange(ref dr3, ref di3, ref dr4, ref di4);
                    ch[CH(i - 1, 0, k)] = cc[CC(i - 1, k, 0)] + dr2 + dr3;
                    ch[CH(i, 0, k)] = cc[CC(i, k, 0)] + di2 + di3;
                    double tr2 = cc[CC(i - 1, k, 0)] + tr11 * dr2 + tr12 * dr3;
                    double ti2 = cc[CC(i, k, 0)] + tr11 * di2 + tr12 * di3;
                    double tr3 = cc[CC(i - 1, k, 0)] + tr12 * dr2 + tr11 * dr3;
                    double ti3 = cc[CC(i, k, 0)] + tr12 * di2 + tr11 * di3;
                    double tr5 = ti11 * dr5 + ti12 * dr4;
                    double ti5 = ti11 * di5 + ti12 * di4;
                    double tr4 = ti12 * dr5 - ti11 * dr4;
                    double ti4 = ti12 * di5 - ti11 * di4;
                    ch[CH(i - 1, 2, k)] = tr2 + tr5;
                    ch[CH(ic - 1, 1, k)] = tr2 - tr5;
                    ch[CH(i, 2, k)] = ti5 + ti2;
                    ch[CH(ic, 1, k)] = ti5 - ti2;
                    ch[CH(i - 1, 4, k)] = tr3 + tr4;
                    ch[CH(ic - 1, 3, k)] = tr3 - tr4;
                    ch[CH(i, 4, k)] = ti4 + ti3;
                    ch[CH(ic, 3, k)] = ti4 - ti3;
                }
        }

        private static void Radfg(long ido, long ip, long l1, double* cc, double* ch, double* wa, double* csarr)
        {
            long cdim = ip;
            long ipph = (ip + 1) / 2;
            long idl1 = ido * l1;

            long CC(long a, long b, long c) => a + ido * (b + cdim * c);
            long CH(long a, long b, long c) => a + ido * (b + l1 * c);
            long C1(long a, long b, long c) => a + ido * (b + l1 * c);
            long C2(long a, long b) => a + idl1 * b;
            long CH2(long a, long b) => a + idl1 * b;

            if (ido > 1)
            {
                for (long j = 1, jc = ip - 1; j < ipph; ++j, --jc)
                {
                    long is0 = (j - 1) * (ido - 1), is2 = (jc - 1) * (ido - 1);
                    for (long k = 0; k < l1; ++k)
                    {
                        long idij = is0, idij2 = is2;
                        for (long i = 1; i <= ido - 2; i += 2)
                        {
                            double t1 = cc[C1(i, k, j)], t2 = cc[C1(i + 1, k, j)],
                                   t3 = cc[C1(i, k, jc)], t4 = cc[C1(i + 1, k, jc)];
                            double x1 = wa[idij] * t1 + wa[idij + 1] * t2,
                                   x2 = wa[idij] * t2 - wa[idij + 1] * t1,
                                   x3 = wa[idij2] * t3 + wa[idij2 + 1] * t4,
                                   x4 = wa[idij2] * t4 - wa[idij2 + 1] * t3;
                            cc[C1(i, k, j)] = x3 + x1;
                            cc[C1(i + 1, k, jc)] = x3 - x1;
                            cc[C1(i + 1, k, j)] = x2 + x4;
                            cc[C1(i, k, jc)] = x2 - x4;
                            idij += 2; idij2 += 2;
                        }
                    }
                }
            }

            for (long j = 1, jc = ip - 1; j < ipph; ++j, --jc)
                for (long k = 0; k < l1; ++k)
                {
                    // MPINPLACE(C1(0,k,jc), C1(0,k,j)): a=C1(0,k,jc), b=C1(0,k,j); t=a; a=a-b; b=t+b
                    double av = cc[C1(0, k, jc)], bv = cc[C1(0, k, j)];
                    cc[C1(0, k, jc)] = av - bv;
                    cc[C1(0, k, j)] = av + bv;
                }

            for (long l = 1, lc = ip - 1; l < ipph; ++l, --lc)
            {
                for (long ik = 0; ik < idl1; ++ik)
                {
                    ch[CH2(ik, l)] = cc[C2(ik, 0)] + csarr[2 * l] * cc[C2(ik, 1)] + csarr[4 * l] * cc[C2(ik, 2)];
                    ch[CH2(ik, lc)] = csarr[2 * l + 1] * cc[C2(ik, ip - 1)] + csarr[4 * l + 1] * cc[C2(ik, ip - 2)];
                }
                long iang = 2 * l;
                long j = 3, jc = ip - 3;
                for (; j < ipph - 3; j += 4, jc -= 4)
                {
                    iang += l; if (iang >= ip) iang -= ip;
                    double ar1 = csarr[2 * iang], ai1 = csarr[2 * iang + 1];
                    iang += l; if (iang >= ip) iang -= ip;
                    double ar2 = csarr[2 * iang], ai2 = csarr[2 * iang + 1];
                    iang += l; if (iang >= ip) iang -= ip;
                    double ar3 = csarr[2 * iang], ai3 = csarr[2 * iang + 1];
                    iang += l; if (iang >= ip) iang -= ip;
                    double ar4 = csarr[2 * iang], ai4 = csarr[2 * iang + 1];
                    for (long ik = 0; ik < idl1; ++ik)
                    {
                        ch[CH2(ik, l)] += ar1 * cc[C2(ik, j)] + ar2 * cc[C2(ik, j + 1)]
                                        + ar3 * cc[C2(ik, j + 2)] + ar4 * cc[C2(ik, j + 3)];
                        ch[CH2(ik, lc)] += ai1 * cc[C2(ik, jc)] + ai2 * cc[C2(ik, jc - 1)]
                                         + ai3 * cc[C2(ik, jc - 2)] + ai4 * cc[C2(ik, jc - 3)];
                    }
                }
                for (; j < ipph - 1; j += 2, jc -= 2)
                {
                    iang += l; if (iang >= ip) iang -= ip;
                    double ar1 = csarr[2 * iang], ai1 = csarr[2 * iang + 1];
                    iang += l; if (iang >= ip) iang -= ip;
                    double ar2 = csarr[2 * iang], ai2 = csarr[2 * iang + 1];
                    for (long ik = 0; ik < idl1; ++ik)
                    {
                        ch[CH2(ik, l)] += ar1 * cc[C2(ik, j)] + ar2 * cc[C2(ik, j + 1)];
                        ch[CH2(ik, lc)] += ai1 * cc[C2(ik, jc)] + ai2 * cc[C2(ik, jc - 1)];
                    }
                }
                for (; j < ipph; ++j, --jc)
                {
                    iang += l; if (iang >= ip) iang -= ip;
                    double ar = csarr[2 * iang], ai = csarr[2 * iang + 1];
                    for (long ik = 0; ik < idl1; ++ik)
                    {
                        ch[CH2(ik, l)] += ar * cc[C2(ik, j)];
                        ch[CH2(ik, lc)] += ai * cc[C2(ik, jc)];
                    }
                }
            }
            for (long ik = 0; ik < idl1; ++ik)
                ch[CH2(ik, 0)] = cc[C2(ik, 0)];
            for (long j = 1; j < ipph; ++j)
                for (long ik = 0; ik < idl1; ++ik)
                    ch[CH2(ik, 0)] += cc[C2(ik, j)];

            for (long k = 0; k < l1; ++k)
                for (long i = 0; i < ido; ++i)
                    cc[CC(i, 0, k)] = ch[CH(i, k, 0)];

            for (long j = 1, jc = ip - 1; j < ipph; ++j, --jc)
            {
                long j2 = 2 * j - 1;
                for (long k = 0; k < l1; ++k)
                {
                    cc[CC(ido - 1, j2, k)] = ch[CH(0, k, j)];
                    cc[CC(0, j2 + 1, k)] = ch[CH(0, k, jc)];
                }
            }

            if (ido == 1) return;

            for (long j = 1, jc = ip - 1; j < ipph; ++j, --jc)
            {
                long j2 = 2 * j - 1;
                for (long k = 0; k < l1; ++k)
                    for (long i = 1, ic = ido - i - 2; i <= ido - 2; i += 2, ic -= 2)
                    {
                        cc[CC(i, j2 + 1, k)] = ch[CH(i, k, j)] + ch[CH(i, k, jc)];
                        cc[CC(ic, j2, k)] = ch[CH(i, k, j)] - ch[CH(i, k, jc)];
                        cc[CC(i + 1, j2 + 1, k)] = ch[CH(i + 1, k, j)] + ch[CH(i + 1, k, jc)];
                        cc[CC(ic + 1, j2, k)] = ch[CH(i + 1, k, jc)] - ch[CH(i + 1, k, j)];
                    }
            }
        }

        // ============================ backward (hc2r) codelets ============================

        private static void Radb2(long ido, long l1, double* cc, double* ch, double* wa)
        {
            long WA(long x, long i) => i + x * (ido - 1);
            long CC(long a, long b, long c) => a + ido * (b + 2 * c);
            long CH(long a, long b, long c) => a + ido * (b + l1 * c);

            for (long k = 0; k < l1; k++)
            {
                ch[CH(0, k, 0)] = cc[CC(0, 0, k)] + cc[CC(ido - 1, 1, k)];
                ch[CH(0, k, 1)] = cc[CC(0, 0, k)] - cc[CC(ido - 1, 1, k)];
            }
            if ((ido & 1) == 0)
                for (long k = 0; k < l1; k++)
                {
                    ch[CH(ido - 1, k, 0)] = 2 * cc[CC(ido - 1, 0, k)];
                    ch[CH(ido - 1, k, 1)] = -2 * cc[CC(0, 1, k)];
                }
            if (ido <= 2) return;
            for (long k = 0; k < l1; ++k)
                for (long i = 2; i < ido; i += 2)
                {
                    long ic = ido - i;
                    ch[CH(i - 1, k, 0)] = cc[CC(i - 1, 0, k)] + cc[CC(ic - 1, 1, k)];
                    double tr2 = cc[CC(i - 1, 0, k)] - cc[CC(ic - 1, 1, k)];
                    double ti2 = cc[CC(i, 0, k)] + cc[CC(ic, 1, k)];
                    ch[CH(i, k, 0)] = cc[CC(i, 0, k)] - cc[CC(ic, 1, k)];
                    MULPM(out double m1, out double m2, wa[WA(0, i - 2)], wa[WA(0, i - 1)], ti2, tr2);
                    ch[CH(i, k, 1)] = m1;
                    ch[CH(i - 1, k, 1)] = m2;
                }
        }

        private static void Radb3(long ido, long l1, double* cc, double* ch, double* wa)
        {
            const double taur = -0.5, taui = 0.8660254037844386467637231707529362;
            long WA(long x, long i) => i + x * (ido - 1);
            long CC(long a, long b, long c) => a + ido * (b + 3 * c);
            long CH(long a, long b, long c) => a + ido * (b + l1 * c);

            for (long k = 0; k < l1; k++)
            {
                double tr2 = 2 * cc[CC(ido - 1, 1, k)];
                double cr2 = cc[CC(0, 0, k)] + taur * tr2;
                ch[CH(0, k, 0)] = cc[CC(0, 0, k)] + tr2;
                double ci3 = 2 * taui * cc[CC(0, 2, k)];
                ch[CH(0, k, 2)] = cr2 + ci3;
                ch[CH(0, k, 1)] = cr2 - ci3;
            }
            if (ido == 1) return;
            for (long k = 0; k < l1; k++)
                for (long i = 2, ic = ido - 2; i < ido; i += 2, ic -= 2)
                {
                    double tr2 = cc[CC(i - 1, 2, k)] + cc[CC(ic - 1, 1, k)];
                    double ti2 = cc[CC(i, 2, k)] - cc[CC(ic, 1, k)];
                    double cr2 = cc[CC(i - 1, 0, k)] + taur * tr2;
                    double ci2 = cc[CC(i, 0, k)] + taur * ti2;
                    ch[CH(i - 1, k, 0)] = cc[CC(i - 1, 0, k)] + tr2;
                    ch[CH(i, k, 0)] = cc[CC(i, 0, k)] + ti2;
                    double cr3 = taui * (cc[CC(i - 1, 2, k)] - cc[CC(ic - 1, 1, k)]);
                    double ci3 = taui * (cc[CC(i, 2, k)] + cc[CC(ic, 1, k)]);
                    double dr3 = cr2 + ci3, dr2 = cr2 - ci3;
                    double di2 = ci2 + cr3, di3 = ci2 - cr3;
                    MULPM(out double m1, out double m2, wa[WA(0, i - 2)], wa[WA(0, i - 1)], di2, dr2);
                    ch[CH(i, k, 1)] = m1; ch[CH(i - 1, k, 1)] = m2;
                    MULPM(out double n1, out double n2, wa[WA(1, i - 2)], wa[WA(1, i - 1)], di3, dr3);
                    ch[CH(i, k, 2)] = n1; ch[CH(i - 1, k, 2)] = n2;
                }
        }

        private static void Radb4(long ido, long l1, double* cc, double* ch, double* wa)
        {
            const double sqrt2 = 1.414213562373095048801688724209698;
            long WA(long x, long i) => i + x * (ido - 1);
            long CC(long a, long b, long c) => a + ido * (b + 4 * c);
            long CH(long a, long b, long c) => a + ido * (b + l1 * c);

            for (long k = 0; k < l1; k++)
            {
                double tr2 = cc[CC(0, 0, k)] + cc[CC(ido - 1, 3, k)];
                double tr1 = cc[CC(0, 0, k)] - cc[CC(ido - 1, 3, k)];
                double tr3 = 2 * cc[CC(ido - 1, 1, k)];
                double tr4 = 2 * cc[CC(0, 2, k)];
                ch[CH(0, k, 0)] = tr2 + tr3;
                ch[CH(0, k, 2)] = tr2 - tr3;
                ch[CH(0, k, 3)] = tr1 + tr4;
                ch[CH(0, k, 1)] = tr1 - tr4;
            }
            if ((ido & 1) == 0)
                for (long k = 0; k < l1; k++)
                {
                    double ti1 = cc[CC(0, 3, k)] + cc[CC(0, 1, k)];
                    double ti2 = cc[CC(0, 3, k)] - cc[CC(0, 1, k)];
                    double tr2 = cc[CC(ido - 1, 0, k)] + cc[CC(ido - 1, 2, k)];
                    double tr1 = cc[CC(ido - 1, 0, k)] - cc[CC(ido - 1, 2, k)];
                    ch[CH(ido - 1, k, 0)] = tr2 + tr2;
                    ch[CH(ido - 1, k, 1)] = sqrt2 * (tr1 - ti1);
                    ch[CH(ido - 1, k, 2)] = ti2 + ti2;
                    ch[CH(ido - 1, k, 3)] = -sqrt2 * (tr1 + ti1);
                }
            if (ido <= 2) return;
            for (long k = 0; k < l1; ++k)
                for (long i = 2; i < ido; i += 2)
                {
                    long ic = ido - i;
                    double tr2 = cc[CC(i - 1, 0, k)] + cc[CC(ic - 1, 3, k)];
                    double tr1 = cc[CC(i - 1, 0, k)] - cc[CC(ic - 1, 3, k)];
                    double ti1 = cc[CC(i, 0, k)] + cc[CC(ic, 3, k)];
                    double ti2 = cc[CC(i, 0, k)] - cc[CC(ic, 3, k)];
                    double tr4 = cc[CC(i, 2, k)] + cc[CC(ic, 1, k)];
                    double ti3 = cc[CC(i, 2, k)] - cc[CC(ic, 1, k)];
                    double tr3 = cc[CC(i - 1, 2, k)] + cc[CC(ic - 1, 1, k)];
                    double ti4 = cc[CC(i - 1, 2, k)] - cc[CC(ic - 1, 1, k)];
                    ch[CH(i - 1, k, 0)] = tr2 + tr3;
                    double cr3 = tr2 - tr3;
                    ch[CH(i, k, 0)] = ti2 + ti3;
                    double ci3 = ti2 - ti3;
                    double cr4 = tr1 + tr4, cr2 = tr1 - tr4;
                    double ci2 = ti1 + ti4, ci4 = ti1 - ti4;
                    MULPM(out double m1, out double m2, wa[WA(0, i - 2)], wa[WA(0, i - 1)], ci2, cr2);
                    ch[CH(i, k, 1)] = m1; ch[CH(i - 1, k, 1)] = m2;
                    MULPM(out double n1, out double n2, wa[WA(1, i - 2)], wa[WA(1, i - 1)], ci3, cr3);
                    ch[CH(i, k, 2)] = n1; ch[CH(i - 1, k, 2)] = n2;
                    MULPM(out double o1, out double o2, wa[WA(2, i - 2)], wa[WA(2, i - 1)], ci4, cr4);
                    ch[CH(i, k, 3)] = o1; ch[CH(i - 1, k, 3)] = o2;
                }
        }

        private static void Radb5(long ido, long l1, double* cc, double* ch, double* wa)
        {
            const double tr11 = 0.3090169943749474241022934171828191, ti11 = 0.9510565162951535721164393333793821,
                         tr12 = -0.8090169943749474241022934171828191, ti12 = 0.5877852522924731291687059546390728;
            long WA(long x, long i) => i + x * (ido - 1);
            long CC(long a, long b, long c) => a + ido * (b + 5 * c);
            long CH(long a, long b, long c) => a + ido * (b + l1 * c);

            for (long k = 0; k < l1; k++)
            {
                double ti5 = cc[CC(0, 2, k)] + cc[CC(0, 2, k)];
                double ti4 = cc[CC(0, 4, k)] + cc[CC(0, 4, k)];
                double tr2 = cc[CC(ido - 1, 1, k)] + cc[CC(ido - 1, 1, k)];
                double tr3 = cc[CC(ido - 1, 3, k)] + cc[CC(ido - 1, 3, k)];
                ch[CH(0, k, 0)] = cc[CC(0, 0, k)] + tr2 + tr3;
                double cr2 = cc[CC(0, 0, k)] + tr11 * tr2 + tr12 * tr3;
                double cr3 = cc[CC(0, 0, k)] + tr12 * tr2 + tr11 * tr3;
                MULPM(out double ci5, out double ci4, ti5, ti4, ti11, ti12);
                ch[CH(0, k, 4)] = cr2 + ci5;
                ch[CH(0, k, 1)] = cr2 - ci5;
                ch[CH(0, k, 3)] = cr3 + ci4;
                ch[CH(0, k, 2)] = cr3 - ci4;
            }
            if (ido == 1) return;
            for (long k = 0; k < l1; ++k)
                for (long i = 2, ic = ido - 2; i < ido; i += 2, ic -= 2)
                {
                    double tr2 = cc[CC(i - 1, 2, k)] + cc[CC(ic - 1, 1, k)];
                    double tr5 = cc[CC(i - 1, 2, k)] - cc[CC(ic - 1, 1, k)];
                    double ti5 = cc[CC(i, 2, k)] + cc[CC(ic, 1, k)];
                    double ti2 = cc[CC(i, 2, k)] - cc[CC(ic, 1, k)];
                    double tr3 = cc[CC(i - 1, 4, k)] + cc[CC(ic - 1, 3, k)];
                    double tr4 = cc[CC(i - 1, 4, k)] - cc[CC(ic - 1, 3, k)];
                    double ti4 = cc[CC(i, 4, k)] + cc[CC(ic, 3, k)];
                    double ti3 = cc[CC(i, 4, k)] - cc[CC(ic, 3, k)];
                    ch[CH(i - 1, k, 0)] = cc[CC(i - 1, 0, k)] + tr2 + tr3;
                    ch[CH(i, k, 0)] = cc[CC(i, 0, k)] + ti2 + ti3;
                    double cr2 = cc[CC(i - 1, 0, k)] + tr11 * tr2 + tr12 * tr3;
                    double ci2 = cc[CC(i, 0, k)] + tr11 * ti2 + tr12 * ti3;
                    double cr3 = cc[CC(i - 1, 0, k)] + tr12 * tr2 + tr11 * tr3;
                    double ci3 = cc[CC(i, 0, k)] + tr12 * ti2 + tr11 * ti3;
                    MULPM(out double cr5, out double cr4, tr5, tr4, ti11, ti12);
                    MULPM(out double ci5, out double ci4, ti5, ti4, ti11, ti12);
                    double dr4 = cr3 + ci4, dr3 = cr3 - ci4;
                    double di3 = ci3 + cr4, di4 = ci3 - cr4;
                    double dr5 = cr2 + ci5, dr2 = cr2 - ci5;
                    double di2 = ci2 + cr5, di5 = ci2 - cr5;
                    MULPM(out double m1, out double m2, wa[WA(0, i - 2)], wa[WA(0, i - 1)], di2, dr2);
                    ch[CH(i, k, 1)] = m1; ch[CH(i - 1, k, 1)] = m2;
                    MULPM(out double n1, out double n2, wa[WA(1, i - 2)], wa[WA(1, i - 1)], di3, dr3);
                    ch[CH(i, k, 2)] = n1; ch[CH(i - 1, k, 2)] = n2;
                    MULPM(out double o1, out double o2, wa[WA(2, i - 2)], wa[WA(2, i - 1)], di4, dr4);
                    ch[CH(i, k, 3)] = o1; ch[CH(i - 1, k, 3)] = o2;
                    MULPM(out double p1, out double p2, wa[WA(3, i - 2)], wa[WA(3, i - 1)], di5, dr5);
                    ch[CH(i, k, 4)] = p1; ch[CH(i - 1, k, 4)] = p2;
                }
        }

        private static void Radbg(long ido, long ip, long l1, double* cc, double* ch, double* wa, double* csarr)
        {
            long cdim = ip;
            long ipph = (ip + 1) / 2;
            long idl1 = ido * l1;

            long CC(long a, long b, long c) => a + ido * (b + cdim * c);
            long CH(long a, long b, long c) => a + ido * (b + l1 * c);
            long C1(long a, long b, long c) => a + ido * (b + l1 * c);
            long C2(long a, long b) => a + idl1 * b;
            long CH2(long a, long b) => a + idl1 * b;

            for (long k = 0; k < l1; ++k)
                for (long i = 0; i < ido; ++i)
                    ch[CH(i, k, 0)] = cc[CC(i, 0, k)];
            for (long j = 1, jc = ip - 1; j < ipph; ++j, --jc)
            {
                long j2 = 2 * j - 1;
                for (long k = 0; k < l1; ++k)
                {
                    ch[CH(0, k, j)] = 2 * cc[CC(ido - 1, j2, k)];
                    ch[CH(0, k, jc)] = 2 * cc[CC(0, j2 + 1, k)];
                }
            }

            if (ido != 1)
            {
                for (long j = 1, jc = ip - 1; j < ipph; ++j, --jc)
                {
                    long j2 = 2 * j - 1;
                    for (long k = 0; k < l1; ++k)
                        for (long i = 1, ic = ido - i - 2; i <= ido - 2; i += 2, ic -= 2)
                        {
                            ch[CH(i, k, j)] = cc[CC(i, j2 + 1, k)] + cc[CC(ic, j2, k)];
                            ch[CH(i, k, jc)] = cc[CC(i, j2 + 1, k)] - cc[CC(ic, j2, k)];
                            ch[CH(i + 1, k, j)] = cc[CC(i + 1, j2 + 1, k)] - cc[CC(ic + 1, j2, k)];
                            ch[CH(i + 1, k, jc)] = cc[CC(i + 1, j2 + 1, k)] + cc[CC(ic + 1, j2, k)];
                        }
                }
            }
            for (long l = 1, lc = ip - 1; l < ipph; ++l, --lc)
            {
                for (long ik = 0; ik < idl1; ++ik)
                {
                    cc[C2(ik, l)] = ch[CH2(ik, 0)] + csarr[2 * l] * ch[CH2(ik, 1)] + csarr[4 * l] * ch[CH2(ik, 2)];
                    cc[C2(ik, lc)] = csarr[2 * l + 1] * ch[CH2(ik, ip - 1)] + csarr[4 * l + 1] * ch[CH2(ik, ip - 2)];
                }
                long iang = 2 * l;
                long j = 3, jc = ip - 3;
                for (; j < ipph - 3; j += 4, jc -= 4)
                {
                    iang += l; if (iang > ip) iang -= ip;
                    double ar1 = csarr[2 * iang], ai1 = csarr[2 * iang + 1];
                    iang += l; if (iang > ip) iang -= ip;
                    double ar2 = csarr[2 * iang], ai2 = csarr[2 * iang + 1];
                    iang += l; if (iang > ip) iang -= ip;
                    double ar3 = csarr[2 * iang], ai3 = csarr[2 * iang + 1];
                    iang += l; if (iang > ip) iang -= ip;
                    double ar4 = csarr[2 * iang], ai4 = csarr[2 * iang + 1];
                    for (long ik = 0; ik < idl1; ++ik)
                    {
                        cc[C2(ik, l)] += ar1 * ch[CH2(ik, j)] + ar2 * ch[CH2(ik, j + 1)]
                                       + ar3 * ch[CH2(ik, j + 2)] + ar4 * ch[CH2(ik, j + 3)];
                        cc[C2(ik, lc)] += ai1 * ch[CH2(ik, jc)] + ai2 * ch[CH2(ik, jc - 1)]
                                        + ai3 * ch[CH2(ik, jc - 2)] + ai4 * ch[CH2(ik, jc - 3)];
                    }
                }
                for (; j < ipph - 1; j += 2, jc -= 2)
                {
                    iang += l; if (iang > ip) iang -= ip;
                    double ar1 = csarr[2 * iang], ai1 = csarr[2 * iang + 1];
                    iang += l; if (iang > ip) iang -= ip;
                    double ar2 = csarr[2 * iang], ai2 = csarr[2 * iang + 1];
                    for (long ik = 0; ik < idl1; ++ik)
                    {
                        cc[C2(ik, l)] += ar1 * ch[CH2(ik, j)] + ar2 * ch[CH2(ik, j + 1)];
                        cc[C2(ik, lc)] += ai1 * ch[CH2(ik, jc)] + ai2 * ch[CH2(ik, jc - 1)];
                    }
                }
                for (; j < ipph; ++j, --jc)
                {
                    iang += l; if (iang > ip) iang -= ip;
                    double war = csarr[2 * iang], wai = csarr[2 * iang + 1];
                    for (long ik = 0; ik < idl1; ++ik)
                    {
                        cc[C2(ik, l)] += war * ch[CH2(ik, j)];
                        cc[C2(ik, lc)] += wai * ch[CH2(ik, jc)];
                    }
                }
            }
            for (long j = 1; j < ipph; ++j)
                for (long ik = 0; ik < idl1; ++ik)
                    ch[CH2(ik, 0)] += ch[CH2(ik, j)];
            for (long j = 1, jc = ip - 1; j < ipph; ++j, --jc)
                for (long k = 0; k < l1; ++k)
                {
                    // PM(CH(0,k,jc),CH(0,k,j),C1(0,k,j),C1(0,k,jc))
                    double a = cc[C1(0, k, j)], b = cc[C1(0, k, jc)];
                    ch[CH(0, k, jc)] = a + b;
                    ch[CH(0, k, j)] = a - b;
                }

            if (ido == 1) return;

            for (long j = 1, jc = ip - 1; j < ipph; ++j, --jc)
                for (long k = 0; k < l1; ++k)
                    for (long i = 1; i <= ido - 2; i += 2)
                    {
                        ch[CH(i, k, j)] = cc[C1(i, k, j)] - cc[C1(i + 1, k, jc)];
                        ch[CH(i, k, jc)] = cc[C1(i, k, j)] + cc[C1(i + 1, k, jc)];
                        ch[CH(i + 1, k, j)] = cc[C1(i + 1, k, j)] + cc[C1(i, k, jc)];
                        ch[CH(i + 1, k, jc)] = cc[C1(i + 1, k, j)] - cc[C1(i, k, jc)];
                    }

            for (long j = 1; j < ip; ++j)
            {
                long is0 = (j - 1) * (ido - 1);
                for (long k = 0; k < l1; ++k)
                {
                    long idij = is0;
                    for (long i = 1; i <= ido - 2; i += 2)
                    {
                        double t1 = ch[CH(i, k, j)], t2 = ch[CH(i + 1, k, j)];
                        ch[CH(i, k, j)] = wa[idij] * t1 - wa[idij + 1] * t2;
                        ch[CH(i + 1, k, j)] = wa[idij] * t2 + wa[idij + 1] * t1;
                        idij += 2;
                    }
                }
            }
        }

        // ================================ exec ================================

        public void Exec(double* c, double fct, bool r2hc)
        {
            if (length == 1) { c[0] *= fct; return; }
            int nf = fact.Count;
            var chArr = ArrayPool<double>.Shared.Rent((int)length);
            try
            {
                fixed (double* chp = chArr)
                {
                    double* p1 = c, p2 = chp;
                    if (r2hc)
                    {
                        long l1 = length;
                        for (int k1 = 0; k1 < nf; ++k1)
                        {
                            int k = nf - 1 - k1;
                            long ip = fact[k].fct;
                            long ido = length / l1;
                            l1 /= ip;
                            fixed (double* tw = fact[k].tw, tws = fact[k].tws)
                            {
                                if (ip == 4) Radf4(ido, l1, p1, p2, tw);
                                else if (ip == 2) Radf2(ido, l1, p1, p2, tw);
                                else if (ip == 3) Radf3(ido, l1, p1, p2, tw);
                                else if (ip == 5) Radf5(ido, l1, p1, p2, tw);
                                else { Radfg(ido, ip, l1, p1, p2, tw, tws); double* t = p1; p1 = p2; p2 = t; }
                            }
                            { double* t = p1; p1 = p2; p2 = t; }
                        }
                    }
                    else
                    {
                        long l1 = 1;
                        for (int k = 0; k < nf; ++k)
                        {
                            long ip = fact[k].fct;
                            long ido = length / (ip * l1);
                            fixed (double* tw = fact[k].tw, tws = fact[k].tws)
                            {
                                if (ip == 4) Radb4(ido, l1, p1, p2, tw);
                                else if (ip == 2) Radb2(ido, l1, p1, p2, tw);
                                else if (ip == 3) Radb3(ido, l1, p1, p2, tw);
                                else if (ip == 5) Radb5(ido, l1, p1, p2, tw);
                                else Radbg(ido, ip, l1, p1, p2, tw, tws);
                            }
                            { double* t = p1; p1 = p2; p2 = t; }
                            l1 *= ip;
                        }
                    }
                    // copy_and_norm
                    if (p1 != c)
                    {
                        if (fct != 1.0)
                            for (long i = 0; i < length; ++i) c[i] = fct * p1[i];
                        else
                            for (long i = 0; i < length; ++i) c[i] = p1[i];
                    }
                    else if (fct != 1.0)
                        for (long i = 0; i < length; ++i) c[i] *= fct;
                }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(chArr);
            }
        }
    }
}
