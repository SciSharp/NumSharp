using System;
using System.Buffers;

// =============================================================================
// Port of pocketfft's fftblue<T0> — Bluestein's algorithm (chirp-z) for lengths
// with large prime factors (pocketfft_hdronly.h lines 2413-2509). Double engine.
// The chirp b_k and its zero-padded transform bkf are precomputed once; each
// transform is a length-n2 complex convolution via the inner cfftp plan.
// =============================================================================

namespace NumSharp.Fourier
{
    internal sealed unsafe class Fftblue
    {
        private readonly long n, n2;
        private readonly Cfftp plan;
        private readonly Cmplx[] bk;   // length n
        private readonly Cmplx[] bkf;  // length n2/2+1

        public long Length => n;

        public Fftblue(long length)
        {
            n = length;
            n2 = PocketFFTUtil.GoodSizeCmplx(n * 2 - 1);
            plan = new Cfftp(n2);
            bk = new Cmplx[n];
            bkf = new Cmplx[n2 / 2 + 1];

            // initialize b_k
            var tmp = new SinCos2PiByN(2 * n);
            bk[0] = new Cmplx(1.0, 0.0);
            long coeff = 0;
            for (long m = 1; m < n; ++m)
            {
                coeff += 2 * m - 1;
                if (coeff >= 2 * n) coeff -= 2 * n;
                bk[m] = tmp[coeff];
            }

            // initialize the zero-padded, Fourier transformed b_k. Add normalisation.
            var tbkf = new Cmplx[n2];
            double xn2 = 1.0 / (double)n2;
            tbkf[0] = bk[0] * xn2;
            for (long m = 1; m < n; ++m)
                tbkf[m] = tbkf[n2 - m] = bk[m] * xn2;
            for (long m = n; m <= (n2 - n); ++m)
                tbkf[m] = new Cmplx(0.0, 0.0);
            fixed (Cmplx* tp = tbkf)
                plan.Exec(tp, 1.0, true);
            for (long i = 0; i < n2 / 2 + 1; ++i)
                bkf[i] = tbkf[i];
        }

        // pocketfft fftblue::fft<fwd> — the complex chirp-z convolution.
        private void Fft(bool fwd, Cmplx* c, double fct)
        {
            var akfArr = ArrayPool<Cmplx>.Shared.Rent((int)n2);
            try
            {
                fixed (Cmplx* akf = akfArr)
                {
                    // initialize a_k and FFT it
                    for (long m = 0; m < n; ++m)
                        akf[m] = Cmplx.SpecialMul(fwd, c[m], bk[m]);
                    Cmplx zero = akf[0] * 0.0;
                    for (long m = n; m < n2; ++m)
                        akf[m] = zero;

                    plan.Exec(akf, 1.0, true);

                    // do the convolution
                    akf[0] = Cmplx.SpecialMul(!fwd, akf[0], bkf[0]);
                    for (long m = 1; m < (n2 + 1) / 2; ++m)
                    {
                        akf[m] = Cmplx.SpecialMul(!fwd, akf[m], bkf[m]);
                        akf[n2 - m] = Cmplx.SpecialMul(!fwd, akf[n2 - m], bkf[m]);
                    }
                    if ((n2 & 1) == 0)
                        akf[n2 / 2] = Cmplx.SpecialMul(!fwd, akf[n2 / 2], bkf[n2 / 2]);

                    // inverse FFT
                    plan.Exec(akf, 1.0, false);

                    // multiply by b_k
                    for (long m = 0; m < n; ++m)
                        c[m] = Cmplx.SpecialMul(fwd, akf[m], bk[m]) * fct;
                }
            }
            finally
            {
                ArrayPool<Cmplx>.Shared.Return(akfArr);
            }
        }

        public void Exec(Cmplx* c, double fct, bool fwd) => Fft(fwd, c, fct);

        // pocketfft fftblue::exec_r — real transform via the complex chirp-z, with the
        // FFTPACK half-complex layout produced/consumed through the interleaved reinterpret.
        public void ExecR(double* c, double fct, bool fwd)
        {
            var tmpArr = new Cmplx[n];
            fixed (Cmplx* tmp = tmpArr)
            {
                double* td = (double*)tmp; // td[2m]=tmp[m].r, td[2m+1]=tmp[m].i (sequential layout)
                if (fwd)
                {
                    double zero = 0.0 * c[0];
                    for (long m = 0; m < n; ++m)
                        tmp[m] = new Cmplx(c[m], zero);
                    Fft(true, tmp, fct);
                    c[0] = tmp[0].r;
                    // std::copy_n(&tmp[1].r, n-1, &c[1])
                    for (long i = 0; i < n - 1; ++i)
                        c[1 + i] = td[2 + i];
                }
                else
                {
                    tmp[0] = new Cmplx(c[0], c[0] * 0);
                    // std::copy_n(c+1, n-1, &tmp[1].r)
                    for (long i = 0; i < n - 1; ++i)
                        td[2 + i] = c[1 + i];
                    if ((n & 1) == 0) tmp[n / 2].i = 0.0 * c[0];
                    for (long m = 1; 2 * m < n; ++m)
                        tmp[n - m] = new Cmplx(tmp[m].r, -tmp[m].i);
                    Fft(false, tmp, fct);
                    for (long m = 0; m < n; ++m)
                        c[m] = tmp[m].r;
                }
            }
        }
    }
}
