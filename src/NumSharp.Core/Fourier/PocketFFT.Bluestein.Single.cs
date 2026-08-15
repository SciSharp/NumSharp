using System;
using System.Buffers;

// =============================================================================
// SINGLE-PRECISION port of pocketfft's fftblue<T0> — Bluestein's algorithm
// (chirp-z) for lengths with large prime factors, T0 == float. Verbatim
// transcription of PocketFFT.Bluestein.cs with the arithmetic narrowed to float:
// the chirp bk/bkf are cmplx<float> (narrowed from the double twiddle table via
// TwiddleF.At), the normalisation xn2 = 1f/(float)n2 is computed in float exactly
// as pocketfft's `T0(1)/T0(n2)`, and the inner convolution runs on the single
// engine CfftpF. Operation ORDER preserved verbatim for bit-parity with numpy's
// complex64 Bluestein path.
// =============================================================================

namespace NumSharp.Fourier
{
    public sealed unsafe class FftblueF
    {
        private readonly long n, n2;
        private readonly CfftpF plan;
        private readonly CmplxF[] bk;   // length n
        private readonly CmplxF[] bkf;  // length n2/2+1

        public long Length => n;

        public FftblueF(long length)
        {
            n = length;
            n2 = PocketFFTUtil.GoodSizeCmplx(n * 2 - 1);
            plan = new CfftpF(n2);
            bk = new CmplxF[n];
            bkf = new CmplxF[n2 / 2 + 1];

            // initialize b_k
            var tmp = new SinCos2PiByN(2 * n);
            bk[0] = new CmplxF(1f, 0f);
            long coeff = 0;
            for (long m = 1; m < n; ++m)
            {
                coeff += 2 * m - 1;
                if (coeff >= 2 * n) coeff -= 2 * n;
                bk[m] = TwiddleF.At(tmp, coeff);
            }

            // initialize the zero-padded, Fourier transformed b_k. Add normalisation.
            var tbkf = new CmplxF[n2];
            float xn2 = 1f / (float)n2;
            tbkf[0] = bk[0] * xn2;
            for (long m = 1; m < n; ++m)
                tbkf[m] = tbkf[n2 - m] = bk[m] * xn2;
            for (long m = n; m <= (n2 - n); ++m)
                tbkf[m] = new CmplxF(0f, 0f);
            fixed (CmplxF* tp = tbkf)
                plan.Exec(tp, 1f, true);
            for (long i = 0; i < n2 / 2 + 1; ++i)
                bkf[i] = tbkf[i];
        }

        // pocketfft fftblue::fft<fwd> — the complex chirp-z convolution.
        private void Fft(bool fwd, CmplxF* c, float fct)
        {
            var akfArr = ArrayPool<CmplxF>.Shared.Rent((int)n2);
            try
            {
                fixed (CmplxF* akf = akfArr)
                {
                    // initialize a_k and FFT it
                    for (long m = 0; m < n; ++m)
                        akf[m] = CmplxF.SpecialMul(fwd, c[m], bk[m]);
                    CmplxF zero = akf[0] * 0f;
                    for (long m = n; m < n2; ++m)
                        akf[m] = zero;

                    plan.Exec(akf, 1f, true);

                    // do the convolution
                    akf[0] = CmplxF.SpecialMul(!fwd, akf[0], bkf[0]);
                    for (long m = 1; m < (n2 + 1) / 2; ++m)
                    {
                        akf[m] = CmplxF.SpecialMul(!fwd, akf[m], bkf[m]);
                        akf[n2 - m] = CmplxF.SpecialMul(!fwd, akf[n2 - m], bkf[m]);
                    }
                    if ((n2 & 1) == 0)
                        akf[n2 / 2] = CmplxF.SpecialMul(!fwd, akf[n2 / 2], bkf[n2 / 2]);

                    // inverse FFT
                    plan.Exec(akf, 1f, false);

                    // multiply by b_k
                    for (long m = 0; m < n; ++m)
                        c[m] = CmplxF.SpecialMul(fwd, akf[m], bk[m]) * fct;
                }
            }
            finally
            {
                ArrayPool<CmplxF>.Shared.Return(akfArr);
            }
        }

        public void Exec(CmplxF* c, float fct, bool fwd) => Fft(fwd, c, fct);

        // pocketfft fftblue::exec_r — real transform via the complex chirp-z, with the
        // FFTPACK half-complex layout produced/consumed through the interleaved reinterpret.
        public void ExecR(float* c, float fct, bool fwd)
        {
            var tmpArr = new CmplxF[n];
            fixed (CmplxF* tmp = tmpArr)
            {
                float* td = (float*)tmp; // td[2m]=tmp[m].r, td[2m+1]=tmp[m].i (sequential layout)
                if (fwd)
                {
                    float zero = 0f * c[0];
                    for (long m = 0; m < n; ++m)
                        tmp[m] = new CmplxF(c[m], zero);
                    Fft(true, tmp, fct);
                    c[0] = tmp[0].r;
                    // std::copy_n(&tmp[1].r, n-1, &c[1])
                    for (long i = 0; i < n - 1; ++i)
                        c[1 + i] = td[2 + i];
                }
                else
                {
                    tmp[0] = new CmplxF(c[0], c[0] * 0f);
                    // std::copy_n(c+1, n-1, &tmp[1].r)
                    for (long i = 0; i < n - 1; ++i)
                        td[2 + i] = c[1 + i];
                    if ((n & 1) == 0) tmp[n / 2].i = 0f * c[0];
                    for (long m = 1; 2 * m < n; ++m)
                        tmp[n - m] = new CmplxF(tmp[m].r, -tmp[m].i);
                    Fft(false, tmp, fct);
                    for (long m = 0; m < n; ++m)
                        c[m] = tmp[m].r;
                }
            }
        }
    }
}
