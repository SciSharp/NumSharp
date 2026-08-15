using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Fourier
{
    /// <summary>
    ///     Regression coverage for two things the float32/float16 FFT port established (FFT_PARITY.md §7):
    ///
    ///     1. The pre-existing DOUBLE-engine bug in the radix-7 / radix-11 codelets' <c>ido&gt;1</c> branch
    ///        (<c>Cfftp.Pass7</c>/<c>Pass11</c> applied <c>special_mul</c> to <c>ca</c>/<c>cb</c> instead of
    ///        <c>ca±cb</c>), which made <c>np.fft.fft(double, n)</c> silently wrong for any <c>n</c> with a
    ///        radix-7/11 factor at ido&gt;1 (49, 98, 121, 143, 259, …). Verified here against a
    ///        codelet-independent direct O(n²) DFT so the check survives corpus regeneration and needs no
    ///        NumPy. (The differential corpus also covers it now via the added n=49/121 sweep sizes.)
    ///
    ///     2. The float32/float16 result-dtype divergence is dtype-ONLY: NumSharp returns complex128 where
    ///        NumPy returns complex64 (no complex64 dtype — issue #569), but the values round-trip and sit
    ///        within float32 precision of the double transform. The exact bit-parity with NumPy 2.4.2 is
    ///        gated by the differential corpus (harness value-verification); this pins the local invariants.
    /// </summary>
    [TestClass]
    public class NpFftFloat32ParityTest : TestClass
    {
        // Naive DFT in double — the ground truth, independent of pocketfft's codelets.
        private static Complex[] DirectDft(Complex[] x, bool forward)
        {
            int n = x.Length;
            var y = new Complex[n];
            double sign = forward ? -1.0 : 1.0;
            for (int k = 0; k < n; k++)
            {
                Complex s = Complex.Zero;
                for (int m = 0; m < n; m++)
                {
                    double ang = sign * 2.0 * Math.PI * k * m / n;
                    s += x[m] * new Complex(Math.Cos(ang), Math.Sin(ang));
                }
                y[k] = s;
            }
            return y;
        }

        private static Complex[] DeterministicComplex(int n)
        {
            // A fixed, well-conditioned signal (no NumPy dependency) — enough to exercise every butterfly.
            var x = new Complex[n];
            for (int m = 0; m < n; m++)
                x[m] = new Complex(Math.Sin(0.7 * m + 0.1), Math.Cos(0.37 * m - 0.2));
            return x;
        }

        [DataTestMethod]
        // radix-7 (ido>1): 49=7², 98=2·7², 259=7·37, 343=7³. radix-11 (ido>1): 121=11², 143=11·13.
        [DataRow(49)]
        [DataRow(98)]
        [DataRow(121)]
        [DataRow(143)]
        [DataRow(259)]
        [DataRow(343)]
        public void Fft_Pass7_Pass11_Ido_GreaterThan1_MatchesDirectDft(int n)
        {
            var x = DeterministicComplex(n);
            var got = np.fft.fft(np.array(x)).Data<Complex>();
            var want = DirectDft(x, forward: true);

            // Direct DFT vs mixed-radix FFT differ only by floating accumulation (~n·eps); a wide absolute
            // tolerance still catches the pre-fix bug, which was wrong by O(1) on ~half the elements.
            for (int k = 0; k < n; k++)
            {
                got[k].Real.Should().BeApproximately(want[k].Real, 1e-9, $"real[{k}] n={n}");
                got[k].Imaginary.Should().BeApproximately(want[k].Imaginary, 1e-9, $"imag[{k}] n={n}");
            }
        }

        [DataTestMethod]
        [DataRow(49)]
        [DataRow(121)]
        [DataRow(259)]
        public void Ifft_Pass7_Pass11_Ido_GreaterThan1_MatchesDirectDft(int n)
        {
            var x = DeterministicComplex(n);
            var got = np.fft.ifft(np.array(x)).Data<Complex>();
            var raw = DirectDft(x, forward: false);
            for (int k = 0; k < n; k++)
            {
                got[k].Real.Should().BeApproximately(raw[k].Real / n, 1e-9, $"real[{k}] n={n}");
                got[k].Imaginary.Should().BeApproximately(raw[k].Imaginary / n, 1e-9, $"imag[{k}] n={n}");
            }
        }

        [TestMethod]
        [Misaligned]  // documents the dtype-only divergence (complex128 vs NumPy complex64) — issue #569
        public void Fft_Float32_ReturnsComplex128_ValuesNearDouble()
        {
            // float32 input -> NumSharp complex128 (NumPy: complex64). Value bit-parity vs NumPy is gated
            // by the differential corpus; here we pin that (a) the dtype is complex128 and (b) the values
            // are within float32 precision of the double transform (so nothing is grossly wrong locally).
            var xf = new float[8] { 0.5f, -0.13f, 0.64f, 1.52f, -0.23f, -0.23f, 1.57f, 0.76f };
            var a32 = np.array(xf);                          // Single
            a32.typecode.Should().Be(NPTypeCode.Single);

            var y = np.fft.fft(a32);
            y.typecode.Should().Be(NPTypeCode.Complex);      // complex128, the documented divergence

            var yd = np.fft.fft(np.array(xf).astype(NPTypeCode.Double)).Data<Complex>();
            var yv = y.Data<Complex>();
            for (int k = 0; k < 8; k++)
            {
                // float32-precision agreement with the double transform (relative ~1e-6).
                double scale = Math.Max(1.0, Math.Abs(yd[k].Real) + Math.Abs(yd[k].Imaginary));
                Math.Abs(yv[k].Real - yd[k].Real).Should().BeLessThan(1e-5 * scale);
                Math.Abs(yv[k].Imaginary - yd[k].Imaginary).Should().BeLessThan(1e-5 * scale);
            }
        }

        [TestMethod]
        public void Rfft_Irfft_Float32_RoundTrips()
        {
            var xf = new float[6] { 1f, 2f, 3f, 4f, 3f, 2f };
            var a = np.array(xf);                            // Single
            var r = np.fft.irfft(np.fft.rfft(a), n: 6).Data<double>();
            for (int i = 0; i < 6; i++)
                r[i].Should().BeApproximately(xf[i], 1e-4);  // float32 round-trip precision
        }
    }
}
