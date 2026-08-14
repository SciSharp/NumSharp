using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Fourier
{
    /// <summary>
    ///     End-to-end correctness of the wired FFT path (np.fft.* -> RawFft -> PocketFFTDriver ->
    ///     the ported pocketfft engine). Round-trips and the DC-component identity, verified against
    ///     NumPy 2.4.2 behaviour. (Was [OpenBugs] while compute was stubbed; un-gated once wired.)
    /// </summary>
    [TestClass]
    public class NpFftRoundTripTest : TestClass
    {
        [TestMethod]
        public void Fft_Ifft_RoundTrip()
        {
            var x = new NDArray(new double[] { 1, 2, 3, 4, 3, 2 });
            var r = np.fft.ifft(np.fft.fft(x)).Data<Complex>();
            for (int i = 0; i < 6; i++)
                r[i].Real.Should().BeApproximately(x.Data<double>()[i], 1e-9);
        }

        [TestMethod]
        public void Rfft_Irfft_RoundTrip()
        {
            var x = new NDArray(new double[] { 1, 2, 3, 4, 3, 2 });
            var r = np.fft.irfft(np.fft.rfft(x), n: 6).Data<double>();
            for (int i = 0; i < 6; i++)
                r[i].Should().BeApproximately(x.Data<double>()[i], 1e-9);
        }

        [TestMethod]
        public void Fft_Ones_DcComponentOnly()
        {
            // np.fft.fft(ones(4)) == [4+0j, 0, 0, 0]
            var r = np.fft.fft(new NDArray(new double[] { 1, 1, 1, 1 })).Data<Complex>();
            r[0].Real.Should().BeApproximately(4.0, 1e-9);
            for (int i = 1; i < 4; i++)
                r[i].Magnitude.Should().BeApproximately(0.0, 1e-9);
        }
    }
}
