using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Fourier
{
    /// <summary>
    ///     Compute-pending gate for the FFT engine agent. These need the actual transform (they throw
    ///     <see cref="NotImplementedException"/> at the seam today), so they are <c>[OpenBugs]</c> and
    ///     excluded from CI until <c>PocketFFTDriver.Execute</c> is wired — then drop the attribute.
    /// </summary>
    [TestClass]
    public class NpFftComputePendingTest : TestClass
    {
        [TestMethod]
        [OpenBugs]
        public void Fft_Ifft_RoundTrip()
        {
            var x = new NDArray(new double[] { 1, 2, 3, 4, 3, 2 });
            var r = np.fft.ifft(np.fft.fft(x)).Data<Complex>();
            for (int i = 0; i < 6; i++)
                r[i].Real.Should().BeApproximately(x.Data<double>()[i], 1e-9);
        }

        [TestMethod]
        [OpenBugs]
        public void Rfft_Irfft_RoundTrip()
        {
            var x = new NDArray(new double[] { 1, 2, 3, 4, 3, 2 });
            var r = np.fft.irfft(np.fft.rfft(x), n: 6).Data<double>();
            for (int i = 0; i < 6; i++)
                r[i].Should().BeApproximately(x.Data<double>()[i], 1e-9);
        }

        [TestMethod]
        [OpenBugs]
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
