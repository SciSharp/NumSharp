using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Fourier
{
    /// <summary>
    ///     The compute-seam contract: for VALID inputs every transform passes its full validation and
    ///     then throws <see cref="NotImplementedException"/> at the engine seam
    ///     (<c>PocketFFTDriver.Execute</c>) — the point a separate agent wires the compute. These assert
    ///     that validation runs and the seam is reached (NOT the transform's numeric result). When the
    ///     engine lands, these are replaced by value/round-trip tests.
    /// </summary>
    [TestClass]
    public class NpFftSeamTest : TestClass
    {
        private static NDArray R(int n) => np.arange(n);
        private static NDArray C(int n) => new NDArray(np.complex128, n);

        private static void ReachesSeam(Action act, string name)
            => act.Should().Throw<NotImplementedException>()
                  .WithMessage($"np.fft.{name}: FFT compute pending (engine seam) -> PocketFFTDriver.Execute");

        [TestMethod] public void Fft_Valid_ReachesSeam() => ReachesSeam(() => np.fft.fft(R(8)), "fft");
        [TestMethod] public void Ifft_Valid_ReachesSeam() => ReachesSeam(() => np.fft.ifft(R(8)), "ifft");
        [TestMethod] public void Rfft_Valid_ReachesSeam() => ReachesSeam(() => np.fft.rfft(R(8)), "rfft");
        [TestMethod] public void Irfft_Valid_ReachesSeam() => ReachesSeam(() => np.fft.irfft(C(5)), "irfft");

        // hfft = irfft(conjugate(a), ...) — the conjugate computes, the irfft leaf reaches the seam.
        [TestMethod] public void Hfft_Valid_ReachesIrfftSeam() => ReachesSeam(() => np.fft.hfft(R(4)), "irfft");

        // ihfft = conjugate(rfft(a, ...)) — the rfft leaf reaches the seam before the conjugate runs.
        [TestMethod] public void Ihfft_Valid_ReachesRfftSeam() => ReachesSeam(() => np.fft.ihfft(R(6)), "rfft");

        // N-D wrappers decompose to 1-D leaves; the FIRST leaf reaches the seam.
        [TestMethod] public void Fft2_Valid_ReachesFftSeam() => ReachesSeam(() => np.fft.fft2(np.arange(6).reshape(2, 3)), "fft");
        [TestMethod] public void Fftn_Valid_ReachesFftSeam() => ReachesSeam(() => np.fft.fftn(np.arange(6).reshape(2, 3)), "fft");
        [TestMethod] public void Ifftn_Valid_ReachesIfftSeam() => ReachesSeam(() => np.fft.ifftn(np.arange(6).reshape(2, 3)), "ifft");
        [TestMethod] public void Rfftn_Valid_ReachesRfftSeam() => ReachesSeam(() => np.fft.rfftn(np.arange(6).reshape(2, 3)), "rfft");

        // irfftn over a 2-D input runs ifft over the non-last axis first.
        [TestMethod]
        public void Irfftn_Valid_ReachesIfftSeam()
        {
            var m = new NDArray(np.complex128, 4).reshape(2, 2);
            ReachesSeam(() => np.fft.irfftn(m), "ifft");
        }

        // A correctly-shaped out= passes validation and still reaches the seam.
        [TestMethod]
        public void Fft_ValidOut_ReachesSeam()
            => ReachesSeam(() => np.fft.fft(R(4), @out: new NDArray(np.complex128, 4)), "fft");
    }
}
