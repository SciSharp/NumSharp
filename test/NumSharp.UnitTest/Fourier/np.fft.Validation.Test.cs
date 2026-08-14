using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Fourier
{
    /// <summary>
    ///     np.fft transform SHELLS — the validation surface that must fire BEFORE the compute seam.
    ///     Every message verbatim from NumPy 2.4.2. (The actual transform compute is stubbed; see
    ///     <see cref="NpFftSeamTest"/> for the "reached the seam after validation" contract.)
    /// </summary>
    [TestClass]
    public class NpFftValidationTest : TestClass
    {
        private static NDArray R(int n) => np.arange(n);                       // int64 1-D (real)
        private static NDArray C(int n) => new NDArray(np.complex128, n);      // complex 1-D

        // ---- n < 1 : "Invalid number of FFT data points (N) specified." ----

        [TestMethod]
        public void Fft_NZero_Raises()
        {
            Action act = () => np.fft.fft(R(4), n: 0);
            act.Should().Throw<ValueError>().WithMessage("Invalid number of FFT data points (0) specified.");
        }

        [TestMethod]
        public void Ifft_NNegative_Raises()
        {
            Action act = () => np.fft.ifft(R(4), n: -1);
            act.Should().Throw<ValueError>().WithMessage("Invalid number of FFT data points (-1) specified.");
        }

        [TestMethod]
        public void Rfft_NZero_Raises()
        {
            Action act = () => np.fft.rfft(R(4), n: 0);
            act.Should().Throw<ValueError>().WithMessage("Invalid number of FFT data points (0) specified.");
        }

        [TestMethod]
        public void Irfft_NZero_Raises()
        {
            Action act = () => np.fft.irfft(C(4), n: 0);
            act.Should().Throw<ValueError>().WithMessage("Invalid number of FFT data points (0) specified.");
        }

        // ---- bad norm : FORWARD has NO space after the first comma; INVERSE / hermitian have one ----

        [TestMethod]
        public void Fft_BadNorm_Forward_NoSpaceMessage()
        {
            Action act = () => np.fft.fft(R(4), norm: "bad");
            act.Should().Throw<ValueError>()
                .WithMessage("Invalid norm value bad; should be \"backward\",\"ortho\" or \"forward\".");
        }

        [TestMethod]
        public void Rfft_BadNorm_Forward_NoSpaceMessage()
        {
            Action act = () => np.fft.rfft(R(4), norm: "bad");
            act.Should().Throw<ValueError>()
                .WithMessage("Invalid norm value bad; should be \"backward\",\"ortho\" or \"forward\".");
        }

        [TestMethod]
        public void Ifft_BadNorm_Inverse_WithSpaceMessage()
        {
            Action act = () => np.fft.ifft(R(4), norm: "bad");
            act.Should().Throw<ValueError>()
                .WithMessage("Invalid norm value bad; should be \"backward\", \"ortho\" or \"forward\".");
        }

        [TestMethod]
        public void Irfft_BadNorm_Inverse_WithSpaceMessage()
        {
            Action act = () => np.fft.irfft(C(4), norm: "bad");
            act.Should().Throw<ValueError>()
                .WithMessage("Invalid norm value bad; should be \"backward\", \"ortho\" or \"forward\".");
        }

        [TestMethod]
        public void Hfft_BadNorm_WithSpaceMessage()
        {
            Action act = () => np.fft.hfft(R(4), norm: "q");
            act.Should().Throw<ValueError>()
                .WithMessage("Invalid norm value q; should be \"backward\", \"ortho\" or \"forward\".");
        }

        [TestMethod]
        public void Ihfft_BadNorm_WithSpaceMessage()
        {
            Action act = () => np.fft.ihfft(R(4), norm: "q");
            act.Should().Throw<ValueError>()
                .WithMessage("Invalid norm value q; should be \"backward\", \"ortho\" or \"forward\".");
        }

        // ---- axis : IndexError when n is None (a.shape[axis]); AxisError when n is given ----

        [TestMethod]
        public void Fft_AxisOob_NDefault_IndexError()
        {
            Action act = () => np.fft.fft(R(4), axis: 5);
            act.Should().Throw<IndexError>().WithMessage("tuple index out of range");
        }

        [TestMethod]
        public void Fft_AxisNegOob_NDefault_IndexError()
        {
            Action act = () => np.fft.fft(R(4), axis: -2);
            act.Should().Throw<IndexError>().WithMessage("tuple index out of range");
        }

        [TestMethod]
        public void Fft_ZeroD_IndexError()
        {
            Action act = () => np.fft.fft(NDArray.Scalar(5.0));
            act.Should().Throw<IndexError>().WithMessage("tuple index out of range");
        }

        [TestMethod]
        public void Fft_AxisOob_NGiven_AxisError()
        {
            Action act = () => np.fft.fft(R(4), n: 4, axis: 5);
            act.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 1*");
        }

        // ---- out= wrong shape ----

        [TestMethod]
        public void Fft_OutWrongShape_Raises()
        {
            Action act = () => np.fft.fft(R(4), @out: new NDArray(np.complex128, 3));
            act.Should().Throw<ValueError>().WithMessage("output array has wrong shape.");
        }

        [TestMethod]
        public void Fft_OutWrongNdim_Raises()
        {
            Action act = () => np.fft.fft(R(4), @out: new NDArray(np.complex128, 4).reshape(4, 1));
            act.Should().Throw<ValueError>().WithMessage("output array has wrong shape.");
        }

        // ---- N-D : s/axes mismatch, and take-path oob ----

        [TestMethod]
        public void Fft2_SAxesLengthMismatch_Raises()
        {
            var m = np.arange(60).reshape(3, 4, 5);
            Action act = () => np.fft.fft2(m, new[] { 2, 3, 4 });   // s len 3, default axes (-2,-1) len 2
            act.Should().Throw<ValueError>().WithMessage("Shape and axes have different lengths.");
        }

        [TestMethod]
        public void Fftn_SAxesLengthMismatch_Raises()
        {
            var m = np.arange(60).reshape(3, 4, 5);
            Action act = () => np.fft.fftn(m, new[] { 2, 3 }, new[] { 0, 1, 2 });
            act.Should().Throw<ValueError>().WithMessage("Shape and axes have different lengths.");
        }

        [TestMethod]
        public void Fftn_AxesOob_NoS_TakeIndexError()
        {
            var m = np.arange(12).reshape(3, 4);
            Action act = () => np.fft.fftn(m, axes: new[] { 5 });
            act.Should().Throw<IndexError>().WithMessage("index 5 is out of bounds for axis 0 with size 2");
        }
    }
}
