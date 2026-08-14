using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Fourier
{
    /// <summary>
    ///     np.fft helper routines — fully computed (no engine seam). Values probed against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class NpFftHelperTest : TestClass
    {
        // ---- fftfreq / rfftfreq (values) ----

        [TestMethod]
        public void FftFreq_Even()
        {
            // np.fft.fftfreq(8) == [0, .125, .25, .375, -.5, -.375, -.25, -.125]
            np.fft.fftfreq(8).Data<double>().Should().BeEquivalentTo(
                new[] { 0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125 });
        }

        [TestMethod]
        public void FftFreq_WithSpacing()
        {
            // np.fft.fftfreq(10, 0.1) == [0,1,2,3,4,-5,-4,-3,-2,-1]
            np.fft.fftfreq(10, 0.1).Data<double>().Should().BeEquivalentTo(
                new[] { 0.0, 1, 2, 3, 4, -5, -4, -3, -2, -1 });
        }

        [TestMethod]
        public void FftFreq_Odd_Length()
        {
            var r = np.fft.fftfreq(9);
            r.size.Should().Be(9);
            var d = r.Data<double>();
            d[0].Should().Be(0.0);
            d[4].Should().BeApproximately(4.0 / 9.0, 1e-15);
            d[5].Should().BeApproximately(-4.0 / 9.0, 1e-15);
        }

        [TestMethod]
        public void RfftFreq_Even()
        {
            // np.fft.rfftfreq(8) == [0, .125, .25, .375, .5]   (length n//2+1)
            np.fft.rfftfreq(8).Data<double>().Should().BeEquivalentTo(
                new[] { 0.0, 0.125, 0.25, 0.375, 0.5 });
        }

        [TestMethod]
        public void RfftFreq_Odd_Length()
        {
            var r = np.fft.rfftfreq(9);
            r.size.Should().Be(5);   // 9//2 + 1
            r.Data<double>()[4].Should().BeApproximately(4.0 / 9.0, 1e-15);
        }

        [TestMethod]
        public void FftFreq_NonInteger_Raises()
        {
            Action act = () => { var _ = np.fft.fftfreq(2.5); };
            act.Should().Throw<ValueError>().WithMessage("n should be an integer");
        }

        [TestMethod]
        public void RfftFreq_NonInteger_Raises()
        {
            Action act = () => { var _ = np.fft.rfftfreq(3.5); };
            act.Should().Throw<ValueError>().WithMessage("n should be an integer");
        }

        [TestMethod]
        public void FftFreq_Negative_Raises()
        {
            Action act = () => { var _ = np.fft.fftfreq(-3); };
            act.Should().Throw<ValueError>().WithMessage("negative dimensions are not allowed");
        }

        // ---- fftshift / ifftshift (values) ----

        [TestMethod]
        public void FftShift_1D_Even()
        {
            // np.fft.fftshift(arange(10)) == [5,6,7,8,9,0,1,2,3,4]
            np.fft.fftshift(np.arange(10)).Data<long>().Should().BeEquivalentTo(
                new long[] { 5, 6, 7, 8, 9, 0, 1, 2, 3, 4 });
        }

        [TestMethod]
        public void FftShift_1D_Odd()
        {
            // np.fft.fftshift(arange(9)) == [5,6,7,8,0,1,2,3,4]
            np.fft.fftshift(np.arange(9)).Data<long>().Should().BeEquivalentTo(
                new long[] { 5, 6, 7, 8, 0, 1, 2, 3, 4 });
        }

        [TestMethod]
        public void IfftShift_1D_Odd_DiffersByOne()
        {
            // np.fft.ifftshift(arange(9)) == [4,5,6,7,8,0,1,2,3]
            np.fft.ifftshift(np.arange(9)).Data<long>().Should().BeEquivalentTo(
                new long[] { 4, 5, 6, 7, 8, 0, 1, 2, 3 });
        }

        [TestMethod]
        public void FftShift_2D_AllAxes()
        {
            // np.fft.fftshift(arange(6).reshape(2,3)) == [[5,3,4],[2,0,1]]
            var m = np.arange(6).reshape(2, 3);
            np.fft.fftshift(m).Data<long>().Should().BeEquivalentTo(
                new long[] { 5, 3, 4, 2, 0, 1 });
        }

        [TestMethod]
        public void FftShift_2D_SingleAxis_Int()
        {
            // np.fft.fftshift(m, axes=1) == [[2,0,1],[5,3,4]]
            var m = np.arange(6).reshape(2, 3);
            np.fft.fftshift(m, 1).Data<long>().Should().BeEquivalentTo(
                new long[] { 2, 0, 1, 5, 3, 4 });
        }

        [TestMethod]
        public void FftShift_2D_AxesList()
        {
            // np.fft.fftshift(m, axes=(0,)) == [[3,4,5],[0,1,2]]
            var m = np.arange(6).reshape(2, 3);
            np.fft.fftshift(m, new[] { 0 }).Data<long>().Should().BeEquivalentTo(
                new long[] { 3, 4, 5, 0, 1, 2 });
        }

        [TestMethod]
        public void FftShift_IfftShift_RoundTrip()
        {
            var m = np.arange(6).reshape(2, 3);
            np.fft.ifftshift(np.fft.fftshift(m)).Data<long>().Should().BeEquivalentTo(
                new long[] { 0, 1, 2, 3, 4, 5 });
        }
    }
}
