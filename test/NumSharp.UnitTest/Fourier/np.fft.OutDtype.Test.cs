using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Fourier
{
    /// <summary>
    ///     out= dtype flexibility (NumPy ufunc same_kind casting FROM the loop output dtype).
    ///     Loop dtype is complex128 for fft/ifft/rfft, float64 for irfft; out= is accepted iff the loop
    ///     dtype casts to it under 'same_kind'. Values/messages verified verbatim against NumPy 2.4.2.
    ///     (NumSharp raises ArgumentException with NumPy's UFuncTypeError text — the house convention,
    ///     identical to DefaultEngine.UfuncOut.ValidateOutCast for clip/negative/etc.)
    /// </summary>
    [TestClass]
    public class NpFftOutDtypeTest : TestClass
    {
        // --- irfft (loop float64) ACCEPTS a same_kind out: complex128 (imag=0), float32 (narrow) ---
        [TestMethod]
        public void Irfft_ComplexOut_AcceptsRealIntoComplexImagZero()
        {
            var a = np.arange(4).astype(np.float64);
            var half = np.fft.rfft(a);                 // complex half-spectrum
            var o = new NDArray(np.complex128, new Shape(4));
            var r = np.fft.irfft(half, n: 4, @out: o);
            ReferenceEquals(r, o).Should().BeTrue();    // returns the out instance
            for (int i = 0; i < 4; i++)
            {
                o.GetAtIndex<Complex>(i).Real.Should().BeApproximately(i, 1e-9);   // reconstructs [0,1,2,3]
                o.GetAtIndex<Complex>(i).Imaginary.Should().Be(0.0);              // imag exactly 0
            }
        }

        [TestMethod]
        public void Irfft_Float32Out_AcceptsNarrowed()
        {
            var a = np.arange(4).astype(np.float64);
            var half = np.fft.rfft(a);
            var o = new NDArray(np.float32, new Shape(4));
            var r = np.fft.irfft(half, n: 4, @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            o.dtype.Should().Be(typeof(float));
            for (int i = 0; i < 4; i++)
                o.GetAtIndex<float>(i).Should().BeApproximately(i, 1e-5f);
        }

        // --- fft/rfft (loop complex128) still ACCEPT a complex out (regression guard) ---
        [TestMethod]
        public void Fft_ComplexOut_Accepts()
        {
            var a = np.arange(4).astype(np.float64);
            var o = new NDArray(np.complex128, new Shape(4));
            var r = np.fft.fft(a, @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            o.GetAtIndex<Complex>(0).Real.Should().BeApproximately(6.0, 1e-9);
        }

        // --- incompatible dtypes raise NumPy's verbatim ufunc-cast message ---
        [TestMethod]
        public void Fft_Float64Out_RaisesUfuncCast()
            => new Action(() => np.fft.fft(np.arange(4).astype(np.float64), @out: new NDArray(np.float64, new Shape(4))))
                .Should().Throw<ArgumentException>()
                .WithMessage("Cannot cast ufunc 'fft' output from dtype('complex128') to dtype('float64') with casting rule 'same_kind'");

        [TestMethod]
        public void Rfft_Float64Out_RaisesUfuncCast_NamesEvenLoop()
            => new Action(() => np.fft.rfft(np.arange(4).astype(np.float64), @out: new NDArray(np.float64, new Shape(3))))
                .Should().Throw<ArgumentException>()
                .WithMessage("Cannot cast ufunc 'rfft_n_even' output*to dtype('float64')*same_kind*");

        [TestMethod]
        public void Irfft_Int64Out_RaisesUfuncCast()
        {
            var half = np.fft.rfft(np.arange(4).astype(np.float64));
            new Action(() => np.fft.irfft(half, n: 4, @out: new NDArray(np.int64, new Shape(4))))
                .Should().Throw<ArgumentException>()
                .WithMessage("Cannot cast ufunc 'irfft' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'");
        }
    }
}
