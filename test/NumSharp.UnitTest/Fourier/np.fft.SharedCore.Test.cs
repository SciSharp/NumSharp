using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Fourier
{
    /// <summary>
    ///     Central shared-core parity fixes surfaced by the per-group /np-function reviews and applied
    ///     at the RawFft seam / CookNdArgs (not owned by any single group). All messages/types verified
    ///     verbatim against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class NpFftSharedCoreTest : TestClass
    {
        private static NDArray C(int n) => np.ones(new Shape(n)).astype(np.complex128);

        // --- rfft has no complex loop: NumPy rejects complex input (parity: rejects, not silently
        //     computes on the real part). The ufunc name follows n parity. ihfft rejects transitively.
        [TestMethod]
        public void Rfft_ComplexInput_EvenN_RaisesTypeError()
            => new Action(() => np.fft.rfft(C(4))).Should().Throw<TypeError>()
                .WithMessage("ufunc 'rfft_n_even' not supported for the input types*safely coerced*''safe''");

        [TestMethod]
        public void Rfft_ComplexInput_OddN_RaisesTypeError()
            => new Action(() => np.fft.rfft(C(5))).Should().Throw<TypeError>()
                .WithMessage("ufunc 'rfft_n_odd' not supported for the input types*");

        [TestMethod]
        public void Rfftn_ComplexInput_RaisesTypeError()
            => new Action(() => np.fft.rfftn(C(4).reshape(2, 2))).Should().Throw<TypeError>();

        [TestMethod]
        public void Ihfft_ComplexInput_RaisesTypeError()
            => new Action(() => np.fft.ihfft(C(4))).Should().Throw<TypeError>()
                .WithMessage("ufunc 'rfft_n_even' not supported*");

        // --- 0-d input to the 2-D forms: NumPy's np.take(shape, axes) on an empty shape.
        [TestMethod]
        public void Fft2_ScalarInput_RaisesTakeFromEmptyAxes()
            => new Action(() => np.fft.fft2(np.array(3.0))).Should().Throw<IndexError>()
                .WithMessage("cannot do a non-empty take from an empty axes.");

        [TestMethod]
        public void Rfft2_ScalarInput_RaisesTakeFromEmptyAxes()
            => new Action(() => np.fft.rfft2(np.array(3.0))).Should().Throw<IndexError>()
                .WithMessage("cannot do a non-empty take from an empty axes.");

        // --- invreal s-defaulting indexes s[-1]/axes[-1]: NumPy leaks a plain IndexError on 0-d.
        [TestMethod]
        public void Irfftn_ScalarInput_RaisesListIndexOutOfRange()
            => new Action(() => np.fft.irfftn(np.array(3.0))).Should().Throw<IndexError>()
                .WithMessage("list index out of range");

        // irfft2 carries explicit axes=(-2,-1), so a 0-d input hits np.take-from-empty (like rfft2/fft2),
        // NOT the axes-less s[-1] indexing that gives irfftn's "list index out of range". Both match NumPy.
        [TestMethod]
        public void Irfft2_ScalarInput_RaisesTakeFromEmptyAxes()
            => new Action(() => np.fft.irfft2(np.array(3.0))).Should().Throw<IndexError>()
                .WithMessage("cannot do a non-empty take from an empty axes.");

        // --- fft/ifft/irfft still ACCEPT complex input (regression guard for the rfft-only rejection).
        [TestMethod]
        public void Fft_ComplexInput_Computes()
            => new Action(() => np.fft.fft(C(4))).Should().NotThrow();

        [TestMethod]
        public void Irfft_ComplexInput_Computes()
            => new Action(() => np.fft.irfft(C(3), n: 4)).Should().NotThrow();
    }
}
