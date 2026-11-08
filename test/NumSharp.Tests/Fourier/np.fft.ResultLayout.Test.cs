using System;

namespace NumSharp.Tests.Fourier
{
    /// <summary>
    ///     The memory order of an <c>np.fft</c> result: NumPy 2.4.2's <c>_raw_fft</c> (<c>numpy/fft/_pocketfft.py</c>)
    ///     allocates every 1-D pass's output with <c>empty_like(a, shape=..., dtype=...)</c>, whose default <c>order='K'</c>
    ///     keeps the input's memory order, so an F-ordered input gives an F-ordered spectrum, a permuted 3-D input a spectrum
    ///     permuted the same way, a broadcast input an F-ordered one (its stride-0 axis ranks innermost), and the N-D forms
    ///     keep the order through every pass (parity QA-48). The values do not depend on it.
    /// </summary>
    /// <remarks>
    ///     Every expected row is NumPy 2.4.2's <c>(flags.c_contiguous, flags.f_contiguous, shape, strides)</c> for the same
    ///     call (probed on win-amd64). <c>F</c> is <c>np.asfortranarray(np.arange(24.0).reshape(4, 6))</c>, <c>P</c> the
    ///     permuted <c>np.arange(24.0).reshape(2, 3, 4).transpose(2, 0, 1)</c> (strides (8, 96, 32)).
    /// </remarks>
    [TestClass]
    public class FftResultLayoutTests
    {
        /// <summary>The F-ordered (4, 6) float64 input.</summary>
        /// <returns>A fresh F-contiguous array.</returns>
        private static NDArray F() => np.asfortranarray(np.arange(24.0).reshape(4, 6));

        /// <summary>The permuted (4, 2, 3) float64 input: neither C nor F.</summary>
        /// <returns>A transposed view of a fresh C buffer.</returns>
        private static NDArray P() => np.arange(24.0).reshape(2, 3, 4).transpose(new[] { 2, 0, 1 });

        /// <summary>Asserts a result's contiguity flags, shape and byte strides.</summary>
        /// <param name="r">The result.</param>
        /// <param name="c">NumPy's <c>flags.c_contiguous</c>.</param>
        /// <param name="f">NumPy's <c>flags.f_contiguous</c>.</param>
        /// <param name="shape">NumPy's shape.</param>
        /// <param name="strides">NumPy's byte strides.</param>
        private static void Layout(NDArray r, bool c, bool f, long[] shape, long[] strides)
        {
            r.shape.Should().Equal(shape);
            r.strides.Should().Equal(strides);
            r.flags.c_contiguous.Should().Be(c);
            r.flags.f_contiguous.Should().Be(f);
        }

        /// <summary>fft / ifft of an F input along either axis keep F: complex128 (16, 64); with n= the new length too.</summary>
        [TestMethod]
        public void ComplexTransforms_FInput_StayF()
        {
            using var f = F();
            using (var r = np.fft.fft(f, axis: 1)) Layout(r, false, true, new long[] { 4, 6 }, new long[] { 16, 64 });
            using (var r = np.fft.fft(f, axis: 0)) Layout(r, false, true, new long[] { 4, 6 }, new long[] { 16, 64 });
            using (var r = np.fft.ifft(f, axis: 1)) Layout(r, false, true, new long[] { 4, 6 }, new long[] { 16, 64 });
            using (var r = np.fft.fft(f, n: 8, axis: 1)) Layout(r, false, true, new long[] { 4, 8 }, new long[] { 16, 64 });
            using (var r = np.fft.fft(f, n: 3, axis: 0)) Layout(r, false, true, new long[] { 3, 6 }, new long[] { 16, 48 });
        }

        /// <summary>The real transforms change the axis length and keep F: rfft / ihfft (3, 6) at (16, 48), irfft / hfft
        ///     float64 at (8, 32).</summary>
        [TestMethod]
        public void RealTransforms_FInput_StayF()
        {
            using var f = F();
            using (var r = np.fft.rfft(f, axis: 0)) Layout(r, false, true, new long[] { 3, 6 }, new long[] { 16, 48 });
            using (var r = np.fft.ihfft(f, axis: 0)) Layout(r, false, true, new long[] { 3, 6 }, new long[] { 16, 48 });
            using (var half = np.fft.rfft(f, axis: 1))
            using (var r = np.fft.irfft(half, axis: 1))
                Layout(r, false, true, new long[] { 4, 6 }, new long[] { 8, 32 });
            using (var r = np.fft.hfft(f, axis: 1)) Layout(r, false, true, new long[] { 4, 10 }, new long[] { 8, 32 });
        }

        /// <summary>The N-D forms keep the order through every pass: fft2 / rfft2 of F stay F, fftn / rfftn / irfftn of P
        ///     stay permuted like P.</summary>
        [TestMethod]
        public void NDimensionalForms_KeepTheOrderThroughEveryPass()
        {
            using var f = F();
            using var p = P();
            using (var r = np.fft.fft2(f)) Layout(r, false, true, new long[] { 4, 6 }, new long[] { 16, 64 });
            using (var r = np.fft.rfft2(f)) Layout(r, false, true, new long[] { 4, 4 }, new long[] { 16, 64 });
            using (var r = np.fft.fftn(p)) Layout(r, false, false, new long[] { 4, 2, 3 }, new long[] { 16, 192, 64 });
            using (var spectrum = np.fft.rfftn(p))
            {
                Layout(spectrum, false, false, new long[] { 4, 2, 2 }, new long[] { 16, 128, 64 });
                using var back = np.fft.irfftn(spectrum, s: new[] { 4, 2, 3 }, axes: new[] { 0, 1, 2 });
                Layout(back, false, false, new long[] { 4, 2, 3 }, new long[] { 8, 96, 32 });
            }
        }

        /// <summary>A permuted input along one axis, a transposed (F-contiguous) input, a broadcast input (F: its stride-0
        ///     axis ranks innermost) and a C input (C).</summary>
        [TestMethod]
        public void OtherLayouts_FollowEmptyLikesKOrder()
        {
            using var p = P();
            using var t = np.arange(24.0).reshape(6, 4).T;
            using var row = np.arange(6.0);
            using var b = np.broadcast_to(row, new Shape(4, 6));
            using var c = np.arange(24.0).reshape(4, 6);
            using (var r = np.fft.fft(p, axis: 0)) Layout(r, false, false, new long[] { 4, 2, 3 }, new long[] { 16, 192, 64 });
            using (var r = np.fft.fft(t, axis: 1)) Layout(r, false, true, new long[] { 4, 6 }, new long[] { 16, 64 });
            using (var r = np.fft.fft(b, axis: 1)) Layout(r, false, true, new long[] { 4, 6 }, new long[] { 16, 64 });
            using (var r = np.fft.fft(c, axis: 1)) Layout(r, true, false, new long[] { 4, 6 }, new long[] { 96, 16 });
        }

        /// <summary>The layout is not part of the values: an F input's spectrum equals its C copy's, bit for bit
        ///     (NumPy: <c>np.array_equal</c> True).</summary>
        [TestMethod]
        public void Values_DoNotDependOnTheLayout()
        {
            using var f = F();
            using var c = np.ascontiguousarray(f);
            using var fromF = np.fft.fft(f, axis: 1);
            using var fromC = np.fft.fft(c, axis: 1);
            np.array_equal(fromF, fromC).Should().BeTrue();
        }
    }
}
