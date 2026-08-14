using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Fourier
{
    /// <summary>
    ///     Parity gate for the REAL transforms — <c>rfft</c>, <c>irfft</c>, <c>rfft2</c>, <c>irfft2</c>,
    ///     <c>rfftn</c>, <c>irfftn</c> — against NumPy 2.4.2. Every expected value/shape/dtype/error was
    ///     probed from a live NumPy 2.4.2. Focus areas: the <c>rfft</c> output length <c>n//2+1</c>, the
    ///     <c>irfft</c> default output <c>n = 2*(m-1)</c> and explicit <c>n</c>, the FFTPACK half-complex
    ///     packing (verified through the public API for even AND odd <c>n</c>), the N-D decompositions
    ///     (<c>rfftn = rfft(last) + fft(rest)</c>, <c>irfftn = ifft(rest) + irfft(last)</c>), and the
    ///     <c>invreal</c> last-axis <c>2*(m-1)</c> defaulting.
    /// </summary>
    [TestClass]
    public class NpFftRealTest : TestClass
    {
        // ---- helpers -------------------------------------------------------------------------

        private static NDArray Real(params double[] v) => new NDArray(v);
        private static NDArray Cplx(params Complex[] v) => new NDArray(v);

        private static Complex[] FlatC(NDArray r)
        {
            var c = np.ascontiguousarray(r);
            var d = c.Data<Complex>();
            var outp = new Complex[c.size];
            for (int i = 0; i < outp.Length; i++) outp[i] = d[i];
            return outp;
        }

        private static double[] FlatR(NDArray r)
        {
            var c = np.ascontiguousarray(r);
            var d = c.Data<double>();
            var outp = new double[c.size];
            for (int i = 0; i < outp.Length; i++) outp[i] = d[i];
            return outp;
        }

        private static void AssertCloseC(NDArray result, Complex[] expected, double tol = 1e-11)
        {
            result.typecode.Should().Be(NPTypeCode.Complex);
            result.size.Should().Be(expected.Length);
            var flat = FlatC(result);
            for (int i = 0; i < expected.Length; i++)
            {
                flat[i].Real.Should().BeApproximately(expected[i].Real, tol, $"real[{i}]");
                flat[i].Imaginary.Should().BeApproximately(expected[i].Imaginary, tol, $"imag[{i}]");
            }
        }

        private static void AssertCloseR(NDArray result, double[] expected, double tol = 1e-11)
        {
            result.typecode.Should().Be(NPTypeCode.Double);
            result.size.Should().Be(expected.Length);
            var flat = FlatR(result);
            for (int i = 0; i < expected.Length; i++)
                flat[i].Should().BeApproximately(expected[i], tol, $"[{i}]");
        }

        private static void AssertCloseArraysC(NDArray a, NDArray b, double tol = 1e-11)
        {
            a.shape.Should().BeEquivalentTo(b.shape);
            var fa = FlatC(a); var fb = FlatC(b);
            for (int i = 0; i < fa.Length; i++)
            {
                fa[i].Real.Should().BeApproximately(fb[i].Real, tol, $"real[{i}]");
                fa[i].Imaginary.Should().BeApproximately(fb[i].Imaginary, tol, $"imag[{i}]");
            }
        }

        private static void AssertCloseArraysR(NDArray a, NDArray b, double tol = 1e-11)
        {
            a.shape.Should().BeEquivalentTo(b.shape);
            var fa = FlatR(a); var fb = FlatR(b);
            for (int i = 0; i < fa.Length; i++)
                fa[i].Should().BeApproximately(fb[i], tol, $"[{i}]");
        }

        // =====================================================================================
        // 1. rfft — concrete value pins + output length n//2+1 (even AND odd)
        // =====================================================================================

        [TestMethod]
        public void Rfft_EvenLength_MatchesNumpy()
        {
            // np.fft.rfft([1,2,3,4,3,2]) -> length 6//2+1 = 4 ; last term (Nyquist) is REAL.
            AssertCloseC(np.fft.rfft(Real(1, 2, 3, 4, 3, 2)),
                new[]
                {
                    new Complex(15, 0),
                    new Complex(-4, 2.220446049250313e-16),
                    new Complex(0, 2.220446049250313e-16),
                    new Complex(-1, 0)
                });
        }

        [TestMethod]
        public void Rfft_OddLength_MatchesNumpy()
        {
            // np.fft.rfft([1,2,3,4,5]) -> length (5+1)/2 = 3 ; last term is COMPLEX (no Nyquist).
            AssertCloseC(np.fft.rfft(Real(1, 2, 3, 4, 5)),
                new[]
                {
                    new Complex(15, 0),
                    new Complex(-2.5, 3.4409548011779334),
                    new Complex(-2.5, 0.8122992405822659)
                });
        }

        [TestMethod]
        public void Rfft_OutputLength_IsHalfPlusOne()
        {
            // even n -> n/2+1 ; odd n -> (n+1)/2, both == n//2+1
            np.fft.rfft(Real(1, 2, 3, 4)).size.Should().Be(3);        // 4 -> 3
            np.fft.rfft(Real(1, 2, 3, 4, 3, 2)).size.Should().Be(4);  // 6 -> 4
            np.fft.rfft(Real(1, 2, 3, 4, 5)).size.Should().Be(3);     // 5 -> 3
            np.fft.rfft(Real(1, 2, 3, 4, 5, 6, 7)).size.Should().Be(4); // 7 -> 4
        }

        [TestMethod]
        public void Rfft_N_Odd_TruncatesEvenInput_MatchesNumpy()
        {
            // np.fft.rfft([1,2,3,4,3,2], n=5) -> take first 5, length (5+1)/2 = 3
            AssertCloseC(np.fft.rfft(Real(1, 2, 3, 4, 3, 2), n: 5),
                new[]
                {
                    new Complex(13, 0),
                    new Complex(-3.118033988749895, 1.5388417685876266),
                    new Complex(-0.8819660112501051, -0.3632712640026804)
                });
        }

        [TestMethod]
        public void Rfft_N_Pad_MatchesNumpy()
        {
            // zero-padding: np.fft.rfft([1,2,3], n=6) -> length 6//2+1 = 4
            AssertCloseC(np.fft.rfft(Real(1, 2, 3), n: 6),
                new[]
                {
                    new Complex(6.0, 0.0),
                    new Complex(0.4999999999999999, -4.330127018922194),
                    new Complex(-1.5, 0.8660254037844386),
                    new Complex(2.0, 0.0)
                });
        }

        [TestMethod]
        public void Rfft_PrimeLength_Bluestein_MatchesNumpy()
        {
            // n=13 (prime) forces the Bluestein path; pad arange(8) with zeros. length 13//2+1 = 7.
            AssertCloseC(np.fft.rfft(np.arange(8).astype(np.float64), n: 13),
                new[]
                {
                    new Complex(28, 0),
                    new Complex(-15.400021355682835, -11.905180123783321),
                    new Complex(6.065598262451868, 3.7403484872825237),
                    new Complex(-6.233653505680083, -0.2591954576693385),
                    new Complex(3.8170210349709484, -1.8118424509206026),
                    new Complex(-2.8695938204581086, 3.0692382523288315),
                    new Complex(0.6206493843982093, -3.675091918872909)
                }, tol: 1e-10);
        }

        [TestMethod]
        public void Rfft_MatchesFft_HalfSpectrum()
        {
            // rfft(x)[k] == fft(x)[k] for k in 0..n//2 (rfft is the non-negative-freq half of fft).
            var x = Real(1, 2, 3, 4, 3, 2, 5, 1);        // n = 8
            var r = FlatC(np.fft.rfft(x));                // length 5
            var f = FlatC(np.fft.fft(x));                 // length 8
            for (int k = 0; k < r.Length; k++)
            {
                r[k].Real.Should().BeApproximately(f[k].Real, 1e-11, $"real[{k}]");
                r[k].Imaginary.Should().BeApproximately(f[k].Imaginary, 1e-11, $"imag[{k}]");
            }
        }

        [TestMethod]
        public void Rfft_N1_MatchesNumpy()
        {
            // np.fft.rfft([0,1,2,3], n=1) == [0+0j]  (single point, DC = first element)
            AssertCloseC(np.fft.rfft(Real(0, 1, 2, 3), n: 1), new[] { new Complex(0, 0) });
        }

        // =====================================================================================
        // 2. rfft — normalization modes
        // =====================================================================================

        [TestMethod]
        public void Rfft_Norm_Ortho_MatchesNumpy()
        {
            AssertCloseC(np.fft.rfft(Real(1, 2, 3, 4, 3, 2), norm: "ortho"),
                new[]
                {
                    new Complex(6.123724356957946, 0),
                    new Complex(-1.6329931618554523, 9.06493303673679e-17),
                    new Complex(0, 9.06493303673679e-17),
                    new Complex(-0.4082482904638631, 0)
                });
        }

        [TestMethod]
        public void Rfft_Norm_Forward_MatchesNumpy()
        {
            AssertCloseC(np.fft.rfft(Real(1, 2, 3, 4, 3, 2), norm: "forward"),
                new[]
                {
                    new Complex(2.5, 0),
                    new Complex(-0.6666666666666666, 3.700743415417188e-17),
                    new Complex(0, 3.700743415417188e-17),
                    new Complex(-0.16666666666666666, 0)
                });
        }

        [TestMethod]
        public void Rfft_Norm_Null_Equals_Backward()
        {
            AssertCloseArraysC(np.fft.rfft(Real(1, 2, 3, 4, 3, 2)),
                               np.fft.rfft(Real(1, 2, 3, 4, 3, 2), norm: "backward"));
        }

        // =====================================================================================
        // 3. irfft — default output n = 2*(m-1), explicit n, even/odd, imaginary-ignore
        // =====================================================================================

        [TestMethod]
        public void Irfft_DefaultLength_Is2mMinus1_MatchesNumpy()
        {
            // rfft([1,2,3,4,3,2]) has m=4 -> irfft default n = 2*(4-1) = 6, recovers the signal.
            var r = np.fft.rfft(Real(1, 2, 3, 4, 3, 2));
            var y = np.fft.irfft(r);
            y.size.Should().Be(6);
            AssertCloseR(y, new double[] { 1, 2, 3, 4, 3, 2 }, 1e-9);
        }

        [TestMethod]
        public void Irfft_ExplicitN_Shorter_MatchesNumpy()
        {
            // np.fft.irfft(rfft([1,2,3,4,3,2]), n=5)
            var r = np.fft.rfft(Real(1, 2, 3, 4, 3, 2));
            AssertCloseR(np.fft.irfft(r, n: 5),
                new[] { 1.4000000000000001, 2.505572809000084, 4.294427190999916, 4.294427190999916, 2.505572809000084 });
        }

        [TestMethod]
        public void Irfft_ExplicitN_Longer_MatchesNumpy()
        {
            // np.fft.irfft(rfft([1,2,3,4,3,2]), n=7) — odd output length (must be specified)
            var r = np.fft.rfft(Real(1, 2, 3, 4, 3, 2));
            AssertCloseR(np.fft.irfft(r, n: 7),
                new[]
                {
                    0.7142857142857142, 1.687717045847853, 2.2190268382761498,
                    3.23611325873314, 3.23611325873314, 2.2190268382761498, 1.687717045847853
                });
        }

        [TestMethod]
        public void Irfft_OddInput_DefaultLength_MatchesNumpy()
        {
            // rfft([1,2,3,4,5]) has m=3 -> irfft default n = 2*(3-1) = 4 (even, NOT the original 5!)
            var r = np.fft.rfft(Real(1, 2, 3, 4, 5));
            var y = np.fft.irfft(r);
            y.size.Should().Be(4);
            AssertCloseR(y, new[] { 1.875, 2.6545225994110333, 4.375, 6.095477400588967 });
        }

        [TestMethod]
        public void Irfft_OddInput_ExplicitOddN_RecoversSignal()
        {
            // to recover the 5-point signal, n=5 MUST be passed (odd output).
            var r = np.fft.rfft(Real(1, 2, 3, 4, 5));
            AssertCloseR(np.fft.irfft(r, n: 5), new double[] { 1, 2, 3, 4, 5 }, 1e-9);
        }

        [TestMethod]
        public void Irfft_IgnoresImaginary_OfDcAndNyquist()
        {
            // Even output (default n=4 here): irfft treats a[0] (DC) and a[-1] (Nyquist) as REAL,
            // silently dropping their imaginary parts — matching NumPy.
            var full = Cplx(new Complex(3, 9), new Complex(1, 1), new Complex(2, -2));   // m=3 -> n=4
            var expected = new[] { 1.75, -0.25, 0.75, 0.75 };
            AssertCloseR(np.fft.irfft(full), expected);
            // Zeroing the DC imaginary yields the identical result.
            AssertCloseR(np.fft.irfft(Cplx(new Complex(3, 0), new Complex(1, 1), new Complex(2, -2))), expected);
            // Zeroing the Nyquist imaginary too yields the identical result.
            AssertCloseR(np.fft.irfft(Cplx(new Complex(3, 9), new Complex(1, 1), new Complex(2, 0))), expected);
        }

        [TestMethod]
        public void Irfft_OddN_UsesFullLastTerm()
        {
            // With an odd n there is no Nyquist term, so the last input term keeps its imaginary part.
            var full = Cplx(new Complex(3, 9), new Complex(1, 1), new Complex(2, -2));
            AssertCloseR(np.fft.irfft(full, n: 5),
                new[] { 1.8, 0.16619879756593814, -0.4723525162031331, 1.5195661117030912, -0.013412393065896078 });
        }

        [TestMethod]
        public void Irfft_N1_MatchesNumpy()
        {
            // np.fft.irfft([5+0j, 1+1j], n=1) == [5.] (single DC point)
            AssertCloseR(np.fft.irfft(Cplx(new Complex(5, 0), new Complex(1, 1)), n: 1), new[] { 5.0 });
        }

        // =====================================================================================
        // 4. irfft — normalization modes
        // =====================================================================================

        [TestMethod]
        public void Irfft_Norm_Ortho_MatchesNumpy()
        {
            var r = np.fft.rfft(Real(1, 2, 3, 4, 3, 2));
            AssertCloseR(np.fft.irfft(r, norm: "ortho"),
                new[]
                {
                    2.4494897427831783, 4.898979485566357, 7.348469228349535,
                    9.797958971132713, 7.348469228349535, 4.898979485566357
                });
        }

        [TestMethod]
        public void Irfft_Norm_Forward_MatchesNumpy()
        {
            var r = np.fft.rfft(Real(1, 2, 3, 4, 3, 2));
            AssertCloseR(np.fft.irfft(r, norm: "forward"),
                new double[] { 6, 12, 18, 24, 18, 12 });
        }

        // =====================================================================================
        // 5. rfft/irfft round-trips (even AND odd), all norms
        // =====================================================================================

        [TestMethod]
        public void Rfft_Irfft_RoundTrip_Even()
        {
            var x = Real(1, 2, 3, 4, 3, 2, 5, 1);   // n = 8 (even)
            var back = np.fft.irfft(np.fft.rfft(x), n: 8);
            AssertCloseR(back, FlatR(x), 1e-9);
        }

        [TestMethod]
        public void Rfft_Irfft_RoundTrip_Odd()
        {
            var x = Real(1, 2, 3, 4, 3, 2, 5);      // n = 7 (odd) -> n MUST be passed back
            var back = np.fft.irfft(np.fft.rfft(x), n: 7);
            AssertCloseR(back, FlatR(x), 1e-9);
        }

        [TestMethod]
        public void Rfft_Irfft_RoundTrip_AllNorms()
        {
            var x = Real(2, 5, 1, 8, 3, 9);
            foreach (var norm in new[] { null, "backward", "ortho", "forward" })
            {
                var back = np.fft.irfft(np.fft.rfft(x, norm: norm), n: 6, norm: norm);
                AssertCloseR(back, FlatR(x), 1e-9);
            }
        }

        // =====================================================================================
        // 6. dtype policy (int/bool -> complex128; float32 -> complex128 [Misaligned]; irfft -> double)
        // =====================================================================================

        [TestMethod]
        public void Rfft_IntInput_PromotesToComplex128()
        {
            var r = np.fft.rfft(np.arange(4).astype(np.int32));
            r.typecode.Should().Be(NPTypeCode.Complex);
            AssertCloseC(r, new[] { new Complex(6, 0), new Complex(-2, 2), new Complex(-2, 0) });
        }

        [TestMethod]
        [Misaligned]
        public void Rfft_Float32Input_PromotesToComplex128_NumpyGivesComplex64()
        {
            // NumSharp has only complex128, so float32 -> complex128 (compute-in-double).
            // NumPy 2.x returns complex64 here. Values are the correctly-rounded double result.
            var r = np.fft.rfft(np.arange(4).astype(np.float32));
            r.typecode.Should().Be(NPTypeCode.Complex);   // NumPy: complex64
            AssertCloseArraysC(r, np.fft.rfft(np.arange(4).astype(np.float64)));
        }

        [TestMethod]
        public void Irfft_RealOutput_IsFloat64()
        {
            var r = np.fft.rfft(Real(1, 2, 3, 4));
            np.fft.irfft(r).typecode.Should().Be(NPTypeCode.Double);
            // irfft accepts a real (int/float) input too (promoted to complex internally) -> float64 out.
            np.fft.irfft(np.arange(4)).typecode.Should().Be(NPTypeCode.Double);
        }

        // =====================================================================================
        // 7. Memory-layout DOD — the driver reads through strides (compare to a contiguous copy)
        // =====================================================================================

        [TestMethod]
        public void Rfft_Transposed_Axis0_EqualsContiguousCopy()
        {
            var a = np.arange(12).astype(np.float64).reshape(3, 4).T;   // (4,3) F-contiguous view
            AssertCloseArraysC(np.fft.rfft(a, axis: 0), np.fft.rfft(a.copy(), axis: 0));
        }

        [TestMethod]
        public void Rfft_ColumnStrided_Axis0_EqualsContiguousCopy()
        {
            var view = np.arange(12).astype(np.float64).reshape(3, 4)[":, ::2"];  // (3,2) col-strided
            AssertCloseArraysC(np.fft.rfft(view, axis: 0), np.fft.rfft(view.copy(), axis: 0));
        }

        [TestMethod]
        public void Rfft_Reversed_EqualsContiguousCopy()
        {
            var a = np.arange(8).astype(np.float64)["::-1"];   // negative-stride view
            AssertCloseArraysC(np.fft.rfft(a), np.fft.rfft(a.copy()));
        }

        [TestMethod]
        public void Rfft_BroadcastRead_EqualsContiguousCopy()
        {
            var a = np.broadcast_to(np.arange(4).astype(np.float64), new Shape(3, 4));  // read-only broadcast
            AssertCloseArraysC(np.fft.rfft(a), np.fft.rfft(a.copy()));
        }

        [TestMethod]
        public void Irfft_Strided_EqualsContiguousCopy()
        {
            // strided complex half-spectrum input on axis 0.
            var full = np.fft.rfftn(np.arange(24).astype(np.float64).reshape(2, 3, 4));  // (2,3,3) complex
            var view = full["::-1"];   // reverse axis 0
            AssertCloseArraysR(np.fft.irfft(view, axis: 2), np.fft.irfft(view.copy(), axis: 2));
        }

        // =====================================================================================
        // 8. rfft2 / rfftn — shapes, values, decomposition
        // =====================================================================================

        [TestMethod]
        public void Rfft2_MatchesNumpy()
        {
            // np.fft.rfft2(np.mgrid[:5,:5][0]) : each row = its row index. shape (5,3).
            var mg = new NDArray(new double[]
            {
                0, 0, 0, 0, 0,
                1, 1, 1, 1, 1,
                2, 2, 2, 2, 2,
                3, 3, 3, 3, 3,
                4, 4, 4, 4, 4
            }).reshape(5, 5);
            AssertCloseC(np.fft.rfft2(mg),
                new[]
                {
                    new Complex(50, 0), new Complex(0, 0), new Complex(0, 0),
                    new Complex(-12.5, 17.204774005889668), new Complex(0, 0), new Complex(0, 0),
                    new Complex(-12.5, 4.0614962029113295), new Complex(0, 0), new Complex(0, 0),
                    new Complex(-12.5, -4.0614962029113295), new Complex(0, 0), new Complex(0, 0),
                    new Complex(-12.5, -17.204774005889668), new Complex(0, 0), new Complex(0, 0)
                }, tol: 1e-10);
        }

        [TestMethod]
        public void Rfftn_3D_MatchesNumpy()
        {
            // np.fft.rfftn(np.arange(8).reshape(2,2,2)) — last axis 2->2, all real DC-heavy.
            var a = np.arange(8).astype(np.float64).reshape(2, 2, 2);
            AssertCloseC(np.fft.rfftn(a),
                new[]
                {
                    new Complex(28, 0), new Complex(-4, 0),
                    new Complex(-8, 0), new Complex(0, 0),
                    new Complex(-16, 0), new Complex(0, 0),
                    new Complex(0, 0), new Complex(0, 0)
                });
        }

        [TestMethod]
        public void Rfftn_LastAxisReduced_ToHalfPlusOne()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            // default: rfft over last axis (4 -> 3), fft over the rest (kept). shape (2,3,3).
            np.fft.rfftn(a).shape.Should().BeEquivalentTo(new[] { 2, 3, 3 });
            np.fft.rfft2(a).shape.Should().BeEquivalentTo(new[] { 2, 3, 3 });   // rfft2 == rfftn axes (-2,-1)
        }

        [TestMethod]
        public void Rfftn_EqualsRfftLastThenFftRest()
        {
            // rfftn(a) == fft(fft(rfft(a, axis=2), axis=1), axis=0) — the documented decomposition.
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            var manual = np.fft.fft(np.fft.fft(np.fft.rfft(a, axis: 2), axis: 1), axis: 0);
            AssertCloseArraysC(np.fft.rfftn(a), manual);
        }

        [TestMethod]
        public void Rfft2_DefaultAxes_AreLastTwo()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            AssertCloseArraysC(np.fft.rfft2(a), np.fft.rfftn(a, axes: new[] { 1, 2 }));
        }

        [TestMethod]
        public void Rfftn_NegativeAxes_MatchExplicit()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            AssertCloseArraysC(np.fft.rfftn(a, axes: new[] { -1, -3 }), np.fft.rfftn(a, axes: new[] { 2, 0 }));
        }

        [TestMethod]
        public void Rfftn_1D_EqualsRfft()
        {
            // rfftn over a 1-D array (all axes = axis 0) == rfft.
            var x = np.arange(4).astype(np.float64);
            AssertCloseArraysC(np.fft.rfftn(x), np.fft.rfft(x));
        }

        [TestMethod]
        public void Rfftn_S_WithoutAxes_UsesLastLenSAxes()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            // s=[2,2] no axes -> last two axes ; last axis (4->2) reduced to 2//2+1 = 2. shape (2,2,2).
            np.fft.rfftn(a, s: new[] { 2, 2 }).shape.Should().BeEquivalentTo(new[] { 2, 2, 2 });
        }

        [TestMethod]
        public void Rfftn_S_Minus1_UsesFullLength()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            // s=[-1,-1]: keep full axes; last axis 4 -> 3. Equal to default rfftn.
            AssertCloseArraysC(np.fft.rfftn(a, s: new[] { -1, -1 }, axes: new[] { 1, 2 }), np.fft.rfft2(a));
        }

        // =====================================================================================
        // 9. irfft2 / irfftn — shapes, round-trips, invreal 2*(m-1) default
        // =====================================================================================

        [TestMethod]
        public void Irfftn_Rfftn_RoundTrip_3D()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            var back = np.fft.irfftn(np.fft.rfftn(a));
            back.shape.Should().BeEquivalentTo(new[] { 2, 3, 4 });
            AssertCloseArraysR(back, a, 1e-9);
        }

        [TestMethod]
        public void Irfft2_Rfft2_RoundTrip()
        {
            var a = np.arange(20).astype(np.float64).reshape(4, 5);
            var back = np.fft.irfft2(np.fft.rfft2(a), s: new[] { 4, 5 });
            back.shape.Should().BeEquivalentTo(new[] { 4, 5 });
            AssertCloseArraysR(back, a, 1e-9);
        }

        [TestMethod]
        public void Irfftn_DefaultLastAxis_Is2mMinus1()
        {
            // The last-axis default output length comes from _cook_nd_args invreal: 2*(m-1).
            // rfftn(a) last axis is 3 (from 4) -> irfftn default recovers 2*(3-1) = 4.
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            var r = np.fft.rfftn(a);                   // (2,3,3)
            var back = np.fft.irfftn(r);               // last axis 3 -> 2*(3-1) = 4
            back.shape.Should().BeEquivalentTo(new[] { 2, 3, 4 });
        }

        [TestMethod]
        public void Irfftn_EqualsIfftRestThenIrfftLast()
        {
            // irfftn(a) == irfft(ifft(a, axis=0), axis=1) for a 2-axis transform.
            var a = np.fft.rfftn(np.arange(20).astype(np.float64).reshape(4, 5));  // (4,3) complex
            var manual = np.fft.irfft(np.fft.ifft(a, axis: 0), n: 5, axis: 1);
            AssertCloseArraysR(np.fft.irfftn(a, s: new[] { 4, 5 }), manual, 1e-9);
        }

        [TestMethod]
        public void Irfft2_MatchesNumpy_Mgrid()
        {
            // irfft2(rfft2(mgrid[:5,:5][0]), s=(5,5)) recovers the row-index matrix (docstring example).
            var mg = new NDArray(new double[]
            {
                0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4
            }).reshape(5, 5);
            var back = np.fft.irfft2(np.fft.rfft2(mg), s: new[] { 5, 5 });
            AssertCloseArraysR(back, mg, 1e-9);
        }

        // =====================================================================================
        // 10. out= — same instance, write-through; irfft2 ignores out (NumPy passes out=None)
        // =====================================================================================

        [TestMethod]
        public void Rfft_Out_SameInstance_WritesThrough()
        {
            var x = np.arange(6).astype(np.float64);
            var o = new NDArray(np.complex128, new Shape(4));   // 6//2+1 = 4
            var r = np.fft.rfft(x, @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            AssertCloseArraysC(o, np.fft.rfft(x));
        }

        [TestMethod]
        public void Irfft_Out_SameInstance_WritesThrough()
        {
            var rf = np.fft.rfft(np.arange(6).astype(np.float64));
            var o = new NDArray(np.float64, new Shape(6));
            var r = np.fft.irfft(rf, @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            AssertCloseArraysR(o, np.fft.irfft(rf));
        }

        [TestMethod]
        public void Rfftn_Out_SameInstance()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            var o = new NDArray(np.complex128, new Shape(2, 3, 3));
            var r = np.fft.rfftn(a, @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            AssertCloseArraysC(o, np.fft.rfftn(a));
        }

        [TestMethod]
        public void Rfft2_Out_SameInstance()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            var o = new NDArray(np.complex128, new Shape(2, 3, 3));
            var r = np.fft.rfft2(a, @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            AssertCloseArraysC(o, np.fft.rfft2(a));
        }

        [TestMethod]
        public void Irfftn_Out_SameInstance()
        {
            var rn = np.fft.rfftn(np.arange(24).astype(np.float64).reshape(2, 3, 4));
            var o = new NDArray(np.float64, new Shape(2, 3, 4));
            var r = np.fft.irfftn(rn, @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            AssertCloseArraysR(o, np.fft.irfftn(rn), 1e-9);
        }

        [TestMethod]
        public void Irfft2_Out_IsIgnored_LikeNumpy()
        {
            // NumPy's irfft2 hard-codes out=None: the provided array is NOT written and NOT returned.
            var rn = np.fft.rfftn(np.arange(24).astype(np.float64).reshape(2, 3, 4));
            var o = np.zeros(new Shape(2, 3, 4)).astype(np.float64);
            var r = np.fft.irfft2(rn, @out: o);
            ReferenceEquals(r, o).Should().BeFalse();
            foreach (var z in o.Data<double>()) z.Should().Be(0);   // untouched
        }

        [TestMethod]
        public void Rfft_Out_WrongShape_Raises()
        {
            Action act = () => np.fft.rfft(np.arange(6).astype(np.float64), @out: new NDArray(np.complex128, new Shape(3)));
            act.Should().Throw<ValueError>().WithMessage("output array has wrong shape.");
        }

        [TestMethod]
        public void Irfft_Out_WrongShape_Raises()
        {
            var rf = np.fft.rfft(np.arange(6).astype(np.float64));
            Action act = () => np.fft.irfft(rf, @out: new NDArray(np.float64, new Shape(5)));
            act.Should().Throw<ValueError>().WithMessage("output array has wrong shape.");
        }

        // =====================================================================================
        // 11. error taxonomy (probed from NumPy 2.4.2)
        // =====================================================================================

        [TestMethod]
        public void Rfft_NZero_Raises()
        {
            Action act = () => np.fft.rfft(Real(1, 2, 3, 4), n: 0);
            act.Should().Throw<ValueError>().WithMessage("Invalid number of FFT data points (0) specified.");
        }

        [TestMethod]
        public void Irfft_NZero_Raises()
        {
            Action act = () => np.fft.irfft(Cplx(new Complex(1, 0), new Complex(2, 0)), n: 0);
            act.Should().Throw<ValueError>().WithMessage("Invalid number of FFT data points (0) specified.");
        }

        [TestMethod]
        public void Irfft_M1_DefaultN_Raises()
        {
            // m=1 -> default n = 2*(1-1) = 0 -> ValueError.
            Action act = () => np.fft.irfft(Cplx(new Complex(5, 0)));
            act.Should().Throw<ValueError>().WithMessage("Invalid number of FFT data points (0) specified.");
        }

        [TestMethod]
        public void Rfft_BadNorm_Forward_NoSpaceMessage()
        {
            Action act = () => np.fft.rfft(Real(1, 2, 3, 4), norm: "bad");
            act.Should().Throw<ValueError>()
                .WithMessage("Invalid norm value bad; should be \"backward\",\"ortho\" or \"forward\".");
        }

        [TestMethod]
        public void Irfft_BadNorm_Inverse_WithSpaceMessage()
        {
            Action act = () => np.fft.irfft(Cplx(new Complex(1, 0), new Complex(2, 0)), norm: "bad");
            act.Should().Throw<ValueError>()
                .WithMessage("Invalid norm value bad; should be \"backward\", \"ortho\" or \"forward\".");
        }

        [TestMethod]
        public void Rfft_AxisOob_NDefault_IndexError()
        {
            Action act = () => np.fft.rfft(Real(1, 2, 3, 4), axis: 5);
            act.Should().Throw<IndexError>().WithMessage("tuple index out of range");
        }

        [TestMethod]
        public void Rfft_AxisOob_NGiven_AxisError()
        {
            Action act = () => np.fft.rfft(Real(1, 2, 3, 4), n: 4, axis: 5);
            act.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 1*");
        }

        [TestMethod]
        public void Rfft_ZeroD_IndexError()
        {
            Action act = () => np.fft.rfft(NDArray.Scalar(5.0));
            act.Should().Throw<IndexError>().WithMessage("tuple index out of range");
        }

        [TestMethod]
        public void Irfft_ZeroD_IndexError()
        {
            Action act = () => np.fft.irfft(NDArray.Scalar(new Complex(5, 0)));
            act.Should().Throw<IndexError>().WithMessage("tuple index out of range");
        }

        [TestMethod]
        public void Rfftn_SAxesLengthMismatch_Raises()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            Action act = () => np.fft.rfftn(a, s: new[] { 2, 3 }, axes: new[] { 0, 1, 2 });
            act.Should().Throw<ValueError>().WithMessage("Shape and axes have different lengths.");
        }

        [TestMethod]
        public void Rfft2_SAxesLengthMismatch_Raises()
        {
            var a = np.arange(60).astype(np.float64).reshape(3, 4, 5);
            Action act = () => np.fft.rfft2(a, s: new[] { 2, 3, 4 });   // s len 3, default axes (-2,-1) len 2
            act.Should().Throw<ValueError>().WithMessage("Shape and axes have different lengths.");
        }

        [TestMethod]
        public void Rfft2_1D_Input_Raises_IndexError()
        {
            // rfft2 default axes=(-2,-1): take axis -2 of a 1-D array is out of bounds.
            Action act = () => np.fft.rfft2(np.arange(4).astype(np.float64));
            act.Should().Throw<IndexError>().WithMessage("index -2 is out of bounds for axis 0 with size 1");
        }

        // ---- empty / 0-d axes -> IndexError("list index out of range") (fixed guard) ----

        [TestMethod]
        public void Rfftn_EmptyAxes_Raises_IndexError()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            Action act = () => np.fft.rfftn(a, axes: new int[0]);
            act.Should().Throw<IndexError>().WithMessage("list index out of range");
        }

        [TestMethod]
        public void Rfft2_EmptyAxes_Raises_IndexError()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            Action act = () => np.fft.rfft2(a, axes: new int[0]);
            act.Should().Throw<IndexError>().WithMessage("list index out of range");
        }

        [TestMethod]
        public void Rfftn_ZeroD_Raises_IndexError()
        {
            Action act = () => np.fft.rfftn(NDArray.Scalar(5.0));
            act.Should().Throw<IndexError>().WithMessage("list index out of range");
        }

        [TestMethod]
        public void Irfftn_EmptySAndAxes_Raises_IndexError()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            Action act = () => np.fft.irfftn(a, s: new int[0], axes: new int[0]);
            act.Should().Throw<IndexError>().WithMessage("list index out of range");
        }

        // =====================================================================================
        // 12. zero-length dimensions
        // =====================================================================================

        [TestMethod]
        public void Rfft_EmptyNonAxisDim_ReturnsEmpty()
        {
            // shape (3,0), transform axis 0 (len 3): 0 lanes -> empty complex result, axis 0 -> 3//2+1 = 2.
            var z = np.zeros(new Shape(3, 0)).astype(np.float64);
            var r = np.fft.rfft(z, axis: 0);
            r.shape.Should().BeEquivalentTo(new[] { 2, 0 });
            r.typecode.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Rfft_ZeroLengthAxis_Raises()
        {
            var z = np.zeros(new Shape(3, 0)).astype(np.float64);
            Action act = () => np.fft.rfft(z, axis: 1);   // n defaults to 0
            act.Should().Throw<ValueError>().WithMessage("Invalid number of FFT data points (0) specified.");
        }
    }
}
