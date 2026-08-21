using System;
using System.Numerics;

namespace NumSharp.Tests.Fourier
{
    /// <summary>
    ///     Parity gate for the HERMITIAN transforms — <c>hfft</c>, <c>ihfft</c> — against NumPy 2.4.2.
    ///     Both are pure compositions (numpy/fft/_pocketfft.py):
    ///     <c>hfft(a,n,axis,norm) = irfft(conjugate(a), n, axis, _swap_direction(norm))</c> (out= IGNORED),
    ///     <c>ihfft(a,n,axis,norm) = conjugate(rfft(a, n, axis, _swap_direction(norm)), out)</c> (out= USED).
    ///     Every expected value/shape/dtype/error was probed from a live NumPy 2.4.2. Focus areas: the
    ///     double norm-swap, the <c>n</c> defaulting (<c>hfft</c>: <c>2*(m-1)</c>; <c>ihfft</c>: <c>m</c>),
    ///     the real (float64) <c>hfft</c> output vs the complex128 <c>ihfft</c> output, the out= asymmetry,
    ///     and the error taxonomy (IndexError when <c>n</c> defaults on a bad axis vs AxisError when <c>n</c>
    ///     is given; the WITH-space bad-norm message from <c>_swap_direction</c>).
    /// </summary>
    [TestClass]
    public class NpFftHermitianTest : TestClass
    {
        // ---- helpers (mirror NpFftRealTest) --------------------------------------------------

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

        private static void AssertCloseR(NDArray result, double[] expected, double tol = 1e-11)
        {
            result.typecode.Should().Be(NPTypeCode.Double);
            result.size.Should().Be(expected.Length);
            var flat = FlatR(result);
            for (int i = 0; i < expected.Length; i++)
                flat[i].Should().BeApproximately(expected[i], tol, $"[{i}]");
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

        private static void AssertCloseArraysR(NDArray a, NDArray b, double tol = 1e-11)
        {
            a.shape.Should().BeEquivalentTo(b.shape);
            var fa = FlatR(a); var fb = FlatR(b);
            for (int i = 0; i < fa.Length; i++)
                fa[i].Should().BeApproximately(fb[i], tol, $"[{i}]");
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

        // =====================================================================================
        // 1. hfft — value pins (real output; default n = 2*(m-1), explicit n, complex input)
        // =====================================================================================

        [TestMethod]
        public void Hfft_DefaultLength_Is2mMinus1_MatchesNumpy()
        {
            // np.fft.hfft([1,2,3,4]) : m=4 -> n = 2*(4-1) = 6, real output.
            AssertCloseR(np.fft.hfft(Real(1, 2, 3, 4)),
                new double[] { 15, -4, 0, -1, 0, -4 });
        }

        [TestMethod]
        public void Hfft_ExplicitN_TruncatesInput_MatchesNumpy()
        {
            // Docstring example: hfft(signal, 6) with signal=[1,2,3,4,3,2] -> only n//2+1=4 inputs used.
            AssertCloseR(np.fft.hfft(Real(1, 2, 3, 4, 3, 2), n: 6),
                new double[] { 15, -4, 0, -1, 0, -4 });
        }

        [TestMethod]
        public void Hfft_OddN_MatchesNumpy()
        {
            // np.fft.hfft([1,2,3,4], 7) -> odd output length (must be specified).
            AssertCloseR(np.fft.hfft(Real(1, 2, 3, 4), n: 7),
                new[]
                {
                    19.0, -5.048917339522305, -0.3079785283699037, -0.6431041321077902,
                    -0.6431041321077902, -0.3079785283699037, -5.048917339522305
                });
        }

        [TestMethod]
        public void Hfft_ComplexInput_MatchesNumpy()
        {
            // hfft conjugates its input first; complex is legal. m=3 -> default n=4.
            AssertCloseR(np.fft.hfft(Cplx(new Complex(1, 2), new Complex(3, -4), new Complex(5, 0))),
                new double[] { 12, -12, 0, 4 });
        }

        [TestMethod]
        public void Hfft_2D_DocstringExample_MatchesNumpy()
        {
            // np.fft.hfft([[1, 1j], [-1j, 2]]) == [[1, 1], [2, -2]] (docstring).
            var sig = Cplx(new Complex(1, 0), new Complex(0, 1), new Complex(0, -1), new Complex(2, 0)).reshape(2, 2);
            var r = np.fft.hfft(sig);
            r.shape.Should().BeEquivalentTo(new[] { 2, 2 });
            AssertCloseR(r, new double[] { 1, 1, 2, -2 });
        }

        [TestMethod]
        public void Hfft_IntInput_PromotesOutputToFloat64()
        {
            var r = np.fft.hfft(new NDArray(new[] { 1, 2, 3, 4 }));
            r.typecode.Should().Be(NPTypeCode.Double);
            AssertCloseR(r, new double[] { 15, -4, 0, -1, 0, -4 });
        }

        [TestMethod]
        public void Hfft_BoolInput_PromotesOutputToFloat64()
        {
            // np.array([1,2,3,4],dtype=bool) == [T,T,T,T]; hfft -> [6,0,0,0,0,0].
            var r = np.fft.hfft(new NDArray(new[] { true, true, true, true }));
            r.typecode.Should().Be(NPTypeCode.Double);
            AssertCloseR(r, new double[] { 6, 0, 0, 0, 0, 0 });
        }

        // =====================================================================================
        // 2. hfft — normalization modes
        // =====================================================================================

        [TestMethod]
        public void Hfft_Norm_Ortho_MatchesNumpy()
        {
            AssertCloseR(np.fft.hfft(Real(1, 2, 3, 4), norm: "ortho"),
                new[]
                {
                    6.123724356957946, -1.6329931618554523, 0.0,
                    -0.4082482904638631, 0.0, -1.6329931618554523
                });
        }

        [TestMethod]
        public void Hfft_Norm_Forward_MatchesNumpy()
        {
            AssertCloseR(np.fft.hfft(Real(1, 2, 3, 4), norm: "forward"),
                new[]
                {
                    2.5, -0.6666666666666666, 0.0,
                    -0.16666666666666666, 0.0, -0.6666666666666666
                });
        }

        [TestMethod]
        public void Hfft_Norm_Null_Equals_Backward()
        {
            AssertCloseArraysR(np.fft.hfft(Real(1, 2, 3, 4)),
                               np.fft.hfft(Real(1, 2, 3, 4), norm: "backward"));
        }

        // =====================================================================================
        // 3. hfft — n truncation / zero-padding
        // =====================================================================================

        [TestMethod]
        public void Hfft_N_Truncate_MatchesNumpy()
        {
            // np.fft.hfft([1,2,3,4,5], 3) -> uses 3//2+1 = 2 inputs.
            AssertCloseR(np.fft.hfft(Real(1, 2, 3, 4, 5), n: 3), new double[] { 5, -1, -1 });
        }

        [TestMethod]
        public void Hfft_N_Pad_MatchesNumpy()
        {
            // np.fft.hfft([1,2,3], 8) -> zero-pads the half-spectrum.
            AssertCloseR(np.fft.hfft(Real(1, 2, 3), n: 8),
                new[]
                {
                    11.0, 3.8284271247461903, -5.0, -1.8284271247461903,
                    3.0, -1.8284271247461903, -5.0, 3.8284271247461903
                });
        }

        [TestMethod]
        public void Hfft_PrimeLength_Bluestein_MatchesNumpy()
        {
            // n=13 (prime) forces the Bluestein path in the underlying irfft.
            AssertCloseArraysR(np.fft.hfft(np.arange(7).astype(np.float64), n: 13),
                np.fft.irfft(np.conjugate(np.arange(7).astype(np.float64)), n: 13, norm: "forward"), 1e-10);
        }

        // =====================================================================================
        // 4. ihfft — value pins (complex output; default n = m)
        // =====================================================================================

        [TestMethod]
        public void Ihfft_DefaultLength_IsHalfPlusOne_MatchesNumpy()
        {
            // np.fft.ihfft([15,-4,0,-1,0,-4]) : m=6 -> 6//2+1 = 4, recovers [1,2,3,4].
            AssertCloseC(np.fft.ihfft(Real(15, -4, 0, -1, 0, -4)),
                new[]
                {
                    new Complex(1, -0.0),
                    new Complex(2, -3.700743415417188e-17),
                    new Complex(3, -3.700743415417188e-17),
                    new Complex(4, -0.0)
                });
        }

        [TestMethod]
        public void Ihfft_ExplicitOddN_MatchesNumpy()
        {
            // np.fft.ihfft([15,-4,0,-1,0,-4], 5) -> 5//2+1 = 3.
            AssertCloseC(np.fft.ihfft(Real(15, -4, 0, -1, 0, -4), n: 5),
                new[]
                {
                    new Complex(2, -0.0),
                    new Complex(2.9145898033750317, -0.6432881625776283),
                    new Complex(3.5854101966249687, -0.6604395050930093)
                });
        }

        [TestMethod]
        public void Ihfft_IntInput_PromotesToComplex128()
        {
            var r = np.fft.ihfft(new NDArray(new[] { 1, 2, 3, 4 }));
            r.typecode.Should().Be(NPTypeCode.Complex);
            AssertCloseC(r, new[]
            {
                new Complex(2.5, -0.0), new Complex(-0.5, -0.5), new Complex(-0.5, -0.0)
            });
        }

        // =====================================================================================
        // 5. ihfft — normalization modes
        // =====================================================================================

        [TestMethod]
        public void Ihfft_Norm_Ortho_MatchesNumpy()
        {
            AssertCloseC(np.fft.ihfft(Real(15, -4, 0, -1, 0, -4), norm: "ortho"),
                new[]
                {
                    new Complex(2.4494897427831783, -0.0),
                    new Complex(4.898979485566357, -9.06493303673679e-17),
                    new Complex(7.348469228349535, -9.06493303673679e-17),
                    new Complex(9.797958971132713, -0.0)
                });
        }

        [TestMethod]
        public void Ihfft_Norm_Forward_MatchesNumpy()
        {
            AssertCloseC(np.fft.ihfft(Real(15, -4, 0, -1, 0, -4), norm: "forward"),
                new[]
                {
                    new Complex(6, -0.0),
                    new Complex(12, -2.220446049250313e-16),
                    new Complex(18, -2.220446049250313e-16),
                    new Complex(24, -0.0)
                });
        }

        // =====================================================================================
        // 6. hfft / ihfft round-trips (even AND odd; the documented identities)
        // =====================================================================================

        [TestMethod]
        public void Ihfft_Hfft_RoundTrip_Even()
        {
            // even: ihfft(hfft(a, 2*len(a)-2)) == a
            var a = Real(1, 2, 3, 4, 3);   // m = 5
            var back = np.fft.ihfft(np.fft.hfft(a, 2 * 5 - 2));
            back.typecode.Should().Be(NPTypeCode.Complex);
            var flat = FlatC(back);
            var orig = FlatR(a);
            for (int i = 0; i < orig.Length; i++)
            {
                flat[i].Real.Should().BeApproximately(orig[i], 1e-9, $"real[{i}]");
                flat[i].Imaginary.Should().BeApproximately(0.0, 1e-9, $"imag[{i}]");
            }
        }

        [TestMethod]
        public void Ihfft_Hfft_RoundTrip_Odd()
        {
            // odd: ihfft(hfft(a, 2*len(a)-1)) == a
            var a = Real(1, 2, 3, 4, 3);   // m = 5
            var back = np.fft.ihfft(np.fft.hfft(a, 2 * 5 - 1));
            var flat = FlatC(back);
            var orig = FlatR(a);
            for (int i = 0; i < orig.Length; i++)
            {
                flat[i].Real.Should().BeApproximately(orig[i], 1e-9, $"real[{i}]");
                flat[i].Imaginary.Should().BeApproximately(0.0, 1e-9, $"imag[{i}]");
            }
        }

        [TestMethod]
        public void Hfft_Ihfft_AllNorms_Consistent()
        {
            // hfft and ihfft are an inverse pair; check the norm plumbing is symmetric.
            var a = Real(2, 5, 1, 8, 3);   // m=5 -> hfft n=8 -> ihfft recovers 8//2+1=5
            foreach (var norm in new[] { null, "backward", "ortho", "forward" })
            {
                var spec = np.fft.hfft(a, n: 8, norm: norm);
                var back = np.fft.ihfft(spec, n: 8, norm: norm);
                var flat = FlatC(back);
                var orig = FlatR(a);
                for (int i = 0; i < orig.Length; i++)
                    flat[i].Real.Should().BeApproximately(orig[i], 1e-9, $"norm={norm} [{i}]");
            }
        }

        // =====================================================================================
        // 7. dtype policy — float32/float16 -> complex128/float64 ([Misaligned] vs NumPy)
        // =====================================================================================

        [TestMethod]
        [Misaligned]
        public void Hfft_Float32Input_OutputIsFloat64_NumpyGivesFloat32()
        {
            // NumSharp has only complex128, so the whole FFT family computes in double.
            // NumPy 2.x returns float32 for a float32 hfft; values are the double result.
            var r = np.fft.hfft(new NDArray(new float[] { 1, 2, 3, 4 }));
            r.typecode.Should().Be(NPTypeCode.Double);   // NumPy: float32
            AssertCloseR(r, new double[] { 15, -4, 0, -1, 0, -4 });
        }

        [TestMethod]
        [Misaligned]
        public void Ihfft_Float32Input_OutputIsComplex128_NumpyGivesComplex64()
        {
            // NumPy 2.x returns complex64 for a float32 ihfft; NumSharp -> complex128.
            var r = np.fft.ihfft(new NDArray(new float[] { 1, 2, 3, 4 }));
            r.typecode.Should().Be(NPTypeCode.Complex);   // NumPy: complex64
            AssertCloseC(r, new[]
            {
                new Complex(2.5, -0.0), new Complex(-0.5, -0.5), new Complex(-0.5, -0.0)
            });
        }

        // =====================================================================================
        // 8. out= — hfft IGNORES out (NumPy hard-codes out=None into irfft);
        //    ihfft USES out (threads into rfft, conjugates in place) and returns the same instance.
        // =====================================================================================

        [TestMethod]
        public void Hfft_Out_IsIgnored_LikeNumpy()
        {
            // NumPy's hfft passes out=None to irfft: the provided array is neither written nor returned.
            var o = new NDArray(new double[] { 99, 99, 99, 99, 99, 99 });
            var r = np.fft.hfft(Real(1, 2, 3, 4), @out: o);
            ReferenceEquals(r, o).Should().BeFalse();
            foreach (var z in o.Data<double>()) z.Should().Be(99);   // untouched
            AssertCloseR(r, new double[] { 15, -4, 0, -1, 0, -4 });
        }

        [TestMethod]
        public void Ihfft_Out_SameInstance_WritesThrough()
        {
            var spec = Real(15, -4, 0, -1, 0, -4);   // n=6 -> 4
            var o = new NDArray(np.complex128, new Shape(4));
            var r = np.fft.ihfft(spec, @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            AssertCloseArraysC(o, np.fft.ihfft(spec));
        }

        [TestMethod]
        public void Ihfft_Out_WrongShape_Raises()
        {
            var spec = Real(15, -4, 0, -1, 0, -4);
            Action act = () => np.fft.ihfft(spec, @out: new NDArray(np.complex128, new Shape(3)));
            act.Should().Throw<ValueError>().WithMessage("output array has wrong shape.");
        }

        // =====================================================================================
        // 9. 2-D + axis (shape and value pins)
        // =====================================================================================

        [TestMethod]
        public void Hfft_2D_Axis0_MatchesNumpy()
        {
            // np.fft.hfft(arange(12).reshape(3,4), axis=0) : (3,4) -> n=2*(3-1)=4 -> (4,4).
            var a = np.arange(12).astype(np.float64).reshape(3, 4);
            var r = np.fft.hfft(a, axis: 0);
            r.shape.Should().BeEquivalentTo(new[] { 4, 4 });
            AssertCloseR(r, new double[]
            {
                16, 20, 24, 28, -8, -8, -8, -8, 0, 0, 0, 0, -8, -8, -8, -8
            });
        }

        [TestMethod]
        public void Ihfft_2D_Axis1_MatchesNumpy()
        {
            // np.fft.ihfft(arange(12).reshape(3,4), axis=1) : 4//2+1=3 -> (3,3).
            var a = np.arange(12).astype(np.float64).reshape(3, 4);
            var r = np.fft.ihfft(a, axis: 1);
            r.shape.Should().BeEquivalentTo(new[] { 3, 3 });
            AssertCloseC(r, new[]
            {
                new Complex(1.5, -0.0), new Complex(-0.5, -0.5), new Complex(-0.5, -0.0),
                new Complex(5.5, -0.0), new Complex(-0.5, -0.5), new Complex(-0.5, -0.0),
                new Complex(9.5, -0.0), new Complex(-0.5, -0.5), new Complex(-0.5, -0.0)
            });
        }

        [TestMethod]
        public void Hfft_NegativeAxis_EqualsPositive()
        {
            var a = np.arange(12).astype(np.float64).reshape(3, 4);
            AssertCloseArraysR(np.fft.hfft(a, axis: -1), np.fft.hfft(a, axis: 1));
            AssertCloseArraysR(np.fft.hfft(a, axis: -2), np.fft.hfft(a, axis: 0));
        }

        // =====================================================================================
        // 10. Memory-layout DOD — the driver reads through strides (compare to a contiguous copy)
        // =====================================================================================

        [TestMethod]
        public void Hfft_Transposed_Axis0_EqualsContiguousCopy()
        {
            var a = np.arange(12).astype(np.float64).reshape(3, 4).T;   // (4,3) F-view
            AssertCloseArraysR(np.fft.hfft(a, axis: 0), np.fft.hfft(a.copy(), axis: 0));
        }

        [TestMethod]
        public void Hfft_Reversed_EqualsContiguousCopy()
        {
            var a = np.arange(8).astype(np.float64)["::-1"];   // negative-stride view
            AssertCloseArraysR(np.fft.hfft(a), np.fft.hfft(a.copy()));
        }

        [TestMethod]
        public void Hfft_BroadcastRead_EqualsContiguousCopy()
        {
            var a = np.broadcast_to(np.arange(4).astype(np.float64), new Shape(3, 4));  // read-only broadcast
            AssertCloseArraysR(np.fft.hfft(a, axis: 1), np.fft.hfft(a.copy(), axis: 1));
        }

        [TestMethod]
        public void Hfft_ComplexTransposedView_ConjugatesThroughStrides()
        {
            var cc = Cplx(
                new Complex(1, 1), new Complex(2, -2), new Complex(3, 3),
                new Complex(4, -4), new Complex(5, 5), new Complex(6, -6)).reshape(2, 3);
            AssertCloseArraysR(np.fft.hfft(cc.T, axis: 0), np.fft.hfft(cc.T.copy(), axis: 0));
        }

        [TestMethod]
        public void Ihfft_ColumnStrided_Axis0_EqualsContiguousCopy()
        {
            var view = np.arange(12).astype(np.float64).reshape(3, 4)[":, ::2"];  // (3,2) col-strided
            AssertCloseArraysC(np.fft.ihfft(view, axis: 0), np.fft.ihfft(view.copy(), axis: 0));
        }

        [TestMethod]
        public void Ihfft_Fcontiguous_Axis1_EqualsContiguousCopy()
        {
            var a = np.asfortranarray(np.arange(12).astype(np.float64).reshape(3, 4));
            AssertCloseArraysC(np.fft.ihfft(a, axis: 1), np.fft.ihfft(a.copy(), axis: 1));
        }

        // =====================================================================================
        // 11. error taxonomy (probed from NumPy 2.4.2)
        // =====================================================================================

        [TestMethod]
        public void Hfft_BadNorm_WithSpaceMessage()
        {
            // hfft calls _swap_direction(norm) first -> the WITH-space message (like the inverse family).
            Action act = () => np.fft.hfft(Real(1, 2, 3, 4), norm: "bad");
            act.Should().Throw<ValueError>()
                .WithMessage("Invalid norm value bad; should be \"backward\", \"ortho\" or \"forward\".");
        }

        [TestMethod]
        public void Ihfft_BadNorm_WithSpaceMessage()
        {
            Action act = () => np.fft.ihfft(Real(1, 2, 3, 4), norm: "bad");
            act.Should().Throw<ValueError>()
                .WithMessage("Invalid norm value bad; should be \"backward\", \"ortho\" or \"forward\".");
        }

        [TestMethod]
        public void Hfft_NZero_Raises()
        {
            Action act = () => np.fft.hfft(Real(1, 2, 3, 4), n: 0);
            act.Should().Throw<ValueError>().WithMessage("Invalid number of FFT data points (0) specified.");
        }

        [TestMethod]
        public void Ihfft_NZero_Raises()
        {
            Action act = () => np.fft.ihfft(Real(1, 2, 3, 4), n: 0);
            act.Should().Throw<ValueError>().WithMessage("Invalid number of FFT data points (0) specified.");
        }

        [TestMethod]
        public void Hfft_M1_DefaultN_Raises()
        {
            // m=1 -> default n = 2*(1-1) = 0 -> ValueError.
            Action act = () => np.fft.hfft(Real(5));
            act.Should().Throw<ValueError>().WithMessage("Invalid number of FFT data points (0) specified.");
        }

        [TestMethod]
        public void Hfft_AxisOob_NDefault_IndexError()
        {
            // n defaults from a.shape[axis] (tuple indexing) -> IndexError, NOT AxisError.
            Action act = () => np.fft.hfft(Real(1, 2, 3, 4), axis: 5);
            act.Should().Throw<IndexError>().WithMessage("tuple index out of range");
        }

        [TestMethod]
        public void Hfft_AxisOob_NGiven_AxisError()
        {
            // n given -> a.shape[axis] is not read; the axis is validated inside irfft.
            Action act = () => np.fft.hfft(Real(1, 2, 3, 4), n: 4, axis: 5);
            act.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 1*");
        }

        [TestMethod]
        public void Hfft_BadNormAndAxisOob_AxisReportedFirst()
        {
            // n defaults first, so the IndexError (from a.shape[axis]) precedes the norm validation.
            Action act = () => np.fft.hfft(Real(1, 2, 3, 4), axis: 9, norm: "bad");
            act.Should().Throw<IndexError>().WithMessage("tuple index out of range");
        }

        [TestMethod]
        public void Ihfft_AxisOob_NGiven_AxisError()
        {
            Action act = () => np.fft.ihfft(Real(1, 2, 3, 4), n: 4, axis: 5);
            act.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 1*");
        }

        [TestMethod]
        public void Hfft_ZeroD_IndexError()
        {
            // 0-d + default n -> a.shape[-1] is out of range -> IndexError.
            Action act = () => np.fft.hfft(NDArray.Scalar(5.0));
            act.Should().Throw<IndexError>().WithMessage("tuple index out of range");
        }

        [TestMethod]
        public void Ihfft_ZeroD_IndexError()
        {
            Action act = () => np.fft.ihfft(NDArray.Scalar(5.0));
            act.Should().Throw<IndexError>().WithMessage("tuple index out of range");
        }

        [TestMethod]
        public void Hfft_ZeroD_NGiven_AxisError()
        {
            // 0-d + explicit n -> a.shape[axis] not read; the axis fails inside irfft (dim 0).
            Action act = () => np.fft.hfft(NDArray.Scalar(5.0), n: 4);
            act.Should().Throw<AxisError>().WithMessage("axis -1 is out of bounds for array of dimension 0*");
        }
    }
}
