using System;
using System.Numerics;

namespace NumSharp.Tests.Fourier
{
    /// <summary>
    ///     Parity gate for the STANDARD complex-to-complex transforms — <c>fft</c>, <c>ifft</c>,
    ///     <c>fft2</c>, <c>ifft2</c>, <c>fftn</c>, <c>ifftn</c> — against NumPy 2.4.2. Every expected
    ///     value/shape/dtype/error was probed from a live NumPy 2.4.2. Bit-exact value parity for the
    ///     native float64/complex128 path is proven by the differential-fuzz corpus; these tests pin the
    ///     concrete NumPy outputs plus the shell behaviours (n/s truncation-padding, axes/-1 sentinel,
    ///     norms, out=, dtype policy, N-D decomposition, error taxonomy).
    /// </summary>
    [TestClass]
    public class NpFftStandardTest : TestClass
    {
        // ---- helpers -------------------------------------------------------------------------

        private static NDArray Real(params double[] v) => new NDArray(v);
        private static NDArray Cplx(params Complex[] v) => new NDArray(v);

        private static Complex[] Flat(NDArray r)
        {
            var c = np.ascontiguousarray(r);
            var d = c.Data<Complex>();
            var outp = new Complex[c.size];
            for (int i = 0; i < outp.Length; i++) outp[i] = d[i];
            return outp;
        }

        private static void AssertClose(NDArray result, Complex[] expected, double tol = 1e-11)
        {
            result.typecode.Should().Be(NPTypeCode.Complex);
            result.size.Should().Be(expected.Length);
            var flat = Flat(result);
            for (int i = 0; i < expected.Length; i++)
            {
                flat[i].Real.Should().BeApproximately(expected[i].Real, tol, $"real[{i}]");
                flat[i].Imaginary.Should().BeApproximately(expected[i].Imaginary, tol, $"imag[{i}]");
            }
        }

        private static void AssertCloseArrays(NDArray a, NDArray b, double tol = 1e-11)
        {
            a.shape.Should().BeEquivalentTo(b.shape);
            var fa = Flat(a); var fb = Flat(b);
            for (int i = 0; i < fa.Length; i++)
            {
                fa[i].Real.Should().BeApproximately(fb[i].Real, tol, $"real[{i}]");
                fa[i].Imaginary.Should().BeApproximately(fb[i].Imaginary, tol, $"imag[{i}]");
            }
        }

        // =====================================================================================
        // 1. Concrete value pins (NumPy 2.4.2)
        // =====================================================================================

        [TestMethod]
        public void Fft_1D_MatchesNumpy()
        {
            // np.fft.fft([1,2,3,4]) == [10, -2+2j, -2, -2-2j]
            AssertClose(np.fft.fft(Real(1, 2, 3, 4)),
                new[] { new Complex(10, 0), new Complex(-2, 2), new Complex(-2, 0), new Complex(-2, -2) });
        }

        [TestMethod]
        public void Ifft_1D_MatchesNumpy()
        {
            // np.fft.ifft([1,2,3,4]) == [2.5, -0.5-0.5j, -0.5, -0.5+0.5j]
            AssertClose(np.fft.ifft(Real(1, 2, 3, 4)),
                new[] { new Complex(2.5, 0), new Complex(-0.5, -0.5), new Complex(-0.5, 0), new Complex(-0.5, 0.5) });
        }

        [TestMethod]
        public void Fft_ComplexInput_MatchesNumpy()
        {
            // np.fft.fft([1+1j, 2-1j, 0+3j]) probed from NumPy 2.4.2
            var a = Cplx(new Complex(1, 1), new Complex(2, -1), new Complex(0, 3));
            AssertClose(np.fft.fft(a),
                new[]
                {
                    new Complex(3.0, 3.0),
                    new Complex(-3.4641016151377544, -1.7320508075688772),
                    new Complex(3.4641016151377544, 1.7320508075688772)
                });
        }

        [TestMethod]
        public void Fft2_2D_MatchesNumpy()
        {
            // np.fft.fft2([[1,2,3],[4,5,6]])
            var m = np.arange(1, 7).astype(np.float64).reshape(2, 3);
            AssertClose(np.fft.fft2(m),
                new[]
                {
                    new Complex(21, 0), new Complex(-3, 1.7320508075688772), new Complex(-3, -1.7320508075688772),
                    new Complex(-9, 0), new Complex(0, 0), new Complex(0, 0)
                });
        }

        [TestMethod]
        public void Fftn_3D_MatchesNumpy()
        {
            // np.fft.fftn(np.arange(8).reshape(2,2,2)) == [28,-4,-8,0,-16,0,0,0]
            var a = np.arange(8).astype(np.float64).reshape(2, 2, 2);
            AssertClose(np.fft.fftn(a),
                new[]
                {
                    new Complex(28, 0), new Complex(-4, 0), new Complex(-8, 0), new Complex(0, 0),
                    new Complex(-16, 0), new Complex(0, 0), new Complex(0, 0), new Complex(0, 0)
                });
        }

        [TestMethod]
        public void Ifft2_Eye_MatchesNumpy()
        {
            // np.fft.ifft2(4*np.eye(4)) -> anti-diagonal ones (docstring example)
            var a = (np.eye(4) * 4.0).astype(np.float64);
            var r = np.fft.ifft2(a);
            r.shape.Should().BeEquivalentTo(new[] { 4, 4 });
            var f = Flat(r);
            // row0 -> [1,0,0,0]; row1 -> [0,0,0,1]; row2 -> [0,0,1,0]; row3 -> [0,1,0,0]
            f[0].Real.Should().BeApproximately(1, 1e-11);
            f[7].Real.Should().BeApproximately(1, 1e-11);   // (1,3)
            f[10].Real.Should().BeApproximately(1, 1e-11);  // (2,2)
            f[13].Real.Should().BeApproximately(1, 1e-11);  // (3,1)
        }

        // =====================================================================================
        // 2. Normalization modes (null == "backward"; ortho == /sqrt(n); forward == /n)
        // =====================================================================================

        [TestMethod]
        public void Fft_Norm_Ortho_MatchesNumpy()
        {
            AssertClose(np.fft.fft(Real(1, 2, 3, 4), norm: "ortho"),
                new[] { new Complex(5, 0), new Complex(-1, 1), new Complex(-1, 0), new Complex(-1, -1) });
        }

        [TestMethod]
        public void Fft_Norm_Forward_MatchesNumpy()
        {
            AssertClose(np.fft.fft(Real(1, 2, 3, 4), norm: "forward"),
                new[] { new Complex(2.5, 0), new Complex(-0.5, 0.5), new Complex(-0.5, 0), new Complex(-0.5, -0.5) });
        }

        [TestMethod]
        public void Fft_Norm_Null_Equals_Backward()
        {
            AssertCloseArrays(np.fft.fft(Real(1, 2, 3, 4)), np.fft.fft(Real(1, 2, 3, 4), norm: "backward"));
        }

        [TestMethod]
        public void Ifft_Norm_Forward_IsUnscaled()
        {
            // ifft forward-norm == plain fft/... : ifft(x, norm="forward") swaps to "backward" (no scale)
            // -> equals np.fft.fft conjugate-free identity: check against NumPy value.
            AssertClose(np.fft.ifft(Real(1, 2, 3, 4), norm: "forward"),
                new[] { new Complex(10, 0), new Complex(-2, -2), new Complex(-2, 0), new Complex(-2, 2) });
        }

        [TestMethod]
        public void Fftn_Ortho_ParsevalPreserved()
        {
            // ortho fftn preserves the L2 norm (Parseval).
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            var r = np.fft.fftn(a, norm: "ortho");
            double e0 = 0, e1 = 0;
            var av = a.astype(np.complex128).Data<Complex>();
            for (int i = 0; i < a.size; i++) e0 += av[i].Magnitude * av[i].Magnitude;
            var fv = Flat(r);
            for (int i = 0; i < fv.Length; i++) e1 += fv[i].Magnitude * fv[i].Magnitude;
            e1.Should().BeApproximately(e0, 1e-6);
        }

        // =====================================================================================
        // 3. n truncation / zero-padding (fft/ifft)
        // =====================================================================================

        [TestMethod]
        public void Fft_N_Truncate_MatchesNumpy()
        {
            // np.fft.fft([1,2,3,4], n=2) == [3, -1]
            AssertClose(np.fft.fft(Real(1, 2, 3, 4), n: 2),
                new[] { new Complex(3, 0), new Complex(-1, 0) });
        }

        [TestMethod]
        public void Fft_N_Pad_MatchesNumpy()
        {
            // np.fft.fft([1,2,3,4], n=6)
            AssertClose(np.fft.fft(Real(1, 2, 3, 4), n: 6),
                new[]
                {
                    new Complex(10, 0),
                    new Complex(-3.5, -4.330127018922194),
                    new Complex(2.5, 0.8660254037844386),
                    new Complex(-2, 8.881784197001252e-16),
                    new Complex(2.5, -0.8660254037844386),
                    new Complex(-3.5, 4.330127018922193)
                });
        }

        [TestMethod]
        public void Fft_N_PadToPrime_Bluestein_MatchesNumpy()
        {
            // n=11 (prime) forces the Bluestein path; pad [0..7] with zeros.
            var a = np.arange(8).astype(np.float64);
            var r = np.fft.fft(a, n: 11);
            r.shape.Should().BeEquivalentTo(new[] { 11 });
            // DC term == sum(0..7) == 28
            Flat(r)[0].Real.Should().BeApproximately(28, 1e-9);
            // Round-trip via ifft recovers the zero-padded signal.
            var back = Flat(np.fft.ifft(r));
            for (int i = 0; i < 8; i++) back[i].Real.Should().BeApproximately(i, 1e-9);
            for (int i = 8; i < 11; i++) back[i].Magnitude.Should().BeApproximately(0, 1e-9);
        }

        // =====================================================================================
        // 4. Memory-layout DOD — the driver reads through strides
        // =====================================================================================

        [TestMethod]
        public void Fft_ColumnStrided_Axis0_MatchesNumpy()
        {
            // np.fft.fft(np.arange(12).reshape(3,4)[:, ::2], axis=0)
            var view = np.arange(12).astype(np.float64).reshape(3, 4)[":, ::2"];  // (3,2) col-strided
            AssertClose(np.fft.fft(view, axis: 0),
                new[]
                {
                    new Complex(12, 0), new Complex(18, 0),
                    new Complex(-6, 3.4641016151377544), new Complex(-6, 3.4641016151377544),
                    new Complex(-6, -3.4641016151377544), new Complex(-6, -3.4641016151377544)
                });
        }

        [TestMethod]
        public void Fft_Transposed_EqualsContiguousCopy()
        {
            var a = np.arange(12).astype(np.float64).reshape(3, 4).T;   // (4,3) F-contiguous view
            AssertCloseArrays(np.fft.fft(a, axis: 1), np.fft.fft(a.copy(), axis: 1));
            AssertCloseArrays(np.fft.fft(a, axis: 0), np.fft.fft(a.copy(), axis: 0));
        }

        [TestMethod]
        public void Fft_Reversed_EqualsContiguousCopy()
        {
            var a = np.arange(8).astype(np.float64)["::-1"];   // negative-stride view
            AssertCloseArrays(np.fft.fft(a), np.fft.fft(a.copy()));
        }

        [TestMethod]
        public void Fft_BroadcastRead_EqualsContiguousCopy()
        {
            var a = np.broadcast_to(np.arange(4).astype(np.float64), new Shape(3, 4));  // read-only broadcast
            AssertCloseArrays(np.fft.fft(a), np.fft.fft(a.copy()));
        }

        [TestMethod]
        public void Fft2_Transposed_EqualsContiguousCopy()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4).T;  // (4,3,2)
            AssertCloseArrays(np.fft.fft2(a), np.fft.fft2(a.copy()));
        }

        // =====================================================================================
        // 5. axes / axis handling
        // =====================================================================================

        [TestMethod]
        public void Fft2_DefaultAxes_AreLastTwo()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            np.fft.fft2(a).shape.Should().BeEquivalentTo(new[] { 2, 3, 4 });
            // fft2 default (-2,-1) == fftn over axes [1,2]
            AssertCloseArrays(np.fft.fft2(a), np.fft.fftn(a, axes: new[] { 1, 2 }));
        }

        [TestMethod]
        public void Fftn_DefaultAxes_AreAll()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            np.fft.fftn(a).shape.Should().BeEquivalentTo(new[] { 2, 3, 4 });
            AssertCloseArrays(np.fft.fftn(a), np.fft.fftn(a, axes: new[] { 0, 1, 2 }));
        }

        [TestMethod]
        public void Fft2_NegativeAxes_MatchExplicit()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            AssertCloseArrays(np.fft.fft2(a, axes: new[] { -3, -1 }), np.fft.fft2(a, axes: new[] { 0, 2 }));
        }

        [TestMethod]
        public void Fftn_RepeatedAxis_IsDoubleTransform()
        {
            // fftn(m, axes=[0,0]) == fft(fft(m, axis=0), axis=0)
            var m = np.arange(9).astype(np.float64).reshape(3, 3);
            AssertCloseArrays(np.fft.fftn(m, axes: new[] { 0, 0 }),
                              np.fft.fft(np.fft.fft(m, axis: 0), axis: 0));
        }

        [TestMethod]
        public void Fftn_Decomposition_EqualsSequentialFft()
        {
            // fftn == fft along each axis (reverse order irrelevant for values)
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            var seq = np.fft.fft(np.fft.fft(np.fft.fft(a, axis: 2), axis: 1), axis: 0);
            AssertCloseArrays(np.fft.fftn(a), seq);
        }

        [TestMethod]
        public void Fftn_EmptyAxes_PassThrough_PreservesDtype()
        {
            // fftn(a, axes=[]) returns the input unchanged — dtype NOT promoted to complex.
            var a = np.arange(12).astype(np.float64).reshape(3, 4);
            var r = np.fft.fftn(a, axes: new int[0]);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.shape.Should().BeEquivalentTo(new[] { 3, 4 });
            var rv = r.Data<double>();
            for (int i = 0; i < 12; i++) rv[i].Should().Be(i);
        }

        [TestMethod]
        public void Fftn_Scalar_PassThrough()
        {
            // fftn(scalar) — no axes to transform, returns the 0-d input unchanged.
            var r = np.fft.fftn(NDArray.Scalar(5.0));
            r.ndim.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.Double);
        }

        // =====================================================================================
        // 6. s / -1 sentinel / N-D padding-truncation
        // =====================================================================================

        [TestMethod]
        public void Fft2_S_TruncateAndPad_MatchesNumpy()
        {
            var m = np.arange(12).astype(np.float64).reshape(3, 4);
            np.fft.fft2(m, s: new[] { 2, 6 }).shape.Should().BeEquivalentTo(new[] { 2, 6 });
            np.fft.fft2(m, s: new[] { 5, 5 }).shape.Should().BeEquivalentTo(new[] { 5, 5 });
        }

        [TestMethod]
        public void Fft2_S_Minus1_UsesFullLength()
        {
            var m = np.arange(12).astype(np.float64).reshape(3, 4);
            // s=[-1,2]: axis -2 keeps full 3, axis -1 -> 2
            np.fft.fft2(m, s: new[] { -1, 2 }).shape.Should().BeEquivalentTo(new[] { 3, 2 });
            // -1 everywhere == no s
            AssertCloseArrays(np.fft.fft2(m, s: new[] { -1, -1 }), np.fft.fft2(m));
        }

        [TestMethod]
        public void Fftn_S_WithoutAxes_UsesLastLenSAxes()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            // s=[2,2] no axes -> transform last two axes -> (2,2,2)
            np.fft.fftn(a, s: new[] { 2, 2 }).shape.Should().BeEquivalentTo(new[] { 2, 2, 2 });
        }

        [TestMethod]
        public void Fftn_S_WithExplicitAxes()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            // s=[3,5] on axes [0,2] -> axis0 keeps 2? no: s[0]=3 -> axis0->3, s[1]=5 -> axis2->5
            np.fft.fftn(a, s: new[] { 3, 5 }, axes: new[] { 0, 2 }).shape.Should().BeEquivalentTo(new[] { 3, 3, 5 });
        }

        // =====================================================================================
        // 7. out= — same instance, write-through; ifft2 ignores out (NumPy passes out=None)
        // =====================================================================================

        [TestMethod]
        public void Fft_Out_SameInstance_WritesThrough()
        {
            var x = np.arange(8).astype(np.float64);
            var o = new NDArray(np.complex128, new Shape(8));
            var r = np.fft.fft(x, @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            AssertCloseArrays(o, np.fft.fft(x));
        }

        [TestMethod]
        public void Fft2_Out_SameInstance()
        {
            var m = np.arange(16).astype(np.float64).reshape(4, 4);
            var o = new NDArray(np.complex128, new Shape(4, 4));
            var r = np.fft.fft2(m, @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            AssertCloseArrays(o, np.fft.fft2(m));
        }

        [TestMethod]
        public void Fftn_Out_SameInstance()
        {
            var m = np.arange(16).astype(np.float64).reshape(4, 4);
            var o = new NDArray(np.complex128, new Shape(4, 4));
            var r = np.fft.fftn(m, @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            AssertCloseArrays(o, np.fft.fftn(m));
        }

        [TestMethod]
        public void Ifftn_Out_SameInstance()
        {
            var m = np.arange(16).astype(np.float64).reshape(4, 4);
            var o = new NDArray(np.complex128, new Shape(4, 4));
            var r = np.fft.ifftn(m, @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            AssertCloseArrays(o, np.fft.ifftn(m));
        }

        [TestMethod]
        public void Ifft2_Out_IsIgnored_LikeNumpy()
        {
            // NumPy's ifft2 hard-codes out=None: the provided array is NOT written and NOT returned.
            var m = np.arange(16).astype(np.float64).reshape(4, 4);
            var o = (np.zeros(new Shape(4, 4))).astype(np.complex128);
            var r = np.fft.ifft2(m, @out: o);
            ReferenceEquals(r, o).Should().BeFalse();
            // o stays all-zero (untouched)
            foreach (var z in o.Data<Complex>()) z.Magnitude.Should().Be(0);
            AssertCloseArrays(r, np.fft.ifft2(m));
        }

        [TestMethod]
        public void Fft_Out_WrongShape_Raises()
        {
            Action act = () => np.fft.fft(np.arange(4).astype(np.float64), @out: new NDArray(np.complex128, new Shape(3)));
            act.Should().Throw<ValueError>().WithMessage("output array has wrong shape.");
        }

        // =====================================================================================
        // 8. dtype policy (int -> complex128; float32/float16 -> complex128 [Misaligned])
        // =====================================================================================

        [TestMethod]
        public void Fft_IntInput_PromotesToComplex128()
        {
            var r = np.fft.fft(np.arange(4).astype(np.int32));
            r.typecode.Should().Be(NPTypeCode.Complex);
            AssertClose(r, new[] { new Complex(6, 0), new Complex(-2, 2), new Complex(-2, 0), new Complex(-2, -2) });
        }

        [TestMethod]
        [Misaligned]
        public void Fft_Float32Input_PromotesToComplex128_NumpyGivesComplex64()
        {
            // NumSharp has only complex128, so float32 input -> complex128 (compute-in-double).
            // NumPy 2.x returns complex64 here. Values are the correctly-rounded double result.
            var r = np.fft.fft(np.arange(4).astype(np.float32));
            r.typecode.Should().Be(NPTypeCode.Complex);  // NumPy: complex64
            AssertCloseArrays(r, np.fft.fft(np.arange(4).astype(np.float64)));
        }

        [TestMethod]
        [Misaligned]
        public void Fftn_Float32Input_PromotesToComplex128_NumpyGivesComplex64()
        {
            var r = np.fft.fftn(np.arange(8).astype(np.float32).reshape(2, 4));
            r.typecode.Should().Be(NPTypeCode.Complex);  // NumPy: complex64
        }

        // =====================================================================================
        // 9. Round-trip identities
        // =====================================================================================

        [TestMethod]
        public void Ifft_Fft_RoundTrip_AllNorms()
        {
            var x = Real(1, 2, 3, 4, 3, 2, 5, 1);
            foreach (var norm in new[] { null, "backward", "ortho", "forward" })
            {
                var back = Flat(np.fft.ifft(np.fft.fft(x, norm: norm), norm: norm));
                var xv = x.Data<double>();
                for (int i = 0; i < 8; i++) back[i].Real.Should().BeApproximately(xv[i], 1e-9);
            }
        }

        [TestMethod]
        public void Ifft2_Fft2_RoundTrip()
        {
            var a = np.arange(20).astype(np.float64).reshape(4, 5);
            var back = np.fft.ifft2(np.fft.fft2(a));
            AssertCloseArrays(back, a.astype(np.complex128), 1e-9);
        }

        [TestMethod]
        public void Ifftn_Fftn_RoundTrip_3D()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            var back = np.fft.ifftn(np.fft.fftn(a));
            AssertCloseArrays(back, a.astype(np.complex128), 1e-9);
        }

        [TestMethod]
        public void Fft_Ones_DcComponentOnly()
        {
            var r = Flat(np.fft.fft(np.ones(new Shape(6)).astype(np.float64)));
            r[0].Real.Should().BeApproximately(6, 1e-11);
            for (int i = 1; i < 6; i++) r[i].Magnitude.Should().BeApproximately(0, 1e-11);
        }

        // =====================================================================================
        // 10. N-D-specific error taxonomy (probed from NumPy 2.4.2)
        // =====================================================================================

        [TestMethod]
        public void Fft2_1D_Input_Raises_IndexError()
        {
            // fft2 default axes=(-2,-1): take axis -2 of a 1-D array is out of bounds.
            Action act = () => np.fft.fft2(np.arange(4).astype(np.float64));
            act.Should().Throw<IndexError>().WithMessage("index -2 is out of bounds for axis 0 with size 1");
        }

        [TestMethod]
        public void Fftn_SAxesLengthMismatch_Raises()
        {
            var m = np.arange(12).astype(np.float64).reshape(3, 4);
            Action act = () => np.fft.fftn(m, s: new[] { 2, 3 }, axes: new[] { 0, 1, 2 });
            act.Should().Throw<ValueError>().WithMessage("Shape and axes have different lengths.");
        }

        [TestMethod]
        public void Fftn_AxesOob_WithS_Raises_AxisError()
        {
            // s given (len matches) -> the per-axis fft normalizes the bad axis -> AxisError.
            var m = np.arange(12).astype(np.float64).reshape(3, 4);
            Action act = () => np.fft.fftn(m, s: new[] { 2, 3 }, axes: new[] { 0, 5 });
            act.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 2*");
        }

        [TestMethod]
        public void Fftn_AxesOob_NoS_Raises_IndexError()
        {
            // s omitted -> the take(a.shape, axes) path -> IndexError.
            var m = np.arange(12).astype(np.float64).reshape(3, 4);
            Action act = () => np.fft.fftn(m, axes: new[] { 5 });
            act.Should().Throw<IndexError>().WithMessage("index 5 is out of bounds for axis 0 with size 2");
        }

        [TestMethod]
        public void Fftn_SZeroInLoop_Raises_InvalidPoints()
        {
            var m = np.arange(12).astype(np.float64).reshape(3, 4);
            Action act = () => np.fft.fftn(m, s: new[] { 0, 3 }, axes: new[] { 0, 1 });
            act.Should().Throw<ValueError>().WithMessage("Invalid number of FFT data points (0) specified.");
        }

        [TestMethod]
        public void Fftn_SNegativeNotMinus1_Raises_InvalidPoints()
        {
            var m = np.arange(12).astype(np.float64).reshape(3, 4);
            Action act = () => np.fft.fftn(m, s: new[] { -2, 3 }, axes: new[] { 0, 1 });
            act.Should().Throw<ValueError>().WithMessage("Invalid number of FFT data points (-2) specified.");
        }

        [TestMethod]
        public void Fft2_BadNorm_Forward_NoSpaceMessage()
        {
            var m = np.arange(12).astype(np.float64).reshape(3, 4);
            Action act = () => np.fft.fft2(m, norm: "bad");
            act.Should().Throw<ValueError>()
                .WithMessage("Invalid norm value bad; should be \"backward\",\"ortho\" or \"forward\".");
        }

        [TestMethod]
        public void Ifftn_BadNorm_Inverse_WithSpaceMessage()
        {
            var m = np.arange(12).astype(np.float64).reshape(3, 4);
            Action act = () => np.fft.ifftn(m, norm: "bad");
            act.Should().Throw<ValueError>()
                .WithMessage("Invalid norm value bad; should be \"backward\", \"ortho\" or \"forward\".");
        }

        // =====================================================================================
        // 11. Zero-length dimensions
        // =====================================================================================

        [TestMethod]
        public void Fft_EmptyNonAxisDim_ReturnsEmpty()
        {
            // shape (3,0), transform axis 0 (len 3): 0 lanes -> empty (3,0) complex result.
            var z = np.zeros(new Shape(3, 0)).astype(np.float64);
            var r = np.fft.fft(z, axis: 0);
            r.shape.Should().BeEquivalentTo(new[] { 3, 0 });
            r.typecode.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Fft_ZeroLengthAxis_Raises()
        {
            var z = np.zeros(new Shape(3, 0)).astype(np.float64);
            Action act = () => np.fft.fft(z, axis: 1);   // n defaults to 0
            act.Should().Throw<ValueError>().WithMessage("Invalid number of FFT data points (0) specified.");
        }
    }
}
