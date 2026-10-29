using System;
using System.Collections.Generic;
using System.Numerics;
using AwesomeAssertions;
using NumSharp.Backends;

namespace NumSharp.Tests.Fourier
{
    /// <summary>
    ///     The <see cref="TensorEngine.Fft"/> sub-engine (the np-layer dtype plan's phase PE, E3): every <c>np.fft</c>
    ///     transform and frequency helper reaches the <see cref="FftEngine"/> assigned to the cached engine, arrays created
    ///     before the assignment included; a <see cref="DefaultFftEngine"/> subclass that overrides a kernel computes the
    ///     transforms while NumPy's statements (the length, norm, axis and out= checks, the N-D, real and Hermitian
    ///     compositions) stay; null restores the default.
    /// </summary>
    /// <remarks>
    ///     Every test assigns the property of the process-wide cached engine, which every other test's arrays use, so the
    ///     class runs alone (<see cref="DoNotParallelizeAttribute"/>) and restores the default in a <c>finally</c>.
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class FftSubEngineTests
    {
        /// <summary>
        ///     A probe over the default statements: each of the 16 members records its name and then runs the default.
        /// </summary>
        private sealed class RecordingFft : DefaultFftEngine
        {
            /// <summary>The names of the members called, in NumPy's spelling.</summary>
            public readonly List<string> Calls = new();

            public override NDArray Fft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null) { Calls.Add("fft"); return base.Fft(a, n, axis, norm, @out); }
            public override NDArray Ifft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null) { Calls.Add("ifft"); return base.Ifft(a, n, axis, norm, @out); }
            public override NDArray Fft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null) { Calls.Add("fft2"); return base.Fft2(a, s, axes, norm, @out); }
            public override NDArray Ifft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null) { Calls.Add("ifft2"); return base.Ifft2(a, s, axes, norm, @out); }
            public override NDArray Fftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null) { Calls.Add("fftn"); return base.Fftn(a, s, axes, norm, @out); }
            public override NDArray Ifftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null) { Calls.Add("ifftn"); return base.Ifftn(a, s, axes, norm, @out); }
            public override NDArray Rfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null) { Calls.Add("rfft"); return base.Rfft(a, n, axis, norm, @out); }
            public override NDArray Irfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null) { Calls.Add("irfft"); return base.Irfft(a, n, axis, norm, @out); }
            public override NDArray Rfft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null) { Calls.Add("rfft2"); return base.Rfft2(a, s, axes, norm, @out); }
            public override NDArray Rfftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null) { Calls.Add("rfftn"); return base.Rfftn(a, s, axes, norm, @out); }
            public override NDArray Irfft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null) { Calls.Add("irfft2"); return base.Irfft2(a, s, axes, norm, @out); }
            public override NDArray Irfftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null) { Calls.Add("irfftn"); return base.Irfftn(a, s, axes, norm, @out); }
            public override NDArray Hfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null) { Calls.Add("hfft"); return base.Hfft(a, n, axis, norm, @out); }
            public override NDArray Ihfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null) { Calls.Add("ihfft"); return base.Ihfft(a, n, axis, norm, @out); }
            public override NDArray Fftfreq(long n, double d = 1.0, string device = null) { Calls.Add("fftfreq"); return base.Fftfreq(n, d, device); }
            public override NDArray Rfftfreq(long n, double d = 1.0, string device = null) { Calls.Add("rfftfreq"); return base.Rfftfreq(n, d, device); }
        }

        /// <summary>One kernel call as the provider below saw it: the arguments NumPy's statements resolved.</summary>
        /// <param name="Kernel"><c>c2c</c>, <c>r2c</c> or <c>c2r</c>.</param>
        /// <param name="Operand">The operand's dtype as the kernel received it (an N-D transform's later axes see the
        ///     complex intermediate).</param>
        /// <param name="N">The transform length.</param>
        /// <param name="Axis">The axis, already normalized.</param>
        /// <param name="Forward">The direction (always true for r2c, false for c2r).</param>
        /// <param name="Fct">The normalization factor.</param>
        /// <param name="HasOut">Whether an out= reached the kernel.</param>
        /// <param name="SinglePrecision">The precision thread of a float32 / float16 operand.</param>
        /// <param name="UnitFactor">Whether the factor is NumPy's integer 1.</param>
        private sealed record KernelCall(string Kernel, NPTypeCode Operand, int N, int Axis, bool Forward, double Fct,
            bool HasOut, bool SinglePrecision, bool UnitFactor);

        /// <summary>
        ///     A provider in miniature: it records every kernel call with the arguments NumPy's statements resolved, and
        ///     answers either through the default kernels or, for <see cref="Constant"/>, with a recognizable array of the
        ///     result's shape, so a test can see that the public function returns the kernel's result.
        /// </summary>
        private sealed class KernelProvider : DefaultFftEngine
        {
            /// <summary>The kernel calls, in order.</summary>
            public readonly List<KernelCall> Calls = new();

            /// <summary>When set, the complex-to-complex kernel answers <c>Constant + 0j</c> everywhere instead of
            ///     transforming.</summary>
            public double? Constant;

            /// <inheritdoc/>
            /// <remarks>Records the call; answers <see cref="Constant"/> everywhere when it is set, else delegates.</remarks>
            protected override NDArray C2CKernel(NDArray a, int n, int axis, bool forward, double fct, NDArray @out,
                bool singlePrecision, bool unitFactor)
            {
                Calls.Add(new KernelCall("c2c", a.typecode, n, axis, forward, fct, @out is not null, singlePrecision, unitFactor));
                if (Constant is not double k)
                    return base.C2CKernel(a, n, axis, forward, fct, @out, singlePrecision, unitFactor);
                var dims = (long[])a.Shape.dimensions.Clone();
                dims[axis] = n;
                return np.full(Shape.Adopt(dims), new Complex(k, 0), np.complex128);
            }

            /// <inheritdoc/>
            /// <remarks>Records the call and delegates.</remarks>
            protected override NDArray R2CKernel(NDArray a, int n, int axis, double fct, NDArray @out,
                bool singlePrecision, bool unitFactor)
            {
                Calls.Add(new KernelCall("r2c", a.typecode, n, axis, true, fct, @out is not null, singlePrecision, unitFactor));
                return base.R2CKernel(a, n, axis, fct, @out, singlePrecision, unitFactor);
            }

            /// <inheritdoc/>
            /// <remarks>Records the call and delegates.</remarks>
            protected override NDArray C2RKernel(NDArray a, int n, int axis, double fct, NDArray @out,
                bool singlePrecision, bool unitFactor)
            {
                Calls.Add(new KernelCall("c2r", a.typecode, n, axis, false, fct, @out is not null, singlePrecision, unitFactor));
                return base.C2RKernel(a, n, axis, fct, @out, singlePrecision, unitFactor);
            }
        }

        /// <summary>
        ///     Without an assignment the cached engine answers the shared <see cref="DefaultFftEngine"/>; an assignment is
        ///     read back, leaves <see cref="TensorEngine.Linalg"/> alone, and null restores the default.
        /// </summary>
        [TestMethod]
        public void TheDefault_IsTheSharedDefaultFftEngine_AndAssigningNullRestoresIt()
        {
            var engine = BackendFactory.GetEngine();
            var before = engine.Fft;
            before.Should().BeOfType<DefaultFftEngine>("NumSharp.Core installs its own numpy.fft statements");
            var probe = new RecordingFft();
            try
            {
                engine.Fft = probe;
                engine.Fft.Should().BeSameAs(probe);
                engine.Linalg.Should().BeOfType<DefaultLinalgEngine>("the two sub-engines are assigned separately");
                engine.Fft = null;
                engine.Fft.Should().BeSameAs(before, "null stands for the shared default, as a null Blas does for the managed kernels");
            }
            finally
            {
                engine.Fft = null;
            }
        }

        /// <summary>
        ///     Each of the 16 computing np.fft functions calls its member of the engine assigned after its operand was
        ///     created, and the frequency helpers, which take no array, reach the same cached engine.
        /// </summary>
        [TestMethod]
        public void EveryNpFftFunction_CallsTheAssignedEngine_ForArraysCreatedBeforeTheAssignment()
        {
            // Created BEFORE the assignment: the property sits on the cached engine, so it reaches existing arrays. The
            // frequency helpers take no array and use the default engine, which is the same cached instance.
            var v = np.array(new[] {1.0, 2.0, 3.0, 4.0});
            var m = np.arange(6.0).reshape(2, 3);
            var engine = v.TensorEngine;
            BackendFactory.GetEngine().Should().BeSameAs(engine, "fftfreq / rfftfreq reach the operand engine's property");
            var probe = new RecordingFft();
            try
            {
                engine.Fft = probe;

                np.fft.fft(v);
                np.fft.ifft(v);
                np.fft.fft2(m);
                np.fft.ifft2(m);
                np.fft.fftn(m);
                np.fft.ifftn(m);
                np.fft.rfft(v);
                np.fft.irfft(v);
                np.fft.rfft2(m);
                np.fft.irfft2(m);
                np.fft.rfftn(m);
                np.fft.irfftn(m);
                np.fft.hfft(v);
                np.fft.ihfft(v);
                np.fft.fftfreq(8);
                np.fft.rfftfreq(8L);
            }
            finally
            {
                engine.Fft = null;
            }

            new HashSet<string>(probe.Calls).Should().BeEquivalentTo(new[]
            {
                "fft", "ifft", "fft2", "ifft2", "fftn", "ifftn", "rfft", "irfft", "rfft2", "irfft2", "rfftn", "irfftn",
                "hfft", "ihfft", "fftfreq", "rfftfreq",
            }, "each of numpy.fft's 16 computing functions is one Fft member");
            probe.Calls.Should().HaveCount(20, "the four compositions (rfft2, irfft2, hfft, ihfft) call one more member each");
        }

        /// <summary>
        ///     The four compositions NumPy writes over its module functions (hfft, ihfft, rfft2, irfft2) call the np.fft
        ///     facades, so the assigned engine computes the nested function too, in NumPy's order.
        /// </summary>
        [TestMethod]
        public void ANestedCall_GoesThroughTheAssignedEngineToo()
        {
            // NumPy's hfft calls the module's irfft, ihfft its rfft, rfft2 its rfftn and irfft2 its irfftn: the defaults call
            // the public np.fft functions, so the assigned engine computes the nested function as well.
            var v = np.array(new[] {1.0, 2.0, 3.0, 4.0});
            var m = np.arange(6.0).reshape(2, 3);
            var engine = v.TensorEngine;
            var probe = new RecordingFft();
            try
            {
                engine.Fft = probe;
                np.fft.hfft(v);
                np.fft.ihfft(v);
                np.fft.rfft2(m);
                np.fft.irfft2(m);
            }
            finally
            {
                engine.Fft = null;
            }

            probe.Calls.Should().Equal(new[] {"hfft", "irfft", "ihfft", "rfft", "rfft2", "rfftn", "irfft2", "irfftn"});
        }

        /// <summary>
        ///     A kernel override sees what NumPy's statements resolved (the default length, the normalized axis, the factor
        ///     in the operand's precision, the precision thread across an N-D transform), its result is what the public
        ///     function returns, and the checks before it still refuse without calling it.
        /// </summary>
        [TestMethod]
        public void TheKernels_ReceiveTheResolvedArguments_AndEveryTransformReachesThem()
        {
            var v = np.array(new[] {1.0, 2.0, 3.0, 4.0});
            var c = np.array(new[] {1.0, 2.0, 3.0, 4.0}).astype(np.complex128);
            var m = np.arange(6.0).reshape(2, 3);
            var f32 = np.array(new[] {1f, 2f, 3f, 4f, 5f});
            var engine = v.TensorEngine;
            var provider = new KernelProvider();
            try
            {
                engine.Fft = provider;

                // fft: the default length (4), the axis normalized (-1 -> 0), norm None's integer factor.
                var expected = np.fft.fft(v);   // through the default kernel: the provider delegates
                provider.Calls.Should().Equal(new KernelCall("c2c", NPTypeCode.Double, 4, 0, true, 1.0, false, false, true));
                provider.Calls.Clear();

                // ifft with norm None: _swap_direction makes it "forward", so the factor is 1/n and not the integer 1.
                np.fft.ifft(c);
                provider.Calls.Should().Equal(new KernelCall("c2c", NPTypeCode.Complex, 4, 0, false, 0.25, false, false, false));
                provider.Calls.Clear();

                // rfft / irfft: irfft's default length is 2 * (m - 1).
                np.fft.rfft(v);
                np.fft.irfft(c);
                provider.Calls.Should().Equal(
                    new KernelCall("r2c", NPTypeCode.Double, 4, 0, true, 1.0, false, false, true),
                    new KernelCall("c2r", NPTypeCode.Complex, 6, 0, false, 1.0 / 6, false, false, false));
                provider.Calls.Clear();

                // fftn: _raw_fftnd transforms the axes in reverse order; the second axis sees the complex intermediate.
                np.fft.fftn(m);
                provider.Calls.Should().Equal(
                    new KernelCall("c2c", NPTypeCode.Double, 3, 1, true, 1.0, false, false, true),
                    new KernelCall("c2c", NPTypeCode.Complex, 2, 0, true, 1.0, false, false, true));
                provider.Calls.Clear();

                // rfftn: r2c over the last axis, then c2c; irfftn: c2c first, c2r last; hfft / ihfft through irfft / rfft.
                np.fft.rfftn(m);
                np.fft.irfftn(m);
                np.fft.hfft(v);
                np.fft.ihfft(v);
                provider.Calls.ConvertAll(k => k.Kernel).Should().Equal("r2c", "c2c", "c2c", "c2r", "c2r", "r2c");
                provider.Calls.Clear();

                // A float32 operand: the factor is computed in float32 (ortho, n = 5), and the precision thread reaches the
                // N-D transform's complex128 intermediate too.
                np.fft.fft(f32, norm: "ortho");
                provider.Calls.Should().Equal(new KernelCall("c2c", NPTypeCode.Single, 5, 0, true,
                    (double)(1f / MathF.Sqrt(5f)), false, true, false));
                provider.Calls.Clear();
                np.fft.rfftn(np.arange(6.0).reshape(2, 3).astype(np.float32), norm: "forward");
                provider.Calls.ConvertAll(k => (k.Kernel, k.Operand, k.SinglePrecision)).Should().Equal(
                    ("r2c", NPTypeCode.Single, true), ("c2c", NPTypeCode.Complex, true));
                provider.Calls.Clear();

                // The kernel's result is what the public function returns.
                provider.Constant = 7.0;
                np.fft.fft(v).ToArray<Complex>().Should().Equal(new Complex(7, 0), new Complex(7, 0), new Complex(7, 0), new Complex(7, 0));
                provider.Constant = null;
                provider.Calls.Clear();

                // The checks before the kernel still refuse, and a refused call never reaches it.
                new Action(() => np.fft.fft(v, 0)).Should().Throw<ValueError>()
                    .WithMessage("Invalid number of FFT data points (0) specified.");
                new Action(() => np.fft.fft(v, norm: "Ortho")).Should().Throw<ValueError>()
                    .WithMessage("Invalid norm value Ortho; should be \"backward\",\"ortho\" or \"forward\".");
                new Action(() => np.fft.rfft(c)).Should().Throw<TypeError>()
                    .WithMessage("ufunc 'rfft_n_even' not supported for the input types*");
                new Action(() => np.fft.fft(v, null, 3)).Should().Throw<IndexError>()
                    .WithMessage("tuple index out of range");
                new Action(() => np.fft.fft(v, null, -1, null, np.zeros(new Shape(5), np.complex128))).Should().Throw<ValueError>()
                    .WithMessage("output array has wrong shape.");
                provider.Calls.Should().BeEmpty();

                // The default kernel gave the same values the shared default gives.
                engine.Fft = null;
                np.fft.fft(v).ToArray<Complex>().Should().Equal(expected.ToArray<Complex>());
            }
            finally
            {
                engine.Fft = null;
            }
        }

        /// <summary>
        ///     A checked out= reaches the kernel, which writes it and returns it; ihfft threads its out through rfft's kernel
        ///     and the conjugate.
        /// </summary>
        [TestMethod]
        public void TheOut_ReachesTheKernel_AfterTheShapeAndWriteabilityChecks()
        {
            var v = np.array(new[] {1.0, 2.0, 3.0, 4.0});
            var engine = v.TensorEngine;
            var provider = new KernelProvider();
            var @out = np.zeros(new Shape(4), np.complex128);
            try
            {
                engine.Fft = provider;
                var r = np.fft.fft(v, null, -1, null, @out);
                ReferenceEquals(r, @out).Should().BeTrue("the default kernel writes into the given out and returns it");
                provider.Calls.Should().Equal(new KernelCall("c2c", NPTypeCode.Double, 4, 0, true, 1.0, true, false, true));

                // ihfft threads its out through rfft (the kernel sees it) and the conjugate.
                provider.Calls.Clear();
                var half = np.zeros(new Shape(3), np.complex128);
                ReferenceEquals(np.fft.ihfft(v, null, -1, null, half), half).Should().BeTrue();
                provider.Calls.Should().Equal(new KernelCall("r2c", NPTypeCode.Double, 4, 0, true, 0.25, true, false, false));
            }
            finally
            {
                engine.Fft = null;
            }
        }
    }
}
