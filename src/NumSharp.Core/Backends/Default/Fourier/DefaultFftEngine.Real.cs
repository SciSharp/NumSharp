namespace NumSharp.Backends
{
    public partial class DefaultFftEngine
    {
        // Real transforms (rfft: real -> complex half-spectrum; irfft: complex half -> real).
        // Ports of numpy/fft/_pocketfft.py. Rfft/Irfft resolve fully then reach the r2c / c2r kernels;
        // Rfftn/Irfftn run _raw_fft per axis, and the 2-D wrappers call this engine's N-D members (never an np.* facade).

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: <c>n = a.shape[axis]</c> when n is None, then
        ///     <c>_raw_fft(a, n, axis, True, True, norm, out=out)</c>, whose ufunc (and its complex-input refusal) is
        ///     <c>rfft_n_even</c> or <c>rfft_n_odd</c> by the parity of n.
        /// </remarks>
        [NDScoped]
        public override NDArray Rfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
        {
            int nn = n ?? ShapeAt(a, axis);
            return RawFft(a, nn, axis, isReal: true, isForward: true, norm, @out, "rfft", IsSinglePrecision(a));
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: <c>n = (a.shape[axis] - 1) * 2</c> when n is None (so a length-1 axis gives 0, refused by
        ///     <c>_raw_fft</c>), then <c>_raw_fft(a, n, axis, True, False, norm, out=out)</c>.
        /// </remarks>
        [NDScoped]
        public override NDArray Irfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
        {
            int nn = n ?? (ShapeAt(a, axis) - 1) * 2;
            return RawFft(a, nn, axis, isReal: true, isForward: false, norm, @out, "irfft", IsSinglePrecision(a));
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's one statement <c>return rfftn(a, s, axes, norm, out=out)</c> with <c>axes=(-2, -1)</c> for null: this
        ///     engine's <see cref="FftEngine.Rfftn"/> (the member np.fft.rfftn dispatches to), so a subclass that overrides
        ///     it computes rfft2 too. It opens no scope (the result is the callee's own).
        /// </remarks>
        public override NDArray Rfft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
            => Rfftn(a, s, axes ?? new[] { -2, -1 }, norm, @out);

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: <c>_cook_nd_args(a, s, axes)</c>, <c>rfft</c> over the last axis
        ///     (<c>rfft(a, s[-1], axes[-1], norm, out=out)</c>), then <c>fft</c> over the others in reverse order, all with
        ///     the provided out. Each axis runs this engine's <c>_raw_fft</c> with the original operand's precision.
        /// </remarks>
        [NDScoped]
        public override NDArray Rfftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
        {
            bool floatPrec = IsSinglePrecision(a); // float-precision source: thread it so every leaf reproduces its own numpy loop
            var (ss, aa) = CookNdArgs(a, s, axes, invreal: false);
            // NumPy indexes s[-1]/axes[-1] with no axes to transform (0-d input, or an explicit
            // empty `axes`), leaking a bare IndexError("list index out of range"). Reproduce it here
            // rather than let C# surface an IndexOutOfRangeException.
            if (aa.Length == 0)
                throw new IndexError("list index out of range");
            a = RawFft(a, ss[aa.Length - 1], aa[aa.Length - 1], isReal: true, isForward: true, norm, @out, "rfft", floatPrec);
            for (int ii = aa.Length - 2; ii >= 0; ii--)
                a = RawFft(a, ss[ii], aa[ii], isReal: false, isForward: true, norm, @out, "fft", floatPrec);
            return a;
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's one statement <c>return irfftn(a, s, axes, norm, out=None)</c> with <c>axes=(-2, -1)</c> for null: this
        ///     engine's <see cref="FftEngine.Irfftn"/> (the member np.fft.irfftn dispatches to), without the provided out.
        ///     It opens no scope.
        /// </remarks>
        public override NDArray Irfft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
            => Irfftn(a, s, axes ?? new[] { -2, -1 }, norm, @out: null);

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: <c>_cook_nd_args(a, s, axes, invreal=1)</c> (the last default length is
        ///     <c>2 * (m - 1)</c>), <c>ifft</c> over every axis but the last without the provided out, then <c>irfft</c> over
        ///     the last with it. Each axis runs this engine's <c>_raw_fft</c> with the original operand's precision.
        /// </remarks>
        [NDScoped]
        public override NDArray Irfftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
        {
            bool floatPrec = IsSinglePrecision(a); // float-precision source: thread it so every leaf reproduces its own numpy loop
            var (ss, aa) = CookNdArgs(a, s, axes, invreal: true);
            // As in Rfftn: no axes to transform leaks NumPy's IndexError("list index out of range")
            // at the s[-1]/axes[-1] access. (For a shapeless `s`, the invreal defaulting inside
            // CookNdArgs reaches that access first — see the report's shared-core escalation.)
            if (aa.Length == 0)
                throw new IndexError("list index out of range");
            for (int ii = 0; ii < aa.Length - 1; ii++)
                a = RawFft(a, ss[ii], aa[ii], isReal: false, isForward: false, norm, @out: null, "ifft", floatPrec);
            a = RawFft(a, ss[aa.Length - 1], aa[aa.Length - 1], isReal: true, isForward: false, norm, @out, "irfft", floatPrec);
            return a;
        }
    }
}
