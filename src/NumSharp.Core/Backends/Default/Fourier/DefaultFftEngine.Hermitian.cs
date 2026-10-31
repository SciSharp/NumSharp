namespace NumSharp.Backends
{
    public partial class DefaultFftEngine
    {
        // Hermitian transforms — compositions of irfft/rfft + conjugate + the norm swap, exactly as
        // numpy/fft/_pocketfft.py defines them. No kernel of their own: the transforms they call reach the c2r / r2c
        // kernels. They call this engine's own members and the operand engine's Conjugate, never an np.* facade.

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: <c>n = (a.shape[axis] - 1) * 2</c> when n is None, <c>new_norm = _swap_direction(norm)</c>,
        ///     then <c>irfft(conjugate(a), n, axis, norm=new_norm, out=None)</c>: <c>conjugate</c> is the operand engine's
        ///     <see cref="TensorEngine.Conjugate"/> (np.conjugate's body without its class-boundary test: the np.fft facade
        ///     applied that before it called this member, so <paramref name="a"/> is plain), and <c>irfft</c> is this
        ///     engine's <see cref="FftEngine.Irfft"/>, so a subclass that overrides <c>Irfft</c> computes the inverse
        ///     transform here too.
        /// </remarks>
        [NDScoped] // reclaims the conjugate's temporary
        public override NDArray Hfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
        {
            int nn = n ?? (ShapeAt(a, axis) - 1) * 2;
            string newNorm = SwapDirection(norm);
            // NumPy calls irfft with out=None here (the hfft `out` argument is not used).
            return Irfft(a.TensorEngine.Conjugate(a, null, null, null), nn, axis, newNorm, @out: null);
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: <c>n = a.shape[axis]</c> when n is None, <c>new_norm = _swap_direction(norm)</c>,
        ///     <c>out = rfft(a, n, axis, norm=new_norm, out=out)</c> (this engine's <see cref="FftEngine.Rfft"/>), then
        ///     <c>conjugate(out, out=out)</c> (the operand engine's <see cref="TensorEngine.Conjugate"/>), here into the
        ///     caller's out (a new array when none was given).
        /// </remarks>
        [NDScoped]
        public override NDArray Ihfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
        {
            int nn = n ?? ShapeAt(a, axis);
            string newNorm = SwapDirection(norm);
            var outp = Rfft(a, nn, axis, newNorm, @out);
            return outp.TensorEngine.Conjugate(outp, null, @out, null);
        }
    }
}
