namespace NumSharp.Backends
{
    public partial class DefaultFftEngine
    {
        // Hermitian transforms — compositions of irfft/rfft + np.conjugate + the norm swap,
        // exactly as numpy/fft/_pocketfft.py defines them. No kernel of their own: the transforms
        // they call reach the c2r / r2c kernels.

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: <c>n = (a.shape[axis] - 1) * 2</c> when n is None, <c>new_norm = _swap_direction(norm)</c>,
        ///     then <c>irfft(conjugate(a), n, axis, norm=new_norm, out=None)</c> - the np.fft facade, so the engine assigned
        ///     to <see cref="TensorEngine.Fft"/> computes the inverse transform. The np.fft facade applies the class
        ///     boundary (numpy.matrix, numpy.char.chararray) before it calls this member.
        /// </remarks>
        [NDScoped] // reclaims np.conjugate's temporary
        public override NDArray Hfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
        {
            int nn = n ?? (ShapeAt(a, axis) - 1) * 2;
            string newNorm = SwapDirection(norm);
            // NumPy calls irfft with out=None here (the hfft `out` argument is not used).
            return np.fft.irfft(np.conjugate(a), nn, axis, newNorm, @out: null);
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: <c>n = a.shape[axis]</c> when n is None, <c>new_norm = _swap_direction(norm)</c>,
        ///     <c>out = rfft(a, n, axis, norm=new_norm, out=out)</c> (the np.fft facade), then
        ///     <c>conjugate(out, out=out)</c>, here into the caller's out (a new array when none was given).
        /// </remarks>
        [NDScoped]
        public override NDArray Ihfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
        {
            int nn = n ?? ShapeAt(a, axis);
            string newNorm = SwapDirection(norm);
            var outp = np.fft.rfft(a, nn, axis, newNorm, @out);
            return np.conjugate(outp, @out);
        }
    }
}
