namespace NumSharp.Backends
{
    public partial class DefaultFftEngine
    {
        // Standard (complex-to-complex) transforms. Ports of numpy/fft/_pocketfft.py.
        // Fft/Ifft resolve n/axis/norm/out fully then reach the c2c kernel through _raw_fft;
        // Fft2/Ifft2/Fftn/Ifftn are compositions over _raw_fft via _raw_fftnd.

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: <c>n = a.shape[axis]</c> when n is None (Python tuple indexing, so an axis out of range is
        ///     an IndexError here and an AxisError in <c>_raw_fft</c> when n was given), then
        ///     <c>_raw_fft(a, n, axis, False, True, norm, out)</c>.
        /// </remarks>
        [NDScoped]
        public override NDArray Fft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
        {
            int nn = n ?? ShapeAt(a, axis);
            return RawFft(a, nn, axis, isReal: false, isForward: true, norm, @out, "fft", IsSinglePrecision(a));
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: the default length as in <see cref="Fft"/>, then
        ///     <c>_raw_fft(a, n, axis, False, False, norm, out)</c>, which swaps the norm's direction first.
        /// </remarks>
        [NDScoped]
        public override NDArray Ifft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
        {
            int nn = n ?? ShapeAt(a, axis);
            return RawFft(a, nn, axis, isReal: false, isForward: false, norm, @out, "ifft", IsSinglePrecision(a));
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's one statement <c>_raw_fftnd(a, s, axes, fft, norm, out=out)</c> with <c>axes=(-2, -1)</c> for null;
        ///     each axis runs this engine's <c>_raw_fft</c> with the original operand's precision.
        /// </remarks>
        [NDScoped]
        public override NDArray Fft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
        {
            bool floatPrec = IsSinglePrecision(a); // float-precision source: every 1-D leaf reproduces its own numpy loop
            return RawFftNd(a, s, axes ?? new[] { -2, -1 }, (arr, n, ax, nrm, o) => RawFft(arr, n, ax, false, true, nrm, o, "fft", floatPrec), norm, @out);
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's one statement <c>_raw_fftnd(a, s, axes, ifft, norm, out=None)</c>: the provided out is not threaded
        ///     through the composition.
        /// </remarks>
        [NDScoped]
        public override NDArray Ifft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
        {
            bool floatPrec = IsSinglePrecision(a);
            return RawFftNd(a, s, axes ?? new[] { -2, -1 }, (arr, n, ax, nrm, o) => RawFft(arr, n, ax, false, false, nrm, o, "ifft", floatPrec), norm, @out: null);
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's one statement <c>_raw_fftnd(a, s, axes, fft, norm, out=out)</c>: <c>_cook_nd_args</c> resolves the
        ///     lengths and axes, then the axes are transformed in reverse order.
        /// </remarks>
        [NDScoped]
        public override NDArray Fftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
        {
            bool floatPrec = IsSinglePrecision(a);
            return RawFftNd(a, s, axes, (arr, n, ax, nrm, o) => RawFft(arr, n, ax, false, true, nrm, o, "fft", floatPrec), norm, @out);
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's one statement <c>_raw_fftnd(a, s, axes, ifft, norm, out=out)</c>.
        /// </remarks>
        [NDScoped]
        public override NDArray Ifftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
        {
            bool floatPrec = IsSinglePrecision(a);
            return RawFftNd(a, s, axes, (arr, n, ax, nrm, o) => RawFft(arr, n, ax, false, false, nrm, o, "ifft", floatPrec), norm, @out);
        }
    }
}
