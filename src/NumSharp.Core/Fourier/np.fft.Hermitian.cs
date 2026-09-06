namespace NumSharp
{
    public partial class FourierModule
    {
        // Hermitian transforms — pure compositions of irfft/rfft + np.conjugate + the norm swap,
        // exactly as numpy/fft/_pocketfft.py defines them. No new engine seam of their own.

        /// <summary>
        ///     Compute the FFT of a signal that has Hermitian symmetry (a real spectrum). Defined as
        ///     <c>irfft(conjugate(a), n, axis, norm=_swap_direction(norm))</c>. The output is real; its
        ///     length along the axis is <c>n</c>, or <c>2*(m-1)</c> when <c>n</c> is omitted.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="n">Length of the transformed axis of the output (default <c>2*(m-1)</c>).</param>
        /// <param name="axis">Axis over which to compute the FFT (default the last axis).</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Accepted for signature parity; NumPy passes <c>out=None</c> into irfft.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.hfft.html</remarks>
        [NDScoped]
        public NDArray hfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
        {
            int nn = n ?? (ShapeAt(a, axis) - 1) * 2;
            string newNorm = SwapDirection(norm);
            // NumPy calls irfft with out=None here (the hfft `out` argument is not used).
            return irfft(np.conjugate(a), nn, axis, newNorm, @out: null);
        }

        /// <summary>
        ///     Compute the inverse FFT of a signal that has Hermitian symmetry. Defined as
        ///     <c>conjugate(rfft(a, n, axis, norm=_swap_direction(norm)), out=out)</c>. The output is
        ///     complex; its length along the axis is <c>n//2 + 1</c>.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="n">Length of the inverse FFT along the transform axis (default <c>a.shape[axis]</c>).</param>
        /// <param name="axis">Axis over which to compute the inverse FFT (default the last axis).</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated complex output (threaded into rfft and the conjugate).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.ihfft.html</remarks>
        [NDScoped]
        public NDArray ihfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
        {
            int nn = n ?? ShapeAt(a, axis);
            string newNorm = SwapDirection(norm);
            var outp = rfft(a, nn, axis, newNorm, @out);
            return np.conjugate(outp, @out);
        }
    }
}
