namespace NumSharp
{
    public partial class FourierModule
    {
        // Standard (complex-to-complex) transforms. Ports of numpy/fft/_pocketfft.py.
        // fft/ifft resolve n/axis/norm/out fully then reach the 1-D engine seam;
        // fft2/ifft2/fftn/ifftn are pure compositions over fft/ifft via _raw_fftnd.

        /// <summary>
        ///     Compute the one-dimensional discrete Fourier Transform.
        /// </summary>
        /// <param name="a">Input array, can be complex.</param>
        /// <param name="n">Length of the transformed axis of the output. Cropped/zero-padded from the
        ///     input; defaults to <c>a.shape[axis]</c>.</param>
        /// <param name="axis">Axis over which to compute the FFT (default the last axis).</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated complex output of the appropriate shape.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html</remarks>
        [NDScoped]
        public NDArray fft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
        {
            int nn = n ?? ShapeAt(a, axis);
            return RawFft(a, nn, axis, isReal: false, isForward: true, norm, @out, "fft", IsSinglePrecision(a));
        }

        /// <summary>
        ///     Compute the one-dimensional inverse discrete Fourier Transform (<c>ifft(fft(a)) == a</c>).
        /// </summary>
        /// <param name="a">Input array, can be complex.</param>
        /// <param name="n">Length of the transformed axis of the output (default <c>a.shape[axis]</c>).</param>
        /// <param name="axis">Axis over which to compute the inverse FFT (default the last axis).</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated complex output of the appropriate shape.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html</remarks>
        [NDScoped]
        public NDArray ifft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
        {
            int nn = n ?? ShapeAt(a, axis);
            return RawFft(a, nn, axis, isReal: false, isForward: false, norm, @out, "ifft", IsSinglePrecision(a));
        }

        /// <summary>
        ///     Compute the 2-dimensional discrete Fourier Transform (over the last two axes by default).
        ///     <c>fft2</c> is <c>fftn</c> with a different default for <paramref name="axes"/>.
        /// </summary>
        /// <param name="a">Input array, can be complex.</param>
        /// <param name="s">Shape (length of each transformed axis) of the output; <c>-1</c> uses the full
        ///     input length. Defaults to the input shape along <paramref name="axes"/>.</param>
        /// <param name="axes">Axes over which to compute the FFT. Default <c>(-2, -1)</c>.</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated complex output.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html</remarks>
        [NDScoped]
        public NDArray fft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
        {
            bool floatPrec = IsSinglePrecision(a); // float-precision source: every 1-D leaf reproduces its own numpy loop
            return RawFftNd(a, s, axes ?? new[] { -2, -1 }, (arr, n, ax, nrm, o) => RawFft(arr, n, ax, false, true, nrm, o, "fft", floatPrec), norm, @out);
        }

        /// <summary>
        ///     Compute the 2-dimensional inverse discrete Fourier Transform (over the last two axes by
        ///     default). <c>ifft2</c> is <c>ifftn</c> with a different default for <paramref name="axes"/>.
        ///     Like NumPy, the provided <paramref name="out"/> is not threaded through the composition.
        /// </summary>
        /// <param name="a">Input array, can be complex.</param>
        /// <param name="s">Per-axis output lengths; <c>-1</c> uses the full input length.</param>
        /// <param name="axes">Axes over which to compute the inverse FFT. Default <c>(-2, -1)</c>.</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Accepted for signature parity; NumPy passes <c>out=None</c> internally.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft2.html</remarks>
        [NDScoped]
        public NDArray ifft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
        {
            bool floatPrec = IsSinglePrecision(a);
            return RawFftNd(a, s, axes ?? new[] { -2, -1 }, (arr, n, ax, nrm, o) => RawFft(arr, n, ax, false, false, nrm, o, "ifft", floatPrec), norm, @out: null);
        }

        /// <summary>
        ///     Compute the N-dimensional discrete Fourier Transform over the given axes (all axes by
        ///     default). Pure composition of 1-D <see cref="fft"/> per axis.
        /// </summary>
        /// <param name="a">Input array, can be complex.</param>
        /// <param name="s">Per-axis output lengths; <c>-1</c> uses the full input length. Defaults to the
        ///     input shape along <paramref name="axes"/>.</param>
        /// <param name="axes">Axes over which to compute the FFT (default: all axes, or the last
        ///     <c>len(s)</c> when only <c>s</c> is given).</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated complex output.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.fftn.html</remarks>
        [NDScoped]
        public NDArray fftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
        {
            bool floatPrec = IsSinglePrecision(a);
            return RawFftNd(a, s, axes, (arr, n, ax, nrm, o) => RawFft(arr, n, ax, false, true, nrm, o, "fft", floatPrec), norm, @out);
        }

        /// <summary>
        ///     Compute the N-dimensional inverse discrete Fourier Transform (<c>ifftn(fftn(a)) == a</c>).
        ///     Pure composition of 1-D <see cref="ifft"/> per axis.
        /// </summary>
        /// <param name="a">Input array, can be complex.</param>
        /// <param name="s">Per-axis output lengths; <c>-1</c> uses the full input length.</param>
        /// <param name="axes">Axes over which to compute the inverse FFT (default: all axes).</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated complex output.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftn.html</remarks>
        [NDScoped]
        public NDArray ifftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
        {
            bool floatPrec = IsSinglePrecision(a);
            return RawFftNd(a, s, axes, (arr, n, ax, nrm, o) => RawFft(arr, n, ax, false, false, nrm, o, "ifft", floatPrec), norm, @out);
        }
    }
}
