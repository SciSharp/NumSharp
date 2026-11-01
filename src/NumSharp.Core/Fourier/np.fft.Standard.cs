using System;

namespace NumSharp
{
    public partial class FourierModule
    {
        // Standard (complex-to-complex) transforms. Ports of numpy/fft/_pocketfft.py: each is one call
        // into the TensorEngine.Fft sub-engine (Backends/FftEngine.cs; the default's statements are
        // Backends/Default/Fourier/DefaultFftEngine.Standard.cs). The result is a plain array, as NumPy's asarray makes it.

        /// <summary>
        ///     Compute the one-dimensional discrete Fourier Transform.
        /// </summary>
        /// <param name="a">Input array, can be complex.</param>
        /// <param name="n">Length of the transformed axis of the output. Cropped/zero-padded from the
        ///     input; defaults to <c>a.shape[axis]</c>.</param>
        /// <param name="axis">Axis over which to compute the FFT (default the last axis).</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated complex output of the appropriate shape.</param>
        /// <returns>The transform, complex128 (<paramref name="out"/> itself when given).</returns>
        /// <exception cref="ValueError"><paramref name="n"/> below 1, an unknown <paramref name="norm"/>, or an
        ///     <paramref name="out"/> of the wrong shape or read-only.</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> out of range with <paramref name="n"/> omitted.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range with <paramref name="n"/> given.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        /// <exception cref="ArgumentException">An <paramref name="out"/> dtype complex128 cannot be cast to under
        ///     same_kind.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html
        ///     <para>Computed by <see cref="TensorEngine.Fft"/> of the operand's engine (<see cref="FftEngine.Fft"/>).</para></remarks>
        public NDArray fft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            return a.TensorEngine.Fft.Fft(a, n, axis, norm, @out);
        }

        /// <summary>
        ///     Compute the one-dimensional inverse discrete Fourier Transform (<c>ifft(fft(a)) == a</c>).
        /// </summary>
        /// <param name="a">Input array, can be complex.</param>
        /// <param name="n">Length of the transformed axis of the output (default <c>a.shape[axis]</c>).</param>
        /// <param name="axis">Axis over which to compute the inverse FFT (default the last axis).</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated complex output of the appropriate shape.</param>
        /// <returns>The inverse transform, complex128 (<paramref name="out"/> itself when given).</returns>
        /// <exception cref="ValueError"><paramref name="n"/> below 1, an unknown <paramref name="norm"/>, or an
        ///     <paramref name="out"/> of the wrong shape or read-only.</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> out of range with <paramref name="n"/> omitted.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range with <paramref name="n"/> given.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        /// <exception cref="ArgumentException">An <paramref name="out"/> dtype complex128 cannot be cast to under
        ///     same_kind.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html
        ///     <para>Computed by <see cref="TensorEngine.Fft"/> of the operand's engine (<see cref="FftEngine.Ifft"/>).</para></remarks>
        public NDArray ifft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            return a.TensorEngine.Fft.Ifft(a, n, axis, norm, @out);
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
        /// <returns>The transform, complex128.</returns>
        /// <exception cref="ValueError"><paramref name="s"/> and <paramref name="axes"/> of different lengths, a length
        ///     below 1, an unknown norm, or an unfit out.</exception>
        /// <exception cref="IndexError">An axis out of range.</exception>
        /// <exception cref="AxisError">An axis out of range for a one-dimensional transform.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
        ///     <para>Computed by <see cref="TensorEngine.Fft"/> of the operand's engine (<see cref="FftEngine.Fft2"/>).</para></remarks>
        public NDArray fft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            return a.TensorEngine.Fft.Fft2(a, s, axes, norm, @out);
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
        /// <returns>The inverse transform, complex128.</returns>
        /// <exception cref="ValueError"><paramref name="s"/> and <paramref name="axes"/> of different lengths, a length
        ///     below 1, or an unknown norm.</exception>
        /// <exception cref="IndexError">An axis out of range.</exception>
        /// <exception cref="AxisError">An axis out of range for a one-dimensional transform.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft2.html
        ///     <para>Computed by <see cref="TensorEngine.Fft"/> of the operand's engine (<see cref="FftEngine.Ifft2"/>).</para></remarks>
        public NDArray ifft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            return a.TensorEngine.Fft.Ifft2(a, s, axes, norm, @out);
        }

        /// <summary>
        ///     Compute the N-dimensional discrete Fourier Transform over the given axes (all axes by
        ///     default). Composition of 1-D <see cref="fft"/> per axis.
        /// </summary>
        /// <param name="a">Input array, can be complex.</param>
        /// <param name="s">Per-axis output lengths; <c>-1</c> uses the full input length. Defaults to the
        ///     input shape along <paramref name="axes"/>.</param>
        /// <param name="axes">Axes over which to compute the FFT (default: all axes, or the last
        ///     <c>len(s)</c> when only <c>s</c> is given).</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated complex output.</param>
        /// <returns>The transform, complex128.</returns>
        /// <exception cref="ValueError"><paramref name="s"/> and <paramref name="axes"/> of different lengths, a length
        ///     below 1, an unknown norm, or an unfit out.</exception>
        /// <exception cref="IndexError">An axis out of range.</exception>
        /// <exception cref="AxisError">An axis out of range for a one-dimensional transform.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.fftn.html
        ///     <para>Computed by <see cref="TensorEngine.Fft"/> of the operand's engine (<see cref="FftEngine.Fftn"/>).</para></remarks>
        public NDArray fftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            return a.TensorEngine.Fft.Fftn(a, s, axes, norm, @out);
        }

        /// <summary>
        ///     Compute the N-dimensional inverse discrete Fourier Transform (<c>ifftn(fftn(a)) == a</c>).
        ///     Composition of 1-D <see cref="ifft"/> per axis.
        /// </summary>
        /// <param name="a">Input array, can be complex.</param>
        /// <param name="s">Per-axis output lengths; <c>-1</c> uses the full input length.</param>
        /// <param name="axes">Axes over which to compute the inverse FFT (default: all axes).</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated complex output.</param>
        /// <returns>The inverse transform, complex128.</returns>
        /// <exception cref="ValueError"><paramref name="s"/> and <paramref name="axes"/> of different lengths, a length
        ///     below 1, an unknown norm, or an unfit out.</exception>
        /// <exception cref="IndexError">An axis out of range.</exception>
        /// <exception cref="AxisError">An axis out of range for a one-dimensional transform.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftn.html
        ///     <para>Computed by <see cref="TensorEngine.Fft"/> of the operand's engine (<see cref="FftEngine.Ifftn"/>).</para></remarks>
        public NDArray ifftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            return a.TensorEngine.Fft.Ifftn(a, s, axes, norm, @out);
        }
    }
}
