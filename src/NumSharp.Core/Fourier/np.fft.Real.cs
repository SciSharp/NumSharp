using System;

namespace NumSharp
{
    public partial class FourierModule
    {
        // Real transforms (rfft: real -> complex half-spectrum; irfft: complex half -> real).
        // Ports of numpy/fft/_pocketfft.py: each is one call into the TensorEngine.Fft sub-engine
        // (Backends/FftEngine.cs; the default's statements are Fourier/DefaultFftEngine.Real.cs).

        /// <summary>
        ///     Compute the one-dimensional discrete Fourier Transform for real input. The output length
        ///     along the axis is <c>n//2 + 1</c> (the non-negative-frequency half).
        /// </summary>
        /// <param name="a">Input array (taken to be real).</param>
        /// <param name="n">Number of points along the transform axis to use (default <c>a.shape[axis]</c>).</param>
        /// <param name="axis">Axis over which to compute the FFT (default the last axis).</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated complex output of shape <c>(..., n//2+1, ...)</c>.</param>
        /// <returns>The half spectrum, complex128 (<paramref name="out"/> itself when given).</returns>
        /// <exception cref="ValueError"><paramref name="n"/> below 1, an unknown <paramref name="norm"/>, or an
        ///     <paramref name="out"/> of the wrong shape or read-only.</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> out of range with <paramref name="n"/> omitted.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range with <paramref name="n"/> given.</exception>
        /// <exception cref="TypeError">A complex input (<c>ufunc 'rfft_n_even'</c> / <c>'rfft_n_odd' not supported for
        ///     the input types, ...</c>).</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        /// <exception cref="ArgumentException">An <paramref name="out"/> dtype complex128 cannot be cast to under
        ///     same_kind.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html
        ///     <para>Computed by <see cref="TensorEngine.Fft"/> of the operand's engine (<see cref="FftEngine.Rfft"/>).</para></remarks>
        public NDArray rfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
            => np.EngineOf(a).Fft.Rfft(a, n, axis, norm, @out);

        /// <summary>
        ///     Compute the inverse of <see cref="rfft"/> — a real-valued output. For <c>n</c> output
        ///     points, <c>n//2+1</c> input points are used; when <c>n</c> is omitted it defaults to
        ///     <c>2*(m-1)</c> where <c>m</c> is the input length along the axis.
        /// </summary>
        /// <param name="a">Input array (the non-negative-frequency half-spectrum).</param>
        /// <param name="n">Length of the transformed (real) axis of the output. Default <c>2*(m-1)</c>.</param>
        /// <param name="axis">Axis over which to compute the inverse FFT (default the last axis).</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated real (float64) output of length <c>n</c> on the axis.</param>
        /// <returns>The real signal, float64 (<paramref name="out"/> itself when given).</returns>
        /// <exception cref="ValueError"><paramref name="n"/> below 1 (an input of length 1 with <paramref name="n"/>
        ///     omitted gives 0), an unknown <paramref name="norm"/>, or an <paramref name="out"/> of the wrong shape or
        ///     read-only.</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> out of range with <paramref name="n"/> omitted.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range with <paramref name="n"/> given.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        /// <exception cref="ArgumentException">An <paramref name="out"/> dtype float64 cannot be cast to under
        ///     same_kind.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft.html
        ///     <para>Computed by <see cref="TensorEngine.Fft"/> of the operand's engine (<see cref="FftEngine.Irfft"/>).</para></remarks>
        public NDArray irfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
            => np.EngineOf(a).Fft.Irfft(a, n, axis, norm, @out);

        /// <summary>
        ///     Compute the 2-dimensional FFT of a real array. <c>rfft2</c> is <c>rfftn</c> with the
        ///     default <paramref name="axes"/> of <c>(-2, -1)</c>.
        /// </summary>
        /// <param name="a">Input array (taken to be real).</param>
        /// <param name="s">Per-axis output lengths; <c>-1</c> uses the full input length.</param>
        /// <param name="axes">Axes over which to compute the FFT. Default <c>(-2, -1)</c>.</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated complex output.</param>
        /// <returns>The transform, complex128, the last transformed axis halved.</returns>
        /// <exception cref="ValueError"><paramref name="s"/> and <paramref name="axes"/> of different lengths, a length
        ///     below 1, an unknown norm, or an unfit out.</exception>
        /// <exception cref="IndexError">An axis out of range, or no axis to transform.</exception>
        /// <exception cref="AxisError">An axis out of range for a one-dimensional transform.</exception>
        /// <exception cref="TypeError">A complex input.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft2.html
        ///     <para>Computed by <see cref="TensorEngine.Fft"/> of the operand's engine (<see cref="FftEngine.Rfft2"/>).</para></remarks>
        public NDArray rfft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
            => np.EngineOf(a).Fft.Rfft2(a, s, axes, norm, @out);

        /// <summary>
        ///     Compute the N-dimensional discrete Fourier Transform for real input: <see cref="rfft"/>
        ///     over the LAST transformed axis, then <see cref="fft"/> over the remaining axes.
        /// </summary>
        /// <param name="a">Input array (taken to be real).</param>
        /// <param name="s">Per-axis output lengths; the final element is <c>n</c> for <c>rfft</c>, the
        ///     rest are <c>n</c> for <c>fft</c>. <c>-1</c> uses the full input length.</param>
        /// <param name="axes">Axes over which to compute the FFT (default: all axes).</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated complex output.</param>
        /// <returns>The transform, complex128, the last transformed axis halved.</returns>
        /// <exception cref="ValueError"><paramref name="s"/> and <paramref name="axes"/> of different lengths, a length
        ///     below 1, an unknown norm, or an unfit out.</exception>
        /// <exception cref="IndexError">An axis out of range, or no axis to transform (<c>list index out of
        ///     range</c>).</exception>
        /// <exception cref="AxisError">An axis out of range for a one-dimensional transform.</exception>
        /// <exception cref="TypeError">A complex input.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftn.html
        ///     <para>Computed by <see cref="TensorEngine.Fft"/> of the operand's engine (<see cref="FftEngine.Rfftn"/>).</para></remarks>
        public NDArray rfftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
            => np.EngineOf(a).Fft.Rfftn(a, s, axes, norm, @out);

        /// <summary>
        ///     Compute the inverse of <see cref="rfft2"/>. <c>irfft2</c> is <c>irfftn</c> with the
        ///     default <paramref name="axes"/> of <c>(-2, -1)</c>. Like NumPy, the provided
        ///     <paramref name="out"/> is not threaded through the composition.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="s">Per-axis output lengths; <c>-1</c> uses the full input length.</param>
        /// <param name="axes">Axes over which to compute the inverse FFT. Default <c>(-2, -1)</c>.</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Accepted for signature parity; NumPy passes <c>out=None</c> internally.</param>
        /// <returns>The real result, float64.</returns>
        /// <exception cref="ValueError"><paramref name="s"/> and <paramref name="axes"/> of different lengths, a length
        ///     below 1, or an unknown norm.</exception>
        /// <exception cref="IndexError">An axis out of range, or no axis to transform.</exception>
        /// <exception cref="AxisError">An axis out of range for a one-dimensional transform.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft2.html
        ///     <para>Computed by <see cref="TensorEngine.Fft"/> of the operand's engine (<see cref="FftEngine.Irfft2"/>).</para></remarks>
        public NDArray irfft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
            => np.EngineOf(a).Fft.Irfft2(a, s, axes, norm, @out);

        /// <summary>
        ///     Compute the inverse of <see cref="rfftn"/>: <see cref="ifft"/> over all axes but the last,
        ///     then <see cref="irfft"/> over the last axis (real output). The last-axis default length is
        ///     <c>2*(m-1)</c>.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="s">Per-axis output lengths; <c>-1</c> uses the full input length.</param>
        /// <param name="axes">Axes over which to compute the inverse FFT (default: all axes).</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated real (float64) output for the final transform.</param>
        /// <returns>The real result, float64.</returns>
        /// <exception cref="ValueError"><paramref name="s"/> and <paramref name="axes"/> of different lengths, a length
        ///     below 1, an unknown norm, or an unfit out.</exception>
        /// <exception cref="IndexError">An axis out of range, or no axis to transform (<c>tuple</c> / <c>list index out
        ///     of range</c>).</exception>
        /// <exception cref="AxisError">An axis out of range for a one-dimensional transform.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.irfftn.html
        ///     <para>Computed by <see cref="TensorEngine.Fft"/> of the operand's engine (<see cref="FftEngine.Irfftn"/>).</para></remarks>
        public NDArray irfftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
            => np.EngineOf(a).Fft.Irfftn(a, s, axes, norm, @out);
    }
}
