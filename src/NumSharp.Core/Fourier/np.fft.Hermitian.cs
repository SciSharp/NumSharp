namespace NumSharp
{
    public partial class FourierModule
    {
        // Hermitian transforms — compositions of irfft/rfft + conjugate + the norm swap, exactly as
        // numpy/fft/_pocketfft.py defines them: each is one call into the TensorEngine.Fft sub-engine
        // (Backends/FftEngine.cs; the default's statements are Backends/Default/Fourier/DefaultFftEngine.Hermitian.cs,
        // which call the conjugate ufunc's engine member, not np.conjugate), hfft behind NumPy's asarray class
        // boundary (a numpy.matrix or numpy.char.chararray operand gives a plain array).

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
        /// <returns>The real result, float64; a plain array for a <c>numpy.matrix</c> or <c>numpy.char.chararray</c>
        ///     operand too.</returns>
        /// <exception cref="ValueError">A length below 1, or an unknown <paramref name="norm"/> (the inverse
        ///     transforms' text).</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> out of range with <paramref name="n"/> omitted.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range with <paramref name="n"/> given.</exception>
        /// <exception cref="TypeError">A text input (np.conjugate has no string loop).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.hfft.html</remarks>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.
        ///     <para>Computed by <see cref="TensorEngine.Fft"/> of the operand's engine (<see cref="FftEngine.Hfft"/>),
        ///     after the class boundary: the member's <c>np.conjugate</c> would keep a matrix.</para>
        /// </remarks>
        [NDScoped] // the class boundary's temporaries; the Fft member scopes the computation's own
        public NDArray hfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
        {
            // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Plain((self: this, a, n, axis, norm, @out), static s => s.self.hfft(s.a, s.n, s.axis, s.norm, s.@out));
            // a numpy.char.chararray operand too: NumPy's asarray drops either class (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.Plain((self: this, a, n, axis, norm, @out), static s => s.self.hfft(s.a, s.n, s.axis, s.norm, s.@out));
            return np.EngineOf(a).Fft.Hfft(a, n, axis, norm, @out);
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
        /// <returns>The conjugated half spectrum, complex128 (<paramref name="out"/> itself when given).</returns>
        /// <exception cref="ValueError">A length below 1, an unknown <paramref name="norm"/>, or an
        ///     <paramref name="out"/> of the wrong shape or read-only.</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> out of range with <paramref name="n"/> omitted.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range with <paramref name="n"/> given.</exception>
        /// <exception cref="TypeError">A complex input (rfft's refusal).</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        /// <exception cref="System.ArgumentException">An <paramref name="out"/> dtype complex128 cannot be cast to under
        ///     same_kind.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.ihfft.html
        ///     <para>Computed by <see cref="TensorEngine.Fft"/> of the operand's engine (<see cref="FftEngine.Ihfft"/>).</para></remarks>
        public NDArray ihfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
            => np.EngineOf(a).Fft.Ihfft(a, n, axis, norm, @out);
    }
}
