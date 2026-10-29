// DefaultFftEngine's kernels: the three _pocketfft_umath ufuncs NumPy's _raw_fft calls last (numpy/fft/_pocketfft_umath.cpp:
// fft / ifft, rfft_n_even / rfft_n_odd, irfft), each one call into NumSharp's managed pocketfft port
// (PocketFFTDriver.Execute, whose isReal / isForward arguments select the same three transforms). A provider (MKL, FFTW,
// cuFFT) overrides them; the public members keep NumPy's statements around them (np-layer dtype plan, phase PE, E3).

using System;

namespace NumSharp.Backends
{
    public partial class DefaultFftEngine
    {
        /// <summary>
        ///     The complex-to-complex transform, NumPy's <c>fft</c> / <c>ifft</c> ufunc: every lane of
        ///     <paramref name="a"/> along <paramref name="axis"/> cropped or zero-padded to <paramref name="n"/> points,
        ///     transformed, and scaled by <paramref name="fct"/>. fft, ifft, the N-D complex transforms, every axis but the
        ///     last of rfftn and irfftn, and hfft / ihfft through them all end here.
        /// </summary>
        /// <param name="a">The operand, real or complex of any numeric dtype (taken as complex128); the statements have
        ///     refused text.</param>
        /// <param name="n">The transform and output length along the axis, at least 1.</param>
        /// <param name="axis">The transformed axis, already normalized to <c>[0, a.ndim)</c>.</param>
        /// <param name="forward">True for the forward transform (exponent sign -1), false for the inverse.</param>
        /// <param name="fct">The normalization factor applied to the result: 1, <c>1/sqrt(n)</c> or <c>1/n</c>, computed in
        ///     the operand's real dtype and carried as a double.</param>
        /// <param name="out">The output the caller passed, its shape checked against <c>a.shape</c> with the axis
        ///     replaced by <paramref name="n"/> and its writeability checked; null to allocate one.</param>
        /// <param name="singlePrecision">The public call's operand is float32 or float16, so NumPy's result is complex64:
        ///     reproduce its values (NumPy computes in double and rounds, or, for a complex64-precision operand with a
        ///     float factor, runs its single-precision loop).</param>
        /// <param name="unitFactor">The factor is NumPy's Python int 1 (norm None or "backward"), which makes NumPy's loop
        ///     resolution take the double loop even for a complex64-precision operand.</param>
        /// <returns>A new complex128 array of <c>a.shape</c> with the axis replaced by <paramref name="n"/>, or
        ///     <paramref name="out"/> itself holding the result.</returns>
        /// <exception cref="ArgumentException">An <paramref name="out"/> whose dtype complex128 cannot be cast to under
        ///     same_kind (<c>Cannot cast ufunc 'fft' output from dtype('complex128') to ...</c>).</exception>
        protected virtual NDArray C2CKernel(NDArray a, int n, int axis, bool forward, double fct, NDArray @out,
            bool singlePrecision, bool unitFactor)
            => PocketFFTDriver.Execute(a, n, axis, isReal: false, isForward: forward, fct, @out, singlePrecision, unitFactor);

        /// <summary>
        ///     The real-to-complex transform, NumPy's <c>rfft_n_even</c> / <c>rfft_n_odd</c> ufunc: every lane of
        ///     <paramref name="a"/> along <paramref name="axis"/> cropped or zero-padded to <paramref name="n"/> real points
        ///     and transformed forward, keeping the <c>n // 2 + 1</c> non-negative frequencies, scaled by
        ///     <paramref name="fct"/>. rfft, the last axis of rfftn / rfft2, and ihfft end here.
        /// </summary>
        /// <param name="a">The real operand of any non-complex numeric dtype (taken as float64); the statements have
        ///     refused complex and text operands.</param>
        /// <param name="n">The number of input points per lane, at least 1.</param>
        /// <param name="axis">The transformed axis, already normalized to <c>[0, a.ndim)</c>.</param>
        /// <param name="fct">The normalization factor applied to the result, computed in the operand's real dtype.</param>
        /// <param name="out">The output the caller passed, its shape checked (the axis of length <c>n // 2 + 1</c>) and its
        ///     writeability checked; null to allocate one.</param>
        /// <param name="singlePrecision">The public call's operand is float32 or float16, so NumPy's result is complex64:
        ///     reproduce its values (a float32 operand with a float factor runs NumPy's single-precision loop; the others
        ///     compute in double and round).</param>
        /// <param name="unitFactor">The factor is NumPy's Python int 1 (norm None or "backward"), which makes NumPy take the
        ///     double loop.</param>
        /// <returns>A new complex128 array of <c>a.shape</c> with the axis replaced by <c>n // 2 + 1</c>, or
        ///     <paramref name="out"/> itself holding the result.</returns>
        /// <exception cref="ArgumentException">An <paramref name="out"/> whose dtype complex128 cannot be cast to under
        ///     same_kind (the message names <c>rfft_n_even</c> or <c>rfft_n_odd</c> by the parity of
        ///     <paramref name="n"/>).</exception>
        protected virtual NDArray R2CKernel(NDArray a, int n, int axis, double fct, NDArray @out,
            bool singlePrecision, bool unitFactor)
            => PocketFFTDriver.Execute(a, n, axis, isReal: true, isForward: true, fct, @out, singlePrecision, unitFactor);

        /// <summary>
        ///     The complex-to-real transform, NumPy's <c>irfft</c> ufunc: every lane of <paramref name="a"/> along
        ///     <paramref name="axis"/> read as the non-negative half of a Hermitian spectrum (its first <c>n // 2 + 1</c>
        ///     terms, zero-padded when shorter), transformed backward to <paramref name="n"/> real points and scaled by
        ///     <paramref name="fct"/>. irfft, the last axis of irfftn / irfft2, and hfft end here.
        /// </summary>
        /// <param name="a">The half spectrum, of any numeric dtype (taken as complex128); the statements have refused
        ///     text.</param>
        /// <param name="n">The output length per lane, at least 1.</param>
        /// <param name="axis">The transformed axis, already normalized to <c>[0, a.ndim)</c>.</param>
        /// <param name="fct">The normalization factor applied to the result, computed in the operand's real dtype.</param>
        /// <param name="out">The output the caller passed, its shape checked (the axis of length <paramref name="n"/>) and
        ///     its writeability checked; null to allocate one.</param>
        /// <param name="singlePrecision">The public call's operand is float32 or float16, so NumPy's result is float32 (or
        ///     float16 for a float16 operand): reproduce its values.</param>
        /// <param name="unitFactor">The factor is NumPy's Python int 1 (norm None or "backward"), which makes NumPy take the
        ///     double loop.</param>
        /// <returns>A new float64 array of <c>a.shape</c> with the axis replaced by <paramref name="n"/>, or
        ///     <paramref name="out"/> itself holding the result (a same_kind cast of float64, such as complex128 or
        ///     float32).</returns>
        /// <exception cref="ArgumentException">An <paramref name="out"/> whose dtype float64 cannot be cast to under
        ///     same_kind (<c>Cannot cast ufunc 'irfft' output from dtype('float64') to ...</c>).</exception>
        protected virtual NDArray C2RKernel(NDArray a, int n, int axis, double fct, NDArray @out,
            bool singlePrecision, bool unitFactor)
            => PocketFFTDriver.Execute(a, n, axis, isReal: true, isForward: false, fct, @out, singlePrecision, unitFactor);
    }
}
