// numpy.fft's functions as a sub-engine, reached through TensorEngine.Fft (docs/plans/np-layer-dtype-logic.md, section 1
// "Which O" and phase PE, E3): NumPy writes them in Python (numpy/fft/_pocketfft.py, numpy/fft/_helper.py) around the
// _pocketfft_umath ufuncs, and DefaultFftEngine runs those statements over protected virtual c2c / r2c / c2r kernels. The
// rolls fftshift and ifftshift are not members: they are functions of the array object, facades over np.roll.

using System;

namespace NumSharp
{
    /// <summary>
    ///     The contract of <c>numpy.fft</c>: one member per function that computes (16 of NumPy's 18; <c>fftshift</c> and
    ///     <c>ifftshift</c> are rolls of the array object). An engine reaches it through <see cref="TensorEngine.Fft"/>, and
    ///     every public <c>np.fft</c> function applies NumPy's class boundary and C#'s conversions, then calls one member.
    /// </summary>
    /// <remarks>
    ///     <para><b>The default</b> is <see cref="Backends.DefaultFftEngine"/>: NumPy's statements in NumPy's order (the
    ///     length and norm checks, <c>_cook_nd_args</c>, <c>_raw_fftnd</c>'s axis order, the Hermitian compositions,
    ///     <c>fftfreq</c>'s arithmetic) over three kernels that stand for NumPy's <c>_pocketfft_umath</c> ufuncs (complex to
    ///     complex, real to complex, complex to real), computed by NumSharp's managed port of pocketfft, bit for bit with
    ///     NumPy 2.4.2's on the double path.</para>
    ///     <para><b>Providing one.</b> Derive from <see cref="Backends.DefaultFftEngine"/> and override its kernels to compute
    ///     the transforms another way (MKL, FFTW, cuFFT) while NumPy's statements stay; derive from this class to replace
    ///     whole functions. Assign the instance to <see cref="TensorEngine.Fft"/> of the cached engine: it reaches every array
    ///     bound to that engine, existing ones included.</para>
    ///     <para><b>Operands.</b> A member receives plain arrays: the <c>np.fft</c> functions decide the <c>numpy.matrix</c>
    ///     and <c>numpy.char.chararray</c> classes before they call it (NumPy's <c>asarray</c> drops both), so an override
    ///     never sees those classes from them.</para>
    ///     <para><b>Results.</b> NumSharp has one complex dtype (complex128, issue #569): where NumPy returns complex64 /
    ///     float32 / float16 for a float32 / float16 input, the result dtype is complex128 / float64 and the values are
    ///     NumPy's, rounded to NumPy's precision.</para>
    /// </remarks>
    public abstract class FftEngine
    {
        /// <summary>
        ///     NumPy's <c>np.fft.fft(a, n=None, axis=-1, norm=None, out=None)</c>: the one-dimensional discrete Fourier
        ///     transform along <paramref name="axis"/>.
        /// </summary>
        /// <param name="a">The input, real or complex (a real input is taken as complex).</param>
        /// <param name="n">The transform length: the axis is cropped or zero-padded to it; null for the input's length.</param>
        /// <param name="axis">The transformed axis (negative counts from the end).</param>
        /// <param name="norm">null or <c>"backward"</c> (no scaling forward), <c>"ortho"</c> (<c>1/sqrt(n)</c>) or
        ///     <c>"forward"</c> (<c>1/n</c>).</param>
        /// <param name="out">A pre-allocated output of the result's shape (its dtype a same_kind cast of complex128), or
        ///     null.</param>
        /// <returns>The transform, complex128 (<paramref name="out"/> itself when given).</returns>
        /// <exception cref="ValueError"><paramref name="n"/> below 1 (<c>Invalid number of FFT data points (0)
        ///     specified.</c>), an unknown <paramref name="norm"/>, or an <paramref name="out"/> of the wrong shape or
        ///     read-only.</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> out of range with <paramref name="n"/> omitted (NumPy's
        ///     <c>a.shape[axis]</c>: <c>tuple index out of range</c>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range with <paramref name="n"/> given.</exception>
        /// <exception cref="DTypePromotionError">A text input (<c>result_type(a.real.dtype, 1.0)</c> has no common DType).</exception>
        /// <exception cref="ArgumentException">An <paramref name="out"/> dtype the complex128 result cannot be cast to under
        ///     same_kind (NumPy's UFuncTypeError text).</exception>
        public abstract NDArray Fft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.fft.ifft(a, n=None, axis=-1, norm=None, out=None)</c>: the one-dimensional inverse discrete
        ///     Fourier transform, <c>ifft(fft(a)) == a</c> to rounding.
        /// </summary>
        /// <param name="a">The input, real or complex.</param>
        /// <param name="n">The transform length (cropped or zero-padded); null for the input's length.</param>
        /// <param name="axis">The transformed axis.</param>
        /// <param name="norm">null or <c>"backward"</c> (<c>1/n</c> on the inverse), <c>"ortho"</c> or <c>"forward"</c> (no
        ///     scaling on the inverse).</param>
        /// <param name="out">A pre-allocated output of the result's shape, or null.</param>
        /// <returns>The inverse transform, complex128 (<paramref name="out"/> itself when given).</returns>
        /// <exception cref="ValueError"><paramref name="n"/> below 1, an unknown <paramref name="norm"/> (the inverse's
        ///     text, with a space after "backward",), or an <paramref name="out"/> of the wrong shape or read-only.</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> out of range with <paramref name="n"/> omitted.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range with <paramref name="n"/> given.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        /// <exception cref="ArgumentException">An <paramref name="out"/> dtype the complex128 result cannot be cast to under
        ///     same_kind.</exception>
        public abstract NDArray Ifft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.fft.fft2(a, s=None, axes=(-2, -1), norm=None, out=None)</c>: <see cref="Fftn"/> over the last two
        ///     axes by default.
        /// </summary>
        /// <param name="a">The input, real or complex.</param>
        /// <param name="s">The output length per transformed axis (<c>-1</c> keeps the input's), or null for the input's.</param>
        /// <param name="axes">The transformed axes; null stands for NumPy's default <c>(-2, -1)</c>.</param>
        /// <param name="norm">The normalization mode, as for <see cref="Fft"/>.</param>
        /// <param name="out">A pre-allocated output handed to every one-dimensional transform (NumPy's <c>out=out</c>), or
        ///     null.</param>
        /// <returns>The transform, complex128.</returns>
        /// <exception cref="ValueError"><paramref name="s"/> and <paramref name="axes"/> of different lengths (<c>Shape and
        ///     axes have different lengths.</c>), a length below 1, an unknown norm, or an unfit out.</exception>
        /// <exception cref="IndexError">An axis out of range (NumPy's <c>take</c> / tuple-index texts; a 0-d input:
        ///     <c>cannot do a non-empty take from an empty axes.</c>).</exception>
        /// <exception cref="AxisError">An axis out of range for a one-dimensional transform.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        public abstract NDArray Fft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.fft.ifft2(a, s=None, axes=(-2, -1), norm=None, out=None)</c>: <see cref="Ifftn"/> over the last two
        ///     axes by default. Like NumPy's, it hands <c>out=None</c> to the transforms.
        /// </summary>
        /// <param name="a">The input, real or complex.</param>
        /// <param name="s">The output length per transformed axis, or null.</param>
        /// <param name="axes">The transformed axes; null stands for <c>(-2, -1)</c>.</param>
        /// <param name="norm">The normalization mode, as for <see cref="Ifft"/>.</param>
        /// <param name="out">Accepted for NumPy's signature; NumPy's ifft2 does not use it.</param>
        /// <returns>The inverse transform, complex128.</returns>
        /// <exception cref="ValueError">Mismatched <paramref name="s"/> / <paramref name="axes"/>, a length below 1, or an
        ///     unknown norm.</exception>
        /// <exception cref="IndexError">An axis out of range.</exception>
        /// <exception cref="AxisError">An axis out of range for a one-dimensional transform.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        public abstract NDArray Ifft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.fft.fftn(a, s=None, axes=None, norm=None, out=None)</c>: the N-dimensional transform, one
        ///     <see cref="Fft"/> per axis in reverse order (<c>_raw_fftnd</c>).
        /// </summary>
        /// <param name="a">The input, real or complex.</param>
        /// <param name="s">The output length per transformed axis (<c>-1</c> keeps the input's); null for the input's. Given
        ///     without <paramref name="axes"/>, the last <c>len(s)</c> axes are transformed.</param>
        /// <param name="axes">The transformed axes, or null for all of them.</param>
        /// <param name="norm">The normalization mode, as for <see cref="Fft"/>.</param>
        /// <param name="out">A pre-allocated output handed to every one-dimensional transform, or null.</param>
        /// <returns>The transform, complex128.</returns>
        /// <exception cref="ValueError">Mismatched <paramref name="s"/> / <paramref name="axes"/>, a length below 1, an
        ///     unknown norm, or an unfit out.</exception>
        /// <exception cref="IndexError">An axis out of range.</exception>
        /// <exception cref="AxisError">An axis out of range for a one-dimensional transform.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        public abstract NDArray Fftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.fft.ifftn(a, s=None, axes=None, norm=None, out=None)</c>: the N-dimensional inverse transform, one
        ///     <see cref="Ifft"/> per axis in reverse order.
        /// </summary>
        /// <param name="a">The input, real or complex.</param>
        /// <param name="s">The output length per transformed axis, or null.</param>
        /// <param name="axes">The transformed axes, or null for all of them.</param>
        /// <param name="norm">The normalization mode, as for <see cref="Ifft"/>.</param>
        /// <param name="out">A pre-allocated output handed to every one-dimensional transform, or null.</param>
        /// <returns>The inverse transform, complex128.</returns>
        /// <exception cref="ValueError">Mismatched <paramref name="s"/> / <paramref name="axes"/>, a length below 1, an
        ///     unknown norm, or an unfit out.</exception>
        /// <exception cref="IndexError">An axis out of range.</exception>
        /// <exception cref="AxisError">An axis out of range for a one-dimensional transform.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        public abstract NDArray Ifftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.fft.rfft(a, n=None, axis=-1, norm=None, out=None)</c>: the one-dimensional transform of a real
        ///     input, the non-negative-frequency half (<c>n // 2 + 1</c> terms).
        /// </summary>
        /// <param name="a">The real input (a complex one is refused: NumPy's rfft has no complex loop).</param>
        /// <param name="n">The number of input points used (cropped or zero-padded); null for the input's length.</param>
        /// <param name="axis">The transformed axis.</param>
        /// <param name="norm">The normalization mode, as for <see cref="Fft"/>.</param>
        /// <param name="out">A pre-allocated complex output of length <c>n // 2 + 1</c> on the axis, or null.</param>
        /// <returns>The half spectrum, complex128 (<paramref name="out"/> itself when given).</returns>
        /// <exception cref="ValueError"><paramref name="n"/> below 1, an unknown norm, or an unfit out.</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> out of range with <paramref name="n"/> omitted.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range with <paramref name="n"/> given.</exception>
        /// <exception cref="TypeError">A complex input (<c>ufunc 'rfft_n_even' not supported for the input types, ...</c>,
        ///     <c>rfft_n_odd</c> for an odd <paramref name="n"/>).</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        /// <exception cref="ArgumentException">An <paramref name="out"/> dtype the complex128 result cannot be cast to under
        ///     same_kind.</exception>
        public abstract NDArray Rfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.fft.irfft(a, n=None, axis=-1, norm=None, out=None)</c>: the inverse of <see cref="Rfft"/>, a real
        ///     output of <paramref name="n"/> points from the half spectrum's first <c>n // 2 + 1</c> terms.
        /// </summary>
        /// <param name="a">The half spectrum.</param>
        /// <param name="n">The output length; null for NumPy's <c>2 * (m - 1)</c>, <c>m</c> the input's length.</param>
        /// <param name="axis">The transformed axis.</param>
        /// <param name="norm">The normalization mode, as for <see cref="Ifft"/>.</param>
        /// <param name="out">A pre-allocated output of length <paramref name="n"/> on the axis (float64, or a same_kind cast
        ///     of it such as complex128), or null.</param>
        /// <returns>The real signal, float64 (<paramref name="out"/> itself when given).</returns>
        /// <exception cref="ValueError"><paramref name="n"/> below 1 (an input of length 1 with <paramref name="n"/> omitted
        ///     gives 0), an unknown norm, or an unfit out.</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> out of range with <paramref name="n"/> omitted.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range with <paramref name="n"/> given.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        /// <exception cref="ArgumentException">An <paramref name="out"/> dtype the float64 result cannot be cast to under
        ///     same_kind.</exception>
        public abstract NDArray Irfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.fft.rfft2(a, s=None, axes=(-2, -1), norm=None, out=None)</c>: <see cref="Rfftn"/> over the last two
        ///     axes by default.
        /// </summary>
        /// <param name="a">The real input.</param>
        /// <param name="s">The output length per transformed axis, or null.</param>
        /// <param name="axes">The transformed axes; null stands for <c>(-2, -1)</c>.</param>
        /// <param name="norm">The normalization mode, as for <see cref="Fft"/>.</param>
        /// <param name="out">A pre-allocated output, or null.</param>
        /// <returns>The transform, complex128, the last axis halved.</returns>
        /// <exception cref="ValueError">Mismatched <paramref name="s"/> / <paramref name="axes"/>, a length below 1, an
        ///     unknown norm, or an unfit out.</exception>
        /// <exception cref="IndexError">An axis out of range.</exception>
        /// <exception cref="AxisError">An axis out of range for a one-dimensional transform.</exception>
        /// <exception cref="TypeError">A complex input.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        public abstract NDArray Rfft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.fft.rfftn(a, s=None, axes=None, norm=None, out=None)</c>: <see cref="Rfft"/> over the last
        ///     transformed axis, then <see cref="Fft"/> over the others in reverse order.
        /// </summary>
        /// <param name="a">The real input.</param>
        /// <param name="s">The output length per transformed axis (the last one is rfft's <c>n</c>), or null.</param>
        /// <param name="axes">The transformed axes, or null for all of them.</param>
        /// <param name="norm">The normalization mode, as for <see cref="Fft"/>.</param>
        /// <param name="out">A pre-allocated output handed to every one-dimensional transform, or null.</param>
        /// <returns>The transform, complex128, the last transformed axis halved.</returns>
        /// <exception cref="ValueError">Mismatched <paramref name="s"/> / <paramref name="axes"/>, a length below 1, an
        ///     unknown norm, or an unfit out.</exception>
        /// <exception cref="IndexError">An axis out of range, or no axis to transform (a 0-d input, an empty
        ///     <paramref name="axes"/>: <c>list index out of range</c>).</exception>
        /// <exception cref="AxisError">An axis out of range for a one-dimensional transform.</exception>
        /// <exception cref="TypeError">A complex input.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        public abstract NDArray Rfftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.fft.irfft2(a, s=None, axes=(-2, -1), norm=None, out=None)</c>: <see cref="Irfftn"/> over the last two
        ///     axes by default. Like NumPy's, it hands <c>out=None</c> on.
        /// </summary>
        /// <param name="a">The input spectrum.</param>
        /// <param name="s">The output length per transformed axis, or null.</param>
        /// <param name="axes">The transformed axes; null stands for <c>(-2, -1)</c>.</param>
        /// <param name="norm">The normalization mode, as for <see cref="Ifft"/>.</param>
        /// <param name="out">Accepted for NumPy's signature; NumPy's irfft2 does not use it.</param>
        /// <returns>The real result, float64.</returns>
        /// <exception cref="ValueError">Mismatched <paramref name="s"/> / <paramref name="axes"/>, a length below 1, or an
        ///     unknown norm.</exception>
        /// <exception cref="IndexError">An axis out of range, or no axis to transform.</exception>
        /// <exception cref="AxisError">An axis out of range for a one-dimensional transform.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        public abstract NDArray Irfft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.fft.irfftn(a, s=None, axes=None, norm=None, out=None)</c>: <see cref="Ifft"/> over every transformed
        ///     axis but the last, then <see cref="Irfft"/> over the last (a real output whose default length is
        ///     <c>2 * (m - 1)</c>).
        /// </summary>
        /// <param name="a">The input spectrum.</param>
        /// <param name="s">The output length per transformed axis, or null.</param>
        /// <param name="axes">The transformed axes, or null for all of them.</param>
        /// <param name="norm">The normalization mode, as for <see cref="Ifft"/>.</param>
        /// <param name="out">A pre-allocated real output for the final transform, or null.</param>
        /// <returns>The real result, float64.</returns>
        /// <exception cref="ValueError">Mismatched <paramref name="s"/> / <paramref name="axes"/>, a length below 1, an
        ///     unknown norm, or an unfit out.</exception>
        /// <exception cref="IndexError">An axis out of range, or no axis to transform (NumPy's <c>tuple</c> / <c>list index
        ///     out of range</c>).</exception>
        /// <exception cref="AxisError">An axis out of range for a one-dimensional transform.</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        public abstract NDArray Irfftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.fft.hfft(a, n=None, axis=-1, norm=None, out=None)</c>: the transform of a signal with Hermitian
        ///     symmetry, <c>irfft(conjugate(a), n, axis, norm=_swap_direction(norm))</c>; the output is real.
        /// </summary>
        /// <param name="a">The input (the half of a Hermitian-symmetric signal).</param>
        /// <param name="n">The output length; null for <c>2 * (m - 1)</c>.</param>
        /// <param name="axis">The transformed axis.</param>
        /// <param name="norm">The normalization mode, swapped before the inverse transform.</param>
        /// <param name="out">Accepted for NumPy's signature; NumPy's hfft hands <c>out=None</c> to irfft.</param>
        /// <returns>The real result, float64.</returns>
        /// <exception cref="ValueError">A length below 1, or an unknown norm (the inverse's text).</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> out of range with <paramref name="n"/> omitted.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range with <paramref name="n"/> given.</exception>
        /// <exception cref="TypeError">A text input (np.conjugate has no string loop).</exception>
        public abstract NDArray Hfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.fft.ihfft(a, n=None, axis=-1, norm=None, out=None)</c>: the inverse of <see cref="Hfft"/>,
        ///     <c>conjugate(rfft(a, n, axis, norm=_swap_direction(norm), out=out), out=out)</c>; the output is complex.
        /// </summary>
        /// <param name="a">The real input.</param>
        /// <param name="n">The number of input points used; null for the input's length.</param>
        /// <param name="axis">The transformed axis.</param>
        /// <param name="norm">The normalization mode, swapped before the forward transform.</param>
        /// <param name="out">A pre-allocated complex output, threaded through rfft and the conjugate, or null.</param>
        /// <returns>The half spectrum's conjugate, complex128 (<paramref name="out"/> itself when given).</returns>
        /// <exception cref="ValueError">A length below 1, an unknown norm, or an unfit out.</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> out of range with <paramref name="n"/> omitted.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range with <paramref name="n"/> given.</exception>
        /// <exception cref="TypeError">A complex input (rfft's refusal).</exception>
        /// <exception cref="DTypePromotionError">A text input.</exception>
        /// <exception cref="ArgumentException">An <paramref name="out"/> dtype the complex128 result cannot be cast to.</exception>
        public abstract NDArray Ihfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.fft.fftfreq(n, d=1.0, device=None)</c>: the sample frequencies of an <paramref name="n"/>-point
        ///     transform, <c>[0, 1, ..., (n-1)//2, -(n//2), ..., -1] / (d * n)</c>.
        /// </summary>
        /// <param name="n">The window length.</param>
        /// <param name="d">The sample spacing.</param>
        /// <param name="device">null or <c>"cpu"</c> (the Array API's device argument).</param>
        /// <returns>A new float64 array of <paramref name="n"/> frequencies.</returns>
        /// <exception cref="DivideByZeroException"><c>n * d</c> is zero (Python's <c>float division by zero</c>, checked
        ///     first).</exception>
        /// <exception cref="ValueError">A device other than <c>"cpu"</c>, then a negative <paramref name="n"/>
        ///     (<c>negative dimensions are not allowed</c>).</exception>
        public abstract NDArray Fftfreq(long n, double d = 1.0, string device = null);

        /// <summary>
        ///     NumPy's <c>np.fft.rfftfreq(n, d=1.0, device=None)</c>: the non-negative sample frequencies of an
        ///     <paramref name="n"/>-point real transform, <c>[0, 1, ..., n//2] / (d * n)</c>.
        /// </summary>
        /// <param name="n">The window length (a negative one gives an empty result, as NumPy's <c>arange</c> does).</param>
        /// <param name="d">The sample spacing.</param>
        /// <param name="device">null or <c>"cpu"</c>.</param>
        /// <returns>A new float64 array of <c>n // 2 + 1</c> frequencies.</returns>
        /// <exception cref="DivideByZeroException"><c>n * d</c> is zero.</exception>
        /// <exception cref="ValueError">A device other than <c>"cpu"</c>.</exception>
        public abstract NDArray Rfftfreq(long n, double d = 1.0, string device = null);
    }
}
