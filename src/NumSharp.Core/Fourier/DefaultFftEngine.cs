namespace NumSharp.Backends
{
    /// <summary>
    ///     NumSharp's <see cref="FftEngine"/>, the default of <see cref="TensorEngine.Fft"/>: NumPy's <c>numpy/fft/_pocketfft.py</c>
    ///     and <c>numpy/fft/_helper.py</c> statements in NumPy's order - <c>_raw_fft</c>'s length, norm, axis and <c>out=</c>
    ///     checks, <c>_cook_nd_args</c>, <c>_raw_fftnd</c>'s axis order, the Hermitian compositions, <c>fftfreq</c>'s
    ///     arithmetic - over three <c>protected virtual</c> kernels that stand for NumPy's <c>_pocketfft_umath</c> ufuncs:
    ///     <see cref="C2CKernel"/> (<c>fft</c> / <c>ifft</c>), <see cref="R2CKernel"/> (<c>rfft_n_even</c> /
    ///     <c>rfft_n_odd</c>) and <see cref="C2RKernel"/> (<c>irfft</c>).
    /// </summary>
    /// <remarks>
    ///     <para><b>The kernels.</b> Each runs <see cref="PocketFFTDriver.Execute"/>, NumSharp's managed port of pocketfft
    ///     (bit for bit with NumPy 2.4.2's on the double path; float32 / float16 operands reproduce NumPy's complex64 /
    ///     float32 / float16 values in complex128 / float64).</para>
    ///     <para><b>Deriving.</b> Override a kernel to compute the transforms another way (MKL, FFTW, cuFFT): the public
    ///     members keep running NumPy's statements around it, so the override receives a checked length, a normalized axis,
    ///     the normalization factor and a shape-checked <c>out=</c>, and every N-D, real and Hermitian function reaches it.
    ///     Call the base kernel for the operands it does not serve. Assign the instance to <see cref="TensorEngine.Fft"/> of
    ///     the cached engine.</para>
    ///     <para><b>Nested calls.</b> A member that calls another numpy.fft function where NumPy's statement calls the module
    ///     function calls its public facade (hfft's <c>irfft</c>, ihfft's <c>rfft</c>, rfft2's <c>rfftn</c>, irfft2's
    ///     <c>irfftn</c>), so the engine assigned to <see cref="TensorEngine.Fft"/> computes it. The N-D transforms
    ///     (fft2, ifft2, fftn, ifftn, rfftn, irfftn) run this engine's own <c>_raw_fft</c> statements per axis where NumPy
    ///     calls its module <c>fft</c> / <c>rfft</c> / ...: NumSharp's intermediate is complex128 where NumPy's is
    ///     complex64, so only the statements can carry the original operand's precision to every axis.</para>
    ///     <para>The statements were the <c>np.fft</c> module's own bodies (<c>FourierModule</c>, until phase PE's E3 of the
    ///     np-layer dtype plan). The files sit beside the <c>np.fft</c> facades: the statements call np.* functions
    ///     (<c>np.conjugate</c>, <c>np.arange</c>, <c>np.concatenate</c>), which ratchet R2 counts against engine files
    ///     under <c>Backends/</c>, and the kernels sit beside the pocketfft port they call.</para>
    /// </remarks>
    public partial class DefaultFftEngine : FftEngine
    {
        /// <summary>
        ///     The instance <see cref="TensorEngine.Fft"/> returns while nothing else is assigned: shared by every engine,
        ///     because the class holds no state.
        /// </summary>
        internal static readonly DefaultFftEngine Shared = new();
    }
}
