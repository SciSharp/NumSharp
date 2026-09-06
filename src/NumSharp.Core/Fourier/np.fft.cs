namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Discrete Fourier Transform namespace — the port of Python's <c>numpy.fft</c> module.
        ///     Accessed exactly like Python: <c>np.fft.fft(x)</c>, <c>np.fft.rfft(x)</c>,
        ///     <c>np.fft.fftfreq(n)</c>, etc. Mirrors the <see cref="random"/> facade shape (a lowercase
        ///     property returning a module object whose methods are the functions).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/routines.fft.html</remarks>
        public static FourierModule fft { get; } = new FourierModule();
    }

    /// <summary>
    ///     The <c>numpy.fft</c> module surface, reachable as <see cref="np.fft"/>. Holds the 18 public
    ///     transforms/helpers (standard <c>fft/ifft/fft2/ifft2/fftn/ifftn</c>, real
    ///     <c>rfft/irfft/rfft2/irfft2/rfftn/irfftn</c>, hermitian <c>hfft/ihfft</c>, and helpers
    ///     <c>fftfreq/rfftfreq/fftshift/ifftshift</c>).
    ///
    ///     <para>The helpers (<c>fftfreq</c>/<c>rfftfreq</c>/<c>fftshift</c>/<c>ifftshift</c>) are pure
    ///     compositions of existing <c>np.*</c> functions. The transforms validate and resolve
    ///     everything NumPy's Python layer does — n/axis/norm/shape/dtype and the N-D→1-D decomposition
    ///     — then compute through <c>PocketFFTDriver.Execute</c>, the managed port of pocketfft's 1-D
    ///     engine (bit-identical to NumPy 2.4.2 on the double/complex128 path). The N-D wrappers are
    ///     pure compositions of the 1-D transforms.</para>
    /// </summary>
    /// <remarks>https://numpy.org/doc/stable/reference/routines.fft.html</remarks>
    [ModuleName("np.fft")]
    public partial class FourierModule
    {
        internal FourierModule() { }
    }
}
