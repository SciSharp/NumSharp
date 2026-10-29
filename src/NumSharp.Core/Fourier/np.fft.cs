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
    ///     <para>The transforms and <c>fftfreq</c>/<c>rfftfreq</c> are members of the <see cref="TensorEngine.Fft"/>
    ///     sub-engine (<see cref="FftEngine"/>): each method here keeps its signature and its documentation and makes one
    ///     call into the sub-engine of its operand's engine (hfft behind the class boundary its <c>np.conjugate</c> needs),
    ///     so an engine assigned to <see cref="TensorEngine.Fft"/> computes every function of this module. The default,
    ///     <see cref="Backends.DefaultFftEngine"/>, validates and resolves everything NumPy's Python layer does —
    ///     n/axis/norm/shape/dtype and the N-D→1-D decomposition — then computes through
    ///     <c>PocketFFTDriver.Execute</c>, the managed port of pocketfft's 1-D engine (bit-identical to NumPy 2.4.2 on the
    ///     double/complex128 path). <c>fftshift</c>/<c>ifftshift</c> roll the array object and stay compositions over
    ///     <c>np.roll</c> here.</para>
    /// </summary>
    /// <remarks>https://numpy.org/doc/stable/reference/routines.fft.html</remarks>
    [ModuleName("np.fft")]
    public partial class FourierModule
    {
        internal FourierModule() { }
    }
}
