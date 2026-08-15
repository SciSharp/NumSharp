using System;
using System.Runtime.CompilerServices;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.Interop.OpenBLAS
{
    /// <summary>
    ///     Installs (and describes) the external-CBLAS matrix-product backend.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Referencing this package is the opt-in.</b> A module initializer runs
    ///     <see cref="TryEnable"/> when the assembly loads: if a CBLAS library can be found, an
    ///     <see cref="OpenBlasBackend"/> is assigned to the engine's <see cref="TensorEngine.Blas"/>
    ///     property and <c>np.dot</c> / <c>np.matmul</c> compute float32/float64 products through
    ///     it; if none can be found NOTHING changes and NumSharp keeps doing its own managed math.
    ///     Loading never throws at module load — an app that merely references the package cannot
    ///     be broken by a missing native library.
    ///     </para>
    ///     <para>
    ///     Call <see cref="Enable"/> explicitly to pin a specific library or thread count — which is
    ///     what byte-parity with a given NumPy requires, since the result bits depend on both. Call
    ///     <see cref="Disable"/> to go back to NumSharp's own SIMD GEMM.
    ///     </para>
    ///     <example>
    ///     <code>
    ///     // Byte-identical with a pip-installed numpy: name that wheel's own BLAS.
    ///     OpenBlasEngine.Enable(@"…\site-packages\numpy.libs\libscipy_openblas64_-74a4….dll", threads: 1);
    ///     Console.WriteLine(OpenBlasEngine.Info);
    ///     OpenBlasEngine.Disable();                  // back to the built-in managed engine
    ///     </code>
    ///     </example>
    /// </remarks>
    public static partial class OpenBlasEngine
    {
        /// <summary>
        ///     True when this package's backend is installed on the default engine AND has a CBLAS
        ///     library to call.
        /// </summary>
        /// <remarks>
        ///     Both halves are load-bearing. The backend declines every operand when no library is
        ///     loaded, so an installed-but-unbound backend silently computes nothing at all — every
        ///     product falls back to NumSharp's managed kernels and the answer stays plausible.
        ///     Reporting true there would make this property claim parity it is not delivering.
        /// </remarks>
        public static bool Enabled
            => OpenBlasNative.IsLoaded && BackendFactory.GetEngine().Blas is OpenBlasBackend;

        /// <summary>
        ///     True when the loaded library also provides the LAPACK LU routines the <c>np.linalg</c>
        ///     factorisations need (<c>det</c>/<c>slogdet</c>/<c>solve</c>/<c>inv</c>, and everything
        ///     built on them — <c>matrix_power</c> with a negative exponent, <c>tensorinv</c>,
        ///     <c>tensorsolve</c>).
        /// </summary>
        /// <remarks>
        ///     A full OpenBLAS (including the bundled scipy-openblas) exports LAPACK; a bare reference
        ///     CBLAS does not. When it does not, those functions raise exactly as they would with no
        ///     backend installed, while <c>dot</c>/<c>matmul</c> still accelerate — the factorisations
        ///     have no managed fallback, unlike the products.
        /// </remarks>
        public static bool LapackAvailable => OpenBlasNative.IsLoaded && OpenBlasNative.IsLapackLoaded;

        /// <summary>
        ///     Path, symbol scheme, integer width, thread count and build string of the loaded
        ///     library, or null when none is loaded.
        /// </summary>
        public static string Info
        {
            get
            {
                if (!OpenBlasNative.IsLoaded)
                    return null;

                var config = OpenBlasNative.GetConfig();
                return $"{OpenBlasNative.LibraryPath} [symbols {OpenBlasNative.SymbolScheme}, " +
                       $"{(OpenBlasNative.IsIlp64 ? "ILP64" : "LP64")}, threads {OpenBlasNative.GetNumThreads()}]" +
                       (config == null ? string.Empty : " " + config);
            }
        }

        /// <summary>
        ///     Loads a CBLAS library and installs it as the engine's matrix-product backend.
        /// </summary>
        /// <param name="library">
        ///     Path of the CBLAS shared library (or a directory holding one), or null to
        ///     auto-discover: the <c>NUMSHARP_OPENBLAS_LIBRARY</c> environment variable (binding), then
        ///     the override path(s) (<c>NUMSHARP_OPENBLAS_SEARCH_PATH</c> / a build-recorded
        ///     <c>OpenBlasPath</c>), then a build-staged version override (required — a miss
        ///     throws), then an explicit OpenBLAS root (<c>OPENBLAS_HOME</c>/<c>OPENBLAS_ROOT</c>),
        ///     then the bundled runtime asset, then ambient machine tooling (system install
        ///     directories, bare loader names, a PATH sweep).
        ///     <b>A named library is binding</b> — it is used as given and never silently replaced
        ///     with another, because parity is a claim about one specific binary.
        /// </param>
        /// <param name="threads">
        ///     BLAS worker threads to pin, or 0 to leave the current setting. <b>The result bits
        ///     depend on this</b> — measured: 1, 2, 4 and 24 threads give four different answers —
        ///     so a parity run must pin both sides to the same count.
        /// </param>
        /// <remarks>
        ///     <b>A failed call is a no-op.</b> When the requested library cannot be loaded this
        ///     throws having changed nothing — an already-working parity setup keeps working, and a
        ///     process that had none still has none. (It used to unload the good library first, so
        ///     a mistyped path demoted every subsequent product back to the managed kernels while
        ///     <see cref="Enabled"/> still said true.)
        /// </remarks>
        /// <param name="coreType">
        ///     The OpenBLAS <c>DYNAMIC_ARCH</c> micro-kernel to force (e.g. <see cref="CoreType"/>),
        ///     or null to let OpenBLAS pick the best one this CPU supports — which is what NumPy
        ///     does, and therefore the right default for matching a NumPy on the same machine.
        ///     Pin it only to make results reproducible across DIFFERENT machines; see
        ///     <see cref="CoreType"/> for the trade-off and the hazard.
        /// </param>
        /// <exception cref="DllNotFoundException">No CBLAS library could be located or loaded.</exception>
        /// <exception cref="EntryPointNotFoundException">The library is not a CBLAS provider.</exception>
        /// <exception cref="PlatformNotSupportedException">
        ///     <paramref name="coreType"/> names kernels this CPU cannot execute.
        /// </exception>
        public static void Enable(string library = null, int threads = 0, string coreType = null)
        {
            if (!OpenBlasNative.IsLoaded || (library != null && library != OpenBlasNative.LibraryPath))
            {
                OpenBlasNative.Load(library, coreType);
            }
            else if (!string.IsNullOrWhiteSpace(coreType))
            {
                // The library is already loaded, so its micro-kernel is already chosen and this
                // request cannot be honoured. Dropping it silently would leave the caller believing
                // the kernel is pinned while the bits are whatever this CPU picked — so it either
                // already holds or it throws.
                OpenBlasNative.VerifyCoreTypeAlreadyInEffect(coreType);
            }

            if (threads > 0)
                OpenBlasNative.TrySetNumThreads(threads);

            if (!(BackendFactory.GetEngine().Blas is OpenBlasBackend))
                BackendFactory.GetEngine().Blas = new OpenBlasBackend();
        }

        /// <summary>
        ///     The OpenBLAS micro-kernel the committed <c>matmul_parity</c> corpus was generated
        ///     with, and the one to pin for results that reproduce on another machine.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     Bundling the binary fixes only two of the three things the result bits depend on.
        ///     The library build is now pinned by the package and the worker-thread count by
        ///     <see cref="Enable"/>'s <c>threads</c> argument — but OpenBLAS is built
        ///     <c>DYNAMIC_ARCH</c>, so it still chooses a micro-kernel from the CPU it finds itself
        ///     on, and different kernels accumulate in a different order. Measured on one host by
        ///     forcing the choice: Haswell, Nehalem, Sandybridge and Katmai each produced different
        ///     bytes for the same float32 product. So "same binary" is necessary and not sufficient.
        ///     </para>
        ///     <para>
        ///     Passing this as <c>coreType</c> closes that last gap for any x86-64 CPU with AVX2 and
        ///     FMA (Haswell 2013 and later, and every AMD Zen), which is what makes the host-pinned
        ///     corpus replayable on a machine that would otherwise dispatch a newer kernel.
        ///     </para>
        ///     <para>
        ///     <b>It is NOT the default, deliberately.</b> The package's main promise is matching the
        ///     NumPy on the same machine, and that NumPy dispatches by CPU — so pinning here would
        ///     CREATE a divergence on any host newer than Haswell. Pin both sides or neither: the
        ///     same <c>OPENBLAS_CORETYPE</c> variable drives NumPy too, because it is the same
        ///     library.
        ///     </para>
        ///     <para>
        ///     <b>Hazard:</b> this overrides OpenBLAS' CPU detection rather than requesting a
        ///     preference, so naming kernels the CPU cannot run terminates the process with an
        ///     illegal instruction. <see cref="Enable"/> rejects the combinations it can recognise
        ///     with a <see cref="PlatformNotSupportedException"/> first.
        ///     </para>
        ///     <para>
        ///     Only effective when the call actually loads the library: OpenBLAS reads the variable
        ///     while initialising and never re-reads it, so it cannot be changed afterwards.
        ///     </para>
        /// </remarks>
        public const string CoreType = "Haswell";

        /// <summary>
        ///     The <c>DYNAMIC_ARCH</c> micro-kernel family OpenBLAS actually dispatched to
        ///     (e.g. "Haswell"), or null when no library is loaded or it does not report one.
        /// </summary>
        public static string CoreName => OpenBlasNative.IsLoaded ? OpenBlasNative.GetCoreName() : null;

        /// <summary>
        ///     Full path of the CBLAS library currently loaded, or null when none is.
        /// </summary>
        /// <remarks>
        ///     Exposed because "which binary answered?" is a correctness question here, not a
        ///     diagnostic one — the file this points at is what a parity claim is about, and it is
        ///     the input to reproducing or hashing a run. <see cref="Info"/> embeds it in prose;
        ///     this is the machine-readable form.
        /// </remarks>
        public static string LibraryPath => OpenBlasNative.IsLoaded ? OpenBlasNative.LibraryPath : null;

        /// <summary>
        ///     True when the loaded library is the one this package bundles (as opposed to one
        ///     discovered on the machine, staged by a build-time override, or named by the caller).
        /// </summary>
        /// <remarks>
        ///     A build-staged version/path override lands in the same
        ///     <c>runtimes/&lt;rid&gt;/native/</c> layout as the bundle — and the delivery design's
        ///     "same binary" invariant can make the two content-identical — so the folder heuristic
        ///     alone cannot tell them apart. The source marker the build writes
        ///     (<c>openblas.source.json</c>) is what does: a folder it declares an override
        ///     read-location is an override, not the bundle.
        /// </remarks>
        public static bool IsBundledLibrary
        {
            get
            {
                var path = LibraryPath;
                if (string.IsNullOrEmpty(path))
                    return false;

                var dir = System.IO.Path.GetDirectoryName(path);
                if (string.IsNullOrEmpty(dir) ||
                    !System.IO.Path.GetFileName(dir).Equals("native", StringComparison.OrdinalIgnoreCase) ||
                    dir.IndexOf("runtimes", StringComparison.OrdinalIgnoreCase) < 0)
                    return false;

                return !OpenBlasSourceMarker.DeclaresOverrideFor(dir);
            }
        }

        /// <summary>
        ///     Uninstalls the backend: NumSharp goes back to its own managed SIMD GEMM. The native
        ///     library stays loaded (unloading a BLAS mid-process is not worth the risk), so a later
        ///     <see cref="Enable"/> is free.
        /// </summary>
        public static void Disable()
        {
            var engine = BackendFactory.GetEngine();
            if (engine.Blas is OpenBlasBackend)
                engine.Blas = null;
        }

        /// <summary>
        ///     <see cref="Enable"/> that reports failure instead of throwing — the module-load path.
        /// </summary>
        /// <returns>True when the backend is installed.</returns>
        /// <remarks>
        ///     Catches EVERYTHING on purpose. This runs from a <c>[ModuleInitializer]</c>, where an
        ///     escaping exception is not a failed opt-in but a <c>TypeInitializationException</c> on
        ///     every later touch of this assembly — i.e. merely referencing the package would break
        ///     the app, which is the one thing the design promises it cannot do. Library discovery
        ///     walks PATH and the filesystem, so the reachable failures are not limited to the
        ///     three loader exceptions: an unreadable directory alone raises
        ///     <see cref="System.UnauthorizedAccessException"/>.
        /// </remarks>
        public static bool TryEnable(string library = null, int threads = 0)
        {
            try
            {
                Enable(library, threads);
                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }

        /// <summary>
        ///     Runs when this assembly is loaded: referencing the package is all the wiring there
        ///     is. Named for its defining default job — making the BUNDLED runtime asset work with
        ///     zero configuration — though it runs the whole discovery (override path → staged
        ///     version override → bundle → machine tooling). Deliberately silent on failure — a
        ///     missing native library leaves NumSharp exactly as it was, computing everything
        ///     itself — with ONE exception: a version override the build staged as a hard
        ///     requirement that fails to load is reported to stderr (the backend still stays
        ///     uninstalled rather than substituted, and an explicit <see cref="Enable"/> throws
        ///     <see cref="OpenBlasRequiredOverrideException"/> with the same message).
        /// </summary>
        /// <remarks>
        ///     Opt out with <c>NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL=0</c>. (The pre-rename
        ///     <c>NUMSHARP_BLAS_BUNDLE_AUTOINSTALL</c> and the earlier <c>NUMSHARP_BLAS_AUTOINSTALL</c>
        ///     spellings are RETIRED and deliberately ignored — each rename shipped unreleased, so
        ///     there is no installed base to migrate.)
        /// </remarks>
        [ModuleInitializer]
        internal static void BundleAutoinstall()
        {
            if (!EnvVars.OpenBlasBundleAutoinstall)
                return;

            try
            {
                Enable();
            }
            catch (OpenBlasRequiredOverrideException e)
            {
                // The consumer PINNED a version at build time; silently running with different bits
                // is the one failure a pin exists to prevent. Throwing here would surface as a
                // TypeInitializationException at some unrelated first touch of this assembly, so:
                // loud on stderr, backend left uninstalled (never substituted), and any explicit
                // OpenBlasEngine.Enable() raises the full exception.
                Console.Error.WriteLine("NumSharp.Interop.OpenBLAS: " + e.Message);
            }
            catch (Exception)
            {
                // Everything else is the ordinary no-BLAS-anywhere case: an app that merely
                // references the package cannot be broken by a missing native library.
            }
        }
    }
}
