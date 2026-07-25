using System;
using System.Runtime.CompilerServices;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.Interop.Blas
{
    /// <summary>
    ///     Installs (and describes) the external-CBLAS matrix-product engine.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Referencing this package is the opt-in.</b> A module initializer runs
    ///     <see cref="TryEnable"/> when the assembly loads: if a CBLAS library can be found it
    ///     becomes <see cref="BackendFactory.Default"/> and <c>np.dot</c> / <c>np.matmul</c> compute
    ///     float32/float64 products through it; if none can be found NOTHING changes and NumSharp
    ///     keeps doing its own managed math. Loading never throws at module load — an app that
    ///     merely references the package cannot be broken by a missing native library.
    ///     </para>
    ///     <para>
    ///     Call <see cref="Enable"/> explicitly to pin a specific library or thread count — which is
    ///     what byte-parity with a given NumPy requires, since the result bits depend on both. Call
    ///     <see cref="Disable"/> to go back to NumSharp's own SIMD GEMM.
    ///     </para>
    ///     <example>
    ///     <code>
    ///     // Byte-identical with a pip-installed numpy: name that wheel's own BLAS.
    ///     Blas.Enable(@"…\site-packages\numpy.libs\libscipy_openblas64_-74a4….dll", threads: 1);
    ///     Console.WriteLine(Blas.Info);
    ///     Blas.Disable();                  // back to the built-in managed engine
    ///     </code>
    ///     </example>
    /// </remarks>
    public static class Blas
    {
        /// <summary>True when a <see cref="BlasEngine"/> is currently the default engine.</summary>
        public static bool Enabled => BackendFactory.Default is BlasEngine;

        /// <summary>
        ///     Path, symbol scheme, integer width, thread count and build string of the loaded
        ///     library, or null when none is loaded.
        /// </summary>
        public static string Info
        {
            get
            {
                if (!CBlasNative.IsLoaded)
                    return null;

                var config = CBlasNative.GetConfig();
                return $"{CBlasNative.LibraryPath} [symbols {CBlasNative.SymbolScheme}, " +
                       $"{(CBlasNative.IsIlp64 ? "ILP64" : "LP64")}, threads {CBlasNative.GetNumThreads()}]" +
                       (config == null ? string.Empty : " " + config);
            }
        }

        /// <summary>
        ///     Loads a CBLAS library and makes it NumSharp's matrix-product backend.
        /// </summary>
        /// <param name="library">
        ///     Path of the CBLAS shared library (or a directory holding one), or null to
        ///     auto-discover: the <c>NUMSHARP_PARITY_BLAS</c> environment variable, then the
        ///     <c>numpy.libs</c> folder of any python on PATH, then bare loader names.
        ///     <b>A named library is binding</b> — it is used as given and never silently replaced
        ///     with another, because parity is a claim about one specific binary.
        /// </param>
        /// <param name="threads">
        ///     BLAS worker threads to pin, or 0 to leave the current setting. <b>The result bits
        ///     depend on this</b> — measured: 1, 2, 4 and 24 threads give four different answers —
        ///     so a parity run must pin both sides to the same count.
        /// </param>
        /// <exception cref="DllNotFoundException">No CBLAS library could be located or loaded.</exception>
        /// <exception cref="EntryPointNotFoundException">The library is not a CBLAS provider.</exception>
        public static void Enable(string library = null, int threads = 0)
        {
            if (!CBlasNative.IsLoaded || (library != null && library != CBlasNative.LibraryPath))
                CBlasNative.Load(library);

            if (threads > 0)
                CBlasNative.TrySetNumThreads(threads);

            if (!(BackendFactory.Default is BlasEngine))
                BackendFactory.Default = new BlasEngine();
        }

        /// <summary>
        ///     Uninstalls the engine: NumSharp goes back to its own managed SIMD GEMM. The native
        ///     library stays loaded (unloading a BLAS mid-process is not worth the risk), so a later
        ///     <see cref="Enable"/> is free.
        /// </summary>
        public static void Disable()
        {
            if (BackendFactory.Default is BlasEngine)
                BackendFactory.Default = null;
        }

        /// <summary>
        ///     <see cref="Enable"/> that reports failure instead of throwing — the module-load path.
        /// </summary>
        /// <returns>True when the engine is installed.</returns>
        public static bool TryEnable(string library = null, int threads = 0)
        {
            try
            {
                Enable(library, threads);
                return true;
            }
            catch (DllNotFoundException)
            {
                return false;
            }
            catch (EntryPointNotFoundException)
            {
                return false;
            }
            catch (BadImageFormatException)
            {
                return false;
            }
        }

        /// <summary>
        ///     Runs when this assembly is loaded: referencing the package is all the wiring there is.
        ///     Deliberately silent on failure — a missing native library leaves NumSharp exactly as
        ///     it was, computing everything itself.
        /// </summary>
        [ModuleInitializer]
        internal static void AutoInstall()
        {
            if (string.Equals(Environment.GetEnvironmentVariable("NUMSHARP_BLAS_AUTOINSTALL"), "0",
                    StringComparison.Ordinal))
                return;

            TryEnable();
        }
    }
}
