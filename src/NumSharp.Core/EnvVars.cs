using System;

namespace NumSharp
{
    /// <summary>
    ///     Central, typed accessor for every environment variable NumSharp reads. Each property RETURNS
    ///     THE VALUE (not the name), read through the shared <see cref="Get"/> / <see cref="GetBool"/>
    ///     helpers, so null/blank handling, trimming and boolean parsing live in ONE place. Read-sites
    ///     use e.g. <c>if (EnvVars.DebugGuardPages)</c> or <c>var lib = EnvVars.OpenBlasLibrary;</c>
    ///     instead of repeating <c>Environment.GetEnvironmentVariable(...)</c> and its parsing.
    /// </summary>
    /// <remarks>
    ///     Grouped by area. <c>NUMSHARP_*</c> names are NumSharp's own; the <b>discovery</b> section reads
    ///     third-party conventions (conda / vcpkg / OpenBLAS / Python.NET) that NumSharp honours. The
    ///     <b>build-time</b> variables are normally consumed by MSBuild (the package's
    ///     <c>buildTransitive</c> props/targets), not by managed code; their accessors here simply read
    ///     whatever is in the current process environment, for completeness and diagnostics.
    ///     <para>This type only READS. Setting/clearing a variable is done by NAME through
    ///     <c>Environment.SetEnvironmentVariable</c> (e.g. in tests) and is deliberately out of scope.
    ///     The retired, now-ignored spellings <c>NUMSHARP_BLAS_BUNDLE_AUTOINSTALL</c> /
    ///     <c>NUMSHARP_BLAS_AUTOINSTALL</c> have no accessor — they carry no value.</para>
    /// </remarks>
    public static class EnvVars
    {
        /// <summary>Shared reader: the trimmed value of <paramref name="name"/>, or <c>null</c> when unset or blank.</summary>
        private static string Get(string name)
        {
            string v = Environment.GetEnvironmentVariable(name);
            return string.IsNullOrWhiteSpace(v) ? null : v.Trim();
        }

        /// <summary>
        ///     Shared boolean reader: returns <paramref name="defaultValue"/> when the variable is unset,
        ///     blank, or unrecognized; <c>1</c>/<c>true</c>/<c>yes</c>/<c>on</c> → <c>true</c>;
        ///     <c>0</c>/<c>false</c>/<c>no</c>/<c>off</c> → <c>false</c> (all case-insensitive).
        /// </summary>
        private static bool GetBool(string name, bool defaultValue)
        {
            string v = Get(name);
            if (v is null) return defaultValue;
            return v.ToLowerInvariant() switch
            {
                "1" or "true" or "yes" or "on" => true,
                "0" or "false" or "no" or "off" => false,
                _ => defaultValue,
            };
        }

        /// <summary>
        ///     Shared integer reader: returns <paramref name="defaultValue"/> when the variable is unset,
        ///     blank, or not a valid invariant-culture integer.
        /// </summary>
        private static int GetInt(string name, int defaultValue)
        {
            string v = Get(name);
            if (v is null) return defaultValue;
            return int.TryParse(v, System.Globalization.NumberStyles.Integer,
                System.Globalization.CultureInfo.InvariantCulture, out int parsed)
                ? parsed
                : defaultValue;
        }

        #region Core — runtime

        /// <summary>
        ///     Opt-in Windows page-heap diagnostic for the unmanaged buffer pool (env
        ///     <c>NUMSHARP_DEBUG_GUARD_PAGES=1</c>): every pooled buffer is handed back with its last byte
        ///     abutting an inaccessible guard page, so an out-of-bounds access faults INSTANTLY instead of
        ///     silently corrupting a neighbour. <b>Default:</b> <c>false</c> (Windows only — no effect
        ///     elsewhere).
        /// </summary>
        public static bool DebugGuardPages => GetBool("NUMSHARP_DEBUG_GUARD_PAGES", false);

        /// <summary>
        ///     Whether the unmanaged buffer pool may PACE the GC — request a forced gen-0 collection when
        ///     the pool is starving (a Take misses while enough bytes have been handed out since the last
        ///     paced collection). This is NumSharp's stand-in for CPython's refcount-prompt free: dropped,
        ///     undisposed result arrays return their native buffers only via finalizers, which run only
        ///     after a GC — without pacing a tight allocating loop starves the pool for thousands of calls
        ///     and pays a cold alloc + first-touch page faults per op. Env <c>NUMSHARP_POOL_GC_PACING</c>.
        ///     <b>Default:</b> <c>true</c>; set <c>=0</c> to never trigger collections from the pool
        ///     (deterministic <c>Dispose</c>/<c>NDScope</c> workloads don't need pacing — they also never
        ///     starve, so pacing self-disables there anyway).
        /// </summary>
        public static bool PoolGcPacing => GetBool("NUMSHARP_POOL_GC_PACING", true);

        /// <summary>
        ///     The pool-pacing window in MiB — how many bytes the pool hands out between pool-initiated
        ///     gen-0 collections (and, equally, the per-bucket resident budget for pooled warm buffers).
        ///     Larger = fewer forced collections but a larger warm set and longer starvation bursts.
        ///     Env <c>NUMSHARP_POOL_GC_PACING_MB</c>. <b>Default:</b> <c>16</c> (clamped to 1..1024).
        /// </summary>
        public static int PoolGcPacingMB => GetInt("NUMSHARP_POOL_GC_PACING_MB", 16);

        #endregion

        #region OpenBLAS backend — runtime (package: NumSharp.Interop.OpenBLAS)

        /// <summary>
        ///     Absolute path to ONE specific OpenBLAS/CBLAS shared library to bind — a parity pin: the
        ///     named binary is used exactly as given and is NEVER silently substituted, not even by the
        ///     bundled copy. Env <c>NUMSHARP_OPENBLAS_LIBRARY</c>. <b>Default:</b> <c>null</c> → normal
        ///     multi-tier discovery runs.
        ///     <para><b>Value examples</b> (a full path to the library FILE, not a directory):</para>
        ///     <list type="bullet">
        ///       <item><description>Windows: <c>C:\opt\openblas\scipy_openblas64.dll</c></description></item>
        ///       <item><description>Linux: <c>/usr/lib/x86_64-linux-gnu/libopenblas.so.0</c></description></item>
        ///       <item><description>macOS: <c>/opt/homebrew/opt/openblas/lib/libopenblas.dylib</c></description></item>
        ///     </list>
        /// </summary>
        public static string OpenBlasLibrary => Get("NUMSHARP_OPENBLAS_LIBRARY");

        /// <summary>
        ///     Extra, highest-priority (but NON-binding) directories to search for an OpenBLAS library,
        ///     tried before the bundled asset; a match here carries no parity guarantee. Env
        ///     <c>NUMSHARP_OPENBLAS_SEARCH_PATH</c>. <b>Default:</b> <c>null</c> → no extra search dirs.
        ///     <para><b>Value examples</b> (a directory — or several joined by the platform path
        ///     separator: <c>;</c> on Windows, <c>:</c> elsewhere): <c>C:\opt\openblas\bin</c>,
        ///     <c>/opt/openblas/lib</c>. Also read at BUILD time by the buildTransitive targets.</para>
        /// </summary>
        public static string OpenBlasSearchPath => Get("NUMSHARP_OPENBLAS_SEARCH_PATH");

        /// <summary>
        ///     Whether the bundled OpenBLAS asset participates in discovery as one tier. Env
        ///     <c>NUMSHARP_OPENBLAS_USE_BUNDLED</c>. <b>Default:</b> <c>true</c> (the bundle is included —
        ///     the parity default); set <c>=0</c> to drop the bundled tier entirely. A toggle for the
        ///     bundled TIER, not a "use ONLY the bundle" switch.
        /// </summary>
        public static bool OpenBlasUseBundled => GetBool("NUMSHARP_OPENBLAS_USE_BUNDLED", true);

        /// <summary>
        ///     Whether the module initializer auto-installs the bundled OpenBLAS on load. Env
        ///     <c>NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL</c>. <b>Default:</b> <c>true</c>; set <c>=0</c> to
        ///     skip it (bind later via <c>OpenBlasEngine.Enable</c> or a discovery variable above).
        /// </summary>
        public static bool OpenBlasBundleAutoinstall => GetBool("NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL", true);

        #endregion

        #region OpenBLAS backend — build-time (normally MSBuild buildTransitive; accessor reads the process env)

        /// <summary>Build-time: scipy-openblas version to download and stage over the bundle (e.g. <c>0.3.31.22.0</c>). Env <c>NUMSHARP_OPENBLAS_VERSION</c>, beats the <c>&lt;OpenBlasVersion&gt;</c> MSBuild property. <b>Default:</b> <c>null</c> → the pinned bundled version.</summary>
        public static string OpenBlasVersion => Get("NUMSHARP_OPENBLAS_VERSION");

        /// <summary>Build-time: distribution to fetch — <c>64</c> (ILP64) or <c>32</c> (LP64). Env <c>NUMSHARP_OPENBLAS_DISTRIBUTION</c>. <b>Default:</b> <c>null</c> → the per-RID default.</summary>
        public static string OpenBlasDistribution => Get("NUMSHARP_OPENBLAS_DISTRIBUTION");

        /// <summary>Build-time: PyPI mirror base URL for the scipy-openblas wheel (needs a matching <see cref="OpenBlasSha256"/>). Env <c>NUMSHARP_OPENBLAS_PYPI_FEED_URL</c>. <b>Default:</b> <c>null</c> → the default PyPI feed. Example: <c>https://pypi.org/simple</c>.</summary>
        public static string OpenBlasPypiFeedUrl => Get("NUMSHARP_OPENBLAS_PYPI_FEED_URL");

        /// <summary>Build-time: expected sha256 of the extracted OpenBLAS library, verified after download. Env <c>NUMSHARP_OPENBLAS_SHA256</c>. <b>Default:</b> <c>null</c> → the checked-in manifest pin.</summary>
        public static string OpenBlasSha256 => Get("NUMSHARP_OPENBLAS_SHA256");

        /// <summary>Build-time: delivery mode — <c>none</c>, <c>build</c>, or <c>package</c>. Env <c>NUMSHARP_OPENBLAS_DELIVERY</c>. <b>Default:</b> <c>null</c> → <c>build</c>.</summary>
        public static string OpenBlasDelivery => Get("NUMSHARP_OPENBLAS_DELIVERY");

        #endregion

        #region OpenBLAS discovery — third-party conventions NumSharp honours (names not owned by NumSharp)

        /// <summary>External <c>OPENBLAS_HOME</c>: an explicit OpenBLAS install root; its base, <c>/lib</c> and <c>/bin</c> are searched ABOVE the bundle (not byte-parity with NumPy). <b>Default:</b> <c>null</c>. Example: <c>/opt/OpenBLAS</c>.</summary>
        public static string OpenBlasHome => Get("OPENBLAS_HOME");

        /// <summary>External <c>OPENBLAS_ROOT</c>: alternate spelling of <see cref="OpenBlasHome"/>. <b>Default:</b> <c>null</c>.</summary>
        public static string OpenBlasRoot => Get("OPENBLAS_ROOT");

        /// <summary>
        ///     External <c>OPENBLAS_CORETYPE</c>: pins scipy-openblas' DYNAMIC_ARCH kernel (e.g.
        ///     <c>Haswell</c>). <b>WARNING:</b> the native library reads this through the CRT at load, and
        ///     .NET keeps its own env table — so THIS accessor may not reflect what the native lib saw.
        ///     Exposed for diagnostics; to actually pin the kernel, set it through the CRT before load.
        /// </summary>
        public static string OpenBlasCoreType => Get("OPENBLAS_CORETYPE");

        /// <summary>External <c>CONDA_PREFIX</c>: an active conda prefix; its <c>lib</c> / <c>Library\bin</c> / <c>Library\lib</c> are scanned as ambient tooling (below the bundle). <b>Default:</b> <c>null</c>.</summary>
        public static string CondaPrefix => Get("CONDA_PREFIX");

        /// <summary>External <c>VCPKG_ROOT</c> (Windows): a vcpkg root; its <c>installed\{triplet}\bin</c> dirs are scanned as ambient tooling (below the bundle). <b>Default:</b> <c>null</c>.</summary>
        public static string VcpkgRoot => Get("VCPKG_ROOT");

        /// <summary>External <c>PATH</c>: the process search path, swept dead-last for a CBLAS whose file name is not a bare loader name.</summary>
        public static string SystemPath => Get("PATH");

        #endregion

        #region Python.NET interop (test suite: NumSharp.Tests.Interop)

        /// <summary>External <c>PYTHONNET_PYDLL</c>: absolute path to the libpython shared library Python.NET should host. <b>Default:</b> <c>null</c> → the suite probes common interpreters for one that imports numpy. Example: <c>/usr/lib/x86_64-linux-gnu/libpython3.12.so</c>.</summary>
        public static string PythonNetPyDll => Get("PYTHONNET_PYDLL");

        /// <summary>
        ///     Whether an unavailable Python+numpy engine is a HARD test failure rather than an
        ///     Inconclusive skip. Env <c>NUMSHARP_PYTHONNET_REQUIRE_ENGINE</c>. <b>Default:</b> <c>true</c>
        ///     — the interop suite exists to exercise the engine, so an absent one means discovery is
        ///     broken and must go red, not green-by-skipping. Set <c>=0</c> / <c>false</c> / <c>no</c> /
        ///     <c>off</c> from outside to allow a graceful skip on a machine without Python.
        /// </summary>
        public static bool PythonnetRequireEngine => GetBool("NUMSHARP_PYTHONNET_REQUIRE_ENGINE", true);

        #endregion

        #region Benchmarks

        /// <summary>Env <c>NUMSHARP_BENCH_NDITER_SECTION</c>: selects a single section of the nditer benchmark to run in its own process. <b>Default:</b> <c>null</c> → callers treat as <c>"all"</c> (run every section). Example: a section name printed by the benchmark's own index.</summary>
        public static string BenchNditerSection => Get("NUMSHARP_BENCH_NDITER_SECTION");

        #endregion
    }
}
