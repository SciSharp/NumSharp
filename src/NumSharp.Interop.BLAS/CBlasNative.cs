using System;
using NumSharp;
using NumSharp.Backends;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;

namespace NumSharp.Interop.Blas
{
    /// <summary>Row/column storage order — mirrors <c>enum CBLAS_ORDER</c> in npy_cblas.h.</summary>
    internal enum CBlasOrder
    {
        RowMajor = 101,
        ColMajor = 102
    }

    /// <summary>Transposition flag — mirrors <c>enum CBLAS_TRANSPOSE</c> in npy_cblas.h.</summary>
    internal enum CBlasTranspose
    {
        NoTrans = 111,
        Trans = 112,
        ConjTrans = 113
    }

    /// <summary>Triangle selector — mirrors <c>enum CBLAS_UPLO</c> in npy_cblas.h.</summary>
    internal enum CBlasUpLo
    {
        Upper = 121,
        Lower = 122
    }

    /// <summary>
    ///     Binding to an external CBLAS shared library, used exclusively by the opt-in
    ///     byte-parity matrix-product backend this package installs (<see cref="Blas"/> →
    ///     <see cref="OpenBlasBackend"/>).
    /// </summary>
    /// <remarks>
    ///     NumSharp's own GEMM is never routed through here — this exists so a caller can hand
    ///     NumSharp the SAME BLAS binary a given NumPy build calls, which is the only way to get
    ///     bit-identical <c>np.dot</c> / <c>np.matmul</c> results out of the two stacks (OpenBLAS'
    ///     multi-accumulator kernels are arch- and thread-count-specific and are not reproducible
    ///     by any portable algorithm — see <c>docs/GEMM_PARITY.md</c>).
    ///     <para>
    ///     Two symbol schemes are supported. NumPy's pip wheels ship an ILP64 build with the
    ///     symbols renamed <c>scipy_</c><i>name</i><c>64_</c> (e.g. <c>scipy_cblas_sgemm64_</c>),
    ///     where the BLAS integer is 64-bit; a stock LP64 OpenBLAS/reference CBLAS exports plain
    ///     <c>cblas_sgemm</c> with a 32-bit integer. Both are probed at load time and the widths
    ///     are marshalled accordingly — see <see cref="IsIlp64"/>.
    ///     </para>
    /// </remarks>
    internal static unsafe class CBlasNative
    {
        private static readonly object SyncRoot = new object();

        private static IntPtr _handle;

        /// <summary>Path of the currently loaded CBLAS library, or null when none is loaded.</summary>
        internal static string LibraryPath { get; private set; }

        /// <summary>True when the loaded library uses 64-bit BLAS integers (numpy's wheels do).</summary>
        internal static bool IsIlp64 { get; private set; }

        /// <summary>The symbol prefix/suffix pair that resolved, for diagnostics (e.g. "scipy_*64_").</summary>
        internal static string SymbolScheme { get; private set; }

        /// <summary>True when a library is loaded and every required symbol bound.</summary>
        internal static bool IsLoaded => _handle != IntPtr.Zero;

        #region function pointers — ILP64 (64-bit BLAS int)

        private static delegate* unmanaged[Cdecl]<int, int, int, long, long, long, float, float*, long, float*, long, float, float*, long, void> _sgemm64;
        private static delegate* unmanaged[Cdecl]<int, int, int, long, long, long, double, double*, long, double*, long, double, double*, long, void> _dgemm64;
        private static delegate* unmanaged[Cdecl]<int, int, long, long, float, float*, long, float*, long, float, float*, long, void> _sgemv64;
        private static delegate* unmanaged[Cdecl]<int, int, long, long, double, double*, long, double*, long, double, double*, long, void> _dgemv64;
        private static delegate* unmanaged[Cdecl]<int, int, int, long, long, float, float*, long, float, float*, long, void> _ssyrk64;
        private static delegate* unmanaged[Cdecl]<int, int, int, long, long, double, double*, long, double, double*, long, void> _dsyrk64;
        private static delegate* unmanaged[Cdecl]<long, float*, long, float*, long, float> _sdot64;
        private static delegate* unmanaged[Cdecl]<long, double*, long, double*, long, double> _ddot64;
        private static delegate* unmanaged[Cdecl]<long, float, float*, long, float*, long, void> _saxpy64;
        private static delegate* unmanaged[Cdecl]<long, double, double*, long, double*, long, void> _daxpy64;

        #endregion

        #region function pointers — LP64 (32-bit BLAS int)

        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, int, float, float*, int, float*, int, float, float*, int, void> _sgemm32;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, int, double, double*, int, double*, int, double, double*, int, void> _dgemm32;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, float, float*, int, float*, int, float, float*, int, void> _sgemv32;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, double, double*, int, double*, int, double, double*, int, void> _dgemv32;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, float, float*, int, float, float*, int, void> _ssyrk32;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, double, double*, int, double, double*, int, void> _dsyrk32;
        private static delegate* unmanaged[Cdecl]<int, float*, int, float*, int, float> _sdot32;
        private static delegate* unmanaged[Cdecl]<int, double*, int, double*, int, double> _ddot32;
        private static delegate* unmanaged[Cdecl]<int, float, float*, int, float*, int, void> _saxpy32;
        private static delegate* unmanaged[Cdecl]<int, double, double*, int, double*, int, void> _daxpy32;

        #endregion

        private static delegate* unmanaged[Cdecl]<int, void> _setNumThreads;
        private static delegate* unmanaged[Cdecl]<int> _getNumThreads;
        private static delegate* unmanaged[Cdecl]<IntPtr> _getConfig;

        /// <summary>
        ///     The BLAS integer ceiling — NumPy's <c>BLAS_MAXSIZE</c>, "-1 to be conservative, in case
        ///     blas internally uses a for loop with an inclusive upper bound".
        /// </summary>
        internal static long BlasMaxSize => IsIlp64 ? long.MaxValue - 1 : int.MaxValue - 1;

        /// <summary>
        ///     NumPy's <c>NPY_CBLAS_CHUNK</c> — the greatest power of two below the BLAS integer max,
        ///     used to split long <c>?dot</c> reductions when npy_intp is wider than the BLAS int.
        /// </summary>
        internal static long CBlasChunk => IsIlp64 ? long.MaxValue : int.MaxValue / 2 + 1;

        /// <summary>
        ///     Loads a CBLAS shared library and binds every symbol the parity backend needs.
        /// </summary>
        /// <param name="path">
        ///     Explicit file path, a directory to search, or null to auto-discover (see
        ///     <see cref="AutoCandidates"/>).
        /// </param>
        /// <remarks>
        ///     <b>A failed load is a no-op.</b> Nothing here touches the currently bound library
        ///     until a replacement has resolved every required symbol, so <c>Load</c> either
        ///     succeeds or leaves the previous binding exactly as it was. Tearing the working
        ///     library down first — which is what this used to do — meant a mistyped path silently
        ///     demoted every subsequent product back to NumSharp's managed kernels while
        ///     <c>Blas.Enabled</c> still said true: the worst outcome a parity feature has, since
        ///     the answer stays plausible.
        /// </remarks>
        /// <exception cref="DllNotFoundException">No candidate library could be loaded.</exception>
        /// <exception cref="EntryPointNotFoundException">The library loaded but is not a CBLAS provider.</exception>
        internal static void Load(string path)
        {
            lock (SyncRoot)
            {
                // An explicitly named library is BINDING: parity is a claim about one specific
                // binary, so quietly falling back to whatever else is installed would produce
                // confidently wrong bits. Only the unattended case searches.
                string requested = !string.IsNullOrWhiteSpace(path)
                    ? path
                    : Environment.GetEnvironmentVariable("NUMSHARP_PARITY_BLAS");
                bool strict = !string.IsNullOrWhiteSpace(requested);

                var tried = new List<string>();
                foreach (var candidate in strict ? Expand(requested) : AutoCandidates())
                {
                    tried.Add(candidate);
                    IntPtr handle;
                    if (!NativeLibrary.TryLoad(candidate, out handle))
                        continue;

                    var previous = _handle;
                    try
                    {
                        // Commits the statics only once every required symbol resolved.
                        Bind(handle, candidate);
                    }
                    catch
                    {
                        NativeLibrary.Free(handle);
                        throw; // previous binding untouched
                    }

                    if (previous != IntPtr.Zero && previous != handle)
                        NativeLibrary.Free(previous);

                    return;
                }

                if (strict)
                    throw new DllNotFoundException(
                        $"Blas.Enable: '{requested}' could not be loaded as a CBLAS library" +
                        (tried.Count == 0
                            ? " (no such file, and no BLAS library in it if it is a directory)."
                            : $" (tried: {string.Join(", ", tried)}).") +
                        " The named library is used as given — parity is a claim about one specific " +
                        "binary, so no other one is substituted. Clear the path (or the " +
                        "NUMSHARP_PARITY_BLAS environment variable) to auto-discover instead.");

                throw new DllNotFoundException(
                    "Blas.Enable: no CBLAS library could be loaded. Pass the path of the very " +
                    "same BLAS binary your NumPy build uses — for a pip wheel that is " +
                    "<python>/Lib/site-packages/numpy.libs/libscipy_openblas64_-*.dll (Windows) or " +
                    "numpy.libs/libscipy_openblas64_-*.so (Linux) — or set the NUMSHARP_PARITY_BLAS " +
                    "environment variable. Tried: " + string.Join(", ", tried));
            }
        }

        /// <summary>Releases the loaded library and clears every bound symbol.</summary>
        /// <remarks>
        ///     Deliberately NOT called by <see cref="Load"/> — replacing a library must not tear the
        ///     working one down before the replacement is bound, and <c>Blas.Disable()</c> only
        ///     uninstalls the backend (freeing a BLAS a thread may be executing in is a fatal
        ///     access violation, not a catchable error). Kept for an explicit teardown.
        /// </remarks>
        internal static void Unload()
        {
            lock (SyncRoot)
            {
                if (_handle != IntPtr.Zero)
                    NativeLibrary.Free(_handle);

                _handle = IntPtr.Zero;
                LibraryPath = null;
                SymbolScheme = null;
                IsIlp64 = false;
                _sgemm64 = null; _dgemm64 = null; _sgemv64 = null; _dgemv64 = null;
                _ssyrk64 = null; _dsyrk64 = null; _sdot64 = null; _ddot64 = null;
                _saxpy64 = null; _daxpy64 = null;
                _sgemm32 = null; _dgemm32 = null; _sgemv32 = null; _dgemv32 = null;
                _ssyrk32 = null; _dsyrk32 = null; _sdot32 = null; _ddot32 = null;
                _saxpy32 = null; _daxpy32 = null;
                _setNumThreads = null; _getNumThreads = null; _getConfig = null;
            }
        }

        /// <summary>
        ///     Resolves every needed entry point in <paramref name="handle"/>, trying the ILP64
        ///     (numpy-wheel) scheme first and falling back to plain LP64 CBLAS names.
        /// </summary>
        /// <remarks>
        ///     Every symbol is resolved into a local BEFORE a single static is written, so this
        ///     either binds completely or not at all. A partial bind is not merely untidy: the
        ///     caller frees the handle when this throws, so a half-written <see cref="_handle"/>
        ///     would leave <see cref="IsLoaded"/> reporting true over freed memory — the next
        ///     <c>Blas.Enable()</c> would skip loading and call into it, and the next
        ///     <see cref="Load"/> would free it a second time.
        /// </remarks>
        private static void Bind(IntPtr handle, string path)
        {
            // (prefix, suffix, ilp64) tuples, most specific first. numpy's wheels rename symbols so
            // an ILP64 OpenBLAS can coexist with a system LP64 one in the same process.
            var schemes = new[]
            {
                new ValueTuple<string, string, bool>("scipy_", "64_", true),
                new ValueTuple<string, string, bool>("", "64_", true),
                new ValueTuple<string, string, bool>("", "_64", true),
                new ValueTuple<string, string, bool>("", "", false)
            };

            foreach (var scheme in schemes)
            {
                string prefix = scheme.Item1, suffix = scheme.Item2;
                bool ilp64 = scheme.Item3;

                IntPtr probe;
                if (!NativeLibrary.TryGetExport(handle, prefix + "cblas_sgemm" + suffix, out probe))
                    continue;

                // Resolve first, commit second. Require() throws when a symbol is missing, and at
                // that point not one static may have been written — see the remarks above.
                IntPtr dgemm = Require(handle, prefix + "cblas_dgemm" + suffix);
                IntPtr sgemv = Require(handle, prefix + "cblas_sgemv" + suffix);
                IntPtr dgemv = Require(handle, prefix + "cblas_dgemv" + suffix);
                IntPtr ssyrk = Require(handle, prefix + "cblas_ssyrk" + suffix);
                IntPtr dsyrk = Require(handle, prefix + "cblas_dsyrk" + suffix);
                IntPtr sdot = Require(handle, prefix + "cblas_sdot" + suffix);
                IntPtr ddot = Require(handle, prefix + "cblas_ddot" + suffix);
                IntPtr saxpy = Require(handle, prefix + "cblas_saxpy" + suffix);
                IntPtr daxpy = Require(handle, prefix + "cblas_daxpy" + suffix);

                _handle = handle;
                LibraryPath = path;
                IsIlp64 = ilp64;
                SymbolScheme = prefix + "*" + suffix;

                if (ilp64)
                {
                    _sgemm64 = (delegate* unmanaged[Cdecl]<int, int, int, long, long, long, float, float*, long, float*, long, float, float*, long, void>)probe;
                    _dgemm64 = (delegate* unmanaged[Cdecl]<int, int, int, long, long, long, double, double*, long, double*, long, double, double*, long, void>)dgemm;
                    _sgemv64 = (delegate* unmanaged[Cdecl]<int, int, long, long, float, float*, long, float*, long, float, float*, long, void>)sgemv;
                    _dgemv64 = (delegate* unmanaged[Cdecl]<int, int, long, long, double, double*, long, double*, long, double, double*, long, void>)dgemv;
                    _ssyrk64 = (delegate* unmanaged[Cdecl]<int, int, int, long, long, float, float*, long, float, float*, long, void>)ssyrk;
                    _dsyrk64 = (delegate* unmanaged[Cdecl]<int, int, int, long, long, double, double*, long, double, double*, long, void>)dsyrk;
                    _sdot64 = (delegate* unmanaged[Cdecl]<long, float*, long, float*, long, float>)sdot;
                    _ddot64 = (delegate* unmanaged[Cdecl]<long, double*, long, double*, long, double>)ddot;
                    _saxpy64 = (delegate* unmanaged[Cdecl]<long, float, float*, long, float*, long, void>)saxpy;
                    _daxpy64 = (delegate* unmanaged[Cdecl]<long, double, double*, long, double*, long, void>)daxpy;
                }
                else
                {
                    _sgemm32 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, int, float, float*, int, float*, int, float, float*, int, void>)probe;
                    _dgemm32 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, int, double, double*, int, double*, int, double, double*, int, void>)dgemm;
                    _sgemv32 = (delegate* unmanaged[Cdecl]<int, int, int, int, float, float*, int, float*, int, float, float*, int, void>)sgemv;
                    _dgemv32 = (delegate* unmanaged[Cdecl]<int, int, int, int, double, double*, int, double*, int, double, double*, int, void>)dgemv;
                    _ssyrk32 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, float, float*, int, float, float*, int, void>)ssyrk;
                    _dsyrk32 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, double, double*, int, double, double*, int, void>)dsyrk;
                    _sdot32 = (delegate* unmanaged[Cdecl]<int, float*, int, float*, int, float>)sdot;
                    _ddot32 = (delegate* unmanaged[Cdecl]<int, double*, int, double*, int, double>)ddot;
                    _saxpy32 = (delegate* unmanaged[Cdecl]<int, float, float*, int, float*, int, void>)saxpy;
                    _daxpy32 = (delegate* unmanaged[Cdecl]<int, double, double*, int, double*, int, void>)daxpy;
                }

                // Optional OpenBLAS extensions — absent on a reference CBLAS, which is fine.
                IntPtr p;
                foreach (var name in new[] { prefix + "openblas_set_num_threads" + suffix, "openblas_set_num_threads", "goto_set_num_threads" })
                    if (NativeLibrary.TryGetExport(handle, name, out p)) { _setNumThreads = (delegate* unmanaged[Cdecl]<int, void>)p; break; }
                foreach (var name in new[] { prefix + "openblas_get_num_threads" + suffix, "openblas_get_num_threads" })
                    if (NativeLibrary.TryGetExport(handle, name, out p)) { _getNumThreads = (delegate* unmanaged[Cdecl]<int>)p; break; }
                foreach (var name in new[] { prefix + "openblas_get_config" + suffix, "openblas_get_config" })
                    if (NativeLibrary.TryGetExport(handle, name, out p)) { _getConfig = (delegate* unmanaged[Cdecl]<IntPtr>)p; break; }

                return;
            }

            throw new EntryPointNotFoundException(
                $"Blas.Enable: '{path}' loaded but exports no recognizable cblas_sgemm symbol " +
                "(tried the scipy_*64_, *64_, *_64 and plain CBLAS naming schemes). It is probably " +
                "not a CBLAS provider.");
        }

        private static IntPtr Require(IntPtr handle, string name)
        {
            IntPtr p;
            if (!NativeLibrary.TryGetExport(handle, name, out p))
                throw new EntryPointNotFoundException(
                    $"Blas.Enable: the loaded BLAS library is missing the required symbol '{name}'.");
            return p;
        }

        /// <summary>
        ///     Candidates for the unattended case (no library named): the numpy wheels reachable
        ///     from any python on PATH, then bare loader names.
        /// </summary>
        private static IEnumerable<string> AutoCandidates()
        {
            foreach (var dir in PythonLibDirectories())
                foreach (var p in Expand(dir))
                    yield return p;

            // Bare names — resolved by the platform loader search path.
            yield return "libscipy_openblas64_";
            yield return "libopenblas64_";
            yield return "libopenblas";
            yield return "openblas";
            yield return "libblas";
        }

        /// <summary>Expands a path that may be a file, or a directory holding a BLAS library.</summary>
        private static IEnumerable<string> Expand(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                yield break;

            if (File.Exists(path))
            {
                yield return path;
                yield break;
            }

            if (!Directory.Exists(path))
                yield break;

            // Newest first: a machine can hold several wheels; the freshest is the likeliest match.
            var patterns = new[] { "*scipy_openblas*", "*openblas*", "*blas*" };
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var pattern in patterns)
            {
                // Discovery runs from a [ModuleInitializer]; an unreadable directory on PATH must
                // not become a TypeInitializationException for the whole assembly.
                string[] found;
                try
                {
                    found = Directory.GetFiles(path, pattern);
                }
                catch (IOException)
                {
                    yield break;
                }
                catch (UnauthorizedAccessException)
                {
                    yield break;
                }

                var files = new List<string>();
                foreach (var f in found)
                {
                    var ext = Path.GetExtension(f);
                    if (ext.Equals(".dll", StringComparison.OrdinalIgnoreCase) ||
                        ext.Equals(".so", StringComparison.OrdinalIgnoreCase) ||
                        ext.Equals(".dylib", StringComparison.OrdinalIgnoreCase) ||
                        f.Contains(".so.", StringComparison.OrdinalIgnoreCase))
                        files.Add(f);
                }

                files.Sort(StringComparer.OrdinalIgnoreCase);
                for (int i = files.Count - 1; i >= 0; i--)
                    if (seen.Add(files[i]))
                        yield return files[i];
            }
        }

        /// <summary>
        ///     Directories that plausibly hold a numpy wheel's bundled BLAS, derived from every
        ///     python interpreter reachable through PATH (plus PYTHONHOME / VIRTUAL_ENV).
        /// </summary>
        private static IEnumerable<string> PythonLibDirectories()
        {
            var roots = new List<string>();
            foreach (var key in new[] { "VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONHOME" })
            {
                var v = Environment.GetEnvironmentVariable(key);
                if (!string.IsNullOrWhiteSpace(v))
                    roots.Add(v);
            }

            var pathVar = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
            bool windows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
            string exe = windows ? "python.exe" : "python3";
            foreach (var entry in pathVar.Split(Path.PathSeparator))
            {
                if (string.IsNullOrWhiteSpace(entry))
                    continue;

                string candidate;
                try
                {
                    candidate = Path.Combine(entry, exe);
                }
                catch (ArgumentException)
                {
                    continue; // malformed PATH entry
                }

                if (File.Exists(candidate))
                {
                    roots.Add(entry);
                    var parent = Path.GetDirectoryName(entry.TrimEnd(Path.DirectorySeparatorChar));
                    if (!string.IsNullOrEmpty(parent))
                        roots.Add(parent); // <venv>/Scripts/python.exe or <prefix>/bin/python3
                }
            }

            foreach (var root in roots)
            {
                // Windows layout.
                yield return Path.Combine(root, "Lib", "site-packages", "numpy.libs");
                // POSIX layout — the python3.X component is unknown, so probe lib/*/site-packages.
                var lib = Path.Combine(root, "lib");
                if (!Directory.Exists(lib))
                    continue;

                string[] pyDirs;
                try
                {
                    pyDirs = Directory.GetDirectories(lib, "python3.*");
                }
                catch (IOException)
                {
                    continue;
                }
                catch (UnauthorizedAccessException)
                {
                    continue;
                }

                foreach (var pyDir in pyDirs)
                    yield return Path.Combine(pyDir, "site-packages", "numpy.libs");
            }
        }

        /// <summary>Pins the BLAS worker-thread count (OpenBLAS only); results depend on it.</summary>
        internal static bool TrySetNumThreads(int threads)
        {
            if (_setNumThreads == null)
                return false;

            _setNumThreads(threads);
            return true;
        }

        /// <summary>The BLAS worker-thread count, or -1 when the library does not report one.</summary>
        internal static int GetNumThreads()
        {
            return _getNumThreads == null ? -1 : _getNumThreads();
        }

        /// <summary>The library's self-description (OpenBLAS build string), or null.</summary>
        internal static string GetConfig()
        {
            return _getConfig == null ? null : Marshal.PtrToStringAnsi(_getConfig());
        }

        #region call wrappers (widths marshalled per IsIlp64)

        internal static void Sgemm(CBlasOrder order, CBlasTranspose transA, CBlasTranspose transB,
            long m, long n, long k, float alpha, float* a, long lda, float* b, long ldb, float beta, float* c, long ldc)
        {
            if (IsIlp64)
                _sgemm64((int)order, (int)transA, (int)transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            else
                _sgemm32((int)order, (int)transA, (int)transB, (int)m, (int)n, (int)k, alpha, a, (int)lda, b, (int)ldb, beta, c, (int)ldc);
        }

        internal static void Dgemm(CBlasOrder order, CBlasTranspose transA, CBlasTranspose transB,
            long m, long n, long k, double alpha, double* a, long lda, double* b, long ldb, double beta, double* c, long ldc)
        {
            if (IsIlp64)
                _dgemm64((int)order, (int)transA, (int)transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            else
                _dgemm32((int)order, (int)transA, (int)transB, (int)m, (int)n, (int)k, alpha, a, (int)lda, b, (int)ldb, beta, c, (int)ldc);
        }

        internal static void Sgemv(CBlasOrder order, CBlasTranspose trans, long m, long n,
            float alpha, float* a, long lda, float* x, long incX, float beta, float* y, long incY)
        {
            if (IsIlp64)
                _sgemv64((int)order, (int)trans, m, n, alpha, a, lda, x, incX, beta, y, incY);
            else
                _sgemv32((int)order, (int)trans, (int)m, (int)n, alpha, a, (int)lda, x, (int)incX, beta, y, (int)incY);
        }

        internal static void Dgemv(CBlasOrder order, CBlasTranspose trans, long m, long n,
            double alpha, double* a, long lda, double* x, long incX, double beta, double* y, long incY)
        {
            if (IsIlp64)
                _dgemv64((int)order, (int)trans, m, n, alpha, a, lda, x, incX, beta, y, incY);
            else
                _dgemv32((int)order, (int)trans, (int)m, (int)n, alpha, a, (int)lda, x, (int)incX, beta, y, (int)incY);
        }

        internal static void Ssyrk(CBlasOrder order, CBlasUpLo uplo, CBlasTranspose trans,
            long n, long k, float alpha, float* a, long lda, float beta, float* c, long ldc)
        {
            if (IsIlp64)
                _ssyrk64((int)order, (int)uplo, (int)trans, n, k, alpha, a, lda, beta, c, ldc);
            else
                _ssyrk32((int)order, (int)uplo, (int)trans, (int)n, (int)k, alpha, a, (int)lda, beta, c, (int)ldc);
        }

        internal static void Dsyrk(CBlasOrder order, CBlasUpLo uplo, CBlasTranspose trans,
            long n, long k, double alpha, double* a, long lda, double beta, double* c, long ldc)
        {
            if (IsIlp64)
                _dsyrk64((int)order, (int)uplo, (int)trans, n, k, alpha, a, lda, beta, c, ldc);
            else
                _dsyrk32((int)order, (int)uplo, (int)trans, (int)n, (int)k, alpha, a, (int)lda, beta, c, (int)ldc);
        }

        internal static float Sdot(long n, float* x, long incX, float* y, long incY)
        {
            return IsIlp64 ? _sdot64(n, x, incX, y, incY) : _sdot32((int)n, x, (int)incX, y, (int)incY);
        }

        internal static double Ddot(long n, double* x, long incX, double* y, long incY)
        {
            return IsIlp64 ? _ddot64(n, x, incX, y, incY) : _ddot32((int)n, x, (int)incX, y, (int)incY);
        }

        internal static void Saxpy(long n, float alpha, float* x, long incX, float* y, long incY)
        {
            if (IsIlp64)
                _saxpy64(n, alpha, x, incX, y, incY);
            else
                _saxpy32((int)n, alpha, x, (int)incX, y, (int)incY);
        }

        internal static void Daxpy(long n, double alpha, double* x, long incX, double* y, long incY)
        {
            if (IsIlp64)
                _daxpy64(n, alpha, x, incX, y, incY);
            else
                _daxpy32((int)n, alpha, x, (int)incX, y, (int)incY);
        }

        #endregion
    }
}
