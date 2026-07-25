using System;
using NumSharp.Backends.Kernels.Blas;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Enable or disable the byte-parity matrix-product backend: <c>np.dot</c> and
        ///     <c>np.matmul</c> then reproduce NumPy's float32 / float64 results BIT-FOR-BIT by
        ///     calling the very same CBLAS binary NumPy calls, through a route-for-route port of
        ///     NumPy 2.4.2's two matrix-product dispatchers.
        /// </summary>
        /// <param name="enabled">
        ///     True to route <c>np.dot</c> / <c>np.matmul</c> through the parity backend, false to go
        ///     back to NumSharp's own SIMD GEMM (the default).
        /// </param>
        /// <param name="library">
        ///     Path of the CBLAS shared library (or a directory holding one). Null auto-discovers:
        ///     the <c>NUMSHARP_PARITY_BLAS</c> environment variable, then the <c>numpy.libs</c>
        ///     folder of any python on PATH, then bare loader names (<c>libopenblas</c>, …).
        ///     For parity with a pip-installed NumPy this MUST be that wheel's own
        ///     <c>numpy.libs/libscipy_openblas64_-*.dll</c> (<c>.so</c> on Linux).
        /// </param>
        /// <param name="threads">
        ///     BLAS worker threads to pin, or 0 to leave the current setting (the library's own
        ///     default until something changes it). <b>The result bits depend on this</b> —
        ///     measured: 1, 2, 4 and 24 threads give four different answers — so
        ///     both sides must agree (NumPy's default is also the library default; pin both to 1 with
        ///     <c>OPENBLAS_NUM_THREADS=1</c> / <c>threads: 1</c> if you want a host-independent run).
        /// </param>
        /// <remarks>
        ///     <para>
        ///     Why this exists: NumPy computes float matrix products with cblas, and scipy-openblas'
        ///     <c>sgemm</c> accumulates in an arch-specific multi-accumulator register scheme that
        ///     matches neither a sequential mul+add chain nor a sequential FMA chain. No portable
        ///     algorithm reproduces those bits, so a workload that must agree with NumPy to the last
        ///     bit (e.g. training a network twice and byte-comparing the weights) has to call the
        ///     same binary. NumSharp's own GEMM is untouched and remains the default and the fast
        ///     path; this is an opt-in seam for parity harnesses.
        ///     </para>
        ///     <para>
        ///     Covered: <c>float32</c> and <c>float64</c>, every route of both dispatchers — gemm
        ///     (incl. the <c>a @ a.T</c> syrk shortcut and the copy-if-not-blasable rule), both gemv
        ///     directions, the row·column <c>?dot</c>, the portable no-blas loop, N-D <c>np.dot</c>'s
        ///     per-element <c>?dot</c>, and stacked <c>np.matmul</c>. Any other dtype (including
        ///     <c>Complex</c> and <c>Half</c>) silently keeps the native path — integer products are
        ///     bit-exact by construction anyway.
        ///     </para>
        ///     <example>
        ///     <code>
        ///     np.parity_matmul(true);                       // auto-discover numpy's OpenBLAS
        ///     np.parity_matmul(true, @"C:\...\numpy.libs\libscipy_openblas64_-74a4….dll", threads: 1);
        ///     np.parity_matmul(false);                      // back to NumSharp's SIMD GEMM
        ///     </code>
        ///     </example>
        /// </remarks>
        /// <exception cref="DllNotFoundException">No CBLAS library could be located or loaded.</exception>
        /// <exception cref="EntryPointNotFoundException">The library is not a CBLAS provider.</exception>
        public static void parity_matmul(bool enabled, string library = null, int threads = 0)
        {
            if (!enabled)
            {
                BlasParity.Enabled = false;
                return;
            }

            if (!CBlasNative.IsLoaded || (library != null && library != CBlasNative.LibraryPath))
                CBlasNative.Load(library);

            if (threads > 0)
                CBlasNative.TrySetNumThreads(threads);

            BlasParity.Enabled = true;
        }

        /// <summary>
        ///     Whether the byte-parity matrix-product backend is currently servicing <c>np.dot</c>
        ///     and <c>np.matmul</c>.
        /// </summary>
        public static bool parity_matmul_enabled => BlasParity.Enabled;

        /// <summary>
        ///     A one-line description of the loaded parity BLAS — path, symbol scheme, integer width,
        ///     thread count and the library's own build string — or null when none is loaded.
        /// </summary>
        public static string parity_matmul_info()
        {
            if (!CBlasNative.IsLoaded)
                return null;

            var config = CBlasNative.GetConfig();
            return $"{CBlasNative.LibraryPath} [symbols {CBlasNative.SymbolScheme}, " +
                   $"{(CBlasNative.IsIlp64 ? "ILP64" : "LP64")}, threads {CBlasNative.GetNumThreads()}]" +
                   (config == null ? string.Empty : " " + config);
        }
    }
}
