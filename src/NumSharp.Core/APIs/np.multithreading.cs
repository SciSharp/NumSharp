using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Enable or disable NumSharp's multithreaded kernels and cap the worker thread count.
        /// </summary>
        /// <param name="enabled">Whether kernels are allowed to use more than one thread.</param>
        /// <param name="max_threads">
        ///     Upper bound on worker threads (clamped to at least 1 and to the processor count).
        ///     Defaults to 8.
        /// </param>
        /// <remarks>
        ///     Two classes of kernel are controlled here:
        ///     <para>
        ///     ORDER-SENSITIVE kernels (the fused 1-D dot product,
        ///     <see cref="dot(NDArray,NDArray)"/> vector·vector on contiguous <c>float</c>/<c>double</c>)
        ///     are <b>disabled by default</b> — enabling them sums per-chunk and combines, so the result
        ///     may differ in the last few ULPs (the same reordering NumPy's threaded BLAS exhibits).
        ///     Only large vectors parallelize; small/medium ones stay single-threaded.
        ///     </para>
        ///     <para>
        ///     ORDER-INDEPENDENT reductions (integer <c>min</c>/<c>max</c>/<c>amin</c>/<c>amax</c>) are
        ///     <b>parallelized by default</b> above a large-array threshold, because their result is
        ///     bit-IDENTICAL at any thread count — there is no summation order to preserve — and NumPy's
        ///     reductions are single-threaded, leaving most of the socket's DRAM bandwidth idle. Passing
        ///     <c>enabled: false</c> turns these off too, for callers/benchmarks that want strict
        ///     single-thread execution.
        ///     </para>
        ///     <example>
        ///     <code>
        ///     // (default)                         // int min/max parallel on large arrays; dot single
        ///     np.multithreading(true);          // also enable the order-sensitive dot, up to 8 threads
        ///     np.multithreading(true, 16);      // ... up to 16 threads
        ///     np.multithreading(false);         // strict single-threaded (both classes off)
        ///     </code>
        ///     </example>
        /// </remarks>
        public static void multithreading(bool enabled, int max_threads = 8)
        {
            MultiThread.Enabled = enabled;
            MultiThread.UnorderedEnabled = enabled;
            MultiThread.MaxThreads = max_threads;
        }
    }
}
