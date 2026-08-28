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
        ///     Multithreading is disabled by default, so the default behavior — and the exact
        ///     summation order — is unchanged unless you opt in.
        ///     <para>
        ///     Currently this controls the fused 1-D dot product (<see cref="dot(NDArray,NDArray)"/>
        ///     for vector·vector) on contiguous <c>float</c> / <c>double</c> inputs. Only large
        ///     vectors are parallelized; small and medium ones stay single-threaded because thread
        ///     fan-out would cost more than it saves. With multithreading on, the inner product is
        ///     summed per-chunk and combined, so results may differ from the single-threaded path in
        ///     the last few ULPs (the same floating-point reordering NumPy's threaded BLAS exhibits).
        ///     </para>
        ///     <example>
        ///     <code>
        ///     np.multithreading(true);          // enable, up to 8 threads
        ///     np.multithreading(true, 16);      // enable, up to 16 threads
        ///     np.multithreading(false);         // back to single-threaded
        ///     </code>
        ///     </example>
        /// </remarks>
        public static void multithreading(bool enabled, int max_threads = 8)
        {
            MultiThread.Enabled = enabled;
            MultiThread.MaxThreads = max_threads;
        }
    }
}
