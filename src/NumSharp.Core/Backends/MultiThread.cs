using System;

namespace NumSharp.Backends
{
    /// <summary>
    /// Global configuration for NumSharp's multithreaded kernels.
    ///
    /// Currently governs the fused 1-D dot product (numpy.dot vector·vector) for
    /// contiguous float / double inputs; other kernels remain single-threaded.
    /// Disabled by default — enable via <see cref="NumSharp.np.multithreading(bool,int)"/>
    /// so existing behavior (and bit-for-bit summation order) is unchanged unless
    /// the caller opts in.
    ///
    /// Parallelism is gated on work size: tiny and medium reductions stay on one
    /// thread because thread fan-out (a few microseconds) would dominate. Only when
    /// there is enough work to amortize that cost are chunks dispatched across cores.
    /// </summary>
    public static class MultiThread
    {
        private static volatile bool _enabled = false;
        private static volatile int _maxThreads = 8;

        /// <summary>Whether parallel kernels may use more than one thread. Default: <c>false</c>.</summary>
        public static bool Enabled
        {
            get => _enabled;
            set => _enabled = value;
        }

        /// <summary>
        /// Upper bound on worker threads, clamped to at least 1. The effective count
        /// is additionally capped by <see cref="Environment.ProcessorCount"/> and by the
        /// available work. Default: 8.
        /// </summary>
        public static int MaxThreads
        {
            get => _maxThreads;
            set => _maxThreads = value < 1 ? 1 : value;
        }

        /// <summary>
        /// Below this many elements a reduction is never parallelized — fan-out overhead
        /// would outweigh any gain (see the 32-thread regression at n=100k in the POC).
        /// </summary>
        internal const long MinTotalWork = 50_000;

        /// <summary>Each worker thread is given at least this many elements of work.</summary>
        internal const long MinWorkPerThread = 32_000;

        /// <summary>
        /// Effective number of threads for a contiguous element-wise reduction over
        /// <paramref name="n"/> elements. Returns 1 when multithreading is disabled or the
        /// work is too small for parallelism to pay off; otherwise
        /// <c>min(MaxThreads, ProcessorCount, n / MinWorkPerThread)</c>.
        /// </summary>
        public static int DegreeOfParallelism(long n)
        {
            if (!_enabled || n < MinTotalWork)
                return 1;
            long byWork = n / MinWorkPerThread;
            if (byWork < 1)
                return 1;
            int cap = Math.Min(_maxThreads, Environment.ProcessorCount);
            return (int)Math.Min(byWork, cap);
        }
    }
}
