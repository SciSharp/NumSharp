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
        private static volatile bool _unorderedEnabled = true;
        private static volatile int _maxThreads = 8;

        /// <summary>Whether ORDER-SENSITIVE parallel kernels may use more than one thread. Default:
        /// <c>false</c>. Gates kernels whose float result depends on the reduction order (the fused
        /// dot product) — opt-in so the summation order (and its last ULPs) is unchanged unless the
        /// caller asks. Set together with <see cref="UnorderedEnabled"/> by
        /// <see cref="NumSharp.np.multithreading(bool,int)"/>.</summary>
        public static bool Enabled
        {
            get => _enabled;
            set => _enabled = value;
        }

        /// <summary>Whether ORDER-INDEPENDENT parallel reductions may use more than one thread.
        /// Default: <c>true</c>. Gates reductions whose result is bit-IDENTICAL at any thread count
        /// — integer min/max (associative, commutative, no NaN/-0) — so parallelizing them changes
        /// nothing but the wall-clock: NumPy's reductions are single-threaded, so the socket's spare
        /// DRAM bandwidth (measured several× a single core) is otherwise wasted. On by default because
        /// the summation-order concern that keeps <see cref="Enabled"/> off does not exist here.
        /// <c>np.multithreading(false)</c> turns it off for callers/benches wanting strict
        /// single-thread execution.</summary>
        public static bool UnorderedEnabled
        {
            get => _unorderedEnabled;
            set => _unorderedEnabled = value;
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
        /// Below this many elements an ORDER-INDEPENDENT reduction (min/max) is never parallelized:
        /// under ~1M the buffer is L2/L3-resident, a single core already saturates it in tens of µs,
        /// and thread fan-out would cost more than it saves (measured crossover). Higher than
        /// <see cref="MinTotalWork"/> because these run by DEFAULT, so the floor must be safely past
        /// the point where parallelism is a clear win.
        /// </summary>
        internal const long MinUnorderedWork = 1_000_000;

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

        /// <summary>
        /// Effective thread count for an ORDER-INDEPENDENT contiguous reduction (integer min/max),
        /// whose result is bit-identical at any thread count. Unlike <see cref="DegreeOfParallelism"/>
        /// this does NOT gate on <see cref="Enabled"/> (that gate exists to preserve float summation
        /// order, which min/max have none of) — it is on by default via <see cref="UnorderedEnabled"/>.
        /// Engaged only above <see cref="MinUnorderedWork"/> and bounded by MaxThreads, ProcessorCount
        /// and a per-thread work floor.
        /// </summary>
        public static int DegreeOfParallelismUnordered(long n)
        {
            if (!_unorderedEnabled || n < MinUnorderedWork)
                return 1;
            long byWork = n / MinWorkPerThread;
            if (byWork < 1)
                return 1;
            int cap = Math.Min(_maxThreads, Environment.ProcessorCount);
            return (int)Math.Min(byWork, cap);
        }
    }
}
