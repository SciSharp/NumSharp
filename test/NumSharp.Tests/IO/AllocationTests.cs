using System;

namespace NumSharp.Tests.IO
{
    /// <summary>
    ///     Helper for tests that assert on how much an operation allocates.
    /// </summary>
    internal static class AllocationTests
    {
        /// <summary>
        ///     The smallest managed allocation <paramref name="action"/> made on the calling thread across several runs.
        /// </summary>
        /// <param name="action">The operation to measure. It must do its allocating work on the calling thread, which
        ///     every NumSharp operation does (the library is single-threaded); work it hands to another thread is not
        ///     counted.</param>
        /// <param name="rounds">How many measured runs to take the minimum of (after one unmeasured warm-up run).</param>
        /// <returns>The fewest managed bytes one run allocated on this thread.</returns>
        /// <remarks>
        ///     Counted with <see cref="GC.GetAllocatedBytesForCurrentThread"/>, not the process-wide
        ///     <see cref="GC.GetTotalAllocatedBytes(bool)"/>: tests run in parallel, so the process-wide counter
        ///     would also count whatever the other tests allocate during the window — megabytes per run, against
        ///     thresholds of a few kilobytes. The per-thread counter sees only this operation (and, unlike the precise
        ///     process-wide count, needs no pass over every thread's allocation context). A finalizer or a background
        ///     thread can no longer inflate a reading either; the minimum of several runs is kept anyway, because a
        ///     run can still allocate one-off lazily (a cache filled on first use), and noise of that kind only ever
        ///     adds — the same best-of-rounds rule the repo's benchmarks use.
        ///
        ///     The first run is discarded: it pays for JIT and first-touch, which are real but one-off.
        /// </remarks>
        public static long MinAllocated(Action action, int rounds = 5)
        {
            action(); // warm: JIT + first-touch

            long min = long.MaxValue;
            for (int i = 0; i < rounds; i++)
            {
                long before = GC.GetAllocatedBytesForCurrentThread();
                action();
                long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
                if (allocated < min)
                    min = allocated;
            }

            return min;
        }
    }
}
