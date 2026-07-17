using System;

namespace NumSharp.UnitTest.IO
{
    /// <summary>
    ///     Helper for tests that assert on how much an operation allocates.
    /// </summary>
    internal static class AllocationTests
    {
        /// <summary>
        ///     The smallest managed allocation <paramref name="action"/> made across several runs.
        /// </summary>
        /// <remarks>
        ///     <see cref="GC.GetTotalAllocatedBytes(bool)"/> is process-wide, not per-operation, so a
        ///     background or finalizer thread allocating inside the measurement window inflates a single
        ///     reading — the difference between a real regression and a neighbour's garbage is invisible
        ///     from one sample. Noise can only ever ADD, so the minimum of several runs is the closest
        ///     thing to the operation's true cost and cannot be pushed over a threshold by unrelated
        ///     work. This is the same best-of-rounds rule the repo's benchmarks use.
        ///
        ///     The first run is discarded: it pays for JIT and first-touch, which are real but one-off.
        /// </remarks>
        public static long MinAllocated(Action action, int rounds = 5)
        {
            action(); // warm: JIT + first-touch

            long min = long.MaxValue;
            for (int i = 0; i < rounds; i++)
            {
                long before = GC.GetTotalAllocatedBytes(precise: true);
                action();
                long allocated = GC.GetTotalAllocatedBytes(precise: true) - before;
                if (allocated < min)
                    min = allocated;
            }

            return min;
        }
    }
}
