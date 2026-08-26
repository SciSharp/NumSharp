using System;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     Undisposed-intermediate detector over <see cref="SizeBucketedBufferPool"/>'s public
    ///     counters.
    ///
    ///     PRINCIPLE — in a fully-disposed region, pool takes and returns must balance: every
    ///     buffer an op allocates is either (a) inside its RESULT, returned when the harness
    ///     disposes that result, or (b) an internal temporary, returned by the library's own
    ///     deterministic scoping ([NDScoped] weaver / using) before the op returns. What is left,
    ///     <c>takes − returns</c>, is (c): buffers taken, dropped, and reclaimable only by a
    ///     future GC + finalizer pass — undisposed intermediates the weaver failed to cover.
    ///     A NEGATIVE balance is the mirror defect: a result buffer allocated OUTSIDE the pool
    ///     (bypassing Take and its warm reuse) whose Dispose nevertheless Returns into it.
    ///
    ///     VALIDITY REQUIREMENTS, each owned by a caller rule:
    ///     <list type="bullet">
    ///       <item>counters are process-global → the measuring test class is [DoNotParallelize];</item>
    ///       <item>a GC inside the region masks escapes (finalizers Return them mid-window) →
    ///             a region that observed a collection is retried after a settle, never trusted;
    ///             persistent interference yields null, never a false verdict;</item>
    ///       <item>one-time caches (FFT plans, emitted kernels) take buffers they legitimately
    ///             retain → callers run one un-measured warm invocation first;</item>
    ///       <item>the takes side must include the calloc path: TakeZeroed counts in
    ///             <see cref="SizeBucketedBufferPool.ZeroedAllocs"/>, not Hits/Misses. (Guard-pages
    ///             mode double-counts zeroed takes as misses; it is env-opt-in and off in CI.)</item>
    ///     </list>
    /// </summary>
    internal static class ScopeAudit
    {
        /// <summary>Every pool hand-out path: bucket hits + cold allocs + the calloc fast path.</summary>
        public static long Takes =>
            SizeBucketedBufferPool.Hits + SizeBucketedBufferPool.Misses + SizeBucketedBufferPool.ZeroedAllocs;

        /// <summary>Every give-back path: pooled returns + returns freed (bucket full / out-of-range).</summary>
        public static long Returns =>
            SizeBucketedBufferPool.Returns + SizeBucketedBufferPool.ReturnsFreed;

        public static int Collections =>
            GC.CollectionCount(0) + GC.CollectionCount(1) + GC.CollectionCount(2);

        /// <summary>
        ///     Drain the finalizer backlog so pending returns from earlier (undisposing) tests
        ///     cannot land inside a measured region as spurious returns.
        /// </summary>
        public static void Settle()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        /// <summary>
        ///     Escaped-buffer count of one execution of <paramref name="region"/>, or null when a
        ///     collection landed inside the region on every attempt (escapes indistinguishable —
        ///     reported as inconclusive by callers, never as a failure). The region must be
        ///     re-executable: attempts after an interfered one re-run it post-settle, so a region
        ///     must produce the SAME escape count on every execution (true for op-invocations and
        ///     for the deliberate-escape teeth tests, which escape per execution).
        /// </summary>
        public static long? Measure(Action region, int attempts = 3)
        {
            for (int attempt = 0; attempt < attempts; attempt++)
            {
                long t0 = Takes, r0 = Returns;
                int g0 = Collections;
                region();
                long escaped = (Takes - t0) - (Returns - r0);
                if (Collections == g0)
                    return escaped;
                Settle();
            }
            return null;
        }
    }
}
