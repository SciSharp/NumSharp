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
    ///             a region that observed a collection is retried after a <see cref="Quiesce"/>,
    ///             never trusted; persistent interference yields null, never a false verdict;</item>
    ///       <item>per-measurement hygiene must NOT be a full collection: a forced full GC marks the
    ///             whole live heap, and this process's heap holds the replayed corpus and the test
    ///             framework's bookkeeping — 363 full <see cref="Settle"/> calls were 14 s of a 37 s
    ///             oracle run (2026-09-24). Retries, confirmations and the sweep's periodic hygiene use
    ///             the young-generation <see cref="Quiesce"/>; <see cref="Settle"/> is for sweep/test
    ///             starts only. Exactness never depended on the full collection — see
    ///             <see cref="MeasureExactTraffic"/> and <c>GcQuiescence</c>;</item>
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
        ///     FULL drain of the finalizer backlog (every generation, twice) so pending returns from
        ///     earlier (undisposing) tests cannot land inside a measured region as spurious returns.
        ///     Call it once at the start of a sweep or a teeth test — never per measurement: its cost
        ///     scales with the whole live heap (~40 ms here), which is what <see cref="Quiesce"/> avoids.
        /// </summary>
        public static void Settle() => GcQuiescence.CollectFull();

        /// <summary>
        ///     Cheap per-measurement hygiene: a young-generation collection plus a finalizer drain
        ///     (<see cref="GcQuiescence.CollectYoung"/>). Finalizes the garbage recent regions left —
        ///     escaped buffers' wrappers, allocated moments ago and therefore young — so their returns
        ///     land NOW rather than inside a later region, at a cost independent of heap size.
        /// </summary>
        /// <remarks>
        ///     Older garbage it leaves unvisited cannot finalize until some collection finds it, and a
        ///     collection inside a measured region is exactly what the region's validity check rejects
        ///     (<see cref="MeasureTraffic"/>'s collection count, <see cref="MeasureExactTraffic"/>'s full
        ///     epoch). So the weaker hygiene can cost an occasional retry, never a wrong verdict.
        /// </remarks>
        public static void Quiesce() => GcQuiescence.CollectYoung();

        /// <summary>The pool traffic one region execution produced. <see cref="Escaped"/> is the
        /// undisposed-intermediate count; <see cref="Takes"/> == 0 with a fresh non-scalar result
        /// disposed is the FULL-BYPASS signature (allocated AND freed outside the bucketed pool —
        /// invisible to the balance alone, which is why callers get both sides).</summary>
        public readonly record struct Traffic(long Takes, long Returns)
        {
            public long Escaped => Takes - Returns;
        }

        /// <summary>
        ///     Pool traffic of one execution of <paramref name="region"/>, or null when a
        ///     collection landed inside the region on every attempt (escapes indistinguishable —
        ///     reported as inconclusive by callers, never as a failure). The region must be
        ///     re-executable: attempts after an interfered one re-run it post-<see cref="Quiesce"/>,
        ///     so a region must produce the SAME traffic on every execution (true for op-invocations
        ///     and for the deliberate-escape teeth tests, which escape per execution).
        /// </summary>
        /// <param name="region">The re-executable region to measure.</param>
        /// <param name="attempts">How many executions may be discarded for GC interference.</param>
        /// <returns>The region's traffic, or null when every attempt saw a collection.</returns>
        /// <remarks>
        ///     This is the cheap SCREEN: no finalizer drain before the region (a drain per case would
        ///     cost ~20–50 µs × every corpus case) and a collection-count check only. It is exact for a
        ///     zero reading in the direction that matters (see <see cref="MeasureConfirmedTraffic"/>);
        ///     a non-zero reading is re-measured by <see cref="MeasureExactTraffic"/>.
        /// </remarks>
        public static Traffic? MeasureTraffic(Action region, int attempts = 3)
        {
            for (int attempt = 0; attempt < attempts; attempt++)
            {
                long t0 = Takes, r0 = Returns;
                int g0 = Collections;
                region();
                var traffic = new Traffic(Takes - t0, Returns - r0);
                if (Collections == g0)
                    return traffic;
                // The collection that disturbed this attempt queued finalizers; drain them (and the
                // young garbage around them) before the retry. Young is enough — see Quiesce.
                Quiesce();
            }
            return null;
        }

        /// <summary>
        ///     The EXACT measurement: pool traffic of one execution of <paramref name="region"/> in a
        ///     window no finalizer could touch, or null when a collection disturbed every attempt.
        /// </summary>
        /// <param name="region">The re-executable region to measure.</param>
        /// <param name="attempts">How many executions may be discarded for GC interference.</param>
        /// <returns>The region's own traffic, or null (inconclusive) when every attempt was disturbed.</returns>
        /// <remarks>
        ///     Each attempt runs <see cref="Quiesce"/> (hygiene), then opens a
        ///     <see cref="GcQuiescence.OpenWindow"/> window — epoch first, finalizer drain second — and
        ///     accepts the reading only if <see cref="GcQuiescence.Undisturbed"/> holds. That closes two
        ///     holes the former confirm (a full <see cref="Settle"/> followed by a count-only
        ///     <see cref="MeasureTraffic"/>) left open: a collection landing between the settle's last
        ///     drain and the count snapshot queued finalizers that then ran inside the region unseen, and
        ///     a background collection that started earlier and FINISHED inside the region queued
        ///     finalizers without moving the collection count at all.
        /// </remarks>
        public static Traffic? MeasureExactTraffic(Action region, int attempts = 3)
        {
            for (int attempt = 0; attempt < attempts; attempt++)
            {
                Quiesce();
                var window = GcQuiescence.OpenWindow();
                long t0 = Takes, r0 = Returns;
                region();
                var traffic = new Traffic(Takes - t0, Returns - r0);
                if (GcQuiescence.Undisturbed(window))
                    return traffic;
            }
            return null;
        }

        /// <summary>Escaped-buffer count of one region execution (see <see cref="MeasureTraffic"/>).</summary>
        public static long? Measure(Action region, int attempts = 3)
            => MeasureTraffic(region, attempts)?.Escaped;

        /// <summary>
        ///     The SCREEN-then-CONFIRM measurement every corpus/catalogue replay uses: one
        ///     <see cref="MeasureTraffic"/> screen, and — only when that screen reads a non-zero
        ///     balance — an exact confirming measurement (<see cref="MeasureExactTraffic"/>) whose
        ///     verdict is returned instead.
        /// </summary>
        /// <remarks>
        ///     Why a non-zero screen is never trusted on its own: a replay runs over a library that
        ///     may carry leaks, so escaped buffers accumulate; a pacing or natural GC collects them
        ///     and the finalizer thread then RETURNS their buffers asynchronously across LATER regions.
        ///     That drain is not a collection, so <see cref="MeasureTraffic"/>'s GC-count detection
        ///     cannot see it, and it produces phantom negative balances (down to −262 at the gate's
        ///     landing) or masks a real positive one. With the finalizer queue drained when the
        ///     confirming window opens and no collection inside it, the confirmed balance is exact. A
        ///     zero screen needs no confirmation: a drain can only ADD returns, so a region that
        ///     genuinely escaped a buffer can read zero only if a drain landed in it — which the next
        ///     (drained) measurement of the same op would expose, and the harness quiesces
        ///     periodically anyway.
        /// </remarks>
        /// <param name="region">The re-executable region (see <see cref="MeasureTraffic"/>'s contract:
        /// every execution must produce the same pool traffic).</param>
        /// <param name="attempts">Per-measurement retry budget for GC interference.</param>
        /// <returns>The confirmed traffic, or null when a GC landed inside every attempt of the
        /// measurement that decides the verdict (inconclusive — callers count it, never fail on it).</returns>
        public static Traffic? MeasureConfirmedTraffic(Action region, int attempts = 3)
        {
            var traffic = MeasureTraffic(region, attempts);
            if (traffic is not null && traffic.Value.Escaped != 0)
                traffic = MeasureExactTraffic(region, attempts);
            return traffic;
        }

        /// <summary>Confirmed escaped-buffer count (see <see cref="MeasureConfirmedTraffic"/>).</summary>
        /// <param name="region">The re-executable region to measure.</param>
        /// <param name="attempts">Per-measurement retry budget for GC interference.</param>
        /// <returns>The confirmed escape count, or null when every attempt was GC-interfered.</returns>
        public static long? MeasureConfirmed(Action region, int attempts = 3)
            => MeasureConfirmedTraffic(region, attempts)?.Escaped;
    }
}
