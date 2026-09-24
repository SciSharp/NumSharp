using System;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Cheap, exact garbage-collector hygiene for tests that read PROCESS-GLOBAL state around a region of
    ///     code — the buffer pool's take/return counters (<c>SizeBucketedBufferPool</c>), or an ARC reference
    ///     count after forcing a finalizer. Use <see cref="CollectYoung"/> between measurements,
    ///     <see cref="OpenWindow"/> / <see cref="Undisturbed"/> to prove a measured region saw no asynchronous
    ///     finalizer, and <see cref="CollectFull"/> only once per test to clear what EARLIER tests left behind.
    /// </summary>
    /// <remarks>
    ///     <para><b>The cost this removes.</b> These tests used to "drain" with
    ///     <c>GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect()</c> before every measured region. A forced
    ///     FULL, blocking collection marks the whole live heap, and in a full test run that heap is dominated by
    ///     the test framework's own per-test bookkeeping (MSTest keeps every TestCase with its property
    ///     dictionaries, and a TestContext per executed test, alive until the run ends): measured on
    ///     2026-09-24 (net10.0, 16,334 tests) the drain at <c>BufferReleaseSweepTests</c>' start promoted
    ///     121 MiB and cost 78 ms; the same sweep run alone (5 tests loaded) spent under 9 ms per window, drain
    ///     AND measured ops included. The sweep ran 840 drains (68 s of a 262 s suite), ArcLifecycle's finalizer
    ///     loop 200 (19 s), and the oracle's leak sweep 363 <c>ScopeAudit.Settle</c> calls (14 s of a 37 s run).
    ///     Worse, the cost grows with EVERY test added to the suite, because every test grows the heap each full
    ///     drain has to mark.</para>
    ///     <para><b>Why a young collection is enough — exactness never needed a full one.</b> A measured
    ///     window is exact when no finalizer runs inside it. Finalizers run only after a collection queues them,
    ///     so it suffices that (1) the finalizer queue is empty when the window opens —
    ///     <see cref="GC.WaitForPendingFinalizers"/>, in <see cref="OpenWindow"/> — and (2) no collection queued
    ///     anything while it was open — checked afterwards by <see cref="Undisturbed"/>. Neither requires
    ///     collecting gen 2: a garbage object no collection has visited cannot run its finalizer, so old garbage
    ///     the young collection leaves behind is inert until some collection finds it, and that collection is
    ///     exactly what <see cref="Undisturbed"/> detects. The collection before a window is therefore HYGIENE,
    ///     not correctness: it finalizes the young garbage the previous window left (so those buffers are home
    ///     again) and resets the gen-0 budget (so a natural collection rarely lands in the next window). A
    ///     gen-1 collection does that in well under a millisecond whatever the size of gen 2.</para>
    ///     <para><b>The background-GC trap.</b> <see cref="GC.CollectionCount(int)"/> alone is NOT a sufficient
    ///     window check, although it is what the older harnesses used. A background (concurrent) gen-2
    ///     collection increments the count when it STARTS but queues its finalizers when it ENDS — measured
    ///     0.26 ms vs ~8 ms after the request on a 200K-object heap (net10.0). One that started before the window
    ///     and finishes inside it runs finalizers mid-window without moving the count. <see cref="Epoch"/>
    ///     therefore also records <c>GC.GetGCMemoryInfo(GCKind.Background).Index</c>, which advances only when a
    ///     background collection COMPLETES. (A blocking collection cannot straddle a window boundary: it suspends
    ///     the measuring thread, so the thread's reads fall wholly before or wholly after it.)</para>
    ///     <para>Linked into <c>NumSharp.Tests.Oracle</c> as well (see its csproj), so the leak sweep's
    ///     <c>ScopeAudit</c> and this project's lifetime tests share one implementation of the protocol.</para>
    /// </remarks>
    internal static class GcQuiescence
    {
        /// <summary>
        ///     A snapshot of every signal that can make a finalizer run asynchronously: the total number of
        ///     collections STARTED (any generation) and the index of the last background collection COMPLETED.
        ///     Two equal epochs taken around a region mean no collection began, and no background collection
        ///     finished, in between.
        /// </summary>
        /// <param name="Collections">Sum of <see cref="GC.CollectionCount(int)"/> over generations 0–2. A gen-N
        /// collection also counts as a collection of every younger generation, so any collection moves it.</param>
        /// <param name="BackgroundGcIndex">The <see cref="GCMemoryInfo.Index"/> of the last COMPLETED background
        /// collection (0 before the first one); it moves when a background GC ends, i.e. when it queues its
        /// finalizers — the half of a background GC the collection count cannot see.</param>
        internal readonly record struct Epoch(int Collections, long BackgroundGcIndex)
        {
            /// <summary>Reads both signals now.</summary>
            /// <returns>The current epoch.</returns>
            /// <remarks>
            ///     Allocates one small <see cref="GCMemoryInfo"/> payload; call it OUTSIDE the measured region (as
            ///     <see cref="OpenWindow"/> and <see cref="Undisturbed"/> do) when the region counts managed
            ///     allocations.
            /// </remarks>
            public static Epoch Capture()
                => new(GC.CollectionCount(0) + GC.CollectionCount(1) + GC.CollectionCount(2),
                       GC.GetGCMemoryInfo(GCKind.Background).Index);
        }

        /// <summary>
        ///     Hygiene between measurements: a forced, blocking, non-compacting collection of generations 0 and 1,
        ///     then a wait for the finalizers it queued. Finalizes the garbage the previous window just created
        ///     (so its buffers return to the pool before the next window starts counting) at a cost independent of
        ///     how big the rest of the heap has grown.
        /// </summary>
        /// <remarks>
        ///     Generation 1 rather than 0 so garbage that a collection promoted once — e.g. a wrapper that was
        ///     still referenced when the pool's own gen-0 pacing collection ran mid-call — is found too.
        ///     Non-compacting because compaction buys nothing here and only lengthens the pause (the pool's own
        ///     pacing collection makes the same choice). Garbage that reached gen 2 is deliberately left alone:
        ///     it cannot finalize until a collection visits it, and any such collection inside a window is
        ///     caught by <see cref="Undisturbed"/>.
        /// </remarks>
        public static void CollectYoung()
        {
            GC.Collect(1, GCCollectionMode.Forced, blocking: true, compacting: false);
            GC.WaitForPendingFinalizers();
        }

        /// <summary>
        ///     The expensive full drain: two forced full collections, each followed by a finalizer wait, so that
        ///     garbage whose finalizer drops the last reference to OTHER finalizable objects is finalized as well.
        ///     Use it once per test (or once per sweep) to clear the backlog earlier tests left, never per
        ///     measurement — its cost scales with the whole test run's live heap (see the type remarks).
        /// </summary>
        public static void CollectFull()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        /// <summary>
        ///     Opens an exact measurement window: takes the <see cref="Epoch"/> FIRST, then waits for the
        ///     finalizer queue to empty. Read the counters after this returns, run the region, then confirm with
        ///     <see cref="Undisturbed"/>.
        /// </summary>
        /// <returns>The epoch the window opened at, for <see cref="Undisturbed"/>.</returns>
        /// <remarks>
        ///     The order is load-bearing. Anything a collection queued BEFORE the snapshot has finished running
        ///     when the wait returns; anything queued AFTER it came from a collection that the snapshot comparison
        ///     reports. Waiting first and snapshotting second would leave a gap in which a collection could queue
        ///     finalizers that then run inside the window unobserved.
        /// </remarks>
        public static Epoch OpenWindow()
        {
            var opened = Epoch.Capture();
            GC.WaitForPendingFinalizers();
            return opened;
        }

        /// <summary>
        ///     Whether no collection started, and no background collection completed, since
        ///     <paramref name="opened"/> — i.e. whether every counter change observed since
        ///     <see cref="OpenWindow"/> was produced synchronously by the measured region itself.
        /// </summary>
        /// <param name="opened">The epoch returned by <see cref="OpenWindow"/>.</param>
        /// <returns>True when the window is exact; false when it must be discarded and re-run.</returns>
        public static bool Undisturbed(Epoch opened) => Epoch.Capture() == opened;
    }
}
