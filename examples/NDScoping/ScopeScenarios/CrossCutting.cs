using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using NumSharp;
using NDScoping.Instrumentation;

namespace NDScoping
{
    /// <summary>
    ///     CROSS-CUTTING — the contracts that are not a single result-shape but govern correct use at
    ///     every level: nesting, the caller-side <c>Attach</c> hot loop, <c>Detach</c> for a cache,
    ///     granularity, and threading. These map 1:1 to the "Ownership rules become structural",
    ///     "Nesting", "Granularity" and "Threading" paragraphs of <see cref="NDScope"/>'s XML doc — the
    ///     example is the executable form of that doc.
    /// </summary>
    internal static class CrossCutting
    {
        private static NDArray A() => np.array(new double[] { 1, 2, 3 });
        private static NDArray B() => np.array(new double[] { 10, 20, 30 });

        public static List<Report> RunAll(bool stress) => new()
        {
            Nesting(),
            HotLoop_Attach(),
            Detach_ForCache(),
            Granularity(),
            Threading(),
        };

        [NDScoped]
        private static NDArray InnerScoped(NDArray a, NDArray b) => a + b;

        // The enclosing scope reclaims an inner scoped call's result the caller drops: Returns
        // re-parents the yielded array into the parent scope.
        private static Report Nesting()
        {
            NDArray inner;
            bool aliveInside;
            using (var outer = NDScope.Open())
            {
                inner = InnerScoped(A(), B());   // Returns re-parents a+b into OUTER
                aliveInside = !inner.IsDisposed; // alive while outer is open
            }

            bool reclaimedByOuter = inner.IsDisposed;   // outer reclaimed the dropped inner result
            bool ok = aliveInside && reclaimedByOuter;
            return ReclamationProbe.Verdict("Nesting", ok,
                $"inner result alive inside outer {ReclamationProbe.Mark(aliveInside)}   " +
                $"reclaimed by the enclosing scope when dropped {ReclamationProbe.Mark(reclaimedByOuter)}");
        }

        // The two-audience rule made concrete: results a caller RECEIVES (built with no ambient scope)
        // linger; NDScope.Attach adopts each into the caller's own scope so it is reclaimed per loop.
        private static Report HotLoop_Attach()
        {
            var received = new List<NDArray>();
            for (int i = 0; i < 5; i++) received.Add(A() + B());   // no scope -> untracked -> linger
            bool naiveLinger = received.All(x => !x.IsDisposed);

            var adopted = new List<NDArray>();
            for (int i = 0; i < 5; i++) adopted.Add(A() + B());
            using (var loop = NDScope.Open())
                foreach (var r in adopted)
                    NDScope.Attach(r);                             // adopt into the caller's scope
            bool attachedReclaimed = adopted.All(x => x.IsDisposed);

            bool ok = naiveLinger && attachedReclaimed;
            return ReclamationProbe.Verdict("HotLoop_Attach", ok,
                $"naive received results linger {ReclamationProbe.Mark(naiveLinger)}   " +
                $"Attach reclaims them per loop {ReclamationProbe.Mark(attachedReclaimed)}");
        }

        private static NDArray _cache;

        [NDScoped]
        private static NDArray Detach_Producer(NDArray a, NDArray b, List<NDArray> temps)
        {
            var temp = a + b;         // ordinary temp -> reclaimed
            temps.Add(temp);
            var cached = a * b;       // must survive into the static cache
            NDScope.Detach(cached);   // remove from the scope so it OUTLIVES the boundary
            _cache = cached;
            return temp * 2.0;        // result
        }

        private static Report Detach_ForCache()
        {
            var temps = new List<NDArray>();
            var result = Detach_Producer(A(), B(), temps);

            bool tempReclaimed = temps.TrueForAll(t => t.IsDisposed);
            bool resultAlive = !result.IsDisposed;
            bool cacheAlive = _cache is not null && !_cache.IsDisposed;   // the detached array outlived the scope
            bool valueOk = resultAlive && Math.Abs(result.GetDouble(0) - 22.0) < 1e-9;

            bool ok = tempReclaimed && resultAlive && cacheAlive && valueOk;
            return ReclamationProbe.Verdict("Detach_ForCache", ok,
                $"ordinary temp reclaimed {ReclamationProbe.Mark(tempReclaimed)}   detached cache survives " +
                $"{ReclamationProbe.Mark(cacheAlive)}   result alive {ReclamationProbe.Mark(resultAlive)}");
        }

        [NDScoped]
        private static NDArray ScopedProduceTemp(List<NDArray> sink)
        {
            var t = A() + B();
            sink.Add(t);
            return t * 2.0;
        }

        // Scope a CALL, not a caller LOOP. Scoping the loop holds all N temps alive simultaneously
        // (the pool-bucket-overflow hazard); scoping the call reclaims each immediately.
        private static Report Granularity()
        {
            const int N = 2000;

            var perCall = new List<NDArray>();
            for (int i = 0; i < N; i++)
            {
                var r = ScopedProduceTemp(perCall);   // each call reclaims its own temp at its scope exit
                GC.KeepAlive(r);
            }

            int aliveAfterCall = perCall.Count(t => !t.IsDisposed);

            var perLoop = new List<NDArray>();
            int aliveInsideLoop;
            using (var loop = NDScope.Open())
            {
                for (int i = 0; i < N; i++) perLoop.Add(A() + B());   // all tracked by ONE scope
                aliveInsideLoop = perLoop.Count(t => !t.IsDisposed);
            }

            int aliveAfterLoop = perLoop.Count(t => !t.IsDisposed);

            bool ok = aliveAfterCall == 0 && aliveInsideLoop == N && aliveAfterLoop == 0;
            return ReclamationProbe.Verdict("Granularity", ok,
                $"scope-the-call peak temps ~1 (alive after {N} calls: {aliveAfterCall})   " +
                $"scope-the-loop holds all {N} at once (alive inside: {aliveInsideLoop}, after: {aliveAfterLoop})");
        }

        // The scope stack is [ThreadStatic]: a worker thread sees no ambient scope (finalizer fallback)
        // unless it opens its own. Both outcomes here are CORRECT behaviour, not a bug. We use a genuine
        // dedicated Thread rather than Task.Run — a threadpool task's body can be INLINED onto the
        // calling thread by a blocking .Result, which would (correctly, but confusingly) run the work
        // under the main scope. A dedicated thread guarantees a distinct [ThreadStatic] slot.
        private static Report Threading()
        {
            NDArray fromWorker = null;
            using (var main = NDScope.Open())
            {
                var worker = new Thread(() => fromWorker = A() + B());   // built with no ambient scope
                worker.Start();
                worker.Join();
            }

            bool notTrackedByMain = !fromWorker.IsDisposed;      // correct: cross-thread, outside main's scope

            var workerTemps = new List<NDArray>();
            var scopedWorker = new Thread(() =>
            {
                using var s = NDScope.Open();
                var t = A() + B();
                workerTemps.Add(t);   // reclaimed at s dispose (end of this thread body)
            });
            scopedWorker.Start();
            scopedWorker.Join();

            bool reclaimedInWorker = workerTemps.TrueForAll(t => t.IsDisposed);

            bool ok = notTrackedByMain && reclaimedInWorker;
            return ReclamationProbe.Verdict("Threading", ok,
                $"worker temp not tracked by the main scope {ReclamationProbe.Mark(notTrackedByMain)}   " +
                $"a worker's own scope reclaims {ReclamationProbe.Mark(reclaimedInWorker)}   (both correct — [ThreadStatic])");
        }
    }
}
