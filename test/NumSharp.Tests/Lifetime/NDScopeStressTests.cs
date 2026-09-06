using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Generic;

namespace NumSharp.Tests.Lifetime
{
    /// <summary>
    ///     Cross-thread stress proof of <see cref="NDScope"/>. The scope stack is
    ///     <c>[ThreadStatic]</c>, so the contracts under fire here are: full isolation between
    ///     threads (no locks, no cross-talk), correct values under sustained concurrent mixed-op
    ///     load, zero per-call buffer strands with every op internally scoped, safe hand-off of
    ///     yielded results between threads, nested-scope re-parenting under concurrency, and
    ///     stability while an antagonist thread forces GC/finalizer churn mid-flight.
    /// </summary>
    [TestClass]
    [DoNotParallelize]   // pool counters are process-wide; the load itself must own the machine
    public class NDScopeStressTests
    {
        private static void DrainGC()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }

        private static long Acquisitions => SizeBucketedBufferPool.Hits + SizeBucketedBufferPool.Misses + SizeBucketedBufferPool.ZeroedAllocs;
        private static long Releases => SizeBucketedBufferPool.Returns + SizeBucketedBufferPool.ReturnsFreed;

        /// <summary>One full mixed cycle over the scoped surface; returns the number of value errors.</summary>
        private static int MixedCycle(NDArray a, NDArray b, NDArray d, NDArray idx, NDArray<bool> mask, int salt)
        {
            int errors = 0;

            using (var r = np.logical_and(a, b))
                if (r.GetBoolean(0) || !r.GetBoolean(1)) errors++;

            using (var r = np.roll(a, 1 + (salt & 3)))
                if (r.size != a.size) errors++;

            using (var r = np.isin(a, idx))
                if (!r.GetBoolean(0) || r.GetBoolean(1)) errors++;   // idx contains 0, not 1

            using (var r = np.sort(d))
                if (r.GetDouble(0) != 1.0 || r.GetDouble(4) != 5.0) errors++;

            using (var r = np.argsort(d))
                if (r.GetInt64(0) != 1) errors++;                    // d[1] == 1.0 is the minimum

            using (var r = np.ptp(a))
                if (r.GetInt64(0) != 63) errors++;

            using (var r = np.median(d))
                if (r.GetDouble(0) != 3.0) errors++;

            using (var r = d[mask])                                  // boolean-mask gather
                if (r.size != 3 || r.GetDouble(0) != 3.0) errors++;

            using (var r = d[idx])                                   // fancy gather
                if (r.size != 2 || r.GetDouble(0) != 3.0) errors++;

            var tup = np.nonzero(mask);                              // tuple of column views
            if (tup[0].size != 3) errors++;
            foreach (var t in tup) t.Dispose();

            return errors;
        }

        private static (NDArray a, NDArray b, NDArray d, NDArray idx, NDArray<bool> mask) MakeOperands()
        {
            var a = np.arange(64);   // int64: matches the GetInt64 assertions
            var b = np.ones(new Shape(64), NPTypeCode.Int32);
            var d = np.array(new double[] { 3, 1, 2, 5, 4 });
            var idx = np.array(new int[] { 0, 2 });
            var mask = np.array(new[] { true, false, true, false, true }).MakeGeneric<bool>();
            return (a, b, d, idx, mask);
        }

        private static void DisposeOperands((NDArray a, NDArray b, NDArray d, NDArray idx, NDArray<bool> mask) ops)
        {
            ops.a.Dispose(); ops.b.Dispose(); ops.d.Dispose(); ops.idx.Dispose(); ops.mask.Dispose();
        }

        [TestMethod]
        public void EightThreads_MixedOps_ZeroErrors_ZeroStrandSlope()
        {
            const int Threads = 8;
            const int Iterations = 1_500;

            // Warm all kernels/caches once so one-time setup stays out of the measured window.
            var warm = MakeOperands();
            Assert.AreEqual(0, MixedCycle(warm.a, warm.b, warm.d, warm.idx, warm.mask, 0));
            DisposeOperands(warm);
            DrainGC();
            SizeBucketedBufferPool.ResetCounters();

            long errors = 0;
            Parallel.For(0, Threads, t =>
            {
                var ops = MakeOperands();          // per-thread operands: no shared mutable state
                long local = 0;
                for (int i = 0; i < Iterations; i++)
                    local += MixedCycle(ops.a, ops.b, ops.d, ops.idx, ops.mask, i);
                DisposeOperands(ops);
                Interlocked.Add(ref errors, local);
            });

            long deficit = Acquisitions - Releases;
            long calls = (long)Threads * Iterations * 10;   // 10 scoped ops per cycle

            Assert.AreEqual(0L, errors, "scoped ops must stay value-correct under concurrency");
            // A single stranded buffer per call would read as `calls`. Allow a tiny constant
            // (lazily-initialised per-thread machinery), never a slope.
            Assert.IsTrue(deficit <= Threads * 8,
                $"stranded {deficit} buffers over {calls} scoped calls (acq {Acquisitions}, rel {Releases})");
        }

        [TestMethod]
        public void EightThreads_NestedScopes_DropThroughParent_AllReclaimed()
        {
            const int Threads = 8;
            const int Iterations = 400;

            long aliveAfter = 0, errors = 0;
            Parallel.For(0, Threads, t =>
            {
                var input = np.arange(32);
                var seen = new List<NDArray>(Iterations);
                using (var outer = NDScope.Open())
                {
                    for (int i = 0; i < Iterations; i++)
                    {
                        NDArray dropped;
                        using (var inner = NDScope.Open())
                            dropped = inner.Returns(input * 2);   // re-parents into `outer`
                        if (dropped.IsDisposed) Interlocked.Increment(ref errors);
                        if (dropped.GetInt64(3) != 6) Interlocked.Increment(ref errors);
                        seen.Add(dropped);                        // never disposed by hand
                    }
                }
                foreach (var r in seen)
                    if (!r.IsDisposed)
                        Interlocked.Increment(ref aliveAfter);
                if (input.IsDisposed) Interlocked.Increment(ref errors);
                input.Dispose();
            });

            Assert.AreEqual(0L, errors, "dropped results must stay usable until the parent scope closes");
            Assert.AreEqual(0L, aliveAfter, "every dropped result must be reclaimed by its parent scope");
        }

        [TestMethod]
        public void ProducersAndConsumers_CrossThreadHandoff()
        {
            const int Producers = 4;
            const int PerProducer = 800;

            var queue = new ConcurrentQueue<NDArray>();
            long errors = 0;
            using var done = new CountdownEvent(Producers);

            var consumers = new Thread[2];
            long consumed = 0;
            for (int c = 0; c < consumers.Length; c++)
            {
                consumers[c] = new Thread(() =>
                {
                    while (!done.IsSet || !queue.IsEmpty)
                    {
                        if (!queue.TryDequeue(out var r)) { Thread.SpinWait(200); continue; }
                        // Produced under a scope on ANOTHER thread, yielded with no parent ->
                        // fully detached: this thread owns it outright.
                        if (r.IsDisposed || r.GetInt64(0) != 62 || r.GetInt64(2) != 0)
                            Interlocked.Increment(ref errors);
                        r.Dispose();
                        Interlocked.Increment(ref consumed);
                    }
                });
                consumers[c].Start();
            }

            Parallel.For(0, Producers, p =>
            {
                var a = np.arange(64);
                for (int i = 0; i < PerProducer; i++)
                    queue.Enqueue(np.roll(a, 2));   // scoped internally; result detached to the caller
                a.Dispose();
                done.Signal();
            });

            foreach (var c in consumers) c.Join();

            Assert.AreEqual(0L, errors, "cross-thread consumed results must be alive and correct");
            Assert.AreEqual((long)Producers * PerProducer, consumed);
        }

        [TestMethod]
        public void MixedOps_UnderForcedGCChurn_StayCorrect()
        {
            using var stop = new ManualResetEventSlim(false);
            var antagonist = new Thread(() =>
            {
                while (!stop.IsSet)
                {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    Thread.Sleep(10);
                }
            }) { IsBackground = true };
            antagonist.Start();

            long errors = 0;
            try
            {
                Parallel.For(0, 4, t =>
                {
                    var ops = MakeOperands();
                    long local = 0;
                    for (int i = 0; i < 600; i++)
                        local += MixedCycle(ops.a, ops.b, ops.d, ops.idx, ops.mask, i);
                    DisposeOperands(ops);
                    Interlocked.Add(ref errors, local);
                });
            }
            finally
            {
                stop.Set();
                antagonist.Join();
            }

            Assert.AreEqual(0L, errors, "forced GC/finalizer churn must not perturb scoped results");
        }
    }
}
