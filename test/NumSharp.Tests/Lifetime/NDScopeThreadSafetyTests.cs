using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Tests.Lifetime
{
    /// <summary>
    ///     Thread-safety invariants of <see cref="NDScope"/>. The scope stack is <c>[ThreadStatic]</c>
    ///     (both <c>t_current</c> and the single-slot <c>t_pool</c>), and every instance field
    ///     (<c>_parent</c>/<c>_tracked</c>/<c>_disposed</c>/<c>_threadId</c>) is touched only by the
    ///     owning thread — there is NO shared mutable state across threads and no lock. The model is
    ///     therefore thread-CONFINED (each scope lives entirely on one thread), which these tests pin
    ///     from the angles <see cref="NDScopeStressTests"/> (mixed-op load, GC churn) does not:
    ///     <list type="bullet">
    ///     <item>a SHARED read-only input is never tracked or corrupted by concurrent scoped ops;</item>
    ///     <item>a worker thread does not see another thread's open scope, and a scope never reclaims
    ///           an array built on a different thread (direct proof of thread-local isolation);</item>
    ///     <item>the full cross-thread hand-off cycle detach → hand off → <see cref="NDScope.Attach"/>
    ///           into the RECEIVER's scope → reclaim on close;</item>
    ///     <item>the woven NESTED op (roll axis=null → roll) stays correct and strand-free when run
    ///           concurrently on many threads.</item>
    ///     </list>
    /// </summary>
    [TestClass]
    [DoNotParallelize]   // the counter-based tests read the process-wide pool counters
    public class NDScopeThreadSafetyTests
    {
        private static void DrainGC()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }

        private static long Acquisitions => SizeBucketedBufferPool.Hits + SizeBucketedBufferPool.Misses + SizeBucketedBufferPool.ZeroedAllocs;
        private static long Releases => SizeBucketedBufferPool.Returns + SizeBucketedBufferPool.ReturnsFreed;

        // ------------------------------------------------------- shared read-only input, many threads

        [TestMethod]
        public void SharedReadOnlyInput_ConcurrentScopedOps_UntouchedAndCorrect()
        {
            var shared = np.arange(64).reshape(8, 8);   // built with no scope open -> never tracked
            Assert.IsNull(shared.TrackingScope, "a plainly-constructed array is tracked by no scope");

            const int Threads = 8, Iterations = 500;
            long errors = 0;
            Parallel.For(0, Threads, _ =>
            {
                long local = 0;
                for (int i = 0; i < Iterations; i++)
                {
                    int k = 1 + (i & 7);                    // 1..8
                    using var r = np.roll(shared, k);       // woven scoped op, reads `shared` only
                    // np.roll(arange(64).reshape(8,8), k) flat[0] == 64-k
                    if (r.GetInt64(0, 0) != 64 - k) local++;
                    if (shared.TrackingScope != null) local++;   // a worker's scope must NEVER adopt the input
                    if (shared.IsDisposed) local++;              // and must never reclaim it
                }
                Interlocked.Add(ref errors, local);
            });

            Assert.AreEqual(0L, errors, "a shared read-only input must stay untracked, alive and correct under concurrent scoped ops");
            Assert.IsFalse(shared.IsDisposed);
            Assert.IsNull(shared.TrackingScope);
            Assert.AreEqual(0L, shared.GetInt64(0, 0), "shared data intact");
            Assert.AreEqual(63L, shared.GetInt64(7, 7));
            shared.Dispose();
        }

        // --------------------------------------------------------------- thread-local scope isolation

        [TestMethod]
        public void ThreadLocalIsolation_WorkerDoesNotSeeMainScope_NorIsReclaimed()
        {
            NDArray workerArray = null;
            bool workerSawAScope = true;
            Exception workerError = null;

            using (var outer = NDScope.Open())
            {
                Assert.AreSame(outer, NDScope.Current, "the opening thread's scope is current here");
                var mainTemp = np.arange(10);                         // tracked by outer
                Assert.AreSame(outer, mainTemp.TrackingScope);

                var worker = new Thread(() =>
                {
                    try
                    {
                        workerSawAScope = NDScope.Current != null;    // MUST be false — its own thread-static
                        workerArray = np.arange(5);                   // built off-main -> not tracked by outer
                    }
                    catch (Exception e) { workerError = e; }
                });
                worker.Start();
                worker.Join();

                Assert.IsNull(workerError);
                Assert.IsFalse(workerSawAScope, "a worker thread must NOT see the main thread's open scope (thread-static stack)");
                Assert.IsNotNull(workerArray);
                Assert.IsNull(workerArray.TrackingScope, "an array built on another thread is not tracked by main's scope");
                Assert.IsFalse(workerArray.IsDisposed);
            }

            // outer closed: it reclaimed mainTemp only; the off-thread array was never its responsibility.
            Assert.IsFalse(workerArray.IsDisposed, "main's scope must not reclaim an array built on another thread");
            Assert.AreEqual(4L, workerArray.GetInt64(4), "the off-thread array's data is intact");
            workerArray.Dispose();
        }

        [TestMethod]
        public void ConcurrentScopes_EachThreadSeesOnlyItsOwnStack()
        {
            const int Threads = 8, Iterations = 300;
            long errors = 0;

            // Eight threads open/nest/close scopes concurrently. Each must ALWAYS see its own scope
            // as current and its own temps tracked by its own scope — never another thread's — which
            // is exactly what the [ThreadStatic] stack guarantees under overlap.
            Parallel.For(0, Threads, t =>
            {
                for (int i = 0; i < Iterations; i++)
                {
                    using var mine = NDScope.Open();
                    if (!ReferenceEquals(NDScope.Current, mine)) Interlocked.Increment(ref errors);
                    var temp = np.arange(4 + t);
                    if (!ReferenceEquals(temp.TrackingScope, mine)) Interlocked.Increment(ref errors);

                    using (var inner = NDScope.Open())
                    {
                        if (!ReferenceEquals(NDScope.Current, inner)) Interlocked.Increment(ref errors);
                        var t2 = np.arange(3);
                        if (!ReferenceEquals(t2.TrackingScope, inner)) Interlocked.Increment(ref errors);
                    }

                    if (!ReferenceEquals(NDScope.Current, mine)) Interlocked.Increment(ref errors);   // inner popped back to mine
                }
            });

            Assert.AreEqual(0L, errors, "each thread must see only its own scope stack while all threads run concurrently");
            Assert.IsNull(NDScope.Current, "no scope leaks onto the calling thread after the parallel region");
        }

        // ------------------------------------------------------ cross-thread hand-off + receiver Attach

        [TestMethod]
        public void CrossThreadHandoff_ReceiverAttachesIntoScope_Reclaims()
        {
            const int Producers = 4, PerProducer = 500;
            var queue = new ConcurrentQueue<NDArray>();
            using var done = new CountdownEvent(Producers);
            long errors = 0, reclaimed = 0;

            var consumer = new Thread(() =>
            {
                while (!done.IsSet || !queue.IsEmpty)
                {
                    var batch = new List<NDArray>();
                    using (var s = NDScope.Open())
                    {
                        for (int k = 0; k < 32 && queue.TryDequeue(out var r); k++)
                        {
                            // Produced under a scope on ANOTHER thread, yielded with no parent -> fully
                            // detached (TrackingScope == null): safe to read and to adopt here.
                            if (r.IsDisposed || r.GetInt64(0) != 62 || r.TrackingScope != null)
                                Interlocked.Increment(ref errors);
                            NDScope.Attach(r);            // adopt the handed-off result into THIS thread's scope
                            batch.Add(r);
                        }
                    }   // scope close reclaims every attached result

                    foreach (var r in batch)
                    {
                        if (!r.IsDisposed) Interlocked.Increment(ref errors);
                        Interlocked.Increment(ref reclaimed);
                    }
                    if (batch.Count == 0) Thread.SpinWait(200);
                }
            });
            consumer.Start();

            Parallel.For(0, Producers, p =>
            {
                var a = np.arange(64);
                for (int i = 0; i < PerProducer; i++)
                    queue.Enqueue(np.roll(a, 2));         // scoped internally; result detached to the caller
                a.Dispose();
                done.Signal();
            });
            consumer.Join();

            Assert.AreEqual(0L, errors, "detached cross-thread results must be adoptable via Attach and reclaimed by the receiver's scope");
            Assert.AreEqual((long)Producers * PerProducer, reclaimed);
        }

        // --------------------------------------------------- woven NESTED op under concurrent threads

        [TestMethod]
        public void ManyThreads_NestedWoven_Concurrent_ZeroErrors_BoundedStrands()
        {
            const int Threads = 8, Iterations = 800;

            var warm = np.arange(36).reshape(6, 6);
            using (var _ = np.roll(warm, 1)) { }
            warm.Dispose();
            DrainGC();
            SizeBucketedBufferPool.ResetCounters();

            long errors = 0;
            Parallel.For(0, Threads, _ =>
            {
                var a = np.arange(36).reshape(6, 6);       // per-thread operand: no shared mutable state
                long local = 0;
                for (int i = 0; i < Iterations; i++)
                {
                    int k = 1 + (i % 5);
                    using var r = np.roll(a, k);           // axis=null -> woven roll nests woven roll
                    if (r.GetInt64(0, 0) != 36 - k) local++;
                    if (a.IsDisposed) local++;
                }
                a.Dispose();
                Interlocked.Add(ref errors, local);
            });

            long deficit = Acquisitions - Releases;
            long calls = (long)Threads * Iterations;
            Assert.AreEqual(0L, errors, "nested woven ops must stay value-correct under concurrency");
            // A single per-call strand from either nested scope would read as ~`calls`; allow a tiny
            // per-thread constant for lazily-initialised machinery, never a slope.
            Assert.IsTrue(deficit <= Threads * 8,
                $"nested woven under concurrency stranded {deficit} buffers over {calls} calls (acq {Acquisitions}, rel {Releases})");
        }
    }
}
