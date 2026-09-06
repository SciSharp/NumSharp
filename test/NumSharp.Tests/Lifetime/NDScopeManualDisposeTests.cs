using System;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Tests.Lifetime
{
    /// <summary>
    ///     What happens when we dispose an array OURSELVES inside a scope — the interaction between a
    ///     manual <see cref="NDArray.Dispose"/> on a TRACKED array and the scope's later reclamation
    ///     sweep, which will call <c>Dispose()</c> on the same array again. The load-bearing property
    ///     is <b>freed exactly once</b>: <see cref="NDArray.Dispose"/> is CAS-idempotent, so the sweep's
    ///     second dispose is a no-op and the buffer is never double-returned to the pool (a double-free
    ///     would hand the same buffer to two later allocations — silent aliasing corruption). These
    ///     tests pin that across the manual/scope, manual/view, manual/exception, double-manual, the
    ///     mixed manual+scope reclaim pattern (DISPOSAL-GUIDELINES §6's direct-Dispose form), and a
    ///     hostile aliasing probe.
    /// </summary>
    [TestClass]
    [DoNotParallelize]   // reads process-wide pool counters
    public class NDScopeManualDisposeTests
    {
        private static void DrainGC()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }

        private static long Acquisitions => SizeBucketedBufferPool.Hits + SizeBucketedBufferPool.Misses + SizeBucketedBufferPool.ZeroedAllocs;
        private static long Releases => SizeBucketedBufferPool.Returns + SizeBucketedBufferPool.ReturnsFreed;

        [TestMethod]
        public void ManualDisposeOfTrackedTemp_ScopeSweepIsNoOp_FreedExactlyOnce()
        {
            NDArray temp;
            long relAfterManual;
            DrainGC();
            using (var scope = NDScope.Open())
            {
                temp = np.arange(64);
                Assert.AreSame(scope, temp.TrackingScope, "the temp is tracked by the open scope");
                Assert.IsFalse(temp.IsDisposed);

                long relBefore = Releases;
                temp.Dispose();                                // MANUAL dispose, inside the scope
                relAfterManual = Releases;

                Assert.IsTrue(temp.IsDisposed);
                Assert.IsTrue(temp.Storage.InternalArray.IsReleased, "the manual dispose freed the buffer");
                Assert.IsTrue(relAfterManual > relBefore, "the manual dispose returned exactly this buffer to the pool");
                Assert.AreSame(scope, temp.TrackingScope, "manual Dispose does NOT un-track — the sweep will still visit it");
            }
            // The scope sweep called temp.Dispose() a SECOND time. It must be a no-op.
            Assert.AreEqual(relAfterManual, Releases, "the scope sweep must NOT return the manually-disposed buffer a second time");
            Assert.IsTrue(temp.IsDisposed);
            Assert.IsNull(temp.TrackingScope, "the sweep clears the tracking back-pointer");
            Assert.IsTrue(temp.Storage.InternalArray.IsReleased);
        }

        [TestMethod]
        public void DoubleManualDispose_ThenScopeSweep_FreedOnce()
        {
            NDArray temp;
            long relAfterFirst;
            DrainGC();
            using (var scope = NDScope.Open())
            {
                temp = np.arange(32);
                temp.Dispose();
                relAfterFirst = Releases;
                temp.Dispose();                                // second manual dispose
                Assert.AreEqual(relAfterFirst, Releases, "a second manual Dispose must not free again");
            }
            // third dispose, via the sweep
            Assert.AreEqual(relAfterFirst, Releases, "the scope sweep (third dispose) must not free again");
            Assert.IsTrue(temp.IsDisposed);
            Assert.IsTrue(temp.Storage.InternalArray.IsReleased);
        }

        [TestMethod]
        public void ManualDisposeBase_YieldedView_StaysValid_FreedOnceAtViewDispose()
        {
            NDArray view;
            using (var scope = NDScope.Open())
            {
                var baseArr = np.arange(6);                    // tracked owner
                view = baseArr.reshape(2, 3);                  // tracked view — shares the buffer (refcount 2)
                var slice = baseArr.Storage.InternalArray;

                baseArr.Dispose();                             // MANUAL dispose of the base (refcount 2 -> 1)
                Assert.IsTrue(baseArr.IsDisposed);
                Assert.IsFalse(slice.IsReleased, "the view still references the buffer — it must stay alive");
                Assert.AreEqual(5L, view.GetInt64(1, 2), "the view reads correct data after the base is manually disposed");

                scope.Returns(view);                           // yield the view out of the scope
            }
            // sweep: baseArr.Dispose() again (no-op); the yielded view was not swept.
            Assert.IsFalse(view.IsDisposed, "the yielded view survives the scope");
            Assert.IsFalse(view.Storage.InternalArray.IsReleased, "buffer still alive via the view");
            Assert.AreEqual(5L, view.GetInt64(1, 2));

            view.Dispose();                                    // the LAST reference -> freed now, exactly once
            Assert.IsTrue(view.Storage.InternalArray.IsReleased);
        }

        [TestMethod]
        public void ManualDisposeThenReturnsSame_Graceful_NoCorruption()
        {
            // Disposing the very array you then return is a USER bug. It must degrade gracefully — a
            // disposed array handed back, never a crash or a double-free of a live buffer.
            static NDArray SelfDispose(NDArray x)
            {
                using var s = NDScope.Open();
                var t = x + 1;
                t.Dispose();                                   // dispose what we are about to return
                return s.Returns(t);
            }

            var input = np.arange(4);
            var r = SelfDispose(input);

            Assert.IsTrue(r.IsDisposed, "returning a manually-disposed temp yields a disposed array — graceful, not a crash");
            Assert.IsFalse(input.IsDisposed, "the input is untouched");
            Assert.AreEqual(3L, input.GetInt64(3), "input data intact");
            input.Dispose();
        }

        [TestMethod]
        public void ExceptionAfterManualDispose_ScopeFinallySweep_FreedOnce_InputUntouched()
        {
            var input = np.arange(5);                          // built before any scope -> never tracked
            NDArray temp = null;
            long relAfterManual = 0;
            DrainGC();
            try
            {
                using var scope = NDScope.Open();
                temp = np.arange(30);                          // tracked; exactly one buffer, no hidden transient
                long before = Releases;
                temp.Dispose();                                // MANUAL dispose
                relAfterManual = Releases;
                Assert.IsTrue(relAfterManual > before, "manual dispose freed the temp");
                throw new InvalidOperationException("boom after the manual dispose");
            }
            catch (InvalidOperationException)
            {
            }

            // temp was the ONLY tracked temp; the finally swept it (already disposed) -> no second free.
            Assert.AreEqual(relAfterManual, Releases, "the finally sweep must not free the manually-disposed temp a second time");
            Assert.IsTrue(temp.IsDisposed);
            Assert.IsTrue(temp.Storage.InternalArray.IsReleased);
            Assert.IsFalse(input.IsDisposed, "the input is untouched on the exception path");
            Assert.AreEqual(2L, input.GetInt64(2));            // arange(5)[2] == 2
            input.Dispose();
        }

        [TestMethod]
        public void MixedManualAndScopeReclaim_Balanced_NoDoubleFree_NoLeak()
        {
            // The §6 "direct Dispose() for a mid-method lifetime" pattern, mixed with scope reclamation:
            // some temps disposed by hand, the rest reclaimed by the sweep. Every buffer must be freed
            // EXACTLY once — a double-free reads as Releases > Acquisitions (deficit < 0), a leak as
            // deficit >> 0.
            void Cycle()
            {
                using var scope = NDScope.Open();
                var a = np.arange(48);                         // tracked
                var b = a - 1;                                 // tracked; last use of `a`
                a.Dispose();                                   // MANUAL dispose of a
                var c = b + b;                                 // tracked; last use of `b`
                b.Dispose();                                   // MANUAL dispose of b
                // c is left to the scope sweep. No Returns -> everything reclaimed.
                _ = c.size;
            }

            for (int i = 0; i < 3; i++) Cycle();
            DrainGC();
            SizeBucketedBufferPool.ResetCounters();
            const int N = 200;
            for (int i = 0; i < N; i++) Cycle();

            long deficit = Acquisitions - Releases;
            Assert.IsTrue(deficit >= 0,
                $"Releases exceeded Acquisitions by {-deficit} — a DOUBLE-FREE (manual dispose + scope sweep both returned a buffer)");
            Assert.IsTrue(deficit <= 8,
                $"mixed manual+scope reclaim stranded {deficit} buffers over {N} cycles (acq {Acquisitions}, rel {Releases})");
        }

        [TestMethod]
        public void ManualDisposeThenScopeSweep_NoAliasing_HostileProbe()
        {
            // The strong double-free detector: if a manually-disposed buffer were freed AGAIN by the
            // sweep, the pool's free list would hold it twice and hand it to two later allocations —
            // writes to one would corrupt the other. Churn the manual-dispose-inside-scope pattern,
            // then allocate two arrays and prove they are independent.
            const int Size = 37;   // odd size -> its own pool bucket, easy to collide on
            for (int round = 0; round < 100; round++)
            {
                using var scope = NDScope.Open();
                var t = np.arange(Size);
                t.Dispose();                                   // manual dispose; sweep will dispose again
            }

            var a = np.zeros(new Shape(Size), NPTypeCode.Int64);
            var b = np.zeros(new Shape(Size), NPTypeCode.Int64);
            for (int i = 0; i < Size; i++) a.SetInt64(7L, i);
            for (int i = 0; i < Size; i++) b.SetInt64(3L, i);   // if a and b alias, this overwrites a too

            for (int i = 0; i < Size; i++)
                Assert.AreEqual(7L, a.GetInt64(i), $"a[{i}] was corrupted — a and b alias the same buffer (double-free)");
            for (int i = 0; i < Size; i++)
                Assert.AreEqual(3L, b.GetInt64(i), $"b[{i}] wrong");

            a.Dispose();
            b.Dispose();
        }
    }
}
