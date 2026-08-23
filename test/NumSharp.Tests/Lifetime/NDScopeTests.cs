using System;
using System.Collections.Generic;
using System.Threading;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Generic;

namespace NumSharp.Tests.Lifetime
{
    /// <summary>
    ///     Functional proof of <see cref="NDScope"/> — the ambient reclamation scope that lets a
    ///     boundary method keep its original body while every internal intermediate is eagerly
    ///     returned to the buffer pool. Each test pins one contract from the design:
    ///     tracking, <c>Returns</c> yield/no-op/re-parenting, <c>out</c>-parameter egress,
    ///     view-over-base ARC safety, exception paths, <c>Detach</c>, pooling, and the
    ///     thread-affinity rules. Cross-thread stress lives in <see cref="NDScopeStressTests"/>.
    /// </summary>
    [TestClass]
    [DoNotParallelize]   // several tests read the process-wide pool counters
    public class NDScopeTests
    {
        private static void DrainGC()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }

        private static long Acquisitions => SizeBucketedBufferPool.Hits + SizeBucketedBufferPool.Misses + SizeBucketedBufferPool.ZeroedAllocs;
        private static long Releases => SizeBucketedBufferPool.Returns + SizeBucketedBufferPool.ReturnsFreed;

        // ------------------------------------------------------------------ core contracts

        [TestMethod]
        public void TempsReclaimed_ResultSurvives()
        {
            NDArray temp = null;
            NDArray result;
            using (var scope = NDScope.Open())
            {
                temp = np.arange(5);              // tracked
                result = scope.Returns(temp + 1); // fresh, yielded
            }

            Assert.IsTrue(temp.IsDisposed, "the internal temp must be reclaimed at scope exit");
            Assert.IsFalse(result.IsDisposed, "the yielded result must survive");
            Assert.AreEqual(4L, result.GetInt64(3));
            result.Dispose();
        }

        [TestMethod]
        public void Returns_UntrackedArray_IsNoOp_AndNotAdoptedByParent()
        {
            var outside = np.arange(3);           // constructed before any scope
            using (var outer = NDScope.Open())
            {
                using (var inner = NDScope.Open())
                {
                    var passedThrough = inner.Returns(outside);
                    Assert.AreSame(outside, passedThrough);
                }
            }                                     // if inner wrongly re-tracked it, outer would dispose it here

            Assert.IsFalse(outside.IsDisposed, "an untracked array must pass through Returns untouched");
            Assert.AreEqual(2L, outside.GetInt64(2));
            outside.Dispose();
        }

        [TestMethod]
        public void Returns_Reparents_DroppedInnerResult_ReclaimedByOuter()
        {
            static NDArray Doubled(NDArray x)
            {
                using var s = NDScope.Open();
                return s.Returns(x * 2);
            }

            var input = np.arange(4);
            NDArray dropped;
            using (var outer = NDScope.Open())
            {
                dropped = Doubled(input);
                Assert.IsFalse(dropped.IsDisposed, "yielded result must be alive inside the parent scope");
                Assert.AreEqual(6L, dropped.GetInt64(3));
            }

            Assert.IsTrue(dropped.IsDisposed, "the parent scope must reclaim a dropped inner result");
            Assert.IsFalse(input.IsDisposed, "inputs are never tracked");
            input.Dispose();
        }

        [TestMethod]
        public void Returns_WithNoParent_FullyDetaches()
        {
            static NDArray Doubled(NDArray x)
            {
                using var s = NDScope.Open();
                return s.Returns(x * 2);
            }

            var input = np.arange(4);
            var r = Doubled(input);               // no enclosing scope: caller owns it outright
            Assert.IsFalse(r.IsDisposed);
            Assert.AreEqual(2L, r.GetInt64(1));
            r.Dispose();
            input.Dispose();
        }

        [TestMethod]
        public void Returns_ArrayOverload_YieldsWholeTuple()
        {
            var m = np.array(new int[,] { { 0, 1 }, { 2, 0 } });
            var tup = np.nonzero(m);              // NDScope inside NonZero yields the column views

            Assert.AreEqual(2, tup.Length);
            Assert.IsFalse(tup[0].IsDisposed);
            Assert.IsFalse(tup[1].IsDisposed);
            Assert.AreEqual(0L, tup[0].GetInt64(0));
            Assert.AreEqual(1L, tup[0].GetInt64(1));
            Assert.AreEqual(1L, tup[1].GetInt64(0));
            Assert.AreEqual(0L, tup[1].GetInt64(1));

            foreach (var t in tup) t.Dispose();
            m.Dispose();
        }

        [TestMethod]
        public void OutParameter_Egress_ViaReturns()
        {
            static void Twice(NDArray x, out NDArray result, out NDArray leakedTemp)
            {
                using var s = NDScope.Open();
                leakedTemp = x + 0;               // internal temp we deliberately observe
                result = s.Returns(x + x);        // egress through the out slot
            }

            var input = np.arange(4);
            Twice(input, out var doubled, out var temp);

            Assert.IsFalse(doubled.IsDisposed, "the out-parameter value must survive the scope");
            Assert.IsTrue(temp.IsDisposed, "non-yielded internals must be reclaimed");
            Assert.AreEqual(6L, doubled.GetInt64(3));
            doubled.Dispose();
            input.Dispose();
        }

        [TestMethod]
        public void CallerOut_RelayedBack_NeedsNoYield()
        {
            static NDArray Relay(NDArray x, NDArray @out)
            {
                using var s = NDScope.Open();
                np.copyto(@out, x);
                return s.Returns(@out);           // untracked -> provable no-op, so the blanket rule is safe
            }

            var src = np.arange(6);
            var dst = np.empty(new Shape(6), NPTypeCode.Int32);
            var back = Relay(src, dst);

            Assert.AreSame(dst, back);
            Assert.IsFalse(dst.IsDisposed);
            Assert.AreEqual(5, dst.GetInt32(5));
            src.Dispose();
            dst.Dispose();
        }

        [TestMethod]
        public void YieldedView_OverTrackedBase_KeepsDataAlive()
        {
            NDArray view;
            using (var scope = NDScope.Open())
            {
                var flat = np.arange(6);          // tracked base
                view = scope.Returns(flat.reshape(2, 3));
            }                                     // base wrapper released; ARC keeps the buffer via the view

            Assert.IsFalse(view.IsDisposed);
            Assert.AreEqual(5L, view.GetInt64(1, 2));
            view.Dispose();
        }

        [TestMethod]
        public void ExceptionPath_ReclaimsTemps_InputsUnharmed()
        {
            var input = np.arange(4);
            NDArray temp = null;
            try
            {
                using var scope = NDScope.Open();
                temp = input * 3;
                throw new InvalidOperationException("boom");
            }
            catch (InvalidOperationException)
            {
            }

            Assert.IsTrue(temp.IsDisposed, "temps must be reclaimed on the throw path");
            Assert.IsFalse(input.IsDisposed, "inputs must never be touched");
            Assert.AreEqual(3L, input.GetInt64(3));
            input.Dispose();
        }

        [TestMethod]
        public void Detach_ArraySurvivesEveryScope()
        {
            NDArray cached;
            using (var scope = NDScope.Open())
            {
                cached = np.arange(4);            // tracked...
                NDScope.Detach(cached);           // ...then permanently removed from scope ownership
            }

            Assert.IsFalse(cached.IsDisposed, "a detached array must survive the scope");
            Assert.AreEqual(2L, cached.GetInt64(2));
            cached.Dispose();
        }

        [TestMethod]
        public void ScopeDispose_IsIdempotent()
        {
            var scope = NDScope.Open();
            var t = np.arange(3);
            scope.Dispose();
            scope.Dispose();                       // second dispose must be a no-op
            Assert.IsTrue(t.IsDisposed);
        }

        [TestMethod]
        public void ScopePooling_ManyOpenClose_NoCorruption()
        {
            for (int i = 0; i < 10_000; i++)
            {
                using var scope = NDScope.Open();
                var t = np.arange(4);
                var r = scope.Returns(t + i);
                if (r.GetInt64(3) != 3 + i)
                    Assert.Fail($"iteration {i}: wrong value");
                r.Dispose();
            }
        }

        [TestMethod]
        public void BroadcastReadOnlyInput_Unharmed()
        {
            var b = np.broadcast_to(np.arange(3), new Shape(4, 3));   // read-only view input
            var ones = np.ones(new Shape(4, 3), NPTypeCode.Int32);
            using var mask = np.logical_and(b, ones);

            Assert.IsFalse(b.IsDisposed);
            Assert.IsFalse(b.Shape.IsWriteable);
            Assert.IsFalse(mask.GetBoolean(0, 0));   // 0 AND 1
            Assert.IsTrue(mask.GetBoolean(0, 1));    // 1 AND 1
        }

        [TestMethod]
        public void CrossThread_YieldedResult_UsableOnAnotherThread()
        {
            var a = np.arange(64).astype(NPTypeCode.Int32);
            var b = np.ones(new Shape(64), NPTypeCode.Int32);
            NDArray crossed = null;
            Exception threadError = null;

            var th = new Thread(() =>
            {
                try { crossed = np.logical_and(a, b); }   // scope on THIS thread, no parent -> detached
                catch (Exception e) { threadError = e; }
            });
            th.Start();
            th.Join();

            Assert.IsNull(threadError);
            Assert.IsFalse(crossed.IsDisposed);
            Assert.IsFalse(crossed.GetBoolean(0));
            Assert.IsTrue(crossed.GetBoolean(1));
            crossed.Dispose();
            a.Dispose();
            b.Dispose();
        }

        // ------------------------------------------------------------------ counters: zero strands

        [TestMethod]
        public void ScopedSurface_ZeroStrandsPerCall()
        {
            var a = np.arange(64).astype(NPTypeCode.Int32);
            var b = np.ones(new Shape(64), NPTypeCode.Int32);
            var d = np.array(new double[] { 3, 1, 2, 5, 4 });
            var idx = np.array(new int[] { 0, 2, 4 });
            var mask = np.array(new[] { true, false, true, false, true });
            var maskT = mask.MakeGeneric<bool>();   // hoisted: an in-loop MakeGeneric would itself strand

            void Cycle()
            {
                using (var r = np.logical_and(a, b)) { }
                using (var r = np.logical_or(a, b)) { }
                using (var r = np.logical_not(a)) { }
                using (var r = np.logical_xor(a, b)) { }
                using (var r = np.roll(a, 7)) { }
                using (var r = np.isin(a, idx)) { }
                using (var r = np.sort(d)) { }
                using (var r = np.argsort(d)) { }
                using (var r = np.ptp(a)) { }
                using (var r = np.clip(d, (NDArray)1.5, (NDArray)4.5)) { }
                using (var r = np.median(d)) { }
                using (var r = d[maskT]) { }
                using (var r = d[idx]) { }
                var tup = np.nonzero(mask);
                foreach (var t in tup) t.Dispose();
            }

            // Warm every kernel/cache first so one-time setup is outside the measured window.
            for (int i = 0; i < 3; i++) Cycle();
            DrainGC();

            SizeBucketedBufferPool.ResetCounters();
            const int N = 200;
            for (int i = 0; i < N; i++) Cycle();
            long deficit = Acquisitions - Releases;

            // 14 ops per cycle; a single stranded buffer per call would read as >= N.
            Assert.IsTrue(deficit <= 8,
                $"scoped surface stranded {deficit} buffers over {N} cycles (acq {Acquisitions}, rel {Releases})");

            a.Dispose(); b.Dispose(); d.Dispose(); idx.Dispose(); maskT.Dispose(); mask.Dispose();
        }

        [TestMethod]
        public void NestedDrop_ParentReclaimsEverything()
        {
            static NDArray Tripled(NDArray x)
            {
                using var s = NDScope.Open();
                return s.Returns(x * 3);
            }

            var input = np.arange(16);
            var dropped = new List<NDArray>(200);
            using (var outer = NDScope.Open())
            {
                for (int i = 0; i < 200; i++)
                    dropped.Add(Tripled(input));   // never disposed by hand
            }

            foreach (var r in dropped)
                Assert.IsTrue(r.IsDisposed, "outer scope must reclaim every dropped inner result");
            Assert.IsFalse(input.IsDisposed);
            input.Dispose();
        }

        // ------------------------------------------------------------------ scoped ops: value smoke

        [TestMethod]
        public void ScopedOps_ValuesCorrect()
        {
            using (var r = np.roll(np.arange(5), 2))
            {
                Assert.AreEqual(3L, r.GetInt64(0));
                Assert.AreEqual(4L, r.GetInt64(1));
                Assert.AreEqual(0L, r.GetInt64(2));
            }

            using (var r = np.isin(np.array(new[] { 1, 2, 3, 4 }), np.array(new[] { 2, 4 })))
            {
                Assert.IsFalse(r.GetBoolean(0));
                Assert.IsTrue(r.GetBoolean(1));
                Assert.IsFalse(r.GetBoolean(2));
                Assert.IsTrue(r.GetBoolean(3));
            }

            using (var r = np.sort(np.array(new double[] { 3, 1, 2 })))
            {
                Assert.AreEqual(1.0, r.GetDouble(0));
                Assert.AreEqual(2.0, r.GetDouble(1));
                Assert.AreEqual(3.0, r.GetDouble(2));
            }

            using (var r = np.argsort(np.array(new double[] { 3, 1, 2 })))
            {
                Assert.AreEqual(1L, r.GetInt64(0));
                Assert.AreEqual(2L, r.GetInt64(1));
                Assert.AreEqual(0L, r.GetInt64(2));
            }

            using (var r = np.ptp(np.arange(5)))
                Assert.AreEqual(4L, r.GetInt64(0));

            using (var r = np.median(np.arange(5)))
                Assert.AreEqual(2.0, r.GetDouble(0));

            using (var r = np.clip(np.array(new double[] { 0, 2, 9 }), (NDArray)1.0, (NDArray)5.0))
            {
                Assert.AreEqual(1.0, r.GetDouble(0));
                Assert.AreEqual(2.0, r.GetDouble(1));
                Assert.AreEqual(5.0, r.GetDouble(2));
            }

            using (var r = np.correlate(np.array(new double[] { 1, 2, 3 }), np.array(new double[] { 1, 1 }), "valid"))
            {
                Assert.AreEqual(2, r.size);
                Assert.AreEqual(3.0, r.GetDouble(0));
                Assert.AreEqual(5.0, r.GetDouble(1));
            }

            using (var r = np.convolve(np.array(new double[] { 1, 2, 3 }), np.array(new double[] { 1, 1 })))
            {
                Assert.AreEqual(4, r.size);
                Assert.AreEqual(1.0, r.GetDouble(0));
                Assert.AreEqual(3.0, r.GetDouble(1));
                Assert.AreEqual(5.0, r.GetDouble(2));
                Assert.AreEqual(3.0, r.GetDouble(3));
            }

            using (var r = np.corrcoef(np.array(new double[,] { { 1, 2, 3 }, { 1, 2, 3 } })))
            {
                Assert.AreEqual(1.0, r.GetDouble(0, 0), 1e-12);
                Assert.AreEqual(1.0, r.GetDouble(0, 1), 1e-12);
            }

            using (var r = np.cov(np.array(new double[] { 0, 1, 2 })))
                Assert.AreEqual(1.0, r.GetDouble(0), 1e-12);

            // typed generic operators + indexer (the alias-handoff sites)
            var ba = np.array(new[] { true, true, false }).MakeGeneric<bool>();
            var bb = np.array(new[] { true, false, false }).MakeGeneric<bool>();
            using (NDArray<bool> and = ba & bb)
            {
                Assert.IsTrue(and[0]);
                Assert.IsFalse(and[1]);
                Assert.IsFalse(and[2]);
            }
            using (NDArray<bool> slice = ba["0:2"])
            {
                Assert.AreEqual(2, slice.size);
                Assert.IsTrue(slice[0]);
            }
            ba.Dispose();
            bb.Dispose();

            // fancy set through the scoped dispatcher, then fancy get
            var target = np.arange(6).astype(NPTypeCode.Int32);
            var where = np.array(new int[] { 1, 4 });
            target[where] = np.array(new int[] { 100, 200 });
            Assert.AreEqual(100, target.GetInt32(1));
            Assert.AreEqual(200, target.GetInt32(4));
            using (var got = target[where])
            {
                Assert.AreEqual(100, got.GetInt32(0));
                Assert.AreEqual(200, got.GetInt32(1));
            }
            target.Dispose();
            where.Dispose();
        }
    }
}
