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
        public void Attach_AdoptsUntrackedArray_ReclaimedAtExit()
        {
            var received = np.arange(6);          // constructed before any scope — untracked
            using (var scope = NDScope.Open())
            {
                NDScope.Attach(received);         // the hot-loop pattern: adopt instead of using/Dispose
                Assert.AreEqual(5L, received.GetInt64(5));
            }

            Assert.IsTrue(received.IsDisposed, "an attached array must be reclaimed at scope exit");
        }

        [TestMethod]
        public void Attach_MovesOwnership_OldScopeNoLongerDisposes()
        {
            using (var outer = NDScope.Open())
            {
                var x = np.arange(3);             // tracked by OUTER
                using (var inner = NDScope.Open())
                {
                    NDScope.Attach(x);            // current is inner -> MOVES x: outer's slot cleared
                    Assert.IsFalse(x.IsDisposed);
                }                                  // inner (the new owner) reclaims it

                Assert.IsTrue(x.IsDisposed, "the adopting scope must reclaim the moved array");

                var fresh = np.arange(4);          // tracked by outer (current again)
                NDScope.Attach(fresh);             // already tracked by the current scope -> no-op
                Assert.IsFalse(fresh.IsDisposed);
            }                                      // outer's cleared slot for x is skipped; fresh reclaimed
        }

        [TestMethod]
        public void Attach_NoOpenScope_IsNoOp()
        {
            var free = np.arange(3);
            NDScope.Attach(free);                  // nothing to adopt into — must not throw or track
            Assert.IsFalse(free.IsDisposed);
            Assert.AreEqual(1L, free.GetInt64(1));
            free.Dispose();
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
        public void Dispose_OutOfOrder_OuterBeforeInner_StackStaysConsistent()
        {
            Assert.IsNull(NDScope.Current, "no scope must be open at test start");

            var outer = NDScope.Open();
            var inner = NDScope.Open();
            var t = np.arange(4);                  // tracked by inner
            NDArray r = null;
            try
            {
                Assert.AreSame(inner, NDScope.Current);

                outer.Dispose();                   // OUT OF ORDER: splice outer; current stays inner
                Assert.AreSame(inner, NDScope.Current,
                    "disposing the outer scope must leave the innermost scope current");
                Assert.IsFalse(t.IsDisposed, "inner's temp must survive the outer's out-of-order dispose");

                r = inner.Returns(t + 1);          // inner is now parentless -> r fully detached
            }
            finally
            {
                inner.Dispose();
                outer.Dispose();                   // idempotent no-op
            }

            Assert.IsNull(NDScope.Current, "the scope stack must be empty and uncorrupted");
            Assert.IsTrue(t.IsDisposed, "inner's non-yielded temp must be reclaimed");
            Assert.IsFalse(r.IsDisposed, "the yielded, detached result must survive");
            Assert.AreEqual(4L, r.GetInt64(3));
            r.Dispose();
        }

        [TestMethod]
        public void Dispose_MiddleScope_SplicedOut_InnerPopsToOuter()
        {
            Assert.IsNull(NDScope.Current, "no scope must be open at test start");

            var a = NDScope.Open();   // outer
            var b = NDScope.Open();   // middle
            var c = NDScope.Open();   // inner
            try
            {
                Assert.AreSame(c, NDScope.Current);

                b.Dispose();          // splice the MIDDLE out; current stays c; c re-parents onto a
                Assert.AreSame(c, NDScope.Current, "disposing a middle scope must not change the innermost");

                c.Dispose();          // must pop to a (the spliced-out b must not resurface)
                Assert.AreSame(a, NDScope.Current, "inner must pop to the outer, skipping the spliced-out middle");
            }
            finally
            {
                c.Dispose();
                b.Dispose();
                a.Dispose();
            }

            Assert.IsNull(NDScope.Current, "the stack must be fully unwound and uncorrupted");
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
            var d2 = np.array(new double[] { 2, 4, 6, 1, 3 });
            var f2d = np.arange(24).reshape(4, 6).astype(NPTypeCode.Single);
            var dn = np.array(new double[] { 1, double.NaN, 3, 2 });
            var idx = np.array(new int[] { 0, 2, 4 });
            var mask = np.array(new[] { true, false, true, false, true });
            var maskT = mask.MakeGeneric<bool>();   // hoisted: an in-loop MakeGeneric would itself strand
            var cond = d > d2;                      // hoisted where-condition (operand, not measured garbage)
            var gatherIdx = np.argsort(f2d, 1);     // hoisted take_along_axis indices

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

                // ---- the widened scoped surface (Phase A adoption) ----
                using (var r = np.where(cond, d, d2)) { }
                using (var r = np.where(cond, d, 0)) { }          // scalar y -> asanyarray under the scope
                using (var r = np.union1d(d, d2)) { }
                using (var r = np.intersect1d(d, d2)) { }
                using (var r = np.setxor1d(d, d2)) { }
                using (var r = np.setdiff1d(d, d2)) { }
                using (var r = np.std(f2d, 1, dtype: NPTypeCode.Single)) { }   // axis IL path + f32 cast
                using (var r = np.var(f2d, 1, dtype: NPTypeCode.Single)) { }
                using (var r = np.average(d)) { }
                using (var r = np.take_along_axis(f2d, gatherIdx, 1)) { }
                using (var r = (NDArray)np.unique(d)) { }
                foreach (var t in (NDArray[])np.unique_counts(a)) t.Dispose();
                using (var r = np.nanargmax(dn, 0)) { }
                using (var r = np.nancumsum(dn)) { }
                np.count_nonzero(a);                               // long return; MakeGeneric alias scoped
            }

            // Warm every kernel/cache first so one-time setup is outside the measured window.
            for (int i = 0; i < 3; i++) Cycle();
            DrainGC();

            SizeBucketedBufferPool.ResetCounters();
            const int N = 200;
            for (int i = 0; i < N; i++) Cycle();
            long deficit = Acquisitions - Releases;

            // 29 ops per cycle; a single stranded buffer per call would read as >= N.
            Assert.IsTrue(deficit <= 8,
                $"scoped surface stranded {deficit} buffers over {N} cycles (acq {Acquisitions}, rel {Releases})");

            a.Dispose(); b.Dispose(); d.Dispose(); d2.Dispose(); f2d.Dispose(); dn.Dispose();
            idx.Dispose(); maskT.Dispose(); mask.Dispose(); cond.Dispose(); gatherIdx.Dispose();
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

            // ---- the Phase A widened surface ----
            var cond = np.array(new[] { true, false, true });
            using (var r = np.where(cond, np.array(new double[] { 1, 2, 3 }), np.array(new double[] { 9, 8, 7 })))
            {
                Assert.AreEqual(1.0, r.GetDouble(0));
                Assert.AreEqual(8.0, r.GetDouble(1));
                Assert.AreEqual(3.0, r.GetDouble(2));
            }
            using (var r = np.where(cond, np.array(new double[] { 1, 2, 3 }), 0))
            {
                Assert.AreEqual(1.0, r.GetDouble(0));
                Assert.AreEqual(0.0, r.GetDouble(1));
            }
            cond.Dispose();

            using (var r = np.union1d(np.array(new[] { 3, 1 }), np.array(new[] { 2, 3 })))
            {
                Assert.AreEqual(3, r.size);
                Assert.AreEqual(1, r.GetInt32(0));
                Assert.AreEqual(2, r.GetInt32(1));
                Assert.AreEqual(3, r.GetInt32(2));
            }
            using (var r = np.intersect1d(np.array(new[] { 3, 1, 2 }), np.array(new[] { 2, 3 })))
            {
                Assert.AreEqual(2, r.size);
                Assert.AreEqual(2, r.GetInt32(0));
                Assert.AreEqual(3, r.GetInt32(1));
            }
            using (var r = np.setxor1d(np.array(new[] { 3, 1, 2 }), np.array(new[] { 2, 3 })))
            {
                Assert.AreEqual(1, r.size);
                Assert.AreEqual(1, r.GetInt32(0));
            }
            using (var r = np.setdiff1d(np.array(new[] { 3, 1, 2 }), np.array(new[] { 2 })))
            {
                Assert.AreEqual(2, r.size);
                Assert.AreEqual(1, r.GetInt32(0));
                Assert.AreEqual(3, r.GetInt32(1));
            }

            using (var r = np.std(np.array(new double[,] { { 1, 3 }, { 2, 4 } }), 1))
            {
                Assert.AreEqual(1.0, r.GetDouble(0), 1e-12);
                Assert.AreEqual(1.0, r.GetDouble(1), 1e-12);
            }
            using (var r = np.var(np.array(new double[,] { { 1, 3 }, { 2, 4 } }), 1))
                Assert.AreEqual(1.0, r.GetDouble(0), 1e-12);

            using (var r = np.average(np.array(new double[] { 1, 2, 3, 4 })))
                Assert.AreEqual(2.5, r.GetDouble(0), 1e-12);

            var src2d = np.array(new double[,] { { 3, 1, 2 }, { 6, 5, 4 } });
            var order = np.argsort(src2d, 1);
            using (var r = np.take_along_axis(src2d, order, 1))
            {
                Assert.AreEqual(1.0, r.GetDouble(0, 0));
                Assert.AreEqual(2.0, r.GetDouble(0, 1));
                Assert.AreEqual(3.0, r.GetDouble(0, 2));
                Assert.AreEqual(4.0, r.GetDouble(1, 0));
            }
            src2d.Dispose();
            order.Dispose();

            using (NDArray r = np.unique(np.array(new[] { 2, 1, 2, 3 })))
            {
                Assert.AreEqual(3, r.size);
                Assert.AreEqual(1, r.GetInt32(0));
            }
            {
                var uc = np.unique_counts(np.array(new[] { 2, 1, 2, 3 }));
                Assert.AreEqual(3, uc.values.size);
                Assert.AreEqual(2L, uc.counts.GetInt64(1));   // value 2 appears twice
                foreach (var t in (NDArray[])uc) t.Dispose();
            }

            Assert.AreEqual(2L, np.nanargmax(np.array(new double[] { 1, double.NaN, 3, 2 })));
            using (var r = np.nancumsum(np.array(new double[] { 1, double.NaN, 3 })))
            {
                Assert.AreEqual(1.0, r.GetDouble(0));
                Assert.AreEqual(1.0, r.GetDouble(1));   // NaN treated as 0
                Assert.AreEqual(4.0, r.GetDouble(2));
            }

            Assert.AreEqual(2L, np.count_nonzero(np.array(new[] { 0, 5, 0, 7 })));
        }
    }
}
