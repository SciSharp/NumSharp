using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Tests.Lifetime
{
    /// <summary>
    ///     Behavioral proof that the <c>[NDScoped]</c> weaver correctly weaves methods returning a
    ///     <b>carrier</b> — the two shapes beyond a bare <see cref="NDArray"/>/<see cref="NDArray"/><c>[]</c>:
    ///     <list type="bullet">
    ///     <item>a <b>ValueTuple</b> of NDArrays (<c>np.polydiv</c> → <c>(q, r)</c>): the weaver yields both
    ///           components through the <see cref="NDScope.Returns{T1,T2}(ValueTuple{T1,T2})"/> overload;</item>
    ///     <item>a <b>result struct</b> (<c>np.unique_counts</c>/<c>unique_all</c>/<c>unique_inverse</c> →
    ///           <c>UniqueCountsResult</c> …): the weaver calls the struct's own
    ///           <c>INDArrayCarrier.YieldTo(scope)</c>, which re-parents every NDArray the struct holds.</item>
    ///     </list>
    ///     For each shape this asserts the three carrier invariants: every member SURVIVES the woven method's
    ///     reclamation (rule R1 — never dispose what you return), a dropped result is RE-PARENTED into an
    ///     enclosing scope (and reclaimed by it, not stranded), and the woven method's internal temporaries
    ///     leave ZERO net buffer strands. <see cref="NDScopeWeaveTests"/> proves these methods carry the
    ///     woven scope local; this proves the carrier egress itself is correct.
    /// </summary>
    [TestClass]
    [DoNotParallelize]   // the zero-strand tests read the process-wide pool counters
    public class NDScopeWeaveCarrierTests
    {
        private static void DrainGC()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }

        private static long Acquisitions => SizeBucketedBufferPool.Hits + SizeBucketedBufferPool.Misses + SizeBucketedBufferPool.ZeroedAllocs;
        private static long Releases => SizeBucketedBufferPool.Returns + SizeBucketedBufferPool.ReturnsFreed;

        // ---------------------------------------------------------------- 1) tuple carrier: np.polydiv

        [TestMethod]
        public void TupleCarrier_Polydiv_BothMembersSurvive_ValueCorrect_InputsUntouched()
        {
            var u = np.array(new double[] { 3, 5, 2 });   // 3x^2 + 5x + 2
            var v = np.array(new double[] { 1, 2 });      // x + 2
            var (q, r) = np.polydiv(u, v);                // numpy: q=[3,-1], r=[4]

            Assert.IsFalse(q.IsDisposed, "the woven tuple's first component must survive (yielded)");
            Assert.IsFalse(r.IsDisposed, "the woven tuple's second component must survive (yielded)");
            Assert.AreEqual(3.0, q.GetDouble(0), 1e-9);
            Assert.AreEqual(-1.0, q.GetDouble(1), 1e-9);
            Assert.AreEqual(4.0, r.GetDouble(0), 1e-9);
            Assert.IsFalse(u.IsDisposed, "inputs are constructed before the scope — untouched");
            Assert.IsFalse(v.IsDisposed);
            Assert.AreEqual(3.0, u.GetDouble(0), "input data intact");

            q.Dispose(); r.Dispose(); u.Dispose(); v.Dispose();
        }

        [TestMethod]
        public void TupleCarrier_Polydiv_DroppedUnderEnclosingScope_BothReparentedAndReclaimed()
        {
            var u = np.array(new double[] { 1, 0, -1, 2 });
            var v = np.array(new double[] { 1, 1 });

            NDArray q, r;
            using (var outer = NDScope.Open())
            {
                (q, r) = np.polydiv(u, v);   // woven; its scope re-parents q and r up into `outer`
                Assert.IsFalse(q.IsDisposed, "a dropped tuple component must stay alive inside the enclosing scope");
                Assert.IsFalse(r.IsDisposed);
            }

            Assert.IsTrue(q.IsDisposed, "the enclosing scope must reclaim the dropped quotient on close");
            Assert.IsTrue(r.IsDisposed, "the enclosing scope must reclaim the dropped remainder on close");
            Assert.IsFalse(u.IsDisposed, "the input is untouched by the enclosing scope");
            u.Dispose(); v.Dispose();
        }

        [TestMethod]
        public void TupleCarrier_Polydiv_ZeroStrands()
        {
            var u = np.array(new double[] { 6, 11, 6, 1, 4, 2, 7 });   // reused across cycles
            var v = np.array(new double[] { 1, 2, 1 });
            void Cycle()
            {
                var (q, r) = np.polydiv(u, v);
                q.Dispose();
                r.Dispose();
            }

            for (int i = 0; i < 3; i++) Cycle();   // warm one-time kernel/cache setup out of the window
            DrainGC();

            SizeBucketedBufferPool.ResetCounters();
            const int N = 200;
            for (int i = 0; i < N; i++) Cycle();
            long deficit = Acquisitions - Releases;

            // polydiv floats u/v, and per output builds a scaled quotient term and a `d*v` product; a
            // leaked internal temp would read as a multiple of N. A tiny constant is lazy machinery.
            Assert.IsTrue(deficit <= 8,
                $"woven polydiv stranded {deficit} buffers over {N} cycles (acq {Acquisitions}, rel {Releases})");
            Assert.IsFalse(u.IsDisposed);
            u.Dispose(); v.Dispose();
        }

        // ------------------------------------------------------- 2) struct carrier: np.unique_counts/all

        [TestMethod]
        public void StructCarrier_UniqueCounts_BothMembersSurvive_ValueCorrect()
        {
            var x = np.array(new int[] { 3, 1, 1, 3, 3, 2 });
            var res = np.unique_counts(x);   // sorted [1,2,3], counts [2,1,3]

            Assert.IsFalse(res.values.IsDisposed, "the carrier's values must survive (yielded via YieldTo)");
            Assert.IsFalse(res.counts.IsDisposed, "the carrier's counts must survive (yielded via YieldTo)");
            Assert.AreEqual(1, res.values.GetInt32(0));
            Assert.AreEqual(3, res.values.GetInt32(2));
            Assert.AreEqual(2L, res.counts.GetInt64(0));
            Assert.AreEqual(3L, res.counts.GetInt64(2));
            Assert.IsFalse(x.IsDisposed, "the input is untouched");

            res.values.Dispose(); res.counts.Dispose(); x.Dispose();
        }

        [TestMethod]
        public void StructCarrier_UniqueAll_AllFourMembers_DroppedUnderEnclosingScope_Reparented()
        {
            var x = np.array(new int[] { 5, 5, 7, 6, 7 });

            np.UniqueAllResult res;
            using (var outer = NDScope.Open())
            {
                res = np.unique_all(x);   // woven; YieldTo re-parents all four members into `outer`
                Assert.IsFalse(res.values.IsDisposed, "a dropped carrier's members stay alive inside the enclosing scope");
                Assert.IsFalse(res.indices.IsDisposed);
                Assert.IsFalse(res.inverse_indices.IsDisposed);
                Assert.IsFalse(res.counts.IsDisposed);
            }

            Assert.IsTrue(res.values.IsDisposed, "the enclosing scope must reclaim every dropped member on close");
            Assert.IsTrue(res.indices.IsDisposed);
            Assert.IsTrue(res.inverse_indices.IsDisposed);
            Assert.IsTrue(res.counts.IsDisposed);
            Assert.IsFalse(x.IsDisposed, "the input is untouched");
            x.Dispose();
        }

        [TestMethod]
        public void StructCarrier_UniqueInverse_ReshapedInverseView_SurvivesBaseReclamation()
        {
            // The woven method yields the RESHAPED inverse view; the un-yielded flat base is reclaimed by
            // the scope, its buffer kept alive by the view's ARC reference. The view must remain valid.
            var x = np.array(new int[] { 9, 8, 9 });
            var res = np.unique_inverse(x);   // inverse over [9,8,9] -> [1,0,1]

            Assert.IsFalse(res.values.IsDisposed, "values must survive");
            Assert.IsFalse(res.inverse_indices.IsDisposed, "the reshaped inverse view must survive (ARC keeps its buffer)");
            Assert.AreEqual(1L, res.inverse_indices.GetInt64(0));
            Assert.AreEqual(0L, res.inverse_indices.GetInt64(1));
            Assert.AreEqual(1L, res.inverse_indices.GetInt64(2));

            res.values.Dispose(); res.inverse_indices.Dispose(); x.Dispose();
        }

        [TestMethod]
        public void StructCarrier_UniqueCounts_ZeroStrands()
        {
            var x = np.array(new int[] { 4, 4, 1, 2, 2, 2, 3, 1, 4, 0 });   // reused across cycles
            void Cycle()
            {
                var res = np.unique_counts(x);
                res.values.Dispose();
                res.counts.Dispose();
            }

            for (int i = 0; i < 3; i++) Cycle();   // warm hash/sort kernels out of the window
            DrainGC();

            SizeBucketedBufferPool.ResetCounters();
            const int N = 200;
            for (int i = 0; i < N; i++) Cycle();
            long deficit = Acquisitions - Releases;

            // The hash/sort internals allocate several scratch buffers per call; a leak would read ~N·k.
            Assert.IsTrue(deficit <= 8,
                $"woven unique_counts stranded {deficit} buffers over {N} cycles (acq {Acquisitions}, rel {Releases})");
            Assert.IsFalse(x.IsDisposed);
            x.Dispose();
        }

        // ------------------------------------------- 3) the general ITuple egress (up to 8, mixed)
        //
        // A ValueTuple of 2..4 NDArrays takes the strongly-typed Returns overload; ANY other tuple —
        // arity 5..8, a non-NDArray component, or a reference-type Tuple — takes Returns(ITuple). No
        // in-tree [NDScoped] method returns such a tuple today, so (exactly like the weaver's
        // out-NDArray[] egress branch) its runtime semantics are pinned here on the underlying NDScope
        // method, called in the EXACT box+callvirt shape the weaver emits: `scope.Returns((ITuple)t)`.

        [TestMethod]
        public void ITupleEgress_HighArityMixed_EveryNDArrayYielded_NonArraySkipped()
        {
            NDArray a, b, c, dropped;
            using (var scope = NDScope.Open())
            {
                a = np.arange(3); b = np.arange(3); c = np.arange(3);   // tracked by scope
                dropped = np.arange(3);                                 // tracked, NOT yielded (control)

                // A 5-element mixed tuple (NDArray, int, NDArray, string, NDArray) — the shape the weaver
                // boxes and hands to Returns(ITuple). The non-NDArray components must be skipped, not crash.
                ITuple t = (a, 42, b, "x", c);
                ITuple ret = scope.Returns(t);
                Assert.AreSame(t, ret, "Returns(ITuple) returns the same tuple");
            }

            Assert.IsFalse(a.IsDisposed, "NDArray component 0 was yielded → survives the scope");
            Assert.IsFalse(b.IsDisposed, "NDArray component 2 was yielded → survives");
            Assert.IsFalse(c.IsDisposed, "NDArray component 4 was yielded → survives");
            Assert.IsTrue(dropped.IsDisposed, "an un-yielded array in the same scope is still reclaimed (the scope worked)");

            a.Dispose(); b.Dispose(); c.Dispose();
        }

        [TestMethod]
        public void ITupleEgress_DroppedUnderEnclosingScope_Reparented()
        {
            NDArray a, b;
            using (var outer = NDScope.Open())
            {
                using (var inner = NDScope.Open())
                {
                    a = np.arange(4); b = np.arange(4);   // tracked by inner
                    inner.Returns((ITuple)(a, 7L, b));    // re-parents the two NDArrays up into `outer`
                }
                Assert.IsFalse(a.IsDisposed, "re-parented into the enclosing scope, survives inner's close");
                Assert.IsFalse(b.IsDisposed);
            }

            Assert.IsTrue(a.IsDisposed, "the enclosing scope reclaims the re-parented arrays on close");
            Assert.IsTrue(b.IsDisposed);
        }

        [TestMethod]
        public void ITupleEgress_ReferenceTuple_AndNull_AreHandled()
        {
            // A reference-type System.Tuple also implements ITuple (the weaver passes it without boxing).
            NDArray a, b;
            using (var scope = NDScope.Open())
            {
                a = np.arange(2); b = np.arange(2);
                ITuple ret = scope.Returns((ITuple)Tuple.Create(a, b));
                Assert.IsNotNull(ret);

                Assert.AreSame(null, scope.Returns((ITuple)null), "a null tuple is a safe no-op");
            }
            Assert.IsFalse(a.IsDisposed, "reference-Tuple components are yielded too");
            Assert.IsFalse(b.IsDisposed);
            a.Dispose(); b.Dispose();
        }

        // Returns(ITuple) dispatches each component by its RUNTIME type through the same family the
        // weaver emits for direct returns — so a nested tuple, an NDArray[] component, a carrier
        // struct component and a bare-buffer component are all seen through, not skipped. (Before
        // the recursive dispatch, ((a, b), c) handed the caller a and b DISPOSED — the exact
        // silent-corruption class the leak analyzer's carrier model says a scope covers.)

        [TestMethod]
        public void ITupleEgress_NestedTuple_ComponentsYieldedRecursively()
        {
            NDArray a, b, c, dropped;
            using (var scope = NDScope.Open())
            {
                a = np.arange(3); b = np.arange(4); c = np.arange(5);
                dropped = np.arange(6);                       // control: the scope still reclaims
                scope.Returns((ITuple)((a, b), c));           // the woven box+callvirt shape
            }

            Assert.IsFalse(a.IsDisposed, "a nested tuple's inner component must be yielded");
            Assert.IsFalse(b.IsDisposed, "a nested tuple's inner component must be yielded");
            Assert.IsFalse(c.IsDisposed, "the outer bare component must be yielded");
            Assert.IsTrue(dropped.IsDisposed, "an un-yielded array in the same scope is still reclaimed");
            Assert.AreEqual(3L, b.GetInt64(3), "the surviving inner buffer is readable");
            a.Dispose(); b.Dispose(); c.Dispose();

            // A reference Tuple nesting a ValueTuple dispatches identically.
            NDArray x, y;
            using (var scope = NDScope.Open())
            {
                x = np.arange(2); y = np.arange(3);
                scope.Returns((ITuple)Tuple.Create((object)(x, y), (object)7));
            }

            Assert.IsFalse(x.IsDisposed, "a ValueTuple nested inside a reference Tuple is seen through");
            Assert.IsFalse(y.IsDisposed);
            x.Dispose(); y.Dispose();
        }

        [TestMethod]
        public void ITupleEgress_NDArrayArrayComponent_ElementsYielded()
        {
            NDArray a, b, c;
            using (var scope = NDScope.Open())
            {
                a = np.arange(3); b = np.arange(4); c = np.arange(5);
                scope.Returns((ITuple)(new[] { a, b }, c));
            }

            Assert.IsFalse(a.IsDisposed, "an NDArray[] component's elements must be yielded");
            Assert.IsFalse(b.IsDisposed);
            Assert.IsFalse(c.IsDisposed);
            a.Dispose(); b.Dispose(); c.Dispose();
        }

        [TestMethod]
        public void ITupleEgress_CarrierComponent_MembersYielded()
        {
            NDArray a, c;
            using (var scope = NDScope.Open())
            {
                a = np.arange(3); c = np.arange(5);
                scope.Returns((ITuple)(new TupleBox { V = a }, c));   // INDArrayCarrier rides inside the tuple
            }

            Assert.IsFalse(a.IsDisposed, "a carrier-struct component yields through its own YieldTo");
            Assert.IsFalse(c.IsDisposed);
            a.Dispose(); c.Dispose();
        }

        [TestMethod]
        public void ITupleEgress_SliceComponent_CountedRefSurvivesScope()
        {
            IArraySlice slice;
            NDArray c;
            using (var scope = NDScope.Open())
            {
                var nd = np.arange(4);                                 // int64; tracked; NOT yielded
                slice = nd.Storage.InternalArray;
                c = np.arange(2);
                scope.Returns((ITuple)(slice, c));                     // bare-buffer component takes a counted ref
            }

            Assert.IsFalse(slice.IsReleased, "a bare-buffer component must be counted-ref protected");
            Assert.AreEqual(3L, slice.GetIndex<long>(3), "the surviving buffer is readable");
            Assert.IsFalse(c.IsDisposed);
            c.Dispose();
        }

        private struct TupleBox : INDArrayCarrier
        {
            public NDArray V;
            void INDArrayCarrier.YieldTo(NDScope scope) => scope.Returns(V);
        }

        // ------------------------------------- 4) bare lower-layer buffer egress (IArraySlice/UnmanagedStorage)
        //
        // The scope reclaims NDArrays via the DETERMINISTIC Release path (eager free at refcount 0). A bare
        // IArraySlice / UnmanagedStorage returned from a scoped method is an UNCOUNTED alias — the scope's
        // Release of the intermediate NDArray that shares its buffer would free it out from under the caller.
        // Returns(slice)/Returns(storage) take a counted reference (TryAddRef) so the buffer survives. As with
        // the ITuple path there is no woven consumer today, so these pin the underlying NDScope method the
        // weaver targets. NOTE: this depends on the ARC contract (see the ndarray-finalizer memory) — the
        // deterministic Release path is stable, but the finalizer path is under active rework in a worktree.

        [TestMethod]
        public void StorageEgress_ReturnedSlice_CountedRefSurvivesScopeReclamation()
        {
            IArraySlice slice;
            using (var scope = NDScope.Open())
            {
                var nd = np.arange(4);                                 // int64 [0,1,2,3]; tracked; refcount 1
                slice = scope.Returns(nd.Storage.InternalArray);       // TryAddRef → 2 (nd NOT yielded)
            }                                                          // scope Releases nd → 1; buffer survives

            Assert.IsFalse(slice.IsReleased, "the yielded slice's buffer survives the scope's deterministic Release of its NDArray");
            Assert.AreEqual(0L, slice.GetIndex<long>(0), "the surviving buffer is still readable");
            Assert.AreEqual(3L, slice.GetIndex<long>(3));
        }

        [TestMethod]
        public void StorageEgress_WithoutProtection_ScopeReleaseFreesTheBareAlias()
        {
            // The control that gives the protection above its meaning: an un-yielded bare alias IS freed by
            // the scope's deterministic Release (this is the Release contract — "no alias outlives").
            IArraySlice slice;
            using (var scope = NDScope.Open())
            {
                var nd = np.arange(4);
                slice = nd.Storage.InternalArray;   // bare uncounted alias — NOT protected
            }                                       // scope Releases nd → refcount 0 → eager free

            Assert.IsTrue(slice.IsReleased, "without Returns' counted-ref protection the scope frees the buffer under the bare alias");
        }

        [TestMethod]
        public void StorageEgress_ReturnedUnmanagedStorage_CountedRefSurvives()
        {
            UnmanagedStorage storage;
            using (var scope = NDScope.Open())
            {
                var nd = np.arange(5);                 // int64 [0,1,2,3,4]
                storage = scope.Returns(nd.Storage);   // TryAddRef's the storage's InternalArray
            }

            Assert.IsFalse(storage.InternalArray.IsReleased, "the yielded storage's buffer survives the scope");
            Assert.AreEqual(4L, storage.InternalArray.GetIndex<long>(4), "the surviving buffer is still readable");
        }
    }
}
