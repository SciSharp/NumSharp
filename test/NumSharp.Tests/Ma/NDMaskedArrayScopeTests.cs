using System;
using System.Linq;

namespace NumSharp.Tests.Ma
{
    /// <summary>
    ///     Pins the np.ma OWNERSHIP rules the leak-audit work introduced (<c>docs/plans/leak-audit-completion.md</c>):
    ///     every np.ma method scopes its own temporaries BY HAND (the weaver cannot yield a masked array), so the
    ///     arrays that must outlive a scope — module singletons, a mask stored into a FIELD, an argument a
    ///     callback returned verbatim — are exactly the ones a wrong scope rule would dispose. Each test builds
    ///     the survivor INSIDE an open <see cref="NDScope"/> (or runs the mutation inside one) and then reads it
    ///     after the scope closed: a regression is a use-after-dispose, not a leak. Every value is NumPy 2.4.2's.
    /// </summary>
    [TestClass]
    public class NDMaskedArrayScopeTests
    {
        /// <summary>Full boolean mask as a managed array (never the nomask sentinel).</summary>
        /// <param name="r">The masked array.</param>
        /// <returns>The mask, all-False when <paramref name="r"/> is nomask.</returns>
        private static bool[] M(NDMaskedArray r) => np.ma.getmaskarray(r).ToArray<bool>();

        /// <summary>The data buffer as float64, in logical C-order.</summary>
        /// <param name="r">The masked array.</param>
        /// <returns>The data values (masked slots included).</returns>
        private static double[] D(NDMaskedArray r) => r.data.ToArray<double>();

        /// <summary>
        ///     <c>MaskedArrayModule.Own</c> is the field-store egress: it removes the array from EVERY open scope,
        ///     so neither the scope that created it nor an enclosing one can dispose it — the array then lives as
        ///     long as whatever holds it. (A <c>Returns</c> would re-track it into the enclosing scope, which would
        ///     then release a mask the target still reads — the poly1d lesson.)
        /// </summary>
        [TestMethod]
        public void Own_DetachesAFieldStoreFromEveryOpenScope()
        {
            NDArray owned;
            using (NDScope.Open())
            {
                using (NDScope.Open())
                {
                    owned = MaskedArrayModule.Own(np.array(new[] { true, false, true }));
                }
            }

            Assert.IsFalse(owned.IsDisposed);
            Assert.IsTrue(owned.ToArray<bool>().SequenceEqual(new[] { true, false, true }));
            owned.Dispose();
        }

        /// <summary>
        ///     The module's process-wide singletons (<c>nomask</c>, the <c>masked</c> constant's data and mask) are
        ///     created by the module's field initializers — which run wherever the FIRST touch of <c>np</c> happens,
        ///     possibly inside a caller's open scope. Built like that, they must survive the scope: before the fix
        ///     the scope disposed them, and every later <c>np.ma.nomask</c>/<c>np.ma.masked</c> read was a
        ///     use-after-dispose for the rest of the process. A fresh module built inside a scope reproduces the
        ///     first-touch conditions without needing a fresh process.
        /// </summary>
        [TestMethod]
        public void ModuleSingletons_BuiltInsideAScope_SurviveIt()
        {
            MaskedArrayModule module;
            using (NDScope.Open())
            {
                module = new MaskedArrayModule();
            }

            Assert.IsFalse(module.nomask.IsDisposed);
            Assert.IsFalse(module.NDMasked.data.IsDisposed);
            Assert.IsFalse(module.NDMasked.mask.IsDisposed);
            Assert.IsFalse(module.nomask.GetAtIndex<bool>(0));
            Assert.IsTrue(module.NDMasked.mask.GetAtIndex<bool>(0));
        }

        /// <summary>
        ///     A hard-mask <c>put</c> ORs the restored mask INTO the live mask buffer (NumPy writes <c>self._mask</c>
        ///     in place too): the masked array keeps the very same mask object, so no replacement array is created
        ///     and the old one is never orphaned. NumPy 2.4.2:
        ///     <c>x = ma.array([1,2,3,4], mask=[0,1,0,0], hard_mask=True); ma.put(x, [0,1], [9,9])</c> →
        ///     data <c>[9,2,3,4]</c>, mask <c>[F,T,F,F]</c>, <c>x._mask is m0</c>.
        /// </summary>
        [TestMethod]
        public void Put_HardMask_KeepsTheLiveMaskObject()
        {
            var x = np.ma.array(np.array(new double[] { 1, 2, 3, 4 }), np.array(new[] { false, true, false, false }), hard_mask: true);
            var before = x._mask;

            np.ma.put(x, np.array(new long[] { 0, 1 }), np.array(new double[] { 9, 9 }));

            Assert.IsTrue(ReferenceEquals(before, x._mask));
            Assert.IsTrue(D(x).SequenceEqual(new double[] { 9, 2, 3, 4 }));
            Assert.IsTrue(M(x).SequenceEqual(new[] { false, true, false, false }));
        }

        /// <summary>
        ///     A hard-mask <c>putmask</c> with a MASKED value ORs the value's mask in place (NumPy's
        ///     <c>a.mask |= m</c>), keeping the mask object, while the data is written everywhere the condition
        ///     holds. NumPy 2.4.2: <c>x = ma.array([1,2,3,4], mask=[0,1,0,0], hard_mask=True);
        ///     ma.putmask(x, [T,T,T,F], ma.array([7,7,7,7], mask=[0,0,1,0]))</c> → data <c>[7,7,7,4]</c>, mask
        ///     <c>[F,T,T,F]</c>, <c>x._mask is m0</c>.
        /// </summary>
        [TestMethod]
        public void PutMask_HardMask_OrsTheValueMaskInPlace()
        {
            var x = np.ma.array(np.array(new double[] { 1, 2, 3, 4 }), np.array(new[] { false, true, false, false }), hard_mask: true);
            var before = x._mask;
            var v = np.ma.array(np.array(new double[] { 7, 7, 7, 7 }), np.array(new[] { false, false, true, false }));

            np.ma.putmask(x, np.array(new[] { true, true, true, false }), v);

            Assert.IsTrue(ReferenceEquals(before, x._mask));
            Assert.IsTrue(D(x).SequenceEqual(new double[] { 7, 7, 7, 4 }));
            Assert.IsTrue(M(x).SequenceEqual(new[] { false, true, true, false }));
        }

        /// <summary>
        ///     <c>putmask</c> on a <c>nomask</c> target with a masked value CREATES the target's mask — a field
        ///     store. Run inside a caller's scope, that mask must outlive the scope (it is detached, not tracked).
        ///     NumPy 2.4.2: <c>x = ma.array([1,2,3]); ma.putmask(x, [T,T,F], ma.array([5,6,7], mask=[0,1,0]))</c>
        ///     → data <c>[5,6,3]</c>, mask <c>[F,T,F]</c>.
        /// </summary>
        [TestMethod]
        public void PutMask_NomaskTarget_CreatedMaskOutlivesTheCallersScope()
        {
            var x = np.ma.array(np.array(new double[] { 1, 2, 3 }));
            var v = np.ma.array(np.array(new double[] { 5, 6, 7 }), np.array(new[] { false, true, false }));

            using (NDScope.Open())
            {
                np.ma.putmask(x, np.array(new[] { true, true, false }), v);
            }

            Assert.IsNotNull(x._mask);
            Assert.IsFalse(x._mask.IsDisposed);
            Assert.IsTrue(D(x).SequenceEqual(new double[] { 5, 6, 3 }));
            Assert.IsTrue(M(x).SequenceEqual(new[] { false, true, false }));
        }

        /// <summary>
        ///     A hard-mask indexer assignment of a MASKED value writes data only where neither the existing mask
        ///     nor the value's mask is set — a slot the value masks KEEPS its old data — and the mask is OR'd in
        ///     place. NumPy 2.4.2: <c>x = ma.array([1,2,3,4], mask=[0,1,0,0], hard_mask=True);
        ///     x[0:3] = ma.array([7,7,7], mask=[0,0,1])</c> → data <c>[7,2,3,4]</c>, mask <c>[F,T,T,F]</c>,
        ///     <c>x._mask is m0</c>. (The restore used to run under the OLD mask only, leaving data 7 in slot 2.)
        /// </summary>
        [TestMethod]
        public void SetItem_HardMask_MaskedValue_KeepsDataUnderEveryMaskedSlot()
        {
            var x = np.ma.array(np.array(new double[] { 1, 2, 3, 4 }), np.array(new[] { false, true, false, false }), hard_mask: true);
            var before = x._mask;

            x["0:3"] = np.ma.array(np.array(new double[] { 7, 7, 7 }), np.array(new[] { false, false, true }));

            Assert.IsTrue(ReferenceEquals(before, x._mask));
            Assert.IsTrue(D(x).SequenceEqual(new double[] { 7, 2, 3, 4 }));
            Assert.IsTrue(M(x).SequenceEqual(new[] { false, true, true, false }));
        }

        /// <summary>
        ///     A slice of a hard-masked array is hard (it inherits the flag), so writes through it run the hard
        ///     branch of <c>SetItem</c> INSIDE the caller's scope. Ownership must hold on that path: the parent was
        ///     built before the scope, so its data and mask survive it and the mask keeps its identity; the slice's
        ///     data and mask were handed back to the caller's scope by the indexer, so they are reclaimed with it
        ///     rather than stranded. NumPy 2.4.2: <c>x = ma.array([1.,2,3,4], mask=[0,1,0,0], hard_mask=True);
        ///     v = x[0:3]; v[1] = 99; v[0] = 11; v[...] = 50; v[2] = ma.masked</c> → data <c>[50,2,50,4]</c>, mask
        ///     <c>[F,T,T,F]</c>, <c>x._mask is m0</c>.
        /// </summary>
        [TestMethod]
        public void HardSlice_WritesInsideACallersScope_ParentSurvives_SliceReclaimedWithTheScope()
        {
            var x = np.ma.array(np.array(new double[] { 1, 2, 3, 4 }), np.array(new[] { false, true, false, false }), hard_mask: true);
            var data0 = x._data;
            var mask0 = x._mask;

            NDMaskedArray v;
            using (NDScope.Open())
            {
                v = (NDMaskedArray)x["0:3"];
                Assert.IsTrue(v.hardmask);
                v[1] = 99.0;
                v[0] = 11.0;
                v["..."] = 50.0;
                v[2] = np.ma.masked;
            }

            Assert.IsFalse(data0.IsDisposed);
            Assert.IsFalse(mask0.IsDisposed);
            Assert.IsTrue(ReferenceEquals(mask0, x._mask));
            Assert.IsTrue(D(x).SequenceEqual(new double[] { 50, 2, 50, 4 }));
            Assert.IsTrue(M(x).SequenceEqual(new[] { false, true, true, false }));
            Assert.IsTrue(v._data.IsDisposed);
            Assert.IsTrue(v._mask.IsDisposed);
        }

        /// <summary>
        ///     <c>SetItem</c> carries <c>[NDScoped]</c>, so its own scope releases the hard-mask snapshots (a data
        ///     copy and a mask copy per write) before it returns, even when the caller has NO scope open. Without that
        ///     scope each hard write would leave two pooled buffers to the finalizer: a hard write through a slice
        ///     copies the whole slice, so a loop of writes would hold that memory until a collection. Measured in an
        ///     exact pool window (re-run when a collection lands in it); the slice is taken outside the window, so
        ///     only the writes are counted.
        /// </summary>
        [TestMethod]
        [DoNotParallelize] // reads the process-wide pool counters
        public void HardSlice_SetItem_ReleasesItsSnapshots_WithNoEnclosingScope()
        {
            const int n = 4096; // large enough that both snapshots come from the bucketed pool
            var x = np.ma.array(np.arange(n).astype(np.float64), np.equal(np.mod(np.arange(n), 7), 1), hard_mask: true);
            var v = (NDMaskedArray)x["0:4000"];
            v[1] = 99.0; // warm-up: kernels and the pool bucket exist before the window

            long acquired = 0, released = 0;
            bool exact = false;
            for (int attempt = 0; attempt < 5 && !exact; attempt++)
            {
                // Hygiene: finalize the previous attempt's garbage and start with a fresh gen-0 budget.
                GcQuiescence.CollectYoung();
                var window = GcQuiescence.OpenWindow();
                long a0 = Acquisitions, r0 = Releases;
                for (int i = 0; i < 4; i++)
                {
                    v[1] = 99.0;  // masked slot: ignored
                    v[0] = 11.0;  // unmasked slot: written
                }
                acquired = Acquisitions - a0;
                released = Releases - r0;
                // Exact only when no collection (and so no finalizer) ran inside the window.
                exact = GcQuiescence.Undisturbed(window);
            }

            Assert.IsTrue(acquired > 0, "the hard writes must take snapshots from the pool, or the window measures nothing");
            if (exact)
                Assert.AreEqual(acquired, released, "every snapshot buffer is released by SetItem's own scope");
            else
                // A collection landed in every attempt: a finalizer can only ADD releases, so this reading errs
                // toward innocence and still fails when the scope released nothing.
                Assert.IsTrue(released >= acquired, "every snapshot buffer is released by SetItem's own scope");
            Assert.AreEqual(1.0, x._data.GetAtIndex<double>(1)); // the masked slot kept its data
            Assert.AreEqual(11.0, x._data.GetAtIndex<double>(0));
        }

        /// <summary>Pooled buffers handed out so far (process-wide).</summary>
        private static long Acquisitions => NumSharp.Backends.Unmanaged.Pooling.SizeBucketedBufferPool.Hits
                                            + NumSharp.Backends.Unmanaged.Pooling.SizeBucketedBufferPool.Misses
                                            + NumSharp.Backends.Unmanaged.Pooling.SizeBucketedBufferPool.ZeroedAllocs;

        /// <summary>Pooled buffers given back so far (process-wide).</summary>
        private static long Releases => NumSharp.Backends.Unmanaged.Pooling.SizeBucketedBufferPool.Returns
                                        + NumSharp.Backends.Unmanaged.Pooling.SizeBucketedBufferPool.ReturnsFreed;

        /// <summary>
        ///     <c>fromfunction</c> releases the index grids it handed to the callback — except one the callback
        ///     returned VERBATIM, which is the result's data. NumPy 2.4.2:
        ///     <c>ma.fromfunction(lambda i, j: i, (2, 3))</c> → data <c>[[0,0,0],[1,1,1]]</c> (float64), nomask.
        /// </summary>
        [TestMethod]
        public void FromFunction_CallbackReturningItsArgument_KeepsTheResult()
        {
            // Built after this epoch, so CollectSince decides these objects' fate as a full collection would, with
            // a young one whenever no collection ran in between (GcQuiescence.CollectSince).
            var since = GcQuiescence.Epoch.Capture();
            var r = np.ma.fromfunction((i, j) => i, new Shape(2, 3));

            // A collection proves the result does not merely survive by luck of timing.
            GcQuiescence.CollectSince(since);

            Assert.IsFalse(r.data.IsDisposed);
            Assert.IsTrue(D(r).SequenceEqual(new double[] { 0, 0, 0, 1, 1, 1 }));
            Assert.IsTrue(ReferenceEquals(np.ma.getmask(r), np.ma.nomask));
        }

        /// <summary>
        ///     A reduction called inside a caller's scope hands its result's parts BACK to that scope (the
        ///     masked-array Yield): the result is readable for the scope's whole lifetime and reclaimed WITH it.
        ///     Both failure modes are caught — parts disposed by the method's own scope (the read inside fails)
        ///     and parts escaping every scope (still alive after the caller's scope closed). NumPy 2.4.2:
        ///     <c>ma.array([[1,2],[3,4]], mask=[[0,1],[0,0]]).sum(0)</c> → <c>[4, 4]</c>.
        /// </summary>
        [TestMethod]
        public void Reduction_InsideACallersScope_ResultLivesExactlyAsLongAsThatScope()
        {
            NDMaskedArray s;
            using (NDScope.Open())
            {
                var x = np.ma.array(np.array(new double[] { 1, 2, 3, 4 }).reshape(2, 2), np.array(new[] { false, true, false, false }).reshape(2, 2));
                s = np.ma.sum(x, 0);
                // Column 0: 1 + 3 = 4; column 1: only the 4 is unmasked.
                Assert.IsFalse(s.data.IsDisposed);
                Assert.IsTrue(D(s).SequenceEqual(new double[] { 4, 4 }));
            }

            Assert.IsTrue(s.data.IsDisposed);
        }
    }
}
