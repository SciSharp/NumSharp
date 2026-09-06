using System;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests.Lifetime
{
    /// <summary>
    ///     Proves the woven <c>[NDScoped]</c> np.* methods that touch <see cref="Backends.UnmanagedStorage"/>
    ///     internally — the ones that explicitly <c>Dispose()</c> a transient (<c>np.diff</c>, <c>np.ediff1d</c>,
    ///     the <c>materialized?.Dispose()</c> ARC-release in <c>nonzero</c>) or alias a buffer into a returned
    ///     view (<c>nonzero</c>) — never double-release a buffer.
    /// <para>
    ///     The concern: the weaver injects a scope that reclaims every tracked <see cref="NDArray"/> at method
    ///     exit. A method that ALSO disposes one of those transients itself would, if unguarded, release the
    ///     same buffer reference twice — a double-free / premature-free under a live view.
    /// </para>
    /// <para>
    ///     Why it cannot happen (structural): (1) <see cref="NDArray.Dispose"/> is idempotent — an
    ///     <see cref="System.Threading.Interlocked"/> <c>_disposed</c> guard makes each NDArray release its
    ///     one buffer reference AT MOST ONCE, so an explicit dispose followed by the scope's dispose is a
    ///     single release; and (2) no <c>[NDScoped]</c> method releases a buffer DIRECTLY (never
    ///     <c>InternalArray.Release()</c> / <c>DangerousFree()</c> — audited), so the only releases come
    ///     through the idempotent <see cref="NDArray.Dispose"/>. These tests pin that behaviour at runtime.
    /// </para>
    /// </summary>
    [TestClass]
    public class NDScopeDoubleDisposeTests
    {
        // --------------------------------------------------------- the mechanism, made observable

        /// <summary>
        ///     Replicates the exact woven shape — a scope tracks a transient, the method disposes it itself
        ///     (the <c>np.diff</c>/<c>nonzero</c> pattern), and a VIEW that shares the transient's buffer is
        ///     yielded. If the scope's end-of-method dispose double-released the base (bypassing idempotency),
        ///     the buffer would be freed under the live view — the readback would be a use-after-free.
        /// </summary>
        [TestMethod]
        public void ExplicitTransientDispose_PlusScopeDispose_ReleasesBufferExactlyOnce()
        {
            NDArray view;
            IArraySlice buffer;
            using (var scope = NDScope.Open())
            {
                var baseArr = np.arange(50);        // int64 [0..49]; tracked; buffer refcount 1
                view = baseArr["10:15"];            // tracked view; SHARES baseArr's buffer → refcount 2
                buffer = view.Storage.InternalArray;

                baseArr.Dispose();                  // EXPLICIT dispose (the np.diff / nonzero pattern) → refcount 1
                Assert.IsTrue(baseArr.IsDisposed, "explicit dispose marks the base disposed");
                Assert.IsFalse(buffer.IsReleased, "the view keeps the shared buffer alive after the base is disposed");

                scope.Returns(view);                // yield the view — it must survive the scope
            }                                       // scope disposes baseArr AGAIN — must be an idempotent no-op

            Assert.IsFalse(buffer.IsReleased,
                "the buffer survived scope exit: the scope did NOT double-release the already-disposed base under the live view");
            Assert.AreEqual(10L, view.GetInt64(0), "the yielded view still reads correct data (no use-after-free)");
            Assert.AreEqual(14L, view.GetInt64(4));

            view.Dispose();                         // the LAST owner
            Assert.IsTrue(buffer.IsReleased, "the buffer is freed exactly once — when its final reference (the view) drops");
        }

        /// <summary>
        ///     A method that disposes a transient AND yields a view onto the SAME buffer: the view must
        ///     outlive both the explicit dispose and the scope. Mirrors <c>nonzero</c>, whose result columns
        ///     alias the multi-index buffer while an intermediate (<c>materialized</c>) is disposed in a
        ///     <c>finally</c>.
        /// </summary>
        [TestMethod]
        public void YieldedViewOntoDisposedBase_SurvivesBothDisposes()
        {
            NDArray col;
            using (var scope = NDScope.Open())
            {
                var matrix = np.arange(12).reshape(3, 4);   // tracked
                col = scope.Returns(matrix[":, 1"]);        // a strided view sharing matrix's buffer, yielded
                matrix.Dispose();                           // the base disposed while the view is live + yielded
            }
            // matrix was disposed explicitly AND by the scope; the yielded column view must still be intact.
            Assert.AreEqual(1L, col.GetInt64(0));
            Assert.AreEqual(5L, col.GetInt64(1));
            Assert.AreEqual(9L, col.GetInt64(2));
            col.Dispose();
        }

        // ------------------------------------------------ the real methods, under the reproducing conditions

        /// <summary>
        ///     Drives the woven methods that explicitly dispose transients (<c>np.diff</c>, <c>np.ediff1d</c>)
        ///     and alias storage into returned views (<c>np.nonzero</c>, on a NON-contiguous input so it takes
        ///     the <c>materialized?.Dispose()</c> + <c>Storage.Alias</c> path) many times under GC pressure —
        ///     the exact conditions <c>Default.NonZero</c> documents as reproducing a double-free
        ///     <c>AccessViolationException</c> in Release. A double-release introduced by the weave would
        ///     surface here as a crash or as garbage read from a freed-and-reused buffer.
        /// </summary>
        [TestMethod]
        public void WovenStorageMethods_UnderGCPressure_NoDoubleFree()
        {
            var seq = np.arange(20);
            var noncontig = np.arange(24).reshape(4, 6).T;   // transposed → non-contiguous
            var square = np.arange(9).reshape(3, 3);         // for the einsum diagonal-view path
            var cond = np.array(new bool[] { true, false, true, false, true });
            var wx = np.arange(5);
            var wy = np.arange(5) + 100;
            const int N = 12000;

            for (int i = 0; i < N; i++)
            {
                using (var d = np.diff(seq))                 // explicit disposes of hi/lo/current transients
                {
                    Assert.AreEqual(1L, d.GetInt64(0), "np.diff stayed correct");
                    Assert.AreEqual(1L, d.GetInt64(18));
                }
                using (var e = np.ediff1d(np.arange(15)))    // many internal explicit Disposes
                {
                    Assert.AreEqual(14L, e.size, "np.ediff1d stayed correct");
                    Assert.AreEqual(1L, e.GetInt64(0));
                }

                var nz = np.nonzero(noncontig);              // materialized.Dispose() + Storage.Alias views
                long nzCount = nz[0].size;
                foreach (var col in nz)
                    col.Dispose();
                Assert.AreEqual(23L, nzCount, "np.nonzero stayed correct (23 of 24 non-zero)");

                using (var w = np.where(cond, wx, wy))       // NDIterRef.Dispose under the scope
                {
                    Assert.AreEqual(0L, w.GetInt64(0), "np.where stayed correct");
                    Assert.AreEqual(101L, w.GetInt64(1));
                }
                using (var diag = np.einsum("ii->i", square))// a writeable view aliasing the operand's storage
                {
                    Assert.AreEqual(0L, diag.GetInt64(0), "np.einsum diagonal view stayed correct");
                    Assert.AreEqual(4L, diag.GetInt64(1));
                    Assert.AreEqual(8L, diag.GetInt64(2));
                }

                if ((i & 511) == 0)
                {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    GC.Collect();
                }
            }

            seq.Dispose();
            noncontig.Dispose();
            square.Dispose();
            cond.Dispose();
            wx.Dispose();
            wy.Dispose();
        }
    }
}
