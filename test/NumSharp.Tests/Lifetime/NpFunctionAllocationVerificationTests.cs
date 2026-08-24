using System;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Tests.Lifetime
{
    /// <summary>
    ///     Detailed per-function verification of TEN np.* functions on two axes at once:
    ///     <list type="number">
    ///     <item><b>Correctness</b> — outputs match NumPy across representative cases;</item>
    ///     <item><b>Allocation discipline</b> — every buffer the function allocates (its internal
    ///           transients AND its result) is reclaimed. Measured as pool acquisitions − releases
    ///           over 200 cycles with the result disposed each cycle: a clean function nets ≈ 0,
    ///           while a SINGLE undisposed internal per call would read as ~200. Inputs are also
    ///           asserted untouched (never disposed, never adopted into the function's scope).</item>
    ///     </list>
    ///     Five are heavy COMPOSITIONS that drive many other np.* functions internally
    ///     (<c>cross</c>, <c>corrcoef</c>, <c>union1d</c>, <c>kron</c>, <c>vander</c>) — the sites
    ///     where "did every temporary get disposed?" actually bites; five are kernel/manipulation
    ///     ops (<c>clip</c>, <c>sort</c>, <c>take_along_axis</c>, <c>roll</c>, <c>where</c>).
    /// </summary>
    [TestClass]
    [DoNotParallelize]   // the allocation measurement reads the process-wide pool counters
    public class NpFunctionAllocationVerificationTests
    {
        // ------------------------------------------------------------------ measurement harness

        private static void DrainGC()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }

        private static long Acquisitions => SizeBucketedBufferPool.Hits + SizeBucketedBufferPool.Misses + SizeBucketedBufferPool.ZeroedAllocs;
        private static long Releases => SizeBucketedBufferPool.Returns + SizeBucketedBufferPool.ReturnsFreed;

        private const int Cycles = 200;

        /// <summary>
        ///     Runs <paramref name="op"/> <see cref="Cycles"/> times, disposing each result, and
        ///     asserts the pool did not accumulate un-returned buffers — i.e. every internal
        ///     transient the function allocated was reclaimed by its scope, not left to the finalizer.
        ///     A per-call leak nets ~<see cref="Cycles"/>; the tolerance only absorbs lazily-initialised
        ///     machinery. The <paramref name="op"/> must NOT allocate operands itself (hoist them), or
        ///     those undisposed operands would themselves read as strands.
        /// </summary>
        private static void AssertReclaimsEverything(string name, Func<NDArray> op)
        {
            for (int i = 0; i < 3; i++) op().Dispose();       // warm kernels/caches out of the window
            DrainGC();

            SizeBucketedBufferPool.ResetCounters();
            for (int i = 0; i < Cycles; i++) op().Dispose();
            long deficit = Acquisitions - Releases;

            Assert.IsTrue(deficit <= 8,
                $"{name}: {deficit} buffers stranded over {Cycles} cycles (a single undisposed internal per call " +
                $"reads as ~{Cycles}); acq {Acquisitions}, rel {Releases}");
        }

        private static void AssertInputsUntouched(params NDArray[] inputs)
        {
            foreach (var x in inputs)
            {
                Assert.IsFalse(x.IsDisposed, "an input must never be disposed by the function");
                Assert.IsNull(x.TrackingScope, "an input must never be adopted (tracked) into the function's scope");
            }
        }

        // ==================================================================================
        //  FIVE COMPOSITIONS — use many other np.* functions internally
        // ==================================================================================

        [TestMethod]
        public void Composition_Cross()
        {
            // internally: astype / moveaxis / multiply / subtract / negative / concatenate / stack
            var a = np.array(new double[] { 1, 2, 3 });
            var b = np.array(new double[] { 4, 5, 6 });

            using (var r = np.cross(a, b))
            {
                Assert.AreEqual(3, r.size);
                Assert.AreEqual(-3.0, r.GetDouble(0));   // 2*6 - 3*5
                Assert.AreEqual(6.0, r.GetDouble(1));    // 3*4 - 1*6
                Assert.AreEqual(-3.0, r.GetDouble(2));   // 1*5 - 2*4
                Assert.IsFalse(r.IsDisposed);
            }
            using (var x = np.array(new double[] { 1, 0, 0 }))
            using (var y = np.array(new double[] { 0, 1, 0 }))
            using (var r = np.cross(x, y))
                Assert.AreEqual(1.0, r.GetDouble(2), "x × y = z");

            AssertInputsUntouched(a, b);
            AssertReclaimsEverything("np.cross", () => np.cross(a, b));
            a.Dispose(); b.Dispose();
        }

        [TestMethod]
        public void Composition_Corrcoef()
        {
            // internally: cov (dot/mean/subtract/...) / sqrt / divide / outer / clip
            var anti = np.array(new double[,] { { 1, 2, 3 }, { 3, 2, 1 } });   // anti-correlated rows

            using (var r = np.corrcoef(anti))
            {
                Assert.AreEqual(1.0, r.GetDouble(0, 0), 1e-12);
                Assert.AreEqual(-1.0, r.GetDouble(0, 1), 1e-12);
                Assert.AreEqual(-1.0, r.GetDouble(1, 0), 1e-12);
                Assert.AreEqual(1.0, r.GetDouble(1, 1), 1e-12);
            }
            using (var pos = np.array(new double[,] { { 1, 2, 3 }, { 1, 2, 3 } }))
            using (var r = np.corrcoef(pos))
                Assert.AreEqual(1.0, r.GetDouble(0, 1), 1e-12, "identical rows correlate to +1");

            AssertInputsUntouched(anti);
            AssertReclaimsEverything("np.corrcoef", () => np.corrcoef(anti));
            anti.Dispose();
        }

        [TestMethod]
        public void Composition_Union1d()
        {
            // internally: ravel / concatenate / unique (sort + mask)
            var a = np.array(new[] { 3, 1, 1 });
            var b = np.array(new[] { 2, 3, 3 });

            using (var r = np.union1d(a, b))
            {
                Assert.AreEqual(3, r.size);
                Assert.AreEqual(1, r.GetInt32(0));
                Assert.AreEqual(2, r.GetInt32(1));
                Assert.AreEqual(3, r.GetInt32(2));
            }
            using (var e1 = np.array(new int[0]))
            using (var r = np.union1d(e1, b))
                Assert.AreEqual(2, r.size, "union with empty is unique(b)");

            AssertInputsUntouched(a, b);
            AssertReclaimsEverything("np.union1d", () => np.union1d(a, b));
            a.Dispose(); b.Dispose();
        }

        [TestMethod]
        public void Composition_Kron()
        {
            // internally: multiply / tile / reshape / broadcast_to
            var a = np.array(new double[] { 1, 2 });
            var b = np.array(new double[] { 3, 4 });

            using (var r = np.kron(a, b))   // [1*3, 1*4, 2*3, 2*4]
            {
                Assert.AreEqual(4, r.size);
                Assert.AreEqual(3.0, r.GetDouble(0));
                Assert.AreEqual(4.0, r.GetDouble(1));
                Assert.AreEqual(6.0, r.GetDouble(2));
                Assert.AreEqual(8.0, r.GetDouble(3));
            }
            using (var i2 = np.array(new double[,] { { 1, 0 }, { 0, 1 } }))
            using (var r = np.kron(i2, i2))
                Assert.AreEqual(16, r.size, "kron of 2×2 identities is 4×4");

            AssertInputsUntouched(a, b);
            AssertReclaimsEverything("np.kron", () => np.kron(a, b));
            a.Dispose(); b.Dispose();
        }

        [TestMethod]
        public void Composition_Vander()
        {
            // internally: per-column power / concatenate / reshape
            var x = np.array(new double[] { 1, 2, 3 });

            using (var r = np.vander(x))   // decreasing powers, N=3
            {
                Assert.AreEqual(9.0, r.GetDouble(2, 0));   // 3^2
                Assert.AreEqual(3.0, r.GetDouble(2, 1));   // 3^1
                Assert.AreEqual(1.0, r.GetDouble(2, 2));   // 3^0
                Assert.AreEqual(1.0, r.GetDouble(0, 0));   // 1^2
                Assert.AreEqual(4.0, r.GetDouble(1, 0));   // 2^2
            }
            using (var r = np.vander(x, 2))
            {
                Assert.AreEqual(2.0, r.GetDouble(1, 0));   // 2^1
                Assert.AreEqual(1.0, r.GetDouble(1, 1));   // 2^0
            }

            AssertInputsUntouched(x);
            AssertReclaimsEverything("np.vander", () => np.vander(x));
            x.Dispose();
        }

        // ==================================================================================
        //  FIVE KERNEL / MANIPULATION OPS
        // ==================================================================================

        [TestMethod]
        public void Kernel_Clip()
        {
            var a = np.array(new double[] { 0, 2, 9 });
            var lo = (NDArray)1.0;   // hoisted so the closure allocates nothing itself
            var hi = (NDArray)5.0;

            using (var r = np.clip(a, lo, hi))
            {
                Assert.AreEqual(1.0, r.GetDouble(0));   // clamped up
                Assert.AreEqual(2.0, r.GetDouble(1));   // untouched
                Assert.AreEqual(5.0, r.GetDouble(2));   // clamped down
            }

            AssertInputsUntouched(a, lo, hi);
            AssertReclaimsEverything("np.clip", () => np.clip(a, lo, hi));
            a.Dispose(); lo.Dispose(); hi.Dispose();
        }

        [TestMethod]
        public void Kernel_Sort()
        {
            var a = np.array(new double[] { 3, 1, 2 });

            using (var r = np.sort(a))
            {
                Assert.AreEqual(1.0, r.GetDouble(0));
                Assert.AreEqual(2.0, r.GetDouble(1));
                Assert.AreEqual(3.0, r.GetDouble(2));
            }
            Assert.AreEqual(3.0, a.GetDouble(0), "np.sort returns a copy — the input order is preserved");

            AssertInputsUntouched(a);
            AssertReclaimsEverything("np.sort", () => np.sort(a));
            a.Dispose();
        }

        [TestMethod]
        public void Kernel_TakeAlongAxis()
        {
            var a = np.array(new double[,] { { 3, 1, 2 }, { 6, 5, 4 } });
            var idx = np.argsort(a, 1);   // hoisted indices

            using (var r = np.take_along_axis(a, idx, 1))   // == sort(a, axis=1)
            {
                Assert.AreEqual(1.0, r.GetDouble(0, 0));
                Assert.AreEqual(2.0, r.GetDouble(0, 1));
                Assert.AreEqual(3.0, r.GetDouble(0, 2));
                Assert.AreEqual(4.0, r.GetDouble(1, 0));
                Assert.AreEqual(6.0, r.GetDouble(1, 2));
            }

            AssertInputsUntouched(a, idx);
            AssertReclaimsEverything("np.take_along_axis", () => np.take_along_axis(a, idx, 1));
            a.Dispose(); idx.Dispose();
        }

        [TestMethod]
        public void Kernel_Roll()
        {
            var a = np.arange(5);
            var m = np.arange(6).reshape(2, 3);   // axis=null path -> woven roll nests woven roll

            using (var r = np.roll(a, 2))
            {
                Assert.AreEqual(3L, r.GetInt64(0));
                Assert.AreEqual(4L, r.GetInt64(1));
                Assert.AreEqual(0L, r.GetInt64(2));
            }
            using (var r = np.roll(m, 1))   // [[5,0,1],[2,3,4]]
            {
                Assert.AreEqual(5L, r.GetInt64(0, 0));
                Assert.AreEqual(4L, r.GetInt64(1, 2));
            }

            AssertInputsUntouched(a, m);
            AssertReclaimsEverything("np.roll(2d)", () => np.roll(m, 1));   // nested reclamation
            a.Dispose(); m.Dispose();
        }

        [TestMethod]
        public void Kernel_Where()
        {
            var cond = np.array(new[] { true, false, true });
            var x = np.array(new double[] { 1, 2, 3 });
            var y = np.array(new double[] { 9, 8, 7 });

            using (var r = np.where(cond, x, y))   // [1, 8, 3]
            {
                Assert.AreEqual(1.0, r.GetDouble(0));
                Assert.AreEqual(8.0, r.GetDouble(1));
                Assert.AreEqual(3.0, r.GetDouble(2));
            }

            AssertInputsUntouched(cond, x, y);
            AssertReclaimsEverything("np.where", () => np.where(cond, x, y));
            cond.Dispose(); x.Dispose(); y.Dispose();
        }
    }
}
