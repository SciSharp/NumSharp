using System;
using System.Collections.Generic;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Tests.Lifetime
{
    /// <summary>
    ///     Behavioral proof that the <c>[NDScoped]</c> WEAVER's output COMPOSES under nesting.
    ///     <see cref="NDScopeWeaveTests"/> proves every attributed method was woven (structural);
    ///     <see cref="NDScopeTests"/> pins the scope CONTRACT with hand-written scopes. This class
    ///     drives the woven Core methods themselves through three nesting shapes and asserts value
    ///     correctness, zero net buffer strands, and untouched inputs for each:
    ///     <list type="bullet">
    ///     <item>a woven method that internally calls another woven method — real IL nesting:
    ///           <c>np.roll(a)</c> with <c>axis=null</c> recurses into the woven
    ///           <c>roll(ravel, shift, 0)</c>, so TWO woven scopes nest with no hand-written scope
    ///           anywhere;</item>
    ///     <item>an enclosing scope over MANY woven calls whose results are dropped — each woven
    ///           method's scope is a child of the enclosing one and re-parents its result up, which
    ///           the enclosing scope then reclaims on close;</item>
    ///     <item>a DEEP chain of nested functions mixing the weaver's own injected pattern
    ///           (reproduced by hand in <see cref="NestLevel"/>) with a real woven leaf, plus the
    ///           exception path through the whole nest.</item>
    ///     </list>
    /// </summary>
    [TestClass]
    [DoNotParallelize]   // the zero-strand tests read the process-wide pool counters
    public class NDScopeWeaveNestingTests
    {
        private static void DrainGC()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }

        private static long Acquisitions => SizeBucketedBufferPool.Hits + SizeBucketedBufferPool.Misses + SizeBucketedBufferPool.ZeroedAllocs;
        private static long Releases => SizeBucketedBufferPool.Returns + SizeBucketedBufferPool.ReturnsFreed;

        // ---------------------------------------------------------- 1) woven calls woven (real IL)

        [TestMethod]
        public void RollAxisNull_WovenCallsWoven_ValueCorrect_InputUntouched()
        {
            var a = np.arange(6).reshape(2, 3);   // C-contiguous 2-D
            // axis=null path: roll(2-D) -> roll(ravel, shift, 0).reshape(...). Two woven scopes nest,
            // the inner result re-parents into the outer roll's scope, the reshape view is yielded up.
            var r = np.roll(a, 1);                 // numpy: [[5,0,1],[2,3,4]]

            Assert.IsFalse(r.IsDisposed, "the outermost woven result must survive the nested reclamation");
            Assert.AreEqual(5L, r.GetInt64(0, 0));
            Assert.AreEqual(0L, r.GetInt64(0, 1));
            Assert.AreEqual(1L, r.GetInt64(0, 2));
            Assert.AreEqual(4L, r.GetInt64(1, 2));
            Assert.IsFalse(a.IsDisposed, "the input must never be reclaimed by the nested woven scopes");
            Assert.AreEqual(0L, a.GetInt64(0, 0), "input data intact");

            r.Dispose();
            a.Dispose();
        }

        [TestMethod]
        public void RollAxisNull_WovenNesting_ZeroStrands()
        {
            var a = np.arange(64).reshape(8, 8);   // reused across cycles
            void Cycle()
            {
                using (var r = np.roll(a, 3)) { }  // BOTH nested woven scopes must reclaim their temps
            }

            for (int i = 0; i < 3; i++) Cycle();   // warm one-time kernel/cache setup out of the window
            DrainGC();

            SizeBucketedBufferPool.ResetCounters();
            const int N = 200;
            for (int i = 0; i < N; i++) Cycle();
            long deficit = Acquisitions - Releases;

            // If the INNER woven scope leaked its result buffer, the deficit would read ~N; a tiny
            // constant is only lazy per-thread machinery.
            Assert.IsTrue(deficit <= 8,
                $"nested woven roll stranded {deficit} buffers over {N} cycles (acq {Acquisitions}, rel {Releases})");
            Assert.IsFalse(a.IsDisposed);
            a.Dispose();
        }

        // ---------------------------------------------- 2) enclosing scope over many woven functions

        [TestMethod]
        public void EnclosingScope_ManyNestedWovenCalls_AllReparentAndReclaim()
        {
            var v1 = np.array(new double[] { 1, 2, 3 });
            var v2 = np.array(new double[] { 4, 5, 6 });
            var m = np.arange(9).reshape(3, 3).astype(NPTypeCode.Double);
            var p = np.array(new double[] { 1, -3, 2 });   // polynomial coefficients

            var dropped = new List<NDArray>();
            using (var outer = NDScope.Open())
            {
                // Each woven method opens a CHILD scope of `outer`, computes, and re-parents its
                // result up to `outer`. We drop every result (never dispose by hand); the enclosing
                // scope is now solely responsible for reclaiming them.
                dropped.Add(np.cross(v1, v2));         // woven
                dropped.Add(np.diff(v1));              // woven
                dropped.Add(np.roll(m, 2));            // woven -> woven (axis=null recursion) under `outer`
                dropped.Add(np.kron(v1, v2));          // woven
                dropped.Add(np.outer(v1, v2));         // woven
                dropped.Add(np.vander(v1));            // woven
                dropped.Add(np.polyval(p, v1));        // woven
                dropped.Add(np.linalg.norm(m));        // woven

                foreach (var r in dropped)
                    Assert.IsFalse(r.IsDisposed, "a dropped nested-woven result must stay alive inside the enclosing scope");
            }

            foreach (var r in dropped)
                Assert.IsTrue(r.IsDisposed, "the enclosing scope must reclaim every dropped nested-woven result on close");
            Assert.IsFalse(v1.IsDisposed, "inputs are constructed before the scope and must be untouched");
            Assert.IsFalse(v2.IsDisposed);
            Assert.IsFalse(m.IsDisposed);
            Assert.IsFalse(p.IsDisposed);

            v1.Dispose(); v2.Dispose(); m.Dispose(); p.Dispose();
        }

        [TestMethod]
        public void WovenResultsFlowBetweenWovenFunctions_UnderScope_ValueCorrect()
        {
            var a = np.array(new double[] { 1, 2, 4, 7 });
            using (var outer = NDScope.Open())
            {
                var d = np.diff(a);        // woven          -> [1, 2, 3]
                var r = np.roll(d, 1);     // woven, CONSUMES the woven result d -> [3, 1, 2]

                Assert.IsFalse(r.IsDisposed);
                Assert.AreEqual(3.0, r.GetDouble(0));
                Assert.AreEqual(1.0, r.GetDouble(1));
                Assert.AreEqual(2.0, r.GetDouble(2));
                Assert.IsFalse(d.IsDisposed, "a woven result feeding another woven call stays alive under the scope");
            }
            Assert.IsFalse(a.IsDisposed, "the input survives the enclosing scope");
            a.Dispose();
        }

        // -------------------------------------------------- 3) deep nested functions + woven leaf

        /// <summary>
        ///     A hand-written stand-in for a woven method: it reproduces the EXACT pattern the weaver
        ///     injects (open a scope, allocate an owned temp, yield the egress through
        ///     <see cref="NDScope.Returns{T}(T)"/>) and either recurses — another nested function — or
        ///     calls a REAL woven method at the leaf. A deep chain of these therefore exercises many
        ///     nested scopes AND many nested functions with genuine woven Core code at the bottom.
        /// </summary>
        private static NDArray NestLevel(NDArray x, int depth)
        {
            using var s = NDScope.Open();
            var t = x + 1;                            // an owned temp at this level (must be reclaimed)
            if (depth == 0)
                return s.Returns(np.roll(t, 1));      // real woven leaf (which itself nests roll -> roll)
            return s.Returns(NestLevel(t, depth - 1));
        }

        [TestMethod]
        public void DeepNestedFunctions_MixHandAndWoven_ValueCorrect_InputUntouched()
        {
            var input = np.arange(5).astype(NPTypeCode.Double);   // [0,1,2,3,4]
            var r = NestLevel(input, 5);   // six +1's across depths 5..0, then a woven roll by 1

            Assert.IsFalse(r.IsDisposed, "the result must survive every level of the nest");
            Assert.IsFalse(input.IsDisposed, "the input must be untouched through the whole chain");
            // [0..4] + 6 = [6,7,8,9,10]; roll 1 -> [10,6,7,8,9]
            Assert.AreEqual(10.0, r.GetDouble(0));
            Assert.AreEqual(6.0, r.GetDouble(1));
            Assert.AreEqual(9.0, r.GetDouble(4));

            r.Dispose();
            input.Dispose();
        }

        [TestMethod]
        public void DeepNestedFunctions_ZeroStrands()
        {
            var input = np.arange(16).astype(NPTypeCode.Double);
            void Cycle()
            {
                using (var r = NestLevel(input, 6)) { }
            }

            for (int i = 0; i < 3; i++) Cycle();
            DrainGC();

            SizeBucketedBufferPool.ResetCounters();
            const int N = 200;
            for (int i = 0; i < N; i++) Cycle();
            long deficit = Acquisitions - Releases;

            // 7 hand levels (a temp each) + the woven roll leaf per cycle; a per-level strand would
            // read as a multiple of N.
            Assert.IsTrue(deficit <= 8,
                $"deep nested chain stranded {deficit} buffers over {N} cycles (acq {Acquisitions}, rel {Releases})");
            Assert.IsFalse(input.IsDisposed);
            input.Dispose();
        }

        [TestMethod]
        public void ExceptionInNestedWovenChain_ReclaimsAtEveryLevel_InputsUntouched()
        {
            var input = np.arange(6).reshape(2, 3).astype(NPTypeCode.Double);
            NDArray nestedResult = null;
            NDArray crossResult = null;
            try
            {
                using var outer = NDScope.Open();
                nestedResult = np.roll(input, 1);   // woven -> woven, re-parented into `outer`
                crossResult = np.cross(              // woven; its inline inputs are tracked by `outer` too
                    np.array(new double[] { 1, 2, 3 }),
                    np.array(new double[] { 4, 5, 6 }));

                Assert.IsFalse(nestedResult.IsDisposed);
                Assert.IsFalse(crossResult.IsDisposed);
                throw new InvalidOperationException("boom after the nested woven calls");
            }
            catch (InvalidOperationException)
            {
            }

            Assert.IsTrue(nestedResult.IsDisposed, "the enclosing finally must reclaim nested-woven temps on the throw path");
            Assert.IsTrue(crossResult.IsDisposed, "every nested-woven result the scope owned must be reclaimed on throw");
            Assert.IsFalse(input.IsDisposed, "inputs must be untouched on the exception path");
            Assert.AreEqual(0.0, input.GetDouble(0, 0));
            Assert.AreEqual(5.0, input.GetDouble(1, 2));

            input.Dispose();
        }
    }
}
