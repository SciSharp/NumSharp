using System.Runtime.CompilerServices;

namespace NumSharp.Tests.Lifetime
{
    /// <summary>
    ///     Contracts for the <see cref="NDScopedExitAttribute"/> escape hatch — a parameter the callee
    ///     RETAINS, which the caller's <see cref="NDScope"/> must therefore NOT reclaim. Two layers:
    ///     the runtime <see cref="NDScope.Detach(NDArray)"/> overloads the weave emits (tested by direct
    ///     call — no weave needed), and the CALLEE-detaches-its-own-argument semantics the weaver injects
    ///     at method entry (pinned here by a hand-written method whose body IS the woven shape — a
    ///     <c>Detach</c> at the top — exactly as <see cref="NDScopeTests.OutParameter_Egress_ViaReturns"/>
    ///     pins the out-parameter egress by hand). The emitted IL is a structural twin of the proven
    ///     <c>Returns(ITuple)</c>/<c>Detach</c> patterns, so this behavioural pin plus the self-weave's
    ///     integration coverage is the same bar the codebase holds the out-param branch to.
    /// </summary>
    [TestClass]
    [DoNotParallelize]   // the retained-array statics below are shared across the tests
    public class NDScopeExitTests
    {
        private static NDArray _retained;

        // The woven shape of `void Retain([NDScopedExit] NDArray a) { _retained = a; }` — the weaver
        // prepends `NDScope.Detach(a)` at method entry, and the callee runs inside the caller's ambient
        // scope, so this reaches into the caller's scope and un-tracks the argument.
        private static void RetainWithDetach(NDArray a)
        {
            NDScope.Detach(a);
            _retained = a;
        }

        // The UNWOVEN shape — no detach — used as the teeth: the caller's scope DOES dispose it.
        private static void RetainWithoutDetach(NDArray a)
        {
            _retained = a;
        }

        // ------------------------------------------------------------------ the runtime Detach overloads

        [TestMethod]
        public void Detach_NDArrayArray_AllElementsSurviveScope()
        {
            NDArray[] items;
            using (var scope = NDScope.Open())
            {
                items = new[] { np.arange(4), np.arange(4) };   // both tracked by this scope
                NDScope.Detach(items);                           // NDArray[] overload — detach each
            }                                                    // scope disposes; the detached arrays are not in it

            Assert.IsFalse(items[0].IsDisposed, "a detached array must survive scope disposal");
            Assert.IsFalse(items[1].IsDisposed);
            Assert.AreEqual(2L, items[0].GetInt64(2), "the surviving buffer is still readable");
            items[0].Dispose();
            items[1].Dispose();
        }

        [TestMethod]
        public void Detach_NDArrayArray_WithoutDetach_AreReclaimed_Teeth()
        {
            NDArray[] items;
            using (var scope = NDScope.Open())
            {
                items = new[] { np.arange(4), np.arange(4) };   // tracked, NOT detached
            }

            Assert.IsTrue(items[0].IsDisposed, "without Detach the scope reclaims the tracked arrays");
            Assert.IsTrue(items[1].IsDisposed);
        }

        [TestMethod]
        public void Detach_ValueTuple_ViaITuple_AllComponentsSurvive()
        {
            (NDArray, NDArray) tup;
            using (var scope = NDScope.Open())
            {
                tup = (np.arange(4), np.arange(4));
                NDScope.Detach((ITuple)tup);   // the exact box-to-ITuple the weaver emits for a ValueTuple param
            }

            Assert.IsFalse(tup.Item1.IsDisposed, "each NDArray a tuple carries must survive");
            Assert.IsFalse(tup.Item2.IsDisposed);
            Assert.AreEqual(1L, tup.Item2.GetInt64(1));
            tup.Item1.Dispose();
            tup.Item2.Dispose();
        }

        [TestMethod]
        public void Detach_MixedTuple_SkipsNonNDArrayComponents()
        {
            (NDArray, int) tup;
            using (var scope = NDScope.Open())
            {
                tup = (np.arange(4), 5);
                NDScope.Detach((ITuple)tup);   // the int component is skipped, no crash
            }

            Assert.IsFalse(tup.Item1.IsDisposed, "the NDArray component survives");
            Assert.AreEqual(5, tup.Item2, "the non-NDArray component is untouched");
            tup.Item1.Dispose();
        }

        [TestMethod]
        public void Detach_NullAndNullElements_AreNoOps()
        {
            NDScope.Detach((NDArray)null);
            NDScope.Detach((NDArray[])null);
            NDScope.Detach((ITuple)null);
            NDScope.Detach(new NDArray[] { null, null });   // null elements skipped
            // no throw is the assertion
        }

        // ------------------------------------------------------------------ the [NDScopedExit] weave semantics

        [TestMethod]
        public void ExitParam_RetainedArgument_SurvivesCallerScope()
        {
            NDArray a;
            using (var scope = NDScope.Open())
            {
                a = np.arange(4);       // constructed under the caller's scope -> tracked
                RetainWithDetach(a);    // callee detaches its own argument, then retains it
            }                           // caller's scope disposes -> `a` was detached, so it is spared

            Assert.IsFalse(a.IsDisposed, "an [NDScopedExit] argument the callee keeps must not be reclaimed by the caller's scope");
            Assert.AreEqual(3L, _retained.GetInt64(3), "the retained reference is a live, correct array");

            a.Dispose();
            _retained = null;
        }

        [TestMethod]
        public void ExitParam_WithoutDetach_RetainedArgumentDangles_Teeth()
        {
            NDArray a;
            using (var scope = NDScope.Open())
            {
                a = np.arange(4);
                RetainWithoutDetach(a);   // no detach — the exact bug [NDScopedExit] closes
            }

            Assert.IsTrue(a.IsDisposed, "without the detach the caller's scope reclaims the retained argument (a use-after-free waiting to happen)");
            _retained = null;
        }

        [TestMethod]
        public void ExitParam_NoAmbientScope_DetachIsHarmless()
        {
            var a = np.arange(4);     // no scope open -> untracked
            RetainWithDetach(a);      // Detach is a no-op on an untracked array
            Assert.IsFalse(a.IsDisposed);
            Assert.AreEqual(2L, _retained.GetInt64(2));
            a.Dispose();
            _retained = null;
        }
    }
}
