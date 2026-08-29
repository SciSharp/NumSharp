using System;

namespace NumSharp.Tests.Build.Analyzer.Fixtures
{
    // Documented analyzer divergences (COVERAGE_PLAN §3.1 CF-1 + the mixed coalesce, §3.2 EC-13, §3.4
    // SC-5, §3.5 AM-1). NumSharp.Tests.Build.Analyzer asserts THIS file under [TestCategory("Misaligned")],
    // pinning the analyzer's CURRENT (imperfect) behavior EXACTLY — so improving any of these flips a tag
    // and turns the pin red, i.e. a change is noticed rather than silent. (CF-4/CF-11 — the dropped
    // ternary/switch of owning temps — used to be pinned here and are FIXED: every-branch-produces
    // conditionals are owned now; see ControlFlowScenarios.)
    //
    //   • FALSE NEGATIVES (a real leak the per-local/per-expression analysis misses) produce NOTHING
    //     today and are therefore UNtagged — a future fix would add a warning and fail the exact match.
    //   • FALSE POSITIVES (a value that is actually kept alive, but looks abandoned) are TAGGED — a
    //     future fix would remove the warning and fail the exact match.
    public static class KnownLimitationScenarios
    {
        private static NDArray _a;

        // ---- FALSE NEGATIVES (should warn, currently silent — untagged) ----

        // CF-1: the first owning value is orphaned by the reassignment, never reclaimed. Catching it
        // needs per-assignment (SSA) tracking; the analysis is per-local, so it is silent here.
        public static NDArray ReassignmentOrphan(NDArray a, NDArray b, NDArray c, NDArray d)
        {
            var t = a + b;   // orphaned — a real leak the per-local analysis does not see
            t = c - d;
            return t;
        }

        // A MIXED coalesce: the right side leaks whenever x is null, but the left side is the caller's
        // value, so under the every-branch-produces rule the result is conservatively NOT owned (the
        // ternary/switch mixed twins pin the same rule in ControlFlowScenarios).
        public static void CoalesceMixedDropped(NDArray x, NDArray a, NDArray b)
        {
            var t = x ?? (a + b);   // leaks when x is null; not flagged
        }

        // An `out var` result is untracked: whether the callee handed out a FRESH buffer or an alias
        // of something it retains is not visible per-method, so ownership is conservatively not
        // assumed — the dropped result is silent even when the callee really did construct it.
        private static void MakeOut(NDArray a, out NDArray x) { x = a + 1.0; }
        public static void OutVarDropped(NDArray a)
        {
            MakeOut(a, out var t);   // t owns a fresh buffer here; not flagged
        }

        // A declaration-pattern local (`expr is NDArray t`) is untracked — the produced input escapes
        // into the pattern match. Rare enough that tracking it is not worth the plumbing.
        public static void IsPatternUnused(NDArray a, NDArray b)
        {
            if ((a + b) is NDArray t) { }   // a+b is always constructed; not flagged
        }

        // SC-5: a lambda inside a scoped method rides the enclosing [NDScoped] exemption, so its temp is
        // not analyzed — yet the lambda may run AFTER the scope closes. Silent today.
        [NDScoped]
        public static NDArray LambdaInScopedMethod(NDArray a, NDArray b)
        {
            Action f = () => { var t = a + b; };   // deferred execution; the ambient scope may be gone
            f();
            return a.copy();
        }

        // ---- FALSE POSITIVES (kept alive, but flagged — tagged) ----

        // EC-13: np.reshape returns a VIEW that keeps the (a + b) buffer alive, so returning it does not
        // leak — but a NumSharp op that returns an NDArray is indistinguishable from one that consumes
        // its input, so the temp is flagged. (Fresh-vs-view is undecidable syntactically.)
        public static NDArray ReshapeViewOfTemp(NDArray a, NDArray b)
        {
            return np.reshape(a + b, new int[] { 1, -1 });  // [NDW012]  accepted false positive
        }

        // AM-1: this helper's temp is reclaimed by the ambient scope its [NDScoped] CALLER opens, but a
        // per-method analysis cannot see that call-graph coverage, so it over-reports.
        public static NDArray AmbientCoveredHelper(NDArray a)
        {
            var idx = a + 1.0;                               // [NDW012]  covered by the caller's scope, still flagged
            return np.minimum(idx, a);
        }

        // AM-2: the same helper marked [NDScoped] — the fix — is clean.
        [NDScoped]
        public static NDArray AmbientCoveredHelperScoped(NDArray a)
        {
            var idx = a + 1.0;
            return np.minimum(idx, a);
        }
    }
}
