using System.Collections.Generic;

namespace NumSharp.Tests.Build.Analyzer.Fixtures
{
    // NDW012 control-flow / data-flow scenarios (COVERAGE_PLAN §3.1) whose behavior is deterministic
    // and correct today. A line tagged with the diagnostic id MUST draw exactly one NDW012; every
    // untagged line MUST stay clean. The remaining known-limitation cases (CF-1 reassignment orphan,
    // mixed coalesce) live in KnownLimitationScenarios.cs (documented false negatives), not here.
    public static class ControlFlowScenarios
    {
        // CF-2: both branches assign an owning value, the local is never read -> the local leaks.
        // Reported on the declaration (the local is the owner; both assignments feed it).
        public static void CondBothOwning(bool p, NDArray a, NDArray b, NDArray c, NDArray d)
        {
            NDArray t;                                      // [NDW012]  t is only ever assigned, never read
            if (p) t = a + b;
            else t = c - d;
        }

        // CF-3: one branch aliases the input -> the local is dropped from the owning set (R2-safe),
        // so even though it is returned it draws nothing. Pins the conservative alias handling.
        public static NDArray CondOneBranchAliases(bool p, NDArray a, NDArray b)
        {
            NDArray t;
            if (p) t = a + b;
            else t = a;                                     // aliases the input -> not this method's to reclaim
            return t.copy();
        }

        // CF-5: a temp created inside a loop leaks on every iteration.
        public static void LoopTemp(int n, NDArray a, NDArray b)
        {
            for (int i = 0; i < n; i++) { var t = a + b; }  // [NDW012]  leaks each iteration
        }

        // CF-6: the accumulator seed is the input (an alias), so the local is conservatively clean.
        // The per-iteration `acc + x` intermediates do leak, but catching them needs SSA the per-local
        // analysis deliberately does not do — pinned as a clean case (documented in the plan).
        public static NDArray Accumulator(NDArray a, IEnumerable<NDArray> xs)
        {
            NDArray acc = a;
            foreach (var x in xs) acc = acc + x;
            return acc;
        }

        // CF-7: a `using` STATEMENT (not a `using var` declaration) reclaims the local.
        public static void UsingStatement(NDArray a, NDArray b)
        {
            using (var t = a + b) { }
        }

        // CF-8: a `using` on a bare owning expression (no local) reclaims the resource value.
        public static void UsingBareExpression(NDArray a, NDArray b)
        {
            using (a + b) { }
        }

        // CF-9: a `try/finally` that disposes the temp reclaims it.
        public static void TryFinallyDispose(NDArray a, NDArray b)
        {
            var t = a + b;
            try { }
            finally { t.Dispose(); }
        }

        // CF-10: an explicit discard drops the owned buffer on the floor — a leak, not a store.
        public static void ExplicitDiscard(NDArray a, NDArray b)
        {
            _ = a + b;                                      // [NDW012]  discarded, never reclaimed
        }

        // CF-10 (negative): discarding a bare INPUT owns nothing, so it must NOT warn.
        public static void DiscardOfInput(NDArray a)
        {
            _ = a;                                          // an alias of the input — nothing to reclaim
        }

        // CF-12: constructed then dropped by an unconditional throw — still built, still leaks.
        public static void ThrownBeforeUse(NDArray a, NDArray b)
        {
            var t = a + b;                                  // [NDW012]  constructed, then abandoned
            throw new System.Exception();
        }

        // CF-4: a conditional value whose EVERY branch produces owns its result whichever way the
        // branch goes — dropping it is a leak on all paths.
        public static void TernaryBothOwningDropped(bool p, NDArray a, NDArray b, NDArray c, NDArray d)
        {
            var t = p ? a + b : c - d;                      // [NDW012]  both branches produce, dropped
        }

        // CF-11: the switch-expression twin of CF-4.
        public static void SwitchBothOwningDropped(int n, NDArray a, NDArray b, NDArray c, NDArray d)
        {
            var t = n switch { 0 => a + b, _ => c - d };    // [NDW012]  every arm produces, dropped
        }

        // The coalesce twin: both sides produce, so the result is owned whichever is taken.
        public static void CoalesceBothOwningDropped(NDArray a, NDArray b, NDArray c, NDArray d)
        {
            var t = (a + b) ?? (c - d);                     // [NDW012]  both sides produce, dropped
        }

        // A discarded conditional drops the owned result exactly like a discarded operator.
        public static void TernaryDiscarded(bool p, NDArray a, NDArray b, NDArray c, NDArray d)
        {
            _ = p ? a + b : c - d;                          // [NDW012]  discarded, owned on every path
        }

        // CF-4 (negative): a MIXED conditional may hand back the caller's value, so it is
        // conservatively NOT owned (the same rule as the CF-3 aliasing if/else branch).
        public static void TernaryMixedDropped(bool p, NDArray a, NDArray b)
        {
            var t = p ? a + b : a;
        }

        public static void SwitchMixedDropped(int n, NDArray a, NDArray b)
        {
            var t = n switch { 0 => a + b, _ => a };
        }

        // Returned conditionals are ordinary escapes.
        public static NDArray TernaryReturnedViaLocal(bool p, NDArray a, NDArray b, NDArray c, NDArray d)
        {
            var t = p ? a + b : c - d;
            return t;
        }
    }
}
