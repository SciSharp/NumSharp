using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     COVERAGE_PLAN §3.1 — control-flow / data-flow leak scenarios with deterministic, correct
    ///     behavior: conditional assignment, loop temps, `using` statement/bare-expression, try/finally
    ///     dispose, explicit discard (the CF-10 fix), and throw-before-use. (The known-limitation cases
    ///     CF-1/CF-4/CF-11 live in <see cref="KnownLimitationTests"/>.)
    /// </summary>
    [TestClass]
    public class ControlFlowTests
    {
        [TestMethod]
        public Task ControlFlowScenarios_MatchTagsExactly()
            => AnalyzerTestHarness.AssertExactAsync("ControlFlowScenarios.cs");

        [TestMethod]
        public async Task LeakingControlFlow_IsFlagged()
        {
            var flagged = await FixtureFacts.FlaggedLineTexts("ControlFlowScenarios.cs");
            FixtureFacts.AnyStartsWith(flagged, "NDArray t;", "a conditionally-assigned local, never read");
            FixtureFacts.AnyStartsWith(flagged, "for (int i = 0;", "a temp created inside a loop");
            FixtureFacts.AnyStartsWith(flagged, "_ = a + b;", "an explicit discard (CF-10)");
            FixtureFacts.AnyStartsWith(flagged, "var t = a + b;", "a temp abandoned by a throw");
            // the conditional-value family (CF-4/CF-11 + coalesce): every branch produces -> owned
            FixtureFacts.AnyStartsWith(flagged, "var t = p ? a + b : c - d;", "a dropped both-owning ternary");
            FixtureFacts.AnyStartsWith(flagged, "var t = n switch { 0 => a + b, _ => c - d };", "a dropped both-owning switch expression");
            FixtureFacts.AnyStartsWith(flagged, "var t = (a + b) ?? (c - d);", "a dropped both-owning coalesce");
            FixtureFacts.AnyStartsWith(flagged, "_ = p ? a + b : c - d;", "a discarded both-owning ternary");
            // throw branches are vacuous for the every-branch rule
            FixtureFacts.AnyStartsWith(flagged, "var x = p ? a + b : throw", "a dropped produce-or-throw ternary");
            FixtureFacts.AnyStartsWith(flagged, "var t = n switch { 0 => a + b, _ => throw", "a dropped switch with a throw arm");
            // both the seed and the += result die unread
            FixtureFacts.AnyStartsWith(flagged, "var t = a + b;                                      // [NDW012]  seed",
                "a compound-assigned local, dropped");
        }

        [TestMethod]
        public async Task CleanControlFlow_IsNotFlagged()
        {
            var flagged = await FixtureFacts.FlaggedLineTexts("ControlFlowScenarios.cs");
            FixtureFacts.NoneContains(flagged, "else t = a;", "a conditional branch that aliases the input");
            FixtureFacts.NoneContains(flagged, "acc = acc + x", "an accumulator seeded by the input");
            FixtureFacts.NoneContains(flagged, "using (var t = a + b)", "a `using` statement");
            FixtureFacts.NoneContains(flagged, "using (a + b)", "a `using` on a bare owning expression");
            FixtureFacts.NoneContains(flagged, "t.Dispose();", "a try/finally dispose");
            FixtureFacts.NoneContains(flagged, "_ = a;", "a discard of a bare input (no owned buffer)");
            // MIXED conditionals may hand back the caller's value -> conservatively not owned
            FixtureFacts.NoneContains(flagged, "var t = p ? a + b : a;", "a dropped MIXED ternary (alias branch)");
            FixtureFacts.NoneContains(flagged, "var t = n switch { 0 => a + b, _ => a };", "a dropped MIXED switch expression");
            FixtureFacts.NoneContains(flagged, "return t;", "a returned conditional result");
            // an alias completing branch beside a throw stays conservative
            FixtureFacts.NoneContains(flagged, "var t = p ? a : throw", "an alias-or-throw ternary");
            // every reclaim spelling counts
            FixtureFacts.NoneContains(flagged, "t?.Dispose();", "a conditional dispose");
            FixtureFacts.NoneContains(flagged, "((System.IDisposable)t).Dispose();", "a dispose through an interface cast");
            FixtureFacts.NoneContains(flagged, "catch { t.Dispose(); }", "a catch-only dispose (path-insensitive)");
            FixtureFacts.NoneContains(flagged, "using ((System.IDisposable)(a + b))", "a using on a cast bare expression");
            FixtureFacts.NoneContains(flagged, "t += c;\n            return t;", "a compound-assigned local that is returned");
        }
    }
}
