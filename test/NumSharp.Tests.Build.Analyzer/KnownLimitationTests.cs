using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     COVERAGE_PLAN §3.5 + the documented false-negatives/positives (CF-1, the mixed coalesce,
    ///     SC-5, EC-13, AM-1). These PIN the analyzer's current, deliberately-imperfect behavior so a
    ///     future improvement flips the outcome and is NOTICED (the test turns red) rather than passing
    ///     silently. Categorised Misaligned — they document a divergence, not a defect to fix now.
    ///     (CF-4/CF-11 were pinned here until the every-branch-produces conditional fix landed; their
    ///     warn cases now live in ControlFlowScenarios.)
    /// </summary>
    [TestClass]
    [TestCategory("Misaligned")]
    public class KnownLimitationTests
    {
        // Exact-match: the false negatives produce NOTHING (untagged) and the two accepted false
        // positives produce their tagged NDW012 — pinned to the line.
        [TestMethod]
        public Task KnownLimitationScenarios_MatchTagsExactly()
            => AnalyzerTestHarness.AssertExactAsync("KnownLimitationScenarios.cs");

        [TestMethod]
        public async Task FalseNegatives_AreCurrentlySilent()
        {
            var flagged = await FixtureFacts.FlaggedLineTexts("KnownLimitationScenarios.cs");
            // CF-1 reassignment orphan
            FixtureFacts.NoneContains(flagged, "var t = a + b;   // orphaned",
                "the reassignment-orphaned first value (CF-1, a documented false negative)");
            // mixed coalesce (the every-branch-produces rule's residue)
            FixtureFacts.NoneContains(flagged, "var t = x ?? (a + b);",
                "a dropped mixed coalesce (leaks when x is null; a documented false negative)");
            // SC-5 lambda inside a scoped method
            FixtureFacts.NoneContains(flagged, "Action f = () => { var t = a + b; };",
                "a lambda temp inside a scoped method (SC-5, a documented false negative)");
            // out-var result (callee-fresh vs alias is undecidable per-method)
            FixtureFacts.NoneContains(flagged, "MakeOut(a, out var t);",
                "a dropped out-var result (a documented false negative)");
            // declaration-pattern local
            FixtureFacts.NoneContains(flagged, "if ((a + b) is NDArray t)",
                "a produced value matched into an unused pattern local (a documented false negative)");
        }

        [TestMethod]
        public async Task FalsePositives_AreAcceptedAndPinned()
        {
            var flagged = await FixtureFacts.FlaggedLineTexts("KnownLimitationScenarios.cs");
            // EC-13 view-of-temp
            FixtureFacts.AnyStartsWith(flagged, "return np.reshape(a + b,",
                "a temp handed to a view-returning NumSharp op (EC-13, accepted false positive)");
            // AM-1 ambient-covered helper
            FixtureFacts.AnyStartsWith(flagged, "var idx = a + 1.0;",
                "a helper temp covered by the caller's ambient scope (AM-1, accepted false positive)");
            // AM-2: the same helper, scoped, is clean (the fix).
            var scopedIsFlaggedTwice = flagged.FindAll(t => t.StartsWith("var idx = a + 1.0;")).Count;
            Assert.AreEqual(1, scopedIsFlaggedTwice,
                "only the UN-scoped ambient helper warns; the [NDScoped] twin (AM-2) must be clean");
        }
    }
}
