using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     COVERAGE_PLAN §3.4 — scope / exemption nuances: a scoped accessor, a hand-opened NDScope
    ///     (statement form, and in a nested block), NDScope.Detach, an async-scoped method, and the one
    ///     case that is NOT exempt — a local function inside a non-scoped method (its own block).
    /// </summary>
    [TestClass]
    public class ScopeTests
    {
        [TestMethod]
        public Task ScopeScenarios_MatchTagsExactly()
            => AnalyzerTestHarness.AssertExactAsync("ScopeScenarios.cs");

        [TestMethod]
        public async Task ScopedAndHandScoped_AreExempt()
        {
            var flagged = await FixtureFacts.FlaggedLineTexts("ScopeScenarios.cs");
            FixtureFacts.NoneContains(flagged, "get { var t = _a + 1.0;", "a [NDScoped] property getter");
            FixtureFacts.NoneContains(flagged, "using (NDScope.Open())", "a hand-opened scope");
            FixtureFacts.NoneContains(flagged, "{ using (NDScope.Open()) { } }", "a nested hand-opened scope");
            FixtureFacts.NoneContains(flagged, "NDScope.Detach(t);", "a detached value");
            FixtureFacts.NoneContains(flagged, "var t = a + b;\n", "the temp after the nested scope");
        }

        [TestMethod]
        public async Task LocalFunctionInNonScopedMethod_IsAnalyzed()
        {
            var flagged = await FixtureFacts.FlaggedLineTexts("ScopeScenarios.cs");
            FixtureFacts.AnyStartsWith(flagged, "void L() { var t = a + b; }", "a leaking local function");
        }
    }
}
