using System.Linq;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     Proves the harness and the fixtures are non-vacuous: the harness genuinely runs the
    ///     analyzer (a real leak warns, clean code does not, broken code is reported), and the fixture
    ///     files actually carry the tags they are gated on (so a stripped fixture fails rather than
    ///     passing silently).
    /// </summary>
    [TestClass]
    public class HarnessSelfTests
    {
        private const string Header = "using NumSharp;\npublic static class T {\n";
        private const string Footer = "\n}";

        [TestMethod]
        public async Task Harness_DetectsARealLeak()
        {
            var src = Header + "  public static void M(NDArray a, NDArray b) { var t = a + b; }" + Footer;
            var r = await AnalyzerTestHarness.RunAsync(src, "inline_leak.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "inline source must compile");
            Assert.AreEqual(1, r.CountOf("NDW012"), "a dropped 'a + b' must draw exactly one NDW012");
        }

        [TestMethod]
        public async Task Harness_LeavesCleanCodeAlone()
        {
            var src = Header + "  public static NDArray M(NDArray a, NDArray b) { return a + b; }" + Footer;
            var r = await AnalyzerTestHarness.RunAsync(src, "inline_clean.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "inline source must compile");
            Assert.AreEqual(0, r.Ndw.Length, "a returned value must draw no diagnostics");
        }

        [TestMethod]
        public async Task Harness_ReportsCompileErrors()
        {
            var r = await AnalyzerTestHarness.RunAsync(Header + "  this is not valid C#" + Footer, "inline_broken.cs");
            Assert.IsFalse(r.CompileErrors.IsEmpty, "a broken fixture must surface as compile errors, never pass silently");
        }

        [TestMethod]
        public async Task Harness_ExemptsScopedMethod()
        {
            var src = Header + "  [NDScoped] public static NDArray M(NDArray a, NDArray b) { var t = a + b; return t.copy(); }" + Footer;
            var r = await AnalyzerTestHarness.RunAsync(src, "inline_scoped.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty);
            Assert.AreEqual(0, r.CountOf("NDW012"), "a [NDScoped] method must be exempt from NDW012");
        }

        [TestMethod]
        public async Task Harness_ExemptsNDScopedCoveredMethod()
        {
            // Metamorphic pair — the SAME dead-local leak, and the ONLY difference is the attribute:
            // [NDScopedCovered] asserts the method always runs under a caller's ambient scope, so the
            // leak analyzer must exempt it (nothing is woven; the caller's scope reclaims the temp).
            const string body = "public static NDArray H(NDArray a, NDArray b) { var t = a + b; return a.copy(); }";

            var bare = await AnalyzerTestHarness.RunAsync(Header + "  " + body + Footer, "inline_helper_bare.cs");
            Assert.IsTrue(bare.CompileErrors.IsEmpty, "control source must compile");
            Assert.AreEqual(1, bare.CountOf("NDW012"), "control: the dead local 't' must leak without the attribute");

            var covered = await AnalyzerTestHarness.RunAsync(Header + "  [NDScopedCovered] " + body + Footer, "inline_covered.cs");
            Assert.IsTrue(covered.CompileErrors.IsEmpty, "covered source must compile");
            Assert.AreEqual(0, covered.CountOf("NDW012"), "a [NDScopedCovered] method must be exempt from NDW012");
        }

        [TestMethod]
        public async Task Covered_IsHonored_UnderNDScopedAsyncBoundary()
        {
            // "Supported by [NDScopedAsync]": a [NDScopedCovered] helper riding an ASYNC boundary's
            // ambient scope. The async method is exempt (its own attribute) and the covered helper is
            // exempt (its attribute), so a helper leak under an async boundary draws nothing. Dropping
            // [NDScopedCovered] surfaces the dead-local leak, so the pair is non-vacuous.
            const string helper = "static NDArray Helper(NDArray a, NDArray b) { var t = a + b; return a.copy(); }";
            const string boundary = "[NDScopedAsync] public static async System.Threading.Tasks.Task<NDArray> " +
                                    "Boundary(NDArray a, NDArray b) { await System.Threading.Tasks.Task.Yield(); return Helper(a, b); }";

            var bare = await AnalyzerTestHarness.RunAsync(Header + "  " + helper + "\n  " + boundary + Footer, "inline_covered_async_bare.cs");
            Assert.IsTrue(bare.CompileErrors.IsEmpty, "control source must compile");
            Assert.AreEqual(1, bare.CountOf("NDW012"), "control: the helper's dead local leaks without [NDScopedCovered]");

            var covered = await AnalyzerTestHarness.RunAsync(Header + "  [NDScopedCovered] " + helper + "\n  " + boundary + Footer, "inline_covered_async.cs");
            Assert.IsTrue(covered.CompileErrors.IsEmpty, "covered source must compile");
            Assert.AreEqual(0, covered.CountOf("NDW012"), "[NDScopedCovered] helper under an [NDScopedAsync] boundary must be exempt");
        }

        [DataTestMethod]
        [DataRow("LeakScenarios.cs", 6)]
        [DataRow("CarrierScenarios.cs", 12)]
        [DataRow("GateScenarios.cs", 7)]
        [DataRow("ControlFlowScenarios.cs", 11)]
        [DataRow("BodyKindScenarios.cs", 8)]
        [DataRow("OwningScenarios.cs", 3)]
        [DataRow("EscapeScenarios.cs", 1)]
        [DataRow("ScopeScenarios.cs", 1)]
        [DataRow("GateNegativeScenarios.cs", 4)]
        [DataRow("KnownLimitationScenarios.cs", 2)]
        [DataRow("HolderTypeScenarios.cs", 25)]
        [DataRow("DisposePathScenarios.cs", 14)]
        [DataRow("ContagionScenarios.cs", 9)]
        public void Fixtures_CarryTheTagsTheyAreGatedOn(string fileName, int minTags)
        {
            var markers = DiagnosticMarkers.Parse(AnalyzerTestHarness.FixturePath(fileName));
            Assert.IsTrue(markers.Count >= minTags,
                $"'{fileName}' should carry at least {minTags} // [NDWxxx] tags (found {markers.Count}) — " +
                "a stripped fixture must not pass vacuously");
        }

        [TestMethod]
        public void AllExpectedCodes_ArePresentAcrossFixtures()
        {
            var codes = new[] { "LeakScenarios.cs", "CarrierScenarios.cs", "GateScenarios.cs", "HolderTypeScenarios.cs", "DisposePathScenarios.cs", "ContagionScenarios.cs" }
                .SelectMany(f => DiagnosticMarkers.Parse(AnalyzerTestHarness.FixturePath(f)))
                .Select(m => m.Code)
                .Distinct()
                .OrderBy(c => c)
                .ToArray();
            foreach (var expected in new[] { "NDW002", "NDW003", "NDW005", "NDW006", "NDW009", "NDW010", "NDW011", "NDW012", "NDW016", "NDW017" })
                CollectionAssert.Contains(codes, expected, $"the fixtures must cover {expected}");
        }
    }
}
