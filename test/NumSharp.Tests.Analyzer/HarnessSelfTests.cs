using System.Linq;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Analyzer
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
        public async Task Harness_ExemptsNDScopedHelperMethod()
        {
            // Metamorphic pair — the SAME dead-local leak, and the ONLY difference is the attribute:
            // [NDScopedHelper] asserts the method always runs under a caller's ambient scope, so the
            // leak analyzer must exempt it (nothing is woven; the caller's scope reclaims the temp).
            const string body = "public static NDArray H(NDArray a, NDArray b) { var t = a + b; return a.copy(); }";

            var bare = await AnalyzerTestHarness.RunAsync(Header + "  " + body + Footer, "inline_helper_bare.cs");
            Assert.IsTrue(bare.CompileErrors.IsEmpty, "control source must compile");
            Assert.AreEqual(1, bare.CountOf("NDW012"), "control: the dead local 't' must leak without the attribute");

            var helped = await AnalyzerTestHarness.RunAsync(Header + "  [NDScopedHelper] " + body + Footer, "inline_helper.cs");
            Assert.IsTrue(helped.CompileErrors.IsEmpty, "helper source must compile");
            Assert.AreEqual(0, helped.CountOf("NDW012"), "a [NDScopedHelper] method must be exempt from NDW012");
        }

        [DataTestMethod]
        [DataRow("LeakScenarios.cs", 6)]
        [DataRow("CarrierScenarios.cs", 4)]
        [DataRow("GateScenarios.cs", 7)]
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
            var codes = new[] { "LeakScenarios.cs", "CarrierScenarios.cs", "GateScenarios.cs" }
                .SelectMany(f => DiagnosticMarkers.Parse(AnalyzerTestHarness.FixturePath(f)))
                .Select(m => m.Code)
                .Distinct()
                .OrderBy(c => c)
                .ToArray();
            foreach (var expected in new[] { "NDW002", "NDW003", "NDW005", "NDW006", "NDW009", "NDW010", "NDW011", "NDW012" })
                CollectionAssert.Contains(codes, expected, $"the fixtures must cover {expected}");
        }
    }
}
