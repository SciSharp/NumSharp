using System.Linq;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Weaver.Analyzer;

namespace NumSharp.Tests.Analyzer
{
    /// <summary>
    ///     Pins the analyzer CONTRACT — most importantly that a leak (NDW012) is reported as a
    ///     <b>Warning</b>, never an Error. A leak is a performance nudge (the finalizer still reclaims
    ///     the buffer), so it must never break a build; the target-gate diagnostics, by contrast, ARE
    ///     errors because a mis-woven method would silently misbehave.
    /// </summary>
    [TestClass]
    public class AnalyzerContractTests
    {
        [TestMethod]
        public void Ndw012_DefaultSeverity_IsWarning_And_EnabledByDefault()
        {
            var d = new NDArrayLeakAnalyzer().SupportedDiagnostics.Single(x => x.Id == "NDW012");
            Assert.AreEqual(DiagnosticSeverity.Warning, d.DefaultSeverity,
                "NDW012 must be a WARNING, never an Error — a leak is a nudge, not a build-breaker");
            Assert.IsTrue(d.IsEnabledByDefault, "NDW012 must be enabled by default");
        }

        [TestMethod]
        public async Task Ndw012_OnARealLeak_IsReportedAsWarning_NotError()
        {
            const string src = "using NumSharp;\npublic static class T {\n" +
                               "  public static void M(NDArray a, NDArray b) { var t = a + b; }\n}";
            var r = await AnalyzerTestHarness.RunAsync(src, "severity_leak.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "inline source must compile");

            var leak = r.Ndw.Single(x => x.Id == "NDW012");
            Assert.AreEqual(DiagnosticSeverity.Warning, leak.Severity,
                "a detected leak must surface as a Warning");
            Assert.AreEqual(0, r.Ndw.Count(x => x.Severity == DiagnosticSeverity.Error),
                "no diagnostic produced for a leak may be an Error");
        }

        [DataTestMethod]
        [DataRow("NDW002")]
        [DataRow("NDW003")]
        [DataRow("NDW005")]
        [DataRow("NDW006")]
        [DataRow("NDW009")]
        [DataRow("NDW010")]
        [DataRow("NDW011")]
        public void GateDiagnostics_AreErrors_ByDesign(string id)
        {
            // The target gate is a different contract: a wrong/unsupported [NDScoped] target is a
            // hard error (the same one the weaver emits). This pins the leak/gate severity split.
            var d = new NDScopedTargetAnalyzer().SupportedDiagnostics.Single(x => x.Id == id);
            Assert.AreEqual(DiagnosticSeverity.Error, d.DefaultSeverity,
                $"{id} is a target-gate error, distinct from the NDW012 leak warning");
        }
    }
}
