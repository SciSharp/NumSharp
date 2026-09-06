using System.Linq;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Build.Analyzer;

namespace NumSharp.Tests.Build.Analyzer
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
        [DataRow("NDW016")]
        [DataRow("NDW017")]
        public void OwnershipDiagnostics_DefaultSeverity_IsWarning_And_EnabledByDefault(string id)
        {
            // The type-level ownership diagnostics are nudges of the same kind as NDW012: a stored
            // array that is never disposed still reaches the finalizer, so they must never break a build.
            var d = new NDArrayHolderAnalyzer().SupportedDiagnostics.Single(x => x.Id == id);
            Assert.AreEqual(DiagnosticSeverity.Warning, d.DefaultSeverity, $"{id} must be a WARNING, never an Error");
            Assert.IsTrue(d.IsEnabledByDefault, $"{id} must be enabled by default");
            Assert.IsTrue(d.HelpLinkUri.EndsWith("#" + id.ToLowerInvariant()), $"{id} links its own anchor on the build-compiler page");
        }

        [TestMethod]
        public async Task OwnershipDiagnostics_OnRealCases_AreReportedAsWarnings_NotErrors()
        {
            const string src = "using NumSharp;\n" +
                               "public class Stores { NDArray _a; }\n" +
                               "public class Forgets : System.IDisposable { NDArray _a; public void Dispose() { } }";
            var r = await AnalyzerTestHarness.RunAsync(src, "severity_ownership.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "inline source must compile");
            Assert.AreEqual(DiagnosticSeverity.Warning, r.Ndw.Single(x => x.Id == "NDW016").Severity);
            Assert.AreEqual(DiagnosticSeverity.Warning, r.Ndw.Single(x => x.Id == "NDW017").Severity);
            Assert.AreEqual(0, r.Ndw.Count(x => x.Severity == DiagnosticSeverity.Error), "no ownership finding may be an Error");
        }

        [TestMethod]
        public void TheThreeAnalyzers_CoverDisjointDiagnosticIds()
        {
            var leak = new NDArrayLeakAnalyzer().SupportedDiagnostics.Select(d => d.Id).ToArray();
            var gate = new NDScopedTargetAnalyzer().SupportedDiagnostics.Select(d => d.Id).ToArray();
            var holder = new NDArrayHolderAnalyzer().SupportedDiagnostics.Select(d => d.Id).ToArray();
            CollectionAssert.AreEquivalent(new[] { "NDW012" }, leak);
            CollectionAssert.AreEquivalent(new[] { "NDW016", "NDW017" }, holder);
            Assert.AreEqual(0, leak.Intersect(gate).Count() + leak.Intersect(holder).Count() + gate.Intersect(holder).Count(),
                "no id is owned by two analyzers");
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

        [TestMethod]
        public void Ndw013_WeaverMissing_IsAWarning_LikeItsMsBuildTwin()
        {
            // The analyzer's NDW013 — an own [NDScoped]/[NDScopedAsync], an inherited one, or an
            // [NDScopedExit] parameter, while the build says the weaver is inactive — shares its id and
            // severity with the fallback MSBuild warning in NumSharp's build/NumSharp.targets: a nudge
            // about inert attributes, never a build error. Three descriptors carry the one id (three
            // sentences for three shapes); every one of them is a default-on warning.
            var ds = new NDScopedTargetAnalyzer().SupportedDiagnostics.Where(x => x.Id == "NDW013").ToList();
            Assert.AreEqual(3, ds.Count, "own attribute / inherited attribute / [NDScopedExit] — one NDW013 descriptor each");
            foreach (var d in ds)
            {
                Assert.AreEqual(DiagnosticSeverity.Warning, d.DefaultSeverity, $"NDW013 '{d.Title}' is a warning");
                Assert.IsTrue(d.IsEnabledByDefault, $"NDW013 '{d.Title}' is on by default");
                StringAssert.Contains(d.MessageFormat.ToString(), "dotnet add package NumSharp.Build", $"NDW013 '{d.Title}' names the fix");
            }
        }
    }
}
