using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     Asserts the source-detectable <c>[NDScoped]</c>/<c>[NDScopedAsync]</c> TARGET gate
    ///     (NDW002/003/005/006/009/010/011) fires on exactly the tagged fixture lines. These are the
    ///     same diagnostics the IL weaver emits post-compile — the analyzer surfaces them at compile
    ///     time and preempts the weave.
    /// </summary>
    [TestClass]
    public class GateAnalyzerTests
    {
        [TestMethod]
        public Task GateScenarios_MatchTagsExactly() => AnalyzerTestHarness.AssertExactAsync("GateScenarios.cs");

        /// <summary>
        ///     OPEN BUG: an <c>[NDScoped]</c> method returning a managed array that cannot hold an NDArray
        ///     (<c>double[]</c>, <c>int[]</c>, <c>string[]</c>, <c>double[][]</c>) draws NDW003 as an "unsupported
        ///     carrier"; it carries no NDArray and must stay clean, like a scalar return. Found by Bigtree.Algorithm's
        ///     upgrade to NumSharp 0.80.0 (its count tables leave a scoped computation as <c>double[]</c>), which now
        ///     scopes those methods by hand. Remove <c>OpenBugs</c> when the gate (and the weaver's twin) treat an
        ///     array whose element type holds no NDArray as a non-carrier.
        /// </summary>
        /// <returns>The assertion task.</returns>
        [TestMethod]
        [TestCategory("OpenBugs")]
        public Task ManagedArrayReturns_AreNotGated() => AnalyzerTestHarness.AssertExactAsync("GateManagedArrayScenarios.cs");

        [DataTestMethod]
        [DataRow("NDW002")] // ref NDArray hidden egress
        [DataRow("NDW003")] // unsupported carrier return (List<NDArray>)
        [DataRow("NDW005")] // abstract / no body
        [DataRow("NDW006")] // setter-only property
        [DataRow("NDW009")] // async method under [NDScoped]
        [DataRow("NDW010")] // sync method under [NDScopedAsync]
        [DataRow("NDW011")] // both attributes
        public async Task EachGateCode_IsProducedExactlyOnce(string code)
        {
            var result = await AnalyzerTestHarness.RunFileAsync("GateScenarios.cs");
            Assert.IsTrue(result.CompileErrors.IsEmpty, "gate fixture must compile");
            Assert.AreEqual(1, result.CountOf(code), $"expected exactly one {code} in the gate fixture");
        }

        [TestMethod]
        public async Task GateFixture_ProducesNoLeakWarnings()
        {
            // Every gate method is [NDScoped]/[NDScopedAsync], so the LEAK analyzer must exempt them all.
            var result = await AnalyzerTestHarness.RunFileAsync("GateScenarios.cs");
            Assert.AreEqual(0, result.CountOf("NDW012"),
                "gate methods are all scoped, so none may draw NDW012");
        }
    }
}
