using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     COVERAGE_PLAN §4 — the gate's negative half (supported carriers stay CLEAN) plus two more
    ///     rejections, and the T4 weaver-parity table: a curated set of return types asserted to be
    ///     gated (NDW003) exactly when the weaver would reject them, driven through the real gate
    ///     analyzer so the analyzer's carrier classification cannot drift from the weaver's contract.
    /// </summary>
    [TestClass]
    public class GateNegativeTests
    {
        [TestMethod]
        public Task GateNegativeScenarios_MatchTagsExactly()
            => AnalyzerTestHarness.AssertExactAsync("GateNegativeScenarios.cs");

        [TestMethod]
        public async Task GateFixture_HasExactlyTheExpectedGateErrors()
        {
            var r = await AnalyzerTestHarness.RunFileAsync("GateNegativeScenarios.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "gate-negative fixture must compile");
            Assert.AreEqual(2, r.CountOf("NDW002"), "two hidden ref/in egresses");
            Assert.AreEqual(2, r.CountOf("NDW003"), "two unsupported carriers");
            Assert.AreEqual(0, r.CountOf("NDW012"), "every method here is scoped -> no leak warnings");
            Assert.AreEqual(4, r.Ndw.Length, "the supported-carrier methods must add NO diagnostics");
        }

        // ---- T4: the analyzer's carrier classification must match the weaver's Classify ----

        // Supported shapes: a [NDScoped]/[NDScopedAsync] method returning one of these is NOT gated.
        [DataTestMethod]
        [DataRow("NDArray", false)]
        [DataRow("NDArray[]", false)]
        [DataRow("(NDArray, NDArray)", false)]
        [DataRow("(NDArray, NDArray, NDArray)", false)]
        [DataRow("(NDArray, NDArray, NDArray, NDArray, NDArray)", false)]
        [DataRow("NumSharp.Generic.NDArray<double>", false)]
        [DataRow("NumSharp.Backends.Unmanaged.IArraySlice", false)]
        [DataRow("NumSharp.Backends.UnmanagedStorage", false)]
        [DataRow("int", false)]
        [DataRow("double", false)]
        [DataRow("bool", false)]
        [DataRow("void", false)]
        // Unsupported shapes: gated with NDW003.
        [DataRow("System.Collections.Generic.List<NDArray>", true)]
        [DataRow("object", true)]
        public async Task SyncCarrierParity(string returnType, bool gated)
        {
            var src = "using NumSharp;\npublic static class T {\n" +
                      $"  [NDScoped] public static {returnType} M(NDArray a) => throw null;\n}}";
            var r = await AnalyzerTestHarness.RunAsync(src, "sync_carrier.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "carrier snippet must compile:\n" + src);
            Assert.AreEqual(gated ? 1 : 0, r.CountOf("NDW003"),
                $"return type '{returnType}' should {(gated ? "" : "NOT ")}be gated (NDW003)");
        }

        // Task/ValueTask-of-carrier shapes under [NDScopedAsync]: the carrier is unwrapped once.
        [DataTestMethod]
        [DataRow("System.Threading.Tasks.Task<NDArray>", false)]
        [DataRow("System.Threading.Tasks.ValueTask<NDArray>", false)]
        [DataRow("System.Threading.Tasks.Task<NDArray[]>", false)]
        [DataRow("System.Threading.Tasks.Task<(NDArray, NDArray)>", false)]
        [DataRow("System.Threading.Tasks.Task", false)]
        [DataRow("System.Threading.Tasks.ValueTask", false)]
        [DataRow("System.Threading.Tasks.Task<System.Collections.Generic.List<NDArray>>", true)]
        [DataRow("System.Threading.Tasks.Task<object>", true)]
        public async Task AsyncCarrierParity(string returnType, bool gated)
        {
            var src = "using NumSharp;\npublic static class T {\n" +
                      $"  [NDScopedAsync] public static {returnType} M(NDArray a) => throw null;\n}}";
            var r = await AnalyzerTestHarness.RunAsync(src, "async_carrier.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "carrier snippet must compile:\n" + src);
            Assert.AreEqual(gated ? 1 : 0, r.CountOf("NDW003"),
                $"return type '{returnType}' should {(gated ? "" : "NOT ")}be gated (NDW003)");
        }
    }
}
