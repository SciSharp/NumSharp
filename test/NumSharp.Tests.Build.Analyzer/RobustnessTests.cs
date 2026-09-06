using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     COVERAGE_PLAN §6 (technique T6) — the analyzer must never crash and must hold its invariants
    ///     on hostile input: malformed source, a very large body, recursive local functions, and generic
    ///     host types / methods. A returned value is never flagged; a dropped one always is.
    /// </summary>
    [TestClass]
    public class RobustnessTests
    {
        [TestMethod]
        public async Task MalformedSource_DoesNotCrash_ReportsCompileErrors()
        {
            var r = await AnalyzerTestHarness.RunAsync(
                "using NumSharp;\npublic static class T {\n  public static void M(NDArray a) { var t = a + ", "broken.cs");
            Assert.IsFalse(r.CompileErrors.IsEmpty, "a broken body must surface as compile errors, not throw");
        }

        [TestMethod]
        public async Task UnresolvedSymbols_DoNotProduceBogusWarnings()
        {
            // `Nonexistent` does not resolve; the analyzer must not invent an NDW on this.
            var r = await AnalyzerTestHarness.RunAsync(
                "using NumSharp;\npublic static class T {\n  public static void M(NDArray a) { var t = Nonexistent(a); }\n}", "unresolved.cs");
            Assert.AreEqual(0, r.CountOf("NDW012"), "an unresolved call is not a known NumSharp producer");
        }

        [TestMethod]
        [Timeout(60000)]
        public async Task VeryLargeBody_Completes()
        {
            var sb = new StringBuilder("using NumSharp;\npublic static class T {\n");
            sb.Append("  public static void M(NDArray a, NDArray b) {\n");
            const int n = 400;
            for (int i = 0; i < n; i++)
                sb.Append($"    var t{i} = a + b;\n");   // each a dropped leak
            sb.Append("  }\n}");
            var r = await AnalyzerTestHarness.RunAsync(sb.ToString(), "large.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "the large body must compile");
            Assert.AreEqual(n, r.CountOf("NDW012"), "every dropped temp in the large body must be flagged");
        }

        [TestMethod]
        public async Task RecursiveLocalFunction_DoesNotHang()
        {
            var r = await AnalyzerTestHarness.RunAsync(
                "using NumSharp;\npublic static class T {\n" +
                "  public static void M(NDArray a, NDArray b) {\n" +
                "    void L(int n) { if (n > 0) L(n - 1); var t = a + b; }\n" +
                "    L(3);\n  }\n}", "recursive.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "recursive local function must compile");
            Assert.AreEqual(1, r.CountOf("NDW012"), "the recursive local function leaks its temp exactly once");
        }

        [TestMethod]
        public async Task GenericHostType_Resolves()
        {
            var r = await AnalyzerTestHarness.RunAsync(
                "using NumSharp;\npublic class G<TX> {\n  public void M(NDArray a, NDArray b) { var t = a + b; }\n}", "generic_type.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty);
            Assert.AreEqual(1, r.CountOf("NDW012"), "a leak in a generic host type is still flagged");
        }

        [TestMethod]
        public async Task GenericMethod_Resolves()
        {
            var r = await AnalyzerTestHarness.RunAsync(
                "using NumSharp;\npublic static class T {\n  public static void M<TX>(NDArray a, NDArray b) { var t = a + b; }\n}", "generic_method.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty);
            Assert.AreEqual(1, r.CountOf("NDW012"), "a leak in a generic method is still flagged");
        }

        [DataTestMethod]
        [DataRow("return a + b;", 0)]              // returned -> clean
        [DataRow("var t = a + b; t.Dispose();", 0)] // disposed -> clean
        [DataRow("using var t = a + b;", 0)]        // using -> clean
        [DataRow("var t = a + b;", 1)]              // dropped -> leaks
        [DataRow("_ = a + b;", 1)]                  // discarded -> leaks
        [DataRow("var t = (a + b, b - a);", 1)]     // dropped tuple literal of temps -> leaks (once)
        [DataRow("var t = (a, b);", 0)]             // tuple of aliases -> owns nothing (R2)
        [DataRow("var g = new[] { a + b };", 1)]    // dropped array literal of a temp -> leaks (once)
        [DataRow("var g = new[] { a, b };", 0)]     // array of aliases -> owns nothing (R2)
        [DataRow("var t = a.size > 0 ? a + b : b - a;", 1)] // both-owning ternary dropped -> leaks
        [DataRow("var t = a.size > 0 ? a + b : a;", 0)]     // mixed ternary -> conservatively clean
        [DataRow("NDArray[] g = [a + b];", 1)]              // dropped collection expression -> leaks
        [DataRow("NDArray[] g = [a, b];", 0)]               // collection expression of aliases -> clean
        [DataRow("var t = a + b; ((System.IDisposable)t).Dispose();", 0)] // cast-dispose -> clean
        [DataRow("var t = a.size > 0 ? a + b : throw new System.Exception();", 1)] // produce-or-throw -> leaks
        public async Task Invariant_ReturnedOrDisposedNeverFlagged(string stmt, int expected)
        {
            var ret = stmt.StartsWith("return") ? "NDArray" : "void";
            var src = $"using NumSharp;\npublic static class T {{\n  public static {ret} M(NDArray a, NDArray b) {{ {stmt} }}\n}}";
            var r = await AnalyzerTestHarness.RunAsync(src, "invariant.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "invariant snippet must compile:\n" + src);
            Assert.AreEqual(expected, r.CountOf("NDW012"), $"'{stmt}' should produce {expected} NDW012");
        }
    }
}
