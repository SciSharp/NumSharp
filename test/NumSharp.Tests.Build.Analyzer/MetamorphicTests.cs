using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     COVERAGE_PLAN technique T3 — metamorphic pairs. Each pair is the SAME leaking body with one
    ///     minimal edit that should flip it clean (add <c>using</c>, add <c>[NDScoped]</c>, return the
    ///     result, hand it to a foreign sink, dispose it). Proving the DELTA catches an analyzer that
    ///     "warns for the wrong reason" — a change that flips the diagnostic for an unrelated cause.
    /// </summary>
    [TestClass]
    public class MetamorphicTests
    {
        private const string H = "using NumSharp;\npublic static class T {\n" +
                                 "  static void Sink(NDArray x) { }\n";
        private const string F = "\n}";

        private static async Task<int> Leaks(string body)
        {
            var r = await AnalyzerTestHarness.RunAsync(H + body + F, "meta.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "metamorphic snippet must compile:\n" + body);
            return r.CountOf("NDW012");
        }

        [TestMethod]
        public async Task Dispose_FlipsDeadLocalClean()
        {
            Assert.AreEqual(1, await Leaks("  public static void M(NDArray a, NDArray b) { var t = a + b; }"));
            Assert.AreEqual(0, await Leaks("  public static void M(NDArray a, NDArray b) { using var t = a + b; }"));
        }

        [TestMethod]
        public async Task ScopedAttribute_FlipsDeadLocalClean()
        {
            Assert.AreEqual(1, await Leaks("  public static void M(NDArray a, NDArray b) { var t = a + b; }"));
            Assert.AreEqual(0, await Leaks("  [NDScoped] public static void M(NDArray a, NDArray b) { var t = a + b; }"));
        }

        [TestMethod]
        public async Task Return_FlipsDiscardedResultClean()
        {
            Assert.AreEqual(1, await Leaks("  public static void M(NDArray a, NDArray b) { np.add(a, b); }"));
            Assert.AreEqual(0, await Leaks("  public static NDArray M(NDArray a, NDArray b) { return np.add(a, b); }"));
        }

        [TestMethod]
        public async Task ForeignSink_FlipsDiscardClean()
        {
            Assert.AreEqual(1, await Leaks("  public static void M(NDArray a, NDArray b) { _ = a + b; }"));
            Assert.AreEqual(0, await Leaks("  public static void M(NDArray a, NDArray b) { Sink(a + b); }"));
        }

        [TestMethod]
        public async Task ExplicitDispose_FlipsDeadLocalClean()
        {
            Assert.AreEqual(1, await Leaks("  public static void M(NDArray a, NDArray b) { var t = a + b; }"));
            Assert.AreEqual(0, await Leaks("  public static void M(NDArray a, NDArray b) { var t = a + b; t.Dispose(); }"));
        }
    }
}
