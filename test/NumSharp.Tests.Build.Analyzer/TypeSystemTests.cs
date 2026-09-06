using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     COVERAGE_PLAN §5 — type-system &amp; resolution robustness: the analyzer keys off SYMBOLS, not
    ///     spellings, so a subclass, a consumer's own carrier struct, a nullable annotation, a using
    ///     alias, a fully-qualified name and a global using all resolve; and with NumSharp not referenced
    ///     at all it no-ops.
    /// </summary>
    [TestClass]
    public class TypeSystemTests
    {
        private static async Task<int> Leaks(string src)
        {
            var r = await AnalyzerTestHarness.RunAsync(src, "types.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "type-system snippet must compile:\n" + src);
            return r.CountOf("NDW012");
        }

        [TestMethod]
        public async Task TY1_SubclassOfNDArray_Leaks()
        {
            Assert.AreEqual(1, await Leaks(
                "using NumSharp;\npublic static class T {\n" +
                "  class MyND : NDArray { public MyND() : base(typeof(double), new Shape(1)) {} }\n" +
                "  public static void M() { var t = new MyND(); }\n}"));
        }

        [TestMethod]
        public async Task TY2_ConsumerCarrierStruct_Leaks()
        {
            Assert.AreEqual(1, await Leaks(
                "using NumSharp;\n" +
                "public readonly struct P : INDArrayCarrier {\n" +
                "  public readonly NDArray A;\n" +
                "  public P(NDArray a) { A = a; }\n" +
                "  void INDArrayCarrier.YieldTo(NDScope scope) { scope.Returns(A); }\n" +
                "}\n" +
                "public static class T {\n" +
                "  static P Make(NDArray a) => new P(a + 1.0);\n" +
                "  public static void M(NDArray a) { var p = Make(a); }\n}"));
        }

        [TestMethod]
        public async Task TY3_NullableAnnotation_Leaks()
        {
            Assert.AreEqual(1, await Leaks(
                "#nullable enable\nusing NumSharp;\npublic static class T {\n" +
                "  public static void M(NDArray a, NDArray b) { NDArray? t = a + b; }\n}"));
        }

        [TestMethod]
        public async Task TY4_UsingAlias_Leaks()
        {
            Assert.AreEqual(1, await Leaks(
                "using ND = NumSharp.NDArray;\npublic static class T {\n" +
                "  public static void M(ND a, ND b) { var t = a + b; }\n}"));
        }

        [TestMethod]
        public async Task TY5_FullyQualified_LeaksOnArgument()
        {
            Assert.AreEqual(1, await Leaks(
                "public static class T {\n" +
                "  public static NumSharp.NDArray M(NumSharp.NDArray a, NumSharp.NDArray b) { return NumSharp.np.sum(a * b); }\n}"));
        }

        [TestMethod]
        public async Task TY6_NumSharpNotReferenced_NoOps()
        {
            var r = await AnalyzerTestHarness.RunWithoutNumSharpAsync(
                "public class NDArray { }\npublic static class T {\n" +
                "  public static void M() { var t = new NDArray(); }\n}", "notref.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "the NumSharp-free snippet must compile");
            Assert.AreEqual(0, r.Ndw.Length, "with NumSharp unreferenced the analyzer must produce no NDW");
        }

        [TestMethod]
        public async Task TY7_GlobalUsing_Leaks()
        {
            Assert.AreEqual(1, await Leaks(
                "global using NumSharp;\npublic static class T {\n" +
                "  public static void M(NDArray a, NDArray b) { var t = a + b; }\n}"));
        }
    }
}
