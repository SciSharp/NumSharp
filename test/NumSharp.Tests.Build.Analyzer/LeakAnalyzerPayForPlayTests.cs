using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     NDW012's pay-for-play path: a method body that holds no operation of an owned type is skipped
    ///     after one walk, and the "is this type owned" answer is memoized per type for the compilation. Both
    ///     are exact, not heuristics, and these tests pin that they drop no report: a leak whose only owned
    ///     value sits in a lambda, a local function, an async lambda, a tuple, an owning disposable, a generic
    ///     method over a constrained type parameter, or in any kind of member body (accessors, operators,
    ///     conversions, event accessors, finalizers, static and instance constructors, indexers) is still
    ///     reported; owned shapes are recognized however often a type recurs; a single leak among hundreds of
    ///     NDArray-free methods is found; and a concurrent multi-file run over shared types reports exactly what
    ///     a sequential run does. Every expectation is master's output for the same source (the early exit and
    ///     the memo changed no report).
    /// </summary>
    [TestClass]
    public class LeakAnalyzerPayForPlayTests
    {
        /// <summary>The usings every snippet starts with: five lines, so body line k is file line k + 5.</summary>
        private const string H = "using System;\nusing System.Collections.Generic;\nusing System.Linq;\nusing System.Threading.Tasks;\nusing NumSharp;\n";

        /// <summary>The number of lines <see cref="H"/> takes.</summary>
        private const int HeaderLines = 5;

        /// <summary>Compiles a snippet with every analyzer and returns the body-relative lines that draw NDW012, sorted.</summary>
        /// <param name="body">The declarations, compiled after <see cref="H"/>.</param>
        /// <param name="name">The file name (without extension).</param>
        /// <returns>The NDW012 lines, numbered from the body's first line.</returns>
        /// <exception cref="AssertFailedException">The snippet does not compile, or an analyzer threw.</exception>
        private static async Task<List<int>> LeakLines(string body, string name)
        {
            var r = await AnalyzerTestHarness.RunAsync(H + body, name + ".cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile:\n" + string.Join("\n", r.CompileErrors));
            return r.Ndw.Where(d => d.Id == "NDW012").Select(d => AnalyzerTestHarness.LineOf(d) - HeaderLines).OrderBy(l => l).ToList();
        }

        /// <summary>A leak inside a lambda is the method's only owned value: the body is not skipped, and the leak is reported on its line.</summary>
        [TestMethod]
        public async Task LeakInsideALambda_IsStillReported()
        {
            var lines = await LeakLines("public static class C {\n  public static void M(NDArray a) {\n    Action f = () => { var t = a + 1.0; };\n    f();\n  }\n}", "lambda_leak");
            CollectionAssert.AreEqual(new[] { 3 }, lines);
        }

        /// <summary>A leak inside a local function is the method's only owned value, and is reported.</summary>
        [TestMethod]
        public async Task LeakInsideALocalFunction_IsStillReported()
        {
            var lines = await LeakLines("public static class C {\n  public static void M(NDArray a) {\n    void Inner() { var t = a + 1.0; }\n    Inner();\n  }\n}", "local_function_leak");
            CollectionAssert.AreEqual(new[] { 3 }, lines);
        }

        /// <summary>A leak inside an async lambda handed to Task.Run is reported.</summary>
        [TestMethod]
        public async Task LeakInsideAnAsyncLambda_IsStillReported()
        {
            var lines = await LeakLines("public static class C {\n  public static async Task M(NDArray a) {\n    await Task.Run(() => { var t = a + 1.0; });\n  }\n}", "async_lambda_leak");
            CollectionAssert.AreEqual(new[] { 3 }, lines);
        }

        /// <summary>The body's only owned value is a tuple whose element is a fresh NDArray: the tuple type counts as owned, and the dead local is reported.</summary>
        [TestMethod]
        public async Task OnlyOwnedValueIsATuple_IsStillReported()
        {
            var lines = await LeakLines("public static class C {\n  public static void M(NDArray a) {\n    var t = (a + 1.0, 1);\n  }\n}", "tuple_leak");
            CollectionAssert.AreEqual(new[] { 3 }, lines);
        }

        /// <summary>The body's only owned value is an NDArray-owning disposable (its constructor argument is handed to it): the dropped instance is reported.</summary>
        [TestMethod]
        public async Task OnlyOwnedValueIsAnOwningDisposable_IsStillReported()
        {
            var lines = await LeakLines(
                "public sealed class Holder : IDisposable { NDArray _x; public Holder(NDArray x) { _x = x; } public void Dispose() => _x?.Dispose(); }\n" +
                "public static class C {\n  public static void M(NDArray a) {\n    new Holder(a + 1.0);\n  }\n}", "holder_leak");
            CollectionAssert.AreEqual(new[] { 4 }, lines);
        }

        /// <summary>A generic method over <c>T : NDArray</c>: the arithmetic resolves to NDArray's operator, the result is owned, and the dead local is reported.</summary>
        [TestMethod]
        public async Task GenericMethodOverAConstrainedType_IsStillReported()
        {
            var lines = await LeakLines("public static class C {\n  public static void M<T>(T a) where T : NDArray {\n    var t = a + 1.0;\n  }\n}", "generic_method_leak");
            CollectionAssert.AreEqual(new[] { 3 }, lines);
        }

        /// <summary>
        ///     Every kind of member body is analyzed: a property getter, an operator, a conversion operator, an
        ///     event accessor, a finalizer, a static constructor, an indexer getter and an instance constructor
        ///     each leak one temporary, and each leak is reported.
        /// </summary>
        [TestMethod]
        public async Task EveryKindOfMemberBody_IsAnalyzed()
        {
            var lines = await LeakLines(
                "public class S {\n" +                                  // 1
                "  NDArray _a;\n" +                                     // 2
                "  public NDArray P { get {\n" +                       // 3
                "    var t = _a + 1.0;\n" +                             // 4 getter
                "    return _a; } }\n" +                                // 5
                "  public static S operator +(S x, S y) {\n" +          // 6
                "    var t = x._a + y._a;\n" +                          // 7 operator
                "    return x; }\n" +                                   // 8
                "  public static explicit operator int(S s) {\n" +      // 9
                "    var t = s._a + 1.0;\n" +                           // 10 conversion
                "    return 0; }\n" +                                   // 11
                "  public event Action E { add {\n" +                   // 12
                "    var t = _a + 1.0; } remove { } }\n" +              // 13 event accessor
                "  ~S() {\n" +                                          // 14
                "    var t = _a + 1.0; }\n" +                           // 15 finalizer
                "  static S() {\n" +                                    // 16
                "    var t = np.arange(3) + 1.0; }\n" +                 // 17 static constructor
                "  public NDArray this[int i] { get {\n" +              // 18
                "    var t = _a + i;\n" +                               // 19 indexer getter
                "    return _a; } }\n" +                                // 20
                "  public S() {\n" +                                    // 21
                "    var t = np.arange(3) + 1.0; }\n" +                 // 22 instance constructor
                "}", "member_kinds");
            CollectionAssert.AreEqual(new[] { 4, 7, 10, 13, 15, 17, 19, 22 }, lines);
        }

        /// <summary>
        ///     Mentions of NDArray that produce nothing — <c>nameof</c>, <c>typeof</c>, a <c>default</c> NDArray,
        ///     <c>nameof</c> of a parameter — draw nothing (the body has an operation of NDArray type, the default
        ///     literal, so it is not skipped; there is simply nothing to reclaim).
        /// </summary>
        [TestMethod]
        public async Task NameofTypeofAndDefault_DrawNothing()
        {
            var lines = await LeakLines("public static class C {\n  public static void M(NDArray a) {\n    var n = nameof(NDArray);\n    var t = typeof(NDArray);\n    NDArray d = default;\n    var s = nameof(a);\n  }\n}", "nameof_typeof_default");
            Assert.AreEqual(0, lines.Count);
        }

        /// <summary>
        ///     Owned shapes recognized through the memo: an NDArray array of a fresh element, a nested tuple, an
        ///     owning disposable, an array of owning disposables and a typed NDArray view are reported; a list
        ///     that receives the element, a scalar tuple and a delegate returning the element are not.
        /// </summary>
        [TestMethod]
        public async Task OwnedShapes_AreRecognized_LookalikesAreNot()
        {
            var lines = await LeakLines(
                "public sealed class Holder : IDisposable { NDArray _x; public Holder(NDArray x) { _x = x; } public void Dispose() => _x?.Dispose(); }\n" + // 1
                "public static class C {\n" +                                   // 2
                "  public static void M(NDArray a) {\n" +                       // 3
                "    var arr = new NDArray[] { a + 1.0 };\n" +                  // 4 owned array
                "    var nested = ((a + 1.0, 1), \"x\");\n" +                   // 5 nested tuple
                "    var h = new Holder(a + 1.0);\n" +                           // 6 owning disposable
                "    var list = new List<NDArray> { a + 1.0 };\n" +             // 7 handed to Add: not owned here
                "    var pair = (1, 2);\n" +                                     // 8 no NDArray
                "    Func<NDArray> f = () => a + 1.0;\n" +                      // 9 returned by the lambda
                "    var holders = new[] { new Holder(a) };\n" +                 // 10 array of owning disposables
                "    var g = np.arange(3).MakeGeneric<double>();\n" +           // 11 typed NDArray
                "  }\n" +
                "}", "owned_shapes");
            CollectionAssert.AreEqual(new[] { 4, 5, 6, 10, 11 }, lines);
        }

        /// <summary>
        ///     One leak among 400 NDArray-free methods (LINQ, strings, dictionaries — the bodies the early exit
        ///     skips) and 100 methods that return their NDArray: exactly the one leak is reported, on its line.
        /// </summary>
        [TestMethod]
        public async Task OneLeakAmongHundredsOfMethods_IsTheOnlyReport()
        {
            var sb = new StringBuilder("public static class C {\n");
            int line = 1;
            int leakLine = -1;
            for (int i = 0; i < 500; i++)
            {
                if (i == 321)
                {
                    sb.Append("  public static void Leak(NDArray a) { var t = a + 1.0; }\n");
                    leakLine = ++line;
                }
                if (i % 5 == 0)
                    sb.Append("  public static NDArray Keep").Append(i).Append("(NDArray a) { var t = a + 1.0; return t; }\n");
                else
                    sb.Append("  public static string Free").Append(i).Append("(List<int> xs, int k) => string.Join(\",\", xs.Where(x => x > k).OrderBy(x => x).Select(x => x.ToString()));\n");
                line++;
            }
            sb.Append("}\n");
            var lines = await LeakLines(sb.ToString(), "one_leak_among_many");
            CollectionAssert.AreEqual(new[] { leakLine }, lines);
        }

        /// <summary>
        ///     The same owned types in 150 methods — a third leak, a third return, a third use <c>using</c>:
        ///     every leak is reported and nothing else, so the memoized answer is the same in every body.
        /// </summary>
        [TestMethod]
        public async Task SameOwnedTypesInManyMethods_EveryLeakIsReported()
        {
            var sb = new StringBuilder("public static class C {\n");
            var expected = new List<int>();
            for (int i = 0; i < 150; i++)
            {
                int line = i + 2;
                switch (i % 3)
                {
                    case 0:
                        sb.Append("  public static void L").Append(i).Append("(NDArray a) { var t = (a + 1.0, a * 2.0); }\n");
                        expected.Add(line);
                        break;
                    case 1:
                        sb.Append("  public static (NDArray, NDArray) R").Append(i).Append("(NDArray a) { var t = (a + 1.0, a * 2.0); return t; }\n");
                        break;
                    default:
                        sb.Append("  public static void U").Append(i).Append("(NDArray a) { using var t = a + 1.0; }\n");
                        break;
                }
            }
            sb.Append("}\n");
            var lines = await LeakLines(sb.ToString(), "same_types_many_methods");
            CollectionAssert.AreEqual(expected, lines);
        }

        /// <summary>
        ///     A 30-file project whose methods leak, return and scope the same owned types (NDArray, a tuple,
        ///     an owning disposable) with NDArray-free code between them, analyzed sequentially and then four
        ///     times concurrently: the shared memo is filled from many threads at once, and every run reports
        ///     exactly the sequential run's leaks.
        /// </summary>
        [TestMethod]
        [Timeout(300000)]
        public async Task ConcurrentFilesOverSharedTypes_ReportExactlyTheSequentialLeaks()
        {
            var files = new List<(string Path, string Source)>
            {
                ("holder.cs", H + "public sealed class Holder : IDisposable { NDArray _x; public Holder(NDArray x) { _x = x; } public void Dispose() => _x?.Dispose(); }\n"),
            };
            int expectedLeaks = 0;
            for (int f = 0; f < 30; f++)
            {
                var sb = new StringBuilder(H).Append("public static class C").Append(f).Append(" {\n");
                for (int m = 0; m < 12; m++)
                {
                    switch ((f + m) % 6)
                    {
                        case 0: sb.Append("  public static void A").Append(m).Append("(NDArray a) { var t = a + 1.0; }\n"); expectedLeaks++; break;
                        case 1: sb.Append("  public static void B").Append(m).Append("(NDArray a) { var t = (a + 1.0, 1); }\n"); expectedLeaks++; break;
                        case 2: sb.Append("  public static void D").Append(m).Append("(NDArray a) { new Holder(a + 1.0); }\n"); expectedLeaks++; break;
                        case 3: sb.Append("  public static NDArray E").Append(m).Append("(NDArray a) { var t = a + 1.0; return t; }\n"); break;
                        case 4: sb.Append("  public static void F").Append(m).Append("(NDArray a) { using var h = new Holder(a + 1.0); }\n"); break;
                        default: sb.Append("  public static int G").Append(m).Append("(List<int> xs) => xs.Where(x => x % 2 == 0).Sum();\n"); break;
                    }
                }
                sb.Append("}\n");
                files.Add(($"c{f:D2}.cs", sb.ToString()));
            }

            var sequential = await AnalyzerTestHarness.RunProjectAsync(files, "leak_project", concurrentAnalysis: false);
            Assert.IsTrue(sequential.CompileErrors.IsEmpty, string.Join("\n", sequential.CompileErrors.Take(10)));
            Assert.AreEqual(expectedLeaks, sequential.CountOf("NDW012"), "each leaking method draws one NDW012");
            var baseline = sequential.Ndw.Select(d => $"{d.Location.SourceTree?.FilePath}:{AnalyzerTestHarness.LineOf(d)}:{d.Id}").ToList();
            for (int run = 0; run < 4; run++)
            {
                var concurrent = await AnalyzerTestHarness.RunProjectAsync(files, "leak_project", concurrentAnalysis: true);
                CollectionAssert.AreEqual(baseline, concurrent.Ndw.Select(d => $"{d.Location.SourceTree?.FilePath}:{AnalyzerTestHarness.LineOf(d)}:{d.Id}").ToList(), $"concurrent run {run}");
            }
        }
    }
}
