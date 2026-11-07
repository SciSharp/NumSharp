using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Operations;
using Microsoft.CodeAnalysis.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Build.Analyzer;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     The analyzers' generated-code test (<see cref="GeneratedCodeFilter"/>) against the one it replaces: the three
    ///     analyzers register with <c>GeneratedCodeAnalysisFlags.ReportDiagnostics</c>, which switches the driver's own
    ///     per-diagnostic test off, and drop a diagnostic themselves wherever the driver would have. These tests pin that
    ///     "wherever" is exactly the driver's answer: a probe analyzer that keeps the driver's test reports at every
    ///     marked identifier of a corpus covering every rule (generated file names, auto-generated headers, the
    ///     <c>generated_code</c> option both ways, <c>#line hidden</c>, <c>[GeneratedCode]</c> on every kind of declaration,
    ///     partial declarations, aliases, Unicode escapes), and the filter must call suppressed exactly the marks the
    ///     driver suppressed. End-to-end cases then show the three analyzers stay silent in generated code.
    /// </summary>
    [TestClass]
    public class GeneratedCodeFilterTests
    {
        /// <summary>A diagnostic analyzer that keeps the driver's generated-code test: it reports at every identifier spelled <c>MARK...</c>, and whatever survives is what the driver lets through.</summary>
        [DiagnosticAnalyzer(LanguageNames.CSharp)]
        private sealed class DriverProbe : DiagnosticAnalyzer
        {
            /// <summary>The probe's one diagnostic.</summary>
            public static readonly DiagnosticDescriptor Probe = new DiagnosticDescriptor(
                "PROBE01", "probe", "probe", "test", DiagnosticSeverity.Warning, isEnabledByDefault: true);

            /// <inheritdoc />
            public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics => ImmutableArray.Create(Probe);

            /// <inheritdoc />
            public override void Initialize(AnalysisContext context)
            {
                // The flags the NumSharp analyzers used before: the driver suppresses every diagnostic located in
                // generated code. A compilation action is not skipped for generated trees, so every mark is reported.
                context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
                context.RegisterCompilationAction(c =>
                {
                    foreach (var tree in c.Compilation.SyntaxTrees)
                        foreach (var token in tree.GetRoot().DescendantTokens())
                            if (token.IsKind(SyntaxKind.IdentifierToken) && token.ValueText.StartsWith("MARK", StringComparison.Ordinal))
                                c.ReportDiagnostic(Diagnostic.Create(Probe, token.GetLocation()));
                });
            }
        }

        /// <summary>Analyzer-config options per tree path: the channel of the <c>generated_code</c> option.</summary>
        private sealed class PerTreeOptions : AnalyzerConfigOptionsProvider
        {
            private readonly Dictionary<string, AnalyzerConfigOptions> _byPath;
            private static readonly AnalyzerConfigOptions Empty = new MapOptions(new Dictionary<string, string>());

            /// <summary>Maps each listed path to its options; every other tree gets none.</summary>
            /// <param name="byPath">Tree path → (key → value).</param>
            public PerTreeOptions(Dictionary<string, Dictionary<string, string>> byPath)
            {
                _byPath = byPath.ToDictionary(kv => kv.Key, kv => (AnalyzerConfigOptions)new MapOptions(kv.Value));
            }

            /// <inheritdoc />
            public override AnalyzerConfigOptions GlobalOptions => Empty;

            /// <inheritdoc />
            public override AnalyzerConfigOptions GetOptions(SyntaxTree tree) => _byPath.TryGetValue(tree.FilePath, out var o) ? o : Empty;

            /// <inheritdoc />
            public override AnalyzerConfigOptions GetOptions(AdditionalText textFile) => Empty;
        }

        /// <summary>One tree's options.</summary>
        private sealed class MapOptions : AnalyzerConfigOptions
        {
            private readonly Dictionary<string, string> _map;

            /// <summary>Wraps a key → value map.</summary>
            /// <param name="map">The options.</param>
            public MapOptions(Dictionary<string, string> map) { _map = map; }

            /// <inheritdoc />
            public override bool TryGetValue(string key, out string value) => _map.TryGetValue(key, out value);
        }

        /// <summary>The corpus: file path → source. Every rule the driver applies has marks on both sides of it.</summary>
        private static readonly (string Path, string Source)[] Corpus =
        {
            ("user.cs",
                "using System.CodeDom.Compiler;\n" +
                "using GC = System.CodeDom.Compiler.GeneratedCodeAttribute;\n" +
                "namespace P {\n" +
                "  class MARK_plain { int MARK_field; void MARK_method() { int MARK_local = 0; } }\n" +
                "  [GeneratedCode(\"t\", \"1\")] class MARK_genclass { int MARK_genclass_field; class MARK_nested_in_gen { } }\n" +
                "  partial class MARK_partial_gen { int MARK_partial_member_a; }\n" +
                "  [GeneratedCode(\"t\", \"1\")] partial class MARK_partial_gen { int MARK_partial_member_b; }\n" +
                "  class MARK_host {\n" +
                "    [GeneratedCode(\"t\", \"1\")] int MARK_gen_field1, MARK_gen_field2;\n" +
                "    [GeneratedCode(\"t\", \"1\")] int MARK_gen_prop { get; set; }\n" +
                "    [GeneratedCode(\"t\", \"1\")] void MARK_gen_method() { int MARK_in_gen_method = 0; }\n" +
                "    void MARK_host_method([GeneratedCode(\"t\", \"1\")] int MARK_gen_param) {\n" +
                "      [GeneratedCode(\"t\", \"1\")] void MARK_gen_local_fn() { int MARK_in_gen_local_fn = 0; }\n" +
                "      int MARK_after_local_fn = 0;\n" +
                "    }\n" +
                "    int MARK_accessor_host { [GeneratedCode(\"t\", \"1\")] get { int MARK_in_gen_getter = 0; return MARK_in_gen_getter; } }\n" +
                "  }\n" +
                "  [GC(\"t\", \"1\")] class MARK_alias_gen { int MARK_alias_member; }\n" +
                "  enum MARK_enum_host { MARK_enum_plain, [GeneratedCode(\"t\", \"1\")] MARK_enum_gen }\n" +
                "  [GeneratedCode(\"t\", \"1\")] delegate void MARK_gen_delegate(int MARK_gen_delegate_param);\n" +
                "  class MARK_host2 {\n" +
                "    [GeneratedCode(\"t\", \"1\")] event System.Action MARK_gen_event { add { int MARK_in_gen_add = 0; } remove { } }\n" +
                "    [GeneratedCode(\"t\", \"1\")] int this[int MARK_gen_indexer_param] => MARK_gen_indexer_param;\n" +
                "    [GeneratedCode(\"t\", \"1\")] int MARK_gen_arrow_prop => 0;\n" +
                "    [GeneratedCode(\"t\", \"1\")] MARK_host2() { int MARK_in_gen_ctor = 0; }\n" +
                "    [GeneratedCode(\"t\", \"1\")] public static MARK_host2 operator +(MARK_host2 a, MARK_host2 b) { int MARK_in_gen_op = 0; return a; }\n" +
                "    void MARK_lambda_host() { System.Action f = [GeneratedCode(\"t\", \"1\")] () => { int MARK_in_gen_lambda = 0; }; }\n" +
                "  }\n" +
                "  [GeneratedCode(\"t\", \"1\")] record MARK_gen_record(int MARK_gen_record_param);\n" +
                "}\n"),
            ("escape.cs", "[System.CodeDom.Compiler.Generated\\u0043ode(\"t\", \"1\")] class MARK_escaped { int MARK_escaped_member; }\n"),
            ("comment_only.cs", "// GeneratedCode is only mentioned here\nclass MARK_comment_only { }\n"),
            ("aliases.cs", "global using GGC = System.CodeDom.Compiler.GeneratedCodeAttribute;\n"),
            ("uses_alias.cs", "[GGC(\"t\", \"1\")] class MARK_global_alias_gen { }\n"),
            ("gen1.g.cs", "class MARK_g { }\n"),
            ("gen2.designer.cs", "class MARK_designer { }\n"),
            ("gen3.generated.cs", "class MARK_generated { }\n"),
            ("gen4.g.i.cs", "class MARK_gi { }\n"),
            ("TemporaryGeneratedFile_123.cs", "class MARK_temp { }\n"),
            ("upper.G.CS", "class MARK_upper { }\n"),
            ("file.g", "class MARK_no_cs_extension { }\n"),
            ("sub.g/plain_in_g_dir.cs", "class MARK_in_g_dir { }\n"),
            ("hdr1.cs", "// <auto-generated/>\nclass MARK_hdr1 { }\n"),
            ("hdr2.cs", "/* <autogenerated> */ class MARK_hdr2 { }\n"),
            ("hdr3.cs", "/// <auto-generated/>\nclass MARK_hdr3_doc_comment { }\n"),
            ("hdr4.cs", "#pragma warning disable\n// <auto-generated/>\nclass MARK_hdr4 { }\n"),
            ("hdr5.cs", "using System;\n// <auto-generated/>\nclass MARK_hdr5_not_leading { }\n"),
            ("hidden.cs",
                "class MARK_hidden_host {\n  void M() {\n#line hidden\n    int MARK_hidden_local = 0;\n#line default\n" +
                "    int MARK_visible_local = 0;\n  }\n}\n"),
            ("opt_true.cs", "class MARK_opt_true { }\n"),
            ("opt_false.g.cs", "class MARK_opt_false { }\n"),
            ("opt_bad.g.cs", "class MARK_opt_bad { }\n"),
        };

        /// <summary>The <c>generated_code</c> option per path (an unparseable value falls back to the file rules).</summary>
        private static Dictionary<string, Dictionary<string, string>> CorpusOptions() => new Dictionary<string, Dictionary<string, string>>
        {
            ["opt_true.cs"] = new Dictionary<string, string> { ["generated_code"] = "true" },
            ["opt_false.g.cs"] = new Dictionary<string, string> { ["generated_code"] = "false" },
            ["opt_bad.g.cs"] = new Dictionary<string, string> { ["generated_code"] = "maybe" },
        };

        /// <summary>
        ///     Differential: for every mark of the corpus, the filter's verdict equals the driver's suppression, and both
        ///     sides of every rule occur (so a filter that answered one constant would fail).
        /// </summary>
        [TestMethod]
        public async Task Filter_AgreesWithTheDriver_OnEveryMark()
        {
            var (suppressed, kept) = await AssertAgreement("GeneratedCorpus", Corpus, CorpusOptions());
            Assert.IsTrue(suppressed >= 25 && kept >= 15, $"the corpus must exercise both answers (suppressed {suppressed}, kept {kept})");
        }

        /// <summary>
        ///     An assembly-level <c>[GeneratedCode]</c> makes every declaration generated by its containing chain - but
        ///     only in a tree whose text names the attribute, because the driver looks for generated symbols only there.
        ///     The filter follows both halves.
        /// </summary>
        [TestMethod]
        public async Task Filter_AgreesWithTheDriver_OnAnAssemblyLevelAttribute()
        {
            var (suppressed, kept) = await AssertAgreement("AssemblyGenerated", new[]
            {
                ("asm_attr.cs", "[assembly: System.CodeDom.Compiler.GeneratedCode(\"t\", \"1\")]\nclass MARK_with_token { }\n"),
                ("no_token.cs", "class MARK_without_token { }\n"),
            }, new Dictionary<string, Dictionary<string, string>>());
            Assert.AreEqual(1, suppressed, "the declaration in the attribute's own tree is generated");
            Assert.AreEqual(1, kept, "a tree that never names the attribute is not searched");
        }

        /// <summary>
        ///     Compiles a corpus, runs <see cref="DriverProbe"/> (the driver's own test) and the filter over every mark,
        ///     and asserts they agree.
        /// </summary>
        /// <param name="name">The compilation's name.</param>
        /// <param name="files">The corpus: path and source per tree.</param>
        /// <param name="treeOptions">The analyzer-config options per tree path.</param>
        /// <returns>How many marks the driver suppressed and kept.</returns>
        /// <exception cref="AssertFailedException">The corpus does not compile, or the filter disagrees with the driver on a mark.</exception>
        private static async Task<(int Suppressed, int Kept)> AssertAgreement(string name, (string Path, string Source)[] files,
            Dictionary<string, Dictionary<string, string>> treeOptions)
        {
            var trees = files.Select(f => CSharpSyntaxTree.ParseText(SourceText.From(f.Source), path: f.Path)).ToArray();
            var comp = CSharpCompilation.Create(name, trees, AnalyzerTestHarness.References,
                new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
            var errors = comp.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error).ToList();
            Assert.AreEqual(0, errors.Count, "corpus must compile:\n" + string.Join("\n", errors));

            var options = new AnalyzerOptions(ImmutableArray<AdditionalText>.Empty, new PerTreeOptions(treeOptions));
            var reported = await comp.WithAnalyzers(ImmutableArray.Create<DiagnosticAnalyzer>(new DriverProbe()), options)
                .GetAnalyzerDiagnosticsAsync();
            var survivors = new HashSet<(string, int)>(reported.Where(d => d.Id == "PROBE01")
                .Select(d => (d.Location.SourceTree.FilePath, d.Location.SourceSpan.Start)));

            var filter = new GeneratedCodeFilter(comp);
            var mismatches = new List<string>();
            int suppressed = 0, kept = 0;
            foreach (var tree in comp.SyntaxTrees)
                foreach (var token in tree.GetRoot().DescendantTokens())
                {
                    if (!token.IsKind(SyntaxKind.IdentifierToken) || !token.ValueText.StartsWith("MARK", StringComparison.Ordinal))
                        continue;
                    var location = token.GetLocation();
                    bool driverSuppressed = !survivors.Contains((tree.FilePath, location.SourceSpan.Start));
                    bool filterSuppresses = filter.IsInGeneratedCode(location, options, CancellationToken.None);
                    if (driverSuppressed) suppressed++; else kept++;
                    if (driverSuppressed != filterSuppresses)
                        mismatches.Add($"{tree.FilePath} {token.ValueText}: driver {(driverSuppressed ? "suppresses" : "keeps")}, filter {(filterSuppresses ? "suppresses" : "keeps")}");
                }

            Assert.AreEqual(0, mismatches.Count, "filter disagrees with the driver:\n  " + string.Join("\n  ", mismatches));
            return (suppressed, kept);
        }

        /// <summary>The filter answers false for a location outside source and for <see cref="Location.None"/>, as the driver does.</summary>
        [TestMethod]
        public void Filter_LocationOutsideSource_IsNotGenerated()
        {
            var comp = CSharpCompilation.Create("Empty", Array.Empty<SyntaxTree>(), AnalyzerTestHarness.References);
            var filter = new GeneratedCodeFilter(comp);
            var options = new AnalyzerOptions(ImmutableArray<AdditionalText>.Empty);
            Assert.IsFalse(filter.IsInGeneratedCode(Location.None, options, CancellationToken.None));
            Assert.IsFalse(filter.IsInGeneratedCode(null, options, CancellationToken.None));
        }

        /// <summary>
        ///     NDW016 is reported at a type's FIRST declaration: when that is the generated part of a partial type, the
        ///     driver would have dropped it, and so does the filter; with the user part first, it is reported.
        /// </summary>
        [TestMethod]
        public async Task Ndw016_AtAGeneratedFirstPart_IsDropped_AtAUserFirstPart_IsReported()
        {
            const string generatedPart = "// <auto-generated/>\nusing NumSharp;\npublic partial class Hold { NDArray _x; }\n";
            const string userPart = "using NumSharp;\npublic partial class Hold { public NDArray Get() => _x; }\n";

            var generatedFirst = await AnalyzerTestHarness.RunProjectAsync(
                new[] { ("Hold.g.cs", generatedPart), ("Hold.cs", userPart) }, "gen_first", concurrentAnalysis: true);
            Assert.IsTrue(generatedFirst.CompileErrors.IsEmpty, string.Join("\n", generatedFirst.CompileErrors));
            Assert.AreEqual(0, generatedFirst.Ndw.Count(d => d.Id == "NDW016"), string.Join("\n", generatedFirst.Ndw));

            var userFirst = await AnalyzerTestHarness.RunProjectAsync(
                new[] { ("Hold.cs", userPart), ("Hold.g.cs", generatedPart) }, "user_first", concurrentAnalysis: true);
            Assert.AreEqual(1, userFirst.Ndw.Count(d => d.Id == "NDW016"), string.Join("\n", userFirst.Ndw));
        }

        /// <summary>NDW017 on a holding member declared in the generated part of a disposable partial type is dropped; one declared in the user part is reported.</summary>
        [TestMethod]
        public async Task Ndw017_OnAMemberOfAGeneratedPart_IsDropped()
        {
            var r = await AnalyzerTestHarness.RunProjectAsync(new[]
            {
                ("Res.cs", "using System;\nusing NumSharp;\npublic partial class Res : IDisposable {\n  NDArray _user;\n  public void Dispose() { }\n}\n"),
                ("Res.designer.cs", "using NumSharp;\npublic partial class Res {\n  NDArray _designer;\n}\n"),
            }, "ndw017_generated_part", concurrentAnalysis: true);
            Assert.IsTrue(r.CompileErrors.IsEmpty, string.Join("\n", r.CompileErrors));
            var ndw017 = r.Ndw.Where(d => d.Id == "NDW017").ToList();
            Assert.AreEqual(1, ndw017.Count, string.Join("\n", r.Ndw));
            Assert.AreEqual("Res.cs", ndw017[0].Location.SourceTree.FilePath);
        }

        /// <summary>
        ///     A leak in a <c>#line hidden</c> region is dropped; the same leak after <c>#line default</c> is reported. (The
        ///     file opens with <c>#line default</c>: in a tree with a hidden region, the lines before the first directive
        ///     are hidden too, and a method declared there would be skipped as generated code altogether.)
        /// </summary>
        [TestMethod]
        public async Task Ndw012_InAHiddenRegion_IsDropped()
        {
            var r = await AnalyzerTestHarness.RunAsync(
                "using NumSharp;\n#line default\npublic static class C {\n  public static void M(NDArray a) {\n#line hidden\n    var hidden = a + 1.0;\n#line default\n    var visible = a + 2.0;\n  }\n}\n",
                "hidden_leak.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, string.Join("\n", r.CompileErrors));
            CollectionAssert.AreEqual(new[] { 8 }, r.Ndw.Where(d => d.Id == "NDW012").Select(AnalyzerTestHarness.LineOf).ToArray());
        }

        /// <summary>
        ///     A <c>[GeneratedCode]</c> local function does not make its body generated code for the driver (its
        ///     generated-symbol set holds member-level declarations only), so a leak inside it is reported, as it was
        ///     before the analyzers took the test over.
        /// </summary>
        [TestMethod]
        public async Task Ndw012_InAGeneratedCodeLocalFunction_IsStillReported()
        {
            var r = await AnalyzerTestHarness.RunAsync(
                "using System.CodeDom.Compiler;\nusing NumSharp;\npublic static class C {\n  public static void M(NDArray a) {\n" +
                "    [GeneratedCode(\"t\", \"1\")] void Gen() { var g = a + 1.0; }\n    Gen();\n    var mine = a + 2.0;\n  }\n}\n",
                "generated_local_function.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, string.Join("\n", r.CompileErrors));
            CollectionAssert.AreEqual(new[] { 5, 7 }, r.Ndw.Where(d => d.Id == "NDW012").Select(AnalyzerTestHarness.LineOf).ToArray());
        }

        /// <summary>A leak inside a <c>[GeneratedCode]</c> method of a user type is never analyzed (the driver skips the method as generated code), with or without the filter.</summary>
        [TestMethod]
        public async Task Ndw012_InAGeneratedCodeMethod_IsNotReported()
        {
            var r = await AnalyzerTestHarness.RunAsync(
                "using System.CodeDom.Compiler;\nusing NumSharp;\npublic static class C {\n" +
                "  [GeneratedCode(\"t\", \"1\")] public static void Gen(NDArray a) { var g = a + 1.0; }\n" +
                "  public static void Mine(NDArray a) { var m = a + 2.0; }\n}\n",
                "generated_method.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, string.Join("\n", r.CompileErrors));
            CollectionAssert.AreEqual(new[] { 5 }, r.Ndw.Where(d => d.Id == "NDW012").Select(AnalyzerTestHarness.LineOf).ToArray());
        }

        /// <summary>
        ///     <c>generated_code = false</c> makes a <c>*.g.cs</c> file user code (analyzed and reported), and
        ///     <c>generated_code = true</c> makes a plain file generated code (neither analyzed nor reported).
        /// </summary>
        [TestMethod]
        public async Task GeneratedCodeOption_OverridesTheFileName_BothWays()
        {
            const string leak = "using NumSharp;\npublic static class {0} {{\n  public static void M(NDArray a) {{ var t = a + 1.0; }}\n}}\n";
            var trees = new[]
            {
                CSharpSyntaxTree.ParseText(SourceText.From(string.Format(leak, "MarkedUser")), path: "marked_user.g.cs"),
                CSharpSyntaxTree.ParseText(SourceText.From(string.Format(leak, "MarkedGenerated")), path: "marked_generated.cs"),
            };
            var comp = CSharpCompilation.Create("GeneratedOption", trees, AnalyzerTestHarness.References,
                new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
            var provider = new PerTreeOptions(new Dictionary<string, Dictionary<string, string>>
            {
                ["marked_user.g.cs"] = new Dictionary<string, string> { ["generated_code"] = "false" },
                ["marked_generated.cs"] = new Dictionary<string, string> { ["generated_code"] = "true" },
            });
            var all = await comp.WithAnalyzers(ImmutableArray.Create<DiagnosticAnalyzer>(
                    new NDArrayLeakAnalyzer(), new NDScopedTargetAnalyzer(), new NDArrayHolderAnalyzer()),
                new AnalyzerOptions(ImmutableArray<AdditionalText>.Empty, provider)).GetAnalyzerDiagnosticsAsync();
            var files = all.Where(d => d.Id == "NDW012").Select(d => d.Location.SourceTree.FilePath).ToList();
            CollectionAssert.AreEqual(new[] { "marked_user.g.cs" }, files);
        }
    }

    /// <summary>
    ///     The name index (<see cref="NumSharpNameIndex"/>) never skips work whose answer is not empty. An ORACLE runs over
    ///     every analyzer fixture and a corpus of index-specific shapes: a method body containing any operation or local
    ///     whose type holds NDArrays (judged by an ownership model WITHOUT the index) must pass
    ///     <see cref="NumSharpNameIndex.BodyMayInvolveNumSharp"/>; a type with a holding member must pass
    ///     <see cref="NumSharpNameIndex.SourceTypeMayHold"/>; a method that inherits a scope or an <c>[NDScopedExit]</c>
    ///     must pass <see cref="NumSharpNameIndex.MayInheritScope"/>. End-to-end cases then pin the routes the index
    ///     follows to reach a body that never names NDArray: aliases, <c>using static</c>, extension methods of a
    ///     referenced library, operators, events, primary constructors, generic constraints, partial and nested types.
    /// </summary>
    [TestClass]
    public class NameIndexSoundnessTests
    {
        /// <summary>The usings every snippet starts with.</summary>
        private const string H = "using System;\nusing System.Collections.Generic;\nusing System.Linq;\nusing System.Threading.Tasks;\nusing NumSharp;\n";

        /// <summary>Index-specific shapes for the oracle (each must compile on its own).</summary>
        private static readonly string[] Shapes =
        {
            "using Arr = NumSharp.NDArray;\npublic static class A { public static Arr Make() => np.zeros(3); }\npublic static class B { public static void M() { var t = A.Make(); } }",
            "using static NumSharp.np;\npublic static class C { public static void M() { var t = zeros(3); } }",
            "public sealed class V { public static NDArray operator +(V a, V b) => np.zeros(1); public static V Get() => new V(); }\npublic static class D { public static void M() { var t = V.Get() + V.Get(); } }",
            "public sealed class E { public event Func<NDArray> Made; public NDArray Fire() => Made?.Invoke(); }\npublic static class F { public static void M(E e) { var t = e.Fire(); } }",
            "public sealed class P(NDArray seed) { public NDArray Twice() => seed + seed; }\npublic static class G { public static void M(P p) { var t = p.Twice(); } }",
            "public static class K { public static void M<T>(T x) where T : NDArray { var t = x + 1.0; } }",
            "public partial class L { public NDArray Make() => np.ones(2); }\npublic partial class L { public void Use() { var t = Make(); } }",
            "public class Outer { public class Inner { public static NDArray Make() => np.ones(2); } }\npublic static class N { public static void M() { var t = Outer.Inner.Make(); } }",
            "public record R(NDArray Data);\npublic static class Q { public static void M() { var r = new R(np.ones(2)); } }",
            "public class Box<T> { public T Value; }\npublic class Holds { Box<NDArray> _box; }\npublic class Plain { Box<int> _box; }",
            "public class BaseH { protected NDArray _x; }\npublic class DerivedH : BaseH { }\npublic class Unrelated { int _y; }",
            "public interface IScoped { [NDScoped] NDArray Run(NDArray a); }\npublic sealed class Impl : IScoped { public NDArray Run(NDArray a) { var t = a + 1.0; return t * 2.0; } }",
            "public abstract class BaseS { [NDScoped] public virtual NDArray Run(NDArray a) => a; }\npublic sealed class DerS : BaseS { public override NDArray Run(NDArray a) { var t = a + 1.0; return t; } }",
            "public static class T9 { public static void M(NDArray a) { var t = (1, 2, 3, 4, 5, 6, 7, 8, a + 1.0); } }",
            "public static class Plain2 { public static int Count(List<int> xs) => xs.Count(); public static int Sum(int[] xs) => xs.Sum(); }",
        };

        /// <summary>
        ///     The oracle: over every fixture and every shape, no body with an NDArray-holding operation or local fails the
        ///     body gate, no type with a holding member fails the holder gate, and no method that inherits a scope (or an
        ///     exit parameter) fails the inheritance gate.
        /// </summary>
        [TestMethod]
        public void Index_NeverSkipsWhatTheFullAnalysisNeeds()
        {
            var sources = new List<(string Name, string Source)>();
            foreach (var path in Directory.GetFiles(Path.Combine(AppContext.BaseDirectory, "Fixtures"), "*.cs"))
                sources.Add((Path.GetFileName(path), File.ReadAllText(path)));
            for (int i = 0; i < Shapes.Length; i++)
                sources.Add(($"shape{i}.cs", H + Shapes[i]));

            var violations = new List<string>();
            int bodiesChecked = 0, typesChecked = 0;
            foreach (var (name, source) in sources)
            {
                var tree = CSharpSyntaxTree.ParseText(SourceText.From(source), path: name);
                var comp = CSharpCompilation.Create("Oracle_" + Path.GetFileNameWithoutExtension(name), new[] { tree },
                    AnalyzerTestHarness.References, new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, allowUnsafe: true));
                var known = KnownTypes.Resolve(comp);
                Assert.IsNotNull(known, name + ": NumSharp must resolve");
                var index = new NumSharpNameIndex(comp, known);
                var truth = new OwnershipModel(known, comp, names: null); // binds every type: the unpruned answers
                var semantic = comp.GetSemanticModel(tree);

                foreach (var method in SourceMethods(comp))
                {
                    if (BodyHoldsNDArrays(method, semantic, truth))
                    {
                        bodiesChecked++;
                        if (!index.BodyMayInvolveNumSharp(method))
                            violations.Add($"{name}: body of {method.ToDisplayString()} holds NDArrays but the index skips it");
                    }

                    if (!method.IsStatic && (ScopeInheritance.Inherited(method, known) != null || ScopeInheritance.InheritedExitSource(method, known) != null)
                        && !index.MayInheritScope(method))
                        violations.Add($"{name}: {method.ToDisplayString()} inherits a scope but the index says it cannot");
                }

                foreach (var symbol in comp.GetSymbolsWithName(_ => true, SymbolFilter.Type))
                {
                    if (!(symbol is INamedTypeSymbol type))
                        continue;
                    if (truth.HolderMembersOf(type).Count > 0)
                    {
                        typesChecked++;
                        if (!index.SourceTypeMayHold(type))
                            violations.Add($"{name}: {type.ToDisplayString()} has holding members but the index says it cannot hold");
                    }
                }
            }

            Assert.AreEqual(0, violations.Count, "unsound index:\n  " + string.Join("\n  ", violations));
            Assert.IsTrue(bodiesChecked > 100 && typesChecked > 20, $"the oracle must see real work (bodies {bodiesChecked}, types {typesChecked})");
        }

        /// <summary>Every explicitly declared method of the compilation's source, accessors of properties, indexers and events included (each owns an operation block of its own).</summary>
        /// <param name="comp">The compilation.</param>
        /// <returns>The methods, each once.</returns>
        private static IEnumerable<IMethodSymbol> SourceMethods(Compilation comp)
        {
            var seen = new HashSet<IMethodSymbol>(SymbolEqualityComparer.Default);
            foreach (var symbol in comp.GetSymbolsWithName(_ => true, SymbolFilter.Member))
            {
                IEnumerable<IMethodSymbol> methods = symbol switch
                {
                    IMethodSymbol m => new[] { m },
                    IPropertySymbol p => new[] { p.GetMethod, p.SetMethod },
                    IEventSymbol e => new[] { e.AddMethod, e.RemoveMethod },
                    _ => Array.Empty<IMethodSymbol>(),
                };
                foreach (var m in methods)
                    if (m != null && !m.IsImplicitlyDeclared && seen.Add(m))
                        yield return m;
            }
        }

        /// <summary>Whether a method's body contains an operation or declares a local whose type holds NDArrays (by the unpruned model).</summary>
        /// <param name="method">A source method.</param>
        /// <param name="semantic">The tree's semantic model.</param>
        /// <param name="truth">The ownership model without the index.</param>
        /// <returns>True when the body has an NDArray-holding value.</returns>
        private static bool BodyHoldsNDArrays(IMethodSymbol method, SemanticModel semantic, OwnershipModel truth)
        {
            foreach (var reference in method.DeclaringSyntaxReferences)
            {
                if (reference.SyntaxTree != semantic.SyntaxTree)
                    continue;
                var root = semantic.GetOperation(reference.GetSyntax());
                if (root == null)
                    continue;
                foreach (var op in root.DescendantsAndSelf())
                {
                    if (op.Type != null && truth.HoldsNDArrays(op.Type))
                        return true;
                    if (op is IVariableDeclaratorOperation d && truth.HoldsNDArrays(d.Symbol.Type))
                        return true;
                }
            }

            return false;
        }

        /// <summary>Compiles a snippet and returns the NDW012 lines (numbered from the snippet's first line, after the usings).</summary>
        /// <param name="body">The declarations.</param>
        /// <param name="name">The file name (without extension).</param>
        /// <param name="references">Extra metadata references (a referenced library).</param>
        /// <returns>The sorted NDW012 lines.</returns>
        private static async Task<List<int>> LeakLines(string body, string name, ImmutableArray<MetadataReference> references = default)
        {
            var r = await AnalyzerTestHarness.RunAsync(H + body, name + ".cs",
                references.IsDefault ? ImmutableArray<MetadataReference>.Empty : references, null);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile:\n" + string.Join("\n", r.CompileErrors));
            return r.Ndw.Where(d => d.Id == "NDW012").Select(d => AnalyzerTestHarness.LineOf(d) - 5).OrderBy(l => l).ToList();
        }

        /// <summary>A body that reaches NDArray only through a member typed by an alias of it is analyzed (the alias, then the member, join the relevant names).</summary>
        [TestMethod]
        public async Task LeakThroughAnAlias_IsReported()
        {
            var lines = await LeakLines("using Arr = NumSharp.NDArray;\n" +
                                        "public static class A { public static Arr Make() => np.zeros(3); }\n" +
                                        "public static class B {\n  public static void M() { var t = A.Make() + 1.0; }\n}", "alias_route");
            CollectionAssert.AreEqual(new[] { 4 }, lines);
        }

        /// <summary>A body that calls a statically imported NumSharp function by its bare name is analyzed.</summary>
        [TestMethod]
        public async Task LeakThroughUsingStatic_IsReported()
        {
            var r = await AnalyzerTestHarness.RunAsync(
                "using static NumSharp.np;\npublic static class C {\n  public static void M() { var t = zeros(3); }\n}\n", "using_static_route.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, string.Join("\n", r.CompileErrors));
            CollectionAssert.AreEqual(new[] { 3 }, r.Ndw.Where(d => d.Id == "NDW012").Select(AnalyzerTestHarness.LineOf).ToArray());
        }

        /// <summary>
        ///     A referenced library's extension method returning an NDArray, called on a value of a type that is not
        ///     NumSharp's (so the body names nothing of NumSharp), is a route the index seeds; one returning an int is not
        ///     needed, and a body using only it draws nothing.
        /// </summary>
        [TestMethod]
        public async Task LeakThroughALibraryExtension_IsReported()
        {
            var library = AnalyzerTestHarness.CompileToReference(
                "using System;\nusing NumSharp;\npublic static class Ext {\n  public static NDArray ToArr(this int[] xs) => np.array(xs);\n" +
                "  public static int Count(this ReadOnlySpan<char> s) => s.Length;\n}\n", "ExtLib");
            var lines = await LeakLines("public static class C {\n  public static void M() { var t = new[] { 1, 2 }.ToArr(); }\n" +
                                        "  public static int N(string s) => s.AsSpan().Count();\n}", "extension_route", ImmutableArray.Create(library));
            CollectionAssert.AreEqual(new[] { 2 }, lines);
        }

        /// <summary>A user-defined operator of a type that is not an NDArray, returning one, produces an owned value in a body that only names the operand type.</summary>
        [TestMethod]
        public async Task LeakThroughAUserOperator_IsReported()
        {
            var lines = await LeakLines("public sealed class V { public static NDArray operator +(V a, V b) => np.zeros(1); public static V Get() => new V(); }\n" +
                                        "public static class D {\n  public static void M() { var t = V.Get() + V.Get(); }\n}", "operator_route");
            CollectionAssert.AreEqual(new[] { 3 }, lines);
        }

        /// <summary>A primary-constructor parameter of type NDArray makes the whole type relevant: a body using it bare is analyzed.</summary>
        [TestMethod]
        public async Task LeakThroughAPrimaryConstructorParameter_IsReported()
        {
            var lines = await LeakLines("public sealed class P(NDArray seed) {\n  public void M() { var t = seed + 1.0; }\n}", "primary_ctor_route");
            CollectionAssert.AreEqual(new[] { 2 }, lines);
        }

        /// <summary>A member declared in another part of a partial type is reachable unqualified: the body is analyzed.</summary>
        [TestMethod]
        public async Task LeakThroughAPartialTypesOtherPart_IsReported()
        {
            var lines = await LeakLines("public partial class L { public NDArray Make() => np.ones(2); }\n" +
                                        "public partial class L {\n  public void Use() { var t = Make() + 1.0; }\n}", "partial_route");
            CollectionAssert.AreEqual(new[] { 3 }, lines);
        }

        /// <summary>A tuple of nine elements whose NINTH is a fresh NDArray is owned (its rest tuple is walked).</summary>
        [TestMethod]
        public async Task NineElementTupleWithAnNDArrayLast_IsOwned()
        {
            var lines = await LeakLines("public static class T9 {\n  public static void M(NDArray a) { var t = (1, 2, 3, 4, 5, 6, 7, 8, a + 1.0); }\n}", "tuple9");
            CollectionAssert.AreEqual(new[] { 2 }, lines);
        }

        /// <summary>A body of NDArray-free code in a project that references NumSharp draws nothing, even when it calls LINQ methods that share a name with NumSharp's.</summary>
        [TestMethod]
        public async Task NDArrayFreeBodies_DrawNothing()
        {
            var lines = await LeakLines("public static class Plain {\n  public static int Count(List<int> xs) => xs.Count();\n  public static int Sum(int[] xs) => xs.Sum();\n}", "plain");
            Assert.AreEqual(0, lines.Count);
        }
    }
}
