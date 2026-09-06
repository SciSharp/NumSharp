using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     Declaration-site INHERITANCE of the scope attributes, as both analyzers see it: an override or
    ///     implementation of a <c>[NDScoped]</c>/<c>[NDScopedAsync]</c>/<c>[NDScopedCovered]</c> virtual,
    ///     abstract or interface member inherits the attribute (the weaver's <c>ScopeInheritance</c>
    ///     rule) — so the NDW012 leak pass exempts it, the target gate reads its declaration's attribute,
    ///     an abstract declaration is a contract rather than NDW005, and every inheriting override /
    ///     implementation — a shape no metadata text scan can see, whether the declaration lives in
    ///     THIS assembly or another — draws NDW013 from the analyzer when the build says the weaver is
    ///     inactive (the analyzer is the primary NDW013 reporter; see <see cref="Ndw013AnalyzerTests"/>).
    /// </summary>
    [TestClass]
    public class InheritanceAnalyzerTests
    {
        [TestMethod]
        public Task InheritanceScenarios_MatchTagsExactly() => AnalyzerTestHarness.AssertExactAsync("InheritanceScenarios.cs");

        [TestMethod]
        public async Task InheritanceFixture_HasExactlyTheExpectedDiagnostics()
        {
            var r = await AnalyzerTestHarness.RunFileAsync("InheritanceScenarios.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "inheritance fixture must compile");
            Assert.AreEqual(5, r.CountOf("NDW012"), "exactly the five UNSCOPED controls leak (an unscoped virtual's override, an unscoped generic base member, the unscoped interface member implemented implicitly and explicitly, an unrelated same-named method)");
            Assert.AreEqual(2, r.CountOf("NDW002"), "the hidden egress is reported on the scoped declaration AND on the override that inherits it");
            Assert.AreEqual(0, r.CountOf("NDW005"), "an abstract declaration carrying the attribute is the contract, never a body-less target");
            Assert.AreEqual(0, r.CountOf("NDW013"), "no build property -> the weaver's presence is unknown -> NDW013 stays silent");
            Assert.AreEqual(7, r.Ndw.Length, "every inheriting override/implementation must add NO diagnostics");
        }

        [TestMethod]
        public async Task GateFixture_AbstractDeclaration_IsAContract_NotNdw005()
        {
            // The NDW005 rejection is now an EXTERN's alone; the abstract declaration right below it in the
            // gate fixture carries the attribute as the contract its overrides inherit and stays clean.
            var flagged = await FixtureFacts.FlaggedLineTexts("GateScenarios.cs", "NDW005");
            Assert.AreEqual(1, flagged.Count, "exactly one NDW005 in the gate fixture");
            FixtureFacts.AnyStartsWith(flagged, "public static extern", "the extern declaration");
            FixtureFacts.NoneContains(flagged, "AbstractContract", "the abstract [NDScoped] declaration");
        }

        // ---------------------------------------------------------------- the metamorphic delta

        [DataTestMethod]
        [DataRow("[NDScoped] ", 0)]
        [DataRow("[NDScopedAsync] ", 0)]
        [DataRow("[NDScopedCovered] ", 0)]
        [DataRow("", 1)]
        public async Task BaseAttribute_FlipsTheOverridesVerdict(string baseAttribute, int expectedLeaks)
        {
            // Identical override bodies; the ONLY variable is the attribute on the base declaration.
            // [NDScopedAsync] on a Task-returning virtual is the async model; the override stays non-async.
            bool isAsync = baseAttribute.Contains("Async");
            var ret = isAsync ? "System.Threading.Tasks.Task<NDArray>" : "NDArray";
            var body = isAsync ? "return System.Threading.Tasks.Task.FromResult(a.copy());" : "return a.copy();";
            var src = "using NumSharp;\n" +
                      "public abstract class B { " + baseAttribute + "public abstract " + ret + " M(NDArray a); }\n" +
                      "public class D : B { public override " + ret + " M(NDArray a) { var t = a + 1.0; " + body + " } }\n";
            var r = await AnalyzerTestHarness.RunAsync(src, "inherit_delta.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "metamorphic pair must compile");
            Assert.AreEqual(expectedLeaks, r.CountOf("NDW012"), $"base '{baseAttribute.Trim()}' -> override leaks: {expectedLeaks}");
            Assert.AreEqual(0, r.CountOf("NDW005"), "the abstract declaration is a contract");
        }

        [TestMethod]
        public async Task ExplicitAttributeOnTheOverride_Wins()
        {
            // The override's own [NDScopedCovered] opts out of the inherited [NDScoped]: still exempt
            // (covered), and — the weaver-facing half — it is no longer a weave target. An own [NDScoped]
            // on the override of an UNSCOPED base makes it a target on its own.
            var src = "using NumSharp;\n" +
                      "public abstract class B { [NDScoped] public abstract NDArray M(NDArray a); public virtual NDArray P(NDArray a) => a.copy(); }\n" +
                      "public class D : B {\n" +
                      "  [NDScopedCovered] public override NDArray M(NDArray a) { var t = a + 1.0; return a.copy(); }\n" +
                      "  [NDScoped] public override NDArray P(NDArray a) { var t = a + 1.0; return a.copy(); }\n" +
                      "}\n";
            var r = await AnalyzerTestHarness.RunAsync(src, "inherit_explicit.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "explicit-wins snippet must compile");
            Assert.AreEqual(0, r.Ndw.Length, "both overrides are exempt by their OWN attribute");
        }

        [TestMethod]
        public async Task InheritedGate_ReportsTheDeclarationAndTheOverride()
        {
            // A wrong-model declaration ([NDScoped] on a Task-returning virtual) is NDW009 on the
            // declaration, and — because the override inherits the same attribute and the weaver would
            // try to weave it under it — NDW009 on the override as well, labelled with its source.
            var src = "using NumSharp;\nusing System.Threading.Tasks;\n" +
                      "public class B { [NDScoped] public virtual Task<NDArray> M(NDArray a) => Task.FromResult(a.copy()); }\n" +
                      "public class D : B { public override Task<NDArray> M(NDArray a) => Task.FromResult(a.copy()); }\n";
            var r = await AnalyzerTestHarness.RunAsync(src, "inherit_gate.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "inherited-gate snippet must compile");
            Assert.AreEqual(2, r.CountOf("NDW009"), "the declaration and the inheriting override are both rejected");
            StringAssert.Contains(r.Ndw[1].GetMessage(), "inherited from", "the override's diagnostic names the declaration it inherits from");
        }

        // ---------------------------------------------------------------- across an assembly boundary

        private const string BaseLibrary =
            "using NumSharp;\nusing System.Threading.Tasks;\n" +
            "namespace Lib {\n" +
            "  public abstract class Base {\n" +
            "    [NDScoped] public abstract NDArray Compute(NDArray a);\n" +
            "    [NDScoped] public virtual NDArray Twice(NDArray a) => a * 2.0;\n" +
            "    [NDScopedAsync] public abstract Task<NDArray> ComputeAsync(NDArray a);\n" +
            "    [NDScoped] public abstract NDArray Value { get; }\n" +
            "    public virtual NDArray Plain(NDArray a) => a.copy();\n" +
            "  }\n" +
            "  public interface IOp { [NDScoped] NDArray Apply(NDArray a); }\n" +
            "  public abstract class GBase<T> { [NDScoped] public virtual NDArray Map(T x, NDArray a) => a.copy(); }\n" +
            "}\n";

        private const string Consumer =
            "using NumSharp;\nusing System.Threading.Tasks;\n" +
            "public class Derived : Lib.Base, Lib.IOp {\n" +
            "  private static NDArray _a;\n" +
            "  public override NDArray Compute(NDArray a) { var t = a + 1.0; return a.copy(); }\n" +
            "  public override NDArray Twice(NDArray a) { var t = a * 2.0; return a.copy(); }\n" +
            "  public override async Task<NDArray> ComputeAsync(NDArray a) { var t = a + 1.0; await Task.Yield(); return a.copy(); }\n" +
            "  public override NDArray Value { get { var t = _a + 1.0; return _a.copy(); } }\n" +
            "  public NDArray Apply(NDArray a) { var t = a + 1.0; return a.copy(); }\n" +
            "  public override NDArray Plain(NDArray a) { var t = a + 1.0; return a.copy(); }\n" +   // the control
            "}\n" +
            "public class Leaf : Lib.GBase<int> { public override NDArray Map(int x, NDArray a) { var t = a + x; return a.copy(); } }\n";

        private static ImmutableArray<MetadataReference> BaseRef()
            => ImmutableArray.Create(AnalyzerTestHarness.CompileToReference(BaseLibrary, "InheritBaseLib"));

        [TestMethod]
        public async Task CrossAssembly_OverridesInherit_TheReferencedDeclaration()
        {
            var r = await AnalyzerTestHarness.RunAsync(Consumer, "inherit_xasm.cs", BaseRef(), null);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "cross-assembly consumer must compile:\n" + string.Join("\n", r.CompileErrors));
            Assert.AreEqual(1, r.CountOf("NDW012"), "only the override of the UNSCOPED base member leaks");
            Assert.AreEqual(1, r.Ndw.Length, "the six inheriting members (abstract, virtual, async, property, interface, generic) add nothing");
            Assert.AreEqual(0, r.CountOf("NDW013"), "no build property -> the weaver's presence is unknown -> silent");
        }

        [TestMethod]
        public async Task CrossAssembly_Ndw013_WhenTheBuildSaysTheWeaverIsInactive()
        {
            var inactive = AnalyzerTestHarness.BuildProperties(("NumSharpBuildActive", "false"));
            var r = await AnalyzerTestHarness.RunAsync(Consumer, "inherit_xasm_inactive.cs", BaseRef(), inactive);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "cross-assembly consumer must compile");
            Assert.AreEqual(6, r.CountOf("NDW013"), "every member inheriting a scope attribute from the OTHER assembly is warned: no attribute text of its own, the MSBuild scan cannot see it");
            Assert.AreEqual(DiagnosticSeverity.Warning, r.Ndw.First(d => d.Id == "NDW013").Severity, "NDW013 is a warning, like its MSBuild twin");
            StringAssert.Contains(r.Ndw.First(d => d.Id == "NDW013").GetMessage(), "NumSharp.Build", "the fix names the weaver package");
        }

        [DataTestMethod]
        [DataRow("true", "", 0)]      // the weaver is installed and active
        [DataRow("false", "true", 0)] // the documented opt-out knob
        [DataRow("", "", 6)]          // declared but empty reads as inactive (NumSharp.targets defaults it to 'false' for exactly this reason)
        public async Task CrossAssembly_Ndw013_HonoursTheBuildKnobs(string active, string disable, int expected)
        {
            var options = AnalyzerTestHarness.BuildProperties(
                ("NumSharpBuildActive", active), ("NumSharpDisableWeaverMissingWarning", disable));
            var r = await AnalyzerTestHarness.RunAsync(Consumer, "inherit_xasm_knobs.cs", BaseRef(), options);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "cross-assembly consumer must compile");
            Assert.AreEqual(expected, r.CountOf("NDW013"), $"NumSharpBuildActive='{active}', NumSharpDisableWeaverMissingWarning='{disable}'");
        }

        [TestMethod]
        public async Task SameAssembly_Inheritance_DrawsNdw013_WhenTheBuildSaysTheWeaverIsInactive()
        {
            // The analyzer is the PRIMARY NDW013 reporter (the MSBuild text scan yields to it), so every
            // member the weaver would have woven is warned at its own declaration — an own attribute AND
            // an inherited one, same assembly included. Over InheritanceScenarios.cs that is exactly 13:
            //   own attribute, body, gate-clean ...... InheritanceBase.Virtual, InheritanceOptOut.ComputeAsync,
            //                                         GenericBase<T>.Map                                    (3)
            //   inherited, body, gate-clean .......... InheritanceDerived.{Compute, ComputeAsync, Value.get, Virtual},
            //                                         InheritanceOptOut.Value.get, LeafGeneric.Map,
            //                                         ImplicitImpl.{Apply, Current.get}, ExplicitImpl.{Apply, Current.get} (10)
            // and NOT: the abstract/interface declarations (contracts, no body), the [NDScopedCovered] members
            // and the override inheriting one (the opt-out), the unscoped controls, and RefEgress + its
            // override (NDW002 rejections draw their error alone).
            var inactive = AnalyzerTestHarness.BuildProperties(("NumSharpBuildActive", "false"));
            var r = await AnalyzerTestHarness.RunAsync(
                System.IO.File.ReadAllText(AnalyzerTestHarness.FixturePath("InheritanceScenarios.cs")),
                AnalyzerTestHarness.FixturePath("InheritanceScenarios.cs"), ImmutableArray<MetadataReference>.Empty, inactive);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "inheritance fixture must compile");
            Assert.AreEqual(13, r.CountOf("NDW013"), "every weavable member — own or inherited attribute — is warned once:\n  " +
                string.Join("\n  ", r.Ndw.Where(d => d.Id == "NDW013").Select(d => $"L{AnalyzerTestHarness.LineOf(d)}: {d.GetMessage()}")));
            Assert.AreEqual(3, r.Ndw.Count(d => d.Id == "NDW013" && d.GetMessage().Contains(" on '")), "three own-attribute reports");
            Assert.AreEqual(10, r.Ndw.Count(d => d.Id == "NDW013" && d.GetMessage().Contains(" inherits ")), "ten inherited-attribute reports");
            Assert.AreEqual(2, r.CountOf("NDW002"), "the rejections are unchanged by the inactive weaver");
        }
    }
}
