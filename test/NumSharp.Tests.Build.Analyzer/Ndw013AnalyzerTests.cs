using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     The analyzer half of NDW013 — the weaver-missing guard — as the PRIMARY reporter: when the
    ///     build has declared the weaver inactive (<c>build_property.NumSharpBuildActive</c> present and
    ///     not <c>true</c>, the state NumSharp's <c>build/NumSharp.targets</c> puts a consumer in when the
    ///     NumSharp.Build package is not installed), every member the weaver WOULD have woven draws one
    ///     NDW013 warning at its declaration: a member carrying <c>[NDScoped]</c>/<c>[NDScopedAsync]</c>
    ///     itself (method or property), an override/implementation inheriting one — from THIS assembly
    ///     too, not only from another — and a method with an own or inherited <c>[NDScopedExit]</c>
    ///     parameter. Abstract/interface declarations are contracts (nothing is woven there), a target the
    ///     gate rejects draws its error alone, <c>[NDScopedCovered]</c> is the opt-out, and the three knobs
    ///     (active / undeclared / disabled) silence it. The MSBuild text scan is now the fallback for a
    ///     compile the analyzer does not reach (<see cref="Ndw013BuildTests"/>).
    /// </summary>
    [TestClass]
    public class Ndw013AnalyzerTests
    {
        private static AnalyzerConfigOptionsProvider Inactive => AnalyzerTestHarness.BuildProperties(("NumSharpBuildActive", "false"));

        private static Task<AnalyzerResult> Run(string source, string name, AnalyzerConfigOptionsProvider options)
            => AnalyzerTestHarness.RunAsync(source, name, ImmutableArray<MetadataReference>.Empty, options);

        private const string DirectSync =
            "using NumSharp;\n" +
            "public static class S {\n" +
            "  [NDScoped] public static NDArray F(NDArray a) => a + 1.0;\n" +
            "}\n";

        // ---------------------------------------------------------------- own attribute

        [TestMethod]
        public async Task OwnAttribute_Sync_WarnsAtTheMethod_WhenTheWeaverIsInactive()
        {
            var r = await Run(DirectSync, "n13_direct.cs", Inactive);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile:\n" + string.Join("\n", r.CompileErrors));
            Assert.AreEqual(1, r.CountOf("NDW013"), "exactly one NDW013 — the one attributed method");
            var d = r.Ndw.Single(x => x.Id == "NDW013");
            Assert.AreEqual(3, AnalyzerTestHarness.LineOf(d), "reported AT the attributed method, not project-wide");
            Assert.AreEqual(DiagnosticSeverity.Warning, d.Severity, "a nudge, never a build error");
            StringAssert.Contains(d.GetMessage(), "[NDScoped] on 'S.F(NumSharp.NDArray)'", "names the attribute and the member");
            StringAssert.Contains(d.GetMessage(), "dotnet add package NumSharp.Build", "the fix names the weaver package");
        }

        [TestMethod]
        public async Task OwnAttribute_Async_Warns()
        {
            var src = "using NumSharp;\nusing System.Threading.Tasks;\n" +
                      "public static class S {\n" +
                      "  [NDScopedAsync] public static async Task<NDArray> G(NDArray a) { await Task.Yield(); return a + 1.0; }\n" +
                      "}\n";
            var r = await Run(src, "n13_async.cs", Inactive);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile");
            Assert.AreEqual(1, r.CountOf("NDW013"));
            StringAssert.Contains(r.Ndw.Single(x => x.Id == "NDW013").GetMessage(), "[NDScopedAsync] on 'S.G(NumSharp.NDArray)'");
        }

        [TestMethod]
        public async Task OwnAttribute_PropertyLevel_WarnsAtTheProperty_Once()
        {
            // A property-level attribute weaves the GETTER; the report sits on the property (where the
            // attribute is) and the getter symbol does not add a second one.
            var src = "using NumSharp;\n" +
                      "public class P {\n" +
                      "  private static NDArray _a;\n" +
                      "  [NDScoped] public NDArray V => _a + 1.0;\n" +
                      "}\n";
            var r = await Run(src, "n13_property.cs", Inactive);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile");
            Assert.AreEqual(1, r.CountOf("NDW013"), "one per member — the property, not property AND getter");
            var d = r.Ndw.Single(x => x.Id == "NDW013");
            Assert.AreEqual(4, AnalyzerTestHarness.LineOf(d));
            StringAssert.Contains(d.GetMessage(), "[NDScoped] on 'P.V'");
        }

        [DataTestMethod]
        [DataRow("true", "", 0)]      // the weaver is installed and active
        [DataRow("false", "true", 0)] // the documented opt-out knob
        [DataRow("false", "", 1)]     // declared inactive → warns
        [DataRow("", "", 1)]          // declared but EMPTY reads as inactive (NumSharp.targets defaults it to 'false' for exactly this reason)
        public async Task OwnAttribute_HonoursTheBuildKnobs(string active, string disable, int expected)
        {
            var options = AnalyzerTestHarness.BuildProperties(
                ("NumSharpBuildActive", active), ("NumSharpDisableWeaverMissingWarning", disable));
            var r = await Run(DirectSync, "n13_knobs.cs", options);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile");
            Assert.AreEqual(expected, r.CountOf("NDW013"), $"NumSharpBuildActive='{active}', NumSharpDisableWeaverMissingWarning='{disable}'");
        }

        [TestMethod]
        public async Task OwnAttribute_IsSilent_WhenTheBuildPropertyIsUndeclared()
        {
            // A source-mode build that imports neither targets file (a bare ProjectReference to
            // NumSharp.Core) declares nothing — the weaver's presence is UNKNOWN, and unknown stays silent.
            var r = await AnalyzerTestHarness.RunAsync(DirectSync, "n13_undeclared.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile");
            Assert.AreEqual(0, r.CountOf("NDW013"), "no build property -> unknown -> silent");
        }

        [TestMethod]
        public async Task AbstractAndInterfaceDeclarations_AreContracts_NotWarned()
        {
            // Nothing is woven at a body-less declaration; its inheritors are warned one by one instead.
            var src = "using NumSharp;\n" +
                      "public abstract class B { [NDScoped] public abstract NDArray M(NDArray a); [NDScoped] public abstract NDArray V { get; } }\n" +
                      "public interface I { [NDScoped] NDArray N(NDArray a); }\n";
            var r = await Run(src, "n13_contracts.cs", Inactive);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile");
            Assert.AreEqual(0, r.CountOf("NDW013"), "abstract / interface declarations carry the contract, nothing to weave there");
        }

        [TestMethod]
        public async Task Covered_IsNeverWarned()
        {
            // [NDScopedCovered] needs no weaver (inert at runtime, analyzer-only) — the opt-out.
            var src = "using NumSharp;\n" +
                      "public static class S { [NDScopedCovered] public static NDArray F(NDArray a) => a.copy(); }\n";
            var r = await Run(src, "n13_covered.cs", Inactive);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile");
            Assert.AreEqual(0, r.CountOf("NDW013"));
        }

        [TestMethod]
        public async Task RejectedTarget_DrawsItsErrorAlone()
        {
            // A target the gate refuses would not have been woven either way — the error owns the line.
            var src = "using System.Collections.Generic;\nusing NumSharp;\n" +
                      "public static class Bad { [NDScoped] public static List<NDArray> F(NDArray a) => new List<NDArray> { a + 1 }; }\n";
            var r = await Run(src, "n13_rejected.cs", Inactive);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile");
            Assert.AreEqual(1, r.CountOf("NDW003"), "the unsupported carrier is rejected");
            Assert.AreEqual(0, r.CountOf("NDW013"), "no weaver-missing nudge on top of a rejection");
        }

        // ---------------------------------------------------------------- inherited attribute

        [TestMethod]
        public async Task SameAssembly_Inheritor_Warns_WithProvenance()
        {
            // The override spells no attribute of its own, so no metadata text scan can see it — this is
            // the analyzer's to report, whether the declaration lives in this assembly or another.
            var src = "using NumSharp;\n" +
                      "public abstract class B { [NDScoped] public abstract NDArray M(NDArray a); }\n" +
                      "public class D : B { public override NDArray M(NDArray a) => a + 1.0; }\n";
            var r = await Run(src, "n13_inherit.cs", Inactive);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile");
            Assert.AreEqual(1, r.CountOf("NDW013"), "the abstract declaration is a contract; only the override is warned");
            var d = r.Ndw.Single(x => x.Id == "NDW013");
            Assert.AreEqual(3, AnalyzerTestHarness.LineOf(d), "at the override");
            StringAssert.Contains(d.GetMessage(), "inherits [NDScoped] from 'B.M(NumSharp.NDArray)'", "names the declaration it inherits from");
            StringAssert.Contains(d.GetMessage(), "[NDScopedCovered]", "the opt-out is offered");
        }

        [TestMethod]
        public async Task Inheritor_WithOwnCovered_IsNotWarned()
        {
            var src = "using NumSharp;\n" +
                      "public abstract class B { [NDScoped] public abstract NDArray M(NDArray a); }\n" +
                      "public class D : B { [NDScopedCovered] public override NDArray M(NDArray a) => a.copy(); }\n";
            var r = await Run(src, "n13_inherit_covered.cs", Inactive);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile");
            Assert.AreEqual(0, r.CountOf("NDW013"), "[NDScopedCovered] on the override opts out of the inherited weave");
        }

        // ---------------------------------------------------------------- [NDScopedExit]

        [TestMethod]
        public async Task ExitParameter_Own_Warns()
        {
            var src = "using NumSharp;\n" +
                      "public class H : System.IDisposable {\n" +
                      "  private NDArray _w;\n" +
                      "  public void Adopt([NDScopedExit] NDArray w) => _w = w;\n" +
                      "  public void Dispose() => _w?.Dispose();\n" +
                      "}\n";
            var r = await Run(src, "n13_exit.cs", Inactive);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile");
            Assert.AreEqual(1, r.CountOf("NDW013"));
            var d = r.Ndw.Single(x => x.Id == "NDW013");
            Assert.AreEqual(4, AnalyzerTestHarness.LineOf(d), "at the method whose parameter carries the attribute");
            StringAssert.Contains(d.GetMessage(), "[NDScopedExit] on parameter 'w' of 'H.Adopt(NumSharp.NDArray)'");
            StringAssert.Contains(d.GetMessage(), "NDScope.Detach", "the hand alternative is named");
        }

        [TestMethod]
        public async Task ExitParameter_OnASetter_Warns()
        {
            var src = "using NumSharp;\n" +
                      "public class H : System.IDisposable {\n" +
                      "  private NDArray _w;\n" +
                      "  public NDArray W { get => _w; [param: NDScopedExit] set => _w = value; }\n" +
                      "  public void Dispose() => _w?.Dispose();\n" +
                      "}\n";
            var r = await Run(src, "n13_exit_setter.cs", Inactive);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile");
            Assert.AreEqual(1, r.CountOf("NDW013"), "the setter's value parameter counts");
            StringAssert.Contains(r.Ndw.Single(x => x.Id == "NDW013").GetMessage(), "'value'");
        }

        [TestMethod]
        public async Task ExitParameter_OnAConstructor_Warns()
        {
            var src = "using NumSharp;\n" +
                      "public class H : System.IDisposable {\n" +
                      "  private readonly NDArray _w;\n" +
                      "  public H([NDScopedExit] NDArray w) { _w = w; }\n" +
                      "  public void Dispose() => _w?.Dispose();\n" +
                      "}\n";
            var r = await Run(src, "n13_exit_ctor.cs", Inactive);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile");
            Assert.AreEqual(1, r.CountOf("NDW013"));
        }

        [TestMethod]
        public async Task ExitParameter_Inherited_Warns_WithProvenance()
        {
            // The weaver inherits [NDScopedExit] by parameter position from the nearest declaration that
            // marks one; the abstract declaration itself is a contract (no body) and is not warned.
            var src = "using NumSharp;\n" +
                      "public abstract class B { public abstract void Adopt([NDScopedExit] NDArray w); }\n" +
                      "public class D : B, System.IDisposable {\n" +
                      "  private NDArray _w;\n" +
                      "  public override void Adopt(NDArray kept) => _w = kept;\n" +
                      "  public void Dispose() => _w?.Dispose();\n" +
                      "}\n";
            var r = await Run(src, "n13_exit_inherit.cs", Inactive);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile");
            Assert.AreEqual(1, r.CountOf("NDW013"), "only the override (the abstract declaration has no body)");
            var d = r.Ndw.Single(x => x.Id == "NDW013");
            Assert.AreEqual(5, AnalyzerTestHarness.LineOf(d), "at the override");
            StringAssert.Contains(d.GetMessage(), "'kept'", "the override's OWN parameter name at the inherited position");
            StringAssert.Contains(d.GetMessage(), "(inherited from 'B.Adopt(NumSharp.NDArray)')");
        }

        [TestMethod]
        public async Task ScopeAndExit_OnOneMethod_WarnOnce()
        {
            var src = "using NumSharp;\n" +
                      "public class H : System.IDisposable {\n" +
                      "  private NDArray _w;\n" +
                      "  [NDScoped] public NDArray Keep([NDScopedExit] NDArray w) { _w = w; return w + 1.0; }\n" +
                      "  public void Dispose() => _w?.Dispose();\n" +
                      "}\n";
            var r = await Run(src, "n13_scope_and_exit.cs", Inactive);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile");
            Assert.AreEqual(1, r.CountOf("NDW013"), "one NDW013 per member, whichever attribute earned it");
            StringAssert.Contains(r.Ndw.Single(x => x.Id == "NDW013").GetMessage(), "[NDScoped] on", "the scope attribute's report wins");
        }

        [TestMethod]
        public async Task ExitParameter_IsSilent_WhenActive()
        {
            var src = "using NumSharp;\n" +
                      "public class H : System.IDisposable {\n" +
                      "  private NDArray _w;\n" +
                      "  public void Adopt([NDScopedExit] NDArray w) => _w = w;\n" +
                      "  public void Dispose() => _w?.Dispose();\n" +
                      "}\n";
            var r = await Run(src, "n13_exit_active.cs", AnalyzerTestHarness.BuildProperties(("NumSharpBuildActive", "true")));
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile");
            Assert.AreEqual(0, r.CountOf("NDW013"));
        }
    }
}
