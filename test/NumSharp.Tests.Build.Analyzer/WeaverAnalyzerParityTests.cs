using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Mono.Cecil;
using CecilKind = NumSharp.Build.ScopeKind;
using CecilScope = NumSharp.Build.ScopeInheritance;
using RoslynKind = NumSharp.Build.Analyzer.ScopeKind;
using RoslynKnown = NumSharp.Build.Analyzer.KnownTypes;
using RoslynScope = NumSharp.Build.Analyzer.ScopeInheritance;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     The weaver (Mono.Cecil, <c>tools/NumSharp.Build/ScopeInheritance.cs</c>) and the analyzer
    ///     (Roslyn, <c>tools/NumSharp.Build.Analyzer/ScopeInheritance.cs</c>) each walk a method's
    ///     override/implementation graph to find the scope attribute in effect. They are two
    ///     implementations of ONE rule, and a divergence is the worst kind of bug: the analyzer would
    ///     exempt a method the weaver leaves unwoven (a silent leak) or flag one it weaves (noise that
    ///     trains people to ignore NDW012). This gate runs BOTH resolvers over every method of the same
    ///     fixtures — same-assembly hierarchies, interfaces, generics, properties, covariance, and a
    ///     cross-assembly base library — and asserts they agree on the kind, on whether it was inherited
    ///     and on the declaration it came from; then ties the Cecil answer to the actual weave output
    ///     (a scope local exactly where the resolver says there is a target) and the Roslyn answer to
    ///     the actual leak diagnostics (an NDW012 only ever lands inside a method the weaver did not weave).
    /// </summary>
    [TestClass]
    public class WeaverAnalyzerParityTests
    {
        private const string Prelude =
            "using System;\nusing System.Collections.Generic;\nusing System.Threading.Tasks;\nusing NumSharp;\nusing NumSharp.Generic;\n";

        private const string Chains = Prelude +
            "public abstract class Base { [NDScoped] public abstract NDArray A(NDArray a); [NDScoped] public virtual NDArray B(NDArray a) => a.copy(); public virtual NDArray C(NDArray a) => a.copy(); [NDScopedCovered] public virtual NDArray D(NDArray a) => a.copy(); [NDScopedAsync] public abstract Task<NDArray> E(NDArray a); [NDScoped] public virtual NDArray Hidden(NDArray a) => a.copy(); }\n" +
            "public abstract class Mid : Base { public abstract override NDArray B(NDArray a); [NDScoped] public override NDArray C(NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
            "public class Leaf : Mid {\n" +
            "  public override NDArray A(NDArray a) { var t = a + 1.0; return a.copy(); }\n" +
            "  public override NDArray B(NDArray a) { var t = a + 1.0; return a.copy(); }\n" +
            "  public override NDArray C(NDArray a) { var t = a + 1.0; return a.copy(); }\n" +
            "  [NDScopedCovered] public override NDArray D(NDArray a) { var t = a + 1.0; return a.copy(); }\n" +
            "  public override async Task<NDArray> E(NDArray a) { var t = a + 1.0; await Task.Yield(); return a.copy(); }\n" +
            "  public new NDArray Hidden(NDArray a) { var t = a + 1.0; return a.copy(); }\n" +
            "  public NDArray Unrelated(NDArray a) { var t = a + 1.0; return a.copy(); }\n" +
            "}\n" +
            "public abstract class GBase<T> { [NDScoped] public virtual NDArray M(T x, NDArray a) => a.copy(); public virtual NDArray P(T x, NDArray a) => a.copy(); [NDScoped] public abstract NDArray Ovl(int x, NDArray a); public abstract NDArray Ovl(double x, NDArray a); }\n" +
            "public abstract class GMid : GBase<int> { }\n" +
            "public class GLeaf : GMid { public override NDArray M(int x, NDArray a) { var t = a + x; return a.copy(); } public override NDArray P(int x, NDArray a) { var t = a + x; return a.copy(); } public override NDArray Ovl(int x, NDArray a) { var t = a + x; return a.copy(); } public override NDArray Ovl(double x, NDArray a) { var t = a + x; return a.copy(); } }\n" +
            "public class GOpen<V> : GBase<V> { public override NDArray M(V x, NDArray a) { var t = a + 1.0; return a.copy(); } public override NDArray Ovl(int x, NDArray a) => a.copy(); public override NDArray Ovl(double x, NDArray a) => a.copy(); }\n";

        private const string Interfaces = Prelude +
            "public interface IOp { [NDScoped] NDArray Apply(NDArray a); [NDScoped] NDArray Value { get; } NDArray Plain(NDArray a); }\n" +
            "public interface IOp2 : IOp { }\n" +
            "public interface IHide : IOp { new NDArray Apply(NDArray a); }\n" +
            "public class Implicit : IOp2 { public NDArray Apply(NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray Value { get { var t = np.arange(2); return t.copy(); } } public NDArray Plain(NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
            "public class Explicit : IOp { NDArray IOp.Apply(NDArray a) { var t = a + 1.0; return a.copy(); } NDArray IOp.Value { get { var t = np.arange(2); return t.copy(); } } NDArray IOp.Plain(NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
            "public struct SImpl : IOp { public NDArray Apply(NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray Value { get { var t = np.arange(2); return t.copy(); } } public NDArray Plain(NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
            "public interface IGen<T> { [NDScoped] NDArray Map(T x, NDArray a); }\n" +
            "public class GImpl<T> : IGen<T> { NDArray IGen<T>.Map(T x, NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
            "public class GImplInt : IGen<int> { public NDArray Map(int x, NDArray a) { var t = a + x; return a.copy(); } }\n" +
            "public interface IDim { [NDScoped] NDArray Twice(NDArray a) { var t = a * 2.0; return a.copy(); } }\n" +
            "public class DimOverride : IDim { public NDArray Twice(NDArray a) { var t = a * 3.0; return a.copy(); } }\n" +
            "public class HideImpl : IHide { public NDArray Apply(NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray Value => np.arange(2); public NDArray Plain(NDArray a) => a.copy(); }\n" +
            "public class Unrelated { public NDArray Apply(NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
            "public class A { public virtual NDArray Apply(NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +   // the known limitation, on both sides
            "public class B : A, IOp { public NDArray Value => np.arange(2); public NDArray Plain(NDArray a) => a.copy(); }\n" +
            "public class C : A, IOp { public override NDArray Apply(NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray Value => np.arange(2); public NDArray Plain(NDArray a) => a.copy(); }\n";

        private const string PropertiesAndCovariance = Prelude +
            "public abstract class P { [NDScoped] public abstract NDArray V1 { get; } public abstract NDArray V2 { [NDScoped] get; } [NDScoped] public virtual NDArray V3 { get => np.arange(3); set { } } public virtual NDArray V4 { get => np.arange(3); } }\n" +
            "public class PD : P { public override NDArray V1 { get { var t = np.arange(3); return t.copy(); } } public override NDArray V2 { get { var t = np.arange(3); return t.copy(); } } public override NDArray V3 { get { var t = np.arange(3); return t.copy(); } set { var t = value + 1.0; } } public override NDArray V4 { get { var t = np.arange(3); return t.copy(); } } }\n" +
            "public class CB { [NDScoped] public virtual NDArray M(NDArray a) => a.copy(); }\n" +
            "public class CD : CB { public override NDArray<double> M(NDArray a) { var t = a + 1.0; return (a.copy()).MakeGeneric<double>(); } }\n" +
            "public static class Outer { public abstract class NB { [NDScoped] public abstract NDArray M(NDArray a); } public class ND : NB { public override NDArray M(NDArray a) { var t = a + 1.0; return a.copy(); } } }\n";

        private const string BaseLibrary = Prelude +
            "namespace Lib { public abstract class Base { [NDScoped] public abstract NDArray Compute(NDArray a); [NDScoped] public virtual NDArray Twice(NDArray a) => a * 2.0; [NDScopedAsync] public abstract Task<NDArray> ComputeAsync(NDArray a); [NDScoped] public abstract NDArray Value { get; } public virtual NDArray Plain(NDArray a) => a.copy(); [NDScopedCovered] public virtual NDArray Covered(NDArray a) => a.copy(); }\n" +
            "  public interface IOp { [NDScoped] NDArray Apply(NDArray a); }\n" +
            "  public abstract class GBase<T> { [NDScoped] public virtual NDArray Map(T x, NDArray a) => a.copy(); } }\n";

        private const string Consumer = Prelude +
            "public class Derived : Lib.Base, Lib.IOp {\n" +
            "  public override NDArray Compute(NDArray a) { var t = a + 1.0; return a.copy(); }\n" +
            "  public override NDArray Twice(NDArray a) { var t = a * 3.0; return a.copy(); }\n" +
            "  public override async Task<NDArray> ComputeAsync(NDArray a) { var t = a + 1.0; await Task.Yield(); return a.copy(); }\n" +
            "  public override NDArray Value { get { var t = np.arange(3); return t.copy(); } }\n" +
            "  public override NDArray Plain(NDArray a) { var t = a + 1.0; return a.copy(); }\n" +
            "  public override NDArray Covered(NDArray a) { var t = a + 1.0; return a.copy(); }\n" +
            "  public NDArray Apply(NDArray a) { var t = a - 1.0; return a.copy(); }\n" +
            "}\n" +
            "public class Leaf : Lib.GBase<int> { public override NDArray Map(int x, NDArray a) { var t = a + x; return a.copy(); } }\n";

        /// <summary>The fixtures by short name (the DataRow carries only the name, so a failure reads as one line, not a source dump).</summary>
        private static (string Source, string BaseLibrary) Fixture(string name) => name switch
        {
            "Chains" => (Chains, null),
            "Interfaces" => (Interfaces, null),
            "PropertiesAndCovariance" => (PropertiesAndCovariance, null),
            "CrossAssembly" => (Consumer, BaseLibrary),
            _ => throw new ArgumentOutOfRangeException(nameof(name), name, "unknown parity fixture"),
        };

        [DataTestMethod]
        [DataRow("Chains")]
        [DataRow("Interfaces")]
        [DataRow("PropertiesAndCovariance")]
        [DataRow("CrossAssembly")]
        public async Task BothResolvers_AgreeOnEveryMethod_AndMatchTheirOwnOutputs(string name)
        {
            var (source, baseLibrary) = Fixture(name);
            // The referenced "base library", when the fixture has one: a real on-disk assembly both sides read.
            var extra = ImmutableArray<string>.Empty;
            if (baseLibrary != null)
                extra = ImmutableArray.Create(WeaverTestHarness.Compile(baseLibrary, "ParityLib").DllPath);

            // Roslyn side: the compilation (symbols) + the analyzer's own leak diagnostics on it.
            var parse = new CSharpParseOptions(LanguageVersion.Latest);
            var tree = CSharpSyntaxTree.ParseText(SourceText.From(source), parse, path: name + ".cs");
            var references = AnalyzerTestHarness.References.AddRange(extra.Select(p => (MetadataReference)MetadataReference.CreateFromFile(p)));
            var comp = CSharpCompilation.Create("Parity_" + name, new[] { tree }, references,
                new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, allowUnsafe: true));
            var errors = comp.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error).ToList();
            Assert.AreEqual(0, errors.Count, "parity fixture must compile:\n  " + string.Join("\n  ", errors));
            var known = RoslynKnown.Resolve(comp);
            Assert.IsNotNull(known, "NumSharp must resolve for the analyzer side");
            var model = comp.GetSemanticModel(tree);
            var analyzed = await AnalyzerTestHarness.RunAsync(source, name + ".cs",
                extra.Select(p => (MetadataReference)MetadataReference.CreateFromFile(p)).ToImmutableArray(), null);

            // Cecil side: the same source compiled to disk and WOVEN.
            var run = WeaverTestHarness.CompileAndWeave(source, "Parity" + name, true, extra.ToArray());
            Assert.AreEqual(0, run.Result.Errors, run.Report);
            using var asm = run.ReadCecil();
            var cecil = new CecilScope();

            int compared = 0, inheritedSeen = 0;
            var failures = new List<string>();
            foreach (var type in AllTypes(comp.Assembly.GlobalNamespace))
            {
                foreach (var method in type.GetMembers().OfType<IMethodSymbol>())
                {
                    if (method.MethodKind is MethodKind.Constructor or MethodKind.StaticConstructor or MethodKind.Destructor or MethodKind.LocalFunction)
                        continue;
                    if (method.IsImplicitlyDeclared)
                        continue;

                    var def = FindCecil(asm.MainModule, type, method);
                    if (def is null)
                    {
                        failures.Add($"{Display(method)}: no Cecil counterpart found");
                        continue;
                    }

                    var r = RoslynScope.Effective(method, known);
                    var c = cecil.EffectiveScope(def);
                    var rKinds = r?.Kinds ?? RoslynKind.None;
                    var cKinds = c?.Kinds ?? CecilKind.None;
                    if ((int)rKinds != (int)cKinds)
                        failures.Add($"{Display(method)}: analyzer says {rKinds}, weaver says {cKinds}");
                    else if (r != null && c != null)
                    {
                        if (r.Value.Inherited != c.Value.Inherited)
                            failures.Add($"{Display(method)}: inherited? analyzer {r.Value.Inherited}, weaver {c.Value.Inherited}");
                        else if (r.Value.Inherited)
                        {
                            inheritedSeen++;
                            // Both sides normalized to "Type`arity.Member": a property-level attribute is
                            // reported on the PROPERTY by Roslyn and on its GETTER by Cecil.
                            var rSource = SourceName(r.Value.Source);
                            var cSource = c.Value.Source.DeclaringType.Name + "." + c.Value.Source.Name.Replace("get_", "");
                            if (rSource != cSource)
                                failures.Add($"{Display(method)}: inherited from analyzer '{rSource}' vs weaver '{cSource}'");
                        }
                    }

                    // The Cecil verdict must be exactly what the weave DID: a scope local iff the effective
                    // attribute is a single scope model and there is a body to weave.
                    bool scopeKind = (cKinds & (CecilKind.Sync | CecilKind.Async)) != 0 && (cKinds & (CecilKind.Sync | CecilKind.Async)) != (CecilKind.Sync | CecilKind.Async);
                    bool expectedWoven = scopeKind && def.HasBody;
                    bool woven = WeaveRun.HasScopeLocal(def);
                    if (expectedWoven != woven)
                        failures.Add($"{Display(method)}: resolver says target={expectedWoven} but woven={woven}");

                    compared++;
                }
            }

            // The Roslyn verdict must be exactly what the leak analyzer DID: every NDW012 sits inside a
            // method the weaver left unwoven (and that has no scope-family attribute in effect).
            foreach (var d in analyzed.Ndw.Where(x => x.Id == "NDW012"))
            {
                var node = tree.GetRoot().FindToken(d.Location.SourceSpan.Start).Parent;
                var declaration = node.AncestorsAndSelf().FirstOrDefault(n =>
                    n is MethodDeclarationSyntax || n is AccessorDeclarationSyntax || n is ArrowExpressionClauseSyntax && n.Parent is PropertyDeclarationSyntax);
                if (declaration is ArrowExpressionClauseSyntax arrow)
                    declaration = arrow.Parent;
                if (declaration is null)
                    continue;
                var symbol = model.GetDeclaredSymbol(declaration) as IMethodSymbol
                             ?? (model.GetDeclaredSymbol(declaration) as IPropertySymbol)?.GetMethod;
                if (symbol is null)
                    continue;
                var def = FindCecil(asm.MainModule, symbol.ContainingType, symbol);
                if (def != null && WeaveRun.HasScopeLocal(def))
                    failures.Add($"{Display(symbol)}: the analyzer reported NDW012 at line {AnalyzerTestHarness.LineOf(d)} inside a method the weaver WOVE");
                if (RoslynScope.Effective(symbol, known) != null)
                    failures.Add($"{Display(symbol)}: NDW012 inside a method the analyzer's own resolver says is scoped/covered");
            }

            Assert.AreEqual(0, failures.Count, $"{name}: weaver/analyzer parity failures:\n  " + string.Join("\n  ", failures));
            Assert.IsTrue(compared >= 8, $"{name}: the parity walk should have compared a real surface (compared {compared})");
            Assert.IsTrue(inheritedSeen >= 3, $"{name}: the fixture should exercise inherited declarations (saw {inheritedSeen})");
        }

        // ---------------------------------------------------------------- symbol ↔ metadata matching

        private static IEnumerable<INamedTypeSymbol> AllTypes(INamespaceSymbol ns)
        {
            foreach (var member in ns.GetMembers())
            {
                if (member is INamespaceSymbol sub)
                    foreach (var t in AllTypes(sub))
                        yield return t;
                else if (member is INamedTypeSymbol type)
                    foreach (var t in WithNested(type))
                        yield return t;
            }
        }

        private static IEnumerable<INamedTypeSymbol> WithNested(INamedTypeSymbol type)
        {
            yield return type;
            foreach (var nested in type.GetTypeMembers())
                foreach (var t in WithNested(nested))
                    yield return t;
        }

        /// <summary>Cecil's spelling of a type: <c>Ns.Outer/Nested</c> with the metadata (arity-suffixed) names.</summary>
        private static string CecilName(INamedTypeSymbol type)
        {
            if (type.ContainingType != null)
                return CecilName(type.ContainingType) + "/" + type.MetadataName;
            var ns = type.ContainingNamespace;
            return ns == null || ns.IsGlobalNamespace ? type.MetadataName : ns.ToDisplayString() + "." + type.MetadataName;
        }

        private static string ParamName(IParameterSymbol p)
        {
            string core = p.Type switch
            {
                IArrayTypeSymbol arr => arr.ElementType.MetadataName + "[]",
                IPointerTypeSymbol ptr => ptr.PointedAtType.MetadataName + "*",
                _ => p.Type.MetadataName,
            };
            return p.RefKind == RefKind.None ? core : core + "&";
        }

        private static MethodDefinition FindCecil(ModuleDefinition module, INamedTypeSymbol type, IMethodSymbol method)
        {
            var typeName = CecilName(type);
            var candidates = WeaveRun.AllMethods(module)
                .Where(m => m.DeclaringType.FullName == typeName && m.Name == method.MetadataName && m.Parameters.Count == method.Parameters.Length)
                .ToList();
            if (candidates.Count == 1)
                return candidates[0];
            return candidates.FirstOrDefault(m => m.Parameters.Select(p => p.ParameterType.Name).SequenceEqual(method.Parameters.Select(ParamName)));
        }

        private static string SourceName(ISymbol source)
        {
            var name = source is IPropertySymbol p ? p.Name : source.Name.Replace("get_", "");
            return source.ContainingType.MetadataName + "." + name;
        }

        private static string Display(IMethodSymbol m) => m.ContainingType.Name + "." + m.MetadataName;
    }
}
