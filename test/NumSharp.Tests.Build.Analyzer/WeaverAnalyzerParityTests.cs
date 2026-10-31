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

        /// <summary>
        ///     Interface hierarchies the weaver deduplicates its search over (it searches each interface
        ///     instantiation once per level of a method's walk): a chain and a diamond lattice with unscoped members at every
        ///     level, one generic interface reached under two instantiations, the same instantiation reached
        ///     along two paths, and a generic implementer instantiating over its own type parameter. The
        ///     analyzer resolves through Roslyn's own deduplicated interface set, so the two must agree.
        /// </summary>
        private static readonly string InterfaceLattices = Prelude +
            "public interface IC0 { [NDScoped] NDArray Apply(NDArray a); NDArray M0(NDArray a); }\n" +
            "public interface IC1 : IC0 { NDArray M1(NDArray a); }\npublic interface IC2 : IC1 { NDArray M2(NDArray a); }\npublic interface IC3 : IC2 { NDArray M3(NDArray a); }\n" +
            "public class ChainImpl : IC3 { public NDArray Apply(NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray M0(NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray M1(NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray M2(NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray M3(NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
            "public interface IRoot { [NDScoped] NDArray Run(NDArray a); }\npublic interface IA1 : IRoot { }\npublic interface IB1 : IRoot { }\npublic interface IA2 : IA1, IB1 { }\npublic interface IB2 : IA1, IB1 { }\n" +
            "public class Lattice : IA2, IB2 { public NDArray Run(NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray Other(NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
            "public interface IBase<T> { [NDScoped] NDArray M(T x, NDArray a); NDArray Plain(T x, NDArray a); }\n" +
            "public interface IA : IBase<int> { }\npublic interface IB : IBase<string> { }\npublic interface IL : IBase<double> { }\npublic interface IR : IBase<double> { }\n" +
            "public class Both : IA, IB { public NDArray M(int x, NDArray a) { var t = a + x; return a.copy(); } public NDArray M(string x, NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray Plain(int x, NDArray a) { var t = a + x; return a.copy(); } public NDArray Plain(string x, NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
            "public class Same : IL, IR { public NDArray M(double x, NDArray a) { var t = a + x; return a.copy(); } public NDArray Plain(double x, NDArray a) { var t = a + x; return a.copy(); } }\n" +
            "public class Open<U> : IA, IBase<U[]> { public NDArray M(int x, NDArray a) { var t = a + x; return a.copy(); } public NDArray M(U[] x, NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray Plain(int x, NDArray a) { var t = a + x; return a.copy(); } public NDArray Plain(U[] x, NDArray a) { var t = a + 1.0; return a.copy(); } }\n";

        /// <summary>
        ///     The instantiation key's edges, for both resolvers: arguments that differ only inside a type
        ///     argument (List&lt;int&gt; / List&lt;string&gt;), array ranks and nesting, nullable and tuple
        ///     arguments, and an interface listed on a generic base class read in the base's context
        ///     (IBase&lt;T&gt; from GBase&lt;int&gt;) beside the derived class's own instantiation.
        /// </summary>
        private static readonly string InterfaceKeys = Prelude +
            "public interface IBase<T> { [NDScoped] NDArray M(T x, NDArray a); NDArray Plain(T x, NDArray a); }\n" +
            "public interface IL1 : IBase<List<int>> { }\npublic interface IL2 : IBase<List<string>> { }\n" +
            "public class Lists : IL1, IL2 { public NDArray M(List<int> x, NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray M(List<string> x, NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray Plain(List<int> x, NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray Plain(List<string> x, NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
            "public interface IR1 : IBase<int[]> { }\npublic interface IR2 : IBase<int[,]> { }\npublic interface IR3 : IBase<int[][]> { }\n" +
            "public class Ranks : IR1, IR2, IR3 { public NDArray M(int[] x, NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray M(int[,] x, NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray M(int[][] x, NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray Plain(int[] x, NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray Plain(int[,] x, NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray Plain(int[][] x, NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
            "public interface IN1 : IBase<int?> { }\npublic interface IN2 : IBase<int> { }\npublic interface IT1 : IBase<(int, string)> { }\npublic interface IT2 : IBase<(string, int)> { }\n" +
            "public class Specials : IN1, IN2, IT1, IT2 { public NDArray M(int? x, NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray M(int x, NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray M((int, string) x, NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray M((string, int) x, NDArray a) { var t = a + 1.0; return a.copy(); }" +
            " public NDArray Plain(int? x, NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray Plain(int x, NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray Plain((int, string) x, NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray Plain((string, int) x, NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
            "public class GBase<T> : IBase<T> { public virtual NDArray M(T x, NDArray a) { var t = a + 1.0; return a.copy(); } public virtual NDArray Plain(T x, NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
            "public class GDerived : GBase<int>, IBase<string> { public override NDArray M(int x, NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray M(string x, NDArray a) { var t = a + 1.0; return a.copy(); } public override NDArray Plain(int x, NDArray a) { var t = a + 1.0; return a.copy(); } public NDArray Plain(string x, NDArray a) { var t = a + 1.0; return a.copy(); } }\n";

        /// <summary>
        ///     Property shapes the weaver's getter gate decides, for both resolvers: a property-level
        ///     [NDScopedCovered] between a scoped interface property and an override (the opt-out must stop
        ///     the inheritance on both sides), a derived type with many plain properties overriding a scoped
        ///     base property, a scoped virtual indexer and its override, a scoped static property, and a scoped
        ///     property of a generic type.
        /// </summary>
        private static readonly string GetterShapes = Prelude +
            "public interface IRoot { [NDScoped] NDArray V { get; } }\n" +
            "public class Mid : IRoot { [NDScopedCovered] public virtual NDArray V { get { var t = np.arange(3) + 1.0; return t.copy(); } } }\n" +
            "public class Leaf : Mid { public override NDArray V { get { var t = np.arange(3) + 2.0; return t.copy(); } } }\n" +
            "public class PBase { [NDScoped] public virtual NDArray W { get { var t = np.arange(3) + 1.0; return t.copy(); } } }\n" +
            "public class PDerived : PBase {" + string.Concat(Enumerable.Range(0, 20).Select(i => $" public int A{i} {{ get; set; }}")) +
            " public override NDArray W { get { var t = np.arange(3) + 2.0; return t.copy(); } } }\n" +
            "public class Grid { [NDScoped] public virtual NDArray this[int i] { get { var t = np.arange(3) + i; return t.copy(); } } }\n" +
            "public class SubGrid : Grid { public override NDArray this[int i] { get { var t = np.arange(3) * i; return t.copy(); } } }\n" +
            "public static class Defaults { [NDScoped] public static NDArray Ramp { get { var t = np.arange(3) + 1.0; return t.copy(); } } }\n" +
            "public class Holder<T> { [NDScoped] public NDArray Ramp { get { var t = np.arange(3) + 1.0; return t.copy(); } } public T Item { get; set; } }\n";

        /// <summary>
        ///     How the CLR maps an interface member onto a method, in the shapes where a name-and-signature
        ///     match and the interface map disagree, for both resolvers (all found by a parity probe on
        ///     2026-10-24; both layers now follow the map):
        ///     <list type="bullet">
        ///         <item>an interface listed on a base that does not declare the slot (<c>Stage : StageRoot, IStage</c>):
        ///         the override below it runs for calls through the interface and inherits — through one and two
        ///         levels, sealed, abstract, a slot two bases up, a generic slot, a property getter and an
        ///         <c>[NDScopedExit]</c> parameter; a <c>new virtual</c> below it is a new slot and inherits nothing.
        ///         The analyzer missed all of these (NDW012 inside methods the weaver wove).</item>
        ///         <item>an explicit implementation beside a public virtual of the same signature
        ///         (<c>Forwarder</c>): the interface runs the explicit one, so the public method — and its
        ///         override — inherit nothing. The weaver wove them by the name match, which disposes any
        ///         array they keep.</item>
        ///         <item>re-implementation: a type listing the interface again implements it with its public
        ///         method (<c>Reimplementer</c>); a type that re-lists it explicitly still inherits through the
        ///         slot's declaring base when that base implements the interface (<c>ExplicitOverrider</c>), and
        ///         inherits nothing when only a base below the slot listed it (<c>ReExplicit</c>, <c>BelowExplicit</c>).</item>
        ///         <item>one generic interface implemented explicitly for one instantiation and implicitly for
        ///         another (<c>Mixed</c>), and an explicit implementation of a base interface's member on a type
        ///         that lists the derived interface (<c>ViaDerived</c>).</item>
        ///     </list>
        /// </summary>
        private static readonly string InterfaceMapping = Prelude +
            "public interface IStage { [NDScoped] NDArray Run(NDArray a); }\n" +
            "public class StageRoot { public virtual NDArray Run(NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
            "public class Stage : StageRoot, IStage { }\n" +
            "public class MyStage : Stage { public override NDArray Run(NDArray a) { var t = a + 2.0; return a.copy(); } }\n" +
            "public class MyStage2 : MyStage { public override NDArray Run(NDArray a) { var t = a + 3.0; return a.copy(); } }\n" +
            "public class SealedStage : Stage { public sealed override NDArray Run(NDArray a) { var t = a + 4.0; return a.copy(); } }\n" +
            "public class NewSlotStage : Stage { public new virtual NDArray Run(NDArray a) { var t = a + 5.0; return a.copy(); } }\n" +
            "public abstract class AbstractRoot { public abstract NDArray Run(NDArray a); }\n" +
            "public abstract class AbstractStage : AbstractRoot, IStage { }\n" +
            "public class ConcreteStage : AbstractStage { public override NDArray Run(NDArray a) { var t = a + 6.0; return a.copy(); } }\n" +
            "public class GapRoot { public virtual NDArray Run(NDArray a) { var t = a + 7.0; return a.copy(); } }\n" +
            "public class GapMiddle : GapRoot { }\npublic class GapStage : GapMiddle, IStage { }\n" +
            "public class GapLeaf : GapStage { public override NDArray Run(NDArray a) { var t = a + 8.0; return a.copy(); } }\n" +
            "public interface IGStage<T> { [NDScoped] NDArray Run(T x, NDArray a); }\n" +
            "public class GRoot<T> { public virtual NDArray Run(T x, NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
            "public class GStage : GRoot<int>, IGStage<int> { }\n" +
            "public class GLeaf : GStage { public override NDArray Run(int x, NDArray a) { var t = a + x; return a.copy(); } }\n" +
            "public interface IProp { [NDScoped] NDArray Value { get; } }\n" +
            "public class PropRoot { public virtual NDArray Value { get { var t = np.arange(3) + 1.0; return t.copy(); } } }\n" +
            "public class PropStage : PropRoot, IProp { }\n" +
            "public class PropLeaf : PropStage { public override NDArray Value { get { var t = np.arange(3) + 2.0; return t.copy(); } } }\n" +
            "public interface IKeep { void Keep([NDScopedExit] NDArray w); }\n" +
            "public class KeepRoot { public NDArray Kept; public virtual void Keep(NDArray w) => Kept = w; }\n" +
            "public class KeepStage : KeepRoot, IKeep { }\n" +
            "public class KeepLeaf : KeepStage { public override void Keep(NDArray w) => Kept = w; }\n" +
            "public interface IOp { [NDScoped] NDArray Apply(NDArray a); }\n" +
            "public class Forwarder : IOp { NDArray IOp.Apply(NDArray a) { var t = a + 1.0; return Apply(a).copy(); } public virtual NDArray Apply(NDArray a) { var t = a + 2.0; return a.copy(); } }\n" +
            "public class ForwarderLeaf : Forwarder { public override NDArray Apply(NDArray a) { var t = a + 3.0; return a.copy(); } }\n" +
            "public class ExplicitBase : IOp { NDArray IOp.Apply(NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
            "public class Reimplementer : ExplicitBase, IOp { public NDArray Apply(NDArray a) { var t = a + 2.0; return a.copy(); } }\n" +
            "public class ImplRoot : IOp { public virtual NDArray Apply(NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
            "public class ExplicitOverrider : ImplRoot, IOp { NDArray IOp.Apply(NDArray a) { var t = a + 2.0; return a.copy(); } public override NDArray Apply(NDArray a) { var t = a + 3.0; return a.copy(); } }\n" +
            "public class PlainRoot { public virtual NDArray Apply(NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
            "public class ExplicitListing : PlainRoot, IOp { NDArray IOp.Apply(NDArray a) { var t = a + 2.0; return a.copy(); } }\n" +
            "public class BelowExplicit : ExplicitListing { public override NDArray Apply(NDArray a) { var t = a + 3.0; return a.copy(); } }\n" +
            "public class ImplicitListing : PlainRoot, IOp { }\n" +
            "public class ReExplicit : ImplicitListing, IOp { NDArray IOp.Apply(NDArray a) { var t = a + 4.0; return a.copy(); } public override NDArray Apply(NDArray a) { var t = a + 5.0; return a.copy(); } }\n" +
            "public interface IGOp<T> { [NDScoped] NDArray Map(T x, NDArray a); }\n" +
            "public class Mixed : IGOp<int>, IGOp<string> { NDArray IGOp<int>.Map(int x, NDArray a) { var t = a + x; return a.copy(); } public virtual NDArray Map(int x, NDArray a) { var t = a + x; return a.copy(); } public virtual NDArray Map(string x, NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
            "public interface IOpEx : IOp { }\n" +
            "public class ViaDerived : IOpEx { NDArray IOp.Apply(NDArray a) { var t = a + 1.0; return a.copy(); } public virtual NDArray Apply(NDArray a) { var t = a + 2.0; return a.copy(); } }\n";

        /// <summary>The number of override levels in <see cref="DepthBoundary"/>: enough to put the root's interface past the walk's 64-level limit for the three most derived types.</summary>
        internal const int DepthBoundaryLevels = 67;

        /// <summary>
        ///     A <see cref="DepthBoundaryLevels"/>-level override chain whose ROOT implements an interface with
        ///     a scoped method, a scoped property and an <c>[NDScopedExit]</c> parameter: from <c>C{k}</c> the
        ///     root's interface is <c>66 - k</c> levels up, and both walks give up past their 64-level limit at
        ///     the same place — C0 to C2 inherit nothing, C3 and up inherit all three. The analyzer used to
        ///     search interfaces one level deeper than the weaver, and called C2 scoped (and dropped its
        ///     NDW012) while the weave left it unwoven (found by a parity probe on 2026-10-24).
        /// </summary>
        /// <returns>The fixture's source.</returns>
        internal static string DepthBoundary()
        {
            int top = DepthBoundaryLevels - 1;
            var sb = new System.Text.StringBuilder(Prelude);
            sb.AppendLine("public interface IDeep { [NDScoped] NDArray Apply(NDArray a); [NDScoped] NDArray Value { get; } void Keep([NDScopedExit] NDArray w); }");
            sb.AppendLine($"public class C{top} : IDeep {{ public NDArray Kept; public virtual NDArray Apply(NDArray a) {{ var t = a + 1.0; return a.copy(); }} public virtual NDArray Value {{ get {{ var t = np.arange(3) + 1.0; return t.copy(); }} }} public virtual void Keep(NDArray w) => Kept = w; }}");
            for (int k = top - 1; k >= 0; k--)
                sb.AppendLine($"public class C{k} : C{k + 1} {{ public override NDArray Apply(NDArray a) {{ var t = a + {k}.0; return a.copy(); }} public override NDArray Value {{ get {{ var t = np.arange(3) + {k}.0; return t.copy(); }} }} public override void Keep(NDArray w) => Kept = w; }}");
            return sb.ToString();
        }

        /// <summary>The fixtures by short name (the DataRow carries only the name, so a failure reads as one line, not a source dump).</summary>
        private static (string Source, string BaseLibrary) Fixture(string name) => name switch
        {
            "Chains" => (Chains, null),
            "Interfaces" => (Interfaces, null),
            "PropertiesAndCovariance" => (PropertiesAndCovariance, null),
            "CrossAssembly" => (Consumer, BaseLibrary),
            "InterfaceLattices" => (InterfaceLattices, null),
            "InterfaceKeys" => (InterfaceKeys, null),
            "GetterShapes" => (GetterShapes, null),
            "InterfaceMapping" => (InterfaceMapping, null),
            "DepthBoundary" => (DepthBoundary(), null),
            _ => throw new ArgumentOutOfRangeException(nameof(name), name, "unknown parity fixture"),
        };

        /// <summary>
        ///     For every method of the named fixture, the weaver's and the analyzer's resolvers agree on the
        ///     scope kind, on whether it was inherited and on the declaration it came from; the weave puts a
        ///     scope local exactly where the weaver's resolver sees a target, and NDW012 lands only in
        ///     methods the weave left alone. A divergence would be a silent leak or warning noise.
        ///     <para>
        ///     The second walk over the same graph — <c>[NDScopedExit]</c> parameters, inherited by position —
        ///     is compared the same way: the effective positions, whether they were inherited and from which
        ///     declaration, and the weave's own output (a <c>NDScope.Detach</c> call exactly where the weaver's
        ///     resolver sees retained parameters and there is a body).
        ///     </para>
        /// </summary>
        /// <param name="name">The fixture's short name (see <see cref="Fixture"/>).</param>
        /// <exception cref="AssertFailedException">The fixture does not compile or weave, or any method diverges.</exception>
        [DataTestMethod]
        [DataRow("Chains")]
        [DataRow("Interfaces")]
        [DataRow("PropertiesAndCovariance")]
        [DataRow("CrossAssembly")]
        [DataRow("InterfaceLattices")]
        [DataRow("InterfaceKeys")]
        [DataRow("GetterShapes")]
        [DataRow("InterfaceMapping")]
        [DataRow("DepthBoundary")]
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

            int compared = 0, inheritedSeen = 0, exitInheritedSeen = 0;
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

                    // The [NDScopedExit] walk: the same graph, a different test for what counts as a declaration.
                    var rExit = AnalyzerExitPositions(method, known, out var rExitSource);
                    var cExit = cecil.EffectiveExitParameters(def);
                    var cExitSource = cecil.ExitParametersSource(def);
                    if (!rExit.SequenceEqual(cExit))
                        failures.Add($"{Display(method)}: [NDScopedExit] positions analyzer [{string.Join(",", rExit)}] vs weaver [{string.Join(",", cExit)}]");
                    else if ((rExitSource != null) != (cExitSource != null))
                        failures.Add($"{Display(method)}: [NDScopedExit] inherited? analyzer {rExitSource != null}, weaver {cExitSource != null}");
                    else if (rExitSource != null)
                    {
                        exitInheritedSeen++;
                        var rName = rExitSource.ContainingType.MetadataName + "." + rExitSource.MetadataName;
                        var cName = cExitSource.DeclaringType.Name + "." + cExitSource.Name;
                        if (rName != cName)
                            failures.Add($"{Display(method)}: [NDScopedExit] inherited from analyzer '{rName}' vs weaver '{cName}'");
                    }

                    // ... and the weave's output: a Detach call iff the resolver sees retained parameters and
                    // there is a body to put it in.
                    bool expectDetach = cExit.Length > 0 && def.HasBody;
                    if (expectDetach != WeaveRun.CallsDetach(def))
                        failures.Add($"{Display(method)}: exit resolver says detach={expectDetach} but the weave {(expectDetach ? "has no" : "has a")} NDScope.Detach call");

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
            if (name == "InterfaceMapping" || name == "DepthBoundary")
                Assert.IsTrue(exitInheritedSeen >= 1, $"{name}: the fixture should exercise inherited [NDScopedExit] parameters (saw {exitInheritedSeen})");
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

        /// <summary>
        ///     Cecil's <c>FullName</c> spelling of a type as it appears in a signature: namespace-qualified, nested
        ///     types joined by '/', generic arguments in angle brackets without spaces, arrays with their rank
        ///     (Cecil's "0..." lower bounds are stripped from the Cecil side before comparing), type parameters
        ///     by name. Precise enough to tell <c>List&lt;int&gt;</c> from <c>List&lt;string&gt;</c> and <c>int[,]</c>
        ///     from <c>int[][]</c>, which <see cref="ParamName"/>'s bare names cannot.
        /// </summary>
        /// <param name="t">A parameter type.</param>
        /// <returns>The Cecil-style full name.</returns>
        private static string CecilTypeName(ITypeSymbol t) => t switch
        {
            IArrayTypeSymbol arr => CecilTypeName(arr.ElementType) + "[" + new string(',', arr.Rank - 1) + "]",
            IPointerTypeSymbol ptr => CecilTypeName(ptr.PointedAtType) + "*",
            ITypeParameterSymbol tp => tp.Name,
            INamedTypeSymbol named when named.IsGenericType => CecilName(named.OriginalDefinition) + "<" + string.Join(",", named.TypeArguments.Select(CecilTypeName)) + ">",
            INamedTypeSymbol named => CecilName(named),
            _ => t.MetadataName,
        };

        /// <summary>A parameter's <see cref="CecilTypeName"/>, with Cecil's '&amp;' for a by-reference parameter.</summary>
        /// <param name="p">The parameter.</param>
        /// <returns>The Cecil-style full name of its type.</returns>
        private static string ParamFullName(IParameterSymbol p)
        {
            string core = CecilTypeName(p.Type);
            return p.RefKind == RefKind.None ? core : core + "&";
        }

        /// <summary>
        ///     The Cecil method behind a Roslyn method: same declaring type, name and parameter count; among
        ///     overloads, the one whose parameter types match by full name, and failing that (a signature the
        ///     full-name spelling does not cover, such as an <c>in</c> parameter's modreq) by bare name — the
        ///     original rule, so a fixture that paired before still pairs the same way.
        /// </summary>
        /// <param name="module">The woven module.</param>
        /// <param name="type">The Roslyn declaring type.</param>
        /// <param name="method">The Roslyn method.</param>
        /// <returns>The Cecil method, or null when none matches.</returns>
        private static MethodDefinition FindCecil(ModuleDefinition module, INamedTypeSymbol type, IMethodSymbol method)
        {
            var typeName = CecilName(type);
            var candidates = WeaveRun.AllMethods(module)
                .Where(m => m.DeclaringType.FullName == typeName && m.Name == method.MetadataName && m.Parameters.Count == method.Parameters.Length)
                .ToList();
            if (candidates.Count == 1)
                return candidates[0];
            var exact = candidates.FirstOrDefault(m => m.Parameters.Select(p => p.ParameterType.FullName.Replace("0...", "")).SequenceEqual(method.Parameters.Select(ParamFullName)));
            return exact ?? candidates.FirstOrDefault(m => m.Parameters.Select(p => p.ParameterType.Name).SequenceEqual(method.Parameters.Select(ParamName)));
        }

        /// <summary>
        ///     The analyzer's inherited declaration spelled the way the weaver's is compared: "Type`arity.Member",
        ///     a property by its METADATA name — an indexer is "this[]" to Roslyn but "Item" in metadata, where
        ///     the weaver finds it through its getter <c>get_Item</c>.
        /// </summary>
        /// <param name="source">The declaration the analyzer inherited from (a method, or the property of a property-level attribute).</param>
        /// <returns>The normalized name.</returns>
        private static string SourceName(ISymbol source)
        {
            var name = source is IPropertySymbol p ? p.MetadataName : source.Name.Replace("get_", "");
            return source.ContainingType.MetadataName + "." + name;
        }

        private static string Display(IMethodSymbol m) => m.ContainingType.Name + "." + m.MetadataName;

        /// <summary>
        ///     The analyzer's effective <c>[NDScopedExit]</c> positions for <paramref name="method"/>, computed
        ///     the way the weaver's <c>EffectiveExitParameters</c> defines them: its own marks when it marks any,
        ///     else the nearest inherited declaration's marks, clamped to its own parameter count.
        /// </summary>
        /// <param name="method">The method asked about.</param>
        /// <param name="known">The resolved NumSharp attribute types.</param>
        /// <param name="source">Receives the declaration the positions were inherited from, or null when they are the method's own (or there are none).</param>
        /// <returns>The positions, ascending; empty when nothing is retained.</returns>
        private static int[] AnalyzerExitPositions(IMethodSymbol method, RoslynKnown known, out IMethodSymbol source)
        {
            source = null;
            var own = RoslynScope.OwnExitParameters(method, known);
            if (own.Length > 0)
                return own;
            source = RoslynScope.InheritedExitSource(method, known);
            return source == null
                ? Array.Empty<int>()
                : RoslynScope.OwnExitParameters(source, known).Where(i => i < method.Parameters.Length).ToArray();
        }
    }
}
