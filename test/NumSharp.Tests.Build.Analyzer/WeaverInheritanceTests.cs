using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     The weaver's declaration-site INHERITANCE walk (<c>tools/NumSharp.Build/ScopeInheritance.cs</c>),
    ///     pinned in-process shape by shape: what inherits (class chains through non-declaring levels,
    ///     hiding vs overriding, sealed/abstract overrides, explicit-wins, generic instantiations and
    ///     open generic derived types, by-ref/pointer/params signatures, overload discrimination,
    ///     implicit/explicit/struct/generic/default-implementation interface members, base interfaces,
    ///     property-level and accessor-level attributes, covariant returns, async and iterator
    ///     overrides, inherited <c>[NDScopedExit]</c> parameters, nested types, cross-assembly
    ///     declarations), what must NOT (a <c>new</c> hiding member, an unscoped overload, an opted-out
    ///     override, an unresolvable base), the error shapes routed through inherited targets, and the
    ///     walk's scale. Every positive case also RUNS the woven code and asserts the dropped temporary
    ///     is reclaimed while the result survives.
    /// </summary>
    [TestClass]
    public class WeaverInheritanceTests
    {
        private const string Prelude =
            "using System;\nusing System.Collections.Generic;\nusing System.Threading.Tasks;\nusing NumSharp;\nusing NumSharp.Generic;\n";

        private static NDArray Input() => np.arange(3).astype(np.float64);

        private static void AssertReclaimed(WeaveRun run, string type, string field, string what)
        {
            var temp = (NDArray)run.GetStatic(type, field);
            Assert.IsNotNull(temp, $"{what}: the fixture did not record its temporary");
            Assert.IsTrue(temp.IsDisposed, $"{what}: the dropped temporary must be reclaimed by the woven scope");
        }

        // ---------------------------------------------------------------- class chains

        [TestMethod]
        public void AbstractContract_ThroughNonDeclaringLevel_OverrideWoven()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public abstract class Base { [NDScoped] public abstract NDArray Compute(NDArray a); }\n" +
                "public abstract class Mid : Base { }\n" +
                "public class Leaf : Mid { public static NDArray Last; public override NDArray Compute(NDArray a) { var t = a + 1.0; Last = t; return t * 2.0; } }\n",
                "Chain");
            run.AssertWoven(1);
            StringAssert.Contains(run.Stdout, "contract", "the abstract declaration is reported as a contract, not woven");
            StringAssert.Contains(run.Stdout, "inherited from Base::Compute", "the verbose line names the declaration");
            Assert.IsTrue(run.HasScopeLocal("Leaf", "Compute"));
            Assert.IsTrue(run.HasFinallyDispose("Leaf", "Compute"));
            Assert.AreEqual(1, run.CountScopeCalls("Leaf", "Compute", "Returns"));

            var result = (NDArray)run.Invoke("Leaf", "Compute", run.New("Leaf"), Input());
            CollectionAssert.AreEqual(new[] { 2.0, 4.0, 6.0 }, result.ToArray<double>());
            AssertReclaimed(run, "Leaf", "Last", "Leaf.Compute");
            Assert.IsFalse(result.IsDisposed);

            // Reflection agrees with the weaver (Inherited = true): the override reports the base's attribute.
            var m = run.LoadType("Leaf").GetMethod("Compute");
            Assert.IsNotNull(m.GetCustomAttribute<NDScopedAttribute>(inherit: true), "GetCustomAttribute(inherit: true) sees the contract on the override");
            Assert.IsNull(m.GetCustomAttribute<NDScopedAttribute>(inherit: false), "…and it is not the override's own");
        }

        [TestMethod]
        public void VirtualWithBody_BaseAndOverrideWoven_BaseCallComposes()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public class B { public static NDArray BaseTemp; [NDScoped] public virtual NDArray M(NDArray a) { var t = a + 1.0; BaseTemp = t; return t.copy(); } }\n" +
                "public class D : B { public static NDArray BaseResult; public override NDArray M(NDArray a) { var u = base.M(a); BaseResult = u; return u * 2.0; } }\n",
                "BaseCall");
            run.AssertWoven(2);
            Assert.IsTrue(run.HasScopeLocal("B", "M") && run.HasScopeLocal("D", "M"));

            var result = (NDArray)run.Invoke("D", "M", run.New("D"), Input());
            CollectionAssert.AreEqual(new[] { 2.0, 4.0, 6.0 }, result.ToArray<double>());
            AssertReclaimed(run, "B", "BaseTemp", "base's own temp (base scope)");
            AssertReclaimed(run, "D", "BaseResult", "base's result, dropped by the override (re-tracked into the override's scope by Returns)");
            Assert.IsFalse(result.IsDisposed);
        }

        [TestMethod]
        public void NewHidingMember_IsNotAnOverride_NotWoven()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public class B { [NDScoped] public virtual NDArray M(NDArray a) => a.copy(); [NDScoped] public virtual NDArray N(NDArray a) => a.copy(); }\n" +
                "public class D : B { public new NDArray M(NDArray a) { var t = a + 1.0; return t.copy(); } public new virtual NDArray N(NDArray a) { var t = a + 1.0; return t.copy(); } }\n",
                "Hiding");
            run.AssertWoven(2);
            Assert.IsTrue(run.HasScopeLocal("B", "M") && run.HasScopeLocal("B", "N"), "the attributed bodies");
            Assert.IsFalse(run.HasScopeLocal("D", "M"), "a non-virtual `new` member hides the slot — it inherits nothing");
            Assert.IsFalse(run.HasScopeLocal("D", "N"), "a `new virtual` starts a fresh slot (newslot) — it inherits nothing");
        }

        [TestMethod]
        public void SealedOverride_AndAbstractOverrideChain()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public abstract class A { [NDScoped] public virtual NDArray M(NDArray a) => a.copy(); [NDScoped] public virtual NDArray S(NDArray a) => a.copy(); }\n" +
                "public abstract class B : A { public abstract override NDArray M(NDArray a); }\n" +
                "public class C : B { public static NDArray Last; public override NDArray M(NDArray a) { var t = a + 1.0; Last = t; return t.copy(); } public sealed override NDArray S(NDArray a) { var t = a + 1.0; return t.copy(); } }\n",
                "Sealed");
            run.AssertWoven(4);
            Assert.IsTrue(run.HasScopeLocal("A", "M") && run.HasScopeLocal("A", "S"));
            Assert.IsFalse(run.HasScopeLocal("B", "M"), "an abstract override has no body — it passes the contract through");
            Assert.IsTrue(run.HasScopeLocal("C", "M"), "the leaf inherits through the abstract re-declaration");
            Assert.IsTrue(run.HasScopeLocal("C", "S"), "a sealed override inherits");
            run.Invoke("C", "M", run.New("C"), Input());
            AssertReclaimed(run, "C", "Last", "C.M");
        }

        [TestMethod]
        public void ExplicitAttributeOnOverride_Wins()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public abstract class B { [NDScoped] public abstract NDArray Scoped(NDArray a); public virtual NDArray Unscoped(NDArray a) => a.copy(); [NDScoped] public abstract NDArray Redundant(NDArray a); [NDScopedCovered] public virtual NDArray Covered(NDArray a) => a.copy(); }\n" +
                "public class D : B {\n" +
                "  [NDScopedCovered] public override NDArray Scoped(NDArray a) { var t = a + 1.0; return t.copy(); }\n" +   // opt-out
                "  [NDScoped] public override NDArray Unscoped(NDArray a) { var t = a + 1.0; return t.copy(); }\n" +       // own attribute
                "  [NDScoped] public override NDArray Redundant(NDArray a) { var t = a + 1.0; return t.copy(); }\n" +      // re-stated
                "  public override NDArray Covered(NDArray a) { var t = a + 1.0; return t.copy(); }\n" +                   // inherits Covered
                "}\n", "Explicit");
            run.AssertWoven(2);
            Assert.IsFalse(run.HasScopeLocal("D", "Scoped"), "[NDScopedCovered] on the override opts out of the inherited [NDScoped]");
            Assert.IsTrue(run.HasScopeLocal("D", "Unscoped"), "an own [NDScoped] on the override of an unscoped base weaves");
            Assert.IsTrue(run.HasScopeLocal("D", "Redundant"), "a re-stated attribute weaves exactly once");
            Assert.AreEqual(1, run.CountScopeCalls("D", "Redundant", "Open"));
            Assert.IsFalse(run.HasScopeLocal("D", "Covered"), "an inherited [NDScopedCovered] is not a weave target");
            Assert.IsFalse(run.HasScopeLocal("B", "Covered"));
        }

        // ---------------------------------------------------------------- signatures & generics

        [TestMethod]
        public void GenericBases_InstantiationsOpenDerivedAndNestedArguments()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public abstract class GBase<T> {\n" +
                "  [NDScoped] public virtual NDArray M(T x, NDArray a) => a.copy();\n" +
                "  [NDScoped] public abstract NDArray Arr(T[] xs, NDArray a);\n" +
                "  [NDScoped] public abstract NDArray Gen<U>(U u, T t, NDArray a);\n" +
                "  public virtual NDArray Plain(T x, NDArray a) => a.copy();\n" +
                "  [NDScoped] public abstract NDArray Ovl(int x, NDArray a);\n" +
                "  public abstract NDArray Ovl(double x, NDArray a);\n" +
                "}\n" +
                "public class GInt : GBase<int> {\n" +
                "  public static NDArray Last;\n" +
                "  public override NDArray M(int x, NDArray a) { var t = a + x; Last = t; return t.copy(); }\n" +
                "  public override NDArray Arr(int[] xs, NDArray a) { var t = a + xs.Length; return t.copy(); }\n" +
                "  public override NDArray Gen<U>(U u, int t, NDArray a) { var q = a + t; return q.copy(); }\n" +
                "  public override NDArray Plain(int x, NDArray a) { var t = a + x; return t.copy(); }\n" +
                "  public override NDArray Ovl(int x, NDArray a) { var t = a + x; return t.copy(); }\n" +
                "  public override NDArray Ovl(double x, NDArray a) { var t = a + x; return t.copy(); }\n" +
                "}\n" +
                "public class GOpen<V> : GBase<V> { public override NDArray M(V x, NDArray a) { var t = a + 1.0; return t.copy(); } public override NDArray Arr(V[] xs, NDArray a) => a.copy(); public override NDArray Gen<U>(U u, V t, NDArray a) => a.copy(); public override NDArray Ovl(int x, NDArray a) => a.copy(); public override NDArray Ovl(double x, NDArray a) => a.copy(); }\n" +
                "public class GList : GBase<List<int>> { public override NDArray M(List<int> x, NDArray a) { var t = a + x.Count; return t.copy(); } public override NDArray Arr(List<int>[] xs, NDArray a) => a.copy(); public override NDArray Gen<U>(U u, List<int> t, NDArray a) => a.copy(); public override NDArray Ovl(int x, NDArray a) => a.copy(); public override NDArray Ovl(double x, NDArray a) => a.copy(); }\n",
                "Generics");
            // GBase<T>.M (body) + GInt: M, Arr, Gen, Ovl(int) + GOpen: M, Arr, Gen, Ovl(int) + GList: M, Arr, Gen, Ovl(int) = 13
            run.AssertWoven(13);
            Assert.IsTrue(run.HasScopeLocal("GBase`1", "M"));
            Assert.IsTrue(run.HasScopeLocal("GInt", "M") && run.HasScopeLocal("GInt", "Arr") && run.HasScopeLocal("GInt", "Gen"));
            Assert.IsTrue(OverloadWoven(run, "GInt", "Int32"), "Ovl(int) inherits the scoped overload");
            Assert.IsFalse(OverloadWoven(run, "GInt", "Double"), "Ovl(double) overrides the UNSCOPED overload — signature matching must not conflate the two");
            Assert.IsFalse(run.HasScopeLocal("GInt", "Plain"), "the unscoped generic member inherits nothing");
            Assert.IsTrue(run.HasScopeLocal("GOpen`1", "M"), "an open generic derived type (Derived<V> : Base<V>) inherits");
            Assert.IsTrue(run.HasScopeLocal("GList", "M"), "a nested generic instantiation (Base<List<int>>) inherits");

            var result = (NDArray)run.Invoke("GInt", "M", run.New("GInt"), 10, Input());
            CollectionAssert.AreEqual(new[] { 10.0, 11.0, 12.0 }, result.ToArray<double>());
            AssertReclaimed(run, "GInt", "Last", "GInt.M");
        }

        private static bool OverloadWoven(WeaveRun run, string type, string firstParamTypeName)
        {
            using var asm = run.ReadCecil();
            var m = WeaveRun.AllMethods(asm.MainModule).Single(x => x.DeclaringType.FullName == type && x.Name == "Ovl" && x.Parameters[0].ParameterType.Name == firstParamTypeName);
            return WeaveRun.HasScopeLocal(m);
        }

        [TestMethod]
        public void ByRefPointerParamsAndOptionalSignatures_Match()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public abstract class R {\n" +
                "  [NDScoped] public abstract NDArray InP(in int x, NDArray a);\n" +
                "  [NDScoped] public abstract NDArray RefP(ref double x, NDArray a);\n" +
                "  [NDScoped] public abstract NDArray OutP(out int x, NDArray a);\n" +
                "  [NDScoped] public abstract unsafe NDArray Ptr(int* p, NDArray a);\n" +
                "  [NDScoped] public abstract NDArray Par(NDArray a, params int[] xs);\n" +
                "  [NDScoped] public abstract NDArray Opt(NDArray a, int x = 3);\n" +
                "}\n" +
                "public class RD : R {\n" +
                "  public override NDArray InP(in int x, NDArray a) { var t = a + x; return t.copy(); }\n" +
                "  public override NDArray RefP(ref double x, NDArray a) { x += 1; var t = a + x; return t.copy(); }\n" +
                "  public override NDArray OutP(out int x, NDArray a) { x = 7; var t = a + x; return t.copy(); }\n" +
                "  public override unsafe NDArray Ptr(int* p, NDArray a) { var t = a + *p; return t.copy(); }\n" +
                "  public override NDArray Par(NDArray a, params int[] xs) { var t = a + xs.Length; return t.copy(); }\n" +
                "  public override NDArray Opt(NDArray a, int x = 5) { var t = a + x; return t.copy(); }\n" +
                "}\n", "ByRef");
            run.AssertWoven(6);
            foreach (var name in new[] { "InP", "RefP", "OutP", "Ptr", "Par", "Opt" })
                Assert.IsTrue(run.HasScopeLocal("RD", name), $"RD.{name} inherits through a {name} signature");
        }

        // ---------------------------------------------------------------- interfaces

        [TestMethod]
        public void InterfaceMembers_ImplicitExplicitStructGenericDefaultAndBaseInterface()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public interface IOp { [NDScoped] NDArray Apply(NDArray a); [NDScoped] NDArray Value { get; } NDArray Plain(NDArray a); }\n" +
                "public interface IOp2 : IOp { }\n" +
                "public class Implicit : IOp2 { public static NDArray Last; public NDArray Apply(NDArray a) { var t = a + 1.0; Last = t; return t.copy(); } public NDArray Value { get { var t = np.arange(2); return t.copy(); } } public NDArray Plain(NDArray a) { var t = a + 1.0; return t.copy(); } }\n" +
                "public class Explicit : IOp { NDArray IOp.Apply(NDArray a) { var t = a + 1.0; return t.copy(); } NDArray IOp.Value { get { var t = np.arange(2); return t.copy(); } } NDArray IOp.Plain(NDArray a) { var t = a + 1.0; return t.copy(); } }\n" +
                "public struct SImpl : IOp { public NDArray Apply(NDArray a) { var t = a + 1.0; return t.copy(); } public NDArray Value { get { var t = np.arange(2); return t.copy(); } } public NDArray Plain(NDArray a) { var t = a + 1.0; return t.copy(); } }\n" +
                "public interface IGen<T> { [NDScoped] NDArray Map(T x, NDArray a); }\n" +
                "public class GImpl<T> : IGen<T> { NDArray IGen<T>.Map(T x, NDArray a) { var t = a + 1.0; return t.copy(); } }\n" +
                "public class GImplInt : IGen<int> { public NDArray Map(int x, NDArray a) { var t = a + x; return t.copy(); } }\n" +
                "public interface IDim { [NDScoped] NDArray Twice(NDArray a) { var t = a * 2.0; return t.copy(); } }\n" +
                "public class DimUser : IDim { }\n" +
                "public class DimOverride : IDim { public NDArray Twice(NDArray a) { var t = a * 3.0; return t.copy(); } }\n" +
                "public interface IHide : IOp { new NDArray Apply(NDArray a); }\n" +
                "public class HideImpl : IHide { public NDArray Apply(NDArray a) { var t = a + 1.0; return t.copy(); } public NDArray Value => np.arange(2); public NDArray Plain(NDArray a) => a.copy(); }\n",
                "Ifaces");
            // Implicit 2 + Explicit 2 + SImpl 2 + GImpl 1 + GImplInt 1 + IDim.Twice 1 + DimOverride 1 + HideImpl (Apply + get_Value) 2 = 12
            run.AssertWoven(12);
            Assert.IsTrue(run.HasScopeLocal("HideImpl", "get_Value"), "an expression-bodied getter implementing the scoped interface property inherits too");
            Assert.IsTrue(run.HasScopeLocal("Implicit", "Apply") && run.HasScopeLocal("Implicit", "get_Value"), "implicit implementation through a derived interface");
            Assert.IsFalse(run.HasScopeLocal("Implicit", "Plain"), "the unscoped interface member inherits nothing");
            Assert.IsTrue(run.HasScopeLocal("Explicit", "IOp.Apply") && run.HasScopeLocal("Explicit", "IOp.get_Value"), "explicit implementations (the .override table)");
            Assert.IsFalse(run.HasScopeLocal("Explicit", "IOp.Plain"));
            Assert.IsTrue(run.HasScopeLocal("SImpl", "Apply") && run.HasScopeLocal("SImpl", "get_Value"), "a struct's implicit implementations");
            Assert.IsTrue(run.HasScopeLocal("GImpl`1", "IGen<T>.Map"), "an explicit implementation of a generic interface in a generic class");
            Assert.IsTrue(run.HasScopeLocal("GImplInt", "Map"), "an implicit implementation of an instantiated generic interface");
            Assert.IsTrue(run.HasScopeLocal("IDim", "Twice"), "a default interface implementation with a body is woven in the interface");
            Assert.IsTrue(run.HasScopeLocal("DimOverride", "Twice"), "a class overriding the default implementation inherits");
            Assert.IsTrue(run.HasScopeLocal("HideImpl", "Apply"), "the implementation of a hiding re-declaration also implements the scoped base-interface member");

            run.Invoke("Implicit", "Apply", run.New("Implicit"), Input());
            AssertReclaimed(run, "Implicit", "Last", "Implicit.Apply");
            var viaStruct = (NDArray)run.Invoke("SImpl", "Apply", run.New("SImpl"), Input());
            CollectionAssert.AreEqual(new[] { 1.0, 2.0, 3.0 }, viaStruct.ToArray<double>());
            var dim = (NDArray)run.Invoke("IDim", "Twice", run.New("DimUser"), Input());
            CollectionAssert.AreEqual(new[] { 0.0, 2.0, 4.0 }, dim.ToArray<double>(), "the woven default implementation runs for a class that does not override it");
        }

        [TestMethod]
        [TestCategory("KnownLimitation")]
        public void InterfaceListedOnDerived_ImplementedByBaseMethod_IsNotWoven_KnownLimitation()
        {
            // The CLR maps IOp.Apply onto A.Apply when B lists IOp, but the walk only checks the
            // interfaces of the DECLARING type and its bases — A never lists IOp — so A.Apply is left
            // unwoven. Pinned so a change (in either direction) is noticed; the analyzer twin classifies
            // it the same way (WeaverAnalyzerParityTests), so at least the two layers agree.
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public interface IOp { [NDScoped] NDArray Apply(NDArray a); }\n" +
                "public class A { public virtual NDArray Apply(NDArray a) { var t = a + 1.0; return t.copy(); } }\n" +
                "public class B : A, IOp { }\n", "IfaceOnDerived");
            run.AssertWoven(0);
            Assert.IsFalse(run.HasScopeLocal("A", "Apply"));
        }

        // ---------------------------------------------------------------- properties & covariance

        [TestMethod]
        public void Properties_PropertyLevelAccessorLevel_GetterOnly()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public abstract class P { [NDScoped] public abstract NDArray V1 { get; } public abstract NDArray V2 { [NDScoped] get; } [NDScoped] public virtual NDArray V3 { get => np.arange(3); set { } } }\n" +
                "public class PD : P { public static NDArray Last;\n" +
                "  public override NDArray V1 { get { var t = np.arange(3) + 1.0; Last = t; return t.copy(); } }\n" +
                "  public override NDArray V2 { get { var t = np.arange(3); return t.copy(); } }\n" +
                "  public override NDArray V3 { get { var t = np.arange(3); return t.copy(); } set { var t = value + 1.0; } }\n" +
                "}\n", "Props");
            // P.get_V3 (body) + PD.get_V1 + PD.get_V2 + PD.get_V3 = 4; no setter
            run.AssertWoven(4);
            Assert.IsTrue(run.HasScopeLocal("PD", "get_V1"), "property-level attribute on the base property reaches the getter override");
            Assert.IsTrue(run.HasScopeLocal("PD", "get_V2"), "accessor-level attribute on the base getter reaches the getter override");
            Assert.IsTrue(run.HasScopeLocal("PD", "get_V3") && run.HasScopeLocal("P", "get_V3"));
            Assert.IsFalse(run.HasScopeLocal("PD", "set_V3"), "a setter never inherits a property-level attribute");
            Assert.IsFalse(run.HasScopeLocal("P", "set_V3"));

            var v1 = (NDArray)run.GetProperty("PD", "V1", run.New("PD"));
            CollectionAssert.AreEqual(new[] { 1.0, 2.0, 3.0 }, v1.ToArray<double>());
            AssertReclaimed(run, "PD", "Last", "PD.V1 getter");
        }

        [TestMethod]
        public void CovariantReturnOverride_ReachesTheBaseThroughTheOverrideTable()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public class B { [NDScoped] public virtual NDArray M(NDArray a) => a.copy(); }\n" +
                "public class D : B { public static NDArray Last; public override NDArray<double> M(NDArray a) { var t = a + 1.0; Last = t; return (t * 2.0).MakeGeneric<double>(); } }\n",
                "Covariant");
            run.AssertWoven(2);
            Assert.IsTrue(run.HasScopeLocal("D", "M"), "a C# 9 covariant-return override is pinned to its base slot by a .override entry");
            var result = (NDArray)run.Invoke("D", "M", run.New("D"), Input());
            CollectionAssert.AreEqual(new[] { 2.0, 4.0, 6.0 }, result.ToArray<double>());
            AssertReclaimed(run, "D", "Last", "D.M (covariant)");
            Assert.IsFalse(result.IsDisposed);
        }

        // ---------------------------------------------------------------- async / iterators

        [TestMethod]
        public void AsyncAndIteratorOverrides_WeaveThroughTheirOwnStateMachines()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public abstract class AB { [NDScopedAsync] public abstract Task<NDArray> Async(NDArray a); [NDScopedAsync] public abstract Task<NDArray> Deferred(NDArray a); [NDScoped] public abstract IEnumerable<NDArray> Items(NDArray a); }\n" +
                "public class AD : AB { public static NDArray T1, T2, T3;\n" +
                "  public override async Task<NDArray> Async(NDArray a) { var t = a + 1.0; T1 = t; await Task.Yield(); return t * 2.0; }\n" +
                "  public override Task<NDArray> Deferred(NDArray a) { var t = a + 2.0; T2 = t; return Task.FromResult(t - 1.0); }\n" +
                "  public override IEnumerable<NDArray> Items(NDArray a) { var t = a + 1.0; T3 = t; yield return t + 1.0; yield return t * 3.0; }\n" +
                "}\n", "AsyncIter");
            run.AssertWoven(3);
            Assert.IsTrue(run.HasScopeLocal("AD", "Async"), "the async override's MoveNext carries the scope");
            Assert.IsTrue(run.HasScopeLocal("AD", "Deferred"), "the non-async Task override gets the deferral egress");
            Assert.IsTrue(run.HasScopeLocal("AD", "Items"), "the iterator override's MoveNext carries the scope");

            var inst = run.New("AD");
            var async = ((Task<NDArray>)run.Invoke("AD", "Async", inst, Input())).GetAwaiter().GetResult();
            CollectionAssert.AreEqual(new[] { 2.0, 4.0, 6.0 }, async.ToArray<double>());
            AssertReclaimed(run, "AD", "T1", "async override temp (reclaimed at completion)");
            Assert.IsFalse(async.IsDisposed);

            var deferred = ((Task<NDArray>)run.Invoke("AD", "Deferred", inst, Input())).GetAwaiter().GetResult();
            CollectionAssert.AreEqual(new[] { 1.0, 2.0, 3.0 }, deferred.ToArray<double>());
            AssertReclaimed(run, "AD", "T2", "non-async Task override temp");
            Assert.IsFalse(deferred.IsDisposed);

            var items = ((IEnumerable<NDArray>)run.Invoke("AD", "Items", inst, Input())).ToList();
            Assert.AreEqual(2, items.Count);
            CollectionAssert.AreEqual(new[] { 2.0, 3.0, 4.0 }, items[0].ToArray<double>());
            CollectionAssert.AreEqual(new[] { 3.0, 6.0, 9.0 }, items[1].ToArray<double>());
            AssertReclaimed(run, "AD", "T3", "iterator override temp (reclaimed at the final MoveNext)");
            Assert.IsFalse(items[0].IsDisposed || items[1].IsDisposed, "yielded elements survive");
        }

        [TestMethod]
        public void NonIteratorOverride_OfAnIteratorContract_IsRejectedWithProvenance()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public abstract class AB { [NDScoped] public abstract IEnumerable<NDArray> Items(NDArray a); }\n" +
                "public class AD : AB { public override IEnumerable<NDArray> Items(NDArray a) => new List<NDArray> { a + 1.0 }; }\n",
                "IterReject");
            run.AssertRejected("NDW003");
            StringAssert.Contains(run.Stderr, "inherited from AB::Items", "the rejection names the declaration the attribute came from");
        }

        // ---------------------------------------------------------------- [NDScopedExit]

        [TestMethod]
        public void ExitParameters_InheritByPosition_OwnMarksWin()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public abstract class Keeper { public NDArray Kept; public abstract void Adopt([NDScopedExit] NDArray w); public abstract void AdoptSecond(NDArray first, [NDScopedExit] NDArray second); }\n" +
                "public class K1 : Keeper { public override void Adopt(NDArray w) => Kept = w; public override void AdoptSecond(NDArray first, NDArray second) => Kept = second; }\n" +
                "public class K2 : Keeper { public override void Adopt([NDScopedExit] NDArray w) => Kept = w; public override void AdoptSecond([NDScopedExit] NDArray first, NDArray second) => Kept = first; }\n" +
                "public interface IKeep { void Keep([NDScopedExit] NDArray w); }\n" +
                "public class K3 : IKeep { public NDArray Kept; public void Keep(NDArray w) => Kept = w; }\n" +
                "public abstract class GK<T> { public abstract void Keep(T x, [NDScopedExit] NDArray w); }\n" +
                "public class GK1 : GK<int> { public NDArray Kept; public override void Keep(int x, NDArray w) => Kept = w; }\n" +
                "public abstract class Both { [NDScoped] public abstract NDArray Take([NDScopedExit] NDArray w, NDArray a); }\n" +
                "public class Both1 : Both { public NDArray Kept; public static NDArray Last; public override NDArray Take(NDArray w, NDArray a) { Kept = w; var t = a + 1.0; Last = t; return t.copy(); } }\n",
                "Exit");
            run.AssertWoven(7);
            foreach (var (type, method) in new[] { ("K1", "Adopt"), ("K1", "AdoptSecond"), ("K2", "Adopt"), ("K2", "AdoptSecond"), ("K3", "Keep"), ("GK1", "Keep"), ("Both1", "Take") })
                Assert.IsTrue(run.CallsDetach(type, method), $"{type}.{method} must detach its retained argument");
            Assert.IsTrue(run.HasScopeLocal("Both1", "Take"), "a method inheriting BOTH a scope and an exit parameter gets both");
            StringAssert.Contains(run.Stdout, "inherited from Keeper::Adopt", "the verbose line names the exit parameter's source");

            // K1.AdoptSecond inherits position 1 (second); K2.AdoptSecond marks position 0 (first) itself — own marks win.
            var k1 = run.New("K1");
            var k2 = run.New("K2");
            NDArray k1Second, k2First, k2Second;
            using (NDScope.Open())
            {
                var w1 = Input() + 100.0;
                run.Invoke("K1", "Adopt", k1, w1);
                var f1 = Input() + 1.0;
                k1Second = Input() + 2.0;
                run.Invoke("K1", "AdoptSecond", k1, f1, k1Second);
                k2First = Input() + 3.0;
                k2Second = Input() + 4.0;
                run.Invoke("K2", "AdoptSecond", k2, k2First, k2Second);
                GC.KeepAlive(f1);
            }

            Assert.IsFalse(k1Second.IsDisposed, "K1.AdoptSecond: the INHERITED position (second) is detached and survives the caller's scope");
            Assert.IsFalse(k2First.IsDisposed, "K2.AdoptSecond: the override's OWN mark (first) is detached");
            Assert.IsTrue(k2Second.IsDisposed, "K2.AdoptSecond: the base's position (second) is NOT inherited once the override marks its own — reclaimed");
            Assert.IsFalse(((NDArray)run.GetField(k1, "Kept")).IsDisposed);

            NDArray kept3, kept4, keptBoth;
            using (NDScope.Open())
            {
                var w3 = Input() + 5.0;
                run.Invoke("K3", "Keep", run.New("K3"), w3);
                kept3 = w3;
                var w4 = Input() + 6.0;
                run.Invoke("GK1", "Keep", run.New("GK1"), 1, w4);
                kept4 = w4;
                var w5 = Input() + 7.0;
                run.Invoke("Both1", "Take", run.New("Both1"), w5, Input());
                keptBoth = w5;
            }

            Assert.IsFalse(kept3.IsDisposed, "interface [NDScopedExit] inherited by the implicit implementation");
            Assert.IsFalse(kept4.IsDisposed, "generic base [NDScopedExit] inherited under instantiation");
            Assert.IsFalse(keptBoth.IsDisposed, "retained argument of a method that also inherits its scope");
            AssertReclaimed(run, "Both1", "Last", "Both1.Take's own temp");
        }

        // ---------------------------------------------------------------- errors through inherited targets

        [TestMethod]
        public void BothAttributesOnDeclaration_ReportedOnce_NotPropagated()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public class B { [NDScoped] [NDScopedAsync] public virtual Task<NDArray> M(NDArray a) => Task.FromResult(a.copy()); }\n" +
                "public class D : B { public override Task<NDArray> M(NDArray a) => Task.FromResult(a.copy()); }\n", "Both");
            run.AssertRejected("NDW011");
            Assert.AreEqual(1, Regex.Matches(run.Stderr, "error NDW011").Count, "the declaration's error is not repeated for the override:\n" + run.Report);
            Assert.AreEqual(1, run.Result.Errors);
        }

        [TestMethod]
        public void HiddenEgressOnTheContract_IsRejectedAtTheOverride()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public abstract class B { [NDScoped] public abstract void RefEgress(ref NDArray a); }\n" +
                "public class D : B { public override void RefEgress(ref NDArray a) { a = a + 1.0; } }\n", "RefEgress");
            run.AssertRejected("NDW002");
            StringAssert.Contains(run.Stderr, "inherited from B::RefEgress");
        }

        [TestMethod]
        public void SetterOnlyPropertyContract_IsNDW006_AtTheDeclaration()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public abstract class B { [NDScoped] public abstract NDArray V { set; } }\n" +
                "public class D : B { public override NDArray V { set { var t = value + 1.0; } } }\n", "SetterOnly");
            run.AssertRejected("NDW006");
            Assert.IsFalse(run.HasScopeLocal("D", "set_V"));
        }

        [TestMethod]
        public void WrongModelOnTheContract_IsRejectedOnDeclarationAndOverride()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public class B { [NDScoped] public virtual Task<NDArray> M(NDArray a) => Task.FromResult(a.copy()); }\n" +
                "public class D : B { public override async Task<NDArray> M(NDArray a) { await Task.Yield(); return a.copy(); } }\n", "WrongModel");
            run.AssertRejected("NDW009");
            Assert.AreEqual(2, Regex.Matches(run.Stderr, "error NDW009").Count, "the declaration (Task-returning) and the async override are each rejected under [NDScoped]:\n" + run.Report);
            StringAssert.Contains(run.Stderr, "inherited from B::M");
        }

        // ---------------------------------------------------------------- across an assembly boundary

        private const string BaseLibrary = Prelude +
            "namespace Lib {\n" +
            "  public abstract class Base { [NDScoped] public abstract NDArray Compute(NDArray a); [NDScoped] public virtual NDArray Twice(NDArray a) => a * 2.0; [NDScopedAsync] public abstract Task<NDArray> ComputeAsync(NDArray a); [NDScoped] public abstract NDArray Value { get; } public virtual NDArray Plain(NDArray a) => a.copy(); public abstract void Adopt([NDScopedExit] NDArray w); }\n" +
            "  public interface IOp { [NDScoped] NDArray Apply(NDArray a); }\n" +
            "  public abstract class GBase<T> { [NDScoped] public virtual NDArray Map(T x, NDArray a) => a.copy(); }\n" +
            "}\n";

        private const string Consumer = Prelude +
            "public class Derived : Lib.Base, Lib.IOp { public static NDArray Last; public NDArray Kept;\n" +
            "  public override NDArray Compute(NDArray a) { var t = a + 1.0; Last = t; return t.copy(); }\n" +
            "  public override NDArray Twice(NDArray a) { var t = a * 3.0; return t.copy(); }\n" +
            "  public override async Task<NDArray> ComputeAsync(NDArray a) { var t = a + 1.0; await Task.Yield(); return t.copy(); }\n" +
            "  public override NDArray Value { get { var t = np.arange(3); return t.copy(); } }\n" +
            "  public override NDArray Plain(NDArray a) { var t = a + 1.0; return t.copy(); }\n" +
            "  public NDArray Apply(NDArray a) { var t = a - 1.0; return t.copy(); }\n" +
            "  public override void Adopt(NDArray w) => Kept = w;\n" +
            "}\n" +
            "public class Leaf : Lib.GBase<int> { public override NDArray Map(int x, NDArray a) { var t = a + x; return t.copy(); } }\n";

        [TestMethod]
        public void CrossAssembly_DeclarationsInAReferencedLibrary_AreInherited()
        {
            var lib = WeaverTestHarness.CompileAndWeave(BaseLibrary, "XLib").AssertWoven(2); // Twice + GBase<T>.Map
            var run = WeaverTestHarness.CompileAndWeave(Consumer, "XApp", true, lib.Fixture.DllPath);
            run.AssertWoven(7); // Compute, Twice, ComputeAsync, get_Value, Apply, Leaf.Map + Adopt's detach
            foreach (var name in new[] { "Compute", "Twice", "ComputeAsync", "get_Value", "Apply" })
                Assert.IsTrue(run.HasScopeLocal("Derived", name), $"Derived.{name} inherits from the referenced library");
            Assert.IsTrue(run.HasScopeLocal("Leaf", "Map"));
            Assert.IsFalse(run.HasScopeLocal("Derived", "Plain"), "the unscoped library member inherits nothing");
            Assert.IsTrue(run.CallsDetach("Derived", "Adopt"));
            StringAssert.Contains(run.Stdout, "inherited from Lib.Base::Compute");

            var inst = run.New("Derived");
            var result = (NDArray)run.Invoke("Derived", "Compute", inst, Input());
            CollectionAssert.AreEqual(new[] { 1.0, 2.0, 3.0 }, result.ToArray<double>());
            AssertReclaimed(run, "Derived", "Last", "Derived.Compute");
            NDArray kept;
            using (NDScope.Open())
            {
                kept = Input() + 100.0;
                run.Invoke("Derived", "Adopt", inst, kept);
            }

            Assert.IsFalse(kept.IsDisposed, "the retained argument survives the caller's scope (inherited exit parameter across assemblies)");
        }

        [TestMethod]
        public void CrossAssembly_UnresolvableBase_EndsTheWalkGracefully()
        {
            var lib = WeaverTestHarness.Compile(BaseLibrary, "XLibMissing");
            var app = WeaverTestHarness.Compile(Consumer, "XAppMissing", true, lib.DllPath);
            // Weave WITHOUT the base library in the reference list (and not beside the assembly):
            // every base-chain walk into Lib.Base dead-ends. Nothing must throw, nothing is a target,
            // the file is untouched — the pre-inheritance behaviour, not a crash.
            var detached = new CompiledFixture(app.AssemblyName, app.DllPath, app.Directory, ImmutableArray<string>.Empty, true);
            var run = WeaverTestHarness.Weave(detached);
            Assert.AreEqual(0, run.Result.Errors, run.Report);
            Assert.AreEqual(0, run.Result.Woven, "an unresolvable declaration cannot be inherited from");
            Assert.IsFalse(run.Rewrote);
        }

        // ---------------------------------------------------------------- structure & scale

        [TestMethod]
        public void NestedTypes_AreWalkedLikeTopLevelOnes()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public static class Outer { public abstract class Base { [NDScoped] public abstract NDArray M(NDArray a); }\n" +
                "  public class Derived : Base { public static NDArray Last; public override NDArray M(NDArray a) { var t = a + 1.0; Last = t; return t.copy(); } }\n" +
                "  public class Grand : Derived { public override NDArray M(NDArray a) { var t = a + 2.0; return t.copy(); } } }\n", "Nested");
            run.AssertWoven(2);
            Assert.IsTrue(run.HasScopeLocal("Outer/Derived", "M") && run.HasScopeLocal("Outer/Grand", "M"));
            run.Invoke("Outer/Derived", "M", run.New("Outer/Derived"), Input());
            AssertReclaimed(run, "Outer/Derived", "Last", "Outer.Derived.M");
        }

        [TestMethod]
        public void SecondWeave_OfInheritedTargets_IsIdempotent()
        {
            var fixture = WeaverTestHarness.Compile(Prelude +
                "public abstract class Base { [NDScoped] public abstract NDArray M(NDArray a); public abstract void Adopt([NDScopedExit] NDArray w); }\n" +
                "public class D : Base { public NDArray Kept; public override NDArray M(NDArray a) { var t = a + 1.0; return t.copy(); } public override void Adopt(NDArray w) => Kept = w; }\n", "Idem");
            var first = WeaverTestHarness.Weave(fixture).AssertWoven(2);
            var second = WeaverTestHarness.Weave(fixture);
            Assert.AreEqual(0, second.Result.Errors, second.Report);
            Assert.AreEqual(0, second.Result.Woven);
            Assert.AreEqual(2, second.Result.Skipped, "both inherited targets are recognised as already woven");
            Assert.IsFalse(second.Rewrote);
            Assert.AreEqual(1, second.CountScopeCalls("D", "M", "Open"));
            Assert.IsTrue(first.Rewrote);
        }

        [TestMethod]
        [Timeout(120000)]
        public void DeepAndWideHierarchies_WeaveCompletely_InBoundedTime()
        {
            const int depth = 12, width = 40;
            var sb = new StringBuilder(Prelude);
            sb.Append("public class L0 {");
            for (int j = 0; j < width; j++)
                sb.Append($" [NDScoped] public virtual NDArray M{j}(NDArray a) {{ var t = a + {j}.0; return t.copy(); }}");
            sb.Append(" }\n");
            for (int i = 1; i < depth; i++)
            {
                sb.Append($"public class L{i} : L{i - 1} {{");
                for (int j = 0; j < width; j++)
                    sb.Append($" public override NDArray M{j}(NDArray a) {{ var t = a + {i}.0; return t.copy(); }}");
                sb.Append(" }\n");
            }

            var sw = Stopwatch.StartNew();
            var run = WeaverTestHarness.CompileAndWeave(sb.ToString(), "Scale");
            sw.Stop();
            run.AssertWoven(depth * width);
            Assert.IsTrue(run.HasScopeLocal($"L{depth - 1}", $"M{width - 1}"), "the deepest override of the last member inherits through every level");
            Assert.IsTrue(sw.ElapsedMilliseconds < 60_000, $"{depth * width} inherited targets took {sw.ElapsedMilliseconds} ms");
        }

        [TestMethod]
        public void DeepInterfaceChain_ReachesTheRootDeclaration()
        {
            const int depth = 20;
            var sb = new StringBuilder(Prelude);
            sb.AppendLine("public interface I0 { [NDScoped] NDArray Apply(NDArray a); }");
            for (int i = 1; i < depth; i++)
                sb.AppendLine($"public interface I{i} : I{i - 1} {{ }}");
            sb.AppendLine($"public class Impl : I{depth - 1} {{ public static NDArray Last; public NDArray Apply(NDArray a) {{ var t = a + 1.0; Last = t; return t.copy(); }} }}");
            sb.AppendLine($"public class ExplicitImpl : I{depth - 1} {{ NDArray I0.Apply(NDArray a) {{ var t = a + 1.0; return t.copy(); }} }}");
            var run = WeaverTestHarness.CompileAndWeave(sb.ToString(), "DeepIface");
            run.AssertWoven(2);
            Assert.IsTrue(run.HasScopeLocal("Impl", "Apply"), "the implicit implementation inherits through 19 interface levels");
            Assert.IsTrue(run.HasScopeLocal("ExplicitImpl", "I0.Apply"), "the explicit implementation names the root member directly");
            run.Invoke("Impl", "Apply", run.New("Impl"), Input());
            AssertReclaimed(run, "Impl", "Last", "Impl.Apply");
        }

        [TestMethod]
        [TestCategory("KnownLimitation")]
        public void StaticAbstractInterfaceMember_IsNotInherited_Pinned()
        {
            // A static abstract interface member (C# 11) has no instance slot: the walk considers only
            // virtual instance methods, so its static implementation inherits nothing. Pinned: a static
            // implementation that wants scoping carries the attribute itself.
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public interface IFactory<TSelf> where TSelf : IFactory<TSelf> { [NDScoped] static abstract NDArray Create(NDArray a); }\n" +
                "public class F : IFactory<F> { public static NDArray Create(NDArray a) { var t = a + 1.0; return t.copy(); } }\n" +
                "public class G : IFactory<G> { [NDScoped] public static NDArray Create(NDArray a) { var t = a + 1.0; return t.copy(); } }\n", "StaticAbstract");
            run.AssertWoven(1);
            Assert.IsFalse(run.HasScopeLocal("F", "Create"), "a static implementation does not inherit the static abstract member's attribute");
            Assert.IsTrue(run.HasScopeLocal("G", "Create"), "…but its own attribute weaves it");
        }

        [TestMethod]
        public void EveryMethodWithAnInheritedAttribute_HasAScopeLocal_CoverageInvariant()
        {
            // The NDScopeWeaveTests invariant, applied to a hierarchy-heavy fixture: reflection's
            // inherit:true view of the attribute and the weaver's scope local must agree on every
            // method with a body (abstract declarations excluded; [NDScopedCovered] overrides excluded).
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public interface IOp { [NDScoped] NDArray Apply(NDArray a); }\n" +
                "public abstract class B { [NDScoped] public abstract NDArray M(NDArray a); [NDScoped] public virtual NDArray N(NDArray a) => a.copy(); public virtual NDArray P(NDArray a) => a.copy(); }\n" +
                "public class D : B, IOp { public override NDArray M(NDArray a) => a.copy(); public override NDArray N(NDArray a) => a.copy(); public override NDArray P(NDArray a) => a.copy(); public NDArray Apply(NDArray a) => a.copy(); }\n" +
                "public class E : D { [NDScopedCovered] public override NDArray M(NDArray a) => a.copy(); public override NDArray N(NDArray a) => a.copy(); }\n", "Coverage");
            run.AssertWoven(5); // B.N, D.M, D.N, E.N (+ D.Apply)
            var asm = run.Load();
            const BindingFlags all = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;
            int checkedMethods = 0;
            using var cecil = run.ReadCecil();
            foreach (var type in asm.GetTypes())
            foreach (var m in type.GetMethods(all))
            {
                if (m.IsAbstract || m.IsSpecialName)
                    continue;
                bool expected = m.GetCustomAttribute<NDScopedCoveredAttribute>(inherit: false) == null &&
                                m.GetCustomAttribute<NDScopedAttribute>(inherit: true) != null;
                // Interface implementations are NOT reachable through reflection's inherit:true (it walks
                // class overrides only), so the interface case is asserted separately.
                if (m.Name == "Apply")
                    expected = true;
                var def = WeaveRun.Find(cecil.MainModule, type.FullName.Replace('+', '/'), m.Name, m.GetParameters().Length);
                Assert.AreEqual(expected, WeaveRun.HasScopeLocal(def), $"{type.Name}.{m.Name}: reflection-inherited attribute vs woven scope local");
                checkedMethods++;
            }

            Assert.IsTrue(checkedMethods >= 8, $"the invariant must have covered the hierarchy (checked {checkedMethods})");
        }
    }
}
