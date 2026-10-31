using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Mono.Cecil;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     Which interface member a method implements — and so whose scope attribute it inherits — is the
    ///     CLR's interface map, not a name-and-signature match. Both layers follow it, and these tests pin
    ///     the EXPECTED verdict of each shape on both (the parity test proves only that they agree, which a
    ///     change breaking both alike would still pass):
    ///     <list type="bullet">
    ///         <item>an interface listed on a base that does not declare the slot maps its member onto the
    ///         inherited slot, so an override below that base runs for calls through the interface and
    ///         inherits — the analyzer missed this and reported NDW012 inside methods the weaver wove;</item>
    ///         <item>a member a type implements explicitly maps onto that explicit implementation, so a public
    ///         virtual of the same signature (and its overrides) implements nothing — the weaver wove them by
    ///         the name match, which disposes any array such a method keeps;</item>
    ///         <item>each type's own map decides at its level of the walk: a type re-implementing an interface
    ///         explicitly still inherits through the base declaring its slot when that base implements the
    ///         interface;</item>
    ///         <item>at the walk's 64-level limit both layers stop at the same place.</item>
    ///     </list>
    ///     All of these were found by a parity probe on 2026-10-24; on the commit before, every test here
    ///     fails on one layer or the other.
    /// </summary>
    [TestClass]
    public class WeaverInterfaceMappingTests
    {
        /// <summary>The usings every fixture of this class starts with.</summary>
        private const string Prelude = "using System;\nusing System.Collections.Generic;\nusing NumSharp;\n";

        /// <summary>A small float64 input array.</summary>
        /// <returns>A fresh <c>[0., 1., 2.]</c>.</returns>
        private static NumSharp.NDArray Input() => NumSharp.np.arange(3).astype(NumSharp.np.float64);

        /// <summary>Whether the single method of <paramref name="type"/> that <paramref name="pick"/> selects got a scope local.</summary>
        /// <param name="run">The woven assembly.</param>
        /// <param name="type">The declaring type's full name.</param>
        /// <param name="pick">Selects exactly one method of the type.</param>
        /// <returns>True when the method was woven with a scope.</returns>
        /// <exception cref="InvalidOperationException">The predicate selects no method or several.</exception>
        private static bool Woven(WeaveRun run, string type, Func<MethodDefinition, bool> pick)
        {
            using var asm = run.ReadCecil();
            return WeaveRun.HasScopeLocal(WeaveRun.AllMethods(asm.MainModule).Single(m => m.DeclaringType.FullName == type && pick(m)));
        }

        /// <summary>Selects the explicit implementation of an interface member named <paramref name="member"/> (C# names it <c>Interface.Member</c> and pins it with an <c>.override</c>).</summary>
        /// <param name="member">The interface member's name.</param>
        /// <returns>The predicate.</returns>
        private static Func<MethodDefinition, bool> ExplicitImplementation(string member) => m => m.HasOverrides && m.Name.EndsWith("." + member, StringComparison.Ordinal);

        /// <summary>
        ///     The types whose declaration line carries an NDW012 — every fixture of this class declares one type
        ///     per line, so the line names the method's type.
        /// </summary>
        /// <param name="source">The fixture.</param>
        /// <param name="displayPath">The file name the analyzer reports.</param>
        /// <returns>The type names, sorted, one entry per NDW012.</returns>
        /// <exception cref="AssertFailedException">The fixture does not compile.</exception>
        private static async Task<List<string>> LeakingTypes(string source, string displayPath)
        {
            var r = await AnalyzerTestHarness.RunAsync(source, displayPath);
            Assert.IsTrue(r.CompileErrors.IsEmpty, string.Join("\n", r.CompileErrors));
            var lines = source.Split('\n');
            return r.Ndw.Where(d => d.Id == "NDW012")
                .Select(d => Regex.Match(lines[AnalyzerTestHarness.LineOf(d) - 1], @"(?:class|struct) (\w+)").Groups[1].Value)
                .OrderBy(n => n, StringComparer.Ordinal)
                .ToList();
        }

        /// <summary>
        ///     <c>Stage : StageRoot, IStage</c> lists the interface but declares no <c>Run</c>: the CLR implements
        ///     <c>IStage.Run</c> with the inherited <c>StageRoot.Run</c> slot, so for an instance of an override
        ///     below <c>Stage</c> a call through <c>IStage</c> runs that override, and it inherits [NDScoped] —
        ///     sealed, two bases below the slot, abstract slot, generic slot. <c>StageRoot.Run</c> itself (it
        ///     knows no interface) and a <c>new virtual</c> below the listing (a new slot) inherit nothing. The
        ///     analyzer agrees: NDW012 only in the three that are not woven.
        /// </summary>
        [TestMethod]
        public async Task InterfaceListedOnAnIntermediateBase_TheOverrideBelowIsWoven_AndDrawsNoNDW012()
        {
            const string source = Prelude +
                "public interface IStage { [NDScoped] NDArray Run(NDArray a); }\n" +
                "public class StageRoot { public virtual NDArray Run(NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
                "public class Stage : StageRoot, IStage { }\n" +
                "public class MyStage : Stage { public override NDArray Run(NDArray a) { var t = a + 2.0; return a.copy(); } }\n" +
                "public class SealedStage : Stage { public sealed override NDArray Run(NDArray a) { var t = a + 3.0; return a.copy(); } }\n" +
                "public class NewSlotStage : Stage { public new virtual NDArray Run(NDArray a) { var t = a + 4.0; return a.copy(); } }\n" +
                "public class GapRoot { public virtual NDArray Run(NDArray a) { var t = a + 5.0; return a.copy(); } }\n" +
                "public class GapMiddle : GapRoot { }\n" +
                "public class GapStage : GapMiddle, IStage { }\n" +
                "public class GapLeaf : GapStage { public override NDArray Run(NDArray a) { var t = a + 6.0; return a.copy(); } }\n" +
                "public abstract class AbstractRoot { public abstract NDArray Run(NDArray a); }\n" +
                "public abstract class AbstractStage : AbstractRoot, IStage { }\n" +
                "public class ConcreteStage : AbstractStage { public override NDArray Run(NDArray a) { var t = a + 7.0; return a.copy(); } }\n" +
                "public interface IGStage<T> { [NDScoped] NDArray Run(T x, NDArray a); }\n" +
                "public class GRoot<T> { public virtual NDArray Run(T x, NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
                "public class GStage : GRoot<int>, IGStage<int> { }\n" +
                "public class GLeaf : GStage { public override NDArray Run(int x, NDArray a) { var t = a + x; return a.copy(); } }\n";
            var run = WeaverTestHarness.CompileAndWeave(source, "IntermediateListing");
            Assert.IsTrue(run.HasScopeLocal("MyStage", "Run"), "an override below the listing base runs for IStage.Run");
            Assert.IsTrue(run.HasScopeLocal("SealedStage", "Run"), "a sealed override is an override");
            Assert.IsTrue(run.HasScopeLocal("GapLeaf", "Run"), "the slot two bases above the listing changes nothing");
            Assert.IsTrue(run.HasScopeLocal("ConcreteStage", "Run"), "an abstract slot implements the interface for the abstract listing base");
            Assert.IsTrue(run.HasScopeLocal("GLeaf", "Run"), "a generic slot instantiated as the interface is");
            Assert.IsFalse(run.HasScopeLocal("StageRoot", "Run"), "the slot's own declaration knows no interface (the documented limitation)");
            Assert.IsFalse(run.HasScopeLocal("GapRoot", "Run"));
            Assert.IsFalse(Woven(run, "GRoot`1", m => m.Name == "Run"));
            Assert.IsFalse(run.HasScopeLocal("NewSlotStage", "Run"), "a new slot implements nothing");

            CollectionAssert.AreEqual(new[] { "GRoot", "GapRoot", "NewSlotStage", "StageRoot" }, await LeakingTypes(source, "intermediate_listing.cs"),
                "the analyzer reports exactly the methods the weave left alone");
        }

        /// <summary>
        ///     A property getter and an <c>[NDScopedExit]</c> parameter take the same route through a listing base:
        ///     the getter override is woven (and draws no NDW012 for the temporary it drops, while the root's
        ///     getter does), and the <c>Keep</c> override detaches its argument, which then survives the caller's
        ///     scope.
        /// </summary>
        [TestMethod]
        public async Task GetterAndExitParameter_ThroughAnIntermediateListing_AreInherited()
        {
            const string source = Prelude +
                "public interface IProp { [NDScoped] NDArray Value { get; } }\n" +
                "public class PropRoot { public virtual NDArray Value { get { var t = np.arange(3) + 1.0; return np.zeros(3); } } }\n" +
                "public class PropStage : PropRoot, IProp { }\n" +
                "public class PropLeaf : PropStage { public override NDArray Value { get { var t = np.arange(3) + 2.0; return np.zeros(3); } } }\n" +
                "public interface IKeep { void Keep([NDScopedExit] NDArray w); }\n" +
                "public class KeepRoot { public NDArray Kept; public virtual void Keep(NDArray w) => Kept = w; }\n" +
                "public class KeepStage : KeepRoot, IKeep { }\n" +
                "public class KeepLeaf : KeepStage { public override void Keep(NDArray w) => Kept = w; }\n";
            var run = WeaverTestHarness.CompileAndWeave(source, "ListingGetterExit");
            Assert.IsTrue(run.HasScopeLocal("PropLeaf", "get_Value"), "the getter override inherits the interface property's attribute");
            Assert.IsFalse(run.HasScopeLocal("PropRoot", "get_Value"));
            Assert.IsTrue(run.CallsDetach("KeepLeaf", "Keep"), "the Keep override inherits the retained parameter");
            Assert.IsFalse(run.CallsDetach("KeepRoot", "Keep"));

            var leaf = run.New("KeepLeaf");
            NumSharp.NDArray kept;
            using (NumSharp.NDScope.Open())
            {
                kept = Input() + 1.0;
                run.Invoke("KeepLeaf", "Keep", leaf, kept);
            }

            Assert.IsFalse(kept.IsDisposed, "the inherited [NDScopedExit] argument is detached from the caller's scope");
            CollectionAssert.AreEqual(new[] { "PropRoot" }, await LeakingTypes(source, "listing_getter_exit.cs"),
                "the root's getter drops its temporary unscoped; the override below the listing is scoped");
        }

        /// <summary>
        ///     <c>Forwarder</c> implements <c>IOp.Apply</c> explicitly and also has a public virtual <c>Apply</c> of
        ///     the same signature (the classic forwarding shape): the interface runs the explicit one, which is
        ///     woven; the public method and its override implement nothing and are left alone — so an array
        ///     they keep in a field survives the call. The weaver used to weave them by the name match, and
        ///     their own scope disposed the kept array at exit.
        /// </summary>
        [TestMethod]
        public async Task ExplicitImplementationBesideAPublicVirtual_ThePublicMethodIsNotWoven_AndKeepsItsArrays()
        {
            const string source = Prelude +
                "public interface IOp { [NDScoped] NDArray Apply(NDArray a); }\n" +
                "public class Forwarder : IOp { public NDArray Last; NDArray IOp.Apply(NDArray a) { var t = a + 1.0; return Apply(a).copy(); } public virtual NDArray Apply(NDArray a) { Last = a * 2.0; return a.copy(); } }\n" +
                "public class ForwarderLeaf : Forwarder { public override NDArray Apply(NDArray a) { Last = a * 3.0; return a.copy(); } }\n";
            var run = WeaverTestHarness.CompileAndWeave(source, "ExplicitBesidePublic");
            Assert.IsTrue(Woven(run, "Forwarder", ExplicitImplementation("Apply")), "the explicit implementation is what IOp.Apply runs");
            Assert.IsFalse(run.HasScopeLocal("Forwarder", "Apply"), "the public method implements nothing");
            Assert.IsFalse(run.HasScopeLocal("ForwarderLeaf", "Apply"), "nor does its override");

            foreach (var type in new[] { "Forwarder", "ForwarderLeaf" })
            {
                var instance = run.New(type);
                var input = Input();
                run.Invoke(type, "Apply", instance, input);
                var last = (NumSharp.NDArray)run.GetField(instance, "Last");
                Assert.IsFalse(last.IsDisposed, $"{type}.Apply keeps the array it stored — no scope of its own reclaims it");
                double factor = type == "Forwarder" ? 2.0 : 3.0; // Last = a * 2.0 / a * 3.0, a = [0, 1, 2]
                Assert.AreEqual(factor, last.GetDouble(1), "the kept array is intact");
            }

            // The analyzer's verdict on the same shape: the public method is not scoped, so a temporary it
            // drops leaks (NDW012), while the explicit implementation inherits and draws nothing.
            const string leaking = Prelude +
                "public interface IOp { [NDScoped] NDArray Apply(NDArray a); }\n" +
                "public class ExplicitScoped : IOp { NDArray IOp.Apply(NDArray a) { var t = a + 1.0; return a.copy(); } public virtual NDArray Apply(NDArray b) => b.copy(); }\n" +
                "public class PublicLeaks : IOp { NDArray IOp.Apply(NDArray a) => a.copy(); public virtual NDArray Apply(NDArray a) { var t = a + 2.0; return a.copy(); } }\n";
            CollectionAssert.AreEqual(new[] { "PublicLeaks" }, await LeakingTypes(leaking, "explicit_beside_public.cs"));
        }

        /// <summary>
        ///     Each type's own interface map decides at its level of the walk:
        ///     <list type="bullet">
        ///         <item><c>ExplicitOverrider : ImplRoot, IOp</c> re-implements <c>IOp</c> explicitly, so its public
        ///         <c>Apply</c> is not <c>IOp.Apply</c> at its own level — but it overrides <c>ImplRoot.Apply</c>, which
        ///         implements <c>IOp.Apply</c> in <c>ImplRoot</c>'s map, so it inherits through that base. (A walk-wide
        ///         memory of searched interfaces hid this once the explicit rule existed.)</item>
        ///         <item><c>ReExplicit : ImplicitListing, IOp</c> re-implements explicitly over a base that listed
        ///         <c>IOp</c> implicitly, and its slot's declaring base (<c>PlainRoot</c>) knows no interface: nothing
        ///         inherited — and the kept array survives.</item>
        ///         <item><c>BelowExplicit : ExplicitListing</c>, where the listing base implements the member
        ///         explicitly: nothing inherited.</item>
        ///         <item><c>Reimplementer : ExplicitBase, IOp</c> lists the interface again with a public method: that
        ///         method is the implementation now, and inherits.</item>
        ///     </list>
        /// </summary>
        [TestMethod]
        public async Task ReimplementationAndExplicitListings_EachLevelReadsItsOwnInterfaceMap()
        {
            const string source = Prelude +
                "public interface IOp { [NDScoped] NDArray Apply(NDArray a); }\n" +
                "public class ImplRoot : IOp { public virtual NDArray Apply(NDArray a) { var t = a + 1.0; return a.copy(); } }\n" +
                "public class ExplicitOverrider : ImplRoot, IOp { NDArray IOp.Apply(NDArray a) { var t = a + 2.0; return a.copy(); } public override NDArray Apply(NDArray a) { var t = a + 3.0; return a.copy(); } }\n" +
                "public class PlainRoot { public NDArray Last; public virtual NDArray Apply(NDArray a) { var t = a + 4.0; return a.copy(); } }\n" +
                "public class ImplicitListing : PlainRoot, IOp { }\n" +
                "public class ReExplicit : ImplicitListing, IOp { NDArray IOp.Apply(NDArray a) { var t = a + 5.0; return a.copy(); } public override NDArray Apply(NDArray a) { Last = a * 2.0; var t = a + 6.0; return a.copy(); } }\n" +
                "public class ExplicitListing : PlainRoot, IOp { NDArray IOp.Apply(NDArray a) { var t = a + 7.0; return a.copy(); } }\n" +
                "public class BelowExplicit : ExplicitListing { public override NDArray Apply(NDArray a) { var t = a + 8.0; return a.copy(); } }\n" +
                "public class ExplicitBase : IOp { NDArray IOp.Apply(NDArray a) { var t = a + 9.0; return a.copy(); } }\n" +
                "public class Reimplementer : ExplicitBase, IOp { public NDArray Apply(NDArray a) { var t = a + 10.0; return a.copy(); } }\n";
            var run = WeaverTestHarness.CompileAndWeave(source, "ReimplementationLevels");
            Assert.IsTrue(run.HasScopeLocal("ImplRoot", "Apply"));
            Assert.IsTrue(Woven(run, "ExplicitOverrider", ExplicitImplementation("Apply")));
            Assert.IsTrue(run.HasScopeLocal("ExplicitOverrider", "Apply"), "inherits IOp.Apply through ImplRoot.Apply, the slot it overrides");
            Assert.IsTrue(Woven(run, "ReExplicit", ExplicitImplementation("Apply")));
            Assert.IsFalse(run.HasScopeLocal("ReExplicit", "Apply"), "its own map sends IOp.Apply to its explicit implementation, and PlainRoot knows no interface");
            Assert.IsTrue(Woven(run, "ExplicitListing", ExplicitImplementation("Apply")));
            Assert.IsFalse(run.HasScopeLocal("BelowExplicit", "Apply"), "the listing base sends IOp.Apply to its explicit implementation");
            Assert.IsFalse(run.HasScopeLocal("PlainRoot", "Apply"));
            Assert.IsTrue(run.HasScopeLocal("Reimplementer", "Apply"), "listing IOp again makes the public method the implementation");

            var re = run.New("ReExplicit");
            run.Invoke("ReExplicit", "Apply", re, Input());
            Assert.IsFalse(((NumSharp.NDArray)run.GetField(re, "Last")).IsDisposed, "an unscoped override keeps what it stores");

            CollectionAssert.AreEqual(new[] { "BelowExplicit", "PlainRoot", "ReExplicit" }, await LeakingTypes(source, "reimplementation_levels.cs"),
                "the analyzer reports exactly the methods the weave left alone");
        }

        /// <summary>
        ///     One generic interface, two instantiations: <c>IGOp&lt;int&gt;.Map</c> implemented explicitly,
        ///     <c>IGOp&lt;string&gt;.Map</c> implicitly. The explicit implementation and <c>Map(string)</c> inherit;
        ///     the public <c>Map(int)</c> does not — the explicit test compares instantiations, so the explicit
        ///     <c>IGOp&lt;int&gt;</c> implementation does not hide the implicit <c>IGOp&lt;string&gt;</c> one.
        /// </summary>
        [TestMethod]
        public async Task GenericInterface_ExplicitForOneInstantiation_TheOtherIsStillImplicit()
        {
            const string source = Prelude +
                "public interface IGOp<T> { [NDScoped] NDArray Map(T x, NDArray a); }\n" +
                "public class Mixed : IGOp<int>, IGOp<string> { NDArray IGOp<int>.Map(int x, NDArray a) { var t = a + x; return a.copy(); } public virtual NDArray Map(int x, NDArray a) { var t = a + x; return a.copy(); } public virtual NDArray Map(string x, NDArray a) { var t = a + 1.0; return a.copy(); } }\n";
            var run = WeaverTestHarness.CompileAndWeave(source, "GenericExplicitMixed");
            Assert.IsTrue(Woven(run, "Mixed", ExplicitImplementation("Map")), "the explicit IGOp<int>.Map");
            Assert.IsTrue(Woven(run, "Mixed", m => m.Name == "Map" && m.Parameters[0].ParameterType.FullName == "System.String"), "Map(string) implements IGOp<string>.Map");
            Assert.IsFalse(Woven(run, "Mixed", m => m.Name == "Map" && m.Parameters[0].ParameterType.FullName == "System.Int32"), "Map(int) implements nothing: IGOp<int>.Map is explicit");

            var r = await AnalyzerTestHarness.RunAsync(source, "generic_explicit_mixed.cs");
            Assert.AreEqual(1, r.Ndw.Count(d => d.Id == "NDW012"), "only Map(int) leaks its temporary:\n" + string.Join("\n", r.Ndw.Select(d => d.GetMessage())));
        }

        /// <summary>
        ///     The type lists <c>IOpEx</c>, a derived interface, and implements the BASE interface's member
        ///     explicitly (<c>IOp.Apply</c>): the explicit rule reaches the member through the derived listing too,
        ///     so the public <c>Apply</c> is not woven.
        /// </summary>
        [TestMethod]
        public async Task ExplicitImplementationOfABaseInterfaceMember_ThroughADerivedListing_IsHonoured()
        {
            const string source = Prelude +
                "public interface IOp { [NDScoped] NDArray Apply(NDArray a); }\n" +
                "public interface IOpEx : IOp { }\n" +
                "public class ViaDerived : IOpEx { NDArray IOp.Apply(NDArray a) { var t = a + 1.0; return a.copy(); } public virtual NDArray Apply(NDArray a) { var t = a + 2.0; return a.copy(); } }\n";
            var run = WeaverTestHarness.CompileAndWeave(source, "ExplicitViaDerived");
            Assert.IsTrue(Woven(run, "ViaDerived", ExplicitImplementation("Apply")));
            Assert.IsFalse(run.HasScopeLocal("ViaDerived", "Apply"));
            CollectionAssert.AreEqual(new[] { "ViaDerived" }, await LeakingTypes(source, "explicit_via_derived.cs"), "the public Apply leaks; the explicit one is scoped");
        }

        /// <summary>
        ///     The walk's 64-level limit, for the scope walk, the getter and the <c>[NDScopedExit]</c> walk at
        ///     once (<see cref="WeaverAnalyzerParityTests.DepthBoundary"/>: a 67-level override chain whose root
        ///     implements an interface): C3 and up inherit all three members, C0 to C2 none — the root's
        ///     interface is past the limit for them — and the analyzer reports NDW012 exactly in C0 to C2's
        ///     <c>Apply</c> and <c>Value</c> getter (each drops a temporary: <c>Apply</c> its sum, the getter the
        ///     <c>np.arange(3)</c> operand). The analyzer used to search interfaces one level deeper and called C2
        ///     scoped.
        /// </summary>
        [TestMethod]
        [Timeout(300000)]
        public async Task OverrideChainAtTheDepthLimit_InheritsWithinTheLimitOnly_OnBothLayers()
        {
            var source = WeaverAnalyzerParityTests.DepthBoundary();
            var run = WeaverTestHarness.CompileAndWeave(source, "DepthLimitChain");
            foreach (int k in new[] { 0, 1, 2 })
            {
                Assert.IsFalse(run.HasScopeLocal($"C{k}", "Apply"), $"C{k}.Apply: the root's interface is past the limit");
                Assert.IsFalse(run.HasScopeLocal($"C{k}", "get_Value"), $"C{k}.get_Value");
                Assert.IsFalse(run.CallsDetach($"C{k}", "Keep"), $"C{k}.Keep");
            }

            foreach (int k in new[] { 3, 4, 40, WeaverAnalyzerParityTests.DepthBoundaryLevels - 1 })
            {
                Assert.IsTrue(run.HasScopeLocal($"C{k}", "Apply"), $"C{k}.Apply inherits within the limit");
                Assert.IsTrue(run.HasScopeLocal($"C{k}", "get_Value"), $"C{k}.get_Value");
                Assert.IsTrue(run.CallsDetach($"C{k}", "Keep"), $"C{k}.Keep");
            }

            var leaks = await LeakingTypes(source, "depth_limit_chain.cs");
            CollectionAssert.AreEqual(new[] { "C0", "C0", "C1", "C1", "C2", "C2" }, leaks,
                "the analyzer stops where the weaver does — Apply and the Value getter of C0 to C2 each drop a temporary: " + string.Join(",", leaks));
        }
    }
}
