using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Build.Analyzer;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     Which members of a type that sits on a cycle are its NDArray HOLDERS (NDW016's member list,
    ///     NDW017's per-member warning). The rule: a member is a holder when its type holds NDArrays
    ///     WITHOUT passing back through the declaring type. A parent back-pointer, a linked-list link,
    ///     or a reference into the rest of a cycle that reaches arrays only through the declaring type
    ///     stores nothing that type would have to dispose on its own account, so it is not reported —
    ///     a warning there would ask a child to dispose its parent (NumSharp's own <c>NDScope._parent</c>
    ///     and <c>NDIterator._nestedChild</c> are that shape). A member whose type holds arrays through
    ///     another route IS a holder, cycle or not.
    ///     <para>
    ///     Every answer must also be independent of the order the analyzer callbacks ask in: the
    ///     model is shared by concurrently running callbacks, and an answer that depended on what was
    ///     memoized first would make the same code produce different warnings build to build.
    ///     </para>
    /// </summary>
    [TestClass]
    public partial class OwnershipCycleTests
    {
        /// <summary>The usings every snippet of this class is compiled with.</summary>
        private const string H = "using System;\nusing System.Collections.Generic;\nusing NumSharp;\n";

        /// <summary>
        ///     NDW016's text up to its member list. The list is the point of these tests: the exact-match gate
        ///     of the fixture tests compares line and id only, so a member wrongly added to or dropped from
        ///     the list is visible nowhere else.
        /// </summary>
        private static readonly Regex Ndw016Members = new Regex(@"^'(?<type>[^']+)' stores NDArrays in (?<list>.*?) but implements neither", RegexOptions.Compiled);

        /// <summary>NDW017's text: the disposable type and the one holder member it never disposes.</summary>
        private static readonly Regex Ndw017Member = new Regex(@"^'(?<type>[^']+)' implements .*? but never disposes '(?<member>[^']+)'", RegexOptions.Compiled);

        /// <summary>A quoted name inside a member list.</summary>
        private static readonly Regex Quoted = new Regex(@"'([^']+)'", RegexOptions.Compiled);

        /// <summary>The NDW016 member list and the NDW017 members of every type, keyed by type name.</summary>
        private sealed class Findings
        {
            /// <summary>Each NDW016's member names, comma-joined in the message's order, by declaring type.</summary>
            public readonly Dictionary<string, string> Ndw016 = new Dictionary<string, string>(StringComparer.Ordinal);

            /// <summary>The members each type's NDW017s name (one diagnostic per member), by declaring type.</summary>
            public readonly Dictionary<string, SortedSet<string>> Ndw017 = new Dictionary<string, SortedSet<string>>(StringComparer.Ordinal);

            /// <summary>How many NDW diagnostics of any id the snippet drew; pins that nothing else fired.</summary>
            public int Total;

            /// <summary>Every diagnostic as "id: message", one per line — the failure message's evidence.</summary>
            public string All;

            /// <summary>The members <paramref name="type"/>'s NDW016 names, comma-joined.</summary>
            /// <param name="type">The declaring type's name.</param>
            /// <returns>The member list, or null when the type drew no NDW016 (distinct from an empty list).</returns>
            public string Members16(string type) => Ndw016.TryGetValue(type, out var list) ? list : null;

            /// <summary>The members <paramref name="type"/>'s NDW017s name, sorted and comma-joined.</summary>
            /// <param name="type">The declaring type's name.</param>
            /// <returns>The member list; empty when the type drew no NDW017.</returns>
            public string Members17(string type) => Ndw017.TryGetValue(type, out var set) ? string.Join(",", set) : "";
        }

        /// <summary>Compiles a snippet with every analyzer and parses its NDW016/NDW017 member lists.</summary>
        /// <param name="body">The C# declarations, compiled after <see cref="H"/>.</param>
        /// <param name="name">The file name (without extension) the snippet is compiled as.</param>
        /// <returns>The parsed findings.</returns>
        /// <exception cref="AssertFailedException">The snippet does not compile, or an NDW016/NDW017 text no longer matches the parser.</exception>
        private static async Task<Findings> Analyze(string body, string name)
        {
            var r = await AnalyzerTestHarness.RunAsync(H + body, name + ".cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile:\n" + body + "\n" + string.Join("\n", r.CompileErrors));
            return Parse(r);
        }

        /// <summary>
        ///     Parses the NDW016 member lists and NDW017 members out of an analyzer result — for a run whose
        ///     compile errors the caller has already judged (a deliberately broken snippet, a multi-file
        ///     project).
        /// </summary>
        /// <param name="r">The analyzer result.</param>
        /// <returns>The parsed findings.</returns>
        /// <exception cref="AssertFailedException">An NDW016/NDW017 text no longer matches the parser.</exception>
        private static Findings Parse(AnalyzerResult r)
        {
            var f = new Findings { Total = r.Ndw.Length, All = string.Join("\n", r.Ndw.Select(d => d.Id + ": " + d.GetMessage())) };
            foreach (var d in r.Ndw)
            {
                var msg = d.GetMessage();
                if (d.Id == "NDW016")
                {
                    var m = Ndw016Members.Match(msg);
                    Assert.IsTrue(m.Success, "unexpected NDW016 text: " + msg);
                    f.Ndw016[m.Groups["type"].Value] = string.Join(",", Quoted.Matches(m.Groups["list"].Value).Cast<Match>().Select(q => q.Groups[1].Value));
                }
                else if (d.Id == "NDW017")
                {
                    var m = Ndw017Member.Match(msg);
                    Assert.IsTrue(m.Success, "unexpected NDW017 text: " + msg);
                    var type = m.Groups["type"].Value;
                    if (!f.Ndw017.TryGetValue(type, out var set))
                        f.Ndw017[type] = set = new SortedSet<string>(StringComparer.Ordinal);
                    set.Add(m.Groups["member"].Value);
                }
            }
            return f;
        }

        // ---------------------------------------------------------------- back-references are not holders

        /// <summary>
        ///     A node's link to the next node of its own type reaches NDArrays only through the node itself,
        ///     so NDW016 names <c>_v</c> alone. PR #632's commit listed <c>'_next' and '_v'</c>.
        /// </summary>
        [TestMethod]
        public async Task SelfReference_LinkedListLink_IsNotAHolder()
        {
            var f = await Analyze("public class Node { Node _next; NDArray _v; }", "self_ref");
            Assert.AreEqual("_v", f.Members16("Node"), "the link to the next node of the same type reaches arrays only through Node itself:\n" + f.All);
        }

        /// <summary>
        ///     A disposable node that disposes its own array draws nothing: its link to the next node is not
        ///     an array it owns, and an NDW017 on it (PR #632's commit drew one) would demand that every node
        ///     dispose the rest of the list.
        /// </summary>
        [TestMethod]
        public async Task SelfReference_DisposableThatDisposesItsArray_IsClean()
        {
            var f = await Analyze("public class Node : IDisposable { Node _next; NDArray _v; public void Dispose() => _v?.Dispose(); }", "self_ref_disposable");
            Assert.AreEqual(0, f.Total, "disposing the node's own array is the whole job; the link is not its array:\n" + f.All);
        }

        /// <summary>
        ///     A tree node's parent pointer, child list and sibling array, all of its own type, draw nothing
        ///     once the node disposes its own data. An NDW017 on <c>_parent</c> would tell a child to dispose
        ///     its parent.
        /// </summary>
        [TestMethod]
        public async Task ParentBackPointer_AndChildrenOfTheSameType_AreNotHolders()
        {
            var f = await Analyze(
                "public class TreeNode : IDisposable { TreeNode _parent; List<TreeNode> _children = new List<TreeNode>(); TreeNode[] _siblings; NDArray _data;\n" +
                "  public void Dispose() => _data?.Dispose(); }", "tree");
            Assert.AreEqual(0, f.Total, "a parent pointer, a child list and a sibling array of the node's own type all reach arrays only through TreeNode:\n" + f.All);
        }

        /// <summary>
        ///     NumSharp's own NDScope shape, a list of tracked arrays plus a link to the parent scope, is
        ///     clean. On PR #632's commit this shape gave NumSharp.Core's build an NDW017 on
        ///     <c>NDScope._parent</c>.
        /// </summary>
        [TestMethod]
        public async Task NDScopeShape_ParentScope_IsNotAHolder()
        {
            // NumSharp's own NDScope: a scope tracks the arrays it reclaims and links to its parent scope.
            // A warning on _parent would tell the child scope to dispose its parent.
            var f = await Analyze(
                "public sealed class Scope : IDisposable { private Scope _parent; private readonly List<NDArray> _tracked = new List<NDArray>();\n" +
                "  public void Dispose() { foreach (var a in _tracked) a.Dispose(); } }", "scope_shape");
            Assert.AreEqual(0, f.Total, f.All);
        }

        /// <summary>
        ///     NumSharp's np.nditer iterator shape, an operand plus a nested child iterator of the same type,
        ///     is clean. On PR #632's commit it gave NumSharp.Core an NDW017 on
        ///     <c>NDIterator._nestedChild</c>.
        /// </summary>
        [TestMethod]
        public async Task NDIteratorShape_NestedChild_IsNotAHolder()
        {
            var f = await Analyze(
                "public sealed class Iter : IDisposable { private Iter _nestedChild; private NDArray _operand; public void Dispose() => _operand?.Dispose(); }", "iter_shape");
            Assert.AreEqual(0, f.Total, f.All);
        }

        /// <summary>
        ///     Every container of the declaring type (a source generic box, a list, one- and two-dimensional
        ///     arrays, a tuple, a dictionary value, a Lazy) leads back to the type, so only the type's own
        ///     array is a holder; the unconstrained <c>Box&lt;T&gt;</c> itself draws nothing.
        /// </summary>
        [TestMethod]
        public async Task ContainersOfTheOwnType_AreNotHolders()
        {
            var f = await Analyze(
                "public class Box<T> { public T Value; }\n" +
                "public class Node { Box<Node> _boxed; List<Node> _list; Node[] _arr; Node[,] _grid; (Node, int) _pair; Dictionary<string, Node> _byName; Lazy<Node> _lazy; NDArray _v; }",
                "self_containers");
            Assert.AreEqual("_v", f.Members16("Node"), "every container of Node reaches arrays only through Node:\n" + f.All);
            Assert.AreEqual(1, f.Total, "Box<T> stores an unconstrained T, so Node's warning is the only one:\n" + f.All);
        }

        /// <summary>
        ///     In a ring A → B → C → A where only C stores an array, A's and B's members lead forward to it
        ///     and are holders, while C's <c>_a</c> reaches arrays only by coming back to C, so C, which
        ///     disposes its array, draws no NDW017.
        /// </summary>
        [TestMethod]
        public async Task Ring_TheMemberClosingTheRing_IsNotAHolder()
        {
            var f = await Analyze(
                "public class A { B _b; }\npublic class B { C _c; }\npublic class C : IDisposable { A _a; NDArray _x; public void Dispose() => _x?.Dispose(); }", "ring3");
            Assert.AreEqual("_b", f.Members16("A"), f.All);
            Assert.AreEqual("_c", f.Members16("B"), f.All);
            Assert.AreEqual("", f.Members17("C"), "C's '_a' reaches arrays only by coming back to C:\n" + f.All);
            Assert.AreEqual(2, f.Total, f.All);
        }

        /// <summary>
        ///     The HolderTypeScenarios fixture's cycle pair, member by member. The fixture states that B's
        ///     back-reference adds nothing, but its exact-match gate compares lines and ids, never the member
        ///     list in a message, so PR #632's regression on this very pair passed there.
        /// </summary>
        [TestMethod]
        public async Task CycleWithArrayFixture_BackReference_AddsNothing()
        {
            // HolderTypeScenarios.cs states it in a comment ("its back-reference to A adds nothing"); the
            // exact-match gate there pins lines only, so the member list is pinned here.
            var f = await Analyze(
                "public class CycleWithArrayA { private CycleWithArrayB _b; }\npublic class CycleWithArrayB { private CycleWithArrayA _a; private NDArray _x; }", "fixture_cycle");
            Assert.AreEqual("_b", f.Members16("CycleWithArrayA"), f.All);
            Assert.AreEqual("_x", f.Members16("CycleWithArrayB"), f.All);
        }

        /// <summary>
        ///     A base type with a member of its own derived type. A base type is an edge, so Derived holds
        ///     only through Base and <c>Base._d</c> comes back to Base: NDW016 names <c>_a</c> alone, and
        ///     Derived, which declares nothing, draws nothing.
        /// </summary>
        [TestMethod]
        public async Task BaseTypeCycle_DerivedMemberOfTheBase_IsNotAHolder()
        {
            // Derived holds only through its base (the base type is an edge), so the base's member of type
            // Derived comes back to Base.
            var f = await Analyze("public class Base { Derived _d; NDArray _a; }\npublic class Derived : Base { }", "base_cycle");
            Assert.AreEqual("_a", f.Members16("Base"), f.All);
            Assert.IsNull(f.Members16("Derived"), "Derived declares no member of its own:\n" + f.All);
        }

        /// <summary>
        ///     A source subclass of NDArray holds by itself but keeps its member edges: its member of a
        ///     wrapper that only points back at it is not a holder (the NDW017s name <c>_extra</c> alone),
        ///     while the wrapper's member of the subclass is one, a MyArray being an NDArray. Without the
        ///     member edges the wrapper would look like an outside holder and <c>_w</c> would be reported.
        /// </summary>
        [TestMethod]
        public async Task SourceNDArraySubclass_WrapperPointingBack_IsNotAHolder()
        {
            // A source subclass of NDArray holds by itself, but its members still form edges: a wrapper that
            // only points back at it reaches arrays only through it. (NDArray's own Dispose is inherited, so
            // the holders get the inherited-Dispose NDW017.)
            var f = await Analyze(
                "public class MyArray : NDArray { public MyArray() : base(typeof(double), new Shape(1)) { } Wrapper _w; NDArray _extra; }\n" +
                "public class Wrapper { MyArray _a; }", "ndarray_subclass");
            Assert.AreEqual("_extra", f.Members17("MyArray"), f.All);
            Assert.AreEqual("_a", f.Members16("Wrapper"), "a MyArray is itself an NDArray:\n" + f.All);
        }

        // ---------------------------------------------------------------- members that hold on their own

        /// <summary>
        ///     Two types that point at each other and each store an array of their own: each member holds
        ///     through the OTHER type's own array, so both are holders. Being on a cycle never hides a
        ///     holder by itself.
        /// </summary>
        [TestMethod]
        public async Task MutualReferences_EachHoldingItsOwnArray_AreHolders()
        {
            var f = await Analyze("public class A { B _b; NDArray _x; }\npublic class B { A _a; NDArray _y; }", "mutual");
            Assert.AreEqual("_b,_x", f.Members16("A"), "B holds _y on its own, not through A:\n" + f.All);
            Assert.AreEqual("_a,_y", f.Members16("B"), "A holds _x on its own, not through B:\n" + f.All);
        }

        /// <summary>
        ///     A child's back-reference to an owner that stores arrays itself IS a holder: the owner's arrays
        ///     do not come back through the child. The author settles this one with [NDBorrowed]; the rule
        ///     must not hide it just because the reference closes a cycle.
        /// </summary>
        [TestMethod]
        public async Task ParentChild_ChildsOwnerHoldingArraysDirectly_IsAHolder()
        {
            // Child._owner is a back-reference, but Parent stores _x itself: the owner's arrays do not come
            // back through Child, so the member is a holder (the author marks it [NDBorrowed]).
            var f = await Analyze(
                "public class Parent : IDisposable { List<Child> _kids = new List<Child>(); NDArray _x; public void Dispose() { _x?.Dispose(); foreach (var k in _kids) k.Dispose(); } }\n" +
                "public class Child : IDisposable { Parent _owner; NDArray _y; public void Dispose() => _y?.Dispose(); }", "parent_child");
            Assert.AreEqual("_owner", f.Members17("Child"), f.All);
            Assert.AreEqual("", f.Members17("Parent"), f.All);
        }

        /// <summary>
        ///     T ↔ A, where A also stores a holding G: T's <c>_a</c> reaches G without T (a holder), A's
        ///     <c>_t</c> reaches arrays only through A (not a holder). The type a route must avoid is each
        ///     member's own declaring type, not one fixed type of the cycle.
        /// </summary>
        [TestMethod]
        public async Task GateOtherThanTheOwner_DecidesThroughTheRestOfTheCycle()
        {
            // T <-> A, and A also stores G, which holds. T's member _a reaches G without T: a holder. A's
            // member _t reaches arrays only through A (T's only edge is back to A): not a holder.
            var f = await Analyze(
                "public class T { A _a; }\npublic class A { T _t; G _g; }\npublic class G { NDArray _x; }", "gate");
            Assert.AreEqual("_a", f.Members16("T"), f.All);
            Assert.AreEqual("_g", f.Members16("A"), f.All);
            Assert.AreEqual("_x", f.Members16("G"), f.All);
        }

        /// <summary>
        ///     The cycle T → L → J → T, with T also reaching a holder outside it (T → R → K): a member is a
        ///     holder exactly when its type reaches arrays along SOME route that avoids the member's
        ///     declaring type. T's <c>_l</c> has none (L's only route runs back through T); L's <c>_j</c>
        ///     and J's <c>_t</c> have one.
        /// </summary>
        [TestMethod]
        public async Task TwoRoutes_OneThroughTheOwnerOneNot_IsAHolder()
        {
            // T -> L -> J -> T, T -> R -> K, and K holds. From J the arrays are reachable through T only
            // (J -> T -> R -> K), so J's _t is not a holder. From L: L -> J -> T -> ...: through L? No —
            // L's member _j reaches arrays via J -> T -> R -> K, which avoids L: a holder.
            var f = await Analyze(
                "public class T { L _l; R _r; }\npublic class L { J _j; }\npublic class J { T _t; }\npublic class R { K _k; }\npublic class K { NDArray _x; }", "two_routes");
            Assert.AreEqual("_r", f.Members16("T"), "L reaches arrays only through T:\n" + f.All);
            Assert.AreEqual("_j", f.Members16("L"), f.All);
            Assert.AreEqual("_t", f.Members16("J"), "J's only route runs through T, not through J:\n" + f.All);
            Assert.AreEqual("_k", f.Members16("R"), f.All);
        }

        /// <summary>
        ///     A ring T → A → B → T in which both T and B store arrays: every member is a holder, each one
        ///     reaching the OTHER array without passing through its declaring type. Two anchors inside one
        ///     cycle give the dominator tree two roots below its virtual root.
        /// </summary>
        [TestMethod]
        public async Task SecondAnchorInsideTheCycle_KeepsTheMemberAHolder()
        {
            // A ring T -> A -> B -> T where both T and B hold an array of their own: A's _b holds without A,
            // and T's _a holds without T (through B), and B's _t holds without B (T's own _x).
            var f = await Analyze(
                "public class T { A _a; NDArray _x; }\npublic class A { B _b; }\npublic class B { T _t; NDArray _y; }", "two_anchors");
            Assert.AreEqual("_a,_x", f.Members16("T"), f.All);
            Assert.AreEqual("_b", f.Members16("A"), f.All);
            Assert.AreEqual("_t,_y", f.Members16("B"), f.All);
        }

        // ---------------------------------------------------------------- order independence (the model, directly)

        /// <summary>
        ///     The model the order tests query: a tree with a parent pointer, a three-type ring, a base/derived
        ///     cycle, a parent/child pair whose child's owner holds by itself, the two-route cycle, and a
        ///     self-referencing type that holds nothing — one of each shape whose answer an order-dependent
        ///     memo could flip.
        /// </summary>
        private const string OrderModel =
            "using System;\nusing System.Collections.Generic;\nusing NumSharp;\n" +
            "public class TreeNode { TreeNode _parent; List<TreeNode> _children; NDArray _data; }\n" +
            "public class A { B _b; }\npublic class B { C _c; }\npublic class C { A _a; NDArray _x; }\n" +
            "public class Base { Derived _d; NDArray _a; }\npublic class Derived : Base { }\n" +
            "public class P { List<Q> _kids; NDArray _x; }\npublic class Q { P _owner; NDArray _y; }\n" +
            "public class T { L _l; R _r; }\npublic class L { J _j; }\npublic class J { T _t; }\npublic class R { K _k; }\npublic class K { NDArray _x; }\n" +
            "public class Free { int _n; string _s; Free _self; }\n";

        /// <summary>Compiles <see cref="OrderModel"/> for direct <see cref="OwnershipModel"/> queries, outside any analyzer callback.</summary>
        /// <returns>
        ///     The compilation, its resolved NumSharp types, the source types ordered by name (what a snapshot
        ///     reads), and every type the walks can meet — the source types plus their field and base types —
        ///     for the warm-up orders to draw from.
        /// </returns>
        /// <exception cref="AssertFailedException">The model does not compile, or NumSharp's types cannot be resolved from it.</exception>
        private static (CSharpCompilation comp, KnownTypes known, INamedTypeSymbol[] types, ITypeSymbol[] allTypes) BuildOrderModel()
        {
            var tree = CSharpSyntaxTree.ParseText(SourceText.From(OrderModel), path: "order_model.cs");
            var comp = CSharpCompilation.Create("OrderModel", new[] { tree }, AnalyzerTestHarness.References,
                new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
            Assert.AreEqual(0, comp.GetDiagnostics().Count(d => d.Severity == DiagnosticSeverity.Error), "order model must compile");
            var known = KnownTypes.Resolve(comp);
            Assert.IsNotNull(known);
            var types = comp.GlobalNamespace.GetTypeMembers().Where(t => OwnershipModel.IsInSource(t)).OrderBy(t => t.Name, StringComparer.Ordinal).ToArray();

            // Every type the walks can meet: the source types and their member and base types.
            var all = new List<ITypeSymbol>();
            foreach (var t in types)
            {
                all.Add(t);
                foreach (var f in t.GetMembers().OfType<IFieldSymbol>())
                    all.Add(f.Type);
                if (t.BaseType != null)
                    all.Add(t.BaseType);
            }
            return (comp, known, types, all.Distinct<ITypeSymbol>(SymbolEqualityComparer.Default).ToArray());
        }

        /// <summary>Every answer the model gives about <paramref name="types"/>, in one comparable string.</summary>
        /// <param name="model">The model to read (reading may complete its memo).</param>
        /// <param name="types">The types to report, in a fixed order.</param>
        /// <returns>"Name+:holder,holder " per type, '+' when it holds NDArrays and '-' when it does not.</returns>
        private static string Snapshot(OwnershipModel model, INamedTypeSymbol[] types)
        {
            var sb = new StringBuilder();
            foreach (var t in types)
                sb.Append(t.Name).Append(model.HoldsNDArrays(t) ? "+" : "-").Append(':')
                  .Append(string.Join(",", model.HolderMembersOf(t).Select(h => h.Name))).Append(' ');
            return sb.ToString();
        }

        /// <summary>
        ///     The model gives the same answers whatever it was asked before: after each of 200 random warm-up
        ///     orders (member types first, owners first, mixed with HolderMembersOf) every answer equals the
        ///     cold baseline. Master fails this: it memoized a whole-graph answer that HolderMembersOf
        ///     trusted, so whether <c>TreeNode._parent</c> was a holder depended on whether something, such
        ///     as a Dispose body elsewhere, had asked about TreeNode first.
        /// </summary>
        [TestMethod]
        public void HolderMembers_DoNotDependOnWhatWasAskedFirst()
        {
            var (comp, known, types, all) = BuildOrderModel();
            string baseline = Snapshot(new OwnershipModel(known, comp), types);
            StringAssert.Contains(baseline, "TreeNode+:_data ", "baseline sanity: " + baseline);
            StringAssert.Contains(baseline, "C+:_x ", "baseline sanity: " + baseline);
            StringAssert.Contains(baseline, "Q+:_owner,_y ", "baseline sanity: " + baseline);
            StringAssert.Contains(baseline, "Free-: ", "baseline sanity: " + baseline);

            // Warm the model up in many different orders (member types first, owners first, mixed with
            // HolderMembersOf calls), then read everything: the answers must equal the cold baseline.
            for (int seed = 0; seed < 200; seed++)
            {
                var rng = new Random(seed);
                var model = new OwnershipModel(known, comp);
                foreach (var t in all.OrderBy(_ => rng.Next()).Take(rng.Next(all.Length + 1)))
                {
                    if (rng.Next(3) == 0 && t is INamedTypeSymbol named)
                        model.HolderMembersOf(named);
                    else
                        model.HoldsNDArrays(t);
                }
                Assert.AreEqual(baseline, Snapshot(model, types), $"seed {seed}: the answers depend on the query order");
            }
        }

        /// <summary>
        ///     Sixteen threads query one fresh model in random orders, 25 rounds: every thread's answers equal
        ///     the single-threaded ones. The analyzers share one model per compilation and fill it from
        ///     concurrent callbacks, so a reader must never see a half-decided component.
        /// </summary>
        [TestMethod]
        public void ConcurrentQueries_AgreeWithTheSingleThreadedAnswers()
        {
            var (comp, known, types, all) = BuildOrderModel();
            string baseline = Snapshot(new OwnershipModel(known, comp), types);
            for (int round = 0; round < 25; round++)
            {
                var model = new OwnershipModel(known, comp);
                var results = new string[16];
                Parallel.For(0, results.Length, new ParallelOptions { MaxDegreeOfParallelism = 16 }, i =>
                {
                    var rng = new Random(round * 100 + i);
                    foreach (var t in all.OrderBy(_ => rng.Next()))
                        if (t is INamedTypeSymbol named && rng.Next(2) == 0)
                            model.HolderMembersOf(named);
                        else
                            model.HoldsNDArrays(t);
                    results[i] = Snapshot(model, types);
                });
                foreach (var r in results)
                    Assert.AreEqual(baseline, r, $"round {round}: a concurrent reader saw a different answer");
            }
        }

        // ---------------------------------------------------------------- the walk's stack

        /// <summary>
        ///     A 10,000-level chain of types completes and flags every level. A recursive walk overflowed the
        ///     thread's stack, which ends the compiler PROCESS rather than failing one analyzer: at 2,000
        ///     levels on master and at 1,500 on PR #632's commit.
        /// </summary>
        [TestMethod]
        [Timeout(180000)]
        public async Task VeryDeepChain_DoesNotOverflowTheStack()
        {
            // The walk keeps its own stack: a recursive walk overflowed the thread's stack (taking the
            // compiler process down) at a few thousand levels.
            const int depth = 10000;
            var sb = new StringBuilder();
            for (int i = 0; i < depth; i++)
                sb.Append("public class L").Append(i).Append(" { ")
                  .Append(i + 1 < depth ? "L" + (i + 1) + " _next;" : "NDArray _leaf;")
                  .AppendLine(" }");
            var r = await AnalyzerTestHarness.RunAsync(H + sb, "very_deep_chain.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty);
            Assert.AreEqual(depth, r.CountOf("NDW016"), "every level stores the leaf transitively");
        }

        /// <summary>
        ///     An entity model of <paramref name="count"/> types with three navigation properties each (one
        ///     reference, one collection, one reference), wired at random: a large strongly connected
        ///     model, the shape a real domain model has.
        /// </summary>
        /// <param name="count">How many entity types to declare.</param>
        /// <param name="seed">The wiring's random seed.</param>
        /// <param name="arrayHolder">The index of the one entity that also stores an NDArray; -1 for none.</param>
        /// <returns>The declarations, one class per line.</returns>
        private static string RandomEntityModel(int count, int seed, int arrayHolder)
        {
            var rng = new Random(seed);
            var sb = new StringBuilder();
            for (int i = 0; i < count; i++)
            {
                sb.Append("public class E").Append(i).Append(" {")
                  .Append(" public E").Append(rng.Next(count)).Append(" A { get; set; }")
                  .Append(" public List<E").Append(rng.Next(count)).Append("> B { get; set; }")
                  .Append(" public E").Append(rng.Next(count)).Append(" C { get; set; }");
                if (i == arrayHolder)
                    sb.Append(" NDArray _x;");
                sb.AppendLine(" }");
            }
            return sb.ToString();
        }

        /// <summary>
        ///     A random 3,000-entity model that stores no NDArray completes and draws nothing. A model of this
        ///     shape and size overflowed the stack on master and on PR #632's commit alike (the PR's commit
        ///     already at 2,000 entities): a project that never uses NDArray lost its build.
        /// </summary>
        [TestMethod]
        [Timeout(180000)]
        public async Task LargeCyclicEntityModel_WithoutArrays_CompletesClean()
        {
            var r = await AnalyzerTestHarness.RunAsync(H + RandomEntityModel(3000, 7, -1), "entity_model_clean.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty);
            Assert.AreEqual(0, r.Ndw.Length, "a model that stores no NDArray draws nothing");
        }

        /// <summary>
        ///     The same model with an NDArray in one entity: most entities reach it through the random wiring,
        ///     so more than half and at most all of them draw NDW016, and the analysis completes.
        /// </summary>
        [TestMethod]
        [Timeout(180000)]
        public async Task LargeCyclicEntityModel_WithOneArray_FlagsTheTypesThatReachIt()
        {
            const int count = 3000;
            var r = await AnalyzerTestHarness.RunAsync(H + RandomEntityModel(count, 7, count / 2), "entity_model_array.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty);
            int flagged = r.CountOf("NDW016");
            Assert.IsTrue(flagged > count / 2 && flagged <= count, $"most of a random strongly connected model reaches the one array ({flagged} of {count} flagged)");
        }
    }
}
