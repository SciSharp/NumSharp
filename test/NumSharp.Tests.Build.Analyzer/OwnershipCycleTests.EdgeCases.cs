using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    // The edge cases of the holder rule, as the analyzers report them. Every expectation below was worked out
    // by hand from the rule ("a member holds when its type reaches a stored NDArray without passing through
    // the declaring type") and then run against master: these pin the rule's behavior on shapes the first test
    // pass did not reach (generics, records, structs, partial and nested types, borrowed cycles, several
    // components, scale, messages, disposal and contagion, broken code, and determinism across files and
    // threads). Master agrees on every scenario but three: the long ring and the deep chain through generic
    // containers overflow its stack (the test host crashes), and the random multi-file project times out —
    // its per-path walk is exponential on a model that dense, the problem PR #632 set out to fix. Against PR
    // #632's commit most of the back-reference scenarios fail (it reported the back-references), and the same
    // two deep shapes crash it.
    public partial class OwnershipCycleTests
    {
        /// <summary>Cuts a long diagnostic dump for a failure message, so a scale test does not build megabyte strings.</summary>
        /// <param name="s">The text.</param>
        /// <returns>At most its first 2,000 characters.</returns>
        private static string Head(string s) => s.Length <= 2000 ? s : s.Substring(0, 2000) + " …";

        // ---------------------------------------------------------------- more shapes that only lead back

        /// <summary>A doubly linked node: neither the next nor the previous link is a holder — each reaches arrays only through the node itself.</summary>
        [TestMethod]
        public async Task DoublyLinkedList_BothLinks_AreNotHolders()
        {
            var f = await Analyze("public class Node { Node _next; Node _prev; NDArray _v; }", "doubly_linked");
            Assert.AreEqual("_v", f.Members16("Node"), f.All);
            Assert.AreEqual(1, f.Total, f.All);
        }

        /// <summary>
        ///     Nullable-annotated self-references are the same type to the model (nullability is not part of
        ///     a type's identity there), so <c>Node? _next</c> and <c>List&lt;Node?&gt;? _all</c> are links,
        ///     not holders, and <c>NDArray? _v</c> still holds.
        /// </summary>
        [TestMethod]
        public async Task NullableAnnotatedSelfReferences_AreNotHolders()
        {
            var f = await Analyze("#nullable enable\npublic class Node { Node? _next; List<Node?>? _all; NDArray? _v; }\n#nullable restore", "nullable_self");
            Assert.AreEqual("_v", f.Members16("Node"), f.All);
            Assert.AreEqual(1, f.Total, f.All);
        }

        /// <summary>
        ///     Containers of the declaring type that the first pass did not cover — a jagged array, a list of
        ///     arrays, a dictionary of lists, a nullable tuple, a lazy list, an <c>IEnumerable&lt;T&gt;</c>, a
        ///     <c>KeyValuePair</c> — each a chain of intermediate generic nodes back to the type: none is a holder.
        /// </summary>
        [TestMethod]
        public async Task JaggedNestedAndBclContainersOfTheOwnType_AreNotHolders()
        {
            var f = await Analyze(
                "public class Node { Node[][] _jagged; List<Node[]> _listOfArrays; Dictionary<string, List<Node>> _index; (Node, int)? _maybe;\n" +
                "  Lazy<List<Node>> _lazy; IEnumerable<Node> _seq; KeyValuePair<Node, Node> _kv; NDArray _v; }", "containers_of_self");
            Assert.AreEqual("_v", f.Members16("Node"), f.All);
            Assert.AreEqual(1, f.Total, f.All);
        }

        /// <summary>
        ///     A tuple that stores an NDArray beside a link to the declaring type IS a holder: the array is
        ///     stored directly, not reached through the type. A tuple of the type and a scalar, or of the type
        ///     and a nested tuple of it, is a link.
        /// </summary>
        [TestMethod]
        public async Task TupleCarryingAnArrayBesideTheOwnType_IsAHolder()
        {
            var f = await Analyze("public class Node { (Node next, NDArray data) _pair; (Node, int) _plain; (Node, (Node, string)) _nested; }", "tuple_with_array");
            Assert.AreEqual("_pair", f.Members16("Node"), f.All);
        }

        /// <summary>
        ///     Static fields, constants, field-like events and computed properties store no per-instance value
        ///     of the type, so none of them is an edge — the type's only holder is its own array, and the link
        ///     beside them is still a link.
        /// </summary>
        [TestMethod]
        public async Task StaticEventAndComputedSelfReferences_AreNotEdges()
        {
            var f = await Analyze(
                "public class Node { static Node s_root; const int Limit = 3; public event Action<Node> Changed; public Node Self => this; public Node Twin { get => this; }\n" +
                "  Node _next; NDArray _v; }", "non_edges");
            Assert.AreEqual("_v", f.Members16("Node"), f.All);
            Assert.AreEqual(1, f.Total, f.All);
        }

        /// <summary>A property over an explicit backing field: the FIELD is the stored link (and not a holder); the property itself stores nothing.</summary>
        [TestMethod]
        public async Task ExplicitBackingField_IsTheEdge_ThePropertyIsNot()
        {
            var f = await Analyze("public class Node { private Node _backing; public Node Next { get => _backing; set => _backing = value; } NDArray _v; }", "backing_field");
            Assert.AreEqual("_v", f.Members16("Node"), f.All);
        }

        /// <summary>The members of every part of a partial class are judged together, and the type draws one NDW016 however many declarations it has.</summary>
        [TestMethod]
        public async Task PartialClass_MembersOfEveryPart_AreJudgedTogether()
        {
            var f = await Analyze("public partial class Node { Node _next; }\npublic partial class Node { NDArray _v; List<Node> _all; }", "partial_parts");
            Assert.AreEqual("_v", f.Members16("Node"), f.All);
            Assert.AreEqual(1, f.Total, "one type, one NDW016:\n" + f.All);
        }

        /// <summary>
        ///     A member typed by a SOURCE interface is not an edge: a source interface stores nothing (its
        ///     properties have no backing field), so it holds no array. Pinned so a change to that rule is a
        ///     deliberate one; the node's own array is still its holder.
        /// </summary>
        [TestMethod]
        public async Task InterfaceTypedLink_IsNotAnEdge()
        {
            var f = await Analyze("public interface INode { NDArray Data { get; } }\npublic class Node : INode { INode _next; List<INode> _children; NDArray _v; public NDArray Data => _v; }", "interface_link");
            Assert.AreEqual("_v", f.Members16("Node"), f.All);
            Assert.AreEqual(1, f.Total, f.All);
        }

        /// <summary>
        ///     Records: a positional property of the record's own type (a record class, a sealed record) or an
        ///     array of it (a record struct) is a link, reported nowhere; the positional NDArray property is the
        ///     holder, named as the property the author wrote.
        /// </summary>
        [TestMethod]
        public async Task Records_PositionalSelfReference_IsNotAHolder()
        {
            var f = await Analyze(
                "public record Node(Node Next, NDArray Value);\npublic record struct Point(NDArray Coords, Point[] Neighbors);\n" +
                "public sealed record Leaf(Leaf Parent, List<Leaf> Children, NDArray Data);", "records");
            Assert.AreEqual("Value", f.Members16("Node"), f.All);
            Assert.AreEqual("Coords", f.Members16("Point"), f.All);
            Assert.AreEqual("Data", f.Members16("Leaf"), f.All);
            Assert.AreEqual(3, f.Total, f.All);
        }

        /// <summary>A struct cannot contain itself, but it can store arrays and lists of itself: those are links, and its own array is its holder.</summary>
        [TestMethod]
        public async Task Struct_ArraysAndListsOfItself_AreNotHolders()
        {
            var f = await Analyze("public struct Cell { Cell[] _neighbors; List<Cell> _history; Cell[][] _grid; NDArray _value; }", "struct_self");
            Assert.AreEqual("_value", f.Members16("Cell"), f.All);
        }

        // ---------------------------------------------------------------- generics

        /// <summary>
        ///     A generic tree over an unconstrained payload: the definition cannot know whether it holds and
        ///     draws nothing; <c>TreeNode&lt;NDArray&gt;</c> is a holder in its owner and <c>TreeNode&lt;int&gt;</c>
        ///     is not.
        /// </summary>
        [TestMethod]
        public async Task GenericTree_UnconstrainedPayload_OnlyTheInstantiationHolds()
        {
            var f = await Analyze(
                "public class TreeNode<T> { TreeNode<T> _parent; List<TreeNode<T>> _children = new List<TreeNode<T>>(); T _payload; }\n" +
                "public class Tree { TreeNode<NDArray> _root; TreeNode<int> _counts; }", "generic_tree");
            Assert.IsNull(f.Members16("TreeNode<T>"), "an unconstrained T cannot know whether it holds:\n" + f.All);
            Assert.AreEqual("_root", f.Members16("Tree"), "TreeNode<NDArray> holds its payload; TreeNode<int> holds nothing:\n" + f.All);
            Assert.AreEqual(1, f.Total, f.All);
        }

        /// <summary>
        ///     A generic tree whose payload is constrained to NDArray holds in its definition: the payload is
        ///     the holder (T reaches NDArray through its constraint, never through the tree), the parent and the
        ///     children are links.
        /// </summary>
        [TestMethod]
        public async Task GenericTree_PayloadConstrainedToNDArray_OnlyThePayloadHolds()
        {
            var f = await Analyze("public class TreeNode<T> where T : NDArray { TreeNode<T> _parent; List<TreeNode<T>> _children; T _payload; }", "generic_tree_constrained");
            Assert.AreEqual("_payload", f.Members16("TreeNode<T>"), f.All);
        }

        /// <summary>
        ///     The curiously recurring base (<c>Entity&lt;TSelf&gt; where TSelf : Entity&lt;TSelf&gt;</c>), common in
        ///     domain models: <c>TSelf</c> leads back to the entity through its constraint, so the self-typed
        ///     parent and children are links. A subclass that only adds a member of its own type holds through
        ///     its base and lists nothing; a subclass storing other entities lists them.
        /// </summary>
        [TestMethod]
        public async Task CuriouslyRecurringBase_SelfTypedParameter_IsNotAHolder()
        {
            var f = await Analyze(
                "public abstract class Entity<TSelf> where TSelf : Entity<TSelf> { TSelf _parent; List<TSelf> _children; NDArray _features; }\n" +
                "public class Person : Entity<Person> { Person _spouse; }\n" +
                "public class Company : Entity<Company> { List<Person> _staff; }", "crtp");
            Assert.AreEqual("_features", f.Members16("Entity<TSelf>"), f.All);
            Assert.IsNull(f.Members16("Person"), "Person holds only through its base; its spouse is its own type:\n" + f.All);
            Assert.AreEqual("_staff", f.Members16("Company"), f.All);
            Assert.AreEqual(2, f.Total, f.All);
        }

        /// <summary>
        ///     Another instantiation of the declaring generic is another type: <c>Node&lt;int&gt;</c> inside
        ///     <c>Node&lt;T&gt;</c> holds its own array without passing through <c>Node&lt;T&gt;</c>, so it is a
        ///     holder, while <c>Node&lt;T&gt;</c> itself (the owner) is a link.
        /// </summary>
        [TestMethod]
        public async Task DifferentInstantiationOfTheOwnGeneric_IsAHolder()
        {
            var f = await Analyze("public class Node<T> { Node<int> _other; Node<T> _next; T _value; NDArray _data; }", "different_instantiation");
            Assert.AreEqual("_other,_data", f.Members16("Node<T>"), f.All);
        }

        /// <summary>A nested type and its outer type on one cycle: the outer type's member of the nested type holds (the nested type stores the array); the nested type's link back to the outer type does not.</summary>
        [TestMethod]
        public async Task NestedTypes_OuterAndInner_FormOneCycle()
        {
            var f = await Analyze("public class Outer { Inner _i; public class Inner { Outer _o; NDArray _x; } }", "nested_types");
            Assert.AreEqual("_i", f.Members16("Outer"), f.All);
            Assert.AreEqual("_x", f.Members16("Inner"), "Outer reaches arrays only through Inner:\n" + f.All);
        }

        // ---------------------------------------------------------------- [NDBorrowed] inside a cycle

        /// <summary>
        ///     A borrowed member is not an edge, so it CUTS the cycle: with A's member of B borrowed, A no longer
        ///     reaches B, B's link to A is an ordinary member of another component, and A holds its own array —
        ///     so B's link IS a holder.
        /// </summary>
        [TestMethod]
        public async Task BorrowedMember_CutsTheCycle()
        {
            var f = await Analyze("public class A { [NDBorrowed] B _b; NDArray _x; }\npublic class B { A _a; }", "borrowed_member_cycle");
            Assert.AreEqual("_x", f.Members16("A"), f.All);
            Assert.AreEqual("_a", f.Members16("B"), f.All);
        }

        /// <summary>
        ///     A borrowed TYPE on a cycle holds nothing and is never reported, so a member of that type is not a
        ///     holder either — neither A's link into the cycle nor the link of C, a type outside it.
        ///     <para>
        ///     C is the part that tests the model: the reporting site skips an [NDBorrowed] type on its own, and
        ///     A's link would not be a holder even if B were expanded (A dominates B, the cycle's only path to
        ///     an array). Only C, in another component, sees B's answer directly: if the model expanded a
        ///     borrowed type, B would reach A's array and C's '_b' would become a holder. Without C the test
        ///     passed with that rule removed (a mutation run, 2026-10-24).
        ///     </para>
        /// </summary>
        [TestMethod]
        public async Task BorrowedType_InTheCycle_HoldsNothing()
        {
            var f = await Analyze("public class A { B _b; NDArray _x; }\n[NDBorrowed] public class B { A _a; }\npublic class C { B _b; }", "borrowed_type_cycle");
            Assert.AreEqual("_x", f.Members16("A"), f.All);
            Assert.IsNull(f.Members16("C"), "C stores only the borrowed type, which holds nothing:\n" + f.All);
            Assert.AreEqual(1, f.Total, f.All);
        }

        // ---------------------------------------------------------------- several anchors, several components

        /// <summary>A ring in which every type stores its own array: every link reaches the NEXT type's array without passing through its owner, so every link is a holder.</summary>
        [TestMethod]
        public async Task RingWhereEveryTypeHoldsItsOwnArray_EveryLinkIsAHolder()
        {
            var f = await Analyze("public class A { B _b; NDArray _x; }\npublic class B { C _c; NDArray _y; }\npublic class C { A _a; NDArray _z; }", "all_anchors_ring");
            Assert.AreEqual("_b,_x", f.Members16("A"), f.All);
            Assert.AreEqual("_c,_y", f.Members16("B"), f.All);
            Assert.AreEqual("_a,_z", f.Members16("C"), f.All);
        }

        /// <summary>
        ///     Two cycles joined by one edge (A ↔ B → C ↔ D, D storing the array): B's link back to A reaches
        ///     arrays only through B, but its link to C is in another component that holds; D's link back to C
        ///     leads only back to D.
        /// </summary>
        [TestMethod]
        public async Task TwoCyclesJoinedByOneEdge_EachJudgedOnItsOwnComponent()
        {
            var f = await Analyze("public class A { B _b; }\npublic class B { A _a; C _c; }\npublic class C { D _d; }\npublic class D { C _c; NDArray _x; }", "two_cycles");
            Assert.AreEqual("_b", f.Members16("A"), f.All);
            Assert.AreEqual("_c", f.Members16("B"), f.All);
            Assert.AreEqual("_d", f.Members16("C"), f.All);
            Assert.AreEqual("_x", f.Members16("D"), f.All);
        }

        /// <summary>
        ///     A base type with a member of its derived type, and the derived type with a member of the base:
        ///     the base's member of the derived type comes back to the base (a base type is an edge), but the
        ///     derived type's member of the base reaches the base's own array — a holder.
        /// </summary>
        [TestMethod]
        public async Task BaseAndDerivedPointingAtEachOther_DerivedPeerIsAHolder()
        {
            var f = await Analyze("public class Base { Derived _d; NDArray _a; }\npublic class Derived : Base { Base _peer; }", "base_peer");
            Assert.AreEqual("_a", f.Members16("Base"), f.All);
            Assert.AreEqual("_peer", f.Members16("Derived"), f.All);
        }

        // ---------------------------------------------------------------- scale

        /// <summary>
        ///     A ring of 2,000 types with the array in T0: every link but T0's reaches T0 without passing its
        ///     owner, and T0's link leads all the way round back to T0. Master and PR #632's commit overflow the
        ///     stack on this ring (the test host crashes).
        /// </summary>
        [TestMethod]
        [Timeout(180000)]
        public async Task LongRing_OneAnchor_EveryLinkButTheAnchorsIsAHolder()
        {
            const int n = 2000;
            var sb = new StringBuilder();
            for (int i = 0; i < n; i++)
                sb.Append("public class T").Append(i).Append(" { T").Append((i + 1) % n).Append(" _next;").Append(i == 0 ? " NDArray _x;" : "").AppendLine(" }");
            var f = await Analyze(sb.ToString(), "long_ring");
            Assert.AreEqual("_x", f.Members16("T0"), Head(f.All));
            for (int i = 1; i < n; i++)
                if (f.Members16("T" + i) != "_next")
                    Assert.Fail($"T{i}'s link reaches T0 without passing T{i}; got [{f.Members16("T" + i)}]");
            Assert.AreEqual(n, f.Total);
        }

        /// <summary>
        ///     The star of a hub and <paramref name="spokes"/> spoke types, each spoke linking back to the hub.
        /// </summary>
        /// <param name="spokes">How many spoke types.</param>
        /// <param name="everySpokeHolds">True to give every spoke its own array; false to give one only to S0.</param>
        /// <returns>The declarations.</returns>
        private static string StarHub(int spokes, bool everySpokeHolds)
        {
            var sb = new StringBuilder("public class Hub {");
            for (int i = 0; i < spokes; i++)
                sb.Append(" S").Append(i).Append(" _s").Append(i).Append(';');
            sb.AppendLine(" }");
            for (int i = 0; i < spokes; i++)
                sb.Append("public class S").Append(i).Append(" { Hub _hub;").Append(everySpokeHolds || i == 0 ? " NDArray _x;" : "").AppendLine(" }");
            return sb.ToString();
        }

        /// <summary>
        ///     A hub with 300 spokes of which only S0 stores an array: the hub's only holder is its member of S0
        ///     (every other spoke reaches S0 only through the hub), S0's link to the hub comes back to S0, and
        ///     every other spoke's link reaches S0 through the hub without passing that spoke — a holder. The
        ///     dominator tree decides all 600 answers at once; PR #632's commit called every hub member a holder.
        /// </summary>
        [TestMethod]
        public async Task StarHub_OnlyOneSpokeHolds_OnlyItsLinkIsTheHubsHolder()
        {
            const int n = 300;
            var f = await Analyze(StarHub(n, everySpokeHolds: false), "star_one");
            Assert.AreEqual("_s0", f.Members16("Hub"), Head(f.All));
            Assert.AreEqual("_x", f.Members16("S0"), Head(f.All));
            for (int i = 1; i < n; i++)
                if (f.Members16("S" + i) != "_hub")
                    Assert.Fail($"S{i}'s link reaches S0 through the hub without passing S{i}; got [{f.Members16("S" + i)}]");
            Assert.AreEqual(n + 1, f.Total);
        }

        /// <summary>A hub whose 300 spokes all store arrays: every hub member and every spoke's link is a holder (the hub reaches the other spokes' arrays), and the hub's message names three and counts the rest.</summary>
        [TestMethod]
        public async Task StarHub_EverySpokeHolds_EveryLinkIsAHolder()
        {
            const int n = 300;
            var f = await Analyze(StarHub(n, everySpokeHolds: true), "star_all");
            StringAssert.Contains(f.All, "'Hub' stores NDArrays in '_s0', '_s1', '_s2' and 297 more but", Head(f.All));
            for (int i = 0; i < n; i++)
                if (f.Members16("S" + i) != "_hub,_x")
                    Assert.Fail($"S{i}: the hub reaches the other spokes' arrays; got [{f.Members16("S" + i)}]");
        }

        /// <summary>
        ///     An inheritance chain of 3,000 classes, each with a member of its own type and the array only in
        ///     the root: every level holds through its base, but its own member is a self-reference, so only the
        ///     root draws NDW016. PR #632's commit drew all 3,000.
        /// </summary>
        [TestMethod]
        [Timeout(180000)]
        public async Task DeepInheritanceChain_SelfReferencesAtEveryLevel_OnlyTheRootHolds()
        {
            const int depth = 3000;
            var sb = new StringBuilder("public class L0 { NDArray _x; L0 _self; }\n");
            for (int i = 1; i < depth; i++)
                sb.Append("public class L").Append(i).Append(" : L").Append(i - 1).Append(" { L").Append(i).AppendLine(" _self; }");
            var f = await Analyze(sb.ToString(), "deep_inheritance");
            Assert.AreEqual("_x", f.Members16("L0"), Head(f.All));
            Assert.AreEqual(1, f.Total, Head(f.All));
        }

        /// <summary>
        ///     A chain of 8,000 classes linked through <c>List&lt;T&gt;</c> (16,000 nodes on the walk, every list an
        ///     intermediate node), the array at the end: every level holds. Master and PR #632's commit overflow
        ///     the stack on it.
        /// </summary>
        [TestMethod]
        [Timeout(180000)]
        public async Task DeepChainThroughGenericContainers_DoesNotOverflowTheStack()
        {
            const int depth = 8000;
            var sb = new StringBuilder();
            for (int i = 0; i < depth; i++)
                sb.AppendLine(i + 1 < depth ? $"public class L{i} {{ List<L{i + 1}> _next; }}" : $"public class L{i} {{ NDArray _leaf; }}");
            var r = await AnalyzerTestHarness.RunAsync(H + sb, "deep_list_chain.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty);
            Assert.AreEqual(depth, r.CountOf("NDW016"), "every level stores the leaf through a list of the next level");
        }

        // ---------------------------------------------------------------- messages

        /// <summary>The NDW016 text names and counts only holders, in declaration order: four arrays among three links read "'_a', '_b', '_c' and 1 more".</summary>
        [TestMethod]
        public async Task Message_NamesAndCountsOnlyTheHolders()
        {
            var f = await Analyze("public class Node { NDArray _a; Node _next; NDArray _b; Node _prev; NDArray _c; NDArray _d; List<Node> _all; }", "message_count");
            StringAssert.Contains(f.All, "'Node' stores NDArrays in '_a', '_b', '_c' and 1 more but", f.All);
        }

        // ---------------------------------------------------------------- disposal and contagion through cycles

        /// <summary>A parent that holds only through its children and disposes them, and children that dispose their own arrays: nothing to report — the child's back-pointer is not its array.</summary>
        [TestMethod]
        public async Task ParentHoldingOnlyThroughItsChildren_DisposingThem_IsClean()
        {
            var f = await Analyze(
                "public class Parent : IDisposable { List<Child> _kids = new List<Child>(); public void Dispose() { foreach (var k in _kids) k.Dispose(); } }\n" +
                "public class Child : IDisposable { Parent _p; NDArray _y; public void Dispose() => _y?.Dispose(); }", "parent_through_children");
            Assert.AreEqual(0, f.Total, f.All);
        }

        /// <summary>
        ///     The same parent disposing nothing: its children ARE its holders (the children's arrays are not
        ///     reached through the parent), so the parent draws NDW017 for them, and the child still draws
        ///     nothing for its back-pointer.
        /// </summary>
        [TestMethod]
        public async Task ParentHoldingOnlyThroughItsChildren_DisposingNothing_FlagsOnlyTheChildren()
        {
            var f = await Analyze(
                "public class Parent : IDisposable { List<Child> _kids = new List<Child>(); public void Dispose() { } }\n" +
                "public class Child : IDisposable { Parent _p; NDArray _y; public void Dispose() => _y?.Dispose(); }", "parent_through_children_undisposed");
            Assert.AreEqual("_kids", f.Members17("Parent"), f.All);
            Assert.AreEqual("", f.Members17("Child"), f.All);
            Assert.AreEqual(1, f.Total, f.All);
        }

        /// <summary>A disposable node that disposes nothing draws exactly one NDW017, for its array: its link and its array of links are not holders.</summary>
        [TestMethod]
        public async Task DisposableSelfReference_DisposingNothing_FlagsOnlyTheArray()
        {
            var f = await Analyze("public class Node : IDisposable { Node _next; Node[] _all; NDArray _v; public void Dispose() { } }", "self_disposable_nothing");
            Assert.AreEqual("_v", f.Members17("Node"), f.All);
            Assert.AreEqual(1, f.Total, f.All);
        }

        /// <summary>
        ///     Contagion (NDW012) through cycles: dropping a disposable that holds arrays only THROUGH a cycle
        ///     (A ↔ B, B storing the array) leaks; dropping a disposable on a cycle that stores no array (C ↔ D)
        ///     does not. No NDW016/NDW017: B's link back to A is not a holder, and every holder is disposed.
        /// </summary>
        [TestMethod]
        public async Task Contagion_ThroughACycle_FlagsTheDroppedHolder_NotTheArrayFreeCycle()
        {
            var r = await AnalyzerTestHarness.RunAsync(H +
                "public class A : IDisposable { B _b; public void Dispose() => _b?.Dispose(); }\n" +
                "public class B : IDisposable { A _a; NDArray _x; public void Dispose() => _x?.Dispose(); }\n" +
                "public class C : IDisposable { D _d; public void Dispose() { } }\n" +
                "public class D { C _c; }\n" +
                "public static class Use {\n" +
                "  public static void M(NDArray a) {\n" +
                "    new A();\n" +
                "    new C();\n" +
                "  }\n" +
                "}", "contagion_cycle.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, string.Join("\n", r.CompileErrors));
            // The header H is three lines, so body line k is file line k + 3; line 7 is "new A();".
            var leaks = r.Ndw.Where(d => d.Id == "NDW012").Select(d => AnalyzerTestHarness.LineOf(d) - 3).ToList();
            CollectionAssert.AreEqual(new[] { 7 }, leaks, string.Join("\n", r.Ndw.Select(d => d.Id + ": " + d.GetMessage())));
            Assert.AreEqual(1, r.Ndw.Length, string.Join("\n", r.Ndw.Select(d => d.Id + ": " + d.GetMessage())));
        }

        // ---------------------------------------------------------------- broken code

        /// <summary>
        ///     Unresolved types on a cycle (a broken build the IDE analyzes while the author types): the analyzer
        ///     must not throw — the harness fails on any AD0001 — and the resolvable members are judged as usual,
        ///     an unknown type being no edge at all.
        /// </summary>
        [TestMethod]
        public async Task UnresolvedTypesInACycle_DoNotCrash_AndResolvableMembersAreJudged()
        {
            var r = await AnalyzerTestHarness.RunAsync(H +
                "public class Node { Node _next; MissingType _m; NDArray _v; }\n" +
                "public class A { B _b; }\n" +
                "public class B { A _a; Missing<A> _m; NDArray _x; }", "unresolved_types.cs");
            Assert.AreEqual(2, r.CompileErrors.Count(e => e.Id == "CS0246"), "broken on purpose: two unknown types\n" + string.Join("\n", r.CompileErrors));
            var f = Parse(r);
            Assert.AreEqual("_v", f.Members16("Node"), f.All);
            Assert.AreEqual("_b", f.Members16("A"), f.All);
            Assert.AreEqual("_x", f.Members16("B"), f.All);
            Assert.AreEqual(3, f.Total, f.All);
        }

        // ---------------------------------------------------------------- determinism across files and threads

        /// <summary>
        ///     The order trap master fell into, end to end: an earlier file's Dispose body hands its TreeNode
        ///     to a call, and the hand-off check asks the shared model whether TreeNode holds before (or while)
        ///     TreeNode's own symbol is analyzed. TreeNode's holder list must not change — sequentially, and in
        ///     five concurrent runs. (The sink is generic on purpose: the hand-off check judges the argument's
        ///     own type, and an <c>object</c> parameter would make that type <c>object</c>, never TreeNode.)
        /// </summary>
        [TestMethod]
        public async Task HandOffInAnEarlierFile_DoesNotChangeTheHoldersOfTheTypeItAsksAbout()
        {
            var files = new List<(string Path, string Source)>
            {
                ("a.cs", H + "public class Owner : IDisposable { TreeNode _root; public void Dispose() { Sink.Take(_root); } }\npublic static class Sink { public static void Take<T>(T o) { } }\n"),
                ("b.cs", H + "public class TreeNode { TreeNode _parent; List<TreeNode> _children; NDArray _data; }\n"),
            };
            for (int run = 0; run < 6; run++)
            {
                bool concurrent = run > 0;
                var r = await AnalyzerTestHarness.RunProjectAsync(files, "handoff_first", concurrent);
                Assert.IsTrue(r.CompileErrors.IsEmpty, string.Join("\n", r.CompileErrors));
                var f = Parse(r);
                Assert.AreEqual("_data", f.Members16("TreeNode"), $"run {run} (concurrent {concurrent}):\n{f.All}");
                Assert.AreEqual("", f.Members17("Owner"), $"run {run}: the hand-off disposes Owner's only holder:\n{f.All}");
                Assert.AreEqual(1, f.Total, $"run {run}:\n{f.All}");
            }
        }

        /// <summary>
        ///     A random project of 240 classes in 40 files — cross-file cycles of entity links, arrays in some
        ///     classes, disposables whose Dispose bodies forget some members and hand others off (each hand-off
        ///     asks the model about a type, from whichever thread runs that body) — analyzed once sequentially
        ///     and six times concurrently: every run must give exactly the sequential diagnostics, text for text.
        /// </summary>
        [TestMethod]
        [Timeout(300000)]
        public async Task RandomMultiFileProject_ConcurrentRuns_EqualTheSequentialRun()
        {
            var files = RandomProject(seed: 11, fileCount: 40, typesPerFile: 6);
            var sequential = await AnalyzerTestHarness.RunProjectAsync(files, "random_project", concurrentAnalysis: false);
            Assert.IsTrue(sequential.CompileErrors.IsEmpty, string.Join("\n", sequential.CompileErrors.Take(10)));
            var baseline = Signature(sequential);
            Assert.IsTrue(sequential.CountOf("NDW016") > 20 && sequential.CountOf("NDW017") > 20,
                $"the project must exercise both diagnostics (NDW016 {sequential.CountOf("NDW016")}, NDW017 {sequential.CountOf("NDW017")})");
            for (int run = 0; run < 6; run++)
            {
                var concurrent = Signature(await AnalyzerTestHarness.RunProjectAsync(files, "random_project", concurrentAnalysis: true));
                CollectionAssert.AreEqual(baseline, concurrent, $"concurrent run {run} differs from the sequential run");
            }
        }

        /// <summary>
        ///     The random project of <see cref="RandomMultiFileProject_ConcurrentRuns_EqualTheSequentialRun"/>:
        ///     classes linking to random classes of any file (direct, list and array links), a tenth of them
        ///     storing an array, half of them disposable with a Dispose body that forgets or hands off each member.
        /// </summary>
        /// <param name="seed">The random seed.</param>
        /// <param name="fileCount">How many files.</param>
        /// <param name="typesPerFile">How many classes per file.</param>
        /// <returns>The files.</returns>
        private static List<(string Path, string Source)> RandomProject(int seed, int fileCount, int typesPerFile)
        {
            var rng = new Random(seed);
            int n = fileCount * typesPerFile;
            var files = new List<(string Path, string Source)>();
            for (int file = 0; file < fileCount; file++)
            {
                var sb = new StringBuilder(H);
                if (file == 0)
                    sb.AppendLine("public static class Sink { public static void Take<T>(T o) { } }"); // generic: the hand-off check sees the member's own type
                for (int k = 0; k < typesPerFile; k++)
                {
                    int i = file * typesPerFile + k;
                    bool disposable = rng.Next(2) == 0;
                    sb.Append("public class E").Append(i).Append(disposable ? " : IDisposable" : "").Append(" {");
                    var names = new List<string>();
                    int members = 1 + rng.Next(3);
                    for (int m = 0; m < members; m++)
                    {
                        int j = rng.Next(n);
                        string name = "_m" + m;
                        switch (rng.Next(3))
                        {
                            case 0: sb.Append(" E").Append(j).Append(' ').Append(name).Append(';'); break;
                            case 1: sb.Append(" List<E").Append(j).Append("> ").Append(name).Append(';'); break;
                            default: sb.Append(" E").Append(j).Append("[] ").Append(name).Append(';'); break;
                        }
                        names.Add(name);
                    }
                    if (rng.Next(10) == 0)
                    {
                        sb.Append(" NDArray _data;");
                        names.Add("_data");
                    }
                    if (disposable)
                    {
                        sb.Append(" public void Dispose() {");
                        foreach (var name in names)
                        {
                            int choice = rng.Next(3);
                            if (choice == 0)
                                continue; // forgotten: an NDW017 when the member is a holder
                            if (name == "_data" && choice == 2)
                                sb.Append(" _data?.Dispose();");
                            else
                                sb.Append(" Sink.Take(").Append(name).Append(");"); // a hand-off: asks whether the member's type holds
                        }
                        sb.Append(" }");
                    }
                    sb.AppendLine(" }");
                }
                files.Add(($"f{file:D2}.cs", sb.ToString()));
            }
            return files;
        }

        /// <summary>One comparable line per diagnostic: file, line, id and message.</summary>
        /// <param name="r">An analyzer result.</param>
        /// <returns>The lines, in the result's (run-independent) order.</returns>
        private static List<string> Signature(AnalyzerResult r)
            => r.Ndw.Select(d => $"{d.Location.SourceTree?.FilePath}:{AnalyzerTestHarness.LineOf(d)}:{d.Id}:{d.GetMessage()}").ToList();
    }
}
