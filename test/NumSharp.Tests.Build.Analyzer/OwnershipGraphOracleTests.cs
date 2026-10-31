using System;
using System.Collections.Generic;
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
    ///     A differential gate for the ownership graph: seeded random type GRAPHS — cycles,
    ///     self-references, base types, containers (<c>List&lt;T&gt;</c>, arrays, tuples, a source
    ///     <c>Box&lt;T&gt;</c>, auto-properties) next to members that never carry ownership (borrowed,
    ///     delegates, comparers, statics, computed properties, scalars) — checked member by member against
    ///     an ORACLE written independently of the analyzer: a breadth-first search over the generator's
    ///     own adjacency lists.
    ///     <para>
    ///     The rule the oracle encodes: a member of type <c>T</c> is a holder when it is an NDArray, or
    ///     its type reaches a type that stores an NDArray along a path that never passes through
    ///     <c>T</c> (a member that holds only by coming back to <c>T</c> is a back-reference). Every
    ///     disposable type draws one NDW017 per holder (its Dispose disposes nothing), every other type
    ///     one NDW016 naming its holders — so the diagnostics expose the exact holder set of every type.
    ///     </para>
    ///     <para>
    ///     The EXTENDED palette adds the shapes where the member type is not a graph type but a chain of
    ///     generic nodes in between: a source <c>Pair&lt;A, B&gt;</c> and <c>KeyValuePair&lt;K, V&gt;</c>
    ///     with two graph targets, nested and BCL containers, jagged arrays, nullable tuples, a pair that
    ///     carries an NDArray beside a graph type, and a class whose BASE is <c>Box&lt;T&gt;</c>. Those
    ///     intermediate nodes join the strongly connected components, so they test that the dominator
    ///     rule sees through them exactly as the oracle's "is there a path that avoids the owner" does.
    ///     The direct-model test then runs graphs of hundreds of classes through
    ///     <see cref="OwnershipModel"/> itself, where components of dozens of types give the dominator
    ///     tree real depth.
    ///     </para>
    /// </summary>
    [TestClass]
    public class OwnershipGraphOracleTests
    {
        /// <summary>
        ///     How a generated member relates to its target type. The first six are EDGES (a value of the
        ///     target is stored, so the target's arrays become the member's), <see cref="Shape.NDArray"/> is an
        ///     array member, and the next six store no value of the target (all but <see cref="Shape.Scalar"/>
        ///     still name it) — each a rule of the model the oracle must agree with. The shapes after
        ///     <see cref="Shape.Computed"/> are the extended palette: every one is an edge through one or
        ///     more intermediate generic nodes.
        /// </summary>
        private enum Shape
        {
            /// <summary>A field of the target type.</summary>
            Direct,

            /// <summary>A <c>List&lt;T&gt;</c> of the target (a collection's elements are stored).</summary>
            List,

            /// <summary>An array of the target.</summary>
            Array,

            /// <summary>A tuple with the target as an element.</summary>
            Tuple,

            /// <summary>A source <c>Box&lt;T&gt;</c> of the target (a user generic whose field stores a T).</summary>
            Box,

            /// <summary>An auto-property of the target type (its backing field stores one).</summary>
            AutoProperty,

            /// <summary>An NDArray field: holds by itself.</summary>
            NDArray,

            /// <summary>An <c>[NDBorrowed]</c> field of the target: never owned.</summary>
            Borrowed,

            /// <summary>A <c>Func&lt;T&gt;</c>: a delegate stores no value of T.</summary>
            Func,

            /// <summary>An <c>IComparer&lt;T&gt;</c>: a non-holding generic family.</summary>
            Comparer,

            /// <summary>A static field: not per-instance state.</summary>
            Static,

            /// <summary>An <c>int</c> field: mentions no type of the graph.</summary>
            Scalar,

            /// <summary>A computed (expression-bodied) property: no backing field.</summary>
            Computed,

            /// <summary>Extended: a source <c>Pair&lt;A, B&gt;</c> of two targets (two edges through one node).</summary>
            Pair,

            /// <summary>Extended: a <c>List&lt;List&lt;T&gt;&gt;</c> (two intermediate nodes).</summary>
            NestedList,

            /// <summary>Extended: a <c>Dictionary&lt;string, T&gt;</c> (the value type is stored).</summary>
            DictValue,

            /// <summary>Extended: a jagged array <c>T[][]</c>.</summary>
            Jagged,

            /// <summary>Extended: a nullable tuple <c>(T, int)?</c> (Nullable over a ValueTuple).</summary>
            NullableTuple,

            /// <summary>Extended: a <c>Lazy&lt;T&gt;</c> (a BCL generic outside the non-holding families).</summary>
            LazyOf,

            /// <summary>Extended: an <c>IEnumerable&lt;T&gt;</c> (a BCL interface: its type argument is the edge).</summary>
            Sequence,

            /// <summary>Extended: a <c>KeyValuePair&lt;K, V&gt;</c> of two targets (a BCL struct, generic arguments only).</summary>
            KeyValue,

            /// <summary>Extended: a <c>Pair&lt;T, NDArray&gt;</c> — an edge that also stores an array directly, so always a holder.</summary>
            PairWithArray
        }

        /// <summary>One generated member and the facts the oracle reads about it.</summary>
        private sealed class Member
        {
            /// <summary>The member's name as declared (and as diagnostics quote it).</summary>
            public string Name;

            /// <summary>Its C# declaration.</summary>
            public string Declaration;

            /// <summary>The graph types the member's type contains, when it is an edge (two for the two-argument shapes); empty otherwise.</summary>
            public int[] Targets = System.Array.Empty<int>();

            /// <summary>True when the member stores values of its <see cref="Targets"/> (an ownership edge).</summary>
            public bool Edge;

            /// <summary>True for a member that stores an NDArray directly (an NDArray field, a pair carrying one), which holds by itself.</summary>
            public bool Array;
        }

        /// <summary>One generated class.</summary>
        private sealed class GraphType
        {
            /// <summary>The class name, prefixed per graph so that graphs compiled together share no type.</summary>
            public string Name;

            /// <summary>True when the class itself implements IDisposable (with a Dispose that disposes nothing).</summary>
            public bool DeclaresDisposable;

            /// <summary>The index of its base class in the graph, -1 for none; always a lower index, so inheritance is acyclic.</summary>
            public int Base = -1;

            /// <summary>Extended palette: the graph type <c>T</c> of a <c>Box&lt;T&gt;</c> base class, -1 for none (never set together with <see cref="Base"/>).</summary>
            public int GenericBase = -1;

            /// <summary>Its members in declaration order.</summary>
            public readonly List<Member> Members = new List<Member>();
        }

        /// <summary>A generated graph: its seed, its types, and their C# source.</summary>
        private sealed class Graph
        {
            /// <summary>The seed that reproduces it.</summary>
            public int Seed;

            /// <summary>Its classes, indexed as members and bases refer to them.</summary>
            public GraphType[] Types;

            /// <summary>The C# declarations of <see cref="Types"/>.</summary>
            public string Source;
        }

        /// <summary>
        ///     The original thirteen shapes, in declaration order. The original tests draw from exactly this
        ///     array, so their seeds keep producing the graphs they always produced.
        /// </summary>
        private static readonly Shape[] OriginalShapes =
        {
            Shape.Direct, Shape.List, Shape.Array, Shape.Tuple, Shape.Box, Shape.AutoProperty, Shape.NDArray,
            Shape.Borrowed, Shape.Func, Shape.Comparer, Shape.Static, Shape.Scalar, Shape.Computed
        };

        /// <summary>Every shape, the original and the extended ones, drawn uniformly by the extended tests.</summary>
        private static readonly Shape[] ExtendedShapes = (Shape[])Enum.GetValues(typeof(Shape));

        /// <summary>
        ///     The declarations every compiled batch starts with: the usings the shapes need and the two
        ///     source generics they use. Neither generic holds by itself (their type parameters are
        ///     unconstrained), so they add no diagnostic of their own.
        /// </summary>
        private const string Header =
            "using System;\nusing System.Collections.Generic;\nusing NumSharp;\n" +
            "public class Box<T> { public T Value; }\npublic class Pair<A, B> { public A First; public B Second; }\n";

        /// <summary>Generates one random graph of classes: members of every shape of the palette pointing at random classes (cycles and self-references included), acyclic base classes, random IDisposable.</summary>
        /// <param name="seed">The random seed; the same seed gives the same graph.</param>
        /// <param name="prefix">Prefixed to every class name, unique per graph.</param>
        /// <param name="minTypes">The fewest classes.</param>
        /// <param name="maxTypes">The most classes.</param>
        /// <param name="maxMembers">The most members per class (each has between 0 and this many).</param>
        /// <param name="extended">
        ///     Draw from <see cref="ExtendedShapes"/> and give some classes a <c>Box&lt;T&gt;</c> base. False
        ///     draws exactly the random numbers the generator always drew, so existing seeds are stable.
        /// </param>
        /// <returns>The graph with its source.</returns>
        private static Graph Generate(int seed, string prefix, int minTypes, int maxTypes, int maxMembers, bool extended = false)
        {
            var rng = new Random(seed);
            var palette = extended ? ExtendedShapes : OriginalShapes;
            int n = minTypes + rng.Next(maxTypes - minTypes + 1);
            var types = new GraphType[n];
            for (int i = 0; i < n; i++)
            {
                var t = types[i] = new GraphType { Name = prefix + "T" + i, DeclaresDisposable = rng.Next(2) == 0 };
                if (i > 0 && rng.Next(4) == 0)
                    t.Base = rng.Next(i); // a lower index: inheritance stays acyclic
                // Only the extended generator draws these numbers, so the original stream is unchanged.
                if (extended && t.Base < 0 && rng.Next(6) == 0)
                    t.GenericBase = rng.Next(n); // Box<T> is not a graph type, so any target is acyclic
                int members = rng.Next(maxMembers + 1);
                for (int k = 0; k < members; k++)
                {
                    var shape = palette[rng.Next(palette.Length)];
                    int j = rng.Next(n);
                    int j2 = extended ? rng.Next(n) : -1;
                    string target = prefix + "T" + j;
                    string target2 = j2 >= 0 ? prefix + "T" + j2 : null;
                    string name = (shape == Shape.AutoProperty ? "P" : shape == Shape.Computed ? "V" : "_m") + i + "_" + k;
                    var m = new Member { Name = name };
                    switch (shape)
                    {
                        case Shape.Direct: m.Declaration = $"{target} {name};"; m.Edge = true; break;
                        case Shape.List: m.Declaration = $"List<{target}> {name};"; m.Edge = true; break;
                        case Shape.Array: m.Declaration = $"{target}[] {name};"; m.Edge = true; break;
                        case Shape.Tuple: m.Declaration = $"({target}, int) {name};"; m.Edge = true; break;
                        case Shape.Box: m.Declaration = $"Box<{target}> {name};"; m.Edge = true; break;
                        case Shape.AutoProperty: m.Declaration = $"public {target} {name} {{ get; set; }}"; m.Edge = true; break;
                        case Shape.NDArray: m.Declaration = $"NDArray {name};"; m.Array = true; break;
                        case Shape.Borrowed: m.Declaration = $"[NDBorrowed] {target} {name};"; break;
                        case Shape.Func: m.Declaration = $"Func<{target}> {name};"; break;
                        case Shape.Comparer: m.Declaration = $"IComparer<{target}> {name};"; break;
                        case Shape.Static: m.Declaration = $"static {target} {name};"; break;
                        case Shape.Scalar: m.Declaration = $"int {name};"; break;
                        case Shape.Computed: m.Declaration = $"public {target} {name} => null;"; break;
                        case Shape.Pair: m.Declaration = $"Pair<{target}, {target2}> {name};"; m.Edge = true; m.Targets = new[] { j, j2 }; break;
                        case Shape.NestedList: m.Declaration = $"List<List<{target}>> {name};"; m.Edge = true; break;
                        case Shape.DictValue: m.Declaration = $"Dictionary<string, {target}> {name};"; m.Edge = true; break;
                        case Shape.Jagged: m.Declaration = $"{target}[][] {name};"; m.Edge = true; break;
                        case Shape.NullableTuple: m.Declaration = $"({target}, int)? {name};"; m.Edge = true; break;
                        case Shape.LazyOf: m.Declaration = $"Lazy<{target}> {name};"; m.Edge = true; break;
                        case Shape.Sequence: m.Declaration = $"IEnumerable<{target}> {name};"; m.Edge = true; break;
                        case Shape.KeyValue: m.Declaration = $"KeyValuePair<{target}, {target2}> {name};"; m.Edge = true; m.Targets = new[] { j, j2 }; break;
                        case Shape.PairWithArray: m.Declaration = $"Pair<{target}, NDArray> {name};"; m.Edge = true; m.Array = true; break;
                    }
                    if (m.Edge && m.Targets.Length == 0)
                        m.Targets = new[] { j };
                    t.Members.Add(m);
                }
            }

            var sb = new StringBuilder();
            foreach (var t in types)
            {
                sb.Append("public class ").Append(t.Name);
                var bases = new List<string>();
                if (t.Base >= 0)
                    bases.Add(types[t.Base].Name);
                if (t.GenericBase >= 0)
                    bases.Add("Box<" + types[t.GenericBase].Name + ">");
                if (t.DeclaresDisposable)
                    bases.Add("IDisposable");
                if (bases.Count > 0)
                    sb.Append(" : ").Append(string.Join(", ", bases));
                sb.Append(" {");
                foreach (var m in t.Members)
                    sb.Append(' ').Append(m.Declaration);
                if (t.DeclaresDisposable)
                    sb.Append(" public void Dispose() { }");
                sb.AppendLine(" }");
            }
            return new Graph { Seed = seed, Types = types, Source = sb.ToString() };
        }

        /// <summary>The graph types a value of type <paramref name="t"/> stores directly: its edges' targets, its graph base and its <c>Box&lt;T&gt;</c> base's argument.</summary>
        /// <param name="t">The type.</param>
        /// <returns>The successor indexes (repeats allowed).</returns>
        private static IEnumerable<int> Successors(GraphType t)
        {
            foreach (var m in t.Members)
                if (m.Edge)
                    foreach (int target in m.Targets)
                        yield return target;
            if (t.Base >= 0)
                yield return t.Base;
            if (t.GenericBase >= 0)
                yield return t.GenericBase;
        }

        /// <summary>The oracle: does a value of graph type <paramref name="start"/> reach a stored NDArray without passing through <paramref name="avoid"/>?</summary>
        /// <param name="types">The graph.</param>
        /// <param name="start">The index of the type the search starts from.</param>
        /// <param name="avoid">The index of the type no path may pass through (or start at); -1 for none.</param>
        /// <returns>True when some type reachable from <paramref name="start"/> along edges and base classes, avoiding <paramref name="avoid"/>, declares a member that stores an NDArray directly.</returns>
        private static bool ReachesArraysAvoiding(GraphType[] types, int start, int avoid)
        {
            if (start == avoid)
                return false;
            var seen = new bool[types.Length];
            var queue = new Queue<int>();
            seen[start] = true;
            queue.Enqueue(start);
            while (queue.Count > 0)
            {
                var t = types[queue.Dequeue()];
                foreach (var m in t.Members)
                    if (m.Array)
                        return true;
                foreach (int j in Successors(t))
                    if (j != avoid && !seen[j])
                    {
                        seen[j] = true;
                        queue.Enqueue(j);
                    }
            }
            return false;
        }

        /// <summary>The members of type <paramref name="i"/> the rule calls holders: its direct array storage, and each edge with a target that reaches arrays without passing through type <paramref name="i"/>.</summary>
        /// <param name="types">The graph.</param>
        /// <param name="i">The index of the declaring type.</param>
        /// <returns>The holder names, sorted ordinally.</returns>
        private static SortedSet<string> ExpectedHolders(GraphType[] types, int i)
        {
            var set = new SortedSet<string>(StringComparer.Ordinal);
            foreach (var m in types[i].Members)
                if (m.Array || (m.Edge && m.Targets.Any(t => ReachesArraysAvoiding(types, t, i))))
                    set.Add(m.Name);
            return set;
        }

        /// <summary>
        ///     Whether member <paramref name="m"/> of type <paramref name="i"/> is one the rule EXCLUDES: an
        ///     edge whose targets reach arrays, but only through type <paramref name="i"/> itself. Counted so
        ///     a test proves it exercised the rule rather than only easy cases.
        /// </summary>
        /// <param name="types">The graph.</param>
        /// <param name="i">The index of the declaring type.</param>
        /// <param name="m">One of its members.</param>
        /// <returns>True for a back-reference.</returns>
        private static bool IsBackReference(GraphType[] types, int i, Member m)
            => m.Edge && !m.Array &&
               !m.Targets.Any(t => ReachesArraysAvoiding(types, t, i)) &&
               m.Targets.Any(t => ReachesArrays(types, t));

        /// <summary>Whether type <paramref name="i"/> is IDisposable, itself or through a base class — which decides between NDW017 (one per holder) and NDW016 (one listing the holders).</summary>
        /// <param name="types">The graph.</param>
        /// <param name="i">The index of the type.</param>
        /// <returns>True when the type or a graph base class declares IDisposable (a <c>Box&lt;T&gt;</c> base is not disposable).</returns>
        private static bool EffectivelyDisposable(GraphType[] types, int i)
        {
            for (int t = i; t >= 0; t = types[t].Base)
                if (types[t].DeclaresDisposable)
                    return true;
            return false;
        }

        /// <summary>NDW016's text up to its member list.</summary>
        private static readonly Regex Ndw016 = new Regex(@"^'(?<type>[^']+)' stores NDArrays in (?<list>.*?) but implements neither", RegexOptions.Compiled);

        /// <summary>NDW017's text: the disposable type and the one holder it never disposes.</summary>
        private static readonly Regex Ndw017 = new Regex(@"^'(?<type>[^']+)' implements .*? but never disposes '(?<member>[^']+)'", RegexOptions.Compiled);

        /// <summary>A quoted name inside a member list.</summary>
        private static readonly Regex Quoted = new Regex(@"'([^']+)'", RegexOptions.Compiled);

        /// <summary>The count NDW016 appends when it names only the first three holders.</summary>
        private static readonly Regex More = new Regex(@"and (\d+) more$", RegexOptions.Compiled);

        /// <summary>Compiles the graphs together (they share no type), runs the analyzers, and compares every type with the oracle.</summary>
        /// <param name="graphs">The graphs to compile as one file.</param>
        /// <param name="label">The file name and the prefix of every failure message.</param>
        /// <returns>How many holders the oracle expected across the batch (the non-vacuity count).</returns>
        /// <exception cref="AssertFailedException">The source does not compile, a diagnostic other than NDW016/NDW017 fires, or any type's holder set differs from the oracle's.</exception>
        private static async Task<int> CheckBatch(List<Graph> graphs, string label)
        {
            var source = new StringBuilder(Header);
            foreach (var g in graphs)
                source.Append(g.Source);
            var r = await AnalyzerTestHarness.RunAsync(source.ToString(), label + ".cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, $"{label}: generated graphs must compile:\n{string.Join("\n", r.CompileErrors.Take(10))}");

            var seen16 = new Dictionary<string, string>(StringComparer.Ordinal);
            var seen17 = new Dictionary<string, SortedSet<string>>(StringComparer.Ordinal);
            foreach (var d in r.Ndw)
            {
                var msg = d.GetMessage();
                if (d.Id == "NDW016")
                {
                    var m = Ndw016.Match(msg);
                    Assert.IsTrue(m.Success, "unexpected NDW016 text: " + msg);
                    seen16[m.Groups["type"].Value] = m.Groups["list"].Value;
                }
                else if (d.Id == "NDW017")
                {
                    var m = Ndw017.Match(msg);
                    Assert.IsTrue(m.Success, "unexpected NDW017 text: " + msg);
                    if (!seen17.TryGetValue(m.Groups["type"].Value, out var set))
                        seen17[m.Groups["type"].Value] = set = new SortedSet<string>(StringComparer.Ordinal);
                    set.Add(m.Groups["member"].Value);
                }
                else
                {
                    Assert.Fail($"{label}: only NDW016/NDW017 are expected over member declarations, got {d.Id}: {msg}");
                }
            }

            int holders = 0;
            foreach (var g in graphs)
                for (int i = 0; i < g.Types.Length; i++)
                {
                    var t = g.Types[i];
                    var expected = ExpectedHolders(g.Types, i);
                    holders += expected.Count;
                    string where = $"{label}, seed {g.Seed}, type {t.Name}: expected holders [{string.Join(", ", expected)}]\n{g.Source}";
                    if (EffectivelyDisposable(g.Types, i))
                    {
                        seen17.TryGetValue(t.Name, out var actual);
                        Assert.AreEqual(string.Join(",", expected), actual == null ? "" : string.Join(",", actual), "NDW017 members differ — " + where);
                        Assert.IsFalse(seen16.ContainsKey(t.Name), "a disposable type draws no NDW016 — " + where);
                    }
                    else
                    {
                        Assert.IsFalse(seen17.ContainsKey(t.Name), "a non-disposable type draws no NDW017 — " + where);
                        if (expected.Count == 0)
                        {
                            Assert.IsFalse(seen16.ContainsKey(t.Name), "a type without holders draws no NDW016 — " + where);
                            continue;
                        }
                        Assert.IsTrue(seen16.TryGetValue(t.Name, out var list), "NDW016 missing — " + where);
                        var named = Quoted.Matches(list).Cast<Match>().Select(q => q.Groups[1].Value).ToList();
                        if (expected.Count <= 3)
                        {
                            CollectionAssert.AreEquivalent(expected.ToList(), named, "NDW016 members differ — " + where);
                        }
                        else
                        {
                            Assert.AreEqual(3, named.Count, "NDW016 names three holders — " + where);
                            Assert.IsTrue(named.All(expected.Contains), "NDW016 names a non-holder — " + where);
                            var more = More.Match(list);
                            Assert.IsTrue(more.Success && int.Parse(more.Groups[1].Value) == expected.Count - 3, "NDW016 counts the rest — " + where);
                        }
                    }
                }
            return holders;
        }

        /// <summary>
        ///     600 random graphs of 2 to 8 classes: every class's holder set matches the oracle. The test also
        ///     counts the edges the rule excludes (a member that reaches arrays only through its own type)
        ///     and requires more than 50 of them, so it cannot pass by generating only the easy cases.
        /// </summary>
        [TestMethod]
        [Timeout(300000)]
        public async Task RandomGraphs_HolderSets_MatchTheIndependentOracle()
        {
            const int batches = 15, perBatch = 40;
            int holders = 0, backReferences = 0;
            for (int b = 0; b < batches; b++)
            {
                var graphs = new List<Graph>();
                for (int k = 0; k < perBatch; k++)
                {
                    int seed = b * perBatch + k;
                    var g = Generate(seed, "G" + seed, minTypes: 2, maxTypes: 8, maxMembers: 5);
                    graphs.Add(g);
                    // Count the members the rule excludes (a container edge that holds only through its owner),
                    // so the gate proves it exercised the rule, not only the easy cases.
                    for (int i = 0; i < g.Types.Length; i++)
                        foreach (var m in g.Types[i].Members)
                            if (IsBackReference(g.Types, i, m))
                                backReferences++;
                }
                holders += await CheckBatch(graphs, "graph_oracle_batch" + b);
            }
            Assert.IsTrue(holders > 500, $"the generator must produce plenty of holders (got {holders})");
            Assert.IsTrue(backReferences > 50, $"the generator must produce members that hold only through their owner (got {backReferences})");
        }

        /// <summary>
        ///     30 random graphs of 15 to 40 classes: larger strongly connected components with several anchors
        ///     each, where the dominator tree has real depth, and the holder sets still match the oracle.
        /// </summary>
        [TestMethod]
        [Timeout(300000)]
        public async Task LargerRandomGraphs_HolderSets_MatchTheIndependentOracle()
        {
            for (int b = 0; b < 6; b++)
            {
                var graphs = new List<Graph>();
                for (int k = 0; k < 5; k++)
                {
                    int seed = 100000 + b * 5 + k;
                    graphs.Add(Generate(seed, "L" + seed, minTypes: 15, maxTypes: 40, maxMembers: 6));
                }
                await CheckBatch(graphs, "graph_oracle_large" + b);
            }
        }

        /// <summary>
        ///     360 random graphs over the EXTENDED palette — two-target generics, nested and BCL containers,
        ///     jagged arrays, nullable tuples, pairs carrying an array, <c>Box&lt;T&gt;</c> base classes — whose
        ///     holder sets, read from the diagnostics, match the oracle. Non-vacuity: the run must contain
        ///     back-references, two-target edges with distinct targets, and generic base classes in numbers,
        ///     or a generator change that silently dropped a shape would let the test pass without them.
        /// </summary>
        [TestMethod]
        [Timeout(300000)]
        public async Task ExtendedShapes_RandomGraphs_HolderSets_MatchTheIndependentOracle()
        {
            const int batches = 12, perBatch = 30;
            int holders = 0, backReferences = 0, twoTargets = 0, genericBases = 0;
            for (int b = 0; b < batches; b++)
            {
                var graphs = new List<Graph>();
                for (int k = 0; k < perBatch; k++)
                {
                    int seed = 200000 + b * perBatch + k;
                    var g = Generate(seed, "X" + seed, minTypes: 2, maxTypes: 8, maxMembers: 5, extended: true);
                    graphs.Add(g);
                    for (int i = 0; i < g.Types.Length; i++)
                    {
                        if (g.Types[i].GenericBase >= 0)
                            genericBases++;
                        foreach (var m in g.Types[i].Members)
                        {
                            if (IsBackReference(g.Types, i, m))
                                backReferences++;
                            if (m.Targets.Length == 2 && m.Targets[0] != m.Targets[1])
                                twoTargets++;
                        }
                    }
                }
                holders += await CheckBatch(graphs, "graph_oracle_extended" + b);
            }
            Assert.IsTrue(holders > 300, $"the extended generator must produce plenty of holders (got {holders})");
            Assert.IsTrue(backReferences > 30, $"the extended generator must produce back-references (got {backReferences})");
            Assert.IsTrue(twoTargets > 50, $"the extended generator must produce two-target edges (got {twoTargets})");
            Assert.IsTrue(genericBases > 20, $"the extended generator must produce Box<T> base classes (got {genericBases})");
        }

        /// <summary>
        ///     Twelve graphs of 100 to 300 classes over the extended palette, asked of
        ///     <see cref="OwnershipModel"/> directly — the whole holder set of every class (no message
        ///     truncation in between) and every class's own "holds" answer — in a seeded random order with
        ///     member types asked first now and then. The run must reach a strongly connected component of at
        ///     least 30 classes, so the dominator tree is exercised at a depth the small graphs never reach.
        /// </summary>
        [TestMethod]
        [Timeout(300000)]
        public void DirectModel_LargeRandomGraphs_MatchTheIndependentOracle()
        {
            int largestComponent = 0, checkedTypes = 0, holders = 0;
            for (int k = 0; k < 12; k++)
            {
                int seed = 300000 + k;
                var g = Generate(seed, "D" + seed, minTypes: 100, maxTypes: 300, maxMembers: 4, extended: true);
                largestComponent = Math.Max(largestComponent, LargestComponent(g.Types));
                var (comp, model) = CompileForModel(g.Source, "graph_oracle_direct" + k);

                var rng = new Random(seed);
                foreach (int i in Enumerable.Range(0, g.Types.Length).OrderBy(_ => rng.Next()))
                {
                    var type = comp.GetTypeByMetadataName(g.Types[i].Name);
                    Assert.IsNotNull(type, $"seed {seed}: type {g.Types[i].Name} not found");
                    if (rng.Next(3) == 0)
                    {
                        // Warm a member type first now and then: the answers must not depend on it.
                        var memberType = type.GetMembers().OfType<IFieldSymbol>().Select(f => f.Type).FirstOrDefault();
                        model.HoldsNDArrays(memberType);
                    }

                    var expected = ExpectedHolders(g.Types, i);
                    var actual = new SortedSet<string>(model.HolderMembersOf(type).Select(h => h.Name), StringComparer.Ordinal);
                    string where = $"seed {seed}, type {g.Types[i].Name}";
                    Assert.AreEqual(string.Join(",", expected), string.Join(",", actual), "holder set differs — " + where);
                    Assert.AreEqual(ReachesArrays(g.Types, i), model.HoldsNDArrays(type), "holds answer differs — " + where);
                    holders += expected.Count;
                    checkedTypes++;
                }
            }

            Assert.IsTrue(checkedTypes > 1200, $"the direct run must cover many classes (checked {checkedTypes})");
            Assert.IsTrue(holders > 1000, $"the direct run must see plenty of holders (got {holders})");
            Assert.IsTrue(largestComponent >= 30, $"the direct run must reach a large strongly connected component (largest {largestComponent})");
        }

        /// <summary>Compiles one graph (after <see cref="Header"/>) for direct <see cref="OwnershipModel"/> questions, outside any analyzer.</summary>
        /// <param name="source">The graph's declarations.</param>
        /// <param name="name">The compilation's assembly name.</param>
        /// <returns>The compilation and a fresh model over it.</returns>
        /// <exception cref="AssertFailedException">The graph does not compile, or NumSharp's types cannot be resolved.</exception>
        private static (CSharpCompilation comp, OwnershipModel model) CompileForModel(string source, string name)
        {
            var tree = CSharpSyntaxTree.ParseText(SourceText.From(Header + source), path: name + ".cs");
            var comp = CSharpCompilation.Create(name, new[] { tree }, AnalyzerTestHarness.References,
                new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
            var errors = comp.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error).Take(10).ToList();
            Assert.AreEqual(0, errors.Count, $"{name}: the graph must compile:\n{string.Join("\n", errors)}");
            var known = KnownTypes.Resolve(comp);
            Assert.IsNotNull(known, "NumSharp must resolve");
            return (comp, new OwnershipModel(known, comp));
        }

        /// <summary>
        ///     The size of the largest strongly connected component of the graph's own edges (members'
        ///     targets and base classes): Kosaraju's two passes, iterative. Used only to prove a test reached
        ///     big components; the oracle itself never needs components.
        /// </summary>
        /// <param name="types">The graph.</param>
        /// <returns>The number of classes in the largest component.</returns>
        private static int LargestComponent(GraphType[] types)
        {
            int n = types.Length;
            var forward = new List<int>[n];
            var backward = new List<int>[n];
            for (int i = 0; i < n; i++)
            {
                forward[i] = new List<int>();
                backward[i] = new List<int>();
            }
            for (int i = 0; i < n; i++)
                foreach (int j in Successors(types[i]))
                {
                    forward[i].Add(j);
                    backward[j].Add(i);
                }

            // Pass 1: finishing order of a depth-first walk over the forward edges.
            var order = new List<int>(n);
            var visited = new bool[n];
            for (int s = 0; s < n; s++)
            {
                if (visited[s])
                    continue;
                var stack = new Stack<(int node, int next)>();
                stack.Push((s, 0));
                visited[s] = true;
                while (stack.Count > 0)
                {
                    var (node, next) = stack.Pop();
                    if (next < forward[node].Count)
                    {
                        stack.Push((node, next + 1));
                        int child = forward[node][next];
                        if (!visited[child])
                        {
                            visited[child] = true;
                            stack.Push((child, 0));
                        }
                    }
                    else
                    {
                        order.Add(node);
                    }
                }
            }

            // Pass 2: walk the reversed edges in reverse finishing order; each walk is one component.
            var assigned = new bool[n];
            int largest = 0;
            for (int o = n - 1; o >= 0; o--)
            {
                int s = order[o];
                if (assigned[s])
                    continue;
                int size = 0;
                var stack = new Stack<int>();
                stack.Push(s);
                assigned[s] = true;
                while (stack.Count > 0)
                {
                    int node = stack.Pop();
                    size++;
                    foreach (int p in backward[node])
                        if (!assigned[p])
                        {
                            assigned[p] = true;
                            stack.Push(p);
                        }
                }
                largest = Math.Max(largest, size);
            }
            return largest;
        }

        /// <summary>The oracle's plain reachability (no type avoided), for the non-vacuity count.</summary>
        /// <param name="types">The graph.</param>
        /// <param name="start">The index of the type the search starts from.</param>
        /// <returns>True when a value of the type reaches a stored NDArray by any route.</returns>
        private static bool ReachesArrays(GraphType[] types, int start) => ReachesArraysAvoiding(types, start, -1);
    }
}
