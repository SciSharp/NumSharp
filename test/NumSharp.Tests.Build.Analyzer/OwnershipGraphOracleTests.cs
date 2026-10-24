using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

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
    /// </summary>
    [TestClass]
    public class OwnershipGraphOracleTests
    {
        /// <summary>
        ///     How a generated member relates to its target type. The first six are EDGES (a value of the
        ///     target is stored, so the target's arrays become the member's), <see cref="Shape.NDArray"/> is an
        ///     array member, and the last six store no value of the target (all but <see cref="Shape.Scalar"/>
        ///     still name it) — each a rule of the model the oracle must agree with.
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
            Computed
        }

        /// <summary>One generated member and the facts the oracle reads about it.</summary>
        private sealed class Member
        {
            /// <summary>The member's name as declared (and as diagnostics quote it).</summary>
            public string Name;

            /// <summary>Its C# declaration.</summary>
            public string Declaration;

            /// <summary>The graph type the member's type contains, when it is an edge; -1 otherwise.</summary>
            public int Target = -1;

            /// <summary>True when the member stores values of <see cref="Target"/> (an ownership edge).</summary>
            public bool Edge;

            /// <summary>True for an NDArray member, which holds by itself.</summary>
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

        /// <summary>Every member shape, drawn uniformly.</summary>
        private static readonly Shape[] Shapes = (Shape[])Enum.GetValues(typeof(Shape));

        /// <summary>Generates one random graph of classes: members of every <see cref="Shape"/> pointing at random classes (cycles and self-references included), acyclic base classes, random IDisposable.</summary>
        /// <param name="seed">The random seed; the same seed gives the same graph.</param>
        /// <param name="prefix">Prefixed to every class name, unique per graph.</param>
        /// <param name="minTypes">The fewest classes.</param>
        /// <param name="maxTypes">The most classes.</param>
        /// <param name="maxMembers">The most members per class (each has between 0 and this many).</param>
        /// <returns>The graph with its source.</returns>
        private static Graph Generate(int seed, string prefix, int minTypes, int maxTypes, int maxMembers)
        {
            var rng = new Random(seed);
            int n = minTypes + rng.Next(maxTypes - minTypes + 1);
            var types = new GraphType[n];
            for (int i = 0; i < n; i++)
            {
                var t = types[i] = new GraphType { Name = prefix + "T" + i, DeclaresDisposable = rng.Next(2) == 0 };
                if (i > 0 && rng.Next(4) == 0)
                    t.Base = rng.Next(i); // a lower index: inheritance stays acyclic
                int members = rng.Next(maxMembers + 1);
                for (int k = 0; k < members; k++)
                {
                    var shape = Shapes[rng.Next(Shapes.Length)];
                    int j = rng.Next(n);
                    string target = prefix + "T" + j;
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
                    }
                    if (m.Edge)
                        m.Target = j;
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

        /// <summary>The oracle: does a value of graph type <paramref name="start"/> reach a stored NDArray without passing through <paramref name="avoid"/>?</summary>
        /// <param name="types">The graph.</param>
        /// <param name="start">The index of the type the search starts from.</param>
        /// <param name="avoid">The index of the type no path may pass through (or start at); -1 for none.</param>
        /// <returns>True when some type reachable from <paramref name="start"/> along edges and base classes, avoiding <paramref name="avoid"/>, declares an NDArray member.</returns>
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
                var next = t.Members.Where(m => m.Edge).Select(m => m.Target).ToList();
                if (t.Base >= 0)
                    next.Add(t.Base);
                foreach (int j in next)
                    if (j != avoid && !seen[j])
                    {
                        seen[j] = true;
                        queue.Enqueue(j);
                    }
            }
            return false;
        }

        /// <summary>The members of type <paramref name="i"/> the rule calls holders: its NDArrays, and each edge whose target reaches arrays without passing through type <paramref name="i"/>.</summary>
        /// <param name="types">The graph.</param>
        /// <param name="i">The index of the declaring type.</param>
        /// <returns>The holder names, sorted ordinally.</returns>
        private static SortedSet<string> ExpectedHolders(GraphType[] types, int i)
        {
            var set = new SortedSet<string>(StringComparer.Ordinal);
            foreach (var m in types[i].Members)
                if (m.Array || (m.Edge && ReachesArraysAvoiding(types, m.Target, i)))
                    set.Add(m.Name);
            return set;
        }

        /// <summary>Whether type <paramref name="i"/> is IDisposable, itself or through a base class — which decides between NDW017 (one per holder) and NDW016 (one listing the holders).</summary>
        /// <param name="types">The graph.</param>
        /// <param name="i">The index of the type.</param>
        /// <returns>True when the type or a base class declares IDisposable.</returns>
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
            var source = new StringBuilder("using System;\nusing System.Collections.Generic;\nusing NumSharp;\npublic class Box<T> { public T Value; }\n");
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
                            if (m.Edge && !ReachesArraysAvoiding(g.Types, m.Target, i) && ReachesArrays(g.Types, m.Target))
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

        /// <summary>The oracle's plain reachability (no type avoided), for the non-vacuity count.</summary>
        /// <param name="types">The graph.</param>
        /// <param name="start">The index of the type the search starts from.</param>
        /// <returns>True when a value of the type reaches a stored NDArray by any route.</returns>
        private static bool ReachesArrays(GraphType[] types, int start) => ReachesArraysAvoiding(types, start, -1);
    }
}
