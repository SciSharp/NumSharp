using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz.RandomApi
{
    /// <summary>
    ///     Gates G2–G6 of the random-API oracle (docs/plans/random-oracle-coverage.md §1, §6.6): the committed corpus
    ///     (<c>random_api*.jsonl</c>) measured against the reflected inventory (<see cref="RandomApiSurface"/>, itself pinned
    ///     by G1), so a new overload, parameter, engine or seed that no case exercises turns CI red instead of going
    ///     unnoticed.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The gates read the corpus only (no replay): they state what the corpus COVERS; the replay tests
    ///     (<c>FuzzCorpusTests.RandomApi</c>) state that NumSharp MATCHES it. Both are needed — a gate alone passes a corpus
    ///     NumSharp fails, a replay alone passes a corpus that forgot an overload.
    ///     </para>
    ///     <para>
    ///     Every exemption carries its reason and is self-retiring: an exempted member that gains a case, an exempted
    ///     rule that the corpus now satisfies, or an allow-list entry that is no longer needed fails the gate, so the
    ///     tables cannot silently outlive the gap they excuse.
    ///     </para>
    ///     <para>
    ///     The corpus is indexed once per process (<see cref="CorpusIndex"/>): each case reduced to its signature,
    ///     receiver, outcome, state presence and per-argument value keys — never materialized as full cases (the
    ///     coverage-gate rule of the Oracle project: coverage questions do not parse replay payloads they do not need).
    ///     </para>
    /// </remarks>
    [TestClass]
    public class RandomApiCoverageTests
    {
        /// <summary>The four random-API tiers every gate reads (all of them: a member may live in any tier).</summary>
        internal static readonly string[] Tiers =
            { "random_api.jsonl", "random_api_host.jsonl", "random_api_mvn.jsonl", "random_api_lp64.jsonl" };

        /// <summary>The five engines (plan §5).</summary>
        internal static readonly string[] Engines = { "MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64" };

        /// <summary>The ten fixed seeds of the committed corpus (plan §5), as the corpus spells them (decimal text).</summary>
        internal static readonly string[] FixedSeeds =
            { "0", "1", "7", "42", "1234", "65535", "2147483647", "2147483648", "987654321", "4294967295" };

        /// <summary>
        ///     G2's exemptions: members with no NumPy counterpart (plan §7), by exact signature. Mirrors the generator's
        ///     <c>exempt(...)</c> calls; the gate fails when an entry gains a case or names a removed member.
        /// </summary>
        internal static readonly Dictionary<string, string> OverloadExemptions = new(StringComparer.Ordinal)
        {
            ["NativeRandomState.ctor(byte[])"] = "obsolete NumSharp constructor that always throws; no NumPy counterpart",
            ["NumPyRandom.Seed.get"] = "NumSharp bookkeeping of the last legacy integer seed; no NumPy attribute",
            ["NumPyRandom.Seed.set"] = "NumSharp bookkeeping of the last legacy integer seed; no NumPy attribute",
            ["BitGenerator.lock.get"] = "a threading primitive; nothing NumPy-observable (unit-tested: lock identity, concurrency)",
        };

        /// <summary>
        ///     Members whose result does not depend on the receiver they are invoked on — the factories on <c>np.random</c>
        ///     (<c>RandomState(...)</c>, <c>default_rng(...)</c>) build a NEW generator from their arguments. G4/G5 do not
        ///     demand them on every receiver engine and seed; G6 still checks that the receiver is left untouched, and G4
        ///     demands every engine among their ENGINE-BEARING arguments. Keyed by <c>Type.name</c>.
        /// </summary>
        internal static readonly Dictionary<string, string> ReceiverIndependent = new(StringComparer.Ordinal)
        {
            ["NumPyRandom.RandomState"] = "np.random.RandomState(...) builds a new RandomState from its seed argument",
            ["NumPyRandom.default_rng"] = "np.random.default_rng(...) builds a Generator from its seed argument",
            // numpy.random's classes through the module: each factory builds what the constructor builds, from its own
            // arguments (the cases record the receiver's state to prove it is never drawn from).
            ["NumPyRandom.MT19937"] = "np.random.MT19937(...) builds a new engine from its seed argument",
            ["NumPyRandom.PCG64"] = "np.random.PCG64(...) builds a new engine from its seed argument",
            ["NumPyRandom.PCG64DXSM"] = "np.random.PCG64DXSM(...) builds a new engine from its seed argument",
            ["NumPyRandom.Philox"] = "np.random.Philox(...) builds a new engine from its seed/counter/key arguments",
            ["NumPyRandom.SFC64"] = "np.random.SFC64(...) builds a new engine from its seed argument",
            ["NumPyRandom.SeedSequence"] = "np.random.SeedSequence(...) builds a new sequence from its entropy argument",
            ["NumPyRandom.Generator"] = "np.random.Generator(...) wraps its bit_generator argument",
        };

        /// <summary>
        ///     G3 parameter-rule exemptions, keyed <c>sig|parameter|rule</c> (rules: <c>omitted</c>, <c>non-default</c>,
        ///     <c>two-values</c>, <c>null</c>, <c>non-null</c>, <c>params-lengths</c>). Empty today: every parameter of every
        ///     generated overload meets every rule. A future entry needs a reason, and fails once the rule is met.
        /// </summary>
        internal static readonly Dictionary<string, string> ParameterExemptions = new(StringComparer.Ordinal);

        /// <summary>
        ///     Methods and constructors of the NumPy-mirroring types that have no NumPy counterpart at all, keyed by
        ///     <c>Type.name</c> — the parameter-name gate's allow-list (fails when an entry gains a counterpart).
        /// </summary>
        internal static readonly Dictionary<string, string> NumSharpOnlyMembers = new(StringComparer.Ordinal)
        {
            ["NumPyRandom.bernoulli"] = "NumSharp's own sampler (NumPy has no bernoulli); oracle-checked against its documented composition",
            ["NumPyRandom.ToString"] = "Python's str(), no parameters",
            ["Generator.ToString"] = "Python's str(), no parameters",
            ["SeedSequence.ToString"] = "Python's repr(), no parameters",
        };

        /// <summary>
        ///     Parameter-name exemptions, by exact signature: overloads that exist only in NumSharp although their NAME has a
        ///     NumPy counterpart, so their parameter names cannot follow NumPy's.
        /// </summary>
        internal static readonly Dictionary<string, string> ParameterNameExemptions = new(StringComparer.Ordinal)
        {
            ["NumPyRandom.RandomState(NativeRandomState)"] =
                "NumSharp's restoring constructor (a RandomState from a legacy state tuple); NumPy's RandomState(seed) refuses a tuple",
        };

        /// <summary>The C# types whose methods and constructors mirror a NumPy object (the parameter-name gate's scope).</summary>
        internal static readonly HashSet<string> NumPyMirroringTypes = new(StringComparer.Ordinal)
        {
            "NumPyRandom", "Generator", "BitGenerator", "MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64",
            "SeedSequence", "SeedlessSeedSequence", "ISeedSequence", "ISpawnableSeedSequence",
        };

        /// <summary>
        ///     An enumerated parameter (plan §1.2): every value NumPy accepts must appear in a case NumPy answers, and — unless
        ///     <paramref name="NoRejection"/> says why not — at least one value outside the set must appear in a case NumPy
        ///     rejects (with NumPy's error, which the replay compares).
        /// </summary>
        /// <param name="Member">The member, <c>Type.name</c> (every overload having the parameter is checked on its own).</param>
        /// <param name="Parameter">The parameter name.</param>
        /// <param name="Accepted">The values NumPy accepts (a DType by its NumPy name, a string by its text).</param>
        /// <param name="NoRejection">Null when NumPy rejects some values; otherwise why no value is rejected.</param>
        internal sealed record Enumeration(string Member, string Parameter, string[] Accepted, string NoRejection = null);

        /// <summary>The enumerated parameters of the random world and the values NumPy 2.4.2 accepts for each.</summary>
        internal static readonly Enumeration[] Enumerations =
        {
            new("Generator.integers", "dtype", new[] { "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool" }),
            new("NumPyRandom.randint", "dtype", new[] { "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool" }),
            new("Generator.random", "dtype", new[] { "float64", "float32" }),
            new("Generator.standard_normal", "dtype", new[] { "float64", "float32" }),
            new("Generator.standard_exponential", "dtype", new[] { "float64", "float32" }),
            new("Generator.standard_gamma", "dtype", new[] { "float64", "float32" }),
            new("SeedSequence.generate_state", "dtype", new[] { "uint32", "uint64" }),
            new("ISeedSequence.generate_state", "dtype", new[] { "uint32", "uint64" }),
            new("Generator.multivariate_normal", "method", new[] { "svd", "eigh", "cholesky" }),
            new("Generator.multivariate_normal", "check_valid", new[] { "warn", "raise", "ignore" }),
            new("NumPyRandom.multivariate_normal", "check_valid", new[] { "warn", "raise", "ignore" }),
            new("Generator.multivariate_hypergeometric", "method", new[] { "marginals", "count" }),
            new("Generator.standard_exponential", "method", new[] { "zig", "inv" },
                NoRejection: "NumPy takes every method other than 'zig' as the inverse transform; nothing is rejected"),
        };

        /// <summary>The receiver kinds whose state a case records after the call (G6).</summary>
        private static readonly HashSet<string> StatefulReceivers = new(StringComparer.Ordinal)
        {
            "RandomState", "Generator", "BitGenerator", "legacy_tuple", "rs_dict", "bgstate", "rs_bitgen", "seedseq", "seedless",
        };

        /// <summary>The stream receiver kinds (G5: their engine x seed sweep).</summary>
        private static readonly HashSet<string> StreamReceivers = new(StringComparer.Ordinal) { "RandomState", "Generator", "BitGenerator" };

        /// <summary>The corpus index, built once per process.</summary>
        private static readonly Lazy<CorpusIndex> Corpus = new(CorpusIndex.Load);

        /// <summary>The reflected inventory, once per process.</summary>
        private static readonly Lazy<IReadOnlyList<RandomApiSurface.Member>> Surface = new(RandomApiSurface.Reflect);

        // ---------------------------------------------------------------------------------------------------------------
        // G2 — overloads
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     G2: every inventory member is invoked by at least one case (by its exact signature) or is exempt with a
        ///     reason; an exempt member has NO case; every case names an inventory member; every exemption names one.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void G2_EveryOverloadIsReplayed_OrExempt()
        {
            var problems = new List<string>();
            var bySig = Corpus.Value.BySig;
            var inventory = Surface.Value.Select(m => m.Sig).ToHashSet(StringComparer.Ordinal);
            foreach (var m in Surface.Value)
            {
                int n = bySig.TryGetValue(m.Sig, out var cs) ? cs.Count : 0;
                if (OverloadExemptions.TryGetValue(m.Sig, out var reason))
                {
                    if (n > 0)
                        problems.Add($"{m.Sig}: exempt ({reason}) but {n} case(s) exercise it — remove the exemption");
                }
                else if (n == 0)
                    problems.Add($"{m.Sig}: no case (add a family to gen_random_oracle.py, or an exemption with a reason)");
            }
            foreach (var sig in bySig.Keys)
                if (!inventory.Contains(sig))
                    problems.Add($"{sig}: the corpus names a member the inventory does not have (regenerate the corpus)");
            foreach (var sig in OverloadExemptions.Keys)
                if (!inventory.Contains(sig))
                    problems.Add($"{sig}: exemption for a member that no longer exists");
            Report("G2 overload coverage", problems);
        }

        // ---------------------------------------------------------------------------------------------------------------
        // G3 — parameters
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     G3: for every overload, every optional parameter is exercised omitted AND with a non-default value; every
        ///     required parameter takes at least two distinct values; every nullable parameter takes null (explicitly, or
        ///     omitted where the default IS null) and a non-null value; every <c>params</c> array takes zero and several
        ///     elements.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void G3_EveryParameterIsExercised()
        {
            var problems = new List<string>();
            var satisfiedExemptions = new List<string>();
            foreach (var m in Surface.Value)
            {
                if (OverloadExemptions.ContainsKey(m.Sig) || m.Parameters.Count == 0)
                    continue;
                var cases = Corpus.Value.BySig[m.Sig];
                var infos = ParameterInfos(m);
                for (int i = 0; i < m.Parameters.Count; i++)
                {
                    var p = m.Parameters[i];
                    var args = new List<IndexedArg>(cases.Count);
                    foreach (var c in cases)
                    {
                        if (c.Args.Length != m.Parameters.Count || c.Args[i].Name != p.Name)
                        {
                            problems.Add($"{c.Id}: argument {i} is not '{p.Name}' — the case does not match the overload");
                            goto nextMember;
                        }
                        args.Add(c.Args[i]);
                    }

                    void Rule(string rule, bool met, string detail)
                    {
                        string key = $"{m.Sig}|{p.Name}|{rule}";
                        if (ParameterExemptions.ContainsKey(key))
                        {
                            if (met)
                                satisfiedExemptions.Add(key);
                        }
                        else if (!met)
                            problems.Add($"{m.Sig} '{p.Name}' ({p.Type}): {detail}");
                    }

                    if (p.Optional)
                    {
                        Rule("omitted", args.Any(a => a.Kind == ArgKind.Omitted), "never omitted (the C# default path is untested)");
                        Rule("non-default", args.Any(a => IsNonDefault(a, p)), $"never passed a value other than its default {p.Default}");
                    }
                    else
                    {
                        int distinct = args.Where(a => a.Kind != ArgKind.Omitted).Select(a => a.Key).Distinct(StringComparer.Ordinal).Count();
                        Rule("two-values", distinct >= 2, $"takes {distinct} distinct value(s); a required parameter needs at least two");
                    }
                    var t = infos[i].ParameterType;
                    if (!t.IsValueType || Nullable.GetUnderlyingType(t) != null)
                    {
                        Rule("null", args.Any(a => a.Kind == ArgKind.Null || (a.Kind == ArgKind.Omitted && p.Default == "null")),
                            "never null (NumPy's None)");
                        Rule("non-null", args.Any(a => a.Kind == ArgKind.Value), "never non-null");
                    }
                    if (p.IsParams)
                    {
                        var lengths = args.Where(a => a.Kind == ArgKind.Value && a.Length.HasValue).Select(a => a.Length!.Value).ToHashSet();
                        Rule("params-lengths", lengths.Contains(0) && lengths.Any(l => l >= 2),
                            $"params array lengths {{{string.Join(",", lengths.OrderBy(l => l))}}}: needs 0 and 2+");
                    }
                }
                nextMember: ;
            }
            foreach (var key in satisfiedExemptions)
                problems.Add($"{key}: exempt, but the corpus now meets the rule — remove the exemption");
            foreach (var key in ParameterExemptions.Keys)
                if (!Surface.Value.Any(m => key.StartsWith(m.Sig + "|", StringComparison.Ordinal)))
                    problems.Add($"{key}: exemption for a member that no longer exists");
            Report("G3 parameter coverage", problems);
        }

        /// <summary>
        ///     G3, enumerated parameters: per overload, every value NumPy accepts appears in a case NumPy answers, and a
        ///     value outside the set appears in a case NumPy rejects (see <see cref="Enumerations"/>).
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void G3_EnumeratedParametersTakeEveryAcceptedValue()
        {
            var problems = new List<string>();
            foreach (var e in Enumerations)
            {
                var overloads = Surface.Value.Where(m => $"{m.DeclaringType}.{m.Name}" == e.Member
                                                         && m.Parameters.Any(p => p.Name == e.Parameter)).ToList();
                if (overloads.Count == 0)
                {
                    problems.Add($"{e.Member} '{e.Parameter}': no such parameter in the inventory (stale table entry)");
                    continue;
                }
                foreach (var m in overloads)
                {
                    int i = m.Parameters.Select((p, k) => (p, k)).First(x => x.p.Name == e.Parameter).k;
                    var cases = Corpus.Value.BySig.TryGetValue(m.Sig, out var cs) ? cs : new List<IndexedCase>();
                    foreach (var v in e.Accepted)
                        if (!cases.Any(c => !c.Throws && c.Args[i].Kind == ArgKind.Value && c.Args[i].Scalar == v))
                            problems.Add($"{m.Sig} '{e.Parameter}': NumPy accepts '{v}', but no case NumPy answers passes it");
                    if (e.NoRejection == null
                        && !cases.Any(c => c.Throws && c.Args[i].Kind == ArgKind.Value && c.Args[i].Scalar != null
                                           && !e.Accepted.Contains(c.Args[i].Scalar)))
                        problems.Add($"{m.Sig} '{e.Parameter}': no case passes a value NumPy rejects");
                }
            }
            Report("G3 enumerated values", problems);
        }

        /// <summary>
        ///     G3, parameter names: every method/constructor parameter of the NumPy-mirroring types carries one of NumPy's
        ///     parameter names for that member, in NumPy's order (so a ported call with keyword arguments binds), checked
        ///     against <c>test/oracle/random_numpy_signatures.json</c> (written by the generator from NumPy 2.4.2). A NumPy
        ///     <c>*args</c> member accepts any names (its parameters cannot be passed by name). Members without a NumPy
        ///     counterpart must be allow-listed (<see cref="NumSharpOnlyMembers"/>).
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void G3_ParameterNamesFollowNumPy()
        {
            string path = Path.Combine(RandomApiSurface.RepoRoot(), "test", "oracle", "random_numpy_signatures.json");
            Assert.IsTrue(File.Exists(path), $"{path} is missing; regenerate it with python test/oracle/gen_random_oracle.py");
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            var table = new Dictionary<string, (string Numpy, string[] Names)>(StringComparer.Ordinal);
            foreach (var entry in doc.RootElement.GetProperty("members").EnumerateObject())
                table[entry.Name] = (entry.Value.GetProperty("numpy").GetString(),
                    entry.Value.GetProperty("params").EnumerateArray().Select(x => x.GetString()).ToArray());

            var problems = new List<string>();
            var used = new HashSet<string>(StringComparer.Ordinal);
            foreach (var m in Surface.Value)
            {
                if ((m.Kind != "method" && m.Kind != "ctor") || !NumPyMirroringTypes.Contains(m.DeclaringType))
                    continue;
                string key = $"{m.DeclaringType}.{m.Name}";
                if (!table.TryGetValue(key, out var np))
                {
                    if (!NumSharpOnlyMembers.ContainsKey(key))
                        problems.Add($"{m.Sig}: no NumPy counterpart in the signature table and not allow-listed as NumSharp-only");
                    continue;
                }
                used.Add(key);
                if (np.Names.Contains("*") || ParameterNameExemptions.ContainsKey(m.Sig))
                    continue;
                var names = m.Parameters.Select(p => p.Name).ToArray();
                var foreign = names.Where(n => !np.Names.Contains(n)).ToArray();
                if (foreign.Length > 0)
                {
                    problems.Add($"{m.Sig}: parameter(s) {string.Join(", ", foreign)} are not NumPy's ({np.Numpy}({string.Join(", ", np.Names)}))");
                    continue;
                }
                var order = names.Select(n => Array.IndexOf(np.Names, n)).ToArray();
                for (int k = 1; k < order.Length; k++)
                    if (order[k] < order[k - 1])
                    {
                        problems.Add($"{m.Sig}: parameters out of NumPy's order ({np.Numpy}({string.Join(", ", np.Names)}))");
                        break;
                    }
            }
            foreach (var key in table.Keys)
                if (!used.Contains(key))
                    problems.Add($"{key}: signature-table entry matches no member (stale; regenerate the table)");
            foreach (var key in NumSharpOnlyMembers.Keys)
                if (table.ContainsKey(key))
                    problems.Add($"{key}: allow-listed as NumSharp-only, but NumPy has it — remove the allow-list entry");
            foreach (var sig in ParameterNameExemptions.Keys)
            {
                var m = Surface.Value.FirstOrDefault(x => x.Sig == sig);
                if (m == null)
                    problems.Add($"{sig}: parameter-name exemption for a member that no longer exists");
                else if (table.TryGetValue($"{m.DeclaringType}.{m.Name}", out var np)
                         && m.Parameters.All(p => np.Names.Contains(p.Name)))
                    problems.Add($"{sig}: parameter-name exemption no longer needed (its names are NumPy's)");
            }
            Report("G3 parameter names", problems);
        }

        // ---------------------------------------------------------------------------------------------------------------
        // G4 — engines
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     G4: every <c>Generator</c> member runs on all five engines; every legacy <c>RandomState</c> member on the
        ///     legacy-seeded MT19937 and on each engine; every <c>BitGenerator</c> member, <c>BitGeneratorState</c> member and
        ///     <c>NumPyRandom.State</c> member over every engine's state; and every member taking an engine-bearing argument
        ///     (a bit generator, a Generator, a RandomState) takes one of every engine.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void G4_EveryEngineRunsEveryMember()
        {
            var problems = new List<string>();
            var legacyAndEngines = new[] { "legacy" }.Concat(Engines).ToArray();
            foreach (var m in Surface.Value)
            {
                if (OverloadExemptions.ContainsKey(m.Sig))
                    continue;
                var cases = Corpus.Value.BySig[m.Sig];
                string owner = $"{m.DeclaringType}.{m.Name}";
                bool independent = ReceiverIndependent.ContainsKey(owner);

                void Need(string what, IEnumerable<string> have, IEnumerable<string> want)
                {
                    var missing = want.Except(have.Where(x => x != null), StringComparer.Ordinal).ToArray();
                    if (missing.Length > 0)
                        problems.Add($"{m.Sig}: {what} lacks {string.Join(", ", missing)}");
                }

                if (!independent)
                {
                    switch (m.DeclaringType)
                    {
                        case "Generator" when m.Kind != "ctor":
                            Need("Generator receivers", cases.Where(c => c.RecvKind == "Generator").Select(c => c.RecvEngine), Engines);
                            break;
                        case "NumPyRandom" when !m.IsStatic:
                            Need("RandomState receivers",
                                cases.Where(c => c.RecvKind == "RandomState").Select(c => c.RecvEngine ?? "legacy"), legacyAndEngines);
                            break;
                        case "BitGenerator":
                            Need("bit generator receivers",
                                cases.Where(c => c.RecvKind == "BitGenerator").Select(c => c.RecvEngine), Engines);
                            break;
                        case "BitGeneratorState":
                            Need("engine-state receivers", cases.Where(c => c.RecvKind == "bgstate").Select(c => c.RecvEngine), Engines);
                            break;
                        case "NumPyRandom.State" when m.Kind != "ctor":
                            Need("RandomState dict receivers",
                                cases.Where(c => c.RecvKind == "rs_dict").Select(c => c.RecvEngine ?? "legacy"), legacyAndEngines);
                            break;
                    }
                }
                // Engine-bearing arguments (bit generators, Generators, RandomStates): every engine among them.
                var argEngines = cases.SelectMany(c => c.Args).Where(a => a.Engine != null).Select(a => a.Engine).ToList();
                if (argEngines.Count > 0)
                    Need("engine-bearing arguments", argEngines, Engines);
            }
            Report("G4 engine coverage", problems);
        }

        // ---------------------------------------------------------------------------------------------------------------
        // G5 — seeds
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     G5: every stream member that NumPy answers on a receiver engine is answered there under all ten fixed seeds
        ///     (per receiver kind and engine: a Generator member on each of the five engines, a legacy member on the
        ///     legacy MT19937 and each engine, a bit-generator member on each engine).
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void G5_EveryStreamMemberRunsUnderTheTenSeeds()
        {
            var problems = new List<string>();
            foreach (var m in Surface.Value)
            {
                if (OverloadExemptions.ContainsKey(m.Sig) || ReceiverIndependent.ContainsKey($"{m.DeclaringType}.{m.Name}"))
                    continue;
                var groups = Corpus.Value.BySig[m.Sig]
                    .Where(c => !c.Throws && StreamReceivers.Contains(c.RecvKind))
                    .GroupBy(c => (c.RecvKind, Engine: c.RecvEngine ?? "legacy"));
                foreach (var g in groups)
                {
                    var missing = FixedSeeds.Except(g.Select(c => c.RecvSeed), StringComparer.Ordinal).ToArray();
                    if (missing.Length > 0)
                        problems.Add($"{m.Sig} on {g.Key.RecvKind}({g.Key.Engine}): no answered case under seed(s) {string.Join(", ", missing)}");
                }
            }
            Report("G5 seed coverage", problems);
        }

        // ---------------------------------------------------------------------------------------------------------------
        // G6 — state after the call
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     G6: every answered case on a receiver that has state (a RandomState, a Generator, a bit generator, a state
        ///     object, a seed sequence) records that state after the call, so over- or under-drawing is caught and not only
        ///     wrong values.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void G6_EveryStatefulCaseRecordsTheStateAfterTheCall()
        {
            var problems = Corpus.Value.All
                .Where(c => !c.Throws && StatefulReceivers.Contains(c.RecvKind) && !c.HasState)
                .Select(c => $"{c.Id}: no expected.state")
                .ToList();
            Report("G6 state observation", problems);
        }

        // ---------------------------------------------------------------------------------------------------------------
        // Helpers
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>Fails with every problem listed (capped for the message; all of them go to the test output).</summary>
        /// <param name="gate">The gate's name.</param>
        /// <param name="problems">The problems (empty: pass).</param>
        private static void Report(string gate, List<string> problems)
        {
            if (problems.Count == 0)
                return;
            if (problems.Count > 60)
                Console.WriteLine($"[{gate}] all {problems.Count} problems:\n  " + string.Join("\n  ", problems));
            Assert.Fail($"{gate}: {problems.Count} problem(s):\n  " + string.Join("\n  ", problems.Take(60)));
        }

        /// <summary>The reflected parameters of a member (a setter's single <c>value</c>; none for getters and fields).</summary>
        /// <param name="m">The member.</param>
        /// <returns>The parameters, in declaration order.</returns>
        private static ParameterInfo[] ParameterInfos(RandomApiSurface.Member m) => m.Info switch
        {
            MethodBase mb => mb.GetParameters(),
            PropertyInfo p when m.Kind == "set" => p.SetMethod!.GetParameters(),
            _ => Array.Empty<ParameterInfo>(),
        };

        /// <summary>
        ///     Whether a passed argument differs from the parameter's declared default (the inventory's canonical default
        ///     text): a non-null value for a null default, a size for a <c>default(Shape)</c>, another number / bool /
        ///     string, or any structured value (an array, an operand, an object) for a scalar default.
        /// </summary>
        /// <param name="a">The argument.</param>
        /// <param name="p">The optional parameter.</param>
        /// <returns>True for a value other than the default.</returns>
        private static bool IsNonDefault(IndexedArg a, RandomApiSurface.Parameter p)
        {
            if (a.Kind != ArgKind.Value)
                return false;
            string d = p.Default;
            switch (d)
            {
                case "null":
                    return true;
                case "default":
                    return !a.IsNone;
                case "true":
                case "false":
                    return a.Scalar != d;
            }
            if (d.StartsWith("\"", StringComparison.Ordinal))
                return a.Scalar != d.Trim('"');
            if (a.Bits != null)
                return BitConverter.Int64BitsToDouble(unchecked((long)ulong.Parse(a.Bits, NumberStyles.HexNumber)))
                       != double.Parse(d, CultureInfo.InvariantCulture);
            if (a.Scalar != null && BigInteger.TryParse(a.Scalar, NumberStyles.Integer, CultureInfo.InvariantCulture, out var iv))
                return d.IndexOfAny(new[] { '.', 'E', 'e' }) >= 0
                    ? (double)iv != double.Parse(d, CultureInfo.InvariantCulture)
                    : iv != BigInteger.Parse(d, CultureInfo.InvariantCulture);
            return true;
        }

        // ---------------------------------------------------------------------------------------------------------------
        // The corpus index
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>How an argument was given.</summary>
        internal enum ArgKind
        {
            /// <summary>Left out: the C# default (NumPy called without it).</summary>
            Omitted,

            /// <summary>An explicit null (Python's None).</summary>
            Null,

            /// <summary>An explicit non-null value.</summary>
            Value,
        }

        /// <summary>One argument of an indexed case.</summary>
        /// <param name="Name">The parameter name.</param>
        /// <param name="Kind">How it was given.</param>
        /// <param name="Key">A canonical identity of the value (an operand by the SHA-256 of its serialized form), for
        ///     counting distinct values.</param>
        /// <param name="Scalar">A scalar spelling when the value has one: a DType's NumPy name, or a string / bool / integer
        ///     as text.</param>
        /// <param name="Bits">A double's IEEE bits (16 hex digits), or null.</param>
        /// <param name="IsNone">Whether the value is a <c>default(Shape)</c> (<c>size=None</c>).</param>
        /// <param name="Length">An array value's element count, or null.</param>
        /// <param name="Engine">The engine of an engine-bearing object argument (bit generator, Generator, RandomState:
        ///     <c>legacy</c> for a legacy-seeded one), or null.</param>
        internal sealed record IndexedArg(string Name, ArgKind Kind, string Key, string Scalar, string Bits, bool IsNone,
            int? Length, string Engine);

        /// <summary>One corpus case, reduced to what the gates read.</summary>
        /// <param name="Id">The case id.</param>
        /// <param name="Sig">The overload's canonical signature.</param>
        /// <param name="RecvKind">The receiver kind (<c>none</c>, <c>RandomState</c>, <c>Generator</c>, …).</param>
        /// <param name="RecvEngine">The receiver's engine (for an object receiver, its source's), null for the legacy
        ///     MT19937 or none.</param>
        /// <param name="RecvSeed">The receiver's seed (for an object receiver, its source's), or null.</param>
        /// <param name="Throws">Whether NumPy raised (an error case).</param>
        /// <param name="HasState">Whether the case records the receiver's state after the call.</param>
        /// <param name="Args">The arguments, in parameter order.</param>
        internal sealed record IndexedCase(string Id, string Sig, string RecvKind, string RecvEngine, string RecvSeed,
            bool Throws, bool HasState, IndexedArg[] Args);

        /// <summary>
        ///     The random-API corpus reduced to <see cref="IndexedCase"/>s: every tier read line by line, each line parsed
        ///     once and discarded.
        /// </summary>
        internal sealed class CorpusIndex
        {
            /// <summary>Every case of every tier.</summary>
            internal List<IndexedCase> All { get; } = new();

            /// <summary>The cases by signature.</summary>
            internal Dictionary<string, List<IndexedCase>> BySig { get; } = new(StringComparer.Ordinal);

            /// <summary>Reads the four tiers from the test output's corpus folder.</summary>
            /// <returns>The index.</returns>
            /// <exception cref="FileNotFoundException">A tier file is missing (the csproj copies the corpus at build).</exception>
            internal static CorpusIndex Load()
            {
                var index = new CorpusIndex();
                foreach (var tier in Tiers)
                {
                    string path = FuzzCorpus.CorpusPath(tier);
                    if (!File.Exists(path))
                        throw new FileNotFoundException($"random-API tier {tier} is missing from the test output", path);
                    foreach (var line in File.ReadLines(path))
                    {
                        if (line.Length == 0)
                            continue;
                        using var doc = JsonDocument.Parse(line);
                        var c = Reduce(doc.RootElement);
                        index.All.Add(c);
                        if (!index.BySig.TryGetValue(c.Sig, out var list))
                            index.BySig[c.Sig] = list = new List<IndexedCase>();
                        list.Add(c);
                    }
                }
                return index;
            }

            /// <summary>Reduces one corpus line to an <see cref="IndexedCase"/>.</summary>
            /// <param name="root">The parsed line.</param>
            /// <returns>The reduced case.</returns>
            private static IndexedCase Reduce(JsonElement root)
            {
                var prm = root.GetProperty("params");
                var recv = prm.GetProperty("recv");
                // An object receiver (a state object, a RandomState's engine) names its source receiver under "of".
                var source = recv.TryGetProperty("of", out var of) ? of : recv;
                string engine = source.TryGetProperty("engine", out var e) ? e.GetString() : null;
                string seed = source.TryGetProperty("seed", out var s) ? s.GetString() : null;
                bool throws = root.TryGetProperty("expects_throw", out var t) && t.ValueKind == JsonValueKind.True;
                bool hasState = root.TryGetProperty("expected", out var exp) && exp.TryGetProperty("state", out var st)
                                && st.ValueKind == JsonValueKind.String;
                var operands = root.TryGetProperty("operands", out var ops) ? ops : default;
                var args = new List<IndexedArg>();
                foreach (var a in prm.GetProperty("args").EnumerateArray())
                    args.Add(ReduceArg(a, operands));
                return new IndexedCase(root.GetProperty("id").GetString(), prm.GetProperty("sig").GetString(),
                    recv.GetProperty("k").GetString(), engine, seed, throws, hasState, args.ToArray());
            }

            /// <summary>Reduces one argument (see <see cref="IndexedArg"/>).</summary>
            /// <param name="a">The argument object.</param>
            /// <param name="operands">The case's operand array (for an operand's identity).</param>
            /// <returns>The reduced argument.</returns>
            private static IndexedArg ReduceArg(JsonElement a, JsonElement operands)
            {
                string name = a.GetProperty("n").GetString();
                if (a.TryGetProperty("omit", out _))
                    return new IndexedArg(name, ArgKind.Omitted, "omit", null, null, false, null, null);
                if (a.TryGetProperty("null", out _))
                    return new IndexedArg(name, ArgKind.Null, "null", null, null, false, null, null);
                string key;
                if (a.TryGetProperty("op", out var op))
                    key = "op:" + Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(operands[op.GetInt32()].GetRawText())));
                else
                {
                    // The value fields without the name/type labels, in the generator's (deterministic) order.
                    var sb = new StringBuilder();
                    foreach (var prop in a.EnumerateObject())
                        if (prop.Name != "n" && prop.Name != "t")
                            sb.Append(prop.Name).Append('=').Append(prop.Value.GetRawText()).Append(';');
                    key = sb.ToString();
                }
                string scalar = null;
                if (a.TryGetProperty("name", out var dt))
                    scalar = dt.GetString();
                else if (a.TryGetProperty("v", out var v))
                    scalar = v.ValueKind switch
                    {
                        JsonValueKind.String => v.GetString(),
                        JsonValueKind.True => "true",
                        JsonValueKind.False => "false",
                        _ => null,
                    };
                string bits = a.TryGetProperty("bits", out var b) ? b.GetString() : null;
                bool isNone = a.TryGetProperty("none", out _);
                int? length = a.TryGetProperty("v", out var arr) && arr.ValueKind == JsonValueKind.Array ? arr.GetArrayLength() : null;
                string engine = null;
                if (a.TryGetProperty("obj", out var obj))
                {
                    foreach (var kind in new[] { "bitgen", "generator", "randomstate" })
                        if (obj.TryGetProperty(kind, out var spec))
                            engine = spec.TryGetProperty("engine", out var en) ? en.GetString() : "legacy";
                }
                return new IndexedArg(name, ArgKind.Value, key, scalar, bits, isNone, length, engine);
            }
        }
    }
}
