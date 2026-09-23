using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>What kind of public member a <see cref="SurfaceMember"/> is.</summary>
    internal enum SurfaceKind
    {
        /// <summary>A named method group (all overloads of one name).</summary>
        Method,

        /// <summary>A property without index parameters (read — and, when settable, round-trip written — by the reflective read gate).</summary>
        Property,

        /// <summary>An indexer property (<c>this[...]</c>): needs index arguments, so it is covered by a corpus family or a catalogue entry.</summary>
        Indexer,

        /// <summary>A public field (read by the reflective read gate).</summary>
        Field,

        /// <summary>An operator (<c>op_*</c> special-name method group).</summary>
        Operator,
    }

    /// <summary>
    ///     One member of the NumPy-facing public surface the leak gate must cover.
    /// </summary>
    /// <param name="Id">The coverage id: <c>&lt;owner&gt;.&lt;name&gt;</c> — the [ModuleName] for module
    /// members (<c>np.histogram</c>, <c>ndarray.GetData</c>, <c>np.ma.sum</c>), the owner's short name
    /// for object members and operators (<c>Generator.normal</c>, <c>NDMaskedArray.op_Addition</c>).</param>
    /// <param name="Owner">The owner id (module name or object short name).</param>
    /// <param name="Name">The member name.</param>
    /// <param name="Kind">The member kind.</param>
    /// <param name="OwnerType">The reflected owner type.</param>
    /// <param name="Members">Every reflected member behind the name (all overloads / the property / field).</param>
    internal sealed record SurfaceMember(string Id, string Owner, string Name, SurfaceKind Kind, Type OwnerType, MemberInfo[] Members);

    /// <summary>
    ///     Enumerates the surface the leak gate covers, with the SAME discovery rules as
    ///     <c>coverage/NumSharp.Tools.ApiInventory</c> (the repo's authoritative API inventory):
    ///     <list type="bullet">
    ///       <item><b>Modules</b> — every exported type carrying <c>[ModuleName]</c> (np, ndarray,
    ///             np.random, np.fft, np.linalg, np.emath, np.ma, np.dtypes), members reflected
    ///             <c>Public | Static | Instance | DeclaredOnly</c>, special names excluded — the inventory
    ///             tool's <c>InspectType</c>, verbatim.</item>
    ///       <item><b>Operators</b> — the <c>op_*</c> special names of <see cref="NDArray"/>,
    ///             <see cref="NumSharp.Generic.NDArray{TDType}"/>, <see cref="NDMaskedArray"/> and every object
    ///             owner that declares any (<see cref="DType"/>, <see cref="NDArrayFlags"/>, <see cref="poly1d"/>),
    ///             which the inventory filters out by construction but which run code — and, for the array
    ///             and polynomial types, allocate — on every use.</item>
    ///       <item><b>Object surfaces</b> — the NumPy object types <c>coverage/object_surfaces.py</c> maps
    ///             to CLR owners (Generator, SeedSequence, the bit generators, the iterator objects,
    ///             DType, finfo/iinfo, NDArrayFlags, poly1d, NpzFile), plus <see cref="NDMaskedArray"/>
    ///             and <c>NDArray&lt;T&gt;</c>, reflected the way the inventory's <c>objectTypes</c> is
    ///             (inherited NumSharp members included, System.Object's excluded).</item>
    ///     </list>
    /// </summary>
    internal static class LeakSurface
    {
        /// <summary>
        ///     The object-surface owners: <c>coverage/object_surfaces.py</c>'s CLR-mapped NumPy objects
        ///     (the ones NumSharp exports — PCG64DXSM/Philox/SFC64 have no NumSharp type) plus the two
        ///     array types whose instance surface is the masked/typed half of ndarray. Keyed by the
        ///     short owner id used in coverage ids.
        /// </summary>
        public static readonly (string Id, Type Type, bool DeclaredOnly)[] ObjectOwners =
        {
            ("Generator", typeof(Generator), false),
            ("SeedSequence", typeof(SeedSequence), false),
            ("BitGenerator", typeof(BitGenerator), false),
            ("PCG64", typeof(PCG64), false),
            ("MT19937", typeof(MT19937), false),
            ("NDIterator", typeof(np.NDIterator), false),
            ("FlatIterator", typeof(np.FlatIterator), false),
            ("NDIndex", typeof(np.NDIndex), false),
            ("NDEnumerate", typeof(np.NDEnumerate), false),
            ("Broadcast", typeof(np.Broadcast), false),
            ("DType", typeof(DType), false),
            ("finfo", typeof(finfo), false),
            ("iinfo", typeof(iinfo), false),
            ("NDArrayFlags", typeof(NDArrayFlags), false),
            ("poly1d", typeof(poly1d), false),
            ("NpzFile", typeof(NumSharp.IO.NpzFile), false),
            ("NDMaskedArray", typeof(NDMaskedArray), false),
            // NDArray<T>'s inherited NDArray surface is the ndarray module's; only its OWN members count.
            ("NDArray<T>", typeof(NumSharp.Generic.NDArray<>), true),
        };

        /// <summary>
        ///     The types whose operators are enumerated, with their owner id: the array types, plus every
        ///     <see cref="ObjectOwners"/> type that declares <c>op_*</c> members. The completeness gate fails when
        ///     an object owner declares an operator but is missing here (see
        ///     <see cref="LeakSurfaceCoverageTests.EveryInventoryMember_IsLeakAudited"/>), so a new operator on,
        ///     say, <see cref="finfo"/> cannot ship leak-unaudited.
        /// </summary>
        public static readonly (string Id, Type Type)[] OperatorOwners =
        {
            ("ndarray", typeof(NDArray)),
            ("NDArray<T>", typeof(NumSharp.Generic.NDArray<>)),
            ("NDMaskedArray", typeof(NDMaskedArray)),
            // Object owners with operators: DType's conversions and safe-cast ordering, the flags object's
            // equality, and poly1d's arithmetic (which allocates a new polynomial per call).
            ("DType", typeof(DType)),
            ("NDArrayFlags", typeof(NDArrayFlags)),
            ("poly1d", typeof(poly1d)),
        };

        /// <summary>Whether <paramref name="type"/> declares (not inherits) at least one public static <c>op_*</c> member.</summary>
        /// <param name="type">The owner type.</param>
        /// <returns>True when the type declares an operator or conversion.</returns>
        public static bool DeclaresOperators(Type type)
            => type.GetMethods(BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly)
                   .Any(m => m.IsSpecialName && m.Name.StartsWith("op_", StringComparison.Ordinal));

        /// <summary>Every surface member, in a stable (owner, name) order.</summary>
        /// <returns>The full list.</returns>
        public static IReadOnlyList<SurfaceMember> Enumerate()
        {
            var core = typeof(np).Assembly;
            var list = new List<SurfaceMember>();

            // ---- [ModuleName] modules: the inventory tool's discovery, verbatim ----
            const BindingFlags ModuleFlags = BindingFlags.Public | BindingFlags.Static | BindingFlags.Instance | BindingFlags.DeclaredOnly;
            foreach (var type in core.GetExportedTypes())
            {
                var module = type.GetCustomAttribute<ModuleNameAttribute>(inherit: false);
                if (module is null)
                    continue;
                AddMembers(list, module.Name, type, ModuleFlags, core);
            }

            // ---- operators ----
            foreach (var (id, type) in OperatorOwners)
            {
                foreach (var g in type.GetMethods(BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly)
                             .Where(m => m.IsSpecialName && m.Name.StartsWith("op_", StringComparison.Ordinal))
                             .GroupBy(m => m.Name, StringComparer.Ordinal))
                    list.Add(new SurfaceMember($"{id}.{g.Key}", id, g.Key, SurfaceKind.Operator, type, g.Cast<MemberInfo>().ToArray()));
            }

            // ---- object surfaces ----
            foreach (var (id, type, declaredOnly) in ObjectOwners)
            {
                var flags = BindingFlags.Public | BindingFlags.Static | BindingFlags.Instance |
                            (declaredOnly ? BindingFlags.DeclaredOnly : BindingFlags.FlattenHierarchy);
                AddMembers(list, id, type, flags, core);
            }

            return list.OrderBy(m => m.Owner, StringComparer.Ordinal).ThenBy(m => m.Name, StringComparer.Ordinal).ToList();
        }

        /// <summary>Adds one owner's methods, properties and fields (declared in the Core assembly, special names excluded).</summary>
        /// <param name="list">The accumulating list.</param>
        /// <param name="owner">The owner id.</param>
        /// <param name="type">The owner type.</param>
        /// <param name="flags">Reflection flags (DeclaredOnly for modules, FlattenHierarchy for objects).</param>
        /// <param name="core">The NumSharp.Core assembly (members declared elsewhere — System.Object's — are excluded).</param>
        private static void AddMembers(List<SurfaceMember> list, string owner, Type type, BindingFlags flags, Assembly core)
        {
            foreach (var g in type.GetMethods(flags)
                         .Where(m => !m.IsSpecialName && m.DeclaringType?.Assembly == core)
                         .GroupBy(m => m.Name, StringComparer.Ordinal))
                list.Add(new SurfaceMember($"{owner}.{g.Key}", owner, g.Key, SurfaceKind.Method, type, g.Cast<MemberInfo>().ToArray()));

            foreach (var g in type.GetProperties(flags)
                         .Where(p => p.DeclaringType?.Assembly == core)
                         .GroupBy(p => p.Name, StringComparer.Ordinal))
            {
                bool indexer = g.Any(p => p.GetIndexParameters().Length > 0);
                list.Add(new SurfaceMember($"{owner}.{g.Key}", owner, g.Key, indexer ? SurfaceKind.Indexer : SurfaceKind.Property,
                                           type, g.Cast<MemberInfo>().ToArray()));
            }

            foreach (var g in type.GetFields(flags)
                         .Where(f => f.DeclaringType?.Assembly == core)
                         .GroupBy(f => f.Name, StringComparer.Ordinal))
                list.Add(new SurfaceMember($"{owner}.{g.Key}", owner, g.Key, SurfaceKind.Field, type, g.Cast<MemberInfo>().ToArray()));
        }
    }

    /// <summary>
    ///     Everything the completeness gate may credit a surface member with — the observations of the four
    ///     runs of the process — so <see cref="LeakSurfaceCoverageTests.Resolve"/> can only ever credit what
    ///     was actually MEASURED, never what was merely declared (a catalogue entry that could not run, a
    ///     property whose target could not be built, a LAPACK op key listed for a backend pass that did not
    ///     measure it).
    /// </summary>
    /// <param name="Corpus">The shared corpus sweep (<see cref="UndisposedIntermediateTests.SharedSweep"/>).</param>
    /// <param name="Catalogue">The shared catalogue run (<see cref="UndisposedIntermediateTests.SharedCatalogue"/>).</param>
    /// <param name="Properties">The shared property/field read run (<see cref="UndisposedIntermediateTests.SharedPropertyReads"/>).</param>
    /// <param name="Backend">The shared backend pass (<see cref="UndisposedIntermediateTests.SharedBackendSweep"/>).</param>
    internal sealed record LeakCoverageEvidence(
        UndisposedIntermediateTests.SweepResult Corpus,
        UndisposedIntermediateTests.DirectRunResult Catalogue,
        UndisposedIntermediateTests.DirectRunResult Properties,
        UndisposedIntermediateTests.BackendSweepResult Backend)
    {
        /// <summary>
        ///     Whether the corpus sweep exercised <paramref name="key"/> on a SUCCESS path — a confirmed
        ///     measurement, or one a GC made inconclusive (the case ran; never red). An op whose every case
        ///     raised is NOT credited: its success path went unaudited.
        /// </summary>
        /// <param name="key">An op key or derived coverage key (<c>rnd:…</c>, <c>grnd:…</c>, <c>index.get</c>); null is never covered.</param>
        /// <returns>True when the corpus audited the key's success path.</returns>
        public bool CorpusAudited(string key)
            => key != null && (Corpus.MeasuredByOp.GetValueOrDefault(key) > 0 || Corpus.InconclusiveIds.Contains(key));

        /// <summary>
        ///     The route label crediting backend-only op key <paramref name="op"/>: measured by the backend pass,
        ///     or — on a host where no CBLAS/LAPACK library loads — credited as skipped, exactly as the host-pinned
        ///     parity tiers go Inconclusive there instead of red. Null when the pass ran and did NOT measure it.
        /// </summary>
        /// <param name="op">A corpus op key in <see cref="UndisposedIntermediateTests.BackendOnlyOpKeys"/>.</param>
        /// <returns>The route label, or null when uncovered.</returns>
        public string BackendRoute(string op)
        {
            if (Backend.SkipReason != null)
                return "corpus: backend pass (skipped: no CBLAS/LAPACK library)";
            return Backend.Attempted(op) ? "corpus: backend pass" : null;
        }
    }

    /// <summary>
    ///     COMPLETENESS gate for the leak audit: every member of the NumPy-facing public surface
    ///     (<see cref="LeakSurface"/> — the ApiInventory modules, the operators, the mapped object
    ///     surfaces) must be LEAK-MEASURED somewhere — a measured corpus case (any family of
    ///     <see cref="UndisposedIntermediateTests.SharedSweep"/>, or the backend pass for the LAPACK-only
    ///     ops), a <see cref="LeakCatalogue"/> entry, or (properties and fields) the reflective read gate.
    ///     A new public API therefore cannot ship leak-unaudited: it fails here until it gains corpus
    ///     rows or a catalogue entry.
    /// </summary>
    /// <remarks>
    ///     <para>Credit follows MEASUREMENT, not declaration: a catalogue entry or property read counts only when
    ///     its run actually exercised the member (<see cref="UndisposedIntermediateTests.DirectRunResult.Attempted"/>),
    ///     and a LAPACK-only op only when the backend pass measured it — an entry whose warm invocation throws
    ///     therefore leaves its member uncovered here as well as failing its own gate.</para>
    ///     <para>The resolution maps below are the ONLY place a member is credited through a differently
    ///     named corpus key; each one is checked against the reflected surface (a renamed member makes
    ///     its mapping fail instead of rotting), and every catalogue id must name a real member.</para>
    /// </remarks>
    [TestClass]
    [DoNotParallelize]   // reads the shared sweep, which measures process-global pool counters
    public class LeakSurfaceCoverageTests
    {
        /// <summary>
        ///     np.linalg member → the corpus op key whose case invokes it (OpRegistry spells the linalg
        ///     products by their bare names; a key named like an np.* function — <c>trace</c>, <c>outer</c> —
        ///     calls the np.* one, which is why the Array-API forms are catalogue entries, not mapped here).
        /// </summary>
        internal static readonly Dictionary<string, string> LinalgOpKeys = new(StringComparer.Ordinal)
        {
            ["cholesky"] = "cholesky", ["cond"] = "cond", ["det"] = "det", ["eig"] = "eig", ["eigh"] = "eigh",
            ["eigvals"] = "eigvals", ["eigvalsh"] = "eigvalsh", ["inv"] = "inv", ["lstsq"] = "lstsq",
            ["matrix_norm"] = "matrix_norm", ["matrix_power"] = "matrix_power", ["matrix_rank"] = "matrix_rank",
            ["multi_dot"] = "multi_dot", ["norm"] = "norm", ["pinv"] = "pinv", ["qr"] = "qr",
            ["slogdet"] = "slogdet", ["solve"] = "solve", ["svd"] = "svd", ["svdvals"] = "svdvals",
            ["tensorinv"] = "tensorinv", ["tensorsolve"] = "tensorsolve", ["vector_norm"] = "vector_norm",
        };

        /// <summary>
        ///     np.random (NumPyRandom) members credited through a stream/state corpus key other than
        ///     <c>rnd:&lt;name&gt;</c>: the state ops have their own keys, <c>random</c> is the
        ///     <c>random_sample</c> alias, and <c>bytes</c>/<c>random_integers</c> ride the Generator tier's
        ///     RandomState helpers (<c>OpRegistry.GeneratorDraw</c>).
        /// </summary>
        internal static readonly Dictionary<string, string> RandomOpKeys = new(StringComparer.Ordinal)
        {
            ["random"] = "rnd:random_sample",
            ["seed"] = "seed",
            ["set_state"] = "set_state",
            ["get_state"] = "get_state",
            ["bytes"] = "grnd:rs_bytes",
            ["random_integers"] = "grnd:random_integers",
        };

        /// <summary>
        ///     The completeness verdict: lists every surface member no measurement covers, then every
        ///     stale mapping/catalogue id and every object owner whose operators are not enumerated. See the
        ///     class remarks for the resolution order.
        /// </summary>
        /// <remarks>
        ///     Reads all four shared runs (corpus sweep, catalogue, property reads, backend pass), so run in
        ///     isolation it pays for each of them once; in a ScopeAudit/FuzzMatrix run they are already built.
        /// </remarks>
        /// <exception cref="AssertFailedException">A member is unaudited, a mapping/catalogue id is stale, or an
        /// object owner declares operators outside <see cref="LeakSurface.OperatorOwners"/>.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void EveryInventoryMember_IsLeakAudited()
        {
            var evidence = new LeakCoverageEvidence(
                UndisposedIntermediateTests.SharedSweep.Value,
                UndisposedIntermediateTests.SharedCatalogue.Value,
                UndisposedIntermediateTests.SharedPropertyReads.Value,
                UndisposedIntermediateTests.SharedBackendSweep.Value);
            var declaredCatalogue = LeakCatalogue.ApiIds;
            var surface = LeakSurface.Enumerate();
            var ids = new HashSet<string>(surface.Select(m => m.Id), StringComparer.Ordinal);

            var missing = new List<string>();
            var byRoute = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var m in surface)
            {
                string route = Resolve(m, evidence);
                if (route == null)
                {
                    // Say WHY when a declaration exists without a measurement — that is a broken entry or
                    // target, not a missing one, and the fix is different.
                    string why = declaredCatalogue.Contains(m.Id)
                        ? "a catalogue entry is DECLARED but was never measured (it is a harness error — see the catalogue gate's list)"
                        : m.Kind is SurfaceKind.Property or SurfaceKind.Field
                            ? "the reflective read never exercised it (no read target could be built — see the read gate's harness errors)"
                            : "no measured corpus case, backend-pass op, catalogue entry or reflective read covers it";
                    missing.Add($"{m.Id} ({m.Kind}) — {why}");
                }
                else
                    byRoute[route] = byRoute.GetValueOrDefault(route) + 1;
            }

            // Self-retirement: every mapping and catalogue id must name a live member.
            var stale = new List<string>();
            foreach (var id in declaredCatalogue)
                if (!ids.Contains(id))
                    stale.Add($"catalogue entry '{id}' names no surface member (renamed/removed API, or a typo)");
            foreach (var name in LinalgOpKeys.Keys)
                if (!ids.Contains("np.linalg." + name))
                    stale.Add($"LinalgOpKeys entry '{name}' names no np.linalg member");
            foreach (var name in RandomOpKeys.Keys)
                if (!ids.Contains("np.random." + name))
                    stale.Add($"RandomOpKeys entry '{name}' names no np.random member");
            foreach (var op in UndisposedIntermediateTests.BackendOnlyOpKeys)
                if (!LinalgOpKeys.ContainsValue(op) && !ids.Contains("np." + op))
                    stale.Add($"BackendOnlyOpKeys entry '{op}' maps to no np / np.linalg member");

            // Operator enumeration must track the owners: an object owner that gains an operator (or a
            // conversion) outside OperatorOwners would be invisible to this gate, not merely uncovered.
            foreach (var (id, type, _) in LeakSurface.ObjectOwners)
                if (LeakSurface.DeclaresOperators(type) && !LeakSurface.OperatorOwners.Any(o => o.Type == type))
                    stale.Add($"object owner '{id}' declares op_* members but is missing from LeakSurface.OperatorOwners");

            Console.WriteLine($"[leak-surface] members={surface.Count} covered={surface.Count - missing.Count} " +
                              $"missing={missing.Count} catalogueIds={declaredCatalogue.Count} " +
                              $"measuredOpKeys={evidence.Corpus.MeasuredByOp.Count}" +
                              (evidence.Backend.SkipReason != null ? $" backendSkipped=({evidence.Backend.SkipReason})" : "") + "\n  " +
                              string.Join("\n  ", byRoute.OrderByDescending(k => k.Value).Select(k => $"{k.Key}: {k.Value}")));

            var problems = missing.Concat(stale).ToList();
            if (problems.Count > 0)
                Assert.Fail($"{missing.Count} surface members are not leak-audited and {stale.Count} mappings are stale:\n  " +
                            string.Join("\n  ", problems.Take(400)));
        }

        /// <summary>
        ///     Resolves which MEASUREMENT covers <paramref name="m"/>, or null when none does. Order:
        ///     properties/fields → the reflective read gate (it attempts EVERY one); then, for every kind, a
        ///     catalogue entry that ran; indexers → the index tiers (ndarray only); operators → catalogue only;
        ///     methods → a measured corpus key (per-owner naming + the alias maps), then the backend pass.
        /// </summary>
        /// <param name="m">The surface member.</param>
        /// <param name="ev">The four shared runs' observations.</param>
        /// <returns>A route label (for the rollup), or null when uncovered.</returns>
        internal static string Resolve(SurfaceMember m, LeakCoverageEvidence ev)
        {
            bool Has(string key) => ev.CorpusAudited(key);

            if (m.Kind is SurfaceKind.Property or SurfaceKind.Field)
            {
                if (ev.Properties.Attempted(m.Id))
                    return "reflective read";
                if (ev.Properties.Skipped(m.Id))
                    return "reflective read (backend skipped)";
                // Not read (its target could not be built): a catalogue entry may still cover it below.
            }

            if (ev.Catalogue.Attempted(m.Id))
                return "catalogue";
            if (ev.Catalogue.Skipped(m.Id))
                return "catalogue (backend skipped)";

            if (m.Kind is SurfaceKind.Property or SurfaceKind.Field)
                return null;

            if (m.Kind == SurfaceKind.Indexer)
                return m.Owner == "ndarray" && Has("index.get") && Has("index.set") ? "corpus: index tiers" : null;

            if (m.Kind == SurfaceKind.Operator)
                return null;   // operators are catalogue-only (the corpus drives np.* functions, never operators)

            switch (m.Owner)
            {
                case "np":
                    if (Has(m.Name))
                        return "corpus: np";
                    if (OracleSurfaceCoverageTests.EquivalentAliases.TryGetValue(m.Name, out var npCanonical) && Has(npCanonical))
                        return "corpus: np alias";
                    if (UndisposedIntermediateTests.BackendOnlyOpKeys.Contains(m.Name))
                        return ev.BackendRoute(m.Name);
                    return null;
                case "ndarray":
                    if (Has("ndarray." + m.Name))
                        return "corpus: ndarray";
                    if (OracleSurfaceCoverageTests.NdarrayAliases.TryGetValue(m.Name, out var ndTarget) && Has(ndTarget))
                        return "corpus: ndarray alias";
                    return null;
                case "np.ma":
                    if (Has("ma." + m.Name))
                        return "corpus: ma";
                    if (OracleSurfaceCoverageTests.MaAliases.TryGetValue(m.Name, out var maTarget) && Has(maTarget))
                        return "corpus: ma alias";
                    return null;
                case "np.emath":
                    return Has("emath." + m.Name) ? "corpus: emath" : null;
                case "np.fft":
                    return Has(m.Name) ? "corpus: fft" : null;
                case "np.linalg":
                    if (LinalgOpKeys.TryGetValue(m.Name, out var linKey))
                    {
                        if (Has(linKey))
                            return "corpus: linalg";
                        if (UndisposedIntermediateTests.BackendOnlyOpKeys.Contains(linKey))
                            return ev.BackendRoute(linKey);
                    }
                    return null;
                case "np.random":
                    if (Has("rnd:" + m.Name))
                        return "corpus: rnd stream";
                    if (RandomOpKeys.TryGetValue(m.Name, out var rndKey) && Has(rndKey))
                        return "corpus: random state/helper";
                    return null;
                case "Generator":
                    return Has("grnd:" + m.Name) ? "corpus: grnd stream" : null;
                default:
                    return null;
            }
        }
    }
}
