using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     Completeness guard for the PUBLIC NumSharp NumPy surface. Value coverage is discovered
    ///     from the committed corpus; everything that is not a direct corpus op must be explicitly
    ///     classified as an equivalent alias, a sibling-gate responsibility, a compatibility-only
    ///     extension, or a pinned known gap. A newly-added public API therefore cannot silently miss
    ///     the oracle.
    /// </summary>
    [TestClass]
    public class OracleSurfaceCoverageTests
    {
        /// <summary>
        ///     np.* spellings whose value path IS another corpus op (member name → canonical op key) —
        ///     pure delegations such as <c>absolute → abs</c>. <c>internal</c> because the leak gate
        ///     (<see cref="LeakSurfaceCoverageTests"/>) resolves coverage through the SAME map: a
        ///     delegating alias allocates exactly what its canonical op allocates, so a measured
        ///     canonical op leak-covers the alias too.
        /// </summary>
        internal static readonly Dictionary<string, string> EquivalentAliases = new()
        {
            ["absolute"] = "abs",
            ["amax"] = "max",
            ["amin"] = "min",
            ["around"] = "round_",
            ["bitwise_not"] = "invert",
            ["broadcast"] = "broadcast_values",
            ["common_type_code"] = "common_type",
            ["concat"] = "concatenate",
            ["degrees"] = "rad2deg",
            // np.fix IS np.trunc (numpy/lib/_ufunclike_impl.py delegates verbatim: trunc(x, out=out));
            // it has no where/dtype of its own, so trunc's unary-tier corpus already gates its arithmetic.
            // Fix-specific API (the out= slot, the no-where/dtype signature) is gated by np.fix.Test.cs.
            ["fix"] = "trunc",
            ["radians"] = "deg2rad",
            ["remainder"] = "mod",
            ["result_type"] = "result_type_arrays",
            ["true_divide"] = "divide",
        };

        // These have a stronger or more appropriate gate elsewhere, or have no deterministic
        // value bytes to compare in an operand/result corpus.
        private static readonly HashSet<string> SiblingOwned = new()
        {
            "array2string",
            // apply_along_axis / apply_over_axes take a CALLABLE (a C# Func) applied per 1-D slice /
            // per axis — there is no NumPy-serializable operand form for an arbitrary delegate, so
            // the differential corpus (which replays operand bytes) cannot express them (same reason
            // as np.evaluate, which takes an NDExpr). Gated by the dedicated np.apply_along_axis.Test.cs
            // / np.apply_over_axes.Test.cs suites, verified bit-exact against NumPy 2.4.2.
            "apply_along_axis", "apply_over_axes",
            // vectorize / frompyfunc wrap an arbitrary C# DELEGATE (returning a Func / a Vectorized
            // object, not an array), exactly like apply_along_axis — there is no NumPy-serializable
            // operand form for a user function, so the differential corpus (which replays operand
            // bytes) cannot express them. Gated by the dedicated np.vectorize.Test.cs suite (both
            // modes, multi-output, otypes, broadcasting, and the error parity), verified bit-exact
            // against NumPy 2.4.2. The element-wise path additionally rides np.evaluate's fused
            // NDExpr.Call machinery, already fuzzed via evaluate.jsonl.
            "vectorize", "frompyfunc",
            // broadcast_shapes takes SHAPES (tuples of ints), not array operands, so it has no entry
            // in the (dtype,shape,strides,bytes)->result operand corpus. Gated by the dedicated
            // np.broadcast_shapes.Test.cs suite (happy path across argument forms + verbatim error
            // messages), verified against NumPy 2.4.2.
            "broadcast_shapes",
            // getbufsize/setbufsize configure the ufunc/NDIter default buffer size (NumPy's
            // NPY_BUFSIZE, 8192). They are a thread-local config getter/setter with no array
            // result and no deterministic value bytes — buffering only changes internal chunking,
            // never a computed value — so there is nothing for an operand/result corpus to
            // bit-compare (same rationale as get_printoptions/set_printoptions). Gated by the
            // dedicated np.bufsize.Test.cs suite, verified against NumPy 2.4.2.
            "getbufsize", "setbufsize",
            // bmat is pure block-assembly (concatenation) over the already-fuzzed concatenate +
            // asmatrix — its nested-block ([[A,B],[C,D]]) and string+dict inputs have no single-operand
            // corpus representation (the same reason block needs a bespoke multi-operand oracle path);
            // gated by the dedicated np.bmat.Tests.cs suite verified against NumPy 2.4.2.
            "bmat",
            // binary_repr / base_repr are SCALAR integer -> string formatters (NumPy takes a single
            // Python int via operator.index / int(), not an array), so they have no operand form in the
            // (dtype,shape,strides,bytes)->result corpus — the same reason format_float_positional /
            // format_float_scientific are sibling-owned. Gated by the dedicated np.base_repr.Test.cs suite
            // (both funcs), differentially verified against NumPy 2.4.2 across ±2^100 magnitudes, every
            // width/base/padding and the ValueError/TypeError edges.
            "binary_repr", "base_repr",
            // typename is a type-CODE -> description dict lookup (NumPy's _namefromtype[char],
            // numpy/lib/_type_check_impl.py): its input is a single array-protocol type-code STRING
            // ('i', 'D', 'S1', ...), not an array, so it has no (dtype,shape,strides,bytes)->result
            // operand form in the corpus — the same reason binary_repr/base_repr are sibling-owned.
            // Gated by the dedicated np.typename.Test.cs suite (all 22 codes + case-sensitivity, the
            // 'S1'/'S' split, the absent-'e' parity, and the KeyError message on every miss incl. null),
            // verified against NumPy 2.4.2.
            "typename",
            // datetime_data reads the (unit, count) parameter off a datetime64/timedelta64 DESCRIPTOR — a dtype-level
            // accessor with no array operand or result bytes (the datetime classes have no storage yet, Stage A of
            // docs/plans/dtype-system.md). Gated by test/NumSharp.Tests/DTypes/*, probed against NumPy 2.4.2.
            "datetime_data",
            "finfo", "flat",
            // The histogram family returns TUPLES over POLYMORPHIC bins (int count / estimator string /
            // explicit edge array) and, for dd/2d, MULTI-array samples — none of which fit the single-operand
            // (dtype,shape,strides,bytes)->result corpus model (the same reason bmat and block need bespoke
            // paths). They are gated by the dedicated np.histogram.Test.cs suite plus a large live .npy
            // round-trip differential (histogram 1-D value/edges bit-exact across 11 dtypes × int/estimator/
            // edge bins × range/weights/density; histogramdd + histogram2d H bit-exact across 1-3D × bin modes),
            // all verified against NumPy 2.4.2.
            "histogram", "histogram_bin_edges", "histogramdd", "histogram2d",
            "format_float_positional", "format_float_scientific",
            "get_printoptions", "iinfo", "load", "load_npy", "load_npz",
            "nditer_chunks", "printoptions", "save", "savez",
            "savez_compressed", "set_printoptions",
            // isposinf/isneginf/nan_to_num: gated by the dedicated byte-exact suites
            // np.isposinf_isneginf.Test.cs / np.nan_to_num.Test.cs, differentially verified against
            // NumPy 2.4.2 (isposinf/isneginf byte-identical across float64/float32/float16 × C/
            // reversed/stepped layouts; nan_to_num byte-identical across float64/float32/float16/
            // complex128 × default/scalar fills, plus array-fill and copy-semantics coverage). The
            // predicates additionally ride the same fused predicate kernel already fuzzed for
            // isinf/isnan (complex rejected up front with NumPy's verbatim TypeError).
            "isposinf", "isneginf", "nan_to_num",
            // may_share_memory / shares_memory are MEMORY-LAYOUT predicates: their answer depends on the
            // two operands being views of ONE underlying buffer (or not). The differential corpus replays
            // each operand from a serialized (dtype,shape,strides,offset,bytes) tuple and reconstructs them
            // as INDEPENDENT arrays, so it has no way to express "b is a view of a" — the exact relationship
            // these functions test — and every replayed pair would trivially not overlap. Same structural
            // reason the iteration protocols (nditer/ndindex/ndenumerate) are absent. Gated by the dedicated
            // np.shares_memory.Test.cs suite, verified against NumPy 2.4.2 across view/copy/interleaved/
            // transposed layouts and the max_work / TooHardError / ValueError edges.
            "may_share_memory", "shares_memory",
            // array_equiv's VALUE path is identical to array_equal's — both reduce all(a == b) — and
            // array_equal is already a direct corpus op (logic.jsonl via ALLCLOSE_OPS), so the elementwise
            // comparison + all-reduction is fuzzed across every dtype/layout there. array_equiv's ONLY
            // distinguishing behavior is the broadcast-shape gate (shape-consistent rather than exact),
            // which the single-operand-per-slot differential corpus cannot express (it reconstructs each
            // operand independently and can't stage a genuine (M,)-vs-(N,M) broadcast pair) — the same
            // structural reason the set routines and broadcast predicates are sibling-owned. Gated by the
            // dedicated np.array_equiv.Test.cs suite (broadcast-consistent / non-broadcastable / scalar +
            // column broadcast / empty / NaN / mixed-dtype), verified against NumPy 2.4.2.
            "array_equiv",
        };

        // NumSharp compatibility/convenience APIs with no NumPy 2.4.2 callable of the same name.
        private static readonly HashSet<string> CompatibilityOnly = new()
        {
            "are_broadcastable", "asscalar", "find_common_type", "issctype", "issubsctype",
            "maximum_sctype", "multithreading", "ndarray", "save_version", "sctype2char",
        };

        private static readonly HashSet<string> RandomStateSurface = new()
        {
            "RandomState",
        };

        // Non-stream Generator/RandomState API surface: these return a Generator, a byte[], or are a
        // deterministic randint alias, so they have no per-draw "rnd" stream corpus entry. Each is
        // gated by dedicated unit tests (np.random.default_rng.Test.cs / np.random.bytes.Test.cs /
        // np.random.random_integers.Test.cs) verified byte-exact against NumPy 2.4.2.
        private static readonly HashSet<string> GeneratorApiSurface = new()
        {
            "default_rng", "bytes", "random_integers",
        };

        // Stream algorithms already carved and pinned under OpenBugs.Random.cs. Re-adding any one
        // to random_parity(_host).jsonl automatically moves it to direct coverage and this set entry
        // becomes stale/fails below.
        private static readonly HashSet<string> RandomKnownGaps = new()
        {
            "binomial", "f", "multinomial", "multivariate_normal", "negative_binomial",
            "pareto", "standard_cauchy",
        };

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void EveryPublicNumpySurface_IsCoveredOrExplicitlyClassified()
        {
            var corpusOps = new HashSet<string>(StringComparer.Ordinal);
            var randomDists = new HashSet<string>(StringComparer.Ordinal);
            string directory = Path.GetDirectoryName(FuzzCorpus.CorpusPath("unused"));
            foreach (string path in Directory.EnumerateFiles(directory, "*.jsonl"))
            {
                if (path.EndsWith(".host.jsonl", StringComparison.Ordinal))
                    continue;
                foreach (var c in FuzzCorpus.Load(Path.GetFileName(path)))
                {
                    if (!string.IsNullOrEmpty(c.Op))
                        corpusOps.Add(c.Op);
                    if (c.Op == "rnd" && c.Params != null && c.Params.TryGetValue("dist", out var dist))
                        randomDists.Add(dist.GetString());
                }
            }

            var failures = new List<string>();
            var npNames = Surface(typeof(np), BindingFlags.Static);
            foreach (string name in npNames)
            {
                if (corpusOps.Contains(name) || SiblingOwned.Contains(name) || CompatibilityOnly.Contains(name))
                    continue;
                if (EquivalentAliases.TryGetValue(name, out string canonical))
                {
                    if (!corpusOps.Contains(canonical))
                        failures.Add($"np.{name}: alias target corpus op '{canonical}' is absent");
                    continue;
                }
                failures.Add($"np.{name}: unclassified public surface");
            }

            foreach (string name in Surface(typeof(np.linalg), BindingFlags.Static))
                if (!corpusOps.Contains(name))
                    failures.Add($"np.linalg.{name}: no direct corpus op");

            foreach (string name in Surface(typeof(FourierModule), BindingFlags.Instance))
                if (!corpusOps.Contains(name))
                    failures.Add($"np.fft.{name}: no direct corpus op");

            foreach (string name in Surface(typeof(NumPyRandom), BindingFlags.Instance))
            {
                if (corpusOps.Contains(name))
                    continue;
                if (randomDists.Contains(name))
                    continue;
                if (name == "random" && randomDists.Contains("random_sample"))
                    continue; // documented alias
                if (RandomStateSurface.Contains(name))
                    continue; // dedicated state/seed tests
                if (GeneratorApiSurface.Contains(name))
                    continue; // Generator factory / bytes / random_integers — dedicated byte-exact tests
                if (name == "bernoulli")
                    continue; // NumSharp extension; dedicated tests (NumPy spells it binomial(1,p))
                if (RandomKnownGaps.Contains(name))
                    continue;
                failures.Add($"np.random.{name}: neither stream corpus nor explicit classification");
            }

            // Classification entries must self-retire: a renamed/deleted method or a newly direct
            // random stream should make this manifest fail instead of accumulating stale prose.
            foreach (string name in EquivalentAliases.Keys.Concat(SiblingOwned).Concat(CompatibilityOnly))
            {
                if (!npNames.Contains(name))
                    failures.Add($"stale np classification: {name}");
                if (corpusOps.Contains(name))
                    failures.Add($"stale np classification: {name} now has a direct corpus op");
            }
            foreach (string name in RandomKnownGaps)
                if (randomDists.Contains(name))
                    failures.Add($"stale random known-gap classification: {name} is now in the stream corpus");

            Console.WriteLine($"[OracleSurface] np={npNames.Length}, corpus_ops={corpusOps.Count}, " +
                              $"random_streams={randomDists.Count}, aliases={EquivalentAliases.Count}, " +
                              $"sibling={SiblingOwned.Count}, compatibility={CompatibilityOnly.Count}, " +
                              $"random_open_gaps={RandomKnownGaps.Count}");

            if (failures.Count > 0)
                Assert.Fail($"{failures.Count} oracle surface coverage classification failures:\n  " +
                            string.Join("\n  ", failures));
        }

        private static string[] Surface(Type type, BindingFlags flags)
            => type.GetMethods(flags | BindingFlags.Public | BindingFlags.DeclaredOnly)
                .Where(method => !method.IsSpecialName)
                .Select(method => method.Name)
                .Distinct(StringComparer.Ordinal)
                .OrderBy(name => name, StringComparer.Ordinal)
                .ToArray();

        /// <summary>Declared public property names — np.dtypes and the ndarray property row are
        ///     property-only surfaces that <see cref="Surface"/> (methods) cannot see.</summary>
        /// <param name="type">The facade type to reflect.</param>
        /// <param name="flags">Instance or Static, per the facade's shape.</param>
        /// <returns>Distinct ordinal-sorted property names.</returns>
        private static string[] Properties(Type type, BindingFlags flags)
            => type.GetProperties(flags | BindingFlags.Public | BindingFlags.DeclaredOnly)
                .Select(p => p.Name)
                .Distinct(StringComparer.Ordinal)
                .OrderBy(name => name, StringComparer.Ordinal)
                .ToArray();

        /// <summary>All distinct op keys in the committed corpus (host pins excluded) — the same
        ///     discovery the np-surface gate uses, shared by the four facade gates below.</summary>
        /// <returns>The op-key set, ordinal-compared.</returns>
        private static HashSet<string> CorpusOps()
        {
            var ops = new HashSet<string>(StringComparer.Ordinal);
            string directory = Path.GetDirectoryName(FuzzCorpus.CorpusPath("unused"));
            foreach (string path in Directory.EnumerateFiles(directory, "*.jsonl"))
            {
                if (path.EndsWith(".host.jsonl", StringComparison.Ordinal))
                    continue;
                foreach (var c in FuzzCorpus.Load(Path.GetFileName(path)))
                    if (!string.IsNullOrEmpty(c.Op))
                        ops.Add(c.Op);
            }
            return ops;
        }

        // ================= ndarray (the instance surface, coverage plan §A1 / row G0) =========

        // NumSharp conveniences with NO NumPy 2.4.2 ndarray member of that name, whose VALUE path
        // is nevertheless a corpus op (the np.* twin or an instance key): the alias target must
        // exist, so a renamed op breaks this map instead of rotting. internal: the leak gate
        // (LeakSurfaceCoverageTests) resolves ndarray coverage through the same delegations.
        internal static readonly Dictionary<string, string> NdarrayAliases = new()
        {
            ["amax"] = "ndarray.max",           // alias of ndarray.max (np.amax == np.max)
            ["amin"] = "ndarray.min",
            ["conjugate"] = "ndarray.conj",     // same method, NumPy exposes both names
            // partition/argpartition ARE NumPy methods, but whole-output bytes between kth anchors
            // are introselect-implementation-specific on BOTH sides — the np.* corpus pins the
            // DERIVED kth-values instead, and the instance spelling delegates to that same kernel.
            ["argpartition"] = "argpartition",
            ["partition"] = "partition",
            // NumSharp-only instance conveniences (no ndarray.<name> in NumPy) delegating to the
            // fuzzed np.* op of the same name:
            ["array_equal"] = "array_equal",
            ["convolve"] = "convolve",
            ["correlate"] = "correlate",
            ["delete"] = "delete",
            ["dstack"] = "dstack",
            ["hstack"] = "hstack",
            ["vstack"] = "vstack",
            ["matrix_power"] = "matrix_power",
            ["negative"] = "negative",
            ["positive"] = "positive",
            ["roll"] = "roll",
            ["unique"] = "unique",
        };

        // Members whose stronger gate lives elsewhere, or which the operand corpus structurally
        // cannot express (IO, Python-protocol plumbing, device placement).
        private static readonly HashSet<string> NdarraySiblingOwned = new()
        {
            // IO / host-object conversions: text bytes and nested CLR lists have their own
            // byte-exact suites (PrintfFormatter/tofile; ToJaggedArray/tolist tests).
            "tofile", "tolist",
            // flags mutation is gated by the dedicated flags oracle (#5, FlagsOracleTests).
            "setflags",
            // setfield writes THROUGH a dtype window — gated with getfield by the
            // NDArray.getfield/setfield suite (the read half IS corpus-gated: ndarray.getfield).
            "setfield",
            // Array-API device placement: no value bytes (np.device conformance suite).
            "to_device",
            // Python protocol: indexing rides the advanced-indexing oracle (#2); iteration order
            // rides the iter tier (nditer/flatiter semantics); containment/hash are container
            // tests (NDArray.Container).
            "__getitem__", "__setitem__", "__iter__", "__contains__", "__hash__",
        };

        // NumSharp-only members with no NumPy 2.4.2 counterpart at all (itemset was REMOVED in
        // NumPy 2.0; negate/reshape_unsafe are C#-side conveniences).
        private static readonly HashSet<string> NdarrayCompatibilityOnly = new()
        {
            "itemset", "negate", "reshape_unsafe",
        };

        // Property row (plan §D4): NumPy-named properties NOT carried as ndarray.<prop> corpus
        // keys, each with a sibling gate; NumSharp-only properties are compatibility surface.
        private static readonly HashSet<string> NdarrayPropSiblingOwned = new()
        {
            "base",     // view lineage: layout-parity oracle (#6) + view/property validation suite
            "data",     // memoryview bridge: NDArray.data/MemoryView suite
            "device",   // Array-API conformance suite
            "dtype",    // the DTypes suite + every corpus case's expected-dtype comparison
            "flags",    // the flags oracle (#5)
            "shape",    // asserted by EVERY corpus case's result-shape comparison
        };

        private static readonly HashSet<string> NdarrayPropCompatibilityOnly = new()
        {
            "dtypesize", "typecode", "order", "flatiter",
        };

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void NdarraySurface_IsCoveredOrExplicitlyClassified()
        {
            var corpusOps = CorpusOps();
            var failures = new List<string>();

            // Methods. The C#-infrastructure half of NDArray (GetAtIndex/SetData/Clone/ToString/…)
            // is uniformly PascalCase while every NumPy-named member is lowercase (NumPy's own
            // convention; T/mT are PROPERTIES, handled below) — so an uppercase first letter IS
            // the infra classification, and a future NumPy-named member (always lowercase) can
            // never hide behind it.
            var methods = Surface(typeof(NDArray), BindingFlags.Instance | BindingFlags.Static);
            foreach (string name in methods)
            {
                if (char.IsUpper(name[0]))
                    continue;                                   // C# infrastructure by convention
                if (corpusOps.Contains("ndarray." + name))
                    continue;                                   // direct instance coverage
                if (NdarrayAliases.TryGetValue(name, out string target))
                {
                    if (!corpusOps.Contains(target))
                        failures.Add($"ndarray.{name}: alias target corpus op '{target}' is absent");
                    continue;
                }
                if (NdarraySiblingOwned.Contains(name) || NdarrayCompatibilityOnly.Contains(name))
                    continue;
                failures.Add($"ndarray.{name}: unclassified instance surface");
            }

            // Properties (the D4 row): reads are ordinary values, so NumPy-named ones must be
            // corpus keys (ndarray.T/mT/real/imag/flat/nbytes/itemsize/ndim/size/strides) or
            // explicitly sibling-owned; uppercase C#-infra properties (Shape/Unsafe/…) are exempt
            // EXCEPT T, which is NumPy's transpose property and must be corpus-covered.
            foreach (string name in Properties(typeof(NDArray), BindingFlags.Instance | BindingFlags.Static))
            {
                if (corpusOps.Contains("ndarray." + name))
                    continue;
                if (name != "T" && char.IsUpper(name[0]))
                    continue;
                if (NdarrayPropSiblingOwned.Contains(name) || NdarrayPropCompatibilityOnly.Contains(name))
                    continue;
                failures.Add($"ndarray.{name} (property): unclassified instance surface");
            }

            // Classification self-retirement (the A3 rule): a stale entry fails instead of rotting.
            foreach (string name in NdarrayAliases.Keys
                         .Concat(NdarraySiblingOwned).Concat(NdarrayCompatibilityOnly))
            {
                if (!methods.Contains(name))
                    failures.Add($"stale ndarray classification: {name}");
                if (corpusOps.Contains("ndarray." + name))
                    failures.Add($"stale ndarray classification: {name} now has a direct instance corpus op");
            }

            Console.WriteLine($"[NdarraySurface] methods={methods.Length}, " +
                              $"instance_ops={corpusOps.Count(o => o.StartsWith("ndarray.", StringComparison.Ordinal))}, " +
                              $"aliases={NdarrayAliases.Count}, sibling={NdarraySiblingOwned.Count}, " +
                              $"compat={NdarrayCompatibilityOnly.Count}");
            if (failures.Count > 0)
                Assert.Fail($"{failures.Count} ndarray surface classification failures:\n  " +
                            string.Join("\n  ", failures));
        }

        // ================= np.emath (coverage plan §A2/E5) =====================================

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void EmathSurface_IsFullyCorpusCovered()
        {
            // The whole scimath module is promoted into the differential corpus (emath.jsonl) —
            // no sibling/alias classifications at all, so ANY new emath member must gain corpus
            // cases before this gate passes.
            var corpusOps = CorpusOps();
            var missing = Surface(typeof(EmathModule), BindingFlags.Instance)
                .Where(name => !corpusOps.Contains("emath." + name))
                .ToArray();
            Assert.AreEqual(0, missing.Length,
                "np.emath members without an emath.* corpus op: " + string.Join(", ", missing));
        }

        // ================= np.ma (coverage plan §A2) ===========================================

        // NumPy-side aliases whose canonical spelling carries the ma corpus cases. internal: the
        // leak gate (LeakSurfaceCoverageTests) resolves np.ma coverage through the same aliases.
        internal static readonly Dictionary<string, string> MaAliases = new()
        {
            ["alltrue"] = "ma.all",
            ["sometrue"] = "ma.any",
            ["amax"] = "ma.max",
            ["amin"] = "ma.min",
            ["anomalies"] = "ma.anom",
            ["product"] = "ma.prod",
            ["round"] = "ma.around",
            ["round_"] = "ma.around",
            ["row_stack"] = "ma.vstack",
            ["innerproduct"] = "ma.inner",
            ["outerproduct"] = "ma.outer",
            ["asanyarray"] = "ma.asarray",
            // The returned=True twin of ma.average — same weighted-average value path, plus the
            // sum-of-weights slot pinned by NDMaskedArrayTests.
            ["average_returned"] = "ma.average",
        };

        // Gated by the NDMaskedArrayTests suite + the ledger in docs/MA_ORACLE_DESIGN.md: creation
        // helpers returning plain/unmasked arrays, callable-taking wrappers, fill-value plumbing,
        // mask-structure helpers with no serializable operand form, and predicates.
        private static readonly HashSet<string> MaSiblingOwned = new()
        {
            "apply_along_axis", "apply_over_axes",              // C# callable — not serializable
            "arange", "empty", "empty_like", "ones", "ones_like", "zeros", "zeros_like",
            "identity", "indices", "frombuffer", "fromfunction", // creation (plain np twins fuzzed)
            "clump_masked", "clump_unmasked", "column_stack", "common_fill_value",
            "compress_nd", "compress_rowcols", "convolve", "corrcoef", "correlate", "cov",
            "default_fill_value", "diagflat", "dstack",
            "flatnotmasked_contiguous", "flatnotmasked_edges",
            "getmask",                                          // nomask (null) has no byte form; getmaskarray is corpus-gated
            "harden_mask", "hsplit", "ids", "isMA", "isMaskedArray", "is_mask", "isarray",
            "masked_object", "maximum_fill_value", "minimum_fill_value", "moveaxis",
            "ndenumerate", "ndim", "nonzero", "notmasked_contiguous", "notmasked_edges",
            "polyfit", "putmask", "resize", "set_fill_value", "shape", "shrink_mask", "size",
            "soften_mask",
        };

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MaSurface_IsCoveredOrExplicitlyClassified()
        {
            var corpusOps = CorpusOps();
            var failures = new List<string>();
            var methods = Surface(typeof(MaskedArrayModule), BindingFlags.Instance | BindingFlags.Static);
            foreach (string name in methods)
            {
                if (corpusOps.Contains("ma." + name))
                    continue;
                if (MaAliases.TryGetValue(name, out string target))
                {
                    if (!corpusOps.Contains(target))
                        failures.Add($"np.ma.{name}: alias target corpus op '{target}' is absent");
                    continue;
                }
                if (MaSiblingOwned.Contains(name))
                    continue;
                failures.Add($"np.ma.{name}: unclassified surface");
            }

            // Self-retirement: an entry for a member that no longer exists — or that gained a
            // direct corpus op — must fail so the classification tracks reality.
            foreach (string name in MaAliases.Keys.Concat(MaSiblingOwned))
            {
                if (!methods.Contains(name))
                    failures.Add($"stale np.ma classification: {name}");
                if (corpusOps.Contains("ma." + name))
                    failures.Add($"stale np.ma classification: {name} now has a direct corpus op");
            }

            Console.WriteLine($"[MaSurface] methods={methods.Length}, " +
                              $"ma_ops={corpusOps.Count(o => o.StartsWith("ma.", StringComparison.Ordinal))}, " +
                              $"aliases={MaAliases.Count}, sibling={MaSiblingOwned.Count}");
            if (failures.Count > 0)
                Assert.Fail($"{failures.Count} np.ma surface classification failures:\n  " +
                            string.Join("\n  ", failures));
        }

        // ================= np.dtypes (coverage plan §A2) =======================================

        // The 29 DType-class accessor properties, all gated by the DTypes suites
        // (test/NumSharp.Tests/DTypes/*) and the dtype_text tier's np.dtype(string) coverage —
        // they return the registry's class singletons, not array values, so the operand corpus
        // has nothing to bit-compare. A NEW property must be added here (with its gate) or fail.
        private static readonly HashSet<string> DtypesKnownProperties = new()
        {
            "BoolDType", "ByteDType", "CLongDoubleDType", "CharDType", "Complex128DType",
            "DateTime64DType", "DecimalDType", "Float16DType", "Float32DType", "Float64DType",
            "Int16DType", "Int32DType", "Int64DType", "Int8DType", "IntDType", "LongDType",
            "LongDoubleDType", "LongLongDType", "ShortDType", "TimeDelta64DType", "UByteDType",
            "UInt16DType", "UInt32DType", "UInt64DType", "UInt8DType", "UIntDType", "ULongDType",
            "ULongLongDType", "UShortDType",
        };

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void DtypesSurface_IsExplicitlyClassified()
        {
            var props = Properties(typeof(np.dtypes), BindingFlags.Static);
            var failures = new List<string>();
            foreach (string name in props)
                if (!DtypesKnownProperties.Contains(name))
                    failures.Add($"np.dtypes.{name}: unclassified DType-class property (add its gate + classification)");
            foreach (string name in DtypesKnownProperties)
                if (!props.Contains(name))
                    failures.Add($"stale np.dtypes classification: {name}");
            Assert.AreEqual(0, failures.Count,
                string.Join("\n  ", failures));
        }
    }
}
