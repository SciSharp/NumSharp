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
        private static readonly Dictionary<string, string> EquivalentAliases = new()
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
            // datetime_data reads the (unit, count) parameter off a datetime64/timedelta64 DESCRIPTOR — a dtype-level
            // accessor with no array operand or result bytes (the datetime classes have no storage yet, Stage A of
            // docs/plans/dtype-system.md). Gated by test/NumSharp.Tests/DTypes/*, probed against NumPy 2.4.2.
            "datetime_data",
            "evaluate", "finfo", "flat",
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
    }
}
