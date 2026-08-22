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
            ["radians"] = "deg2rad",
            ["result_type"] = "result_type_arrays",
            ["true_divide"] = "divide",
        };

        // These have a stronger or more appropriate gate elsewhere, or have no deterministic
        // value bytes to compare in an operand/result corpus.
        private static readonly HashSet<string> SiblingOwned = new()
        {
            "array2string",
            // bmat is pure block-assembly (concatenation) over the already-fuzzed concatenate +
            // asmatrix — its nested-block ([[A,B],[C,D]]) and string+dict inputs have no single-operand
            // corpus representation (the same reason block needs a bespoke multi-operand oracle path);
            // gated by the dedicated np.bmat.Tests.cs suite verified against NumPy 2.4.2.
            "bmat",
            "evaluate", "finfo", "flat",
            "format_float_positional", "format_float_scientific",
            "get_printoptions", "iinfo", "load", "load_npy", "load_npz",
            "nditer_chunks", "printoptions", "save", "savez",
            "savez_compressed", "set_printoptions",
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
