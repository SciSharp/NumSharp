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
    ///     Permanent coverage receipt for the public NumPy callables owned by a production source
    ///     file changed between master (61506de1) and journey3 (aaa41ef2), plus the current journey3
    ///     completion work. File ownership is intentionally conservative: every callable in a
    ///     touched file is in scope even when only a neighbouring overload changed.
    /// </summary>
    [TestClass]
    public class Journey3TouchedOracleCoverageTests
    {
        private static readonly string[] TouchedNp =
        {
            "acosh", "all", "angle", "any", "arange", "arccosh", "arcsinh", "arctanh",
            "argpartition", "array", "array_split", "asanyarray", "asarray", "ascontiguousarray", "asfortranarray", "asinh",
            "atanh", "bincount", "block", "broadcast_arrays", "broadcast_to", "choose", "clip", "concatenate",
            "conj", "conjugate", "copyto", "corrcoef", "correlate", "cov", "cross", "cumprod",
            "cumsum", "delete", "diag", "diag_indices", "diag_indices_from", "diagflat", "digitize", "dot",
            "einsum", "einsum_path", "empty", "empty_like", "expand_dims", "eye", "fill_diagonal", "frombuffer",
            "fromfile", "fromstring", "full", "full_like", "identity", "imag", "inner", "insert",
            "intersect1d", "isclose", "isfinite", "isfortran", "isin", "isinf", "isnan", "isscalar",
            "iterable", "ix_", "kron", "lexsort", "linspace", "loadtxt", "mask_indices", "matmul",
            "matvec", "meshgrid", "mintypecode", "nanargmax", "nanargmin", "nanmean", "nanstd", "nanvar",
            "nditer", "nested_iters", "ones", "ones_like", "outer", "partition", "place", "poly",
            "polyadd", "polyder", "polydiv", "polyfit", "polyint", "polymul", "polysub", "polyval",
            "ptp", "put", "ravel_multi_index", "real", "require", "roots", "savetxt", "searchsorted",
            "select", "setdiff1d", "setxor1d", "sort_complex", "split", "squeeze", "take", "take_along_axis",
            "tensordot", "trace", "tri", "tril", "tril_indices", "tril_indices_from", "triu", "triu_indices",
            "triu_indices_from", "union1d", "unique", "unique_all", "unique_counts", "unique_inverse", "unique_values", "vander",
            "vdot", "vecdot", "vecmat", "zeros", "zeros_like",
        };

        private static readonly string[] TouchedLinalg =
        {
            "cholesky", "cond", "cross", "det", "diagonal", "eig", "eigh", "eigvals",
            "eigvalsh", "inv", "lstsq", "matmul", "matrix_norm", "matrix_power", "matrix_rank", "matrix_transpose",
            "multi_dot", "norm", "outer", "pinv", "qr", "slogdet", "solve", "svd",
            "svdvals", "tensordot", "tensorinv", "tensorsolve", "trace", "vecdot", "vector_norm",
        };

        private static readonly string[] TouchedFft =
        {
            "fft", "fft2", "fftfreq", "fftn", "fftshift", "hfft", "ifft", "ifft2",
            "ifftn", "ifftshift", "ihfft", "irfft", "irfft2", "irfftn", "rfft", "rfft2",
            "rfftfreq", "rfftn",
        };

        private static readonly string[] TouchedRandom =
        {
            "get_state", "seed", "set_state", "shuffle",
        };

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void EveryJourney3TouchedCallable_HasDirectOracleCases()
        {
            var counts = new Dictionary<string, int>(StringComparer.Ordinal);
            var randomCounts = new Dictionary<string, int>(StringComparer.Ordinal);
            string directory = Path.GetDirectoryName(FuzzCorpus.CorpusPath("unused"));
            foreach (string path in Directory.EnumerateFiles(directory, "*.jsonl"))
            {
                if (path.EndsWith(".host.jsonl", StringComparison.Ordinal))
                    continue;
                foreach (var c in FuzzCorpus.Load(Path.GetFileName(path)))
                {
                    if (!string.IsNullOrEmpty(c.Op))
                        counts[c.Op] = counts.TryGetValue(c.Op, out int old) ? old + 1 : 1;
                    if (c.Op == "rnd" && c.Params != null && c.Params.TryGetValue("dist", out var dist))
                    {
                        string name = dist.GetString();
                        randomCounts[name] = randomCounts.TryGetValue(name, out int rold) ? rold + 1 : 1;
                    }
                }
            }

            var failures = new List<string>();
            CheckSurface("np", TouchedNp, Surface(typeof(np), BindingFlags.Static), counts, null, failures);
            CheckSurface("np.linalg", TouchedLinalg, Surface(typeof(np.linalg), BindingFlags.Static), counts, null, failures);
            CheckSurface("np.fft", TouchedFft, Surface(typeof(FourierModule), BindingFlags.Instance), counts, null, failures);
            CheckSurface("np.random", TouchedRandom, Surface(typeof(NumPyRandom), BindingFlags.Instance), counts, randomCounts, failures);

            int total = TouchedNp.Length + TouchedLinalg.Length + TouchedFft.Length + TouchedRandom.Length;
            Assert.AreEqual(186, total, "Journey3 coverage receipt changed size; update its audited inventory.");
            Console.WriteLine($"[Journey3Oracle] {total}/{total} touched callables have direct committed cases: " +
                              $"np={TouchedNp.Length}, linalg={TouchedLinalg.Length}, fft={TouchedFft.Length}, " +
                              $"random={TouchedRandom.Length}");
            if (failures.Count > 0)
                Assert.Fail($"{failures.Count} journey3 touched-callable coverage failures:\n  " +
                            string.Join("\n  ", failures));
        }

        private static void CheckSurface(
            string label, IEnumerable<string> expected, HashSet<string> actual,
            IReadOnlyDictionary<string, int> corpusCounts, IReadOnlyDictionary<string, int> randomCounts,
            ICollection<string> failures)
        {
            foreach (string name in expected)
            {
                if (!actual.Contains(name))
                    failures.Add($"{label}.{name}: no longer exists on the public surface");
                int count = corpusCounts.TryGetValue(name, out int direct) ? direct : 0;
                if (randomCounts != null && randomCounts.TryGetValue(name, out int stream))
                    count += stream;
                if (count == 0)
                    failures.Add($"{label}.{name}: no direct corpus op/distribution cases");
            }
        }

        private static HashSet<string> Surface(Type type, BindingFlags flags)
            => type.GetMethods(flags | BindingFlags.Public | BindingFlags.DeclaredOnly)
                .Where(method => !method.IsSpecialName)
                .Select(method => method.Name)
                .ToHashSet(StringComparer.Ordinal);
    }
}
