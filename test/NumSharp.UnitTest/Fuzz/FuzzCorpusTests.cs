using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Fuzz;

namespace NumSharp.UnitTest.Fuzz
{
    /// <summary>
    ///     Differential matrix: replay every committed NumPy oracle case through NumSharp and
    ///     bit-compare. One test per corpus file; a failure lists every divergent cell so the
    ///     whole matrix is visible at once (not first-failure-wins). Runs in CI under FuzzMatrix.
    /// </summary>
    [TestClass]
    public class FuzzCorpusTests
    {
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Astype_Smoke() => RunCorpus("astype_smoke.jsonl");

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Astype_Full() => RunCorpus("astype_full.jsonl");

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Binary_Arith() => RunCorpus("binary_arith.jsonl");

        // KNOWN-FAILING bug reproduction (excluded from CI via [OpenBugs]; remove the tag when fixed).
        // floor_divide / mod / power diverge from NumPy: integer ÷0 and mod-0 throw or return garbage
        // instead of 0; float //0 yields NaN instead of ±inf; mixed-precision mod; complex power.
        [TestMethod]
        [TestCategory("OpenBugs")]
        public void Binary_DivModPower_KnownDivergences() => RunCorpus("binary_divmod_power.jsonl");

        private static void RunCorpus(string file)
        {
            var cases = FuzzCorpus.Load(file);
            Assert.IsTrue(cases.Count > 0, $"corpus '{file}' has no cases (was it generated/copied?)");

            var failures = new List<string>();
            var documented = new Dictionary<string, int>();   // intended divergence reason -> count
            var empty = System.Array.Empty<BitDiff.Diff>();

            foreach (var c in cases)
            {
                try
                {
                    var operands = new NumSharp.NDArray[c.Operands.Length];
                    for (int i = 0; i < operands.Length; i++)
                        operands[i] = FuzzCorpus.Reconstruct(c.Operands[i]);

                    var result = OpRegistry.Apply(c.Op, c.Params, operands);
                    var tc = FuzzCorpus.DtypeToTC(c.Expected.Dtype);

                    // NEP50: result dtype must match NumPy exactly (the headline promotion failure).
                    if (result.typecode != tc)
                    {
                        var reason = MisalignedRegistry.Classify(c, DivergenceKind.Dtype, null, null, tc, empty);
                        if (reason != null) Bump(documented, reason);
                        else failures.Add($"{c.Id} [{c.Layout}]: result dtype {result.typecode} != NumPy {c.Expected.Dtype}");
                        continue;
                    }
                    // Broadcasting: result shape must match NumPy.
                    if (!ShapeEquals(result.Shape.dimensions, c.Expected.Shape))
                    {
                        failures.Add($"{c.Id} [{c.Layout}]: result shape [{string.Join(",", result.Shape.dimensions)}] " +
                                     $"!= NumPy [{string.Join(",", c.Expected.Shape)}]");
                        continue;
                    }

                    var actual = FuzzCorpus.ResultBytes(result);
                    var expected = FuzzCorpus.FromHex(c.Expected.Buffer);

                    var diffs = BitDiff.Compare(expected, actual, tc);
                    if (diffs.Count > 0)
                    {
                        var reason = MisalignedRegistry.Classify(c, DivergenceKind.Value, expected, actual, tc, diffs);
                        if (reason != null) Bump(documented, reason);
                        else failures.Add($"{c.Id} [{c.Layout}]: " +
                            string.Join(", ", diffs.Take(3).Select(d => $"@{d.Index} exp {d.Expected} act {d.Actual}")) +
                            (diffs.Count > 3 ? $" (+{diffs.Count - 3} more)" : ""));
                    }
                }
                catch (Exception e)
                {
                    failures.Add($"{c.Id} [{c.Layout}]: THREW {e.GetType().Name}: {e.Message}");
                }
            }

            // Never silent: surface documented (intended) divergences even when the test passes.
            if (documented.Count > 0)
                Console.WriteLine($"[{file}] documented Misaligned divergences excused: " +
                                  string.Join("; ", documented.Select(kv => $"{kv.Value}x {kv.Key}")));

            if (failures.Count > 0)
                Assert.Fail($"{failures.Count}/{cases.Count} cases diverged from NumPy (unexpected):\n  " +
                            string.Join("\n  ", failures.Take(60)));
        }

        private static void Bump(Dictionary<string, int> d, string key) => d[key] = d.TryGetValue(key, out var n) ? n + 1 : 1;

        private static bool ShapeEquals(long[] actual, long[] expected)
        {
            if (actual.Length != expected.Length)
                return false;
            for (int i = 0; i < actual.Length; i++)
                if (actual[i] != expected[i])
                    return false;
            return true;
        }
    }
}
