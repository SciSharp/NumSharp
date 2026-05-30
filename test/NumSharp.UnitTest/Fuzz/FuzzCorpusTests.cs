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

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Comparison() => RunCorpus("comparison.jsonl");

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Unary() => RunCorpus("unary.jsonl");

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Reduce() => RunCorpus("reduce.jsonl");

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Where() => RunCorpus("where.jsonl");

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Place() => RunCorpus("place.jsonl");

        // T8 linear algebra: matmul / dot / outer across the gufunc shape space (2-D, 1-D promotion,
        // batched/broadcast stacks), 6 dtypes, and C/F operand layouts.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Matmul() => RunCorpus("matmul.jsonl");

        // T9 bitwise & shift: bitwise_and/or/xor (& | ^), invert (~), left/right_shift across
        // integer + bool dtypes, pairwise layouts, and shift-count edges that straddle the bit width.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Bitwise() => RunCorpus("bitwise.jsonl");

        // W3 unary stragglers: exp2/expm1/log2/log10/log1p/sinh/cosh/tanh/arcsin/arccos/arctan/
        // deg2rad/rad2deg/positive across all 13 dtypes and all 25 single-array layouts.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void UnaryExtra() => RunCorpus("unary_extra.jsonl");

        // W4 NaN-aware reductions (T10): nansum/nanprod/nanmax/nanmin/nanmean/nanstd/nanvar/
        // nanmedian over NaN-laced float operands — must IGNORE NaN per NumPy contract.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void NanReduce() => RunCorpus("nanreduce.jsonl");

        // Seeded random fuzzer corpus (offline-generated; reproducible from its seed).
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void FuzzRandom() => RunCorpus("random_smoke.jsonl");

        // Shrunk reproductions of divergences found by the nightly soak. Empty until the soak finds one.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void FuzzRegression()
        {
            var dir = System.IO.Path.Combine(AppContext.BaseDirectory, "Fuzz", "corpus", "regressions");
            if (!System.IO.Directory.Exists(dir))
                return;
            foreach (var f in System.IO.Directory.GetFiles(dir, "*.jsonl"))
                RunCorpus(System.IO.Path.Combine("regressions", System.IO.Path.GetFileName(f)));
        }

        // floor_divide / mod are bit-exact with NumPy as of Phase 1 F1 (integer ÷0 -> 0, float //0 ->
        // ±inf/nan, signed floor toward -inf, MIN/-1 wrap, mixed-precision promotion). The only
        // remaining divergence in this corpus is complex `power` (~ULP + inf/NaN edge), excused by the
        // registry's complex-binary branch pending Phase 1 F5. CI-gated so a floor_divide/mod
        // regression fails immediately.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Binary_DivModPower() => RunCorpus("binary_divmod_power.jsonl");

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
                        var reason = MisalignedRegistry.Classify(c, DivergenceKind.Shape, null, null, tc, empty);
                        if (reason != null) Bump(documented, reason);
                        else failures.Add($"{c.Id} [{c.Layout}]: result shape [{string.Join(",", result.Shape.dimensions)}] " +
                                          $"!= NumPy [{string.Join(",", c.Expected.Shape)}]");
                        continue;
                    }

                    var actual = FuzzCorpus.ResultBytes(result);
                    var expected = FuzzCorpus.FromHex(c.Expected.Buffer);

                    var diffs = BitDiff.Compare(expected, actual, tc);
                    if (diffs.Count > 0)
                    {
                        var reason = MisalignedRegistry.Classify(c, DivergenceKind.Value, expected, actual, tc, diffs);
                        if (reason != null)
                        {
                            Bump(documented, reason);
                        }
                        else
                        {
                            var shrunk = Shrinker.ShrinkElementwise(c, diffs[0].Index);
                            failures.Add($"{c.Id} [{c.Layout}]: " +
                                string.Join(", ", diffs.Take(3).Select(d => $"@{d.Index} exp {d.Expected} act {d.Actual}")) +
                                (diffs.Count > 3 ? $" (+{diffs.Count - 3} more)" : "") +
                                (shrunk != null ? $"\n      minimal repro: {shrunk}" : ""));
                        }
                    }
                }
                catch (Exception e)
                {
                    var reason = MisalignedRegistry.Classify(c, DivergenceKind.Threw, null, null, default, empty);
                    if (reason != null) Bump(documented, reason);
                    else failures.Add($"{c.Id} [{c.Layout}]: THREW {e.GetType().Name}: {e.Message}");
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
