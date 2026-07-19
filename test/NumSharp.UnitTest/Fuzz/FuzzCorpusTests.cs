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

        // Char dtype (NumSharp-only, bit-identical to uint16) is WOVEN into the applicable tiers:
        // each Char op is generated through the uint16 NumPy proxy and relabelled uint16->char
        // (gen_oracle.char_tier), appended into its native tier file (binary_arith/divmod_power/
        // comparison/unary/unary_extra/bitwise/reduce/scan/stat/manip/sort/tail/astype_full/
        // where/logic/matmul/rounding/copyto — 18 tier files). So the existing per-tier gates
        // assert NumSharp's Char ≡ uint16. The verified Char bugs (promote(Char,Byte)->Byte,
        // reciprocal/power/invert, dot 1-D) are carved out of the green corpus and reproduced in
        // OpenBugs.Char.cs (OpenBugsCharTests) / OpenBugs.FuzzGaps.cs under [OpenBugs].

        // W3 unary stragglers: exp2/expm1/log2/log10/log1p/sinh/cosh/tanh/arcsin/arccos/arctan/
        // deg2rad/rad2deg/positive across all 13 dtypes and all 26 single-array layouts.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void UnaryExtra() => RunCorpus("unary_extra.jsonl");

        // W4 NaN-aware reductions (T10): nansum/nanprod/nanmax/nanmin/nanmean/nanstd/nanvar/
        // nanmedian over NaN-laced float operands — must IGNORE NaN per NumPy contract.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void NanReduce() => RunCorpus("nanreduce.jsonl");

        // W5 cumulative (T11): cumsum/cumprod (axis None + per-axis, NEP50 accumulator) and diff
        // (n=1,2; axis 0/last; output shrinks by n) across int/uint/float/complex dtypes.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Scan() => RunCorpus("scan.jsonl");

        // W6 statistics (T12): median/average/ptp (axis+keepdims), count_nonzero, percentile/
        // quantile (q in {0,25,50,75,100}/{0,.25,.5,.75,1}, axis None/0/last), clip (a,min,max).
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Stat() => RunCorpus("stat.jsonl");

        // W7 logic (T13): isnan/isinf/isfinite (unary->bool), maximum/minimum (NaN-propagating),
        // fmax/fmin (NaN-ignoring), isclose (binary->bool) — NaN/inf-laced operands.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Logic() => RunCorpus("logic.jsonl");

        // W8 multi-output (T15): np.modf -> (fractional, integral), each output bit-compared,
        // with C-standard signed-zero/inf edges from the float pools.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Modf() => RunCorpus("modf.jsonl");

        // W9 manipulation (T7): ravel/transpose/expand_dims/squeeze/roll/repeat/tile/reshape/
        // swapaxes/moveaxis/delete/atleast_* (single-array, all layouts) + concatenate/stack/
        // hstack/vstack/dstack (two-array) + pad (constant/edge/reflect/wrap).
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Manip() => RunCorpus("manip.jsonl");

        // W10 sorting/searching (T14): argsort (1-D/2-D, axis 0/1/-1, distinct values),
        // searchsorted (side left/right), nonzero (1-D int64 indices).
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Sort() => RunCorpus("sort.jsonl");

        // Group A: np.round_/around with decimals in {0,1,2,-1} over every layout (banker's rounding;
        // int + negative-decimals genuinely rounds to tens).
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Rounding() => RunCorpus("rounding.jsonl");

        // Group A Batches 4-6: shape (flatten/rollaxis/append/insert), selection (take/compress/
        // extract), math (convolve), multi-output split (one case per output piece).
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void GroupA() => RunCorpus("groupa.jsonl");

        // W13 SIMD-tail boundaries: add/sub/mul/negative/abs/sqrt/sum/prod/max/min over 1-D arrays
        // sized 1..129 straddling the V128/V256/V512 lane counts (7/8/9, 15/16/17, 31/32/33, ...).
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Tail() => RunCorpus("tail.jsonl");

        // W12 parameter sweep: middle + negative axes (-1/-2/-3) for all reductions, ddof=1
        // sample std/var, and order='F' ravel across C/transposed/F-contiguous sources.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Params() => RunCorpus("params.jsonl");

        // W11 operand-relationship flags (section C): input aliasing (a op a, same buffer) and
        // in-place out= (maximum/minimum/clip writing into an input operand).
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Aliasing() => RunCorpus("aliasing.jsonl");

        // W14 error parity: cases where NumPy raises (int**neg, broadcast/core-dim mismatch,
        // bitwise/shift on float, bad reshape, axis-out-of-range, ...) must also throw in NumSharp.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Errors() => RunCorpus("errors.jsonl");

        // W15 copyto: same-dtype OVERLAPPING copies (shift/reverse/strided write-before-read/2-D
        // transpose — NumPy's COPY_IF_OVERLAP) + cross-dtype copyto into contiguous/strided dst and
        // scalar-broadcast src (the cast-into-non-contiguous-dst path astype never exercises).
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Copyto() => RunCorpus("copyto.jsonl");

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

        // Decimal — the ONE NumSharp numeric dtype with no NumPy analog (System.Decimal: 16-byte,
        // base-10). NumPy cannot be the oracle, so these corpora are generated by an INDEPENDENT C#
        // oracle (test/oracle/gen_decimal_oracle.cs) that computes every expected value with naive
        // scalar System.Decimal arithmetic — NO NumSharp kernels. The replay runs the same operand
        // through NumSharp's decimal KERNELS and value-compares (BitDiff tokenizes decimal by
        // canonical value, so benign scale 1.0m≡1.00m doesn't false-fail). A divergence is a real
        // decimal kernel bug (strided/broadcast/reduction/scan iteration or accumulation).
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void DecimalUnary() => RunCorpus("decimal_unary.jsonl");

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void DecimalBinary() => RunCorpus("decimal_binary.jsonl");

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void DecimalReduce() => RunCorpus("decimal_reduce.jsonl");

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void DecimalScan() => RunCorpus("decimal_scan.jsonl");

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void DecimalPower() => RunCorpus("decimal_power.jsonl");

        // var = mean((x-mean)^2) is exact decimal arithmetic; std = sqrt(var) is oracled by an
        // INDEPENDENT Newton decimal sqrt (gen_decimal_oracle.DecSqrt), not NumSharp's DecimalMath.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void DecimalVarStd() => RunCorpus("decimal_varstd.jsonl");

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void DecimalMatmul() => RunCorpus("decimal_matmul.jsonl");

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void DecimalAstype() => RunCorpus("decimal_astype.jsonl");

        // clip (elementwise Max(lo,Min(hi,x))) + the order statistics median/ptp/percentile/quantile
        // (axis=None -> scalar) over the strided/broadcast decimal layouts. Oracle: naive sort +
        // NumPy 'linear' interpolation in exact decimal (gen_decimal_oracle.Quantile/Median).
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void DecimalStat() => RunCorpus("decimal_stat.jsonl");

        // where(cond, a, b) — the 16-byte conditional-copy kernel over contiguous AND strided decimal.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void DecimalWhere() => RunCorpus("decimal_where.jsonl");

        // sort along an axis (1-D/2-D, contiguous + strided) — independent Array.Sort oracle.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void DecimalSort() => RunCorpus("decimal_sort.jsonl");

        // ravel/transpose/reshape — value-preserving reindex; forces the strided decimal
        // materialize/copy path (result compared C-contiguous via ascontiguousarray).
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void DecimalManip() => RunCorpus("decimal_manip.jsonl");

        // B9 (F26): minimum case-count floor per corpus file, ~80 % of the committed count at
        // 2026-07-07 (post G1-G5/G11/G12 regenerations). `Count > 0` alone would let a silently
        // TRUNCATED regeneration (encoding hiccup, generator early-exit, partial copy) pass the
        // gate with a fraction of its cells. Regenerations only grow tiers, so these floors stay
        // conservative; bump them after intentional expansions (WS-DOCS re-checks at D8).
        // Unknown files (e.g. regressions/*.jsonl) keep the floor of 1.
        private static readonly Dictionary<string, int> MinCases = new()
        {
            ["aliasing.jsonl"] = 70,
            ["astype_full.jsonl"] = 4056,
            ["astype_smoke.jsonl"] = 249,
            ["binary_arith.jsonl"] = 1296,
            ["binary_divmod_power.jsonl"] = 793,
            ["bitwise.jsonl"] = 590,
            ["comparison.jsonl"] = 1857,
            ["copyto.jsonl"] = 91,
            ["decimal_astype.jsonl"] = 17,
            ["decimal_binary.jsonl"] = 78,
            ["decimal_manip.jsonl"] = 28,
            ["decimal_matmul.jsonl"] = 3,
            ["decimal_power.jsonl"] = 16,
            ["decimal_reduce.jsonl"] = 48,
            ["decimal_scan.jsonl"] = 36,
            ["decimal_sort.jsonl"] = 5,
            ["decimal_stat.jsonl"] = 136,
            ["decimal_unary.jsonl"] = 72,
            ["decimal_varstd.jsonl"] = 17,
            ["decimal_where.jsonl"] = 3,
            ["errors.jsonl"] = 8,
            ["groupa.jsonl"] = 82,
            ["logic.jsonl"] = 1775,
            ["manip.jsonl"] = 4244,
            ["matmul.jsonl"] = 769,
            ["modf.jsonl"] = 51,
            ["nanreduce.jsonl"] = 6692,
            ["params.jsonl"] = 966,
            ["place.jsonl"] = 12,
            ["random_smoke.jsonl"] = 1600,
            ["reduce.jsonl"] = 9004,
            ["rounding.jsonl"] = 665,
            ["scan.jsonl"] = 907,
            ["sort.jsonl"] = 401,
            ["stat.jsonl"] = 3412,
            ["tail.jsonl"] = 1872,
            ["unary.jsonl"] = 4222,
            ["unary_extra.jsonl"] = 4305,
            ["where.jsonl"] = 75,
        };

        private static void RunCorpus(string file)
        {
            var cases = FuzzCorpus.Load(file);
            int floor = MinCases.TryGetValue(file, out var f) ? f : 1;
            Assert.IsTrue(cases.Count >= floor,
                $"corpus '{file}' has {cases.Count} cases, below the committed floor of {floor} " +
                "(truncated/partial regeneration? see MinCases in FuzzCorpusTests)");

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

                    // W11 input aliasing: pass the single stored operand as BOTH binary arguments
                    // through the SAME reference (true a-op-a aliasing, not two equal copies).
                    if (c.Alias && operands.Length == 1)
                        operands = new[] { operands[0], operands[0] };

                    // W14 error parity: NumPy raised here; NumSharp must throw too (not silently
                    // produce a result). A throw of ANY kind passes; a normal return is the divergence.
                    if (c.Expects_Throw)
                    {
                        bool threw = false;
                        try { _ = OpRegistry.Apply(c.Op, c.Params, operands); }
                        catch { threw = true; }
                        if (!threw)
                        {
                            var reason = MisalignedRegistry.Classify(c, DivergenceKind.Value, null, null, default, empty);
                            if (reason != null) Bump(documented, reason);
                            else failures.Add($"{c.Id} [{c.Layout}]: NumPy raises but NumSharp produced a result (error-parity gap)");
                        }
                        continue;
                    }

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
