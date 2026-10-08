using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The MASKED-ARRAY half of the differential matrix. <see cref="np.ma"/> is a separate TYPE
    ///     (data + optional bool mask), so its operands and results cannot ride the single-buffer
    ///     NDArray corpus — they get their own <c>ma_*.jsonl</c> tiers, replayed here.
    ///
    ///     <para><b>Contract.</b> A masked operand is (data descriptor + optional <c>mask</c> hex);
    ///     a masked result is compared as <c>filled(0)</c> data bytes (BitDiff, NaN-tokenized) PLUS
    ///     <c>getmaskarray</c> bytes (bit-exact bool). Masked slots collapse to 0 on both sides, so the
    ///     underlying masked-slot data is a wildcard (arbitrary for unique/set-ops, input-restored for
    ///     ufuncs), while every unmasked value AND the mask position are gated. This is exactly the
    ///     pass-13 "M-token + hex" scheme (docs/MA_MODULE_AUDIT.md) now committed and CI-gated.</para>
    ///
    ///     <para><b>Excuses.</b> np.ma delegates to the identical <c>np.*</c> op on the data, so every
    ///     value divergence is routed through the SHARED <see cref="MisalignedRegistry"/> with the
    ///     <c>ma.</c> op-name prefix STRIPPED — the same ledger that excuses e.g. the complex-unary
    ///     ≤3-ULP envelope for the plain ops. ma.mean / ma.var / ma.std are numpy.ma's statements since
    ///     2026-10-05 (plan api-parameter-parity A6) and no excuse of this file or of the ledger's blankets
    ///     takes them. A MASK divergence is never excused: NumSharp claims bit-exact masks, so a wrong
    ///     mask is always red.</para>
    /// </summary>
    public partial class FuzzCorpusTests
    {
        // ---- tiers -------------------------------------------------------------------------------

        // Unary ufunc family (domain-free + domained) + complex accessors. Host-libm pinned like the
        // plain Unary tier: the transcendental/complex kernels are win-amd64 ucrtbase-dependent, so an
        // off-Windows divergence is a platform artifact (Inconclusive), not an ma defect.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MaUnary() => RunHostLibmMaCorpus("ma_unary.jsonl");

        // Binary ufunc family + domained-binary + comparison/logical/bitwise/shift + maximum/minimum +
        // arctan2/hypot/power. Host-libm pinned (power/arctan2/hypot reach libm at float64).
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MaBinary() => RunHostLibmMaCorpus("ma_binary.jsonl");

        // Reductions (sum/prod/mean/min/max/ptp/var/std/all/any/anom/median/average) + count/argmin/
        // argmax(->array). Float sum/prod accumulation order is excused via the shared reduce branch
        // (prefix-stripped); mean/var/std are held to NumPy's bits (plan api-parameter-parity A6) but for
        // the reversed-view axis layout they inherit from np.sum (finding 5); the MASK is bit-exact.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MaReduce() => RunMaCorpus("ma_reduce.jsonl");

        /// <summary>
        ///     ma.mean / ma.var / ma.std / ma.anom held to numpy.ma's bits with NO excuse (plan api-parameter-parity A6):
        ///     every catalog layout × the 13 NumPy dtypes × the mask classes, keepdims, ddof, dtype= loops, the mean= forms,
        ///     special values, sizes past the pairwise block, empty and 0-d inputs, and NumPy's errors (type and text).
        ///     Beyond filled(0) and getmaskarray each case pins the raw data at EVERY slot (masked ones included) and which
        ///     object NumPy returned (the masked singleton, a scalar, an array with nomask or with a mask array).
        /// </summary>
        /// <remarks>
        ///     The cells that inherit np.sum's residuals (findings 5 and 10, X12) are carved by the generator
        ///     (<c>gen_ma_oracle._mv_carved</c>), never excused here; a failure is a real divergence of the port.
        /// </remarks>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MaMeanVar() => RunMaCorpus("ma_meanvar.jsonl");

        // Scans: cumsum/cumprod/diff/ediff1d — per-position mask; portable arithmetic (bit-exact).
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MaScan() => RunMaCorpus("ma_scan.jsonl");

        // Shape manipulation (ravel/reshape/transpose/stack family/diag/…/hsplit->tuple) +
        // compressed/getdata/getmask/getmaskarray/filled(->array). Pure reindex/copy (bit-exact).
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MaManip() => RunMaCorpus("ma_manip.jsonl");

        // Masked constructors: masked_where/masked_<cmp>/masked_values/masked_inside/masked_outside/
        // masked_invalid/fix_invalid/masked_all(_like)/array/masked_array — the mask-CREATION surface.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MaConstruct() => RunMaCorpus("ma_construct.jsonl");

        // Selection: take/choose/compress/clip/where + put/putmask(mutated-operand result).
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MaSelect() => RunMaCorpus("ma_select.jsonl");

        // Sort + set operations: sort/argsort(->array)/unique + intersect1d/union1d/setxor1d/
        // setdiff1d/isin/in1d — the masked-to-END sort key + the presence-rule masked trailing entry.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MaSortSetops() => RunMaCorpus("ma_sortsetops.jsonl");

        /// <summary>
        ///     numpy.ma's arraysetops as NumPy's statements (ma_setops.jsonl, gen_ma_oracle.py <c>setops</c>): unique (with
        ///     return_index / return_inverse), in1d, isin, intersect1d, setxor1d, union1d, setdiff1d over every dtype, value
        ///     pool, mask class, hard masks, plain operands, layouts and mixed dtypes, NumPy's errors included. Held to ZERO
        ///     excuses (valueclass <c>masetops</c>), with the result's form checked, and the raw flag data of in1d / isin.
        /// </summary>
        /// <remarks>
        ///     The cases whose NumPy answer depends on its unstable quicksort's order among equal keys are dropped by the
        ///     generator (it replays a stable copy of NumPy's statements, test/oracle/ma_setops_reference.py), never
        ///     excused here.
        /// </remarks>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MaSetops() => RunMaCorpus("ma_setops.jsonl");

        /// <summary>
        ///     numpy.ma's put as NumPy's statements (ma_put.jsonl, gen_ma_oracle.py <c>put</c>): MaskedArray.put / ma.put
        ///     over every dtype, target layout, mask kind, hard and soft masks, plain targets, index kinds, value kinds and
        ///     modes, NumPy's errors included (its hard-mask branch raises IndexError for every N-D target and ValueError for
        ///     masked values). Held to ZERO excuses (valueclass <c>maput</c>), with the mutated target's form and raw data
        ///     checked and the exception TYPE checked strictly (<see cref="StrictErrorTypes"/>).
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MaPut() => RunMaCorpus("ma_put.jsonl");

        // Extras: small exact products (dot/inner/outer) + trace/vander(->array) + triangular-mask
        // (mask_rows/cols/rowcols, compress_rows/cols) + predicates (is_masked/allclose/allequal->scalar)
        // + mask helpers (make_mask/make_mask_none/mask_or/flatten_mask->array, make_mask_descr->dtype).
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MaExtras() => RunMaCorpus("ma_extras.jsonl");

        // ---- driver ------------------------------------------------------------------------------

        /// <summary>Minimum committed case counts per ma tier — a truncated regeneration fails loudly.</summary>
        private static readonly Dictionary<string, int> MaMinCases = new()
        {
            // ~80% of the committed counts (pass 2, 2026-09-18: edge layouts + params expanded the corpus
            // from 26,586 to 68,860 cases): a truncated/partial regeneration fails loudly.
            ["ma_unary.jsonl"] = 8650,
            ["ma_binary.jsonl"] = 9370,
            ["ma_reduce.jsonl"] = 23020,
            ["ma_scan.jsonl"] = 2910,
            ["ma_manip.jsonl"] = 5880,
            ["ma_construct.jsonl"] = 3690,
            ["ma_select.jsonl"] = 680,
            ["ma_sortsetops.jsonl"] = 790,
            ["ma_extras.jsonl"] = 58,
            // ~80% of the 26,328 committed cases (plan api-parameter-parity A6, 2026-10-06).
            ["ma_meanvar.jsonl"] = 21000,
            // ~95% of the 24,021 kept cases of the numpy.ma surface tier (gen_ma_surface_oracle.py, 2026-10-08; its
            // 10,123 carved cases live in gaps/ma_surface_gaps.jsonl).
            ["ma_surface.jsonl"] = 22800,
            // ~80% of the 6,976 committed cases (the arraysetops port, 2026-10-08).
            ["ma_setops.jsonl"] = 5580,
            // ~80% of the 3,660 committed cases (the put port, 2026-10-08).
            ["ma_put.jsonl"] = 2900,
        };

        /// <summary>
        ///     Host-libm variant of <see cref="RunMaCorpus"/>: Inconclusive off-Windows (the tier's
        ///     transcendental/complex cells are ucrtbase-dependent), strict on win-amd64.
        /// </summary>
        private static void RunHostLibmMaCorpus(string file)
        {
            if (!System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                    System.Runtime.InteropServices.OSPlatform.Windows))
                Assert.Inconclusive(
                    $"{file} is authored against the win-amd64 CRT libm; np.ma delegates to the same " +
                    "transcendental/complex kernels as the plain ops, so an off-Windows divergence is a " +
                    "platform artifact, not an ma defect (the Unary/Fft tier policy).");
            RunMaCorpus(file);
        }

        /// <summary>
        ///     Replay a committed <c>ma_*.jsonl</c> tier: rebuild each masked operand (data view +
        ///     optional contiguous bool mask -> <see cref="NDMaskedArray"/>), run the <c>ma.*</c> op, and
        ///     compare by result kind. A failure lists every divergent cell (not first-failure-wins).
        /// </summary>
        /// <param name="file">The corpus file name under Fuzz/corpus/.</param>
        private static void RunMaCorpus(string file)
        {
            // Streamed like RunCorpus: counted up front, parsed one case at a time (see CorpusFile).
            using var cases = FuzzCorpus.Open(file);
            int floor = MaMinCases.TryGetValue(file, out var f) ? f : 1;
            Assert.IsTrue(cases.Count >= floor,
                $"corpus '{file}' has {cases.Count} cases, below the committed floor of {floor} " +
                "(truncated/partial regeneration? see MaMinCases in FuzzCorpusTests.Ma)");

            var failures = new List<string>();
            var documented = new Dictionary<string, int>();

            foreach (var c in cases)
                ReplayMaOne(c, failures, documented);
            // NUMSHARP_ORACLE_FAILURE_DUMP: every failure line, for drawing a new tier's carve (diagnostic only).
            DumpFailures(file, failures);

            if (documented.Count > 0)
                Console.WriteLine($"[{file}] documented Misaligned divergences excused: " +
                                  string.Join("; ", documented.Select(kv => $"{kv.Value}x {kv.Key}")));

            if (failures.Count > 0)
                Assert.Fail($"{failures.Count}/{cases.Count} ma cases diverged from NumPy (unexpected):\n  " +
                            string.Join("\n  ", failures.Take(60)));
        }

        /// <summary>
        ///     Replays ONE masked case: rebuilds its masked operands (data view + optional contiguous bool mask), runs the
        ///     <c>ma.*</c> op through <see cref="OpRegistry"/> and compares the result by its declared kind, appending one
        ///     line per unexcused divergence to <paramref name="failures"/>.
        /// </summary>
        /// <param name="c">The corpus case.</param>
        /// <param name="failures">Collects the divergence lines (empty after the call when the case matches NumPy).</param>
        /// <param name="documented">Counts the excused divergences by reason.</param>
        /// <remarks>
        ///     Shared by <see cref="RunMaCorpus"/> (one list for the whole tier) and <see cref="RunMaGapCorpus"/> (one list
        ///     per carved case, so each case gets its own verdict). The result kinds: <c>masked</c> / <c>masked_tuple</c>
        ///     (filled(0) + mask, and the form / raw data when recorded), <c>array</c> / <c>scalar</c> (CompareArray),
        ///     <c>tuple</c> (CompareArray per slot, arity checked), <c>text</c> (a Python repr, ordinal string equality)
        ///     and <c>dtype</c> (the dtype name). Any exception from the replay itself counts as a THREW divergence.
        /// </remarks>
        private static void ReplayMaOne(FuzzCorpus.Case c, List<string> failures, Dictionary<string, int> documented)
        {
            var empty = Array.Empty<BitDiff.Diff>();
            // Classify against the SHARED ledger under the bare np op name (np.ma delegates to it).
            var sc = StripMa(c);
            try
            {
                var ops = new NDMaskedArray[c.Operands.Length];
                for (int i = 0; i < ops.Length; i++)
                {
                    var data = FuzzCorpus.Reconstruct(c.Operands[i]);
                    var mask = FuzzCorpus.ReconstructMask(c.Operands[i].Shape, c.Operands[i].Mask);
                    // mask==null keeps _mask==null (NumPy's nomask fast path); an explicit mask pairs
                    // with the data view exactly as the generator's np.ma.array(view, mask=…) does.
                    ops[i] = np.ma.masked_array(data, mask);
                }

                // NumPy raised (ma_meanvar.jsonl and ma_surface.jsonl record their errors; the older ma tiers skip such
                // cells). The op must throw too, with a plausible type and NumPy's text.
                if (c.Expects_Throw)
                {
                    CheckMaError(sc, ops, failures, documented);
                    return;
                }

                switch (c.Expected.KindOrArray)
                {
                    case "masked":
                        CompareMasked(sc, (NDMaskedArray)OpRegistry.ApplyMasked(c.Op, c.Params, ops),
                                      c.Expected, null, failures, documented);
                        break;
                    case "masked_tuple":
                    {
                        var got = OpRegistry.ApplyMaskedTuple(c.Op, c.Params, ops);
                        var want = c.Expected.Slots ?? Array.Empty<FuzzCorpus.Expected>();
                        if (got.Length != want.Length)
                            failures.Add($"{sc.Id} [{sc.Layout}]: masked tuple arity {got.Length} != NumPy {want.Length}");
                        else
                            for (int i = 0; i < want.Length; i++)
                                CompareMasked(sc, got[i], want[i], $"slot[{i}]", failures, documented);
                        break;
                    }
                    case "array":
                    case "scalar":
                        CompareArray(sc, ToNDArray(OpRegistry.ApplyMasked(c.Op, c.Params, ops)),
                                     c.Expected, null, failures, documented);
                        break;
                    case "tuple":
                    {
                        // A tuple of plain arrays (ma.nonzero's index arrays): arity first, then each slot as an array.
                        var got = OpRegistry.ApplyMaskedArrayTuple(c.Op, c.Params, ops);
                        var want = c.Expected.Slots ?? Array.Empty<FuzzCorpus.Expected>();
                        if (got.Length != want.Length)
                            failures.Add($"{sc.Id} [{sc.Layout}]: tuple arity {got.Length} != NumPy {want.Length}");
                        else
                            for (int i = 0; i < want.Length; i++)
                                CompareArray(sc, got[i], want[i], $"slot[{i}]", failures, documented);
                        break;
                    }
                    case "text":
                    {
                        // A Python object NumPy returns (a fill value, a list of slices, an edge tuple, a shape tuple,
                        // None) as its normalized repr (gen_ma_surface_oracle.pyrepr / OpRegistry.MaRepr).
                        string got = OpRegistry.ApplyMaskedText(c.Op, c.Params, ops);
                        if (!string.Equals(got, c.Expected.Value, StringComparison.Ordinal))
                            failures.Add($"{sc.Id} [{sc.Layout}]: text {Show(got)} != NumPy {Show(c.Expected.Value)}");
                        break;
                    }
                    case "dtype":
                    {
                        var got = ((DType)OpRegistry.ApplyMasked(c.Op, c.Params, ops)).name;
                        if (!string.Equals(got, c.Expected.Value, StringComparison.Ordinal))
                            failures.Add($"{sc.Id} [{sc.Layout}]: dtype result '{got}' != NumPy '{c.Expected.Value}'");
                        break;
                    }
                    default:
                        failures.Add($"{sc.Id}: unknown ma expected.kind '{c.Expected.Kind}'");
                        break;
                }
            }
            catch (Exception e)
            {
                var reason = MisalignedRegistry.Classify(sc, DivergenceKind.Threw, null, null, default, empty);
                if (reason != null) Bump(documented, reason);
                else failures.Add($"{sc.Id} [{sc.Layout}]: THREW {e.GetType().Name}: {e.Message}");
            }
        }

        /// <summary>
        ///     Replays a carved masked-array gap file (<c>corpus/gaps/*.jsonl</c>): every case must carry a gap id, and
        ///     the test passes only when NO carved case still diverges, so it stays red (an [OpenBugs] test) until the
        ///     gaps are fixed, and prints per gap how many of its cases now match NumPy (a fix shrinks the carve).
        /// </summary>
        /// <param name="fileName">The file name under <c>corpus/gaps/</c>.</param>
        /// <exception cref="AssertFailedException">A case without a gap id, or any carved case that still diverges.</exception>
        private static void RunMaGapCorpus(string fileName)
        {
            using var cases = FuzzCorpus.Open(System.IO.Path.Combine("gaps", fileName));
            var byGap = new SortedDictionary<string, (int total, int open, string sample)>(StringComparer.Ordinal);
            var orphans = new List<string>();
            foreach (var c in cases)
            {
                if (string.IsNullOrEmpty(c.Gap))
                {
                    orphans.Add(c.Id);
                    continue;
                }
                // A fresh list per case: the verdict is whether THIS case still diverges.
                var failures = new List<string>();
                ReplayMaOne(c, failures, new Dictionary<string, int>());
                var (total, open, sample) = byGap.TryGetValue(c.Gap, out var e) ? e : (0, 0, null);
                byGap[c.Gap] = (total + 1, open + (failures.Count > 0 ? 1 : 0), sample ?? (failures.Count > 0 ? failures[0] : null));
            }

            foreach (var (gap, (total, open, sample)) in byGap)
                Console.WriteLine($"[{fileName}] {gap}: {open}/{total} carved cases still diverge" +
                                  (open < total ? $", {total - open} now match NumPy" : "") +
                                  (sample != null ? $"\n    e.g. {sample}" : ""));

            Assert.AreEqual(0, orphans.Count, "carved cases without a gap id: " + string.Join(", ", orphans.Take(10)));
            int stillOpen = byGap.Values.Sum(v => v.open);
            Assert.AreEqual(0, stillOpen,
                $"{stillOpen} carved cases still diverge from NumPy across {byGap.Count(kv => kv.Value.open > 0)} open gaps " +
                $"({string.Join(", ", byGap.Where(kv => kv.Value.open > 0).Select(kv => kv.Key))})");
        }

        // ---- comparators -------------------------------------------------------------------------

        /// <summary>
        ///     Compare a masked result: data dtype + shape + <c>filled(0)</c> bytes (BitDiff, NaN
        ///     tokenized) + <c>getmaskarray</c> bytes (bit-exact). Value divergences route through the
        ///     shared registry under the bare op name; a mask divergence is a HARD failure. When the case
        ///     records them (ma_meanvar.jsonl), the result's FORM (which object came back) and its RAW data at
        ///     every slot are compared too, and a case of valueclass <c>mameanvar</c> gets no excuse at all.
        /// </summary>
        /// <param name="sc">The case, its op name stripped of <c>ma.</c> (the registry's key).</param>
        /// <param name="result">NumSharp's result.</param>
        /// <param name="exp">NumPy's recorded result (a whole case's, or one tuple slot's).</param>
        /// <param name="slot">The tuple slot label for failure lines, or null.</param>
        /// <param name="failures">Collects one line per unexcused divergence.</param>
        /// <param name="documented">Counts the excused divergences by reason.</param>
        private static void CompareMasked(
            FuzzCorpus.Case sc, NDMaskedArray result, FuzzCorpus.Expected exp, string slot,
            List<string> failures, Dictionary<string, int> documented)
        {
            var empty = Array.Empty<BitDiff.Diff>();
            string at = slot == null ? "" : $" {slot}";

            // ma_meanvar.jsonl, ma_surface.jsonl, ma_setops.jsonl and ma_put.jsonl are held to ZERO excuses: MisalignedRegistry
            // returns null for their valueclasses, and none of this file's own excuses below run for them either.
            bool strict = sc.Valueclass is "mameanvar" or "masurface" or "masetops" or "maput";

            // FORM: the masked singleton, a scalar, an array with nomask or with a mask array. getmaskarray cannot tell
            // these apart (an all-False mask reads like nomask, the singleton like a masked 0-d array), so the object
            // kind is checked before any value. NUMSHARP_MA_SKIP_FORM=1 (diagnostic only, never set in CI) skips it, so a
            // replay shows the value and mask divergences a form gap's carve would otherwise hide.
            if (exp.Form != null && Environment.GetEnvironmentVariable("NUMSHARP_MA_SKIP_FORM") != "1")
            {
                string gotForm = MaResultForm(result);
                if (!string.Equals(gotForm, exp.Form, StringComparison.Ordinal))
                {
                    failures.Add($"{sc.Id} [{sc.Layout}]{at}: result form '{gotForm}' != NumPy '{exp.Form}'");
                    return;
                }
            }

            NPTypeCode tc;
            try { tc = FuzzCorpus.DtypeToTC(exp.Dtype); }
            catch (NotSupportedException)
            {
                failures.Add($"{sc.Id} [{sc.Layout}]{at}: NumPy dtype '{exp.Dtype}' has no NumSharp analog");
                return;
            }

            // filled(0): masked slots -> 0 on both sides; unmasked slots keep the data. This is the
            // observable value, and its dtype/shape are the result's data dtype/shape.
            NDArray data0 = result.filled((object)0);

            if (data0.typecode != tc)
            {
                var reason = MisalignedRegistry.Classify(sc, DivergenceKind.Dtype, null, null, tc, empty);
                // The shared registry's NEP 50 weak-scalar excuse covers np.*'s house rule (a 0-d operand against an array
                // is read as a Python scalar). numpy.ma's getdata keeps a 0-d operand STRONG and NDMaskedArray does the
                // same (PromoteMaScalar), so that excuse never applies to an ma case: a 0-d dtype divergence here is a bug.
                if (reason is not null && reason.StartsWith("NEP50 weak-scalar", StringComparison.Ordinal))
                    reason = null;
                reason ??= strict ? null : MaDtypeExcuse(sc);
                if (reason != null) Bump(documented, reason);
                else failures.Add($"{sc.Id} [{sc.Layout}]{at}: data dtype {data0.typecode} != NumPy {exp.Dtype}");
                return;
            }
            if (!ShapeEquals(data0.Shape.dimensions, exp.Shape))
            {
                var reason = MisalignedRegistry.Classify(sc, DivergenceKind.Shape, null, null, tc, empty);
                if (reason != null) Bump(documented, reason);
                else failures.Add($"{sc.Id} [{sc.Layout}]{at}: shape [{string.Join(",", data0.Shape.dimensions)}] " +
                                  $"!= NumPy [{string.Join(",", exp.Shape)}]");
                return;
            }

            // MASK: bit-exact, save the ONE documented inherited edge below.
            var actualMask = FuzzCorpus.ResultBytes(np.ma.getmaskarray(result));
            var expectedMask = FuzzCorpus.FromHex(exp.Mask);
            var maskDiff = MaskDiffIndices(actualMask, expectedMask);
            if (maskDiff != null)
            {
                // A masked op ORs `~isfinite(result)` into its mask, so it inherits the underlying
                // complex-unary op's special-value FINITENESS: NumPy's msun gives a FINITE value at a
                // nan/inf special input where NumSharp's System.Numerics/NDComplexMath gives nan (e.g.
                // arctanh(nan+inf·j) = 0+π/2·j in NumPy, nan in NumSharp). That flips the mask bit at
                // that slot ONLY. It is the documented complex-unary special-value edge (the plain-op
                // MisalignedRegistry branch 7), not an ma mask bug — excused precisely: every differing
                // slot must be a NON-FINITE complex INPUT of a complex-unary op. Any other mask diff
                // (a real mask-propagation bug) still fails.
                var maskReason = strict ? null : MaComplexEdgeMaskExcuse(sc, maskDiff) ?? MaReductionMaskExcuse(sc);
                if (maskReason != null)
                {
                    // The mask AND the filled(0) data both diverge at that slot (NumSharp masks -> 0,
                    // NumPy keeps the finite special value), and both are one inherited edge — excuse
                    // the whole case, exactly as the plain-op branch 7 excuses the whole complex case.
                    Bump(documented, maskReason);
                    return;
                }
                failures.Add($"{sc.Id} [{sc.Layout}]{at}: MASK differs — " +
                             $"NumSharp [{MaskStr(actualMask)}] != NumPy [{MaskStr(expectedMask)}]");
                return;
            }

            // DATA (filled 0): unmasked values bit-exact (NaN tokenized). Masked slots are 0 both sides.
            var actual = FuzzCorpus.ResultBytes(data0);
            var expected = FuzzCorpus.FromHex(exp.Buffer);
            var diffs = BitDiff.Compare(expected, actual, tc, false);
            if (diffs.Count == 0)
            {
                CompareMaskedRaw(sc, result, exp, tc, at, failures);
                return;
            }

            // Signed-zero ±0 differences are non-contractual (IEEE, order/algorithm dependent) — the K13
            // class. The domained divide/mod fill-back and numpy's masked-machinery canonicalization land
            // on different zero signs at a division-to-zero; excuse when EVERY diff is a pure ±0 flip.
            // mean / var / std reproduce that fill-back statement for statement (plan api-parameter-parity A6),
            // so a zero sign they miss is a real bug (a flat ddof=2 variance of one element is -0.0 in NumPy).
            if (!strict && sc.Op is not ("mean" or "var" or "std" or "anom") && AllSignedZeroFlips(expected, actual, diffs, tc))
            {
                Bump(documented, "signed-zero ±0 non-contractual (masked domained divide/mod fill-back) [documented]");
                return;
            }

            // complex128 cumprod is a chain of complex MULTIPLIES; NumPy's npy_cmul is FMA-contracted under
            // MSVC while .NET's System.Numerics.Complex operator* is not, so the chain drifts a few ULP per
            // component — the same non-portable host-FMA class the main nanscan tier CARVES complex128 out of
            // nancumprod for. Bounded so a gross cumprod bug (wrong exponent) still fails.
            if (!strict && sc.Op == "cumprod" && tc == NPTypeCode.Complex &&
                diffs.All(d => ComplexComponentsWithinUlp(expected, actual, d.Index, 256)))
            {
                Bump(documented, "complex128 cumprod ~ULP: host-FMA-contracted npy_cmul vs .NET Complex operator* " +
                                 "[documented, the nanscan complex-cumprod carve-out class]");
                return;
            }

            var truth = exp.Truth == null ? null : FuzzCorpus.FromHex(exp.Truth);
            var vreason = MisalignedRegistry.Classify(sc, DivergenceKind.Value, expected, actual, tc, diffs, truth)
                          ?? (strict ? null : MaValueExcuse(sc));
            if (vreason != null)
            {
                Bump(documented, vreason);
                return;
            }

            failures.Add($"{sc.Id} [{sc.Layout}]{at}: filled(0) " +
                string.Join(", ", diffs.Take(3).Select(d => $"@{d.Index} exp {d.Expected} act {d.Actual}")) +
                (diffs.Count > 3 ? $" (+{diffs.Count - 3} more)" : ""));
        }

        /// <summary>
        ///     The RAW comparison of ma_meanvar.jsonl: the result's data at EVERY slot, masked ones included (NumPy's
        ///     divide-domain fills, the minuend's put-back data), NaN tokenized like the filled(0) comparison. A case
        ///     without a recorded <see cref="FuzzCorpus.Expected.Raw"/> (every other ma tier) skips it.
        /// </summary>
        /// <param name="sc">The case (for the failure line).</param>
        /// <param name="result">NumSharp's result; the masked singleton's data is its 0-d float64 0.0, like NumPy's.</param>
        /// <param name="exp">NumPy's recorded result.</param>
        /// <param name="tc">The data dtype, already checked equal to NumPy's.</param>
        /// <param name="at">The tuple-slot suffix of the failure line.</param>
        /// <param name="failures">Collects the failure line.</param>
        private static void CompareMaskedRaw(
            FuzzCorpus.Case sc, NDMaskedArray result, FuzzCorpus.Expected exp, NPTypeCode tc, string at, List<string> failures)
        {
            if (exp.Raw == null)
                return;
            var expected = FuzzCorpus.FromHex(exp.Raw);
            var actual = FuzzCorpus.ResultBytes(result.data);
            var diffs = BitDiff.Compare(expected, actual, tc, false);
            if (diffs.Count == 0)
                return;
            failures.Add($"{sc.Id} [{sc.Layout}]{at}: raw data " +
                string.Join(", ", diffs.Take(3).Select(d => $"@{d.Index} exp {d.Expected} act {d.Actual}")) +
                (diffs.Count > 3 ? $" (+{diffs.Count - 3} more)" : ""));
        }

        /// <summary>
        ///     The object kind of a masked result, in the corpus's <c>form</c> words: <c>masked</c> (the
        ///     <see cref="np.ma"/> singleton), <c>scalar</c> (a 0-d result without a mask: NumPy's scalar), <c>nomask</c>
        ///     (an array whose mask is nomask) or <c>mask</c> (an array with a mask array, an all-False one included).
        /// </summary>
        /// <param name="r">The result.</param>
        /// <returns>The form word.</returns>
        private static string MaResultForm(NDMaskedArray r)
        {
            if (ReferenceEquals(r, np.ma.NDMasked))
                return "masked";
            if (r._mask is null)
                return r.data.ndim == 0 ? "scalar" : "nomask";
            return "mask";
        }

        /// <summary>
        ///     Error parity for a masked case whose NumPy call raised (<see cref="FuzzCorpus.Case.Expects_Throw"/>): the op
        ///     must throw, as a type <see cref="ErrorTypeMap"/> pairs with NumPy's, with NumPy's text (the same three
        ///     checks as the ordinary tiers' CheckError). A silent result is always a failure.
        /// </summary>
        /// <param name="sc">The case, its op name stripped of <c>ma.</c>.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="failures">Collects one line per unexcused mismatch.</param>
        /// <param name="documented">Counts the excused mismatches by reason.</param>
        /// <summary>
        ///     The exception types a STRICT ma tier accepts besides NumPy's own name: OverflowError is .NET's
        ///     <see cref="OverflowException"/> (the house type for NumPy's OverflowError), and a ValueError may be the
        ///     house <see cref="IncorrectShapeException"/> that NumSharp's <c>ndarray.resize</c> raises with NumPy's texts.
        ///     An IndexError or a TypeError must be NumSharp's <see cref="IndexError"/> / <see cref="TypeError"/>.
        /// </summary>
        private static readonly Dictionary<string, string[]> StrictErrorTypes = new()
        {
            ["OverflowError"] = new[] { "OverflowException" },
            ["ValueError"] = new[] { "IncorrectShapeException" },
        };

        private static void CheckMaError(
            FuzzCorpus.Case sc, NDMaskedArray[] ops, List<string> failures, Dictionary<string, int> documented)
        {
            var empty = Array.Empty<BitDiff.Diff>();
            Exception caught = null;
            try
            {
                // An error case keeps its result kind (ma_surface.jsonl), so the call goes through the dispatcher that
                // kind's success case would use: a text or tuple op is not registered in ApplyMasked.
                switch (sc.Expected?.KindOrArray)
                {
                    case "text": _ = OpRegistry.ApplyMaskedText(sc.Op, sc.Params, ops); break;
                    case "tuple": _ = OpRegistry.ApplyMaskedArrayTuple(sc.Op, sc.Params, ops); break;
                    case "masked_tuple": _ = OpRegistry.ApplyMaskedTuple(sc.Op, sc.Params, ops); break;
                    default: _ = OpRegistry.ApplyMasked(sc.Op, sc.Params, ops); break;
                }
            }
            catch (Exception e)
            {
                caught = e;
            }

            if (caught == null)
            {
                var reason = MisalignedRegistry.Classify(sc, DivergenceKind.Value, null, null, default, empty);
                if (reason != null) Bump(documented, reason);
                else failures.Add($"{sc.Id} [{sc.Layout}]: NumPy raises {sc.Error?.Type ?? "an error"} but NumSharp " +
                                  "produced a result (error-parity gap)");
                return;
            }
            if (sc.Error == null)
                return;

            string gotType = caught.GetType().Name;
            string gotText = NormalizeMessage(caught.Message);
            string wantText = (sc.Error.Text ?? "").Trim();
            // An exception named after NumPy's (Exceptions/{ValueError,TypeError,AxisError}.cs) is the strongest match. The
            // strict tiers accept only NumPy's own type or the house type NumSharp documents for it (StrictErrorTypes).
            bool typeOk = sc.Valueclass == "maput"
                ? gotType == sc.Error.Type || (StrictErrorTypes.TryGetValue(sc.Error.Type, out var strictAllowed)
                                               && strictAllowed.Contains(gotType))
                : gotType == sc.Error.Type
                  || !ErrorTypeMap.TryGetValue(sc.Error.Type, out var allowed)
                  || allowed.Contains(gotType);
            bool textOk = string.Equals(gotText, wantText, StringComparison.Ordinal);
            if (typeOk && textOk)
                return;

            var er = MisalignedRegistry.Classify(sc, DivergenceKind.ErrorText, null, null, default, empty);
            if (er != null)
            {
                Bump(documented, er);
                return;
            }
            failures.Add($"{sc.Id} [{sc.Layout}]: error mismatch\n" +
                         $"      numpy:    {sc.Error.Type}: {Show(wantText)}\n" +
                         $"      numsharp: {gotType}: {Show(gotText)}");
        }

        // ---- helpers -----------------------------------------------------------------------------

        /// <summary>A shallow copy of the case with the <c>ma.</c> op prefix stripped, so
        /// <see cref="MisalignedRegistry"/> matches it against the bare np op name it delegates to.</summary>
        private static FuzzCorpus.Case StripMa(FuzzCorpus.Case c) => new()
        {
            Id = c.Id,
            Op = c.Op.StartsWith("ma.", StringComparison.Ordinal) ? c.Op.Substring(3) : c.Op,
            Params = c.Params,
            Operands = c.Operands,
            Expected = c.Expected,
            Layout = c.Layout,
            Valueclass = c.Valueclass,
            Alias = c.Alias,
            Expects_Throw = c.Expects_Throw,
            Error = c.Error,
        };

        /// <summary>
        ///     KNOWN-GAP return-dtype divergences (tightly scoped to the two ops that have one), documented
        ///     in docs/MA_ORACLE_DESIGN.md "Known gaps". A dtype diff on any OTHER op still fails.
        /// </summary>
        private static string MaDtypeExcuse(FuzzCorpus.Case sc) => sc.Op switch
        {
            "average" => "[known gap] ma.average promotes the result to float64 where NumPy keeps the input float width",
            "median" => "[known gap] ma.median widens a narrow-float / all-masked result to float64 where NumPy keeps the width",
            _ => null,
        };

        /// <summary>
        ///     KNOWN-GAP mask divergences on NON-FINITE unmasked data: average masks a nan/inf
        ///     reduction result where NumSharp leaves it unmasked (the masked-op composition differs; var/std/mean/
        ///     anom were in this class until they became numpy.ma's statements, plan api-parameter-parity A6), and
        ///     ma.sort places a masked slot differently when the UNMASKED data carries NaN (NaN-ordering vs
        ///     strict masked-to-end). Scoped to those ops at a float/complex dtype so a real mask-propagation
        ///     bug in any other op still fails. Documented in docs/MA_ORACLE_DESIGN.md "Known gaps".
        /// </summary>
        private static string MaReductionMaskExcuse(FuzzCorpus.Case sc)
        {
            // Complex power: the result's finiteness at a pathological complex input differs (Complex.Pow
            // vs npy_cpow's host cpow — the F5 class the VALUE diffs are already excused under), which flips
            // the ~isfinite mask bit at that one slot. Scoped to power with a complex operand.
            if (sc.Op == "power" && sc.Operands.Any(o => o.Dtype == "complex128"))
                return "[documented F5] complex power result finiteness at a pathological input differs " +
                       "(Complex.Pow vs npy_cpow), flipping the mask bit — same class as the excused value edge";

            bool floatc = sc.Operands.Length >= 1 &&
                          sc.Operands[0].Dtype is "float16" or "float32" or "float64" or "complex128";
            if (!floatc) return null;
            if (sc.Op == "average")
                return "[known gap] ma.average masks a non-finite (nan/inf) reduction result " +
                       "where NumSharp leaves it unmasked (nan/inf present in the UNMASKED data)";
            if (sc.Op == "sort")
                return "[known gap] ma.sort masked-slot position differs when the UNMASKED data carries NaN " +
                       "(NaN-ordering vs strict masked-to-end)";
            return null;
        }

        /// <summary>
        ///     KNOWN-GAP value divergences the mask/dtype branches do not already cover: ma.average over
        ///     non-finite unmasked data (NumPy masks the slot -> filled(0)=0, NumSharp keeps nan/inf).
        ///     var/std/mean/anom have no value excuse (numpy.ma's statements since plan api-parameter-parity A6).
        ///     Scoped to average at a float/complex dtype. Documented in docs/MA_ORACLE_DESIGN.md.
        /// </summary>
        private static string MaValueExcuse(FuzzCorpus.Case sc)
        {
            // median VALUE diverges on a NON-CONTIGUOUS input (1-D strided/negstride/offset AND the 2-D
            // strided/negstride edge layouts) — a real strided-reduction gap (its internal sort over a strided
            // view reads the wrong stride), any dtype; contiguous/F/transposed are byte-exact. anom shared it until
            // it became numpy.ma's statements (plan api-parameter-parity A6). Documented in docs/MA_ORACLE_DESIGN.md.
            if (sc.Op == "median" &&
                (sc.Layout is "strided_step2_1d" or "negstride_1d" or "simple_slice_offset_1d"
                    or "strided_2d_cols" or "negstride_2d_offset"))
                return "[known gap] ma.median value diverges on a non-contiguous (strided/negstride/offset) input";

            bool floatc = sc.Operands.Length >= 1 &&
                          sc.Operands[0].Dtype is "float16" or "float32" or "float64" or "complex128";
            if (!floatc) return null;
            if (sc.Op == "average")
                return "[known gap] ma.average value differs on non-finite unmasked data (NumPy masks -> 0, " +
                       "NumSharp keeps nan/inf); the class var/std/mean/anom left in plan api-parameter-parity A6";
            if (sc.Op == "median")
                return "[known gap] ma.median value differs on a slice carrying non-finite (nan/inf) unmasked data " +
                       "(sort/interpolation of inf; NumSharp finite vs NumPy inf)";
            if (sc.Op == "ptp" && sc.Operands[0].Dtype == "complex128")
                return "[documented] complex ptp (max-min) NaN propagation: complex ordering with NaN is " +
                       "implementation-defined (NumPy NaN+NaNj vs NumSharp NaN+0j) — the complex-reduction NaN class";
            return null;
        }

        /// <summary>
        ///     True when EVERY diff is a pure ±0 sign flip. For a real dtype this defers to
        ///     <see cref="BitDiff.IsSignedZeroFlip"/>; for complex128 each 16-byte element is split into its
        ///     real and imaginary doubles and each component must be bit-equal OR a ±0 flip (the complex
        ///     domained-divide fill-back lands -0 in a lane where NumPy has +0).
        /// </summary>
        private static bool AllSignedZeroFlips(byte[] e, byte[] a, List<BitDiff.Diff> diffs, NPTypeCode tc)
        {
            if (tc != NPTypeCode.Complex)
                return diffs.All(d => BitDiff.IsSignedZeroFlip(e, a, d.Index, tc));
            foreach (var d in diffs)
            {
                int off = d.Index * 16;   // element index -> byte offset (complex128 = 2 doubles)
                if (off + 16 > e.Length || off + 16 > a.Length) return false;
                if (!ComponentEqualOrPmZero(e, a, off) || !ComponentEqualOrPmZero(e, a, off + 8))
                    return false;
            }
            return true;
        }

        /// <summary>One 8-byte double component is bit-identical, or both sides are zero (a ±0 sign flip).</summary>
        private static bool ComponentEqualOrPmZero(byte[] e, byte[] a, int off)
        {
            if (BitConverter.ToInt64(e, off) == BitConverter.ToInt64(a, off)) return true;
            return BitConverter.ToDouble(e, off) == 0.0 && BitConverter.ToDouble(a, off) == 0.0;
        }

        /// <summary>Both doubles of a complex128 element (real + imag) are within <paramref name="maxUlp"/>
        /// of each other, by sign-magnitude ULP distance (NaN components are treated as matching — the
        /// tokenized-NaN policy). Bounds the complex-multiply FMA-contraction drift excuse.</summary>
        private static bool ComplexComponentsWithinUlp(byte[] e, byte[] a, int elemIndex, long maxUlp)
        {
            int off = elemIndex * 16;
            if (off + 16 > e.Length || off + 16 > a.Length) return false;
            return DoubleUlp(e, a, off, maxUlp) && DoubleUlp(e, a, off + 8, maxUlp);
        }

        /// <summary>Sign-magnitude ULP distance of the two doubles at <paramref name="off"/> is ≤ maxUlp
        /// (a shared NaN passes; a finite-vs-NaN mismatch fails).</summary>
        private static bool DoubleUlp(byte[] e, byte[] a, int off, long maxUlp)
        {
            double de = BitConverter.ToDouble(e, off), da = BitConverter.ToDouble(a, off);
            if (double.IsNaN(de) && double.IsNaN(da)) return true;
            if (double.IsNaN(de) || double.IsNaN(da)) return false;
            long le = BitConverter.ToInt64(e, off), la = BitConverter.ToInt64(a, off);
            if (le < 0) le = long.MinValue - le;   // map to a monotone signed-magnitude order
            if (la < 0) la = long.MinValue - la;
            long d = le > la ? le - la : la - le;
            return d >= 0 && d <= maxUlp;          // d<0 only on overflow (opposite huge magnitudes) -> reject
        }

        /// <summary>Coerce an <c>array</c>/<c>scalar</c>-kind ma result to an NDArray for CompareArray.</summary>
        private static NDArray ToNDArray(object r) => r switch
        {
            NDArray a => a,
            bool b => NDArray.Scalar(b),
            long l => NDArray.Scalar(l),
            int i => NDArray.Scalar((long)i),
            NDMaskedArray m => m.filled((object)0),    // defensive: an array-kind case that yields an NDMaskedArray
            _ => throw new NotSupportedException($"ma array/scalar result of type {r?.GetType().Name ?? "null"}"),
        };

        /// <summary>Indices where two mask byte buffers differ; null when identical. A length
        /// mismatch is reported as the sentinel [-1] (a hard, non-excusable difference).</summary>
        private static List<int> MaskDiffIndices(byte[] a, byte[] b)
        {
            if (a.Length != b.Length) return new List<int> { -1 };
            List<int> idx = null;
            for (int i = 0; i < a.Length; i++)
                if (a[i] != b[i]) (idx ??= new List<int>()).Add(i);
            return idx;
        }

        /// <summary>The complex-unary ops whose msun special-value handling (finite result at a nan/inf
        /// special input) NumSharp's System.Numerics/NDComplexMath does not fully reproduce — so a
        /// masked wrapper's <c>~isfinite</c> term flips at exactly those slots. The plain-op counterpart
        /// is MisalignedRegistry branch 7.</summary>
        private static readonly HashSet<string> MaComplexEdgeOps = new()
        {
            "arctanh", "arccosh", "arcsin", "arccos", "arctan", "arcsinh",
            "sqrt", "log", "log2", "log10", "tan", "sin", "cos", "sinh", "cosh", "tanh", "exp",
        };

        /// <summary>
        ///     Excuse a MASK divergence that is entirely the inherited complex-unary special-value edge:
        ///     a complex-unary op whose every differing mask slot has a NON-FINITE complex INPUT (so the
        ///     divergence is only whether NumSharp's <c>op(nan/inf)</c> is finite like NumPy's). Any mask
        ///     diff at a finite input — a genuine mask-propagation bug — returns null and fails.
        /// </summary>
        private static string MaComplexEdgeMaskExcuse(FuzzCorpus.Case sc, List<int> diffIdx)
        {
            if (diffIdx == null || diffIdx.Contains(-1)) return null;
            if (!MaComplexEdgeOps.Contains(sc.Op)) return null;
            if (sc.Operands.Length != 1 || sc.Operands[0].Dtype != "complex128") return null;

            System.Numerics.Complex[] flat;
            try
            {
                // Materialize the input in logical C-order so a flat position matches the result's.
                var input = FuzzCorpus.Reconstruct(sc.Operands[0]);
                flat = np.ascontiguousarray(input).ToArray<System.Numerics.Complex>();
            }
            catch { return null; }

            foreach (var i in diffIdx)
                if (i < 0 || i >= flat.Length ||
                    (double.IsFinite(flat[i].Real) && double.IsFinite(flat[i].Imaginary)))
                    return null;   // a mask diff at a FINITE input is a real bug, not this edge
            return "ma complex-unary domain mask: NumSharp op(complex) finiteness at a nan/inf special "
                 + "input differs from NumPy (inherited complex-unary special-value edge, plain-op branch 7) [documented]";
        }

        /// <summary>Render a mask byte buffer as T/F for a legible failure line (capped).</summary>
        private static string MaskStr(byte[] m)
        {
            var take = Math.Min(m.Length, 32);
            var s = string.Concat(Enumerable.Range(0, take).Select(i => m[i] != 0 ? "T" : "F"));
            return m.Length > take ? s + $"…({m.Length})" : s;
        }
    }
}
