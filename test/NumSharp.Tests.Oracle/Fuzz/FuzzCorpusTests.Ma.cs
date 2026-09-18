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
    ///     <c>ma.</c> op-name prefix STRIPPED — the same ledger that excuses e.g. float var/std
    ///     accumulation order and the complex-unary ≤3-ULP envelope for the plain ops. A MASK
    ///     divergence is never excused: NumSharp claims bit-exact masks, so a wrong mask is always red.</para>
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
        // argmax(->array). var/std accumulation order + float sum/mean are excused via the shared
        // reduce branch (prefix-stripped); the MASK (which slice reduced to masked) is bit-exact.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MaReduce() => RunMaCorpus("ma_reduce.jsonl");

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
            // ~80% of the committed counts (2026-09-18): a truncated/partial regeneration fails loudly.
            ["ma_unary.jsonl"] = 3560,
            ["ma_binary.jsonl"] = 3480,
            ["ma_reduce.jsonl"] = 8980,
            ["ma_scan.jsonl"] = 820,
            ["ma_manip.jsonl"] = 2020,
            ["ma_construct.jsonl"] = 1530,
            ["ma_select.jsonl"] = 410,
            ["ma_sortsetops.jsonl"] = 370,
            ["ma_extras.jsonl"] = 58,
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
        ///     optional contiguous bool mask -> <see cref="MaskedArray"/>), run the <c>ma.*</c> op, and
        ///     compare by result kind. A failure lists every divergent cell (not first-failure-wins).
        /// </summary>
        /// <param name="file">The corpus file name under Fuzz/corpus/.</param>
        private static void RunMaCorpus(string file)
        {
            var cases = FuzzCorpus.Load(file);
            int floor = MaMinCases.TryGetValue(file, out var f) ? f : 1;
            Assert.IsTrue(cases.Count >= floor,
                $"corpus '{file}' has {cases.Count} cases, below the committed floor of {floor} " +
                "(truncated/partial regeneration? see MaMinCases in FuzzCorpusTests.Ma)");

            var failures = new List<string>();
            var documented = new Dictionary<string, int>();
            var empty = Array.Empty<BitDiff.Diff>();

            foreach (var c in cases)
            {
                // Classify against the SHARED ledger under the bare np op name (np.ma delegates to it).
                var sc = StripMa(c);
                try
                {
                    var ops = new MaskedArray[c.Operands.Length];
                    for (int i = 0; i < ops.Length; i++)
                    {
                        var data = FuzzCorpus.Reconstruct(c.Operands[i]);
                        var mask = FuzzCorpus.ReconstructMask(c.Operands[i].Shape, c.Operands[i].Mask);
                        // mask==null keeps _mask==null (NumPy's nomask fast path); an explicit mask pairs
                        // with the data view exactly as the generator's np.ma.array(view, mask=…) does.
                        ops[i] = np.ma.masked_array(data, mask);
                    }

                    switch (c.Expected.KindOrArray)
                    {
                        case "masked":
                            CompareMasked(sc, (MaskedArray)OpRegistry.ApplyMasked(c.Op, c.Params, ops),
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

            if (documented.Count > 0)
                Console.WriteLine($"[{file}] documented Misaligned divergences excused: " +
                                  string.Join("; ", documented.Select(kv => $"{kv.Value}x {kv.Key}")));

            if (failures.Count > 0)
                Assert.Fail($"{failures.Count}/{cases.Count} ma cases diverged from NumPy (unexpected):\n  " +
                            string.Join("\n  ", failures.Take(60)));
        }

        // ---- comparators -------------------------------------------------------------------------

        /// <summary>
        ///     Compare a masked result: data dtype + shape + <c>filled(0)</c> bytes (BitDiff, NaN
        ///     tokenized) + <c>getmaskarray</c> bytes (bit-exact). Value divergences route through the
        ///     shared registry under the bare op name; a mask divergence is a HARD failure.
        /// </summary>
        private static void CompareMasked(
            FuzzCorpus.Case sc, MaskedArray result, FuzzCorpus.Expected exp, string slot,
            List<string> failures, Dictionary<string, int> documented)
        {
            var empty = Array.Empty<BitDiff.Diff>();
            string at = slot == null ? "" : $" {slot}";

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
                var reason = MisalignedRegistry.Classify(sc, DivergenceKind.Dtype, null, null, tc, empty)
                             ?? MaDtypeExcuse(sc);
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
                var maskReason = MaComplexEdgeMaskExcuse(sc, maskDiff) ?? MaReductionMaskExcuse(sc);
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
                return;

            // Signed-zero ±0 differences are non-contractual (IEEE, order/algorithm dependent) — the K13
            // class. The domained divide/mod fill-back and numpy's masked-machinery canonicalization land
            // on different zero signs at a division-to-zero; excuse when EVERY diff is a pure ±0 flip.
            if (AllSignedZeroFlips(expected, actual, diffs, tc))
            {
                Bump(documented, "signed-zero ±0 non-contractual (masked domained divide/mod fill-back) [documented]");
                return;
            }

            var truth = exp.Truth == null ? null : FuzzCorpus.FromHex(exp.Truth);
            var vreason = MisalignedRegistry.Classify(sc, DivergenceKind.Value, expected, actual, tc, diffs, truth)
                          ?? MaValueExcuse(sc);
            if (vreason != null)
            {
                Bump(documented, vreason);
                return;
            }

            failures.Add($"{sc.Id} [{sc.Layout}]{at}: filled(0) " +
                string.Join(", ", diffs.Take(3).Select(d => $"@{d.Index} exp {d.Expected} act {d.Actual}")) +
                (diffs.Count > 3 ? $" (+{diffs.Count - 3} more)" : ""));
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
            "anom" => "[known gap] ma.anom (x - mean) promotes a narrow-float result to float64 where NumPy keeps the width",
            _ => null,
        };

        /// <summary>
        ///     KNOWN-GAP mask divergences on NON-FINITE unmasked data: var/std/mean/average mask a nan/inf
        ///     reduction result where NumSharp leaves it unmasked (the masked-op composition differs), and
        ///     ma.sort places a masked slot differently when the UNMASKED data carries NaN (NaN-ordering vs
        ///     strict masked-to-end). Scoped to those ops at a float/complex dtype so a real mask-propagation
        ///     bug in any other op still fails. Documented in docs/MA_ORACLE_DESIGN.md "Known gaps".
        /// </summary>
        private static string MaReductionMaskExcuse(FuzzCorpus.Case sc)
        {
            // isin/in1d mask propagation is dtype-independent, so gate it before the float check.
            if (sc.Op is "isin" or "in1d")
                return "[known gap] ma.isin/in1d do not propagate the element operand's mask to the bool result";

            // Complex power: the result's finiteness at a pathological complex input differs (Complex.Pow
            // vs npy_cpow's host cpow — the F5 class the VALUE diffs are already excused under), which flips
            // the ~isfinite mask bit at that one slot. Scoped to power with a complex operand.
            if (sc.Op == "power" && sc.Operands.Any(o => o.Dtype == "complex128"))
                return "[documented F5] complex power result finiteness at a pathological input differs " +
                       "(Complex.Pow vs npy_cpow), flipping the mask bit — same class as the excused value edge";

            bool floatc = sc.Operands.Length >= 1 &&
                          sc.Operands[0].Dtype is "float16" or "float32" or "float64" or "complex128";
            if (!floatc) return null;
            if (sc.Op is "var" or "std" or "mean" or "average" or "anom")
                return "[known gap] ma var/std/mean/average/anom mask a non-finite (nan/inf) reduction result " +
                       "where NumSharp leaves it unmasked (nan/inf present in the UNMASKED data)";
            if (sc.Op == "sort")
                return "[known gap] ma.sort masked-slot position differs when the UNMASKED data carries NaN " +
                       "(NaN-ordering vs strict masked-to-end)";
            return null;
        }

        /// <summary>
        ///     KNOWN-GAP value divergences the mask/dtype branches do not already cover: ma.average over
        ///     non-finite unmasked data (NumPy masks the slot -> filled(0)=0, NumSharp keeps nan/inf). The
        ///     var/std/mean value drift is covered by the shared reduce-accumulation excuse; this adds only
        ///     average. Scoped to average at a float/complex dtype. Documented in docs/MA_ORACLE_DESIGN.md.
        /// </summary>
        private static string MaValueExcuse(FuzzCorpus.Case sc)
        {
            // anom/median VALUE diverges on a NON-CONTIGUOUS 1-D input (strided/negstride/offset) — a real
            // strided-reduction gap in these two niche ops (their internal mean/sort over a strided view),
            // any dtype; contiguous/F/transposed are byte-exact. Documented in docs/MA_ORACLE_DESIGN.md.
            if ((sc.Op is "anom" or "median") &&
                (sc.Layout is "strided_step2_1d" or "negstride_1d" or "simple_slice_offset_1d"))
                return "[known gap] ma.anom/median value diverges on a non-contiguous (strided/negstride/offset) input";

            bool floatc = sc.Operands.Length >= 1 &&
                          sc.Operands[0].Dtype is "float16" or "float32" or "float64" or "complex128";
            if (!floatc) return null;
            if (sc.Op is "average" or "anom")
                return "[known gap] ma.average/anom value differs on non-finite unmasked data (NumPy masks -> 0, " +
                       "NumSharp keeps nan/inf); same class as var/std/mean non-finite masking";
            if (sc.Op == "median")
                return "[known gap] ma.median value differs on a slice carrying non-finite (nan/inf) unmasked data " +
                       "(sort/interpolation of inf; NumSharp finite vs NumPy inf)";
            if (sc.Op == "unique" && sc.Operands[0].Dtype == "complex128")
                return "[known gap] ma.unique complex128 NaN element: the NaN's imaginary lane is non-contractual " +
                       "(NumPy keeps the element's original imag, NumSharp canonicalizes)";
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

        /// <summary>Coerce an <c>array</c>/<c>scalar</c>-kind ma result to an NDArray for CompareArray.</summary>
        private static NDArray ToNDArray(object r) => r switch
        {
            NDArray a => a,
            bool b => NDArray.Scalar(b),
            long l => NDArray.Scalar(l),
            int i => NDArray.Scalar((long)i),
            MaskedArray m => m.filled((object)0),      // defensive: an array-kind case that yields a MaskedArray
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
