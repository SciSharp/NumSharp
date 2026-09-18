using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     Anti-coverage-theater gates. Surface coverage proves an op is named; these checks prove
    ///     it is represented by a small matrix rather than one duplicate happy-path row.
    /// </summary>
    [TestClass]
    public class OracleCoverageStrengthTests
    {
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void EveryOrdinaryOracleOp_HasFourCasesAndAChangingAxis()
        {
            var byOp = new Dictionary<string, Strength>(StringComparer.Ordinal);
            string directory = Path.GetDirectoryName(FuzzCorpus.CorpusPath("unused"));
            foreach (string path in Directory.EnumerateFiles(directory, "*.jsonl"))
            {
                string file = Path.GetFileName(path);
                if (file.EndsWith(".host.jsonl", StringComparison.Ordinal) ||
                    file.StartsWith("index_", StringComparison.Ordinal) ||
                    file.StartsWith("ma_", StringComparison.Ordinal))
                    continue; // host metadata, the advanced-indexing schema, and the masked-array
                              // schema (MaskedArray operands + ApplyMasked; gated by FuzzCorpusTests.Ma
                              // with its own coverage model — many ma helper ops legitimately have <4 cases)

                foreach (var c in FuzzCorpus.Load(file))
                {
                    if (string.IsNullOrEmpty(c.Op))
                        continue;
                    if (!byOp.TryGetValue(c.Op, out var s))
                        byOp[c.Op] = s = new Strength();
                    s.Count++;
                    s.Layouts.Add(c.Layout ?? "");
                    s.ValueClasses.Add(c.Valueclass ?? "");
                    s.DtypeSignatures.Add(string.Join(",", (c.Operands ?? Array.Empty<FuzzCorpus.Operand>())
                        .Select(o => o.Dtype ?? "")));
                    s.ParameterSignatures.Add(c.Params == null ? "{}" : string.Join("|",
                        c.Params.OrderBy(kv => kv.Key, StringComparer.Ordinal)
                            .Select(kv => kv.Key + "=" + kv.Value.GetRawText())));
                    s.Outcomes.Add(c.Error != null || c.Expects_Throw
                        ? "error"
                        : c.Expected?.KindOrArray ?? "array");
                }
            }

            var thin = byOp.Where(kv => kv.Value.Count < 4)
                .Select(kv => $"{kv.Key}={kv.Value.Count}").OrderBy(x => x).ToArray();
            var duplicateOnly = byOp.Where(kv => !kv.Value.HasVariation)
                .Select(kv => kv.Key).OrderBy(x => x).ToArray();

            Assert.AreEqual(0, thin.Length,
                "Every ordinary op needs at least four committed cases; thin: " + string.Join(", ", thin));
            Assert.AreEqual(0, duplicateOnly.Length,
                "Every ordinary op must vary layout, params, operand dtype, value class, or outcome; duplicates only: " +
                string.Join(", ", duplicateOnly));

            int underTen = byOp.Count(kv => kv.Value.Count < 10);
            int oneLayout = byOp.Count(kv => kv.Value.Layouts.Count == 1);
            int oneDtypeSignature = byOp.Count(kv => kv.Value.DtypeSignatures.Count == 1);
            int errorCovered = byOp.Count(kv => kv.Value.Outcomes.Contains("error"));
            Console.WriteLine($"[OracleStrength] ops={byOp.Count}, min_cases={byOp.Min(kv => kv.Value.Count)}, " +
                              $"under10={underTen}, one_layout={oneLayout}, " +
                              $"one_dtype_signature={oneDtypeSignature}, error_covered={errorCovered}");
        }

        // ================= dtype-spread gate (the "all 15 dtypes" validation) ==================
        //
        // NumSharp supports 15 dtypes and the DOD says every op handles all of them or documents
        // the exception. The corpus expresses 13 directly, Char through the uint16 proxy weave,
        // and Decimal through the independent C# oracle (decimal_*.jsonl) — so "covered" here
        // means the op key carries the dtype as an operand (value OR error cell; a gated
        // rejection IS coverage of the combination), with the params/expected dtype as the
        // fallback for zero-operand creation ops. Two rules, both ratchets:
        //   1. every op reaches >= 4 distinct dtypes, or carries a one-line entry in
        //      FixedDtypeOps saying why its dtype axis is narrower BY DESIGN — self-retiring in
        //      both directions (a listed op that disappears, or that grows past the floor, fails
        //      so the ledger tracks reality);
        //   2. per-dtype GLOBAL op-count floors (like MinCases: filled spread only ratchets up),
        //      so a regeneration that silently drops a dtype axis turns the gate red.
        // ma_* files are skipped exactly as in the case-count gate above (own coverage model).

        /// <summary>
        ///     Ops whose dtype axis is legitimately narrower than four dtypes, each with the
        ///     reviewed one-line reason (the MisalignedRegistry discipline applied to the dtype
        ///     axis). Grouped by cause; every entry is asserted stale when the op vanishes or its
        ///     spread grows to >= 4.
        /// </summary>
        private static readonly Dictionary<string, string> FixedDtypeOps = new(StringComparer.Ordinal)
        {
            // Fixed-output GENERATORS: scalar parameters in, one blessed dtype out.
            ["bartlett"] = "window generator: scalar M -> float64 taper (NumPy forces >= float64)",
            ["blackman"] = "window generator: scalar M -> float64 taper",
            ["hamming"] = "window generator: scalar M -> float64 taper",
            ["hanning"] = "window generator: scalar M -> float64 taper",
            ["kaiser"] = "window generator: scalar (M, beta) -> float64 taper",
            ["fftfreq"] = "frequency generator: scalar (n, d) -> float64 grid",
            ["rfftfreq"] = "frequency generator: scalar (n, d) -> float64 grid",
            ["diag_indices"] = "index generator: n/ndim -> intp coordinate arrays",
            ["diag_indices_from"] = "index accessor: reads only the operand's SHAPE (dtype-blind)",
            ["tril_indices"] = "index generator: (n, k, m) -> intp coordinate arrays",
            ["tril_indices_from"] = "index accessor: reads only the operand's SHAPE (dtype-blind)",
            ["triu_indices"] = "index generator: (n, k, m) -> intp coordinate arrays",
            ["triu_indices_from"] = "index accessor: reads only the operand's SHAPE (dtype-blind)",
            ["mask_indices"] = "index generator: (n, mask_func, k) -> intp coordinate arrays",
            ["indices"] = "coordinate-grid generator; dtype axis is the param sweep int32/float64 (default intp + the float request)",
            ["indices_sparse"] = "coordinate-grid generator; same param sweep as indices",
            ["ndindex"] = "index-space odometer: shape in, int64 indices out — no operand dtype exists",
            ["ravel_multi_index"] = "coordinate<->flat converter: intp indices by contract",
            ["unravel_index"] = "coordinate<->flat converter: intp indices by contract",
            ["unravel_index_all"] = "coordinate<->flat converter: intp indices by contract",
            ["unpackbits"] = "uint8-only by NumPy contract (packbits' inverse)",

            // Dtype axis lives in PARAMS, not operands: the promotion/dtype-predicate family
            // sweeps dtype PAIRS as strings, which the operand-dtype metric cannot see.
            ["can_cast"] = "dtype-pair predicate: the axis is the (from, to, casting) PARAM sweep",
            ["issubdtype"] = "dtype-pair predicate: the axis is the (a, b) PARAM sweep",
            ["promote_types"] = "dtype-pair function: the axis is the (a, b) PARAM sweep",
            ["result_type_dtypes"] = "dtype-pair function: the axis is the (a, b) PARAM sweep",
            ["min_scalar_type"] = "scalar-value function: the axis is the VALUE param sweep",
            ["mintypecode"] = "typechar-string function: the axis is the typechars PARAM sweep",

            // PRNG protocol: the stream IS the contract; dtype is the distribution's own.
            ["rnd"] = "seeded RandomState stream bytes (int32/int64/float64 draws are the API's own dtypes)",
            ["seed"] = "seeding protocol case (float64 draw pinned after seeding)",
            ["get_state"] = "state-tuple serialization (text kind) — no operand dtype",
            ["set_state"] = "state-tuple round-trip (uint32 key vector by MT19937 contract)",

            // Host-pinned LAPACK/CBLAS parity family: float32/float64/complex128 are the three
            // dtypes NumPy itself routes through the backend; int/bool ride the managed tiers.
            ["cond"] = "LAPACK family: f32/f64/c128 (linalg_parity host pin)",
            ["corrcoef"] = "normalized cov: f64/c128 (weighted lanes are 1-ULP battle-tests)",
            ["cov"] = "GEMM-bound composition: int64/f64/c128 span the exactness classes",
            ["dot_aat"] = "matmul_parity syrk-shortcut probe: f32/f64/c128 (the BLAS dtypes)",
            ["dot_ata"] = "matmul_parity syrk-shortcut probe: f32/f64/c128",
            ["eig"] = "LAPACK family: f32/f64/c128",
            ["eigh"] = "LAPACK family: f32/f64/c128",
            ["eigvals"] = "LAPACK family: f32/f64/c128",
            ["matmul_aat"] = "matmul_parity syrk-shortcut probe: f32/f64/c128",
            ["matmul_ata"] = "matmul_parity syrk-shortcut probe: f32/f64/c128",
            ["norm"] = "SVD-defined matrix orders: f64/c128 (other orders ride the products tier)",
            ["pinv"] = "LAPACK family: f32/f64/c128",
            ["polyfit"] = "lstsq-backed: f32/f64 (the gelsd dtypes exercised)",
            ["roots"] = "eigvals-backed: f32/f64/c128",
            ["svd"] = "LAPACK family: f32/f64/c128",
            ["svdvals"] = "LAPACK family: f32/f64/c128",
            ["tensorinv"] = "LU-backed: f32/f64/c128",
            ["tensorsolve"] = "LU-backed: f32/f64/c128",

            // Variant keys whose PRIMARY op key carries the full dtype axis.
            ["average_returned"] = "tuple twin of average (arity gate); average itself carries the full axis",
            ["modf"] = "tuple form (arity gate); the per-piece modf_frac/modf_int keys carry 13 dtypes",
            ["std_ddof"] = "ddof=1 variant; std itself carries the full axis",
            ["var_ddof"] = "ddof=1 variant; var itself carries the full axis",

            // Deliberately narrow by op semantics.
            ["einsum"] = "small-EXACT contractions (int64/f64/c128): wider float lanes route through matmul and lose byte-parity",
            ["einsum_path"] = "contraction PLANNER: shape-derived info string, operand dtype irrelevant",
            ["interp"] = "1-D linear interpolation: int64->f64 promotion, f64, and the complex fp lane — NumPy's own loop set",
            ["meshgrid"] = "dtype-PRESERVING coordinate construction: int32/f64/c128 span the width classes",
            ["nested_iters"] = "iteration-ORDER protocol: traversal is dtype-independent (int32/f64/c128 span widths)",
            ["nditer_values"] = "iteration-ORDER protocol: the value stream's ORDER is dtype-independent",
            ["ndarray.getfield"] = "dtype-window accessor: complex128->float64 offsets are the canonical NumPy use (same-size reinterprets ride ndarray.view)",
            ["unique_values"] = "float lanes only: int/complex unique_values is NumPy-hash-order divergent [Misaligned]; the sorted twins gate those",
        };

        /// <summary>
        ///     Per-dtype GLOBAL floors: how many distinct op keys must carry each dtype (gate
        ///     scope: ma_/index_/host files excluded). ~95% of the measured 2026-09-18 spread
        ///     after the dtype-axis widening pass (instance 13 dtypes + char weave, emath
        ///     unsigned lanes, modf all-non-complex, nanquantile integer lanes, out_where
        ///     f16/uint8) — bump after intentional expansions, never lower silently.
        /// </summary>
        private static readonly Dictionary<string, int> DtypeOpFloors = new(StringComparer.Ordinal)
        {
            ["bool"] = 298, ["int8"] = 279, ["uint8"] = 327, ["int16"] = 261, ["uint16"] = 272,
            ["int32"] = 385, ["uint32"] = 261, ["int64"] = 331, ["uint64"] = 261, ["char"] = 216,
            ["float16"] = 299, ["float32"] = 345, ["float64"] = 417, ["decimal"] = 117,
            ["complex128"] = 369,
        };

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void EveryOrdinaryOp_MeetsItsDtypeSpreadFloor()
        {
            var opDtypes = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
            string directory = Path.GetDirectoryName(FuzzCorpus.CorpusPath("unused"));
            foreach (string path in Directory.EnumerateFiles(directory, "*.jsonl"))
            {
                string file = Path.GetFileName(path);
                if (file.EndsWith(".host.jsonl", StringComparison.Ordinal) ||
                    file.StartsWith("index_", StringComparison.Ordinal) ||
                    file.StartsWith("ma_", StringComparison.Ordinal))
                    continue;

                foreach (var c in FuzzCorpus.Load(file))
                {
                    if (string.IsNullOrEmpty(c.Op))
                        continue;
                    if (!opDtypes.TryGetValue(c.Op, out var set))
                        opDtypes[c.Op] = set = new HashSet<string>(StringComparer.Ordinal);
                    bool any = false;
                    foreach (var o in c.Operands ?? Array.Empty<FuzzCorpus.Operand>())
                        if (DtypeOpFloors.ContainsKey(o.Dtype ?? ""))
                        {
                            set.Add(o.Dtype);
                            any = true;
                        }
                    if (!any)
                    {
                        // Zero-operand creation ops: the dtype request/result is the axis.
                        if (c.Params != null && c.Params.TryGetValue("dtype", out var pd)
                            && pd.ValueKind == System.Text.Json.JsonValueKind.String
                            && DtypeOpFloors.ContainsKey(pd.GetString()))
                            set.Add(pd.GetString());
                        else if (c.Expected?.Dtype != null && DtypeOpFloors.ContainsKey(c.Expected.Dtype))
                            set.Add(c.Expected.Dtype);
                    }
                }
            }

            var failures = new List<string>();

            // Rule 1: >= 4 distinct dtypes per op, or an explicit reviewed reason.
            foreach (var kv in opDtypes.OrderBy(k => k.Key, StringComparer.Ordinal))
            {
                if (kv.Value.Count >= 4 || FixedDtypeOps.ContainsKey(kv.Key))
                    continue;
                failures.Add($"{kv.Key}: only {kv.Value.Count} dtypes " +
                             $"[{string.Join(",", kv.Value.OrderBy(x => x, StringComparer.Ordinal))}] " +
                             "— widen the generator's dtype axis or add a reviewed FixedDtypeOps reason");
            }

            // Ledger hygiene: entries self-retire when the op vanishes or outgrows the floor.
            foreach (var kv in FixedDtypeOps)
            {
                if (!opDtypes.TryGetValue(kv.Key, out var set))
                    failures.Add($"stale FixedDtypeOps entry: '{kv.Key}' has no corpus cases");
                else if (set.Count >= 4)
                    failures.Add($"stale FixedDtypeOps entry: '{kv.Key}' now spreads {set.Count} dtypes — retire the reason (\"{kv.Value}\")");
            }

            // Rule 2: per-dtype global ratchets.
            foreach (var kv in DtypeOpFloors)
            {
                int count = opDtypes.Values.Count(s => s.Contains(kv.Key));
                if (count < kv.Value)
                    failures.Add($"dtype '{kv.Key}' spread regressed: {count} ops carry it, floor {kv.Value} " +
                                 "(a regeneration dropped a dtype axis?)");
            }

            Console.WriteLine($"[DtypeSpread] ops={opDtypes.Count}, fixed_dtype_ledger={FixedDtypeOps.Count}, " +
                              "per-dtype=" + string.Join(" ", DtypeOpFloors.Keys.Select(d =>
                                  $"{d}:{opDtypes.Values.Count(s => s.Contains(d))}")));

            if (failures.Count > 0)
                Assert.Fail($"{failures.Count} dtype-spread failures:\n  " + string.Join("\n  ", failures));
        }

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void AdvancedIndexingCorpora_KeepTheirMatrixFloors()
        {
            var floors = new Dictionary<string, int>(StringComparer.Ordinal)
            {
                ["index_curated.jsonl"] = 2_000,
                ["index_dtype.jsonl"] = 100,
                ["index_setter_dtype.jsonl"] = 10,
                ["index_random_20240626.jsonl"] = 10_000,
            };

            foreach (var pair in floors)
            {
                int count = File.ReadLines(FuzzCorpus.CorpusPath(pair.Key)).Count(line =>
                    !string.IsNullOrWhiteSpace(line));
                Assert.IsTrue(count >= pair.Value,
                    $"{pair.Key} has {count} cases; expected at least {pair.Value}");
            }
        }

        private sealed class Strength
        {
            public int Count;
            public readonly HashSet<string> Layouts = new(StringComparer.Ordinal);
            public readonly HashSet<string> ValueClasses = new(StringComparer.Ordinal);
            public readonly HashSet<string> DtypeSignatures = new(StringComparer.Ordinal);
            public readonly HashSet<string> ParameterSignatures = new(StringComparer.Ordinal);
            public readonly HashSet<string> Outcomes = new(StringComparer.Ordinal);

            public bool HasVariation => Layouts.Count > 1 || ValueClasses.Count > 1 ||
                                        DtypeSignatures.Count > 1 || ParameterSignatures.Count > 1 ||
                                        Outcomes.Count > 1;
        }
    }
}
