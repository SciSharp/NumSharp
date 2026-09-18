using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The APPLICABILITY-MATRIX gate (coverage plan §5/M1–M3): the declarative record of which
    ///     assertion-kinds each taxonomy group of ops is expected to carry, diffed against what the
    ///     committed corpus ACTUALLY carries — so "which cells are still empty" is a gate, not a
    ///     memory, and a filled cell can never silently drain.
    ///
    ///     <para><b>Model.</b> Coverage is a product, not a list: op keys × assertion-kinds ×
    ///     parameters × dtype/layout. The surface gates prove an op is NAMED and the strength gate
    ///     that it has a small matrix; this gate proves each GROUP is asserted in every MODE that
    ///     applies to it — value bytes, error message parity, out=/where= two-slot, the NaN grid,
    ///     IEEE specials, 0-d operands, tuple arity, in-place mutation.</para>
    ///
    ///     <para><b>Cell status.</b> <see cref="CellStatus.Enforced"/> asserts a minimum op-count
    ///     floor (like MinCases: filled cells only ratchet up); <see cref="CellStatus.Warn"/>
    ///     prints the missing ops — the driven checklist for the next expansion session — without
    ///     failing; <see cref="CellStatus.NotApplicable"/> REQUIRES a reason and fails if coverage
    ///     appears anyway (a stale N/A must be reclassified, mirroring MisalignedRegistry's
    ///     explicit-excuse rule). Kinds a row does not mention at all are structurally
    ///     inapplicable by the row's shape; each row's <c>InapplicableNote</c> says why in one
    ///     line, so silence is a reviewed statement.</para>
    /// </summary>
    [TestClass]
    public class OracleApplicabilityTests
    {
        /// <summary>How strongly a manifest cell is held.</summary>
        private enum CellStatus
        {
            /// <summary>Filled: at least <see cref="Cell.MinOps"/> of the row's ops carry the kind — red below the floor.</summary>
            Enforced,

            /// <summary>Applicable but not (fully) filled: the gate PRINTS the missing ops; never red.</summary>
            Warn,

            /// <summary>Reviewed as not applicable — reason required; coverage appearing anyway is red (stale N/A).</summary>
            NotApplicable,
        }

        /// <summary>One (assertion-kind, expectation) cell of a row.</summary>
        /// <param name="Kind">The applied-kind token computed from the corpus (see <see cref="AppliedKinds"/>).</param>
        /// <param name="Status">How the cell is held.</param>
        /// <param name="MinOps">Enforced floor: how many of the row's ops must carry the kind (default 1).</param>
        /// <param name="Reason">Required for <see cref="CellStatus.NotApplicable"/> — the one-line review.</param>
        private sealed record Cell(string Kind, CellStatus Status, int MinOps = 1, string Reason = null);

        /// <summary>One taxonomy-group row of the manifest.</summary>
        /// <param name="Group">Display name (the plan's G-number + label).</param>
        /// <param name="Ops">Explicit corpus op keys in the group (null when <paramref name="Prefix"/> drives membership).</param>
        /// <param name="Cells">The applicable cells, each with its status.</param>
        /// <param name="Prefix">Optional op-key prefix ("ma.", "ndarray.", "emath.") that defines membership instead of <paramref name="Ops"/>.</param>
        /// <param name="InapplicableNote">One line covering every kind the row does NOT mention (the M3 rule at row granularity).</param>
        private sealed record Row(string Group, string[] Ops, Cell[] Cells, string Prefix = null,
                                  string InapplicableNote = null);

        // ---- the manifest -------------------------------------------------------------------

        private static readonly Row[] Manifest =
        {
            new("G1 unary math",
                new[]
                {
                    "negative", "abs", "fabs", "sign", "sqrt", "cbrt", "square", "reciprocal",
                    "floor", "ceil", "trunc", "rint", "exp", "log", "sin", "cos", "tan", "exp2",
                    "expm1", "log2", "log10", "log1p", "sinh", "cosh", "tanh", "arcsin", "arccos",
                    "arctan", "arcsinh", "arccosh", "arctanh", "deg2rad", "rad2deg", "positive",
                },
                new[]
                {
                    new Cell("array", CellStatus.Enforced, MinOps: 34),
                    new Cell("zerod", CellStatus.Enforced, MinOps: 30),
                    new Cell("outwhere", CellStatus.Enforced, MinOps: 10),
                    new Cell("nan", CellStatus.Enforced, MinOps: 25),
                    new Cell("specials", CellStatus.Enforced, MinOps: 30),
                    new Cell("error", CellStatus.Enforced, MinOps: 5),
                    new Cell("precision", CellStatus.Warn),
                },
                InapplicableNote: "single-array float ops: no tuple/dtype/text result, no in-place form"),

            new("G2 binary arithmetic",
                new[]
                {
                    "add", "subtract", "multiply", "divide", "floor_divide", "mod", "fmod",
                    "power", "float_power", "gcd", "lcm", "logaddexp", "logaddexp2", "hypot",
                    "arctan2", "copysign", "nextafter", "heaviside",
                },
                new[]
                {
                    new Cell("array", CellStatus.Enforced, MinOps: 18),
                    new Cell("outwhere", CellStatus.Enforced, MinOps: 14),
                    new Cell("nan", CellStatus.Enforced, MinOps: 15),   // §B3: the binary NaN cross-grid (gcd/lcm are integer-only — no NaN; float_power rides power's float lanes)
                    new Cell("specials", CellStatus.Enforced, MinOps: 8),
                    new Cell("error", CellStatus.Enforced, MinOps: 4),
                    new Cell("tuple", CellStatus.Warn),                 // divmod lives in G-multi (tuple) — here only via out=
                    new Cell("precision", CellStatus.Warn),
                },
                InapplicableNote: "binary elementwise: no dtype/text result; in-place is the aliasing tier's out=an-input form"),

            new("G3 comparison / logic",
                new[]
                {
                    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
                    "isclose", "allclose", "logical_and", "logical_or", "logical_xor",
                    "logical_not", "maximum", "minimum", "fmax", "fmin", "isnan", "isinf",
                    "isfinite", "signbit",
                },
                new[]
                {
                    new Cell("array", CellStatus.Enforced, MinOps: 20),
                    new Cell("outwhere", CellStatus.Enforced, MinOps: 8),
                    new Cell("specials", CellStatus.Enforced, MinOps: 12),
                    new Cell("nan", CellStatus.Enforced, MinOps: 4),
                    new Cell("error", CellStatus.Warn),
                    new Cell("zerod", CellStatus.Enforced, MinOps: 10),
                },
                InapplicableNote: "bool-result predicates/comparisons: no tuple/dtype/text result, no in-place form, no precision axis"),

            new("G4 bitwise / shift",
                new[]
                {
                    "bitwise_and", "bitwise_or", "bitwise_xor", "invert", "left_shift",
                    "right_shift", "bitwise_count",
                },
                new[]
                {
                    new Cell("array", CellStatus.Enforced, MinOps: 7),
                    new Cell("outwhere", CellStatus.Enforced, MinOps: 3),
                    new Cell("error", CellStatus.Enforced, MinOps: 4),
                    new Cell("zerod", CellStatus.Enforced, MinOps: 5),
                },
                InapplicableNote: "integer-only ops: NaN/specials/precision have no meaning; no tuple/dtype/text/in-place"),

            new("G5 reductions",
                new[]
                {
                    "sum", "prod", "min", "max", "mean", "std", "var", "argmax", "argmin", "all",
                    "any", "median", "average", "ptp", "count_nonzero", "percentile", "quantile",
                },
                new[]
                {
                    new Cell("array", CellStatus.Enforced, MinOps: 17),
                    new Cell("error", CellStatus.Enforced, MinOps: 7),
                    new Cell("specials", CellStatus.Enforced, MinOps: 8),
                    new Cell("precision", CellStatus.Enforced, MinOps: 3),
                    new Cell("zerod", CellStatus.Enforced, MinOps: 10),
                    new Cell("multiaxis", CellStatus.Enforced, MinOps: 2),  // §C1: median/average int[]-axis cells
                    // §B2 residue: the sum/prod/min/max family has NO out= overload in NumSharp
                    // yet — a FEATURE gap, so the cell warns (out_nanarg gates the arg family).
                    new Cell("outwhere", CellStatus.Warn),
                },
                InapplicableNote: "reductions: no text/dtype result; the tuple forms (average_returned) live in multioutput"),

            new("G5b scans",
                new[] { "cumsum", "cumprod", "nancumsum", "nancumprod", "diff", "ediff1d", "unwrap" },
                new[]
                {
                    new Cell("array", CellStatus.Enforced, MinOps: 7),
                    new Cell("outwhere", CellStatus.Enforced, MinOps: 2),  // §B2: cumsum/cumprod out= (out_scan)
                    new Cell("zerod", CellStatus.Warn),
                    new Cell("error", CellStatus.Warn),
                },
                InapplicableNote: "prefix scans: NaN handling IS the nanscan tier's pools; no tuple/dtype/text/in-place"),

            new("G6 NaN reductions",
                new[]
                {
                    "nansum", "nanprod", "nanmax", "nanmin", "nanmean", "nanstd", "nanvar",
                    "nanmedian", "nanargmax", "nanargmin", "nanpercentile", "nanquantile",
                },
                new[]
                {
                    new Cell("array", CellStatus.Enforced, MinOps: 12),
                    new Cell("error", CellStatus.Warn),
                    new Cell("specials", CellStatus.Enforced, MinOps: 4),
                    new Cell("outwhere", CellStatus.Enforced, MinOps: 2),   // §B2: nanargmax/nanargmin out=
                    new Cell("multiaxis", CellStatus.Enforced, MinOps: 1),  // §C1: nanmedian int[]-axis
                },
                InapplicableNote: "NaN-ignoring reductions: the whole tier IS the NaN axis; no tuple/dtype/text/in-place"),

            new("G7 manipulation",
                new[]
                {
                    "reshape", "transpose", "ravel", "squeeze", "expand_dims", "swapaxes",
                    "moveaxis", "roll", "repeat", "tile", "flip", "fliplr", "flipud", "rot90",
                    "concatenate", "stack", "delete", "insert", "append", "pad", "broadcast_to",
                },
                new[]
                {
                    new Cell("array", CellStatus.Enforced, MinOps: 21),
                    new Cell("error", CellStatus.Enforced, MinOps: 4),
                    new Cell("zerod", CellStatus.Enforced, MinOps: 8),
                },
                InapplicableNote: "pure reindex/copy: value axes (NaN/specials/precision/outwhere) do not apply; the in-place forms are G0's"),

            new("G9 products",
                new[]
                {
                    "matmul", "dot", "inner", "vdot", "vecdot", "matvec", "vecmat", "outer",
                    "tensordot", "kron", "cross", "trace", "einsum",
                },
                new[]
                {
                    new Cell("array", CellStatus.Enforced, MinOps: 13),
                    new Cell("specials", CellStatus.Enforced, MinOps: 3),
                    new Cell("precision", CellStatus.Enforced, MinOps: 2),
                    new Cell("error", CellStatus.Warn),
                },
                InapplicableNote: "contractions: NaN-grid rides specials; no tuple/text; out= is a feature gap tracked in the plan"),

            new("G10 factorizations",
                new[]
                {
                    "cholesky", "eig", "eigh", "eigvals", "eigvalsh", "svd", "svdvals", "qr",
                    "lstsq", "pinv", "matrix_rank", "cond", "norm", "solve", "inv", "det",
                    "slogdet",
                },
                new[]
                {
                    new Cell("array", CellStatus.Enforced, MinOps: 10),
                    new Cell("tuple", CellStatus.Enforced, MinOps: 5),
                    new Cell("error", CellStatus.Enforced, MinOps: 3),
                },
                InapplicableNote: "host-pinned LAPACK family: value axes beyond bytes+arity are the interop live-parity suite's"),

            new("G11 selection / scatter",
                new[]
                {
                    "take", "put", "place", "putmask", "select", "compress", "extract", "choose",
                    "where", "copyto", "take_along_axis", "put_along_axis",
                },
                new[]
                {
                    new Cell("array", CellStatus.Enforced, MinOps: 12),
                    new Cell("error", CellStatus.Enforced, MinOps: 2),
                    new Cell("inplace", CellStatus.Enforced, MinOps: 2),   // put/putmask mutate their operand
                },
                InapplicableNote: "gather/scatter: index bounds are the error cells; masks make NaN/specials inapplicable; no tuple/text"),

            new("G12 sort / search",
                new[]
                {
                    "sort", "argsort", "partition", "argpartition", "lexsort", "searchsorted",
                    "nonzero", "argwhere", "flatnonzero", "unique", "digitize", "bincount",
                    "sort_complex",
                },
                new[]
                {
                    new Cell("array", CellStatus.Enforced, MinOps: 13),
                    new Cell("error", CellStatus.Enforced, MinOps: 2),
                },
                InapplicableNote: "orderings: NaN placement is pinned inside the sort tiers' pools; the in-place "
                                + "spelling is G0's ndarray.sort; no out=/tuple/text axes"),

            new("G13 dtype / promotion",
                new[]
                {
                    "result_type_arrays", "result_type_dtypes", "promote_types",
                    "min_scalar_type", "can_cast", "dtype", "common_type", "issubdtype", "isdtype",
                },
                new[]
                {
                    new Cell("dtype", CellStatus.Enforced, MinOps: 5),
                    new Cell("scalar", CellStatus.Enforced, MinOps: 3),
                    new Cell("error", CellStatus.Warn),
                },
                InapplicableNote: "promotion helpers: results are dtypes/bools, so every array-value axis is out of scope"),

            new("G14 printing",
                new[] { "array_str", "array_repr", "savetxt", "mintypecode" },
                new[]
                {
                    new Cell("text", CellStatus.Enforced, MinOps: 4),
                },
                InapplicableNote: "text results: byte-exact string comparison IS the whole contract"),

            new("G15 iteration",
                new[]
                {
                    "nditer", "nditer_values", "nditer_index", "nditer_multi_index",
                    "nditer_extloop", "nditer_pair", "ndindex", "ndenumerate", "nested_iters",
                    "broadcast_values", "broadcast_shape",
                },
                new[]
                {
                    new Cell("array", CellStatus.Enforced, MinOps: 5),
                    new Cell("tuple", CellStatus.Enforced, MinOps: 4),
                    new Cell("error", CellStatus.Enforced, MinOps: 2),
                },
                InapplicableNote: "materialized traversal ORDER is the artifact; value axes belong to the ops iterated"),

            new("G16 FFT",
                new[]
                {
                    "fft", "ifft", "rfft", "irfft", "hfft", "ihfft", "fft2", "ifft2", "fftn",
                    "ifftn", "rfft2", "irfft2", "rfftn", "irfftn", "fftfreq", "rfftfreq",
                    "fftshift", "ifftshift",
                },
                new[]
                {
                    new Cell("array", CellStatus.Enforced, MinOps: 18),
                    new Cell("error", CellStatus.Enforced, MinOps: 2),
                    // np.fft out= exists in NumPy 2.x; NumSharp's fft carries out= on the 1-D core —
                    // corpus cells are a tracked expansion (plan §B2 tail).
                    new Cell("outwhere", CellStatus.Warn),
                },
                InapplicableNote: "spectral transforms: NaN/specials propagate trivially (inputs are finite pools); no tuple/text"),

            new("G17 random",
                new[] { "rnd", "grnd" },
                new[]
                {
                    new Cell("array", CellStatus.Enforced, MinOps: 2),
                    new Cell("text", CellStatus.Enforced, MinOps: 0),   // get_state string rides the text kind
                },
                InapplicableNote: "seeded stream bytes ARE the contract; every other axis is meaningless for a PRNG draw"),

            new("G18 evaluate",
                new[] { "evaluate" },
                new[]
                {
                    new Cell("array", CellStatus.Enforced),
                    new Cell("tuple", CellStatus.Enforced),   // out= two-slot cells
                    new Cell("error", CellStatus.Enforced),
                },
                InapplicableNote: "fused trees: the node-level axes are gated through the ops each node mirrors"),

            new("G19 polynomial",
                new[]
                {
                    "poly", "polyval", "vander", "polyder", "polyint", "polyadd", "polysub",
                    "polymul", "polydiv", "poly1d_coeffs", "poly1d_fromroots",
                },
                new[]
                {
                    new Cell("array", CellStatus.Enforced, MinOps: 10),
                    new Cell("tuple", CellStatus.Enforced, MinOps: 1),   // polydiv (q, r)
                },
                InapplicableNote: "portable arithmetic compositions; roots/polyfit ride the host-pinned G10 tier"),

            new("G0 ndarray instance",
                null,
                new[]
                {
                    new Cell("array", CellStatus.Enforced, MinOps: 30),
                    new Cell("scalar", CellStatus.Enforced, MinOps: 5),   // item/__len__/nbytes/itemsize/ndim/size
                    new Cell("tuple", CellStatus.Enforced, MinOps: 4),    // nonzero + the in-place two-slot mutators
                    new Cell("inplace", CellStatus.Enforced, MinOps: 4),  // sort/fill/put/resize
                    new Cell("error", CellStatus.Enforced, MinOps: 3),    // item(size>1), len(0-d), mT(<2-d)
                    new Cell("zerod", CellStatus.Enforced, MinOps: 20),
                },
                Prefix: "ndarray.",
                InapplicableNote: "instance spellings of ops whose value axes are gated on the np.* twin rows"),

            new("G20 emath",
                null,
                new[]
                {
                    new Cell("array", CellStatus.Enforced, MinOps: 9),
                    new Cell("zerod", CellStatus.Enforced, MinOps: 5),
                },
                Prefix: "emath.",
                InapplicableNote: "scimath: the promotion DECISION is the contract; complex64 lanes are sibling-owned (#569)"),

            new("G21 masked arrays",
                null,
                new[]
                {
                    new Cell("array", CellStatus.Enforced, MinOps: 100),
                    new Cell("error", CellStatus.Warn),
                },
                Prefix: "ma.",
                InapplicableNote: "masked results are (filled(0), mask) pairs with their own comparator; other axes ride the np.* twins"),
        };

        // ---- applied-kind computation -------------------------------------------------------

        /// <summary>Ops whose corpus cases MUTATE their first operand (the in-place kind).</summary>
        private static readonly HashSet<string> InplaceOps = new(StringComparer.Ordinal)
        {
            "put", "putmask", "place", "fill_diagonal", "put_along_axis", "copyto",
            "ndarray.sort", "ndarray.fill", "ndarray.put", "ndarray.resize",
        };

        /// <summary>
        ///     Scan the whole committed corpus once and compute, per op key, the set of applied
        ///     assertion-kind tokens: <c>array</c>/<c>scalar</c>/<c>dtype</c>/<c>text</c>/
        ///     <c>tuple</c> from the recorded result kind, <c>error</c> for raising cells,
        ///     <c>outwhere</c>/<c>nan</c>/<c>specials</c>/<c>precision</c> from the tier that
        ///     carries the case, <c>zerod</c> for any case with a 0-d operand, <c>multiaxis</c>
        ///     for an int[]-axis ("axes") param, and <c>inplace</c> from <see cref="InplaceOps"/>.
        /// </summary>
        /// <returns>op key → applied kind tokens.</returns>
        private static Dictionary<string, HashSet<string>> AppliedKinds()
        {
            var applied = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
            string directory = Path.GetDirectoryName(FuzzCorpus.CorpusPath("unused"));
            foreach (string path in Directory.EnumerateFiles(directory, "*.jsonl"))
            {
                string file = Path.GetFileName(path);
                if (file.EndsWith(".host.jsonl", StringComparison.Ordinal) ||
                    file.StartsWith("index_", StringComparison.Ordinal))
                    continue;   // host pins carry no cases; the index oracle has its own schema

                foreach (var c in FuzzCorpus.Load(file))
                {
                    if (string.IsNullOrEmpty(c.Op))
                        continue;
                    if (!applied.TryGetValue(c.Op, out var kinds))
                        applied[c.Op] = kinds = new HashSet<string>(StringComparer.Ordinal);

                    // The masked-array comparator kinds are the (filled, mask) spelling of the
                    // ordinary array/tuple contract — normalize so group floors see them.
                    string kind = c.Error != null || c.Expects_Throw ? "error"
                        : c.Expected?.KindOrArray ?? "array";
                    kinds.Add(kind switch { "masked" => "array", "masked_tuple" => "tuple", _ => kind });

                    if (file == "out_where.jsonl" || c.Valueclass == "outwhere")
                    {
                        kinds.Add("outwhere");
                        // The out_* vehicle keys (out_binary/out_unary/out_scan/out_nanarg) carry
                        // the REAL ufunc in params — credit that op so the group rows (whose keys
                        // are the ufunc names) see their out=/where= coverage.
                        if (c.Params != null && c.Params.TryGetValue("ufunc", out var uf)
                            && uf.ValueKind == System.Text.Json.JsonValueKind.String)
                        {
                            string target = uf.GetString();
                            if (!applied.TryGetValue(target, out var tk))
                                applied[target] = tk = new HashSet<string>(StringComparer.Ordinal);
                            tk.Add("outwhere");
                        }
                        // out_clip has no ufunc param; its target op is clip itself.
                        if (c.Op == "out_clip")
                        {
                            if (!applied.TryGetValue("clip", out var ck))
                                applied["clip"] = ck = new HashSet<string>(StringComparer.Ordinal);
                            ck.Add("outwhere");
                        }
                    }
                    if (file == "nan.jsonl")
                        kinds.Add("nan");
                    if (file == "specials.jsonl")
                        kinds.Add("specials");
                    if (c.Expected?.Truth != null)
                        kinds.Add("precision");
                    if (c.Params != null && c.Params.ContainsKey("axes")
                        && (c.Op is "median" or "average" or "nanmedian"))
                        kinds.Add("multiaxis");
                    if (InplaceOps.Contains(c.Op))
                        kinds.Add("inplace");
                    if (c.Operands != null && c.Operands.Any(o => o.Shape != null && o.Shape.Length == 0))
                        kinds.Add("zerod");
                    if (c.Layout != null && c.Layout.Contains("scalar_0d"))
                        kinds.Add("zerod");
                }
            }
            return applied;
        }

        // ---- the gate -----------------------------------------------------------------------

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void ApplicabilityMatrix_EnforcedCellsHold_WarnCellsPrint()
        {
            var applied = AppliedKinds();
            var failures = new List<string>();
            var warnings = new List<string>();

            foreach (var row in Manifest)
            {
                string[] ops = row.Prefix != null
                    ? applied.Keys.Where(k => k.StartsWith(row.Prefix, StringComparison.Ordinal)).ToArray()
                    : row.Ops;

                // Manifest hygiene: an explicit op that no longer exists in the corpus is a drift
                // the row must not silently absorb (the op was renamed/retired — update the row).
                if (row.Prefix == null)
                    foreach (string op in ops)
                        if (!applied.ContainsKey(op))
                            failures.Add($"{row.Group}: manifest op '{op}' has no corpus cases at all");

                foreach (var cell in row.Cells)
                {
                    var covered = ops.Where(op => applied.TryGetValue(op, out var k) && k.Contains(cell.Kind))
                                     .ToArray();
                    switch (cell.Status)
                    {
                        case CellStatus.Enforced:
                            if (covered.Length < cell.MinOps)
                                failures.Add($"{row.Group} × {cell.Kind}: enforced floor {cell.MinOps}, " +
                                             $"only {covered.Length} ops covered — missing: " +
                                             string.Join(", ", ops.Except(covered).Take(12)));
                            break;
                        case CellStatus.Warn:
                            if (covered.Length < ops.Length)
                                warnings.Add($"{row.Group} × {cell.Kind}: {covered.Length}/{ops.Length} ops — " +
                                             $"empty for: {string.Join(", ", ops.Except(covered).Take(12))}");
                            break;
                        case CellStatus.NotApplicable:
                            if (cell.Reason == null)
                                failures.Add($"{row.Group} × {cell.Kind}: NotApplicable requires a reason (M3)");
                            else if (covered.Length > 0)
                                failures.Add($"{row.Group} × {cell.Kind}: marked N/A (\"{cell.Reason}\") but " +
                                             $"{covered.Length} ops now carry it — reclassify the cell");
                            break;
                    }
                }
            }

            // Visibility (never a failure): corpus ops assigned to no manifest row — candidates
            // for the next taxonomy pass, printed so the matrix's blind spot is explicit.
            var assigned = new HashSet<string>(StringComparer.Ordinal);
            foreach (var row in Manifest)
            {
                if (row.Prefix != null)
                    assigned.UnionWith(applied.Keys.Where(k => k.StartsWith(row.Prefix, StringComparison.Ordinal)));
                else
                    assigned.UnionWith(row.Ops);
            }
            var unassigned = applied.Keys.Except(assigned).OrderBy(x => x, StringComparer.Ordinal).ToArray();

            Console.WriteLine($"[Applicability] rows={Manifest.Length}, corpus_ops={applied.Count}, " +
                              $"unassigned_ops={unassigned.Length}");
            foreach (var w in warnings)
                Console.WriteLine("  WARN " + w);
            if (unassigned.Length > 0)
                Console.WriteLine("  unassigned: " + string.Join(", ", unassigned.Take(60)) +
                                  (unassigned.Length > 60 ? $" (+{unassigned.Length - 60} more)" : ""));

            if (failures.Count > 0)
                Assert.Fail($"{failures.Count} applicability-matrix failures:\n  " +
                            string.Join("\n  ", failures));
        }
    }
}
