using System;
using System.Collections.Generic;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     Maps a corpus op-name to the NumSharp call that produces the operand result.
    ///     The matching NumPy call lives in test/oracle/gen_oracle.py; this is the C# side of that pair.
    ///     New op tiers (binary arith, comparison, unary, reductions, where/place) extend this switch.
    /// </summary>
    public static partial class OpRegistry
    {
        public static NDArray Apply(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            switch (op)
            {
                case "astype":
                    return ops[0].astype(FuzzCorpus.DtypeToTC(p["dtype"].GetString()));

                // Binary arithmetic (NEP50 promotion). NumPy is the oracle for the result dtype.
                case "add": return ops[0] + ops[1];
                case "subtract": return ops[0] - ops[1];
                case "multiply": return ops[0] * ops[1];
                case "divide": return ops[0] / ops[1];
                case "floor_divide": return np.floor_divide(ops[0], ops[1]);
                case "mod": return np.mod(ops[0], ops[1]);
                case "power": return np.power(ops[0], ops[1]);

                // Unary.
                case "negative": return np.negative(ops[0]);
                case "abs": return np.abs(ops[0]);
                case "sign": return np.sign(ops[0]);
                case "sqrt": return np.sqrt(ops[0]);
                case "cbrt": return np.cbrt(ops[0]);
                case "square": return np.square(ops[0]);
                case "reciprocal": return np.reciprocal(ops[0]);
                case "floor": return np.floor(ops[0]);
                case "ceil": return np.ceil(ops[0]);
                case "trunc": return np.trunc(ops[0]);
                case "rint": return np.rint(ops[0]);
                case "sin": return np.sin(ops[0]);
                case "cos": return np.cos(ops[0]);
                case "tan": return np.tan(ops[0]);
                case "exp": return np.exp(ops[0]);
                case "log": return np.log(ops[0]);

                // Unary stragglers (W3): transcendental / hyperbolic / inverse-trig / angle conv.
                case "exp2": return np.exp2(ops[0]);
                case "expm1": return np.expm1(ops[0]);
                case "log2": return np.log2(ops[0]);
                case "log10": return np.log10(ops[0]);
                case "log1p": return np.log1p(ops[0]);
                case "sinh": return np.sinh(ops[0]);
                case "cosh": return np.cosh(ops[0]);
                case "tanh": return np.tanh(ops[0]);
                case "arcsin": return np.arcsin(ops[0]);
                case "arccos": return np.arccos(ops[0]);
                case "arctan": return np.arctan(ops[0]);
                case "arcsinh": return np.arcsinh(ops[0]);
                case "arccosh": return np.arccosh(ops[0]);
                case "arctanh": return np.arctanh(ops[0]);
                case "deg2rad": return np.deg2rad(ops[0]);
                case "rad2deg": return np.rad2deg(ops[0]);
                case "positive": return np.positive(ops[0]);

                // Bitwise & shift (T9). Integer + bool dtypes; NumPy is the oracle.
                case "bitwise_and": return ops[0] & ops[1];
                case "bitwise_or": return ops[0] | ops[1];
                case "bitwise_xor": return ops[0] ^ ops[1];
                case "invert": return np.invert(ops[0]);
                case "left_shift": return np.left_shift(ops[0], ops[1]);
                case "right_shift": return np.right_shift(ops[0], ops[1]);

                // Comparison -> bool result.
                case "equal": return ops[0] == ops[1];
                case "not_equal": return ops[0] != ops[1];
                case "less": return ops[0] < ops[1];
                case "greater": return ops[0] > ops[1];
                case "less_equal": return ops[0] <= ops[1];
                case "greater_equal": return ops[0] >= ops[1];

                // Shape manipulation (T7).
                case "ravel": return np.ravel(ops[0]);
                case "transpose": return np.transpose(ops[0]);
                case "expand_dims": return np.expand_dims(ops[0], p["axis"].GetInt32());
                case "squeeze": return np.squeeze(ops[0]);
                case "roll": return np.roll(ops[0], p["shift"].GetInt32());
                case "repeat": return np.repeat(ops[0], p["repeats"].GetInt32());
                case "tile": return np.tile(ops[0], p["reps"].GetInt32());
                case "reshape": return np.reshape(ops[0], ParseIntArray(p["shape"]));
                case "swapaxes": return np.swapaxes(ops[0], p["a1"].GetInt32(), p["a2"].GetInt32());
                case "moveaxis": return np.moveaxis(ops[0], p["src"].GetInt32(), p["dst"].GetInt32());
                case "rot90": return np.rot90(ops[0], p["k"].GetInt32(),
                    p.ContainsKey("axes") ? ParseIntArray(p["axes"]) : null);
                // flip family + transpose aliases + trim_zeros. "axes" -> int[] overload, "axis" -> int overload.
                case "flip":
                    if (p.ContainsKey("axes")) return np.flip(ops[0], ParseIntArray(p["axes"]));
                    return p.ContainsKey("axis") ? np.flip(ops[0], p["axis"].GetInt32()) : np.flip(ops[0]);
                case "fliplr": return np.fliplr(ops[0]);
                case "flipud": return np.flipud(ops[0]);
                // ndarray.byteswap (no np.byteswap) — not-inplace returns a fresh byte-swapped copy.
                case "byteswap": return ops[0].byteswap();
                case "permute_dims": return np.permute_dims(ops[0],
                    p.ContainsKey("axes") ? ParseIntArray(p["axes"]) : null);
                case "matrix_transpose": return np.matrix_transpose(ops[0]);
                case "trim_zeros":
                    if (p.ContainsKey("axes")) return np.trim_zeros(ops[0], p["trim"].GetString(), ParseIntArray(p["axes"]));
                    return p.ContainsKey("axis")
                        ? np.trim_zeros(ops[0], p["trim"].GetString(), p["axis"].GetInt32())
                        : np.trim_zeros(ops[0], p["trim"].GetString());
                case "delete": return np.delete(ops[0], p["obj"].GetInt32(), p["axis"].GetInt32());
                case "atleast_1d": return np.atleast_1d(ops[0]);
                case "atleast_2d": return np.atleast_2d(ops[0]);
                case "atleast_3d": return np.atleast_3d(ops[0]);
                case "concatenate": return np.concatenate((ops[0], ops[1]), p["axis"].GetInt32());
                case "stack": return np.stack(new[] { ops[0], ops[1] }, p["axis"].GetInt32());
                case "hstack": return np.hstack(ops[0], ops[1]);
                case "vstack": return np.vstack(ops[0], ops[1]);
                case "dstack": return np.dstack(ops[0], ops[1]);
                case "pad": return np.pad(ops[0], p["pad_width"].GetInt32(), p["mode"].GetString());

                // Multi-output (T15): modf split into its two outputs (fractional / integral).
                case "modf_frac": return np.modf(ops[0]).Item1;
                case "modf_int": return np.modf(ops[0]).Item2;

                // Cumulative scans + finite differences (T11).
                case "cumsum": return np.cumsum(ops[0], ParseAxis(p));
                case "cumprod": return np.cumprod(ops[0], ParseAxis(p));
                case "diff": return np.diff(ops[0], p["n"].GetInt32(), p["axis"].GetInt32());

                // In-place out= aliasing (W11): the output buffer IS an input operand.
                case "maximum_out": np.maximum(ops[0], ops[1], ops[0]); return ops[0];
                case "minimum_out": np.minimum(ops[0], ops[1], ops[0]); return ops[0];
                case "clip_out": np.clip(ops[0], ops[1], ops[2], ops[0]); return ops[0];

                // Parameter sweep (W12): ddof=1 std/var, order='F' ravel.
                case "std_ddof": { var ax = ParseAxis(p); int dd = p["ddof"].GetInt32();
                    return ax.HasValue ? np.std(ops[0], ax.Value, false, dd) : np.std(ops[0], false, dd); }
                case "var_ddof": { var ax = ParseAxis(p); int dd = p["ddof"].GetInt32();
                    return ax.HasValue ? np.var(ops[0], ax.Value, false, dd) : np.var(ops[0], false, dd); }
                case "ravel_f": return np.ravel(ops[0], 'F');

                // Statistics (T12).
                case "median": return np.median(ops[0], ParseAxis(p), keepdims: ParseKeepdims(p));
                case "average": return np.average(ops[0], ParseAxis(p), null, ParseKeepdims(p));
                case "ptp": return np.ptp(ops[0], ParseAxis(p), null, ParseKeepdims(p));
                case "count_nonzero": return np.count_nonzero(ops[0], ParseAxis(p).Value, ParseKeepdims(p));
                case "percentile": return np.percentile(ops[0], p["q"].GetDouble(), ParseAxis(p),
                    keepdims: ParseKeepdims(p));
                case "quantile": return np.quantile(ops[0], p["q"].GetDouble(), ParseAxis(p),
                    keepdims: ParseKeepdims(p));
                case "clip": return np.clip(ops[0], ops[1], ops[2]);

                // Logic & element-wise extrema (T13).
                case "isnan": return np.isnan(ops[0]);
                case "isinf": return np.isinf(ops[0]);
                case "isfinite": return np.isfinite(ops[0]);
                case "maximum": return np.maximum(ops[0], ops[1]);
                case "minimum": return np.minimum(ops[0], ops[1]);
                case "fmax": return np.fmax(ops[0], ops[1]);
                case "fmin": return np.fmin(ops[0], ops[1]);
                case "isclose": return np.isclose(ops[0], ops[1]);

                // Group A Batch 1: boolean logic + binary arctan2.
                case "logical_and": return np.logical_and(ops[0], ops[1]);
                case "logical_or": return np.logical_or(ops[0], ops[1]);
                case "logical_xor": return np.logical_xor(ops[0], ops[1]);
                case "logical_not": return np.logical_not(ops[0]);
                case "arctan2": return np.arctan2(ops[0], ops[1]);

                // Group A Batch 3: predicates + whole-array bool reductions (wrapped to 0-D bool).
                case "iscomplex": return np.iscomplex(ops[0]);
                case "isreal": return np.isreal(ops[0]);
                case "allclose": return NDArray.Scalar(np.allclose(ops[0], ops[1]));
                case "array_equal": return NDArray.Scalar(np.array_equal(ops[0], ops[1]));

                // Selection.
                case "where": return np.where(ops[0], ops[1], ops[2]);
                case "place": np.place(ops[0], ops[1], ops[2]); return ops[0]; // mutates arr; result IS arr

                // select — operands are [cond0..cond_{nc-1}, choice0..choice_{nc-1}, default];
                // params "nc" gives the condition count. Choices are strong NDArrays here
                // (weak-scalar dtype resolution is covered by unit tests, not the corpus).
                case "select":
                {
                    int nc = p["nc"].GetInt32();
                    var conds = new NDArray[nc];
                    var choices = new object[nc];
                    for (int i = 0; i < nc; i++) { conds[i] = ops[i]; choices[i] = ops[nc + i]; }
                    return np.select(conds, choices, ops[2 * nc]);
                }

                // choose — operands are [index, choice0..choice_{nc-1}]; params "nc" gives the choice
                // count and "mode" the clip mode (default raise). Choices are strong NDArrays here
                // (weak-scalar dtype resolution is covered by unit tests, not the corpus).
                case "choose":
                {
                    int nc = p["nc"].GetInt32();
                    var choices = new NDArray[nc];
                    for (int i = 0; i < nc; i++) choices[i] = ops[1 + i];
                    string mode = p.TryGetValue("mode", out var cm) ? cm.GetString() : "raise";
                    return np.choose(ops[0], choices, null, mode);
                }

                // W15 copyto: cross-dtype / strided-dst / scalar-broadcast-src. dst (ops[0]) is mutated
                // in place and IS the result; casting rule comes from params (default same_kind).
                case "copyto":
                    np.copyto(ops[0], ops[1], p.TryGetValue("casting", out var cast) ? cast.GetString() : "same_kind");
                    return ops[0];

                // W15 copyto OVERLAP: dst and src are two views of ONE buffer (operand 0). Rebuild both
                // from params so they genuinely alias, exercising NumPy's COPY_IF_OVERLAP path.
                case "copyto_overlap":
                {
                    var storage = ops[0].Storage;
                    long bufSize = ops[0].size;
                    var dst = new NDArray(storage, ShapeFrom(p["dst"], bufSize));
                    var src = new NDArray(storage, ShapeFrom(p["src"], bufSize));
                    np.copyto(dst, src);
                    return dst;
                }

                // Sorting / searching (T14).
                case "argsort": return ApplyArgsort(ops[0], p["axis"].GetInt32());
                case "sort": return np.sort(ops[0], p["axis"].GetInt32());          // Group A B2: value sort
                case "searchsorted": return np.searchsorted(ops[0], ops[1], p["side"].GetString(),
                                                             ops.Length >= 3 ? ops[2] : null);   // ops[2] = optional sorter
                case "digitize": return np.digitize(ops[0], ops[1], p["right"].GetBoolean()); // searchsorted + monotonicity
                case "nonzero": return np.nonzero(ops[0])[0]; // 1-D: single int64 index array
                case "bincount": // count (1 operand) or weighted sum (2 operands); minlength always recorded
                    return ops.Length >= 2
                        ? np.bincount(ops[0], ops[1], p["minlength"].GetInt32())
                        : np.bincount(ops[0], null, p["minlength"].GetInt32());
                case "flatnonzero": return np.flatnonzero(ops[0]);                  // Group A B3
                case "argwhere": return np.argwhere(ops[0]);                        // Group A B3
                case "unique": return np.unique(ops[0]);                            // Group A B3

                // G12 (issue #623): partition/argpartition ride the DERIVED kth-values compare —
                // the corpus stores take(partition(a, ks), ks) because the arrangement between kth
                // anchors is introselect-implementation-specific on BOTH sides (only the kth values
                // and the two-sided invariant are contractual; the invariant is unit-test-pinned).
                case "partition":
                {
                    var ks = ParseIntArray(p["kth"]);
                    int? ax = ParseAxis(p);
                    var part = np.partition(ops[0], ks, ax);
                    if (ax is null)
                        return np.take(part, np.array(ks));
                    int axv = ax.Value < 0 ? ax.Value + part.ndim : ax.Value;
                    return np.take(part, np.array(ks), axv);
                }
                case "argpartition":
                {
                    var ks = ParseIntArray(p["kth"]);
                    int? ax = ParseAxis(p);
                    if (ax is null)
                        return np.take(np.take(ops[0].ravel(), np.argpartition(ops[0], ks, null)), np.array(ks));
                    var g = np.argpartition(ops[0], ks, ax.Value);
                    var vals = np.take_along_axis(ops[0], g, ax.Value);
                    int axv = ax.Value < 0 ? ax.Value + vals.ndim : ax.Value;
                    return np.take(vals, np.array(ks), axv);
                }
                case "lexsort": return np.lexsort(ops, p["axis"].GetInt32());       // all operands are keys
                case "sort_complex": return np.sort_complex(ops[0]);

                // Group A Batches 4-6: shape / selection / convolve / split.
                case "flatten": return ops[0].flatten();
                case "rollaxis": return np.rollaxis(ops[0], p["axis"].GetInt32(), p["start"].GetInt32());
                case "take": return np.take(ops[0], ops[1], p["axis"].GetInt32(),
                    mode: p.TryGetValue("mode", out var takeMode) ? takeMode.GetString() : "raise");
                case "take_along_axis":
                {
                    var axEl = p["axis"];
                    int? tlaAx = axEl.ValueKind == JsonValueKind.Null ? (int?)null : axEl.GetInt32();
                    return np.take_along_axis(ops[0], ops[1], tlaAx);
                }
                case "compress": return np.compress(ops[0], ops[1], p["axis"].GetInt32());
                case "extract": return np.extract(ops[0], ops[1]);
                case "convolve": return np.convolve(ops[0], ops[1], p["mode"].GetString());
                case "correlate": return np.correlate(ops[0], ops[1], p["mode"].GetString());
                case "append": return p.ContainsKey("axis")
                    ? np.append(ops[0], ops[1], p["axis"].GetInt32())
                    : np.append(ops[0], ops[1]);
                case "insert": return np.insert(ops[0], p["obj"].GetInt32(), ops[1], p["axis"].GetInt32());
                case "split": return np.split(ops[0], p["sections"].GetInt32(), p["axis"].GetInt32())[p["piece"].GetInt32()];
                case "hsplit": return np.hsplit(ops[0], p["sections"].GetInt32())[p["piece"].GetInt32()];
                case "vsplit": return np.vsplit(ops[0], p["sections"].GetInt32())[p["piece"].GetInt32()];
                case "dsplit": return np.dsplit(ops[0], p["sections"].GetInt32())[p["piece"].GetInt32()];
                case "put": np.put(ops[0], ops[1], ops[2],
                    mode: p.TryGetValue("mode", out var putMode) ? putMode.GetString() : "raise");
                    return ops[0]; // mutates ops[0], IS the result
                case "ravel_multi_index": return np.ravel_multi_index(new[] { ops[0], ops[1] }, ParseIntArray(p["dims"]));
                case "unravel_index": return np.unravel_index(ops[0], ParseIntArray(p["shape"]))[p["piece"].GetInt32()];

                // ---- isin / set operations (arraysetops). ops[0]=element/ar1, ops[1]=test/ar2.
                // Value-dependent membership: fixtures overlap so each op bites. intersect1d's
                // return_indices (a tuple) is unit-tested, not corpus-gated (single-array here).
                case "isin": return np.isin(ops[0], ops[1],
                    assume_unique: p.ContainsKey("assume_unique") && p["assume_unique"].GetBoolean(),
                    invert: p.ContainsKey("invert") && p["invert"].GetBoolean(),
                    kind: p.TryGetValue("kind", out var isinKind) ? isinKind.GetString() : null);
                case "union1d": return np.union1d(ops[0], ops[1]);
                case "intersect1d": return np.intersect1d(ops[0], ops[1],
                    p.ContainsKey("assume_unique") && p["assume_unique"].GetBoolean());
                case "setxor1d": return np.setxor1d(ops[0], ops[1],
                    p.ContainsKey("assume_unique") && p["assume_unique"].GetBoolean());
                case "setdiff1d": return np.setdiff1d(ops[0], ops[1],
                    p.ContainsKey("assume_unique") && p["assume_unique"].GetBoolean());

                // Linear algebra (T8). NumPy is the oracle for value, result dtype, and broadcast shape.
                case "matmul": return np.matmul(ops[0], ops[1]);
                case "dot": return np.dot(ops[0], ops[1]);
                case "outer": return np.outer(ops[0], ops[1]);
                case "kron": return np.kron(ops[0], ops[1]);

                // CBLAS product family (products tier): first VALUE gate for these — previously
                // only their error contracts were tested. vecdot's axis/keepdims params select
                // the overload; tensordot's "axes" (int) vs "axesA"/"axesB" (lists) mirror
                // NumPy's two call forms; multi_dot consumes ALL operands as the chain.
                case "inner": return np.inner(ops[0], ops[1]);
                case "vdot": return np.vdot(ops[0], ops[1]);
                // vecdot MUST pass axis BY NAME: np.vecdot is the ufunc kwargs form (out, axes,
                // axis?, keepdims) — a positional int would implicitly convert to NDArray and
                // bind `out` (the tier caught exactly that). The int-axis positional form is
                // np.linalg.vecdot (Array-API), not this one.
                case "vecdot":
                    return p.ContainsKey("keepdims")
                        ? np.vecdot(ops[0], ops[1], keepdims: p["keepdims"].GetBoolean())
                        : np.vecdot(ops[0], ops[1], axis: p.TryGetValue("axis", out var vdax) ? vdax.GetInt32() : -1);
                case "matvec": return np.matvec(ops[0], ops[1]);
                case "vecmat": return np.vecmat(ops[0], ops[1]);
                case "tensordot":
                    return p.ContainsKey("axesA")
                        ? np.tensordot(ops[0], ops[1], ParseIntArray(p["axesA"]), ParseIntArray(p["axesB"]))
                        : np.tensordot(ops[0], ops[1], p["axes"].GetInt32());
                case "multi_dot": return np.linalg.multi_dot(ops);
                case "matrix_power": return np.linalg.matrix_power(ops[0], p["n"].GetInt32());

                // A matrix times its OWN transpose — the syrk shortcut both of NumPy's matrix
                // -product dispatchers take when the two operands share a data pointer. The
                // corpus gives every operand its own buffer, so the self-product cannot be
                // expressed as two operands: the op name carries the transpose and the product
                // is formed here from the single stored operand (matmul_parity tier).
                case "dot_aat": return np.dot(ops[0], ops[0].T);
                case "dot_ata": return np.dot(ops[0].T, ops[0]);
                case "matmul_aat": return np.matmul(ops[0], ops[0].T);
                case "matmul_ata": return np.matmul(ops[0].T, ops[0]);
                case "trace": return np.trace(ops[0]);                              // Group A
                case "diagonal": return np.diagonal(ops[0]);                        // Group A

                // ---- LAPACK factorisation family (linalg_parity tier; ARRAY results) ----
                // Computable ONLY through NumSharp.Interop.OpenBLAS — the host-pinned gate
                // enables that backend before replay, then Disable()s. Tuple results
                // (svd/eig/eigh/qr) live in OpRegistry.Kinds.cs::ApplyTuple. Every param
                // here pairs 1:1 with gen_oracle.py::gen_linalg_parity.
                case "cholesky":
                    return np.linalg.cholesky(ops[0], p.TryGetValue("upper", out var chu) && chu.GetBoolean());
                case "eigvals": return np.linalg.eigvals(ops[0]);
                case "eigvalsh": return np.linalg.eigvalsh(ops[0], ParseUplo(p));
                case "svdvals": return np.linalg.svdvals(ops[0]);
                case "pinv":
                    return p.ContainsKey("rcond")
                        ? np.linalg.pinv(ops[0], rcond: p["rcond"].GetDouble())
                        : np.linalg.pinv(ops[0]);
                case "matrix_rank":
                    if (p.ContainsKey("tol")) return np.linalg.matrix_rank(ops[0], tol: p["tol"].GetDouble());
                    if (p.ContainsKey("rtol")) return np.linalg.matrix_rank(ops[0], rtol: p["rtol"].GetDouble());
                    return np.linalg.matrix_rank(ops[0]);
                case "cond":                                                        // SVD orders only (None/2/-2)
                    return p.ContainsKey("p") ? np.linalg.cond(ops[0], p["p"].GetInt32()) : np.linalg.cond(ops[0]);
                case "norm":                                                        // matrix orders 2/-2/'nuc'
                    return p.ContainsKey("axis")
                        ? np.linalg.norm(ops[0], ParseOrd(p["ord"]), ParseIntArray(p["axis"]),
                                         p.TryGetValue("keepdims", out var nkd) && nkd.GetBoolean())
                        : np.linalg.norm(ops[0], ParseOrd(p["ord"]));
                // svd(compute_uv=false) -> just S; the (U,S,Vh) tuple form is in ApplyTuple.
                case "svd": return np.linalg.svd(ops[0], full_matrices: p["full_matrices"].GetBoolean(), compute_uv: false).S;
                // qr(mode='r') -> just R; reduced/complete/raw tuples are in ApplyTuple.
                case "qr": return np.linalg.qr(ops[0], "r").R;

                // ---- LU-factorisation family (linalg_parity tier; ARRAY results) --------
                // solve/inv/det/tensorinv/tensorsolve reach getrf/gesv (deterministic partial
                // pivoting -> byte-reproducible, no eigenvector-phase ambiguity). slogdet's
                // (sign, logabsdet) TUPLE is in OpRegistry.Kinds.cs::ApplyTuple. `det` of a
                // single matrix is a 0-D scalar; a stack is 1-D. Params pair 1:1 with
                // gen_oracle.py::gen_linalg_parity's LU-factorisation block.
                case "inv": return np.linalg.inv(ops[0]);
                case "det": return np.linalg.det(ops[0]);
                case "solve": return np.linalg.solve(ops[0], ops[1]);
                case "tensorinv":
                    return np.linalg.tensorinv(ops[0], p.TryGetValue("ind", out var tiv) ? tiv.GetInt32() : 2);
                case "tensorsolve":
                    return p.ContainsKey("axes")
                        ? np.linalg.tensorsolve(ops[0], ops[1], ParseIntArray(p["axes"]))
                        : np.linalg.tensorsolve(ops[0], ops[1]);

                // ---- cross / cov / corrcoef (products tier: dot-based value gate) ----------
                // cross is multiply-subtract (no reduction) -> portable. cov/corrcoef are
                // normalized dot products -> byte-exact for the small operands here. A SECOND
                // operand is cov/corrcoef's `y` variable (keyed off the operand count).
                case "cross":
                    return p.ContainsKey("axisa")
                        ? np.cross(ops[0], ops[1], axisa: p["axisa"].GetInt32(),
                                   axisb: p["axisb"].GetInt32(), axisc: p["axisc"].GetInt32())
                        : np.cross(ops[0], ops[1]);
                case "cov":
                    return np.cov(ops[0], ops.Length > 1 ? ops[1] : null,
                                  rowvar: !p.TryGetValue("rowvar", out var cvr) || cvr.GetBoolean(),
                                  bias: p.TryGetValue("bias", out var cvb) && cvb.GetBoolean(),
                                  ddof: p.TryGetValue("ddof", out var cvd) ? cvd.GetInt32() : (int?)null);
                case "corrcoef":
                    return np.corrcoef(ops[0], ops.Length > 1 ? ops[1] : null,
                                       rowvar: !p.TryGetValue("rowvar", out var ccr) || ccr.GetBoolean());

                // ---- einsum (portable small-exact; float/int contractions + the view path) --
                case "einsum": return np.einsum(p["subscripts"].GetString(), ops);

                // ---- polynomial family --------------------------------------------------
                // Pure/portable (poly.jsonl): polyval/vander/polyder/polyint/polyadd/polysub/
                // polymul/poly1d, and poly of a 1-D root sequence. Backend (linalg_parity):
                // roots (eigvals), polyfit (lstsq), poly of a 2-D matrix (eigvals) — `poly`
                // dispatches on the operand rank inside np.poly, so ONE case serves both.
                case "poly": return np.poly(ops[0]);
                case "roots": return np.roots(ops[0]);
                case "polyfit": return np.polyfit(ops[0], ops[1], p["deg"].GetInt32()).coeffs;
                case "polyval": return np.polyval(ops[0], ops[1]);
                case "vander":
                    return np.vander(ops[0],
                                     p.TryGetValue("N", out var vN) ? vN.GetInt32() : (int?)null,
                                     p.TryGetValue("increasing", out var vI) && vI.GetBoolean());
                case "polyder": return np.polyder(ops[0], p.TryGetValue("m", out var pdm) ? pdm.GetInt32() : 1);
                case "polyint":
                    return p.ContainsKey("k")
                        ? np.polyint(ops[0], p.TryGetValue("m", out var pim) ? pim.GetInt32() : 1, p["k"].GetDouble())
                        : np.polyint(ops[0], p.TryGetValue("m", out var pim2) ? pim2.GetInt32() : 1);
                case "polyadd": return np.polyadd(ops[0], ops[1]);
                case "polysub": return np.polysub(ops[0], ops[1]);
                case "polymul": return np.polymul(ops[0], ops[1]);
                case "poly1d_coeffs": return new poly1d(ops[0]).coeffs;
                case "poly1d_fromroots": return new poly1d(ops[0], r: true).coeffs;

                // ---- diag / tri family ----------------------------------------------------
                // `tri` is a pure generator: ops[0] is a 1-element carrier whose dtype selects
                // tri's dtype (the corpus loops dtype through it), N/M/k come from params.
                case "tri":
                    return np.tri(p["N"].GetInt32(),
                                  p["M"].ValueKind == JsonValueKind.Null ? (int?)null : p["M"].GetInt32(),
                                  p["k"].GetInt32(), ops[0].typecode);
                case "diag": return np.diag(ops[0], ParseK(p));
                case "diagflat": return np.diagflat(ops[0], ParseK(p));
                case "tril": return np.tril(ops[0], ParseK(p));
                case "triu": return np.triu(ops[0], ParseK(p));

                // Mutating: fill_diagonal writes into ops[0] and the mutated operand IS the result.
                // The value is handed over as a RAW long[] rather than an NDArray on purpose —
                // `val` is object-typed and NumPy accepts any array_like, so this keeps the
                // asanyarray/tiling path (not just the NDArray one) under the differential gate.
                case "fill_diagonal":
                    np.fill_diagonal(ops[0], ParseLongArray(p["val"]), p["wrap"].GetBoolean());
                    return ops[0];

                // Index-tuple generators — `which` selects the recorded tuple element (as nonzero does).
                case "diag_indices":
                    return np.diag_indices(p["n"].GetInt32(), p["ndim"].GetInt32())[p["which"].GetInt32()];
                case "tril_indices":
                    return np.tril_indices(p["n"].GetInt32(), p["k"].GetInt32(), ParseNullableInt(p, "m"))[p["which"].GetInt32()];
                case "triu_indices":
                    return np.triu_indices(p["n"].GetInt32(), p["k"].GetInt32(), ParseNullableInt(p, "m"))[p["which"].GetInt32()];
                case "diag_indices_from":
                    return np.diag_indices_from(ops[0])[p["which"].GetInt32()];
                case "tril_indices_from":
                    return np.tril_indices_from(ops[0], p["k"].GetInt32())[p["which"].GetInt32()];
                case "triu_indices_from":
                    return np.triu_indices_from(ops[0], p["k"].GetInt32())[p["which"].GetInt32()];
                case "mask_indices":
                {
                    // The corpus serialises the mask FUNCTION by name; re-bind it here.
                    var fname = p["func"].GetString();
                    Func<NDArray, int, NDArray> mask = fname switch
                    {
                        "triu" => (a, kk) => np.triu(a, kk),
                        "tril" => (a, kk) => np.tril(a, kk),
                        "diag" => (a, kk) => np.diag(a, kk),
                        _ => throw new NotSupportedException($"mask_indices func '{fname}'")
                    };
                    return np.mask_indices(p["n"].GetInt32(), mask, p["k"].GetInt32())[p["which"].GetInt32()];
                }

                // ---- index-expression DSL (r_ / c_ / ix_) ---------------------------------
                // The non-array parts of the index expression ride in params; the array parts
                // are ordinary operands. Rebuild NumPy's key in order:
                //   [directive?] + slice expressions + operands + weak scalars
                // The scalars are boxed as long/double/bool ON PURPOSE — that is what makes
                // them NEP50-weak on the NumSharp side, so the promotion matrix is gated here
                // rather than only in the hand-written tests.
                case "r_":
                case "c_":
                {
                    var key = new List<object>();
                    if (p["directive"].ValueKind != JsonValueKind.Null)
                        key.Add(p["directive"].GetString());
                    foreach (var e in p["exprs"].EnumerateArray())
                        key.Add(e.GetString());
                    key.AddRange(ops);
                    foreach (var s in p["scalars"].EnumerateArray())
                    {
                        var kind = s[0].GetString();
                        key.Add(kind switch
                        {
                            "i" => s[1].GetInt64(),
                            // "u" is the only way to carry a literal past long.MaxValue; boxing it
                            // as ulong is what lifts the weak-integer default to uint64.
                            "u" => s[1].GetUInt64(),
                            "f" => s[1].GetDouble(),
                            "b" => (object)s[1].GetBoolean(),
                            _ => throw new NotSupportedException($"scalar kind '{kind}'")
                        });
                    }

                    var arr = key.ToArray();
                    return p["kind"].GetString() == "r" ? np.r_[arr] : np.c_[arr];
                }

                // ix_ returns one array per sequence; `which` selects the recorded slot.
                case "ix_":
                {
                    var seqs = new object[ops.Length];
                    for (int i = 0; i < ops.Length; i++)
                        seqs[i] = ops[i];
                    return np.ix_(seqs)[p["which"].GetInt32()];
                }

                // Group A: rounding + flattened diff + nan order-statistics.
                case "round_": return np.round_(ops[0], p["decimals"].GetInt32());
                case "ediff1d": return np.ediff1d(ops[0]);
                case "nanpercentile": return np.nanpercentile(ops[0], p["q"].GetDouble(), ParseAxis(p), keepdims: ParseKeepdims(p));
                case "nanquantile": return np.nanquantile(ops[0], p["q"].GetDouble(), ParseAxis(p), keepdims: ParseKeepdims(p));

                // ---- np.fft.* (Fourier transforms + helpers) -----------------------------
                // Pairs 1:1 with gen_oracle.gen_fft. The transforms take (a, n|s, axis|axes, norm);
                // float32/float16 inputs return complex128/float64 where NumPy returns complex64/
                // float32 (NumSharp has no complex64) — the dtype-only divergence excused in
                // MisalignedRegistry. fftfreq/rfftfreq are operand-less pure generators.
                case "fft": return np.fft.fft(ops[0], ParseNullableInt(p, "n"), p["axis"].GetInt32(), ParseNorm(p));
                case "ifft": return np.fft.ifft(ops[0], ParseNullableInt(p, "n"), p["axis"].GetInt32(), ParseNorm(p));
                case "rfft": return np.fft.rfft(ops[0], ParseNullableInt(p, "n"), p["axis"].GetInt32(), ParseNorm(p));
                case "irfft": return np.fft.irfft(ops[0], ParseNullableInt(p, "n"), p["axis"].GetInt32(), ParseNorm(p));
                case "hfft": return np.fft.hfft(ops[0], ParseNullableInt(p, "n"), p["axis"].GetInt32(), ParseNorm(p));
                case "ihfft": return np.fft.ihfft(ops[0], ParseNullableInt(p, "n"), p["axis"].GetInt32(), ParseNorm(p));
                case "fft2": return np.fft.fft2(ops[0], ParseNullableIntArray(p, "s"), ParseNullableIntArray(p, "axes"), ParseNorm(p));
                case "ifft2": return np.fft.ifft2(ops[0], ParseNullableIntArray(p, "s"), ParseNullableIntArray(p, "axes"), ParseNorm(p));
                case "fftn": return np.fft.fftn(ops[0], ParseNullableIntArray(p, "s"), ParseNullableIntArray(p, "axes"), ParseNorm(p));
                case "ifftn": return np.fft.ifftn(ops[0], ParseNullableIntArray(p, "s"), ParseNullableIntArray(p, "axes"), ParseNorm(p));
                case "rfft2": return np.fft.rfft2(ops[0], ParseNullableIntArray(p, "s"), ParseNullableIntArray(p, "axes"), ParseNorm(p));
                case "irfft2": return np.fft.irfft2(ops[0], ParseNullableIntArray(p, "s"), ParseNullableIntArray(p, "axes"), ParseNorm(p));
                case "rfftn": return np.fft.rfftn(ops[0], ParseNullableIntArray(p, "s"), ParseNullableIntArray(p, "axes"), ParseNorm(p));
                case "irfftn": return np.fft.irfftn(ops[0], ParseNullableIntArray(p, "s"), ParseNullableIntArray(p, "axes"), ParseNorm(p));
                case "fftfreq": return np.fft.fftfreq(p["n"].GetInt32(), p["d"].GetDouble());
                case "rfftfreq": return np.fft.rfftfreq(p["n"].GetInt32(), p["d"].GetDouble());
                case "fftshift":
                    return p.ContainsKey("axis")
                        ? np.fft.fftshift(ops[0], p["axis"].GetInt32())
                        : np.fft.fftshift(ops[0], ParseNullableIntArray(p, "axes"));
                case "ifftshift":
                    return p.ContainsKey("axis")
                        ? np.fft.ifftshift(ops[0], p["axis"].GetInt32())
                        : np.fft.ifftshift(ops[0], ParseNullableIntArray(p, "axes"));

                // Reductions (axis/keepdims params).
                case "sum": case "prod": case "min": case "max": case "mean":
                case "std": case "var": case "argmax": case "argmin": case "all": case "any":
                // NaN-aware reductions (T10; nanargmax/nanargmin are G12 / issue #623).
                case "nansum": case "nanprod": case "nanmax": case "nanmin": case "nanmean":
                case "nanstd": case "nanvar": case "nanmedian":
                case "nanargmax": case "nanargmin":
                    return ApplyReduce(op, ParseAxis(p), ParseKeepdims(p), ops[0]);

                // np.random byte-parity (random_parity tiers): seed -> draw -> compare the raw
                // stream bytes against NumPy's recorded sequence. "draws" > 1 pins stream
                // ADVANCEMENT (the recorded block is the LAST of N identical draws). The global
                // np.random instance is safe here because the corpus runner is sequential and
                // the tier's test method is [DoNotParallelize].
                case "rnd":
                {
                    np.random.seed(p["seed"].GetInt32());
                    int draws = p.TryGetValue("draws", out var dr) ? dr.GetInt32() : 1;
                    NDArray r = null;
                    for (int k = 0; k < draws; k++)
                        r = RndDraw(p);
                    return r;
                }

                // Array/scalar-result ops added with the result-kind upgrade (iterator traces,
                // scalar-returning predicates) live in OpRegistry.Kinds.cs.
                default:
                    return ApplyExtended(op, p, ops);
            }
        }

        private static NDArray ApplyArgsort(NDArray a, int axis) => a.typecode switch
        {
            NPTypeCode.Byte => np.argsort<byte>(a, axis),
            NPTypeCode.SByte => np.argsort<sbyte>(a, axis),
            NPTypeCode.Int16 => np.argsort<short>(a, axis),
            NPTypeCode.UInt16 => np.argsort<ushort>(a, axis),
            NPTypeCode.Int32 => np.argsort<int>(a, axis),
            NPTypeCode.UInt32 => np.argsort<uint>(a, axis),
            NPTypeCode.Int64 => np.argsort<long>(a, axis),
            NPTypeCode.UInt64 => np.argsort<ulong>(a, axis),
            NPTypeCode.Char => np.argsort<char>(a, axis),
            NPTypeCode.Single => np.argsort<float>(a, axis),
            NPTypeCode.Double => np.argsort<double>(a, axis),
            NPTypeCode.Half => np.argsort<Half>(a, axis),
            NPTypeCode.Boolean => np.argsort<bool>(a, axis),
            NPTypeCode.Decimal => np.argsort<decimal>(a, axis),
            NPTypeCode.Complex => np.argsort<System.Numerics.Complex>(a, axis),
            _ => throw new NotSupportedException($"argsort<{a.typecode}> not wired in OpRegistry")
        };

        private static int[] ParseIntArray(JsonElement arr)
        {
            var list = new List<int>();
            foreach (var e in arr.EnumerateArray())
                list.Add(e.GetInt32());
            return list.ToArray();
        }

        /// <summary>UPLO for the symmetric/Hermitian eigensolvers — 'L' (default) or 'U'.</summary>
        private static char ParseUplo(IReadOnlyDictionary<string, JsonElement> p)
            => p.TryGetValue("UPLO", out var u) ? u.GetString()[0] : 'L';

        /// <summary>norm/cond order: a string ('nuc'/'fro') or a boxed int (2/-2/…), as np.linalg expects.</summary>
        private static object ParseOrd(JsonElement e)
            => e.ValueKind == JsonValueKind.String ? (object)e.GetString() : e.GetInt32();

        private static double[] ParseDoubleArray(JsonElement arr)
        {
            var list = new List<double>();
            foreach (var e in arr.EnumerateArray())
                list.Add(e.GetDouble());
            return list.ToArray();
        }

        /// <summary>One distribution draw for the "rnd" op — dist name + positional args in
        /// params (arrays like pvals/alpha/cov ride named keys). Pairs 1:1 with
        /// gen_oracle.gen_random_parity's `run` dispatcher.</summary>
        private static NDArray RndDraw(IReadOnlyDictionary<string, JsonElement> p)
        {
            string dist = p["dist"].GetString();
            double A(int i) => p["args"][i].GetDouble();
            Shape S() => new Shape(ParseIntArray(p["size"]));
            long[] SL()
            {
                var ints = ParseIntArray(p["size"]);
                var longs = new long[ints.Length];
                for (int i = 0; i < ints.Length; i++) longs[i] = ints[i];
                return longs;
            }
            switch (dist)
            {
                // ---- portable: pure MT19937 bits + exactly-rounded arithmetic ----
                case "uniform": return np.random.uniform(A(0), A(1), S());
                case "rand": return np.random.rand(S());
                case "random_sample": return np.random.random_sample(SL());
                case "randint": return np.random.randint((long)A(0), (long)A(1), S());
                case "permutation": return np.random.permutation((int)A(0));
                case "shuffle":
                {
                    var arr = np.arange((int)A(0));
                    np.random.shuffle(arr);
                    return arr;
                }
                case "choice":
                    return np.random.choice((int)A(0), S(), true,
                        p.TryGetValue("p", out var pw) ? ParseDoubleArray(pw) : null);

                // ---- host-libm: transform / rejection samplers ----
                case "normal": return np.random.normal(A(0), A(1), S());
                case "randn": return np.random.randn(SL());
                case "standard_normal": return np.random.standard_normal(S());
                case "standard_exponential": return np.random.standard_exponential(S());
                case "standard_cauchy": return np.random.standard_cauchy(S());
                case "standard_t": return np.random.standard_t(A(0), S());
                case "standard_gamma": return np.random.standard_gamma(A(0), S());
                case "lognormal": return np.random.lognormal(A(0), A(1), S());
                case "exponential": return np.random.exponential(A(0), S());
                case "gamma": return np.random.gamma(A(0), A(1), S());
                case "beta": return np.random.beta(A(0), A(1), S());
                case "chisquare": return np.random.chisquare(A(0), S());
                case "f": return np.random.f(A(0), A(1), S());
                case "gumbel": return np.random.gumbel(A(0), A(1), S());
                case "laplace": return np.random.laplace(A(0), A(1), S());
                case "logistic": return np.random.logistic(A(0), A(1), S());
                case "pareto": return np.random.pareto(A(0), S());
                case "power": return np.random.power(A(0), S());
                case "rayleigh": return np.random.rayleigh(A(0), S());
                case "triangular": return np.random.triangular(A(0), A(1), A(2), S());
                case "vonmises": return np.random.vonmises(A(0), A(1), S());
                case "wald": return np.random.wald(A(0), A(1), S());
                case "weibull": return np.random.weibull(A(0), S());
                case "poisson": return np.random.poisson(A(0), S());
                case "binomial": return np.random.binomial((int)A(0), A(1), S());
                case "negative_binomial": return np.random.negative_binomial(A(0), A(1), S());
                case "geometric": return np.random.geometric(A(0), S());
                case "zipf": return np.random.zipf(A(0), S());
                case "logseries": return np.random.logseries(A(0), S());
                case "noncentral_chisquare": return np.random.noncentral_chisquare(A(0), A(1), S());
                case "noncentral_f": return np.random.noncentral_f(A(0), A(1), A(2), S());
                case "hypergeometric": return np.random.hypergeometric((long)A(0), (long)A(1), (long)A(2), S());
                case "multinomial": return np.random.multinomial((int)A(0), ParseDoubleArray(p["pvals"]), S());
                case "dirichlet": return np.random.dirichlet(ParseDoubleArray(p["alpha"]), S());
                case "multivariate_normal":
                {
                    var mean = ParseDoubleArray(p["mean"]);
                    var flat = ParseDoubleArray(p["cov"]);
                    int d = mean.Length;
                    var cov = new double[d, d];
                    for (int i = 0; i < d; i++)
                        for (int j = 0; j < d; j++)
                            cov[i, j] = flat[i * d + j];
                    return np.random.multivariate_normal(mean, cov, ParseIntArray(p["size"]));
                }
                default:
                    throw new NotSupportedException($"rnd dist '{dist}' not wired in OpRegistry");
            }
        }

        private static long[] ParseLongArray(JsonElement arr)
        {
            var list = new List<long>();
            foreach (var e in arr.EnumerateArray())
                list.Add(e.GetInt64());
            return list.ToArray();
        }

        // Rebuild an aliasing view (shape, element-strides, element-offset) over a shared buffer —
        // the overlap-case counterpart to FuzzCorpus.Reconstruct (which owns its buffer).
        private static Shape ShapeFrom(JsonElement v, long bufferSize)
            => new Shape(ParseLongArray(v.GetProperty("shape")), ParseLongArray(v.GetProperty("strides")),
                         v.GetProperty("offset").GetInt64(), bufferSize);

        /// <summary>Diagonal offset; absent means the default 0.</summary>
        private static int ParseK(IReadOnlyDictionary<string, JsonElement> p)
            => p.TryGetValue("k", out var k) ? k.GetInt32() : 0;

        /// <summary>A param that is either an int or JSON null (NumPy's `M=None` / `m=None`).</summary>
        private static int? ParseNullableInt(IReadOnlyDictionary<string, JsonElement> p, string name)
            => p.TryGetValue(name, out var v) && v.ValueKind != JsonValueKind.Null ? v.GetInt32() : (int?)null;

        private static int? ParseAxis(IReadOnlyDictionary<string, JsonElement> p)
            => p.TryGetValue("axis", out var ax) && ax.ValueKind != JsonValueKind.Null ? ax.GetInt32() : (int?)null;

        /// <summary>np.fft `norm` — the string or null (backward). Absent/JSON-null both mean null.</summary>
        private static string ParseNorm(IReadOnlyDictionary<string, JsonElement> p)
            => p.TryGetValue("norm", out var nrm) && nrm.ValueKind != JsonValueKind.Null ? nrm.GetString() : null;

        /// <summary>An int[] param that may be JSON null (np.fft `s` / `axes` = None).</summary>
        private static int[] ParseNullableIntArray(IReadOnlyDictionary<string, JsonElement> p, string key)
            => p.TryGetValue(key, out var v) && v.ValueKind != JsonValueKind.Null ? ParseIntArray(v) : null;

        private static bool ParseKeepdims(IReadOnlyDictionary<string, JsonElement> p)
            => p.TryGetValue("keepdims", out var kd) && kd.GetBoolean();

        private static NDArray ApplyReduce(string op, int? axis, bool keepdims, NDArray a)
        {
            switch (op)
            {
                case "sum": return np.sum(a, axis, keepdims);
                case "prod": return np.prod(a, axis, (Type)null, keepdims);
                case "min": return np.min(a, axis, keepdims);
                case "max": return np.max(a, axis, keepdims);
                case "mean": return axis.HasValue ? np.mean(a, axis.Value, keepdims) : np.mean(a, keepdims);
                case "std": return axis.HasValue ? np.std(a, axis.Value, keepdims) : np.std(a, keepdims);
                case "var": return axis.HasValue ? np.var(a, axis.Value, keepdims) : np.var(a, keepdims);
                // axis=None (keepdims=False only — the flat long form has no keepdims): wraps the
                // scalar back to the 0-d int64 NumPy returns. G13: this is the path whose Decimal
                // IL compare was silently wrong and whose Char case threw — now oracle-gated.
                case "argmax": return axis.HasValue ? np.argmax(a, axis.Value, keepdims) : NDArray.Scalar(np.argmax(a));
                case "argmin": return axis.HasValue ? np.argmin(a, axis.Value, keepdims) : NDArray.Scalar(np.argmin(a));
                case "all": return np.all(a, axis, null, keepdims);
                case "any": return np.any(a, axis, null, keepdims);
                case "nansum": return np.nansum(a, axis, keepdims);
                case "nanprod": return np.nanprod(a, axis, keepdims);
                case "nanmax": return np.nanmax(a, axis, keepdims);
                case "nanmin": return np.nanmin(a, axis, keepdims);
                // keepdims by NAME: positional slot 3 is now `out=` (an implicit bool->NDArray
                // conversion exists, so a positional bool would silently bind the out parameter).
                case "nanargmax": return np.nanargmax(a, axis, keepdims: keepdims);
                case "nanargmin": return np.nanargmin(a, axis, keepdims: keepdims);
                case "nanmean": return np.nanmean(a, axis, keepdims);
                case "nanstd": return np.nanstd(a, axis, keepdims);
                case "nanvar": return np.nanvar(a, axis, keepdims);
                case "nanmedian": return np.nanmedian(a, axis, keepdims: keepdims);
                default: throw new NotSupportedException(op);
            }
        }
    }
}
