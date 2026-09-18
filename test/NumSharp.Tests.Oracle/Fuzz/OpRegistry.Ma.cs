using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The MASKED-ARRAY half of the registry: maps a <c>ma.*</c> corpus op-name to the
    ///     <see cref="np.ma"/> call it names, over reconstructed <see cref="MaskedArray"/> operands.
    ///     Pairs 1:1 with gen_oracle.py's <c>gen_ma_*</c> generators, exactly as <see cref="Apply"/>
    ///     pairs with the ordinary tiers.
    ///
    ///     <para>
    ///     Op keys are prefixed <c>ma.</c> so they never collide with the identically-named <c>np.*</c>
    ///     key (<c>sum</c> lives in BOTH registries). The prefix is stripped once here and the bare
    ///     name switched; the harness (<see cref="FuzzCorpusTests"/>.RunMaCorpus) then compares the
    ///     returned object by its declared result kind — a <see cref="MaskedArray"/> (masked),
    ///     a <see cref="MaskedArray"/>[] (masked_tuple), a bare <see cref="NDArray"/> (array), or a
    ///     boxed scalar / <see cref="DType"/>.
    ///     </para>
    /// </summary>
    public static partial class OpRegistry
    {
        /// <summary>
        ///     Dispatch a masked op. Returns the raw op result (MaskedArray / MaskedArray[] / NDArray /
        ///     boxed bool / DType) — the caller compares it against the recorded NumPy result by kind.
        ///     Mutating ops (<c>put</c>/<c>putmask</c>) return the mutated operand, so the corpus can
        ///     bit-compare the post-mutation masked array (the place/put/copyto pattern in Apply).
        /// </summary>
        /// <param name="op">The <c>ma.</c>-prefixed corpus op key.</param>
        /// <param name="p">The op params (axis/keepdims/value/…), read the same way as <see cref="Apply"/>.</param>
        /// <param name="ops">The reconstructed MaskedArray operands, in corpus order.</param>
        /// <returns>The op result as <see cref="object"/>, dispatched on kind by the caller.</returns>
        /// <exception cref="NotSupportedException">The op key is not registered (a corpus/registry drift).</exception>
        public static object ApplyMasked(string op, IReadOnlyDictionary<string, JsonElement> p, MaskedArray[] ops)
        {
            // Every masked key carries the "ma." namespace prefix; switch on the bare name below.
            string name = op.StartsWith("ma.", StringComparison.Ordinal) ? op.Substring(3) : op;
            switch (name)
            {
                // ---- unary ufuncs (domain-free + domained) -> masked ------------------------------
                case "negative": return np.ma.negative(ops[0]);
                case "abs": return np.ma.abs(ops[0]);
                case "absolute": return np.ma.absolute(ops[0]);
                case "fabs": return np.ma.fabs(ops[0]);
                case "conjugate": return np.ma.conjugate(ops[0]);
                case "sqrt": return np.ma.sqrt(ops[0]);
                case "exp": return np.ma.exp(ops[0]);
                case "log": return np.ma.log(ops[0]);
                case "log2": return np.ma.log2(ops[0]);
                case "log10": return np.ma.log10(ops[0]);
                case "sin": return np.ma.sin(ops[0]);
                case "cos": return np.ma.cos(ops[0]);
                case "tan": return np.ma.tan(ops[0]);
                case "sinh": return np.ma.sinh(ops[0]);
                case "cosh": return np.ma.cosh(ops[0]);
                case "tanh": return np.ma.tanh(ops[0]);
                case "arcsin": return np.ma.arcsin(ops[0]);
                case "arccos": return np.ma.arccos(ops[0]);
                case "arctan": return np.ma.arctan(ops[0]);
                case "arcsinh": return np.ma.arcsinh(ops[0]);
                case "arccosh": return np.ma.arccosh(ops[0]);
                case "arctanh": return np.ma.arctanh(ops[0]);
                case "floor": return np.ma.floor(ops[0]);
                case "ceil": return np.ma.ceil(ops[0]);
                case "angle": return np.ma.angle(ops[0]);
                case "logical_not": return np.ma.logical_not(ops[0]);
                case "around": return np.ma.around(ops[0], p.TryGetValue("decimals", out var dc) ? dc.GetInt32() : 0);

                // ---- binary ufuncs + domained-binary + comparison + logical + bitwise -> masked ---
                case "add": return np.ma.add(ops[0], ops[1]);
                case "subtract": return np.ma.subtract(ops[0], ops[1]);
                case "multiply": return np.ma.multiply(ops[0], ops[1]);
                case "divide": return np.ma.divide(ops[0], ops[1]);
                case "true_divide": return np.ma.true_divide(ops[0], ops[1]);
                case "floor_divide": return np.ma.floor_divide(ops[0], ops[1]);
                case "mod": return np.ma.mod(ops[0], ops[1]);
                case "remainder": return np.ma.remainder(ops[0], ops[1]);
                case "fmod": return np.ma.fmod(ops[0], ops[1]);
                case "power": return np.ma.power(ops[0], ops[1]);
                case "arctan2": return np.ma.arctan2(ops[0], ops[1]);
                case "hypot": return np.ma.hypot(ops[0], ops[1]);
                case "equal": return np.ma.equal(ops[0], ops[1]);
                case "not_equal": return np.ma.not_equal(ops[0], ops[1]);
                case "less": return np.ma.less(ops[0], ops[1]);
                case "less_equal": return np.ma.less_equal(ops[0], ops[1]);
                case "greater": return np.ma.greater(ops[0], ops[1]);
                case "greater_equal": return np.ma.greater_equal(ops[0], ops[1]);
                case "logical_and": return np.ma.logical_and(ops[0], ops[1]);
                case "logical_or": return np.ma.logical_or(ops[0], ops[1]);
                case "logical_xor": return np.ma.logical_xor(ops[0], ops[1]);
                case "bitwise_and": return np.ma.bitwise_and(ops[0], ops[1]);
                case "bitwise_or": return np.ma.bitwise_or(ops[0], ops[1]);
                case "bitwise_xor": return np.ma.bitwise_xor(ops[0], ops[1]);
                case "left_shift": return np.ma.left_shift(ops[0], ops[1]);
                case "right_shift": return np.ma.right_shift(ops[0], ops[1]);
                case "maximum": return np.ma.maximum(ops[0], ops[1]);
                case "minimum": return np.ma.minimum(ops[0], ops[1]);

                // ---- reductions -> masked (count/argmin/argmax -> array) --------------------------
                case "sum": return np.ma.sum(ops[0], ParseAxis(p), null, ParseKeepdims(p));
                case "prod": return np.ma.prod(ops[0], ParseAxis(p), null, ParseKeepdims(p));
                case "product": return np.ma.product(ops[0], ParseAxis(p), null, ParseKeepdims(p));
                case "mean": return np.ma.mean(ops[0], ParseAxis(p), null, ParseKeepdims(p));
                case "min": return np.ma.min(ops[0], ParseAxis(p), null, ParseKeepdims(p));
                case "max": return np.ma.max(ops[0], ParseAxis(p), null, ParseKeepdims(p));
                case "amin": return np.ma.amin(ops[0], ParseAxis(p), null, ParseKeepdims(p));
                case "amax": return np.ma.amax(ops[0], ParseAxis(p), null, ParseKeepdims(p));
                case "ptp": return np.ma.ptp(ops[0], ParseAxis(p), null, ParseKeepdims(p));
                case "var": return np.ma.var(ops[0], ParseAxis(p), null, ParseDdof(p), ParseKeepdims(p));
                case "std": return np.ma.std(ops[0], ParseAxis(p), null, ParseDdof(p), ParseKeepdims(p));
                case "all": return np.ma.all(ops[0], ParseAxis(p), ParseKeepdims(p));
                case "any": return np.ma.any(ops[0], ParseAxis(p), ParseKeepdims(p));
                case "anom": return np.ma.anom(ops[0], ParseAxis(p), null);
                case "median": return np.ma.median(ops[0], ParseAxis(p), ParseKeepdims(p));
                case "average": return np.ma.average(ops[0], ParseAxis(p), null, ParseKeepdims(p));
                case "count": return np.ma.count(ops[0], ParseAxis(p), ParseKeepdims(p));
                case "count_masked": return np.ma.count_masked(ops[0], ParseAxis(p));
                case "argmin": return np.ma.argmin(ops[0], ParseAxis(p), null);
                case "argmax": return np.ma.argmax(ops[0], ParseAxis(p), null);

                // ---- scans -> masked --------------------------------------------------------------
                case "cumsum": return np.ma.cumsum(ops[0], ParseAxis(p), null);
                case "cumprod": return np.ma.cumprod(ops[0], ParseAxis(p), null);
                case "diff": return np.ma.diff(ops[0], p.TryGetValue("n", out var nn) ? nn.GetInt32() : 1,
                                               p.TryGetValue("axis", out var da) ? da.GetInt32() : -1);
                case "ediff1d": return np.ma.ediff1d(ops[0]);

                // ---- manipulation -> masked (compressed/getdata/... -> array; hsplit -> tuple) ----
                case "ravel": return np.ma.ravel(ops[0]);
                case "flatten": return np.ma.flatten(ops[0]);
                case "reshape": return np.ma.reshape(ops[0], ParseIntArray(p["shape"]));
                case "transpose": return np.ma.transpose(ops[0], ParseNullableIntArray(p, "axes"));
                case "swapaxes": return np.ma.swapaxes(ops[0], p["axis1"].GetInt32(), p["axis2"].GetInt32());
                case "moveaxis": return np.ma.moveaxis(ops[0], p["source"].GetInt32(), p["destination"].GetInt32());
                case "squeeze": return np.ma.squeeze(ops[0], ParseAxis(p));
                case "expand_dims": return np.ma.expand_dims(ops[0], p["axis"].GetInt32());
                case "repeat": return np.ma.repeat(ops[0], p["repeats"].GetInt32(), ParseAxis(p));
                case "diagonal": return np.ma.diagonal(ops[0], p.TryGetValue("offset", out var of) ? of.GetInt32() : 0);
                case "diag": return np.ma.diag(ops[0], p.TryGetValue("k", out var dk) ? dk.GetInt32() : 0);
                case "diagflat": return np.ma.diagflat(ops[0], p.TryGetValue("k", out var dk2) ? dk2.GetInt32() : 0);
                case "atleast_1d": return np.ma.atleast_1d(ops[0]);
                case "atleast_2d": return np.ma.atleast_2d(ops[0]);
                case "atleast_3d": return np.ma.atleast_3d(ops[0]);
                case "concatenate": return np.ma.concatenate(ops.Cast<object>().ToArray(), ParseAxisInt(p));
                case "stack": return np.ma.stack(ops.Cast<object>().ToArray(), ParseAxisInt(p));
                case "hstack": return np.ma.hstack(ops.Cast<object>().ToArray());
                case "vstack": return np.ma.vstack(ops.Cast<object>().ToArray());
                case "dstack": return np.ma.dstack(ops.Cast<object>().ToArray());
                case "column_stack": return np.ma.column_stack(ops.Cast<object>().ToArray());
                case "append": return np.ma.append(ops[0], ops[1], ParseAxis(p));
                case "resize": return np.ma.resize(ops[0], new Shape(ParseLongArray(p["shape"])));
                case "compressed": return np.ma.compressed(ops[0]);
                case "getdata": return np.ma.getdata(ops[0]);
                case "getmask": return np.ma.getmask(ops[0]);
                case "getmaskarray": return np.ma.getmaskarray(ops[0]);
                case "filled": return np.ma.filled(ops[0], p.TryGetValue("fill", out var fv) ? (object)fv.GetDouble() : null);

                // ---- masked constructors -> masked ------------------------------------------------
                case "masked_where": return np.ma.masked_where(ops[0], ops[1]);   // ops[0]=condition, ops[1]=a
                case "masked_equal": return np.ma.masked_equal(ops[0], p["value"].GetDouble());
                case "masked_not_equal": return np.ma.masked_not_equal(ops[0], p["value"].GetDouble());
                case "masked_greater": return np.ma.masked_greater(ops[0], p["value"].GetDouble());
                case "masked_greater_equal": return np.ma.masked_greater_equal(ops[0], p["value"].GetDouble());
                case "masked_less": return np.ma.masked_less(ops[0], p["value"].GetDouble());
                case "masked_less_equal": return np.ma.masked_less_equal(ops[0], p["value"].GetDouble());
                case "masked_inside": return np.ma.masked_inside(ops[0], p["v1"].GetDouble(), p["v2"].GetDouble());
                case "masked_outside": return np.ma.masked_outside(ops[0], p["v1"].GetDouble(), p["v2"].GetDouble());
                case "masked_invalid": return np.ma.masked_invalid(ops[0]);
                case "masked_values": return np.ma.masked_values(ops[0], p["value"].GetDouble());
                case "fix_invalid": return np.ma.fix_invalid(ops[0]);
                case "masked_all": return np.ma.masked_all(new Shape(ParseLongArray(p["shape"])),
                                                            FuzzCorpus.DtypeToTC(p["dtype"].GetString()));
                case "masked_all_like": return np.ma.masked_all_like(ops[0]);
                case "array": return np.ma.array(ops[0]);
                case "masked_array": return np.ma.masked_array(ops[0]);
                case "asarray": return np.ma.asarray(ops[0]);
                case "copy": return np.ma.copy(ops[0]);

                // ---- selection -> masked (put/putmask -> mutated operand) -------------------------
                case "take": return np.ma.take(ops[0], ops[1].data, ParseAxis(p),
                                                p.TryGetValue("mode", out var tm) ? tm.GetString() : "raise");
                case "choose": return np.ma.choose(ops[0], ops.Skip(1).Cast<object>().ToArray(),
                                                   p.TryGetValue("mode", out var cm) ? cm.GetString() : "raise");
                case "compress": return np.ma.compress(ops[0], ops[1], ParseAxis(p));   // ops[0]=condition, ops[1]=a
                case "clip": return np.ma.clip(ops[0], p["min"].GetDouble(), p["max"].GetDouble());
                case "where": return np.ma.where(ops[0], ops[1], ops[2]);
                case "put":
                    ops[0].put(ops[1].data, ops[2], p.TryGetValue("mode", out var pm) ? pm.GetString() : "raise");
                    return ops[0];
                case "putmask":
                    ops[0].putmask(ops[1], ops[2]);
                    return ops[0];

                // ---- sort / set operations -> masked (argsort -> array) --------------------------
                case "sort": return np.ma.sort(ops[0], p.TryGetValue("axis", out var sa) ? sa.GetInt32() : -1,
                                               p.TryGetValue("endwith", out var ew) ? ew.GetBoolean() : true);
                case "argsort": return np.ma.argsort(ops[0], p.TryGetValue("axis", out var aa) ? aa.GetInt32() : -1,
                                                     p.TryGetValue("endwith", out var aew) ? aew.GetBoolean() : true);
                case "unique": return np.ma.unique(ops[0]);
                case "intersect1d": return np.ma.intersect1d(ops[0], ops[1]);
                case "union1d": return np.ma.union1d(ops[0], ops[1]);
                case "setxor1d": return np.ma.setxor1d(ops[0], ops[1]);
                case "setdiff1d": return np.ma.setdiff1d(ops[0], ops[1]);
                case "isin": return np.ma.isin(ops[0], ops[1]);
                case "in1d": return np.ma.in1d(ops[0], ops[1]);

                // ---- extras: products / triangular-mask / predicates / mask helpers --------------
                case "dot": return np.ma.dot(ops[0], ops[1]);
                case "inner": return np.ma.inner(ops[0], ops[1]);
                case "outer": return np.ma.outer(ops[0], ops[1]);
                case "trace": return np.ma.trace(ops[0]);
                case "vander": return np.ma.vander(ops[0], p.TryGetValue("n", out var vn) ? (int?)vn.GetInt32() : null);
                case "mask_rows": return np.ma.mask_rows(ops[0]);
                case "mask_cols": return np.ma.mask_cols(ops[0]);
                case "mask_rowcols": return np.ma.mask_rowcols(ops[0], ParseAxis(p));
                case "compress_rows": return np.ma.compress_rows(ops[0]);
                case "compress_cols": return np.ma.compress_cols(ops[0]);
                case "is_masked": return np.ma.is_masked(ops[0]);
                case "isMaskedArray": return np.ma.isMaskedArray(ops[0]);
                case "allclose": return np.ma.allclose(ops[0], ops[1]);
                case "allequal": return np.ma.allequal(ops[0], ops[1]);
                case "make_mask": return np.ma.make_mask(ops[0].data);
                case "make_mask_none": return np.ma.make_mask_none(new Shape(ParseLongArray(p["shape"])));
                case "mask_or": return np.ma.mask_or(np.ma.getmaskarray(ops[0]), np.ma.getmaskarray(ops[1]));
                case "flatten_mask": return np.ma.flatten_mask(np.ma.getmaskarray(ops[0]));
                case "make_mask_descr": return np.ma.make_mask_descr(FuzzCorpus.DtypeToTC(p["dtype"].GetString()));

                default:
                    throw new NotSupportedException($"ma op '{op}' not registered in OpRegistry.ApplyMasked");
            }
        }

        /// <summary>Dispatch a masked op whose result is a tuple of masked arrays (currently <c>hsplit</c>).</summary>
        /// <param name="op">The <c>ma.</c>-prefixed corpus op key.</param>
        /// <param name="p">Op params (the split count/indices).</param>
        /// <param name="ops">The reconstructed MaskedArray operands.</param>
        /// <returns>The per-slot MaskedArray results, in NumPy tuple order.</returns>
        /// <exception cref="NotSupportedException">The op key is not a registered masked-tuple op.</exception>
        public static MaskedArray[] ApplyMaskedTuple(string op, IReadOnlyDictionary<string, JsonElement> p, MaskedArray[] ops)
        {
            string name = op.StartsWith("ma.", StringComparison.Ordinal) ? op.Substring(3) : op;
            switch (name)
            {
                case "hsplit": return np.ma.hsplit(ops[0], p["sections"].GetInt32());
                default:
                    throw new NotSupportedException($"ma tuple op '{op}' not registered in OpRegistry.ApplyMaskedTuple");
            }
        }

        /// <summary>ddof param for var/std (default 0), read like the ordinary registry's scalar ints.</summary>
        private static int ParseDdof(IReadOnlyDictionary<string, JsonElement> p)
            => p.TryGetValue("ddof", out var v) ? v.GetInt32() : 0;

        /// <summary>A non-nullable axis (default 0), for the stack/concatenate family that has no None axis.</summary>
        private static int ParseAxisInt(IReadOnlyDictionary<string, JsonElement> p)
            => p.TryGetValue("axis", out var v) ? v.GetInt32() : 0;
    }
}
