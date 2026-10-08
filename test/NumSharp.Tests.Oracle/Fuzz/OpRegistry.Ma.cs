using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The MASKED-ARRAY half of the registry: maps a <c>ma.*</c> corpus op-name to the
    ///     <see cref="np.ma"/> call it names, over reconstructed <see cref="NDMaskedArray"/> operands.
    ///     Pairs 1:1 with gen_oracle.py's <c>gen_ma_*</c> generators, exactly as <see cref="Apply"/>
    ///     pairs with the ordinary tiers.
    ///
    ///     <para>
    ///     Op keys are prefixed <c>ma.</c> so they never collide with the identically-named <c>np.*</c>
    ///     key (<c>sum</c> lives in BOTH registries). The prefix is stripped once here and the bare
    ///     name switched; the harness (<see cref="FuzzCorpusTests"/>.RunMaCorpus) then compares the
    ///     returned object by its declared result kind — a <see cref="NDMaskedArray"/> (masked),
    ///     a <see cref="NDMaskedArray"/>[] (masked_tuple), a bare <see cref="NDArray"/> (array), or a
    ///     boxed scalar / <see cref="DType"/>.
    ///     </para>
    /// </summary>
    public static partial class OpRegistry
    {
        /// <summary>
        ///     Dispatch a masked op. Returns the raw op result (NDMaskedArray / NDMaskedArray[] / NDArray /
        ///     boxed bool / DType) — the caller compares it against the recorded NumPy result by kind.
        ///     Mutating ops (<c>put</c>/<c>putmask</c>) return the mutated operand, so the corpus can
        ///     bit-compare the post-mutation masked array (the place/put/copyto pattern in Apply).
        /// </summary>
        /// <param name="op">The <c>ma.</c>-prefixed corpus op key.</param>
        /// <param name="p">The op params (axis/keepdims/value/…), read the same way as <see cref="Apply"/>.</param>
        /// <param name="ops">The reconstructed NDMaskedArray operands, in corpus order.</param>
        /// <returns>The op result as <see cref="object"/>, dispatched on kind by the caller.</returns>
        /// <exception cref="NotSupportedException">The op key is not registered (a corpus/registry drift).</exception>
        public static object ApplyMasked(string op, IReadOnlyDictionary<string, JsonElement> p, NDMaskedArray[] ops)
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
                // NumPy's np.ma.around IS the round ufunc (around == round). NumSharp's ONE-ARG around(a)
                // is the matching ufunc (it collapses a 0-d all-masked input to the float64 `masked`
                // singleton like NumPy); the TWO-ARG around(a, decimals)=round path does NOT (a latent
                // 0-d dtype inconsistency, see docs/MA_ORACLE_DESIGN.md). Route decimals=0 to the ufunc.
                case "around":
                    return p.TryGetValue("decimals", out var dc) && dc.GetInt32() != 0
                        ? np.ma.around(ops[0], dc.GetInt32())
                        : np.ma.around(ops[0]);

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
                // ma_meanvar.jsonl gives mean / var / std / anom a dtype= and var / std a mean= (ParseMaDtype, ParseMaMean);
                // the older tiers carry neither, which reads as null.
                case "mean": return np.ma.mean(ops[0], ParseAxis(p), ParseMaDtype(p), ParseKeepdims(p));
                case "min": return np.ma.min(ops[0], ParseAxis(p), null, ParseKeepdims(p));
                case "max": return np.ma.max(ops[0], ParseAxis(p), null, ParseKeepdims(p));
                case "amin": return np.ma.amin(ops[0], ParseAxis(p), null, ParseKeepdims(p));
                case "amax": return np.ma.amax(ops[0], ParseAxis(p), null, ParseKeepdims(p));
                case "ptp": return np.ma.ptp(ops[0], ParseAxis(p), null, ParseKeepdims(p));
                case "var": return np.ma.var(ops[0], ParseAxis(p), ParseMaDtype(p), ParseDdof(p), ParseKeepdims(p), ParseMaMean(p, ops));
                case "std": return np.ma.std(ops[0], ParseAxis(p), ParseMaDtype(p), ParseDdof(p), ParseKeepdims(p), ParseMaMean(p, ops));
                case "all": return np.ma.all(ops[0], ParseAxis(p), ParseKeepdims(p));
                case "any": return np.ma.any(ops[0], ParseAxis(p), ParseKeepdims(p));
                case "anom": return np.ma.anom(ops[0], ParseAxis(p), ParseMaDtype(p));
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
                // ma_surface.jsonl's getmask cases pass the operand's plain data when "plain" is set.
                case "getmask": return p.ContainsKey("plain") ? np.ma.getmask(ops[0].data) : np.ma.getmask(ops[0]);
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
                {
                    // ma_put.jsonl says how NumPy received each operand: `plain` (the target was the plain data view: ma.put
                    // runs ndarray.put), `hard` (a hardened target), and `vkind` for the values (PutValues). The ma_select
                    // cases carry none of them: NumPy put a plain ndarray of values into a soft-masked array.
                    // Both arms cast to object: a conditional of NDArray and NDMaskedArray takes one arm's type, and the implicit
                    // NDArray -> NDMaskedArray conversion would wrap the plain target in a NEW masked array (whose filled data
                    // then hid the slots the put masked).
                    object target = Flag(p, "plain") ? (object)ops[0].data : ops[0];
                    if (Flag(p, "hard"))
                        ops[0].harden_mask();
                    var values = PutValues(p, ops, out var valuesCopy);
                    try
                    {
                        np.ma.put(target, ops[1].data, values, p.TryGetValue("mode", out var pm) ? pm.GetString() : "raise");
                    }
                    finally
                    {
                        // The replay's own copy of an owned values operand is not part of the op: released here, so the
                        // leak sweep audits np.ma.put alone.
                        valuesCopy?.Dispose();
                    }
                    return target;
                }
                case "putmask":
                    // ma.putmask(a, mask.data, values): values from the third operand, or a Python scalar / the masked
                    // singleton param; "hard" hardens the target first (OpRegistry.MaSurface.cs).
                    return MaPutmask(p, ops);

                // ---- sort / set operations -> masked (argsort -> array) --------------------------
                case "sort": return np.ma.sort(ops[0], p.TryGetValue("axis", out var sa) ? sa.GetInt32() : -1,
                                               p.TryGetValue("endwith", out var ew) ? ew.GetBoolean() : true);
                case "argsort": return np.ma.argsort(ops[0], p.TryGetValue("axis", out var aa) ? aa.GetInt32() : -1,
                                                     p.TryGetValue("endwith", out var aew) ? aew.GetBoolean() : true);
                // The arraysetops (ma_sortsetops.jsonl, ma_setops.jsonl): `plain` / `hard` params say which operands NumPy
                // received as a plain ndarray or hardened (SetOpOperand), the rest are NumPy's keyword arguments.
                case "unique": return np.ma.unique(SetOpOperand(p, ops, 0));
                case "intersect1d": return np.ma.intersect1d(SetOpOperand(p, ops, 0), SetOpOperand(p, ops, 1), Flag(p, "assume_unique"));
                case "union1d": return np.ma.union1d(SetOpOperand(p, ops, 0), SetOpOperand(p, ops, 1));
                case "setxor1d": return np.ma.setxor1d(SetOpOperand(p, ops, 0), SetOpOperand(p, ops, 1), Flag(p, "assume_unique"));
                case "setdiff1d": return np.ma.setdiff1d(SetOpOperand(p, ops, 0), SetOpOperand(p, ops, 1), Flag(p, "assume_unique"));
                case "isin":
                    return np.ma.isin(SetOpOperand(p, ops, 0), SetOpOperand(p, ops, 1), Flag(p, "assume_unique"), Flag(p, "invert"));
                case "in1d":
                    return np.ma.in1d(SetOpOperand(p, ops, 0), SetOpOperand(p, ops, 1), Flag(p, "assume_unique"), Flag(p, "invert"));

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
                // ma_surface.jsonl's predicate cases name the argument form ("arg": data, list, the singleton, ...).
                case "isMaskedArray": return p.ContainsKey("arg") ? ApplyMaskedSurface(name, p, ops) : np.ma.isMaskedArray(ops[0]);
                case "allclose": return np.ma.allclose(ops[0], ops[1]);
                case "allequal": return np.ma.allequal(ops[0], ops[1]);
                case "make_mask": return np.ma.make_mask(ops[0].data);
                case "make_mask_none": return np.ma.make_mask_none(new Shape(ParseLongArray(p["shape"])));
                case "mask_or": return np.ma.mask_or(np.ma.getmaskarray(ops[0]), np.ma.getmaskarray(ops[1]));
                case "flatten_mask":
                {
                    // getmaskarray MATERIALIZES an all-False mask for an unmasked operand — a temporary of this
                    // replay expression (the generator spells the same call inline), which the leak sweep would
                    // otherwise count against flatten_mask. A masked operand hands back its OWN mask, which the
                    // fixture still needs, so only the fresh stand-in is released — after flatten_mask has copied
                    // it (flatten_mask always returns a new array, never its argument).
                    var full = np.ma.getmaskarray(ops[0]);
                    try
                    {
                        return np.ma.flatten_mask(full);
                    }
                    finally
                    {
                        if (ops[0]._mask is null)
                            full.Dispose();
                    }
                }
                case "make_mask_descr": return np.ma.make_mask_descr(FuzzCorpus.DtypeToTC(p["dtype"].GetString()));

                default:
                    // The numpy.ma surface tier's ops (ma_surface.jsonl) live in OpRegistry.MaSurface.cs, which throws for
                    // an unknown name.
                    return ApplyMaskedSurface(name, p, ops);
            }
        }

        /// <summary>Dispatch a masked op whose result is a tuple of masked arrays (currently <c>hsplit</c>).</summary>
        /// <param name="op">The <c>ma.</c>-prefixed corpus op key.</param>
        /// <param name="p">Op params (the split count/indices).</param>
        /// <param name="ops">The reconstructed NDMaskedArray operands.</param>
        /// <returns>The per-slot NDMaskedArray results, in NumPy tuple order.</returns>
        /// <exception cref="NotSupportedException">The op key is not a registered masked-tuple op.</exception>
        public static NDMaskedArray[] ApplyMaskedTuple(string op, IReadOnlyDictionary<string, JsonElement> p, NDMaskedArray[] ops)
        {
            string name = op.StartsWith("ma.", StringComparison.Ordinal) ? op.Substring(3) : op;
            switch (name)
            {
                // NumPy's indices_or_sections: an int (equal sections) or a list of cut points (the int[] overload).
                case "hsplit":
                    return p.TryGetValue("indices", out var hi)
                        ? np.ma.hsplit(ops[0], hi.EnumerateArray().Select(x => x.GetInt32()).ToArray())
                        : np.ma.hsplit(ops[0], p["sections"].GetInt32());
                case "unique":
                {
                    // NumPy's (MaskedArray, ndarray[, ndarray]) tuple: the index arrays compare as nomask slots.
                    var r = np.ma.unique(SetOpOperand(p, ops, 0), Flag(p, "return_index"), Flag(p, "return_inverse"));
                    var slots = new NDMaskedArray[r.Length];
                    slots[0] = r.values;
                    int k = 1;
                    if (r.indices is not null)
                        slots[k++] = np.ma.masked_array(r.indices);
                    if (r.inverse_indices is not null)
                        slots[k] = np.ma.masked_array(r.inverse_indices);
                    return slots;
                }
                default:
                    throw new NotSupportedException($"ma tuple op '{op}' not registered in OpRegistry.ApplyMaskedTuple");
            }
        }

        /// <summary>
        ///     A set operation's operand as NumPy received it: the plain data view when the case lists the operand in
        ///     <c>plain</c> (NumPy took the ndarray, so <c>ma.unique</c> runs <c>np.unique</c> proper), else the masked array,
        ///     HARDENED when listed in <c>hard</c> (NumPy's <c>m.harden_mask()</c>, which changes the in-place masked sort).
        /// </summary>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="i">The operand's position.</param>
        /// <returns>An <see cref="NDArray"/> or an <see cref="NDMaskedArray"/>.</returns>
        private static object SetOpOperand(IReadOnlyDictionary<string, JsonElement> p, NDMaskedArray[] ops, int i)
        {
            if (Listed(p, "plain", i))
                return ops[i].data;
            if (Listed(p, "hard", i))
                ops[i].harden_mask();
            return ops[i];
        }

        /// <summary>Whether the int-list param <paramref name="key"/> holds <paramref name="i"/>.</summary>
        /// <param name="p">The case params.</param>
        /// <param name="key">The list's name.</param>
        /// <param name="i">The operand position.</param>
        /// <returns>True when listed.</returns>
        private static bool Listed(IReadOnlyDictionary<string, JsonElement> p, string key, int i)
        {
            if (!p.TryGetValue(key, out var list) || list.ValueKind != JsonValueKind.Array)
                return false;
            foreach (var e in list.EnumerateArray())
                if (e.GetInt32() == i)
                    return true;
            return false;
        }

        /// <summary>A bool keyword param (NumPy's default False when absent).</summary>
        /// <param name="p">The case params.</param>
        /// <param name="key">The keyword.</param>
        /// <returns>The value, false when absent.</returns>
        private static bool Flag(IReadOnlyDictionary<string, JsonElement> p, string key)
            => p.TryGetValue(key, out var v) && v.ValueKind == JsonValueKind.True;

        /// <summary>
        ///     The values of a ma.put case as NumPy received them (the <c>vkind</c> param): an ndarray (<c>array</c>, the
        ///     default: the third operand's data — copied when NumPy owned it, <c>vown</c>, since a reconstructed operand
        ///     never owns its data and the hard-mask branch's resize distinguishes the two), a Python list (<c>list</c>: a
        ///     C# typed array of the list's NumPy dtype — the house map's ndarray, which NumPy's in-range list values cast
        ///     like), a Python scalar (<c>scalar</c>: <c>value</c> + <c>vtype</c>), a masked array (<c>masked</c>: the
        ///     third operand itself) or the masked constant (<c>constant</c>).
        /// </summary>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The reconstructed operands (target, indices, values).</param>
        /// <param name="copy">The array this method made (the owned copy of an <c>array</c> operand), which the caller
        ///     disposes after the put; null when it made none.</param>
        /// <returns>The values object to pass to <c>np.ma.put</c>.</returns>
        /// <exception cref="NotSupportedException">An unknown <c>vkind</c> / <c>vtype</c>, or a list of another dtype
        ///     (a corpus / registry drift).</exception>
        private static object PutValues(IReadOnlyDictionary<string, JsonElement> p, NDMaskedArray[] ops, out NDArray copy)
        {
            copy = null;
            string kind = p.TryGetValue("vkind", out var vk) ? vk.GetString() : "array";
            switch (kind)
            {
                case "array":
                    // copy('K') keeps the operand's layout (an F-order array stays F, whose memory order the resize keeps).
                    if (!Flag(p, "vown"))
                        return ops[2].data;
                    copy = ops[2].data.copy('K');
                    return copy;
                case "list":
                {
                    var d = ops[2].data;
                    return d.typecode switch
                    {
                        NPTypeCode.Int64 => d.ToArray<long>(),
                        NPTypeCode.Double => d.ToArray<double>(),
                        NPTypeCode.Boolean => d.ToArray<bool>(),
                        NPTypeCode.Complex => d.ToArray<System.Numerics.Complex>(),
                        _ => throw new NotSupportedException($"ma.put list of {d.typecode}"),
                    };
                }
                case "masked":
                    return ops[2];
                case "constant":
                    return np.ma.masked;
                case "scalar":
                {
                    var v = p["value"];
                    double Num(JsonElement e) => e.ValueKind == JsonValueKind.String
                        ? e.GetString() switch { "nan" => double.NaN, "inf" => double.PositiveInfinity, _ => double.NegativeInfinity }
                        : e.GetDouble();
                    return p["vtype"].GetString() switch
                    {
                        "int" => (object)v.GetInt64(),
                        "float" => Num(v),
                        "bool" => v.GetBoolean(),
                        "complex" => new System.Numerics.Complex(Num(v[0]), Num(v[1])),
                        var t => throw new NotSupportedException($"ma.put scalar vtype {t}"),
                    };
                }
                default:
                    throw new NotSupportedException($"ma.put vkind {kind}");
            }
        }

        /// <summary>
        ///     The <c>dtype=</c> of ma.mean / ma.var / ma.std / ma.anom: a NumPy dtype name, or null when the case has none
        ///     (NumPy's <c>dtype=None</c>; the older ma tiers never set one).
        /// </summary>
        /// <param name="p">The case params.</param>
        /// <returns>The dtype, or null.</returns>
        /// <exception cref="NotSupportedException">The name is not a NumSharp dtype (a corpus/registry drift).</exception>
        private static DType ParseMaDtype(IReadOnlyDictionary<string, JsonElement> p)
            => p.TryGetValue("dtype", out var v) && v.ValueKind != JsonValueKind.Null ? np.dtype(v.GetString()) : null;

        /// <summary>
        ///     The <c>mean=</c> of ma.var / ma.std as the C# argument that stands for NumPy's: a Python float / int / complex is
        ///     a boxed <see cref="double"/> / <see cref="long"/> / <see cref="System.Numerics.Complex"/> (the house's weak
        ///     Python scalars), <c>np.float32</c> / <c>np.float64</c> a strong 0-d array, and an array form the case's second
        ///     operand — its plain data for an ndarray, the masked array itself for a MaskedArray. Null when the case has no
        ///     mean= (the computed mean).
        /// </summary>
        /// <param name="p">The case params; <c>mean</c> is an object with a <c>kind</c>.</param>
        /// <param name="ops">The reconstructed operands; an array form reads <c>ops[1]</c>.</param>
        /// <returns>The mean argument, or null.</returns>
        /// <exception cref="NotSupportedException">An unknown mean kind (a corpus/registry drift).</exception>
        private static object ParseMaMean(IReadOnlyDictionary<string, JsonElement> p, NDMaskedArray[] ops)
        {
            if (!p.TryGetValue("mean", out var m) || m.ValueKind == JsonValueKind.Null)
                return null;
            return m.GetProperty("kind").GetString() switch
            {
                "pyfloat" => m.GetProperty("value").GetDouble(),
                "pyint" => m.GetProperty("value").GetInt64(),
                "pycomplex" => new System.Numerics.Complex(m.GetProperty("real").GetDouble(), m.GetProperty("imag").GetDouble()),
                "npf32" => NDArray.Scalar((float)m.GetProperty("value").GetDouble()),
                "npf64" => NDArray.Scalar(m.GetProperty("value").GetDouble()),
                // The harness wraps every operand in a masked array; an ndarray mean= passes its data, as NumPy's does.
                "array" => ops[1].data,
                "masked" => ops[1],
                var k => throw new NotSupportedException($"ma mean= kind '{k}' not registered in OpRegistry.ParseMaMean"),
            };
        }

        /// <summary>ddof param for var/std (default 0), read like the ordinary registry's scalar ints.</summary>
        private static int ParseDdof(IReadOnlyDictionary<string, JsonElement> p)
            => p.TryGetValue("ddof", out var v) ? v.GetInt32() : 0;

        /// <summary>A non-nullable axis (default 0), for the stack/concatenate family that has no None axis.</summary>
        private static int ParseAxisInt(IReadOnlyDictionary<string, JsonElement> p)
            => p.TryGetValue("axis", out var v) ? v.GetInt32() : 0;
    }
}
