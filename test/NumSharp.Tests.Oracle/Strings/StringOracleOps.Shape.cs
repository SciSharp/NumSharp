using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for sections K and L (<c>gen_shape_decided</c> / <c>gen_validation_order</c>): the text
    ///     calls whose answer the SHAPE decides (an empty input, lines of one element, a pair the comparison operators cannot
    ///     compare) and the order of the checks NumPy makes before it converts or compares text (interp, digitize, repeat,
    ///     empty conversions, place / putmask with ndarray values, insert / delete with an array obj, searchsorted). Each op
    ///     replays the C# call a ported NumPy line makes, with the generator's arguments.
    /// </summary>
    internal static partial class StringOracleOps
    {
        /// <summary>
        ///     Replays one section-K / section-L op, or reports that <paramref name="op"/> is not one of them.
        /// </summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="result">The call's result (an array, a tuple, a bool, or the mutated operand's owner).</param>
        /// <returns>True when <paramref name="op"/> is a section-K / section-L op.</returns>
        /// <exception cref="InvalidOperationException">A name the op does not know (a generator / table mismatch).</exception>
        private static bool TryInvokeShape(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            NDArray a = ops.Length > 0 ? ops[0] : null;
            NDArray b = ops.Length > 1 ? ops[1] : null;
            switch (op)
            {
                // ---- section K: answers the shape decides ----
                case "shape_reduce":
                    result = ShapeReduce(Str(p, "name"), a, NullableInt(p, "axis"), p["keepdims"].GetBoolean());
                    return true;
                case "truth_out":
                    // The out operand is compared as the mutated operand; the return value is the out itself.
                    result = Str(p, "name") == "any" ? np.any(a, NullableInt(p, "axis"), @out: b) : np.all(a, NullableInt(p, "axis"), @out: b);
                    return true;
                case "truth_where":
                    result = Str(p, "name") == "any" ? np.any(a, NullableInt(p, "axis"), @where: b) : np.all(a, NullableInt(p, "axis"), @where: b);
                    return true;
                case "shape_sort":
                {
                    int? axis = NullableInt(p, "axis");
                    int? kth = NullableInt(p, "kth");
                    result = Str(p, "name") switch
                    {
                        "sort" => np.sort(a, axis),
                        "argsort" => np.argsort(a, axis),
                        "partition" => np.partition(a, kth.Value, axis),
                        "argpartition" => np.argpartition(a, kth.Value, axis),
                        var n => throw new InvalidOperationException($"the string oracle has no sort '{n}'"),
                    };
                    return true;
                }
                case "index_search":
                    result = Str(p, "name") switch
                    {
                        // Every arm is cast to object: a switch over NDArray[] and NDArray arms is typed NDArray through
                        // NumSharp's implicit Array -> NDArray conversion, which cannot hold NDArray elements.
                        "nonzero" => (object)np.nonzero(a).Select(x => (NDArray)x).ToArray(),
                        "where1" => (object)np.where(a).Select(x => (NDArray)x).ToArray(),
                        "flatnonzero" => (object)(NDArray)np.flatnonzero(a),
                        "argwhere" => (object)np.argwhere(a),
                        var n => throw new InvalidOperationException($"the string oracle has no index search '{n}'"),
                    };
                    return true;
                case "unique":
                {
                    bool ri = p["ri"].GetBoolean(), rv = p["rv"].GetBoolean(), rc = p["rc"].GetBoolean();
                    var r = np.unique(a, ri, rv, rc, NullableInt(p, "axis"));
                    // NumPy returns the bare values without a flag, the tuple with one.
                    result = ri || rv || rc ? (object)(NDArray[])r : (NDArray)r;
                    return true;
                }
                case "unique_api":
                    result = Str(p, "name") switch
                    {
                        "unique_values" => (object)np.unique_values(a),
                        "unique_counts" => (object)(NDArray[])np.unique_counts(a),
                        "unique_inverse" => (object)(NDArray[])np.unique_inverse(a),
                        "unique_all" => (object)(NDArray[])np.unique_all(a),
                        var n => throw new InvalidOperationException($"the string oracle has no unique form '{n}'"),
                    };
                    return true;
                case "isin":
                {
                    string kind = p.TryGetValue("kind", out var k) && k.ValueKind == JsonValueKind.String ? k.GetString() : null;
                    result = np.isin(a, b, p["assume_unique"].GetBoolean(), p["invert"].GetBoolean(), kind);
                    return true;
                }
                case "setop":
                {
                    bool au = p.TryGetValue("assume_unique", out var u) && u.GetBoolean();
                    bool ri = p.TryGetValue("return_indices", out var ix) && ix.GetBoolean();
                    result = Str(p, "name") switch
                    {
                        "union1d" => np.union1d(a, b),
                        "intersect1d" => ri ? (object)np.intersect1d(a, b, assume_unique: au, return_indices: true) : np.intersect1d(a, b, au),
                        "setxor1d" => np.setxor1d(a, b, au),
                        "setdiff1d" => np.setdiff1d(a, b, au),
                        var n => throw new InvalidOperationException($"the string oracle has no set routine '{n}'"),
                    };
                    return true;
                }
                case "equality":
                {
                    bool same = p["same"].GetBoolean();
                    result = Str(p, "name") == "array_equal"
                        ? np.array_equal(a, same ? a : b, p["equal_nan"].GetBoolean())
                        : np.array_equiv(a, b);
                    return true;
                }
                case "lexsort":
                    result = np.lexsort(ops, Int(p, "axis"));
                    return true;
                case "sort_complex":
                    result = np.sort_complex(a);
                    return true;
                case "trim_zeros":
                {
                    var ax = p["axis"];
                    string trim = Str(p, "trim");
                    result = ax.ValueKind switch
                    {
                        JsonValueKind.Null => np.trim_zeros(a, trim),
                        JsonValueKind.Number => np.trim_zeros(a, trim, (int?)ax.GetInt32()),
                        _ => np.trim_zeros(a, trim, ax.EnumerateArray().Select(x => x.GetInt32()).ToArray()),
                    };
                    return true;
                }
                case "diff":
                    result = np.diff(a, Int(p, "n"), Int(p, "axis"), prepend: p["prepend"].GetBoolean() ? b : null);
                    return true;

                // ---- section L: the checks before a conversion or a comparison ----
                case "interp":
                {
                    double? period = p.TryGetValue("period", out var pe) && pe.ValueKind == JsonValueKind.Number ? pe.GetDouble() : null;
                    if (p.TryGetValue("complex_left", out var cl) && cl.GetBoolean())
                    {
                        var l = p["left"].EnumerateArray().Select(x => x.GetDouble()).ToArray();
                        result = np.interp(a, b, ops[2], (Complex?)new Complex(l[0], l[1]), null, period);
                    }
                    else
                    {
                        double? left = p.TryGetValue("left", out var le) && le.ValueKind == JsonValueKind.Number ? le.GetDouble() : null;
                        double? right = p.TryGetValue("right", out var ri) && ri.ValueKind == JsonValueKind.Number ? ri.GetDouble() : null;
                        result = np.interp(a, b, ops[2], left, right, period);
                    }
                    return true;
                }
                case "digitize":
                    result = np.digitize(a, b, p["right"].GetBoolean());
                    return true;
                case "repeat_arr":
                    result = np.repeat(a, b, NullableInt(p, "axis"));
                    return true;
                case "where3":
                    result = np.where(a, b, ops[2]);
                    return true;
                case "concat_kw":
                {
                    int? outAt = NullableInt(p, "out");
                    var arrays = outAt is { } k ? ops.Take(k).ToArray() : ops;
                    string casting = p["casting"].ValueKind == JsonValueKind.String ? Str(p, "casting") : "same_kind";
                    result = np.concatenate(arrays, 0, outAt is { } o ? ops[o] : null, Dt(p, "dtype"), casting);
                    return true;
                }
                case "choose_mode":
                    result = np.choose(np.array(Longs(p["index"])), ops, null, Str(p, "mode"));
                    return true;
                case "place_arr":
                case "putmask_arr":
                {
                    // The values are an ndarray (the NDArray overload: NumPy's 'safe' rule); the mask comes from the
                    // parameters, or is the third operand (a text mask).
                    NDArray mask = p["mask"].ValueKind == JsonValueKind.Null ? ops[2] : MaskOf(p["mask"]);
                    if (op == "place_arr")
                        np.place(a, mask, b);
                    else
                        np.putmask(a, mask, b);
                    result = a;
                    return true;
                }
                case "insert_obj":
                    result = np.insert(a, b, ops[2], NullableInt(p, "axis"));
                    return true;
                case "delete_obj":
                    result = np.delete(a, b, NullableInt(p, "axis"));
                    return true;
                case "searchsorted_kw":
                    result = np.searchsorted(a, b, Str(p, "side"), p["sorter"].GetBoolean() ? ops[2] : null);
                    return true;
            }
            result = null;
            return false;
        }

        /// <summary>A truth test, count or arg-reduction with NumPy's <c>axis</c> / <c>keepdims</c>.</summary>
        /// <param name="name">The NumPy function name.</param>
        /// <param name="a">The input.</param>
        /// <param name="axis">The axis, or null.</param>
        /// <param name="keepdims">NumPy's keepdims.</param>
        /// <returns>The call's result (a bool / long where NumPy returns a scalar).</returns>
        /// <exception cref="InvalidOperationException">An unknown name.</exception>
        private static object ShapeReduce(string name, NDArray a, int? axis, bool keepdims)
        {
            bool flat = axis is null && !keepdims;
            switch (name)
            {
                case "any": return flat ? np.any(a) : np.any(a, axis, keepdims: keepdims);
                case "all": return flat ? np.all(a) : np.all(a, axis, keepdims: keepdims);
                case "count_nonzero": return flat ? np.count_nonzero(a) : np.count_nonzero(a, axis.Value, keepdims);
                case "argmax": return flat ? np.argmax(a) : np.argmax(a, axis.Value, keepdims);
                case "argmin": return flat ? np.argmin(a) : np.argmin(a, axis.Value, keepdims);
                case "nanargmax": return np.nanargmax(a, axis, keepdims: keepdims);
                case "nanargmin": return np.nanargmin(a, axis, keepdims: keepdims);
            }
            throw new InvalidOperationException($"the string oracle has no shape reduction '{name}'");
        }

        /// <summary>A JSON list of bools (possibly empty) as a 1-D bool mask.</summary>
        /// <param name="e">The list.</param>
        /// <returns>The mask.</returns>
        private static NDArray MaskOf(JsonElement e)
            => np.array(e.EnumerateArray().Select(x => x.GetBoolean()).ToArray());
    }
}
