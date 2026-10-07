using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for section J (<c>gen_string_requests</c> / <c>gen_text_precedence</c>): an unsized
    ///     string <c>dtype=</c> request on the ufuncs, reductions, gufuncs, <c>np.clip</c> and <c>np.einsum</c>, and the
    ///     order in which NumPy reports the problems of a call that involves text (out= / where= / axis / q / method /
    ///     kth / weights / concatenation). Each op replays the C# call a ported NumPy line makes, with the generator's
    ///     arguments.
    /// </summary>
    internal static partial class StringOracleOps
    {
        /// <summary>
        ///     Replays one section-J op, or reports that <paramref name="op"/> is not one of them.
        /// </summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="result">The call's result.</param>
        /// <returns>True when <paramref name="op"/> is a section-J op (its result is in <paramref name="result"/>).</returns>
        /// <exception cref="InvalidOperationException">A name the op does not know (a generator / table mismatch).</exception>
        private static bool TryInvokePrecedence(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            NDArray a = ops.Length > 0 ? ops[0] : null;
            NDArray b = ops.Length > 1 ? ops[1] : null;
            switch (op)
            {
                case "reduce":
                    result = Reduce(Str(p, "name"), a, p);
                    return true;
                case "trace":
                    result = np.trace(np.stack(new[] { a, a }), dtype: Dt(p, "dtype"));
                    return true;
                case "gufunc":
                    result = Gufunc(Str(p, "name"), a, b, Dt(p, "dtype"), NullableInt(p, "axis"));
                    return true;
                case "clip":
                    result = Str(p, "bounds") switch
                    {
                        "both" => np.clip(a, ops[1], ops[2], dtype: Dt(p, "dtype")),
                        "min" => np.clip(a, ops[1], null, dtype: Dt(p, "dtype")),
                        "max" => np.clip(a, null, ops[1], dtype: Dt(p, "dtype")),
                        _ => np.clip(a, null, null, dtype: Dt(p, "dtype")),
                    };
                    return true;
                case "einsum":
                    result = np.einsum(Str(p, "subscripts"), ops, dtype: Dt(p, "dtype"), casting: Str(p, "casting"));
                    return true;
                case "ufunc_kw":
                    result = UfuncKw(p, ops);
                    return true;
                case "divmod_out":
                    result = np.divmod(a, b, @out: (ops[2], ops[3]));
                    return true;
                case "quantile":
                    result = Quantile(p, a);
                    return true;
                case "average_w":
                    result = np.average(a, NullableInt(p, "axis"), weights: b);
                    return true;
                case "cov":
                    result = Cov(p, ops);
                    return true;
                case "partition":
                    result = Str(p, "name") == "partition"
                        ? np.partition(a, Int(p, "kth"), NullableInt(p, "axis"))
                        : np.argpartition(a, Int(p, "kth"), NullableInt(p, "axis"));
                    return true;
                case "sort_inplace":
                    a.sort(Int(p, "axis"));
                    result = a;
                    return true;
            }

            result = null;
            return false;
        }

        /// <summary>
        ///     A reduction with the generator's <c>axis</c> (int or None) or <c>axes</c> (a tuple) and optional
        ///     <c>dtype</c>, spelled as NumSharp's NumPy-shaped overloads (<c>maximum.reduce</c> is NumSharp's
        ///     <c>np.amax(dtype:)</c> extension, which is NumPy's <c>np.maximum.reduce(dtype=)</c>).
        /// </summary>
        /// <param name="name">The NumPy function name.</param>
        /// <param name="a">The array.</param>
        /// <param name="p">The case parameters.</param>
        /// <returns>The call's result.</returns>
        /// <exception cref="InvalidOperationException">An unknown name.</exception>
        private static object Reduce(string name, NDArray a, Dictionary<string, JsonElement> p)
        {
            var dt = Dt(p, "dtype");
            if (p.TryGetValue("axes", out var axesElement))
            {
                int[] axes = axesElement.EnumerateArray().Select(x => x.GetInt32()).ToArray();
                return name switch
                {
                    "all" => np.all(a, axes, null),
                    "any" => np.any(a, axes, null),
                    "ptp" => np.ptp(a, axes),
                    "median" => np.median(a, axes),
                    "nanmedian" => np.nanmedian(a, axes),
                    "average" => np.average(a, axes),
                    "percentile" => np.percentile(a, 50.0, axes),
                    "quantile" => np.quantile(a, 0.5, axes),
                    "nanpercentile" => np.nanpercentile(a, 50.0, axes),
                    "nanquantile" => np.nanquantile(a, 0.5, axes),
                    _ => throw new InvalidOperationException($"the string oracle has no tuple-axis reduction '{name}'"),
                };
            }

            int? axis = NullableInt(p, "axis");
            switch (name)
            {
                case "sum": return np.sum(a, axis, dt);
                case "prod": return np.prod(a, axis, dt);
                // NumSharp's mean takes dtype= with an explicit axis only; the generator passes axis 0 for a 1-D input.
                case "mean": return dt is null ? (axis is { } m ? np.mean(a, m) : np.mean(a)) : np.mean(a, axis ?? 0, dt);
                case "var": return axis is { } v ? np.var(a, v, dtype: dt) : np.var(a, dtype: dt);
                case "std": return axis is { } s ? np.std(a, s, dtype: dt) : np.std(a, dtype: dt);
                case "nansum": return np.nansum(a, axis);
                case "nanprod": return np.nanprod(a, axis);
                case "nanmean": return np.nanmean(a, axis, dtype: dt);
                case "nanvar": return np.nanvar(a, axis, dtype: dt);
                case "nanstd": return np.nanstd(a, axis, dtype: dt);
                case "cumsum": return np.cumsum(a, axis, dt);
                case "cumprod": return np.cumprod(a, axis, dt);
                case "nancumsum": return np.nancumsum(a, axis, dt);
                case "nancumprod": return np.nancumprod(a, axis, dt);
                case "maximum.reduce": return np.amax(a, axis, dtype: dt);
                case "minimum.reduce": return np.amin(a, axis, dtype: dt);
                case "max": return np.max(a, axis: axis);
                case "min": return np.min(a, axis: axis);
                case "amax": return np.amax(a, axis: axis);
                case "amin": return np.amin(a, axis: axis);
                case "nanmax": return np.nanmax(a, axis: axis);
                case "nanmin": return np.nanmin(a, axis: axis);
                case "ptp": return np.ptp(a, axis: axis);
                case "argmax": return axis is { } am ? np.argmax(a, am) : NDArray.Scalar(np.argmax(a));
                case "argmin": return axis is { } an ? np.argmin(a, an) : NDArray.Scalar(np.argmin(a));
                case "nanargmax": return np.nanargmax(a, axis: axis);
                case "nanargmin": return np.nanargmin(a, axis: axis);
                case "median": return np.median(a, axis: axis);
                case "nanmedian": return np.nanmedian(a, axis: axis);
                case "average": return np.average(a, axis: axis);
                case "any": return np.any(a, axis: axis);
                case "all": return np.all(a, axis: axis);
                case "count_nonzero": return axis is { } cn ? np.count_nonzero(a, cn) : NDArray.Scalar(np.count_nonzero(a));
                case "sort": return np.sort(a, axis);
                case "argsort": return np.argsort(a, axis);
                case "percentile": return np.percentile(a, 50.0, axis);
                case "quantile": return np.quantile(a, 0.5, axis);
                case "nanpercentile": return np.nanpercentile(a, 50.0, axis);
                case "nanquantile": return np.nanquantile(a, 0.5, axis);
                case "partition": return np.partition(a, 0, axis);
                case "argpartition": return np.argpartition(a, 0, axis);
            }
            throw new InvalidOperationException($"the string oracle has no reduction '{name}'");
        }

        /// <summary>A gufunc call with the generator's dtype and optional axis.</summary>
        /// <param name="name">The gufunc name.</param>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="dt">The dtype request, or null.</param>
        /// <param name="axis">The axis argument, or null.</param>
        /// <returns>The call's result.</returns>
        /// <exception cref="InvalidOperationException">An unknown name.</exception>
        private static object Gufunc(string name, NDArray x1, NDArray x2, DType dt, int? axis) => name switch
        {
            "matmul" => np.matmul(x1, x2, axis: axis, dtype: dt),
            "vecdot" => np.vecdot(x1, x2, axis: axis, dtype: dt),
            "matvec" => np.matvec(x1, x2, axis: axis, dtype: dt),
            "vecmat" => np.vecmat(x1, x2, axis: axis, dtype: dt),
            _ => throw new InvalidOperationException($"the string oracle has no gufunc '{name}'"),
        };

        /// <summary>
        ///     A ufunc call with keyword operands: the first <c>nin</c> operands are the inputs, <c>out</c> / <c>where</c>
        ///     name the operand positions of those keywords.
        /// </summary>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The operands.</param>
        /// <returns>The call's result.</returns>
        /// <exception cref="InvalidOperationException">An unknown ufunc.</exception>
        private static object UfuncKw(Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            int nin = Int(p, "nin");
            NDArray x = ops[0], y = nin > 1 ? ops[1] : null;
            NDArray @out = NullableInt(p, "out") is { } oi ? ops[oi] : null;
            NDArray where = NullableInt(p, "where") is { } wi ? ops[wi] : null;
            var dt = Dt(p, "dtype");
            switch (Str(p, "name"))
            {
                case "add": return np.add(x, y, @out, where, dt);
                case "subtract": return np.subtract(x, y, @out, where, dt);
                case "multiply": return np.multiply(x, y, @out, where, dt);
                case "equal": return np.equal(x, y, @out, where, dt);
                case "less": return np.less(x, y, @out, where, dt);
                case "logical_and": return np.logical_and(x, y, @out, where, dt);
                case "logical_or": return np.logical_or(x, y, @out, where, dt);
                case "sqrt": return np.sqrt(x, @out, where, dt);
                case "negative": return np.negative(x, @out, where, dt);
                case "logical_not": return np.logical_not(x, @out, where, dt);
                case "isnan": return np.isnan(x, @out, where, dt);
            }
            throw new InvalidOperationException($"the string oracle has no keyword ufunc '{Str(p, "name")}'");
        }

        /// <summary>A quantile-family call: q a number or a list, an axis, a method and keepdims.</summary>
        /// <param name="p">The case parameters.</param>
        /// <param name="a">The array.</param>
        /// <returns>The call's result.</returns>
        /// <exception cref="InvalidOperationException">An unknown name.</exception>
        private static object Quantile(Dictionary<string, JsonElement> p, NDArray a)
        {
            string name = Str(p, "name");
            int? axis = NullableInt(p, "axis");
            string method = Str(p, "method");
            bool keepdims = p.TryGetValue("keepdims", out var k) && k.GetBoolean();
            var q = p["q"];
            if (q.ValueKind == JsonValueKind.Array)
            {
                double[] qs = q.EnumerateArray().Select(x => x.GetDouble()).ToArray();
                return name switch
                {
                    "percentile" => np.percentile(a, qs, axis, method: method, keepdims: keepdims),
                    "quantile" => np.quantile(a, qs, axis, method: method, keepdims: keepdims),
                    "nanpercentile" => np.nanpercentile(a, qs, axis, method: method, keepdims: keepdims),
                    "nanquantile" => np.nanquantile(a, qs, axis, method: method, keepdims: keepdims),
                    _ => throw new InvalidOperationException($"the string oracle has no quantile '{name}'"),
                };
            }
            double qv = q.GetDouble();
            return name switch
            {
                "percentile" => np.percentile(a, qv, axis, method: method, keepdims: keepdims),
                "quantile" => np.quantile(a, qv, axis, method: method, keepdims: keepdims),
                "nanpercentile" => np.nanpercentile(a, qv, axis, method: method, keepdims: keepdims),
                "nanquantile" => np.nanquantile(a, qv, axis, method: method, keepdims: keepdims),
                _ => throw new InvalidOperationException($"the string oracle has no quantile '{name}'"),
            };
        }

        /// <summary>A <c>np.cov</c> / <c>np.corrcoef</c> call with the generator's keywords.</summary>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The operands (m, then y when <c>y</c> names its position).</param>
        /// <returns>The call's result.</returns>
        private static object Cov(Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            bool corr = p.TryGetValue("corr", out var c) && c.GetBoolean();
            bool rowvar = !p.TryGetValue("rowvar", out var r) || r.GetBoolean();
            NDArray y = NullableInt(p, "y") is { } yi ? ops[yi] : null;
            var dt = Dt(p, "dtype");
            if (corr)
                return np.corrcoef(ops[0], y, rowvar, dt);
            NDArray fw = p.TryGetValue("fweights", out var f) ? np.array(f.EnumerateArray().Select(x => x.GetDouble()).ToArray()) : null;
            NDArray aw = p.TryGetValue("aweights", out var w) ? np.array(w.EnumerateArray().Select(x => x.GetDouble()).ToArray()) : null;
            // NumPy's fweights=[1, 1, 1] is an int list (Python ints): cast the all-integral values to int64.
            if (fw is not null && fw.ToArray<double>().All(v => v == Math.Floor(v)))
                fw = fw.astype(np.int64);
            return np.cov(ops[0], y, rowvar, fweights: fw, aweights: aw, dtype: dt);
        }
    }
}
