using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The C# side of <c>test/oracle/gen_surface_oracle.py</c>: the APIs the coverage inventory lists as available but
    ///     unproven — the np.linalg Array-API forms, the histogram family, array_equiv, the memory predicates,
    ///     broadcast_shapes, apply_along_axis / apply_over_axes, bmat, the text formatters, the ndarray members and the
    ///     attributes of finfo, iinfo, dtype, ndarray.flags and poly1d. Each op key is the API's own name, so the coverage
    ///     join credits the contracts to it.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Every adapter calls the public NumSharp member the way a C# port of the NumPy call would, and converts only
    ///     what C# cannot spell the NumPy way: a function argument arrives as a name from the generator's menu
    ///     (<see cref="AlongAxisFunc"/>, <see cref="OverAxesFunc"/>), two views of one buffer as step recipes
    ///     (<see cref="ApplySteps"/>), a Python list or scalar result as NumPy's <c>repr</c> text (<see cref="PyRepr"/>).
    ///     A converter must never repair a NumSharp answer: it formats or wraps it, so a wrong value stays wrong.
    ///     </para>
    ///     <para>
    ///     Reached from the default branches of <c>ApplyExtended</c>, <c>ApplyTuple</c>, <c>ApplyText</c> and
    ///     <c>ApplyDtype</c>; an unknown key still throws <see cref="NotSupportedException"/> there.
    ///     </para>
    /// </remarks>
    public static partial class OpRegistry
    {
        // ---- array and scalar results -----------------------------------------------------------------------------

        /// <summary>The surface tier's array / scalar ops (a scalar is returned as a 0-d array).</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>NumSharp's result.</returns>
        /// <exception cref="NotSupportedException">The key is not a surface op either (an unregistered key).</exception>
        internal static NDArray ApplySurface(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            switch (op)
            {
                case "linalg.cross": return np.linalg.cross(ops[0], ops[1], SurfInt(p, "axis"));
                case "linalg.diagonal": return np.linalg.diagonal(ops[0], SurfInt(p, "offset"));
                case "linalg.outer": return np.linalg.outer(ops[0], ops[1]);
                case "linalg.trace":
                    return np.linalg.trace(ops[0], SurfInt(p, "offset"), SurfDType(p, "dtype"));
                case "histogram_bin_edges": return HistogramBinEdges(p, ops);
                case "array_equiv": return NDArray.Scalar(np.array_equiv(ops[0], ops[1]));
                case "shares_memory":
                case "may_share_memory":
                    return MemoryPredicate(op, p, ops);
                case "broadcast_shapes":
                {
                    var shapes = p["shapes"].EnumerateArray()
                        .Select(s => new Shape(s.EnumerateArray().Select(x => x.GetInt64()).ToArray())).ToArray();
                    return np.array(np.broadcast_shapes(shapes).dimensions.ToArray());
                }
                case "apply_along_axis":
                {
                    // apply_along_axis leaves every array the callback returns to the CALLER (documented ownership:
                    // a callback may return an array it keeps), so the per-slice results of the menu functions are
                    // released by this scope; only the reassembled result leaves it.
                    using var scope = NDScope.Open();
                    return scope.Returns(np.apply_along_axis(AlongAxisFunc(p["func"].GetString()), SurfInt(p, "axis"),
                                                             ops[0]));
                }
                case "apply_over_axes":
                {
                    // The same ownership rule: each per-axis result the callback returns is the caller's.
                    using var scope = NDScope.Open();
                    var f = OverAxesFunc(p["func"].GetString());
                    var axes = p["axes"];
                    return scope.Returns(axes.ValueKind == JsonValueKind.Array
                        ? np.apply_over_axes(f, ops[0], axes.EnumerateArray().Select(x => x.GetInt32()).ToArray())
                        : np.apply_over_axes(f, ops[0], axes.GetInt32()));
                }
                case "bmat": return Bmat(p, ops);
                case "ndarray.shape": return np.array(ops[0].shape);
                // numpy.shape(a): NumSharp has no top-level function; the inventory credits the shape property.
                case "shape": return np.array(ops[0].shape);
                case "cumulative_sum":
                case "cumulative_prod":
                {
                    // NumSharp has no cumulative_sum / cumulative_prod: the coverage inventory aliases them to
                    // np.cumsum / np.cumprod, and these cases gate that alias. include_initial has no counterpart, so
                    // the port of a call that passes it has nothing to call.
                    if (p.TryGetValue("include_initial", out var ii) && ii.GetBoolean())
                        throw new NotSupportedException(
                            $"numpy.{op}(include_initial=True) has no NumSharp counterpart (the alias is np.{(op == "cumulative_sum" ? "cumsum" : "cumprod")})");
                    int? axis = SurfNullableInt(p, "axis");
                    DType dtype = SurfDType(p, "dtype");
                    return op == "cumulative_sum" ? np.cumsum(ops[0], axis, dtype) : np.cumprod(ops[0], axis, dtype);
                }
                case "ndarray.tofile":
                {
                    using var stream = new MemoryStream();
                    ops[0].tofile(stream, SurfStr(p, "sep"), SurfStr(p, "format"));
                    return np.array(stream.ToArray());
                }
                case "ndarray.to_device":
                    return ops[0].to_device(SurfStr(p, "device"),
                                            p.TryGetValue("stream", out var stream2) && stream2.ValueKind != JsonValueKind.Null
                                                ? stream2.GetInt64()
                                                : null);
                case "ndarray.setfield":
                {
                    object value = p.TryGetValue("value_operand", out var vi) ? ops[vi.GetInt32()] : PyNumber(p["value"]);
                    ops[0].setfield(value, SurfStr(p, "dtype"), SurfInt(p, "offset"));
                    return ops[0];
                }
                // A poly1d OWNS its coefficient array (Dispose releases it), so returning .coeffs hands that ownership
                // to the caller, which disposes the result; a member that returns anything else disposes the polynomial.
                case "poly1d.coeffs": return new poly1d(ops[0]).coeffs;
                case "poly1d.c": return new poly1d(ops[0]).c;
                case "poly1d.coef": return new poly1d(ops[0]).coef;
                case "poly1d.coefficients": return new poly1d(ops[0]).coefficients;
                case "poly1d.order":
                {
                    using var poly = new poly1d(ops[0]);
                    return NDArray.Scalar((long)poly.order);
                }
                case "poly1d.o":
                {
                    using var poly = new poly1d(ops[0]);
                    return NDArray.Scalar((long)poly.o);
                }
                case "poly1d.deriv":
                {
                    using var poly = new poly1d(ops[0]);
                    return poly.deriv(SurfInt(p, "m")).coeffs;
                }
                case "poly1d.integ":
                {
                    // k: NumPy's integration constants (null = the default 0); the array is this adapter's own.
                    using NDArray k = p.TryGetValue("k", out var kv) && kv.ValueKind == JsonValueKind.Array
                        ? np.array(kv.EnumerateArray().Select(x => x.GetDouble()).ToArray())
                        : null;
                    using var poly = new poly1d(ops[0]);
                    return poly.integ(SurfInt(p, "m"), k).coeffs;
                }
            }
            if (op.StartsWith("finfo.", StringComparison.Ordinal))
                return FinfoAttribute(op.Substring("finfo.".Length), np.finfo(SurfStr(p, "dtype")));
            if (op.StartsWith("iinfo.", StringComparison.Ordinal))
                return IinfoAttribute(op.Substring("iinfo.".Length), np.iinfo(SurfStr(p, "dtype")));
            if (op.StartsWith("ndarray.flags.", StringComparison.Ordinal))
                return FlagAttribute(op.Substring("ndarray.flags.".Length), ops[0].flags);
            // The iteration objects, the buffer size, ndarray.data and np.frompyfunc (OpRegistry.Surface.Objects.cs).
            if (TrySurfaceObject(op, p, ops, out var objectResult))
                return objectResult;
            throw new NotSupportedException($"op '{op}' is not registered in OpRegistry");
        }

        // ---- tuple results ----------------------------------------------------------------------------------------

        /// <summary>The surface tier's tuple ops: the histogram family and ndarray.base.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The slots, in NumPy's tuple order.</returns>
        /// <exception cref="NotSupportedException">The key is not a surface tuple op either.</exception>
        internal static NDArray[] ApplySurfaceTuple(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            switch (op)
            {
                case "histogram":
                {
                    var r = Histogram(p, ops);
                    return new[] { r.hist, r.bin_edges };
                }
                case "histogram2d":
                {
                    var r = Histogram2d(p, ops);
                    return new[] { r.H, r.xedges, r.yedges };
                }
                case "histogramdd":
                {
                    var r = Histogramdd(p, ops);
                    return new[] { r.H }.Concat(r.edges).ToArray();
                }
                case "ndarray.base":
                {
                    var view = ApplySteps(ops[0], p["views"][0]);
                    try
                    {
                        var b = view.@base;
                        return b is null ? Array.Empty<NDArray>() : new[] { b };
                    }
                    finally
                    {
                        ReleaseView(view, ops[0]);   // base holds its own reference to the owner's storage
                    }
                }
                // The iteration objects' tuple results and np.frompyfunc's two-output form; unknown keys throw there.
                default:
                    return SurfaceObjectTuple(op, p, ops);
            }
        }

        // ---- text results -----------------------------------------------------------------------------------------

        /// <summary>The surface tier's text ops: the formatters, tolist / tofile text, device and the str-valued
        /// attributes.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>NumSharp's text (a Python list or tuple as NumPy's <c>repr</c>).</returns>
        /// <exception cref="NotSupportedException">The key is not a surface text op either.</exception>
        internal static string ApplySurfaceText(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            switch (op)
            {
                case "array2string":
                    return np.array2string(ops[0],
                        max_line_width: SurfNullableInt(p, "max_line_width"),
                        precision: SurfNullableInt(p, "precision"),
                        suppress_small: p.TryGetValue("suppress_small", out var ss) ? ss.GetBoolean() : null,
                        separator: p.TryGetValue("separator", out var sep) ? sep.GetString() : " ",
                        prefix: p.TryGetValue("prefix", out var pre) ? pre.GetString() : "",
                        threshold: SurfNullableInt(p, "threshold"),
                        edgeitems: SurfNullableInt(p, "edgeitems"),
                        sign: p.TryGetValue("sign", out var sg) ? sg.GetString()[0] : null,
                        floatmode: p.TryGetValue("floatmode", out var fm) ? fm.GetString() : null,
                        suffix: p.TryGetValue("suffix", out var suf) ? suf.GetString() : "");
                case "format_float_positional":
                    return np.format_float_positional(ScalarAsDouble(ops[0]),
                        precision: SurfNullableInt(p, "precision"),
                        unique: !p.TryGetValue("unique", out var u) || u.GetBoolean(),
                        fractional: !p.TryGetValue("fractional", out var fr) || fr.GetBoolean(),
                        trim: p.TryGetValue("trim", out var tr) ? tr.GetString()[0] : 'k',
                        sign: p.TryGetValue("sign", out var sp) && sp.GetBoolean(),
                        pad_left: SurfNullableInt(p, "pad_left"),
                        pad_right: SurfNullableInt(p, "pad_right"),
                        min_digits: SurfNullableInt(p, "min_digits"));
                case "format_float_scientific":
                    return np.format_float_scientific(ScalarAsDouble(ops[0]),
                        precision: SurfNullableInt(p, "precision"),
                        unique: !p.TryGetValue("unique", out var u2) || u2.GetBoolean(),
                        trim: p.TryGetValue("trim", out var tr2) ? tr2.GetString()[0] : 'k',
                        sign: p.TryGetValue("sign", out var sp2) && sp2.GetBoolean(),
                        pad_left: SurfNullableInt(p, "pad_left"),
                        exp_digits: SurfNullableInt(p, "exp_digits"),
                        min_digits: SurfNullableInt(p, "min_digits"));
                case "base_repr":
                    return np.base_repr(PyInt(p["number"]), SurfInt(p, "base"), SurfInt(p, "padding"));
                case "binary_repr":
                    return np.binary_repr(PyInt(p["num"]), SurfNullableInt(p, "width"));
                case "typename": return np.typename(SurfStr(p, "char"));
                case "datetime_data":
                {
                    var (unit, count) = np.datetime_data(SurfStr(p, "dtype"));
                    return "(" + PyRepr(unit) + ", " + PyRepr(count) + ")";
                }
                case "ndarray.tolist": return PyRepr(ops[0].tolist());
                case "ndarray.tofile":
                {
                    using var stream = new MemoryStream();
                    ops[0].tofile(stream, SurfStr(p, "sep"), SurfStr(p, "format"));
                    return Encoding.Latin1.GetString(stream.ToArray());
                }
                case "ndarray.device": return ops[0].device;
                case "iinfo.kind": return np.iinfo(SurfStr(p, "dtype")).kind.ToString();
                case "poly1d.variable":
                {
                    using var poly = new poly1d(ops[0], variable: SurfNullableStr(p, "variable"));
                    return poly.variable;
                }
            }
            if (op.StartsWith("dtype.", StringComparison.Ordinal))
                return DTypeAttribute(op.Substring("dtype.".Length), np.dtype(SurfStr(p, "dtype")));
            // flatiter.coords, nditer's text members, the print options and ndarray.data's text members.
            if (TrySurfaceObjectText(op, p, ops, out var objectText))
                return objectText;
            throw new NotSupportedException($"text op '{op}' is not registered in OpRegistry");
        }

        // ---- dtype results ----------------------------------------------------------------------------------------

        /// <summary>The surface tier's dtype ops: ndarray.dtype, finfo.dtype, iinfo.dtype.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The dtype's type code (compared by NumPy name).</returns>
        /// <exception cref="NotSupportedException">The key is not a surface dtype op either.</exception>
        internal static NPTypeCode ApplySurfaceDtype(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
            => op switch
            {
                "ndarray.dtype" => ops[0].dtype.typecode,
                "finfo.dtype" => np.finfo(SurfStr(p, "dtype")).dtype.typecode,
                "iinfo.dtype" => np.iinfo(SurfStr(p, "dtype")).dtype.typecode,
                _ => throw new NotSupportedException($"dtype op '{op}' is not registered in OpRegistry"),
            };

        // ---- the histogram family ---------------------------------------------------------------------------------

        /// <summary>np.histogram with NumPy's polymorphic <c>bins</c> (an int, an estimator name or an edge array).</summary>
        /// <param name="p">The case params (<c>bins</c> / <c>bins_operand</c>, <c>range</c>, <c>density</c>,
        /// <c>weights_operand</c>).</param>
        /// <param name="ops">Operand 0 is the sample; the others are edges / weights named by index.</param>
        /// <returns>NumSharp's (hist, bin_edges).</returns>
        private static np.HistogramResult Histogram(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var range = SurfRange(p);
            bool density = p.TryGetValue("density", out var d) && d.ValueKind == JsonValueKind.True;
            var weights = p.TryGetValue("weights_operand", out var w) ? ops[w.GetInt32()] : null;
            if (p.TryGetValue("bins_operand", out var bo))
                return np.histogram(ops[0], ops[bo.GetInt32()], range, density, weights);
            var bins = p["bins"];
            return bins.ValueKind switch
            {
                JsonValueKind.String => np.histogram(ops[0], bins.GetString(), range, density, weights),
                JsonValueKind.Null => np.histogram(ops[0], 10, range, density, weights),
                _ => np.histogram(ops[0], bins.GetInt32(), range, density, weights),
            };
        }

        /// <summary>np.histogram_bin_edges with the same <c>bins</c> forms as <see cref="Histogram"/>.</summary>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The operands.</param>
        /// <returns>NumSharp's edges.</returns>
        private static NDArray HistogramBinEdges(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var range = SurfRange(p);
            var weights = p.TryGetValue("weights_operand", out var w) ? ops[w.GetInt32()] : null;
            if (p.TryGetValue("bins_operand", out var bo))
                return np.histogram_bin_edges(ops[0], ops[bo.GetInt32()], range, weights);
            var bins = p["bins"];
            return bins.ValueKind switch
            {
                JsonValueKind.String => np.histogram_bin_edges(ops[0], bins.GetString(), range, weights),
                JsonValueKind.Null => np.histogram_bin_edges(ops[0], 10, range, weights),
                _ => np.histogram_bin_edges(ops[0], bins.GetInt32(), range, weights),
            };
        }

        /// <summary>np.histogram2d with an int, a per-axis int pair, one edge array or a pair of edge arrays.</summary>
        /// <param name="p">The case params.</param>
        /// <param name="ops">Operands 0 and 1 are x and y; the others are edges / weights named by index.</param>
        /// <returns>NumSharp's (H, xedges, yedges).</returns>
        private static np.Histogram2dResult Histogram2d(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var range = SurfRanges(p);
            bool density = p.TryGetValue("density", out var d) && d.ValueKind == JsonValueKind.True;
            var weights = p.TryGetValue("weights_operand", out var w) ? ops[w.GetInt32()] : null;
            if (p.TryGetValue("bins_operand", out var bo))
                return np.histogram2d(ops[0], ops[1], ops[bo.GetInt32()], range, density, weights);
            if (p.TryGetValue("bins_operands", out var bos))
                return np.histogram2d(ops[0], ops[1], bos.EnumerateArray().Select(i => ops[i.GetInt32()]).ToArray(),
                                      range, density, weights);
            // One edge list (a Python float list of length != 2) for both axes: NumSharp's double[] overload.
            if (p.TryGetValue("bins_edges", out var be))
                return np.histogram2d(ops[0], ops[1], be.EnumerateArray().Select(PyFloat).ToArray(), range, density,
                                      weights);
            var bins = p["bins"];
            return bins.ValueKind == JsonValueKind.Array
                ? np.histogram2d(ops[0], ops[1], bins.EnumerateArray().Select(x => x.GetInt32()).ToArray(), range,
                                 density, weights)
                : np.histogram2d(ops[0], ops[1], bins.GetInt32(), range, density, weights);
        }

        /// <summary>np.histogramdd over an (N, D) sample or a list of D 1-D samples.</summary>
        /// <param name="p">The case params (<c>sample_list</c> makes every operand one coordinate array).</param>
        /// <param name="ops">The operands.</param>
        /// <returns>NumSharp's (H, edges).</returns>
        private static np.HistogramddResult Histogramdd(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var range = SurfRanges(p);
            bool density = p.TryGetValue("density", out var d) && d.ValueKind == JsonValueKind.True;
            var weights = p.TryGetValue("weights_operand", out var w) ? ops[w.GetInt32()] : null;
            if (p.TryGetValue("sample_list", out var sl) && sl.ValueKind == JsonValueKind.True)
                return np.histogramdd(ops, p["bins"].GetInt32(), range, density, weights);
            if (p.TryGetValue("bins_operands", out var bos))
                return np.histogramdd(ops[0], bos.EnumerateArray().Select(i => ops[i.GetInt32()]).ToArray(), range,
                                      density, weights);
            var bins = p["bins"];
            return bins.ValueKind == JsonValueKind.Array
                ? np.histogramdd(ops[0], bins.EnumerateArray().Select(x => x.GetInt32()).ToArray(), range, density,
                                 weights)
                : np.histogramdd(ops[0], bins.GetInt32(), range, density, weights);
        }

        /// <summary>A <c>range</c> param <c>[lo, hi]</c> (or null) as np.histogram's tuple.</summary>
        private static (double, double)? SurfRange(IReadOnlyDictionary<string, JsonElement> p)
        {
            if (!p.TryGetValue("range", out var r) || r.ValueKind == JsonValueKind.Null)
                return null;
            return (PyFloat(r[0]), PyFloat(r[1]));
        }

        /// <summary>A per-axis <c>range</c> param <c>[[lo, hi], …]</c> (or null) as np.histogram2d / dd's array.</summary>
        private static (double, double)?[] SurfRanges(IReadOnlyDictionary<string, JsonElement> p)
        {
            if (!p.TryGetValue("range", out var r) || r.ValueKind == JsonValueKind.Null)
                return null;
            return r.EnumerateArray().Select(x => ((double, double)?)(PyFloat(x[0]), PyFloat(x[1]))).ToArray();
        }

        // ---- memory predicates, view recipes, bmat ----------------------------------------------------------------

        /// <summary>np.shares_memory / np.may_share_memory over two views of operand 0 (an owner).</summary>
        /// <param name="op">"shares_memory" or "may_share_memory".</param>
        /// <param name="p">The case params: <c>views</c> (two step recipes) and an optional <c>max_work</c>.</param>
        /// <param name="ops">Operand 0 is the owner both views are cut from.</param>
        /// <returns>The answer as a 0-d bool array.</returns>
        private static NDArray MemoryPredicate(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var a = ApplySteps(ops[0], p["views"][0]);
            var b = ApplySteps(ops[0], p["views"][1]);
            try
            {
                bool hasWork = p.TryGetValue("max_work", out var mw) && mw.ValueKind != JsonValueKind.Null;
                bool r = op == "shares_memory"
                    ? (hasWork ? np.shares_memory(a, b, mw.GetInt64()) : np.shares_memory(a, b))
                    : (hasWork ? np.may_share_memory(a, b, mw.GetInt64()) : np.may_share_memory(a, b));
                return NDArray.Scalar(r);
            }
            finally
            {
                // The views are this adapter's own (an empty slice may even own a fresh buffer); the operand is the
                // caller's, so a recipe without steps (the operand itself) is left alone.
                ReleaseView(a, ops[0]);
                ReleaseView(b, ops[0]);
            }
        }

        /// <summary>Dispose a view this adapter made, unless it is the operand itself (an empty recipe).</summary>
        /// <param name="view">The view.</param>
        /// <param name="operand">The operand the recipe started from.</param>
        private static void ReleaseView(NDArray view, NDArray operand)
        {
            if (!ReferenceEquals(view, operand))
                view.Dispose();
        }

        /// <summary>
        ///     A view recipe (the generator's <c>apply_steps</c>): <c>["s", "1:5, ::2"]</c> indexes with a slice string,
        ///     <c>["T"]</c> transposes, <c>["r", [dims]]</c> reshapes, <c>["sw", a, b]</c> swaps two axes, <c>["rev"]</c>
        ///     reverses axis 0.
        /// </summary>
        /// <param name="a">The array the recipe starts from.</param>
        /// <param name="steps">The recipe (a JSON array of steps).</param>
        /// <returns>The resulting array (a view wherever NumSharp makes one): <paramref name="a"/> itself for an empty
        ///     recipe, else a new array the caller disposes (<see cref="ReleaseView"/>).</returns>
        /// <exception cref="NotSupportedException">A step name the generator does not define.</exception>
        /// <remarks>Each intermediate view is disposed once the next step has its own reference to the storage, so a
        ///     multi-step recipe leaves only its result behind.</remarks>
        private static NDArray ApplySteps(NDArray a, JsonElement steps)
        {
            var current = a;
            foreach (var st in steps.EnumerateArray())
            {
                string kind = st[0].GetString();
                var next = kind switch
                {
                    "s" => current[st[1].GetString()],
                    "T" => current.T,
                    "r" => current.reshape(st[1].EnumerateArray().Select(x => x.GetInt64()).ToArray()),
                    "sw" => np.swapaxes(current, st[1].GetInt32(), st[2].GetInt32()),
                    "rev" => current["::-1"],
                    _ => throw new NotSupportedException($"view step '{kind}'"),
                };
                if (!ReferenceEquals(current, a) && !ReferenceEquals(current, next))
                    current.Dispose();
                current = next;
            }
            return current;
        }

        /// <summary>np.bmat over a nested list of operands (by index) or a string with name dictionaries.</summary>
        /// <param name="p">The case params: <c>layout</c> (nested indices) or <c>string</c> + <c>ldict</c> / <c>gdict</c>.</param>
        /// <param name="ops">The block operands.</param>
        /// <returns>NumSharp's 2-D result.</returns>
        private static NDArray Bmat(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            if (p.TryGetValue("string", out var s))
            {
                Dictionary<string, NDArray> Dict(string key)
                    => p[key].EnumerateObject().ToDictionary(kv => kv.Name, kv => ops[kv.Value.GetInt32()]);
                return np.bmat(s.GetString(), Dict("ldict"), Dict("gdict"));
            }
            var layout = p["layout"];
            if (layout.GetArrayLength() > 0 && layout[0].ValueKind == JsonValueKind.Array)
                return np.bmat(layout.EnumerateArray()
                    .Select(row => row.EnumerateArray().Select(i => ops[i.GetInt32()]).ToArray()).ToArray());
            return np.bmat(layout.EnumerateArray().Select(i => ops[i.GetInt32()]).ToArray());
        }

        // ---- the function menus -----------------------------------------------------------------------------------

        /// <summary>
        ///     The generator's <c>APPLY_FUNCS</c> menu as NumSharp computations on a 1-D slice. Each entry is the C# spelling
        ///     of the same NumPy call; the values are gated elsewhere, so a case here gates apply_along_axis' iteration,
        ///     result shape and dtype.
        /// </summary>
        /// <param name="name">The menu name.</param>
        /// <returns>The slice function.</returns>
        /// <exception cref="NotSupportedException">A name the generator does not define.</exception>
        private static Func<NDArray, NDArray> AlongAxisFunc(string name) => name switch
        {
            "sum" => v => np.sum(v),
            "mean" => v => np.mean(v),
            "max" => v => np.max(v),
            "cumsum" => v => np.cumsum(v),
            "sort" => v => np.sort(v),
            "first" => v => v[0],
            "outer" => v => np.outer(v, v),
            "diff" => v => np.diff(v),
            "argmax" => v => NDArray.Scalar(np.argmax(v)),
            "double" => v => v * 2,
            _ => throw new NotSupportedException($"apply_along_axis function '{name}'"),
        };

        /// <summary>The generator's <c>OVER_FUNCS</c> menu as NumSharp <c>(array, axis)</c> computations.</summary>
        /// <param name="name">The menu name.</param>
        /// <returns>The function apply_over_axes calls once per axis.</returns>
        /// <exception cref="NotSupportedException">A name the generator does not define.</exception>
        private static Func<NDArray, int, NDArray> OverAxesFunc(string name) => name switch
        {
            "sum" => (x, axis) => np.sum(x, axis),
            "mean" => (x, axis) => np.mean(x, axis),
            "max" => (x, axis) => np.max(x, axis),
            "min" => (x, axis) => np.min(x, axis),
            "prod" => (x, axis) => np.prod(x, axis),
            "cumsum" => (x, axis) => np.cumsum(x, axis),
            _ => throw new NotSupportedException($"apply_over_axes function '{name}'"),
        };

        // ---- object attributes ------------------------------------------------------------------------------------

        /// <summary>One finfo attribute: the float ones as a 0-d float64 (NumSharp's finfo carries double), the int ones
        /// as a 0-d int64.</summary>
        private static NDArray FinfoAttribute(string attr, finfo f) => attr switch
        {
            "eps" => NDArray.Scalar(f.eps),
            "epsneg" => NDArray.Scalar(f.epsneg),
            "max" => NDArray.Scalar(f.max),
            "min" => NDArray.Scalar(f.min),
            "tiny" => NDArray.Scalar(f.tiny),
            "smallest_normal" => NDArray.Scalar(f.smallest_normal),
            "smallest_subnormal" => NDArray.Scalar(f.smallest_subnormal),
            "resolution" => NDArray.Scalar(f.resolution),
            "bits" => NDArray.Scalar((long)f.bits),
            "precision" => NDArray.Scalar((long)f.precision),
            "maxexp" => NDArray.Scalar((long)f.maxexp),
            "minexp" => NDArray.Scalar((long)f.minexp),
            _ => throw new NotSupportedException($"finfo attribute '{attr}'"),
        };

        /// <summary>
        ///     One iinfo attribute as a 0-d array. <c>max</c> of an unsigned dtype reads <see cref="iinfo.maxUnsigned"/>:
        ///     NumSharp's <see cref="iinfo.max"/> is a <c>long</c> by design and documents maxUnsigned as the true value.
        /// </summary>
        private static NDArray IinfoAttribute(string attr, iinfo info) => attr switch
        {
            "bits" => NDArray.Scalar((long)info.bits),
            "min" => NDArray.Scalar(info.min),
            "max" => info.kind == 'u' ? NDArray.Scalar(info.maxUnsigned) : NDArray.Scalar(info.max),
            _ => throw new NotSupportedException($"iinfo attribute '{attr}'"),
        };

        /// <summary>One ndarray.flags attribute: a 0-d bool, or a 0-d int64 for <c>num</c>.</summary>
        private static NDArray FlagAttribute(string attr, NDArrayFlags f) => attr switch
        {
            "aligned" => NDArray.Scalar(f.aligned),
            "behaved" => NDArray.Scalar(f.behaved),
            "c_contiguous" => NDArray.Scalar(f.c_contiguous),
            "carray" => NDArray.Scalar(f.carray),
            "contiguous" => NDArray.Scalar(f.contiguous),
            "f_contiguous" => NDArray.Scalar(f.f_contiguous),
            "farray" => NDArray.Scalar(f.farray),
            "fnc" => NDArray.Scalar(f.fnc),
            "forc" => NDArray.Scalar(f.forc),
            "fortran" => NDArray.Scalar(f.fortran),
            "num" => NDArray.Scalar((long)f.num),
            "owndata" => NDArray.Scalar(f.owndata),
            "writeable" => NDArray.Scalar(f.writeable),
            "writebackifcopy" => NDArray.Scalar(f.writebackifcopy),
            _ => throw new NotSupportedException($"flags attribute '{attr}'"),
        };

        /// <summary>One dtype attribute as the text the generator records: a str attribute raw, anything else as NumPy's
        /// <c>repr</c>.</summary>
        private static string DTypeAttribute(string attr, DType d) => attr switch
        {
            "alignment" => PyRepr(d.alignment),
            "base" => d.@base.ToString(true),
            "byteorder" => d.byteorder.ToString(),
            "char" => d.@char.ToString(),
            "descr" => PyRepr(d.descr.Select(t => (object)(t.name, t.typestr)).ToList()),
            "fields" => PyRepr(d.fields),
            "flags" => PyRepr(d.flags),
            "hasobject" => PyRepr(d.hasobject),
            "isbuiltin" => PyRepr(d.isbuiltin),
            "isnative" => PyRepr(d.isnative),
            "itemsize" => PyRepr(d.itemsize),
            "kind" => d.kind.ToString(),
            "metadata" => PyRepr(d.metadata),
            "name" => d.name,
            "names" => PyRepr(d.names),
            "ndim" => PyRepr(d.ndim),
            "num" => PyRepr(d.num),
            "shape" => PyTuple(d.shape.Select(x => (object)x)),
            "str" => d.str,
            "subdtype" => PyRepr(d.subdtype),
            "newbyteorder" => d.newbyteorder().ToString(true),
            "type" => ScalarClassRepr(d.type),
            _ => throw new NotSupportedException($"dtype attribute '{attr}'"),
        };

        // ---- Python repr ------------------------------------------------------------------------------------------

        /// <summary>
        ///     Python's <c>repr</c> of a value NumSharp returns where NumPy returns a Python object: a nested list (tolist),
        ///     a number, a bool, a string, None, a tuple. Floats follow CPython's <c>float_repr</c> (the shortest
        ///     round-trip digits, fixed notation for a decimal exponent in [-4, 16), otherwise <c>1e+16</c> style; ".0"
        ///     on an integral fixed value); complex numbers CPython's <c>complex_repr</c>.
        /// </summary>
        /// <param name="v">The value: an <see cref="IList"/> (any C# array or list) is a Python list, an
        ///     <see cref="ITuple"/> (a value tuple) a Python tuple.</param>
        /// <returns>The repr text.</returns>
        /// <remarks>Half and float values are widened exactly to double first, as NumPy's tolist hands back a Python
        /// float of the same value.</remarks>
        private static string PyRepr(object v)
        {
            switch (v)
            {
                case null: return "None";
                case bool b: return b ? "True" : "False";
                case string s: return "'" + s.Replace("\\", "\\\\").Replace("'", "\\'") + "'";
                case char c: return "'" + c + "'";
                case sbyte or byte or short or ushort or int or uint or long or ulong:
                    return Convert.ToString(v, CultureInfo.InvariantCulture);
                case BigInteger bi: return bi.ToString(CultureInfo.InvariantCulture);
                case Half h: return PyFloat((double)h, true);
                case float f: return PyFloat(f, true);
                case double d: return PyFloat(d, true);
                case decimal m: return m.ToString(CultureInfo.InvariantCulture);
                case Complex z: return PyComplex(z);
                case IList list:
                {
                    var sb = new StringBuilder("[");
                    for (int i = 0; i < list.Count; i++)
                    {
                        if (i > 0) sb.Append(", ");
                        sb.Append(PyRepr(list[i]));
                    }
                    return sb.Append(']').ToString();
                }
                case ITuple t:
                    return PyTuple(Enumerable.Range(0, t.Length).Select(i => t[i]));
                default:
                    return Convert.ToString(v, CultureInfo.InvariantCulture);
            }
        }

        /// <summary>Python's repr of a tuple: <c>()</c>, <c>(x,)</c> or <c>(x, y)</c>.</summary>
        private static string PyTuple(IEnumerable<object> items)
        {
            var parts = items.Select(x => PyRepr(x)).ToList();
            return parts.Count == 1 ? "(" + parts[0] + ",)" : "(" + string.Join(", ", parts) + ")";
        }

        /// <summary>
        ///     CPython's float formatting in mode 'r' (repr): shortest round-trip digits; scientific when the decimal
        ///     exponent is below -4 or above 15 (<c>1e-05</c>, <c>1e+16</c>), fixed otherwise.
        /// </summary>
        /// <param name="x">The value.</param>
        /// <param name="addDot0">Append ".0" to an integral fixed value (float repr does; complex repr does not).</param>
        /// <returns>The text.</returns>
        private static string PyFloat(double x, bool addDot0)
        {
            if (double.IsNaN(x)) return "nan";
            if (double.IsPositiveInfinity(x)) return "inf";
            if (double.IsNegativeInfinity(x)) return "-inf";
            bool neg = x < 0 || (x == 0 && double.IsNegative(x));
            if (x == 0) return (neg ? "-0" : "0") + (addDot0 ? ".0" : "");
            // "E16" of the shortest round-trip string: .NET's "R" gives the shortest digits; re-derive them with an
            // explicit exponent so the decimal point position (decpt) is exact.
            string r = Math.Abs(x).ToString("R", CultureInfo.InvariantCulture);
            int e = 0;
            int ei = r.IndexOfAny(new[] { 'E', 'e' });
            string mant = r;
            if (ei >= 0)
            {
                e = int.Parse(r.Substring(ei + 1), CultureInfo.InvariantCulture);
                mant = r.Substring(0, ei);
            }
            int dot = mant.IndexOf('.');
            string digits = dot >= 0 ? mant.Remove(dot, 1) : mant;
            int decpt = (dot >= 0 ? dot : mant.Length) + e;
            int lead = 0;
            while (lead < digits.Length - 1 && digits[lead] == '0') { lead++; decpt--; }
            digits = digits.Substring(lead).TrimEnd('0');
            if (digits.Length == 0) digits = "0";
            string body;
            if (decpt <= -4 || decpt > 16)
            {
                int exp = decpt - 1;
                body = digits[0] + (digits.Length > 1 ? "." + digits.Substring(1) : "") +
                       "e" + (exp < 0 ? "-" : "+") + Math.Abs(exp).ToString("00", CultureInfo.InvariantCulture);
            }
            else if (decpt <= 0)
                body = "0." + new string('0', -decpt) + digits;
            else if (decpt >= digits.Length)
                body = digits + new string('0', decpt - digits.Length) + (addDot0 ? ".0" : "");
            else
                body = digits.Substring(0, decpt) + "." + digits.Substring(decpt);
            return (neg ? "-" : "") + body;
        }

        /// <summary>CPython's complex repr: <c>{imag}j</c> for a +0 real part, else <c>({real}{±imag}j)</c>.</summary>
        private static string PyComplex(Complex z)
        {
            string im = PyFloat(z.Imaginary, false);
            if (z.Real == 0 && !double.IsNegative(z.Real))
                return im + "j";
            string sign = im.StartsWith("-", StringComparison.Ordinal) ? "" : "+";
            return "(" + PyFloat(z.Real, false) + sign + im + "j)";
        }

        // ---- params -----------------------------------------------------------------------------------------------

        /// <summary>An int param.</summary>
        private static int SurfInt(IReadOnlyDictionary<string, JsonElement> p, string key) => p[key].GetInt32();

        /// <summary>An optional int param (absent or null → null).</summary>
        private static int? SurfNullableInt(IReadOnlyDictionary<string, JsonElement> p, string key)
            => p.TryGetValue(key, out var v) && v.ValueKind != JsonValueKind.Null ? v.GetInt32() : null;

        /// <summary>A string param.</summary>
        private static string SurfStr(IReadOnlyDictionary<string, JsonElement> p, string key) => p[key].GetString();

        /// <summary>An optional string param (absent or null → null).</summary>
        private static string SurfNullableStr(IReadOnlyDictionary<string, JsonElement> p, string key)
            => p.TryGetValue(key, out var v) && v.ValueKind != JsonValueKind.Null ? v.GetString() : null;

        /// <summary>An optional dtype param (absent or null → null, NumPy's <c>dtype=None</c>).</summary>
        private static DType SurfDType(IReadOnlyDictionary<string, JsonElement> p, string key)
        {
            string s = SurfNullableStr(p, key);
            return s is null ? null : np.dtype(s);
        }

        /// <summary>A Python int param of any size.</summary>
        private static BigInteger PyInt(JsonElement v) => BigInteger.Parse(v.GetRawText(), CultureInfo.InvariantCulture);

        /// <summary>
        ///     A Python float param: a JSON number (an integer reads as its value), or the generator's string form of a
        ///     non-finite float (<c>"nan"</c>, <c>"inf"</c>, <c>"-inf"</c> — strict JSON has no such numbers).
        /// </summary>
        /// <param name="v">The param element.</param>
        /// <returns>The value.</returns>
        /// <exception cref="FormatException">A string that is not one of the three non-finite forms (a corpus bug).</exception>
        private static double PyFloat(JsonElement v) => v.ValueKind == JsonValueKind.String
            ? v.GetString() switch
            {
                "nan" => double.NaN,
                "inf" => double.PositiveInfinity,
                "-inf" => double.NegativeInfinity,
                var s => throw new FormatException($"float param '{s}'"),
            }
            : v.GetDouble();

        /// <summary>A Python number param as its C# literal: an integral JSON number as long, any other as double.</summary>
        private static object PyNumber(JsonElement v)
            => v.TryGetInt64(out long l) ? l : v.GetDouble();

        /// <summary>
        ///     A 0-d float operand's value as the <c>double</c> NumSharp's float formatters take (their only overload).
        ///     The widening is exact, so a float16 / float32 value reaches the formatter unchanged; NumPy formats such a
        ///     value at its own precision, which a double-only API cannot know.
        /// </summary>
        /// <param name="a">The 0-d operand.</param>
        /// <returns>Its value as a double.</returns>
        /// <remarks><see cref="Half"/> does not implement <see cref="IConvertible"/>, hence the explicit cases.</remarks>
        private static double ScalarAsDouble(NDArray a) => a.GetAtIndex(0) switch
        {
            Half h => (double)h,
            float f => f,
            double d => d,
            var o => Convert.ToDouble(o, CultureInfo.InvariantCulture),
        };
    }
}
