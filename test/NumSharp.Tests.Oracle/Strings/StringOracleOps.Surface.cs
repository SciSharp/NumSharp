using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for section M (<c>gen_surface</c>): the rest of NumPy's surface over text operands —
    ///     <c>np.linalg</c>, the products and sliding windows, random seeding and integer bounds, the histogram family,
    ///     quantile <c>q</c>, reductions and arg-reductions with <c>out=</c>, polynomials, <c>np.emath</c>, index
    ///     conversions, dtype predicates, <c>np.ediff1d</c>, text <c>where=</c> masks, 0-d conversions to C# scalars and
    ///     <c>fill</c> with a sequence. Each op replays the C# call a ported NumPy line makes, with the generator's
    ///     arguments.
    /// </summary>
    internal static partial class StringOracleOps
    {
        /// <summary>
        ///     Replays one section-M op, or reports that <paramref name="op"/> is not one of them.
        /// </summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="result">The call's result (an array, a tuple of arrays, a bool, a dtype, or the mutated operand).</param>
        /// <returns>True when <paramref name="op"/> is a section-M op.</returns>
        /// <exception cref="InvalidOperationException">A name the op does not know (a generator / table mismatch).</exception>
        private static bool TryInvokeSurface(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            NDArray a = ops.Length > 0 ? ops[0] : null;
            NDArray b = ops.Length > 1 ? ops[1] : null;
            switch (op)
            {
                case "linalg1":
                    result = Linalg1(Str(p, "name"), a);
                    return true;
                case "linalg2":
                    result = Linalg2(Str(p, "name"), a, b);
                    return true;
                case "product":
                    result = Product(Str(p, "name"), a, b);
                    return true;
                case "sliding_mode":
                    result = Str(p, "name") == "convolve" ? np.convolve(a, b, Str(p, "mode")) : np.correlate(a, b, Str(p, "mode"));
                    return true;
                case "dot_out":
                    // The out operand is compared as the mutated operand (NumPy returns the out itself).
                    result = np.dot(a, b, @out: ops[2]);
                    return true;

                // ---- random ----
                case "seed_seq":
                {
                    // np.random.SeedSequence(text).pool + .generate_state(4), as one uint32 array.
                    var ss = np.random.SeedSequence((object)a);
                    result = np.concatenate(new[] { ss.pool, ss.generate_state(4) });
                    return true;
                }
                case "seed_spawn":
                {
                    // Both arms cast to object: an object[] arm and an NDArray arm would type the ?: as NDArray through
                    // NumSharp's implicit Array -> NDArray conversion, which cannot hold an NDArray element.
                    object key = p["list"].GetBoolean() ? (object)new object[] { a } : (object)a;
                    result = np.random.SeedSequence(5, spawn_key: key).generate_state(4);
                    return true;
                }
                case "default_rng":
                    result = np.random.default_rng(a).random(new Shape(3));
                    return true;
                case "philox_counter":
                {
                    var state = (Philox.State)np.random.Philox(1, counter: a).state;
                    result = np.array(state.counter);
                    return true;
                }
                case "randint":
                {
                    DType dt = np.dtype(Str(p, "dtype"));
                    result = Str(p, "api") switch
                    {
                        "legacy" => np.random.RandomState(1).randint(a, b, dtype: dt),
                        "legacy_size0" => np.random.RandomState(1).randint(a, b, size: new Shape(0), dtype: dt),
                        "gen" => np.random.default_rng(1).integers(a, b, dtype: dt),
                        "gen_closed" => np.random.default_rng(1).integers(a, b, dtype: dt, endpoint: true),
                        var api => throw new InvalidOperationException($"the string oracle has no randint api '{api}'"),
                    };
                    return true;
                }
                case "randint_onearg":
                    result = np.random.RandomState(1).randint(a, dtype: np.dtype(Str(p, "dtype")));
                    return true;
                case "choice_p":
                    result = Str(p, "api") == "gen"
                        ? np.random.default_rng(1).choice(3L, p: a)
                        : np.random.RandomState(1).choice(3, p: a);
                    return true;

                // ---- the histogram family ----
                case "hist":
                    result = Histogram(p, ops);
                    return true;
                case "hdd_bins":
                    result = HistogramddBins(p, ops);
                    return true;

                // ---- statistics, reductions, arg-reductions ----
                case "quantile_q":
                    result = Str(p, "name") switch
                    {
                        "percentile" => np.percentile(a, b),
                        "nanpercentile" => np.nanpercentile(a, b),
                        "quantile" => np.quantile(a, b),
                        "nanquantile" => np.nanquantile(a, b),
                        var n => throw new InvalidOperationException($"the string oracle has no quantile '{n}'"),
                    };
                    return true;
                case "reduce_out":
                    result = ReduceOut(Str(p, "name"), a, b);
                    return true;
                case "argreduce_out":
                {
                    int? axis = NullableInt(p, "axis");
                    result = Str(p, "name") == "nanargmax" ? np.nanargmax(a, axis, @out: b) : np.nanargmin(a, axis, @out: b);
                    return true;
                }

                // ---- polynomials, emath, index conversions ----
                case "poly":
                    result = Str(p, "name") switch
                    {
                        "poly" => np.poly(a),
                        "roots" => np.roots(a),
                        "polymul" => np.polymul(a, b),
                        "polyadd" => np.polyadd(a, b),
                        "polyval" => np.polyval(a, b),
                        "polyder" => np.polyder(a),
                        "polyint" => np.polyint(a),
                        "polydiv" => (object)np.polydiv(a, b),
                        var n => throw new InvalidOperationException($"the string oracle has no polynomial call '{n}'"),
                    };
                    return true;
                case "emath":
                    result = Str(p, "name") switch
                    {
                        "sqrt" => np.emath.sqrt(a),
                        "log" => np.emath.log(a),
                        "log2" => np.emath.log2(a),
                        "log10" => np.emath.log10(a),
                        "arccos" => np.emath.arccos(a),
                        "arcsin" => np.emath.arcsin(a),
                        "arctanh" => np.emath.arctanh(a),
                        "logn" => np.emath.logn(a, b),
                        "power" => np.emath.power(a, b),
                        var n => throw new InvalidOperationException($"the string oracle has no emath call '{n}'"),
                    };
                    return true;
                case "unravel_ix":
                    result = np.unravel_index(a, Ints(p["shape"]));
                    return true;
                case "ravel_mi":
                    result = np.ravel_multi_index(ops, Ints(p["dims"]));
                    return true;

                // ---- dtype predicates ----
                case "common_type":
                    result = np.common_type(ops);
                    return true;
                case "min_scalar_type":
                    result = np.min_scalar_type(a);
                    return true;
                case "can_cast_arr":
                    result = np.can_cast(a, np.dtype(Str(p, "to")), Str(p, "casting"));
                    return true;

                // ---- ediff1d, where= masks, scalar conversions, fill ----
                case "ediff1d":
                {
                    NDArray end = NullableInt(p, "end") is int e ? ops[e] : null;
                    NDArray begin = NullableInt(p, "begin") is int g ? ops[g] : null;
                    result = np.ediff1d(a, to_end: end, to_begin: begin);
                    return true;
                }
                case "where_mask":
                    result = WhereMask(Str(p, "name"), a, b, ops[2]);
                    return true;
                case "to_scalar":
                    // NumPy's complex(a) / float(a) / int(a) of a 0-d array: the C# explicit conversions.
                    result = Str(p, "to") switch
                    {
                        "complex" => NDArray.Scalar((Complex)a),
                        "float" => NDArray.Scalar((double)a),
                        "int" => NDArray.Scalar((long)a),
                        var t => throw new InvalidOperationException($"the string oracle has no conversion to '{t}'"),
                    };
                    return true;
                case "fill_seq":
                    a.fill(b);
                    result = a;
                    return true;
            }
            result = null;
            return false;
        }

        /// <summary>One <c>np.linalg</c> call over one operand, by the generator's name.</summary>
        /// <param name="name">The generator's name (the function and its non-default arguments).</param>
        /// <param name="a">The operand.</param>
        /// <returns>The call's result (an array, a tuple result, or a 0-d count).</returns>
        /// <exception cref="InvalidOperationException">An unknown name.</exception>
        private static object Linalg1(string name, NDArray a) => name switch
        {
            // Every arm is cast to object: the tuple-result arms (slogdet, eig, svd, qr) and the NDArray arms have no
            // common type, and an NDArray-typed switch would route the tuples through NumSharp's implicit conversions.
            "cholesky" => (object)np.linalg.cholesky(a),
            "cholesky_up" => (object)np.linalg.cholesky(a, upper: true),
            "det" => (object)np.linalg.det(a),
            "slogdet" => (object)np.linalg.slogdet(a),
            "inv" => (object)np.linalg.inv(a),
            "pinv" => (object)np.linalg.pinv(a),
            "pinv_h" => (object)np.linalg.pinv(a, hermitian: true),
            "eig" => (object)np.linalg.eig(a),
            "eigvals" => (object)np.linalg.eigvals(a),
            "eigh" => (object)np.linalg.eigh(a),
            "eigh_U" => (object)np.linalg.eigh(a, 'U'),
            "eigvalsh" => (object)np.linalg.eigvalsh(a),
            "eigvalsh_U" => (object)np.linalg.eigvalsh(a, 'U'),
            "svd" => (object)np.linalg.svd(a),
            "svd_nuv" => (object)np.linalg.svd(a, compute_uv: false),
            "svd_s" => (object)np.linalg.svd(a, full_matrices: false),
            "svd_h" => (object)np.linalg.svd(a, hermitian: true),
            "svdvals" => (object)np.linalg.svdvals(a),
            "matrix_rank" => (object)np.linalg.matrix_rank(a),
            "matrix_rank_tol" => (object)np.linalg.matrix_rank(a, tol: 0.5),
            "matrix_rank_h" => (object)np.linalg.matrix_rank(a, hermitian: true),
            "cond" => (object)np.linalg.cond(a),
            "cond_fro" => (object)np.linalg.cond(a, "fro"),
            "cond_1" => (object)np.linalg.cond(a, 1),
            "cond_2" => (object)np.linalg.cond(a, 2),
            "qr" => (object)np.linalg.qr(a),
            "qr_r" => (object)np.linalg.qr(a, mode: "r"),
            "qr_bad" => (object)np.linalg.qr(a, mode: "x"),
            "norm" => (object)np.linalg.norm(a),
            "norm_fro" => (object)np.linalg.norm(a, "fro"),
            "norm_nuc" => (object)np.linalg.norm(a, "nuc"),
            "norm_2" => (object)np.linalg.norm(a, 2),
            "norm_inf" => (object)np.linalg.norm(a, double.PositiveInfinity),
            "norm_1" => (object)np.linalg.norm(a, 1),
            "norm_ax" => (object)np.linalg.norm(a, axis: 0),
            "norm_bad" => (object)np.linalg.norm(a, "x"),
            "matrix_norm" => (object)np.linalg.matrix_norm(a),
            "matrix_norm_2" => (object)np.linalg.matrix_norm(a, ord: 2),
            "vector_norm" => (object)np.linalg.vector_norm(a),
            "matrix_power0" => (object)np.linalg.matrix_power(a, 0),
            "matrix_power1" => (object)np.linalg.matrix_power(a, 1),
            "matrix_power2" => (object)np.linalg.matrix_power(a, 2),
            "matrix_power3" => (object)np.linalg.matrix_power(a, 3),
            "matrix_powerm1" => (object)np.linalg.matrix_power(a, -1),
            "matrix_powerm2" => (object)np.linalg.matrix_power(a, -2),
            "tensorinv1" => (object)np.linalg.tensorinv(a, 1),
            "tensorinv2" => (object)np.linalg.tensorinv(a, 2),
            "tensorinv0" => (object)np.linalg.tensorinv(a, 0),
            "trace" => (object)np.linalg.trace(a),
            "diagonal" => (object)np.linalg.diagonal(a),
            "matrix_transpose" => (object)np.linalg.matrix_transpose(a),
            _ => throw new InvalidOperationException($"the string oracle has no linalg call '{name}'"),
        };

        /// <summary>One <c>np.linalg</c> call over two operands, by the generator's name.</summary>
        /// <param name="name">The function name.</param>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The call's result.</returns>
        /// <exception cref="InvalidOperationException">An unknown name.</exception>
        private static object Linalg2(string name, NDArray a, NDArray b) => name switch
        {
            "solve" => (object)np.linalg.solve(a, b),
            "lstsq" => (object)np.linalg.lstsq(a, b),
            "tensorsolve" => (object)np.linalg.tensorsolve(a, b),
            "multi_dot" => (object)np.linalg.multi_dot(new[] { a, b }),
            "matmul" => (object)np.linalg.matmul(a, b),
            "outer" => (object)np.linalg.outer(a, b),
            "tensordot" => (object)np.linalg.tensordot(a, b),
            "vecdot" => (object)np.linalg.vecdot(a, b),
            "cross" => (object)np.linalg.cross(a, b),
            _ => throw new InvalidOperationException($"the string oracle has no linalg call '{name}'"),
        };

        /// <summary>One top-level product or sliding-window call, by NumPy name.</summary>
        /// <param name="name">The function name.</param>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The call's result.</returns>
        /// <exception cref="InvalidOperationException">An unknown name.</exception>
        private static NDArray Product(string name, NDArray a, NDArray b) => name switch
        {
            "dot" => np.dot(a, b),
            "inner" => np.inner(a, b),
            "vdot" => np.vdot(a, b),
            "outer" => np.outer(a, b),
            "tensordot" => np.tensordot(a, b, 1),
            "kron" => np.kron(a, b),
            "matmul" => np.matmul(a, b),
            "cross" => np.cross(a, b),
            "convolve" => np.convolve(a, b),
            "correlate" => np.correlate(a, b),
            _ => throw new InvalidOperationException($"the string oracle has no product '{name}'"),
        };

        /// <summary>
        ///     One histogram-family call: <c>histogram_bin_edges</c> / <c>histogram</c> over <c>a</c>, or
        ///     <c>histogram2d</c> / <c>histogramdd</c> over <c>(a, y)</c>, with an int / estimator-name / array bins and an
        ///     optional range, density and weights.
        /// </summary>
        /// <param name="p">The case parameters (<c>fn</c>, <c>bins</c> or <c>bins_op</c>, <c>range</c>, <c>density</c>,
        ///     <c>weights_op</c>).</param>
        /// <param name="ops">The operands: the data, then (2d / dd) the y column, then the array bins and weights.</param>
        /// <returns>The edges array, or the tuple of arrays (histogramdd's edges flattened after H).</returns>
        /// <exception cref="InvalidOperationException">An unknown function or bins kind.</exception>
        private static object Histogram(Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            string fn = Str(p, "fn");
            NDArray a = ops[0];
            NDArray w = NullableInt(p, "weights_op") is int wi ? ops[wi] : null;
            (double, double)? range = null;
            if (p["range"].ValueKind == JsonValueKind.Array)
            {
                var r = p["range"].EnumerateArray().Select(JsonFloat).ToArray();
                range = (r[0], r[1]);
            }
            NDArray binsArray = NullableInt(p, "bins_op") is int bi ? ops[bi] : null;
            JsonElement bins = binsArray is null ? p["bins"] : default;
            bool density = p.TryGetValue("density", out var d) && d.GetBoolean();
            switch (fn)
            {
                case "histogram_bin_edges":
                    if (binsArray is not null)
                        return np.histogram_bin_edges(a, binsArray, range, w);
                    return bins.ValueKind == JsonValueKind.Number
                        ? np.histogram_bin_edges(a, bins.GetInt32(), range, w)
                        : np.histogram_bin_edges(a, bins.GetString(), range, w);
                case "histogram":
                {
                    np.HistogramResult r = binsArray is not null
                        ? np.histogram(a, binsArray, range, density, w)
                        : bins.ValueKind == JsonValueKind.Number
                            ? np.histogram(a, bins.GetInt32(), range, density, w)
                            : np.histogram(a, bins.GetString(), range, density, w);
                    return new[] { r.hist, r.bin_edges };
                }
                case "histogram2d":
                {
                    NDArray y = ops[1];
                    (double, double)?[] rr = range is null ? null : new (double, double)?[] { range, range };
                    var r = binsArray is not null ? np.histogram2d(a, y, binsArray, rr) : np.histogram2d(a, y, bins.GetInt32(), rr);
                    return new[] { r.H, r.xedges, r.yedges };
                }
                case "histogramdd":
                {
                    NDArray y = ops[1];
                    var sample = new[] { a, y };
                    (double, double)?[] rr = range is null ? null : new (double, double)?[] { range, range };
                    // NumPy's [bins, bins] for 1-D edges; a 0-d array spec is D copies of it (its `D*[bins]`).
                    np.HistogramddResult r = binsArray is null ? np.histogramdd(sample, bins.GetInt32(), rr)
                        : binsArray.ndim >= 1 ? np.histogramdd(sample, new[] { binsArray, binsArray }, rr)
                        : np.histogramdd(sample, new object[] { binsArray, binsArray }, rr);
                    return new[] { r.H }.Concat(r.edges).ToArray();
                }
            }
            throw new InvalidOperationException($"the string oracle has no histogram function '{fn}'");
        }

        /// <summary>
        ///     <c>np.histogramdd([x, y], bins)</c> with an int or list bin specification (<c>fn</c> "dd"), or
        ///     <c>np.histogram2d(x, y, bins)</c> with an array one (<c>fn</c> "2d").
        /// </summary>
        /// <param name="p">The case parameters: <c>fn</c> and the encoded <c>bins</c> (see <see cref="BinsSpec"/>).</param>
        /// <param name="ops">x, y, then the arrays the specification refers to.</param>
        /// <returns>H followed by the edges.</returns>
        private static NDArray[] HistogramddBins(Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            var sample = new[] { ops[0], ops[1] };
            var spec = p["bins"];
            if (Str(p, "fn") == "2d")
            {
                var r2 = np.histogram2d(ops[0], ops[1], ops[spec.GetProperty("op").GetInt32()]);
                return new[] { r2.H, r2.xedges, r2.yedges };
            }
            np.HistogramddResult r = BinsSpec(spec, ops) switch
            {
                int n => np.histogramdd(sample, n),
                int[] ns => np.histogramdd(sample, ns),
                NDArray[] edges => np.histogramdd(sample, edges),
                object[] items => np.histogramdd(sample, items),
                var other => throw new InvalidOperationException($"no histogramdd bins for {other}"),
            };
            return new[] { r.H }.Concat(r.edges).ToArray();
        }

        /// <summary>
        ///     A decoded histogramdd bin specification as the C# argument a ported line passes: an int; a 0-d array as its
        ///     two copies (<c>object[]</c>); a list of ints as <c>int[]</c>, of 1-D arrays as <c>NDArray[]</c>, anything
        ///     else as <c>object[]</c> of int / double / bool / null / string / <see cref="Complex"/> / NDArray.
        /// </summary>
        /// <param name="e">The encoded specification.</param>
        /// <param name="ops">The operands its <c>{"op": k}</c> entries refer to.</param>
        /// <returns>The argument.</returns>
        private static object BinsSpec(JsonElement e, NDArray[] ops)
        {
            if (e.ValueKind == JsonValueKind.Object)
            {
                if (e.TryGetProperty("op", out var k))
                {
                    var nd = ops[k.GetInt32()];
                    return new object[] { nd, nd };
                }
                return (int)e.GetProperty("int").GetInt64();
            }
            var items = e.EnumerateArray().Select(x => BinsItem(x, ops)).ToArray();
            if (items.All(x => x is int))
                return items.Cast<int>().ToArray();
            if (items.All(x => x is NDArray nd && nd.ndim == 1))
                return items.Cast<NDArray>().ToArray();
            return items;
        }

        /// <summary>One entry of a histogramdd bin list.</summary>
        /// <param name="e">The encoded entry.</param>
        /// <param name="ops">The operands its <c>{"op": k}</c> refers to.</param>
        /// <returns>The C# value.</returns>
        /// <exception cref="InvalidOperationException">An unknown entry kind.</exception>
        private static object BinsItem(JsonElement e, NDArray[] ops)
        {
            if (e.TryGetProperty("op", out var k)) return ops[k.GetInt32()];
            if (e.TryGetProperty("int", out var i)) return (int)i.GetInt64();
            if (e.TryGetProperty("float", out var f)) return f.GetDouble();
            if (e.TryGetProperty("bool", out var b)) return b.GetBoolean();
            if (e.TryGetProperty("none", out _)) return null;
            if (e.TryGetProperty("complex", out var c))
            {
                var parts = c.EnumerateArray().Select(x => x.GetDouble()).ToArray();
                return new Complex(parts[0], parts[1]);
            }
            if (e.TryGetProperty("u", out _)) return StringOracleCorpus.Decode(e);
            throw new InvalidOperationException($"unknown histogramdd bin entry {e}");
        }

        /// <summary>One reduction or scan with <c>out=</c>, by NumPy name.</summary>
        /// <param name="name">The function name.</param>
        /// <param name="a">The input.</param>
        /// <param name="o">The out array (returned and compared as the mutated operand).</param>
        /// <returns>The out array.</returns>
        /// <exception cref="InvalidOperationException">An unknown name.</exception>
        private static NDArray ReduceOut(string name, NDArray a, NDArray o) => name switch
        {
            "max" => np.max(a, @out: o),
            "min" => np.min(a, @out: o),
            "amax" => np.amax(a, @out: o),
            "amin" => np.amin(a, @out: o),
            "nanmax" => np.nanmax(a, @out: o),
            "nanmin" => np.nanmin(a, @out: o),
            "ptp" => np.ptp(a, @out: o),
            "median" => np.median(a, @out: o),
            "nanmedian" => np.nanmedian(a, @out: o),
            "cumsum" => np.cumsum(a, @out: o),
            "cumprod" => np.cumprod(a, @out: o),
            "nancumsum" => np.nancumsum(a, @out: o),
            "nancumprod" => np.nancumprod(a, @out: o),
            _ => throw new InvalidOperationException($"the string oracle has no reduction '{name}'"),
        };

        /// <summary>One call with a <c>where=</c> mask, by the generator's name.</summary>
        /// <param name="name">The function name (<c>copyto</c> writes into <paramref name="x"/>).</param>
        /// <param name="x">The first operand (copyto's destination).</param>
        /// <param name="y">The second operand (copyto's source).</param>
        /// <param name="mask">The mask.</param>
        /// <returns>The call's result, or the destination for copyto.</returns>
        /// <exception cref="InvalidOperationException">An unknown name.</exception>
        private static NDArray WhereMask(string name, NDArray x, NDArray y, NDArray mask)
        {
            switch (name)
            {
                case "add": return np.add(x, y, where: mask);
                case "subtract": return np.subtract(x, y, where: mask);
                case "sqrt": return np.sqrt(x, where: mask);
                case "less": return np.less(x, y, where: mask);
                case "logical_and": return np.logical_and(x, y, where: mask);
                case "max": return np.max(x, initial: 0, @where: mask);
                case "copyto":
                    np.copyto(x, y, where: mask);
                    return x;
            }
            throw new InvalidOperationException($"the string oracle has no where= call '{name}'");
        }

        /// <summary>A JSON list of ints as an int array.</summary>
        private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(x => x.GetInt32()).ToArray();

        /// <summary>
        ///     A float parameter: a JSON number, or the generator's string spelling of a value JSON cannot hold
        ///     (<c>"inf"</c>, <c>"-inf"</c>, <c>"nan"</c>).
        /// </summary>
        /// <param name="e">The encoded float.</param>
        /// <returns>The value.</returns>
        /// <exception cref="InvalidOperationException">A string that is none of the three spellings.</exception>
        private static double JsonFloat(JsonElement e)
        {
            if (e.ValueKind == JsonValueKind.Number)
                return e.GetDouble();
            return e.GetString() switch
            {
                "inf" => double.PositiveInfinity,
                "-inf" => double.NegativeInfinity,
                "nan" => double.NaN,
                var s => throw new InvalidOperationException($"unknown float spelling '{s}'"),
            };
        }
    }
}
