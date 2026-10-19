using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for sections <c>strout</c> / <c>strpad</c> (<c>gen_strout</c> / <c>gen_strpad</c>, strings
    ///     phase 2, P2.7): reductions, accumulations and statistics that touch text — a text input, a text <c>out=</c>, a text
    ///     <c>dtype=</c> request — through every entry point that takes <c>out=</c> or <c>dtype=</c> (<c>o_call</c>), the
    ///     numpy.matrix reductions (<c>o_matrix</c>), <c>np.fft</c> over <c>StringDType</c> (<c>o_fft</c>), and <c>np.pad</c>
    ///     with its mode keywords (<c>pad_kw</c>).
    /// </summary>
    /// <remarks>
    ///     Each op calls the NumSharp overload a ported NumPy line binds, passing only the keywords the generator recorded: an
    ///     absent axis is NumPy's <c>axis=None</c>, an absent <c>dtype</c> / <c>out</c> is None, an absent <c>keepdims</c> is
    ///     False. The cases were evaluated one fresh NumPy process each (NumPy's dispatch cache makes a reduction's answer
    ///     depend on the calls its process made before), which is the answer NumSharp ports.
    /// </remarks>
    internal static partial class StringOracleOps
    {
        /// <summary>Replays one section strout / strpad op, or reports that <paramref name="op"/> is not one of them.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case parameters (see each op's method).</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="result">The call's result (the out array itself for an <c>out=</c> call).</param>
        /// <returns>True when <paramref name="op"/> is one of this partial's ops.</returns>
        /// <exception cref="InvalidOperationException">A function name the op does not know (a generator / table mismatch).</exception>
        private static bool TryInvokeTextOut(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            switch (op)
            {
                case "o_call":
                    result = OutCall(p, ops);
                    return true;
                case "o_matrix":
                    result = OutMatrix(p, ops);
                    return true;
                case "o_fft":
                    result = OutFft(Str(p, "name"), ops[0]);
                    return true;
                case "pad_kw":
                    result = PadWithKeywords(ops[0], p);
                    return true;
            }
            result = null;
            return false;
        }

        /// <summary>
        ///     A call of section strout's <c>o_call</c>: <c>name</c>, and the optional <c>axis</c> (an int), <c>out</c> (an
        ///     operand index), <c>dtype</c> (a request), <c>keepdims</c>, <c>q</c> and <c>method</c> (percentile / quantile),
        ///     <c>offset</c> / <c>axis1</c> / <c>axis2</c> (np.trace) and <c>ddof</c> (the var / std forms).
        /// </summary>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The operands (the input first, the out at its index).</param>
        /// <returns>The call's result.</returns>
        /// <exception cref="InvalidOperationException">An unknown name, or a keyword combination no NumSharp overload takes.</exception>
        private static object OutCall(Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            NDArray a = ops[0];
            int? axis = NullableInt(p, "axis");
            NDArray @out = NullableInt(p, "out") is { } oi ? ops[oi] : null;
            DType dt = Dt(p, "dtype");
            bool keepdims = p.TryGetValue("keepdims", out var ke) && ke.GetBoolean();
            int? ddof = NullableInt(p, "ddof");
            string method = p.TryGetValue("method", out var me) ? me.GetString() : "linear";
            switch (Str(p, "name"))
            {
                case "cumsum": return np.cumsum(a, axis, dt, @out);
                case "cumprod": return np.cumprod(a, axis, dt, @out);
                case "nancumsum": return np.nancumsum(a, axis, dt, @out);
                case "nancumprod": return np.nancumprod(a, axis, dt, @out);
                case "sum": return np.sum(a, axis, keepdims, dt);
                case "prod": return np.prod(a, axis, dt, keepdims);
                case "median": return np.median(a, axis, @out, false, keepdims);
                case "nanmedian": return np.nanmedian(a, axis, @out, false, keepdims);
                // An absent initial= / where= is NumPy's no-initial / where=True.
                case "max": return np.max(a, axis, @out, keepdims, null, null);
                case "min": return np.min(a, axis, @out, keepdims, null, null);
                case "nanmax": return np.nanmax(a, axis, @out, keepdims, null, null);
                case "nanmin": return np.nanmin(a, axis, @out, keepdims, null, null);
                case "ptp": return np.ptp(a, axis, @out, keepdims);
                case "argmax": return np.argmax(a, axis, @out, keepdims);
                case "argmin": return np.argmin(a, axis, @out, keepdims);
                case "all": return np.all(a, axis, @out, keepdims);
                case "any": return np.any(a, axis, @out, keepdims);
                case "percentile": return np.percentile(a, p["q"].GetDouble(), axis, @out, false, method, keepdims);
                case "quantile": return np.quantile(a, p["q"].GetDouble(), axis, @out, false, method, keepdims);
                case "trace":
                    return np.trace(a, NullableInt(p, "offset") ?? 0, NullableInt(p, "axis1") ?? 0, NullableInt(p, "axis2") ?? 1,
                        dt, @out);
                case "mean":
                    // NumSharp's np.mean takes dtype= with an explicit axis (the generator always gives one with a request).
                    if (dt is not null)
                        return np.mean(a, axis ?? throw new InvalidOperationException("np.mean's dtype= needs an axis"), dt, keepdims);
                    return axis is { } m ? np.mean(a, m, keepdims) : np.mean(a, keepdims);
                case "var":
                    return axis is { } v ? np.var(a, v, keepdims, ddof, dt) : np.var(a, keepdims, ddof, dt);
                case "std":
                    return axis is { } s ? np.std(a, s, keepdims, ddof, dt) : np.std(a, keepdims, ddof, dt);
                case "nanmean": return np.nanmean(a, axis, keepdims, dt);
                case "nanvar": return np.nanvar(a, axis, keepdims, ddof ?? 0, dt);
                case "nanstd": return np.nanstd(a, axis, keepdims, ddof ?? 0, dt);
                case var other:
                    throw new InvalidOperationException($"the string oracle has no out=/dtype= call '{other}'");
            }
        }

        /// <summary>
        ///     A call of section strout's <c>o_matrix</c>: <c>np.matrix(a).name(axis, dtype, out=np.asmatrix(out), ddof)</c> —
        ///     the out a matrix VIEW of the out operand, so the values land in the operand's buffer as they do in NumPy's.
        /// </summary>
        /// <param name="p">The case parameters: <c>name</c>, <c>axis</c> (an int or None) and the optional <c>dtype</c>,
        ///     <c>out</c> (an operand index) and <c>ddof</c>.</param>
        /// <param name="ops">The operands (the matrix data first, the out at its index).</param>
        /// <returns>The method's result: the out matrix for an axis, the scalar for every element.</returns>
        /// <exception cref="InvalidOperationException">An unknown method name.</exception>
        private static object OutMatrix(Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            NDMatrix m = np.matrix(ops[0]);
            NDMatrix @out = NullableInt(p, "out") is { } oi ? np.asmatrix(ops[oi]) : null;
            int? axis = NullableInt(p, "axis");
            DType dt = Dt(p, "dtype");
            int ddof = NullableInt(p, "ddof") ?? 0;
            return Str(p, "name") switch
            {
                "sum" => m.sum(axis, dt, @out),
                "prod" => m.prod(axis, dt, @out),
                "mean" => m.mean(axis, dt, @out),
                "var" => m.var(axis, dt, @out, ddof),
                "std" => m.std(axis, dt, @out, ddof),
                "max" => m.max(axis, @out),
                "min" => m.min(axis, @out),
                var other => throw new InvalidOperationException($"the string oracle has no matrix reduction '{other}'"),
            };
        }

        /// <summary>A transform of <c>np.fft</c> by its NumPy name (<c>fft.fft</c>, ...), with every keyword at its default.</summary>
        /// <param name="name">The dotted NumPy name.</param>
        /// <param name="a">The input.</param>
        /// <returns>The transform's result.</returns>
        /// <exception cref="InvalidOperationException">An unknown name.</exception>
        private static NDArray OutFft(string name, NDArray a) => name switch
        {
            "fft.fft" => np.fft.fft(a),
            "fft.ifft" => np.fft.ifft(a),
            "fft.rfft" => np.fft.rfft(a),
            "fft.irfft" => np.fft.irfft(a),
            "fft.hfft" => np.fft.hfft(a),
            "fft.ihfft" => np.fft.ihfft(a),
            "fft.fft2" => np.fft.fft2(a),
            "fft.fftn" => np.fft.fftn(a),
            "fft.rfftn" => np.fft.rfftn(a),
            "fft.fftshift" => np.fft.fftshift(a),
            var other => throw new InvalidOperationException($"the string oracle has no fft transform '{other}'"),
        };

        /// <summary>
        ///     A Python value of a <c>np.pad</c> keyword in the C# spelling a ported call uses: a list or tuple of Python ints
        ///     becomes a <c>long[]</c> (<c>stat_length=(1, 2)</c> is <c>new long[] { 1, 2 }</c>), any other list or tuple an
        ///     <c>object[]</c>, a list of equally long lists an <c>object[,]</c> (np.pad's array_like keywords take those, like
        ///     NumPy takes any sequence); a scalar stays the decoded scalar.
        /// </summary>
        /// <param name="value">The decoded value (<see cref="StringOracleCorpus.Decode"/>: a JSON list is a
        ///     <see cref="List{T}"/> of objects).</param>
        /// <returns>The C# argument.</returns>
        private static object PadArgument(object value)
        {
            if (value is not List<object> list)
                return value;
            // Integers only: the C# integer array (np.pad's integral stat_length takes int[] / long[], not object[]).
            if (list.Count > 0 && list.All(x => x is long))
                return list.Select(x => (long)x).ToArray();
            // A rectangular list of lists (end_values=[[1, 2], ['a', 'b']]) is a 2-D table.
            if (list.Count > 0 && list.All(x => x is List<object> row && row.Count == ((List<object>)list[0]).Count))
            {
                int cols = ((List<object>)list[0]).Count;
                var table = new object[list.Count, cols];
                for (int i = 0; i < list.Count; i++)
                    for (int j = 0; j < cols; j++)
                        table[i, j] = ((List<object>)list[i])[j];
                return table;
            }
            return list.Select(PadArgument).ToArray();
        }

        /// <summary>
        ///     <c>np.pad</c> with section strpad's arguments: <c>pad_width</c> (an int, a <c>[before, after]</c> pair, or per-axis
        ///     pairs), <c>mode</c>, and the optional mode keywords <c>constant_values</c> / <c>end_values</c> /
        ///     <c>stat_length</c> (encoded Python values: a list stays a list, NumPy's array_like) and <c>reflect_type</c>.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="p">The case parameters.</param>
        /// <returns>The padded array.</returns>
        private static NDArray PadWithKeywords(NDArray a, Dictionary<string, JsonElement> p)
        {
            string mode = Str(p, "mode");
            object constant = p.TryGetValue("constant_values", out var ce) ? PadArgument(StringOracleCorpus.Decode(ce)) : null;
            object end = p.TryGetValue("end_values", out var ee) ? PadArgument(StringOracleCorpus.Decode(ee)) : null;
            object stat = p.TryGetValue("stat_length", out var se) ? PadArgument(StringOracleCorpus.Decode(se)) : null;
            string reflect = p.TryGetValue("reflect_type", out var re) ? re.GetString() : "even";
            var pw = p["pad_width"];
            if (pw.ValueKind == JsonValueKind.Number)
                return np.pad(a, pw.GetInt32(), mode, constant, end, stat, reflect);
            if (pw[0].ValueKind == JsonValueKind.Number)
                return np.pad(a, (pw[0].GetInt32(), pw[1].GetInt32()), mode, constant, end, stat, reflect);
            var pairs = pw.EnumerateArray().Select(e => e.EnumerateArray().Select(x => x.GetInt32()).ToArray()).ToArray();
            var grid = new int[pairs.Length, 2];
            for (int i = 0; i < pairs.Length; i++)
            {
                grid[i, 0] = pairs[i][0];
                grid[i, 1] = pairs[i][1];
            }
            return np.pad(a, grid, mode, constant, end, stat, reflect);
        }
    }
}
