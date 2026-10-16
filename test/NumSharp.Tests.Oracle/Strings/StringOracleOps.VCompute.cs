using System;
using System.Collections.Generic;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for sections <c>tcomp</c> / <c>tarith</c> / <c>treduce</c> (<c>gen_tcomp</c> /
    ///     <c>gen_tarith</c> / <c>gen_treduce</c>, strings phase 2, P2.T): computing on NEP 55 <c>StringDType</c> text. The
    ///     comparisons, truth readers, binary ufuncs and keyword ufuncs replay through the shared <c>ufunc</c>,
    ///     <c>ufunc_kw</c>, <c>astype</c>, <c>call</c>, <c>truth_axis</c> and <c>operator_text</c> ops; this partial adds the
    ///     reductions with NumPy's keywords (<c>t_red</c>), the statistics (<c>t_stat</c>), <c>np.trace</c> (<c>t_trace</c>)
    ///     and the <c>*</c> operator with a C# integer (<c>t_operator_int</c>).
    /// </summary>
    /// <remarks>
    ///     Each op calls the NumSharp overload a ported NumPy line binds: a keyword the generator left out is not passed (an
    ///     absent <c>initial</c> is NumPy's no-initial, an absent <c>where</c> is <c>where=True</c>), and the C# literals of
    ///     a Python scalar (a <see cref="long"/> count, a <see cref="string"/> initial value) keep their Python meaning
    ///     through NumSharp's literal conversions.
    /// </remarks>
    internal static partial class StringOracleOps
    {
        /// <summary>Replays one section tcomp / tarith / treduce op, or reports that <paramref name="op"/> is not one of them.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case parameters (see each op's method).</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="result">The call's result (the out array itself for an <c>out=</c> call).</param>
        /// <returns>True when <paramref name="op"/> is one of this partial's ops.</returns>
        /// <exception cref="InvalidOperationException">A function name the op does not know (a generator / table mismatch).</exception>
        private static bool TryInvokeVCompute(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            switch (op)
            {
                case "t_red":
                    result = TReduce(p, ops);
                    return true;
                case "t_stat":
                    result = TStatistic(p, ops);
                    return true;
                case "t_trace":
                    result = np.trace(ops[0], Int(p, "offset"), Int(p, "axis1"), Int(p, "axis2"), Dt(p, "dtype"),
                        NullableInt(p, "out") is { } oi ? ops[oi] : null);
                    return true;
                case "t_operator_int":
                {
                    // A C# integer literal is NumPy's Python int: the implicit conversion marks it weak.
                    long value = p["value"].GetInt64();
                    if (Str(p, "op") != "*")
                        throw new InvalidOperationException($"the string oracle has no integer operator '{Str(p, "op")}'");
                    result = Str(p, "side") == "right" ? ops[0] * value : value * ops[0];
                    return true;
                }
            }
            result = null;
            return false;
        }

        /// <summary>
        ///     A reduction of section treduce with the generator's keywords: <c>name</c>, <c>axis</c> (int or None),
        ///     <c>keepdims</c>, and the optional <c>dtype</c>, <c>out</c> (an operand index), <c>initial</c> (an encoded
        ///     Python value) and <c>where</c> (an operand index).
        /// </summary>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The operands (the array first).</param>
        /// <returns>The call's result.</returns>
        /// <exception cref="InvalidOperationException">An unknown name.</exception>
        private static object TReduce(Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            NDArray a = ops[0];
            int? axis = NullableInt(p, "axis");
            bool keepdims = p["keepdims"].GetBoolean();
            var dt = Dt(p, "dtype");
            NDArray @out = p.TryGetValue("out", out var oe) && oe.ValueKind == JsonValueKind.Number ? ops[oe.GetInt32()] : null;
            NDArray where = p.TryGetValue("where", out var we) && we.ValueKind == JsonValueKind.Number ? ops[we.GetInt32()] : null;
            object initial = p.TryGetValue("initial", out var ie) ? StringOracleCorpus.Decode(ie) : null;
            return Str(p, "name") switch
            {
                "sum" => np.sum(a, axis, keepdims, dt),
                "max" => np.max(a, axis, @out, keepdims, initial, where),
                "min" => np.min(a, axis, @out, keepdims, initial, where),
                "amax" => np.amax(a, axis, @out, keepdims, initial, where),
                "amin" => np.amin(a, axis, @out, keepdims, initial, where),
                "nanmax" => np.nanmax(a, axis, @out, keepdims, initial, where),
                "nanmin" => np.nanmin(a, axis, @out, keepdims, initial, where),
                "argmax" => np.argmax(a, axis, @out, keepdims),
                "argmin" => np.argmin(a, axis, @out, keepdims),
                "nanargmax" => np.nanargmax(a, axis, @out, keepdims),
                "nanargmin" => np.nanargmin(a, axis, @out, keepdims),
                "cumsum" => np.cumsum(a, axis, dt, @out),
                "nancumsum" => np.nancumsum(a, axis, dt, @out),
                "cumprod" => np.cumprod(a, axis, dt, @out),
                "nancumprod" => np.nancumprod(a, axis, dt, @out),
                "nansum" => np.nansum(a, axis, keepdims),
                "prod" => np.prod(a, axis, dt, keepdims),
                "nanprod" => np.nanprod(a, axis, keepdims),
                var other => throw new InvalidOperationException($"the string oracle has no StringDType reduction '{other}'"),
            };
        }

        /// <summary>
        ///     A statistic of section treduce with the generator's keywords: <c>name</c>, <c>axis</c> (int or None),
        ///     <c>keepdims</c>, and the optional <c>dtype</c>, <c>ddof</c>, <c>out</c> (an operand index, np.ptp) and
        ///     <c>weights</c> (an operand index, np.average) — spelled as NumSharp's overloads take them (np.mean takes
        ///     <c>dtype=</c> with an explicit axis only, which the generator always gives).
        /// </summary>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The operands (the array first).</param>
        /// <returns>The call's result.</returns>
        /// <exception cref="InvalidOperationException">An unknown name, or a keyword combination no overload takes.</exception>
        private static object TStatistic(Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            NDArray a = ops[0];
            int? axis = NullableInt(p, "axis");
            bool keepdims = p["keepdims"].GetBoolean();
            var dt = Dt(p, "dtype");
            int? ddof = p.TryGetValue("ddof", out var de) && de.ValueKind == JsonValueKind.Number ? de.GetInt32() : null;
            NDArray @out = p.TryGetValue("out", out var oe) && oe.ValueKind == JsonValueKind.Number ? ops[oe.GetInt32()] : null;
            NDArray weights = p.TryGetValue("weights", out var wte) && wte.ValueKind == JsonValueKind.Number ? ops[wte.GetInt32()] : null;
            switch (Str(p, "name"))
            {
                case "mean":
                    if (dt is not null)
                        return np.mean(a, axis ?? throw new InvalidOperationException("np.mean's dtype= needs an axis"), dt, keepdims);
                    return axis is { } m ? np.mean(a, m, keepdims) : np.mean(a, keepdims);
                case "average":
                    return np.average(a, axis, weights, keepdims);
                case "nanmean":
                    return np.nanmean(a, axis, keepdims, dt);
                case "var":
                    return axis is { } v ? np.var(a, v, keepdims, ddof, dt) : np.var(a, keepdims, ddof, dt);
                case "std":
                    return axis is { } s ? np.std(a, s, keepdims, ddof, dt) : np.std(a, keepdims, ddof, dt);
                case "nanvar":
                    return np.nanvar(a, axis, keepdims, ddof ?? 0, dt);
                case "nanstd":
                    return np.nanstd(a, axis, keepdims, ddof ?? 0, dt);
                case "ptp":
                    return np.ptp(a, axis, @out, keepdims);
                case var other:
                    throw new InvalidOperationException($"the string oracle has no StringDType statistic '{other}'");
            }
        }
    }
}
