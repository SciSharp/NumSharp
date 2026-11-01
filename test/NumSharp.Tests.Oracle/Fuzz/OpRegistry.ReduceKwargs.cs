using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The <c>reduce_kwargs</c> tier (gen_oracle.py <c>gen_reduce_kwargs</c>): each reduction called through its
    ///     NumPy-shaped overload with <c>out=</c>, <c>where=</c>, <c>keepdims</c> and every axis form, and NumPy's
    ///     reduce-prologue errors (plan <c>docs/plans/api-parameter-parity.md</c>).
    /// </summary>
    /// <remarks>
    ///     Params: <c>{"fn", "a", "axis", "keepdims", "out"?, "where"?, "initial"?}</c>. <c>axis</c> is JSON null
    ///     (NumPy's <c>axis=None</c>), an integer, or an array (NumPy's tuple axis, which binds the <c>int[]</c>
    ///     overload). The array arguments use the <c>clip_ufunc</c> spec grammar (<see cref="ClipUfuncResolve"/>).
    ///     The result is a tuple: the returned array; then, when <c>out=</c> is given, the whole base buffer behind
    ///     it (a write outside the view fails) and a 0-d bool that is true when the call returned <c>out</c> itself
    ///     (NumPy's <c>r is out</c>).
    /// </remarks>
    public static partial class OpRegistry
    {
        /// <summary>Replays one <c>reduce_kwargs</c> case; see the type remarks for the param grammar.</summary>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>[returned] or [returned, base buffer behind out, returned-is-out].</returns>
        /// <exception cref="NotSupportedException">A function the tier does not know (a generator / registry
        ///     mismatch).</exception>
        private static NDArray[] ReduceKwargsFromCorpus(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            NDArray Arg(string name) => p.TryGetValue(name, out var spec) ? ClipUfuncResolve(spec, ops) : null;

            string fn = p["fn"].GetString();
            var a = Arg("a");
            var target = Arg("out");
            var where = Arg("where");
            bool keepdims = p.TryGetValue("keepdims", out var kd) && kd.GetBoolean();
            JsonElement axisSpec = p.TryGetValue("axis", out var ax) ? ax : default;
            int[] axes = axisSpec.ValueKind == JsonValueKind.Array ? axisSpec.EnumerateArray().Select(e => e.GetInt32()).ToArray() : null;
            int? axis = axisSpec.ValueKind == JsonValueKind.Number ? axisSpec.GetInt32() : null;
            DType dtype = p.TryGetValue("dtype", out var dt) ? (DType)FuzzCorpus.DtypeToTC(dt.GetString()) : null;
            int ddof = p.TryGetValue("ddof", out var dd) ? dd.GetInt32() : 0;
            int? correction = p.TryGetValue("correction", out var cr) ? cr.GetInt32() : null;
            object initial = p.TryGetValue("initial", out var ini) ? ReduceKwargsInitial(ini, ops) : null;
            NDArray mean = p.TryGetValue("mean", out var mn) ? ReduceKwargsMean(mn, ops) : null;

            NDArray returned = fn switch
            {
                "all" => axes is not null ? np.all(a, axes, target, keepdims, where) : np.all(a, axis, target, keepdims, where),
                "any" => axes is not null ? np.any(a, axes, target, keepdims, where) : np.any(a, axis, target, keepdims, where),
                // np-layer dtype plan, P10 item 1: NumPy's whole signature, by name, so a reordered parameter fails here.
                "sum" => np.sum(a, axis, dtype: dtype, @out: target, keepdims: keepdims, initial: initial, @where: where),
                "prod" => np.prod(a, axis, dtype: dtype, @out: target, keepdims: keepdims, initial: initial, @where: where),
                "mean" => np.mean(a, axis, dtype: dtype, @out: target, keepdims: keepdims, @where: where),
                "var" => np.var(a, axis, dtype: dtype, @out: target, ddof: ddof, keepdims: keepdims, @where: where, mean: mean, correction: correction),
                "std" => np.std(a, axis, dtype: dtype, @out: target, ddof: ddof, keepdims: keepdims, @where: where, mean: mean, correction: correction),
                _ => throw new NotSupportedException($"reduce_kwargs: unknown function '{fn}'"),
            };

            if (target is null)
                return new[] { returned };
            int owner = p["out"].GetProperty("op").GetInt32();
            return new[] { returned, BaseBuffer(ops[owner]), NDArray.Scalar(ReferenceEquals(returned, target)) };
        }

        /// <summary>
        ///     An <c>initial=</c> spec as the C# value the caller would pass: an operand (a 0-d array NumPy casts), or a Python
        ///     scalar as its C# twin (a <c>long</c> is a Python int, a <c>double</c> a Python float, a <c>bool</c> a Python
        ///     bool, a <see cref="System.Numerics.Complex"/> a Python complex) — <c>PyArray_Pack</c> reads each kind by its
        ///     own rule, so the value must not become a 0-d array first.
        /// </summary>
        /// <param name="spec">The JSON spec (<c>{"op": k}</c> or <c>{"py": v, "t": kind}</c>; a complex is <c>[re, im]</c>).</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The value.</returns>
        /// <exception cref="NotSupportedException">An unknown scalar kind.</exception>
        private static object ReduceKwargsInitial(JsonElement spec, NDArray[] ops)
        {
            if (spec.TryGetProperty("op", out var op))
                return ops[op.GetInt32()];
            var v = spec.GetProperty("py");
            return spec.GetProperty("t").GetString() switch
            {
                "int" => (object)v.GetInt64(),
                "float" => v.GetDouble(),
                "bool" => v.GetBoolean(),
                "complex" => new System.Numerics.Complex(v[0].GetDouble(), v[1].GetDouble()),
                var t => throw new NotSupportedException($"reduce_kwargs: unknown initial kind '{t}'"),
            };
        }

        /// <summary>
        ///     A <c>mean=</c> spec as the argument the caller would pass: an operand, or a Python scalar through the C#
        ///     literal's implicit conversion (which marks it as NumPy's weak Python scalar, as <c>np.var(a, mean: 0.5)</c>
        ///     does).
        /// </summary>
        /// <param name="spec">The JSON spec.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The mean argument.</returns>
        /// <exception cref="NotSupportedException">An unknown scalar kind.</exception>
        private static NDArray ReduceKwargsMean(JsonElement spec, NDArray[] ops)
        {
            if (spec.TryGetProperty("op", out var op))
                return ops[op.GetInt32()];
            var v = spec.GetProperty("py");
            return spec.GetProperty("t").GetString() switch
            {
                "int" => (NDArray)v.GetInt64(),
                "float" => (NDArray)v.GetDouble(),
                var t => throw new NotSupportedException($"reduce_kwargs: unknown mean kind '{t}'"),
            };
        }
    }
}
