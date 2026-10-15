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

            NDArray returned = fn switch
            {
                "all" => axes is not null ? np.all(a, axes, target, keepdims, where) : np.all(a, axis, target, keepdims, where),
                "any" => axes is not null ? np.any(a, axes, target, keepdims, where) : np.any(a, axis, target, keepdims, where),
                _ => throw new NotSupportedException($"reduce_kwargs: unknown function '{fn}'"),
            };

            if (target is null)
                return new[] { returned };
            int owner = p["out"].GetProperty("op").GetInt32();
            return new[] { returned, BaseBuffer(ops[owner]), NDArray.Scalar(ReferenceEquals(returned, target)) };
        }
    }
}
