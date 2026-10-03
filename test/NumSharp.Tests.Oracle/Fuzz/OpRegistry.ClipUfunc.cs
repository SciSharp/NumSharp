using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The <c>clip_ufunc</c> tier (gen_oracle.py <c>gen_clip_ufunc</c>): np.clip as the ufunc call it routes to —
    ///     array bounds that broadcast, out= (broadcast up into it, other layouts, a cast, read-only, wrong shape),
    ///     where=, an out overlapping an input, NumPy's const-vs-general float loop choice, and the broadcast /
    ///     keyword-rule error texts.
    /// </summary>
    /// <remarks>
    ///     Every argument is named in the case params (absent = NumPy's no-value, passed here as null). A spec is
    ///     <c>{"op": k}</c> (operand k), <c>{"of": k, "view": "dsl"}</c> (a view of operand k, so several arguments
    ///     can alias ONE buffer, which is how the overlap cases are built), or <c>{"py": v, "t": kind}</c> (a Python
    ///     scalar, passed as the C# literal of that kind: int → <see cref="PyIntLiteral"/>, float → a double,
    ///     bool → a bool, none → null). The result is a tuple: the returned array, then — when out= is given — the
    ///     whole base buffer behind out after the call, so a write outside the view or into a masked-off slot fails.
    /// </remarks>
    public static partial class OpRegistry
    {
        /// <summary>Replays one <c>clip_ufunc</c> case; see the type remarks for the param grammar.</summary>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>[returned] or [returned, base buffer behind out].</returns>
        /// <exception cref="NotSupportedException">A spec the grammar does not know (a generator / registry mismatch).</exception>
        private static NDArray[] ClipUfuncFromCorpus(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            NDArray Arg(string name) => p.TryGetValue(name, out var spec) ? ClipUfuncResolve(spec, ops) : null;

            var a = Arg("a");
            var aMin = Arg("a_min");
            var aMax = Arg("a_max");
            var min = Arg("min");
            var max = Arg("max");
            var where = Arg("where");
            var target = Arg("out");
            DType dtype = p.TryGetValue("dtype", out var dt) ? (DType)FuzzCorpus.DtypeToTC(dt.GetString()) : null;

            var returned = np.clip(a, aMin, aMax, target, dtype, min, max, where);
            if (target is null)
                return new[] { returned };

            // The out's OWNER buffer: an overlap case views operand 0, an ordinary one is its own operand.
            var outSpec = p["out"];
            int owner = outSpec.TryGetProperty("op", out var oi) ? oi.GetInt32() : outSpec.GetProperty("of").GetInt32();
            return new[] { returned, BaseBuffer(ops[owner]) };
        }

        /// <summary>One argument spec of the <c>clip_ufunc</c> grammar, resolved against the operands.</summary>
        /// <param name="spec">The JSON spec.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The argument (null for a Python None).</returns>
        /// <exception cref="NotSupportedException">An unknown scalar kind or view step.</exception>
        private static NDArray ClipUfuncResolve(JsonElement spec, NDArray[] ops)
        {
            if (spec.TryGetProperty("op", out var op))
                return ops[op.GetInt32()];
            if (spec.TryGetProperty("of", out var of))
                return ClipUfuncView(ops[of.GetInt32()], spec.GetProperty("view").GetString());

            var v = spec.GetProperty("py");
            return spec.GetProperty("t").GetString() switch
            {
                "none" => null,
                "int" => PyIntLiteral(v),
                "float" => NDArray.Scalar(v.GetDouble()),
                "bool" => NDArray.Scalar(v.GetBoolean()),
                var t => throw new NotSupportedException($"clip_ufunc: unknown scalar kind '{t}'"),
            };
        }

        /// <summary>
        ///     The view DSL of the <c>clip_ufunc</c> params: '|'-separated steps, each a slice (<c>[1:]</c>), <c>T</c>,
        ///     or <c>reshape(d0,d1,...)</c>, applied left to right as gen_oracle.py's <c>_clip_ufunc_apply_view</c> does.
        /// </summary>
        /// <param name="x">The array the steps start from.</param>
        /// <param name="dsl">The steps.</param>
        /// <returns>The view (shares <paramref name="x"/>'s buffer).</returns>
        /// <exception cref="NotSupportedException">An unknown step.</exception>
        private static NDArray ClipUfuncView(NDArray x, string dsl)
        {
            var v = x;
            foreach (var step in dsl.Split('|'))
            {
                if (step == "T")
                    v = v.T;
                else if (step.StartsWith("reshape(", StringComparison.Ordinal))
                    v = v.reshape(step["reshape(".Length..^1].Split(',').Select(t => long.Parse(t, CultureInfo.InvariantCulture)).ToArray());
                else if (step.StartsWith('[') && step.EndsWith(']'))
                    v = v[step[1..^1]];
                else
                    throw new NotSupportedException($"clip_ufunc: unknown view step '{step}'");
            }
            return v;
        }
    }
}
