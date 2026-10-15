using System.Collections.Generic;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for section <c>strsearch</c> (<c>gen_strsearch</c>, strings phase 2, P2.2 milestone B):
    ///     <c>np.argmax</c> / <c>np.argmin</c> / <c>np.nanargmax</c> / <c>np.nanargmin</c> and the ndarray methods with
    ///     NumPy's <c>axis</c> / <c>out</c> / <c>keepdims</c>, and <c>ndarray.searchsorted</c>. The section's
    ///     <c>np.searchsorted</c> and <c>np.digitize</c> cases replay through the ops of section strvalid
    ///     (<c>searchsorted_kw</c>, <c>digitize</c>), which take the same parameters.
    /// </summary>
    internal static partial class StringOracleOps
    {
        /// <summary>Replays one section-strsearch op, or reports that <paramref name="op"/> is not one of them.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case parameters: for <c>argred_kw</c> the function <c>name</c>, <c>axis</c> (int or None),
        ///     <c>keepdims</c>, <c>method</c> (the ndarray method instead of the function) and <c>out</c> (operand 1 is the
        ///     out array); for <c>ndsearchsorted</c> the <c>side</c> and <c>sorter</c> (operand 2 is the sorter).</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="result">The call's result (the out array itself for an <c>out=</c> call, whose view and base the
        ///     replay compares).</param>
        /// <returns>True when <paramref name="op"/> is a section-strsearch op (its result is in <paramref name="result"/>).</returns>
        private static bool TryInvokeSearch(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            switch (op)
            {
                case "argred_kw":
                {
                    NDArray a = ops[0];
                    NDArray @out = p["out"].GetBoolean() ? ops[1] : null;
                    int? axis = NullableInt(p, "axis");
                    bool keepdims = p["keepdims"].GetBoolean();
                    string name = Str(p, "name");
                    if (p["method"].GetBoolean())
                        result = name == "argmax" ? a.argmax(axis, @out, keepdims) : a.argmin(axis, @out, keepdims);
                    else
                        result = name switch
                        {
                            "argmax" => np.argmax(a, axis, @out, keepdims),
                            "argmin" => np.argmin(a, axis, @out, keepdims),
                            "nanargmax" => np.nanargmax(a, axis, @out, keepdims: keepdims),
                            _ => np.nanargmin(a, axis, @out, keepdims: keepdims),
                        };
                    return true;
                }
                case "ndsearchsorted":
                    result = ops[0].searchsorted(ops[1], Str(p, "side"), p["sorter"].GetBoolean() ? ops[2] : null);
                    return true;
            }

            result = null;
            return false;
        }
    }
}
