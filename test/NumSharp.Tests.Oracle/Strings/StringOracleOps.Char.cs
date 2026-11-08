using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for section <c>strchar</c> (<c>gen_strchar</c>, the <c>numpy.char</c> plan's phase 1):
    ///     <c>np.@char</c>'s six stripping comparisons, <c>compare_chararrays</c>, <c>add</c> and <c>multiply</c>. Each
    ///     argument arrives as the generator recorded it (<c>params.args</c>): an operand (<c>{"op": k}</c>, the k-th
    ///     reconstructed array) or a Python value (<c>{"v": ...}</c>) passed as the C# value the <c>numpy.char</c> facade
    ///     reads as that Python object — a str as <see cref="string"/>, bytes as <c>byte[]</c>, a list as the C# array the
    ///     <c>array</c> op builds (<see cref="ManagedArray"/>), an int as <see cref="long"/>, a float as <see cref="double"/>,
    ///     a bool as <see cref="bool"/>, None as null.
    /// </summary>
    internal static partial class StringOracleOps
    {
        /// <summary>Replays one section-strchar op, or reports that <paramref name="op"/> is not one of them.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case parameters: <c>args</c>, and <c>name</c> for a comparison.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="result">The call's result (null for NumPy's <c>NotImplemented</c>).</param>
        /// <returns>True when <paramref name="op"/> is a section-strchar op.</returns>
        /// <exception cref="InvalidOperationException">An unknown comparison name or argument spec (a generator / replay
        ///     mismatch).</exception>
        private static bool TryInvokeChar(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            switch (op)
            {
                case "char_cmp":
                {
                    var args = CharArgs(p, ops);
                    result = Str(p, "name") switch
                    {
                        "equal" => np.@char.equal(args[0], args[1]),
                        "not_equal" => np.@char.not_equal(args[0], args[1]),
                        "less" => np.@char.less(args[0], args[1]),
                        "less_equal" => np.@char.less_equal(args[0], args[1]),
                        "greater" => np.@char.greater(args[0], args[1]),
                        "greater_equal" => np.@char.greater_equal(args[0], args[1]),
                        var other => throw new InvalidOperationException($"np.char has no comparison '{other}'"),
                    };
                    return true;
                }
                case "char_compare":
                {
                    var args = CharArgs(p, ops);
                    bool rstrip = (bool)args[3];
                    // cmp is any Python value (str, bytes, None, a number): the one member reads it as NumPy's "s#" argument.
                    result = np.@char.compare_chararrays(args[0], args[1], args[2], rstrip);
                    return true;
                }
                case "char_add":
                {
                    var args = CharArgs(p, ops);
                    result = np.@char.add(args[0], args[1]);
                    return true;
                }
                case "char_multiply":
                {
                    var args = CharArgs(p, ops);
                    result = np.@char.multiply(args[0], args[1]);
                    return true;
                }
            }
            result = null;
            return false;
        }

        /// <summary>The call's positional arguments from <c>params.args</c> (see the class summary).</summary>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The arguments, in order.</returns>
        /// <exception cref="InvalidOperationException">An argument spec that is neither an operand nor a value.</exception>
        private static object[] CharArgs(Dictionary<string, JsonElement> p, NDArray[] ops)
            => p["args"].EnumerateArray().Select(spec =>
            {
                if (spec.TryGetProperty("op", out var k))
                    return (object)ops[k.GetInt32()];
                if (spec.TryGetProperty("v", out var v))
                    return ManagedArrayOrValue(StringOracleCorpus.Decode(v));
                throw new InvalidOperationException($"the strchar section has no argument spec {spec}");
            }).ToArray();
    }
}
