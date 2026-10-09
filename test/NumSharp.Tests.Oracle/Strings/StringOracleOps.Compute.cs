using System;
using System.Collections.Generic;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for section <c>strcomp</c> (<c>gen_strcomp</c>, strings phase 2 P2.1) that the shared ops do
    ///     not already cover: the C# comparison operators with a <see cref="string"/> operand (NumPy's Python <c>str</c>) and
    ///     the truth reductions along an axis. The section's comparisons, casts, keyword ufuncs and truth calls replay through
    ///     the shared <c>ufunc</c>, <c>ufunc_kw</c>, <c>astype</c> and <c>call</c> ops.
    /// </summary>
    internal static partial class StringOracleOps
    {
        /// <summary>Replays one <c>strcomp</c>-only op, or reports that <paramref name="op"/> is not one of them.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="result">The call's result.</param>
        /// <returns>True when <paramref name="op"/> is a <c>strcomp</c>-only op.</returns>
        /// <exception cref="InvalidOperationException">An unknown operator or truth reduction name.</exception>
        private static bool TryInvokeCompute(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            switch (op)
            {
                case "operator_text":
                    result = OperatorText(Str(p, "op"), (string)StringOracleCorpus.Decode(p["text"]), ops[0], Str(p, "side") == "right");
                    return true;
                case "truth_axis":
                {
                    int axis = Int(p, "axis");
                    bool keepdims = p["keepdims"].GetBoolean();
                    result = Str(p, "name") switch
                    {
                        "any" => np.any(ops[0], axis, keepdims),
                        "all" => np.all(ops[0], axis, keepdims),
                        "count_nonzero" => np.count_nonzero(ops[0], axis, keepdims),
                        var other => throw new InvalidOperationException($"the string oracle has no truth reduction '{other}'"),
                    };
                    return true;
                }
            }
            result = null;
            return false;
        }

        /// <summary>
        ///     A C# comparison operator between an array and a C# <see cref="string"/> — the overloads that read the string as
        ///     NumPy's <c>str</c> scalar against a text array (<c>NDArray.TextComparison.cs</c>).
        /// </summary>
        /// <param name="symbol">The operator: <c>==</c>, <c>!=</c>, <c>&lt;</c>, <c>&lt;=</c>, <c>&gt;</c>, <c>&gt;=</c>.</param>
        /// <param name="text">The string.</param>
        /// <param name="a">The array.</param>
        /// <param name="right">True for <c>a OP text</c>, false for <c>text OP a</c>.</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="InvalidOperationException">An unknown operator.</exception>
        private static object OperatorText(string symbol, string text, NDArray a, bool right) => (symbol, right) switch
        {
            ("==", true) => a == text,
            ("==", false) => text == a,
            ("!=", true) => a != text,
            ("!=", false) => text != a,
            ("<", true) => a < text,
            ("<", false) => text < a,
            ("<=", true) => a <= text,
            ("<=", false) => text <= a,
            (">", true) => a > text,
            (">", false) => text > a,
            (">=", true) => a >= text,
            (">=", false) => text >= a,
            _ => throw new InvalidOperationException($"the string oracle has no text operator '{symbol}'"),
        };
    }
}
