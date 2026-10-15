using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for section <c>strsort</c> (<c>gen_strsort</c>, strings phase 2, P2.2): <c>np.sort</c>,
    ///     <c>np.argsort</c> and <c>ndarray.sort</c> with NumPy's <c>kind</c> / <c>order</c> / <c>stable</c> keywords, and
    ///     <c>np.partition</c> / <c>np.argpartition</c> with a <c>kth</c>. Each op replays the C# call a ported NumPy line
    ///     makes, with the generator's arguments (a keyword NumPy was not given arrives as JSON null and is passed as C#
    ///     null, the methods' "not given").
    /// </summary>
    internal static partial class StringOracleOps
    {
        /// <summary>Replays one section-strsort op, or reports that <paramref name="op"/> is not one of them.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case parameters: <c>axis</c> (int or None), <c>kind</c> (str or None), <c>stable</c>
        ///     (bool or None), <c>order</c> (None, a field name or a list of them), and <c>kth</c> for the partitions.</param>
        /// <param name="ops">The reconstructed operands (the array to sort).</param>
        /// <param name="result">The call's result: the sorted copy or the indices, or the operand itself for the in-place
        ///     <c>ndarray.sort</c> (whose view and base the replay compares).</param>
        /// <returns>True when <paramref name="op"/> is a section-strsort op (its result is in <paramref name="result"/>).</returns>
        private static bool TryInvokeSort(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            NDArray a = ops.Length > 0 ? ops[0] : null;
            switch (op)
            {
                case "sort_kw":
                    result = np.sort(a, NullableInt(p, "axis"), NullableStr(p, "kind"), SortOrder(p), NullableBool(p, "stable"));
                    return true;
                case "argsort_kw":
                    result = np.argsort(a, NullableInt(p, "axis"), NullableStr(p, "kind"), SortOrder(p), NullableBool(p, "stable"));
                    return true;
                case "ndsort_kw":
                    // NumPy's ndarray.sort takes an int axis only; the generator records no None axis for it.
                    a.sort(Int(p, "axis"), NullableStr(p, "kind"), SortOrder(p), NullableBool(p, "stable"));
                    result = a;
                    return true;
                case "partition_kw":
                    result = np.partition(a, Int(p, "kth"), NullableInt(p, "axis"));
                    return true;
                case "argpartition_kw":
                    result = np.argpartition(a, Int(p, "kth"), NullableInt(p, "axis"));
                    return true;
            }

            result = null;
            return false;
        }

        /// <summary>A string-or-None parameter (absent counts as None).</summary>
        /// <param name="p">The case parameters.</param>
        /// <param name="key">The parameter name.</param>
        /// <returns>The string, or null for JSON null or an absent key.</returns>
        private static string NullableStr(Dictionary<string, JsonElement> p, string key)
            => p.TryGetValue(key, out var e) && e.ValueKind == JsonValueKind.String ? e.GetString() : null;

        /// <summary>A bool-or-None parameter (absent counts as None).</summary>
        /// <param name="p">The case parameters.</param>
        /// <param name="key">The parameter name.</param>
        /// <returns>The bool, or null for JSON null or an absent key.</returns>
        private static bool? NullableBool(Dictionary<string, JsonElement> p, string key)
            => p.TryGetValue(key, out var e) && e.ValueKind is JsonValueKind.True or JsonValueKind.False ? e.GetBoolean() : null;

        /// <summary>
        ///     The sort's <c>order</c> keyword as the C# call passes it: null for None, a field name as a string, a list of
        ///     field names as a string array. NumPy refuses every non-None value for an array without fields, which every
        ///     NumSharp dtype is.
        /// </summary>
        /// <param name="p">The case parameters.</param>
        /// <returns>The value.</returns>
        private static object SortOrder(Dictionary<string, JsonElement> p)
        {
            if (!p.TryGetValue("order", out var e) || e.ValueKind == JsonValueKind.Null)
                return null;
            return e.ValueKind == JsonValueKind.Array ? e.EnumerateArray().Select(x => x.GetString()).ToArray() : e.GetString();
        }
    }
}
