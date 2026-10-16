using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for section <c>strcharobj</c> (<c>gen_strcharobj</c>, the <c>numpy.char</c> plan's phase
    ///     4): the object path — <c>np.@char.split</c> / <c>rsplit</c> / <c>splitlines</c> / <c>join</c> (<c>cobj_fn</c>),
    ///     the chararray members over them (<c>cobj_member</c>) and the printing of their object results
    ///     (<c>cobj_print</c>). Each argument arrives as the generator's spec (<c>_cobj_value</c>): an operand, its chararray
    ///     view, a Python value (the C# value the facade reads as that Python object: a str as <see cref="string"/>, bytes as
    ///     <c>byte[]</c>, an int as <see cref="long"/> or a <see cref="System.Numerics.BigInteger"/> past int64, a float as
    ///     <see cref="double"/>, a complex as <see cref="System.Numerics.Complex"/>, None as null, a list as the C# array
    ///     <see cref="ManagedArray"/> builds — an <c>object</c> array when it holds None), a split result, or a basic index of
    ///     one.
    /// </summary>
    internal static partial class StringOracleOps
    {
        /// <summary>The Python parameter names of each module function, in order (a keyword lands in its position).</summary>
        private static readonly Dictionary<string, string[]> s_objectPathParams = new()
        {
            ["split"] = new[] { "a", "sep", "maxsplit" },
            ["rsplit"] = new[] { "a", "sep", "maxsplit" },
            ["splitlines"] = new[] { "a", "keepends" },
            ["join"] = new[] { "sep", "seq" },
        };

        /// <summary>The Python parameter names of each chararray member, in order.</summary>
        private static readonly Dictionary<string, string[]> s_objectPathMemberParams = new()
        {
            ["split"] = new[] { "sep", "maxsplit" },
            ["rsplit"] = new[] { "sep", "maxsplit" },
            ["splitlines"] = new[] { "keepends" },
            ["join"] = new[] { "seq" },
        };

        /// <summary>Replays one section-strcharobj op, or reports that <paramref name="op"/> is not one of them.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case parameters: <c>fn</c> or <c>member</c> / <c>recv</c>, <c>args</c>, <c>kw</c>, and
        ///     optionally <c>index</c> (a basic index of the result) and <c>print</c> (repr when true, str when false).</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="result">The call's result: an <see cref="NDListArray"/>, an array, or the printed text.</param>
        /// <returns>True when <paramref name="op"/> is a section-strcharobj op.</returns>
        /// <exception cref="InvalidOperationException">An unknown function, member or argument spec (a generator / replay
        ///     mismatch).</exception>
        private static bool TryInvokeCharObject(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            if (op is not ("cobj_fn" or "cobj_member" or "cobj_print"))
            {
                result = null;
                return false;
            }
            object r;
            if (op == "cobj_member")
            {
                string member = p["member"].GetString();
                var receiver = (NDCharArray)ops[p["recv"].GetInt32()].view(typeof(NDCharArray));
                var args = ObjectPathArgs(p, s_objectPathMemberParams[member], ops);
                r = member switch
                {
                    "split" => receiver.split(args[0], args[1]),
                    "rsplit" => receiver.rsplit(args[0], args[1]),
                    "splitlines" => receiver.splitlines(args[0]),
                    "join" => args[0] is NDListArray lists ? receiver.join(lists) : receiver.join(args[0]),
                    _ => throw new InvalidOperationException($"section strcharobj has no member '{member}'"),
                };
            }
            else
            {
                string fn = p["fn"].GetString();
                var args = ObjectPathArgs(p, s_objectPathParams[fn], ops);
                r = ObjectPathCall(fn, args);
            }
            if (p.TryGetValue("index", out var index))
            {
                // A full integer index (one integer per axis) is NumPy's element — the Python list itself, which item()
                // returns; any other basic index is a view.
                var lists = (NDListArray)r;
                var text = index.GetString();
                var parts = text.Split(',').Select(s => s.Trim()).ToArray();
                r = parts.Length == lists.ndim && parts.All(s => !s.Contains(':'))
                    ? lists.item(parts.Select(long.Parse).ToArray())
                    : lists[text];
            }
            if (p.TryGetValue("print", out var print))
                r = r is NDListArray l ? l.ToString(print.GetBoolean()) : ((NDArray)r).ToString(print.GetBoolean());
            result = r;
            return true;
        }

        /// <summary>One module function of the object path with its arguments in Python's positions (an omitted one
        ///     null, NumPy's <c>None</c> default).</summary>
        /// <param name="fn">The function name.</param>
        /// <param name="args">The arguments.</param>
        /// <returns>The result.</returns>
        /// <exception cref="InvalidOperationException">An unknown function.</exception>
        private static object ObjectPathCall(string fn, object[] args) => fn switch
        {
            "split" => np.@char.split(args[0], args[1], args[2]),
            "rsplit" => np.@char.rsplit(args[0], args[1], args[2]),
            "splitlines" => np.@char.splitlines(args[0], args[1]),
            // A list array binds the overload that can return NumPy's empty object array (join over a broadcast that
            // holds no element).
            "join" => args[1] is NDListArray lists ? np.@char.join(args[0], lists) : np.@char.join(args[0], args[1]),
            _ => throw new InvalidOperationException($"section strcharobj has no function '{fn}'"),
        };

        /// <summary>The case's positional and keyword arguments placed in the Python parameter order.</summary>
        /// <param name="p">The case parameters.</param>
        /// <param name="names">The parameter names in order.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>One value per parameter (null when not passed).</returns>
        /// <exception cref="InvalidOperationException">An unknown keyword.</exception>
        private static object[] ObjectPathArgs(Dictionary<string, JsonElement> p, string[] names, NDArray[] ops)
        {
            var values = new object[names.Length];
            if (p.TryGetValue("args", out var args))
            {
                int i = 0;
                foreach (var spec in args.EnumerateArray())
                    values[i++] = ObjectPathValue(spec, ops);
            }
            if (p.TryGetValue("kw", out var kw))
            {
                foreach (var prop in kw.EnumerateObject())
                {
                    int at = Array.IndexOf(names, prop.Name);
                    if (at < 0)
                        throw new InvalidOperationException($"section strcharobj has no keyword '{prop.Name}'");
                    values[at] = ObjectPathValue(prop.Value, ops);
                }
            }
            return values;
        }

        /// <summary>
        ///     One argument spec: <c>{"op": k}</c> the operand, <c>{"cha": k}</c> its chararray view, <c>{"v": ...}</c> a
        ///     Python value (<see cref="ObjectPathPythonValue"/>), <c>{"split": [...]}</c> <c>np.@char.split</c> of those
        ///     arguments, <c>{"view": [spec, index]}</c> a basic index of a list array.
        /// </summary>
        /// <param name="spec">The spec.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The C# argument.</returns>
        /// <exception cref="InvalidOperationException">An unknown spec.</exception>
        private static object ObjectPathValue(JsonElement spec, NDArray[] ops)
        {
            if (spec.TryGetProperty("op", out var k))
                return ops[k.GetInt32()];
            if (spec.TryGetProperty("cha", out var c))
                return (NDCharArray)ops[c.GetInt32()].view(typeof(NDCharArray));
            if (spec.TryGetProperty("v", out var v))
                return ObjectPathPythonValue(StringOracleCorpus.Decode(v));
            if (spec.TryGetProperty("split", out var split))
            {
                var a = split.EnumerateArray().Select(s => ObjectPathValue(s, ops)).ToArray();
                return np.@char.split(a[0], a.Length > 1 ? a[1] : null, a.Length > 2 ? a[2] : null);
            }
            if (spec.TryGetProperty("view", out var view))
                return ((NDListArray)ObjectPathValue(view[0], ops))[view[1].GetString()];
            throw new InvalidOperationException($"section strcharobj has no argument spec {spec}");
        }

        /// <summary>A decoded Python value as the C# value the object path reads as it: a list as the managed array
        ///     <see cref="ManagedArray"/> builds (an object array when it holds None), every other value as itself.</summary>
        /// <param name="value">The decoded value.</param>
        /// <returns>The C# value.</returns>
        private static object ObjectPathPythonValue(object value) => value is List<object> ? ManagedArray(value) : value;
    }
}
