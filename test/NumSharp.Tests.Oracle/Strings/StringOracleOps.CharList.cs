using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's op for section <c>strcharlist</c> (<c>gen_strcharlist</c>): the ndarray members of numpy.char's
    ///     object result — <see cref="NDListArray"/>'s transposition, reshaping, copies and views, selection and the attributes
    ///     of the outer array. The receiver is a split result or a basic index of one (<see cref="ObjectPathValue"/>), the
    ///     arguments the generator's Python values; an <see cref="NDListArray"/> result comes back paired with its receiver,
    ///     whose identity the <c>base</c> facet of the comparison needs.
    /// </summary>
    internal static partial class StringOracleOps
    {
        /// <summary>The Python parameter names of each list-array member, in order (a keyword lands in its position).</summary>
        private static readonly Dictionary<string, string[]> s_listMemberParams = new()
        {
            ["to_device"] = new[] { "device", "stream" },
            ["transpose"] = Array.Empty<string>(),
            ["swapaxes"] = new[] { "axis1", "axis2" },
            ["reshape"] = Array.Empty<string>(),
            ["ravel"] = new[] { "order" },
            ["flatten"] = new[] { "order" },
            ["copy"] = new[] { "order" },
            ["squeeze"] = new[] { "axis" },
            ["view"] = Array.Empty<string>(),
            ["take"] = new[] { "indices", "axis", "out", "mode" },
            ["compress"] = new[] { "condition", "axis", "out" },
            ["repeat"] = new[] { "repeats", "axis" },
            ["diagonal"] = new[] { "offset", "axis1", "axis2" },
        };

        /// <summary>Replays one section-strcharlist op, or reports that <paramref name="op"/> is not one.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case parameters: <c>recv</c>, <c>name</c>, <c>args</c>, <c>kw</c>, and optionally <c>attr</c>
        ///     (an attribute, not a call) and <c>is_recv</c> (the result is whether the call returned its receiver).</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="result">The result: a <see cref="SelfResult"/> pairing a list array with its receiver, or a value.</param>
        /// <returns>True when <paramref name="op"/> is the section's op.</returns>
        /// <exception cref="InvalidOperationException">An unknown member (a generator / replay mismatch).</exception>
        private static bool TryInvokeCharList(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            if (op != "clst")
            {
                result = null;
                return false;
            }
            var receiver = (NDListArray)ObjectPathValue(p["recv"], ops);
            string name = p["name"].GetString();
            object r = p.TryGetValue("attr", out var attr) && attr.GetBoolean()
                ? ListAttribute(receiver, name)
                : ListMember(receiver, name, p, ops);
            if (p.TryGetValue("is_recv", out var isRecv) && isRecv.GetBoolean())
                r = ReferenceEquals(r, receiver);
            result = r is NDListArray ? new SelfResult(r, receiver) : r;
            return true;
        }

        /// <summary>One attribute of a list array.</summary>
        /// <param name="receiver">The list array.</param>
        /// <param name="name">The attribute's Python name.</param>
        /// <returns>Its value.</returns>
        /// <exception cref="InvalidOperationException">An unknown attribute.</exception>
        private static object ListAttribute(NDListArray receiver, string name) => name switch
        {
            "T" => receiver.T,
            "mT" => receiver.mT,
            "strides" => receiver.strides,
            "itemsize" => receiver.itemsize,
            "nbytes" => receiver.nbytes,
            "device" => receiver.device,
            "shape" => receiver.shape,
            "ndim" => receiver.ndim,
            "size" => receiver.size,
            _ => throw new InvalidOperationException($"section strcharlist has no attribute '{name}'"),
        };

        /// <summary>One member call of a list array, its arguments in Python's positions.</summary>
        /// <param name="receiver">The list array.</param>
        /// <param name="name">The member's Python name.</param>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The member's result.</returns>
        /// <exception cref="InvalidOperationException">An unknown member.</exception>
        private static object ListMember(NDListArray receiver, string name, Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            // transpose(*axes) and reshape(*shape) take their positions as a sequence; the others name each parameter.
            var positional = p.TryGetValue("args", out var a)
                ? a.EnumerateArray().Select(s => ObjectPathValue(s, ops)).ToArray()
                : Array.Empty<object>();
            // A member named by its parameters takes positional and keyword arguments in Python's positions.
            var names = s_listMemberParams[name];
            var args = names.Length == 0 ? Array.Empty<object>() : ObjectPathArgs(p, names, ops);
            int? Axis(object v) => v is null ? null : checked((int)(long)v);
            char Order(object v) => v is null ? 'C' : ((string)v)[0];
            string Mode(object v) => (string)v ?? "raise";
            return name switch
            {
                "to_device" => receiver.to_device((string)args[0], args[1]),
                "transpose" => receiver.transpose(positional.Select(x => checked((int)(long)x)).ToArray()),
                "swapaxes" => receiver.swapaxes(checked((int)(long)args[0]), checked((int)(long)args[1])),
                "reshape" => ListReshape(receiver, positional, p, ops),
                "ravel" => receiver.ravel(Order(args[0])),
                "flatten" => receiver.flatten(Order(args[0])),
                "copy" => receiver.copy(Order(args[0])),
                "squeeze" => receiver.squeeze(Axis(args[0])),
                "view" => receiver.view(),
                "take" => args[0] is long index
                    ? receiver.take(index, Axis(args[1]), null, Mode(args[3]))
                    : receiver.take(ToArray(args[0]), Axis(args[1]), null, Mode(args[3])),
                "compress" => receiver.compress(ToArray(args[0]), Axis(args[1])),
                "repeat" => args[0] is long count ? receiver.repeat(count, Axis(args[1])) : receiver.repeat(ToArray(args[0]), Axis(args[1])),
                "diagonal" => receiver.diagonal(args[0] is null ? 0 : checked((int)(long)args[0]),
                                                args[1] is null ? 0 : checked((int)(long)args[1]),
                                                args[2] is null ? 1 : checked((int)(long)args[2])),
                _ => throw new InvalidOperationException($"section strcharlist has no member '{name}'"),
            };
        }

        /// <summary><c>reshape(*shape)</c> / <c>reshape(shape, order=...)</c> as a ported line spells it: the integers as
        ///     <c>params long[]</c>, a sequence with an order as a <see cref="Shape"/>.</summary>
        /// <param name="receiver">The list array.</param>
        /// <param name="positional">The positional arguments (integers, or one sequence).</param>
        /// <param name="p">The case parameters (an <c>order</c> keyword).</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The reshaped list array.</returns>
        private static NDListArray ListReshape(NDListArray receiver, object[] positional, Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            long[] dims = positional.Length == 1 && positional[0] is Array seq
                ? seq.Cast<object>().Select(x => Convert.ToInt64(x)).ToArray()
                : positional.Select(x => (long)x).ToArray();
            if (p.TryGetValue("kw", out var kw) && kw.TryGetProperty("order", out var order))
                return receiver.reshape(new Shape(dims), ((string)ObjectPathValue(order, ops))[0]);
            return receiver.reshape(dims);
        }

        /// <summary>An index / condition / count argument as the array a ported line passes: an <see cref="NDArray"/> is
        ///     itself, a typed C# array converts implicitly, and the replay's object array of Python bools / ints (a decoded
        ///     list, <see cref="ManagedArray"/>) becomes the bool / int64 array of its rank that NumPy makes of the list.</summary>
        /// <param name="value">The argument.</param>
        /// <returns>The array.</returns>
        private static NDArray ToArray(object value)
        {
            if (value is NDArray nd)
                return nd;
            var array = (Array)value;
            if (array.GetType().GetElementType() != typeof(object))
                return array;
            var leaves = array.Cast<object>().ToArray();
            bool flags = leaves.Length > 0 && leaves.All(x => x is bool);
            NDArray flat = flags ? leaves.Select(x => (bool)x).ToArray() : leaves.Select(Convert.ToInt64).ToArray();
            if (array.Rank == 1)
                return flat;
            var dims = Enumerable.Range(0, array.Rank).Select(d => (long)array.GetLength(d)).ToArray();
            return flat.reshape(dims);
        }
    }
}
