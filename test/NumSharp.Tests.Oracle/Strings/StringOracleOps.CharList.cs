using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's op for section <c>strcharlist</c> (<c>gen_strcharlist</c>): the ndarray members of numpy.char's
    ///     object result — <see cref="NDListArray"/>'s transposition, reshaping, copies and views, selection and the attributes
    ///     of the outer array (L-A), and its comparisons, truth, conversions, sort family, reductions, arithmetic and
    ///     assignment (L-B / L-C). The receiver is a split result or a basic index of one (<see cref="ObjectPathValue"/>), the
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
            ["view"] = new[] { "dtype", "type" },
            ["take"] = new[] { "indices", "axis", "out", "mode" },
            ["compress"] = new[] { "condition", "axis", "out" },
            ["repeat"] = new[] { "repeats", "axis" },
            ["diagonal"] = new[] { "offset", "axis1", "axis2" },
            // ---- L-B / L-C ----
            ["all"] = new[] { "axis", "out", "keepdims", "where" },
            ["any"] = new[] { "axis", "out", "keepdims", "where" },
            ["nonzero"] = Array.Empty<string>(),
            ["astype"] = new[] { "dtype", "order", "casting", "subok", "copy" },
            ["sort"] = new[] { "axis", "kind", "order", "stable" },
            ["argsort"] = new[] { "axis", "kind", "order", "stable" },
            ["partition"] = new[] { "kth", "axis", "kind", "order" },
            ["argpartition"] = new[] { "kth", "axis", "kind", "order" },
            ["searchsorted"] = new[] { "v", "side", "sorter" },
            ["argmax"] = new[] { "axis", "out", "keepdims" },
            ["argmin"] = new[] { "axis", "out", "keepdims" },
            ["max"] = new[] { "axis", "out", "keepdims", "initial", "where" },
            ["min"] = new[] { "axis", "out", "keepdims", "initial", "where" },
            ["sum"] = new[] { "axis", "dtype", "out", "keepdims", "initial", "where" },
            ["prod"] = new[] { "axis", "dtype", "out", "keepdims", "initial", "where" },
            ["cumsum"] = new[] { "axis", "dtype", "out" },
            ["cumprod"] = new[] { "axis", "dtype", "out" },
            ["trace"] = new[] { "offset", "axis1", "axis2", "dtype", "out" },
            ["clip"] = new[] { "min", "max", "out", "dtype", "where" },
            ["mean"] = new[] { "axis", "dtype", "out", "keepdims", "where" },
            ["var"] = new[] { "axis", "dtype", "out", "ddof", "keepdims", "where", "mean" },
            ["std"] = new[] { "axis", "dtype", "out", "ddof", "keepdims", "where", "mean" },
            ["dot"] = new[] { "b", "out" },
            ["round"] = new[] { "decimals", "out" },
            ["conj"] = new[] { "out" },
            ["conjugate"] = new[] { "out" },
            ["fill"] = new[] { "value" },
            ["put"] = new[] { "indices", "values", "mode" },
            ["resize"] = Array.Empty<string>(),
            ["byteswap"] = new[] { "inplace" },
            ["getfield"] = new[] { "dtype", "offset" },
            ["setfield"] = new[] { "value", "dtype", "offset" },
            ["choose"] = new[] { "choices", "out", "mode" },
        };

        /// <summary>Replays one section-strcharlist op, or reports that <paramref name="op"/> is not one.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case parameters: <c>recv</c>, <c>name</c>, <c>args</c>, <c>kw</c>, and optionally <c>attr</c>
        ///     (an attribute, not a call), <c>is_recv</c> (the result is whether the call returned its receiver),
        ///     <c>binop</c> / <c>other</c> / <c>rev</c> (an operator), <c>setitem</c> (an assignment at that index) and
        ///     <c>inplace</c> (the receiver after the call is the result).</param>
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
            object r;
            // The values the member received, in its Python parameter order (an is_arg case asks whether the result IS
            // one of them: the out= a member must return itself).
            object[] passed = Array.Empty<object>();
            if (p.TryGetValue("binop", out var binop))
            {
                bool rev = p.TryGetValue("rev", out var revJson) && revJson.GetBoolean();
                r = ListBinop(binop.GetString(), receiver, ObjectPathValue(p["other"], ops, receiver), rev);
            }
            else if (p.TryGetValue("setitem", out var index))
            {
                var value = ObjectPathValue(p["args"][0], ops, receiver);
                receiver[index.GetString()] = (NDListArray)value;
                r = receiver;
            }
            else
            {
                r = p.TryGetValue("attr", out var attr) && attr.GetBoolean()
                    ? ListAttribute(receiver, name)
                    : ListMember(receiver, name, p, ops, out passed);
                if (p.TryGetValue("inplace", out var inplace) && inplace.GetBoolean())
                    r = receiver;
            }
            if (p.TryGetValue("is_recv", out var isRecv) && isRecv.GetBoolean())
                r = ReferenceEquals(r, receiver);
            if (p.TryGetValue("is_arg", out var isArg))
            {
                // A positional index or a keyword name; both land in the parameter's slot.
                int slot = isArg.ValueKind == JsonValueKind.Number
                    ? isArg.GetInt32()
                    : Array.IndexOf(s_listMemberParams[name], isArg.GetString());
                r = ReferenceEquals(r, passed[slot]);
            }
            result = r is NDListArray ? new SelfResult(r, receiver) : r;
            return true;
        }

        /// <summary>
        ///     One operator of a list array with another operand, as a ported line writes it: the other operand typed
        ///     <see cref="object"/> binds the <c>(NDListArray, object)</c> / <c>(object, NDListArray)</c> overloads, whose
        ///     dispatch reads its run-time type (a list array, an ndarray, a C# value, null for None).
        /// </summary>
        /// <param name="op">The generator's operator name.</param>
        /// <param name="receiver">The list array.</param>
        /// <param name="other">The other operand.</param>
        /// <param name="rev">True for <c>other op receiver</c>.</param>
        /// <returns>The result.</returns>
        /// <exception cref="InvalidOperationException">An unknown operator.</exception>
        private static object ListBinop(string op, NDListArray receiver, object other, bool rev) => (op, rev) switch
        {
            ("eq", false) => receiver == other,
            ("eq", true) => other == receiver,
            ("ne", false) => receiver != other,
            ("ne", true) => other != receiver,
            ("lt", false) => receiver < other,
            ("lt", true) => other < receiver,
            ("le", false) => receiver <= other,
            ("le", true) => other <= receiver,
            ("gt", false) => receiver > other,
            ("gt", true) => other > receiver,
            ("ge", false) => receiver >= other,
            ("ge", true) => other >= receiver,
            ("add", false) => receiver + other,
            ("add", true) => other + receiver,
            ("mul", false) => receiver * other,
            ("mul", true) => other * receiver,
            _ => throw new InvalidOperationException($"section strcharlist has no operator '{op}'"),
        };

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
            "real" => receiver.real,
            "imag" => receiver.imag,
            _ => throw new InvalidOperationException($"section strcharlist has no attribute '{name}'"),
        };

        /// <summary>One member call of a list array, its arguments in Python's positions.</summary>
        /// <param name="receiver">The list array.</param>
        /// <param name="name">The member's Python name.</param>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="passed">The values passed in the member's parameter slots (empty for a member that takes its
        ///     positions as one sequence).</param>
        /// <returns>The member's result.</returns>
        /// <exception cref="InvalidOperationException">An unknown member.</exception>
        private static object ListMember(NDListArray receiver, string name, Dictionary<string, JsonElement> p, NDArray[] ops,
                                         out object[] passed)
        {
            // transpose(*axes) / reshape(*shape) / resize(*shape) take their positions as a sequence; the others name each
            // parameter.
            var positional = p.TryGetValue("args", out var a)
                ? a.EnumerateArray().Select(s => ObjectPathValue(s, ops, receiver)).ToArray()
                : Array.Empty<object>();
            // A member named by its parameters takes positional and keyword arguments in Python's positions.
            var names = s_listMemberParams[name];
            var args = names.Length == 0 ? Array.Empty<object>() : ObjectPathArgs(p, names, ops, receiver);
            passed = args;
            // Whether the case GAVE the parameter at position i (positionally or by keyword): an explicit None differs from
            // an omitted argument where the default is not None (argsort's axis=-1).
            bool Given(int i) => i < positional.Length
                || (p.TryGetValue("kw", out var kw) && kw.TryGetProperty(names[i], out _));
            int? Axis(object v) => v is null ? null : checked((int)(long)v);
            int AxisOr(int i, int dflt) => Given(i) ? checked((int)(long)args[i]) : dflt;
            int? AxisOrNull(int i, int? dflt) => Given(i) ? Axis(args[i]) : dflt;
            char Order(object v, char dflt = 'C') => v is null ? dflt : ((string)v)[0];
            string Mode(object v) => (string)v ?? "raise";
            bool Flag(object v) => v is bool b && b;
            NDArray ArrayOrNull(object v) => v is null ? null : ToArray(v);
            // A where= list is NumPy's bool mask (an empty list included, which ToArray would read as int64), of the list's
            // own shape: [[]] is (1, 0).
            NDArray MaskOrNull(object v) => v is null ? null
                : v is Array { Length: 0 } empty
                    ? new NDArray(NPTypeCode.Boolean, new Shape(Enumerable.Range(0, empty.Rank).Select(d => (long)empty.GetLength(d)).ToArray()), false)
                    : ToArray(v);
            DType Dtype(object v) => v is null ? null : np.dtype((string)v);
            // A where= Python bool is NumPy's True / False (the C# literal's conversion); anything else is a mask.
            NDArray WhereOrNull(object v) => v is bool b ? b : MaskOrNull(v);
            // An index / count argument: one Python int is NumPy's 0-d array (the C# literal's conversion), a list the array.
            NDArray IndexArg(object v) => v is long i ? i : ToArray(v);
            switch (name)
            {
                case "to_device": return receiver.to_device((string)args[0], args[1]);
                case "transpose": return receiver.transpose(positional.Select(x => checked((int)(long)x)).ToArray());
                case "swapaxes": return receiver.swapaxes(checked((int)(long)args[0]), checked((int)(long)args[1]));
                case "reshape": return ListReshape(receiver, positional, p, ops);
                case "ravel": return receiver.ravel(Order(args[0]));
                case "flatten": return receiver.flatten(Order(args[0]));
                case "copy": return receiver.copy(Order(args[0]));
                case "squeeze": return receiver.squeeze(Axis(args[0]));
                case "view":
                    // view([dtype][, type]): an omitted argument is NumPy's "not given", an explicit None its None.
                    if (!Given(0) && !Given(1))
                        return receiver.view();
                    if (!Given(1))
                        return receiver.view(args[0]);
                    return Given(0) ? receiver.view(args[0], args[1]) : receiver.view(type: args[1]);
                case "take": return receiver.take(IndexArg(args[0]), Axis(args[1]), (NDListArray)args[2], Mode(args[3]));
                case "compress": return receiver.compress(ToArray(args[0]), Axis(args[1]), (NDListArray)args[2]);
                case "repeat": return receiver.repeat(IndexArg(args[0]), Axis(args[1]));
                case "diagonal":
                    return receiver.diagonal(args[0] is null ? 0 : checked((int)(long)args[0]),
                                             args[1] is null ? 0 : checked((int)(long)args[1]),
                                             args[2] is null ? 1 : checked((int)(long)args[2]));
                // ---- L-B / L-C ----
                case "all": return receiver.all(Axis(args[0]), ArrayOrNull(args[1]), Flag(args[2]), MaskOrNull(args[3]));
                case "any": return receiver.any(Axis(args[0]), ArrayOrNull(args[1]), Flag(args[2]), MaskOrNull(args[3]));
                case "nonzero": return receiver.nonzero();
                case "astype":
                    return receiver.astype(Dtype(args[0]), Order(args[1], 'K'), (string)args[2] ?? "unsafe",
                                           args[3] is not bool sub || sub, args[4] is not bool cp || cp);
                case "sort":
                    receiver.sort(AxisOr(0, -1), (string)args[1], args[2], args[3] as bool?);
                    return null;
                case "argsort": return receiver.argsort(AxisOrNull(0, -1), (string)args[1], args[2], args[3] as bool?);
                case "partition":
                    ListPartition(receiver, args[0], AxisOr(1, -1), (string)args[2] ?? "introselect", (string)args[3], arg: false);
                    return null;
                case "argpartition":
                    return ListPartition(receiver, args[0], AxisOr(1, -1), (string)args[2] ?? "introselect", (string)args[3], arg: true);
                case "searchsorted": return receiver.searchsorted(args[0], (string)args[1] ?? "left", ArrayOrNull(args[2]));
                case "argmax": return receiver.argmax(Axis(args[0]), ArrayOrNull(args[1]), Flag(args[2]));
                case "argmin": return receiver.argmin(Axis(args[0]), ArrayOrNull(args[1]), Flag(args[2]));
                case "max": return receiver.max(Axis(args[0]), (NDListArray)args[1], Flag(args[2]), args[3], MaskOrNull(args[4]));
                case "min": return receiver.min(Axis(args[0]), (NDListArray)args[1], Flag(args[2]), args[3], MaskOrNull(args[4]));
                case "sum":
                    return receiver.sum(Axis(args[0]), Dtype(args[1]), (NDListArray)args[2], Flag(args[3]), args[4], WhereOrNull(args[5]));
                case "prod":
                    return receiver.prod(Axis(args[0]), Dtype(args[1]), (NDListArray)args[2], Flag(args[3]), args[4], WhereOrNull(args[5]));
                case "cumsum": return receiver.cumsum(Axis(args[0]), Dtype(args[1]), (NDListArray)args[2]);
                case "cumprod": return receiver.cumprod(Axis(args[0]), Dtype(args[1]), (NDListArray)args[2]);
                case "trace":
                    return receiver.trace(args[0] is null ? 0 : checked((int)(long)args[0]), args[1] is null ? 0 : checked((int)(long)args[1]),
                                          args[2] is null ? 1 : checked((int)(long)args[2]), Dtype(args[3]), (NDListArray)args[4]);
                case "clip": return receiver.clip(args[0], args[1], (NDListArray)args[2], Dtype(args[3]), WhereOrNull(args[4]));
                case "mean":
                    return receiver.mean(Axis(args[0]), Dtype(args[1]), ArrayOrNull(args[2]), Flag(args[3]), WhereOrNull(args[4]));
                case "var":
                    return receiver.var(Axis(args[0]), Dtype(args[1]), ArrayOrNull(args[2]), args[3] is null ? 0 : checked((int)(long)args[3]),
                                        Flag(args[4]), WhereOrNull(args[5]), MeanArg(args[6]));
                case "std":
                    return receiver.std(Axis(args[0]), Dtype(args[1]), ArrayOrNull(args[2]), args[3] is null ? 0 : checked((int)(long)args[3]),
                                        Flag(args[4]), WhereOrNull(args[5]), MeanArg(args[6]));
                case "dot": return receiver.dot(args[0] is Array list ? ToArray(list) : args[0], (NDListArray)args[1]);
                case "round": return receiver.round(args[0] is null ? 0 : checked((int)(long)args[0]), (NDListArray)args[1]);
                case "conj": return receiver.conj((NDListArray)args[0]);
                case "conjugate": return receiver.conjugate((NDListArray)args[0]);
                case "fill":
                    receiver.fill(args[0]);
                    return null;
                case "put":
                    receiver.put(IndexArg(args[0]), (NDListArray)args[1], Mode(args[2]));
                    return null;
                case "resize":
                    ListResize(receiver, positional, p, ops);
                    return null;
                case "byteswap": return receiver.byteswap(Flag(args[0]));
                case "getfield": return receiver.getfield(Dtype(args[0]), args[1] is null ? 0 : checked((int)(long)args[1]));
                case "setfield":
                    receiver.setfield(args[0], Dtype(args[1]), args[2] is null ? 0 : checked((int)(long)args[2]));
                    return null;
                case "choose": return receiver.choose(args[0], ArrayOrNull(args[1]), (string)args[2] ?? "raise");
                default: throw new InvalidOperationException($"section strcharlist has no member '{name}'");
            }
        }

        /// <summary><c>partition</c> / <c>argpartition</c> with NumPy's kth forms: one integer, a list of them (an int64
        ///     array), or any other value as the array NumPy makes of it (a float kth is the TypeError).</summary>
        /// <param name="receiver">The list array.</param>
        /// <param name="kth">The decoded kth.</param>
        /// <param name="axis">The axis.</param>
        /// <param name="kind">The selection kind.</param>
        /// <param name="order">The order argument.</param>
        /// <param name="arg">True for argpartition.</param>
        /// <returns>argpartition's indices, or null (partition sorts in place).</returns>
        private static NDArray ListPartition(NDListArray receiver, object kth, int axis, string kind, string order, bool arg)
        {
            if (kth is long k)
            {
                if (arg)
                    return receiver.argpartition(k, axis, kind, order);
                receiver.partition(k, axis, kind, order);
                return null;
            }
            var array = kth is double d ? NDArray.Scalar(d) : ToArray(kth);
            if (arg)
                return receiver.argpartition(array, axis, kind, order);
            receiver.partition(array, axis, kind, order);
            return null;
        }

        /// <summary><c>resize(*shape, refcheck=...)</c> as a ported line spells it: the integers as a <see cref="Shape"/>.</summary>
        /// <param name="receiver">The list array.</param>
        /// <param name="positional">The positional dimensions.</param>
        /// <param name="p">The case parameters (a <c>refcheck</c> keyword).</param>
        /// <param name="ops">The reconstructed operands.</param>
        private static void ListResize(NDListArray receiver, object[] positional, Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            var dims = positional.Select(x => (long)x).ToArray();
            bool refcheck = true;
            if (p.TryGetValue("kw", out var kw) && kw.TryGetProperty("refcheck", out var rc))
                refcheck = (bool)ObjectPathValue(rc, ops);
            receiver.resize(new Shape(dims), refcheck);
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
            bool hasKw = p.TryGetValue("kw", out var kw);
            bool hasOrder = hasKw && kw.TryGetProperty("order", out _);
            bool hasCopy = hasKw && kw.TryGetProperty("copy", out _);
            if (!hasOrder && !hasCopy)
                return receiver.reshape(dims);
            char order = hasOrder ? ((string)ObjectPathValue(kw.GetProperty("order"), ops))[0] : 'C';
            bool? copy = hasCopy ? (bool?)ObjectPathValue(kw.GetProperty("copy"), ops) : null;
            return receiver.reshape(new Shape(dims), order, copy);
        }

        /// <summary>
        ///     var / std's <c>mean=</c> as a ported line passes it: a list array, an ndarray or a C# value is itself; the replay's
        ///     object array of Python numbers (a decoded list) is the array NumPy's subtraction makes of it.
        /// </summary>
        /// <param name="value">The decoded argument (null: not given).</param>
        /// <returns>The value to pass.</returns>
        private static object MeanArg(object value)
            => value is Array { Rank: >= 1 } list && list.GetType().GetElementType() == typeof(object)
               && list.Cast<object>().All(x => x is bool or long or double)
                ? ToArray(list)
                : value;

        /// <summary>An index / condition / count argument as the array a ported line passes: an <see cref="NDArray"/> is
        ///     itself, a typed C# array converts implicitly, and the replay's object array of Python bools / ints / floats (a
        ///     decoded list, <see cref="ManagedArray"/>) becomes the bool / int64 / float64 array of its rank that NumPy makes
        ///     of the list.</summary>
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
            bool floats = leaves.Any(x => x is double);
            NDArray flat = flags ? leaves.Select(x => (bool)x).ToArray()
                : floats ? leaves.Select(Convert.ToDouble).ToArray()
                : leaves.Select(Convert.ToInt64).ToArray();
            if (array.Rank == 1)
                return flat;
            var dims = Enumerable.Range(0, array.Rank).Select(d => (long)array.GetLength(d)).ToArray();
            return flat.reshape(dims);
        }
    }
}
