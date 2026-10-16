using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for sections <c>strcharops</c> / <c>strcharfn</c> (the <c>numpy.char</c> plan's phase 3):
    ///     <see cref="NDCharArray"/> BEHAVIOR — the operators with Python's dispatch (<c>cha_op</c>), the members phase 3
    ///     adds (<c>cha_call</c>) and the <c>np.*</c> functions with chararray operands (<c>cha_fn</c>). Each argument
    ///     arrives as the generator's spec (<c>_chx_arg</c>): an operand (plain, or its chararray view — one object per
    ///     operand), an element of that view (the stripped stand-in), a Python value, a dtype or a tuple.
    /// </summary>
    internal static partial class StringOracleOps
    {
        /// <summary>Replays one phase-3 op, or reports that <paramref name="op"/> is not one of them.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="result">The call's result: an array, a <see cref="SelfResult"/>, an element stand-in, a list of
        ///     items or a value.</param>
        /// <returns>True when <paramref name="op"/> is a phase-3 op.</returns>
        /// <exception cref="InvalidOperationException">An unknown operator, member, function or argument spec (a generator /
        ///     replay mismatch).</exception>
        private static bool TryInvokeCharBehavior(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            switch (op)
            {
                case "cha_op":
                    result = CharOperator(p, ops);
                    return true;
                case "cha_call":
                    result = CharCall(p, ops);
                    return true;
                case "cha_fn":
                    result = CharFunction(p, ops);
                    return true;
            }
            result = null;
            return false;
        }

        /// <summary>
        ///     The arguments of one phase-3 case: the generator's specs turned into the C# objects a ported NumPy line passes.
        ///     An operand's chararray view is made once, so a call that names it twice passes the same object (NumPy's
        ///     identity and memory sharing are compared against it).
        /// </summary>
        private sealed class BehaviorArgs
        {
            /// <summary>The reconstructed operands.</summary>
            private readonly NDArray[] _ops;

            /// <summary>The chararray view of each operand used so far, by index.</summary>
            private readonly Dictionary<int, NDCharArray> _chars = new();

            /// <summary>Wraps the case's operands.</summary>
            /// <param name="ops">The reconstructed operands.</param>
            internal BehaviorArgs(NDArray[] ops) => _ops = ops;

            /// <summary>Operand <paramref name="k"/> viewed as a chararray (NumPy's <c>view(np.char.chararray)</c>), made on
            ///     first use.</summary>
            /// <param name="k">The operand index.</param>
            /// <returns>The chararray.</returns>
            internal NDCharArray CharArray(int k)
            {
                if (!_chars.TryGetValue(k, out var c))
                    _chars[k] = c = (NDCharArray)_ops[k].view(typeof(NDCharArray));
                return c;
            }

            /// <summary>
            ///     One argument: <c>{"op": k}</c> the operand, <c>{"op": k, "cha": true}</c> its chararray view,
            ///     <c>{"elem": [k, [i...]]}</c> an element of that view (the stripped stand-in, NumPy's Python str / bytes),
            ///     <c>{"v": ...}</c> a Python value (<see cref="PythonValue"/>), <c>{"dt": s}</c> a dtype, <c>{"tuple": [...]}</c>
            ///     an <c>object[]</c>.
            /// </summary>
            /// <param name="spec">The spec.</param>
            /// <returns>The C# argument.</returns>
            /// <exception cref="InvalidOperationException">An unknown spec.</exception>
            internal object Value(JsonElement spec)
            {
                if (spec.TryGetProperty("op", out var k))
                    return spec.TryGetProperty("cha", out var cha) && cha.GetBoolean() ? CharArray(k.GetInt32()) : _ops[k.GetInt32()];
                if (spec.TryGetProperty("elem", out var elem))
                {
                    var chars = CharArray(elem[0].GetInt32());
                    var index = elem[1].EnumerateArray().Select(i => (object)i.GetInt32()).ToArray();
                    return chars[index];
                }
                if (spec.TryGetProperty("v", out var v))
                    return PythonValue(StringOracleCorpus.Decode(v));
                if (spec.TryGetProperty("dt", out var dt))
                    return np.dtype(dt.GetString());
                if (spec.TryGetProperty("tuple", out var tuple))
                    return tuple.EnumerateArray().Select(Value).ToArray();
                throw new InvalidOperationException($"the chararray behavior sections have no argument spec {spec}");
            }
        }

        /// <summary>
        ///     A decoded Python value as the C# value that stands for it: a <c>bytes</c> is the read-only 0-d <c>S</c> stand-in
        ///     (<see cref="NDCharArray"/>'s implicit conversion from <c>byte[]</c>, as a C# caller spells NumPy's bytes); a list
        ///     is the array NumPy's conversion makes of it (<see cref="CharModule.TextOperand"/>: text of its class, numbers);
        ///     a str, an int (<see cref="long"/>), a float, a bool and None stay themselves.
        /// </summary>
        /// <param name="value">The decoded value.</param>
        /// <returns>The C# value.</returns>
        private static object PythonValue(object value) => value switch
        {
            byte[] bytes => (NDCharArray)bytes,
            List<object> list => CharModule.TextOperand(ManagedArrayOrValue(list), "list"),
            _ => value,
        };

        /// <summary>A keyword argument of the case, or <c>null</c> with <paramref name="given"/> false when it is absent.</summary>
        /// <param name="args">The case's arguments.</param>
        /// <param name="p">The case parameters.</param>
        /// <param name="name">The keyword.</param>
        /// <param name="given">Whether the case passes it.</param>
        /// <returns>The value (null for Python's None too).</returns>
        private static object Kw(BehaviorArgs args, Dictionary<string, JsonElement> p, string name, out bool given)
        {
            given = p.TryGetValue("kw", out var kw) && kw.TryGetProperty(name, out var spec);
            return given ? args.Value(kw.GetProperty(name)) : null;
        }

        /// <summary>
        ///     A value as an <see cref="NDArray"/> operand, as a ported line passes it: an array as it is, a Python value as
        ///     the 0-d array its C# literal converts to — marked the way the implicit conversions mark it
        ///     (<see cref="NDArray.FromPythonOperand"/>: a weak Python int / float, a Python bool, a Python str), so a
        ///     function that reads the marks (NEP 50 promotion, <c>where=True</c>) sees NumPy's Python scalar.
        /// </summary>
        /// <param name="v">The value.</param>
        /// <returns>The array.</returns>
        /// <exception cref="InvalidOperationException">A value that is neither an array nor a Python scalar.</exception>
        private static NDArray Nd(object v) => v switch
        {
            NDArray a => a,
            string or long or double or bool => NDArray.FromPythonOperand(v),
            _ => throw new InvalidOperationException($"no array for argument {v}"),
        };

        /// <summary>An optional operand (<c>where=</c>, <c>out=</c>): <c>null</c> stays <c>null</c>, anything else is
        ///     <see cref="Nd"/>'s array (the C# literal <c>true</c> for NumPy's <c>where=True</c>).</summary>
        /// <param name="v">The value, or <c>null</c>.</param>
        /// <returns>The array, or <c>null</c>.</returns>
        private static NDArray OptNd(object v) => v is null ? null : Nd(v);

        /// <summary>An integer argument (a Python int).</summary>
        /// <param name="v">The value.</param>
        /// <returns>The int.</returns>
        private static int I(object v) => checked((int)(long)v);

        /// <summary>An optional axis: Python's None is <c>null</c>.</summary>
        /// <param name="v">The value.</param>
        /// <returns>The axis.</returns>
        private static int? Axis(object v) => v is null ? null : I(v);

        /// <summary>A list or tuple of Python ints as an <c>int[]</c> (an index array argument read back to its values).</summary>
        /// <param name="v">The value: an int64 array (a converted list) or an <c>object[]</c> (a tuple).</param>
        /// <returns>The ints.</returns>
        private static int[] Ints(object v) => v switch
        {
            NDArray a => a.ToArray<long>().Select(x => checked((int)x)).ToArray(),
            object[] t => t.Select(I).ToArray(),
            long l => new[] { checked((int)l) },
            _ => throw new InvalidOperationException($"no int list for argument {v}"),
        };

        /// <summary>A list or tuple of Python ints as a <c>long[]</c> (a shape).</summary>
        /// <param name="v">The value.</param>
        /// <returns>The longs.</returns>
        private static long[] Longs(object v) => v switch
        {
            NDArray a when a.size == 0 => Array.Empty<long>(),
            NDArray a => a.ToArray<long>(),
            object[] t => t.Select(x => (long)x).ToArray(),
            long l => new[] { l },
            _ => throw new InvalidOperationException($"no shape for argument {v}"),
        };

        /// <summary>A tuple axis (NumPy's <c>axis=(0, 1)</c>, decoded as a converted list or an <c>object[]</c>), or null for an
        ///     int or absent axis.</summary>
        /// <param name="v">The axis value.</param>
        /// <returns>The axes, or null.</returns>
        private static int[] TupleAxes(object v) => v is NDArray or object[] ? Ints(v) : null;

        /// <summary>A sequence result as the items list the comparator reads.</summary>
        /// <param name="arrays">The arrays.</param>
        /// <returns>The items.</returns>
        private static object[] Items(IEnumerable<NDArray> arrays) => arrays.Cast<object>().ToArray();

        /// <summary>
        ///     <c>cha_op</c>: one Python operator over the two arguments, bound as C# binds it for their runtime types
        ///     (<c>dynamic</c>): a chararray operand selects <see cref="NDCharArray"/>'s operators, a plain array
        ///     <see cref="NDArray"/>'s, a C# string / number / null the forms a ported line would spell.
        /// </summary>
        /// <param name="p">The case parameters: <c>sym</c>, <c>args</c>.</param>
        /// <param name="ops">The operands.</param>
        /// <returns>The result.</returns>
        /// <exception cref="InvalidOperationException">An unknown operator.</exception>
        private static object CharOperator(Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            var args = new BehaviorArgs(ops);
            var a = p["args"].EnumerateArray().Select(args.Value).ToArray();
            dynamic l = a[0], r = a[1];
            return Str(p, "sym") switch
            {
                "==" => l == r,
                "!=" => l != r,
                "<" => l < r,
                "<=" => l <= r,
                ">" => l > r,
                ">=" => l >= r,
                "+" => l + r,
                "*" => l * r,
                var s => throw new InvalidOperationException($"the strcharops section has no operator '{s}'"),
            };
        }

        /// <summary><c>cha_call</c>: one member of operand <c>recv</c>'s chararray view, returned with that chararray.</summary>
        /// <param name="p">The case parameters: <c>recv</c>, <c>name</c>, <c>args</c>, <c>kw</c>.</param>
        /// <param name="ops">The operands.</param>
        /// <returns>The result paired with the chararray (<c>sort</c>: the sorted chararray itself).</returns>
        /// <exception cref="InvalidOperationException">An unknown member.</exception>
        private static SelfResult CharCall(Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            var args = new BehaviorArgs(ops);
            var c = args.CharArray(p["recv"].GetInt32());
            var a = p["args"].EnumerateArray().Select(args.Value).ToArray();
            object axis = Kw(args, p, "axis", out bool hasAxis);
            object @out = Kw(args, p, "out", out bool hasOut);
            object keep = Kw(args, p, "keepdims", out bool hasKeep);
            object kind = Kw(args, p, "kind", out bool hasKind);
            bool keepdims = hasKeep && (bool)keep;
            object r;
            switch (Str(p, "name"))
            {
                case "argmax":
                case "argmin":
                {
                    bool max = Str(p, "name") == "argmax";
                    if (!hasAxis && !hasOut && !hasKeep)
                        r = max ? c.argmax() : c.argmin();
                    else if (hasAxis && axis is not null && !hasOut)
                        r = max ? c.argmax(I(axis), keepdims) : c.argmin(I(axis), keepdims);
                    else
                        r = max ? c.argmax(Axis(axis), (NDArray)@out, keepdims) : c.argmin(Axis(axis), (NDArray)@out, keepdims);
                    break;
                }
                case "argpartition":
                {
                    int? ax = hasAxis ? Axis(axis) : -1;
                    string k = hasKind ? (string)kind : "introselect";
                    r = a[0] is long kth ? c.argpartition(checked((int)kth), ax, k) : c.argpartition(Ints(a[0]), ax, k);
                    break;
                }
                case "partition":
                {
                    if (!hasAxis)
                    {
                        // chararray.partition(sep): the string partition.
                        r = c.partition(a[0]);
                        break;
                    }
                    if (a[0] is long kth)
                        c.partition(checked((int)kth), Axis(axis));
                    else
                        c.partition(Ints(a[0]), Axis(axis));
                    r = c;
                    break;
                }
                case "sort":
                    c.sort(hasAxis ? Axis(axis) : -1, hasKind ? (string)kind : null);
                    r = c;
                    break;
                case "argsort":
                    r = c.argsort(hasAxis ? Axis(axis) : -1, hasKind ? (string)kind : null);
                    break;
                case "searchsorted":
                {
                    object side = Kw(args, p, "side", out bool hasSide);
                    object sorter = Kw(args, p, "sorter", out _);
                    r = c.searchsorted(Nd(a[0]), hasSide ? (string)side : "left", (NDArray)sorter);
                    break;
                }
                case "nonzero":
                    r = Items(c.nonzero());
                    break;
                default:
                    throw new InvalidOperationException($"the strcharops section has no member '{Str(p, "name")}'");
            }
            return new SelfResult(r, c);
        }

        /// <summary>
        ///     <c>cha_fn</c>: one <c>np.*</c> function with the case's arguments, through the overload a ported NumPy line
        ///     binds; returned with the first argument when that is a chararray (NumPy's identity with its operand).
        /// </summary>
        /// <param name="p">The case parameters: <c>fn</c>, <c>args</c>, <c>kw</c>.</param>
        /// <param name="ops">The operands.</param>
        /// <returns>The result (an array, a list of items, a count), paired with the first chararray argument.</returns>
        /// <exception cref="InvalidOperationException">An unknown function.</exception>
        private static object CharFunction(Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            var args = new BehaviorArgs(ops);
            var a = p["args"].EnumerateArray().Select(args.Value).ToArray();
            object kw(string name, out bool given) => Kw(args, p, name, out given);
            object axis = kw("axis", out bool hasAxis);
            object @out = kw("out", out bool hasOut);
            object where = kw("where", out bool hasWhere);
            object keep = kw("keepdims", out bool hasKeep);
            object method = kw("method", out bool hasMethod);
            object overwrite = kw("overwrite_input", out bool hasOverwrite);
            object dtype = kw("dtype", out _);
            bool keepdims = hasKeep && (bool)keep;
            NDArray x = a.Length > 0 && a[0] is NDArray first ? first : null;
            object r = Str(p, "fn") switch
            {
                "reshape" => np.reshape(x, Longs(a[1])),
                "ravel" => hasOrder(out var o) ? np.ravel(x, o) : np.ravel(x),
                "transpose" => np.transpose(x),
                "permute_dims" => np.permute_dims(x),
                "matrix_transpose" => np.matrix_transpose(x),
                "squeeze" => np.squeeze(x),
                "expand_dims" => np.expand_dims(x, I(a[1])),
                "atleast_1d" => np.atleast_1d(x),
                "atleast_2d" => np.atleast_2d(Nd(a[0])),
                "atleast_3d" => np.atleast_3d(x),
                "flip" => np.flip(x),
                "fliplr" => np.fliplr(x),
                "flipud" => np.flipud(x),
                "rot90" => np.rot90(x),
                "swapaxes" => np.swapaxes(x, I(a[1]), I(a[2])),
                "moveaxis" => np.moveaxis(x, I(a[1]), I(a[2])),
                "rollaxis" => np.rollaxis(x, I(a[1])),
                "roll" => np.roll(x, I(a[1]), hasAxis ? Axis(axis) : null),
                "tile" => np.tile(x, I(a[1])),
                "repeat" => np.repeat(x, I(a[1]), hasAxis ? Axis(axis) : null),
                "sort" => np.sort(x, hasAxis ? Axis(axis) : -1),
                "argsort" => np.argsort(x, hasAxis ? Axis(axis) : -1),
                "require" => hasRequirements(out var req) ? np.require(x, null, req) : np.require(x),
                "asanyarray" => np.asanyarray(x),
                "asarray" => np.asarray(x),
                "ascontiguousarray" => np.ascontiguousarray(x),
                "asfortranarray" => np.asfortranarray(x),
                "copy" => kw("subok", out bool hasSubok) is bool subok && hasSubok ? np.copy(x, 'K', subok) : np.copy(x),
                "array" => np.array(x),
                "diagflat" => np.diagflat(x),
                "diag" => np.diag(x),
                "diagonal" => np.diagonal(x),
                "take" => a[1] is long index
                    ? np.take(x, index, hasAxis ? Axis(axis) : null)
                    : np.take(x, (NDArray)a[1], hasAxis ? Axis(axis) : null),
                "take_along_axis" => np.take_along_axis(x, (NDArray)a[1], Axis(axis)),
                "compress" => np.compress(Nd(a[0]), (NDArray)a[1], hasAxis ? Axis(axis) : null),
                "extract" => np.extract((NDArray)a[0], (NDArray)a[1]),
                "trim_zeros" => np.trim_zeros(x),
                "insert" => np.insert(x, I(a[1]), a[2]),
                "delete" => np.delete(x, I(a[1])),
                "split" => Items(np.split(x, I(a[1]))),
                "array_split" => Items(np.array_split(x, I(a[1]))),
                "hsplit" => Items(np.hsplit(x, I(a[1]))),
                "vsplit" => Items(np.vsplit(x, I(a[1]))),
                "unstack" => Items(np.unstack(x, hasAxis ? I(axis) : 0)),
                "unique" => UniqueOf(x, args, p, hasAxis ? Axis(axis) : null),
                "unique_counts" => Items(np.unique_counts(x).ToArray()),
                "unique_inverse" => Items(np.unique_inverse(x).ToArray()),
                "unique_all" => Items(np.unique_all(x).ToArray()),
                "empty_like" => np.empty_like(x, (DType)dtype),
                "zeros_like" => np.zeros_like(x, (DType)dtype),
                "ones_like" => np.ones_like(x, (DType)dtype),
                "full_like" => np.full_like(x, a[1] is NDCharArray standIn ? standIn.item() : a[1], (DType)dtype),
                "concatenate" => np.concatenate(((object[])a[0]).Cast<NDArray>().ToArray(), Axis(axis)),
                "stack" => np.stack(((object[])a[0]).Cast<NDArray>().ToArray()),
                "hstack" => np.hstack(((object[])a[0]).Cast<NDArray>().ToArray()),
                "vstack" => np.vstack(((object[])a[0]).Cast<NDArray>().ToArray()),
                "dstack" => np.dstack(((object[])a[0]).Cast<NDArray>().ToArray()),
                "column_stack" => np.column_stack(((object[])a[0]).Cast<NDArray>().ToArray()),
                "append" => hasAxis ? np.append(x, Nd(a[1]), Axis(axis)) : np.append(x, Nd(a[1])),
                "tril" => np.tril(x),
                "triu" => np.triu(x),
                "pad" => np.pad(x, I(a[1]), (string)kw("mode", out _)),
                "broadcast_arrays" => Items(np.broadcast_arrays(new[] { Nd(a[0]), Nd(a[1]) })),
                "nonzero" => Items(np.nonzero(x)),
                "argwhere" => np.argwhere(x),
                "flatnonzero" => np.flatnonzero(x),
                "lexsort" => np.lexsort(x),
                "choose" => np.choose(Nd(a[0]), ((object[])a[1]).Cast<NDArray>().ToArray()),
                "meshgrid" => Items(np.meshgrid(Nd(a[0]), Nd(a[1]),
                    kw("indexing", out bool hasIndexing) is string indexing && hasIndexing ? indexing : "xy",
                    kw("sparse", out bool hasSparse) is bool sparse && hasSparse && sparse,
                    !(kw("copy", out bool hasCopy) is bool copy && hasCopy) || copy).ToArray()),
                "where" => np.where(Nd(a[0]), (NDArray)a[1], (NDArray)a[2]),
                "resize" => np.resize(x, new Shape(Longs(a[1]))),
                "broadcast_to" => np.broadcast_to(x, new Shape(Longs(a[1]))),
                "count_nonzero" => np.count_nonzero(x),
                "union1d" => np.union1d((NDArray)a[0], (NDArray)a[1]),
                "intersect1d" => np.intersect1d((NDArray)a[0], (NDArray)a[1]),
                "setxor1d" => np.setxor1d((NDArray)a[0], (NDArray)a[1]),
                "isin" => np.isin((NDArray)a[0], (NDArray)a[1]),
                "setdiff1d" => np.setdiff1d((NDArray)a[0], (NDArray)a[1], kw("assume_unique", out bool hasAu) is bool au && hasAu && au),
                "searchsorted" => np.searchsorted(x, Nd(a[1])),
                "argmax" => IndexFunction(true, false, x, hasAxis, axis, hasOut, @out, keepdims, hasKeep),
                "argmin" => IndexFunction(false, false, x, hasAxis, axis, hasOut, @out, keepdims, hasKeep),
                "nanargmax" => IndexFunction(true, true, x, hasAxis, axis, hasOut, @out, keepdims, hasKeep),
                "nanargmin" => IndexFunction(false, true, x, hasAxis, axis, hasOut, @out, keepdims, hasKeep),
                "argpartition" => np.argpartition(x, I(a[1]), hasAxis ? Axis(axis) : -1),
                "partition" => np.partition(x, I(a[1]), hasAxis ? Axis(axis) : -1),
                "median" => TupleAxes(axis) is { } axes
                    ? np.median(x, axes, (NDArray)@out, hasOverwrite && (bool)overwrite, keepdims)
                    : np.median(x, hasAxis ? Axis(axis) : null, (NDArray)@out, hasOverwrite && (bool)overwrite, keepdims),
                "quantile" => QuantileOf(false, x, a[1], axis, hasAxis, hasMethod ? (string)method : "linear", keepdims),
                "percentile" => QuantileOf(true, x, a[1], axis, hasAxis, hasMethod ? (string)method : "linear", keepdims),
                "add" => np.add(Nd(a[0]), Nd(a[1]), (NDArray)@out, OptNd(where)),
                "multiply" => np.multiply(Nd(a[0]), Nd(a[1]), (NDArray)@out, OptNd(where)),
                "equal" => np.equal(Nd(a[0]), Nd(a[1]), (NDArray)@out, OptNd(where)),
                "less" => np.less(Nd(a[0]), Nd(a[1]), (NDArray)@out, OptNd(where)),
                "logical_and" => np.logical_and(Nd(a[0]), Nd(a[1]), (NDArray)@out, OptNd(where)),
                "logical_or" => np.logical_or(Nd(a[0]), Nd(a[1]), (NDArray)@out, OptNd(where)),
                "logical_xor" => np.logical_xor(Nd(a[0]), Nd(a[1]), (NDArray)@out, OptNd(where)),
                "strings.add" => np.strings.add(Nd(a[0]), Nd(a[1])),
                "strings.multiply" => np.strings.multiply(Nd(a[0]), Nd(a[1])),
                "char.add" => np.@char.add(a[0], a[1]),
                "char.multiply" => np.@char.multiply(a[0], a[1]),
                var f => throw new InvalidOperationException($"the strcharfn section has no function '{f}'"),
            };
            return x is NDCharArray chars && !chars.IsScalarStandIn ? new SelfResult(r, chars) : r;

            bool hasOrder(out char order)
            {
                object v = kw("order", out bool given);
                order = given ? ((string)v)[0] : 'C';
                return given;
            }

            bool hasRequirements(out string requirements)
            {
                object v = kw("requirements", out bool given);
                requirements = given ? (string)v : null;
                return given;
            }
        }

        /// <summary>
        ///     <c>np.unique</c> with the case's flags: NumPy's bare array without flags (<see cref="np.UniqueResult.values"/>),
        ///     its tuple with any (the present outputs, as items).
        /// </summary>
        /// <param name="x">The operand.</param>
        /// <param name="args">The case's arguments.</param>
        /// <param name="p">The case parameters.</param>
        /// <param name="axis">The axis, or null.</param>
        /// <returns>The values, or the items.</returns>
        private static object UniqueOf(NDArray x, BehaviorArgs args, Dictionary<string, JsonElement> p, int? axis)
        {
            bool flag(string name) => Kw(args, p, name, out bool given) is bool b && given && b;
            bool ri = flag("return_index"), rv = flag("return_inverse"), rc = flag("return_counts");
            var u = np.unique(x, ri, rv, rc, axis);
            // Both arms as object: an object[] converts implicitly to NDArray (the Array conversion), which a typed conditional would pick.
            return ri || rv || rc ? (object)Items(u.ToArray()) : u.values;
        }

        /// <summary>The index functions with the case's keywords, through the overload each keyword set binds.</summary>
        /// <param name="max">True for argmax / nanargmax.</param>
        /// <param name="nan">True for the nan forms.</param>
        /// <param name="x">The operand.</param>
        /// <param name="hasAxis">Whether the case passes an axis.</param>
        /// <param name="axis">The axis.</param>
        /// <param name="hasOut">Whether the case passes out=.</param>
        /// <param name="out">The out array.</param>
        /// <param name="keepdims">The keepdims flag.</param>
        /// <param name="hasKeep">Whether the case passes keepdims.</param>
        /// <returns>The index (a <see cref="long"/> for the flat form) or out=.</returns>
        private static object IndexFunction(bool max, bool nan, NDArray x, bool hasAxis, object axis, bool hasOut, object @out,
                                            bool keepdims, bool hasKeep)
        {
            if (!hasAxis && !hasOut && !hasKeep)
                return nan ? (max ? np.nanargmax(x) : np.nanargmin(x)) : (max ? np.argmax(x) : np.argmin(x));
            if (nan)
                return max ? np.nanargmax(x, Axis(axis), (NDArray)@out, keepdims) : np.nanargmin(x, Axis(axis), (NDArray)@out, keepdims);
            return max ? np.argmax(x, Axis(axis), (NDArray)@out, keepdims) : np.argmin(x, Axis(axis), (NDArray)@out, keepdims);
        }

        /// <summary><c>np.quantile</c> / <c>np.percentile</c> with a scalar or list q and an int or tuple axis.</summary>
        /// <param name="percentile">True for np.percentile.</param>
        /// <param name="x">The operand.</param>
        /// <param name="q">The q argument (a double, a Python int, or a converted list).</param>
        /// <param name="axis">The axis: null, an int or a tuple.</param>
        /// <param name="hasAxis">Whether the case passes an axis.</param>
        /// <param name="method">The method.</param>
        /// <param name="keepdims">The keepdims flag.</param>
        /// <returns>The result.</returns>
        private static NDArray QuantileOf(bool percentile, NDArray x, object q, object axis, bool hasAxis, string method,
                                          bool keepdims)
        {
            if (TupleAxes(axis) is { } ax)
            {
                double qs = Convert.ToDouble(q);
                return percentile ? np.percentile(x, qs, ax, null, false, method, keepdims) : np.quantile(x, qs, ax, null, false, method, keepdims);
            }
            int? one = hasAxis ? Axis(axis) : null;
            if (q is NDArray list)
            {
                double[] qs = list.ToArray<double>();
                return percentile ? np.percentile(x, qs, one, null, false, method, keepdims) : np.quantile(x, qs, one, null, false, method, keepdims);
            }
            double qd = Convert.ToDouble(q);
            return percentile ? np.percentile(x, qd, one, null, false, method, keepdims) : np.quantile(x, qd, one, null, false, method, keepdims);
        }
    }
}
