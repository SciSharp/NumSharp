using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for section <c>strchararray</c> (<c>gen_strchararray</c>, the <c>numpy.char</c> plan's
    ///     phase 2): <see cref="NDCharArray"/> as data — <c>np.@char.chararray</c>, <c>np.@char.array</c> /
    ///     <c>asarray</c>, element access through every indexer spelling, iteration, the members that keep the class,
    ///     <c>view(type)</c>, <c>repr</c> / <c>str</c> and the flat iterator. Each op makes the call a ported NumPy line
    ///     makes on a chararray built the way the generator built it (<c>operand.view(np.char.chararray)</c>, then
    ///     optionally <c>.view(np.uint32)</c>: <see cref="CharArrayReceiver"/>).
    /// </summary>
    internal static partial class StringOracleOps
    {
        /// <summary>
        ///     A result together with the object the call was made on, so the replay compares NumPy's
        ///     <c>result is receiver</c> (an <c>"self"</c> expectation): <see cref="StringOracleTests"/> unwraps it before
        ///     comparing the result.
        /// </summary>
        /// <param name="Result">The call's result.</param>
        /// <param name="Receiver">The object the call was made on (null: no identity to compare).</param>
        internal sealed record SelfResult(object Result, object Receiver);

        /// <summary>Replays one section-strchararray op, or reports that <paramref name="op"/> is not one of them.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="result">The call's result: an array, a <see cref="SelfResult"/>, an element, a list of items, a
        ///     Python value or a printed string.</param>
        /// <returns>True when <paramref name="op"/> is a section-strchararray op.</returns>
        /// <exception cref="InvalidOperationException">An unknown member, form or argument spec (a generator / replay
        ///     mismatch).</exception>
        private static bool TryInvokeChararray(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            switch (op)
            {
                case "cha_new":
                    result = CharArrayNew(p, ops);
                    return true;
                case "cha_array":
                case "cha_asarray":
                    result = CharArrayFrom(op == "cha_asarray", p, ops);
                    return true;
                case "cha_item":
                    result = CharArrayItem(p, ops);
                    return true;
                case "cha_iter":
                {
                    // NumPy's list(c): the chararray's own enumerator (stripped elements of a 1-d chararray, chararray rows
                    // of an N-d one, the 0-d TypeError raised before the first item).
                    var items = new List<object>();
                    foreach (var x in CharArrayReceiver(p, ops[0]))
                        items.Add(x);
                    result = items.ToArray();
                    return true;
                }
                case "cha_member":
                    result = CharArrayMember(p, ops[0]);
                    return true;
                case "cha_viewtype":
                    result = CharArrayViewType(p, ops[0]);
                    return true;
                case "cha_repr":
                    result = CharArrayReceiver(p, ops[0]).ToString(true);
                    return true;
                case "cha_str":
                    result = CharArrayReceiver(p, ops[0]).ToString(false);
                    return true;
            }
            result = null;
            return false;
        }

        /// <summary>
        ///     The chararray a case's call is made on: the operand viewed as <see cref="NDCharArray"/> (NumPy's
        ///     <c>a.view(np.char.chararray)</c>), then — when <c>params.recv</c> names a dtype — viewed as that dtype, which
        ///     keeps the class (the finalize rule reads the string dtype the array has).
        /// </summary>
        /// <param name="p">The case parameters.</param>
        /// <param name="a">The operand.</param>
        /// <returns>The chararray.</returns>
        private static NDCharArray CharArrayReceiver(Dictionary<string, JsonElement> p, NDArray a)
        {
            var c = (NDCharArray)a.view(typeof(NDCharArray));
            if (p.TryGetValue("recv", out var recv) && recv.ValueKind == JsonValueKind.String)
                c = c.view(np.dtype(recv.GetString()));
            return c;
        }

        /// <summary>NumPy's <c>np.char.chararray(shape, itemsize, unicode, buffer, offset, strides, order)</c>, the buffer
        ///     as the C# object that stands for NumPy's: a <see cref="string"/> (a filler), <c>ReadOnlyMemory&lt;byte&gt;</c>
        ///     (Python <c>bytes</c>, read-only), <c>byte[]</c> (a <c>bytearray</c>), an operand, or a list.</summary>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The chararray.</returns>
        /// <exception cref="InvalidOperationException">An unknown buffer spec.</exception>
        private static NDCharArray CharArrayNew(Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            var shape = new Shape(Longs(p["shape"]));
            long itemsize = p.TryGetValue("itemsize", out var it) ? it.GetInt64() : 1;
            bool unicode = p.TryGetValue("unicode", out var u) && u.GetBoolean();
            char order = p.TryGetValue("order", out var o) ? o.GetString()[0] : 'C';
            long offset = p.TryGetValue("offset", out var off) ? off.GetInt64() : 0;
            long[] strides = p.TryGetValue("strides", out var st) && st.ValueKind == JsonValueKind.Array ? Longs(st) : null;
            object buffer = null;
            if (p.TryGetValue("buffer", out var b))
            {
                if (b.TryGetProperty("str", out var s))
                    buffer = (string)StringOracleCorpus.Decode(s);
                else if (b.TryGetProperty("bytes", out var hex))
                    buffer = new ReadOnlyMemory<byte>(Convert.FromHexString(hex.GetString()));
                else if (b.TryGetProperty("bytearray", out var hex2))
                    buffer = Convert.FromHexString(hex2.GetString());
                else if (b.TryGetProperty("op", out var k))
                    buffer = ops[k.GetInt32()];
                else if (b.TryGetProperty("list", out var list))
                    buffer = ManagedArray(StringOracleCorpus.Decode(list));
                else
                    throw new InvalidOperationException($"the strchararray section has no buffer spec {b}");
            }
            return np.@char.chararray(shape, itemsize, unicode, buffer, offset, strides, order);
        }

        /// <summary>NumPy's <c>np.char.array(obj, ...)</c> / <c>np.char.asarray(obj, ...)</c> with the keywords the case gives
        ///     (an omitted one is NumPy's default), returned with <c>obj</c> for the identity check.</summary>
        /// <param name="asarray">True for <c>np.char.asarray</c>.</param>
        /// <param name="p">The case parameters: <c>obj</c> and the keywords.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The result paired with <c>obj</c>.</returns>
        private static SelfResult CharArrayFrom(bool asarray, Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            object obj = CharArrayObject(p["obj"], ops);
            long? itemsize = p.TryGetValue("itemsize", out var it) ? it.GetInt64() : null;
            bool? unicode = p.TryGetValue("unicode", out var u) ? u.GetBoolean() : null;
            char? order = p.TryGetValue("order", out var o) ? o.GetString()[0] : null;
            NDCharArray r = asarray
                ? np.@char.asarray(obj, itemsize, unicode, order)
                : np.@char.array(obj, itemsize, !p.TryGetValue("copy", out var cp) || cp.GetBoolean(), unicode, order);
            return new SelfResult(r, obj);
        }

        /// <summary>The <c>obj</c> argument of np.char.array: an operand (<c>{"op": k}</c>, optionally <c>"as":
        ///     "chararray"</c> — viewed as one — or <c>"as": "T"</c> — transposed), or a Python value
        ///     (<c>{"v": ...}</c>, see <see cref="CharArrayValue"/>).</summary>
        /// <param name="spec">The spec.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The C# argument.</returns>
        /// <exception cref="InvalidOperationException">An unknown spec.</exception>
        private static object CharArrayObject(JsonElement spec, NDArray[] ops)
        {
            if (spec.TryGetProperty("op", out var k))
            {
                var a = ops[k.GetInt32()];
                string how = spec.TryGetProperty("as", out var h) ? h.GetString() : null;
                return how switch
                {
                    null => a,
                    "chararray" => a.view(typeof(NDCharArray)),
                    "T" => a.T,
                    _ => throw new InvalidOperationException($"the strchararray section has no operand form '{how}'"),
                };
            }
            if (spec.TryGetProperty("v", out var v))
                return ManagedArrayOrValue(CharArrayValue(v));
            throw new InvalidOperationException($"the strchararray section has no obj spec {spec}");
        }

        /// <summary>A Python value as the generator's <c>_cha_enc</c> wrote it: <see cref="StringOracleCorpus.Decode"/>'s
        ///     encodings plus <c>{"complex": [re, im]}</c> (a <see cref="Complex"/>) and <c>{"bigint": "…"}</c> (a Python
        ///     int past int64, a <see cref="BigInteger"/>); a list stays a list of decoded items.</summary>
        /// <param name="e">The encoded value.</param>
        /// <returns>The decoded value.</returns>
        private static object CharArrayValue(JsonElement e)
        {
            if (e.ValueKind == JsonValueKind.Array)
                return e.EnumerateArray().Select(CharArrayValue).ToList();
            if (e.ValueKind == JsonValueKind.Object)
            {
                if (e.TryGetProperty("complex", out var c))
                    return new Complex(JsonFloat(c[0]), JsonFloat(c[1]));
                if (e.TryGetProperty("bigint", out var big))
                    return BigInteger.Parse(big.GetString(), CultureInfo.InvariantCulture);
            }
            return StringOracleCorpus.Decode(e);
        }

        /// <summary>
        ///     NumPy's <c>c[index]</c> through the indexer the case names: <c>objects</c> — the general
        ///     <c>this[params object[]]</c> (ints, <see cref="Slice"/>s, bools, index arrays, 0-d index arrays, masks);
        ///     <c>notation</c> — the slice-notation string (<c>"0, ::-1"</c>); <c>slices</c> — <c>this[params Slice[]]</c>.
        /// </summary>
        /// <param name="p">The case parameters: <c>index</c> (the generator's entries), <c>form</c>, <c>recv</c>.</param>
        /// <param name="ops">The operands (the chararray's operand first, then any mask an entry names).</param>
        /// <returns>The element stand-in or the chararray.</returns>
        /// <exception cref="InvalidOperationException">An unknown entry or form.</exception>
        private static NDCharArray CharArrayItem(Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            var c = CharArrayReceiver(p, ops[0]);
            var entries = p["index"].EnumerateArray().ToArray();
            switch (Str(p, "form"))
            {
                case "notation":
                    return c[string.Join(", ", entries.Select(NotationOf))];
                case "slices":
                    return c[entries.Select(SliceOf).ToArray()];
                case "objects":
                    return c[entries.Select(e => IndexObjectOf(e, ops)).ToArray()];
                default:
                    throw new InvalidOperationException($"the strchararray section has no index form '{Str(p, "form")}'");
            }
        }

        /// <summary>One basic index entry as slice notation (<c>{"i": 2}</c> → <c>"2"</c>, <c>{"s": [1, null, -1]}</c> →
        ///     <c>"1::-1"</c>, <c>"..."</c>, <c>"np.newaxis"</c>).</summary>
        /// <param name="e">The entry.</param>
        /// <returns>The notation.</returns>
        /// <exception cref="InvalidOperationException">An entry slice notation cannot spell.</exception>
        private static string NotationOf(JsonElement e)
        {
            if (e.TryGetProperty("i", out var i))
                return i.GetInt64().ToString(CultureInfo.InvariantCulture);
            if (e.TryGetProperty("s", out var s))
                return string.Join(":", s.EnumerateArray().Select(x => x.ValueKind == JsonValueKind.Null ? "" : x.GetInt64().ToString(CultureInfo.InvariantCulture)));
            if (e.TryGetProperty("e", out _))
                return "...";
            if (e.TryGetProperty("n", out _))
                return "np.newaxis";
            throw new InvalidOperationException($"no slice notation for index entry {e}");
        }

        /// <summary>One basic index entry as a <see cref="Slice"/>.</summary>
        /// <param name="e">The entry.</param>
        /// <returns>The slice.</returns>
        /// <exception cref="InvalidOperationException">An entry a Slice cannot hold.</exception>
        private static Slice SliceOf(JsonElement e)
        {
            if (e.TryGetProperty("i", out var i))
                return Slice.Index(i.GetInt64());
            if (e.TryGetProperty("s", out var s))
            {
                var f = s.EnumerateArray().Select(x => x.ValueKind == JsonValueKind.Null ? (long?)null : x.GetInt64()).ToArray();
                return new Slice(f[0], f[1], f[2] ?? 1);
            }
            if (e.TryGetProperty("e", out _))
                return Slice.Ellipsis;
            if (e.TryGetProperty("n", out _))
                return Slice.NewAxis;
            throw new InvalidOperationException($"no Slice for index entry {e}");
        }

        /// <summary>One index entry as the object the general indexer takes: an int, a <see cref="Slice"/>,
        ///     <see cref="Slice.Ellipsis"/> / <see cref="Slice.NewAxis"/>, a bool, an int64 index array, a 0-d int64 array,
        ///     or the mask operand.</summary>
        /// <param name="e">The entry.</param>
        /// <param name="ops">The operands (a mask entry names one).</param>
        /// <returns>The index object.</returns>
        /// <exception cref="InvalidOperationException">An unknown entry.</exception>
        private static object IndexObjectOf(JsonElement e, NDArray[] ops)
        {
            if (e.TryGetProperty("i", out var i))
                return i.GetInt32();
            if (e.TryGetProperty("s", out _) || e.TryGetProperty("e", out _) || e.TryGetProperty("n", out _))
                return SliceOf(e);
            if (e.TryGetProperty("b", out var b))
                return b.GetBoolean();
            if (e.TryGetProperty("a", out var a))
                return np.array(a.EnumerateArray().Select(x => x.GetInt64()).ToArray());
            if (e.TryGetProperty("z", out var z))
                return NDArray.Scalar(z.GetInt64());
            if (e.TryGetProperty("m", out var m))
                return ops[m.GetInt32()];
            throw new InvalidOperationException($"unknown index entry {e}");
        }

        /// <summary>
        ///     One member call of section E on the case's chararray (see the generator's <c>CHA_MEMBERS</c>), returned with
        ///     the chararray for NumPy's identities. NumPy's <c>c.flat</c> is NumSharp's <c>c.flatiter</c>.
        /// </summary>
        /// <param name="p">The case parameters: <c>name</c>, <c>args</c> (positional, then an optional <c>{"kw": {...}}</c>),
        ///     <c>recv</c>.</param>
        /// <param name="a">The operand.</param>
        /// <returns>The result paired with the chararray.</returns>
        /// <exception cref="InvalidOperationException">An unknown member.</exception>
        private static SelfResult CharArrayMember(Dictionary<string, JsonElement> p, NDArray a)
        {
            var c = CharArrayReceiver(p, a);
            var all = p["args"].EnumerateArray().ToArray();
            var kw = all.Length > 0 && all[^1].ValueKind == JsonValueKind.Object && all[^1].TryGetProperty("kw", out var k)
                ? k
                : (JsonElement?)null;
            var args = kw is null ? all : all[..^1];
            int? axis = kw is JsonElement kws && kws.TryGetProperty("axis", out var ax) ? ax.GetInt32() : null;

            object r = Str(p, "name") switch
            {
                "T" => c.T,
                "mT" => c.mT,
                "real" => c.real,
                "imag" => c.imag,
                "transpose" => c.transpose(),
                "swapaxes" => c.swapaxes(args[0].GetInt32(), args[1].GetInt32()),
                "reshape" => kw is JsonElement ro && ro.TryGetProperty("order", out var order)
                    ? c.reshape(new Shape(args.Select(x => x.GetInt64()).ToArray()), order.GetString()[0])
                    : c.reshape(args.Select(x => x.GetInt64()).ToArray()),
                "ravel" => args.Length == 0 ? c.ravel() : c.ravel(args[0].GetString()[0]),
                "flatten" => args.Length == 0 ? c.flatten() : c.flatten(args[0].GetString()[0]),
                "squeeze" => c.squeeze(),
                "copy" => args.Length == 0 ? c.copy() : c.copy(args[0].GetString()[0]),
                "take" => args[0].ValueKind == JsonValueKind.Number
                    ? c.take(args[0].GetInt64(), axis)
                    : c.take(RawArray(args[0].GetProperty("list")), axis),
                "repeat" => c.repeat(args[0].GetInt32(), axis),
                "compress" => c.compress(RawArray(args[0].GetProperty("list")), axis),
                "diagonal" => c.diagonal(),
                "view" => args.Length == 0 ? c.view() : MemberView(c, args[0]),
                "astype" => kw is JsonElement ck && ck.TryGetProperty("copy", out var copy)
                    ? c.astype(np.dtype(args[0].GetProperty("dtype").GetString()), copy.GetBoolean())
                    : c.astype(np.dtype(args[0].GetProperty("dtype").GetString())),
                "getfield" => c.getfield(np.dtype(args[0].GetProperty("dtype").GetString()), args[1].GetInt32()),
                "byteswap" => c.byteswap(),
                "item" => c.item(args[0].GetInt64()),
                "tolist" => c.tolist(),
                "to_device" => c.to_device(args[0].GetString()),
                "flat_index" => c.flatiter[args[0].GetInt64()],
                "flat_slice" => c.flatiter[$"{args[0].GetInt64()}:{args[1].GetInt64()}"],
                "flat_fancy" => c.flatiter[args[0].GetProperty("list").EnumerateArray().Select(x => x.GetInt64()).ToArray()],
                "flat_copy" => c.flatiter.copy(),
                var other => throw new InvalidOperationException($"the strchararray section has no member '{other}'"),
            };
            return new SelfResult(r, c);
        }

        /// <summary><c>c.view(x)</c> with one argument: an array class (<c>{"dtype": "ndarray"}</c> → <c>typeof(NDArray)</c>)
        ///     or a dtype string.</summary>
        /// <param name="c">The chararray.</param>
        /// <param name="arg">The argument spec.</param>
        /// <returns>The view.</returns>
        private static NDArray MemberView(NDCharArray c, JsonElement arg)
        {
            var type = ArrayClassOf(arg);
            return type is not null ? c.view(type) : c.view(np.dtype(arg.GetProperty("dtype").GetString()));
        }

        /// <summary>The array class a spec names (<c>ndarray</c> / <c>chararray</c> / <c>matrix</c>), the CLR type of a
        ///     <c>{"type": "int" | "str"}</c> spec, or null for a dtype string.</summary>
        /// <param name="spec">The spec.</param>
        /// <returns>The type, or null.</returns>
        private static Type ArrayClassOf(JsonElement spec)
        {
            if (spec.TryGetProperty("type", out var t))
                return t.GetString() == "int" ? typeof(long) : typeof(string);
            return spec.GetProperty("dtype").GetString() switch
            {
                "ndarray" => typeof(NDArray),
                "chararray" => typeof(NDCharArray),
                "matrix" => typeof(NDMatrix),
                _ => null,
            };
        }

        /// <summary>
        ///     Section F: <c>a.view(type)</c> / <c>a.view(type=...)</c> / <c>a.view(dtype, type)</c> on a plain operand, or on
        ///     its chararray view (<c>params.chararray</c>). A null first argument is the keyword-only <c>type=</c> (C#'s null
        ///     dtype is an omitted one).
        /// </summary>
        /// <param name="p">The case parameters: <c>args</c>.</param>
        /// <param name="a">The operand.</param>
        /// <returns>The view.</returns>
        private static NDArray CharArrayViewType(Dictionary<string, JsonElement> p, NDArray a)
        {
            NDArray src = p.TryGetValue("chararray", out var ch) && ch.GetBoolean() ? a.view(typeof(NDCharArray)) : a;
            var args = p["args"].EnumerateArray().ToArray();
            if (args.Length == 1)
            {
                var type = ArrayClassOf(args[0]);
                return type is not null ? src.view(type) : src.view(np.dtype(args[0].GetProperty("dtype").GetString()));
            }
            DType dtype = args[0].ValueKind == JsonValueKind.Null ? null : np.dtype(args[0].GetProperty("dtype").GetString());
            return src.view(dtype, ArrayClassOf(args[1]));
        }
    }
}
