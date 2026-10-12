using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for section tdtype (<c>gen_tdtype</c>, strings phase 1, P1.2): the NEP 55
    ///     <c>StringDType</c> descriptor without arrays — attributes, equality, promotion, the casting-rule table, the
    ///     scalar hierarchy, <c>np.isdtype</c>, <c>newbyteorder</c> and the ufunc <c>dtype=</c> refusal. A descriptor
    ///     request is either a dtype string (<c>np.dtype("T")</c>) or <c>{"T": {"coerce": bool, "na": value}}</c>, the
    ///     <c>np.dtypes.StringDType</c> call of <see cref="Spec"/>.
    /// </summary>
    internal static partial class StringOracleOps
    {
        /// <summary>
        ///     Replays one section-tdtype op, or reports that <paramref name="op"/> is not one of them.
        /// </summary>
        /// <param name="op">The corpus op key (<c>t_props</c>, <c>t_eq</c>, …).</param>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The reconstructed operands (only <c>t_ufunc_dtype</c> has one).</param>
        /// <param name="result">The call's result: a descriptor, a bool.</param>
        /// <returns>True when <paramref name="op"/> is a section-tdtype op.</returns>
        /// <exception cref="InvalidOperationException">A parameter encoding the decoder does not know.</exception>
        private static bool TryInvokeVString(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            switch (op)
            {
                case "t_props":
                    result = Spec(p["spec"]);
                    return true;
                case "t_eq":
                    if (p.TryGetValue("same", out var same) && same.GetBoolean())
                    {
                        // One sentinel OBJECT behind both descriptors (NumPy's identity shortcut decides an np.float16 NaN).
                        var spec = p["a"];
                        var t = spec.GetProperty("T");
                        object na = t.TryGetProperty("na", out var naE) ? DecodeNA(naE) : null;
                        bool coerce = t.GetProperty("coerce").GetBoolean();
                        bool hasNA = t.TryGetProperty("na", out _);
                        DType Build() => hasNA ? np.dtypes.StringDType(na_object: na, coerce: coerce) : np.dtypes.StringDType(coerce);
                        result = Build().Equals(Build());
                        return true;
                    }
                    result = Spec(p["a"]).Equals(Spec(p["b"]));
                    return true;
                case "t_eq_str":
                    // A dtype STRING on the right: DType.Equals(string) coerces it for the legacy classes and not for
                    // StringDType, as NumPy's __eq__ does.
                    result = Spec(p["a"]).Equals(Str(p, "b"));
                    return true;
                case "t_promote":
                    result = np.promote_types(Spec(p["a"]), Spec(p["b"]));
                    return true;
                case "t_result":
                {
                    // The descriptors in order, then the weak Python literals (decoded as the C# literals that stand for
                    // them: a long is a Python int, a double a Python float, a Complex a Python complex, a bool a bool).
                    var args = p["specs"].EnumerateArray().Select(e => (object)Spec(e))
                        .Concat(p["extra"].EnumerateArray().Select(DecodeNA)).ToArray();
                    result = args.All(x => x is DType) ? np.result_type(args.Cast<DType>().ToArray()) : np.result_type(args);
                    return true;
                }
                case "t_can_cast":
                    result = np.can_cast(Spec(p["from"]), Spec(p["to"]), Str(p, "casting"));
                    return true;
                case "t_issubdtype":
                    result = p.ContainsKey("category")
                        ? np.issubdtype(Spec(p["a"]), Str(p, "category"))
                        : np.issubdtype(Spec(p["a"]), Spec(p["b"]));
                    return true;
                case "t_isdtype":
                {
                    var kinds = p["kinds"].EnumerateArray().Select(e => e.GetString()).ToArray();
                    // One kind is NumPy's plain string argument; zero or several are its tuple.
                    result = kinds.Length == 1 ? np.isdtype(Spec(p["dtype"]), kinds[0]) : np.isdtype(Spec(p["dtype"]), kinds);
                    return true;
                }
                case "t_isdtype_dtypes":
                {
                    var kinds = p["kinds"].EnumerateArray().Select(e => Spec(e)).ToArray();
                    result = kinds.Length == 1 ? np.isdtype(Spec(p["dtype"]), kinds[0]) : np.isdtype(Spec(p["dtype"]), kinds);
                    return true;
                }
                case "t_newbyteorder":
                    result = Spec(p["spec"]).newbyteorder(Str(p, "order"));
                    return true;
                case "t_ufunc_dtype":
                    result = np.add(ops[0], ops[0], dtype: Spec(p["spec"]));
                    return true;

                // ---- sections tstore / tcreate / tmove (P1.3 - P1.5) ----
                case "t_writes":
                {
                    // np.array(values, dtype=T) followed by `a[index] = value` writes in order; the array or its cells.
                    var arr = np.array(ManagedArray(StringOracleCorpus.Decode(p["values"])), Dt(p, "dtype"));
                    foreach (var w in p["writes"].EnumerateArray())
                        SetLiteral(arr, Index(w[0]), StringOracleCorpus.Decode(w[1]));
                    // Both arms as object: a ?: of byte[] and NDArray is typed NDArray (the implicit Array -> NDArray conversion
                    // would turn the cells into a uint8 array).
                    result = Str(p, "result") == "cells" ? (object)arr.tobytes() : arr;
                    return true;
                }
                case "t_sizes":
                {
                    var a = ops[0];
                    var sizes = new List<object> { (long)a.itemsize, (long)a.nbytes };
                    sizes.AddRange(a.strides.Select(s => (object)s));
                    result = sizes;
                    return true;
                }
                case "t_tobytes":
                    result = ops[0].tobytes(p.TryGetValue("order", out var to) ? to.GetString()[0] : 'C');
                    return true;
                case "t_copy_cells":
                {
                    // np.copy(a, order).tobytes(order='A'): the copy's own cells in its memory order, so the arena offsets
                    // show the order the copy duplicated the strings in.
                    using var copy = np.copy(ops[0], Str(p, "order")[0]);
                    result = copy.tobytes('A');
                    return true;
                }
                case "t_setitem":
                    SetLiteral(ops[0], Index(p["index"]), StringOracleCorpus.Decode(p["value"]));
                    result = ops[0];
                    return true;
                case "t_nditer":
                {
                    // [x[()] for x in np.nditer(a, flags=...)]: each step's operand 0 is a 0-d view; its element (a text or
                    // the NA object) is the step's value.
                    var flags = p["flags"].EnumerateArray().Select(e => e.GetString()).ToArray();
                    using var it = np.nditer(ops[0], flags.Length == 0 ? null : flags);
                    var values = new List<object>();
                    foreach (var step in it)
                        values.Add(step[0].Storage.GetAtIndex(0));
                    result = values;
                    return true;
                }
                case "t_view_same":
                    // a.view(a.dtype): the array's own descriptor object.
                    result = ops[0].view(ops[0].dtype);
                    return true;
                case "t_memoryview":
                    result = ops[0].data;
                    return true;
                case "t_frombuffer":
                    result = np.frombuffer(Convert.FromHexString(Str(p, "buffer")), Dt(p, "dtype"));
                    return true;
                case "select":
                {
                    var conds = p["conds"].EnumerateArray().Select(RawArray).ToArray();
                    result = np.select(conds, ops.Cast<object>().ToArray(), StringOracleCorpus.Decode(p["default"]));
                    return true;
                }
                case "fill_diagonal":
                    np.fill_diagonal(ops[0], StringOracleCorpus.Decode(p["value"]), p["wrap"].GetBoolean());
                    result = ops[0];
                    return true;
                case "t_resize_method":
                    ops[0].resize(new Shape(p["shape"].EnumerateArray().Select(e => e.GetInt64()).ToArray()), refcheck: false);
                    result = ops[0];
                    return true;
            }
            result = null;
            return false;
        }

        /// <summary>
        ///     <c>a[index] = value</c> as a ported NumPy line writes it, with the C# value standing for the Python object NumPy
        ///     got: an all-integer index naming ONE element goes through the object-typed <see cref="NDArray.SetValue(object, int[])"/>
        ///     (NumPy's <c>stringdtype_setitem</c> on the element — <c>None</c>, bytes and bools included); any other index
        ///     through the indexer with the value's implicit conversion (a C# literal: a Python str / int / float / bool, and
        ///     <c>null</c> for <c>None</c>).
        /// </summary>
        /// <param name="a">The target.</param>
        /// <param name="index">The indexer arguments (<see cref="Index"/>).</param>
        /// <param name="value">The decoded value.</param>
        /// <exception cref="InvalidOperationException">A value kind the replay has no C# spelling for.</exception>
        private static void SetLiteral(NDArray a, object[] index, object value)
        {
            if (index.Length == a.ndim && index.All(x => x is int))
            {
                // Negative integers count from the end, as NumPy's index does (SetValue takes non-negative coordinates).
                var coords = new int[index.Length];
                for (int i = 0; i < coords.Length; i++)
                {
                    int k = (int)index[i];
                    coords[i] = k < 0 ? k + (int)a.Shape.dimensions[i] : k;
                }
                a.SetValue(value, coords);
                return;
            }
            a[index] = value switch
            {
                null => null,
                string s => s,
                long l => l,
                double d => d,
                bool b => b,
                byte[] bytes => NDArray.Scalar(bytes),
                _ => throw new InvalidOperationException($"no literal spelling for {value.GetType().Name}"),
            };
        }

        /// <summary>
        ///     A descriptor request of section tdtype: a dtype string through <c>np.dtype</c>, or
        ///     <c>{"T": {"coerce": bool, "na": value}}</c> through <c>np.dtypes.StringDType</c> — the <c>na_object</c>
        ///     overload when <c>"na"</c> is present (even for <c>None</c>), the <c>coerce</c>-only overload when absent.
        /// </summary>
        /// <param name="e">The request.</param>
        /// <returns>The descriptor.</returns>
        /// <exception cref="UnicodeEncodeError">A string sentinel with a lone surrogate (NumPy's own error, compared).</exception>
        private static DType Spec(JsonElement e)
        {
            if (e.ValueKind == JsonValueKind.String)
                return np.dtype(e.GetString());
            return VStringDescr(e.GetProperty("T"));
        }

        /// <summary>
        ///     The <c>StringDType</c> descriptor of a request's inner object <c>{"coerce": bool, "na": value}</c> (an operand's
        ///     <c>t</c> member, a result's <c>t</c> member, the inside of a <c>{"T": ...}</c> request): the <c>na_object</c>
        ///     overload when <c>"na"</c> is present (even for <c>None</c>), the <c>coerce</c>-only overload when absent.
        /// </summary>
        /// <param name="t">The inner object.</param>
        /// <returns>A new descriptor.</returns>
        /// <exception cref="UnicodeEncodeError">A string NA object with a lone surrogate.</exception>
        /// <exception cref="InvalidOperationException">An NA encoding the decoder does not know.</exception>
        internal static DType VStringDescr(JsonElement t)
        {
            bool coerce = t.GetProperty("coerce").GetBoolean();
            return t.TryGetProperty("na", out var na)
                ? np.dtypes.StringDType(na_object: DecodeNA(na), coerce: coerce)
                : np.dtypes.StringDType(coerce);
        }

        /// <summary>
        ///     The C# value standing in for a <c>na_object</c> (or a weak Python literal) of the generator's <c>enc_na</c>:
        ///     <c>None</c> → null, a Python int → long (past int64 → <see cref="BigInteger"/>), a float → double (NaN and the
        ///     infinities travel as strings), a complex → <see cref="Complex"/>, an <c>np.float16</c> → <see cref="Half"/>
        ///     (from its bits), str / bytes / bool as the corpus' value encoding, a tuple → a value tuple of one or two
        ///     items, a list → <c>object[]</c>.
        /// </summary>
        /// <param name="e">The encoded value.</param>
        /// <returns>The C# value.</returns>
        /// <exception cref="InvalidOperationException">An encoding the decoder does not know, or a tuple of more than two items.</exception>
        private static object DecodeNA(JsonElement e)
        {
            if (e.TryGetProperty("f2bits", out var bits))
                return BitConverter.UInt16BitsToHalf((ushort)bits.GetInt32());
            if (e.TryGetProperty("complex", out var z))
                return new Complex(JsonFloat(z[0]), JsonFloat(z[1]));
            if (e.TryGetProperty("float", out var f))
                return JsonFloat(f);
            if (e.TryGetProperty("bigint", out var big))
                return BigInteger.Parse(big.GetString(), CultureInfo.InvariantCulture);
            if (e.TryGetProperty("tuple", out var tup))
            {
                var items = tup.EnumerateArray().Select(DecodeNA).ToArray();
                return items.Length switch
                {
                    0 => ValueTuple.Create(),
                    1 => ValueTuple.Create(items[0]),
                    2 => (items[0], items[1]),
                    _ => throw new InvalidOperationException($"the decoder builds tuples of up to two items, not {items.Length}"),
                };
            }
            if (e.TryGetProperty("list", out var list))
                return list.EnumerateArray().Select(DecodeNA).ToArray();
            // The generator's _jf floats (a JSON number, or "nan" / "inf" / "-inf") are read by the surface section's
            // JsonFloat (StringOracleOps.Surface.cs), which knows the same three spellings.
            return StringOracleCorpus.Decode(e);
        }
    }
}
