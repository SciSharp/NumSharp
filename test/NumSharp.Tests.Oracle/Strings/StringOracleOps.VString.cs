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
            }
            result = null;
            return false;
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
            var t = e.GetProperty("T");
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
