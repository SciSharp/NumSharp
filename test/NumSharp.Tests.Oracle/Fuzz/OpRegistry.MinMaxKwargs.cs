using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The <c>minmax_kwargs</c> tier (gen_oracle.py <c>gen_minmax_kwargs</c>): <c>np.max</c> / <c>np.min</c> /
    ///     <c>np.amax</c> / <c>np.amin</c> / <c>np.nanmax</c> / <c>np.nanmin</c> / <c>ndarray.max</c> / <c>ndarray.min</c>
    ///     called through their NumPy-shaped overloads with <c>out=</c>, <c>initial=</c>, <c>where=</c> and every axis form
    ///     (plan <c>docs/plans/api-parameter-parity.md</c>, item A4).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Params: <c>{"fn", "a", "axis", "keepdims", "out"?, "where"?, "initial"?, "bufsize"?}</c>. <c>axis</c> is JSON
    ///     null (NumPy's <c>axis=None</c>) or an integer. <c>a</c>, <c>out</c> and <c>where</c> use the <c>clip_ufunc</c>
    ///     spec grammar (<see cref="ClipUfuncResolve"/>): an operand, a view of one (an out overlapping the input), or a
    ///     Python bool (a C# <c>bool</c> converts to a 0-d bool array, which NumSharp reads as NumPy's <c>where=True</c> /
    ///     <c>where=False</c>). <c>initial</c> is resolved by <see cref="MinMaxInitial"/> into the C# value of the house
    ///     boundary map, so each kind reaches the <c>object initial</c> parameter the way a caller would pass it.
    ///     <c>bufsize</c> runs the call under <see cref="np.setbufsize(long)"/> (the size NumPy's buffer fills used).
    ///     <c>"facet": "strides"</c> replaces the result with its layout record (<see cref="PolyLayoutFacet"/>).
    ///     </para>
    ///     <para>
    ///     The result is a tuple: the returned array; then, when <c>out=</c> is given, the whole base buffer behind it (a
    ///     write outside the view fails) and a 0-d bool that is true when the call returned <c>out</c> itself (NumPy's
    ///     <c>r is out</c>).
    ///     </para>
    /// </remarks>
    public static partial class OpRegistry
    {
        /// <summary>Replays one <c>minmax_kwargs</c> case; see the type remarks for the param grammar.</summary>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>[returned] or [returned, base buffer behind out, returned-is-out].</returns>
        /// <exception cref="NotSupportedException">A function or initial kind the tier does not know (a generator /
        ///     registry mismatch).</exception>
        private static NDArray[] MinMaxKwargsFromCorpus(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            NDArray Arg(string name) => p.TryGetValue(name, out var spec) ? ClipUfuncResolve(spec, ops) : null;

            string fn = p["fn"].GetString();
            var a = Arg("a");
            var target = Arg("out");
            var where = Arg("where");
            object initial = p.TryGetValue("initial", out var ini) ? MinMaxInitial(ini, ops) : null;
            bool keepdims = p.TryGetValue("keepdims", out var kd) && kd.GetBoolean();
            int? axis = p.TryGetValue("axis", out var ax) && ax.ValueKind == JsonValueKind.Number ? ax.GetInt32() : null;

            // NumPy computed the case under np.setbufsize(bufsize): its buffer fills decide where a cast out round-trips and
            // where the calls are cut. np.setbufsize is thread-local, so restoring the previous size keeps other tests clean.
            long previous = p.TryGetValue("bufsize", out var bs) ? np.setbufsize(bs.GetInt64()) : -1;
            NDArray returned;
            try
            {
                returned = fn switch
                {
                    "max" => np.max(a, axis: axis, @out: target, keepdims: keepdims, initial: initial, @where: where),
                    "min" => np.min(a, axis: axis, @out: target, keepdims: keepdims, initial: initial, @where: where),
                    "amax" => np.amax(a, axis: axis, @out: target, keepdims: keepdims, initial: initial, @where: where),
                    "amin" => np.amin(a, axis: axis, @out: target, keepdims: keepdims, initial: initial, @where: where),
                    "nanmax" => np.nanmax(a, axis: axis, @out: target, keepdims: keepdims, initial: initial, @where: where),
                    "nanmin" => np.nanmin(a, axis: axis, @out: target, keepdims: keepdims, initial: initial, @where: where),
                    "ndarray.max" => a.max(axis: axis, @out: target, keepdims: keepdims, initial: initial, @where: where),
                    "ndarray.min" => a.min(axis: axis, @out: target, keepdims: keepdims, initial: initial, @where: where),
                    _ => throw new NotSupportedException($"minmax_kwargs: unknown function '{fn}'"),
                };
            }
            finally
            {
                if (previous >= 0)
                    np.setbufsize(previous);
            }

            // A "facet": "strides" case records WHERE the fresh result's elements live (NumPy's npyiter_new_temp_array
            // layout), through the polynomial tiers' record (PolyLayoutFacet releases the result it replaces).
            if (p.TryGetValue("facet", out var facet) && facet.ValueKind == JsonValueKind.String)
                return new[] { PolyLayoutFacet(p, returned) };
            if (target is null)
                return new[] { returned };
            var outSpec = p["out"];
            int owner = outSpec.TryGetProperty("op", out var op) ? op.GetInt32() : outSpec.GetProperty("of").GetInt32();
            return new[] { returned, BaseBuffer(ops[owner]), NDArray.Scalar(ReferenceEquals(returned, target)) };
        }

        /// <summary>
        ///     The C# value an <c>initial</c> spec of the <c>minmax_kwargs</c> tier stands for — the house boundary map: a
        ///     Python int is the smallest of <c>int</c> / <c>long</c> / <c>ulong</c> / <see cref="BigInteger"/> holding it, a
        ///     Python float a <c>double</c> (exact bits), a Python complex a <see cref="Complex"/>, an np.float16 scalar a
        ///     <see cref="Half"/>, an np.uint16 scalar a <c>char</c>, a str a <c>string</c>, a list an <c>object[]</c>, a
        ///     tuple a <see cref="ValueTuple"/>, an ndarray the operand.
        /// </summary>
        /// <param name="spec">The spec (see gen_oracle.py's minmax_kwargs header).</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The value, or <c>null</c> for <c>initial=None</c>.</returns>
        /// <exception cref="NotSupportedException">An unknown kind.</exception>
        private static object MinMaxInitial(JsonElement spec, NDArray[] ops)
        {
            if (spec.TryGetProperty("op", out var op))
                return ops[op.GetInt32()];

            static double Bits(JsonElement e)
                => BitConverter.Int64BitsToDouble(unchecked((long)ulong.Parse(e.GetString(), NumberStyles.HexNumber, CultureInfo.InvariantCulture)));

            switch (spec.GetProperty("t").GetString())
            {
                case "none":
                    return null;
                case "int":
                    return PyIntObject(BigInteger.Parse(spec.GetProperty("py").GetString(), CultureInfo.InvariantCulture));
                case "float":
                    return Bits(spec.GetProperty("bits"));
                case "bool":
                    return spec.GetProperty("py").GetBoolean();
                case "complex":
                    return new Complex(Bits(spec.GetProperty("re")), Bits(spec.GetProperty("im")));
                case "half":
                    return BitConverter.UInt16BitsToHalf(ushort.Parse(spec.GetProperty("bits").GetString(), NumberStyles.HexNumber,
                        CultureInfo.InvariantCulture));
                case "char":
                    return (char)spec.GetProperty("py").GetInt32();
                case "str":
                    return spec.GetProperty("py").GetString();
                case "list":
                    return spec.GetProperty("py").EnumerateArray().Select(e => PyIntObject(BigInteger.Parse(e.GetRawText(), CultureInfo.InvariantCulture))).ToArray();
                case "tuple":
                {
                    var items = spec.GetProperty("py").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                    if (items.Length != 1)
                        throw new NotSupportedException("minmax_kwargs: only one-item tuples are encoded");
                    return ValueTuple.Create(items[0]);
                }
                default:
                    throw new NotSupportedException($"minmax_kwargs: unknown initial kind '{spec.GetProperty("t").GetString()}'");
            }
        }

        /// <summary>A Python int as the smallest C# integer type holding it (boxed), <see cref="BigInteger"/> past ulong.</summary>
        /// <param name="v">The value.</param>
        /// <returns>The boxed value.</returns>
        private static object PyIntObject(BigInteger v)
        {
            if (v >= int.MinValue && v <= int.MaxValue) return (int)v;
            if (v >= long.MinValue && v <= long.MaxValue) return (long)v;
            if (v >= 0 && v <= ulong.MaxValue) return (ulong)v;
            return v;
        }
    }
}
