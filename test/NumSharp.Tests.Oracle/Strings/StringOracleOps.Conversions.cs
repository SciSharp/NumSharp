using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using NumSharp.Utilities;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for sections <c>cvtext</c> (NumSharp's <see cref="Converts"/> against NumPy's text rules),
    ///     <c>promote</c> (promotion over three or more dtypes, and a Python scalar — a C# literal — beside text and numbers)
    ///     and <c>storenone</c> (Python objects stored into arrays: <c>None</c> into text, <c>ndarray.__setitem__</c>, a store
    ///     through a sub-array of a writeable broadcast).
    /// </summary>
    /// <remarks>
    ///     The <c>cvtext</c> ops run EVERY <see cref="Converts"/> route a caller can take for one conversion (the typed
    ///     overload, the typed overload with a culture, the object overload, <c>ChangeType</c> by <see cref="NPTypeCode"/> and
    ///     the generic <c>ChangeType</c>) and require the routes to agree before the answer is compared with NumPy's: a route
    ///     that gives another value, or raises where another returns, fails the case on its own.
    /// </remarks>
    internal static partial class StringOracleOps
    {
        /// <summary>A culture whose own number format differs from NumPy's (comma decimal separator): Converts ignores it.</summary>
        private static readonly CultureInfo GermanCulture = CultureInfo.GetCultureInfo("de-DE");

        /// <summary>Replays one <c>cvtext</c> / <c>promote</c> / <c>storenone</c> op, or reports that it is not one of them.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="result">The op's result (null for an in-place op, whose operand is compared).</param>
        /// <returns>True when <paramref name="op"/> is one of these sections' ops.</returns>
        private static bool TryInvokeConversions(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            switch (op)
            {
                case "cv_tostring":
                    result = ConvertsToString(p);
                    return true;
                case "cv_fromstring":
                    result = ConvertsFromString(p);
                    return true;
                case "promote_concatenate":
                    result = np.concatenate(ops);
                    return true;
                case "promote_stack":
                    result = np.stack(ops);
                    return true;
                case "promote_vstack":
                    result = np.vstack(ops);
                    return true;
                case "promote_hstack":
                    result = np.hstack(ops);
                    return true;
                case "promote_where_lit":
                {
                    // A C# literal is NumPy's Python scalar (weak): NDArray.FromPythonOperand marks it as such.
                    var literal = NDArray.FromPythonOperand(StringOracleCorpus.Decode(p["literal"]));
                    result = Str(p, "side") == "y" ? np.where(ops[0], ops[1], literal) : np.where(ops[0], literal, ops[1]);
                    return true;
                }
                case "promote_choose_lit":
                    result = np.choose(ops[0], new[] { ops[1], NDArray.FromPythonOperand(StringOracleCorpus.Decode(p["literal"])) });
                    return true;
                case "promote_select_lit":
                    result = np.select(new[] { ops[0] }, new object[] { ops[1] },
                                       NDArray.FromPythonOperand(StringOracleCorpus.Decode(p["literal"])));
                    return true;
                case "promote_r_lit":
                    // np.r_'s entries are objects: the boxed C# literal itself is the Python scalar.
                    result = np.r_[ops[0], StringOracleCorpus.Decode(p["literal"])];
                    return true;
                case "setitem_method":
                    SetItemMethod(ops[0], p["key"], StringOracleCorpus.Decode(p["value"]));
                    result = null;
                    return true;
                case "bcast_subview_store":
                    BroadcastSubviewStore(ops[0], p);
                    result = null;
                    return true;
                case "none_store":
                    result = NoneStore(p, ops);
                    return true;
            }

            result = null;
            return false;
        }

        /// <summary>
        ///     <c>none_store</c> (P1.7b): Python's <c>None</c> through ONE store route, spelled the way a ported NumPy line
        ///     spells it in C# — a <c>null</c> where NumPy passes <c>None</c> (cast to the parameter type where a bare
        ///     <c>null</c> would be ambiguous).
        /// </summary>
        /// <param name="p">The route and its parameters (<c>"route"</c>, plus the indices, mask, casting, dtype, … it takes).</param>
        /// <param name="ops">The target (none for the creation routes).</param>
        /// <returns>The created array for <c>full</c> / <c>full_like</c> / <c>insert</c> / <c>asanyarray</c> /
        ///     <c>asarray</c> / <c>array_list</c>; null for an in-place route (the case compares the mutated target).</returns>
        /// <exception cref="InvalidOperationException">A route the replay does not know.</exception>
        /// <exception cref="Exception">The route's own error, compared with NumPy's (type and text).</exception>
        private static object NoneStore(Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            NDArray a = ops.Length > 0 ? ops[0] : null;
            switch (Str(p, "route"))
            {
                case "fill":
                    a.fill(null);
                    return null;
                case "full":
                    return np.full(ShapeOf(p, "dims"), null, Spec(p["dtype"]));
                case "full_like":
                    return np.full_like(a, null);
                case "copyto":
                {
                    // `where` is NumPy's True (absent), a 0-d bool, or a bool list.
                    NDArray where = p["where"].ValueKind == JsonValueKind.Null ? null : RawArray(p["where"]);
                    np.copyto(a, (NDArray)null, Str(p, "casting"), where);
                    return null;
                }
                case "put":
                    np.put(a, np.array(Longs(p["indices"])), (NDArray)null, Str(p, "mode"));
                    return null;
                case "putmask":
                    np.putmask(a, RawArray(p["mask"]), (NDArray)null);
                    return null;
                case "place":
                    np.place(a, RawArray(p["mask"]), (NDArray)null);
                    return null;
                case "insert":
                    return NoneInsert(a, p["obj"], NullableInt(p, "axis"));
                case "flat_int":
                    a.flatiter[p["index"].GetInt64()] = null;
                    return null;
                case "flat_slice":
                    a.flatiter[Str(p, "index")] = null;
                    return null;
                case "flat_fancy":
                    a.flatiter[Longs(p["index"])] = null;
                    return null;
                case "fill_diagonal":
                    np.fill_diagonal(a, null, p["wrap"].GetBoolean());
                    return null;
                case "put_along_axis":
                    np.put_along_axis(a, RawArray(p["indices"]), null, NullableInt(p, "axis"));
                    return null;
                case "setfield":
                    a.setfield(null, Spec(p["dtype"]), Int(p, "offset"));
                    return null;
                case "asanyarray":
                    // The bare `null` NumPy's np.asanyarray(None, dtype) ports to binds the MemoryView overload in C#.
                    return np.asanyarray(null, Spec(p["dtype"]));
                case "asarray":
                    return np.asarray((NDArray)null, Spec(p["dtype"]));
                case "array_list":
                    // A Python list holding None: an object array of the list's shape (np.array's per-element setitem).
                    return np.array((Array)ManagedArray(StringOracleCorpus.Decode(p["values"])), Spec(p["dtype"]));
                default:
                    throw new InvalidOperationException($"none_store has no route {Str(p, "route")}");
            }
        }

        /// <summary>
        ///     <c>np.insert(a, obj, None, axis)</c> with <c>obj</c> in the C# form a ported NumPy line takes: an int, an
        ///     <c>int[]</c> (a Python list), a <see cref="Slice"/>, an int64 index array or a bool mask array.
        /// </summary>
        /// <param name="a">The array inserted into.</param>
        /// <param name="obj">The generator's position entry (<c>{"int"}</c>, <c>{"list"}</c>, <c>{"slice"}</c>,
        ///     <c>{"arr"}</c>, <c>{"mask"}</c>).</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <returns>The new array.</returns>
        /// <exception cref="InvalidOperationException">A position encoding the replay does not know.</exception>
        private static NDArray NoneInsert(NDArray a, JsonElement obj, int? axis)
        {
            if (obj.TryGetProperty("int", out var i))
                return np.insert(a, i.GetInt32(), (object)null, axis);
            if (obj.TryGetProperty("list", out var list))
                return np.insert(a, list.EnumerateArray().Select(x => x.GetInt32()).ToArray(), (object)null, axis);
            if (obj.TryGetProperty("slice", out var s))
            {
                var f = s.EnumerateArray().Select(x => x.ValueKind == JsonValueKind.Null ? (long?)null : x.GetInt64()).ToArray();
                return np.insert(a, new Slice(f[0], f[1], f[2] ?? 1), (object)null, axis);
            }
            if (obj.TryGetProperty("arr", out var arr))
                return np.insert(a, np.array(Longs(arr)), (object)null, axis);
            if (obj.TryGetProperty("mask", out var mask))
                return np.insert(a, RawArray(mask), (object)null, axis);
            throw new InvalidOperationException($"none_store insert has no position {obj}");
        }

        /// <summary>
        ///     <c>cv_tostring</c>: the text of a number every <see cref="Converts"/> route gives, which must be NumPy's
        ///     <c>str()</c> of the same NumPy scalar. The value is rebuilt from its bytes, so it is exactly NumPy's.
        /// </summary>
        /// <param name="p">The dtype name and the value's bytes as hex.</param>
        /// <returns>The text the routes agree on.</returns>
        /// <exception cref="InvalidOperationException">Two routes gave different texts.</exception>
        private static string ConvertsToString(Dictionary<string, JsonElement> p)
        {
            var dtype = np.dtype(Str(p, "dtype"));
            using var arr = np.frombuffer(Convert.FromHexString(Str(p, "bits")), dtype);
            object v = arr.Storage.GetAtIndex(0);
            var routes = new List<(string Route, string Text)>
            {
                ("ToString(object)", Converts.ToString(v)),
                ("ToString(object, de-DE)", Converts.ToString(v, GermanCulture)),
                ("ChangeType(object, NPTypeCode.String)", (string)Converts.ChangeType(v, NPTypeCode.String)),
                ("ChangeType(object, NPTypeCode.String, de-DE)", (string)Converts.ChangeType(v, NPTypeCode.String, GermanCulture)),
                ("ChangeType<string>(object)", Converts.ChangeType<string>(v)),
                ("typed ToString", v switch
                {
                    bool b => Converts.ToString(b),
                    sbyte x => Converts.ToString(x),
                    byte x => Converts.ToString(x),
                    short x => Converts.ToString(x),
                    ushort x => Converts.ToString(x),
                    int x => Converts.ToString(x),
                    uint x => Converts.ToString(x),
                    long x => Converts.ToString(x),
                    ulong x => Converts.ToString(x),
                    Half x => Converts.ToString(x),
                    float x => Converts.ToString(x),
                    double x => Converts.ToString(x),
                    Complex x => Converts.ToString(x),
                    _ => throw new InvalidOperationException($"cv_tostring has no typed route for {v.GetType().Name}"),
                }),
            };
            string first = routes[0].Text;
            var disagree = routes.Where(r => r.Text != first).ToList();
            if (disagree.Count > 0)
                throw new InvalidOperationException("Converts routes disagree: " +
                                                    string.Join("; ", routes.Select(r => $"{r.Route} -> '{r.Text}'")));
            return first;
        }

        /// <summary>
        ///     <c>cv_fromstring</c>: a text converted to a number by every <see cref="Converts"/> route, as NumPy stores a
        ///     Python str into an array (<c>a[0] = text</c>). The routes must agree on the value (bit for bit) or on the
        ///     exception (type and message); the answer is a one-element array of the dtype, or the routes' exception.
        /// </summary>
        /// <param name="p">The dtype name and the text.</param>
        /// <returns>A new one-element array holding the converted value.</returns>
        /// <exception cref="Exception">The exception every route raised (NumPy's parse / range errors).</exception>
        /// <exception cref="InvalidOperationException">Two routes disagree.</exception>
        private static NDArray ConvertsFromString(Dictionary<string, JsonElement> p)
        {
            var dtype = np.dtype(Str(p, "dtype"));
            var tc = dtype.Meta.TypeCode;
            var text = (string)StringOracleCorpus.Decode(p["text"]);
            var routes = new List<(string Route, Func<object> Convert)>
            {
                ("ChangeType(object, NPTypeCode)", () => Converts.ChangeType((object)text, tc)),
                ("ChangeType(object, NPTypeCode, de-DE)", () => Converts.ChangeType((object)text, tc, GermanCulture)),
                ("ChangeType<string>(string, NPTypeCode)", () => Converts.ChangeType(text, tc)),
                ("typed To*(string)", () => Typed(text, tc, null)),
                ("typed To*(string, de-DE)", () => Typed(text, tc, GermanCulture)),
                ("generic ChangeType<T>(object)", () => Generic(text, tc)),
            };
            var outcomes = routes.Select(r =>
            {
                try
                {
                    return (r.Route, Value: r.Convert(), Error: (Exception)null);
                }
                catch (Exception e)
                {
                    return (r.Route, Value: (object)null, Error: e);
                }
            }).ToList();

            var first = outcomes[0];
            foreach (var o in outcomes.Skip(1))
            {
                bool same = first.Error is null
                    ? o.Error is null && SameBits(first.Value, o.Value)
                    : o.Error is not null && o.Error.GetType() == first.Error.GetType() && o.Error.Message == first.Error.Message;
                if (!same)
                    throw new InvalidOperationException("Converts routes disagree: " + string.Join("; ", outcomes.Select(x =>
                        $"{x.Route} -> {(x.Error is null ? Convert.ToString(x.Value, CultureInfo.InvariantCulture) : x.Error.GetType().Name + ": " + x.Error.Message)}")));
            }

            if (first.Error is not null)
                throw first.Error;
            return OneElement(first.Value, tc);
        }

        /// <summary>The typed <c>Converts.To*(string)</c> overload of a type code (with a culture when given).</summary>
        /// <param name="text">The text.</param>
        /// <param name="tc">The target type code.</param>
        /// <param name="culture">A culture for the provider overload, or null for the plain one.</param>
        /// <returns>The boxed value.</returns>
        /// <exception cref="NotSupportedException">A type code the section does not convert to.</exception>
        private static object Typed(string text, NPTypeCode tc, IFormatProvider culture) => tc switch
        {
            NPTypeCode.Boolean => culture is null ? Converts.ToBoolean(text) : Converts.ToBoolean(text, culture),
            NPTypeCode.SByte => culture is null ? Converts.ToSByte(text) : Converts.ToSByte(text, culture),
            NPTypeCode.Byte => culture is null ? Converts.ToByte(text) : Converts.ToByte(text, culture),
            NPTypeCode.Int16 => culture is null ? Converts.ToInt16(text) : Converts.ToInt16(text, culture),
            NPTypeCode.UInt16 => culture is null ? Converts.ToUInt16(text) : Converts.ToUInt16(text, culture),
            NPTypeCode.Int32 => culture is null ? Converts.ToInt32(text) : Converts.ToInt32(text, culture),
            NPTypeCode.UInt32 => culture is null ? Converts.ToUInt32(text) : Converts.ToUInt32(text, culture),
            NPTypeCode.Int64 => culture is null ? Converts.ToInt64(text) : Converts.ToInt64(text, culture),
            NPTypeCode.UInt64 => culture is null ? Converts.ToUInt64(text) : Converts.ToUInt64(text, culture),
            NPTypeCode.Half => culture is null ? Converts.ToHalf(text) : Converts.ToHalf(text, culture),
            NPTypeCode.Single => culture is null ? Converts.ToSingle(text) : Converts.ToSingle(text, culture),
            NPTypeCode.Double => culture is null ? Converts.ToDouble(text) : Converts.ToDouble(text, culture),
            NPTypeCode.Complex => culture is null ? Converts.ToComplex(text) : Converts.ToComplex(text, culture),
            _ => throw new NotSupportedException($"cv_fromstring does not convert to {tc}"),
        };

        /// <summary>The generic <c>Converts.ChangeType&lt;T&gt;(object)</c> of a type code.</summary>
        /// <param name="text">The text.</param>
        /// <param name="tc">The target type code.</param>
        /// <returns>The boxed value.</returns>
        /// <exception cref="NotSupportedException">A type code the section does not convert to.</exception>
        private static object Generic(string text, NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => Converts.ChangeType<bool>((object)text),
            NPTypeCode.SByte => Converts.ChangeType<sbyte>((object)text),
            NPTypeCode.Byte => Converts.ChangeType<byte>((object)text),
            NPTypeCode.Int16 => Converts.ChangeType<short>((object)text),
            NPTypeCode.UInt16 => Converts.ChangeType<ushort>((object)text),
            NPTypeCode.Int32 => Converts.ChangeType<int>((object)text),
            NPTypeCode.UInt32 => Converts.ChangeType<uint>((object)text),
            NPTypeCode.Int64 => Converts.ChangeType<long>((object)text),
            NPTypeCode.UInt64 => Converts.ChangeType<ulong>((object)text),
            NPTypeCode.Half => Converts.ChangeType<Half>((object)text),
            NPTypeCode.Single => Converts.ChangeType<float>((object)text),
            NPTypeCode.Double => Converts.ChangeType<double>((object)text),
            NPTypeCode.Complex => Converts.ChangeType<Complex>((object)text),
            _ => throw new NotSupportedException($"cv_fromstring does not convert to {tc}"),
        };

        /// <summary>Whether two converted values are the same value of the same type, bit for bit (NaN payloads included).</summary>
        /// <param name="a">One value.</param>
        /// <param name="b">The other.</param>
        /// <returns>True when both have one type and one bit pattern.</returns>
        private static bool SameBits(object a, object b) => (a, b) switch
        {
            (double x, double y) => BitConverter.DoubleToInt64Bits(x) == BitConverter.DoubleToInt64Bits(y),
            (float x, float y) => BitConverter.SingleToInt32Bits(x) == BitConverter.SingleToInt32Bits(y),
            (Half x, Half y) => BitConverter.HalfToInt16Bits(x) == BitConverter.HalfToInt16Bits(y),
            (Complex x, Complex y) => BitConverter.DoubleToInt64Bits(x.Real) == BitConverter.DoubleToInt64Bits(y.Real)
                                      && BitConverter.DoubleToInt64Bits(x.Imaginary) == BitConverter.DoubleToInt64Bits(y.Imaginary),
            _ => a is not null && b is not null && a.GetType() == b.GetType() && a.Equals(b),
        };

        /// <summary>A new owning one-element array of the type code holding <paramref name="v"/> (NumPy's <c>np.zeros(1)</c> after the store).</summary>
        /// <param name="v">The value, of the type code's CLR type.</param>
        /// <param name="tc">The type code.</param>
        /// <returns>The array.</returns>
        /// <exception cref="NotSupportedException">A type code the section does not convert to.</exception>
        private static NDArray OneElement(object v, NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => np.array(new[] { (bool)v }),
            NPTypeCode.SByte => np.array(new[] { (sbyte)v }),
            NPTypeCode.Byte => np.array(new[] { (byte)v }),
            NPTypeCode.Int16 => np.array(new[] { (short)v }),
            NPTypeCode.UInt16 => np.array(new[] { (ushort)v }),
            NPTypeCode.Int32 => np.array(new[] { (int)v }),
            NPTypeCode.UInt32 => np.array(new[] { (uint)v }),
            NPTypeCode.Int64 => np.array(new[] { (long)v }),
            NPTypeCode.UInt64 => np.array(new[] { (ulong)v }),
            NPTypeCode.Half => np.array(new[] { (Half)v }),
            NPTypeCode.Single => np.array(new[] { (float)v }),
            NPTypeCode.Double => np.array(new[] { (double)v }),
            NPTypeCode.Complex => np.array(new[] { (Complex)v }),
            _ => throw new NotSupportedException($"cv_fromstring does not convert to {tc}"),
        };

        /// <summary>
        ///     <c>ndarray.__setitem__(key, value)</c> through NumSharp's <c>__setitem__</c> overloads: an int key takes the
        ///     <c>int</c> overload, a slice key the slice-string one (<c>":2"</c>, <c>"::-1"</c>), the value as the C# object a
        ///     ported line passes (a literal, a string, null for None).
        /// </summary>
        /// <param name="a">The array (written in place).</param>
        /// <param name="key">The generator's index entry (<c>{"int": i}</c> or <c>{"slice": [start, stop, step]}</c>).</param>
        /// <param name="value">The decoded value.</param>
        /// <exception cref="InvalidOperationException">A key encoding the op does not know.</exception>
        private static void SetItemMethod(NDArray a, JsonElement key, object value)
        {
            if (key.TryGetProperty("int", out var i))
            {
                a.__setitem__(i.GetInt32(), value);
                return;
            }

            if (key.TryGetProperty("slice", out var s))
            {
                string Part(JsonElement e) => e.ValueKind == JsonValueKind.Null ? "" : e.GetInt64().ToString(CultureInfo.InvariantCulture);
                var parts = s.EnumerateArray().Select(Part).ToArray();
                a.__setitem__(parts[0] + ":" + parts[1] + (parts[2].Length > 0 ? ":" + parts[2] : ""), value);
                return;
            }

            throw new InvalidOperationException($"setitem_method has no key {key}");
        }

        /// <summary>
        ///     <c>a, b = np.broadcast_arrays(x[:, None], x); (a or b)[row][...] = value</c>: a store through a sub-array of a
        ///     writeable broadcast (stride 0 for a row of <c>a</c>), which NumPy's views inherit WRITEABLE for; <c>x</c> is the
        ///     compared operand.
        /// </summary>
        /// <param name="x">The 1-D source (written through the broadcast).</param>
        /// <param name="p">Which result (<c>"a"</c> / <c>"b"</c>), the row and the value (a number or a list of numbers).</param>
        private static void BroadcastSubviewStore(NDArray x, Dictionary<string, JsonElement> p)
        {
            var (a, b) = np.broadcast_arrays(x[Slice.All, Slice.NewAxis], x);
            var target = Str(p, "which") == "a" ? a : b;
            var row = target[p["row"].GetInt32()];
            object value = StringOracleCorpus.Decode(p["value"]);
            row[Slice.Ellipsis] = value is List<object> list
                ? np.array(list.Select(o => (long)o).ToArray())
                : NDArray.FromPythonOperand(value);
        }
    }
}
