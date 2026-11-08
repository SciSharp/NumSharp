using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    public static partial class OpRegistry
    {
        /// <summary>
        ///     Replays one <c>weak_scalar.jsonl</c> case (gen_oracle.py <c>weak_scalar</c>): a comparison or binary ufunc
        ///     with one operand given as the C# LITERAL that NumPy's Python scalar maps to, so NEP 50's weak promotion is
        ///     under the gate (every other tier passes arrays, which NumPy treats as strong).
        /// </summary>
        /// <param name="p">The case's params: <c>fn</c> (the NumPy ufunc name, or <c>clip</c>); <c>route</c>
        ///     (<c>func</c>: <c>np.fn</c>; <c>op</c>: the C# operator over the 0-d array a literal binds to; <c>opobj</c>:
        ///     the operator's <c>object</c> overload; <c>out</c> / <c>where</c>: <c>np.fn</c> with the next operand as out
        ///     and the one after as the mask); <c>side</c> (<c>array_first</c> / <c>scalar_first</c>); <c>scalar</c> (kind,
        ///     exact value, and <c>cs</c>, the C# type to spell it in); <c>bound</c> for clip (which bound the scalar is).
        ///     A case without <c>scalar</c> (section E) compares its two array operands. Route <c>store</c> (section S)
        ///     stores the literal into operand 0 at the store entry point <c>fn</c> names (<see cref="WeakStore"/>).</param>
        /// <param name="ops">Operand 0 is the array; then the out array and the mask for those routes (section E: the two
        ///     arrays).</param>
        /// <returns>The result, or the out array itself for the out / where routes (its prior pattern shows at a
        ///     masked-off slot).</returns>
        /// <exception cref="NotSupportedException">An unknown <c>fn</c>, route, kind or C# type (a corpus bug).</exception>
        internal static NDArray ApplyWeakScalar(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            string fn = p["fn"].GetString();
            string route = p["route"].GetString();

            if (!p.TryGetValue("scalar", out var sc))
                return route == "op" ? WeakOperator(fn, ops[0], ops[1]) : WeakUfunc(fn, ops[0], ops[1], null, null);

            object literal = WeakLiteral(sc);
            // The 0-d array a C# literal of that type converts to, through the implicit conversion itself (which marks a
            // Python literal / NumPy scalar for the store paths).
            NDArray scalar = LiteralArray(literal);
            bool scalarFirst = p["side"].GetString() == "scalar_first";
            NDArray x = ops[0];

            if (route == "store")
                return WeakStore(fn, x, literal, scalar);

            if (fn == "clip")
                return p["bound"].GetString() == "min" ? np.clip(x, scalar, null) : np.clip(x, null, scalar);

            switch (route)
            {
                case "func":
                    return scalarFirst ? WeakUfunc(fn, scalar, x, null, null) : WeakUfunc(fn, x, scalar, null, null);
                case "op":
                    return scalarFirst ? WeakOperator(fn, scalar, x) : WeakOperator(fn, x, scalar);
                case "opobj":
                    // The object overloads take the boxed value itself (np.asanyarray inside).
                    return scalarFirst ? WeakObjectOperatorLeft(fn, literal, x) : WeakObjectOperatorRight(fn, x, literal);
                case "out":
                {
                    NDArray o = ops[1];
                    var r = scalarFirst ? WeakUfunc(fn, scalar, x, o, null) : WeakUfunc(fn, x, scalar, o, null);
                    return ReferenceEquals(r, o) ? o : throw new InvalidOperationException($"weak_scalar: {fn} did not return its out");
                }
                case "where":
                {
                    NDArray o = ops[1], m = ops[2];
                    var r = scalarFirst ? WeakUfunc(fn, scalar, x, o, m) : WeakUfunc(fn, x, scalar, o, m);
                    return ReferenceEquals(r, o) ? o : throw new InvalidOperationException($"weak_scalar: {fn} did not return its out");
                }
                default:
                    throw new NotSupportedException($"weak_scalar: unknown route {route}");
            }
        }

        /// <summary>
        ///     The boxed C# value a <c>weak_scalar</c> case's literal is spelled in: the corpus <c>cs</c> type with the exact
        ///     value (an int's digits, a float's or complex part's IEEE bits, a float16's bits).
        /// </summary>
        /// <param name="sc">The <c>scalar</c> descriptor.</param>
        /// <returns>The boxed value (<c>int</c>, <c>long</c>, <c>ulong</c>, <c>short</c>, <c>byte</c>, <c>sbyte</c>,
        ///     <c>uint</c>, <c>ushort</c>, <c>double</c>, <c>float</c>, <see cref="Complex"/>, <c>bool</c> or
        ///     <see cref="Half"/>).</returns>
        /// <exception cref="NotSupportedException">An unknown kind or C# type.</exception>
        private static object WeakLiteral(JsonElement sc)
        {
            string kind = sc.GetProperty("kind").GetString();
            string cs = sc.GetProperty("cs").GetString();
            switch (kind)
            {
                case "int":
                {
                    var v = BigInteger.Parse(sc.GetProperty("value").GetString(), CultureInfo.InvariantCulture);
                    return cs switch
                    {
                        "int" => (object)(int)v,
                        "long" => (long)v,
                        "ulong" => (ulong)v,
                        "short" => (short)v,
                        "byte" => (byte)v,
                        "sbyte" => (sbyte)v,
                        "uint" => (uint)v,
                        "ushort" => (ushort)v,
                        // A C# char is NumPy's np.uint16 SCALAR (section S's NumPy-scalar picks).
                        "char" => (char)(ushort)v,
                        _ => throw new NotSupportedException($"weak_scalar: int literal type {cs}"),
                    };
                }
                case "float":
                {
                    double d = LittleEndianDouble(sc.GetProperty("bits").GetString());
                    return cs switch
                    {
                        "double" => (object)d,
                        // The generator only spells a float32-exact value as a C# float: the cast is exact.
                        "float" => (float)d,
                        _ => throw new NotSupportedException($"weak_scalar: float literal type {cs}"),
                    };
                }
                case "complex":
                    return new Complex(LittleEndianDouble(sc.GetProperty("re").GetString()),
                        LittleEndianDouble(sc.GetProperty("im").GetString()));
                case "bool":
                    return sc.GetProperty("value").GetBoolean();
                case "half":
                    return BitConverter.UInt16BitsToHalf((ushort)sc.GetProperty("bits").GetInt32());
                default:
                    throw new NotSupportedException($"weak_scalar: unknown scalar kind {kind}");
            }
        }

        /// <summary>
        ///     The 0-d array a C# value of the literal's type becomes through NDArray's implicit conversion operator — what
        ///     C# source passing that literal to an <see cref="NDArray"/> parameter gets: an integer / <c>float</c> /
        ///     <c>double</c> / <see cref="Complex"/> marked as a Python literal, a <see cref="Half"/> / <c>char</c> as a
        ///     NumPy scalar, a <c>bool</c> unmarked.
        /// </summary>
        /// <param name="literal">The boxed literal (<see cref="WeakLiteral"/>).</param>
        /// <returns>A new 0-d array.</returns>
        /// <exception cref="NotSupportedException">A type with no implicit conversion (a corpus bug).</exception>
        private static NDArray LiteralArray(object literal) => literal switch
        {
            int v => v,
            long v => v,
            ulong v => v,
            short v => v,
            byte v => v,
            sbyte v => v,
            uint v => v,
            ushort v => v,
            char v => v,
            double v => v,
            float v => v,
            Complex v => v,
            bool v => v,
            Half v => v,
            _ => throw new NotSupportedException($"weak_scalar: no implicit conversion from {literal?.GetType().Name}"),
        };

        /// <summary>
        ///     One section-S store (gen_oracle.py <c>_ws_store_call</c>): the literal stored into <paramref name="x"/> at
        ///     the store entry point <paramref name="fn"/>, spelled the way C# source would call it — an <see cref="NDArray"/>
        ///     parameter (the indexers, put, putmask, place, put_along_axis, insert, asarray, the flat slice) receives the
        ///     literal's implicit 0-d array; an <c>object</c> parameter (fill, full, full_like, fill_diagonal, pad, the flat
        ///     index) receives the boxed literal; np.copyto binds its generic scalar overload.
        /// </summary>
        /// <param name="fn">The store (<c>setitem_int</c>, …, <c>copyto</c>).</param>
        /// <param name="x">The destination (the case's operand 0, a fresh copy of the generator's array).</param>
        /// <param name="literal">The boxed C# literal.</param>
        /// <param name="v">The literal's implicit 0-d array.</param>
        /// <returns><paramref name="x"/> after an in-place store, or the new array of a constructing one.</returns>
        /// <exception cref="NotSupportedException">An unknown store (a corpus bug).</exception>
        private static NDArray WeakStore(string fn, NDArray x, object literal, NDArray v)
        {
            // The mask and index arrays are this replay's own temporaries: disposed here, so the zero-leak gate measures
            // only what the store itself strands.
            switch (fn)
            {
                case "setitem_int": x[0] = v; return x;
                case "setitem_neg": x[-1] = v; return x;
                case "setitem_slice": x["1:"] = v; return x;
                case "setitem_empty": x[$"{x.shape[0]}:"] = v; return x;
                case "setitem_ellipsis": x["..."] = v; return x;
                case "setitem_fancy": x[new[] { 0, -1 }] = v; return x;
                case "setitem_mask":
                {
                    using var m = StoreMask(x);
                    x[m] = v;
                    return x;
                }
                case "fill": x.fill(literal); return x;
                case "full": return np.full(new Shape(x.shape), literal, x.dtype);
                case "full_like": return np.full_like(x, literal);
                case "put":
                {
                    using var idx = np.array(new long[] { 0, -1 });
                    np.put(x, idx, v);
                    return x;
                }
                case "putmask":
                {
                    using var m = StoreMask(x);
                    np.putmask(x, m, v);
                    return x;
                }
                case "place":
                {
                    using var m = StoreMask(x);
                    np.place(x, m, v);
                    return x;
                }
                case "put_along_axis":
                {
                    using var idx = np.array(new long[] { 0 });
                    np.put_along_axis(x, idx, v, 0);
                    return x;
                }
                case "insert": return np.insert(x, 1, v);
                case "fill_diagonal": np.fill_diagonal(x, literal); return x;
                case "pad": return np.pad(x, 1, constant_values: literal);
                case "flat_index": x.flatiter[0] = literal; return x;
                case "flat_slice": x.flatiter["0:2"] = v; return x;
                case "asarray": return np.asarray(v, x.dtype);
                case "copyto": CopytoLiteral(x, literal); return x;
                default: throw new NotSupportedException($"weak_scalar: unknown store {fn}");
            }
        }

        /// <summary>The mask section S's masked stores use: every other element in C order (<c>_ws_store_mask</c>).</summary>
        /// <param name="x">The destination.</param>
        /// <returns>A bool array of <paramref name="x"/>'s shape, true at the even C-order positions.</returns>
        private static NDArray StoreMask(NDArray x)
        {
            var bits = new bool[x.size];
            for (long i = 0; i < bits.Length; i += 2)
                bits[i] = true;
            // A fresh array of x's shape: one buffer the caller disposes. The flat array and its reshape VIEW are both
            // released here (a view holds a counted reference on the flat buffer until it is disposed).
            using var flat = np.array(bits);
            using var view = flat.reshape(x.shape);
            return view.copy();
        }

        /// <summary>np.copyto with the literal through the generic scalar overload its C# type binds.</summary>
        /// <param name="x">The destination.</param>
        /// <param name="literal">The boxed literal.</param>
        /// <exception cref="NotSupportedException">A literal type with no overload (a corpus bug).</exception>
        private static void CopytoLiteral(NDArray x, object literal)
        {
            switch (literal)
            {
                case int v: np.copyto(x, v); return;
                case long v: np.copyto(x, v); return;
                case ulong v: np.copyto(x, v); return;
                case short v: np.copyto(x, v); return;
                case byte v: np.copyto(x, v); return;
                case sbyte v: np.copyto(x, v); return;
                case uint v: np.copyto(x, v); return;
                case ushort v: np.copyto(x, v); return;
                case char v: np.copyto(x, v); return;
                case double v: np.copyto(x, v); return;
                case float v: np.copyto(x, v); return;
                case Complex v: np.copyto(x, v); return;
                case bool v: np.copyto(x, v); return;
                case Half v: np.copyto(x, v); return;
                default: throw new NotSupportedException($"weak_scalar: copyto of {literal?.GetType().Name}");
            }
        }

        /// <summary>The NumPy-shaped ufunc <paramref name="fn"/>(x1, x2, out, where).</summary>
        /// <param name="fn">The ufunc name.</param>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="out">The out array, or null.</param>
        /// <param name="where">The mask, or null.</param>
        /// <returns>The ufunc's result.</returns>
        /// <exception cref="NotSupportedException">An unknown ufunc (a corpus bug).</exception>
        private static NDArray WeakUfunc(string fn, NDArray x1, NDArray x2, NDArray @out, NDArray where) => fn switch
        {
            "equal" => np.equal(x1, x2, @out, where),
            "not_equal" => np.not_equal(x1, x2, @out, where),
            "less" => np.less(x1, x2, @out, where),
            "less_equal" => np.less_equal(x1, x2, @out, where),
            "greater" => np.greater(x1, x2, @out, where),
            "greater_equal" => np.greater_equal(x1, x2, @out, where),
            "add" => np.add(x1, x2, @out, where),
            "subtract" => np.subtract(x1, x2, @out, where),
            "multiply" => np.multiply(x1, x2, @out, where),
            "maximum" => np.maximum(x1, x2, @out, where),
            "minimum" => np.minimum(x1, x2, @out, where),
            "fmax" => np.fmax(x1, x2, @out, where),
            "fmin" => np.fmin(x1, x2, @out, where),
            "arctan2" => np.arctan2(x1, x2, @out, where),
            "hypot" => np.hypot(x1, x2, @out, where),
            "copysign" => np.copysign(x1, x2, @out, where),
            "heaviside" => np.heaviside(x1, x2, @out, where),
            "logaddexp" => np.logaddexp(x1, x2, @out, where),
            "bitwise_and" => np.bitwise_and(x1, x2, @out, where),
            "bitwise_or" => np.bitwise_or(x1, x2, @out, where),
            "bitwise_xor" => np.bitwise_xor(x1, x2, @out, where),
            "floor_divide" => np.floor_divide(x1, x2, @out, where),
            "remainder" => np.remainder(x1, x2, @out, where),
            "left_shift" => np.left_shift(x1, x2, @out, where),
            "right_shift" => np.right_shift(x1, x2, @out, where),
            "power" => np.power(x1, x2, @out, where),
            "gcd" => np.gcd(x1, x2, @out, where),
            "fmod" => np.fmod(x1, x2, @out, where),
            _ => throw new NotSupportedException($"weak_scalar: unknown fn {fn}"),
        };

        /// <summary>The C# comparison operator for <paramref name="fn"/> over two arrays (the overload a literal binds).</summary>
        /// <param name="fn">The comparison ufunc's name.</param>
        /// <param name="a">The left operand.</param>
        /// <param name="b">The right operand.</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="NotSupportedException">Not a comparison (a corpus bug).</exception>
        private static NDArray WeakOperator(string fn, NDArray a, NDArray b) => fn switch
        {
            "equal" => a == b,
            "not_equal" => a != b,
            "less" => a < b,
            "less_equal" => a <= b,
            "greater" => a > b,
            "greater_equal" => a >= b,
            _ => throw new NotSupportedException($"weak_scalar: no operator for {fn}"),
        };

        /// <summary>The C# comparison operator's <c>(NDArray, object)</c> overload.</summary>
        /// <param name="fn">The comparison ufunc's name.</param>
        /// <param name="a">The array.</param>
        /// <param name="b">The boxed C# literal.</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="NotSupportedException">Not a comparison (a corpus bug).</exception>
        private static NDArray WeakObjectOperatorRight(string fn, NDArray a, object b) => fn switch
        {
            "equal" => a == b,
            "not_equal" => a != b,
            "less" => a < b,
            "less_equal" => a <= b,
            "greater" => a > b,
            "greater_equal" => a >= b,
            _ => throw new NotSupportedException($"weak_scalar: no operator for {fn}"),
        };

        /// <summary>The C# comparison operator's <c>(object, NDArray)</c> overload.</summary>
        /// <param name="fn">The comparison ufunc's name.</param>
        /// <param name="a">The boxed C# literal.</param>
        /// <param name="b">The array.</param>
        /// <returns>The bool result.</returns>
        /// <exception cref="NotSupportedException">Not a comparison (a corpus bug).</exception>
        private static NDArray WeakObjectOperatorLeft(string fn, object a, NDArray b) => fn switch
        {
            "equal" => a == b,
            "not_equal" => a != b,
            "less" => a < b,
            "less_equal" => a <= b,
            "greater" => a > b,
            "greater_equal" => a >= b,
            _ => throw new NotSupportedException($"weak_scalar: no operator for {fn}"),
        };
    }
}
