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
    ///     The C# twin of <c>test/oracle/gen_matrix_oracle.py</c>: every op key of the matrix tier mapped to the
    ///     <see cref="NDMatrix"/> call NumPy's <c>numpy.matrix</c> call corresponds to.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The matrix semantics of <see cref="NDMatrix"/> bind by STATIC type, so this file is careful about the type
    ///     each expression has: an operand the corpus marks as a matrix arrives as an <see cref="NDMatrix"/> instance
    ///     (<see cref="FuzzCorpus.Reconstruct"/>) and is cast before a member or an operator is used on it, which is
    ///     exactly what ported NumPy code does with a variable typed <see cref="NDMatrix"/>.
    ///     </para>
    ///     <para>
    ///     A result is returned untouched (never copied or re-wrapped), because the replay compares its class and its
    ///     memory facet — view, copy or identity against each operand, flags and strides — not only its values.
    ///     </para>
    /// </remarks>
    public static partial class OpRegistry
    {
        /// <summary>Whether an op key belongs to the matrix tier: the constructor key or a member key.</summary>
        /// <param name="op">The op key.</param>
        /// <returns><c>true</c> for <c>matrix</c> and every <c>matrix.&lt;member&gt;</c>.</returns>
        internal static bool IsMatrixOp(string op)
            => op == "matrix" || op.StartsWith("matrix.", StringComparison.Ordinal);

        /// <summary>The matrix tier's ops whose result is one array (or a scalar as a 0-d array).</summary>
        /// <param name="op">The op key (<c>matrix</c> or <c>matrix.&lt;member&gt;</c>).</param>
        /// <param name="p">The case's params.</param>
        /// <param name="ops">The case's operands.</param>
        /// <returns>NumSharp's result, as the member returned it.</returns>
        /// <exception cref="NotSupportedException">The key is not a matrix-tier key this registry knows.</exception>
        internal static NDArray ApplyMatrix(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            if (op == "matrix")
                return MatrixConstruct(p, ops);

            string name = op.Substring("matrix.".Length);
            // The binary operators: the key names the Python method NumPy dispatched to, params.sym the operator.
            if (p.TryGetValue("sym", out var sym) && name is not ("__neg__" or "__pos__" or "__invert__"))
                return MatrixBinary(sym.GetString(), p, ops);

            switch (name)
            {
                case "__neg__": return -(NDMatrix)ops[0];
                case "__pos__": return +(NDMatrix)ops[0];
                case "__invert__": return ~(NDMatrix)ops[0];
                case "__getitem__": return MatrixGetItem((NDMatrix)ops[0], p, ops);
                case "__setitem__": return MatrixSetItem((NDMatrix)ops[0], p, ops);
                case "__imul__":
                    return p.TryGetValue("py", out var imulPy)
                        ? ((NDMatrix)ops[0]).__imul__(PyBoxed(imulPy))
                        : ((NDMatrix)ops[0]).__imul__(ops[1]);
                case "__pow__": return MatrixPow((NDMatrix)ops[0], p, ops);
                case "__ipow__": return ((NDMatrix)ops[0]).__ipow__(p["n"].GetInt32());
                case "__rpow__":
                    return p.TryGetValue("py", out var rpowPy)
                        ? ((NDMatrix)ops[0]).__rpow__(PyBoxed(rpowPy))
                        : ((NDMatrix)ops[1]).__rpow__(ops[0]);
            }

            var m = (NDMatrix)ops[0];
            switch (name)
            {
                // ---- the matrix-only properties ----
                case "T": return m.T;
                case "H": return m.H;
                case "A": return m.A;
                case "A1": return m.A1;
                case "I": return m.I;
                case "getT": return m.getT();
                case "getH": return m.getH();
                case "getA": return m.getA();
                case "getA1": return m.getA1();
                case "getI": return m.getI();

                // ---- reductions ----
                case "sum": case "prod": case "mean": case "std": case "var": case "any": case "all":
                case "max": case "min": case "argmax": case "argmin": case "ptp":
                    return MatrixReduce(m, name, p, ops);

                // ---- matrix's own overrides ----
                case "squeeze":
                    return p.TryGetValue("axis", out var sqAxis) ? m.squeeze(sqAxis.GetInt32()) : m.squeeze();
                case "flatten": return m.flatten(p["order"].GetString()[0]);
                case "ravel":
                    return p.TryGetValue("order", out var rvOrder) ? m.ravel(rvOrder.GetString()[0]) : m.ravel();

                // ---- inherited methods ----
                case "copy":
                    return p.TryGetValue("order", out var cpOrder) ? m.copy(cpOrder.GetString()[0]) : m.copy();
                case "transpose":
                    return p.TryGetValue("axes", out var trAxes) ? m.transpose(ParseIntArray(trAxes)) : m.transpose();
                case "swapaxes": return m.swapaxes(p["a"].GetInt32(), p["b"].GetInt32());
                case "mT": return m.mT;
                case "view":
                    return p.TryGetValue("dtype", out var vwDtype)
                        ? m.view(FuzzCorpus.DtypeToTC(vwDtype.GetString()))
                        : m.view();
                case "diagonal":
                    return p.TryGetValue("offset", out var dgOffset) ? m.diagonal(dgOffset.GetInt32()) : m.diagonal();
                case "trace": return m.trace();
                case "conj": return m.conj();
                case "conjugate": return m.conjugate();
                case "real": return m.real;
                case "imag": return m.imag;
                case "astype":
                    return p.TryGetValue("copy", out var asCopy)
                        ? m.astype(FuzzCorpus.DtypeToTC(p["dtype"].GetString()), asCopy.GetBoolean())
                        : m.astype(FuzzCorpus.DtypeToTC(p["dtype"].GetString()));
                case "cumsum": return m.cumsum(MatrixAxis(p));
                case "cumprod": return m.cumprod(MatrixAxis(p));
                case "argsort":
                    // No "axis" key is NumPy's default (the last axis); an explicit null is axis=None.
                    return p.ContainsKey("axis") ? m.argsort(MatrixAxis(p)) : m.argsort();
                case "repeat": return m.repeat(p["repeats"].GetInt32(), MatrixAxis(p));
                case "round":
                    return p.TryGetValue("decimals", out var rdDecimals) ? m.round(rdDecimals.GetInt32()) : m.round();
                case "reshape": return m.reshape(ParseLongArray(p["shape"]));
                case "take":
                {
                    string mode = p.TryGetValue("mode", out var tkMode) ? tkMode.GetString() : "raise";
                    // A Python int index is a C# integer, which converts to a 0-d index array.
                    NDArray indices = p.TryGetValue("index", out var tkIndex)
                        ? (NDArray)tkIndex.GetInt64()
                        : ops[p["indices"].GetInt32()];
                    return m.take(indices, MatrixAxis(p), mode);
                }
                case "compress": return m.compress(ops[p["cond"].GetInt32()], MatrixAxis(p));
                case "clip": return m.clip(ops[p["lo"].GetInt32()], ops[p["hi"].GetInt32()]);
                case "dot": return m.dot(ops[1]);
                case "shape":
                    m.shape = ParseLongArray(p["shape"]);
                    return m;
                case "byteswap":
                    return m.byteswap(p.TryGetValue("inplace", out var bsInplace) && bsInplace.GetBoolean());
                default:
                    throw new NotSupportedException(op);
            }
        }

        /// <summary>The matrix tier's ops whose result is a tuple: iteration over the rows.</summary>
        /// <param name="op">The op key.</param>
        /// <param name="p">The case's params.</param>
        /// <param name="ops">The case's operands.</param>
        /// <returns>The rows, in order.</returns>
        /// <exception cref="NotSupportedException">The key is not a tuple key of the matrix tier.</exception>
        internal static NDArray[] ApplyMatrixTuple(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            if (op != "matrix.__iter__")
                throw new NotSupportedException(op);
            var rows = new List<NDArray>();
            foreach (NDArray row in (NDMatrix)ops[0])
                rows.Add(row);
            return rows.ToArray();
        }

        /// <summary>The matrix tier's ops whose result is text: <c>repr()</c> and <c>str()</c>.</summary>
        /// <param name="op">The op key.</param>
        /// <param name="p">The case's params.</param>
        /// <param name="ops">The case's operands.</param>
        /// <returns>The text.</returns>
        /// <exception cref="NotSupportedException">The key is not a text key of the matrix tier.</exception>
        internal static string ApplyMatrixText(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
            => op switch
            {
                "matrix.__repr__" => ops[0].ToString(true),
                "matrix.__str__" => ops[0].ToString(),
                _ => throw new NotSupportedException(op),
            };

        // ---- construction ---------------------------------------------------------------------------------

        /// <summary><c>np.matrix(data, dtype, copy)</c> from an operand, a literal or a Python list.</summary>
        /// <param name="p">Params: <c>src</c> (<c>op</c> / <c>text</c> / <c>list</c>), <c>dtype</c>?, <c>copy</c>?,
        ///     <c>text</c> or <c>data</c>.</param>
        /// <param name="ops">The operands (operand 0 for <c>src: op</c>).</param>
        /// <returns>The matrix.</returns>
        private static NDArray MatrixConstruct(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            DType dtype = p.TryGetValue("dtype", out var d) ? FuzzCorpus.DtypeToTC(d.GetString()) : null;
            bool copy = !p.TryGetValue("copy", out var c) || c.GetBoolean();
            return p["src"].GetString() switch
            {
                "op" => np.matrix(ops[0], dtype, copy),
                "text" => np.matrix(p["text"].GetString(), dtype, copy),
                // The object overload: a Python list is an object[], a Python scalar a boxed long / double / bool.
                _ => np.matrix(PyValue(p["data"]), dtype, copy),
            };
        }

        /// <summary>A Python value the generator tagged (<c>_leaf</c>) as the C# value of the house mapping.</summary>
        /// <param name="e">A JSON array (a list) or a tagged scalar.</param>
        /// <returns>An <c>object[]</c> for a list; a boxed <see cref="bool"/>, <see cref="long"/> (or
        ///     <see cref="BigInteger"/> past int64), <see cref="double"/> or <see cref="Complex"/> for a scalar.</returns>
        private static object PyValue(JsonElement e)
        {
            if (e.ValueKind == JsonValueKind.Array)
                return e.EnumerateArray().Select(PyValue).ToArray();
            if (e.TryGetProperty("b", out var b))
                return b.GetBoolean();
            if (e.TryGetProperty("i", out var i))
            {
                var big = BigInteger.Parse(i.GetString(), CultureInfo.InvariantCulture);
                return big >= long.MinValue && big <= long.MaxValue ? (object)(long)big : big;
            }
            if (e.TryGetProperty("f", out var f))
                return double.Parse(f.GetString(), CultureInfo.InvariantCulture);
            var parts = e.GetProperty("c").EnumerateArray().Select(x => double.Parse(x.GetString(), CultureInfo.InvariantCulture)).ToArray();
            return new Complex(parts[0], parts[1]);
        }

        /// <summary>A Python scalar operand (<c>params.py</c>) as the boxed C# literal of its kind.</summary>
        /// <param name="py"><c>{"k": "int" | "float" | "bool" | "complex", "v": value}</c>.</param>
        /// <returns>A boxed <see cref="long"/>, <see cref="double"/>, <see cref="bool"/> or <see cref="Complex"/>.</returns>
        private static object PyBoxed(JsonElement py)
        {
            var v = py.GetProperty("v");
            return py.GetProperty("k").GetString() switch
            {
                "int" => v.GetInt64(),
                "float" => v.GetDouble(),
                "bool" => v.GetBoolean(),
                _ => new Complex(v[0].GetDouble(), v[1].GetDouble()),
            };
        }

        /// <summary>
        ///     A Python scalar operand as the 0-d <see cref="NDMatrix"/> the C# compiler makes of a literal through
        ///     the implicit conversion (<c>m * 2</c>, <c>m[0] = 1.5</c>).
        /// </summary>
        /// <param name="py">The <c>params.py</c> record.</param>
        /// <returns>The scalar stand-in.</returns>
        /// <exception cref="NotSupportedException">The record names a kind the tier does not emit.</exception>
        private static NDMatrix PyScalar(JsonElement py)
            => PyBoxed(py) switch
            {
                long l => l,
                double d => d,
                bool b => b,
                Complex c => c,
                var other => throw new NotSupportedException(other?.GetType().Name),
            };

        /// <summary>
        ///     A 0-d value operand as the 0-d <see cref="NDMatrix"/> a C# number of the same dtype converts to (the
        ///     typed indexer setters take a matrix or a number).
        /// </summary>
        /// <param name="scalar">A 0-d array.</param>
        /// <returns>The scalar stand-in, of the operand's dtype.</returns>
        /// <exception cref="NotSupportedException">The operand's dtype has no NumPy counterpart (char, decimal).</exception>
        private static NDMatrix NumberOf(NDArray scalar)
            => scalar.typecode switch
            {
                NPTypeCode.Boolean => scalar.GetAtIndex<bool>(0),
                NPTypeCode.SByte => scalar.GetAtIndex<sbyte>(0),
                NPTypeCode.Byte => scalar.GetAtIndex<byte>(0),
                NPTypeCode.Int16 => scalar.GetAtIndex<short>(0),
                NPTypeCode.UInt16 => scalar.GetAtIndex<ushort>(0),
                NPTypeCode.Int32 => scalar.GetAtIndex<int>(0),
                NPTypeCode.UInt32 => scalar.GetAtIndex<uint>(0),
                NPTypeCode.Int64 => scalar.GetAtIndex<long>(0),
                NPTypeCode.UInt64 => scalar.GetAtIndex<ulong>(0),
                NPTypeCode.Half => scalar.GetAtIndex<Half>(0),
                NPTypeCode.Single => scalar.GetAtIndex<float>(0),
                NPTypeCode.Double => scalar.GetAtIndex<double>(0),
                NPTypeCode.Complex => scalar.GetAtIndex<Complex>(0),
                var other => throw new NotSupportedException(other.ToString()),
            };

        /// <summary>The <c>axis</c> param: absent or null is NumPy's <c>None</c>.</summary>
        /// <param name="p">The case's params.</param>
        /// <returns>The axis, or <c>null</c>.</returns>
        private static int? MatrixAxis(IReadOnlyDictionary<string, JsonElement> p)
            => p.TryGetValue("axis", out var a) && a.ValueKind != JsonValueKind.Null ? a.GetInt32() : null;

        // ---- indexing -------------------------------------------------------------------------------------

        /// <summary>One index token as a <see cref="Slice"/> (the tokens of an index of integers and slices).</summary>
        /// <param name="t">An int, <c>"..."</c>, <c>"newaxis"</c> or <c>{"s": [start, stop, step]}</c>.</param>
        /// <returns>The slice.</returns>
        private static Slice MatrixSlice(JsonElement t)
        {
            if (t.ValueKind == JsonValueKind.Number)
                return Slice.Index(t.GetInt64());
            if (t.ValueKind == JsonValueKind.String)
                return t.GetString() == "..." ? Slice.Ellipsis : Slice.NewAxis;
            var s = t.GetProperty("s");
            long? start = s[0].ValueKind == JsonValueKind.Null ? null : s[0].GetInt64();
            long? stop = s[1].ValueKind == JsonValueKind.Null ? null : s[1].GetInt64();
            return new Slice(start, stop, s[2].ValueKind == JsonValueKind.Null ? 1 : s[2].GetInt64());
        }

        /// <summary>One index token as an entry of the general (<c>object[]</c>) indexer.</summary>
        /// <param name="t">Any token of <c>gen_matrix_oracle._tok</c>.</param>
        /// <param name="ops">The operands (an <c>{"op": k}</c> token names operand k).</param>
        /// <returns>A boxed <see cref="int"/>, a <see cref="Slice"/>, an operand, a <c>long[]</c> or a <c>bool[]</c>.</returns>
        private static object MatrixIndexEntry(JsonElement t, NDArray[] ops)
        {
            if (t.ValueKind == JsonValueKind.Number)
                return t.GetInt32();
            if (t.ValueKind == JsonValueKind.Object)
            {
                if (t.TryGetProperty("op", out var k))
                    return ops[k.GetInt32()];
                if (t.TryGetProperty("l", out var l))
                    return l.EnumerateArray().Select(x => x.GetInt64()).ToArray();
                if (t.TryGetProperty("lb", out var lb))
                    return lb.EnumerateArray().Select(x => x.GetBoolean()).ToArray();
            }
            return MatrixSlice(t);
        }

        /// <summary><c>m[index]</c> through the indexer the case names in <c>params.via</c>.</summary>
        /// <param name="m">The matrix.</param>
        /// <param name="p">Params: <c>index</c> (tokens), <c>via</c> (<c>obj</c> / <c>typed</c> / <c>str</c>), <c>str</c>.</param>
        /// <param name="ops">The operands.</param>
        /// <returns>The selection.</returns>
        private static NDArray MatrixGetItem(NDMatrix m, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var tokens = p["index"].EnumerateArray().ToArray();
            switch (p["via"].GetString())
            {
                case "str":
                    return m[p["str"].GetString()];
                case "typed":
                    // The typed indexers a C# caller reaches with integers and slices.
                    if (tokens.Length == 1 && tokens[0].ValueKind == JsonValueKind.Number)
                        return m[tokens[0].GetInt32()];
                    if (tokens.Length == 2 && tokens.All(t => t.ValueKind == JsonValueKind.Number))
                        return m[tokens[0].GetInt32(), tokens[1].GetInt32()];
                    return m[tokens.Select(MatrixSlice).ToArray()];
                default:
                    return m[tokens.Select(t => MatrixIndexEntry(t, ops)).ToArray()];
            }
        }

        /// <summary><c>m[index] = value</c>; the result is the matrix after the assignment.</summary>
        /// <param name="m">The matrix.</param>
        /// <param name="p">Params: <c>index</c>, <c>via</c> (<c>setitem</c> / <c>typed</c> / <c>str</c>), <c>str</c>.</param>
        /// <param name="ops">The operands; the last one is the value.</param>
        /// <returns><paramref name="m"/>.</returns>
        private static NDArray MatrixSetItem(NDMatrix m, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var tokens = p["index"].EnumerateArray().ToArray();
            NDArray value = ops[ops.Length - 1];
            string via = p["via"].GetString();
            if (via == "setitem")
            {
                m.__setitem__(tokens.Select(t => MatrixIndexEntry(t, ops)).ToArray(), value);
                return m;
            }

            // The typed setters: a full integer index stores an NDArray-typed element; every other indexer is typed
            // NDMatrix, so it takes a matrix value or a C# number (a 0-d operand stands for the number).
            if (via == "typed" && tokens.Length == 2 && tokens.All(t => t.ValueKind == JsonValueKind.Number))
            {
                m[tokens[0].GetInt32(), tokens[1].GetInt32()] = value;
                return m;
            }

            NDMatrix typedValue = value as NDMatrix ?? NumberOf(value);
            if (via == "str")
                m[p["str"].GetString()] = typedValue;
            else if (tokens.Length == 1 && tokens[0].ValueKind == JsonValueKind.Number)
                m[tokens[0].GetInt32()] = typedValue;
            else
                m[tokens.Select(MatrixSlice).ToArray()] = typedValue;
            return m;
        }

        // ---- operators ------------------------------------------------------------------------------------

        /// <summary>
        ///     One binary operator applied to (operand 0, operand 1) in that order, through the overload the static
        ///     types of a C# caller would bind: (matrix, matrix), (matrix, array), (array, matrix), or the object
        ///     forms; a Python scalar (<c>params.py</c>) through the implicit conversion to a 0-d matrix.
        /// </summary>
        /// <param name="sym">The C# operator.</param>
        /// <param name="p">Params: <c>py</c> + <c>side</c>, <c>via</c> (<c>object</c> / <c>method</c>).</param>
        /// <param name="ops">The operands.</param>
        /// <returns>The result.</returns>
        private static NDArray MatrixBinary(string sym, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            if (p.TryGetValue("py", out var py))
            {
                var m = (NDMatrix)ops[0];
                NDMatrix literal = PyScalar(py);
                return p["side"].GetString() == "right" ? MatrixMatrix(sym, m, literal) : MatrixMatrix(sym, literal, m);
            }

            NDArray left = ops[0], right = ops[1];
            string via = p.TryGetValue("via", out var v) ? v.GetString() : null;
            if (via == "method")
            {
                // Python's m.__mul__(other) and m.__rmul__(other): the matrix is the receiver.
                return left is NDMatrix receiver ? receiver.__mul__(right) : ((NDMatrix)right).__rmul__(left);
            }

            if (left is NDMatrix lm && right is NDMatrix rm)
                return MatrixMatrix(sym, lm, rm);
            if (left is NDMatrix lm2)
                return via == "object" ? MatrixObject(sym, lm2, right) : MatrixArray(sym, lm2, right);
            return via == "object" ? ObjectMatrix(sym, left, (NDMatrix)right) : ArrayMatrix(sym, left, (NDMatrix)right);
        }

        /// <summary>The (matrix, matrix) operator overloads.</summary>
        /// <param name="sym">The operator.</param>
        /// <param name="a">The left operand.</param>
        /// <param name="b">The right operand.</param>
        /// <returns>The result.</returns>
        /// <exception cref="NotSupportedException">The operator is not one this overload family has.</exception>
        private static NDArray MatrixMatrix(string sym, NDMatrix a, NDMatrix b) => sym switch
        {
            "*" => a * b, "+" => a + b, "-" => a - b, "/" => a / b, "%" => a % b,
            "&" => a & b, "|" => a | b, "^" => a ^ b, "<<" => a << b, ">>" => a >> b,
            "==" => a == b, "!=" => a != b, "<" => a < b, ">" => a > b, "<=" => a <= b, ">=" => a >= b,
            _ => throw new NotSupportedException(sym),
        };

        /// <summary>The (matrix, array) operator overloads.</summary>
        /// <param name="sym">The operator.</param>
        /// <param name="a">The left operand.</param>
        /// <param name="b">The right operand.</param>
        /// <returns>The result.</returns>
        /// <exception cref="NotSupportedException">The operator is not one this overload family has.</exception>
        private static NDArray MatrixArray(string sym, NDMatrix a, NDArray b) => sym switch
        {
            "*" => a * b, "+" => a + b, "-" => a - b, "/" => a / b, "%" => a % b,
            "&" => a & b, "|" => a | b, "^" => a ^ b, "<<" => a << b, ">>" => a >> b,
            "==" => a == b, "!=" => a != b, "<" => a < b, ">" => a > b, "<=" => a <= b, ">=" => a >= b,
            _ => throw new NotSupportedException(sym),
        };

        /// <summary>The (array, matrix) operator overloads (C# allows none for the shifts).</summary>
        /// <param name="sym">The operator.</param>
        /// <param name="a">The left operand.</param>
        /// <param name="b">The right operand.</param>
        /// <returns>The result.</returns>
        /// <exception cref="NotSupportedException">The operator is not one this overload family has.</exception>
        private static NDArray ArrayMatrix(string sym, NDArray a, NDMatrix b) => sym switch
        {
            "*" => a * b, "+" => a + b, "-" => a - b, "/" => a / b, "%" => a % b,
            "&" => a & b, "|" => a | b, "^" => a ^ b,
            "==" => a == b, "!=" => a != b, "<" => a < b, ">" => a > b, "<=" => a <= b, ">=" => a >= b,
            _ => throw new NotSupportedException(sym),
        };

        /// <summary>The (matrix, object) operator overloads.</summary>
        /// <param name="sym">The operator.</param>
        /// <param name="a">The left operand.</param>
        /// <param name="b">The right operand, as an <see cref="object"/>.</param>
        /// <returns>The result.</returns>
        /// <exception cref="NotSupportedException">The operator is not one this overload family has.</exception>
        private static NDArray MatrixObject(string sym, NDMatrix a, object b) => sym switch
        {
            "*" => a * b, "+" => a + b, "-" => a - b, "/" => a / b, "%" => a % b,
            "&" => a & b, "|" => a | b, "^" => a ^ b, "<<" => a << b, ">>" => a >> b,
            "==" => a == b, "!=" => a != b, "<" => a < b, ">" => a > b, "<=" => a <= b, ">=" => a >= b,
            _ => throw new NotSupportedException(sym),
        };

        /// <summary>The (object, matrix) operator overloads.</summary>
        /// <param name="sym">The operator.</param>
        /// <param name="a">The left operand, as an <see cref="object"/>.</param>
        /// <param name="b">The right operand.</param>
        /// <returns>The result.</returns>
        /// <exception cref="NotSupportedException">The operator is not one this overload family has.</exception>
        private static NDArray ObjectMatrix(string sym, object a, NDMatrix b) => sym switch
        {
            "*" => a * b, "+" => a + b, "-" => a - b, "/" => a / b, "%" => a % b,
            "&" => a & b, "|" => a | b, "^" => a ^ b,
            "==" => a == b, "!=" => a != b, "<" => a < b, ">" => a > b, "<=" => a <= b, ">=" => a >= b,
            _ => throw new NotSupportedException(sym),
        };

        /// <summary><c>m ** exponent</c> for an integer, a tagged Python value, or an operand.</summary>
        /// <param name="m">The matrix.</param>
        /// <param name="p">Params: <c>n</c>, or <c>exp</c> (a tagged Python value), or <c>exp_op</c> (an operand).</param>
        /// <param name="ops">The operands.</param>
        /// <returns>The power.</returns>
        private static NDArray MatrixPow(NDMatrix m, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            if (p.TryGetValue("n", out var n))
                return m.__pow__(n.GetInt32());
            if (p.TryGetValue("exp_op", out var k))
                return m.__pow__((object)ops[k.GetInt32()]);
            var exp = p["exp"];
            return exp.GetProperty("k").GetString() == "list"
                ? m.__pow__(exp.GetProperty("v").EnumerateArray().Select(x => x.GetInt64()).ToArray())
                : m.__pow__(PyBoxed(exp));
        }

        // ---- reductions -----------------------------------------------------------------------------------

        /// <summary>One matrix reduction through the overload its arguments bind.</summary>
        /// <param name="m">The matrix.</param>
        /// <param name="name">The reduction.</param>
        /// <param name="p">Params: <c>axis</c> (null = None), <c>dtype</c>?, <c>ddof</c>?, <c>out</c>? (an operand
        ///     position), <c>nullable</c>? (use the <c>int?</c> overload for a given axis).</param>
        /// <param name="ops">The operands.</param>
        /// <returns>The result: a 0-d array for <c>axis=None</c>, the matrix otherwise.</returns>
        /// <exception cref="NotSupportedException">The name is not a matrix reduction.</exception>
        private static NDArray MatrixReduce(NDMatrix m, string name, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            int? axis = MatrixAxis(p);
            DType dtype = p.TryGetValue("dtype", out var d) ? FuzzCorpus.DtypeToTC(d.GetString()) : null;
            int ddof = p.TryGetValue("ddof", out var dd) ? dd.GetInt32() : 0;
            NDArray outArray = p.TryGetValue("out", out var o) ? ops[o.GetInt32()] : null;
            var outMatrix = outArray as NDMatrix;
            // The (int axis) overloads return NDMatrix; the (int? axis) ones the scalar or the matrix.
            bool typed = axis.HasValue && !(p.TryGetValue("nullable", out var nl) && nl.GetBoolean());
            switch (name)
            {
                case "sum": return typed ? m.sum(axis.Value, dtype, outMatrix) : m.sum(axis, dtype, outMatrix);
                case "prod": return typed ? m.prod(axis.Value, dtype, outMatrix) : m.prod(axis, dtype, outMatrix);
                case "mean": return typed ? m.mean(axis.Value, dtype, outMatrix) : m.mean(axis, dtype, outMatrix);
                case "var": return typed ? m.var(axis.Value, dtype, outMatrix, ddof) : m.var(axis, dtype, outMatrix, ddof);
                case "std": return typed ? m.std(axis.Value, dtype, outMatrix, ddof) : m.std(axis, dtype, outMatrix, ddof);
                case "any": return typed ? m.any(axis.Value, outArray) : m.any(axis, outArray);
                case "all": return typed ? m.all(axis.Value, outArray) : m.all(axis, outArray);
                case "max": return typed ? m.max(axis.Value, outMatrix) : m.max(axis, outMatrix);
                case "min": return typed ? m.min(axis.Value, outMatrix) : m.min(axis, outMatrix);
                case "ptp": return typed ? m.ptp(axis.Value, outArray) : m.ptp(axis, outArray);
                // The flat form is the inherited member, which returns a long (NumPy: an int64 scalar).
                case "argmax": return axis.HasValue ? m.argmax(axis.Value, outArray) : NDArray.Scalar(m.argmax());
                case "argmin": return axis.HasValue ? m.argmin(axis.Value, outArray) : NDArray.Scalar(m.argmin());
                default: throw new NotSupportedException(name);
            }
        }
    }
}
