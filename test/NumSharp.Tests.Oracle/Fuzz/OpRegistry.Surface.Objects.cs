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
    ///     The surface tier's object members (<c>gen_surface_oracle.gen_objects</c>): the <c>np.broadcast</c>,
    ///     <c>flatiter</c> and <c>np.nditer</c> attributes after a number of steps, the print options
    ///     (<c>set_printoptions</c> / <c>printoptions</c> / <c>get_printoptions</c>), the buffer size
    ///     (<c>setbufsize</c> / <c>getbufsize</c>), <c>ndarray.data</c> (the memoryview), <c>dtype.type</c> and
    ///     <c>np.frompyfunc</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A case's params name the object's construction (operands, flags, options), the steps to take and the member
    ///     to read; each adapter performs exactly the NumPy statements of the generator's twin (<c>_nditer_attr</c>,
    ///     <c>_bcast_reset</c>, ...) and converts only what C# returns in another shape: a live iterator view is copied
    ///     so it outlives the iterator, a boxed element becomes a 0-d array of its operand's dtype, a CLR type becomes
    ///     NumPy's class repr.
    ///     </para>
    ///     <para>
    ///     Global state (the print options, the buffer size) is restored in a <c>finally</c> / scope however the case
    ///     ends, so one case cannot leak its options into the next — NumPy's twin restores the same way.
    ///     </para>
    /// </remarks>
    public static partial class OpRegistry
    {
        // ---- dispatch ---------------------------------------------------------------------------------------------

        /// <summary>The object members whose result is one array or a scalar (a 0-d array).</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="result">NumSharp's result when the key is one of these members.</param>
        /// <returns>True when the key was handled.</returns>
        private static bool TrySurfaceObject(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops,
                                             out NDArray result)
        {
            result = null;
            switch (op)
            {
                case "broadcast.nd":
                case "broadcast.ndim":
                case "broadcast.size":
                case "broadcast.numiter":
                case "broadcast.index":
                {
                    using var b = BroadcastAfter(p, ops);
                    result = NDArray.Scalar(op switch
                    {
                        "broadcast.nd" => (long)b.nd,
                        "broadcast.ndim" => (long)b.ndim,
                        "broadcast.size" => b.size,
                        "broadcast.numiter" => (long)b.numiter,
                        _ => (long)b.index,
                    });
                    return true;
                }
                case "flatiter.index":
                    result = NDArray.Scalar(FlatAfter(p, ops[0]).index);
                    return true;
                case "flatiter.copy":
                    result = ops[0].flatiter.copy();
                    return true;
                case "flatiter.base":
                    result = ops[0].flatiter.Base;
                    return true;
                case "setbufsize":
                case "getbufsize":
                    result = BufferSize(op, p);
                    return true;
                case "ndarray.data":
                    result = MemoryViewAttribute(ops[0], SurfStr(p, "attr"));
                    return true;
                case "frompyfunc":
                    result = FromPyFunc(p, ops)[0];
                    return true;
            }
            if (op.StartsWith("nditer.", StringComparison.Ordinal))
            {
                result = op == "nditer.close" ? NditerClosed(p, ops)[0] : NditerMember(p, ops)[0];
                return true;
            }
            return false;
        }

        /// <summary>The object members whose result is a tuple of arrays.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The slots, in NumPy's tuple order.</returns>
        /// <exception cref="NotSupportedException">The key is not a surface tuple op (an unregistered key).</exception>
        private static NDArray[] SurfaceObjectTuple(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            switch (op)
            {
                case "broadcast.reset": return BroadcastReset(p, ops);
                case "broadcast.iters": return BroadcastIters(p, ops);
                case "frompyfunc": return FromPyFunc(p, ops);
            }
            if (op == "nditer.close")
                return NditerClosed(p, ops);
            if (op.StartsWith("nditer.", StringComparison.Ordinal))
                return NditerMember(p, ops);
            throw new NotSupportedException($"tuple op '{op}' is not registered in OpRegistry");
        }

        /// <summary>The object members whose result is text.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="text">NumSharp's text when the key is one of these members.</param>
        /// <returns>True when the key was handled.</returns>
        private static bool TrySurfaceObjectText(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops,
                                                 out string text)
        {
            text = null;
            switch (op)
            {
                case "flatiter.coords":
                    text = PyTuple(FlatAfter(p, ops[0]).coords.Select(c => (object)c));
                    return true;
                case "nditer.iterrange":
                case "nditer.dtypes":
                    text = NditerText(p, ops);
                    return true;
                case "set_printoptions":
                case "printoptions":
                case "get_printoptions":
                    text = PrintOptionsCase(op, p, ops);
                    return true;
                case "ndarray.data":
                {
                    var mv = ops[0].data;
                    text = SurfStr(p, "attr") switch
                    {
                        "hex" => mv.hex(),
                        "format" => mv.format,
                        var a => throw new NotSupportedException($"memoryview text attribute '{a}'"),
                    };
                    return true;
                }
            }
            return false;
        }

        // ---- np.broadcast -----------------------------------------------------------------------------------------

        /// <summary>np.broadcast over the operands, advanced <c>params.steps</c> steps (0 when absent).</summary>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The operands (none is a 0-d broadcast).</param>
        /// <returns>The broadcast object; the caller disposes it (it owns its broadcast views).</returns>
        private static np.Broadcast BroadcastAfter(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var b = np.broadcast(ops);
            int steps = p.TryGetValue("steps", out var s) ? s.GetInt32() : 0;
            for (int i = 0; i < steps; i++)
                b.MoveNext();
            return b;
        }

        /// <summary>
        ///     broadcast.reset after the steps: [index] + the first value tuple read after the reset, each value a 0-d array
        ///     of its operand's dtype (no value tuple for an empty broadcast).
        /// </summary>
        private static NDArray[] BroadcastReset(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            using var b = BroadcastAfter(p, ops);
            b.reset();
            var slots = new List<NDArray> { NDArray.Scalar((long)b.index) };
            if (b.size > 0 && b.MoveNext())
                for (int i = 0; i < ops.Length; i++)
                    slots.Add(NDArray.Scalar(b.Current[i], ops[i].dtype));
            return slots.ToArray();
        }

        /// <summary>
        ///     broadcast.iters after the steps: each per-operand iterator's values (what NumPy's flatiter yields from its
        ///     position on), one 1-D array of the operand's dtype per operand.
        /// </summary>
        private static NDArray[] BroadcastIters(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            using var b = BroadcastAfter(p, ops);
            var iters = b.iters;
            var slots = new NDArray[iters.Length];
            for (int i = 0; i < iters.Length; i++)
            {
                var values = iters[i].ToList();
                var r = new NDArray(ops[i].dtype, Shape.Vector(values.Count));
                for (int j = 0; j < values.Count; j++)
                    r.SetAtIndex(values[j], j);
                slots[i] = r;
            }
            return slots;
        }

        // ---- flatiter ---------------------------------------------------------------------------------------------

        /// <summary>a.flatiter (NumPy's a.flat) advanced <c>params.steps</c> steps.</summary>
        private static np.FlatIterator FlatAfter(IReadOnlyDictionary<string, JsonElement> p, NDArray a)
        {
            var f = a.flatiter;
            int steps = p["steps"].GetInt32();
            for (int i = 0; i < steps; i++)
                f.next();
            return f;
        }

        // ---- np.nditer --------------------------------------------------------------------------------------------

        /// <summary>
        ///     np.nditer over the operands with the case's keyword params (flags, op_flags, op_dtypes — one per
        ///     operand, as the generator repeats them —, order, casting); a single operand takes the single-operand
        ///     overload, as <c>np.nditer(a)</c> does.
        /// </summary>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The operands.</param>
        /// <returns>The iterator; the caller disposes it.</returns>
        private static np.NDIterator SurfNditer(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            string[] flags = p["flags"].EnumerateArray().Select(f => f.GetString()).ToArray();
            char order = p.TryGetValue("order", out var o) ? o.GetString()[0] : 'K';
            string casting = p.TryGetValue("casting", out var c) ? c.GetString() : "safe";
            DType[] opDtypes = p.TryGetValue("op_dtypes", out var od) && od.ValueKind == JsonValueKind.Array
                ? Enumerable.Repeat(od.EnumerateArray().Select(d => np.dtype(d.GetString())).ToArray(), ops.Length)
                    .SelectMany(x => x).ToArray()
                : null;
            string[][] opFlags = p.TryGetValue("op_flags", out var of) && of.ValueKind == JsonValueKind.Array
                ? of.EnumerateArray().Select(l => l.EnumerateArray().Select(x => x.GetString()).ToArray()).ToArray()
                : null;
            // buffersize: absent or 0 is NumPy's default (the iterator picks it).
            long bufferSize = p.TryGetValue("buffersize", out var bs) ? bs.GetInt64() : 0;
            return ops.Length == 1
                ? np.nditer(ops[0], flags, opFlags?[0], opDtypes, order, casting, buffersize: bufferSize)
                : np.nditer(ops, flags, opFlags, opDtypes, order, casting, buffersize: bufferSize);
        }

        /// <summary>nditer.value as tuple slots, each copied so it outlives the iterator (the live views alias it).</summary>
        private static List<NDArray> NditerValues(np.NDIterator it) => it.value.Select(v => v.copy()).ToList();

        /// <summary>
        ///     One np.nditer member after the case's steps (<c>params.attr</c>), shaped as the generator's
        ///     <c>_nditer_attr</c> shapes NumPy's: a scalar or an array is a one-slot list.
        /// </summary>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The operands.</param>
        /// <returns>The result slots.</returns>
        /// <exception cref="NotSupportedException">An attribute the generator does not emit.</exception>
        private static NDArray[] NditerMember(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            using var it = SurfNditer(p, ops);
            int steps = p.TryGetValue("steps", out var s) ? s.GetInt32() : 0;
            for (int i = 0; i < steps; i++)
                it.iternext();
            // "next": elements consumed through the iteration protocol — C#'s MoveNext on the iterator's enumerator,
            // NumPy's next(it) (which advances on entry). The enumerator shares the iterator's cursor and owns nothing.
            int nexts = p.TryGetValue("next", out var nx) ? nx.GetInt32() : 0;
            if (nexts > 0)
            {
                var en = it.GetEnumerator();
                for (int i = 0; i < nexts; i++)
                    en.MoveNext();
            }
            string attr = SurfStr(p, "attr");
            switch (attr)
            {
                case "iternext": return new[] { NDArray.Scalar(it.iternext()) };
                case "value": return NditerValues(it).ToArray();
                case "operands": return it.operands.Select(x => x.copy()).ToArray();
                case "itviews": return it.itviews.Select(x => x.copy()).ToArray();
                case "copy":
                {
                    using var c = it.copy();
                    var slots = new List<NDArray> { NDArray.Scalar(c.iterindex) };
                    slots.AddRange(NditerValues(c));
                    return slots.ToArray();
                }
                case "reset":
                {
                    it.reset();
                    var slots = new List<NDArray> { NDArray.Scalar(it.iterindex) };
                    slots.AddRange(NditerValues(it));
                    return slots.ToArray();
                }
                case "remove_axis":
                    it.remove_axis(SurfInt(p, "axis"));
                    return new[] { np.array(it.shape) };
                case "remove_multi_index":
                {
                    // The values at the first four positions after the removal show the order the iterator follows.
                    it.remove_multi_index();
                    var slots = new List<NDArray> { NDArray.Scalar(it.has_multi_index), np.array(it.shape) };
                    for (int i = 0; i < 4; i++)
                    {
                        slots.AddRange(NditerValues(it));
                        it.iternext();
                    }
                    return slots.ToArray();
                }
                case "enable_external_loop":
                    it.enable_external_loop();
                    return NditerValues(it).ToArray();
                case "shape": return new[] { np.array(it.shape) };
                // NumPy's multi_index tuple; NumSharp returns the same coordinates as a long[].
                case "multi_index": return new[] { np.array(it.multi_index) };
                case "finished": return new[] { NDArray.Scalar(it.finished) };
                case "has_index": return new[] { NDArray.Scalar(it.has_index) };
                case "has_multi_index": return new[] { NDArray.Scalar(it.has_multi_index) };
                case "has_delayed_bufalloc": return new[] { NDArray.Scalar(it.has_delayed_bufalloc) };
                case "iterationneedsapi": return new[] { NDArray.Scalar(it.iterationneedsapi) };
                case "ndim": return new[] { NDArray.Scalar((long)it.ndim) };
                case "nop": return new[] { NDArray.Scalar((long)it.nop) };
                case "itersize": return new[] { NDArray.Scalar(it.itersize) };
                case "iterindex": return new[] { NDArray.Scalar(it.iterindex) };
                default: throw new NotSupportedException($"nditer attribute '{attr}'");
            }
        }

        /// <summary>nditer.iterrange / nditer.dtypes as NumPy's tuple repr.</summary>
        private static string NditerText(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            using var it = SurfNditer(p, ops);
            if (SurfStr(p, "attr") == "iterrange")
            {
                var (start, end) = it.iterrange;
                return PyTuple(new object[] { start, end });
            }
            var reprs = it.dtypes.Select(tc => DType.From(tc).ToString(true)).ToList();
            return reprs.Count == 1 ? "(" + reprs[0] + ",)" : "(" + string.Join(", ", reprs) + ")";
        }

        /// <summary>
        ///     np.nditer closed, then one member read (<c>params.after</c>): NumPy's answer for a closed iterator; or, for a
        ///     <c>params.writeback</c> case, the operand after the write-back (<see cref="NditerWriteback"/>).
        /// </summary>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The operands.</param>
        /// <returns>The result slots (one array).</returns>
        /// <exception cref="NotSupportedException">A member the generator does not emit.</exception>
        private static NDArray[] NditerClosed(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            if (p.ContainsKey("writeback"))
                return new[] { NditerWriteback(p, ops) };
            var it = SurfNditer(p, ops);
            it.close();
            return SurfStr(p, "after") switch
            {
                "iternext" => new[] { NDArray.Scalar(it.iternext()) },
                "value" => NditerValues(it).ToArray(),
                "ndim" => new[] { NDArray.Scalar((long)it.ndim) },
                "iterindex" => new[] { NDArray.Scalar(it.iterindex) },
                var a => throw new NotSupportedException($"nditer member after close '{a}'"),
            };
        }

        /// <summary>
        ///     The write-back contract of close(): an iterator over the operand with the case's params (a copy for
        ///     updateifcopy, buffers, or the operand itself), <c>x = x * 2 + 1</c> written through every element (or
        ///     external-loop chunk) it yields, then the operand copied after the loop (<c>writeback: "before"</c> close) or
        ///     after close() called twice (<c>"after"</c>) — the generator's <c>_nditer_writeback</c>.
        /// </summary>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The operands (one).</param>
        /// <returns>A copy of the operand at that point.</returns>
        /// <exception cref="ValueError">A read-only operand with a write flag ("operand array with iterator write flag set
        /// is read-only", NumPy's ValueError text).</exception>
        /// <exception cref="ArgumentException">NumSharp's "Casting between different dtypes requires the BUFFERED flag"
        /// for an unbuffered updateifcopy cast, where NumPy makes the write-back copy (or raises its own TypeError for a
        /// same_kind int32 operand) — gap SU41, carved.</exception>
        /// <exception cref="InvalidCastException">A buffered cast the casting rule refuses ("Iterator operand 0 dtype
        /// could not be cast …").</exception>
        /// <exception cref="NotSupportedException">An element dtype <see cref="SetElement"/> does not list.</exception>
        /// <remarks>The values are small integers, so the double arithmetic here equals NumPy's in the iterator's
        /// float64 (or the operand's own) dtype exactly.</remarks>
        private static NDArray NditerWriteback(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var it = SurfNditer(p, ops);
            foreach (var v in it)
            {
                var x = v[0];
                for (long i = 0; i < x.size; i++)
                    SetElement(x, i, Convert.ToDouble(x.GetAtIndex(i), CultureInfo.InvariantCulture) * 2 + 1);
            }
            if (SurfStr(p, "writeback") == "before")
            {
                // The snapshot comes BEFORE close(): it shows which writes reached the operand during the loop (every one
                // for a plain operand, every buffer flush for a buffered one, none yet for an updateifcopy copy, whose
                // write-back is close()'s job).
                var snapshot = ops[0].copy();
                it.close();
                return snapshot;
            }
            // "after": close() twice — the write-back must happen exactly once and the second close() must be a no-op.
            it.close();
            it.close();
            return ops[0].copy();
        }

        /// <summary>
        ///     Writes <paramref name="value"/> into element <paramref name="i"/> of <paramref name="x"/> as x's OWN element
        ///     type. Never <c>x.SetAtIndex(double, i)</c> on a non-float64 array: the generic SetAtIndex&lt;T&gt; checks T
        ///     against the dtype only with a Debug.Assert, so in Release it writes sizeof(T) bytes at index * sizeof(T) —
        ///     into the wrong element, and past the buffer's end for the last one.
        /// </summary>
        /// <param name="x">The (writeable) array or view the iterator yielded.</param>
        /// <param name="i">The flat element index.</param>
        /// <param name="value">The value (a small integer here, exact in every listed dtype).</param>
        /// <exception cref="NotSupportedException">A dtype the writeback cases do not use.</exception>
        private static void SetElement(NDArray x, long i, double value)
        {
            switch (x.typecode)
            {
                case NPTypeCode.Double: x.SetAtIndex(value, i); break;
                case NPTypeCode.Single: x.SetAtIndex((float)value, i); break;
                case NPTypeCode.Int32: x.SetAtIndex((int)value, i); break;
                case NPTypeCode.Int64: x.SetAtIndex((long)value, i); break;
                case NPTypeCode.Int16: x.SetAtIndex((short)value, i); break;
                case NPTypeCode.Byte: x.SetAtIndex((byte)value, i); break;
                default: throw new NotSupportedException($"writeback element dtype {x.typecode}");
            }
        }

        // ---- print options and buffer size ------------------------------------------------------------------------

        /// <summary>The print options of a case (<c>params.opts</c>) applied by NumSharp's set_printoptions.</summary>
        private static void SetPrintOptions(JsonElement opts)
        {
            np.set_printoptions(
                precision: OptInt(opts, "precision"), threshold: OptInt(opts, "threshold"),
                edgeitems: OptInt(opts, "edgeitems"), linewidth: OptInt(opts, "linewidth"),
                suppress: opts.TryGetProperty("suppress", out var su) ? su.GetBoolean() : null,
                nanstr: opts.TryGetProperty("nanstr", out var ns) ? ns.GetString() : null,
                infstr: opts.TryGetProperty("infstr", out var inf) ? inf.GetString() : null,
                sign: opts.TryGetProperty("sign", out var sg) ? sg.GetString()[0] : null,
                floatmode: opts.TryGetProperty("floatmode", out var fm) ? fm.GetString() : null);
        }

        /// <summary>The same options as np.printoptions' context (restored when the scope is disposed).</summary>
        private static IDisposable PrintOptionsScope(JsonElement opts)
            => np.printoptions(
                precision: OptInt(opts, "precision"), threshold: OptInt(opts, "threshold"),
                edgeitems: OptInt(opts, "edgeitems"), linewidth: OptInt(opts, "linewidth"),
                suppress: opts.TryGetProperty("suppress", out var su) ? su.GetBoolean() : null,
                nanstr: opts.TryGetProperty("nanstr", out var ns) ? ns.GetString() : null,
                infstr: opts.TryGetProperty("infstr", out var inf) ? inf.GetString() : null,
                sign: opts.TryGetProperty("sign", out var sg) ? sg.GetString()[0] : null,
                floatmode: opts.TryGetProperty("floatmode", out var fm) ? fm.GetString() : null);

        /// <summary>An optional int property of an options object.</summary>
        private static int? OptInt(JsonElement opts, string key)
            => opts.TryGetProperty(key, out var v) ? v.GetInt32() : null;

        /// <summary>
        ///     set_printoptions (then array2string, the previous options restored by an outer np.printoptions() scope),
        ///     printoptions (array2string inside the context, or after it exited) and get_printoptions (one value, or the
        ///     sorted key list, inside the context).
        /// </summary>
        private static string PrintOptionsCase(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var opts = p["opts"];
            if (op == "set_printoptions")
            {
                using (np.printoptions())
                {
                    SetPrintOptions(opts);
                    return np.array2string(ops[0]);
                }
            }
            if (op == "printoptions")
            {
                if (SurfStr(p, "phase") == "after")
                {
                    using (PrintOptionsScope(opts)) { }
                    return np.array2string(ops[0]);
                }
                using (PrintOptionsScope(opts))
                    return np.array2string(ops[0]);
            }
            using (PrintOptionsScope(opts))
            {
                var d = np.get_printoptions();
                string key = SurfNullableStr(p, "key");
                return key is null
                    ? PyRepr(d.Keys.OrderBy(k => k, StringComparer.Ordinal).Cast<object>().ToList())
                    : PyRepr(d[key]);
            }
        }

        /// <summary>setbufsize's return value or getbufsize after it; the previous size is restored either way.</summary>
        private static NDArray BufferSize(string op, IReadOnlyDictionary<string, JsonElement> p)
        {
            if (p["size"].ValueKind == JsonValueKind.Null)
                return NDArray.Scalar(np.getbufsize());
            long prev = np.getbufsize();
            try
            {
                long old = np.setbufsize(p["size"].GetInt64());
                return NDArray.Scalar(op == "setbufsize" ? old : np.getbufsize());
            }
            finally
            {
                np.setbufsize(prev);
            }
        }

        // ---- ndarray.data, dtype.type, frompyfunc -----------------------------------------------------------------

        /// <summary>One memoryview attribute of <c>a.data</c> as the generator's <c>_memview_attr</c> records it.</summary>
        private static NDArray MemoryViewAttribute(NDArray a, string attr)
        {
            var mv = a.data;
            return attr switch
            {
                "tobytes" => np.array(mv.tobytes()),
                "shape" => np.array(mv.shape),
                "strides" => np.array(mv.strides),
                "readonly" => NDArray.Scalar(mv.@readonly),
                "c_contiguous" => NDArray.Scalar(mv.c_contiguous),
                "f_contiguous" => NDArray.Scalar(mv.f_contiguous),
                "contiguous" => NDArray.Scalar(mv.contiguous),
                "itemsize" => NDArray.Scalar((long)mv.itemsize),
                "ndim" => NDArray.Scalar((long)mv.ndim),
                "nbytes" => NDArray.Scalar(mv.nbytes),
                _ => throw new NotSupportedException($"memoryview attribute '{attr}'"),
            };
        }

        /// <summary>
        ///     NumPy's <c>repr(dtype.type)</c> for the CLR element type NumSharp's <see cref="DType.type"/> returns (the C#
        ///     counterpart of NumPy's scalar class): <c>&lt;class 'numpy.int32'&gt;</c> for <see cref="int"/>.
        /// </summary>
        /// <param name="t">The CLR element type.</param>
        /// <returns>The class repr, or <c>None</c> for a null type.</returns>
        /// <exception cref="NotSupportedException">A type with no NumPy scalar class (char, decimal).</exception>
        private static string ScalarClassRepr(Type t)
        {
            // A descriptor without a CLR element type (datetime64 / timedelta64 before Stage C) has a null type, which
            // reads as Python's None beside NumPy's class.
            if (t is null)
                return "None";
            string name = t == typeof(bool) ? "bool"
                : t == typeof(sbyte) ? "int8"
                : t == typeof(byte) ? "uint8"
                : t == typeof(short) ? "int16"
                : t == typeof(ushort) ? "uint16"
                : t == typeof(int) ? "int32"
                : t == typeof(uint) ? "uint32"
                : t == typeof(long) ? "int64"
                : t == typeof(ulong) ? "uint64"
                : t == typeof(Half) ? "float16"
                : t == typeof(float) ? "float32"
                : t == typeof(double) ? "float64"
                : t == typeof(Complex) ? "complex128"
                : throw new NotSupportedException($"no NumPy scalar class for {t}");
            return $"<class 'numpy.{name}'>";
        }

        /// <summary>
        ///     np.frompyfunc of a menu function (<c>params.func</c>) applied to the operands. NumSharp's analog is typed:
        ///     each function is the C# delegate over doubles that computes what the generator's Python lambda computes,
        ///     and NumPy's object result is recorded cast to float64.
        /// </summary>
        /// <param name="p">The case params: func, nin, nout.</param>
        /// <param name="ops">The operands.</param>
        /// <returns>The output slots (one for nout 1).</returns>
        /// <exception cref="NotSupportedException">A name the generator does not define.</exception>
        private static NDArray[] FromPyFunc(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            Delegate f = SurfStr(p, "func") switch
            {
                "add" => new Func<double, double, double>((x, y) => x + y),
                "neg" => new Func<double, double>(x => -x),
                "addmul" => new Func<double, double, (double, double)>((x, y) => (x + y, x * y)),
                var n => throw new NotSupportedException($"frompyfunc function '{n}'"),
            };
            var v = np.frompyfunc(f, SurfInt(p, "nin"), SurfInt(p, "nout"));
            return SurfInt(p, "nout") == 1 ? new[] { v.Call(ops) } : v.CallMany(ops);
        }
    }
}
