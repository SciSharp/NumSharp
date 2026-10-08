using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The replay of <c>gen_oracle.py</c>'s <c>gen_nditer_construct</c> (corpus <c>nditer_construct.jsonl</c>):
    ///     np.nditer built the way the case says (flags, op_flags, op_dtypes, casting, order, buffersize), walked with the
    ///     iteration protocol (<c>foreach</c>, NumPy's <c>for x in it</c>), and summarized per operand as three slots —
    ///     the value stream in the ITERATION dtype, the operand right after construction (a copy route's temporary, an
    ///     allocated output, or the operand itself), and the caller's array after the iterator closed.
    /// </summary>
    /// <remarks>
    ///     A <c>write</c> case stores the step index into operand 0 at every step (NumPy's <c>x[...] = k</c>), so a
    ///     write-back route (<c>updateifcopy</c>) and a buffered <c>readwrite</c> loop are observable in the third slot —
    ///     including the last write of every buffer, which a publish-then-advance protocol lost (journey4's gap SU42). The
    ///     corpus' null operands (outputs the iterator allocates) are re-inserted at the positions <c>none_ops</c> names.
    /// </remarks>
    public static partial class OpRegistry
    {
        /// <summary>One <c>nditer_construct</c> case: the per-operand (values, operand, after) slots.</summary>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The operands the corpus serialized (null slots excluded).</param>
        /// <returns>Three slots per operand, operand by operand.</returns>
        internal static NDArray[] NditerConstructFromCorpus(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var full = WithNoneOperands(p, ops);
            string[] flags = p.TryGetValue("flags", out var fl) && fl.ValueKind == JsonValueKind.Array
                ? fl.EnumerateArray().Select(e => e.GetString()).ToArray()
                : null;
            string[][] opFlags = ParseNditerOpFlags(p);
            DType[] opDtypes = p.TryGetValue("op_dtypes", out var od) && od.ValueKind == JsonValueKind.Array
                ? od.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.Null ? null : np.dtype(e.GetString())).ToArray()
                : null;
            string casting = p["casting"].GetString();
            char order = p["order"].GetString()[0];
            long buffersize = p["buffersize"].GetInt64();
            bool write = p["write"].GetBoolean();

            NDArray[] snaps = null;
            List<object>[] streams;
            DType[] dtypes;
            bool[] reads;
            long chunk;
            // The outputs the iterator allocates are the CALLER's (NumPy hands them out through it.operands): the replay
            // disposes them once their values are recorded, as a well-behaved caller would.
            var allocated = new NDArray[full.Length];
            var result = new List<NDArray>();
            try
            {
                using (var it = np.nditer(full, flags, opFlags, opDtypes, order, casting, null, null, buffersize))
                {
                    int nop = it.nop;
                    dtypes = it.dtypes;
                    reads = new bool[nop];
                    for (int i = 0; i < nop; i++)
                        reads[i] = Reads(opFlags, full, i);
                    var operands = it.operands;
                    for (int i = 0; i < nop; i++)
                        if (full[i] is null)
                            allocated[i] = operands[i];

                    // Before the walk: NumPy's it.operands is the temporary of a copy route / the allocated output.
                    snaps = new NDArray[nop];
                    for (int i = 0; i < nop; i++)
                        snaps[i] = reads[i] ? operands[i].copy() : np.empty(Shape.Vector(0), dtypes[i]);

                    streams = new List<object>[nop];
                    for (int i = 0; i < nop; i++)
                        streams[i] = new List<object>();
                    chunk = -1;
                    long k = 0;
                    foreach (var v in it)
                    {
                        for (int i = 0; i < nop; i++)
                        {
                            if (!reads[i])
                                continue;
                            // A step is one element, or one 1-D chunk under external_loop: NumPy's np.array(streams) stacks
                            // equal-length chunks into rows, so the chunk length is recorded once (all chunks must agree).
                            long n = v[i].ndim == 0 ? 0 : v[i].size;
                            if (chunk == -1)
                                chunk = n;
                            else if (chunk != n)
                                throw new NotSupportedException("ragged external-loop chunks: NumPy's np.array cannot stack them");
                            if (n == 0)
                                streams[i].Add(v[i].GetAtIndex(0));
                            else
                                for (long e = 0; e < n; e++)
                                    streams[i].Add(v[i].GetAtIndex(e));
                        }
                        // NumPy's x[...] = k stores a Python int through PyArray_Pack; np.copyto's generic overload is that
                        // same weak-scalar store (unsafe casting lets bool take it, as setitem does). SetAtIndex cannot: its
                        // generic form demands an int64 view and its object form unboxes without converting.
                        if (write)
                            np.copyto(v[0], k, casting: "unsafe");
                        k++;
                    }
                }

                for (int i = 0; i < full.Length; i++)
                {
                    NDArray values = streams[i].Count > 0 ? NditerValues(streams[i], dtypes[i]) : np.empty(Shape.Vector(0), dtypes[i]);
                    // External-loop chunks stack into (steps, chunk) rows, as NumPy's np.array(list of 1-D chunks) does. The
                    // rows are a view of the flat stream: the flat array is released (the view keeps the buffer).
                    if (chunk > 0 && streams[i].Count > 0)
                    {
                        var rows = values.reshape(streams[i].Count / chunk, chunk);
                        values.Dispose();
                        values = rows;
                    }
                    result.Add(values);
                    result.Add(snaps[i]);
                    snaps[i] = null;
                    result.Add(full[i] is null ? np.empty(Shape.Vector(0), dtypes[i]) : full[i].copy());
                }
                return result.ToArray();
            }
            catch
            {
                // A case NumPy refuses mid-walk (a delayed-allocation pull, a cast error): what was already built is released.
                foreach (var r in result)
                    r.Dispose();
                if (snaps != null)
                    foreach (var sn in snaps)
                        sn?.Dispose();
                throw;
            }
            finally
            {
                foreach (var a in allocated)
                    a?.Dispose();
            }
        }

        /// <summary>
        ///     One <c>nditer_protocol</c> case (<c>gen_nditer_protocol</c>): builds the iterator the case names, runs its
        ///     SCRIPT of steps, and renders one line per step — the step's result, or the exception as
        ///     <c>!Type: text</c> — exactly as the generator renders NumPy's.
        /// </summary>
        /// <param name="p">The case parameters (construction arguments plus <c>script</c>).</param>
        /// <param name="ops">The operands the corpus serialized (null slots excluded).</param>
        /// <returns>The trace text.</returns>
        internal static string NditerProtocolFromCorpus(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var full = WithNoneOperands(p, ops);
            string[] flags = p.TryGetValue("flags", out var fl) && fl.ValueKind == JsonValueKind.Array
                ? fl.EnumerateArray().Select(e => e.GetString()).ToArray()
                : null;
            string[][] opFlags = ParseNditerOpFlags(p);
            DType[] opDtypes = p.TryGetValue("op_dtypes", out var od) && od.ValueKind == JsonValueKind.Array
                ? od.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.Null ? null : np.dtype(e.GetString())).ToArray()
                : null;
            var script = p["script"].EnumerateArray().Select(e => e.GetString()).ToArray();
            int[][] opAxes = ParseNditerOpAxes(p);

            var it = np.nditer(full, flags, opFlags, opDtypes, p["order"].GetString()[0], p["casting"].GetString(),
                               opAxes, null, p["buffersize"].GetInt64());
            // The allocated outputs are the caller's (NumPy hands them out through it.operands): released at the end.
            var operands = it.operands;
            var allocated = new List<NDArray>();
            for (int i = 0; i < full.Length; i++)
                if (full[i] is null)
                    allocated.Add(operands[i]);
            var lines = new List<string>();
            try
            {
                foreach (var step in script)
                {
                    string line;
                    try
                    {
                        line = ProtocolStep(it, step);
                    }
                    catch (Exception e)
                    {
                        line = "!" + ProtocolErrorName(e) + ": " + e.Message;
                    }

                    lines.Add(step + ": " + line);
                }
            }
            finally
            {
                it.close();
                foreach (var a in allocated)
                    a.Dispose();
            }

            return string.Join("\n", lines);
        }

        /// <summary>The trace's stand-in for Python's StopIteration (a pull that found no element).</summary>
        private sealed class ProtocolStopIteration : Exception
        {
            /// <summary>An empty message, as Python's StopIteration renders.</summary>
            public ProtocolStopIteration() : base("") { }
        }

        /// <summary>The Python exception name the generator prints for <paramref name="e"/>.</summary>
        /// <param name="e">The exception a step raised.</param>
        /// <returns>NumPy's exception type name for the house mapping (OverflowException is OverflowError).</returns>
        private static string ProtocolErrorName(Exception e) => e switch
        {
            ProtocolStopIteration => "StopIteration",
            OverflowException => "OverflowError",
            _ => e.GetType().Name,
        };

        /// <summary>Runs one script step (the generator's <c>_np_protocol_step</c>) and renders its result.</summary>
        /// <param name="it">The iterator (closed by a <c>close</c> step; later steps then meet the closed contract).</param>
        /// <param name="step">The step token, <c>name</c> or <c>name:argument</c>.</param>
        /// <returns>The rendered result.</returns>
        /// <exception cref="NotSupportedException">An unknown step token (a generator / replay mismatch).</exception>
        private static string ProtocolStep(np.NDIterator it, string step)
        {
            int colon = step.IndexOf(':');
            string name = colon < 0 ? step : step.Substring(0, colon);
            string arg = colon < 0 ? "" : step.Substring(colon + 1);
            var inv = System.Globalization.CultureInfo.InvariantCulture;
            switch (name)
            {
                case "next":
                    if (!it.MoveNext())
                        throw new ProtocolStopIteration();
                    return string.Join(" ", it.Current.Select(ProtocolValue));
                case "walk":
                    return ProtocolWalk(it, arg.Length == 0 ? (int?)null : int.Parse(arg, inv));
                case "walkidx":
                    return ProtocolWalkIndex(it, arg.Length == 0 ? (int?)null : int.Parse(arg, inv));
                case "value":
                    return string.Join(" ", it.value.Select(ProtocolValue));
                case "it":
                    return ProtocolValue(it[int.Parse(arg, inv)]);
                case "writeable":
                    return ProtocolBool(it[int.Parse(arg, inv)].flags.writeable);
                case "finished": return ProtocolBool(it.finished);
                case "iterindex": return it.iterindex.ToString(inv);
                case "index": return it.index.ToString(inv);
                case "itersize": return it.itersize.ToString(inv);
                case "ndim": return it.ndim.ToString(inv);
                case "nop": return it.nop.ToString(inv);
                case "has_index": return ProtocolBool(it.has_index);
                case "has_multi_index": return ProtocolBool(it.has_multi_index);
                case "has_delayed_bufalloc": return ProtocolBool(it.has_delayed_bufalloc);
                case "multi_index": return ProtocolTuple(it.multi_index);
                case "shape": return ProtocolTuple(it.shape);
                case "iterrange":
                {
                    var (start, end) = it.iterrange;
                    return ProtocolTuple(new[] { start, end });
                }
                case "dtypes":
                    return "[" + string.Join(", ", it.dtypes.Select(d => d.ToString(true))) + "]";
                case "operands":
                    return "[" + string.Join(", ", it.operands.Select(ProtocolValue)) + "]";
                case "iternext":
                    return ProtocolBool(it.iternext());
                case "reset":
                    it.reset();
                    return "None";
                case "close":
                    it.close();
                    return "None";
                case "copywalk":
                {
                    var copy = it.copy();
                    try
                    {
                        return ProtocolWalk(copy, null);
                    }
                    finally
                    {
                        copy.close();
                    }
                }
                case "remove_axis":
                    it.remove_axis(int.Parse(arg, inv));
                    return "None";
                case "remove_multi_index":
                    it.remove_multi_index();
                    return "None";
                case "enable_external_loop":
                    it.enable_external_loop();
                    return "None";
                case "set_iterindex":
                    it.iterindex = long.Parse(arg, inv);
                    return "None";
                case "set_index":
                    it.index = long.Parse(arg, inv);
                    return "None";
                case "set_multi_index":
                    it.multi_index = arg.Length == 0
                        ? Array.Empty<long>()
                        : arg.Split(',').Select(t => long.Parse(t, inv)).ToArray();
                    return "None";
                case "set_iterrange":
                {
                    var parts = arg.Split(',');
                    it.iterrange = (long.Parse(parts[0], inv), long.Parse(parts[1], inv));
                    return "None";
                }
                case "setitem":
                {
                    int sep = arg.IndexOf(':');
                    using var value = ProtocolOperandValue(arg.Substring(sep + 1));
                    it[int.Parse(arg.Substring(0, sep), inv)] = value;
                    return "None";
                }
                case "write":
                {
                    // NumPy's view[...] = v: a Python int is packed (range-checked) into the view, a list assigned with
                    // unsafe casting — np.copyto's weak-scalar overload and its array overload under 'unsafe'.
                    int sep = arg.IndexOf(':');
                    var view = it[int.Parse(arg.Substring(0, sep), inv)];
                    string v = arg.Substring(sep + 1);
                    if (v.StartsWith("[", StringComparison.Ordinal))
                    {
                        using var value = ProtocolOperandValue(v);
                        np.copyto(view, value, casting: "unsafe");
                    }
                    else
                        np.copyto(view, long.Parse(v, inv), casting: "unsafe");
                    return "None";
                }
                case "zero_out":
                    np.copyto(it.operands[int.Parse(arg, inv)], 0L, casting: "unsafe");
                    return "None";
                case "accumwalk":
                    // NumPy's reduction idiom, `for x, y in it: y[...] += x`: an in-place add through each step's views.
                    foreach (var xs in it)
                        np.add(xs[1], xs[0], @out: xs[1]);
                    return "None";
                default:
                    throw new NotSupportedException("unknown protocol step " + step);
            }
        }

        /// <summary>The generator's <c>_np_walk</c>: every remaining (or up to <paramref name="limit"/>) pull.</summary>
        /// <param name="it">The iterator.</param>
        /// <param name="limit">The pull limit, or null for all.</param>
        /// <returns>The pulled steps joined by <c> | </c> (an error inside the walk ends it as its last entry).</returns>
        private static string ProtocolWalk(np.NDIterator it, int? limit)
        {
            var steps = new List<string>();
            try
            {
                int k = 0;
                foreach (var xs in it)
                {
                    var row = new List<string>();
                    if (it.has_multi_index)
                        row.Add(ProtocolTuple(it.multi_index));
                    row.AddRange(xs.Select(ProtocolValue));
                    steps.Add(string.Join(" ", row));
                    k++;
                    if (limit != null && k >= limit)
                        break;
                }
            }
            catch (Exception e)
            {
                steps.Add("!" + ProtocolErrorName(e) + ": " + e.Message);
            }

            return steps.Count == 0 ? "[]" : string.Join(" | ", steps);
        }

        /// <summary>
        ///     The generator's <c>_np_walk_index</c>: pulls up to <paramref name="limit"/> elements and records each as
        ///     <c>value@index#iterindex</c> (<c>-</c> for the index when none is tracked). A tracked index that is wrong
        ///     walks the same values, so only this rendering gates the index strides through the axis ordering, the stride
        ///     flips and the coalescing.
        /// </summary>
        /// <param name="it">The iterator, walked from its current position.</param>
        /// <param name="limit">The number of pulls, or null for the rest of the iteration.</param>
        /// <returns>The steps joined with <c> | </c>, a trailing <c>!Error: text</c> when a pull raises, or <c>[]</c>.</returns>
        private static string ProtocolWalkIndex(np.NDIterator it, int? limit)
        {
            var inv = System.Globalization.CultureInfo.InvariantCulture;
            var steps = new List<string>();
            try
            {
                int k = 0;
                foreach (var xs in it)
                {
                    string index = it.has_index ? it.index.ToString(inv) : "-";
                    steps.Add(string.Join(" ", xs.Select(ProtocolValue)) + "@" + index + "#" + it.iterindex.ToString(inv));
                    k++;
                    if (limit != null && k >= limit)
                        break;
                }
            }
            catch (Exception e)
            {
                steps.Add("!" + ProtocolErrorName(e) + ": " + e.Message);
            }

            return steps.Count == 0 ? "[]" : string.Join(" | ", steps);
        }

        /// <summary>A script value: an int literal (a 0-d value, NumPy's Python int) or a <c>[a,b,…]</c> int64 array.</summary>
        /// <param name="text">The token.</param>
        /// <returns>The value array.</returns>
        private static NDArray ProtocolOperandValue(string text)
        {
            var inv = System.Globalization.CultureInfo.InvariantCulture;
            if (!text.StartsWith("[", StringComparison.Ordinal))
                return NDArray.Scalar(long.Parse(text, inv));
            string inner = text.Substring(1, text.Length - 2);
            return np.array(inner.Length == 0 ? Array.Empty<long>() : inner.Split(',').Select(t => long.Parse(t, inv)).ToArray());
        }

        /// <summary>Python's <c>str(bool)</c>.</summary>
        private static string ProtocolBool(bool b) => b ? "True" : "False";

        /// <summary>Python's <c>str(tuple)</c> of integers: <c>(1, 2)</c>, <c>(3,)</c>, <c>()</c>.</summary>
        private static string ProtocolTuple(long[] t)
            => t.Length == 1 ? "(" + t[0] + ",)" : "(" + string.Join(", ", t) + ")";

        /// <summary>Python's <c>repr(a.tolist())</c> for the tier's integer and integral-float values.</summary>
        private static string ProtocolValue(NDArray a)
        {
            if (a.ndim == 0)
                return ProtocolScalar(a.GetAtIndex(0));
            var parts = new List<string>();
            for (long i = 0; i < a.shape[0]; i++)
            {
                // Each row is a view sharing the operand's buffer: released at once, so the buffer goes back to the pool
                // when its owner is disposed (an undisposed view would hold it until the finalizer runs).
                using var row = a[(int)i];
                parts.Add(ProtocolValue(row));
            }
            return "[" + string.Join(", ", parts) + "]";
        }

        /// <summary>
        ///     Python's repr of one <c>tolist()</c> element: an int, a float (every float width becomes a Python float), a
        ///     complex, a bool.
        /// </summary>
        private static string ProtocolScalar(object v) => v switch
        {
            bool b => ProtocolBool(b),
            double d => ProtocolFloat(d),
            float f => ProtocolFloat(f),
            Half h => ProtocolFloat((double)h),
            System.Numerics.Complex c => ProtocolComplex(c),
            _ => Convert.ToString(v, System.Globalization.CultureInfo.InvariantCulture),
        };

        /// <summary>Python's <c>repr(float)</c> for the tier's values: an integral value keeps <c>.0</c>.</summary>
        private static string ProtocolFloat(double d)
            => d == Math.Floor(d) && Math.Abs(d) < 1e16
                ? d.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture)
                : d.ToString("R", System.Globalization.CultureInfo.InvariantCulture);

        /// <summary>
        ///     Python's <c>repr(complex)</c>: <c>3j</c> for a +0 real part, else <c>(re±imj)</c>, each part without the
        ///     float's <c>.0</c> (<c>(3+0j)</c>).
        /// </summary>
        private static string ProtocolComplex(System.Numerics.Complex c)
        {
            static string Part(double x)
            {
                string r = ProtocolFloat(x);
                return r.EndsWith(".0", StringComparison.Ordinal) ? r.Substring(0, r.Length - 2) : r;
            }

            string im = Part(c.Imaginary);
            if (c.Real == 0 && !double.IsNegative(c.Real))
                return im + "j";
            return "(" + Part(c.Real) + (im.StartsWith("-", StringComparison.Ordinal) ? "" : "+") + im + "j)";
        }

        /// <summary>One <c>nditer_ranged</c> case: a ranged, buffered cast iteration over <c>[start, end)</c>.</summary>
        /// <param name="p">The case parameters (op_dtypes, start, end).</param>
        /// <param name="ops">The single operand.</param>
        /// <returns>The value stream in the iteration dtype.</returns>
        internal static NDArray NditerRangedFromCorpus(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var req = np.dtype(p["op_dtypes"][0].GetString());
            long start = p["start"].GetInt64(), end = p["end"].GetInt64();
            using var it = np.nditer(ops[0], flags: new[] { "ranged", "buffered" }, op_dtypes: new[] { req });
            it.iterrange = (start, end);
            var vals = new List<object>();
            foreach (var v in it)
                vals.Add(v[0].GetAtIndex(0));
            return vals.Count > 0 ? NditerValues(vals, it.dtypes[0]) : np.empty(Shape.Vector(0), req);
        }

        /// <summary>The corpus operands with nulls re-inserted where the case's <c>none_ops</c> names an allocated output.</summary>
        private static NDArray[] WithNoneOperands(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            if (!p.TryGetValue("none_ops", out var no) || no.ValueKind != JsonValueKind.Array)
                return ops;
            var none = new HashSet<int>(no.EnumerateArray().Select(e => e.GetInt32()));
            var full = new NDArray[ops.Length + none.Count];
            int src = 0;
            for (int i = 0; i < full.Length; i++)
                full[i] = none.Contains(i) ? null : ops[src++];
            return full;
        }

        /// <summary>
        ///     The case's op_flags as np.nditer's multi-operand parameter: a list of lists per operand, or NumPy's flat list
        ///     (one set for every operand) as the single inner list; absent is null.
        /// </summary>
        private static string[][] ParseNditerOpFlags(IReadOnlyDictionary<string, JsonElement> p)
        {
            if (!p.TryGetValue("op_flags", out var of) || of.ValueKind != JsonValueKind.Array)
                return null;
            var items = of.EnumerateArray().ToArray();
            if (items.Length > 0 && items[0].ValueKind == JsonValueKind.String)
                return new[] { items.Select(e => e.GetString()).ToArray() };
            return items.Select(e => e.EnumerateArray().Select(x => x.GetString()).ToArray()).ToArray();
        }

        /// <summary>
        ///     The case's <c>op_axes</c> (one entry per operand: null for the operand's own axes, or the operand axis of each
        ///     iterator axis with -1 for an axis the operand does not have), or null when the case passes none.
        /// </summary>
        /// <param name="p">The case parameters.</param>
        /// <returns>The per-operand axis maps, or null.</returns>
        private static int[][] ParseNditerOpAxes(IReadOnlyDictionary<string, JsonElement> p)
        {
            if (!p.TryGetValue("op_axes", out var oa) || oa.ValueKind != JsonValueKind.Array)
                return null;
            return oa.EnumerateArray()
                     .Select(e => e.ValueKind == JsonValueKind.Null ? null : e.EnumerateArray().Select(x => x.GetInt32()).ToArray())
                     .ToArray();
        }

        /// <summary>Whether operand <paramref name="i"/> is read under the case's flags (NumPy's defaults when absent).</summary>
        private static bool Reads(string[][] opFlags, NDArray[] full, int i)
        {
            string[] f = opFlags == null
                ? (full[i] is null ? new[] { "writeonly", "allocate" } : new[] { "readonly" })
                : opFlags.Length == 1 ? opFlags[0] : opFlags[i];
            return f.Contains("readonly") || f.Contains("readwrite");
        }

        /// <summary>A value stream as a 1-D array of the iteration descriptor.</summary>
        private static NDArray NditerValues(List<object> vals, DType dtype)
        {
            var ret = np.empty(Shape.Vector(vals.Count), dtype);
            for (int i = 0; i < vals.Count; i++)
                ret.SetAtIndex(vals[i], i);
            return ret;
        }
    }
}
