using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.Json;
using NumSharp.Backends.Strings;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for section <c>striter</c> (<c>gen_striter</c>): <c>np.nditer</c> and
    ///     <c>np.nested_iters</c> over fixed-width strings, replayed as the TRACE the generator records — the iteration
    ///     descriptors, every step's values (with the multi-index / flat index when tracked), the iterator's operands
    ///     before close and the caller's arrays after close, values as hex (<c>u</c> = UTF-32, <c>b</c> = bytes).
    /// </summary>
    /// <remarks>
    ///     A step that raises ends the walk as a <c>!Type: text</c> line, as in the generator; a
    ///     <see cref="StringComputationNotSupportedException"/> is NOT rendered — it propagates, so the harness counts a
    ///     phase-2 case (text parsed into numbers or numbers formatted as text) as deferred rather than as a mismatch.
    /// </remarks>
    internal static partial class StringOracleOps
    {
        /// <summary>Replays one <c>striter</c> op, or reports that <paramref name="op"/> is not one of them.</summary>
        /// <param name="op">The corpus op key (<c>nditer_…</c> or <c>nested_iters</c>).</param>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The reconstructed operands (the allocated outputs excluded).</param>
        /// <param name="result">The trace text.</param>
        /// <returns>True when <paramref name="op"/> is a <c>striter</c> op.</returns>
        private static bool TryInvokeNditer(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            if (op.StartsWith("nditer_", StringComparison.Ordinal))
            {
                result = NditerTrace(p, ops);
                return true;
            }

            if (op == "nested_iters")
            {
                result = NestedTrace(p, ops[0]);
                return true;
            }

            result = null;
            return false;
        }

        /// <summary>The generator's <c>_it_trace</c>: one iteration over the case's operands, rendered line by line.</summary>
        /// <param name="p">The construction arguments, the writes and the range.</param>
        /// <param name="ops">The caller's arrays (null slots are re-inserted from <c>none_ops</c>).</param>
        /// <returns>The trace.</returns>
        private static string NditerTrace(Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            var none = p.TryGetValue("none_ops", out var no) ? new HashSet<int>(no.EnumerateArray().Select(e => e.GetInt32())) : new HashSet<int>();
            var full = new NDArray[ops.Length + none.Count];
            for (int i = 0, src = 0; i < full.Length; i++)
                full[i] = none.Contains(i) ? null : ops[src++];

            string[] flags = p.TryGetValue("flags", out var fl) ? fl.EnumerateArray().Select(e => e.GetString()).ToArray() : null;
            string[][] opFlags = p.TryGetValue("op_flags", out var of)
                ? of.EnumerateArray().Select(e => e.EnumerateArray().Select(x => x.GetString()).ToArray()).ToArray()
                : null;
            DType[] dtypes = p.TryGetValue("op_dtypes", out var od)
                ? od.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.Null ? null : np.dtype(e.GetString())).ToArray()
                : null;
            List<object> writes = p.TryGetValue("writes", out var w) ? (List<object>)StringOracleCorpus.Decode(w) : null;
            int? copyTo = p.TryGetValue("copy_to", out var ct) ? ct.GetInt32() : null;

            // A write-only operand is never read by the iterator (an allocated one starts uninitialized in NumPy): its step
            // values are rendered as "-", as the generator renders them. The defaults are NumPy's: readonly for an array,
            // writeonly + allocate for a null slot.
            var readable = new bool[full.Length];
            for (int i = 0; i < full.Length; i++)
            {
                string[] f = opFlags == null ? null : opFlags.Length == 1 ? opFlags[0] : opFlags[i];
                f ??= full[i] is null ? new[] { "writeonly", "allocate" } : new[] { "readonly" };
                readable[i] = f.Contains("readonly") || f.Contains("readwrite");
            }

            var it = np.nditer(full, flags, opFlags, dtypes, Str(p, "order")[0], Str(p, "casting"), null, null,
                               p["buffersize"].GetInt64());
            var lines = new List<string> { "dtypes " + string.Join(" ", it.dtypes.Select(d => d.str)) };
            try
            {
                if (p.TryGetValue("iterrange", out var rg))
                    it.iterrange = (rg[0].GetInt64(), rg[1].GetInt64());

                int k = 0;
                bool walked = false;
                try
                {
                    foreach (var xs in it)
                    {
                        var row = new List<string>();
                        if (it.has_multi_index)
                            row.Add("m" + string.Join(",", it.multi_index));
                        if (it.has_index)
                            row.Add("x" + it.index.ToString(CultureInfo.InvariantCulture));
                        row.AddRange(xs.Select((x, i) => readable[i] ? TraceValue(x) : "-"));
                        lines.Add(string.Join(" ", row));
                        // NumPy's x[...] = value / y[...] = x.
                        if (writes != null)
                            AssignInto(xs[0], writes[k % writes.Count]);
                        if (copyTo != null)
                            AssignInto(xs[copyTo.Value], xs[0]);
                        k++;
                    }

                    walked = true;
                }
                catch (Exception e) when (e is not StringComputationNotSupportedException)
                {
                    lines.Add("!" + PythonErrorName(e) + ": " + TraceMessage(e.Message));
                }

                // An allocated output a failed walk did not finish writing holds uninitialized elements in NumPy: "-".
                lines.Add("operands " + string.Join(" ", it.operands.Select((o, i) => walked || full[i] is not null ? TraceValue(o) : "-")));
            }
            finally
            {
                it.close();
            }

            lines.Add("after " + string.Join(" ", ops.Select(TraceValue)));
            return string.Join("\n", lines);
        }

        /// <summary>The generator's <c>_nested_trace</c>: <c>np.nested_iters</c>, the inner level walked per outer step.</summary>
        /// <param name="p">The axes and the construction arguments.</param>
        /// <param name="view">The operand.</param>
        /// <returns>The trace.</returns>
        private static string NestedTrace(Dictionary<string, JsonElement> p, NDArray view)
        {
            int[][] axes = p["axes"].EnumerateArray().Select(e => e.EnumerateArray().Select(x => x.GetInt32()).ToArray()).ToArray();
            string[] flags = p.TryGetValue("flags", out var fl) ? fl.EnumerateArray().Select(e => e.GetString()).ToArray() : null;
            DType[] dtypes = p.TryGetValue("op_dtypes", out var od)
                ? od.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.Null ? null : np.dtype(e.GetString())).ToArray()
                : null;

            var iters = np.nested_iters(view, axes, flags, null, dtypes);
            var outer = iters[0];
            var inner = iters[iters.Length - 1];
            var lines = new List<string> { "dtypes " + string.Join(" ", inner.dtypes.Select(d => d.str)) };
            try
            {
                foreach (var _ in outer)
                {
                    var row = new List<string>();
                    foreach (var y in inner)
                        row.Add(TraceValue(y[0]));
                    lines.Add(string.Join(" ", row));
                }
            }
            catch (Exception e) when (e is not StringComputationNotSupportedException)
            {
                lines.Add("!" + PythonErrorName(e) + ": " + TraceMessage(e.Message));
            }
            finally
            {
                foreach (var i in iters)
                    i.close();
            }

            return string.Join("\n", lines);
        }

        /// <summary>
        ///     NumPy's <c>view[...] = value</c> for the trace's writes: text and bytes are 0-d sources of their own width
        ///     (truncated into a narrower view, ASCII-converted into the other class), an array (another step's view) is
        ///     assigned with unsafe casting.
        /// </summary>
        /// <param name="view">The step's view (writeable for a written operand).</param>
        /// <param name="value">A decoded <c>string</c> / <c>byte[]</c>, or an <see cref="NDArray"/>.</param>
        private static void AssignInto(NDArray view, object value)
        {
            switch (value)
            {
                case string s:
                    np.copyto(view, s, casting: "unsafe");
                    break;
                case byte[] b:
                    np.copyto(view, NDArray.Scalar(b), casting: "unsafe");
                    break;
                case NDArray a:
                    np.copyto(view, a, casting: "unsafe");
                    break;
                default:
                    throw new InvalidOperationException($"no write for {value?.GetType().Name ?? "null"}");
            }
        }

        /// <summary>
        ///     The Python exception name the generator records for <paramref name="e"/> (its first well-known class): NumSharp's
        ///     own classes are named after NumPy's; the house .NET equivalents map to theirs.
        /// </summary>
        /// <param name="e">The exception a step raised.</param>
        /// <returns>The Python class name.</returns>
        private static string PythonErrorName(Exception e) => e switch
        {
            UnicodeEncodeError => "UnicodeEncodeError",
            UnicodeDecodeError => "UnicodeDecodeError",
            OverflowException => "OverflowError",
            InvalidCastException => "TypeError",
            _ => e.GetType().Name,
        };

        /// <summary>
        ///     An exception's text as the trace records NumPy's (<c>str(e)</c>): .NET's <c> (Parameter 'x')</c> suffix on an
        ///     argument exception is not part of the message.
        /// </summary>
        /// <param name="m">The exception's message.</param>
        /// <returns>The message without the suffix.</returns>
        private static string TraceMessage(string m)
        {
            if (string.IsNullOrEmpty(m))
                return "";
            int i = m.LastIndexOf(" (Parameter '", StringComparison.Ordinal);
            return i >= 0 && m.EndsWith("')", StringComparison.Ordinal) ? m.Substring(0, i) : m;
        }

        /// <summary>One traced value: an element as <c>u&lt;UTF-32 hex&gt;</c> / <c>b&lt;hex&gt;</c> / a number, an array as a bracketed list.</summary>
        /// <param name="a">A 0-d step view, a 1-D chunk, or a whole operand.</param>
        /// <returns>The rendering the generator's <c>_it_val(x.tolist())</c> produces.</returns>
        private static string TraceValue(NDArray a)
        {
            if (a.ndim == 0)
                return TraceElement(a.Storage.GetAtIndex(0));
            var parts = new List<string>();
            for (long i = 0; i < a.shape[0]; i++)
            {
                // Each row is a view sharing the operand's buffer: released at once, so the buffer goes back to the pool
                // when its owner is disposed (an undisposed view would hold it until the finalizer runs).
                using var row = a[(int)i];
                parts.Add(TraceValue(row));
            }
            return "[" + string.Join(",", parts) + "]";
        }

        /// <summary>One element as the generator renders it.</summary>
        private static string TraceElement(object v) => v switch
        {
            string s => "u" + Utf32Hex(s),
            byte[] b => "b" + Convert.ToHexString(b).ToLowerInvariant(),
            bool bo => bo ? "B1" : "B0",
            double d => "f" + PythonFloatRepr(d),
            float f => "f" + PythonFloatRepr(f),
            Half h => "f" + PythonFloatRepr((double)h),
            IConvertible c => "i" + c.ToString(CultureInfo.InvariantCulture),
            _ => throw new InvalidOperationException($"untraceable value {v?.GetType().Name ?? "null"}"),
        };

        /// <summary>
        ///     The string's code points as UTF-32LE hex — Python's <c>encode("utf-32-le", "surrogatepass")</c>: a pair of
        ///     surrogates is one code point, a lone surrogate passes through as its own unit.
        /// </summary>
        private static string Utf32Hex(string s)
        {
            var sb = new StringBuilder(s.Length * 8);
            for (int i = 0; i < s.Length; i++)
            {
                int cp = s[i];
                if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
                    cp = char.ConvertToUtf32(s[i], s[++i]);
                sb.Append((cp & 0xFF).ToString("x2")).Append(((cp >> 8) & 0xFF).ToString("x2"))
                  .Append(((cp >> 16) & 0xFF).ToString("x2")).Append(((cp >> 24) & 0xFF).ToString("x2"));
            }

            return sb.ToString();
        }

        /// <summary>
        ///     Python's <c>repr(float)</c>: the shortest round-trip digits, positional between 1e-4 and 1e16 (an integral value
        ///     keeps a <c>.0</c>), scientific outside (<c>1e+20</c>, <c>1e-05</c>), and <c>nan</c> / <c>inf</c>.
        /// </summary>
        private static string PythonFloatRepr(double d)
        {
            if (double.IsNaN(d))
                return "nan";
            if (double.IsInfinity(d))
                return d > 0 ? "inf" : "-inf";
            string r = d.ToString("R", CultureInfo.InvariantCulture);
            int e = r.IndexOfAny(new[] { 'E', 'e' });
            string mantissa = e < 0 ? r : r.Substring(0, e);
            int exp = e < 0 ? 0 : int.Parse(r.Substring(e + 1), CultureInfo.InvariantCulture);
            // The decimal exponent of the leading digit decides the form, as Python's float_repr_style 'short' does.
            bool neg = mantissa.StartsWith("-", StringComparison.Ordinal);
            string digits = (neg ? mantissa.Substring(1) : mantissa).Replace(".", "");
            int dot = (neg ? mantissa.Substring(1) : mantissa).IndexOf('.');
            int intLen = dot < 0 ? (neg ? mantissa.Length - 1 : mantissa.Length) : dot;
            string trimmed = digits.TrimStart('0');
            if (trimmed.Length == 0)
                return neg ? "-0.0" : "0.0";
            int lead = intLen - (digits.Length - trimmed.Length) - 1 + exp;
            trimmed = trimmed.TrimEnd('0');
            string sign = neg ? "-" : "";
            if (lead < -4 || lead >= 16)
            {
                string m = trimmed.Length == 1 ? trimmed : trimmed[0] + "." + trimmed.Substring(1);
                return sign + m + "e" + (lead < 0 ? "-" : "+") + Math.Abs(lead).ToString("00", CultureInfo.InvariantCulture);
            }

            if (lead < 0)
                return sign + "0." + new string('0', -lead - 1) + trimmed;
            if (trimmed.Length <= lead + 1)
                return sign + trimmed + new string('0', lead + 1 - trimmed.Length) + ".0";
            return sign + trimmed.Substring(0, lead + 1) + "." + trimmed.Substring(lead + 1);
        }
    }
}
