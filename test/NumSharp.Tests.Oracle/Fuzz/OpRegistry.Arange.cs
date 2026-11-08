using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    public static partial class OpRegistry
    {
        /// <summary>
        ///     Replays one <c>arange_full.jsonl</c> case (gen_oracle.py <c>arange_full</c>, np-layer dtype plan P3's D4 fix):
        ///     <c>np.arange</c> as NumPy's <c>PyArray_ArangeObj</c>, through the overload NumPy's argument kinds select —
        ///     three Python floats bind <c>np.arange(double, double, double, DType)</c>, three Python ints the exact
        ///     <c>np.arange(long, long, long)</c> (int64, NumPy's discovery for ints within int64).
        /// </summary>
        /// <param name="p">The case's params: <c>start</c> / <c>stop</c> / <c>step</c> as Python-literal descriptors
        ///     (<c>kind</c> <c>"int"</c> with a decimal <c>value</c>, or <c>"float"</c> with little-endian IEEE <c>bits</c>, so
        ///     NaN, the infinities and -0.0 survive) and an optional NumPy dtype name.</param>
        /// <returns>The range.</returns>
        /// <exception cref="NotSupportedException">The bounds mix ints and floats, or the ints carry a dtype: the tier
        ///     generates neither (a corpus bug).</exception>
        internal static NDArray ApplyArangeFull(IReadOnlyDictionary<string, JsonElement> p)
        {
            var start = p["start"];
            var stop = p["stop"];
            var step = p["step"];
            string kind = start.GetProperty("kind").GetString();
            if (stop.GetProperty("kind").GetString() != kind || step.GetProperty("kind").GetString() != kind)
                throw new NotSupportedException("arange_full: the bounds mix Python ints and floats");
            if (kind == "int")
            {
                if (p.ContainsKey("dtype"))
                    throw new NotSupportedException("arange_full: NumSharp's (long, long, long) overload has no dtype");
                return np.arange(PythonLong(start), PythonLong(stop), PythonLong(step));
            }
            // A dtype name parses with np.dtype, which knows the text classes too ("U3", "S3", "T").
            DType dtype = p.TryGetValue("dtype", out var d) ? np.dtype(d.GetString()) : null;
            return np.arange(LittleEndianDouble(start.GetProperty("bits").GetString()),
                LittleEndianDouble(stop.GetProperty("bits").GetString()),
                LittleEndianDouble(step.GetProperty("bits").GetString()), dtype);
        }

        /// <summary>A Python-int literal descriptor's value (the tier's ints fit a long).</summary>
        /// <param name="e">The descriptor (<c>kind</c> <c>"int"</c>, decimal <c>value</c>).</param>
        /// <returns>The value.</returns>
        /// <exception cref="OverflowException">The value does not fit a long (a corpus bug).</exception>
        private static long PythonLong(JsonElement e)
            => long.Parse(e.GetProperty("value").GetString(), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
    }
}
