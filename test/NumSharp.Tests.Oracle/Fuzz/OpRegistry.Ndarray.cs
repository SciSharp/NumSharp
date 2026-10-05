using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The <c>ndarray_ctor.jsonl</c> tier: <c>np.ndarray(shape, dtype, buffer, offset, strides, order)</c>, generated
    ///     by <c>gen_oracle.py</c>'s <c>gen_ndarray</c> (mode <c>ndarray</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Arguments.</b> <c>shape</c> is a list, a single number (NumPy's <c>5</c> is <c>(5,)</c>) or null (NumPy's
    ///     <c>None</c>, replayed as <c>default(Shape)</c>); an absent <c>dtype</c>, <c>offset</c>, <c>strides</c> or
    ///     <c>order</c> is NumPy's default. <c>buffer.kind</c> picks the C# analog of NumPy's buffer object:
    ///     <c>bytearray</c> a <c>byte[]</c> of operand 0's bytes, <c>bytes</c> a read-only
    ///     <see cref="ReadOnlyMemory{T}"/> of them, <c>ndarray</c> operand 0 itself, <c>memoryview</c> its
    ///     <see cref="NDArray.data"/>, <c>none</c> null, and the non-buffers <c>list</c> / <c>int</c> / <c>str</c>
    ///     an <c>object[]</c> / <c>7</c> / <c>"abcdefgh"</c>.
    ///     </para>
    ///     <para>
    ///     <b>Facets.</b> <c>values</c> returns the result itself; without a buffer its memory is uninitialized, so it is
    ///     filled with 0 first, as the generator does. <c>layout</c> returns the int64 record
    ///     <c>[C, F, OWNDATA, WRITEABLE, ALIGNED, base-is-set, *byte strides]</c> (the generator's
    ///     <c>_ndarray_layout</c>) and disposes the result, which the harness never sees.
    ///     </para>
    /// </remarks>
    public static partial class OpRegistry
    {
        /// <summary>Replays one <c>ndarray</c> case.</summary>
        /// <param name="p">The case parameters (see the class remarks).</param>
        /// <param name="ops">The operands: the buffer's bytes or array, when the case has one.</param>
        /// <returns>The result (values facet) or its layout record (layout facet).</returns>
        /// <exception cref="NotSupportedException">An unknown buffer kind or facet.</exception>
        private static NDArray NdarrayCtor(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            JsonElement shapeJson = p["shape"];
            Shape shape = shapeJson.ValueKind switch
            {
                JsonValueKind.Null => default,
                JsonValueKind.Number => Shape.Vector(shapeJson.GetInt64()),
                _ => new Shape(ParseLongArray(shapeJson)),
            };

            DType dtype = p.TryGetValue("dtype", out var d) ? FuzzCorpus.DtypeToTC(d.GetString()) : null;
            long offset = p.TryGetValue("offset", out var o) ? o.GetInt64() : 0;
            long[] strides = p.TryGetValue("strides", out var s) && s.ValueKind == JsonValueKind.Array ? ParseLongArray(s) : null;
            char order = p.TryGetValue("order", out var od) && od.ValueKind == JsonValueKind.String ? od.GetString()![0] : 'C';

            object buffer = null;
            if (p.TryGetValue("buffer", out var b))
            {
                string kind = b.GetProperty("kind").GetString();
                buffer = kind switch
                {
                    "none" => null,
                    "bytearray" => FuzzCorpus.ResultBytes(ops[0]),
                    "bytes" => new ReadOnlyMemory<byte>(FuzzCorpus.ResultBytes(ops[0])),
                    "ndarray" => ops[0],
                    "memoryview" => ops[0].data,
                    "list" => Enumerable.Range(0, 6).Select(i => (object)(long)i).ToArray(),
                    "int" => 7,
                    "str" => "abcdefgh",
                    _ => throw new NotSupportedException($"ndarray buffer kind '{kind}'"),
                };
            }

            NDArray r = np.ndarray(shape, dtype, buffer, offset, strides, order);
            switch (p["facet"].GetString())
            {
                case "values":
                    // Uninitialized memory: the generator fills NumPy's result with 0 the same way.
                    if (buffer is null)
                        r.fill(0);
                    return r;

                case "layout":
                {
                    var f = r.flags;
                    var record = new List<long>
                    {
                        f.c_contiguous ? 1 : 0, f.f_contiguous ? 1 : 0, f.owndata ? 1 : 0, f.writeable ? 1 : 0,
                        f.aligned ? 1 : 0, r.Storage.BaseStorage is null ? 0 : 1,
                    };
                    record.AddRange(r.strides);
                    // The record REPLACES the result, so the harness never disposes it: release it here (and with it
                    // the buffer's pin), or the leak gate reads the buffer as escaped.
                    r.Dispose();
                    return np.array(record.ToArray());
                }

                default:
                    throw new NotSupportedException($"ndarray facet '{p["facet"].GetString()}'");
            }
        }
    }
}
