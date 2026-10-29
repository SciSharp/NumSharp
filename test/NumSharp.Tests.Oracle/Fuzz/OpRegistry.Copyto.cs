using System;
using System.Buffers.Binary;
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
        ///     Replays one <c>copyto_full.jsonl</c> case (gen_oracle.py <c>copyto_full</c>, plan
        ///     <c>docs/plans/api-parameter-parity.md</c> X16 + X11): <c>np.copyto(dst, src, casting, where)</c>, returning
        ///     the destination's WHOLE buffer so a write outside the view, or into a masked-off slot, is visible.
        /// </summary>
        /// <param name="p">The case's params: <c>casting</c> (a name, or JSON null for NumPy's <c>casting=None</c>);
        ///     <c>where</c> (absent: no mask; <c>true</c>/<c>false</c>: a C# literal bool, NumPy's <c>where=True</c> /
        ///     <c>where=False</c>; <c>"array"</c>: the LAST operand); and the source as one of <c>src_scalar</c> (a C#
        ///     literal: NumPy's weak Python int / float / complex, or a strong Python bool), <c>src_view</c> (a view of
        ///     the destination's own buffer: overlap and redundant self-copies), or the next operand.</param>
        /// <param name="ops">Operand 0 is the destination view with its PRIOR contents (read-only when the case says
        ///     so); then the source array (unless the source is a literal or a view); then the mask array.</param>
        /// <returns>The destination's base buffer, as a 1-D array over the whole storage.</returns>
        /// <exception cref="NotSupportedException">An unknown <c>src_scalar</c> kind (a corpus bug).</exception>
        internal static NDArray ApplyCopytoFull(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            NDArray dst = ops[0];
            string casting = p.TryGetValue("casting", out var c) && c.ValueKind != JsonValueKind.Null ? c.GetString() : null;

            NDArray where = null;
            if (p.TryGetValue("where", out var w))
            {
                // A C# literal bool binds the implicit bool → 0-d NDArray conversion: exactly NumPy's where=True/False.
                if (w.ValueKind == JsonValueKind.True)
                    where = true;
                else if (w.ValueKind == JsonValueKind.False)
                    where = false;
                else
                    where = ops[ops.Length - 1];
            }

            if (p.TryGetValue("src_scalar", out var sc))
            {
                CopytoLiteral(dst, sc, casting, where);
                return BaseBuffer(dst);
            }

            // A view of the destination's own buffer genuinely aliases it (the harness built operand 0 over that buffer).
            NDArray src = p.TryGetValue("src_view", out var sv)
                ? new NDArray(dst.Storage, ShapeFrom(sv, dst.Shape.bufferSize))
                : ops[1];
            np.copyto(dst, src, casting, where);
            return BaseBuffer(dst);
        }

        /// <summary>
        ///     <c>np.copyto(dst, literal)</c> with the C# literal type NumPy's Python scalar maps to: a Python int as a
        ///     <c>long</c> (a <c>ulong</c> past long.MaxValue, a <see cref="BigInteger"/> past ulong), a float as a
        ///     <c>double</c>, a complex as a <see cref="Complex"/> — all of which bind the weak-scalar overloads — and a
        ///     Python bool as a <c>bool</c> (strong, as in NumPy).
        /// </summary>
        /// <param name="dst">The destination.</param>
        /// <param name="sc">The <c>src_scalar</c> descriptor (<c>kind</c> plus <c>value</c> or the IEEE bits in
        ///     little-endian hex).</param>
        /// <param name="casting">The casting name (or null).</param>
        /// <param name="where">The mask (or null).</param>
        /// <exception cref="NotSupportedException">An unknown kind.</exception>
        private static void CopytoLiteral(NDArray dst, JsonElement sc, string casting, NDArray where)
        {
            switch (sc.GetProperty("kind").GetString())
            {
                case "bool":
                    np.copyto(dst, sc.GetProperty("value").GetBoolean(), casting, where);
                    return;
                case "int":
                {
                    var v = BigInteger.Parse(sc.GetProperty("value").GetString(), CultureInfo.InvariantCulture);
                    if (v >= long.MinValue && v <= long.MaxValue)
                        np.copyto(dst, (long)v, casting, where);
                    else if (v.Sign > 0 && v <= ulong.MaxValue)
                        np.copyto(dst, (ulong)v, casting, where);
                    else
                        np.copyto(dst, v, casting, where);
                    return;
                }
                case "float":
                    np.copyto(dst, LittleEndianDouble(sc.GetProperty("bits").GetString()), casting, where);
                    return;
                case "complex":
                    np.copyto(dst, new Complex(LittleEndianDouble(sc.GetProperty("re").GetString()),
                        LittleEndianDouble(sc.GetProperty("im").GetString())), casting, where);
                    return;
                default:
                    throw new NotSupportedException($"copyto_full: unknown src_scalar kind {sc.GetProperty("kind")}");
            }
        }

        /// <summary>The double whose IEEE bits are the little-endian hex string (NaN payloads and signed zeros kept).</summary>
        /// <param name="hex">16 hex digits, little-endian byte order.</param>
        /// <returns>The value.</returns>
        private static double LittleEndianDouble(string hex)
            => BitConverter.Int64BitsToDouble(BinaryPrimitives.ReadInt64LittleEndian(Convert.FromHexString(hex)));
    }
}
