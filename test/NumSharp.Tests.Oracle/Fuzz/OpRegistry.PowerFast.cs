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
        ///     Replays one <c>power_fast.jsonl</c> case (gen_oracle.py <c>power_fast</c>, plan
        ///     <c>docs/plans/api-parameter-parity.md</c> X13): <c>np.power</c> or <c>np.float_power</c> with NumPy's
        ///     exponent kinds, out=, where=, dtype= and buffer size, returning the result — or, when an out is given, the
        ///     out's WHOLE base buffer, so a masked-off slot or a write outside the out's view is visible.
        /// </summary>
        /// <param name="p">The case's params: <c>fn</c> (<c>"power"</c> / <c>"float_power"</c>); the exponent as
        ///     <c>exp_scalar</c> (a C# literal: NumPy's weak Python int / float) or the next operand; the out as
        ///     <c>out: "array"</c> (the next operand) or <c>out_view</c> (a view of operand 0's own buffer: in place or
        ///     overlapping); <c>where: "array"</c> (the LAST operand); <c>dtype</c> (a NumPy dtype name); and
        ///     <c>bufsize</c> (NumPy's <c>np.setbufsize</c> for the call; the buffering cost model reads it).</param>
        /// <param name="ops">Operand 0 is the bases; then the exponent array (unless a literal); then the out array
        ///     (unless a view or absent); then the mask.</param>
        /// <returns>The result, or the out's base buffer as a 1-D array.</returns>
        /// <exception cref="NotSupportedException">An unknown <c>fn</c> or literal kind (a corpus bug).</exception>
        internal static NDArray ApplyPowerFast(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            NDArray x = ops[0];
            int k = 1;

            // A Python scalar exponent is the C# literal that binds NumSharp's weak 0-d conversion: a long for an int
            // (the Python ints of the tier fit), a double for a float — the latter as raw bits, so NaN and -0.0 survive.
            NDArray e;
            if (p.TryGetValue("exp_scalar", out var sc))
            {
                e = sc.GetProperty("kind").GetString() switch
                {
                    "int" => NDArray.Scalar(long.Parse(sc.GetProperty("value").GetString(), CultureInfo.InvariantCulture)),
                    "float" => NDArray.Scalar(LittleEndianDouble(sc.GetProperty("bits").GetString())),
                    var kind => throw new NotSupportedException($"power_fast: unknown exp_scalar kind {kind}"),
                };
            }
            else
            {
                e = ops[k++];
            }

            NDArray @out = null;
            if (p.TryGetValue("out_view", out var ov))
                @out = new NDArray(x.Storage, ShapeFrom(ov, x.Shape.bufferSize));   // aliases the bases' buffer
            else if (p.TryGetValue("out", out _))
                @out = ops[k++];

            NDArray where = p.TryGetValue("where", out _) ? ops[ops.Length - 1] : null;
            DType dtype = p.TryGetValue("dtype", out var dt) ? np.dtype(dt.GetString()) : null;
            string fn = p["fn"].GetString();

            // np.setbufsize is thread-local in both libraries; restore it whatever the call does.
            long previous = np.getbufsize();
            if (p.TryGetValue("bufsize", out var bs))
                np.setbufsize(bs.GetInt64());
            try
            {
                NDArray r = fn switch
                {
                    "power" => np.power(x, e, @out, where, dtype),
                    "float_power" => np.float_power(x, e, @out, where, dtype),
                    _ => throw new NotSupportedException($"power_fast: unknown fn {fn}"),
                };
                return @out is null ? r : BaseBuffer(@out);
            }
            finally
            {
                np.setbufsize(previous);
            }
        }
    }
}
