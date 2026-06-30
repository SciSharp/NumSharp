using System;
using System.Collections.Generic;
using System.Text.Json;
using NumSharp;

namespace NumSharp.UnitTest.Fuzz
{
    /// <summary>
    ///     Maps a corpus op-name to the NumSharp call that produces the operand result.
    ///     The matching NumPy call lives in test/oracle/gen_oracle.py; this is the C# side of that pair.
    ///     New op tiers (binary arith, comparison, unary, reductions, where/place) extend this switch.
    /// </summary>
    public static class OpRegistry
    {
        public static NDArray Apply(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            switch (op)
            {
                case "astype":
                    return ops[0].astype(FuzzCorpus.DtypeToTC(p["dtype"].GetString()));

                // Binary arithmetic (NEP50 promotion). NumPy is the oracle for the result dtype.
                case "add": return ops[0] + ops[1];
                case "subtract": return ops[0] - ops[1];
                case "multiply": return ops[0] * ops[1];
                case "divide": return ops[0] / ops[1];
                case "floor_divide": return np.floor_divide(ops[0], ops[1]);
                case "mod": return np.mod(ops[0], ops[1]);
                case "power": return np.power(ops[0], ops[1]);

                // Unary.
                case "negative": return np.negative(ops[0]);
                case "abs": return np.abs(ops[0]);
                case "sign": return np.sign(ops[0]);
                case "sqrt": return np.sqrt(ops[0]);
                case "cbrt": return np.cbrt(ops[0]);
                case "square": return np.square(ops[0]);
                case "reciprocal": return np.reciprocal(ops[0]);
                case "floor": return np.floor(ops[0]);
                case "ceil": return np.ceil(ops[0]);
                case "trunc": return np.trunc(ops[0]);
                case "sin": return np.sin(ops[0]);
                case "cos": return np.cos(ops[0]);
                case "tan": return np.tan(ops[0]);
                case "exp": return np.exp(ops[0]);
                case "log": return np.log(ops[0]);

                // Unary stragglers (W3): transcendental / hyperbolic / inverse-trig / angle conv.
                case "exp2": return np.exp2(ops[0]);
                case "expm1": return np.expm1(ops[0]);
                case "log2": return np.log2(ops[0]);
                case "log10": return np.log10(ops[0]);
                case "log1p": return np.log1p(ops[0]);
                case "sinh": return np.sinh(ops[0]);
                case "cosh": return np.cosh(ops[0]);
                case "tanh": return np.tanh(ops[0]);
                case "arcsin": return np.arcsin(ops[0]);
                case "arccos": return np.arccos(ops[0]);
                case "arctan": return np.arctan(ops[0]);
                case "deg2rad": return np.deg2rad(ops[0]);
                case "rad2deg": return np.rad2deg(ops[0]);
                case "positive": return np.positive(ops[0]);

                // Bitwise & shift (T9). Integer + bool dtypes; NumPy is the oracle.
                case "bitwise_and": return ops[0] & ops[1];
                case "bitwise_or": return ops[0] | ops[1];
                case "bitwise_xor": return ops[0] ^ ops[1];
                case "invert": return np.invert(ops[0]);
                case "left_shift": return np.left_shift(ops[0], ops[1]);
                case "right_shift": return np.right_shift(ops[0], ops[1]);

                // Comparison -> bool result.
                case "equal": return ops[0] == ops[1];
                case "not_equal": return ops[0] != ops[1];
                case "less": return ops[0] < ops[1];
                case "greater": return ops[0] > ops[1];
                case "less_equal": return ops[0] <= ops[1];
                case "greater_equal": return ops[0] >= ops[1];

                // Shape manipulation (T7).
                case "ravel": return np.ravel(ops[0]);
                case "transpose": return np.transpose(ops[0]);
                case "expand_dims": return np.expand_dims(ops[0], p["axis"].GetInt32());
                case "squeeze": return np.squeeze(ops[0]);
                case "roll": return np.roll(ops[0], p["shift"].GetInt32());
                case "repeat": return np.repeat(ops[0], p["repeats"].GetInt32());
                case "tile": return np.tile(ops[0], p["reps"].GetInt32());
                case "reshape": return np.reshape(ops[0], ParseIntArray(p["shape"]));
                case "swapaxes": return np.swapaxes(ops[0], p["a1"].GetInt32(), p["a2"].GetInt32());
                case "moveaxis": return np.moveaxis(ops[0], p["src"].GetInt32(), p["dst"].GetInt32());
                case "delete": return np.delete(ops[0], p["obj"].GetInt32(), p["axis"].GetInt32());
                case "atleast_1d": return np.atleast_1d(ops[0]);
                case "atleast_2d": return np.atleast_2d(ops[0]);
                case "atleast_3d": return np.atleast_3d(ops[0]);
                case "concatenate": return np.concatenate((ops[0], ops[1]), p["axis"].GetInt32());
                case "stack": return np.stack(new[] { ops[0], ops[1] }, p["axis"].GetInt32());
                case "hstack": return np.hstack(ops[0], ops[1]);
                case "vstack": return np.vstack(ops[0], ops[1]);
                case "dstack": return np.dstack(ops[0], ops[1]);
                case "pad": return np.pad(ops[0], p["pad_width"].GetInt32(), p["mode"].GetString());

                // Multi-output (T15): modf split into its two outputs (fractional / integral).
                case "modf_frac": return np.modf(ops[0]).Item1;
                case "modf_int": return np.modf(ops[0]).Item2;

                // Cumulative scans + finite differences (T11).
                case "cumsum": return np.cumsum(ops[0], ParseAxis(p));
                case "cumprod": return np.cumprod(ops[0], ParseAxis(p));
                case "diff": return np.diff(ops[0], p["n"].GetInt32(), p["axis"].GetInt32());

                // In-place out= aliasing (W11): the output buffer IS an input operand.
                case "maximum_out": np.maximum(ops[0], ops[1], ops[0]); return ops[0];
                case "minimum_out": np.minimum(ops[0], ops[1], ops[0]); return ops[0];
                case "clip_out": np.clip(ops[0], ops[1], ops[2], ops[0]); return ops[0];

                // Parameter sweep (W12): ddof=1 std/var, order='F' ravel.
                case "std_ddof": { var ax = ParseAxis(p); int dd = p["ddof"].GetInt32();
                    return ax.HasValue ? np.std(ops[0], ax.Value, false, dd) : np.std(ops[0], false, dd); }
                case "var_ddof": { var ax = ParseAxis(p); int dd = p["ddof"].GetInt32();
                    return ax.HasValue ? np.var(ops[0], ax.Value, false, dd) : np.var(ops[0], false, dd); }
                case "ravel_f": return np.ravel(ops[0], 'F');

                // Statistics (T12).
                case "median": return np.median(ops[0], ParseAxis(p), keepdims: ParseKeepdims(p));
                case "average": return np.average(ops[0], ParseAxis(p), null, ParseKeepdims(p));
                case "ptp": return np.ptp(ops[0], ParseAxis(p), null, ParseKeepdims(p));
                case "count_nonzero": return np.count_nonzero(ops[0], ParseAxis(p).Value, ParseKeepdims(p));
                case "percentile": return np.percentile(ops[0], p["q"].GetDouble(), ParseAxis(p),
                    keepdims: ParseKeepdims(p));
                case "quantile": return np.quantile(ops[0], p["q"].GetDouble(), ParseAxis(p),
                    keepdims: ParseKeepdims(p));
                case "clip": return np.clip(ops[0], ops[1], ops[2]);

                // Logic & element-wise extrema (T13).
                case "isnan": return np.isnan(ops[0]);
                case "isinf": return np.isinf(ops[0]);
                case "isfinite": return np.isfinite(ops[0]);
                case "maximum": return np.maximum(ops[0], ops[1]);
                case "minimum": return np.minimum(ops[0], ops[1]);
                case "fmax": return np.fmax(ops[0], ops[1]);
                case "fmin": return np.fmin(ops[0], ops[1]);
                case "isclose": return np.isclose(ops[0], ops[1]);

                // Selection.
                case "where": return np.where(ops[0], ops[1], ops[2]);
                case "place": np.place(ops[0], ops[1], ops[2]); return ops[0]; // mutates arr; result IS arr

                // W15 copyto: cross-dtype / strided-dst / scalar-broadcast-src. dst (ops[0]) is mutated
                // in place and IS the result; casting rule comes from params (default same_kind).
                case "copyto":
                    np.copyto(ops[0], ops[1], p.TryGetValue("casting", out var cast) ? cast.GetString() : "same_kind");
                    return ops[0];

                // W15 copyto OVERLAP: dst and src are two views of ONE buffer (operand 0). Rebuild both
                // from params so they genuinely alias, exercising NumPy's COPY_IF_OVERLAP path.
                case "copyto_overlap":
                {
                    var storage = ops[0].Storage;
                    long bufSize = ops[0].size;
                    var dst = new NDArray(storage, ShapeFrom(p["dst"], bufSize));
                    var src = new NDArray(storage, ShapeFrom(p["src"], bufSize));
                    np.copyto(dst, src);
                    return dst;
                }

                // Sorting / searching (T14).
                case "argsort": return ApplyArgsort(ops[0], p["axis"].GetInt32());
                case "searchsorted": return np.searchsorted(ops[0], ops[1], p["side"].GetString());
                case "nonzero": return np.nonzero(ops[0])[0]; // 1-D: single int64 index array

                // Linear algebra (T8). NumPy is the oracle for value, result dtype, and broadcast shape.
                case "matmul": return np.matmul(ops[0], ops[1]);
                case "dot": return np.dot(ops[0], ops[1]);
                case "outer": return np.outer(ops[0], ops[1]);

                // Reductions (axis/keepdims params).
                case "sum": case "prod": case "min": case "max": case "mean":
                case "std": case "var": case "argmax": case "argmin": case "all": case "any":
                // NaN-aware reductions (T10).
                case "nansum": case "nanprod": case "nanmax": case "nanmin": case "nanmean":
                case "nanstd": case "nanvar": case "nanmedian":
                    return ApplyReduce(op, ParseAxis(p), ParseKeepdims(p), ops[0]);

                default:
                    throw new NotSupportedException($"op '{op}' is not registered in OpRegistry");
            }
        }

        private static NDArray ApplyArgsort(NDArray a, int axis) => a.typecode switch
        {
            NPTypeCode.Byte => np.argsort<byte>(a, axis),
            NPTypeCode.SByte => np.argsort<sbyte>(a, axis),
            NPTypeCode.Int16 => np.argsort<short>(a, axis),
            NPTypeCode.UInt16 => np.argsort<ushort>(a, axis),
            NPTypeCode.Int32 => np.argsort<int>(a, axis),
            NPTypeCode.UInt32 => np.argsort<uint>(a, axis),
            NPTypeCode.Int64 => np.argsort<long>(a, axis),
            NPTypeCode.UInt64 => np.argsort<ulong>(a, axis),
            NPTypeCode.Char => np.argsort<char>(a, axis),
            NPTypeCode.Single => np.argsort<float>(a, axis),
            NPTypeCode.Double => np.argsort<double>(a, axis),
            NPTypeCode.Half => np.argsort<Half>(a, axis),
            NPTypeCode.Boolean => np.argsort<bool>(a, axis),
            NPTypeCode.Decimal => np.argsort<decimal>(a, axis),
            NPTypeCode.Complex => np.argsort<System.Numerics.Complex>(a, axis),
            _ => throw new NotSupportedException($"argsort<{a.typecode}> not wired in OpRegistry")
        };

        private static int[] ParseIntArray(JsonElement arr)
        {
            var list = new List<int>();
            foreach (var e in arr.EnumerateArray())
                list.Add(e.GetInt32());
            return list.ToArray();
        }

        private static long[] ParseLongArray(JsonElement arr)
        {
            var list = new List<long>();
            foreach (var e in arr.EnumerateArray())
                list.Add(e.GetInt64());
            return list.ToArray();
        }

        // Rebuild an aliasing view (shape, element-strides, element-offset) over a shared buffer —
        // the overlap-case counterpart to FuzzCorpus.Reconstruct (which owns its buffer).
        private static Shape ShapeFrom(JsonElement v, long bufferSize)
            => new Shape(ParseLongArray(v.GetProperty("shape")), ParseLongArray(v.GetProperty("strides")),
                         v.GetProperty("offset").GetInt64(), bufferSize);

        private static int? ParseAxis(IReadOnlyDictionary<string, JsonElement> p)
            => p.TryGetValue("axis", out var ax) && ax.ValueKind != JsonValueKind.Null ? ax.GetInt32() : (int?)null;

        private static bool ParseKeepdims(IReadOnlyDictionary<string, JsonElement> p)
            => p.TryGetValue("keepdims", out var kd) && kd.GetBoolean();

        private static NDArray ApplyReduce(string op, int? axis, bool keepdims, NDArray a)
        {
            switch (op)
            {
                case "sum": return np.sum(a, axis, keepdims);
                case "prod": return np.prod(a, axis, (Type)null, keepdims);
                case "min": return np.min(a, axis, keepdims);
                case "max": return np.max(a, axis, keepdims);
                case "mean": return axis.HasValue ? np.mean(a, axis.Value, keepdims) : np.mean(a, keepdims);
                case "std": return axis.HasValue ? np.std(a, axis.Value, keepdims) : np.std(a, keepdims);
                case "var": return axis.HasValue ? np.var(a, axis.Value, keepdims) : np.var(a, keepdims);
                case "argmax": return np.argmax(a, axis.Value, keepdims);
                case "argmin": return np.argmin(a, axis.Value, keepdims);
                case "all": return np.all(a, axis, null, keepdims);
                case "any": return np.any(a, axis, null, keepdims);
                case "nansum": return np.nansum(a, axis, keepdims);
                case "nanprod": return np.nanprod(a, axis, keepdims);
                case "nanmax": return np.nanmax(a, axis, keepdims);
                case "nanmin": return np.nanmin(a, axis, keepdims);
                case "nanmean": return np.nanmean(a, axis, keepdims);
                case "nanstd": return np.nanstd(a, axis, keepdims);
                case "nanvar": return np.nanvar(a, axis, keepdims);
                case "nanmedian": return np.nanmedian(a, axis, keepdims: keepdims);
                default: throw new NotSupportedException(op);
            }
        }
    }
}
