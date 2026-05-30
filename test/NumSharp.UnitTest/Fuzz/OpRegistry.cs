using System;
using System.Collections.Generic;
using System.Text.Json;
using NumSharp;

namespace NumSharp.UnitTest.Fuzz
{
    /// <summary>
    ///     Maps a corpus op-name to the NumSharp call that produces the operand result.
    ///     The matching NumPy call lives in oracle/gen_oracle.py; this is the C# side of that pair.
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

                // Selection.
                case "where": return np.where(ops[0], ops[1], ops[2]);
                case "place": np.place(ops[0], ops[1], ops[2]); return ops[0]; // mutates arr; result IS arr

                // Linear algebra (T8). NumPy is the oracle for value, result dtype, and broadcast shape.
                case "matmul": return np.matmul(ops[0], ops[1]);
                case "dot": return np.dot(ops[0], ops[1]);
                case "outer": return np.outer(ops[0], ops[1]);

                // Reductions (axis/keepdims params).
                case "sum": case "prod": case "min": case "max": case "mean":
                case "std": case "var": case "argmax": case "argmin": case "all": case "any":
                    return ApplyReduce(op, ParseAxis(p), ParseKeepdims(p), ops[0]);

                default:
                    throw new NotSupportedException($"op '{op}' is not registered in OpRegistry");
            }
        }

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
                default: throw new NotSupportedException(op);
            }
        }
    }
}
