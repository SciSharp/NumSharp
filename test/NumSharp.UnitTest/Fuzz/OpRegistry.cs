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

                // Comparison -> bool result.
                case "equal": return ops[0] == ops[1];
                case "not_equal": return ops[0] != ops[1];
                case "less": return ops[0] < ops[1];
                case "greater": return ops[0] > ops[1];
                case "less_equal": return ops[0] <= ops[1];
                case "greater_equal": return ops[0] >= ops[1];

                default:
                    throw new NotSupportedException($"op '{op}' is not registered in OpRegistry");
            }
        }
    }
}
