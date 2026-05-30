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

                default:
                    throw new NotSupportedException($"op '{op}' is not registered in OpRegistry");
            }
        }
    }
}
