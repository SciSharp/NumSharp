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
                default:
                    throw new NotSupportedException($"op '{op}' is not registered in OpRegistry");
            }
        }
    }
}
