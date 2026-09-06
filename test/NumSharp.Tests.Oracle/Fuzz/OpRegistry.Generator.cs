using System;
using System.Collections.Generic;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    public static partial class OpRegistry
    {
        /// <summary>
        ///     The "grnd" op — the modern PCG64 <see cref="Generator"/> (np.random.default_rng) stream
        ///     tiers (generator_parity[.host].jsonl), plus the two new RandomState helpers
        ///     np.random.random_integers / np.random.bytes. Seeds a FRESH generator/state per case so
        ///     replaying the oracle never mutates the global np.random stream. Pairs 1:1 with
        ///     gen_oracle.gen_generator_parity's `run` dispatcher.
        /// </summary>
        internal static NDArray GeneratorDraw(IReadOnlyDictionary<string, JsonElement> p)
        {
            string method = p["method"].GetString();
            long seed = p["seed"].GetInt64();
            int draws = p.TryGetValue("draws", out var dr) ? dr.GetInt32() : 1;

            double A(int i) => p["args"][i].GetDouble();
            long AL(int i) => p["args"][i].GetInt64();
            Shape S() => p.TryGetValue("size", out var sz) ? new Shape(ParseLongArray(sz)) : default;
            Type Dt() => p.TryGetValue("dtype", out var d) ? DtypeFromName(d.GetString()) : null;

            // ---- RandomState helpers (fresh instance) ----
            if (method == "random_integers")
            {
                var rs = np.random.RandomState();
                rs.seed((uint)seed);
                long? high = p["args"].GetArrayLength() < 2 ? (long?)null : AL(1);
                NDArray r = null;
                for (int k = 0; k < draws; k++)
                {
                    r?.Dispose();   // draws>1 pins advancement — dispose each superseded draw
                    r = rs.random_integers(AL(0), high, S());
                }
                return r;
            }
            if (method == "rs_bytes")
            {
                var rs = np.random.RandomState();
                rs.seed((uint)seed);
                NDArray b = null; // np.random.bytes now returns a 1-D uint8 NDArray directly
                for (int k = 0; k < draws; k++)
                {
                    b?.Dispose();   // draws>1 pins advancement — dispose each superseded draw
                    b = rs.bytes(AL(0));
                }
                return b;
            }

            // ---- Generator (PCG64) ----
            var rng = np.random.default_rng(seed);
            NDArray result = null;
            for (int k = 0; k < draws; k++)
            {
                result?.Dispose();   // draws>1 pins advancement — dispose each superseded draw
                switch (method)
                {
                    case "random":
                        result = rng.random(S(), Dt());
                        break;
                    case "integers":
                        result = rng.integers(AL(0), AL(1), S(), Dt() ?? np.int64,
                            p.TryGetValue("endpoint", out var e) && e.GetBoolean());
                        break;
                    case "uniform":
                        result = rng.uniform(A(0), A(1), S());
                        break;
                    case "permutation":
                        result = rng.permutation(AL(0));
                        break;
                    case "shuffle":
                    {
                        var arr = np.arange(AL(0));
                        rng.shuffle(arr);
                        result = arr;
                        break;
                    }
                    case "choice":
                    {
                        NDArray pv = p.TryGetValue("p", out var pj) ? np.array(ParseDoubleArray(pj)) : null;
                        bool replace = !p.TryGetValue("replace", out var rp) || rp.GetBoolean();
                        bool cshuffle = !p.TryGetValue("cshuffle", out var cs) || cs.GetBoolean();
                        result = rng.choice(AL(0), S(), replace, pv, cshuffle);
                        pv?.Dispose();   // harness-built probability array; choice reads it, never retains it
                        break;
                    }
                    case "bytes":
                        result = rng.bytes(AL(0)); // returns a 1-D uint8 NDArray directly
                        break;
                    case "standard_normal":
                        result = rng.standard_normal(S(), Dt());
                        break;
                    case "standard_exponential":
                        result = rng.standard_exponential(S(), Dt(),
                            p.TryGetValue("emethod", out var em) ? em.GetString() : "zig");
                        break;
                    case "normal":
                        result = rng.normal(A(0), A(1), S());
                        break;
                    case "exponential":
                        result = rng.exponential(A(0), S());
                        break;
                    case "standard_gamma":
                        result = rng.standard_gamma(A(0), S(), Dt());
                        break;
                    case "gamma":
                        result = rng.gamma(A(0), A(1), S());
                        break;
                    default:
                        throw new NotSupportedException($"grnd method '{method}' not wired in OpRegistry");
                }
            }
            return result;
        }

        private static Type DtypeFromName(string n) => n switch
        {
            "int8" => np.int8,
            "int16" => np.int16,
            "int32" => np.int32,
            "int64" => np.int64,
            "uint8" => np.uint8,
            "uint16" => np.uint16,
            "uint32" => np.uint32,
            "uint64" => np.uint64,
            "bool" => np.@bool,
            "float32" => np.float32,
            "float64" => np.float64,
            _ => throw new NotSupportedException($"grnd dtype '{n}'"),
        };
    }
}
