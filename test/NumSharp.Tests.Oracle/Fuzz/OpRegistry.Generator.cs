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
        /// <param name="p">The case params.</param>
        /// <param name="ops">The case operands — the array parameters of the <c>bargs</c> broadcast cases; null (or
        ///     empty) for the scalar-parameter stream cases.</param>
        /// <returns>The recorded draw (the last of <c>draws</c> identical calls).</returns>
        internal static NDArray GeneratorDraw(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops = null)
        {
            string method = p["method"].GetString();
            long seed = p["seed"].GetInt64();
            int draws = p.TryGetValue("draws", out var dr) ? dr.GetInt32() : 1;

            double A(int i) => p["args"][i].GetDouble();
            long AL(int i) => p["args"][i].GetInt64();
            Shape S() => p.TryGetValue("size", out var sz) ? new Shape(ParseLongArray(sz)) : default;
            DType Dt() => p.TryGetValue("dtype", out var d) ? DtypeFromName(d.GetString()) : null;

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
                if (p.ContainsKey("bargs"))
                {
                    // Array-valued parameters: the NDArray overloads over the case's operands.
                    result = GeneratorBroadcastDraw(rng, p, ops);
                    continue;
                }
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
                        result = rng.choice(AL(0), S(), replace, pv, shuffle: cshuffle);
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
                    // ---- the distribution surface: NumPy's positional parameters, then size ----
                    case "beta": result = rng.beta(A(0), A(1), S()); break;
                    case "chisquare": result = rng.chisquare(A(0), S()); break;
                    case "f": result = rng.f(A(0), A(1), S()); break;
                    case "noncentral_chisquare": result = rng.noncentral_chisquare(A(0), A(1), S()); break;
                    case "noncentral_f": result = rng.noncentral_f(A(0), A(1), A(2), S()); break;
                    case "standard_cauchy": result = rng.standard_cauchy(S()); break;
                    case "standard_t": result = rng.standard_t(A(0), S()); break;
                    case "vonmises": result = rng.vonmises(A(0), A(1), S()); break;
                    case "pareto": result = rng.pareto(A(0), S()); break;
                    case "weibull": result = rng.weibull(A(0), S()); break;
                    case "power": result = rng.power(A(0), S()); break;
                    case "laplace": result = rng.laplace(A(0), A(1), S()); break;
                    case "gumbel": result = rng.gumbel(A(0), A(1), S()); break;
                    case "logistic": result = rng.logistic(A(0), A(1), S()); break;
                    case "lognormal": result = rng.lognormal(A(0), A(1), S()); break;
                    case "rayleigh": result = rng.rayleigh(A(0), S()); break;
                    case "wald": result = rng.wald(A(0), A(1), S()); break;
                    case "triangular": result = rng.triangular(A(0), A(1), A(2), S()); break;
                    case "binomial": result = rng.binomial(AL(0), A(1), S()); break;
                    case "negative_binomial": result = rng.negative_binomial(A(0), A(1), S()); break;
                    case "poisson": result = rng.poisson(A(0), S()); break;
                    case "zipf": result = rng.zipf(A(0), S()); break;
                    case "geometric": result = rng.geometric(A(0), S()); break;
                    case "hypergeometric": result = rng.hypergeometric(AL(0), AL(1), AL(2), S()); break;
                    case "logseries": result = rng.logseries(A(0), S()); break;
                    case "multinomial":
                    {
                        // Harness-built operands: the sampler reads them and retains nothing, so they die here.
                        using var pvals = NestedArray(p["pvals"], integer: false);
                        if (p.TryGetValue("narr", out var narr))
                        {
                            using var counts = NestedArray(narr, integer: true);
                            result = rng.multinomial(counts, pvals, S());
                        }
                        else
                        {
                            result = rng.multinomial(AL(0), pvals, S());
                        }
                        break;
                    }
                    case "dirichlet":
                        result = rng.dirichlet(ParseDoubleArray(p["alpha"]), S());
                        break;
                    case "multivariate_hypergeometric":
                        result = rng.multivariate_hypergeometric(ParseLongArray(p["colors"]), AL(0), S(),
                            p.TryGetValue("mvmethod", out var mv) ? mv.GetString() : "marginals");
                        break;
                    default:
                        throw new NotSupportedException($"grnd method '{method}' not wired in OpRegistry");
                }
            }
            return result;
        }

        /// <summary>
        ///     A rectangular nested JSON array (<c>[[5], [10]]</c>, <c>[0.2, 0.3]</c>) as an int64 or float64 NDArray of the
        ///     same shape — the array-valued parameters of the grnd multinomial vector path.
        /// </summary>
        /// <param name="e">The JSON array (nested to any depth, rectangular).</param>
        /// <param name="integer">True for an int64 result (counts), false for float64 (probabilities).</param>
        /// <returns>A fresh C-contiguous array the caller owns.</returns>
        private static NDArray NestedArray(JsonElement e, bool integer)
        {
            var dims = new List<long>();
            for (var cur = e; cur.ValueKind == JsonValueKind.Array; cur = cur[0])
            {
                dims.Add(cur.GetArrayLength());
                if (cur.GetArrayLength() == 0)
                    break;
            }
            var flat = new List<double>();
            void Walk(JsonElement x)
            {
                if (x.ValueKind == JsonValueKind.Array)
                    foreach (var y in x.EnumerateArray())
                        Walk(y);
                else
                    flat.Add(x.GetDouble());
            }
            Walk(e);
            var shape = new Shape(dims.ToArray());
            if (integer)
            {
                var longs = new long[flat.Count];
                for (int i = 0; i < longs.Length; i++)
                    longs[i] = (long)flat[i];
                using var flatL = np.array(longs);
                using var viewL = flatL.reshape(shape); // the reshape view is harness-private too
                return viewL.copy();
            }
            using var flatD = np.array(flat.ToArray());
            using var viewD = flatD.reshape(shape);
            return viewD.copy();
        }

        private static DType DtypeFromName(string n) => n switch
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
