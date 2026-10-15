using System;
using System.Collections.Generic;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    public static partial class OpRegistry
    {
        /// <summary>
        ///     One array-parameter ("broadcast") draw of the legacy <see cref="NumPyRandom"/> — the
        ///     <c>params["bargs"]</c> cases of the random_parity tiers (gen_oracle._gen_random_broadcast, api
        ///     "legacy"). Calls the sampler's NDArray overload with the case's positional parameters.
        /// </summary>
        /// <param name="random">A freshly seeded RandomState (the caller seeds it and repeats for <c>draws</c>).</param>
        /// <param name="p">The case params: <c>dist</c>, <c>bargs</c>, optional <c>size</c>.</param>
        /// <param name="ops">The case operands (the array parameters, rebuilt with their exact layouts).</param>
        /// <returns>The draw.</returns>
        /// <exception cref="NotSupportedException">An unknown distribution name (a corpus / registry mismatch).</exception>
        /// <remarks>
        ///     The Python-literal parameters (<c>{"f": x}</c>, <c>{"i": n}</c>) become harness-built 0-d arrays that are
        ///     disposed after the call, so the leak sweep never attributes them to the library.
        /// </remarks>
        internal static NDArray LegacyBroadcastDraw(NumPyRandom random, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            string dist = p["dist"].GetString();
            var (a, owned) = BroadcastArgs(p, ops);
            try
            {
                Shape size = BroadcastSize(p);
                return dist switch
                {
                    "beta" => random.beta(a[0], a[1], size),
                    "exponential" => random.exponential(a[0], size),
                    "uniform" => random.uniform(a[0], a[1], size),
                    "normal" => random.normal(a[0], a[1], size),
                    "standard_gamma" => random.standard_gamma(a[0], size),
                    "gamma" => random.gamma(a[0], a[1], size),
                    "f" => random.f(a[0], a[1], size),
                    "noncentral_f" => random.noncentral_f(a[0], a[1], a[2], size),
                    "chisquare" => random.chisquare(a[0], size),
                    "noncentral_chisquare" => random.noncentral_chisquare(a[0], a[1], size),
                    "standard_t" => random.standard_t(a[0], size),
                    "vonmises" => random.vonmises(a[0], a[1], size),
                    "pareto" => random.pareto(a[0], size),
                    "weibull" => random.weibull(a[0], size),
                    "power" => random.power(a[0], size),
                    "laplace" => random.laplace(a[0], a[1], size),
                    "gumbel" => random.gumbel(a[0], a[1], size),
                    "logistic" => random.logistic(a[0], a[1], size),
                    "lognormal" => random.lognormal(a[0], a[1], size),
                    "rayleigh" => random.rayleigh(a[0], size),
                    "wald" => random.wald(a[0], a[1], size),
                    "triangular" => random.triangular(a[0], a[1], a[2], size),
                    "binomial" => random.binomial(a[0], a[1], size),
                    "negative_binomial" => random.negative_binomial(a[0], a[1], size),
                    "poisson" => random.poisson(a[0], size),
                    "zipf" => random.zipf(a[0], size),
                    "geometric" => random.geometric(a[0], size),
                    "hypergeometric" => random.hypergeometric(a[0], a[1], a[2], size),
                    "logseries" => random.logseries(a[0], size),
                    _ => throw new NotSupportedException($"rnd broadcast dist '{dist}'"),
                };
            }
            finally
            {
                foreach (var x in owned)
                    x.Dispose();
            }
        }

        /// <summary>
        ///     One array-parameter ("broadcast") draw of the PCG64 <see cref="Generator"/> — the <c>params["bargs"]</c>
        ///     cases of the generator_parity tiers (gen_oracle._gen_random_broadcast, api "gen").
        /// </summary>
        /// <param name="rng">A freshly seeded Generator (the caller seeds it and repeats for <c>draws</c>).</param>
        /// <param name="p">The case params: <c>method</c>, <c>bargs</c>, optional <c>size</c> / <c>dtype</c> /
        ///     <c>out</c> ({dtype, shape, order}).</param>
        /// <param name="ops">The case operands (the array parameters, rebuilt with their exact layouts).</param>
        /// <returns>The draw (the <c>out</c> array itself when one is given, as NumPy returns it).</returns>
        /// <exception cref="NotSupportedException">An unknown method name (a corpus / registry mismatch).</exception>
        /// <remarks>
        ///     A harness-built <c>out</c> array is disposed when the call throws (NumPy's refusal cases): otherwise the
        ///     leak sweep would attribute the harness's own buffer to the library. On success it IS the result.
        /// </remarks>
        internal static NDArray GeneratorBroadcastDraw(Generator rng, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            string method = p["method"].GetString();
            var (a, owned) = BroadcastArgs(p, ops);
            NDArray outArr = null;
            try
            {
                Shape size = BroadcastSize(p);
                DType dtype = p.TryGetValue("dtype", out var d) ? DtypeFromName(d.GetString()) : null;
                if (p.TryGetValue("out", out var o))
                    outArr = BuildOut(o);
                NDArray r = method switch
                {
                    "beta" => rng.beta(a[0], a[1], size),
                    "exponential" => rng.exponential(a[0], size),
                    "uniform" => rng.uniform(a[0], a[1], size),
                    "normal" => rng.normal(a[0], a[1], size),
                    "standard_gamma" => rng.standard_gamma(a[0], size, dtype, outArr),
                    "gamma" => rng.gamma(a[0], a[1], size),
                    "f" => rng.f(a[0], a[1], size),
                    "noncentral_f" => rng.noncentral_f(a[0], a[1], a[2], size),
                    "chisquare" => rng.chisquare(a[0], size),
                    "noncentral_chisquare" => rng.noncentral_chisquare(a[0], a[1], size),
                    "standard_t" => rng.standard_t(a[0], size),
                    "vonmises" => rng.vonmises(a[0], a[1], size),
                    "pareto" => rng.pareto(a[0], size),
                    "weibull" => rng.weibull(a[0], size),
                    "power" => rng.power(a[0], size),
                    "laplace" => rng.laplace(a[0], a[1], size),
                    "gumbel" => rng.gumbel(a[0], a[1], size),
                    "logistic" => rng.logistic(a[0], a[1], size),
                    "lognormal" => rng.lognormal(a[0], a[1], size),
                    "rayleigh" => rng.rayleigh(a[0], size),
                    "wald" => rng.wald(a[0], a[1], size),
                    "triangular" => rng.triangular(a[0], a[1], a[2], size),
                    "binomial" => rng.binomial(a[0], a[1], size),
                    "negative_binomial" => rng.negative_binomial(a[0], a[1], size),
                    "poisson" => rng.poisson(a[0], size),
                    "zipf" => rng.zipf(a[0], size),
                    "geometric" => rng.geometric(a[0], size),
                    "hypergeometric" => rng.hypergeometric(a[0], a[1], a[2], size),
                    "logseries" => rng.logseries(a[0], size),
                    _ => throw new NotSupportedException($"grnd broadcast method '{method}'"),
                };
                outArr = null; // handed over: it is the result (or was never used)
                return r;
            }
            finally
            {
                outArr?.Dispose();
                foreach (var x in owned)
                    x.Dispose();
            }
        }

        /// <summary>
        ///     The positional parameters of a broadcast case: <c>{"op": k}</c> is operand k, <c>{"f": x}</c> /
        ///     <c>{"i": n}</c> a Python float / int (a 0-d float64 / int64, as NumPy's <c>PyArray_FROM_OTF</c> converts
        ///     it), <c>{"none": true}</c> Python's <c>None</c> (C# null).
        /// </summary>
        /// <param name="p">The case params (reads <c>bargs</c>).</param>
        /// <param name="ops">The case operands.</param>
        /// <returns>The arguments, and the harness-built 0-d arrays the caller must dispose after the call.</returns>
        private static (NDArray[] Args, List<NDArray> Owned) BroadcastArgs(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var bargs = p["bargs"];
            var args = new NDArray[bargs.GetArrayLength()];
            var owned = new List<NDArray>();
            int i = 0;
            foreach (var e in bargs.EnumerateArray())
            {
                if (e.TryGetProperty("op", out var k))
                    args[i] = ops[k.GetInt32()];
                else if (e.TryGetProperty("f", out var f))
                    owned.Add(args[i] = NDArray.Scalar(f.GetDouble()));
                else if (e.TryGetProperty("i", out var n))
                    owned.Add(args[i] = NDArray.Scalar(n.GetInt64()));
                else
                    args[i] = null; // {"none": true}
                i++;
            }
            return (args, owned);
        }

        /// <summary>
        ///     The case's <c>size</c>: absent is NumPy's <c>None</c> (<c>default</c>), <c>[]</c> is <c>size=()</c>
        ///     (<see cref="Shape.Scalar"/>), anything else the dimensions.
        /// </summary>
        /// <param name="p">The case params.</param>
        /// <returns>The size argument.</returns>
        private static Shape BroadcastSize(IReadOnlyDictionary<string, JsonElement> p)
        {
            if (!p.TryGetValue("size", out var sz))
                return default;
            var dims = ParseLongArray(sz);
            return dims.Length == 0 ? Shape.Scalar : new Shape(dims);
        }

        /// <summary>
        ///     A fresh <c>out</c> array for a Generator broadcast case — NumPy's <c>np.empty(shape, dtype, order)</c>
        ///     (the F-order form built as an F-contiguous copy, its C-order scaffold disposed).
        /// </summary>
        /// <param name="o">The <c>out</c> spec: {dtype, shape, order}.</param>
        /// <returns>The out array.</returns>
        private static NDArray BuildOut(JsonElement o)
        {
            var dims = ParseLongArray(o.GetProperty("shape"));
            var c = new NDArray(DtypeFromName(o.GetProperty("dtype").GetString()), new Shape(dims), true);
            if (o.GetProperty("order").GetString() != "F")
                return c;
            var f = np.asfortranarray(c);
            if (!ReferenceEquals(f, c))
                c.Dispose();
            return f;
        }
    }
}
