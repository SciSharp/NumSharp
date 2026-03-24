using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a negative binomial distribution.
        /// </summary>
        /// <param name="n">Parameter of the distribution, > 0 (number of successes).</param>
        /// <param name="p">Parameter of the distribution, 0 &lt; p &lt;= 1 (probability of success).</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized negative binomial distribution (integers >= 0).</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.negative_binomial.html
        ///     <br/>
        ///     The negative binomial distribution models the number of failures before n successes,
        ///     where each trial has probability p of success.
        ///     <br/>
        ///     For this distribution:
        ///     - mean = n * (1-p) / p
        ///     - variance = n * (1-p) / p^2
        ///     <br/>
        ///     Uses gamma-Poisson mixture: Y ~ Gamma(n, (1-p)/p), X ~ Poisson(Y)
        /// </remarks>
        public NDArray negative_binomial(double n, double p, Shape size = default)
        {
            if (n <= 0)
                throw new ArgumentException("n <= 0", nameof(n));
            if (p <= 0 || p > 1 || double.IsNaN(p))
                throw new ArgumentException("p < 0, p > 1 or p is NaN", nameof(p));

            if (size.IsEmpty)
            {
                return NDArray.Scalar(SampleNegativeBinomial(n, p));
            }

            var ret = new NDArray<long>(size);
            ArraySlice<long> data = ret.Data<long>();

            for (int i = 0; i < ret.size; i++)
            {
                data[i] = SampleNegativeBinomial(n, p);
            }

            return ret;
        }

        /// <summary>
        ///     Draw samples from a negative binomial distribution.
        /// </summary>
        /// <param name="n">Parameter of the distribution, > 0.</param>
        /// <param name="p">Parameter of the distribution, 0 &lt; p &lt;= 1.</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the parameterized negative binomial distribution.</returns>
        public NDArray negative_binomial(double n, double p, params int[] size)
            => negative_binomial(n, p, new Shape(size));

        /// <summary>
        ///     Draw samples from a negative binomial distribution.
        /// </summary>
        /// <param name="n">Parameter of the distribution, > 0.</param>
        /// <param name="p">Parameter of the distribution, 0 &lt; p &lt;= 1.</param>
        /// <param name="size">Output shape as single int.</param>
        /// <returns>Drawn samples from the parameterized negative binomial distribution.</returns>
        public NDArray negative_binomial(double n, double p, int size)
            => negative_binomial(n, p, new Shape(size));

        /// <summary>
        ///     Sample from the negative binomial distribution using gamma-Poisson mixture.
        /// </summary>
        /// <remarks>
        ///     Based on NumPy's random_negative_binomial in distributions.c:
        ///     Y = random_gamma(n, (1-p)/p)
        ///     return random_poisson(Y)
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private long SampleNegativeBinomial(double n, double p)
        {
            // Special case: p = 1 means 0 failures before success
            if (p == 1.0)
                return 0L;

            // Gamma-Poisson mixture
            double scale = (1.0 - p) / p;
            double Y = SampleGamma(n, scale);
            return SamplePoisson(Y);
        }

        /// <summary>
        ///     Sample a single value from the Gamma distribution.
        ///     Uses Marsaglia and Tsang's method.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double SampleGamma(double shape, double scale)
        {
            if (shape < 1.0)
            {
                // For shape < 1, use: Gamma(shape) = Gamma(shape+1) * U^(1/shape)
                double d = shape + 1.0 - 1.0 / 3.0;
                double c = (1.0 / 3.0) / Math.Sqrt(d);
                double u = randomizer.NextDouble();
                return scale * SampleGammaMarsaglia(d, c) * Math.Pow(u, 1.0 / shape);
            }
            else
            {
                double d = shape - 1.0 / 3.0;
                double c = (1.0 / 3.0) / Math.Sqrt(d);
                return scale * SampleGammaMarsaglia(d, c);
            }
        }

        /// <summary>
        ///     Marsaglia and Tsang's method for Gamma sampling.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double SampleGammaMarsaglia(double d, double c)
        {
            while (true)
            {
                double x, t, v;

                do
                {
                    x = NextGaussian();
                    t = 1.0 + c * x;
                    v = t * t * t;
                } while (v <= 0);

                double U = randomizer.NextDouble();
                double x2 = x * x;

                if (U < 1 - 0.0331 * x2 * x2)
                {
                    return d * v;
                }

                if (Math.Log(U) < 0.5 * x2 + d * (1.0 - v + Math.Log(v)))
                {
                    return d * v;
                }
            }
        }

        /// <summary>
        ///     Sample a single value from the Poisson distribution.
        ///     Uses Knuth's algorithm for small lambda, and rejection for large lambda.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private long SamplePoisson(double lambda)
        {
            if (lambda == 0)
                return 0L;

            if (lambda < 30)
            {
                // Knuth's algorithm for small lambda
                double L = Math.Exp(-lambda);
                double p = 1.0;
                long k = 0;

                do
                {
                    k++;
                    p *= randomizer.NextDouble();
                } while (p > L);

                return k - 1;
            }
            else
            {
                // For large lambda, use rejection method (Devroye, 1986)
                // Based on NumPy's implementation
                double sqrtLam = Math.Sqrt(lambda);
                double logLam = Math.Log(lambda);
                double b = 0.931 + 2.53 * sqrtLam;
                double a = -0.059 + 0.02483 * b;
                double vr = 0.9277 - 3.6224 / (b - 2);

                while (true)
                {
                    double U = randomizer.NextDouble() - 0.5;
                    double V = randomizer.NextDouble();
                    double us = 0.5 - Math.Abs(U);
                    long k = (long)Math.Floor((2 * a / us + b) * U + lambda + 0.43);

                    if (k < 0)
                        continue;

                    if (us >= 0.07 && V <= vr)
                        return k;

                    if (us < 0.013 && V > us)
                        continue;

                    double kf = k;
                    double logFac = LogFactorial(k);
                    double p = Math.Exp(-lambda + kf * logLam - logFac);

                    if (V <= p)
                        return k;
                }
            }
        }

        /// <summary>
        ///     Compute log(k!) using Stirling's approximation for large k.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double LogFactorial(long k)
        {
            if (k <= 1)
                return 0.0;

            if (k < 20)
            {
                // Direct computation for small values
                double result = 0.0;
                for (long i = 2; i <= k; i++)
                    result += Math.Log(i);
                return result;
            }

            // Stirling's approximation for larger values
            double kd = k;
            return kd * Math.Log(kd) - kd + 0.5 * Math.Log(2 * Math.PI * kd)
                   + 1.0 / (12.0 * kd) - 1.0 / (360.0 * kd * kd * kd);
        }
    }
}
