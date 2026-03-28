using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a standard Student's t distribution with df degrees of freedom.
        /// </summary>
        /// <param name="df">Degrees of freedom, must be > 0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized standard Student's t distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_t.html
        ///     <br/>
        ///     A special case of the hyperbolic distribution. As df gets large, the result
        ///     resembles that of the standard normal distribution.
        ///     <br/>
        ///     The probability density function is:
        ///     P(x, df) = Gamma((df+1)/2) / (sqrt(pi*df) * Gamma(df/2)) * (1 + x^2/df)^(-(df+1)/2)
        /// </remarks>
        public NDArray standard_t(double df, Shape size)
            => standard_t(df, size.dimensions);

        /// <summary>
        ///     Draw samples from a standard Student's t distribution with df degrees of freedom.
        /// </summary>
        /// <param name="df">Degrees of freedom, must be > 0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized standard Student's t distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_t.html
        ///     <br/>
        ///     A special case of the hyperbolic distribution. As df gets large, the result
        ///     resembles that of the standard normal distribution.
        /// </remarks>
        public NDArray standard_t(double df, params int[] size)
        {
            // Parameter validation (matches NumPy error message)
            if (df <= 0)
                throw new ArgumentException("df <= 0", nameof(df));

            if (size == null || size.Length == 0)
            {
                return NDArray.Scalar(SampleStandardT(df));
            }

            var result = new NDArray<double>(size);
            ArraySlice<double> resultArray = result.Data<double>();

            for (int i = 0; i < result.size; ++i)
                resultArray[i] = SampleStandardT(df);

            result.ReplaceData(resultArray);
            return result;
        }

        /// <summary>
        ///     Sample a single value from the standard Student's t distribution.
        /// </summary>
        /// <remarks>
        ///     Algorithm from NumPy's random_standard_t in distributions.c:
        ///     num = standard_normal()
        ///     denom = standard_gamma(df/2)
        ///     return sqrt(df/2) * num / sqrt(denom)
        /// </remarks>
        private double SampleStandardT(double df)
        {
            // NumPy's exact implementation from distributions.c
            double num = NextGaussian();
            double denom = SampleStandardGamma(df / 2.0);
            return Math.Sqrt(df / 2.0) * num / Math.Sqrt(denom);
        }

        /// <summary>
        ///     Sample a single value from the standard gamma distribution (scale=1).
        ///     Uses NumPy's algorithm: exponential for shape=1, Marsaglia-Tsang otherwise.
        /// </summary>
        private double SampleStandardGamma(double shape)
        {
            if (shape == 1.0)
            {
                // Shape=1 is exponential distribution: -log(1 - U)
                // This matches NumPy's rk_standard_gamma behavior
                return -Math.Log(1.0 - randomizer.NextDouble());
            }
            else if (shape < 1.0)
            {
                // For shape < 1, use: gamma(shape) = gamma(shape+1) * U^(1/shape)
                double d = shape + 1.0 - 1.0 / 3.0;
                double c = (1.0 / 3.0) / Math.Sqrt(d);
                double u = randomizer.NextDouble();
                return SampleMarsaglia(d, c) * Math.Pow(u, 1.0 / shape);
            }
            else
            {
                double d = shape - 1.0 / 3.0;
                double c = (1.0 / 3.0) / Math.Sqrt(d);
                return SampleMarsaglia(d, c);
            }
        }

        /// <summary>
        ///     Marsaglia and Tsang's method for gamma sampling.
        /// </summary>
        private double SampleMarsaglia(double d, double c)
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

                if (U < 1.0 - 0.0331 * x2 * x2)
                {
                    return d * v;
                }

                if (Math.Log(U) < 0.5 * x2 + d * (1.0 - v + Math.Log(v)))
                {
                    return d * v;
                }
            }
        }
    }
}
