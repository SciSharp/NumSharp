using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from the Laplace or double exponential distribution with
        ///     specified location (or mean) and scale (decay).
        /// </summary>
        /// <param name="loc">The position of the distribution peak. Default is 0.</param>
        /// <param name="scale">The exponential decay. Must be non-negative. Default is 1.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized Laplace distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.laplace.html
        ///     <br/>
        ///     The Laplace distribution is similar to the Gaussian/normal distribution,
        ///     but is sharper at the peak and has fatter tails. It represents the
        ///     difference between two independent, identically distributed exponential
        ///     random variables.
        ///     <br/>
        ///     The probability density function is:
        ///     f(x; μ, λ) = (1/2λ) * exp(-|x - μ| / λ)
        ///     <br/>
        ///     where μ is the location parameter and λ is the scale parameter.
        /// </remarks>
        public NDArray laplace(double loc = 0.0, double scale = 1.0, Shape size = default)
        {
            if (scale < 0)
                throw new ArgumentException("scale < 0", nameof(scale));

            if (size.IsEmpty)
            {
                return NDArray.Scalar(SampleLaplace(loc, scale));
            }

            var ret = new NDArray<double>(size);
            ArraySlice<double> data = ret.Data<double>();

            for (int i = 0; i < ret.size; i++)
            {
                data[i] = SampleLaplace(loc, scale);
            }

            return ret;
        }

        /// <summary>
        ///     Draw samples from the Laplace or double exponential distribution.
        /// </summary>
        /// <param name="loc">The position of the distribution peak.</param>
        /// <param name="scale">The exponential decay. Must be non-negative.</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the parameterized Laplace distribution.</returns>
        public NDArray laplace(double loc, double scale, int[] size)
            => laplace(loc, scale, new Shape(size));

        /// <summary>
        ///     Draw samples from the Laplace or double exponential distribution.
        /// </summary>
        /// <param name="loc">The position of the distribution peak.</param>
        /// <param name="scale">The exponential decay. Must be non-negative.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized Laplace distribution.</returns>
        public NDArray laplace(double loc, double scale, params long[] size)
            => laplace(loc, scale, new Shape(size));

        /// <summary>
        ///     Draw samples from the Laplace or double exponential distribution.
        /// </summary>
        /// <param name="loc">The position of the distribution peak.</param>
        /// <param name="scale">The exponential decay. Must be non-negative.</param>
        /// <param name="size">Output shape as single int.</param>
        /// <returns>Drawn samples from the parameterized Laplace distribution.</returns>
        public NDArray laplace(double loc, double scale, int size)
            => laplace(loc, scale, new Shape(size));

        /// <summary>
        ///     Sample from the Laplace distribution using the same algorithm as NumPy.
        /// </summary>
        /// <remarks>
        ///     Based on NumPy's random_laplace in distributions.c:
        ///     U = random_double [0, 1)
        ///     if U >= 0.5: return loc - scale * log(2.0 - U - U)
        ///     else if U > 0: return loc + scale * log(U + U)
        ///     else: reject U == 0.0 and retry
        /// </remarks>
        private double SampleLaplace(double loc, double scale)
        {
            double U;

            do
            {
                U = randomizer.NextDouble();
            } while (U == 0.0);

            if (U >= 0.5)
            {
                return loc - scale * Math.Log(2.0 - U - U);
            }
            else
            {
                return loc + scale * Math.Log(U + U);
            }
        }
    }
}
