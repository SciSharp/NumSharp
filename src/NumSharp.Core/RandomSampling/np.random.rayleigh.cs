using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a Rayleigh distribution.
        /// </summary>
        /// <param name="scale">Scale parameter (also equals the mode). Must be non-negative. Default is 1.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized Rayleigh distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.rayleigh.html
        ///     <br/>
        ///     The probability density function for the Rayleigh distribution is:
        ///     P(x; scale) = (x / scale^2) * exp(-x^2 / (2 * scale^2))
        ///     <br/>
        ///     The Rayleigh distribution arises when the East and North components of wind
        ///     velocity have identical zero-mean Gaussian distributions. Then the wind speed
        ///     would have a Rayleigh distribution.
        ///     <br/>
        ///     For Rayleigh(scale), mean = scale * sqrt(pi/2) ≈ 1.253 * scale
        /// </remarks>
        public NDArray rayleigh(double scale = 1.0, Shape size = default)
        {
            if (scale < 0)
                throw new ArgumentException("scale < 0", nameof(scale));

            if (size.IsEmpty)
            {
                return NDArray.Scalar(SampleRayleigh(scale));
            }

            var ret = new NDArray<double>(size);
            ArraySlice<double> data = ret.Data<double>();

            for (int i = 0; i < ret.size; i++)
            {
                data[i] = SampleRayleigh(scale);
            }

            return ret;
        }

        /// <summary>
        ///     Draw samples from a Rayleigh distribution.
        /// </summary>
        /// <param name="scale">Scale parameter (also equals the mode). Must be non-negative.</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the parameterized Rayleigh distribution.</returns>
        public NDArray rayleigh(double scale, int[] size)
            => rayleigh(scale, new Shape(size));

        /// <summary>
        ///     Draw samples from a Rayleigh distribution.
        /// </summary>
        /// <param name="scale">Scale parameter (also equals the mode). Must be non-negative.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized Rayleigh distribution.</returns>
        public NDArray rayleigh(double scale, params long[] size)
            => rayleigh(scale, new Shape(size));

        /// <summary>
        ///     Draw samples from a Rayleigh distribution.
        /// </summary>
        /// <param name="scale">Scale parameter (also equals the mode). Must be non-negative.</param>
        /// <param name="size">Output shape as single int.</param>
        /// <returns>Drawn samples from the parameterized Rayleigh distribution.</returns>
        public NDArray rayleigh(double scale, int size)
            => rayleigh(scale, new Shape(size));

        /// <summary>
        ///     Sample from the Rayleigh distribution using the same algorithm as NumPy.
        /// </summary>
        /// <remarks>
        ///     Based on NumPy's random_rayleigh in distributions.c:
        ///     return mode * sqrt(2.0 * random_standard_exponential(bitgen_state));
        ///
        ///     where standard_exponential uses inverse transform: -log(1 - U)
        ///     So: mode * sqrt(2.0 * -log(1 - U)) = mode * sqrt(-2.0 * log(1 - U))
        /// </remarks>
        private double SampleRayleigh(double scale)
        {
            if (scale == 0.0)
                return 0.0;

            double U;
            do
            {
                U = randomizer.NextDouble();
            } while (U == 0.0); // Reject U == 0 to avoid log(1) = 0 edge case

            // Standard exponential via inverse transform: -log(1 - U)
            // But since U is uniform [0,1), we can use -log(U) for U in (0,1]
            // Here we use 1-U to match NumPy's convention
            return scale * Math.Sqrt(-2.0 * Math.Log(1.0 - U));
        }
    }
}
