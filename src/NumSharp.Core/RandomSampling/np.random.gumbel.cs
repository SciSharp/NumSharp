using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a Gumbel distribution.
        /// </summary>
        public NDArray gumbel(double loc = 0.0, double scale = 1.0) => gumbel(loc, scale, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a Gumbel distribution (extreme value type I).
        /// </summary>
        /// <param name="loc">The location of the mode of the distribution. Default is 0.</param>
        /// <param name="scale">The scale parameter of the distribution. Must be non-negative. Default is 1.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized Gumbel distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.gumbel.html
        ///     <br/>
        ///     The Gumbel (or Smallest Extreme Value (SEV) or the Smallest Extreme Value Type I)
        ///     distribution is used to model the distribution of the maximum (or minimum) of a
        ///     number of samples of various distributions.
        ///     <br/>
        ///     The probability density function is:
        ///     p(x) = (1/scale) * exp(-(x-loc)/scale) * exp(-exp(-(x-loc)/scale))
        ///     <br/>
        ///     For Gumbel(loc, scale):
        ///     - mean = loc + scale * γ (where γ ≈ 0.5772 is the Euler-Mascheroni constant)
        ///     - std = scale * π / sqrt(6) ≈ 1.283 * scale
        /// </remarks>
        public NDArray gumbel(double loc, double scale, Shape size)
        {
            if (scale < 0)
                throw new ArgumentException("scale < 0", nameof(scale));

            if (size.IsScalar || size.IsEmpty)
                return NDArray.Scalar(SampleGumbel(loc, scale));

            var ret = new NDArray<double>(size);
            ArraySlice<double> data = ret.Data<double>();

            for (int i = 0; i < ret.size; i++)
            {
                data[i] = SampleGumbel(loc, scale);
            }

            return ret;
        }

        /// <summary>
        ///     Sample from the Gumbel distribution using the same algorithm as NumPy.
        /// </summary>
        /// <remarks>
        ///     Based on NumPy's random_gumbel in distributions.c:
        ///     U = 1.0 - next_double();  // U in (0, 1]
        ///     if (U &lt; 1.0) return loc - scale * log(-log(U));
        ///     // Reject U == 1.0 and retry
        /// </remarks>
        private double SampleGumbel(double loc, double scale)
        {
            if (scale == 0.0)
                return loc;

            double U;
            do
            {
                // U = 1.0 - NextDouble() gives U in (0, 1]
                // We need U < 1.0 to avoid log(0) = -inf
                U = 1.0 - randomizer.NextDouble();
            } while (U >= 1.0); // Reject U == 1.0 (which happens when NextDouble() == 0.0)

            return loc - scale * Math.Log(-Math.Log(U));
        }
    }
}
