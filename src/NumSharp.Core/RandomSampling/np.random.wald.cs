using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a Wald, or inverse Gaussian, distribution.
        /// </summary>
        /// <param name="mean">Distribution mean, must be > 0.</param>
        /// <param name="scale">Scale parameter, must be > 0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized Wald distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.wald.html
        ///     <br/>
        ///     The inverse Gaussian distribution was first studied in relationship to
        ///     Brownian motion. As the scale approaches infinity, the distribution
        ///     becomes more like a Gaussian.
        ///     <br/>
        ///     The probability density function is:
        ///     P(x;mean,scale) = sqrt(scale/(2*pi*x^3)) * exp(-scale*(x-mean)^2 / (2*mean^2*x))
        /// </remarks>
        public NDArray wald(double mean, double scale, Shape size)
            => wald(mean, scale, size.dimensions);

        /// <summary>
        ///     Draw samples from a Wald, or inverse Gaussian, distribution.
        /// </summary>
        /// <param name="mean">Distribution mean, must be > 0.</param>
        /// <param name="scale">Scale parameter, must be > 0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized Wald distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.wald.html
        ///     <br/>
        ///     The inverse Gaussian distribution was first studied in relationship to
        ///     Brownian motion. As the scale approaches infinity, the distribution
        ///     becomes more like a Gaussian.
        /// </remarks>
        public NDArray wald(double mean, double scale, params int[] size)
        {
            // Parameter validation (matches NumPy error messages)
            if (mean <= 0)
                throw new ArgumentException("mean <= 0", nameof(mean));
            if (scale <= 0)
                throw new ArgumentException("scale <= 0", nameof(scale));

            if (size == null || size.Length == 0)
            {
                return NDArray.Scalar(SampleWald(mean, scale));
            }

            var result = new NDArray<double>(size);
            ArraySlice<double> resultArray = result.Data<double>();

            for (int i = 0; i < result.size; ++i)
                resultArray[i] = SampleWald(mean, scale);

            result.ReplaceData(resultArray);
            return result;
        }

        /// <summary>
        ///     Sample a single value from the Wald (inverse Gaussian) distribution.
        /// </summary>
        /// <remarks>
        ///     Algorithm from NumPy's random_wald in distributions.c:
        ///     Y = standard_normal()^2 * mean
        ///     d = 1 + sqrt(1 + 4 * scale / Y)
        ///     X = mean * (1 - 2 / d)
        ///     if uniform() <= mean / (mean + X):
        ///         return X
        ///     else:
        ///         return mean^2 / X
        /// </remarks>
        private double SampleWald(double mean, double scale)
        {
            // NumPy's exact implementation from distributions.c
            double Y = NextGaussian();
            Y = mean * Y * Y;
            double d = 1.0 + Math.Sqrt(1.0 + 4.0 * scale / Y);
            double X = mean * (1.0 - 2.0 / d);
            double U = randomizer.NextDouble();

            if (U <= mean / (mean + X))
            {
                return X;
            }
            else
            {
                return mean * mean / X;
            }
        }
    }
}
