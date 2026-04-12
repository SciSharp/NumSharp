using System;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a log-normal distribution.
        /// </summary>
        public NDArray lognormal(double mean = 0.0, double sigma = 1.0) => lognormal(mean, sigma, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a log-normal distribution.
        /// </summary>
        /// <param name="mean">Mean value of the underlying normal distribution. Default is 0.</param>
        /// <param name="sigma">Standard deviation of the underlying normal distribution. Must be non-negative. Default is 1.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized log-normal distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.lognormal.html
        ///     <br/>
        ///     Draw samples from a log-normal distribution with specified mean, standard deviation,
        ///     and array shape. Note that the mean and standard deviation are not the values for
        ///     the distribution itself, but of the underlying normal distribution it is derived from.
        /// </remarks>
        public NDArray lognormal(double mean, double sigma, Shape size)
        {
            if (sigma < 0)
                throw new ArgumentException("sigma < 0", nameof(sigma));

            // NumPy: lognormal = exp(normal(mean, sigma))
            // The parameters are for the underlying normal distribution, not the lognormal
            if (size.IsScalar || size.IsEmpty)
                return NDArray.Scalar(Math.Exp(mean + sigma * NextGaussian()));

            var x = normal(mean, sigma, size);
            return np.exp(x);
        }
    }
}
