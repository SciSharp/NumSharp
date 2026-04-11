using System;

namespace NumSharp
{
    public partial class NumPyRandom
    {
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
        public NDArray lognormal(double mean, double sigma, Shape size) => lognormal(mean, sigma, size.dimensions);

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
        public NDArray lognormal(double mean, double sigma, int[] size) => lognormal(mean, sigma, Shape.ComputeLongShape(size));

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
        public NDArray lognormal(double mean, double sigma, long[] size)
        {
            double zm = mean * mean;
            double zs = sigma * sigma;

            double lmean = Math.Log(zm / Math.Sqrt(zs + zm));
            double lstdv = Math.Sqrt(Math.Log(zs / zm + 1));

            var x = normal(lmean, lstdv, size);
            return np.exp(x);
        }
    }
}
