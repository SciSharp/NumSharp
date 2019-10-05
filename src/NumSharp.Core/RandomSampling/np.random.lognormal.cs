using System;
using System.Threading.Tasks;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a log-normal distribution.
        ///     Draw samples from a log-normal distribution with specified mean, standard deviation, and array shape.
        ///     Note that the mean and standard deviation are not the values for the distribution itself, but of the underlying normal distribution it is derived from.
        /// </summary>
        /// <param name="mean">Mean value of the underlying normal distribution. Default is 0.</param>
        /// <param name="sigma">Standard deviation of the underlying normal distribution. Should be greater than zero. Default is 1.</param>
        /// <param name="shape">Output Shape</param>
        /// <returns>Drawn samples from the parameterized bernoulli distribution.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.random.lognormal.html</remarks>
        public NDArray lognormal(double mean, double sigma, Shape shape) => lognormal(mean, sigma, shape.dimensions);

        /// <summary>
        ///     Draw samples from a log-normal distribution.
        ///     Draw samples from a log-normal distribution with specified mean, standard deviation, and array shape.
        ///     Note that the mean and standard deviation are not the values for the distribution itself, but of the underlying normal distribution it is derived from.
        /// </summary>
        /// <param name="mean">Mean value of the underlying normal distribution. Default is 0.</param>
        /// <param name="sigma">Standard deviation of the underlying normal distribution. Should be greater than zero. Default is 1.</param>
        /// <param name="dims">Output Shape</param>
        /// <returns>Drawn samples from the parameterized bernoulli distribution.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.random.lognormal.html</remarks>
        public NDArray lognormal(double mean, double sigma, params int[] dims)
        {
            double zm = mean * mean;
            double zs = sigma * sigma;

            double lmean = Math.Log(zm / Math.Sqrt(zs + zm));
            double lstdv = Math.Sqrt(Math.Log(zs / zm + 1));

            var x = normal((float)lmean, (float)lstdv, dims);
            return np.exp(x);
        }
    }
}
