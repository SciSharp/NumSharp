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
        ///     Draw samples from a binomial distribution.
        ///     Samples are drawn from a binomial distribution with specified parameters, n trials and p probability of success where n an integer >= 0 and p is in the interval[0, 1]. (n may be input as a float, but it is truncated to an integer in use)
        /// </summary>
        /// <param name="n">Parameter of the distribution, >= 0. Floats are also accepted, but they will be truncated to integers.</param>
        /// <param name="p">Parameter of the distribution, >= 0 and &lt;=1.</param>
        /// <param name="shape">Output Shape</param>
        /// <returns>Drawn samples from the parameterized binomial distribution, where each sample is equal to the number of successes over the n trials.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.binomial.html</remarks>
        public NDArray binomial(int n, double p, Shape shape) => binomial(n, p, shape.dimensions);

        /// <summary>
        ///     Draw samples from a binomial distribution.
        ///     Samples are drawn from a binomial distribution with specified parameters, n trials and p probability of success where n an integer >= 0 and p is in the interval[0, 1]. (n may be input as a float, but it is truncated to an integer in use)
        /// </summary>
        /// <param name="n">Parameter of the distribution, >= 0. Floats are also accepted, but they will be truncated to integers.</param>
        /// <param name="p">Parameter of the distribution, >= 0 and &lt;=1.</param>
        /// <param name="dims">Output Shape</param>
        /// <returns>Drawn samples from the parameterized binomial distribution, where each sample is equal to the number of successes over the n trials.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.binomial.html</remarks>
        public NDArray binomial(int n, double p, params int[] dims)
        {
            var x = np.zeros(dims);
            for (int i = 0; i < n; i++)
            {
                x = x + bernoulli(p, dims);
            }

            return x / n;
        }
    }
}
