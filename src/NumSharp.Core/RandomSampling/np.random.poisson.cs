using System;
using System.Runtime.CompilerServices;
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
        ///     Draw samples from a Poisson distribution. The Poisson distribution is the limit of the binomial distribution for large N.
        /// </summary>
        /// <param name="lam">Expectation of interval, should be >= 0. A sequence of expectation intervals must be broadcastable over the requested size.</param>
        /// <param name="shape">Output Shape</param>
        /// <returns>Drawn samples from the parameterized binomial distribution, where each sample is equal to the number of successes over the n trials.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.random.poisson.html</remarks>
        public NDArray poisson(double lam, Shape shape) => poisson(lam, shape.dimensions);

        /// <summary>
        ///     Draw samples from a Poisson distribution. The Poisson distribution is the limit of the binomial distribution for large N.
        /// </summary>
        /// <param name="lam">Expectation of interval, should be >= 0. A sequence of expectation intervals must be broadcastable over the requested size.</param>
        /// <param name="dims">Output Shape</param>
        /// <returns>Drawn samples from the parameterized binomial distribution, where each sample is equal to the number of successes over the n trials.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.random.poisson.html</remarks>
        public NDArray poisson(double lam, params int[] dims)
        {
            if (lam < 0)
                throw new ArgumentException("lam >= 0");

            var result = new NDArray<double>(dims);
            unsafe
            {
                var len = result.size;
                var resultArray = result.Address;
                for (int i = 0; i < len; i++)
                    resultArray[i] = knuth(lam);
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        private int knuth(double lambda)
        {
            // Knuth, 1969.
            double p = 1.0;
            double L = Math.Exp(-lambda);

            int k;

            for (k = 0; p > L; k++)
                p *= randomizer.NextDouble();

            return k - 1;
        }
    }
}
