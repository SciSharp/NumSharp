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
        /// Draw samples from a Poisson distribution. The Poisson distribution is the limit of the binomial distribution for large N.
        /// </summary>
        /// <param name="lam">Expectation of interval, should be >= 0. A sequence of expectation intervals must be broadcastable over the requested size.</param>
        /// <param name="shape">Output Shape</param>
        /// <returns>Drawn samples from the parameterized binomial distribution, where each sample is equal to the number of successes over the n trials.</returns>
        public NDArray poisson(double lam, Shape shape) => poisson(lam, shape.Dimensions);

        /// <summary>
        /// Draw samples from a Poisson distribution. The Poisson distribution is the limit of the binomial distribution for large N.
        /// </summary>
        /// <param name="lam">Expectation of interval, should be >= 0. A sequence of expectation intervals must be broadcastable over the requested size.</param>
        /// <param name="dims">Output Shape</param>
        /// <returns>Drawn samples from the parameterized binomial distribution, where each sample is equal to the number of successes over the n trials.</returns>
        public NDArray poisson(double lam, params int[] dims)
        {
            if (lam < 0)
                throw new ArgumentException("lam >= 0");

            var result = new NDArray<double>(dims);
            ArraySlice<double> resultArray = result.Data<double>();

            Parallel.For(0, result.size, (i) => {
                resultArray[i] = knuth(lam);
            });

            result.ReplaceData(resultArray); //incase of a view //todo! incase of a view?
            return result;
        }

        private static int knuth(double lambda)
        {
            // Knuth, 1969.
            double p = 1.0;
            double L = Math.Exp(-lambda);

            int k;

            for (k = 0; p > L; k++)
                p *= np.random.randomizer.NextDouble();

            return k - 1;
        }
    }
}
