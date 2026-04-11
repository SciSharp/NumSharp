using System;
using System.Runtime.CompilerServices;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a Poisson distribution.
        /// </summary>
        public NDArray poisson(double lam = 1.0) => poisson(lam, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a Poisson distribution.
        /// </summary>
        /// <param name="lam">Expected number of events occurring in a fixed-time interval, must be >= 0. Default is 1.0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized Poisson distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.poisson.html
        ///     <br/>
        ///     The Poisson distribution is the limit of the binomial distribution for large N.
        /// </remarks>
        public NDArray poisson(double lam, Shape size)
        {
            if (lam < 0)
                throw new ArgumentException("lam must be >= 0", nameof(lam));

            if (size.IsScalar || size.IsEmpty)
                return NDArray.Scalar((double)Knuth(lam));

            var result = new NDArray<double>(size);
            unsafe
            {
                long len = result.size;
                var resultArray = result.Address;
                for (long i = 0; i < len; i++)
                    resultArray[i] = Knuth(lam);
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private int Knuth(double lambda)
        {
            // Knuth algorithm for Poisson distribution
            double p = 1.0;
            double L = Math.Exp(-lambda);
            int k;

            for (k = 0; p > L; k++)
                p *= randomizer.NextDouble();

            return k - 1;
        }
    }
}
