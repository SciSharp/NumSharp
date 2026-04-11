using System;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a Bernoulli distribution.
        /// </summary>
        /// <param name="p">Probability of success (1), must be in [0, 1].</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples (0 or 1) from the Bernoulli distribution.</returns>
        /// <remarks>
        ///     This function is NumSharp-specific and not available in NumPy.
        ///     For NumPy equivalent, use scipy.stats.bernoulli.
        ///     <br/>
        ///     The Bernoulli distribution is a discrete distribution having two possible
        ///     outcomes: 1 (success) with probability p, and 0 (failure) with probability 1-p.
        /// </remarks>
        public NDArray bernoulli(double p, Shape size) => bernoulli(p, size.dimensions);

        /// <summary>
        ///     Draw samples from a Bernoulli distribution.
        /// </summary>
        /// <param name="p">Probability of success (1), must be in [0, 1].</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples (0 or 1) from the Bernoulli distribution.</returns>
        /// <remarks>
        ///     This function is NumSharp-specific and not available in NumPy.
        ///     For NumPy equivalent, use scipy.stats.bernoulli.
        ///     <br/>
        ///     The Bernoulli distribution is a discrete distribution having two possible
        ///     outcomes: 1 (success) with probability p, and 0 (failure) with probability 1-p.
        /// </remarks>
        public NDArray bernoulli(double p, int[] size) => bernoulli(p, Shape.ComputeLongShape(size));

        /// <summary>
        ///     Draw samples from a Bernoulli distribution.
        /// </summary>
        /// <param name="p">Probability of success (1), must be in [0, 1].</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples (0 or 1) from the Bernoulli distribution.</returns>
        /// <remarks>
        ///     This function is NumSharp-specific and not available in NumPy.
        ///     For NumPy equivalent, use scipy.stats.bernoulli.
        ///     <br/>
        ///     The Bernoulli distribution is a discrete distribution having two possible
        ///     outcomes: 1 (success) with probability p, and 0 (failure) with probability 1-p.
        /// </remarks>
        public NDArray bernoulli(double p, long[] size)
        {
            if (size == null || size.Length == 0)
                return NDArray.Scalar(randomizer.NextDouble() < p ? 1.0 : 0.0);

            var result = new NDArray<double>(size);
            unsafe
            {
                var addr = result.Address;
                long len = result.size;
                Func<double> nextDouble = randomizer.NextDouble;
                for (long i = 0; i < len; i++)
                    addr[i] = nextDouble() < p ? 1 : 0;
            }

            return result;
        }
    }
}
