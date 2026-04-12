using System;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from the geometric distribution.
        /// </summary>
        public NDArray geometric(double p) => geometric(p, Shape.Scalar);

        /// <summary>
        ///     Draw samples from the geometric distribution.
        /// </summary>
        /// <param name="p">The probability of success of an individual trial.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized geometric distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.geometric.html
        ///     <br/>
        ///     Bernoulli trials are experiments with one of two outcomes: success or failure
        ///     (an example of such an experiment is flipping a coin). The geometric distribution
        ///     models the number of trials that must be run in order to achieve success.
        ///     It is therefore supported on the positive integers, k = 1, 2, ...
        /// </remarks>
        public NDArray geometric(double p, Shape size)
        {
            // Validate p is in (0, 1]
            if (p <= 0 || p > 1)
                throw new ArgumentOutOfRangeException(nameof(p), "p must be in (0, 1]");

            if (size.IsScalar || size.IsEmpty)
                return NDArray.Scalar(SampleGeometric(p));

            var shape = size;
            NDArray ret = new NDArray(NPTypeCode.Int64, shape, false);

            // Handle empty arrays (any dimension is 0)
            if (shape.size == 0)
                return ret;

            unsafe
            {
                var addr = (long*)ret.Address;
                for (long i = 0; i < ret.size; i++)
                    addr[i] = SampleGeometric(p);
            }

            return ret;
        }

        /// <summary>
        ///     Sample a single value from the geometric distribution.
        /// </summary>
        /// <remarks>
        ///     NumPy uses the search algorithm (random_geometric_search).
        ///     This matches the legacy mtrand implementation exactly.
        /// </remarks>
        private long SampleGeometric(double p)
        {
            double q = 1.0 - p;
            long X = 1;
            double sum = p;
            double prod = p;
            double U = randomizer.NextDouble();

            while (U > sum)
            {
                prod *= q;
                sum += prod;
                X++;
            }

            return X;
        }
    }
}
