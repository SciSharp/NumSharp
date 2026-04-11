namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a binomial distribution.
        /// </summary>
        public NDArray binomial(int n, double p) => binomial(n, p, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a binomial distribution.
        /// </summary>
        /// <param name="n">Parameter of the distribution, >= 0. Number of trials.</param>
        /// <param name="p">Parameter of the distribution, >= 0 and &lt;= 1. Probability of success.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>
        ///     Drawn samples from the parameterized binomial distribution, where each sample
        ///     is equal to the number of successes over the n trials.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html
        ///     <br/>
        ///     Samples are drawn from a binomial distribution with specified parameters,
        ///     n trials and p probability of success where n is an integer >= 0 and p is
        ///     in the interval [0, 1].
        /// </remarks>
        public NDArray binomial(int n, double p, Shape size)
        {
            if (size.IsScalar || size.IsEmpty)
            {
                long count = 0;
                for (int i = 0; i < n; i++)
                    if (randomizer.NextDouble() < p) count++;
                return NDArray.Scalar(count);
            }

            var result = new NDArray(NPTypeCode.Int64, size, false);
            unsafe
            {
                var addr = (long*)result.Address;
                for (long j = 0; j < result.size; j++)
                {
                    long count = 0;
                    for (int i = 0; i < n; i++)
                        if (randomizer.NextDouble() < p) count++;
                    addr[j] = count;
                }
            }
            return result;
        }
    }
}
