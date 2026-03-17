namespace NumSharp
{
    public partial class NumPyRandom
    {
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
        public NDArray binomial(int n, double p, Shape size) => binomial(n, p, size.dimensions);

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
        public NDArray binomial(int n, double p, params int[] size)
        {
            var x = np.zeros(size);
            for (int i = 0; i < n; i++)
            {
                x = x + bernoulli(p, size);
            }
            return x / n;
        }
    }
}
