namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a binomial distribution.
        /// </summary>
        /// <param name="n">Parameter of the distribution, &gt;= 0. Number of trials.</param>
        /// <param name="p">Parameter of the distribution, &gt;= 0 and &lt;= 1. Probability of success.</param>
        /// <returns>A 0-d int64 array holding the number of successes.</returns>
        /// <exception cref="ValueError"><paramref name="p"/> is outside <c>[0, 1]</c> or NaN, or <paramref name="n"/> is negative.</exception>
        public NDArray binomial(long n, double p) => binomial(n, p, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a binomial distribution.
        /// </summary>
        /// <param name="n">Parameter of the distribution, &gt;= 0. Number of trials.</param>
        /// <param name="p">Parameter of the distribution, &gt;= 0 and &lt;= 1. Probability of success.</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>
        ///     Drawn samples from the parameterized binomial distribution, where each sample
        ///     is equal to the number of successes over the n trials (int64).
        /// </returns>
        /// <exception cref="ValueError">
        ///     <paramref name="p"/> is outside <c>[0, 1]</c> or NaN (<c>p &lt; 0, p &gt; 1 or p is NaN</c>) — checked first —
        ///     or <paramref name="n"/> is negative (<c>n &lt; 0</c>), or <paramref name="size"/> has a negative dimension.
        /// </exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html
        ///     <br/>
        ///     Samples are drawn from a binomial distribution with specified parameters,
        ///     n trials and p probability of success where n is an integer &gt;= 0 and p is
        ///     in the interval [0, 1].
        ///     <br/>
        ///     NumPy's <c>legacy_random_binomial</c>: the legacy inversion (<c>q^n = exp(n*log(q))</c>) when
        ///     <c>n*min(p, 1-p) &lt;= 30</c>, BTPE above, both sharing this RandomState's setup cache with
        ///     <see cref="multinomial(int, double[], Shape?)"/> — byte-identical to <c>np.random.RandomState(seed).binomial</c>.
        ///     There is no shortcut for <c>n == 0</c> or <c>p == 0</c>: each still consumes one uniform, as in NumPy.
        ///     The output is int64 (NumPy returns C <c>long</c>: int64 on Linux, int32 on Windows; the values agree).
        ///     Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray binomial(long n, double p, Shape size)
        {
            RandomConstraints.Check(p, "p", ConstraintType.CONS_BOUNDED_0_1);
            RandomConstraints.Check(n, "n", ConstraintType.LEGACY_CONS_NON_NEGATIVE_INBOUNDS_LONG);

            if (IsScalarDraw(size))
                lock (randomizer.@lock)
                    return NDArray.Scalar(LegacyBinomial(p, n));

            var ret = LegacyOutput(NPTypeCode.Int64, size);
            unsafe
            {
                var dst = (long*)ret.Address;
                long count = ret.size;
                lock (randomizer.@lock)
                    for (long i = 0; i < count; i++)
                        dst[i] = LegacyBinomial(p, n);
            }

            return ret;
        }
    }
}
