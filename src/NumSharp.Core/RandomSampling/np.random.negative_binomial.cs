namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a negative binomial distribution.
        /// </summary>
        /// <param name="n">Parameter of the distribution, &gt; 0 (number of successes).</param>
        /// <param name="p">Parameter of the distribution, 0 &lt;= p &lt;= 1 (probability of success).</param>
        /// <returns>A 0-d int64 array holding the number of failures.</returns>
        /// <exception cref="ValueError"><paramref name="n"/> is <c>&lt;= 0</c>, or <paramref name="p"/> is outside <c>[0, 1]</c> or NaN.</exception>
        public NDArray negative_binomial(double n, double p) => negative_binomial(n, p, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a negative binomial distribution.
        /// </summary>
        /// <param name="n">Parameter of the distribution, &gt; 0 (number of successes; NaN is accepted, as in NumPy).</param>
        /// <param name="p">Parameter of the distribution, 0 &lt;= p &lt;= 1 (probability of success).</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized negative binomial distribution (integers &gt;= 0, int64).</returns>
        /// <exception cref="ValueError"><paramref name="n"/> is <c>&lt;= 0</c> (<c>n &lt;= 0</c>, checked first),
        ///     <paramref name="p"/> is outside <c>[0, 1]</c> or NaN (<c>p &lt; 0, p &gt; 1 or p is NaN</c>), or
        ///     <paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.negative_binomial.html
        ///     <br/>
        ///     The negative binomial distribution models the number of failures before n successes,
        ///     where each trial has probability p of success.
        ///     <br/>
        ///     For this distribution:
        ///     - mean = n * (1-p) / p
        ///     - variance = n * (1-p) / p^2
        ///     <br/>
        ///     Uses gamma-Poisson mixture: Y ~ Gamma(n, (1-p)/p), X ~ Poisson(Y) — NumPy's
        ///     <c>legacy_negative_binomial</c> over the legacy gamma and the shared Poisson (PTRS for large means), byte-identical
        ///     to <c>np.random.RandomState(seed).negative_binomial</c>. <c>p = 1</c> still draws the gamma (then Poisson(0)
        ///     = 0). <c>p = 0</c> is accepted as in NumPy and yields C's indefinite integer from the infinite mean
        ///     (<see cref="long.MinValue"/> here; <c>-2147483648</c> on NumPy's win-amd64 build, whose C <c>long</c> is 32-bit).
        ///     int64 output. Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray negative_binomial(double n, double p, Shape size)
        {
            RandomConstraints.Check(n, "n", ConstraintType.CONS_POSITIVE);
            RandomConstraints.Check(p, "p", ConstraintType.CONS_BOUNDED_0_1);

            if (IsScalarDraw(size))
                lock (randomizer.@lock)
                    return NDArray.Scalar(LegacyNegativeBinomial(n, p));

            var ret = LegacyOutput(NPTypeCode.Int64, size);
            unsafe
            {
                var dst = (long*)ret.Address;
                long count = ret.size;
                lock (randomizer.@lock)
                    for (long i = 0; i < count; i++)
                        dst[i] = LegacyNegativeBinomial(n, p);
            }

            return ret;
        }
    }
}
