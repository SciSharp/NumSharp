namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from the geometric distribution.
        /// </summary>
        /// <param name="p">The probability of success of an individual trial, in <c>(0, 1]</c>.</param>
        /// <returns>A 0-d int64 array holding the trial of the first success.</returns>
        /// <exception cref="ValueError"><paramref name="p"/> is outside <c>(0, 1]</c> or NaN.</exception>
        public NDArray geometric(double p) => geometric(p, Shape.Scalar);

        /// <summary>
        ///     Draw samples from the geometric distribution.
        /// </summary>
        /// <param name="p">The probability of success of an individual trial, in <c>(0, 1]</c>.</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized geometric distribution (int64).</returns>
        /// <exception cref="ValueError"><paramref name="p"/> is outside <c>(0, 1]</c> or NaN
        ///     (<c>p &lt;= 0, p &gt; 1 or p contains NaNs</c>; <c>-0.0</c> included), or <paramref name="size"/> has a
        ///     negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.geometric.html
        ///     <br/>
        ///     Bernoulli trials are experiments with one of two outcomes: success or failure
        ///     (an example of such an experiment is flipping a coin). The geometric distribution
        ///     models the number of trials that must be run in order to achieve success.
        ///     It is therefore supported on the positive integers, k = 1, 2, ...
        ///     <br/>
        ///     NumPy's <c>legacy_random_geometric</c>: the CDF search for <c>p &gt;= 1/3</c> and the legacy inversion
        ///     <c>ceil(log1p(-U) / log(1 - p))</c> below it — the former search-only path walked ~1/p steps per draw for small
        ///     p. Byte-identical to <c>np.random.RandomState(seed).geometric</c>; int64 output (NumPy returns C <c>long</c>).
        ///     Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray geometric(double p, Shape size)
        {
            RandomConstraints.Check(p, "p", ConstraintType.CONS_BOUNDED_GT_0_1);

            if (IsScalarDraw(size))
                lock (randomizer.@lock)
                    return NDArray.Scalar(LegacyGeometric(p));

            var ret = LegacyOutput(NPTypeCode.Int64, size);
            unsafe
            {
                var dst = (long*)ret.Address;
                long n = ret.size;
                lock (randomizer.@lock)
                    for (long i = 0; i < n; i++)
                        dst[i] = LegacyGeometric(p);
            }

            return ret;
        }
    }
}
