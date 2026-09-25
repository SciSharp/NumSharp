namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a chi-square distribution.
        /// </summary>
        /// <param name="df">Number of degrees of freedom, must be &gt; 0.</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="df"/> is <c>&lt;= 0</c> (<c>df &lt;= 0</c>).</exception>
        public NDArray chisquare(double df) => chisquare(df, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a chi-square distribution.
        /// </summary>
        /// <param name="df">Number of degrees of freedom, must be &gt; 0 (NaN is accepted and samples NaN, as in NumPy).</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized chi-square distribution (float64).</returns>
        /// <exception cref="ValueError"><paramref name="df"/> is <c>&lt;= 0</c> (<c>df &lt;= 0</c>), or <paramref name="size"/>
        ///     has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.chisquare.html
        ///     <br/>
        ///     When df independent random variables, each with standard normal distributions
        ///     (mean 0, variance 1), are squared and summed, the resulting distribution is
        ///     chi-square. This distribution is often used in hypothesis testing.
        ///     <br/>
        ///     NumPy's <c>legacy_chisquare</c> — <c>2 * legacy_standard_gamma(df / 2)</c> — byte-identical to
        ///     <c>np.random.RandomState(seed).chisquare</c>. Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray chisquare(double df, Shape size)
        {
            RandomConstraints.Check(df, "df", ConstraintType.CONS_POSITIVE);

            if (IsScalarDraw(size))
                lock (randomizer.@lock)
                    return NDArray.Scalar(LegacyChisquare(df));

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                lock (randomizer.@lock)
                    for (long i = 0; i < n; i++)
                        dst[i] = LegacyChisquare(df);
            }

            return ret;
        }
    }
}
