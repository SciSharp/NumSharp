namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a standard Student's t distribution.
        /// </summary>
        /// <param name="df">Degrees of freedom, must be &gt; 0.</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="df"/> is <c>&lt;= 0</c> (<c>df &lt;= 0</c>).</exception>
        public NDArray standard_t(double df) => standard_t(df, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a standard Student's t distribution with df degrees of freedom.
        /// </summary>
        /// <param name="df">Degrees of freedom, must be &gt; 0 (NaN is accepted and samples NaN, as in NumPy).</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized standard Student's t distribution (float64).</returns>
        /// <exception cref="ValueError"><paramref name="df"/> is <c>&lt;= 0</c> (<c>df &lt;= 0</c>), or <paramref name="size"/> has
        ///     a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_t.html
        ///     <br/>
        ///     A special case of the hyperbolic distribution. As df gets large, the result
        ///     resembles that of the standard normal distribution.
        ///     <br/>
        ///     The probability density function is:
        ///     P(x, df) = Gamma((df+1)/2) / (sqrt(pi*df) * Gamma(df/2)) * (1 + x^2/df)^(-(df+1)/2)
        ///     <br/>
        ///     NumPy's <c>legacy_standard_t</c>: <c>sqrt(df/2) * N / sqrt(G)</c> with the cached-polar normal drawn before the
        ///     legacy gamma <c>G(df/2)</c>. Byte-identical to <c>np.random.RandomState(seed).standard_t</c>; holds the bit
        ///     generator's lock for the draws.
        /// </remarks>
        public NDArray standard_t(double df, Shape size)
        {
            RandomConstraints.Check(df, "df", ConstraintType.CONS_POSITIVE);

            if (IsScalarDraw(size))
                lock (randomizer.@lock)
                    return NDArray.Scalar(LegacyStandardT(df));

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                lock (randomizer.@lock)
                    for (long i = 0; i < n; i++)
                        dst[i] = LegacyStandardT(df);
            }

            return ret;
        }
    }
}
