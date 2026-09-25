namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a noncentral chi-square distribution.
        /// </summary>
        /// <param name="df">Degrees of freedom, must be &gt; 0.</param>
        /// <param name="nonc">Non-centrality parameter, must be &gt;= 0.</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="df"/> is <c>&lt;= 0</c>, or <paramref name="nonc"/> is negative (including <c>-0.0</c>).</exception>
        public NDArray noncentral_chisquare(double df, double nonc) => noncentral_chisquare(df, nonc, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a noncentral chi-square distribution.
        /// </summary>
        /// <param name="df">Degrees of freedom, must be &gt; 0 (NaN is accepted and samples NaN, as in NumPy).</param>
        /// <param name="nonc">Non-centrality parameter, must be &gt;= 0 (NaN is accepted and samples NaN).</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized noncentral chi-square distribution (float64).</returns>
        /// <exception cref="ValueError"><paramref name="df"/> is <c>&lt;= 0</c> (<c>df &lt;= 0</c>, checked first),
        ///     <paramref name="nonc"/> is negative including <c>-0.0</c> (<c>nonc &lt; 0</c>), or <paramref name="size"/> has a
        ///     negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.noncentral_chisquare.html
        ///     <br/>
        ///     The noncentral chi-square distribution is a generalization of the chi-square distribution.
        ///     <br/>
        ///     Mean = df + nonc
        ///     <br/>
        ///     NumPy's <c>legacy_noncentral_chisquare</c>: the central chi-square when <c>nonc == 0</c>;
        ///     <c>chi2(df - 1) + (N + sqrt(nonc))^2</c> when <c>df &gt; 1</c>; otherwise a Poisson(<c>nonc/2</c>)-mixed
        ///     chi-square (the shared PTRS Poisson for large <c>nonc</c>). Byte-identical to
        ///     <c>np.random.RandomState(seed).noncentral_chisquare</c>; holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray noncentral_chisquare(double df, double nonc, Shape size)
        {
            RandomConstraints.Check(df, "df", ConstraintType.CONS_POSITIVE);
            RandomConstraints.Check(nonc, "nonc", ConstraintType.CONS_NON_NEGATIVE);

            if (IsScalarDraw(size))
                lock (randomizer.@lock)
                    return NDArray.Scalar(LegacyNoncentralChisquare(df, nonc));

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                lock (randomizer.@lock)
                    for (long i = 0; i < n; i++)
                        dst[i] = LegacyNoncentralChisquare(df, nonc);
            }

            return ret;
        }
    }
}
