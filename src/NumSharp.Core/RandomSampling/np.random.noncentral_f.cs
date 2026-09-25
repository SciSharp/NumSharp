namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from the noncentral F distribution.
        /// </summary>
        /// <param name="dfnum">Numerator degrees of freedom, must be &gt; 0.</param>
        /// <param name="dfden">Denominator degrees of freedom, must be &gt; 0.</param>
        /// <param name="nonc">Non-centrality parameter, must be &gt;= 0.</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError">A degrees-of-freedom parameter is <c>&lt;= 0</c>, or <paramref name="nonc"/> is negative.</exception>
        public NDArray noncentral_f(double dfnum, double dfden, double nonc) => noncentral_f(dfnum, dfden, nonc, Shape.Scalar);

        /// <summary>
        ///     Draw samples from the noncentral F distribution.
        /// </summary>
        /// <param name="dfnum">Numerator degrees of freedom, must be &gt; 0 (NaN is accepted and samples NaN, as in NumPy).</param>
        /// <param name="dfden">Denominator degrees of freedom, must be &gt; 0 (NaN is accepted and samples NaN).</param>
        /// <param name="nonc">Non-centrality parameter, must be &gt;= 0 (NaN is accepted and samples NaN).</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized noncentral F distribution (float64).</returns>
        /// <exception cref="ValueError">In NumPy's order: <c>dfnum &lt;= 0</c>, <c>dfden &lt;= 0</c>, <c>nonc &lt; 0</c>
        ///     (<c>-0.0</c> included); or <paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.noncentral_f.html
        ///     <br/>
        ///     When calculating the power of an experiment, the non-central F statistic
        ///     becomes important. When the null hypothesis is true, the F statistic follows
        ///     a central F distribution. When the null hypothesis is not true, it follows
        ///     a non-central F distribution.
        ///     <br/>
        ///     NumPy's <c>legacy_noncentral_f</c>: <c>(ncchi2(dfnum, nonc) * dfden) / (chi2(dfden) * dfnum)</c> — byte-identical
        ///     to <c>np.random.RandomState(seed).noncentral_f</c>. Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray noncentral_f(double dfnum, double dfden, double nonc, Shape size)
        {
            RandomConstraints.Check(dfnum, "dfnum", ConstraintType.CONS_POSITIVE);
            RandomConstraints.Check(dfden, "dfden", ConstraintType.CONS_POSITIVE);
            RandomConstraints.Check(nonc, "nonc", ConstraintType.CONS_NON_NEGATIVE);

            if (IsScalarDraw(size))
                lock (randomizer.@lock)
                    return NDArray.Scalar(LegacyNoncentralF(dfnum, dfden, nonc));

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                lock (randomizer.@lock)
                    for (long i = 0; i < n; i++)
                        dst[i] = LegacyNoncentralF(dfnum, dfden, nonc);
            }

            return ret;
        }
    }
}
