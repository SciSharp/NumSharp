namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a standard Cauchy distribution.
        /// </summary>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        public NDArray standard_cauchy() => standard_cauchy(Shape.Scalar);

        /// <summary>
        ///     Draw samples from a standard Cauchy distribution with mode = 0.
        /// </summary>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the standard Cauchy distribution (float64).</returns>
        /// <exception cref="ValueError"><paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_cauchy.html
        ///     <br/>
        ///     Also known as the Lorentz distribution. The standard Cauchy distribution
        ///     has location parameter x0=0 and scale parameter gamma=1.
        ///     <br/>
        ///     The Cauchy distribution has no defined mean or variance (infinite tails).
        ///     The median is 0, and the interquartile range is 2 (from -1 to 1).
        ///     <br/>
        ///     Generated as NumPy's <c>legacy_standard_cauchy</c>: the ratio of two cached-polar normals
        ///     (<c>N1 / N2</c>, numerator first) — not the inverse transform <c>tan(pi * (U - 0.5))</c> the former code used,
        ///     which drew a different stream. Byte-identical to <c>np.random.RandomState(seed).standard_cauchy</c>; holds the
        ///     bit generator's lock for the draws.
        /// </remarks>
        public NDArray standard_cauchy(Shape size)
        {
            if (IsScalarDraw(size))
                lock (randomizer.@lock)
                    return NDArray.Scalar(LegacyStandardCauchy());

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                lock (randomizer.@lock)
                    for (long i = 0; i < n; i++)
                        dst[i] = LegacyStandardCauchy();
            }

            return ret;
        }
    }
}
