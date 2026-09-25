namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a log-normal distribution.
        /// </summary>
        /// <param name="mean">Mean value of the underlying normal distribution. Default is 0.</param>
        /// <param name="sigma">Standard deviation of the underlying normal distribution. Must be non-negative. Default is 1.</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="sigma"/> is negative, including <c>-0.0</c> (<c>sigma &lt; 0</c>).</exception>
        public NDArray lognormal(double mean = 0.0, double sigma = 1.0) => lognormal(mean, sigma, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a log-normal distribution.
        /// </summary>
        /// <param name="mean">Mean value of the underlying normal distribution. Default is 0.</param>
        /// <param name="sigma">Standard deviation of the underlying normal distribution. Must be non-negative. Default is 1.</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized log-normal distribution (float64).</returns>
        /// <exception cref="ValueError"><paramref name="sigma"/> is negative, including <c>-0.0</c> (<c>sigma &lt; 0</c>), or
        ///     <paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.lognormal.html
        ///     <br/>
        ///     Draw samples from a log-normal distribution with specified mean, standard deviation,
        ///     and array shape. Note that the mean and standard deviation are not the values for
        ///     the distribution itself, but of the underlying normal distribution it is derived from.
        ///     <br/>
        ///     NumPy's <c>legacy_lognormal</c> — <c>exp(mean + sigma * legacy_gauss)</c>, one fused loop (no intermediate
        ///     normal array). Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray lognormal(double mean, double sigma, Shape size)
        {
            RandomConstraints.Check(sigma, "sigma", ConstraintType.CONS_NON_NEGATIVE);

            if (IsScalarDraw(size))
                lock (randomizer.@lock)
                    return NDArray.Scalar(LegacyLognormal(mean, sigma));

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                lock (randomizer.@lock)
                    for (long i = 0; i < n; i++)
                        dst[i] = LegacyLognormal(mean, sigma);
            }

            return ret;
        }
    }
}
