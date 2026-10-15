namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a Bernoulli distribution.
        /// </summary>
        /// <param name="p">Probability of success (1), must be in [0, 1].</param>
        /// <returns>A scalar (0 or 1) from the Bernoulli distribution.</returns>
        public NDArray bernoulli(double p) => bernoulli(p, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a Bernoulli distribution.
        /// </summary>
        /// <param name="p">Probability of success (1), must be in [0, 1].</param>
        /// <param name="size">Output shape; <c>default</c> draws a single value.</param>
        /// <returns>Drawn samples (0 or 1) from the Bernoulli distribution, as float64.</returns>
        /// <exception cref="ValueError"><paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     This function is NumSharp-specific and not available in NumPy.
        ///     For NumPy equivalent, use scipy.stats.bernoulli.
        ///     <br/>
        ///     The Bernoulli distribution is a discrete distribution having two possible
        ///     outcomes: 1 (success) with probability p, and 0 (failure) with probability 1-p.
        ///     <br/>
        ///     One uniform per value (<c>U &lt; p</c>), bulk-filled into the output and thresholded in place — the same stream
        ///     as drawing them one at a time. Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray bernoulli(double p, Shape size)
        {
            if (IsScalarDraw(size))
                lock (randomizer.@lock)
                    return NDArray.Scalar(randomizer.NextDouble() < p ? 1.0 : 0.0);

            var result = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var addr = (double*)result.Address;
                long len = result.size;
                lock (randomizer.@lock)
                    randomizer.FillDouble(addr, len);
                for (long i = 0; i < len; i++)
                    addr[i] = addr[i] < p ? 1 : 0;
            }

            return result;
        }
    }
}
