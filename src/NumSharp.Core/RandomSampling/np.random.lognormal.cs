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
        ///     NumPy's <c>legacy_lognormal</c> — <c>exp(mean + sigma * legacy_gauss)</c>, byte-identical to
        ///     <c>np.random.RandomState(seed).lognormal</c>: the normals are drawn into the output, then exponentiated in place
        ///     (no intermediate array; the split lets the CPU overlap the <c>exp</c> calls). Holds the bit generator's lock for
        ///     the draws.
        /// </remarks>
        public NDArray lognormal(double mean, double sigma, Shape size)
        {
            RandomConstraints.Check(sigma, "sigma", ConstraintType.CONS_NON_NEGATIVE);

            if (IsScalarDraw(size))
            {
                unsafe
                {
                    // A one-double buffer IS NumPy's per-draw call sequence.
                    double word;
                    var one = new DrawBufferDouble(randomizer, &word, 1);
                    lock (randomizer.@lock)
                        return NDArray.Scalar(LegacyLognormal(ref one, mean, sigma));
                }
            }

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                // Read-ahead draws (bulk-filled by the bit generator), accounted in polar PAIRS inside LegacyGaussFill.
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity);
                // Two passes over legacy_lognormal = exp(mean + sigma * legacy_gauss): the Gaussians first (every draw, by the
                // two-phase polar fill), then exp(mean + sigma * g) of each in place — the same operations per value, so the
                // same bits, but the exp pass is a branch-free run of independent calls the CPU overlaps, where the fused
                // loop stalled each exp behind a polar rejection loop. The exp pass touches no stream state, so it runs
                // after the lock is released.
                lock (randomizer.@lock)
                    LegacyGaussFill(ref src, dst, n);
                for (long i = 0; i < n; i++)
                    dst[i] = System.Math.Exp(mean + sigma * dst[i]);
            }

            return ret;
        }
    }
}
