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
            {
                unsafe
                {
                    // A one-double buffer IS NumPy's per-draw call sequence.
                    double word;
                    var one = new DrawBufferDouble(randomizer, &word, 1);
                    lock (randomizer.@lock)
                        return NDArray.Scalar(LegacyStandardCauchy(ref one));
                }
            }

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                // Read-ahead draws (bulk-filled by the bit generator), consumed by the two-phase polar fill (LegacyGaussFill).
                // legacy_standard_cauchy is num / denom over two consecutive legacy_gauss calls (numerator first), so the
                // values are consecutive Gaussians g[2j] / g[2j+1]: drawn a chunk at a time — the Gaussian stream, its cache
                // included, continues seamlessly across chunks — then divided (the same division, so the same bits).
                const int Chunk = 256;
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                double* gauss = stackalloc double[2 * Chunk];
                var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity);
                lock (randomizer.@lock)
                    for (long i = 0; i < n; i += Chunk)
                    {
                        long m = n - i < Chunk ? n - i : Chunk;
                        LegacyGaussFill(ref src, gauss, 2 * m);
                        for (long j = 0; j < m; j++)
                            dst[i + j] = gauss[2 * j] / gauss[2 * j + 1];
                    }
            }

            return ret;
        }
    }
}
