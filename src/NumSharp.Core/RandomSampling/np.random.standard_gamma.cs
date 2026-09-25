namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a standard Gamma distribution.
        /// </summary>
        /// <param name="shape">The shape of the gamma distribution. Must be &gt;= 0.</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="shape"/> is negative, including <c>-0.0</c> (<c>shape &lt; 0</c>).</exception>
        public NDArray standard_gamma(double shape) => standard_gamma(shape, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a standard Gamma distribution (scale=1).
        /// </summary>
        /// <param name="shape">The shape of the gamma distribution. Must be &gt;= 0 (0 gives zeros without drawing; NaN is
        ///     accepted and samples NaN, as in NumPy).</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the standard gamma distribution (float64).</returns>
        /// <exception cref="ValueError"><paramref name="shape"/> is negative, including <c>-0.0</c> (<c>shape &lt; 0</c>), or
        ///     <paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_gamma.html
        ///     <br/>
        ///     For a different scale, multiply the result: scale * standard_gamma(shape).
        ///     <br/>
        ///     The probability density function is:
        ///     p(x) = x^(shape-1) * e^(-x) / Gamma(shape)
        ///     <br/>
        ///     NumPy's <c>legacy_standard_gamma</c> (see <c>NumPyRandom.LegacyDistributions.cs</c>) — byte-identical to
        ///     <c>np.random.RandomState(seed).standard_gamma</c>. Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray standard_gamma(double shape, Shape size)
        {
            RandomConstraints.Check(shape, "shape", ConstraintType.CONS_NON_NEGATIVE);

            // The per-call setup NumPy recomputes for every value, evaluated once (the same expressions — bit-neutral).
            var setup = new LegacyGammaSetup(shape);

            if (IsScalarDraw(size))
            {
                unsafe
                {
                    // A one-double buffer IS NumPy's per-draw call sequence.
                    double word;
                    var one = new DrawBufferDouble(randomizer, &word, 1);
                    lock (randomizer.@lock)
                        return NDArray.Scalar(LegacyStandardGamma(ref one, in setup));
                }
            }

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                // Read-ahead draws (bulk-filled by the bit generator): every value draws unless shape == 0 (then per-draw).
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, setup.Draws ? DrawBufferDouble.Capacity : 1);
                lock (randomizer.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = LegacyStandardGamma(ref src, in setup);
                    }
            }

            return ret;
        }
    }
}
