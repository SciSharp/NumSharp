namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a Gamma distribution.
        /// </summary>
        /// <param name="shape">The shape of the gamma distribution. Must be non-negative.</param>
        /// <param name="scale">The scale of the gamma distribution. Must be non-negative. Default is 1.0.</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="shape"/> or <paramref name="scale"/> is negative (including <c>-0.0</c>).</exception>
        public NDArray gamma(double shape, double scale = 1.0) => gamma(shape, scale, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a Gamma distribution.
        /// </summary>
        /// <param name="shape">The shape of the gamma distribution. Must be non-negative (0 gives all zeros without drawing;
        ///     NaN is accepted and samples NaN, as in NumPy).</param>
        /// <param name="scale">The scale of the gamma distribution. Must be non-negative. Default is 1.0.</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized gamma distribution (float64).</returns>
        /// <exception cref="ValueError"><paramref name="shape"/> (<c>shape &lt; 0</c>, checked first) or <paramref name="scale"/>
        ///     (<c>scale &lt; 0</c>) is negative — <c>-0.0</c> included, as NumPy tests the sign bit — or <paramref name="size"/>
        ///     has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.gamma.html
        ///     <br/>
        ///     Samples are drawn from a Gamma distribution with specified parameters,
        ///     shape (sometimes designated "k") and scale (sometimes designated "theta"),
        ///     where both parameters are > 0.
        ///     <br/>
        ///     NumPy's <c>legacy_gamma</c> = <c>scale * legacy_standard_gamma(shape)</c>, so <c>shape &lt; 1</c> takes the same
        ///     Johnk/Ahrens-Dieter branch as <see cref="standard_gamma(double, Shape)"/> (the former two-argument path used a
        ///     different <c>shape + 1</c> boost and diverged from NumPy). Byte-identical to
        ///     <c>np.random.RandomState(seed).gamma</c>; holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray gamma(double shape, double scale, Shape size)
        {
            RandomConstraints.Check(shape, "shape", ConstraintType.CONS_NON_NEGATIVE);
            RandomConstraints.Check(scale, "scale", ConstraintType.CONS_NON_NEGATIVE);

            // The per-call setup NumPy recomputes for every value, evaluated once (the same expressions — bit-neutral).
            var setup = new GammaSetup(shape);

            if (IsScalarDraw(size))
            {
                unsafe
                {
                    // A one-double buffer IS NumPy's per-draw call sequence.
                    double word;
                    var one = new DrawBufferDouble(randomizer, &word, 1);
                    lock (randomizer.@lock)
                        return NDArray.Scalar(LegacyGamma(ref one, in setup, scale));
                }
            }

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                // Read-ahead draws (bulk-filled by the bit generator): every value draws unless shape == 0 (then no draws at all — per-draw).
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, setup.Draws ? DrawBufferDouble.Capacity : 1);
                lock (randomizer.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = LegacyGamma(ref src, in setup, scale);
                    }
            }

            return ret;
        }
    }
}
