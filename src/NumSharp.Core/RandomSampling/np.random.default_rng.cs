namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Construct a new <see cref="Generator"/> with the default bit generator (PCG64),
        ///     seeded from fresh, unpredictable OS entropy.
        /// </summary>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.default_rng.html
        ///     <br/>
        ///     This is the recommended constructor for the modern <see cref="Generator"/> API and is
        ///     independent of the legacy global <c>np.random</c> (<c>RandomState</c>) state.
        /// </remarks>
        public Generator default_rng() => new Generator(new PCG64(new SeedSequence()));

        /// <summary>Construct a new <see cref="Generator"/> (PCG64) seeded from a single integer.</summary>
        public Generator default_rng(long seed) => new Generator(new PCG64(new SeedSequence(seed)));

        /// <summary>Construct a new <see cref="Generator"/> (PCG64) seeded from a sequence of integers.</summary>
        public Generator default_rng(int[] seed) => new Generator(new PCG64(new SeedSequence(seed)));

        /// <summary>Construct a new <see cref="Generator"/> (PCG64) seeded from a sequence of integers.</summary>
        public Generator default_rng(long[] seed) => new Generator(new PCG64(new SeedSequence(seed)));

        /// <summary>Construct a new <see cref="Generator"/> (PCG64) from a prepared <see cref="SeedSequence"/>.</summary>
        public Generator default_rng(SeedSequence seed) => new Generator(new PCG64(seed));

        /// <summary>Wrap an existing bit generator in a <see cref="Generator"/> (NumPy passes it through).</summary>
        public Generator default_rng(BitGenerator bitGenerator) => new Generator(bitGenerator);

        /// <summary>Pass an existing <see cref="Generator"/> through unaltered (NumPy behavior).</summary>
        public Generator default_rng(Generator generator) => generator;
    }
}
