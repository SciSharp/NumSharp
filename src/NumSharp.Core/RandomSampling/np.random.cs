using System;

namespace NumSharp
{
    /// <summary>
    ///     A class that serves as numpy.random.RandomState in python.
    ///     Uses MT19937 (Mersenne Twister) for NumPy-compatible random number generation.
    /// </summary>
    /// <remarks>
    ///     https://numpy.org/doc/stable/reference/random/index.html
    ///     <para>
    ///     Seeding follows NumPy's RandomState: an unseeded instance draws 128 bits of OS entropy through
    ///     <see cref="SeedSequence"/> (NumPy's <c>MT19937()</c>), while <c>RandomState(seed)</c> /
    ///     <c>seed(seed)</c> use MT19937's legacy integer/array initializers (<see cref="MT19937._legacy_seeding(long)"/>),
    ///     so seeded streams are byte-identical to NumPy's legacy streams.
    ///     </para>
    /// </remarks>
    [ModuleName("np.random")]
    public partial class NumPyRandom
    {
        /// <summary>
        ///     The MT19937 bit generator (NumPy-compatible).
        /// </summary>
        protected internal MT19937 randomizer;

        /// <summary>
        ///     Cached Gaussian value from Box-Muller transform.
        ///     NumPy caches the second value to maintain state reproducibility.
        /// </summary>
        private bool _hasGauss;
        private double _gaussCache;

        /// <summary>The last explicit integer seed (NumSharp bookkeeping; not part of NumPy's API).</summary>
        public int Seed { get; set; }

        #region Constructors

        /// <summary>Wraps an existing MT19937 bit generator (NumPy's <c>RandomState(bit_generator)</c>).</summary>
        /// <param name="bitGenerator">The engine; draws advance ITS state.</param>
        protected internal NumPyRandom(MT19937 bitGenerator)
        {
            this.randomizer = bitGenerator;
        }

        /// <summary>Creates a RandomState restored from a legacy state tuple.</summary>
        /// <param name="nativeRandomState">The state to restore (key, position and Gaussian cache).</param>
        /// <exception cref="ArgumentException">The key is not 624 words long.</exception>
        protected internal NumPyRandom(NativeRandomState nativeRandomState)
        {
            set_state(nativeRandomState);
        }

        /// <summary>Creates a RandomState with the legacy integer seeding (NumPy's <c>RandomState(seed)</c>).</summary>
        /// <param name="seed">The seed, in <c>[0, 2**32 - 1]</c>.</param>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative (<c>Seed must be between 0 and 2**32 - 1</c>).</exception>
        protected internal NumPyRandom(int seed)
        {
            if (seed < 0)
                throw new ValueError("Seed must be between 0 and 2**32 - 1");
            Seed = seed;
            randomizer = MT19937.LegacySeeded((uint)seed);
        }

        /// <summary>
        ///     Creates a RandomState seeded from fresh OS entropy (NumPy's <c>RandomState()</c>: an
        ///     <c>MT19937()</c> seeded through <see cref="SeedSequence"/>, so its state has <c>key[0] = 0x80000000</c>
        ///     and <c>pos = 623</c>).
        /// </summary>
        /// <remarks>
        ///     Two instances created back to back are independent — the previous clock-tick seeding gave
        ///     identical streams to instances created within the same millisecond.
        /// </remarks>
        protected internal NumPyRandom()
        {
            randomizer = new MT19937();
        }

        #endregion

        #region Gaussian

        /// <summary>
        ///     Returns a random sample from the standard normal distribution (mean=0, std=1).
        ///     Uses the polar method (Marsaglia) matching NumPy's legacy RandomState exactly.
        /// </summary>
        /// <returns>A standard normal draw (the second of each generated pair is cached and returned next).</returns>
        /// <remarks>
        ///     NumPy's legacy RandomState uses the polar method (not Box-Muller) with caching.
        ///     The polar method generates two uniform values in [-1,1], rejects if outside unit circle,
        ///     then transforms to standard normal. The second value is cached.
        ///
        ///     This is critical for matching NumPy's randn() output exactly.
        /// </remarks>
        protected internal double NextGaussian()
        {
            // Return cached value if available (NumPy behavior)
            if (_hasGauss)
            {
                _hasGauss = false;
                return _gaussCache;
            }

            // Polar method (Marsaglia) - matches NumPy's random_standard_normal
            double x, y, r2;
            do
            {
                // Generate x, y uniform in [-1, 1]
                x = 2.0 * randomizer.NextDouble() - 1.0;
                y = 2.0 * randomizer.NextDouble() - 1.0;
                r2 = x * x + y * y;
            } while (r2 >= 1.0 || r2 == 0.0);

            // Polar transform
            double d = Math.Sqrt(-2.0 * Math.Log(r2) / r2);

            // NumPy caches x*d and returns y*d first
            _gaussCache = x * d;
            _hasGauss = true;

            // Return y*d (NumPy convention)
            return y * d;
        }

        #endregion

        #region RandomState

        /// <summary>
        ///     Returns a new <see cref="NumPyRandom"/> seeded from fresh OS entropy (NumPy's <c>RandomState()</c>).
        /// </summary>
        /// <returns>An independently seeded legacy generator.</returns>
        public NumPyRandom RandomState()
        {
            return new NumPyRandom();
        }

        /// <summary>
        ///     Returns a new <see cref="NumPyRandom"/> with the legacy integer seeding (NumPy's <c>RandomState(seed)</c>).
        /// </summary>
        /// <param name="seed">The seed, in <c>[0, 2**31 - 1]</c> for this overload.</param>
        /// <returns>A legacy generator whose stream matches <c>np.random.RandomState(seed)</c>.</returns>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative.</exception>
        public NumPyRandom RandomState(int seed)
        {
            return new NumPyRandom(seed);
        }

        /// <summary>
        ///     Returns a new <see cref="NumPyRandom"/> restored from a legacy state tuple.
        /// </summary>
        /// <param name="state">The state (key, position and Gaussian cache).</param>
        /// <returns>A legacy generator continuing from <paramref name="state"/>.</returns>
        /// <exception cref="ArgumentException">The key is not 624 words long.</exception>
        public NumPyRandom RandomState(NativeRandomState state)
        {
            return new NumPyRandom(state);
        }

        #endregion

        /// <summary>
        ///     Re-seeds from fresh OS entropy (NumPy's <c>seed()</c> / <c>seed(None)</c>).
        /// </summary>
        /// <remarks>
        ///     NumPy's <c>_legacy_seeding(None)</c> fills the key from a new <see cref="SeedSequence"/> and LEAVES the
        ///     position where it was (observable through <see cref="get_state"/>); the Gaussian cache is cleared.
        /// </remarks>
        public void seed()
        {
            randomizer._legacy_seeding();
            ResetGauss();
        }

        /// <summary>
        ///     Seeds the generator with a uint value (full NumPy range).
        ///     It can be called again to re-seed the generator.
        /// </summary>
        /// <param name="seed">Seed value in range [0, 2^32-1].</param>
        /// <remarks>
        ///     This uses the MT19937 algorithm matching NumPy exactly.
        ///     Same seed produces identical sequences to NumPy.
        /// </remarks>
        public void seed(uint seed)
        {
            Seed = (int)seed;
            randomizer._legacy_seeding(seed);
            ResetGauss();
        }

        /// <summary>
        ///     Seeds the generator with an int value.
        ///     Validates that seed is non-negative (NumPy behavior).
        /// </summary>
        /// <param name="seed">Seed value in range [0, 2^31-1].</param>
        /// <exception cref="ValueError">If seed is negative.</exception>
        /// <remarks>
        ///     NumPy accepts 0 to 2^32-1. Negative values throw:
        ///     "Seed must be between 0 and 2**32 - 1"
        /// </remarks>
        public void seed(int seed)
        {
            if (seed < 0)
                throw new ValueError("Seed must be between 0 and 2**32 - 1");
            this.seed((uint)seed);
        }

        /// <summary>
        ///     Seeds the generator with a long value.
        ///     Validates that seed is in range [0, 2^32-1] (NumPy behavior).
        /// </summary>
        /// <param name="seed">Seed value in range [0, 2^32-1].</param>
        /// <exception cref="ValueError">If seed is out of range.</exception>
        public void seed(long seed)
        {
            if (seed < 0 || seed > uint.MaxValue)
                throw new ValueError("Seed must be between 0 and 2**32 - 1");
            this.seed((uint)seed);
        }

        /// <summary>
        ///     Seeds the generator with a ulong value.
        ///     Validates that seed is in range [0, 2^32-1] (NumPy behavior).
        /// </summary>
        /// <param name="seed">Seed value in range [0, 2^32-1].</param>
        /// <exception cref="ValueError">If seed is out of range.</exception>
        public void seed(ulong seed)
        {
            if (seed > uint.MaxValue)
                throw new ValueError("Seed must be between 0 and 2**32 - 1");
            this.seed((uint)seed);
        }

        /// <summary>
        ///     Seeds the generator with an array of uint values (NumPy's <c>init_by_array</c> seeding).
        /// </summary>
        /// <param name="seed">The key words (non-empty).</param>
        /// <exception cref="ValueError"><paramref name="seed"/> is null or empty (<c>Seed must be non-empty</c> — it used to seed 0 silently).</exception>
        public void seed(uint[] seed)
        {
            randomizer._legacy_seeding(seed);
            Seed = (int)seed[0];
            ResetGauss();
        }

        /// <summary>
        ///     Seeds the generator with an array of integers (NumPy's <c>seed([...])</c>): each element must lie in
        ///     <c>[0, 2**32 - 1]</c>.
        /// </summary>
        /// <param name="seed">The key words (non-empty).</param>
        /// <exception cref="ValueError">Empty (<c>Seed must be non-empty</c>) or an element out of range (<c>Seed must be between 0 and 2**32 - 1</c>).</exception>
        public void seed(long[] seed) => this.seed(MT19937.ValidateLegacyArray(seed));

        /// <summary>
        ///     Seeds the generator with an array of integers (NumPy's <c>seed([...])</c>): each element must be non-negative.
        /// </summary>
        /// <param name="seed">The key words (non-empty).</param>
        /// <exception cref="ValueError">Empty (<c>Seed must be non-empty</c>) or a negative element (<c>Seed must be between 0 and 2**32 - 1</c>).</exception>
        public void seed(int[] seed)
        {
            if (seed is null || seed.Length == 0)
                throw new ValueError("Seed must be non-empty");
            var words = new long[seed.Length];
            for (int i = 0; i < seed.Length; i++)
                words[i] = seed[i];
            this.seed(words);
        }

        /// <summary>Clears the cached Gaussian (NumPy's <c>_reset_gauss</c>) — every re-seed does this.</summary>
        private void ResetGauss()
        {
            _hasGauss = false;
            _gaussCache = 0.0;
        }

        /// <summary>
        ///     Set the internal state of the generator from a <see cref="NativeRandomState"/>.
        ///     For use if one has reason to manually (re-)set the internal state of the pseudo-random number generating algorithm.
        /// </summary>
        /// <param name="state">The state to restore onto this <see cref="NumPyRandom"/></param>
        /// <exception cref="ArgumentException">The key is not 624 words long.</exception>
        public void set_state(NativeRandomState state)
        {
            if (state.Key == null || state.Key.Length != 624)
                throw new ArgumentException("Invalid state: key array must be length 624");

            if (randomizer == null)
                randomizer = MT19937.LegacySeeded(0);

            lock (randomizer.@lock)
            {
                randomizer.SetState(state.Key, state.Pos);
                _hasGauss = state.HasGauss != 0;
                _gaussCache = state.CachedGaussian;
            }
        }

        /// <summary>
        ///     Return a <see cref="NativeRandomState"/> representing the internal state of the generator.
        /// </summary>
        /// <returns>The current state, including Gaussian cache.</returns>
        public NativeRandomState get_state()
        {
            lock (randomizer.@lock)
            {
                return new NativeRandomState(
                    key: randomizer.Key,
                    pos: randomizer.Pos,
                    hasGauss: _hasGauss ? 1 : 0,
                    cachedGaussian: _gaussCache
                );
            }
        }
    }
}
