using System;

namespace NumSharp
{
    /// <summary>
    ///     A class that serves as numpy.random.RandomState in python.
    ///     Uses MT19937 (Mersenne Twister) for NumPy-compatible random number generation.
    /// </summary>
    /// <remarks>https://numpy.org/doc/stable/reference/random/index.html</remarks>
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

        public int Seed { get; set; }

        #region Constructors

        protected internal NumPyRandom(MT19937 bitGenerator)
        {
            this.randomizer = bitGenerator;
        }

        protected internal NumPyRandom(NativeRandomState nativeRandomState)
        {
            set_state(nativeRandomState);
        }

        protected internal NumPyRandom(int seed)
        {
            Seed = seed;
            randomizer = new MT19937(seed);
        }

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
        ///     Returns a new instance of <see cref="NumPyRandom"/>.
        /// </summary>
        public NumPyRandom RandomState()
        {
            return new NumPyRandom();
        }

        /// <summary>
        ///     Returns a new instance of <see cref="NumPyRandom"/>.
        /// </summary>
        public NumPyRandom RandomState(int seed)
        {
            return new NumPyRandom(seed);
        }

        /// <summary>
        ///     Returns a new instance of <see cref="NumPyRandom"/>.
        /// </summary>
        public NumPyRandom RandomState(NativeRandomState state)
        {
            return new NumPyRandom(state);
        }

        #endregion

        /// <summary>
        ///     Seeds the generator.
        ///     It can be called again to re-seed the generator.
        /// </summary>
        /// <remarks>
        ///     This uses the MT19937 algorithm matching NumPy exactly.
        ///     Same seed produces identical sequences to NumPy.
        /// </remarks>
        public void seed(int seed)
        {
            Seed = seed;
            randomizer = new MT19937(seed);
            // Clear Gaussian cache on reseed (NumPy behavior)
            _hasGauss = false;
            _gaussCache = 0.0;
        }

        /// <summary>
        ///     Set the internal state of the generator from a <see cref="NativeRandomState"/>.
        ///     For use if one has reason to manually (re-)set the internal state of the pseudo-random number generating algorithm.
        /// </summary>
        /// <param name="state">The state to restore onto this <see cref="NumPyRandom"/></param>
        public void set_state(NativeRandomState state)
        {
            if (state.Key == null || state.Key.Length != 624)
                throw new ArgumentException("Invalid state: key array must be length 624");

            if (randomizer == null)
                randomizer = new MT19937();

            randomizer.SetState(state.Key, state.Pos);
            _hasGauss = state.HasGauss != 0;
            _gaussCache = state.CachedGaussian;
        }

        /// <summary>
        ///     Return a <see cref="NativeRandomState"/> representing the internal state of the generator.
        /// </summary>
        /// <returns>The current state, including Gaussian cache.</returns>
        public NativeRandomState get_state()
        {
            return new NativeRandomState(
                key: (uint[])randomizer.Key.Clone(),
                pos: randomizer.Pos,
                hasGauss: _hasGauss ? 1 : 0,
                cachedGaussian: _gaussCache
            );
        }
    }
}
