using System;

namespace NumSharp
{
    /// <summary>
    ///     A class that serves as numpy.random.RandomState in python.
    ///     Uses MT19937 (Mersenne Twister) for NumPy-compatible random number generation by default, and any
    ///     <see cref="BitGenerator"/> when constructed from one (NumPy's <c>RandomState(bit_generator)</c>).
    /// </summary>
    /// <remarks>
    ///     https://numpy.org/doc/stable/reference/random/index.html
    ///     <para>
    ///     Seeding follows NumPy's RandomState: an unseeded instance draws 128 bits of OS entropy through
    ///     <see cref="SeedSequence"/> (NumPy's <c>MT19937()</c>), while <c>RandomState(seed)</c> /
    ///     <c>seed(seed)</c> use MT19937's legacy integer/array initializers (<see cref="MT19937._legacy_seeding(long)"/>),
    ///     so seeded streams are byte-identical to NumPy's legacy streams.
    ///     </para>
    ///     <para>
    ///     Every sampler validates its parameters with NumPy's constraints and messages, then draws under the bit
    ///     generator's <see cref="BitGenerator.@lock"/> — the same lock a <see cref="Generator"/> sharing the bit generator
    ///     takes — so concurrent callers interleave whole calls, never half-drawn values. The legacy algorithms are ports of
    ///     NumPy's frozen <c>legacy-distributions.c</c> (see <c>NumPyRandom.LegacyDistributions.cs</c>).
    ///     </para>
    /// </remarks>
    [ModuleName("np.random")]
    public partial class NumPyRandom
    {
        /// <summary>
        ///     NumPy's <c>RandomState._poisson_lam_max</c>: the largest <c>lam</c> the (int64) Poisson sampler accepts,
        ///     <c>&lt;double&gt;iinfo(int64).max - sqrt(iinfo(int64).max) * 10</c>.
        /// </summary>
        public const double _poisson_lam_max = RandomConstraints.PoissonLamMax;

        /// <summary>
        ///     The bit generator every draw comes from (NumPy's <c>RandomState._bit_generator</c>) — MT19937 unless the
        ///     instance was built from another <see cref="BitGenerator"/>.
        /// </summary>
        /// <remarks>
        ///     Only MT19937 supports the legacy seeding and the legacy state tuple; <see cref="seed()"/> and
        ///     <see cref="get_state()"/> refuse other generators exactly where NumPy does.
        /// </remarks>
        protected internal BitGenerator randomizer;

        /// <summary>
        ///     Cached Gaussian value from the polar method.
        ///     NumPy caches the second value to maintain state reproducibility.
        /// </summary>
        private bool _hasGauss;
        private double _gaussCache;

        /// <summary>The last explicit integer seed (NumSharp bookkeeping; not part of NumPy's API).</summary>
        public int Seed { get; set; }

        #region Constructors

        /// <summary>
        ///     Wraps an existing bit generator (NumPy's <c>RandomState(bit_generator)</c>); null means a fresh OS-entropy
        ///     MT19937, as NumPy's <c>RandomState(None)</c> does.
        /// </summary>
        /// <param name="bitGenerator">The engine; draws advance ITS state (it is shared, not copied).</param>
        /// <remarks>The Gaussian cache starts empty (NumPy's <c>_initialize_bit_generator</c> resets it).</remarks>
        protected internal NumPyRandom(BitGenerator bitGenerator)
        {
            this.randomizer = bitGenerator ?? new MT19937();
            ResetGauss();
        }

        /// <summary>Creates a RandomState restored from a legacy state tuple.</summary>
        /// <param name="nativeRandomState">The state to restore (key, position and Gaussian cache).</param>
        /// <exception cref="ValueError">The algorithm is not <c>"MT19937"</c>.</exception>
        /// <exception cref="TypeError">The key is null.</exception>
        /// <exception cref="IndexError">The key is shorter than 624 words.</exception>
        protected internal NumPyRandom(NativeRandomState nativeRandomState)
        {
            randomizer = MT19937.LegacySeeded(0);
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

        /// <summary>
        ///     The bit generator this RandomState draws from (NumPy's public <c>RandomState._bit_generator</c> attribute).
        /// </summary>
        /// <remarks>
        ///     Shared, not copied: drawing from it directly (or through a <see cref="Generator"/> wrapped around it) advances
        ///     this RandomState's stream too, and both serialize on its <see cref="BitGenerator.@lock"/>.
        /// </remarks>
        public BitGenerator _bit_generator => randomizer;

        /// <summary>NumPy's <c>str(RandomState)</c>: the class name and the bit generator's, e.g. <c>RandomState(MT19937)</c>.</summary>
        /// <returns>The display string.</returns>
        public override string ToString() => $"RandomState({randomizer.Name})";

        #region Gaussian

        /// <summary>
        ///     Returns a random sample from the standard normal distribution (mean=0, std=1).
        ///     Uses the polar method (Marsaglia) matching NumPy's legacy RandomState exactly (<c>legacy_gauss</c>).
        /// </summary>
        /// <returns>A standard normal draw (the second of each generated pair is cached and returned next).</returns>
        /// <remarks>
        ///     NumPy's legacy RandomState uses the polar method (not Box-Muller) with caching.
        ///     The polar method generates two uniform values in [-1,1], rejects if outside unit circle,
        ///     then transforms to standard normal. The second value is cached.
        ///
        ///     Consuming the cached value also ZEROES it (<c>aug_state-&gt;gauss = 0.0</c>), which
        ///     <see cref="get_state()"/> exposes — a consumed cache reads <c>(has_gauss=0, gauss=0.0)</c>, not the stale value.
        ///     The caller holds the bit generator's lock.
        ///
        ///     This is critical for matching NumPy's randn() output exactly.
        /// </remarks>
        protected internal unsafe double NextGaussian()
        {
            // A one-double buffer is exactly the per-draw call sequence (see DrawBufferDouble).
            double word;
            var src = new DrawBufferDouble(randomizer, &word, 1);
            return LegacyGauss(ref src);
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
        ///     Returns a new <see cref="NumPyRandom"/> with the legacy integer seeding over NumPy's full seed range
        ///     (<c>RandomState(seed)</c> for <c>seed</c> in <c>[0, 2**32 - 1]</c>).
        /// </summary>
        /// <param name="seed">The seed.</param>
        /// <returns>A legacy generator whose stream matches <c>np.random.RandomState(seed)</c>.</returns>
        /// <exception cref="ValueError"><paramref name="seed"/> is outside <c>[0, 2**32 - 1]</c> (<c>Seed must be between 0 and 2**32 - 1</c>).</exception>
        public NumPyRandom RandomState(long seed)
        {
            var bg = new MT19937();
            bg._legacy_seeding(seed);
            return new NumPyRandom(bg) { Seed = (int)seed };
        }

        /// <summary>
        ///     Returns a new <see cref="NumPyRandom"/> seeded with MT19937's <c>init_by_array</c> (NumPy's
        ///     <c>RandomState([w0, w1, ...])</c>).
        /// </summary>
        /// <param name="seed">The key words (non-empty).</param>
        /// <returns>A legacy generator whose stream matches <c>np.random.RandomState(seed)</c>.</returns>
        /// <exception cref="ValueError"><paramref name="seed"/> is null or empty (<c>Seed must be non-empty</c>).</exception>
        public NumPyRandom RandomState(uint[] seed)
        {
            var bg = new MT19937();
            bg._legacy_seeding(seed);
            return new NumPyRandom(bg);
        }

        /// <summary>
        ///     Returns a new <see cref="NumPyRandom"/> seeded with MT19937's <c>init_by_array</c> from signed words (NumPy's
        ///     <c>RandomState([w0, w1, ...])</c>); each word must lie in <c>[0, 2**32 - 1]</c>.
        /// </summary>
        /// <param name="seed">The key words (non-empty, each in range).</param>
        /// <returns>A legacy generator whose stream matches <c>np.random.RandomState(seed)</c>.</returns>
        /// <exception cref="ValueError">Empty (<c>Seed must be non-empty</c>) or an element out of range (<c>Seed must be between 0 and 2**32 - 1</c>).</exception>
        public NumPyRandom RandomState(long[] seed)
        {
            var bg = new MT19937();
            bg._legacy_seeding(seed);
            return new NumPyRandom(bg);
        }

        /// <summary>
        ///     Returns a new <see cref="NumPyRandom"/> seeded with MT19937's <c>init_by_array</c> from signed words (NumPy's
        ///     <c>RandomState([w0, w1, ...])</c>); each word must be non-negative.
        /// </summary>
        /// <param name="seed">The key words (non-empty, each non-negative).</param>
        /// <returns>A legacy generator whose stream matches <c>np.random.RandomState(seed)</c>.</returns>
        /// <exception cref="ValueError">Empty (<c>Seed must be non-empty</c>) or a negative element (<c>Seed must be between 0 and 2**32 - 1</c>).</exception>
        public NumPyRandom RandomState(int[] seed)
        {
            var bg = new MT19937();
            bg._legacy_seeding(seed);
            return new NumPyRandom(bg);
        }

        /// <summary>
        ///     Returns a new <see cref="NumPyRandom"/> drawing from <paramref name="bit_generator"/> — NumPy's
        ///     <c>RandomState(bit_generator)</c>: the legacy samplers on any engine (PCG64, Philox, …).
        /// </summary>
        /// <param name="bit_generator">The engine, shared rather than copied; null gives a fresh OS-entropy MT19937 (NumPy's <c>None</c>).</param>
        /// <returns>A legacy generator over <paramref name="bit_generator"/>.</returns>
        /// <remarks>
        ///     Only an MT19937-backed instance can be re-seeded (<see cref="seed()"/>) or return the legacy state tuple
        ///     (<see cref="get_state()"/>); the dict form (<see cref="get_state(bool)"/> with <c>legacy: false</c>) and
        ///     <see cref="set_state(State)"/> work for every engine.
        /// </remarks>
        public NumPyRandom RandomState(BitGenerator bit_generator)
        {
            return new NumPyRandom(bit_generator);
        }

        /// <summary>
        ///     Returns a new <see cref="NumPyRandom"/> restored from a legacy state tuple.
        /// </summary>
        /// <param name="state">The state (key, position and Gaussian cache).</param>
        /// <returns>A legacy generator continuing from <paramref name="state"/>.</returns>
        /// <exception cref="ValueError">The algorithm is not <c>"MT19937"</c>.</exception>
        /// <exception cref="TypeError">The key is null.</exception>
        /// <exception cref="IndexError">The key is shorter than 624 words.</exception>
        public NumPyRandom RandomState(NativeRandomState state)
        {
            return new NumPyRandom(state);
        }

        #endregion

        /// <summary>
        ///     The MT19937 behind this RandomState, for the legacy seeding and the legacy state tuple — NumPy's
        ///     <c>isinstance(self._bit_generator, MT19937)</c> gate.
        /// </summary>
        /// <returns>The bit generator as MT19937.</returns>
        /// <exception cref="TypeError">The bit generator is not MT19937 (<c>can only re-seed a MT19937 BitGenerator</c>).</exception>
        private MT19937 ReseedableGenerator()
            => randomizer as MT19937 ?? throw new TypeError("can only re-seed a MT19937 BitGenerator");

        /// <summary>
        ///     Re-seeds from fresh OS entropy (NumPy's <c>seed()</c> / <c>seed(None)</c>).
        /// </summary>
        /// <exception cref="TypeError">The bit generator is not MT19937 (<c>can only re-seed a MT19937 BitGenerator</c>).</exception>
        /// <remarks>
        ///     NumPy's <c>_legacy_seeding(None)</c> fills the key from a new <see cref="SeedSequence"/> and LEAVES the
        ///     position where it was (observable through <see cref="get_state()"/>); the Gaussian cache is cleared.
        /// </remarks>
        public void seed()
        {
            var mt = ReseedableGenerator();
            lock (mt.@lock)
            {
                mt._legacy_seeding();
                ResetGauss();
            }
        }

        /// <summary>
        ///     Seeds the generator with a uint value (full NumPy range).
        ///     It can be called again to re-seed the generator.
        /// </summary>
        /// <param name="seed">Seed value in range [0, 2^32-1].</param>
        /// <exception cref="TypeError">The bit generator is not MT19937 (<c>can only re-seed a MT19937 BitGenerator</c>).</exception>
        /// <remarks>
        ///     This uses the MT19937 algorithm matching NumPy exactly.
        ///     Same seed produces identical sequences to NumPy.
        /// </remarks>
        public void seed(uint seed)
        {
            var mt = ReseedableGenerator();
            lock (mt.@lock)
            {
                Seed = (int)seed;
                mt._legacy_seeding(seed);
                ResetGauss();
            }
        }

        /// <summary>
        ///     Seeds the generator with an int value.
        ///     Validates that seed is non-negative (NumPy behavior).
        /// </summary>
        /// <param name="seed">Seed value in range [0, 2^31-1].</param>
        /// <exception cref="TypeError">The bit generator is not MT19937 — checked BEFORE the seed value, as in NumPy.</exception>
        /// <exception cref="ValueError">If seed is negative.</exception>
        /// <remarks>
        ///     NumPy accepts 0 to 2^32-1. Negative values throw:
        ///     "Seed must be between 0 and 2**32 - 1"
        /// </remarks>
        public void seed(int seed)
        {
            ReseedableGenerator();
            if (seed < 0)
                throw new ValueError("Seed must be between 0 and 2**32 - 1");
            this.seed((uint)seed);
        }

        /// <summary>
        ///     Seeds the generator with a long value.
        ///     Validates that seed is in range [0, 2^32-1] (NumPy behavior).
        /// </summary>
        /// <param name="seed">Seed value in range [0, 2^32-1].</param>
        /// <exception cref="TypeError">The bit generator is not MT19937 — checked BEFORE the seed value, as in NumPy.</exception>
        /// <exception cref="ValueError">If seed is out of range.</exception>
        public void seed(long seed)
        {
            ReseedableGenerator();
            if (seed < 0 || seed > uint.MaxValue)
                throw new ValueError("Seed must be between 0 and 2**32 - 1");
            this.seed((uint)seed);
        }

        /// <summary>
        ///     Seeds the generator with a ulong value.
        ///     Validates that seed is in range [0, 2^32-1] (NumPy behavior).
        /// </summary>
        /// <param name="seed">Seed value in range [0, 2^32-1].</param>
        /// <exception cref="TypeError">The bit generator is not MT19937 — checked BEFORE the seed value, as in NumPy.</exception>
        /// <exception cref="ValueError">If seed is out of range.</exception>
        public void seed(ulong seed)
        {
            ReseedableGenerator();
            if (seed > uint.MaxValue)
                throw new ValueError("Seed must be between 0 and 2**32 - 1");
            this.seed((uint)seed);
        }

        /// <summary>
        ///     Seeds the generator with an array of uint values (NumPy's <c>init_by_array</c> seeding).
        /// </summary>
        /// <param name="seed">The key words (non-empty).</param>
        /// <exception cref="TypeError">The bit generator is not MT19937 (<c>can only re-seed a MT19937 BitGenerator</c>).</exception>
        /// <exception cref="ValueError"><paramref name="seed"/> is null or empty (<c>Seed must be non-empty</c> — it used to seed 0 silently).</exception>
        public void seed(uint[] seed)
        {
            var mt = ReseedableGenerator();
            lock (mt.@lock)
            {
                mt._legacy_seeding(seed);
                Seed = (int)seed[0];
                ResetGauss();
            }
        }

        /// <summary>
        ///     Seeds the generator with an array of integers (NumPy's <c>seed([...])</c>): each element must lie in
        ///     <c>[0, 2**32 - 1]</c>.
        /// </summary>
        /// <param name="seed">The key words (non-empty).</param>
        /// <exception cref="TypeError">The bit generator is not MT19937 — checked BEFORE the words, as in NumPy.</exception>
        /// <exception cref="ValueError">Empty (<c>Seed must be non-empty</c>) or an element out of range (<c>Seed must be between 0 and 2**32 - 1</c>).</exception>
        public void seed(long[] seed)
        {
            ReseedableGenerator();
            this.seed(MT19937.ValidateLegacyArray(seed));
        }

        /// <summary>
        ///     Seeds the generator with an array of integers (NumPy's <c>seed([...])</c>): each element must be non-negative.
        /// </summary>
        /// <param name="seed">The key words (non-empty).</param>
        /// <exception cref="TypeError">The bit generator is not MT19937 — checked BEFORE the words, as in NumPy.</exception>
        /// <exception cref="ValueError">Empty (<c>Seed must be non-empty</c>) or a negative element (<c>Seed must be between 0 and 2**32 - 1</c>).</exception>
        public void seed(int[] seed)
        {
            ReseedableGenerator();
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
        ///     Set the internal state of the generator from a <see cref="NativeRandomState"/> — NumPy's
        ///     <c>set_state(('MT19937', key, pos, has_gauss, cached_gaussian))</c>.
        ///     For use if one has reason to manually (re-)set the internal state of the pseudo-random number generating algorithm.
        /// </summary>
        /// <param name="state">The state to restore onto this <see cref="NumPyRandom"/></param>
        /// <exception cref="ValueError">
        ///     The algorithm is not <c>"MT19937"</c> (<c>set_state can only be used with legacy MT19937 state instances.</c>),
        ///     the RandomState wraps another bit generator (<c>state must be for a PCG64 RNG</c>, …), or the position is outside
        ///     <c>[0, 624]</c>.
        /// </exception>
        /// <exception cref="TypeError">The key is null.</exception>
        /// <exception cref="IndexError">The key is shorter than 624 words.</exception>
        /// <remarks>
        ///     Mirrors NumPy's order inside the lock: the Gaussian cache is written FIRST, then the bit generator's state — so
        ///     a state the bit generator rejects still leaves the tuple's cached Gaussian installed, as in NumPy. A tuple built
        ///     with the 3-argument constructor (NumPy's <c>('MT19937', key, pos)</c> form) clears the cache.
        /// </remarks>
        public void set_state(NativeRandomState state)
        {
            if (state.Algorithm != "MT19937")
                throw new ValueError("set_state can only be used with legacy MT19937 state instances.");
            SetStateCore(new MT19937.State(state.Key, state.Pos), state.HasGauss, state.CachedGaussian);
        }

        /// <summary>
        ///     Set the internal state from NumPy's dict form (<see cref="State"/>, what <c>get_state(legacy: false)</c>
        ///     returns) — the only form that restores a non-MT19937 RandomState.
        /// </summary>
        /// <param name="state">The bit generator's state plus the cached Gaussian.</param>
        /// <exception cref="TypeError"><paramref name="state"/> is null (<c>state must be a dict or a tuple.</c>).</exception>
        /// <exception cref="ValueError">
        ///     The dict has no bit generator state (<c>state dictionary is not valid.</c>) or it belongs to another bit
        ///     generator (<c>state must be for a PCG64 RNG</c>, …).
        /// </exception>
        /// <remarks>Same write order as <see cref="set_state(NativeRandomState)"/>: the cached Gaussian first.</remarks>
        public void set_state(State state)
        {
            if (state is null)
                throw new TypeError("state must be a dict or a tuple.");
            if (state.state is null)
                throw new ValueError("state dictionary is not valid.");
            SetStateCore(state.state, state.has_gauss, state.gauss);
        }

        /// <summary>
        ///     Set the internal state from any of NumPy's accepted forms — the dispatch of NumPy's duck-typed
        ///     <c>set_state(state)</c>: a legacy tuple (<see cref="NativeRandomState"/>), the RandomState dict
        ///     (<see cref="State"/>), or a bare bit-generator state dict (<see cref="BitGeneratorState"/>, which NumPy accepts
        ///     with <c>has_gauss</c>/<c>gauss</c> defaulting to 0).
        /// </summary>
        /// <param name="state">The state.</param>
        /// <exception cref="TypeError"><paramref name="state"/> is none of the accepted forms (<c>state must be a dict or a tuple.</c>).</exception>
        /// <exception cref="ValueError">The state is invalid or belongs to another bit generator (see the typed overloads).</exception>
        /// <exception cref="IndexError">A legacy tuple's key is shorter than 624 words.</exception>
        public void set_state(object state)
        {
            switch (state)
            {
                case NativeRandomState tuple:
                    set_state(tuple);
                    return;
                case State dict:
                    set_state(dict);
                    return;
                case BitGeneratorState bare:
                    SetStateCore(bare, 0, 0.0);
                    return;
                default:
                    throw new TypeError("state must be a dict or a tuple.");
            }
        }

        /// <summary>
        ///     The shared tail of every <c>set_state</c> form: under the lock, install the Gaussian cache, then hand the bit
        ///     generator its state (which validates it — type, key, position).
        /// </summary>
        /// <param name="bitGeneratorState">The bit generator's state.</param>
        /// <param name="hasGauss">Non-zero when a Gaussian is cached (NumPy stores the int, any non-zero meaning "cached").</param>
        /// <param name="gauss">The cached Gaussian.</param>
        /// <exception cref="ValueError">The state belongs to another bit generator, or is otherwise invalid.</exception>
        /// <exception cref="TypeError">A required state member is null.</exception>
        /// <exception cref="IndexError">An MT19937 key is shorter than 624 words.</exception>
        private void SetStateCore(BitGeneratorState bitGeneratorState, int hasGauss, double gauss)
        {
            lock (randomizer.@lock)
            {
                // NumPy writes the cache before assigning the bit generator state; a rejected assignment keeps it.
                _gaussCache = gauss;
                _hasGauss = hasGauss != 0;
                randomizer.state = bitGeneratorState;
            }
        }

        /// <summary>
        ///     Return a <see cref="NativeRandomState"/> representing the internal state of the generator — NumPy's legacy
        ///     tuple <c>('MT19937', key, pos, has_gauss, cached_gaussian)</c>, the default <c>get_state()</c>.
        /// </summary>
        /// <returns>The current state, including Gaussian cache.</returns>
        /// <exception cref="ValueError">
        ///     The RandomState wraps a non-MT19937 bit generator. NumPy warns (<c>RuntimeWarning</c>) and returns the dict form
        ///     instead; a C# method cannot change its return type at run time, so this overload refuses — call
        ///     <see cref="get_state(bool)"/> with <c>legacy: false</c> for the dict.
        /// </exception>
        public NativeRandomState get_state()
        {
            if (randomizer is not MT19937 mt)
                throw new ValueError("get_state and legacy can only be used with the MT19937 BitGenerator. "
                                     + $"Call get_state(legacy: false) for the {randomizer.Name} state dict.");
            lock (mt.@lock)
            {
                return new NativeRandomState(
                    key: mt.Key,
                    pos: mt.Pos,
                    hasGauss: _hasGauss ? 1 : 0,
                    cachedGaussian: _gaussCache
                );
            }
        }

        /// <summary>
        ///     Return the state in either of NumPy's forms (<c>get_state(legacy=...)</c>): the legacy
        ///     <see cref="NativeRandomState"/> tuple when <paramref name="legacy"/> is true and the bit generator is MT19937,
        ///     otherwise the <see cref="State"/> dict.
        /// </summary>
        /// <param name="legacy">Prefer the legacy tuple (NumPy's default). Ignored — as NumPy ignores it, after a
        ///     <c>RuntimeWarning</c> — when the bit generator is not MT19937.</param>
        /// <returns>A boxed <see cref="NativeRandomState"/> or a <see cref="State"/>; both round-trip through
        ///     <see cref="set_state(object)"/>.</returns>
        /// <remarks>
        ///     The bit generator state and the Gaussian cache are read under one lock acquisition, so the snapshot is atomic
        ///     (NumPy takes the lock twice; single-threaded the results are identical). NumSharp has no warnings channel, so
        ///     the non-MT19937 <c>legacy=True</c> case silently returns the dict.
        /// </remarks>
        public object get_state(bool legacy)
        {
            lock (randomizer.@lock)
            {
                if (legacy && randomizer is MT19937)
                    return get_state();
                return new State(randomizer.state, _hasGauss ? 1 : 0, _gaussCache);
            }
        }

        /// <summary>
        ///     NumPy's <c>RandomState.get_state(legacy=False)</c> dict: the bit generator's state plus RandomState's cached
        ///     Gaussian — the one state form every bit generator supports.
        /// </summary>
        /// <remarks>
        ///     NumPy's dict is <c>{'bit_generator': name, 'state': {...}, [generator-specific keys], 'has_gauss': int,
        ///     'gauss': float}</c>; here the bit generator's part is its typed <see cref="BitGeneratorState"/> (whose own members
        ///     already flatten <c>'state'</c> and the generator-specific keys), and <see cref="has_gauss"/>/<see cref="gauss"/>
        ///     are the two keys RandomState adds. The object is plain mutable data: edit it and pass it to
        ///     <see cref="set_state(State)"/>.
        /// </remarks>
        public sealed class State
        {
            /// <summary>Creates an empty state (set <see cref="state"/> before passing it to <see cref="set_state(State)"/>).</summary>
            public State() { }

            /// <summary>Creates a state from its parts.</summary>
            /// <param name="state">The bit generator's state (NumPy's <c>'bit_generator'</c>/<c>'state'</c> entries).</param>
            /// <param name="has_gauss">Non-zero when a Gaussian is cached.</param>
            /// <param name="gauss">The cached Gaussian (0.0 when none).</param>
            public State(BitGeneratorState state, int has_gauss = 0, double gauss = 0.0)
            {
                this.state = state;
                this.has_gauss = has_gauss;
                this.gauss = gauss;
            }

            /// <summary>NumPy's <c>'bit_generator'</c> key: the bit generator's class name (null when <see cref="state"/> is unset).</summary>
            public string bit_generator => state?.bit_generator;

            /// <summary>The bit generator's state — NumPy's <c>'state'</c> sub-dict together with the generator's own keys.</summary>
            public BitGeneratorState state { get; set; }

            /// <summary>NumPy's <c>'has_gauss'</c>: non-zero when a Gaussian is cached.</summary>
            public int has_gauss { get; set; }

            /// <summary>NumPy's <c>'gauss'</c>: the cached Gaussian (0.0 when none — a consumed cache is zeroed).</summary>
            public double gauss { get; set; }
        }
    }
}
