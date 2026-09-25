using System;
using System.Numerics;

namespace NumSharp
{
    /// <summary>
    ///     The Mersenne Twister MT19937 bit generator (NumPy's <c>numpy.random.MT19937</c>) — the engine behind the
    ///     legacy <see cref="NumPyRandom"/> (<c>RandomState</c>) and usable with <see cref="Generator"/>.
    /// </summary>
    /// <remarks>
    ///     Port of NumPy 2.4.2's <c>_mt19937.pyx</c> + <c>src/mt19937/mt19937.{c,h}</c> + <c>mt19937-jump.c</c>.
    ///     <para>
    ///     <b>Two seeding schemes, as in NumPy.</b> Constructing an <see cref="MT19937"/> seeds it through
    ///     <see cref="SeedSequence"/> (<c>generate_state(624)</c>, <c>key[0] = 0x80000000</c>, <c>pos = 623</c>), so
    ///     <c>new MT19937(42)</c> matches <c>np.random.MT19937(42)</c>. The classic integer/array seeding of
    ///     <c>RandomState(42)</c> / <c>np.random.seed(42)</c> is <see cref="_legacy_seeding(long)"/> — it produces a
    ///     DIFFERENT stream (<c>1608637542, …</c>) and discards <see cref="BitGenerator.seed_seq"/>.
    ///     </para>
    ///     <para>
    ///     The per-draw primitives are the internal engine contract (<c>next_uint32</c> = one tempered word,
    ///     <c>next_uint64</c> = two words high-first, <c>next_double</c> = 27+26 bits, <c>next_raw</c> = the 32-bit
    ///     word widened); the public surface is NumPy's: <see cref="BitGenerator.random_raw"/>, <see cref="state"/>,
    ///     <see cref="BitGenerator.seed_seq"/>, <see cref="jumped"/>, <see cref="_legacy_seeding(long)"/>.
    ///     </para>
    /// </remarks>
    public sealed partial class MT19937 : BitGenerator
    {
        // Period parameters (mt19937.h).
        private const int N = 624;
        private const int M = 397;
        private const uint MATRIX_A = 0x9908b0dfU;
        private const uint UPPER_MASK = 0x80000000U;
        private const uint LOWER_MASK = 0x7fffffffU;

        // Tempering parameters.
        private const uint TEMPERING_MASK_B = 0x9d2c5680U;
        private const uint TEMPERING_MASK_C = 0xefc60000U;

        // The engine state (mt19937_state): the 624-word key and the index of the next word to temper.
        private readonly uint[] _key = new uint[N];
        private int _pos;

        /// <summary>Constructs an MT19937 seeded from fresh, unpredictable OS entropy (NumPy's <c>MT19937()</c> / <c>seed=None</c>).</summary>
        public MT19937() : this(new SeedSequence()) { }

        /// <summary>Constructs an MT19937 seeded from a non-negative integer through <see cref="SeedSequence"/> (NOT the legacy integer seeding).</summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative (<c>expected non-negative integer</c>).</exception>
        public MT19937(long seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs an MT19937 seeded from a non-negative integer through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed.</param>
        public MT19937(ulong seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs an MT19937 seeded from an arbitrary-size non-negative integer through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative.</exception>
        public MT19937(BigInteger seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs an MT19937 seeded from a sequence of non-negative integers through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed words (each must be non-negative).</param>
        /// <exception cref="ValueError">An element is negative.</exception>
        public MT19937(int[] seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs an MT19937 seeded from a sequence of non-negative integers through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed words (each must be non-negative).</param>
        /// <exception cref="ValueError">An element is negative.</exception>
        public MT19937(long[] seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs an MT19937 seeded from uint32 words through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed words.</param>
        public MT19937(uint[] seed) : this(new SeedSequence(seed)) { }

        /// <summary>
        ///     Constructs an MT19937 seeded from <paramref name="seedSeq"/> exactly as NumPy's <c>MT19937.__init__</c>:
        ///     <c>val = seed_seq.generate_state(624, np.uint32)</c>, <c>key[0] = 0x80000000</c>, <c>key[1:] = val[1:]</c>,
        ///     <c>pos = 623</c>.
        /// </summary>
        /// <param name="seedSeq">The seed sequence — normally a <see cref="SeedSequence"/>; any <see cref="ISeedSequence"/>
        /// works (NumPy uses it as-is when <c>isinstance(seed, ISeedSequence)</c>).</param>
        /// <exception cref="ArgumentNullException"><paramref name="seedSeq"/> is null.</exception>
        /// <exception cref="NotImplementedException"><paramref name="seedSeq"/> is a <see cref="SeedlessSeedSequence"/>.</exception>
        /// <remarks>
        ///     <c>pos</c> is 623, not 624: NumPy's fill loop leaves <c>i = 623</c> behind and stores it, so the first
        ///     draw tempers <c>key[623]</c> before the first twist. That is part of the stream and is kept.
        /// </remarks>
        public MT19937(ISeedSequence seedSeq) : base(seedSeq ?? throw new ArgumentNullException(nameof(seedSeq)))
        {
            SeedFromSequence(seedSeq);
        }

        /// <summary>Constructs an MT19937 in the legacy integer-seeded state without drawing OS entropy (RandomState(seed)).</summary>
        /// <param name="legacySeed">The 32-bit legacy seed (<c>mt19937_seed</c>).</param>
        /// <param name="legacy">Overload discriminator; always true.</param>
        private MT19937(uint legacySeed, bool legacy) : base(null)
        {
            SeedInteger(legacySeed);
        }

        /// <summary>
        ///     Builds an MT19937 exactly like NumPy's <c>RandomState(seed)</c> does — <c>MT19937()</c> followed by
        ///     <c>_legacy_seeding(seed)</c> — without drawing the OS entropy the throwaway constructor would.
        /// </summary>
        /// <param name="seed">The legacy 32-bit seed.</param>
        /// <returns>A legacy-seeded generator (no seed sequence).</returns>
        internal static MT19937 LegacySeeded(uint seed) => new MT19937(seed, true);

        /// <inheritdoc/>
        internal override string Name => "MT19937";

        /// <inheritdoc/>
        private protected override BitGenerator CreateFromSeed(ISeedSequence seed) => new MT19937(seed);

        /// <summary>NumPy's <c>MT19937.__init__</c> seeding body (and the <c>seed=None</c> branch of legacy seeding).</summary>
        /// <param name="seedSeq">The sequence supplying 624 words.</param>
        /// <param name="setPos">Whether to set <c>pos = 623</c> (the constructor does; <c>_legacy_seeding(None)</c> leaves pos alone).</param>
        /// <exception cref="NotImplementedException"><paramref name="seedSeq"/> is a <see cref="SeedlessSeedSequence"/>.</exception>
        private void SeedFromSequence(ISeedSequence seedSeq, bool setPos = true)
        {
            uint[] val = SeedWords32(seedSeq, N);
            // MSB is 1; assuring non-zero initial array.
            _key[0] = 0x80000000U;
            for (int i = 1; i < N; i++)
                _key[i] = val[i];
            if (setPos)
                _pos = N - 1;
        }

        /// <summary>NumPy's <c>mt19937_seed</c> (Knuth's initializer of the reference implementation).</summary>
        /// <param name="seed">The 32-bit seed.</param>
        private void SeedInteger(uint seed)
        {
            for (int pos = 0; pos < N; pos++)
            {
                _key[pos] = seed;
                seed = unchecked(1812433253U * (seed ^ (seed >> 30)) + (uint)pos + 1);
            }
            _pos = N;
        }

        /// <summary>NumPy's <c>mt19937_init_by_array</c> (Matsumoto &amp; Nishimura's <c>init_by_array</c>).</summary>
        /// <param name="initKey">The key words (non-empty).</param>
        private void InitByArray(uint[] initKey)
        {
            SeedInteger(19650218U); // init_genrand(19650218): leaves pos = 624
            int i = 1, j = 0;
            int k = N > initKey.Length ? N : initKey.Length;
            for (; k > 0; k--)
            {
                // non linear
                _key[i] = unchecked((_key[i] ^ ((_key[i - 1] ^ (_key[i - 1] >> 30)) * 1664525U)) + initKey[j] + (uint)j);
                i++;
                j++;
                if (i >= N)
                {
                    _key[0] = _key[N - 1];
                    i = 1;
                }
                if (j >= initKey.Length)
                    j = 0;
            }
            for (k = N - 1; k > 0; k--)
            {
                // non linear
                _key[i] = unchecked((_key[i] ^ ((_key[i - 1] ^ (_key[i - 1] >> 30)) * 1566083941U)) - (uint)i);
                i++;
                if (i >= N)
                {
                    _key[0] = _key[N - 1];
                    i = 1;
                }
            }
            _key[0] = 0x80000000U; // MSB is 1; assuring non-zero initial array
        }

        /// <summary>NumPy's <c>mt19937_gen</c>: the twist that regenerates all 624 words.</summary>
        private void Generate()
        {
            uint y;
            int kk;
            for (kk = 0; kk < N - M; kk++)
            {
                y = (_key[kk] & UPPER_MASK) | (_key[kk + 1] & LOWER_MASK);
                _key[kk] = _key[kk + M] ^ (y >> 1) ^ (unchecked(0u - (y & 1)) & MATRIX_A);
            }
            for (; kk < N - 1; kk++)
            {
                y = (_key[kk] & UPPER_MASK) | (_key[kk + 1] & LOWER_MASK);
                _key[kk] = _key[kk + (M - N)] ^ (y >> 1) ^ (unchecked(0u - (y & 1)) & MATRIX_A);
            }
            y = (_key[N - 1] & UPPER_MASK) | (_key[0] & LOWER_MASK);
            _key[N - 1] = _key[M - 1] ^ (y >> 1) ^ (unchecked(0u - (y & 1)) & MATRIX_A);
            _pos = 0;
        }

        /// <inheritdoc/>
        /// <remarks>NumPy's <c>mt19937_next</c>: twist when the key is exhausted, then temper the next word.</remarks>
        internal override uint NextUInt32()
        {
            if (_pos == N)
                Generate();

            uint y = _key[_pos++];
            y ^= y >> 11;
            y ^= (y << 7) & TEMPERING_MASK_B;
            y ^= (y << 15) & TEMPERING_MASK_C;
            y ^= y >> 18;
            return y;
        }

        /// <inheritdoc/>
        /// <remarks>NumPy's <c>mt19937_next64</c>: <c>(next32 &lt;&lt; 32) | next32</c> — the HIGH half is drawn first.</remarks>
        internal override ulong NextUInt64()
        {
            ulong hi = NextUInt32();
            return (hi << 32) | NextUInt32();
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's <c>mt19937_next_double</c>: 27 bits of one word and 26 of the next — NOT the 53 top bits of a
        ///     64-bit word the base formula uses (the value is identical to RandomState's <c>random_sample</c>).
        /// </remarks>
        internal override double NextDouble()
        {
            int a = (int)(NextUInt32() >> 5);
            int b = (int)(NextUInt32() >> 6);
            return (a * 67108864.0 + b) / 9007199254740992.0;
        }

        /// <inheritdoc/>
        /// <remarks>NumPy's <c>mt19937_raw</c>: the 32-bit word, widened.</remarks>
        internal override ulong NextRaw() => NextUInt32();

        // ---- legacy seeding (NumPy MT19937._legacy_seeding, used by RandomState.seed) ----

        /// <summary>
        ///     Re-seed from fresh OS entropy the way NumPy's <c>_legacy_seeding(None)</c> does: a new
        ///     <see cref="SeedSequence"/> fills <c>key[1:]</c> and <c>key[0] = 0x80000000</c>, the position is left
        ///     UNCHANGED (a NumPy quirk, observable through <see cref="state"/>), and <see cref="BitGenerator.seed_seq"/>
        ///     becomes null.
        /// </summary>
        public void _legacy_seeding()
        {
            lock (@lock)
            {
                SeedFromSequence(new SeedSequence(), setPos: false);
                _seedSeq = null;
            }
        }

        /// <summary>
        ///     Seed the generator with the classic 32-bit integer initializer (NumPy's <c>_legacy_seeding(int)</c> —
        ///     what <c>RandomState(seed)</c> and <c>np.random.seed(seed)</c> do). Discards <see cref="BitGenerator.seed_seq"/>.
        /// </summary>
        /// <param name="seed">The seed, in <c>[0, 2**32 - 1]</c>.</param>
        /// <exception cref="ValueError"><paramref name="seed"/> is outside <c>[0, 2**32 - 1]</c> (<c>Seed must be between 0 and 2**32 - 1</c>).</exception>
        public void _legacy_seeding(long seed)
        {
            if (seed < 0 || seed > uint.MaxValue)
                throw new ValueError("Seed must be between 0 and 2**32 - 1");
            lock (@lock)
            {
                SeedInteger((uint)seed);
                _seedSeq = null;
            }
        }

        /// <summary>
        ///     Seed the generator with <c>init_by_array</c> (NumPy's <c>_legacy_seeding(array)</c>). Discards
        ///     <see cref="BitGenerator.seed_seq"/>.
        /// </summary>
        /// <param name="seed">The key words (non-empty).</param>
        /// <exception cref="ValueError"><paramref name="seed"/> is null or empty (<c>Seed must be non-empty</c>).</exception>
        public void _legacy_seeding(uint[] seed)
        {
            if (seed is null || seed.Length == 0)
                throw new ValueError("Seed must be non-empty");
            lock (@lock)
            {
                InitByArray(seed);
                _seedSeq = null;
            }
        }

        /// <summary>
        ///     Seed the generator with <c>init_by_array</c> from signed words, each validated to <c>[0, 2**32 - 1]</c>
        ///     (NumPy's <c>_legacy_seeding(list)</c>). Discards <see cref="BitGenerator.seed_seq"/>.
        /// </summary>
        /// <param name="seed">The key words (non-empty, each in range).</param>
        /// <exception cref="ValueError">Empty (<c>Seed must be non-empty</c>) or an element out of range (<c>Seed must be between 0 and 2**32 - 1</c>).</exception>
        public void _legacy_seeding(long[] seed) => _legacy_seeding(ValidateLegacyArray(seed));

        /// <summary>
        ///     Seed the generator with <c>init_by_array</c> from signed words, each validated to be non-negative
        ///     (NumPy's <c>_legacy_seeding(list)</c>). Discards <see cref="BitGenerator.seed_seq"/>.
        /// </summary>
        /// <param name="seed">The key words (non-empty, each non-negative).</param>
        /// <exception cref="ValueError">Empty (<c>Seed must be non-empty</c>) or a negative element (<c>Seed must be between 0 and 2**32 - 1</c>).</exception>
        public void _legacy_seeding(int[] seed)
        {
            if (seed is null || seed.Length == 0)
                throw new ValueError("Seed must be non-empty");
            var words = new long[seed.Length];
            for (int i = 0; i < seed.Length; i++)
                words[i] = seed[i];
            _legacy_seeding(words);
        }

        /// <summary>
        ///     NumPy's array-seed validation: non-empty, and every element in <c>[0, 2**32 - 1]</c> (checked after the
        ///     emptiness test, as NumPy's <c>obj.size == 0</c> precedes its range check).
        /// </summary>
        /// <param name="seed">The signed words.</param>
        /// <returns>The words as uint32.</returns>
        /// <exception cref="ValueError">Empty or out of range.</exception>
        internal static uint[] ValidateLegacyArray(long[] seed)
        {
            if (seed is null || seed.Length == 0)
                throw new ValueError("Seed must be non-empty");
            var words = new uint[seed.Length];
            for (int i = 0; i < seed.Length; i++)
            {
                if (seed[i] < 0 || seed[i] > uint.MaxValue)
                    throw new ValueError("Seed must be between 0 and 2**32 - 1");
                words[i] = (uint)seed[i];
            }
            return words;
        }

        // ---- jumped (NumPy MT19937.jumped / mt19937_jump) ----

        /// <summary>
        ///     Returns a new bit generator whose state is this one's jumped ahead as if <c>2**128 * jumps</c> values
        ///     had been drawn (NumPy's <c>MT19937.jumped</c>).
        /// </summary>
        /// <param name="jumps">The number of 2**128-step jumps (zero or negative: no jump, a plain copy of the state).</param>
        /// <returns>A new <see cref="MT19937"/>; this instance is not modified.</returns>
        /// <remarks>
        ///     Built exactly as NumPy builds it — <c>self.__class__()</c> (so the copy carries a FRESH OS-entropy
        ///     <see cref="BitGenerator.seed_seq"/>, not this one's), then <c>state = self.state</c>, then
        ///     <c>mt19937_jump</c> applied <paramref name="jumps"/> times: Horner evaluation of NumPy's precomputed
        ///     2**128-step polynomial over the F2-linear recurrence. Chaining <c>jumped()</c> yields non-overlapping
        ///     streams for parallel work.
        /// </remarks>
        public MT19937 jumped(long jumps = 1)
        {
            var bg = new MT19937();
            lock (@lock)
            {
                Array.Copy(_key, bg._key, N);
                bg._pos = _pos;
            }
            for (long i = 0; i < jumps; i++)
                bg.JumpInPlace();
            return bg;
        }

        // ---- state get/set (numpy MT19937.state) ----

        /// <summary>
        ///     The MT19937 state — NumPy's <c>{'bit_generator': 'MT19937', 'state': {'key': uint32[624], 'pos': int}}</c>
        ///     with the nested <c>'state'</c> dict flattened.
        /// </summary>
        public sealed class State : BitGeneratorState
        {
            /// <summary>Creates an empty state (fill <see cref="key"/> and <see cref="pos"/> before assigning it).</summary>
            public State() { }

            /// <summary>Creates a state from a key and a position.</summary>
            /// <param name="key">The 624-word key (copied by reference; the setter copies its first 624 words).</param>
            /// <param name="pos">The index of the next word to temper, in <c>[0, 624]</c>.</param>
            public State(uint[] key, int pos)
            {
                this.key = key;
                this.pos = pos;
            }

            /// <inheritdoc/>
            public override string bit_generator => "MT19937";

            /// <summary>The 624-word key (NumPy <c>state['state']['key']</c>).</summary>
            public uint[] key { get; set; }

            /// <summary>The index of the next word to temper (NumPy <c>state['state']['pos']</c>); 624 forces a twist first.</summary>
            public int pos { get; set; }

            /// <summary>
            ///     Converts the legacy RandomState state (NumPy's <c>('MT19937', key, pos, has_gauss, gauss)</c> tuple),
            ///     which NumPy's <c>MT19937.state</c> setter also accepts.
            /// </summary>
            /// <param name="legacy">The legacy state.</param>
            /// <returns>The MT19937 part of it (the Gaussian cache belongs to RandomState, not the bit generator).</returns>
            /// <exception cref="ValueError">The tuple is not an MT19937 state (<c>state is not a legacy MT19937 state</c>).</exception>
            public static implicit operator State(NativeRandomState legacy)
            {
                if (legacy.Algorithm != "MT19937")
                    throw new ValueError("state is not a legacy MT19937 state");
                return new State(legacy.Key, legacy.Pos);
            }
        }

        /// <summary>
        ///     Gets or sets the current internal state (NumPy's <c>bit_generator.state</c> property), typed. A legacy
        ///     <see cref="NativeRandomState"/> converts implicitly, as NumPy's setter accepts the legacy tuple.
        /// </summary>
        /// <exception cref="TypeError">Setting null.</exception>
        /// <exception cref="IndexError">The key has fewer than 624 words (NumPy's <c>key[i]</c> indexing error).</exception>
        /// <exception cref="ValueError">The position is outside <c>[0, 624]</c> — a deliberate guard: NumPy stores any position and then
        /// reads past the key (undefined behaviour in C); NumSharp rejects it at the assignment.</exception>
        public new State state
        {
            get => (State)base.state;
            set => base.state = value;
        }

        /// <inheritdoc/>
        private protected override BitGeneratorState GetStateCore() => new State((uint[])_key.Clone(), _pos);

        /// <inheritdoc/>
        private protected override void SetStateCore(BitGeneratorState value)
        {
            if (value is not State s)
                throw new ValueError("state must be for a MT19937 PRNG");
            uint[] key = s.key ?? throw new TypeError("state['state']['key'] must be a sequence of 624 integers");
            if (key.Length < N)
                throw new IndexError($"index {key.Length} is out of bounds for axis 0 with size {key.Length}");
            if (s.pos < 0 || s.pos > N)
                throw new ValueError($"state['state']['pos'] must be in [0, {N}], got {s.pos}");
            Array.Copy(key, _key, N);
            _pos = s.pos;
        }

        /// <summary>Restores key and position directly (the legacy RandomState <c>set_state</c> path; the caller holds the lock).</summary>
        /// <param name="key">The 624-word key.</param>
        /// <param name="pos">The position in <c>[0, 624]</c>.</param>
        /// <exception cref="ArgumentException">The key is not 624 words long.</exception>
        /// <exception cref="ArgumentOutOfRangeException">The position is outside <c>[0, 624]</c>.</exception>
        internal void SetState(uint[] key, int pos)
        {
            if (key == null || key.Length != N)
                throw new ArgumentException($"Key array must be length {N}", nameof(key));
            if (pos < 0 || pos > N)
                throw new ArgumentOutOfRangeException(nameof(pos), $"Position must be in [0, {N}]");
            Array.Copy(key, _key, N);
            _pos = pos;
        }

        /// <summary>A copy of the 624-word key (the legacy RandomState <c>get_state</c> path).</summary>
        internal uint[] Key => (uint[])_key.Clone();

        /// <summary>The current position (the legacy RandomState <c>get_state</c> path).</summary>
        internal int Pos => _pos;

        /// <summary>Re-seeds with the legacy 32-bit initializer without the range check (callers have validated).</summary>
        /// <param name="seed">The 32-bit seed.</param>
        internal void Seed(uint seed)
        {
            SeedInteger(seed);
            _seedSeq = null;
        }

        /// <summary>Re-seeds with <c>init_by_array</c> without validation (callers have validated a non-empty key).</summary>
        /// <param name="initKey">The key words.</param>
        internal void SeedByArray(uint[] initKey)
        {
            InitByArray(initKey);
            _seedSeq = null;
        }
    }
}
