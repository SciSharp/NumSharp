using System;
using System.Numerics;

namespace NumSharp
{
    /// <summary>
    ///     PCG64 (XSL-RR 128/64) bit generator — the default BitGenerator behind
    ///     <c>np.random.default_rng</c>.
    /// </summary>
    /// <remarks>
    ///     Port of NumPy 2.4.2's PCG64 (<c>numpy/random/src/pcg64/pcg64.h</c>, <c>_pcg64.pyx</c>), a 128-bit LCG
    ///     with the XSL-RR 128→64 output permutation. The 128-bit state arithmetic is expressed with .NET's native
    ///     <see cref="UInt128"/>, so the raw stream is byte-for-byte identical to NumPy's (verified against
    ///     <c>random_raw</c>). Seeding goes through <see cref="SeedSequence"/> exactly as NumPy does
    ///     (<c>generate_state(4, uint64)</c> → initial state and increment). <see cref="advance"/> and
    ///     <see cref="jumped(long)"/> move the stream in O(log n).
    /// </remarks>
    public sealed class PCG64 : BitGenerator
    {
        private UInt128 _state;
        private UInt128 _inc;

        // 32-bit output buffering (NumPy pcg64_next32: caches the high half of a 64-bit draw). Kept as the
        // int NumPy stores so a state round-trips whatever value was assigned.
        private int _hasUint32;
        private uint _uinteger;

        /// <summary>Constructs a PCG64 seeded from fresh, unpredictable OS entropy (NumPy's <c>PCG64()</c> / <c>seed=None</c>).</summary>
        public PCG64() : this(new SeedSequence()) { }

        /// <summary>Constructs a PCG64 seeded from a non-negative integer through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative (<c>expected non-negative integer</c>).</exception>
        public PCG64(long seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs a PCG64 seeded from a non-negative integer through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed.</param>
        public PCG64(ulong seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs a PCG64 seeded from an arbitrary-size non-negative integer through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative.</exception>
        public PCG64(BigInteger seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs a PCG64 seeded from a sequence of non-negative integers through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed words (each must be non-negative).</param>
        /// <exception cref="ValueError">An element is negative.</exception>
        public PCG64(int[] seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs a PCG64 seeded from a sequence of non-negative integers through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed words (each must be non-negative; values of 2**32 and above span several words).</param>
        /// <exception cref="ValueError">An element is negative.</exception>
        public PCG64(long[] seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs a PCG64 seeded from uint32 words through <see cref="SeedSequence"/> (NumPy's uint32-array pass-through).</summary>
        /// <param name="seed">The seed words.</param>
        public PCG64(uint[] seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs and seeds a PCG64 from the given seed sequence.</summary>
        /// <param name="seedSeq">The seed sequence; <c>generate_state(4, uint64)</c> supplies the state and the stream.</param>
        /// <exception cref="ArgumentNullException"><paramref name="seedSeq"/> is null.</exception>
        /// <exception cref="NotImplementedException"><paramref name="seedSeq"/> is a <see cref="SeedlessSeedSequence"/>.</exception>
        public PCG64(ISeedSequence seedSeq) : base(seedSeq ?? throw new ArgumentNullException(nameof(seedSeq)))
        {
            ulong[] val = SeedWords64(seedSeq, 4);
            Pcg128.Srandom(((UInt128)val[0] << 64) | val[1], ((UInt128)val[2] << 64) | val[3], out _state, out _inc);
        }

        /// <inheritdoc/>
        internal override string Name => "PCG64";

        /// <inheritdoc/>
        private protected override BitGenerator CreateFromSeed(ISeedSequence seed) => new PCG64(seed);

        /// <inheritdoc/>
        internal override ulong NextUInt64()
        {
            // pcg_setseq_128_xsl_rr_64_random_r: step, then output the new state.
            _state = _state * Pcg128.DefaultMultiplier + _inc;
            return Pcg128.OutputXslRr(_state);
        }

        /// <inheritdoc/>
        internal override uint NextUInt32()
        {
            // pcg64_next32: serve the cached high half, else draw a 64-bit word and cache its high half.
            if (_hasUint32 != 0)
            {
                _hasUint32 = 0;
                return _uinteger;
            }
            ulong next = NextUInt64();
            _hasUint32 = 1;
            _uinteger = (uint)(next >> 32);
            return (uint)next;
        }

        // ---- advance / jumped (numpy pcg64_advance, PCG64.jumped) ----

        /// <summary>
        ///     Advance the underlying RNG as if <paramref name="delta"/> draws had occurred (NumPy's <c>PCG64.advance</c>),
        ///     in O(log delta).
        /// </summary>
        /// <param name="delta">The number of 64-bit draws to skip — any integer, reduced mod 2**128 as NumPy's
        /// <c>wrap_int</c> does, so a negative delta steps backwards (<c>advance(-1)</c> undoes one draw).</param>
        /// <returns>This bit generator (advanced in place).</returns>
        /// <remarks>
        ///     Resets the buffered 32-bit half, as NumPy does "to ensure exact reproducibility" — a distribution's draw
        ///     count is not a count of raw words (rejection sampling, 32-bit halves), so only raw words are skipped.
        ///     Holds the lock.
        /// </remarks>
        public PCG64 advance(BigInteger delta)
        {
            UInt128 d = Pcg128.WrapDelta(delta);
            lock (@lock)
            {
                _state = Pcg128.AdvanceLcg(_state, d, Pcg128.DefaultMultiplier, _inc);
                _hasUint32 = 0;
                _uinteger = 0;
            }
            return this;
        }

        /// <summary>
        ///     Returns a new bit generator with this one's state jumped as if <c>jumps * 210306068529402873165736369884012333109</c>
        ///     draws had been made (NumPy's <c>PCG64.jumped</c>; the step is (phi - 1) * 2**128).
        /// </summary>
        /// <param name="jumps">The number of jumps (any integer; 0 is a plain copy of the state).</param>
        /// <returns>A new <see cref="PCG64"/>; this instance is not modified.</returns>
        /// <remarks>
        ///     Built like NumPy's <c>self.__class__()</c> — the copy carries a FRESH OS-entropy seed sequence, not this
        ///     one's — then <c>state = self.state</c> and <c>advance(step * jumps)</c> (which also clears the buffered half).
        /// </remarks>
        public PCG64 jumped(long jumps = 1) => jumped((BigInteger)jumps);

        /// <summary>Arbitrary-size <see cref="jumped(long)"/>.</summary>
        /// <param name="jumps">The number of jumps (any integer).</param>
        /// <returns>A new, jumped <see cref="PCG64"/>.</returns>
        public PCG64 jumped(BigInteger jumps)
        {
            var bg = new PCG64();
            bg.state = state;
            bg.advance(Pcg128.JumpStep * jumps);
            return bg;
        }

        // ---- state get/set (numpy pcg64_get_state / pcg64_set_state) ----

        /// <summary>
        ///     The PCG64 state — NumPy's <c>{'bit_generator': 'PCG64', 'state': {'state': int, 'inc': int},
        ///     'has_uint32': int, 'uinteger': int}</c> with the nested <c>'state'</c> dict flattened.
        /// </summary>
        public sealed class State : BitGeneratorState
        {
            /// <summary>Creates an all-zero state (fill the members before assigning it).</summary>
            public State() { }

            /// <summary>Creates a state from its four members.</summary>
            /// <param name="state">The 128-bit LCG state.</param>
            /// <param name="inc">The 128-bit LCG increment (stream selector; odd in any seeded state).</param>
            /// <param name="has_uint32">Non-zero when a buffered 32-bit half is pending.</param>
            /// <param name="uinteger">The buffered 32-bit half.</param>
            public State(UInt128 state, UInt128 inc, int has_uint32 = 0, uint uinteger = 0)
            {
                this.state = state;
                this.inc = inc;
                this.has_uint32 = has_uint32;
                this.uinteger = uinteger;
            }

            /// <inheritdoc/>
            public override string bit_generator => "PCG64";

            /// <summary>The 128-bit LCG state (NumPy <c>state['state']['state']</c>).</summary>
            public UInt128 state { get; set; }

            /// <summary>The 128-bit LCG increment (NumPy <c>state['state']['inc']</c>).</summary>
            public UInt128 inc { get; set; }

            /// <summary>Non-zero when a buffered 32-bit half is pending (NumPy stores the int as given).</summary>
            public int has_uint32 { get; set; }

            /// <summary>The buffered 32-bit half (meaningful only when <see cref="has_uint32"/> is non-zero).</summary>
            public uint uinteger { get; set; }
        }

        /// <summary>
        ///     Gets or sets the current internal state (NumPy's <c>bit_generator.state</c> property), typed.
        /// </summary>
        /// <exception cref="TypeError">Setting null.</exception>
        /// <remarks>The getter returns a fresh snapshot; the setter copies the values in. Both hold the lock.</remarks>
        public new State state
        {
            get => (State)base.state;
            set => base.state = value;
        }

        /// <inheritdoc/>
        private protected override BitGeneratorState GetStateCore() => new State(_state, _inc, _hasUint32, _uinteger);

        /// <inheritdoc/>
        private protected override void SetStateCore(BitGeneratorState value)
        {
            if (value is not State s)
                throw new ValueError("state must be for a PCG64 RNG");
            _state = s.state;
            _inc = s.inc;
            _hasUint32 = s.has_uint32;
            _uinteger = s.uinteger;
        }
    }
}
