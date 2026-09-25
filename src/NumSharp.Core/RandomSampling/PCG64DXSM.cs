using System;
using System.Numerics;

namespace NumSharp
{
    /// <summary>
    ///     PCG64DXSM — the 128-bit PCG with the cheap 64-bit multiplier and the stronger "double xorshift multiply"
    ///     output (NumPy's <c>numpy.random.PCG64DXSM</c>), recommended by NumPy for massively parallel use.
    /// </summary>
    /// <remarks>
    ///     Port of NumPy 2.4.2's <c>_pcg64.pyx</c> <c>PCG64DXSM</c> + <c>pcg64.h</c> <c>pcg_cm_random_r</c>. It is seeded
    ///     exactly like <see cref="PCG64"/> (the same <c>pcg64_set_seed</c>, so the two share a state for the same seed) but
    ///     outputs DXSM of the PRE-iterated state and then steps with <c>state * 0xda942042e4dd58b5 + inc</c>; its
    ///     <see cref="advance"/> composes that cheap-multiplier step. The raw stream is byte-identical to NumPy's.
    /// </remarks>
    public sealed class PCG64DXSM : BitGenerator
    {
        private UInt128 _state;
        private UInt128 _inc;
        private int _hasUint32;
        private uint _uinteger;

        /// <summary>Constructs a PCG64DXSM seeded from fresh, unpredictable OS entropy (NumPy's <c>PCG64DXSM()</c>).</summary>
        public PCG64DXSM() : this(new SeedSequence()) { }

        /// <summary>Constructs a PCG64DXSM seeded from a non-negative integer through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative.</exception>
        public PCG64DXSM(long seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs a PCG64DXSM seeded from a non-negative integer through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed.</param>
        public PCG64DXSM(ulong seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs a PCG64DXSM seeded from an arbitrary-size non-negative integer through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative.</exception>
        public PCG64DXSM(BigInteger seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs a PCG64DXSM seeded from a sequence of non-negative integers through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed words (each must be non-negative).</param>
        /// <exception cref="ValueError">An element is negative.</exception>
        public PCG64DXSM(int[] seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs a PCG64DXSM seeded from a sequence of non-negative integers through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed words (each must be non-negative).</param>
        /// <exception cref="ValueError">An element is negative.</exception>
        public PCG64DXSM(long[] seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs a PCG64DXSM seeded from uint32 words through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed words.</param>
        public PCG64DXSM(uint[] seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs and seeds a PCG64DXSM from the given seed sequence (<c>generate_state(4, uint64)</c>).</summary>
        /// <param name="seedSeq">The seed sequence.</param>
        /// <exception cref="ArgumentNullException"><paramref name="seedSeq"/> is null.</exception>
        /// <exception cref="NotImplementedException"><paramref name="seedSeq"/> is a <see cref="SeedlessSeedSequence"/>.</exception>
        public PCG64DXSM(ISeedSequence seedSeq) : base(seedSeq ?? throw new ArgumentNullException(nameof(seedSeq)))
        {
            ulong[] val = SeedWords64(seedSeq, 4);
            Pcg128.Srandom(((UInt128)val[0] << 64) | val[1], ((UInt128)val[2] << 64) | val[3], out _state, out _inc);
        }

        /// <inheritdoc/>
        internal override string Name => "PCG64DXSM";

        /// <inheritdoc/>
        private protected override BitGenerator CreateFromSeed(ISeedSequence seed) => new PCG64DXSM(seed);

        /// <inheritdoc/>
        /// <remarks>NumPy's <c>pcg_cm_random_r</c>: DXSM output of the current state, then the cheap-multiplier step.</remarks>
        internal override ulong NextUInt64()
        {
            ulong ret = Pcg128.OutputDxsm(_state);
            _state = _state * (UInt128)Pcg128.CheapMultiplier + _inc;
            return ret;
        }

        /// <inheritdoc/>
        internal override uint NextUInt32()
        {
            // pcg64_cm_next32: serve the cached high half, else draw a 64-bit word and cache its high half.
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

        /// <summary>
        ///     Advance the underlying RNG as if <paramref name="delta"/> draws had occurred (NumPy's
        ///     <c>PCG64DXSM.advance</c> → <c>pcg64_cm_advance</c>), in O(log delta).
        /// </summary>
        /// <param name="delta">The number of 64-bit draws to skip — any integer, reduced mod 2**128 (negative steps back).</param>
        /// <returns>This bit generator (advanced in place).</returns>
        /// <remarks>Composes the CHEAP-multiplier step; resets the buffered 32-bit half. Holds the lock.</remarks>
        public PCG64DXSM advance(BigInteger delta)
        {
            UInt128 d = Pcg128.WrapDelta(delta);
            lock (@lock)
            {
                _state = Pcg128.AdvanceLcg(_state, d, (UInt128)Pcg128.CheapMultiplier, _inc);
                _hasUint32 = 0;
                _uinteger = 0;
            }
            return this;
        }

        /// <summary>
        ///     Returns a new bit generator with this one's state jumped as if <c>jumps * 210306068529402873165736369884012333109</c>
        ///     draws had been made (NumPy's <c>PCG64DXSM.jumped</c>).
        /// </summary>
        /// <param name="jumps">The number of jumps (any integer; 0 is a plain copy of the state).</param>
        /// <returns>A new <see cref="PCG64DXSM"/> carrying a fresh OS-entropy seed sequence (NumPy's <c>self.__class__()</c>).</returns>
        public PCG64DXSM jumped(long jumps = 1) => jumped((BigInteger)jumps);

        /// <summary>Arbitrary-size <see cref="jumped(long)"/>.</summary>
        /// <param name="jumps">The number of jumps (any integer).</param>
        /// <returns>A new, jumped <see cref="PCG64DXSM"/>.</returns>
        public PCG64DXSM jumped(BigInteger jumps)
        {
            var bg = new PCG64DXSM();
            bg.state = state;
            bg.advance(Pcg128.JumpStep * jumps);
            return bg;
        }

        /// <summary>
        ///     The PCG64DXSM state — NumPy's <c>{'bit_generator': 'PCG64DXSM', 'state': {'state': int, 'inc': int},
        ///     'has_uint32': int, 'uinteger': int}</c> with the nested <c>'state'</c> dict flattened.
        /// </summary>
        public sealed class State : BitGeneratorState
        {
            /// <summary>Creates an all-zero state (fill the members before assigning it).</summary>
            public State() { }

            /// <summary>Creates a state from its four members.</summary>
            /// <param name="state">The 128-bit LCG state.</param>
            /// <param name="inc">The 128-bit LCG increment.</param>
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
            public override string bit_generator => "PCG64DXSM";

            /// <summary>The 128-bit LCG state (NumPy <c>state['state']['state']</c>).</summary>
            public UInt128 state { get; set; }

            /// <summary>The 128-bit LCG increment (NumPy <c>state['state']['inc']</c>).</summary>
            public UInt128 inc { get; set; }

            /// <summary>Non-zero when a buffered 32-bit half is pending.</summary>
            public int has_uint32 { get; set; }

            /// <summary>The buffered 32-bit half.</summary>
            public uint uinteger { get; set; }
        }

        /// <summary>Gets or sets the current internal state (NumPy's <c>bit_generator.state</c>), typed.</summary>
        /// <exception cref="TypeError">Setting null.</exception>
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
                throw new ValueError("state must be for a PCG64DXSM RNG");
            _state = s.state;
            _inc = s.inc;
            _hasUint32 = s.has_uint32;
            _uinteger = s.uinteger;
        }
    }
}
