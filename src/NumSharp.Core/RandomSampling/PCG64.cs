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
    ///     (<c>generate_state(4, uint64)</c> → initial state and increment).
    /// </remarks>
    public sealed class PCG64 : BitGenerator
    {
        // PCG_DEFAULT_MULTIPLIER_128 = (2549297995355413924 << 64) + 4865540595714422341
        private static readonly UInt128 Multiplier =
            new UInt128(2549297995355413924UL, 4865540595714422341UL);

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
        public PCG64(SeedSequence seedSeq) : base(seedSeq ?? throw new ArgumentNullException(nameof(seedSeq)))
        {
            ulong[] val = seedSeq.GenerateState64(4);
            UInt128 initState = ((UInt128)val[0] << 64) | val[1];
            UInt128 initSeq = ((UInt128)val[2] << 64) | val[3];
            Srandom(initState, initSeq);
        }

        /// <inheritdoc/>
        internal override string Name => "PCG64";

        /// <summary>NumPy's <c>pcg_setseq_128_srandom_r</c>: odd increment from the sequence, two steps around the state add.</summary>
        /// <param name="initState">The initial state word.</param>
        /// <param name="initSeq">The stream selector (the increment is <c>2·initSeq + 1</c>).</param>
        private void Srandom(UInt128 initState, UInt128 initSeq)
        {
            _state = UInt128.Zero;
            _inc = (initSeq << 1) | UInt128.One;
            Step();
            _state += initState;
            Step();
        }

        /// <summary>NumPy's <c>pcg_setseq_128_step_r</c>: one LCG step.</summary>
        private void Step() => _state = _state * Multiplier + _inc;

        /// <summary>Rotate right.</summary>
        /// <param name="value">The value.</param>
        /// <param name="rot">The rotation (0-63).</param>
        /// <returns>The rotated value.</returns>
        private static ulong Rotr(ulong value, int rot)
            => (value >> rot) | (value << ((-rot) & 63));

        /// <summary>NumPy's <c>pcg_output_xsl_rr_128_64</c>: xor-fold the halves, rotate by the top 6 bits.</summary>
        /// <param name="state">The 128-bit state.</param>
        /// <returns>The 64-bit output.</returns>
        private static ulong Output(UInt128 state)
        {
            ulong hi = (ulong)(state >> 64);
            ulong lo = (ulong)state;
            return Rotr(hi ^ lo, (int)(hi >> 58));
        }

        /// <inheritdoc/>
        internal override ulong NextUInt64()
        {
            // pcg_setseq_128_xsl_rr_64_random_r: step, then output the new state.
            Step();
            return Output(_state);
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
