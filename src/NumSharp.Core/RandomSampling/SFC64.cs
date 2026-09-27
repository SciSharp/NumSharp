using System;
using System.Numerics;

namespace NumSharp
{
    /// <summary>
    ///     Chris Doty-Humphrey's Small Fast Chaotic PRNG (NumPy's <c>numpy.random.SFC64</c>): four 64-bit words of state,
    ///     one of them a counter that guarantees a minimum period of 2**64 per seed.
    /// </summary>
    /// <remarks>
    ///     Port of NumPy 2.4.2's <c>_sfc64.pyx</c> + <c>src/sfc64/sfc64.{c,h}</c>; the raw stream is byte-identical to NumPy's.
    ///     Seeding: <c>generate_state(3, uint64)</c> fills <c>s[0..2]</c>, the counter <c>s[3]</c> starts at 1, and twelve
    ///     outputs are discarded (<c>sfc64_set_seed</c>). Like NumPy, SFC64 has no <c>advance</c> or <c>jumped</c> —
    ///     use <see cref="BitGenerator.spawn"/> (or distinct seeds) for parallel streams.
    /// </remarks>
    public sealed class SFC64 : BitGenerator
    {
        private ulong _s0, _s1, _s2, _s3;
        private int _hasUint32;
        private uint _uinteger;

        /// <summary>Constructs an SFC64 seeded from fresh, unpredictable OS entropy (NumPy's <c>SFC64()</c>).</summary>
        public SFC64() : this(new SeedSequence()) { }

        /// <summary>Constructs an SFC64 seeded from a non-negative integer through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative.</exception>
        public SFC64(long seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs an SFC64 seeded from a non-negative integer through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed.</param>
        public SFC64(ulong seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs an SFC64 seeded from an arbitrary-size non-negative integer through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative.</exception>
        public SFC64(BigInteger seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs an SFC64 seeded from a sequence of non-negative integers through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed words (each must be non-negative).</param>
        /// <exception cref="ValueError">An element is negative.</exception>
        public SFC64(int[] seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs an SFC64 seeded from a sequence of non-negative integers through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed words (each must be non-negative).</param>
        /// <exception cref="ValueError">An element is negative.</exception>
        public SFC64(long[] seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs an SFC64 seeded from uint32 words through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed words.</param>
        public SFC64(uint[] seed) : this(new SeedSequence(seed)) { }

        /// <summary>Constructs and seeds an SFC64 from the given seed sequence (<c>generate_state(3, uint64)</c> + <c>sfc64_set_seed</c>).</summary>
        /// <param name="seed">The seed sequence — normally a <see cref="SeedSequence"/>; any <see cref="ISeedSequence"/> works;
        ///     null is NumPy's <c>seed=None</c>: a fresh OS-entropy <see cref="SeedSequence"/> (the parameter name is NumPy's, so <c>seed: sequence</c> ports verbatim).</param>
        /// <exception cref="NotImplementedException"><paramref name="seed"/> is a <see cref="SeedlessSeedSequence"/>.</exception>
        public SFC64(ISeedSequence seed) : base(seed ?? new SeedSequence())
        {
            ulong[] val = SeedWords64(seed_seq, 3);
            _s0 = val[0];
            _s1 = val[1];
            _s2 = val[2];
            _s3 = 1;
            // NumPy "conservatively sticks with the original formula": twelve warm-up outputs mix the counter in
            // before the first observable draw. They are part of the stream.
            for (int i = 0; i < 12; i++)
                Next();
        }

        /// <inheritdoc/>
        internal override string Name => "SFC64";

        /// <inheritdoc/>
        private protected override BitGenerator CreateFromSeed(ISeedSequence seed) => new SFC64(seed);

        /// <summary>NumPy's <c>sfc64_next</c>: the output is <c>a + b + counter++</c>, then the chaotic state update.</summary>
        /// <returns>The next 64-bit output.</returns>
        private ulong Next()
        {
            ulong tmp = _s0 + _s1 + _s3++;
            _s0 = _s1 ^ (_s1 >> 11);
            _s1 = _s2 + (_s2 << 3);
            _s2 = BitOperations.RotateLeft(_s2, 24) + tmp;
            return tmp;
        }

        /// <inheritdoc/>
        internal override ulong NextUInt64() => Next();

        /// <inheritdoc/>
        /// <remarks>The four state words live in locals for the whole run (see <see cref="BitGenerator.FillUInt64"/>).</remarks>
        internal override unsafe void FillUInt64(ulong* dst, long n)
        {
            ulong a = _s0, b = _s1, c = _s2, counter = _s3;
            for (long i = 0; i < n; i++)
            {
                ulong tmp = a + b + counter++;
                a = b ^ (b >> 11);
                b = c + (c << 3);
                c = BitOperations.RotateLeft(c, 24) + tmp;
                dst[i] = tmp;
            }
            _s0 = a;
            _s1 = b;
            _s2 = c;
            _s3 = counter;
        }

        /// <inheritdoc/>
        internal override unsafe void FillUInt32(uint* dst, long n) => FillUInt32From64(dst, n, ref _hasUint32, ref _uinteger);

        /// <inheritdoc/>
        internal override unsafe void FillDouble(double* dst, long n) => FillDoubleFrom64(dst, n);

        /// <inheritdoc/>
        internal override unsafe void FillFloat(float* dst, long n) => FillFloatFrom64(dst, n, ref _hasUint32, ref _uinteger);

        /// <inheritdoc/>
        internal override unsafe void FillRaw(ulong* dst, long n) => FillUInt64(dst, n);

        /// <inheritdoc/>
        internal override uint NextUInt32()
        {
            // sfc64_next32: serve the cached high half, else draw a 64-bit word and cache its high half.
            if (_hasUint32 != 0)
            {
                _hasUint32 = 0;
                return _uinteger;
            }
            ulong next = Next();
            _hasUint32 = 1;
            _uinteger = (uint)(next >> 32);
            return (uint)next;
        }

        /// <summary>
        ///     The SFC64 state — NumPy's <c>{'bit_generator': 'SFC64', 'state': {'state': uint64[4]}, 'has_uint32': int,
        ///     'uinteger': int}</c> with the nested <c>'state'</c> dict flattened.
        /// </summary>
        public sealed class State : BitGeneratorState
        {
            /// <summary>Creates an empty state (fill <see cref="state"/> before assigning it).</summary>
            public State() { }

            /// <summary>Creates a state from its members.</summary>
            /// <param name="state">The four state words <c>a, b, c, counter</c> (a single word broadcasts to all four, as NumPy's <c>state_vec[:] = …</c> does).</param>
            /// <param name="has_uint32">Non-zero when a buffered 32-bit half is pending.</param>
            /// <param name="uinteger">The buffered 32-bit half.</param>
            public State(ulong[] state, int has_uint32 = 0, uint uinteger = 0)
            {
                this.state = state;
                this.has_uint32 = has_uint32;
                this.uinteger = uinteger;
            }

            /// <inheritdoc/>
            public override string bit_generator => "SFC64";

            /// <summary>The four state words (NumPy <c>state['state']['state']</c>); the last is the counter.</summary>
            public ulong[] state { get; set; }

            /// <summary>Non-zero when a buffered 32-bit half is pending.</summary>
            public int has_uint32 { get; set; }

            /// <summary>The buffered 32-bit half.</summary>
            public uint uinteger { get; set; }
        }

        /// <summary>Gets or sets the current internal state (NumPy's <c>bit_generator.state</c>), typed.</summary>
        /// <exception cref="TypeError">Setting null, or a state whose word array is null.</exception>
        /// <exception cref="ValueError">The word array neither has 4 elements nor 1 (NumPy's broadcast error).</exception>
        public new State state
        {
            get => (State)base.state;
            set => base.state = value;
        }

        /// <inheritdoc/>
        private protected override BitGeneratorState GetStateCore() => new State(new[] { _s0, _s1, _s2, _s3 }, _hasUint32, _uinteger);

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's setter assigns <c>state_vec[:] = value['state']['state']</c>, a BROADCAST assignment into four
        ///     words: one word fills all four, four copy across, anything else is the broadcast ValueError. The
        ///     misspelt error text (the literal <c>{self.__class__.__name__}</c> — NumPy's message lacks its f-prefix) is
        ///     reproduced verbatim.
        /// </remarks>
        private protected override void SetStateCore(BitGeneratorState value)
        {
            if (value is not State s)
                throw new ValueError("state must be for a {self.__class__.__name__} RNG");
            ulong[] words = s.state ?? throw new TypeError("state['state']['state'] must be a sequence of 4 integers");
            if (words.Length != 4 && words.Length != 1)
                throw new ValueError($"could not broadcast input array from shape ({words.Length},) into shape (4,)");
            if (words.Length == 1)
                _s0 = _s1 = _s2 = _s3 = words[0];
            else
            {
                _s0 = words[0];
                _s1 = words[1];
                _s2 = words[2];
                _s3 = words[3];
            }
            _hasUint32 = s.has_uint32;
            _uinteger = s.uinteger;
        }
    }
}
