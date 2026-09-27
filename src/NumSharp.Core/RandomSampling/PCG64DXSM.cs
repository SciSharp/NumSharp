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
        // The 128-bit state and increment as 64-bit limbs: the hot loops do their mod-2**128 arithmetic on ulongs
        // (Pcg128.MulAdd = Math.BigMul + two 64-bit products) because the JIT does not reliably inline UInt128's
        // operators into a loop body — measured: a four-way fill written over UInt128 ran 2.3x SLOWER than the plain
        // loop, every operator left as a call. UInt128 is still the currency of the state API and of advance().
        private ulong _stateHi, _stateLo;
        private ulong _incHi, _incLo;
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
        /// <param name="seed">The seed sequence — null is NumPy's <c>seed=None</c>: a fresh OS-entropy <see cref="SeedSequence"/> (the parameter name is NumPy's, so <c>seed: sequence</c> ports verbatim).</param>
        /// <exception cref="NotImplementedException"><paramref name="seed"/> is a <see cref="SeedlessSeedSequence"/>.</exception>
        public PCG64DXSM(ISeedSequence seed) : base(seed ?? new SeedSequence())
        {
            ulong[] val = SeedWords64(seed_seq, 4);
            Pcg128.Srandom(((UInt128)val[0] << 64) | val[1], ((UInt128)val[2] << 64) | val[3], out UInt128 state, out UInt128 inc);
            SetState128(state, inc);
        }

        /// <summary>The current state as a <see cref="UInt128"/>.</summary>
        private UInt128 State128 => new UInt128(_stateHi, _stateLo);

        /// <summary>The increment as a <see cref="UInt128"/>.</summary>
        private UInt128 Inc128 => new UInt128(_incHi, _incLo);

        /// <summary>Stores a 128-bit state and increment into the limb fields.</summary>
        /// <param name="state">The state.</param>
        /// <param name="inc">The increment.</param>
        private void SetState128(UInt128 state, UInt128 inc)
        {
            _stateHi = (ulong)(state >> 64);
            _stateLo = (ulong)state;
            _incHi = (ulong)(inc >> 64);
            _incLo = (ulong)inc;
        }

        /// <inheritdoc/>
        internal override string Name => "PCG64DXSM";

        /// <inheritdoc/>
        private protected override BitGenerator CreateFromSeed(ISeedSequence seed) => new PCG64DXSM(seed);

        /// <inheritdoc/>
        /// <remarks>NumPy's <c>pcg_cm_random_r</c>: DXSM output of the current state, then the cheap-multiplier step.</remarks>
        internal override ulong NextUInt64()
        {
            ulong ret = Pcg128.Dxsm(_stateHi, _stateLo);
            Pcg128.StepCheap(_stateHi, _stateLo, _incHi, _incLo, out _stateHi, out _stateLo);
            return ret;
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     The state and increment live in locals for the whole run (see <see cref="BitGenerator.FillUInt64"/>), two
        ///     words per step: the next state comes from the cheap step and the one after it straight from the current
        ///     state through the precomputed two-step multiplier (<see cref="Pcg128.CheapMul2Hi"/>), so the multiply chain
        ///     is half as long. Two-way is the sweet spot here — the output function costs two multiplies of its own, and
        ///     each further jump needs a FULL 128-bit multiplier, so wider unrolls are multiplier-throughput bound.
        ///     Arithmetic mod 2**128 is exact: the words are the sequential ones.
        /// </remarks>
        internal override unsafe void FillUInt64(ulong* dst, long n)
        {
            ulong sh = _stateHi, sl = _stateLo, ih = _incHi, il = _incLo;
            long i = 0;
            if (n >= 4)
            {
                // The two-step additive term inc * (1 + M) depends on this generator's increment.
                Pcg128.Mul(ih, il, Pcg128.CheapIncFactor2Hi, Pcg128.CheapIncFactor2Lo, out ulong c2h, out ulong c2l);
                ulong m2h = Pcg128.CheapMul2Hi, m2l = Pcg128.CheapMul2Lo;
                for (; i + 2 <= n; i += 2)
                {
                    Pcg128.StepCheap(sh, sl, ih, il, out ulong s1h, out ulong s1l);
                    dst[i] = Pcg128.Dxsm(sh, sl);
                    dst[i + 1] = Pcg128.Dxsm(s1h, s1l);
                    Pcg128.MulAdd(sh, sl, m2h, m2l, c2h, c2l, out sh, out sl);
                }
            }
            for (; i < n; i++)
            {
                dst[i] = Pcg128.Dxsm(sh, sl);
                Pcg128.StepCheap(sh, sl, ih, il, out sh, out sl);
            }
            _stateHi = sh;
            _stateLo = sl;
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
                SetState128(Pcg128.AdvanceLcg(State128, d, (UInt128)Pcg128.CheapMultiplier, Inc128), Inc128);
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
        private protected override BitGeneratorState GetStateCore() => new State(State128, Inc128, _hasUint32, _uinteger);

        /// <inheritdoc/>
        private protected override void SetStateCore(BitGeneratorState value)
        {
            if (value is not State s)
                throw new ValueError("state must be for a PCG64DXSM RNG");
            SetState128(s.state, s.inc);
            _hasUint32 = s.has_uint32;
            _uinteger = s.uinteger;
        }
    }
}
