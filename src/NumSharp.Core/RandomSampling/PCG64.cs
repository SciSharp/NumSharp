using System;
using System.Numerics;

namespace NumSharp
{
    /// <summary>
    ///     PCG64 (XSL-RR 128/64) bit generator — the default BitGenerator behind
    ///     <c>np.random.default_rng</c>.
    /// </summary>
    /// <remarks>
    ///     Port of NumPy 2.4.2's PCG64 (<c>numpy/random/src/pcg64/pcg64.h</c>), a 128-bit LCG with
    ///     the XSL-RR 128→64 output permutation. The 128-bit state arithmetic is expressed with
    ///     .NET's native <see cref="UInt128"/>, so the raw stream is byte-for-byte identical to
    ///     NumPy's (verified against <c>bit_generator.random_raw</c>). Seeding goes through
    ///     <see cref="SeedSequence"/> exactly as NumPy does (<c>generate_state(4, uint64)</c>).
    /// </remarks>
    public sealed class PCG64 : BitGenerator
    {
        // PCG_DEFAULT_MULTIPLIER_128 = (2549297995355413924 << 64) + 4865540595714422341
        private static readonly UInt128 Multiplier =
            new UInt128(2549297995355413924UL, 4865540595714422341UL);

        private UInt128 _state;
        private UInt128 _inc;

        // 32-bit output buffering (NumPy pcg64_next32: caches the high half of a 64-bit draw).
        private bool _hasUint32;
        private uint _uinteger;

        private readonly SeedSequence _seedSeq;

        /// <summary>Constructs and seeds a PCG64 from the given seed sequence.</summary>
        public PCG64(SeedSequence seedSeq)
        {
            _seedSeq = seedSeq ?? throw new ArgumentNullException(nameof(seedSeq));
            ulong[] val = seedSeq.GenerateState64(4);
            UInt128 initState = ((UInt128)val[0] << 64) | val[1];
            UInt128 initSeq = ((UInt128)val[2] << 64) | val[3];
            Srandom(initState, initSeq);
        }

        /// <summary>Constructs and seeds a PCG64 from a single integer seed.</summary>
        public PCG64(long seed) : this(new SeedSequence(seed)) { }

        /// <summary>The seed sequence this generator was constructed from (NumPy <c>PCG64.seed_seq</c>).</summary>
        public SeedSequence seed_seq => _seedSeq;

        /// <inheritdoc/>
        internal override string Name => "PCG64";

        // pcg_setseq_128_srandom_r
        private void Srandom(UInt128 initState, UInt128 initSeq)
        {
            _state = UInt128.Zero;
            _inc = (initSeq << 1) | UInt128.One;
            Step();
            _state += initState;
            Step();
        }

        // pcg_setseq_128_step_r
        private void Step() => _state = _state * Multiplier + _inc;

        private static ulong Rotr(ulong value, int rot)
            => (value >> rot) | (value << ((-rot) & 63));

        // pcg_output_xsl_rr_128_64
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
            if (_hasUint32)
            {
                _hasUint32 = false;
                return _uinteger;
            }
            ulong next = NextUInt64();
            _hasUint32 = true;
            _uinteger = (uint)(next >> 32);
            return (uint)next;
        }

        // ---- state get/set (numpy pcg64_get_state / pcg64_set_state) ----

        /// <summary>
        ///     Snapshot of the PCG64 internal state (the typed stand-in for NumPy's
        ///     <c>bit_generator.state</c> dict; field names match its keys).
        /// </summary>
        public readonly struct Pcg64StateData
        {
            /// <summary>The 128-bit LCG state.</summary>
            public UInt128 state { get; }
            /// <summary>The 128-bit LCG increment (stream selector).</summary>
            public UInt128 inc { get; }
            /// <summary>Whether a cached 32-bit word is pending.</summary>
            public bool has_uint32 { get; }
            /// <summary>The cached 32-bit word (valid only when <see cref="has_uint32"/>).</summary>
            public uint uinteger { get; }

            internal Pcg64StateData(UInt128 state, UInt128 inc, bool has_uint32, uint uinteger)
            {
                this.state = state;
                this.inc = inc;
                this.has_uint32 = has_uint32;
                this.uinteger = uinteger;
            }
        }

        /// <summary>
        ///     Gets or sets the current internal state (NumPy's <c>bit_generator.state</c> property).
        /// </summary>
        public Pcg64StateData state
        {
            get => new Pcg64StateData(_state, _inc, _hasUint32, _uinteger);
            set
            {
                _state = value.state;
                _inc = value.inc;
                _hasUint32 = value.has_uint32;
                _uinteger = value.uinteger;
            }
        }
    }
}
