using System;
using System.Numerics;

namespace NumSharp
{
    /// <summary>
    ///     The Philox 4x64-10 counter-based bit generator (NumPy's <c>numpy.random.Philox</c>): a 256-bit counter
    ///     encrypted under a 128-bit key by ten Feistel-like rounds, so any stream position — and any of 2**128 keyed
    ///     streams — is directly addressable.
    /// </summary>
    /// <remarks>
    ///     Port of NumPy 2.4.2's <c>_philox.pyx</c> + <c>src/philox/philox.{c,h}</c> (Salmon et al., "Parallel Random
    ///     Numbers: As Easy as 1, 2, 3", SC11); the raw stream is byte-identical to NumPy's. Each block increments the
    ///     counter (with carry) BEFORE encrypting it and yields four 64-bit words, served from a buffer; 32-bit draws
    ///     split a 64-bit word and cache its high half. Seeding follows NumPy exactly: a seed goes through
    ///     <see cref="SeedSequence"/> (<c>generate_state(2, uint64)</c> becomes the key, the counter starts at zero),
    ///     while an explicit <c>key</c> bypasses seeding altogether — then <see cref="BitGenerator.seed_seq"/> is null
    ///     and <see cref="BitGenerator.spawn"/> is unavailable, as in NumPy. <see cref="advance"/> adds to the counter
    ///     (mod 2**256); <see cref="jumped(long)"/> advances by <c>2**128</c> per jump.
    /// </remarks>
    public sealed class Philox : BitGenerator
    {
        // Philox 4x64 round multipliers and Weyl key-schedule increments (philox.h).
        private const ulong PHILOX_M4x64_0 = 0xD2E7470EE14C6C93UL;
        private const ulong PHILOX_M4x64_1 = 0xCA5A826395121157UL;
        private const ulong PHILOX_W64_0 = 0x9E3779B97F4A7C15UL;
        private const ulong PHILOX_W64_1 = 0xBB67AE8584CAA73BUL;

        /// <summary>NumPy's <c>philox4x64_rounds</c>.</summary>
        private const int Rounds = 10;

        /// <summary>NumPy's <c>PHILOX_BUFFER_SIZE</c>: one encrypted block is four words.</summary>
        private const int BufferSize = 4;

        /// <summary>2**256 - 1: NumPy's <c>wrap_int(delta, 256)</c> mask.</summary>
        private static readonly BigInteger Mask256 = (BigInteger.One << 256) - 1;

        /// <summary>The distance one <c>jumped()</c> step advances: 2**128 draws (NumPy's <c>jump_inplace</c>).</summary>
        private static readonly BigInteger JumpStep = BigInteger.One << 128;

        // The 256-bit counter (little-endian words) and the 128-bit key.
        private ulong _c0, _c1, _c2, _c3;
        private ulong _k0, _k1;

        // The current block and the index of the next unread word (BufferSize = exhausted).
        private readonly ulong[] _buffer = new ulong[BufferSize];
        private int _bufferPos;

        // 32-bit output buffering (the high half of a split 64-bit word).
        private int _hasUint32;
        private uint _uinteger;

        /// <summary>Constructs a Philox seeded from fresh, unpredictable OS entropy (NumPy's <c>Philox()</c>).</summary>
        public Philox() : this((ISeedSequence)new SeedSequence()) { }

        /// <summary>Constructs a Philox keyed from a non-negative integer seed through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative.</exception>
        public Philox(long seed) : this((ISeedSequence)new SeedSequence(seed)) { }

        /// <summary>Constructs a Philox keyed from a non-negative integer seed through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed.</param>
        public Philox(ulong seed) : this((ISeedSequence)new SeedSequence(seed)) { }

        /// <summary>Constructs a Philox keyed from an arbitrary-size non-negative integer seed through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative.</exception>
        public Philox(BigInteger seed) : this((ISeedSequence)new SeedSequence(seed)) { }

        /// <summary>Constructs a Philox keyed from a sequence of non-negative integers through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed words (each must be non-negative).</param>
        /// <exception cref="ValueError">An element is negative.</exception>
        public Philox(int[] seed) : this((ISeedSequence)new SeedSequence(seed)) { }

        /// <summary>Constructs a Philox keyed from a sequence of non-negative integers through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed words (each must be non-negative).</param>
        /// <exception cref="ValueError">An element is negative.</exception>
        public Philox(long[] seed) : this((ISeedSequence)new SeedSequence(seed)) { }

        /// <summary>Constructs a Philox keyed from uint32 words through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed words.</param>
        public Philox(uint[] seed) : this((ISeedSequence)new SeedSequence(seed)) { }

        /// <summary>Constructs a Philox keyed from the given seed sequence (<c>generate_state(2, uint64)</c>), counter zero.</summary>
        /// <param name="seedSeq">The seed sequence — normally a <see cref="SeedSequence"/>; any <see cref="ISeedSequence"/> works.</param>
        /// <exception cref="ArgumentNullException"><paramref name="seedSeq"/> is null.</exception>
        /// <exception cref="NotImplementedException"><paramref name="seedSeq"/> is a <see cref="SeedlessSeedSequence"/>.</exception>
        public Philox(ISeedSequence seedSeq) : this((object)(seedSeq ?? throw new ArgumentNullException(nameof(seedSeq))), null, null) { }

        /// <summary>
        ///     Constructs a Philox with NumPy's full signature, <c>Philox(seed=None, counter=None, key=None)</c>.
        /// </summary>
        /// <param name="seed">
        ///     null (with no <paramref name="key"/>: fresh OS entropy), an <see cref="ISeedSequence"/>, or anything
        ///     <see cref="SeedSequence"/> accepts as entropy; its <c>generate_state(2, uint64)</c> becomes the key.
        /// </param>
        /// <param name="counter">
        ///     The initial 256-bit counter (null = 0): an integer in <c>[0, 2**256)</c> (a real float truncates toward zero,
        ///     a string parses as a Python int), or a four-element array of words, lowest first, cast to uint64 as NumPy's
        ///     <c>astype</c> casts (a negative word wraps, a float word truncates).
        /// </param>
        /// <param name="key">
        ///     An explicit 128-bit key — an integer in <c>[0, 2**128)</c> or a two-element word array, same forms as
        ///     <paramref name="counter"/>. Mutually exclusive with <paramref name="seed"/>; the resulting generator has no
        ///     seed sequence (so it cannot <see cref="BitGenerator.spawn"/>).
        /// </param>
        /// <exception cref="ValueError">Both <paramref name="seed"/> and <paramref name="key"/> given (<c>seed and key cannot be both
        /// used</c>); a key/counter integer out of range (<c>key must be positive and less than 2**128.</c>); an array form of the
        /// wrong size (<c>counter must have 4 elements when using array form</c>); a NaN or unparsable string; an invalid seed.</exception>
        /// <exception cref="OverflowException">An infinite float key/counter; an integer element too large for uint64.</exception>
        /// <exception cref="TypeError">An invalid seed type, or a complex/non-numeric key or counter.</exception>
        /// <remarks>
        ///     Validation follows NumPy's order — the seed/key conflict, then the seed (inside <see cref="SeedSequence"/>),
        ///     then the key, then the counter — so the first bad argument reported matches NumPy's.
        /// </remarks>
        public Philox(object seed = null, object counter = null, object key = null) : base(ResolveSeed(seed, key))
        {
            ulong[] k = key is not null ? IntToArray(key, "key", 128) : SeedWords64(_seedSeq, 2);
            _k0 = k[0];
            _k1 = k[1];

            ulong[] c = IntToArray(counter ?? (object)0, "counter", 256);
            _c0 = c[0];
            _c1 = c[1];
            _c2 = c[2];
            _c3 = c[3];

            ResetStateVariables();
        }

        /// <summary>
        ///     The seed sequence the base constructor stores: none for an explicit key (NumPy discards the one it built —
        ///     "the seed sequence is invalid"), else the given sequence or a <see cref="SeedSequence"/> over the seed.
        /// </summary>
        /// <param name="seed">The seed argument.</param>
        /// <param name="key">The key argument.</param>
        /// <returns>The seed sequence, or null for a keyed generator.</returns>
        /// <exception cref="ValueError">Both are given, or the seed is invalid entropy.</exception>
        /// <exception cref="TypeError">The seed is not integer entropy.</exception>
        private static ISeedSequence ResolveSeed(object seed, object key)
        {
            if (seed is not null && key is not null)
                throw new ValueError("seed and key cannot be both used");
            // NumPy still builds SeedSequence(None) — an OS-entropy draw — before discarding it for a keyed
            // generator; skipping that draw is unobservable.
            if (key is not null)
                return null;
            return seed as ISeedSequence ?? new SeedSequence(seed);
        }

        /// <inheritdoc/>
        internal override string Name => "Philox";

        /// <inheritdoc/>
        private protected override BitGenerator CreateFromSeed(ISeedSequence seed) => new Philox(seed);

        /// <summary>NumPy's <c>_reset_state_variables</c>: no buffered 32-bit half, and an exhausted, zeroed block buffer.</summary>
        private void ResetStateVariables()
        {
            _hasUint32 = 0;
            _uinteger = 0;
            _bufferPos = BufferSize;
            Array.Clear(_buffer, 0, BufferSize);
        }

        /// <inheritdoc/>
        /// <remarks>NumPy's <c>philox_next</c>: serve the buffered block, else increment the counter and encrypt a new one.</remarks>
        internal override ulong NextUInt64()
        {
            if (_bufferPos < BufferSize)
                return _buffer[_bufferPos++];

            // The counter is incremented BEFORE encryption (a fresh generator's first block encrypts counter 1),
            // carrying across all four words; 2**256 - 1 wraps to 0.
            if (++_c0 == 0 && ++_c1 == 0 && ++_c2 == 0)
                ++_c3;
            EncryptCounter();
            _bufferPos = 1;
            return _buffer[0];
        }

        /// <inheritdoc/>
        internal override uint NextUInt32()
        {
            // philox_next32: serve the cached high half, else draw a 64-bit word and cache its high half.
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
        ///     NumPy's <c>philox4x64_R(10, ctr, key)</c>: ten rounds over the counter, the key bumped by the Weyl
        ///     constants between rounds; the result becomes the buffered block.
        /// </summary>
        private void EncryptCounter()
        {
            ulong x0 = _c0, x1 = _c1, x2 = _c2, x3 = _c3;
            ulong k0 = _k0, k1 = _k1;
            for (int r = 0; r < Rounds; r++)
            {
                if (r > 0)
                {
                    k0 += PHILOX_W64_0;
                    k1 += PHILOX_W64_1;
                }
                // _philox4x64round: two 64x64->128 multiplies; the high halves mix with the other words and the key.
                ulong hi0 = Math.BigMul(PHILOX_M4x64_0, x0, out ulong lo0);
                ulong hi1 = Math.BigMul(PHILOX_M4x64_1, x2, out ulong lo1);
                ulong n0 = hi1 ^ x1 ^ k0;
                ulong n2 = hi0 ^ x3 ^ k1;
                x0 = n0;
                x1 = lo1;
                x2 = n2;
                x3 = lo0;
            }
            _buffer[0] = x0;
            _buffer[1] = x1;
            _buffer[2] = x2;
            _buffer[3] = x3;
        }

        /// <summary>
        ///     Advance the underlying RNG as if <paramref name="delta"/> draws had occurred (NumPy's <c>Philox.advance</c>)
        ///     — strictly, as if <paramref name="delta"/> BLOCKS had been encrypted: the value is added to the 256-bit
        ///     counter, and each block holds four 64-bit draws.
        /// </summary>
        /// <param name="delta">The counter increment — any integer, reduced mod 2**256 as NumPy's <c>wrap_int</c> does
        /// (a negative delta steps back).</param>
        /// <returns>This bit generator (advanced in place).</returns>
        /// <remarks>
        ///     Resets the block buffer and the buffered 32-bit half (NumPy "to ensure exact reproducibility"), so a
        ///     partially consumed block is discarded. Holds the lock.
        /// </remarks>
        public Philox advance(BigInteger delta)
        {
            BigInteger wrapped = delta & Mask256;
            ulong s0 = (ulong)(wrapped & ulong.MaxValue);
            ulong s1 = (ulong)((wrapped >> 64) & ulong.MaxValue);
            ulong s2 = (ulong)((wrapped >> 128) & ulong.MaxValue);
            ulong s3 = (ulong)(wrapped >> 192);
            lock (@lock)
            {
                AddToCounter(s0, s1, s2, s3);
                ResetStateVariables();
            }
            return this;
        }

        /// <summary>
        ///     NumPy's <c>philox_advance</c>, transcribed literally: word-wise addition with a carry that is applied to the
        ///     next word before that word's own addition.
        /// </summary>
        /// <param name="s0">The step's lowest word.</param>
        /// <param name="s1">The step's second word.</param>
        /// <param name="s2">The step's third word.</param>
        /// <param name="s3">The step's highest word.</param>
        private void AddToCounter(ulong s0, ulong s1, ulong s2, ulong s3)
        {
            Span<ulong> ctr = stackalloc ulong[] { _c0, _c1, _c2, _c3 };
            ReadOnlySpan<ulong> step = stackalloc ulong[] { s0, s1, s2, s3 };
            int carry = 0;
            for (int i = 0; i < 4; i++)
            {
                if (carry == 1)
                {
                    ctr[i]++;
                    carry = ctr[i] == 0 ? 1 : 0;
                }
                ulong vOrig = ctr[i];
                ctr[i] += step[i];
                if (ctr[i] < vOrig && carry == 0)
                    carry = 1;
            }
            _c0 = ctr[0];
            _c1 = ctr[1];
            _c2 = ctr[2];
            _c3 = ctr[3];
        }

        /// <summary>
        ///     Returns a new bit generator with this one's state jumped as if <c>2**128 * jumps</c> draws had been made
        ///     (NumPy's <c>Philox.jumped</c>): the counter's upper half advances by <paramref name="jumps"/>.
        /// </summary>
        /// <param name="jumps">The number of jumps (any integer; 0 is a copy of the counter and key).</param>
        /// <returns>
        ///     A new <see cref="Philox"/> — built like NumPy's <c>self.__class__()</c>, so it carries a FRESH OS-entropy
        ///     seed sequence — with this one's state, then advanced (which discards the copied block buffer).
        /// </returns>
        public Philox jumped(long jumps = 1) => jumped((BigInteger)jumps);

        /// <summary>Arbitrary-size <see cref="jumped(long)"/>.</summary>
        /// <param name="jumps">The number of jumps (any integer).</param>
        /// <returns>A new, jumped <see cref="Philox"/>.</returns>
        public Philox jumped(BigInteger jumps)
        {
            var bg = new Philox();
            bg.state = state;
            bg.advance(jumps * JumpStep);
            return bg;
        }

        // ---- key / counter coercion (numpy _common.int_to_array) ----

        /// <summary>
        ///     NumPy's <c>int_to_array(value, name, bits, 64)</c>: a scalar becomes <c>bits / 64</c> little-endian words
        ///     after Python's <c>int()</c> and a <c>[0, 2**bits)</c> range check; an array form is cast to uint64 (NumPy's
        ///     unsafe <c>astype</c>) and must have exactly <c>bits / 64</c> elements.
        /// </summary>
        /// <param name="value">The key or counter as given.</param>
        /// <param name="name">The argument name for messages (<c>key</c> / <c>counter</c>).</param>
        /// <param name="bits">128 for the key, 256 for the counter.</param>
        /// <returns>The words, lowest first.</returns>
        /// <exception cref="ValueError">Out of range, wrong element count, NaN, or an unparsable string.</exception>
        /// <exception cref="OverflowException">An infinity, or an integer element beyond uint64.</exception>
        /// <exception cref="TypeError">A complex or non-numeric value.</exception>
        internal static ulong[] IntToArray(object value, string name, int bits)
        {
            int len = bits / 64;

            // A heterogeneous C# list (object[] / List<T>) stands for a Python list: infer the array NumPy's
            // np.asarray would build from it before casting.
            if (value is System.Collections.IList list && (value is object[] || value is not Array))
                value = InferListArray(list, name, len);
            if (value is BigInteger[] big)
                return CheckArrayForm(BigIntegersToWords(big), name, len, new long[] { big.Length });

            NDArray owned = null;
            try
            {
                NDArray nd = value as NDArray;
                if (nd is null && value is Array arr)
                    nd = owned = np.array(arr);

                if (nd is not null && nd.ndim != 0)
                {
                    // `value.astype(np.uint64)` BEFORE the shape check, exactly as NumPy orders them.
                    using NDArray u = nd.astype(DType.UInt64);
                    var words = new ulong[u.size];
                    for (long i = 0; i < words.LongLength; i++)
                        words[i] = u.GetAtIndex<ulong>(i);
                    var shape = new long[u.ndim];
                    for (int d = 0; d < shape.Length; d++)
                        shape[d] = u.shape[d];
                    return CheckArrayForm(words, name, len, shape);
                }

                BigInteger v = PyInt.From(nd is not null ? nd.GetAtIndex(0) : value);
                if (v.Sign < 0 || v >= BigInteger.One << bits)
                    throw new ValueError($"{name} must be positive and less than 2**{bits}.");
                var result = new ulong[len];
                for (int i = 0; i < len; i++)
                {
                    result[i] = (ulong)(v & ulong.MaxValue);
                    v >>= 64;
                }
                return result;
            }
            finally
            {
                owned?.Dispose();
            }
        }

        /// <summary>
        ///     The array NumPy's <c>np.asarray(list)</c> builds from a Python-list-like C# list: all-integer elements (bools
        ///     count as 0/1) become a <see cref="BigInteger"/> array (cast by <see cref="BigIntegersToWords"/>), any real float
        ///     among real numbers makes a float64 array.
        /// </summary>
        /// <param name="list">The list.</param>
        /// <param name="name">The argument name (for the nested-list shape error).</param>
        /// <param name="len">The required element count.</param>
        /// <returns>A <c>BigInteger[]</c> or <c>double[]</c> the array path then casts.</returns>
        /// <exception cref="ValueError">A nested sequence — the array would not be 1-D (NumPy's wrong-size error).</exception>
        /// <exception cref="TypeError">A null, complex, string or other non-real element.</exception>
        private static object InferListArray(System.Collections.IList list, string name, int len)
        {
            bool anyFloat = false;
            foreach (object e in list)
            {
                switch (e)
                {
                    case System.Collections.IList or NDArray:
                        throw new ValueError($"{name} must have {len} elements when using array form");
                    case bool or char or sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger:
                        break;
                    case double or float or Half or decimal:
                        anyFloat = true;
                        break;
                    default:
                        throw new TypeError($"int() argument must be a string, a bytes-like object or a real number, not '{(e is null ? "NoneType" : e.GetType().Name)}'");
                }
            }
            if (anyFloat)
            {
                var doubles = new double[list.Count];
                for (int i = 0; i < doubles.Length; i++)
                {
                    // Integers round through PyInt.ToDouble (Python's correctly rounded float(int) — .NET's
                    // BigInteger/ulong conversions truncate or double-round); Half is not IConvertible.
                    doubles[i] = list[i] switch
                    {
                        bool b => b ? 1.0 : 0.0,
                        Half h => (double)h,
                        double or float or decimal => Convert.ToDouble(list[i], System.Globalization.CultureInfo.InvariantCulture),
                        _ => PyInt.ToDouble(PyInt.From(list[i])),
                    };
                }
                return doubles;
            }
            var ints = new BigInteger[list.Count];
            for (int i = 0; i < ints.Length; i++)
                ints[i] = PyInt.From(list[i]);
            return ints;
        }

        /// <summary>The array-form shape check of <c>int_to_array</c>: exactly <c>(len,)</c>.</summary>
        /// <param name="words">The cast words.</param>
        /// <param name="name">The argument name.</param>
        /// <param name="len">The required element count.</param>
        /// <param name="shape">The array's shape.</param>
        /// <returns><paramref name="words"/>.</returns>
        /// <exception cref="ValueError">The shape is not <c>(len,)</c>.</exception>
        private static ulong[] CheckArrayForm(ulong[] words, string name, int len, long[] shape)
        {
            if (shape.Length != 1 || shape[0] != len)
                throw new ValueError($"{name} must have {len} elements when using array form");
            return words;
        }

        /// <summary>
        ///     The uint64 cast of a list of Python ints, following NumPy's list-to-array inference: each int is int64 when it
        ///     fits <c>[-2**63, 2**63)</c>, uint64 when it fits <c>[2**63, 2**64)</c>, else an object. All int64 → int64
        ///     (negatives wrap), all uint64 → uint64 (exact), an int64/uint64 MIX → float64 (correctly rounded, then cast —
        ///     so <c>[2**63 + 1, 5]</c> loses the low bit exactly as in NumPy), any object → an object array whose
        ///     per-element conversion raises.
        /// </summary>
        /// <param name="big">The integers.</param>
        /// <returns>The uint64 words.</returns>
        /// <exception cref="OverflowException">An element outside <c>[-2**63, 2**64)</c> forces an object array: the first
        /// element that cannot convert raises — <c>int too big to convert</c> beyond the range, <c>Python integer -1 out of
        /// bounds for uint64</c> for a negative.</exception>
        private static ulong[] BigIntegersToWords(BigInteger[] big)
        {
            BigInteger int64Min = long.MinValue, int64Max = long.MaxValue, uint64Max = ulong.MaxValue;
            bool anyInt64 = false, anyUInt64 = false, anyObject = false;
            foreach (BigInteger b in big)
            {
                if (b >= int64Min && b <= int64Max) anyInt64 = true;
                else if (b > int64Max && b <= uint64Max) anyUInt64 = true;
                else anyObject = true;
            }

            var words = new ulong[big.Length];
            if (anyObject)
            {
                // Object array: each Python int converts on its own, in order; the first failure wins.
                foreach (BigInteger b in big)
                {
                    if (b < int64Min || b > uint64Max)
                        throw new OverflowException("int too big to convert");
                    if (b.Sign < 0)
                        throw new OverflowException($"Python integer {b} out of bounds for uint64");
                }
                throw new InvalidOperationException("unreachable: an object element always fails to convert");
            }
            if (!anyUInt64)
            {
                for (int i = 0; i < big.Length; i++)
                    words[i] = unchecked((ulong)(long)big[i]);
                return words;
            }
            if (!anyInt64)
            {
                for (int i = 0; i < big.Length; i++)
                    words[i] = (ulong)big[i];
                return words;
            }
            // int64 + uint64 promote to float64; the cast then follows NumSharp's (MSVC-exact) float64 -> uint64
            // conversion, the same astype the NDArray forms take.
            var doubles = new double[big.Length];
            for (int i = 0; i < big.Length; i++)
                doubles[i] = PyInt.ToDouble(big[i]);
            using NDArray d = np.array(doubles);
            using NDArray u = d.astype(DType.UInt64);
            for (int i = 0; i < big.Length; i++)
                words[i] = u.GetAtIndex<ulong>(i);
            return words;
        }

        // ---- state get/set (numpy Philox.state) ----

        /// <summary>
        ///     The Philox state — NumPy's <c>{'bit_generator': 'Philox', 'state': {'counter': uint64[4], 'key': uint64[2]},
        ///     'buffer': uint64[4], 'buffer_pos': int, 'has_uint32': int, 'uinteger': int}</c> with the nested
        ///     <c>'state'</c> dict flattened.
        /// </summary>
        public sealed class State : BitGeneratorState
        {
            /// <summary>Creates an empty state (fill the members before assigning it).</summary>
            public State() { }

            /// <summary>Creates a state from its members.</summary>
            /// <param name="counter">The 256-bit counter, lowest word first (at least 4 words; extra words are ignored).</param>
            /// <param name="key">The 128-bit key, lowest word first (at least 2 words).</param>
            /// <param name="buffer">The current encrypted block (at least 4 words).</param>
            /// <param name="buffer_pos">The index of the next unread block word; 4 or more means "encrypt a new block first".</param>
            /// <param name="has_uint32">Non-zero when a buffered 32-bit half is pending.</param>
            /// <param name="uinteger">The buffered 32-bit half.</param>
            public State(ulong[] counter, ulong[] key, ulong[] buffer, int buffer_pos, int has_uint32 = 0, uint uinteger = 0)
            {
                this.counter = counter;
                this.key = key;
                this.buffer = buffer;
                this.buffer_pos = buffer_pos;
                this.has_uint32 = has_uint32;
                this.uinteger = uinteger;
            }

            /// <inheritdoc/>
            public override string bit_generator => "Philox";

            /// <summary>The 256-bit counter, lowest word first (NumPy <c>state['state']['counter']</c>).</summary>
            public ulong[] counter { get; set; }

            /// <summary>The 128-bit key, lowest word first (NumPy <c>state['state']['key']</c>).</summary>
            public ulong[] key { get; set; }

            /// <summary>The current encrypted block (NumPy <c>state['buffer']</c>).</summary>
            public ulong[] buffer { get; set; }

            /// <summary>The index of the next unread block word (NumPy <c>state['buffer_pos']</c>); 4 = exhausted.</summary>
            public int buffer_pos { get; set; }

            /// <summary>Non-zero when a buffered 32-bit half is pending.</summary>
            public int has_uint32 { get; set; }

            /// <summary>The buffered 32-bit half.</summary>
            public uint uinteger { get; set; }
        }

        /// <summary>Gets or sets the current internal state (NumPy's <c>bit_generator.state</c>), typed.</summary>
        /// <exception cref="TypeError">Setting null, or a state with a null word array.</exception>
        /// <exception cref="IndexError">A word array shorter than NumPy's loop reads (counter 4, key 2, buffer 4).</exception>
        /// <exception cref="ValueError">A negative <c>buffer_pos</c>.</exception>
        public new State state
        {
            get => (State)base.state;
            set => base.state = value;
        }

        /// <inheritdoc/>
        private protected override BitGeneratorState GetStateCore()
            => new State(new[] { _c0, _c1, _c2, _c3 }, new[] { _k0, _k1 }, (ulong[])_buffer.Clone(), _bufferPos, _hasUint32, _uinteger);

        /// <inheritdoc/>
        /// <remarks>
        ///     Short arrays fail in NumPy's read order — <c>counter[i]</c> and <c>key[i]</c> interleaved for i = 0..3, then
        ///     the buffer — with its IndexError text; the whole state is validated before any of it is applied, so a
        ///     rejected state leaves the generator untouched (NumPy leaves the words it read before the failure written).
        ///     A negative <c>buffer_pos</c> is refused: NumPy would read memory before its buffer (undefined behavior).
        /// </remarks>
        private protected override void SetStateCore(BitGeneratorState value)
        {
            if (value is not State s)
                throw new ValueError("state must be for a Philox PRNG");
            ulong[] counter = s.counter ?? throw new TypeError("state['state']['counter'] must be a sequence of 4 integers");
            ulong[] key = s.key ?? throw new TypeError("state['state']['key'] must be a sequence of 2 integers");
            ulong[] buffer = s.buffer ?? throw new TypeError("state['buffer'] must be a sequence of 4 integers");
            for (int i = 0; i < 4; i++)
            {
                if (i >= counter.Length)
                    throw new IndexError($"index {i} is out of bounds for axis 0 with size {counter.Length}");
                if (i < 2 && i >= key.Length)
                    throw new IndexError($"index {i} is out of bounds for axis 0 with size {key.Length}");
            }
            if (buffer.Length < BufferSize)
                throw new IndexError($"index {buffer.Length} is out of bounds for axis 0 with size {buffer.Length}");
            if (s.buffer_pos < 0)
                throw new ValueError($"state['buffer_pos'] must be non-negative, got {s.buffer_pos}");

            _c0 = counter[0];
            _c1 = counter[1];
            _c2 = counter[2];
            _c3 = counter[3];
            _k0 = key[0];
            _k1 = key[1];
            Array.Copy(buffer, _buffer, BufferSize);
            _hasUint32 = s.has_uint32;
            _uinteger = s.uinteger;
            _bufferPos = s.buffer_pos;
        }
    }
}
