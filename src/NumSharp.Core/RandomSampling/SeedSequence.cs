using System;
using System.Collections.Generic;
using System.Numerics;
using System.Security.Cryptography;

namespace NumSharp
{
    /// <summary>
    ///     Mixes an arbitrary seed into a high-quality initial state for a bit generator.
    /// </summary>
    /// <remarks>
    ///     Port of NumPy 2.4.2's <c>numpy.random.SeedSequence</c>
    ///     (<c>numpy/random/bit_generator.pyx</c>). The entropy-mixing hash and
    ///     <c>generate_state</c> are byte-for-byte identical to NumPy, so
    ///     <see cref="Generator"/>/<see cref="PCG64"/> seeded from the same integer reproduce
    ///     NumPy's exact stream. Spawning and multi-source (spawn-key) entropy are intentionally
    ///     omitted (not needed by <c>default_rng</c>).
    /// </remarks>
    public sealed class SeedSequence
    {
        // Constants from bit_generator.pyx.
        private const uint INIT_A = 0x43b0d7e5;
        private const uint MULT_A = 0x931e8875;
        private const uint INIT_B = 0x8b51f9dd;
        private const uint MULT_B = 0x58f38ded;
        private const uint MIX_MULT_L = 0xca01f9dd;
        private const uint MIX_MULT_R = 0x4973f715;
        private const int XSHIFT = 16;               // uint32 itemsize*8 // 2
        internal const int DEFAULT_POOL_SIZE = 4;

        private readonly uint[] _pool;
        private readonly uint[] _entropy;
        private object _originalEntropy;

        /// <summary>The seed this sequence was constructed from (NumPy <c>SeedSequence.entropy</c>).</summary>
        public object entropy => _originalEntropy;

        /// <summary>The mixed uint32 entropy pool (NumPy <c>SeedSequence.pool</c>).</summary>
        public uint[] pool => (uint[])_pool.Clone();

        /// <summary>The number of uint32 words in the entropy pool (NumPy <c>SeedSequence.pool_size</c>; always 4 here).</summary>
        public int pool_size => _pool.Length;

        /// <summary>Constructs a sequence from fresh, unpredictable OS entropy (NumPy's <c>seed=None</c>).</summary>
        public SeedSequence()
            : this(RandomEntropy()) { }

        /// <summary>Constructs a sequence from a single non-negative integer seed.</summary>
        public SeedSequence(long entropy) : this(FromInteger(entropy)) { _originalEntropy = entropy; }

        /// <summary>Constructs a sequence from a single non-negative integer seed.</summary>
        public SeedSequence(ulong entropy) : this(IntToUint32Array(entropy)) { _originalEntropy = entropy; }

        /// <summary>Constructs a sequence from a single non-negative integer seed of arbitrary size.</summary>
        public SeedSequence(BigInteger entropy) : this(IntToUint32Array(entropy)) { _originalEntropy = entropy; }

        /// <summary>Constructs a sequence from a sequence of non-negative integers.</summary>
        public SeedSequence(int[] entropy) : this(CoerceSequence(entropy)) { _originalEntropy = entropy; }

        /// <summary>Constructs a sequence from a sequence of non-negative integers.</summary>
        public SeedSequence(long[] entropy) : this(CoerceSequence(entropy)) { _originalEntropy = entropy; }

        /// <summary>Constructs a sequence directly from a uint32 word array (NumPy's uint32-ndarray pass-through).</summary>
        public SeedSequence(uint[] entropy)
        {
            _entropy = entropy ?? Array.Empty<uint>();
            _originalEntropy = _entropy;
            _pool = new uint[DEFAULT_POOL_SIZE];
            MixEntropy(_pool, _entropy);
        }

        // ---- entropy coercion (numpy _coerce_to_uint32_array / _int_to_uint32_array) ----

        private static uint[] FromInteger(long entropy)
        {
            if (entropy < 0)
                throw new ValueError("expected non-negative integer");
            return IntToUint32Array((ulong)entropy);
        }

        private static uint[] IntToUint32Array(ulong n)
        {
            if (n == 0)
                return new uint[] { 0u };
            var arr = new List<uint>();
            while (n > 0)
            {
                arr.Add((uint)(n & 0xffffffffUL));
                n >>= 32;
            }
            return arr.ToArray();
        }

        private static uint[] IntToUint32Array(BigInteger n)
        {
            if (n.Sign < 0)
                throw new ValueError("expected non-negative integer");
            if (n.IsZero)
                return new uint[] { 0u };
            var arr = new List<uint>();
            BigInteger mask = 0xffffffffU;
            while (n > 0)
            {
                arr.Add((uint)(n & mask));
                n >>= 32;
            }
            return arr.ToArray();
        }

        private static uint[] CoerceSequence(int[] seq)
        {
            if (seq == null || seq.Length == 0)
                return Array.Empty<uint>();
            var all = new List<uint>();
            foreach (var v in seq)
            {
                if (v < 0)
                    throw new ValueError("expected non-negative integer");
                all.AddRange(IntToUint32Array((ulong)(uint)v));
            }
            return all.ToArray();
        }

        private static uint[] CoerceSequence(long[] seq)
        {
            if (seq == null || seq.Length == 0)
                return Array.Empty<uint>();
            var all = new List<uint>();
            foreach (var v in seq)
            {
                if (v < 0)
                    throw new ValueError("expected non-negative integer");
                all.AddRange(IntToUint32Array((ulong)v));
            }
            return all.ToArray();
        }

        private static uint[] RandomEntropy()
        {
            var bytes = new byte[DEFAULT_POOL_SIZE * 4];
            RandomNumberGenerator.Fill(bytes);
            var words = new uint[DEFAULT_POOL_SIZE];
            for (int i = 0; i < DEFAULT_POOL_SIZE; i++)
                words[i] = BitConverter.ToUInt32(bytes, i * 4);
            return words;
        }

        // ---- mixing (numpy hashmix / mix / mix_entropy) ----

        private static uint Hashmix(uint value, ref uint hashConst)
        {
            value ^= hashConst;
            hashConst *= MULT_A;
            value *= hashConst;
            value ^= value >> XSHIFT;
            return value;
        }

        private static uint Mix(uint x, uint y)
        {
            uint result = MIX_MULT_L * x - MIX_MULT_R * y;
            result ^= result >> XSHIFT;
            return result;
        }

        private static void MixEntropy(uint[] mixer, uint[] entropy)
        {
            uint hashConst = INIT_A;

            // Add in the entropy up to the pool size.
            for (int i = 0; i < mixer.Length; i++)
                mixer[i] = Hashmix(i < entropy.Length ? entropy[i] : 0u, ref hashConst);

            // Mix all bits together so late bits can affect earlier bits.
            for (int iSrc = 0; iSrc < mixer.Length; iSrc++)
                for (int iDst = 0; iDst < mixer.Length; iDst++)
                    if (iSrc != iDst)
                        mixer[iDst] = Mix(mixer[iDst], Hashmix(mixer[iSrc], ref hashConst));

            // Add any remaining entropy, mixing each new entropy word with each pool word.
            for (int iSrc = mixer.Length; iSrc < entropy.Length; iSrc++)
                for (int iDst = 0; iDst < mixer.Length; iDst++)
                    mixer[iDst] = Mix(mixer[iDst], Hashmix(entropy[iSrc], ref hashConst));
        }

        /// <summary>
        ///     Return the requested number of words for PRNG seeding (NumPy
        ///     <c>generate_state(n_words, dtype=np.uint32)</c>). Returns a <c>uint[]</c> for
        ///     <c>uint32</c> (the default) or a <c>ulong[]</c> for <c>uint64</c>.
        /// </summary>
        public Array generate_state(int n_words, Type dtype = null)
        {
            if (dtype == null)
                return GenerateState(n_words);
            var tc = dtype.GetTypeCode();
            if (tc == NPTypeCode.UInt32)
                return GenerateState(n_words);
            if (tc == NPTypeCode.UInt64)
                return GenerateState64(n_words);
            throw new ValueError("only support uint32 or uint64");
        }

        /// <summary>
        ///     Returns <paramref name="nWords"/> uint32 words for PRNG seeding (NumPy
        ///     <c>generate_state(n_words, np.uint32)</c>).
        /// </summary>
        internal uint[] GenerateState(int nWords)
        {
            var state = new uint[nWords];
            uint hashConst = INIT_B;
            int cyc = 0;
            for (int i = 0; i < nWords; i++)
            {
                uint dataVal = _pool[cyc];
                cyc++;
                if (cyc == _pool.Length) cyc = 0;
                dataVal ^= hashConst;
                hashConst *= MULT_B;
                dataVal *= hashConst;
                dataVal ^= dataVal >> XSHIFT;
                state[i] = dataVal;
            }
            return state;
        }

        /// <summary>
        ///     Returns <paramref name="nWords"/> uint64 words for PRNG seeding (NumPy
        ///     <c>generate_state(n_words, np.uint64)</c> — draws twice as many uint32 and views them
        ///     little-endian).
        /// </summary>
        internal ulong[] GenerateState64(int nWords)
        {
            uint[] words = GenerateState(nWords * 2);
            var result = new ulong[nWords];
            for (int i = 0; i < nWords; i++)
                result[i] = words[2 * i] | ((ulong)words[2 * i + 1] << 32);
            return result;
        }
    }
}
