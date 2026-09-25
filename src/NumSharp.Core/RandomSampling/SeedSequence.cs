using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;

namespace NumSharp
{
    /// <summary>
    ///     Mixes sources of entropy into a high-quality initial state for a bit generator, and spawns independent
    ///     children for parallel streams (NumPy's <c>numpy.random.SeedSequence</c>).
    /// </summary>
    /// <remarks>
    ///     Port of NumPy 2.4.2's <c>SeedSequence</c> (<c>numpy/random/bit_generator.pyx</c>). The entropy coercion
    ///     (<c>_coerce_to_uint32_array</c>), the spawn-key assembly, the hash-mixing pool and <see cref="generate_state"/>
    ///     are byte-for-byte NumPy's, so a <see cref="PCG64"/>/<see cref="MT19937"/>/… seeded from the same entropy — or
    ///     from the same position in a <see cref="spawn"/> tree — reproduces NumPy's exact stream.
    /// </remarks>
    public sealed class SeedSequence : ISpawnableSeedSequence
    {
        // Constants from bit_generator.pyx.
        private const uint INIT_A = 0x43b0d7e5;
        private const uint MULT_A = 0x931e8875;
        private const uint INIT_B = 0x8b51f9dd;
        private const uint MULT_B = 0x58f38ded;
        private const uint MIX_MULT_L = 0xca01f9dd;
        private const uint MIX_MULT_R = 0x4973f715;
        private const int XSHIFT = 16;               // uint32 itemsize*8 // 2

        /// <summary>NumPy's <c>DEFAULT_POOL_SIZE</c>: 4 words, a 128-bit pool (also the minimum).</summary>
        internal const int DEFAULT_POOL_SIZE = 4;

        private readonly uint[] _pool;
        private readonly object _entropy;
        private readonly long[] _spawnKey;
        private uint _nChildrenSpawned;

        /// <summary>
        ///     The entropy this sequence was built from (NumPy's <c>SeedSequence.entropy</c>): the integer or sequence as
        ///     given, or — for a sequence built without entropy — the 128-bit (<c>pool_size * 32</c>-bit) random
        ///     <see cref="BigInteger"/> drawn from the OS, which is what to log for reproducibility
        ///     (<c>new SeedSequence(seq.entropy)</c> rebuilds the same pool).
        /// </summary>
        public object entropy => _entropy;

        /// <summary>
        ///     The position of this sequence in its spawn tree (NumPy's <c>spawn_key</c> tuple): empty for a root, and the
        ///     parent's key plus the child index for a spawned child. It is mixed in as additional entropy.
        /// </summary>
        public long[] spawn_key => (long[])_spawnKey.Clone();

        /// <summary>The number of uint32 words in the entropy pool (NumPy's <c>pool_size</c>; at least 4).</summary>
        public int pool_size => _pool.Length;

        /// <summary>The number of children already spawned (NumPy's <c>n_children_spawned</c>, a uint32) — the next child's index.</summary>
        public long n_children_spawned => _nChildrenSpawned;

        /// <summary>The mixed uint32 entropy pool (NumPy's <c>SeedSequence.pool</c>), as a copy.</summary>
        public uint[] pool => (uint[])_pool.Clone();

        /// <summary>
        ///     NumPy's <c>SeedSequence.state</c>: the constructor arguments that rebuild this sequence —
        ///     <c>{'entropy', 'spawn_key', 'pool_size', 'n_children_spawned'}</c>, in NumPy's key order.
        /// </summary>
        public Dictionary<string, object> state => new Dictionary<string, object>
        {
            ["entropy"] = _entropy,
            ["spawn_key"] = spawn_key,
            ["pool_size"] = pool_size,
            ["n_children_spawned"] = n_children_spawned,
        };

        /// <summary>Constructs a sequence from fresh, unpredictable OS entropy (NumPy's <c>SeedSequence()</c>).</summary>
        public SeedSequence() : this((object)null) { }

        /// <summary>Constructs a sequence from a single non-negative integer seed.</summary>
        /// <param name="entropy">The seed.</param>
        /// <exception cref="ValueError"><paramref name="entropy"/> is negative (<c>expected non-negative integer</c>).</exception>
        public SeedSequence(long entropy) : this((object)entropy) { }

        /// <summary>Constructs a sequence from a single non-negative integer seed.</summary>
        /// <param name="entropy">The seed.</param>
        public SeedSequence(ulong entropy) : this((object)entropy) { }

        /// <summary>Constructs a sequence from a single non-negative integer seed of arbitrary size.</summary>
        /// <param name="entropy">The seed.</param>
        /// <exception cref="ValueError"><paramref name="entropy"/> is negative.</exception>
        public SeedSequence(BigInteger entropy) : this((object)entropy) { }

        /// <summary>Constructs a sequence from a sequence of non-negative integers.</summary>
        /// <param name="entropy">The seed words (a value of 2**32 or more spans several words, as in NumPy).</param>
        /// <exception cref="ValueError">An element is negative.</exception>
        public SeedSequence(int[] entropy) : this((object)(entropy ?? Array.Empty<int>())) { }

        /// <summary>Constructs a sequence from a sequence of non-negative integers.</summary>
        /// <param name="entropy">The seed words (a value of 2**32 or more spans several words, as in NumPy).</param>
        /// <exception cref="ValueError">An element is negative.</exception>
        public SeedSequence(long[] entropy) : this((object)(entropy ?? Array.Empty<long>())) { }

        /// <summary>Constructs a sequence directly from uint32 words (NumPy's uint32-ndarray pass-through).</summary>
        /// <param name="entropy">The seed words.</param>
        public SeedSequence(uint[] entropy) : this((object)(entropy ?? Array.Empty<uint>())) { }

        /// <summary>
        ///     Constructs a sequence with NumPy's full signature:
        ///     <c>SeedSequence(entropy=None, *, spawn_key=(), pool_size=4, n_children_spawned=0)</c>.
        /// </summary>
        /// <param name="entropy">
        ///     null for fresh OS entropy; a non-negative integer (any C# integer type, <see cref="BigInteger"/>, or a bool as
        ///     0/1); a sequence of them — a C# array or <see cref="System.Collections.IList"/>, nested sequences flattening in
        ///     order and string elements parsing as Python ints (<c>"0x…"</c> hex, other leading-<c>0</c> octal, else decimal);
        ///     or an integer <see cref="NDArray"/> (flattened in C order; a uint32 array passes through as-is, a
        ///     <c>uint[]</c> standing for one).
        /// </param>
        /// <param name="spawn_key">This sequence's position in a spawn tree (mixed in as extra entropy); null or empty for a root.</param>
        /// <param name="pool_size">The pool size in uint32 words (at least 4; 8 gives a 256-bit pool).</param>
        /// <param name="n_children_spawned">The number of children already spawned — only when rebuilding a serialized sequence.</param>
        /// <exception cref="ValueError"><paramref name="pool_size"/> is below 4; an entropy or spawn-key value is negative.</exception>
        /// <exception cref="TypeError"><paramref name="entropy"/> is not an integer or a sequence of integers (a float, a string,
        /// a float/bool array, a 0-d array, a null element).</exception>
        /// <exception cref="OverflowException"><paramref name="n_children_spawned"/> is outside the uint32 range NumPy stores it in.</exception>
        public SeedSequence(object entropy, long[] spawn_key = null, int pool_size = DEFAULT_POOL_SIZE, long n_children_spawned = 0)
        {
            if (pool_size < DEFAULT_POOL_SIZE)
                throw new ValueError($"The size of the entropy pool should be at least {DEFAULT_POOL_SIZE}");

            if (entropy is null)
            {
                // NumPy: entropy = randbits(pool_size * 32) — a Python int, kept as the reproducible entropy.
                var bytes = new byte[pool_size * 4];
                RandomNumberGenerator.Fill(bytes);
                entropy = new BigInteger(bytes, isUnsigned: true, isBigEndian: false);
            }
            else if (!IsIntegerScalar(entropy) && entropy is not System.Collections.IList && entropy is not NDArray)
            {
                // NumPy's isinstance(entropy, (int, np.integer, list, tuple, range, np.ndarray)) gate: a C# array or
                // list stands for list/tuple; a bare string is rejected here even though a string ELEMENT would parse.
                throw new TypeError($"SeedSequence expects int or sequence of ints for entropy not {PythonInt.Str(entropy)}");
            }

            _entropy = entropy;
            _spawnKey = spawn_key is null ? Array.Empty<long>() : (long[])spawn_key.Clone();
            _nChildrenSpawned = ToUInt32Count(n_children_spawned);

            _pool = new uint[pool_size];
            MixEntropy(_pool, AssembleEntropy());
        }

        // ---- entropy coercion (numpy get_assembled_entropy / _coerce_to_uint32_array / _int_to_uint32_array) ----

        /// <summary>
        ///     NumPy's <c>get_assembled_entropy</c>: the run entropy and the spawn key as uint32 words, the run entropy
        ///     zero-padded to the pool size when a spawn key follows (so a spawned child cannot collide with a root seeded
        ///     from a longer entropy — the gh-16539 fix, applied only when a key is present to keep root streams stable).
        /// </summary>
        /// <returns>The words to mix into the pool.</returns>
        /// <exception cref="ValueError">A negative value; a passed-through uint32 array that is 0-d or not 1-D (NumPy's
        /// <c>np.concatenate</c> errors).</exception>
        /// <exception cref="TypeError">A non-integer entropy element, or a 0-d uint32 array when a spawn key is present
        /// (<c>len() of unsized object</c>).</exception>
        private uint[] AssembleEntropy()
        {
            EntropyPiece run = CoercePiece(_entropy);
            EntropyPiece spawn = CoerceSequence(_spawnKey);
            if (spawn.Words.Length > 0)
            {
                // NumPy evaluates len(run_entropy) only when a key is present — which is why a 0-d uint32 entropy
                // is a TypeError with a key and a concatenate ValueError without one.
                if (run.NDim == 0)
                    throw new TypeError("len() of unsized object");
                if (run.Shape[0] < _pool.Length)
                    run = Concatenate(new[] { run, new EntropyPiece(new uint[_pool.Length - run.Shape[0]]) });
            }
            return Concatenate(new[] { run, spawn }).Words;
        }

        /// <summary>
        ///     One coerced entropy piece: its uint32 words plus the shape NumPy's array has — 1-D for everything except
        ///     a passed-through uint32 array, which keeps its own shape and so can trip <c>np.concatenate</c>.
        /// </summary>
        private readonly struct EntropyPiece
        {
            /// <summary>Creates a 1-D piece.</summary>
            /// <param name="words">The words.</param>
            internal EntropyPiece(uint[] words) : this(words, new long[] { words.Length }) { }

            /// <summary>Creates a piece of an explicit shape (a passed-through uint32 array).</summary>
            /// <param name="words">The words, C order.</param>
            /// <param name="shape">The array's shape.</param>
            internal EntropyPiece(uint[] words, long[] shape)
            {
                Words = words;
                Shape = shape;
            }

            /// <summary>The words, C order.</summary>
            internal readonly uint[] Words;

            /// <summary>The shape NumPy's intermediate array has.</summary>
            internal readonly long[] Shape;

            /// <summary>The number of dimensions.</summary>
            internal int NDim => Shape.Length;
        }

        /// <summary>
        ///     NumPy's <c>_coerce_to_uint32_array</c>: a uint32 array passes through (copied, shape kept); an integer
        ///     becomes its little-endian uint32 limbs; a string parses (hex for <c>0x…</c>, octal for any other leading
        ///     <c>0</c>, else decimal); a sequence coerces each element and concatenates the pieces.
        /// </summary>
        /// <param name="x">The entropy, or one element of it.</param>
        /// <returns>The coerced piece.</returns>
        /// <exception cref="ValueError">A negative value or an unparsable string.</exception>
        /// <exception cref="TypeError">A float/complex (<c>seed must be integer</c>), a 0-d non-uint32 array
        /// (<c>len() of unsized object</c>), a bool array (<c>object of type 'numpy.bool' has no len()</c>) or null
        /// (<c>object of type 'NoneType' has no len()</c>).</exception>
        private static EntropyPiece CoercePiece(object x)
        {
            switch (x)
            {
                // The CLR lets an int[] pass an `is uint[]` test (same-size integer array covariance), so the
                // pass-through is keyed on the EXACT type — a signed int[] must take the sequence path, where a
                // negative element is NumPy's "expected non-negative integer".
                case uint[] words when words.GetType() == typeof(uint[]):
                    return new EntropyPiece((uint[])words.Clone()); // the uint32-array pass-through
                case NDArray nd:
                    return CoerceNDArray(nd);
                case string s:
                    // NumPy picks the base from the RAW text (no strip first): ' 0x10' parses in base 10 and fails.
                    int radix = s.StartsWith("0x", StringComparison.Ordinal) ? 16 : s.StartsWith("0", StringComparison.Ordinal) ? 8 : 10;
                    return new EntropyPiece(IntToUint32Array(PythonInt.Parse(s, radix)));
                case System.Collections.IList list:
                    return CoerceSequence(list);
                case null:
                    throw new TypeError("object of type 'NoneType' has no len()");
                default:
                    if (IsIntegerScalar(x))
                        return new EntropyPiece(IntToUint32Array(ToBigInteger(x)));
                    if (x is float or double or Half or decimal or Complex)
                        throw new TypeError("seed must be integer");
                    throw new TypeError($"object of type '{x.GetType().Name}' has no len()");
            }
        }

        /// <summary>The sequence branch of the coercion: an empty sequence is an empty uint32 array, else each element's piece, concatenated.</summary>
        /// <param name="list">The sequence (a C# array — multi-dimensional ones flatten in C order like nested lists — or list).</param>
        /// <returns>The concatenated piece.</returns>
        /// <exception cref="ValueError">An element failed, or the pieces cannot be concatenated.</exception>
        /// <exception cref="TypeError">An element is not an integer.</exception>
        private static EntropyPiece CoerceSequence(System.Collections.IList list)
        {
            if (list.Count == 0)
                return new EntropyPiece(Array.Empty<uint>());
            var pieces = new List<EntropyPiece>(list.Count);
            foreach (object v in list)
                pieces.Add(CoercePiece(v));
            return Concatenate(pieces);
        }

        /// <summary>
        ///     The <see cref="NDArray"/> branch of the coercion: a uint32 array passes through with its shape; any other
        ///     array takes the sequence path, which starts with <c>len(x)</c> and then iterates its elements.
        /// </summary>
        /// <param name="nd">The entropy array.</param>
        /// <returns>The uint32 piece.</returns>
        /// <exception cref="TypeError">0-d (<c>len() of unsized object</c>), non-empty bool (<c>object of type 'numpy.bool' has
        /// no len()</c>), or non-empty non-integer (<c>seed must be integer</c>).</exception>
        /// <exception cref="ValueError">A negative element.</exception>
        private static EntropyPiece CoerceNDArray(NDArray nd)
        {
            NPTypeCode tc = nd.typecode;
            if (tc == NPTypeCode.UInt32)
            {
                var pass = new uint[nd.size];
                for (long i = 0; i < pass.LongLength; i++)
                    pass[i] = nd.GetAtIndex<uint>(i);
                return new EntropyPiece(pass, ToLongShape(nd));
            }
            if (nd.ndim == 0)
                throw new TypeError("len() of unsized object");
            // Iterating an empty array never reaches an element, so no element check fires (any dtype).
            if (nd.size == 0)
                return new EntropyPiece(Array.Empty<uint>());
            switch (tc)
            {
                case NPTypeCode.Boolean:
                    throw new TypeError("object of type 'numpy.bool' has no len()");
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Char:
                    // Row-by-row recursion over a C-order walk is a C-order flatten; GetAtIndex reads the logical
                    // index through any layout, so no raveled temporary is needed.
                    var all = new List<uint>();
                    for (long i = 0; i < nd.size; i++)
                        all.AddRange(IntToUint32Array(ToBigInteger(nd.GetAtIndex(i))));
                    return new EntropyPiece(all.ToArray());
                default:
                    throw new TypeError("seed must be integer");
            }
        }

        /// <summary>The array's shape as the <c>long[]</c> the concatenate checks compare.</summary>
        /// <param name="nd">The array.</param>
        /// <returns>A fresh copy of its dimensions.</returns>
        private static long[] ToLongShape(NDArray nd)
        {
            var shape = new long[nd.ndim];
            for (int d = 0; d < shape.Length; d++)
                shape[d] = nd.shape[d];
            return shape;
        }

        /// <summary>
        ///     NumPy's <c>np.concatenate(pieces)</c> along axis 0 with its validation order: the first piece must not be
        ///     0-d, then each later piece must match its dimension count and every non-leading extent.
        /// </summary>
        /// <param name="pieces">The pieces (at least one).</param>
        /// <returns>The concatenation.</returns>
        /// <exception cref="ValueError">NumPy's three concatenate errors, verbatim.</exception>
        private static EntropyPiece Concatenate(IReadOnlyList<EntropyPiece> pieces)
        {
            EntropyPiece first = pieces[0];
            int ndim = first.NDim;
            if (ndim == 0)
                throw new ValueError("zero-dimensional arrays cannot be concatenated");
            var shape = (long[])first.Shape.Clone();
            long total = first.Words.Length;
            for (int i = 1; i < pieces.Count; i++)
            {
                EntropyPiece p = pieces[i];
                if (p.NDim != ndim)
                    throw new ValueError($"all the input arrays must have same number of dimensions, but the array at index 0 has {ndim} dimension(s) and the array at index {i} has {p.NDim} dimension(s)");
                for (int d = 0; d < ndim; d++)
                {
                    if (d == 0)
                        shape[0] += p.Shape[0];
                    else if (shape[d] != p.Shape[d])
                        throw new ValueError($"all the input array dimensions except for the concatenation axis must match exactly, but along dimension {d}, the array at index 0 has size {shape[d]} and the array at index {i} has size {p.Shape[d]}");
                }
                total += p.Words.Length;
            }
            if (pieces.Count == 1)
                return first;
            var words = new uint[total];
            long at = 0;
            foreach (EntropyPiece p in pieces)
            {
                Array.Copy(p.Words, 0, words, at, p.Words.Length);
                at += p.Words.Length;
            }
            return new EntropyPiece(words, shape);
        }

        /// <summary>Whether <paramref name="x"/> is an integer scalar NumPy's <c>isinstance(x, (int, np.integer))</c> accepts (bool included).</summary>
        /// <param name="x">The candidate.</param>
        /// <returns>True for C# integer types, <see cref="BigInteger"/>, bool and char.</returns>
        private static bool IsIntegerScalar(object x)
            => x is sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger or bool or char;

        /// <summary>Converts an integer scalar (or bool) to <see cref="BigInteger"/>.</summary>
        /// <param name="v">The value.</param>
        /// <returns>The integer value.</returns>
        /// <exception cref="TypeError">A float or other non-integer element (<c>seed must be integer</c>).</exception>
        private static BigInteger ToBigInteger(object v) => v switch
        {
            bool b => b ? BigInteger.One : BigInteger.Zero,
            char c => c,
            BigInteger bi => bi,
            sbyte or byte or short or ushort or int or uint or long => Convert.ToInt64(v, CultureInfo.InvariantCulture),
            ulong u => u,
            _ => throw new TypeError("seed must be integer"),
        };

        /// <summary>NumPy's <c>_int_to_uint32_array</c>: little-endian uint32 limbs (<c>[0]</c> for zero).</summary>
        /// <param name="n">A non-negative integer.</param>
        /// <returns>The limbs, lowest first.</returns>
        /// <exception cref="ValueError"><paramref name="n"/> is negative (<c>expected non-negative integer</c>).</exception>
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

        /// <summary>Validates a child count against NumPy's uint32 attribute.</summary>
        /// <param name="n">The count.</param>
        /// <returns>The count as uint32.</returns>
        /// <exception cref="OverflowException">Negative, or above 2**32 - 1 (NumPy's OverflowError texts).</exception>
        private static uint ToUInt32Count(long n)
        {
            if (n < 0)
                throw new OverflowException("can't convert negative value to uint32_t");
            if (n > uint.MaxValue)
                throw new OverflowException("Python int too large to convert to C unsigned long");
            return (uint)n;
        }

        // ---- mixing (numpy hashmix / mix / mix_entropy) ----

        /// <summary>NumPy's <c>hashmix</c> (with the running multiplier).</summary>
        /// <param name="value">The word to hash.</param>
        /// <param name="hashConst">The running hash constant (advanced).</param>
        /// <returns>The hashed word.</returns>
        private static uint Hashmix(uint value, ref uint hashConst)
        {
            value ^= hashConst;
            hashConst *= MULT_A;
            value *= hashConst;
            value ^= value >> XSHIFT;
            return value;
        }

        /// <summary>NumPy's <c>mix</c>.</summary>
        /// <param name="x">The pool word.</param>
        /// <param name="y">The hashed word mixed in.</param>
        /// <returns>The mixed word.</returns>
        private static uint Mix(uint x, uint y)
        {
            uint result = MIX_MULT_L * x - MIX_MULT_R * y;
            result ^= result >> XSHIFT;
            return result;
        }

        /// <summary>NumPy's <c>mix_entropy</c>: fill the pool, mix every word with every other, then fold in the rest.</summary>
        /// <param name="mixer">The pool (written).</param>
        /// <param name="entropy">The assembled entropy words.</param>
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

        // ---- output ----

        /// <summary>
        ///     Return the requested number of words for PRNG seeding (NumPy
        ///     <c>generate_state(n_words, dtype=np.uint32)</c>). Returns a <c>uint[]</c> for
        ///     <c>uint32</c> (the default) or a <c>ulong[]</c> for <c>uint64</c>.
        /// </summary>
        /// <param name="n_words">The number of words to produce (0 gives an empty result).</param>
        /// <param name="dtype"><c>uint32</c> (default) or <c>uint64</c>; a uint64 word consumes two uint32 words of the stream.</param>
        /// <returns>The seeding words — the same words for the same pool every call (the sequence is not consumed).</returns>
        /// <exception cref="ValueError"><paramref name="n_words"/> is negative (NumPy's <c>np.zeros(n_words)</c> error), or <paramref name="dtype"/>
        /// is not the NATIVE uint32/uint64 (NumPy compares with <c>==</c>, so a byte-swapped <c>'&gt;u4'</c> is rejected too).</exception>
        public Array generate_state(int n_words, DType dtype = null)
        {
            // Structural equality (class + byte order), as NumPy's `dtype == np.dtype(np.uint32)`.
            NPTypeCode tc = dtype is null || dtype == DType.UInt32 ? NPTypeCode.UInt32
                : dtype == DType.UInt64 ? NPTypeCode.UInt64
                : NPTypeCode.Empty;
            if (tc == NPTypeCode.Empty)
                throw new ValueError("only support uint32 or uint64");
            // NumPy allocates `np.zeros(n_words, dtype=np.uint32)` (twice the words for uint64), so a
            // negative count is the allocator's error, raised after the dtype is accepted.
            if (n_words < 0)
                throw new ValueError("negative dimensions are not allowed");
            return tc == NPTypeCode.UInt64 ? GenerateState64(n_words) : GenerateState(n_words);
        }

        /// <summary>
        ///     Returns <paramref name="nWords"/> uint32 words for PRNG seeding (NumPy
        ///     <c>generate_state(n_words, np.uint32)</c>): the pool cycled through a second hash.
        /// </summary>
        /// <param name="nWords">The word count (non-negative).</param>
        /// <returns>The words.</returns>
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
        /// <param name="nWords">The word count (non-negative).</param>
        /// <returns>The words.</returns>
        internal ulong[] GenerateState64(int nWords)
        {
            uint[] words = GenerateState(nWords * 2);
            var result = new ulong[nWords];
            for (int i = 0; i < nWords; i++)
                result[i] = words[2 * i] | ((ulong)words[2 * i + 1] << 32);
            return result;
        }

        // ---- spawning ----

        /// <summary>
        ///     Spawn <paramref name="n_children"/> child sequences by extending <see cref="spawn_key"/> (NumPy's
        ///     <c>SeedSequence.spawn</c>): child <c>i</c> gets this sequence's entropy and pool size and the key
        ///     <c>spawn_key + (n_children_spawned + i,)</c>, and <see cref="n_children_spawned"/> advances, so later
        ///     calls continue the numbering and never repeat a child.
        /// </summary>
        /// <param name="n_children">The number of children.</param>
        /// <returns>The children (empty for 0).</returns>
        /// <exception cref="OverflowException">The child count would leave the uint32 range NumPy stores it in (a negative
        /// <paramref name="n_children"/> included — NumPy raises after an empty loop, and no children are returned).</exception>
        public SeedSequence[] spawn(int n_children)
        {
            long start = _nChildrenSpawned;
            uint next = ToUInt32Count(start + n_children); // validated first: NumPy's loop is empty for n < 0, then += raises
            var seqs = new SeedSequence[Math.Max(0, n_children)];
            for (int c = 0; c < seqs.Length; c++)
            {
                var key = new long[_spawnKey.Length + 1];
                _spawnKey.CopyTo(key, 0);
                key[_spawnKey.Length] = start + c;
                seqs[c] = new SeedSequence(_entropy, key, _pool.Length);
            }
            _nChildrenSpawned = next;
            return seqs;
        }

        /// <inheritdoc/>
        ISpawnableSeedSequence[] ISpawnableSeedSequence.spawn(int n_children) => spawn(n_children);

        // ---- repr ----

        /// <summary>
        ///     NumPy's <c>repr(SeedSequence)</c>: the entropy, plus spawn_key / pool_size / n_children_spawned when they
        ///     differ from their defaults, one per line.
        /// </summary>
        /// <returns>E.g. <c>SeedSequence(\n    entropy=42,\n)</c>.</returns>
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append("SeedSequence(\n");
            sb.Append("    entropy=").Append(ReprEntropy(_entropy)).Append(",\n");
            if (_spawnKey.Length > 0)
                sb.Append("    spawn_key=").Append(ReprTuple(_spawnKey)).Append(",\n");
            if (_pool.Length != DEFAULT_POOL_SIZE)
                sb.Append("    pool_size=").Append(_pool.Length.ToString(CultureInfo.InvariantCulture)).Append(",\n");
            if (_nChildrenSpawned != 0)
                sb.Append("    n_children_spawned=").Append(_nChildrenSpawned.ToString(CultureInfo.InvariantCulture)).Append(",\n");
            sb.Append(')');
            return sb.ToString();
        }

        /// <summary>
        ///     Python's repr of the entropy: an int, a list <c>[1, 2]</c> (nested lists recursively, string elements
        ///     quoted), a bool, <c>None</c>, or an ndarray repr.
        /// </summary>
        /// <param name="e">The stored entropy (or one element of it).</param>
        /// <returns>The repr text.</returns>
        private static string ReprEntropy(object e)
        {
            switch (e)
            {
                case null:
                    return "None";
                case bool b:
                    return b ? "True" : "False";
                case string str:
                    return PythonInt.Repr(str);
                case uint[] u when u.GetType() == typeof(uint[]):
                    // The uint32 pass-through stands for NumPy's uint32 ndarray (exact type: an int[] also
                    // passes `is uint[]`, and is a Python list).
                    return "array([" + string.Join(", ", u) + "], dtype=uint32)";
                case NDArray nd:
                    return nd.ToString(true);
                case System.Collections.IList list:
                {
                    var parts = new List<string>();
                    foreach (object v in list)
                        parts.Add(ReprEntropy(v));
                    return "[" + string.Join(", ", parts) + "]";
                }
                default:
                    return PythonInt.Str(e);
            }
        }

        /// <summary>Python's repr of a tuple of ints: <c>()</c>, <c>(1,)</c>, <c>(1, 2)</c>.</summary>
        /// <param name="t">The elements.</param>
        /// <returns>The repr text.</returns>
        private static string ReprTuple(long[] t)
            => t.Length == 1 ? $"({t[0].ToString(CultureInfo.InvariantCulture)},)" : "(" + string.Join(", ", t) + ")";
    }
}
