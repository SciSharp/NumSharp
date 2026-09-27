using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
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
        private readonly BigInteger[] _spawnKey;
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
        /// <remarks>
        ///     Python ints, so <see cref="BigInteger"/>s: a key element of any size is legal (NumPy coerces a large one into
        ///     several uint32 words), where a 64-bit type would cap it. A key given to the constructor with string elements or
        ///     nested sequences is stored as the flat ints NumPy's coercion reads (the same pool, a plainer repr). A copy;
        ///     changing it changes nothing here.
        /// </remarks>
        public BigInteger[] spawn_key => (BigInteger[])_spawnKey.Clone();

        /// <summary>The number of uint32 words in the entropy pool (NumPy's <c>pool_size</c>, a <c>Py_ssize_t</c>; at least 4).</summary>
        public long pool_size => _pool.LongLength;

        /// <summary>
        ///     The number of children already spawned — the next child's index (NumPy's <c>n_children_spawned</c>, stored as
        ///     a <c>uint32_t</c>: <see cref="spawn"/> refuses to pass <c>2**32 - 1</c>).
        /// </summary>
        public uint n_children_spawned => _nChildrenSpawned;

        /// <summary>
        ///     The mixed uint32 entropy pool (NumPy's <c>SeedSequence.pool</c>, a uint32 ndarray of <see cref="pool_size"/>
        ///     words), as a fresh copy.
        /// </summary>
        /// <remarks>
        ///     NumPy hands out the pool array itself, so writing into it there changes every later
        ///     <see cref="generate_state"/>; the copy keeps this sequence immutable, as its documentation promises.
        /// </remarks>
        public NDArray pool => np.array((uint[])_pool.Clone());

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
        /// <param name="entropy">The seed words (a value of 2**32 or more spans several words, as in NumPy); null is NumPy's
        ///     <c>None</c> — fresh OS entropy, never the empty list (an empty array is <c>[]</c>, a fixed seed).</param>
        /// <exception cref="ValueError">An element is negative.</exception>
        /// <remarks>
        ///     A null array used to become <c>[]</c>, so a "no seed" array seeded every run with the SAME stream (NumPy's
        ///     <c>SeedSequence([])</c>) — silently, since an entropy-seeded result is not reproducible to begin with. It now
        ///     forwards null to the dynamically-typed overload, which reads OS entropy like <see cref="SeedSequence()"/>.
        /// </remarks>
        public SeedSequence(int[] entropy) : this((object)entropy) { }

        /// <summary>Constructs a sequence from a sequence of non-negative integers.</summary>
        /// <param name="entropy">The seed words (a value of 2**32 or more spans several words, as in NumPy); null is NumPy's
        ///     <c>None</c> — fresh OS entropy, never the empty list.</param>
        /// <exception cref="ValueError">An element is negative.</exception>
        public SeedSequence(long[] entropy) : this((object)entropy) { }

        /// <summary>Constructs a sequence directly from uint32 words (NumPy's uint32-ndarray pass-through).</summary>
        /// <param name="entropy">The seed words; null is NumPy's <c>None</c> — fresh OS entropy, never the empty list.</param>
        /// <remarks>
        ///     The overload a bare <c>null</c> literal binds (<c>new SeedSequence(null)</c>, NumPy's
        ///     <c>SeedSequence(None)</c>): the <c>int[]</c>/<c>long[]</c>/<c>uint[]</c> overloads are equally good targets for
        ///     it under plain C# rules, so this one carries the higher <see cref="OverloadResolutionPriorityAttribute"/> —
        ///     harmless for every non-null argument, since no other array converts to <c>uint[]</c>.
        /// </remarks>
        [OverloadResolutionPriority(1)]
        public SeedSequence(uint[] entropy) : this((object)entropy) { }

        /// <summary>
        ///     Constructs a sequence with NumPy's full signature:
        ///     <c>SeedSequence(entropy=None, *, spawn_key=(), pool_size=4, n_children_spawned=0)</c>.
        /// </summary>
        /// <param name="entropy">
        ///     null for fresh OS entropy; a non-negative integer (any C# integer type, <see cref="BigInteger"/>, or a bool as
        ///     0/1); a sequence of them — a C# array or <see cref="System.Collections.IList"/>, nested sequences flattening in
        ///     order and string elements parsing NumPy's way (<see cref="ParseSeedString"/>: a <c>"0x…"</c> string through
        ///     <c>int(x, 16)</c>, a string that starts with a decimal digit through <c>int(x)</c>, anything else
        ///     <c>unrecognized seed string</c>); or an integer <see cref="NDArray"/> (flattened in C order; a uint32 array
        ///     passes through as-is, a <c>uint[]</c> standing for one).
        /// </param>
        /// <param name="spawn_key">This sequence's position in a spawn tree (mixed in as extra entropy); null or empty for a
        ///     root. Any iterable — NumPy's <c>tuple(spawn_key)</c> — such as a <c>long[]</c>, <c>int[]</c>, <c>uint[]</c>,
        ///     <see cref="BigInteger"/><c>[]</c>, a list or an integer <see cref="NDArray"/>; string elements parse with the
        ///     entropy's string rule (a string key is itself a sequence of one-character strings, as <c>tuple("12")</c> is),
        ///     and nested sequences flatten the way NumPy's coercion flattens them (see <see cref="ToSpawnKey"/>).</param>
        /// <param name="pool_size">The pool size in uint32 words (at least 4; 8 gives a 256-bit pool) — NumPy's <c>Py_ssize_t</c>.</param>
        /// <param name="n_children_spawned">The number of children already spawned — only when rebuilding a serialized
        ///     sequence. NumPy stores it as a <c>uint32_t</c>, so the parameter is one.</param>
        /// <exception cref="ValueError"><paramref name="pool_size"/> is below 4; an entropy or spawn-key value is negative
        ///     (<c>expected non-negative integer</c>); a string that is not a seed literal (<c>unrecognized seed string</c>, or
        ///     Python's <c>invalid literal for int() …</c> for one that starts like a number).</exception>
        /// <exception cref="TypeError"><paramref name="entropy"/> is not an integer or a sequence of integers (a float, a string,
        /// a float/bool array, a 0-d array, a null element); <paramref name="spawn_key"/> is not iterable (NumPy's
        /// <c>'int' object is not iterable</c>), is a 0-d array (<c>iteration over a 0-d array</c>) or holds an element the
        /// coercion refuses (a float: <c>seed must be integer</c>; null: <c>object of type 'NoneType' has no len()</c>).</exception>
        /// <exception cref="OutOfMemoryException"><paramref name="pool_size"/> is larger than an array can hold (NumPy's
        ///     <c>MemoryError</c> from <c>np.zeros(pool_size)</c>).</exception>
        public SeedSequence(object entropy, object spawn_key = null, long pool_size = DEFAULT_POOL_SIZE, uint n_children_spawned = 0)
        {
            if (pool_size < DEFAULT_POOL_SIZE)
                throw new ValueError($"The size of the entropy pool should be at least {DEFAULT_POOL_SIZE}");
            // The pool is a managed uint[] (it is mixed word by word and read by every generate_state): a size past what an
            // array can hold is NumPy's MemoryError from np.zeros(pool_size), refused before any entropy is drawn for it.
            if (pool_size > Array.MaxLength)
                throw new OutOfMemoryException($"Unable to allocate an entropy pool of {pool_size} uint32 words.");

            if (entropy is null)
            {
                // NumPy: entropy = randbits(pool_size * 32) — a Python int, kept as the reproducible entropy.
                var bytes = new byte[checked(pool_size * 4)];
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
            _spawnKey = ToSpawnKey(spawn_key);
            _nChildrenSpawned = n_children_spawned;

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
        ///     becomes its little-endian uint32 limbs; a string parses with <see cref="ParseSeedString"/>; a sequence
        ///     coerces each element and concatenates the pieces.
        /// </summary>
        /// <param name="x">The entropy, or one element of it.</param>
        /// <returns>The coerced piece.</returns>
        /// <exception cref="ValueError">A negative value (<c>expected non-negative integer</c>) or a string that is not a seed
        ///     literal.</exception>
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
                    return new EntropyPiece(IntToUint32Array(ParseSeedString(s)));
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

        /// <summary>
        ///     The string branch of NumPy's <c>_coerce_to_uint32_array</c>: a string that starts with <c>0x</c> is
        ///     <c>int(x, base=16)</c>; one that <c>DECIMAL_RE.match</c>es — <c>[0-9]+</c>, anchored at the START only — is
        ///     <c>int(x)</c> in base 10; anything else is <c>unrecognized seed string</c>.
        /// </summary>
        /// <param name="s">The seed string.</param>
        /// <returns>Its integer value (never negative: a sign is not a digit, so <c>"-5"</c> is unrecognized).</returns>
        /// <exception cref="ValueError">Not a seed literal: <c>unrecognized seed string</c> when it neither starts with <c>0x</c>
        ///     nor with a digit (<c>"a"</c>, <c>""</c>, <c>" 12"</c>, <c>"-5"</c>); Python's <c>invalid literal for int() with
        ///     base 10/16: …</c> when it does but the rest is not a literal (<c>"0b11"</c>, <c>"12a"</c>, <c>"0x"</c>).</exception>
        /// <remarks>
        ///     There is NO octal branch: <c>"012"</c> is decimal 12, because <c>int("012")</c> reads base 10 (a leading zero
        ///     is legal there). NumSharp used to read a leading-<c>0</c> string as octal, so <c>"012"</c> seeded from 10 and
        ///     every stream built on it differed from NumPy's. The prefix test is case-sensitive (<c>"0X10"</c> reaches the
        ///     base-10 parse and fails, as in NumPy), and the base-10 parse accepts Python's underscores and trailing
        ///     whitespace (<c>"1_000 "</c> is 1000) through <see cref="PythonInt.Parse"/>.
        /// </remarks>
        private static BigInteger ParseSeedString(string s)
        {
            if (s.StartsWith("0x", StringComparison.Ordinal))
                return PythonInt.Parse(s, 16);
            if (s.Length > 0 && s[0] >= '0' && s[0] <= '9')
                return PythonInt.Parse(s, 10);
            throw new ValueError("unrecognized seed string");
        }

        /// <summary>
        ///     NumPy's <c>tuple(spawn_key)</c>, flattened to the Python ints its coercion reads: integers are kept as
        ///     <see cref="BigInteger"/>s (negative ones too — <see cref="AssembleEntropy"/> rejects them exactly when NumPy's
        ///     <c>_coerce_to_uint32_array</c> does), strings parse with <see cref="ParseSeedString"/>, and nested sequences and
        ///     arrays contribute their elements in order.
        /// </summary>
        /// <param name="spawnKey">The key as passed; null is NumPy's default <c>()</c>.</param>
        /// <returns>The flat key.</returns>
        /// <exception cref="TypeError">Not iterable (<c>'int' object is not iterable</c> — NumPy's <c>tuple()</c> error), a 0-d
        ///     array key (<c>iteration over a 0-d array</c>), or an element the coercion refuses: a float (<c>seed must be
        ///     integer</c>), null (<c>object of type 'NoneType' has no len()</c>), a bool array
        ///     (<c>object of type 'numpy.bool' has no len()</c>), a 0-d array (<c>len() of unsized object</c>) or another
        ///     object without a length.</exception>
        /// <exception cref="ValueError">A string that is not a seed literal.</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy keeps the tuple exactly as given and coerces it — flattening every nested sequence into one run of uint32
        ///     words — each time the entropy is assembled. The words depend only on the element VALUES in order: a nested
        ///     <c>((1, 2),)</c> assembles to the same words as <c>(1, 2)</c>, and a child <c>((1, 2), 0)</c> to the same as
        ///     <c>(1, 2, 0)</c>. So storing the flattened values reproduces NumPy's pool — and every stream built on it or on
        ///     any descendant — while keeping <see cref="spawn_key"/> a plain tuple of Python ints, the only shape NumPy's own
        ///     <see cref="spawn"/> produces. The visible difference is the key's repr: NumPy shows <c>((1, 2),)</c> and
        ///     <c>('12',)</c> where this shows <c>(1, 2)</c> and <c>(12,)</c>.
        ///     </para>
        ///     <para>
        ///     One exotic input is accepted where NumPy refuses: a uint32 array nested INSIDE the key passes through NumPy's
        ///     coercion with its own shape, so a 0-d or multi-dimensional one fails its <c>np.concatenate</c>; here its
        ///     elements are flattened in C order like any other array's.
        ///     </para>
        /// </remarks>
        private static BigInteger[] ToSpawnKey(object spawnKey)
        {
            if (spawnKey is null)
                return Array.Empty<BigInteger>();
            var key = new List<BigInteger>();
            switch (spawnKey)
            {
                case string text:
                    // tuple("12") is ('1', '2'): a string key is a sequence of one-character seed strings.
                    foreach (char ch in text)
                        key.Add(ParseSeedString(ch.ToString()));
                    break;
                case NDArray nd:
                    // tuple(ndarray) iterates the first axis, which a 0-d array cannot; the elements (numpy scalars or
                    // rows) then take the ndarray coercion, whose words depend only on the C-order values.
                    if (nd.ndim == 0)
                        throw new TypeError("iteration over a 0-d array");
                    AppendArrayValues(nd, key);
                    break;
                case System.Collections.IEnumerable seq:
                    foreach (object v in seq)
                        AppendKeyElement(v, key);
                    break;
                default:
                    throw new TypeError($"'{PythonTypeName(spawnKey)}' object is not iterable");
            }
            return key.ToArray();
        }

        /// <summary>
        ///     One element of a spawn key, as NumPy's <c>_coerce_to_uint32_array</c> reads it: an integer (bool as 0/1), a
        ///     seed string, or a sized sequence / array whose elements follow in order.
        /// </summary>
        /// <param name="v">The element.</param>
        /// <param name="key">The flat key being built (appended to).</param>
        /// <exception cref="TypeError">A float/complex (<c>seed must be integer</c>), null or another unsized object
        ///     (<c>object of type '…' has no len()</c>), or an array NumPy's coercion refuses.</exception>
        /// <exception cref="ValueError">A string that is not a seed literal.</exception>
        private static void AppendKeyElement(object v, List<BigInteger> key)
        {
            switch (v)
            {
                case null:
                    throw new TypeError("object of type 'NoneType' has no len()");
                case string s:
                    key.Add(ParseSeedString(s));
                    return;
                case NDArray nd:
                    AppendArrayValues(nd, key);
                    return;
                case System.Collections.IList list:
                    // The coercion's else-branch: len(x), then every element — so nesting flattens in order.
                    foreach (object e in list)
                        AppendKeyElement(e, key);
                    return;
                default:
                    if (IsIntegerScalar(v))
                    {
                        key.Add(ToBigInteger(v));
                        return;
                    }
                    if (v is float or double or Half or decimal or Complex)
                        throw new TypeError("seed must be integer");
                    // Python's len() on an object that has none (a generator, an arbitrary class instance).
                    throw new TypeError($"object of type '{v.GetType().Name}' has no len()");
            }
        }

        /// <summary>
        ///     The values an array contributes to a spawn key — NumPy's ndarray coercion, read as values: the elements in C
        ///     order for any integer dtype.
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <param name="key">The flat key being built (appended to).</param>
        /// <exception cref="TypeError">A 0-d non-uint32 array (<c>len() of unsized object</c>), a non-empty bool array
        ///     (<c>object of type 'numpy.bool' has no len()</c>: <c>np.bool_</c> is not an integer type), or a non-empty
        ///     float/complex/decimal one (<c>seed must be integer</c>).</exception>
        /// <remarks>
        ///     The same rules as <see cref="CoerceNDArray"/>, which produces the entropy WORDS; this keeps the element values
        ///     so a large element stays one key entry (its words are identical either way).
        /// </remarks>
        private static void AppendArrayValues(NDArray nd, List<BigInteger> key)
        {
            NPTypeCode tc = nd.typecode;
            // A uint32 array passes the coercion untouched (so even 0-d reaches its values here); any other array needs a
            // length first.
            if (tc != NPTypeCode.UInt32 && nd.ndim == 0)
                throw new TypeError("len() of unsized object");
            // Iterating an empty array never reaches an element, so no element check fires (any dtype).
            if (nd.size == 0)
                return;
            switch (tc)
            {
                case NPTypeCode.Boolean:
                    throw new TypeError("object of type 'numpy.bool' has no len()");
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Char:
                    // GetAtIndex reads the logical C-order index through any layout (strided, transposed, broadcast).
                    for (long i = 0; i < nd.size; i++)
                        key.Add(ToBigInteger(nd.GetAtIndex(i)));
                    return;
                default:
                    throw new TypeError("seed must be integer");
            }
        }

        /// <summary>The Python type name NumPy's messages show for a non-iterable spawn key.</summary>
        /// <param name="x">The value.</param>
        /// <returns><c>int</c> for C# integers, <c>float</c> for floating types, <c>str</c> for a string, else the CLR name.</returns>
        private static string PythonTypeName(object x) => x switch
        {
            bool => "bool",
            string => "str",
            float or double or Half or decimal => "float",
            Complex => "complex",
            _ when IsIntegerScalar(x) => "int",
            _ => x.GetType().Name,
        };

        /// <summary>Validates a child count against NumPy's uint32 attribute.</summary>
        /// <param name="n">The count.</param>
        /// <returns>The count as uint32.</returns>
        /// <exception cref="OverflowException">Negative, or above 2**32 - 1 (NumPy's OverflowError texts).</exception>
        /// <remarks>
        ///     Cython converts a Python int to <c>uint32_t</c> through <c>unsigned long</c> where that type is wider, so the
        ///     too-large text is LP64 NumPy's <c>value too large to convert to uint32_t</c> — the platform NumSharp's legacy
        ///     integers follow (NumPy's Windows build, whose <c>unsigned long</c> IS 32-bit, reports
        ///     <c>Python int too large to convert to C unsigned long</c>). The negative text is the same on both.
        /// </remarks>
        private static uint ToUInt32Count(long n)
        {
            if (n < 0)
                throw new OverflowException("can't convert negative value to uint32_t");
            if (n > uint.MaxValue)
                throw new OverflowException("value too large to convert to uint32_t");
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
        ///     <c>generate_state(n_words, dtype=np.uint32)</c>): a 1-D uint32 (the default) or uint64 NDArray of
        ///     <paramref name="n_words"/> words — NumPy's ndarray return, not a CLR array.
        /// </summary>
        /// <param name="n_words">The number of words to produce (0 gives an empty result) — a Python int in NumPy, allocated as
        ///     <c>np.zeros(n_words)</c>, so any 64-bit count the allocator accepts.</param>
        /// <param name="dtype"><c>uint32</c> (default) or <c>uint64</c>; a uint64 word consumes two uint32 words of the stream.</param>
        /// <returns>The seeding words — the same words for the same pool every call (the sequence is not consumed).</returns>
        /// <exception cref="ValueError"><paramref name="n_words"/> is negative (NumPy's <c>np.zeros(n_words)</c> error) or too big to
        /// address (<c>array is too big; ...</c>) — for uint64 the DOUBLED count is what NumPy allocates, so one that leaves
        /// int64 is <c>Maximum allowed dimension exceeded</c> — or <paramref name="dtype"/> is not the NATIVE uint32/uint64
        /// (NumPy compares with <c>==</c>, so a byte-swapped <c>'&gt;u4'</c> is rejected too).</exception>
        /// <exception cref="OutOfMemoryException">A valid count that cannot be allocated (NumPy's <c>MemoryError</c>).</exception>
        /// <remarks>
        ///     NumPy fills <c>np.zeros(n_words * (2 if uint64 else 1), uint32)</c> from the hash cycle and views a uint64 request
        ///     little-endian (<c>astype('&lt;u4').view('&lt;u8')</c>); the uint64 array here is written through the same uint32
        ///     words, which on this little-endian runtime ARE that view. The words live in unmanaged storage, so the count is not
        ///     capped at a managed array's length.
        /// </remarks>
        public NDArray generate_state(long n_words, DType dtype = null)
        {
            // Structural equality (class + byte order), as NumPy's `dtype == np.dtype(np.uint32)`.
            NPTypeCode tc = dtype is null || dtype == DType.UInt32 ? NPTypeCode.UInt32
                : dtype == DType.UInt64 ? NPTypeCode.UInt64
                : NPTypeCode.Empty;
            if (tc == NPTypeCode.Empty)
                throw new ValueError("only support uint32 or uint64");
            // NumPy doubles a uint64 request BEFORE np.zeros (n_words *= 2 on a Python int): a doubled count past npy_intp
            // is the dimension converter's error, not a byte-size one — generate_state(2**62, uint64) reports this where a
            // single-width 2**62 reports "array is too big". The doubled count cannot be formed in a long, so the bound is
            // tested on n_words itself: for integers 2*n > long.MaxValue ⟺ n > long.MaxValue / 2 (MaxValue is odd, its
            // truncated half is 2**62 - 1), and 2*n < long.MinValue ⟺ n < long.MinValue / 2 (MinValue is even, -2**62 exact).
            if (tc == NPTypeCode.UInt64 && (n_words > long.MaxValue / 2 || n_words < long.MinValue / 2))
                throw new ValueError("Maximum allowed dimension exceeded");
            // Otherwise NumPy allocates `np.zeros(n_words, dtype=np.uint32)` (twice the words for uint64 — the same bytes as
            // n_words uint64s), so a negative or unaddressable count is the allocator's error, raised after the dtype check.
            var state = new NDArray(tc, new Shape(n_words), false);
            unsafe
            {
                FillState((uint*)state.Address, tc == NPTypeCode.UInt64 ? 2 * n_words : n_words);
            }
            return state;
        }

        /// <summary>
        ///     Writes <paramref name="count"/> uint32 seeding words — the pool cycled through NumPy's second hash
        ///     (<c>INIT_B</c> / <c>MULT_B</c>) — into <paramref name="dst"/>.
        /// </summary>
        /// <param name="dst">The destination (at least <paramref name="count"/> words).</param>
        /// <param name="count">The word count (non-negative).</param>
        private unsafe void FillState(uint* dst, long count)
        {
            uint hashConst = INIT_B;
            int cyc = 0;
            for (long i = 0; i < count; i++)
            {
                uint dataVal = _pool[cyc];
                if (++cyc == _pool.Length) cyc = 0;
                dataVal ^= hashConst;
                hashConst *= MULT_B;
                dataVal *= hashConst;
                dataVal ^= dataVal >> XSHIFT;
                dst[i] = dataVal;
            }
        }

        /// <summary>
        ///     Returns <paramref name="nWords"/> uint32 words for PRNG seeding (NumPy
        ///     <c>generate_state(n_words, np.uint32)</c>) as a managed array — the engines' seeding fast path, which needs a
        ///     handful of words, not an NDArray.
        /// </summary>
        /// <param name="nWords">The word count (non-negative).</param>
        /// <returns>The words.</returns>
        internal unsafe uint[] GenerateState(int nWords)
        {
            var state = new uint[nWords];
            fixed (uint* p = state)
                FillState(p, nWords);
            return state;
        }

        /// <summary>
        ///     Returns <paramref name="nWords"/> uint64 words for PRNG seeding (NumPy
        ///     <c>generate_state(n_words, np.uint64)</c> — twice as many uint32 words viewed little-endian) as a managed array.
        /// </summary>
        /// <param name="nWords">The word count (non-negative).</param>
        /// <returns>The words.</returns>
        internal unsafe ulong[] GenerateState64(int nWords)
        {
            var result = new ulong[nWords];
            fixed (ulong* p = result)
                FillState((uint*)p, 2L * nWords); // little-endian: word 2i is the low half of result[i]
            return result;
        }

        // ---- spawning ----

        /// <summary>
        ///     Spawn <paramref name="n_children"/> child sequences by extending <see cref="spawn_key"/> (NumPy's
        ///     <c>SeedSequence.spawn</c>): child <c>i</c> gets this sequence's entropy and pool size and the key
        ///     <c>spawn_key + (n_children_spawned + i,)</c>, and <see cref="n_children_spawned"/> advances, so later
        ///     calls continue the numbering and never repeat a child.
        /// </summary>
        /// <param name="n_children">The number of children (a C# <c>int</c>: the result is an array, which cannot hold more).</param>
        /// <returns>The children (empty for 0).</returns>
        /// <exception cref="OverflowException">The child count would leave the uint32 range NumPy stores it in (a negative
        /// <paramref name="n_children"/> included — NumPy raises after an empty loop, and no children are returned).</exception>
        /// <remarks>
        ///     The overflow is refused BEFORE any child is built, which is where NumPy 2.4.2 diverges: its loop counter is a
        ///     <c>uint32_t</c> compared against a wider bound, so a spawn that would reach <c>2**32</c> children wraps the
        ///     counter to 0 and never ends — it appends children until <c>MemoryError</c> (probed: both
        ///     <c>SeedSequence(1, n_children_spawned=2**32 - 1).spawn(1)</c> and <c>spawn(2**32)</c>). The OverflowError
        ///     raised here is the one NumPy's <c>n_children_spawned += n_children</c> would raise if its loop terminated; below
        ///     the limit the children and the advanced count are NumPy's exactly.
        /// </remarks>
        public SeedSequence[] spawn(int n_children)
        {
            long start = _nChildrenSpawned;
            uint next = ToUInt32Count(start + n_children); // validated first: NumPy's loop is empty for n < 0, then += raises
            var seqs = new SeedSequence[Math.Max(0, n_children)];
            for (int c = 0; c < seqs.Length; c++)
            {
                var key = new BigInteger[_spawnKey.Length + 1];
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
        private static string ReprTuple(BigInteger[] t)
            => t.Length == 1 ? $"({t[0].ToString(CultureInfo.InvariantCulture)},)"
                : "(" + string.Join(", ", Array.ConvertAll(t, v => v.ToString(CultureInfo.InvariantCulture))) + ")";
    }
}
