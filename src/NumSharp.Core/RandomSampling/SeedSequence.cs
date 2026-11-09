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
        ///     0/1); a flat sequence of them — a C# array or <see cref="System.Collections.IList"/> whose string elements
        ///     parse NumPy's way (<see cref="ParseSeedString"/>: a <c>"0x…"</c> string through <c>int(x, 16)</c>, a string
        ///     that starts with a decimal digit through <c>int(x)</c>, anything else <c>unrecognized seed string</c>); or a
        ///     1-D integer or text <see cref="NDArray"/> (a uint32 array passes through as-is, a <c>uint[]</c> standing for
        ///     one). A NESTED sequence is refused since NumPy 2.5 (see <see cref="HasPythonLen"/>).
        /// </param>
        /// <param name="spawn_key">This sequence's position in a spawn tree (mixed in as extra entropy); null or empty for a
        ///     root. Any iterable — NumPy's <c>tuple(spawn_key)</c> — such as a <c>long[]</c>, <c>int[]</c>, <c>uint[]</c>,
        ///     <see cref="BigInteger"/><c>[]</c>, a list or a 1-D integer <see cref="NDArray"/>; string elements parse with
        ///     the entropy's string rule (a string key is itself a sequence of one-character strings, as <c>tuple("12")</c>
        ///     is). Nested elements are refused, as for the entropy (see <see cref="ToSpawnKey"/>).</param>
        /// <param name="pool_size">The pool size in uint32 words (at least 4; 8 gives a 256-bit pool) — NumPy's <c>Py_ssize_t</c>.</param>
        /// <param name="n_children_spawned">The number of children already spawned — only when rebuilding a serialized
        ///     sequence. NumPy stores it as a <c>uint32_t</c>, so the parameter is one.</param>
        /// <exception cref="ValueError"><paramref name="pool_size"/> is below 4; an entropy or spawn-key value is negative
        ///     (<c>expected non-negative integer</c>); a string that is not a seed literal (<c>unrecognized seed string</c>, or
        ///     Python's <c>invalid literal for int() …</c> for one that starts like a number).</exception>
        /// <exception cref="TypeError"><paramref name="entropy"/> is not an integer or a sequence of integers (a float, a string,
        /// a float/bool array, a 0-d array, a null element); either holds a nested sequence — a sequence, array or
        /// fixed-width bytes element, or a multi-dimensional array (NumPy 2.5's <c>SeedSequence does not accept nested
        /// sequences.</c>); <paramref name="spawn_key"/> is not iterable (NumPy's <c>'int' object is not iterable</c>), is a
        /// 0-d array (<c>iteration over a 0-d array</c>) or holds an element the coercion refuses (a float: <c>seed must be
        /// integer</c>; null: <c>object of type 'NoneType' has no len()</c>). The checks run in NumPy's order: the key's
        /// iterability, then the entropy, then the key's elements.</exception>
        /// <exception cref="OutOfMemoryException"><paramref name="pool_size"/> is larger than an array can hold (NumPy's
        ///     <c>MemoryError</c> from <c>np.zeros(pool_size)</c>).</exception>
        public SeedSequence(object entropy, object spawn_key = null, long pool_size = DEFAULT_POOL_SIZE, uint n_children_spawned = 0)
        {
            if (pool_size < DEFAULT_POOL_SIZE)
                throw new ValueError(string.Create(CultureInfo.InvariantCulture, $"The size of the entropy pool should be at least {DEFAULT_POOL_SIZE}"));
            // The pool is a managed uint[] (it is mixed word by word and read by every generate_state): a size past what an
            // array can hold is NumPy's MemoryError from np.zeros(pool_size), refused before any entropy is drawn for it.
            if (pool_size > Array.MaxLength)
                throw new OutOfMemoryException(string.Create(CultureInfo.InvariantCulture, $"Unable to allocate an entropy pool of {pool_size} uint32 words."));

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
            // NumPy's `self.spawn_key = tuple(spawn_key)` runs here, so only the key's ITERATION can fail now; its elements
            // are coerced after the entropy (get_assembled_entropy coerces self.entropy first), so an invalid entropy is
            // reported before an invalid key element.
            CheckSpawnKeyIterable(spawn_key);
            _nChildrenSpawned = n_children_spawned;

            _pool = new uint[pool_size];
            EntropyPiece run = CoercePiece(_entropy);
            _spawnKey = ToSpawnKey(spawn_key);
            MixEntropy(_pool, AssembleEntropy(run));
        }

        // ---- entropy coercion (numpy get_assembled_entropy / _coerce_to_uint32_array / _int_to_uint32_array) ----

        /// <summary>
        ///     NumPy's <c>get_assembled_entropy</c>: the run entropy and the spawn key as uint32 words, the run entropy
        ///     zero-padded to the pool size when a spawn key follows (so a spawned child cannot collide with a root seeded
        ///     from a longer entropy — the gh-16539 fix, applied only when a key is present to keep root streams stable).
        /// </summary>
        /// <param name="run">The run entropy, already coerced (<see cref="CoercePiece"/> of the entropy: the constructor
        ///     coerces it before the spawn key's elements, NumPy's error order).</param>
        /// <returns>The words to mix into the pool.</returns>
        /// <exception cref="ValueError">A passed-through uint32 array that is 0-d or not 1-D (NumPy's <c>np.concatenate</c>
        /// errors).</exception>
        /// <exception cref="TypeError">A 0-d uint32 array when a spawn key is present (<c>len() of unsized object</c>).</exception>
        private uint[] AssembleEntropy(EntropyPiece run)
        {
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
        /// <param name="list">The sequence (a C# array — a multi-dimensional one enumerates its elements in C order, as a
        ///     flat list would — or list).</param>
        /// <returns>The concatenated piece.</returns>
        /// <exception cref="ValueError">An element failed, or the pieces cannot be concatenated.</exception>
        /// <exception cref="TypeError">An element is not an integer, or is itself a sequence (<see cref="HasPythonLen"/>:
        ///     <c>SeedSequence does not accept nested sequences.</c>, checked before the element is coerced).</exception>
        private static EntropyPiece CoerceSequence(System.Collections.IList list)
        {
            if (list.Count == 0)
                return new EntropyPiece(Array.Empty<uint>());
            var pieces = new List<EntropyPiece>(list.Count);
            foreach (object v in list)
            {
                // NumPy 2.5's statement before the recursion (gh-31857: a self-referencing list recursed until it crashed).
                if (HasPythonLen(v))
                    throw new TypeError(NestedSequences);
                pieces.Add(CoercePiece(v));
            }
            return Concatenate(pieces);
        }

        /// <summary>NumPy 2.5's refusal of a sequence element that is itself a sequence (<c>_coerce_to_uint32_array</c>).</summary>
        private const string NestedSequences = "SeedSequence does not accept nested sequences.";

        /// <summary>
        ///     NumPy's nested test <c>hasattr(v, '__len__') and not isinstance(v, str)</c> for a C# value: a C# array or any
        ///     other collection (a list, tuple, <c>bytes</c> or <c>range</c> to Python) and every <see cref="NDArray"/> — even a
        ///     0-d one, whose TYPE has <c>__len__</c> although calling it fails — but not a string.
        /// </summary>
        /// <param name="v">A sequence element.</param>
        /// <returns>True when NumPy refuses the element as a nested sequence.</returns>
        private static bool HasPythonLen(object v) => v is NDArray || (v is System.Collections.IEnumerable && v is not string);

        /// <summary>
        ///     The <see cref="NDArray"/> branch of the coercion: a uint32 array passes through with its shape; any other
        ///     array takes the sequence path, which starts with <c>len(x)</c> (an empty first axis is an empty piece) and then
        ///     iterates its first axis — a row of a multi-dimensional array is a nested sequence, an integer element is its
        ///     limbs, a <c>U</c> / <c>StringDType</c> element parses as a seed string (<see cref="AppendTextElement"/>), an
        ///     <c>S</c> element (<c>np.bytes_</c>, which has a length) is a nested sequence.
        /// </summary>
        /// <param name="nd">The entropy array.</param>
        /// <returns>The uint32 piece.</returns>
        /// <exception cref="TypeError">0-d (<c>len() of unsized object</c>), a non-empty first axis of a multi-dimensional or
        /// <c>S</c> array (NumPy 2.5's <c>SeedSequence does not accept nested sequences.</c>), non-empty bool (<c>object of
        /// type 'numpy.bool' has no len()</c>), or non-empty non-integer (<c>seed must be integer</c>).</exception>
        /// <exception cref="ValueError">A negative element, or a <c>U</c> element that is not a seed literal
        ///     (<c>unrecognized seed string</c>, or <c>invalid literal for int() with base 10: np.str_('1a')</c>).</exception>
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
            // `len(x) == 0`: an empty first axis never reaches an element, so no element check fires (any dtype; a (0, 2)
            // array too, whose rows would be nested).
            if (nd.Shape.dimensions[0] == 0)
                return new EntropyPiece(Array.Empty<uint>());
            // The first element of a multi-dimensional array is a row, an ndarray: NumPy 2.5 refuses it as a nested
            // sequence before looking at its dtype (a (2, 0) array included).
            if (nd.ndim > 1)
                throw new TypeError(NestedSequences);
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
                    // The 1-D walk; GetAtIndex reads the logical index through any layout, so no raveled temporary is needed.
                    var all = new List<uint>();
                    for (long i = 0; i < nd.size; i++)
                        all.AddRange(IntToUint32Array(ToBigInteger(nd.GetAtIndex(i))));
                    return new EntropyPiece(all.ToArray());
                case NPTypeCode.String:
                {
                    // An S element is an np.bytes_, which has a length: NumPy 2.5 refuses it as a nested sequence at the
                    // first element (NumPy 2.4 iterated its bytes as integer words).
                    if (ReferenceEquals(nd.dtype.Meta, DTypeRegistry.Bytes))
                        throw new TypeError(NestedSequences);
                    // The 1-D walk; each element is a string scalar (np.str_ or a StringDType str) whose words come from its
                    // parsed text. Every piece is 1-D, so the concatenation is the flat word list.
                    var values = new List<BigInteger>();
                    bool numpyStr = !Backends.Strings.NDStringUfuncs.IsVString(nd);
                    for (long i = 0; i < nd.size; i++)
                        AppendTextElement(nd.GetAtIndex(i), values, numpyStr);
                    var textWords = new List<uint>(values.Count);
                    foreach (BigInteger v in values)
                        textWords.AddRange(IntToUint32Array(v));
                    return new EntropyPiece(textWords.ToArray());
                }
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
                shape[d] = nd.Shape.dimensions[d];
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
                    throw new ValueError(string.Create(CultureInfo.InvariantCulture, $"all the input arrays must have same number of dimensions, but the array at index 0 has {ndim} dimension(s) and the array at index {i} has {p.NDim} dimension(s)"));
                for (int d = 0; d < ndim; d++)
                {
                    if (d == 0)
                        shape[0] += p.Shape[0];
                    else if (shape[d] != p.Shape[d])
                        throw new ValueError(string.Create(CultureInfo.InvariantCulture, $"all the input array dimensions except for the concatenation axis must match exactly, but along dimension {d}, the array at index 0 has size {shape[d]} and the array at index {i} has size {p.Shape[d]}"));
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
        /// <param name="numpyStr">True for an element of a <c>U</c> array (an <c>np.str_</c> scalar in NumPy): only the
        ///     invalid-literal text differs, quoting <c>np.str_('…')</c> where a plain string quotes <c>'…'</c>.</param>
        /// <returns>Its integer value (never negative: a sign is not a digit, so <c>"-5"</c> is unrecognized).</returns>
        /// <exception cref="ValueError">Not a seed literal: <c>unrecognized seed string</c> when it neither starts with <c>0x</c>
        ///     nor with an ASCII digit (<c>"a"</c>, <c>""</c>, <c>" 12"</c>, <c>"-5"</c>, <c>"٣"</c>); Python's <c>invalid
        ///     literal for int() with base 10/16: …</c> when it does but the rest is not a literal (<c>"0b11"</c>,
        ///     <c>"12a"</c>, <c>"0x"</c>); Python's digit-limit error for more than 4300 decimal digits.</exception>
        /// <remarks>
        ///     There is NO octal branch: <c>"012"</c> is decimal 12, because <c>int("012")</c> reads base 10 (a leading zero
        ///     is legal there). NumSharp used to read a leading-<c>0</c> string as octal, so <c>"012"</c> seeded from 10 and
        ///     every stream built on it differed from NumPy's. The prefix test is case-sensitive (<c>"0X10"</c> reaches the
        ///     base-10 parse and fails, as in NumPy), and the parse is Python's <c>int()</c> (<see cref="PythonInt.Parse"/>):
        ///     underscores and trailing whitespace (<c>"1_000 "</c> is 1000) and, after the leading ASCII digit the regex
        ///     demands, decimal digits of any script (<c>"1٢"</c> is 12).
        /// </remarks>
        private static BigInteger ParseSeedString(string s, bool numpyStr = false)
        {
            if (s.StartsWith("0x", StringComparison.Ordinal))
                return PythonInt.Parse(s, 16, numpyStr);
            // DECIMAL_RE is [0-9]+ matched at the start: an ASCII digit only, whatever the rest holds.
            if (s.Length > 0 && s[0] >= '0' && s[0] <= '9')
                return PythonInt.Parse(s, 10, numpyStr);
            throw new ValueError("unrecognized seed string");
        }

        /// <summary>
        ///     The words one element of a <c>U</c> or <c>StringDType</c> array contributes to NumPy's coercion: the element is
        ///     a string (an <c>np.str_</c> scalar or a Python str), so it parses as a seed string. (An <c>S</c> array never
        ///     gets here: its <c>np.bytes_</c> elements have a length, which NumPy 2.5 refuses as a nested sequence.)
        /// </summary>
        /// <param name="element">The element as <see cref="NDArray.GetAtIndex(long)"/> returns it: a <see cref="string"/>; for a
        ///     <c>StringDType</c> NA element its NA object (null for None, a double for a NaN-like NA).</param>
        /// <param name="words">The flat key or word list being built (appended to).</param>
        /// <param name="numpyStr">True for an element of a <c>U</c> array (an <c>np.str_</c> scalar, whose repr NumPy's int()
        ///     error quotes), false for a <c>StringDType</c> element (a Python str).</param>
        /// <exception cref="ValueError">A text element that is not a seed literal (<see cref="ParseSeedString"/>).</exception>
        /// <exception cref="TypeError">A <c>StringDType</c> NA element: None has no length (<c>object of type 'NoneType' has no
        ///     len()</c>), a NaN-like NA is a float (<c>seed must be integer</c>); any other non-string NA object has no length
        ///     either.</exception>
        private static void AppendTextElement(object element, List<BigInteger> words, bool numpyStr = true)
        {
            if (element is string text)
            {
                words.Add(ParseSeedString(text, numpyStr));
                return;
            }
            // A StringDType NA element: _coerce_to_uint32_array(None) takes len(None); a NaN-like NA is a float.
            if (element is null)
                throw new TypeError("object of type 'NoneType' has no len()");
            if (element is double or float)
                throw new TypeError("seed must be integer");
            throw new TypeError($"object of type '{element.GetType().Name}' has no len()");
        }

        /// <summary>
        ///     The part of NumPy's <c>self.spawn_key = tuple(spawn_key)</c> that can fail: the key must be iterable, and an
        ///     array key must have an axis to iterate. Runs before the entropy is coerced, as in NumPy's constructor.
        /// </summary>
        /// <param name="spawnKey">The key as passed; null is NumPy's default <c>()</c>.</param>
        /// <exception cref="TypeError">Not iterable (<c>'int' object is not iterable</c> — NumPy's <c>tuple()</c> error) or a
        ///     0-d array key (<c>iteration over a 0-d array</c>).</exception>
        private static void CheckSpawnKeyIterable(object spawnKey)
        {
            switch (spawnKey)
            {
                case null:
                case string:
                    return;
                case NDArray nd:
                    if (nd.ndim == 0)
                        throw new TypeError("iteration over a 0-d array");
                    return;
                case System.Collections.IEnumerable:
                    return;
                default:
                    throw new TypeError($"'{PythonTypeName(spawnKey)}' object is not iterable");
            }
        }

        /// <summary>
        ///     NumPy's <c>tuple(spawn_key)</c>, coerced to the Python ints its coercion reads: integers are kept as
        ///     <see cref="BigInteger"/>s (negative ones too — <see cref="AssembleEntropy"/> rejects them exactly when NumPy's
        ///     <c>_coerce_to_uint32_array</c> does) and strings parse with <see cref="ParseSeedString"/>; a nested element is
        ///     refused (NumPy 2.5).
        /// </summary>
        /// <param name="spawnKey">The key as passed (already through <see cref="CheckSpawnKeyIterable"/>); null is NumPy's
        ///     default <c>()</c>.</param>
        /// <returns>The key's values.</returns>
        /// <exception cref="TypeError">An element the coercion refuses: a sequence, array or <c>S</c> array element (NumPy 2.5's
        ///     <c>SeedSequence does not accept nested sequences.</c>; so is every row of a multi-dimensional array key), a
        ///     float (<c>seed must be integer</c>), null (<c>object of type 'NoneType' has no len()</c>), a bool array element
        ///     (<c>object of type 'numpy.bool' has no len()</c>) or another object without a length.</exception>
        /// <exception cref="ValueError">A string that is not a seed literal.</exception>
        /// <remarks>
        ///     NumPy keeps the tuple exactly as given and coerces it each time the entropy is assembled; the words depend only
        ///     on the element VALUES in order, so storing the values reproduces NumPy's pool — and every stream built on it or
        ///     on any descendant — while keeping <see cref="spawn_key"/> a plain tuple of Python ints, the only shape NumPy's
        ///     own <see cref="spawn"/> produces. The visible difference is the key's repr for a string element: NumPy shows
        ///     <c>('12',)</c> where this shows <c>(12,)</c>.
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
                    // tuple(ndarray) iterates the first axis (CheckSpawnKeyIterable refused a 0-d array): numpy scalars of a
                    // 1-D array, rows (nested) of a multi-dimensional one.
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
        ///     One element of a spawn key, as NumPy's <c>_coerce_to_uint32_array</c> reads it: an integer (bool as 0/1) or a
        ///     seed string; anything with a length is a nested sequence.
        /// </summary>
        /// <param name="v">The element.</param>
        /// <param name="key">The flat key being built (appended to).</param>
        /// <exception cref="TypeError">A sequence or array (NumPy 2.5's <c>SeedSequence does not accept nested
        ///     sequences.</c>), a float/complex (<c>seed must be integer</c>), or null or another unsized object
        ///     (<c>object of type '…' has no len()</c>).</exception>
        /// <exception cref="ValueError">A string that is not a seed literal.</exception>
        private static void AppendKeyElement(object v, List<BigInteger> key)
        {
            // NumPy 2.5 refuses an element with a length before coercing it (a 0-d array included); NumPy 2.4 flattened it.
            if (HasPythonLen(v))
                throw new TypeError(NestedSequences);
            switch (v)
            {
                case null:
                    throw new TypeError("object of type 'NoneType' has no len()");
                case string s:
                    key.Add(ParseSeedString(s));
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
        ///     The values an array KEY contributes — NumPy's <c>tuple(key)</c> then the coercion of each element: the elements
        ///     of a 1-D integer array; a <c>U</c> / <c>StringDType</c> element's parsed seed string
        ///     (<see cref="AppendTextElement"/>). A row of a multi-dimensional array and an <c>S</c> element are nested.
        /// </summary>
        /// <param name="nd">The key array (at least 1-D: <see cref="CheckSpawnKeyIterable"/> refused a 0-d one).</param>
        /// <param name="key">The flat key being built (appended to).</param>
        /// <exception cref="TypeError">A non-empty first axis of a multi-dimensional or <c>S</c> array (NumPy 2.5's <c>SeedSequence
        ///     does not accept nested sequences.</c>), a non-empty bool array (<c>object of type 'numpy.bool' has no
        ///     len()</c>: <c>np.bool_</c> is not an integer type), or a non-empty float/complex/decimal one (<c>seed must be
        ///     integer</c>).</exception>
        /// <exception cref="ValueError">A <c>U</c> element that is not a seed literal.</exception>
        /// <remarks>
        ///     The same rules as <see cref="CoerceNDArray"/>, which produces the entropy WORDS; this keeps the element values
        ///     so a large element stays one key entry (its words are identical either way).
        /// </remarks>
        private static void AppendArrayValues(NDArray nd, List<BigInteger> key)
        {
            NPTypeCode tc = nd.typecode;
            // tuple() of an empty first axis is (): no element, so no element check fires (any dtype, any rank).
            if (nd.Shape.dimensions[0] == 0)
                return;
            // The elements of a multi-dimensional array are rows (ndarrays): nested.
            if (nd.ndim > 1)
                throw new TypeError(NestedSequences);
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
                case NPTypeCode.String:
                    // An S element is an np.bytes_ (it has a length): nested. NumPy keeps the U / StringDType scalars in the
                    // key and coerces them each time; their words are the parsed value, so the key stores exactly that.
                    if (ReferenceEquals(nd.dtype.Meta, DTypeRegistry.Bytes))
                        throw new TypeError(NestedSequences);
                    bool keyNumpyStr = !Backends.Strings.NDStringUfuncs.IsVString(nd);
                    for (long i = 0; i < nd.size; i++)
                        AppendTextElement(nd.GetAtIndex(i), key, keyNumpyStr);
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
        /// <exception cref="ValueError"><paramref name="n_children"/> is negative: NumPy 2.5's <c>n_children must be
        /// non-negative</c>, raised before anything changes (NumPy 2.4 ran an empty loop and raised the uint32
        /// OverflowError from its count update).</exception>
        /// <exception cref="OverflowException">The child count would leave the uint32 range NumPy stores it in.</exception>
        /// <remarks>
        ///     The overflow is refused BEFORE any child is built, which is where NumPy 2.5.4 diverges: its loop counter is a
        ///     <c>uint32_t</c> compared against a wider bound, so a spawn that would reach <c>2**32</c> children wraps the
        ///     counter to 0 and never ends — it appends children until <c>MemoryError</c> (probed: both
        ///     <c>SeedSequence(1, n_children_spawned=2**32 - 1).spawn(1)</c> and <c>spawn(2**32)</c>). The OverflowError
        ///     raised here is the one NumPy's <c>n_children_spawned += n_children</c> would raise if its loop terminated; below
        ///     the limit the children and the advanced count are NumPy's exactly.
        /// </remarks>
        public SeedSequence[] spawn(int n_children)
        {
            // NumPy 2.5's first statement: the count is checked before the loop and the uint32 update.
            if (n_children < 0)
                throw new ValueError("n_children must be non-negative");
            long start = _nChildrenSpawned;
            uint next = ToUInt32Count(start + n_children); // validated before any child exists (see the remarks)
            var seqs = new SeedSequence[n_children];
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
