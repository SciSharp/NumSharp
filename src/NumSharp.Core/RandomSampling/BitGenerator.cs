using System;

namespace NumSharp
{
    /// <summary>
    ///     Base class for the pseudo-random bit generators that drive <see cref="Generator"/> and the legacy
    ///     <see cref="NumPyRandom"/> (<c>RandomState</c>).
    /// </summary>
    /// <remarks>
    ///     Mirrors NumPy 2.4.2's <c>numpy.random.BitGenerator</c> (<c>bit_generator.pyx</c>): a source of uniform
    ///     64-/32-bit words plus the <c>[0,1)</c> float conversions the distribution kernels are built on, with
    ///     NumPy's public surface — <see cref="random_raw"/>, <see cref="state"/>, <see cref="seed_seq"/> and
    ///     <see cref="@lock"/>. The per-draw primitives (<c>NextUInt64</c> …) are the internal engine contract
    ///     (NumPy's <c>bitgen_t</c> function pointers), so they are <c>internal</c> and the public surface stays
    ///     NumPy-cased. Concrete generators: <see cref="PCG64"/> (the <c>default_rng</c> default),
    ///     <see cref="PCG64DXSM"/> (NumPy's recommendation for massively parallel work), <see cref="Philox"/> (a
    ///     counter-based generator addressable by key/counter), <see cref="SFC64"/> (Chris Doty-Humphrey's small fast
    ///     chaotic generator) and <see cref="MT19937"/> (the legacy RandomState's engine). Each is seeded through an
    ///     <see cref="ISeedSequence"/> (normally a <see cref="SeedSequence"/>) and <see cref="spawn"/>s independent
    ///     children from it.
    /// </remarks>
    public abstract class BitGenerator
    {
        /// <summary>
        ///     The seed sequence this bit generator was seeded from, or null once it was re-seeded in a way that
        ///     discards it (NumPy's <c>MT19937._legacy_seeding</c> and <c>Philox(key=...)</c> set <c>_seed_seq = None</c>).
        /// </summary>
        private protected ISeedSequence _seedSeq;

        /// <summary>Initializes the shared state of a bit generator.</summary>
        /// <param name="seedSeq">The seed sequence the subclass seeds itself from (null for a generator seeded without one).</param>
        private protected BitGenerator(ISeedSequence seedSeq)
        {
            _seedSeq = seedSeq;
        }

        /// <summary>
        ///     The lock every draw from this bit generator holds (NumPy's <c>BitGenerator.lock</c>).
        /// </summary>
        /// <remarks>
        ///     A <see cref="Generator"/> holds this monitor for the whole of a fill, exactly as NumPy's
        ///     <c>with self.lock:</c> does. The engine state (PCG64's two 128-bit words plus its buffered
        ///     32-bit half, MT19937's 624-word key and position) is updated non-atomically, so two threads drawing
        ///     at once tear it and emit values that belong to no seed's stream — measured at 72–99% foreign values
        ///     for four threads before the lock existed. Generators built over the SAME bit generator share this
        ///     object, so their draws serialize together, as NumPy's <c>Generator.lock = bit_generator.lock</c> makes
        ///     them. The monitor is re-entrant (like NumPy's <c>RLock</c>), so a draw that goes through another
        ///     public method (e.g. <c>choice</c> calling <c>integers</c>) re-acquires it without deadlock. Code that
        ///     calls the internal primitives directly must hold it too.
        /// </remarks>
        public object @lock { get; } = new object();

        /// <summary>
        ///     The seed sequence used to initialize the bit generator (NumPy's <c>BitGenerator.seed_seq</c>) — normally a
        ///     <see cref="SeedSequence"/> (cast to read its <c>entropy</c> / <c>spawn_key</c>).
        /// </summary>
        /// <remarks>
        ///     Null for a bit generator re-seeded with <see cref="MT19937._legacy_seeding()"/> or built from an explicit
        ///     <see cref="Philox"/> key (NumPy then reports <c>None</c> too). A <c>jumped()</c> copy carries a FRESH
        ///     OS-entropy sequence, as NumPy's <c>self.__class__()</c> construction gives it — not the parent's.
        /// </remarks>
        public ISeedSequence seed_seq => _seedSeq;

        /// <summary>
        ///     Create new independent child bit generators of this generator's type (NumPy's <c>BitGenerator.spawn</c>),
        ///     each seeded from a child of <see cref="seed_seq"/> — the recommended way to get non-overlapping streams
        ///     for parallel work.
        /// </summary>
        /// <param name="n_children">The number of children.</param>
        /// <returns>The children, of this bit generator's concrete type (NumPy's <c>type(self)(seed=s)</c>).</returns>
        /// <exception cref="TypeError">The seed sequence cannot spawn — e.g. none (legacy-seeded MT19937, keyed Philox): NumPy's
        /// <c>The underlying SeedSequence does not implement spawning.</c></exception>
        /// <exception cref="OverflowException">The seed sequence's child count would leave its uint32 range.</exception>
        /// <remarks>Spawning advances the parent seed sequence's <c>n_children_spawned</c>, so repeated calls never repeat a child.</remarks>
        public BitGenerator[] spawn(int n_children)
        {
            if (_seedSeq is not ISpawnableSeedSequence spawnable)
                throw new TypeError("The underlying SeedSequence does not implement spawning.");
            ISpawnableSeedSequence[] children = spawnable.spawn(n_children);
            var result = new BitGenerator[children.Length];
            for (int i = 0; i < children.Length; i++)
                result[i] = CreateFromSeed(children[i]);
            return result;
        }

        /// <summary>Constructs a new instance of this bit generator's concrete type seeded from <paramref name="seed"/> (NumPy's <c>type(self)(seed=seed)</c>).</summary>
        /// <param name="seed">The seed sequence.</param>
        /// <returns>The new, independently seeded generator.</returns>
        private protected abstract BitGenerator CreateFromSeed(ISeedSequence seed);

        /// <summary>
        ///     Reads <paramref name="n_words"/> uint32 seeding words from any <see cref="ISeedSequence"/> (the fast path for
        ///     <see cref="SeedSequence"/>, the interface contract — a <c>uint[]</c> — otherwise).
        /// </summary>
        /// <param name="seed">The seed sequence.</param>
        /// <param name="n_words">The word count.</param>
        /// <returns>The words.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="seed"/> is null.</exception>
        /// <exception cref="TypeError">A custom sequence returned something other than a <c>uint[]</c>.</exception>
        private protected static uint[] SeedWords32(ISeedSequence seed, int n_words)
        {
            if (seed is null)
                throw new ArgumentNullException(nameof(seed));
            if (seed is SeedSequence ss)
                return ss.GenerateState(n_words);
            return seed.generate_state(n_words, DType.UInt32) as uint[]
                   ?? throw new TypeError("generate_state(n_words, np.uint32) must return a uint32 array");
        }

        /// <summary>
        ///     Reads <paramref name="n_words"/> uint64 seeding words from any <see cref="ISeedSequence"/> (the fast path for
        ///     <see cref="SeedSequence"/>, the interface contract — a <c>ulong[]</c> — otherwise).
        /// </summary>
        /// <param name="seed">The seed sequence.</param>
        /// <param name="n_words">The word count.</param>
        /// <returns>The words.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="seed"/> is null.</exception>
        /// <exception cref="TypeError">A custom sequence returned something other than a <c>ulong[]</c>.</exception>
        private protected static ulong[] SeedWords64(ISeedSequence seed, int n_words)
        {
            if (seed is null)
                throw new ArgumentNullException(nameof(seed));
            if (seed is SeedSequence ss)
                return ss.GenerateState64(n_words);
            return seed.generate_state(n_words, DType.UInt64) as ulong[]
                   ?? throw new TypeError("generate_state(n_words, np.uint64) must return a uint64 array");
        }

        /// <summary>
        ///     Gets or sets the bit generator's state (NumPy's <c>bit_generator.state</c> dict, as a typed
        ///     <see cref="BitGeneratorState"/> whose members are named after the dict's keys).
        /// </summary>
        /// <remarks>
        ///     The getter returns a fresh snapshot; setting it copies the snapshot's values in, so a snapshot can be
        ///     restored any number of times. Each concrete generator also exposes a typed <c>state</c> property of
        ///     its own state class. Reading or writing holds <see cref="@lock"/>.
        /// </remarks>
        /// <exception cref="TypeError">Setting null (NumPy: <c>state must be a dict</c>).</exception>
        /// <exception cref="ValueError">Setting another generator's state (NumPy: <c>state must be for a &lt;name&gt; RNG/PRNG</c>).</exception>
        public BitGeneratorState state
        {
            get
            {
                lock (@lock)
                    return GetStateCore();
            }
            set
            {
                if (value is null)
                    throw new TypeError("state must be a dict");
                lock (@lock)
                    SetStateCore(value);
            }
        }

        /// <summary>Snapshots the engine state.</summary>
        /// <returns>A fresh state object of the concrete generator's state type.</returns>
        private protected abstract BitGeneratorState GetStateCore();

        /// <summary>Restores the engine state from a snapshot (validated against this generator's type).</summary>
        /// <param name="value">The non-null snapshot.</param>
        /// <exception cref="ValueError">The snapshot belongs to another bit generator type.</exception>
        private protected abstract void SetStateCore(BitGeneratorState value);

        /// <summary>
        ///     Return randoms as generated by the underlying bit generator (NumPy's <c>BitGenerator.random_raw</c>).
        /// </summary>
        /// <param name="size">Output shape; default (NumPy's <c>None</c>) returns a single value as a 0-d array.</param>
        /// <param name="output">When false the draws are made but discarded and null is returned (NumPy's benchmark switch).</param>
        /// <returns>
        ///     The raw words as <c>uint64</c> — a 0-d array for <c>size=None</c>, else an array of <paramref name="size"/>;
        ///     null when <paramref name="output"/> is false. Every raw value is widened to 64 bits irrespective of the
        ///     number of bits the engine produces (MT19937's raw words are 32-bit).
        /// </returns>
        /// <exception cref="ValueError"><paramref name="size"/> has a negative dimension (with <paramref name="output"/> true).</exception>
        /// <exception cref="TypeError"><paramref name="size"/> is <c>()</c> with <paramref name="output"/> false — NumPy sums an empty
        /// float array and cannot use the float as a count.</exception>
        /// <remarks>
        ///     With <paramref name="output"/> false NumPy draws <c>np.asarray(size).sum()</c> values — the SUM of the
        ///     dimensions, not their product (<c>(3, 4)</c> draws 7, <c>(0, 3)</c> draws 3, <c>-1</c> draws none). That
        ///     quirk decides how far the stream advances, so it is reproduced. The whole call holds <see cref="@lock"/>.
        /// </remarks>
        public unsafe NDArray random_raw(Shape size = default, bool output = true)
        {
            lock (@lock)
            {
                if (!output)
                {
                    if (size.IsEmpty)
                    {
                        NextRaw();
                        return null;
                    }
                    if (size.NDim == 0)
                        throw new TypeError("'numpy.float64' object cannot be interpreted as an integer");
                    long n = 0;
                    foreach (long d in size.dimensions)
                        n += d;
                    for (long i = 0; i < n; i++)
                        NextRaw();
                    return null;
                }

                if (size.IsEmpty)
                    return NDArray.Scalar(NextRaw());

                var ret = new NDArray(typeof(ulong), size, false);
                FillRaw((ulong*)ret.Address, ret.size);
                return ret;
            }
        }

        /// <summary>The next uniform 64-bit word (NumPy's <c>next_uint64</c>).</summary>
        /// <returns>A uniformly distributed 64-bit value; advances the engine state.</returns>
        internal abstract ulong NextUInt64();

        /// <summary>The next uniform 32-bit word (NumPy's <c>next_uint32</c>).</summary>
        /// <returns>A uniformly distributed 32-bit value; advances (or consumes a buffered half of) the engine state.</returns>
        internal abstract uint NextUInt32();

        /// <summary>A random double in <c>[0, 1)</c> with 53-bit precision (NumPy's <c>next_double</c>).</summary>
        /// <returns>The top 53 bits of one 64-bit word scaled by <c>2**-53</c> (engines with their own formula override).</returns>
        internal virtual double NextDouble() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

        /// <summary>
        ///     Converts one <c>next_uint64</c> draw to the <c>next_double</c> the same stream position would have produced —
        ///     the contract that lets <see cref="DrawBuffer64"/> serve both kinds of draw from one read-ahead buffer.
        /// </summary>
        /// <param name="unit">A word from <see cref="NextUInt64"/> / <see cref="FillUInt64"/>.</param>
        /// <returns><c>(unit &gt;&gt; 11) * 2**-53</c> — <c>uint64_to_double</c>, the <c>next_double</c> of every 64-bit engine
        /// (MT19937 overrides it: its double is built from the two 32-bit words its 64-bit draw concatenates).</returns>
        internal virtual double UnitToDouble(ulong unit) => (unit >> 11) * (1.0 / 9007199254740992.0);

        /// <summary>A random float in <c>[0, 1)</c> with 24-bit precision (NumPy's <c>next_float</c>).</summary>
        /// <returns>The top 24 bits of one 32-bit word scaled by <c>2**-24</c>.</returns>
        internal virtual float NextFloat() => (NextUInt32() >> 8) * (1.0f / 16777216.0f);

        /// <summary>The engine's raw output word (NumPy's <c>next_raw</c>), widened to 64 bits.</summary>
        /// <returns>A full 64-bit word for 64-bit engines; the 32-bit word for MT19937.</returns>
        internal virtual ulong NextRaw() => NextUInt64();

        // ---- bulk fills: the same streams as the per-draw primitives, without a virtual call per draw ----
        //
        // A fill of n draws through the per-draw primitives pays two virtual calls per element (NextDouble ->
        // NextUInt64) and re-reads/re-writes the engine's fields around every destination store (a raw pointer
        // store may alias them, so the JIT cannot keep the state in registers). The engines override these with
        // loops that hold their state in locals for the whole run. Every override MUST leave the engine in
        // exactly the state the equivalent sequence of per-draw calls would — the fills are observable through
        // `state`, and NumPy's streams interleave them freely with single draws.

        /// <summary>Fills <paramref name="n"/> words of <c>next_uint64</c> (the stream <see cref="NextUInt64"/> yields).</summary>
        /// <param name="dst">The destination (at least <paramref name="n"/> words; may be unaligned).</param>
        /// <param name="n">The number of words (non-positive: nothing).</param>
        /// <remarks>The caller holds <see cref="@lock"/>.</remarks>
        internal virtual unsafe void FillUInt64(ulong* dst, long n)
        {
            for (long i = 0; i < n; i++)
                dst[i] = NextUInt64();
        }

        /// <summary>Fills <paramref name="n"/> words of <c>next_uint32</c>, buffered halves included (the stream <see cref="NextUInt32"/> yields).</summary>
        /// <param name="dst">The destination (at least <paramref name="n"/> words).</param>
        /// <param name="n">The number of words (non-positive: nothing).</param>
        /// <remarks>The caller holds <see cref="@lock"/>.</remarks>
        internal virtual unsafe void FillUInt32(uint* dst, long n)
        {
            for (long i = 0; i < n; i++)
                dst[i] = NextUInt32();
        }

        /// <summary>Fills <paramref name="n"/> <c>next_double</c> draws in <c>[0, 1)</c> (the stream <see cref="NextDouble"/> yields).</summary>
        /// <param name="dst">The destination (at least <paramref name="n"/> doubles).</param>
        /// <param name="n">The number of draws (non-positive: nothing).</param>
        /// <remarks>The caller holds <see cref="@lock"/>.</remarks>
        internal virtual unsafe void FillDouble(double* dst, long n)
        {
            for (long i = 0; i < n; i++)
                dst[i] = NextDouble();
        }

        /// <summary>Fills <paramref name="n"/> <c>next_float</c> draws in <c>[0, 1)</c> (the stream <see cref="NextFloat"/> yields).</summary>
        /// <param name="dst">The destination (at least <paramref name="n"/> floats).</param>
        /// <param name="n">The number of draws (non-positive: nothing).</param>
        /// <remarks>The caller holds <see cref="@lock"/>.</remarks>
        internal virtual unsafe void FillFloat(float* dst, long n)
        {
            for (long i = 0; i < n; i++)
                dst[i] = NextFloat();
        }

        /// <summary>Fills <paramref name="n"/> raw words (<c>next_raw</c>, widened to 64 bits) — the stream <see cref="NextRaw"/> yields.</summary>
        /// <param name="dst">The destination (at least <paramref name="n"/> words).</param>
        /// <param name="n">The number of words (non-positive: nothing).</param>
        /// <remarks>The caller holds <see cref="@lock"/>.</remarks>
        internal virtual unsafe void FillRaw(ulong* dst, long n)
        {
            for (long i = 0; i < n; i++)
                dst[i] = NextRaw();
        }

        /// <summary>
        ///     The chunk, in 64-bit words, that the in-place conversions below walk: 2 KB, so the words a chunk's
        ///     <see cref="FillUInt64"/> just wrote are still in L1 when they are converted.
        /// </summary>
        private protected const int FillChunk = 256;

        /// <summary>
        ///     <see cref="FillDouble"/> for engines whose <c>next_double</c> is <c>uint64_to_double(next_uint64)</c>
        ///     (PCG64, PCG64DXSM, Philox, SFC64): the destination is filled with raw words a chunk at a time and each
        ///     word is converted IN PLACE to <c>(w &gt;&gt; 11) * 2**-53</c>.
        /// </summary>
        /// <param name="dst">The destination.</param>
        /// <param name="n">The number of draws.</param>
        /// <remarks>
        ///     Reinterpreting the double slots as words is safe (same size, each word read before its slot is
        ///     rewritten), and <c>w &gt;&gt; 11 &lt; 2**53</c> converts exactly through the signed conversion.
        /// </remarks>
        private protected unsafe void FillDoubleFrom64(double* dst, long n)
        {
            var words = (ulong*)dst;
            while (n > 0)
            {
                long m = n < FillChunk ? n : FillChunk;
                FillUInt64(words, m);
                var d = (double*)words;
                for (long k = 0; k < m; k++)
                    d[k] = (long)(words[k] >> 11) * (1.0 / 9007199254740992.0);
                words += m;
                n -= m;
            }
        }

        /// <summary>
        ///     <see cref="FillUInt32"/> for engines whose <c>next_uint32</c> splits a <c>next_uint64</c> word — low half
        ///     returned, high half buffered (PCG64, PCG64DXSM, Philox, SFC64): a pending half is served first, whole
        ///     words then land straight in the destination (little-endian, so word k's low/high halves ARE the next two
        ///     uint32 slots), and an odd final draw splits one more word and buffers its high half.
        /// </summary>
        /// <param name="dst">The destination (any 4-byte alignment; the word stores may be unaligned).</param>
        /// <param name="n">The number of draws.</param>
        /// <param name="hasUint32">The engine's pending-half flag (read and updated).</param>
        /// <param name="uinteger">The engine's pending half (read and updated).</param>
        private protected unsafe void FillUInt32From64(uint* dst, long n, ref int hasUint32, ref uint uinteger)
        {
            if (n <= 0)
                return;
            if (!BitConverter.IsLittleEndian)
            {
                // The in-place split relies on the little-endian word layout; stay per-draw elsewhere.
                for (long k = 0; k < n; k++)
                    dst[k] = NextUInt32();
                return;
            }
            long i = 0;
            if (hasUint32 != 0)
            {
                hasUint32 = 0;
                dst[i++] = uinteger;
            }
            long pairs = (n - i) >> 1;
            if (pairs > 0)
            {
                FillUInt64((ulong*)(dst + i), pairs);
                i += pairs << 1;
                // Per draw, each word's high half is first buffered into `uinteger` and then consumed — which clears
                // the flag but LEAVES the value (NumPy's next32 never zeroes it), and the state exposes it. So the
                // last pair's high half is the uinteger the per-draw sequence would leave behind.
                uinteger = dst[i - 1];
            }
            if (i < n)
            {
                ulong w;
                FillUInt64(&w, 1);
                dst[i] = (uint)w;
                hasUint32 = 1;
                uinteger = (uint)(w >> 32);
            }
        }

        /// <summary>
        ///     <see cref="FillFloat"/> for the split-word engines: <see cref="FillUInt32From64"/> into the float slots a chunk at
        ///     a time, then each word converted in place to <c>(u &gt;&gt; 8) * 2**-24</c> (NumPy's <c>next_float</c>).
        /// </summary>
        /// <param name="dst">The destination.</param>
        /// <param name="n">The number of draws.</param>
        /// <param name="hasUint32">The engine's pending-half flag (read and updated).</param>
        /// <param name="uinteger">The engine's pending half (read and updated).</param>
        private protected unsafe void FillFloatFrom64(float* dst, long n, ref int hasUint32, ref uint uinteger)
        {
            var words = (uint*)dst;
            while (n > 0)
            {
                long m = n < 2 * FillChunk ? n : 2 * FillChunk;
                FillUInt32From64(words, m, ref hasUint32, ref uinteger);
                var f = (float*)words;
                for (long k = 0; k < m; k++)
                    f[k] = (int)(words[k] >> 8) * (1.0f / 16777216.0f);
                words += m;
                n -= m;
            }
        }

        /// <summary>The bit generator's class name, e.g. <c>"PCG64"</c>. Drives <c>Generator</c>'s repr and the state checks.</summary>
        internal abstract string Name { get; }
    }

    /// <summary>
    ///     A snapshot of a bit generator's state — the typed stand-in for NumPy's <c>bit_generator.state</c> dict.
    /// </summary>
    /// <remarks>
    ///     Each concrete generator has its own state class (<see cref="PCG64.State"/>, <see cref="PCG64DXSM.State"/>,
    ///     <see cref="Philox.State"/>, <see cref="SFC64.State"/>, <see cref="MT19937.State"/>) whose members are named
    ///     after NumPy's dict keys; the nested <c>'state'</c> sub-dict is flattened into the
    ///     same object. Snapshots are plain mutable data: build one, edit one, or assign one back through
    ///     <see cref="BitGenerator.state"/>.
    /// </remarks>
    public abstract class BitGeneratorState
    {
        /// <summary>NumPy's <c>'bit_generator'</c> key: the name of the bit generator class this state belongs to.</summary>
        public abstract string bit_generator { get; }
    }
}
