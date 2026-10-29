using System.Numerics;
using System.Runtime.CompilerServices;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        // numpy.random's classes, reached through the module: NumPy code spells them np.random.PCG64(42),
        // np.random.Generator(np.random.PCG64(seed)), np.random.SeedSequence(entropy) — attribute lookups on the module
        // that happen to be classes. `np.random` is this instance, so each class needs a factory METHOD of the class's
        // name to port verbatim; each one below mirrors one public constructor overload (same parameters, defaults and
        // overload priority), so a call resolves to the constructor `new PCG64(...)` would pick.
        //
        // Inside NumPyRandom these methods shadow the class names in EXPRESSION position and in crefs: a static member
        // access is spelled global::NumSharp.Generator.Log1p here, a cref NumSharp.Generator. Type positions — new
        // PCG64(..), is MT19937, a declaration's type — are unaffected (C# resolves those among types only).
        //
        // BitGenerator is deliberately absent: NumPy's np.random.BitGenerator(seed) raises "BitGenerator is a base
        // class and cannot be instantized", and the C# class is abstract, so the refusal is a compile error here.

        #region np.random.MT19937

        /// <summary>
        ///     NumPy's <c>np.random.MT19937()</c>: NumPy's Mersenne Twister seeded the modern way — through a
        ///     SeedSequence, NOT the legacy integer seeding (<c>np.random.MT19937(42)</c> and
        ///     <c>np.random.RandomState(42)</c> are different streams, exactly as in NumPy), seeded from fresh,
        ///     unpredictable OS entropy (a new <see cref="NumSharp.SeedSequence"/>).
        /// </summary>
        /// <returns>A new <see cref="NumSharp.MT19937"/> — the object <c>new MT19937(…)</c> builds from the same
        ///     arguments.</returns>
        public MT19937 MT19937() => new MT19937();

        /// <summary>
        ///     NumPy's <c>np.random.MT19937(seed)</c> with an integer seed, mixed through a
        ///     <see cref="NumSharp.SeedSequence"/>.
        /// </summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <returns>A new <see cref="NumSharp.MT19937"/> — the object <c>new MT19937(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative (<c>expected non-negative
        ///     integer</c>).</exception>
        public MT19937 MT19937(long seed) => new MT19937(seed);

        /// <summary>
        ///     NumPy's <c>np.random.MT19937(seed)</c> with an integer seed over the full uint64 range.
        /// </summary>
        /// <param name="seed">The seed.</param>
        /// <returns>A new <see cref="NumSharp.MT19937"/> — the object <c>new MT19937(…)</c> builds from the same
        ///     arguments.</returns>
        public MT19937 MT19937(ulong seed) => new MT19937(seed);

        /// <summary>
        ///     NumPy's <c>np.random.MT19937(seed)</c> with an arbitrary-size integer seed (e.g. a logged 128-bit
        ///     entropy).
        /// </summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <returns>A new <see cref="NumSharp.MT19937"/> — the object <c>new MT19937(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative.</exception>
        public MT19937 MT19937(BigInteger seed) => new MT19937(seed);

        /// <summary>
        ///     NumPy's <c>np.random.MT19937(seed)</c> with a sequence of integers, mixed through a
        ///     <see cref="NumSharp.SeedSequence"/>.
        /// </summary>
        /// <param name="seed">The seed words (each must be non-negative); null is NumPy's <c>None</c> — fresh OS
        ///     entropy.</param>
        /// <returns>A new <see cref="NumSharp.MT19937"/> — the object <c>new MT19937(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError">An element is negative.</exception>
        public MT19937 MT19937(int[] seed) => new MT19937(seed);

        /// <summary>
        ///     NumPy's <c>np.random.MT19937(seed)</c> with a sequence of integers, mixed through a
        ///     <see cref="NumSharp.SeedSequence"/>.
        /// </summary>
        /// <param name="seed">The seed words (each must be non-negative; values of 2**32 and above span several words);
        ///     null is NumPy's <c>None</c> — fresh OS entropy.</param>
        /// <returns>A new <see cref="NumSharp.MT19937"/> — the object <c>new MT19937(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError">An element is negative.</exception>
        public MT19937 MT19937(long[] seed) => new MT19937(seed);

        /// <summary>
        ///     NumPy's <c>np.random.MT19937(seed)</c> with uint32 words (NumPy's uint32-array pass-through).
        /// </summary>
        /// <param name="seed">The seed words; null is NumPy's <c>None</c> — fresh OS entropy.</param>
        /// <returns>A new <see cref="NumSharp.MT19937"/> — the object <c>new MT19937(…)</c> builds from the same
        ///     arguments.</returns>
        public MT19937 MT19937(uint[] seed) => new MT19937(seed);

        /// <summary>
        ///     NumPy's <c>np.random.MT19937(seed)</c> with a prepared seed sequence (a
        ///     <see cref="NumSharp.SeedSequence"/> or one of its <c>spawn</c>ed children).
        /// </summary>
        /// <param name="seed">The seed sequence; null is NumPy's <c>None</c> — fresh OS entropy.</param>
        /// <returns>A new <see cref="NumSharp.MT19937"/> — the object <c>new MT19937(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="NotImplementedException"><paramref name="seed"/> is a <see cref="SeedlessSeedSequence"/>
        ///     (NumPy's <c>seedless SeedSequences cannot generate state</c>).</exception>
        /// <remarks>
        ///     The overload a bare <c>null</c> literal binds (<c>np.random.MT19937(null)</c>, NumPy's
        ///     <c>MT19937(None)</c>): the constructor's <c>OverloadResolutionPriority</c>, mirrored so both spellings
        ///     resolve alike.
        /// </remarks>
        [OverloadResolutionPriority(1)]
        public MT19937 MT19937(ISeedSequence seed) => new MT19937(seed);

        #endregion

        #region np.random.PCG64

        /// <summary>
        ///     NumPy's <c>np.random.PCG64()</c>: NumPy's default engine (the one <c>default_rng</c> builds), seeded
        ///     from fresh, unpredictable OS entropy (a new <see cref="NumSharp.SeedSequence"/>).
        /// </summary>
        /// <returns>A new <see cref="NumSharp.PCG64"/> — the object <c>new PCG64(…)</c> builds from the same
        ///     arguments.</returns>
        public PCG64 PCG64() => new PCG64();

        /// <summary>
        ///     NumPy's <c>np.random.PCG64(seed)</c> with an integer seed, mixed through a
        ///     <see cref="NumSharp.SeedSequence"/>.
        /// </summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <returns>A new <see cref="NumSharp.PCG64"/> — the object <c>new PCG64(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative (<c>expected non-negative
        ///     integer</c>).</exception>
        public PCG64 PCG64(long seed) => new PCG64(seed);

        /// <summary>
        ///     NumPy's <c>np.random.PCG64(seed)</c> with an integer seed over the full uint64 range.
        /// </summary>
        /// <param name="seed">The seed.</param>
        /// <returns>A new <see cref="NumSharp.PCG64"/> — the object <c>new PCG64(…)</c> builds from the same
        ///     arguments.</returns>
        public PCG64 PCG64(ulong seed) => new PCG64(seed);

        /// <summary>
        ///     NumPy's <c>np.random.PCG64(seed)</c> with an arbitrary-size integer seed (e.g. a logged 128-bit
        ///     entropy).
        /// </summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <returns>A new <see cref="NumSharp.PCG64"/> — the object <c>new PCG64(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative.</exception>
        public PCG64 PCG64(BigInteger seed) => new PCG64(seed);

        /// <summary>
        ///     NumPy's <c>np.random.PCG64(seed)</c> with a sequence of integers, mixed through a
        ///     <see cref="NumSharp.SeedSequence"/>.
        /// </summary>
        /// <param name="seed">The seed words (each must be non-negative); null is NumPy's <c>None</c> — fresh OS
        ///     entropy.</param>
        /// <returns>A new <see cref="NumSharp.PCG64"/> — the object <c>new PCG64(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError">An element is negative.</exception>
        public PCG64 PCG64(int[] seed) => new PCG64(seed);

        /// <summary>
        ///     NumPy's <c>np.random.PCG64(seed)</c> with a sequence of integers, mixed through a
        ///     <see cref="NumSharp.SeedSequence"/>.
        /// </summary>
        /// <param name="seed">The seed words (each must be non-negative; values of 2**32 and above span several words);
        ///     null is NumPy's <c>None</c> — fresh OS entropy.</param>
        /// <returns>A new <see cref="NumSharp.PCG64"/> — the object <c>new PCG64(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError">An element is negative.</exception>
        public PCG64 PCG64(long[] seed) => new PCG64(seed);

        /// <summary>
        ///     NumPy's <c>np.random.PCG64(seed)</c> with uint32 words (NumPy's uint32-array pass-through).
        /// </summary>
        /// <param name="seed">The seed words; null is NumPy's <c>None</c> — fresh OS entropy.</param>
        /// <returns>A new <see cref="NumSharp.PCG64"/> — the object <c>new PCG64(…)</c> builds from the same
        ///     arguments.</returns>
        public PCG64 PCG64(uint[] seed) => new PCG64(seed);

        /// <summary>
        ///     NumPy's <c>np.random.PCG64(seed)</c> with a prepared seed sequence (a
        ///     <see cref="NumSharp.SeedSequence"/> or one of its <c>spawn</c>ed children).
        /// </summary>
        /// <param name="seed">The seed sequence; null is NumPy's <c>None</c> — fresh OS entropy.</param>
        /// <returns>A new <see cref="NumSharp.PCG64"/> — the object <c>new PCG64(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="NotImplementedException"><paramref name="seed"/> is a <see cref="SeedlessSeedSequence"/>
        ///     (NumPy's <c>seedless SeedSequences cannot generate state</c>).</exception>
        /// <remarks>
        ///     The overload a bare <c>null</c> literal binds (<c>np.random.PCG64(null)</c>, NumPy's
        ///     <c>PCG64(None)</c>): the constructor's <c>OverloadResolutionPriority</c>, mirrored so both spellings
        ///     resolve alike.
        /// </remarks>
        [OverloadResolutionPriority(1)]
        public PCG64 PCG64(ISeedSequence seed) => new PCG64(seed);

        #endregion

        #region np.random.PCG64DXSM

        /// <summary>
        ///     NumPy's <c>np.random.PCG64DXSM()</c>: PCG64 with the DXSM output function (NumPy's recommended upgrade
        ///     for heavily parallel use), seeded from fresh, unpredictable OS entropy (a new
        ///     <see cref="NumSharp.SeedSequence"/>).
        /// </summary>
        /// <returns>A new <see cref="NumSharp.PCG64DXSM"/> — the object <c>new PCG64DXSM(…)</c> builds from the same
        ///     arguments.</returns>
        public PCG64DXSM PCG64DXSM() => new PCG64DXSM();

        /// <summary>
        ///     NumPy's <c>np.random.PCG64DXSM(seed)</c> with an integer seed, mixed through a
        ///     <see cref="NumSharp.SeedSequence"/>.
        /// </summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <returns>A new <see cref="NumSharp.PCG64DXSM"/> — the object <c>new PCG64DXSM(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative (<c>expected non-negative
        ///     integer</c>).</exception>
        public PCG64DXSM PCG64DXSM(long seed) => new PCG64DXSM(seed);

        /// <summary>
        ///     NumPy's <c>np.random.PCG64DXSM(seed)</c> with an integer seed over the full uint64 range.
        /// </summary>
        /// <param name="seed">The seed.</param>
        /// <returns>A new <see cref="NumSharp.PCG64DXSM"/> — the object <c>new PCG64DXSM(…)</c> builds from the same
        ///     arguments.</returns>
        public PCG64DXSM PCG64DXSM(ulong seed) => new PCG64DXSM(seed);

        /// <summary>
        ///     NumPy's <c>np.random.PCG64DXSM(seed)</c> with an arbitrary-size integer seed (e.g. a logged 128-bit
        ///     entropy).
        /// </summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <returns>A new <see cref="NumSharp.PCG64DXSM"/> — the object <c>new PCG64DXSM(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative.</exception>
        public PCG64DXSM PCG64DXSM(BigInteger seed) => new PCG64DXSM(seed);

        /// <summary>
        ///     NumPy's <c>np.random.PCG64DXSM(seed)</c> with a sequence of integers, mixed through a
        ///     <see cref="NumSharp.SeedSequence"/>.
        /// </summary>
        /// <param name="seed">The seed words (each must be non-negative); null is NumPy's <c>None</c> — fresh OS
        ///     entropy.</param>
        /// <returns>A new <see cref="NumSharp.PCG64DXSM"/> — the object <c>new PCG64DXSM(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError">An element is negative.</exception>
        public PCG64DXSM PCG64DXSM(int[] seed) => new PCG64DXSM(seed);

        /// <summary>
        ///     NumPy's <c>np.random.PCG64DXSM(seed)</c> with a sequence of integers, mixed through a
        ///     <see cref="NumSharp.SeedSequence"/>.
        /// </summary>
        /// <param name="seed">The seed words (each must be non-negative; values of 2**32 and above span several words);
        ///     null is NumPy's <c>None</c> — fresh OS entropy.</param>
        /// <returns>A new <see cref="NumSharp.PCG64DXSM"/> — the object <c>new PCG64DXSM(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError">An element is negative.</exception>
        public PCG64DXSM PCG64DXSM(long[] seed) => new PCG64DXSM(seed);

        /// <summary>
        ///     NumPy's <c>np.random.PCG64DXSM(seed)</c> with uint32 words (NumPy's uint32-array pass-through).
        /// </summary>
        /// <param name="seed">The seed words; null is NumPy's <c>None</c> — fresh OS entropy.</param>
        /// <returns>A new <see cref="NumSharp.PCG64DXSM"/> — the object <c>new PCG64DXSM(…)</c> builds from the same
        ///     arguments.</returns>
        public PCG64DXSM PCG64DXSM(uint[] seed) => new PCG64DXSM(seed);

        /// <summary>
        ///     NumPy's <c>np.random.PCG64DXSM(seed)</c> with a prepared seed sequence (a
        ///     <see cref="NumSharp.SeedSequence"/> or one of its <c>spawn</c>ed children).
        /// </summary>
        /// <param name="seed">The seed sequence; null is NumPy's <c>None</c> — fresh OS entropy.</param>
        /// <returns>A new <see cref="NumSharp.PCG64DXSM"/> — the object <c>new PCG64DXSM(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="NotImplementedException"><paramref name="seed"/> is a <see cref="SeedlessSeedSequence"/>
        ///     (NumPy's <c>seedless SeedSequences cannot generate state</c>).</exception>
        /// <remarks>
        ///     The overload a bare <c>null</c> literal binds (<c>np.random.PCG64DXSM(null)</c>, NumPy's
        ///     <c>PCG64DXSM(None)</c>): the constructor's <c>OverloadResolutionPriority</c>, mirrored so both spellings
        ///     resolve alike.
        /// </remarks>
        [OverloadResolutionPriority(1)]
        public PCG64DXSM PCG64DXSM(ISeedSequence seed) => new PCG64DXSM(seed);

        #endregion

        #region np.random.Philox

        /// <summary>
        ///     NumPy's <c>np.random.Philox()</c>: the counter-based Philox 4x64 engine (seedable by a key instead of a
        ///     seed, and jumpable by a counter), seeded from fresh, unpredictable OS entropy (a new
        ///     <see cref="NumSharp.SeedSequence"/>).
        /// </summary>
        /// <returns>A new <see cref="NumSharp.Philox"/> — the object <c>new Philox(…)</c> builds from the same
        ///     arguments.</returns>
        public Philox Philox() => new Philox();

        /// <summary>
        ///     NumPy's <c>np.random.Philox(seed)</c> with an integer seed, mixed through a
        ///     <see cref="NumSharp.SeedSequence"/>.
        /// </summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <returns>A new <see cref="NumSharp.Philox"/> — the object <c>new Philox(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative (<c>expected non-negative
        ///     integer</c>).</exception>
        public Philox Philox(long seed) => new Philox(seed);

        /// <summary>
        ///     NumPy's <c>np.random.Philox(seed)</c> with an integer seed over the full uint64 range.
        /// </summary>
        /// <param name="seed">The seed.</param>
        /// <returns>A new <see cref="NumSharp.Philox"/> — the object <c>new Philox(…)</c> builds from the same
        ///     arguments.</returns>
        public Philox Philox(ulong seed) => new Philox(seed);

        /// <summary>
        ///     NumPy's <c>np.random.Philox(seed)</c> with an arbitrary-size integer seed (e.g. a logged 128-bit
        ///     entropy).
        /// </summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <returns>A new <see cref="NumSharp.Philox"/> — the object <c>new Philox(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative.</exception>
        public Philox Philox(BigInteger seed) => new Philox(seed);

        /// <summary>
        ///     NumPy's <c>np.random.Philox(seed)</c> with a sequence of integers, mixed through a
        ///     <see cref="NumSharp.SeedSequence"/>.
        /// </summary>
        /// <param name="seed">The seed words (each must be non-negative); null is NumPy's <c>None</c> — fresh OS
        ///     entropy.</param>
        /// <returns>A new <see cref="NumSharp.Philox"/> — the object <c>new Philox(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError">An element is negative.</exception>
        public Philox Philox(int[] seed) => new Philox(seed);

        /// <summary>
        ///     NumPy's <c>np.random.Philox(seed)</c> with a sequence of integers, mixed through a
        ///     <see cref="NumSharp.SeedSequence"/>.
        /// </summary>
        /// <param name="seed">The seed words (each must be non-negative; values of 2**32 and above span several words);
        ///     null is NumPy's <c>None</c> — fresh OS entropy.</param>
        /// <returns>A new <see cref="NumSharp.Philox"/> — the object <c>new Philox(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError">An element is negative.</exception>
        public Philox Philox(long[] seed) => new Philox(seed);

        /// <summary>
        ///     NumPy's <c>np.random.Philox(seed)</c> with uint32 words (NumPy's uint32-array pass-through).
        /// </summary>
        /// <param name="seed">The seed words; null is NumPy's <c>None</c> — fresh OS entropy.</param>
        /// <returns>A new <see cref="NumSharp.Philox"/> — the object <c>new Philox(…)</c> builds from the same
        ///     arguments.</returns>
        public Philox Philox(uint[] seed) => new Philox(seed);

        /// <summary>
        ///     NumPy's <c>np.random.Philox(seed)</c> with a prepared seed sequence (a
        ///     <see cref="NumSharp.SeedSequence"/> or one of its <c>spawn</c>ed children).
        /// </summary>
        /// <param name="seed">The seed sequence; null is NumPy's <c>None</c> — fresh OS entropy.</param>
        /// <returns>A new <see cref="NumSharp.Philox"/> — the object <c>new Philox(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="NotImplementedException"><paramref name="seed"/> is a <see cref="SeedlessSeedSequence"/>
        ///     (NumPy's <c>seedless SeedSequences cannot generate state</c>).</exception>
        /// <remarks>
        ///     The overload a bare <c>null</c> literal binds (<c>np.random.Philox(null)</c>, NumPy's
        ///     <c>Philox(None)</c>): the constructor's <c>OverloadResolutionPriority</c>, mirrored so both spellings
        ///     resolve alike.
        /// </remarks>
        [OverloadResolutionPriority(1)]
        public Philox Philox(ISeedSequence seed) => new Philox(seed);

        /// <summary>
        ///     NumPy's full <c>np.random.Philox(seed=None, counter=None, key=None)</c>: a seed, or a key used as-is,
        ///     and an optional starting counter.
        /// </summary>
        /// <param name="seed">null for OS entropy, an integer, a sequence of integers or an
        ///     <see cref="ISeedSequence"/>; must be null when <paramref name="key"/> is given.</param>
        /// <param name="counter">The starting 256-bit counter: null (zero), an integer or up to four uint64
        ///     words.</param>
        /// <param name="key">The 128-bit key used directly (no seeding): an integer or up to two uint64 words.</param>
        /// <returns>A new <see cref="NumSharp.Philox"/> — the object <c>new Philox(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError">Both <paramref name="seed"/> and <paramref name="key"/> are given, or a counter
        ///     or key value is negative or too large for its words.</exception>
        /// <exception cref="TypeError">A value is not an integer or a sequence of integers.</exception>
        public Philox Philox(object seed = null, object counter = null, object key = null) =>
            new Philox(seed, counter, key);

        #endregion

        #region np.random.SFC64

        /// <summary>
        ///     NumPy's <c>np.random.SFC64()</c>: Chris Doty-Humphrey's Small Fast Chaotic engine, seeded from fresh,
        ///     unpredictable OS entropy (a new <see cref="NumSharp.SeedSequence"/>).
        /// </summary>
        /// <returns>A new <see cref="NumSharp.SFC64"/> — the object <c>new SFC64(…)</c> builds from the same
        ///     arguments.</returns>
        public SFC64 SFC64() => new SFC64();

        /// <summary>
        ///     NumPy's <c>np.random.SFC64(seed)</c> with an integer seed, mixed through a
        ///     <see cref="NumSharp.SeedSequence"/>.
        /// </summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <returns>A new <see cref="NumSharp.SFC64"/> — the object <c>new SFC64(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative (<c>expected non-negative
        ///     integer</c>).</exception>
        public SFC64 SFC64(long seed) => new SFC64(seed);

        /// <summary>
        ///     NumPy's <c>np.random.SFC64(seed)</c> with an integer seed over the full uint64 range.
        /// </summary>
        /// <param name="seed">The seed.</param>
        /// <returns>A new <see cref="NumSharp.SFC64"/> — the object <c>new SFC64(…)</c> builds from the same
        ///     arguments.</returns>
        public SFC64 SFC64(ulong seed) => new SFC64(seed);

        /// <summary>
        ///     NumPy's <c>np.random.SFC64(seed)</c> with an arbitrary-size integer seed (e.g. a logged 128-bit
        ///     entropy).
        /// </summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <returns>A new <see cref="NumSharp.SFC64"/> — the object <c>new SFC64(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative.</exception>
        public SFC64 SFC64(BigInteger seed) => new SFC64(seed);

        /// <summary>
        ///     NumPy's <c>np.random.SFC64(seed)</c> with a sequence of integers, mixed through a
        ///     <see cref="NumSharp.SeedSequence"/>.
        /// </summary>
        /// <param name="seed">The seed words (each must be non-negative); null is NumPy's <c>None</c> — fresh OS
        ///     entropy.</param>
        /// <returns>A new <see cref="NumSharp.SFC64"/> — the object <c>new SFC64(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError">An element is negative.</exception>
        public SFC64 SFC64(int[] seed) => new SFC64(seed);

        /// <summary>
        ///     NumPy's <c>np.random.SFC64(seed)</c> with a sequence of integers, mixed through a
        ///     <see cref="NumSharp.SeedSequence"/>.
        /// </summary>
        /// <param name="seed">The seed words (each must be non-negative; values of 2**32 and above span several words);
        ///     null is NumPy's <c>None</c> — fresh OS entropy.</param>
        /// <returns>A new <see cref="NumSharp.SFC64"/> — the object <c>new SFC64(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="ValueError">An element is negative.</exception>
        public SFC64 SFC64(long[] seed) => new SFC64(seed);

        /// <summary>
        ///     NumPy's <c>np.random.SFC64(seed)</c> with uint32 words (NumPy's uint32-array pass-through).
        /// </summary>
        /// <param name="seed">The seed words; null is NumPy's <c>None</c> — fresh OS entropy.</param>
        /// <returns>A new <see cref="NumSharp.SFC64"/> — the object <c>new SFC64(…)</c> builds from the same
        ///     arguments.</returns>
        public SFC64 SFC64(uint[] seed) => new SFC64(seed);

        /// <summary>
        ///     NumPy's <c>np.random.SFC64(seed)</c> with a prepared seed sequence (a
        ///     <see cref="NumSharp.SeedSequence"/> or one of its <c>spawn</c>ed children).
        /// </summary>
        /// <param name="seed">The seed sequence; null is NumPy's <c>None</c> — fresh OS entropy.</param>
        /// <returns>A new <see cref="NumSharp.SFC64"/> — the object <c>new SFC64(…)</c> builds from the same
        ///     arguments.</returns>
        /// <exception cref="NotImplementedException"><paramref name="seed"/> is a <see cref="SeedlessSeedSequence"/>
        ///     (NumPy's <c>seedless SeedSequences cannot generate state</c>).</exception>
        /// <remarks>
        ///     The overload a bare <c>null</c> literal binds (<c>np.random.SFC64(null)</c>, NumPy's
        ///     <c>SFC64(None)</c>): the constructor's <c>OverloadResolutionPriority</c>, mirrored so both spellings
        ///     resolve alike.
        /// </remarks>
        [OverloadResolutionPriority(1)]
        public SFC64 SFC64(ISeedSequence seed) => new SFC64(seed);

        #endregion

        #region np.random.SeedSequence

        /// <summary>
        ///     NumPy's <c>np.random.SeedSequence()</c>: a sequence over fresh, unpredictable OS entropy (log its
        ///     <c>entropy</c> to reproduce the stream later).
        /// </summary>
        /// <returns>A new <see cref="NumSharp.SeedSequence"/> — the object <c>new SeedSequence(…)</c> builds from the
        ///     same arguments.</returns>
        public SeedSequence SeedSequence() => new SeedSequence();

        /// <summary>
        ///     NumPy's <c>np.random.SeedSequence(entropy)</c> with an integer.
        /// </summary>
        /// <param name="entropy">The seed (must be non-negative).</param>
        /// <returns>A new <see cref="NumSharp.SeedSequence"/> — the object <c>new SeedSequence(…)</c> builds from the
        ///     same arguments.</returns>
        /// <exception cref="ValueError"><paramref name="entropy"/> is negative (<c>expected non-negative
        ///     integer</c>).</exception>
        public SeedSequence SeedSequence(long entropy) => new SeedSequence(entropy);

        /// <summary>
        ///     NumPy's <c>np.random.SeedSequence(entropy)</c> with an integer over the full uint64 range.
        /// </summary>
        /// <param name="entropy">The seed.</param>
        /// <returns>A new <see cref="NumSharp.SeedSequence"/> — the object <c>new SeedSequence(…)</c> builds from the
        ///     same arguments.</returns>
        public SeedSequence SeedSequence(ulong entropy) => new SeedSequence(entropy);

        /// <summary>
        ///     NumPy's <c>np.random.SeedSequence(entropy)</c> with an arbitrary-size integer (e.g. a logged 128-bit
        ///     entropy).
        /// </summary>
        /// <param name="entropy">The seed (must be non-negative).</param>
        /// <returns>A new <see cref="NumSharp.SeedSequence"/> — the object <c>new SeedSequence(…)</c> builds from the
        ///     same arguments.</returns>
        /// <exception cref="ValueError"><paramref name="entropy"/> is negative.</exception>
        public SeedSequence SeedSequence(BigInteger entropy) => new SeedSequence(entropy);

        /// <summary>
        ///     NumPy's <c>np.random.SeedSequence(entropy)</c> with a sequence of integers.
        /// </summary>
        /// <param name="entropy">The seed words (a value of 2**32 or more spans several words); null is NumPy's
        ///     <c>None</c> — fresh OS entropy.</param>
        /// <returns>A new <see cref="NumSharp.SeedSequence"/> — the object <c>new SeedSequence(…)</c> builds from the
        ///     same arguments.</returns>
        /// <exception cref="ValueError">An element is negative.</exception>
        public SeedSequence SeedSequence(int[] entropy) => new SeedSequence(entropy);

        /// <summary>
        ///     NumPy's <c>np.random.SeedSequence(entropy)</c> with a sequence of integers.
        /// </summary>
        /// <param name="entropy">The seed words (a value of 2**32 or more spans several words); null is NumPy's
        ///     <c>None</c> — fresh OS entropy.</param>
        /// <returns>A new <see cref="NumSharp.SeedSequence"/> — the object <c>new SeedSequence(…)</c> builds from the
        ///     same arguments.</returns>
        /// <exception cref="ValueError">An element is negative.</exception>
        public SeedSequence SeedSequence(long[] entropy) => new SeedSequence(entropy);

        /// <summary>
        ///     NumPy's <c>np.random.SeedSequence(entropy)</c> with uint32 words (NumPy's uint32-array pass-through).
        /// </summary>
        /// <param name="entropy">The seed words; null is NumPy's <c>None</c> — fresh OS entropy.</param>
        /// <returns>A new <see cref="NumSharp.SeedSequence"/> — the object <c>new SeedSequence(…)</c> builds from the
        ///     same arguments.</returns>
        /// <remarks>
        ///     The overload a bare <c>null</c> literal binds (<c>np.random.SeedSequence(null)</c>, NumPy's
        ///     <c>SeedSequence(None)</c>): the constructor's <c>OverloadResolutionPriority</c>, mirrored so both
        ///     spellings resolve alike.
        /// </remarks>
        [OverloadResolutionPriority(1)]
        public SeedSequence SeedSequence(uint[] entropy) => new SeedSequence(entropy);

        /// <summary>
        ///     NumPy's full <c>np.random.SeedSequence(entropy=None, *, spawn_key=(), pool_size=4,
        ///     n_children_spawned=0)</c>.
        /// </summary>
        /// <param name="entropy">null for OS entropy, an integer, a (nested) sequence of integers, a seed string or an
        ///     integer <see cref="NDArray"/> — every form the matching constructor takes.</param>
        /// <param name="spawn_key">This sequence's position in a spawn tree (mixed in as extra entropy); null or empty
        ///     for a root.</param>
        /// <param name="pool_size">The pool size in uint32 words (at least 4).</param>
        /// <param name="n_children_spawned">The number of children already spawned — only when rebuilding a serialized
        ///     sequence.</param>
        /// <returns>A new <see cref="NumSharp.SeedSequence"/> — the object <c>new SeedSequence(…)</c> builds from the
        ///     same arguments.</returns>
        /// <exception cref="ValueError"><paramref name="pool_size"/> is below 4, or an entropy or spawn-key value is
        ///     negative or an unrecognized seed string.</exception>
        /// <exception cref="TypeError"><paramref name="entropy"/> or <paramref name="spawn_key"/> holds something that
        ///     is not an integer or a sequence of integers.</exception>
        /// <exception cref="OutOfMemoryException"><paramref name="pool_size"/> is larger than an array can
        ///     hold.</exception>
        public SeedSequence SeedSequence(object entropy, object spawn_key = null,
            long pool_size = global::NumSharp.SeedSequence.DEFAULT_POOL_SIZE, uint n_children_spawned = 0) =>
            new SeedSequence(entropy, spawn_key, pool_size, n_children_spawned);

        #endregion

        #region np.random.Generator

        /// <summary>
        ///     NumPy's <c>np.random.Generator(bit_generator)</c>: the modern sampler API over an existing engine —
        ///     <c>np.random.Generator(np.random.PCG64(seed))</c> ports verbatim.
        /// </summary>
        /// <param name="bit_generator">The engine, shared rather than copied: draws from the Generator advance it (a
        ///     primed engine's buffered state carries over, as in NumPy).</param>
        /// <returns>A new <see cref="NumSharp.Generator"/> over <paramref name="bit_generator"/> — the object <c>new
        ///     Generator(bit_generator)</c> builds.</returns>
        /// <exception cref="AttributeError"><paramref name="bit_generator"/> is null (NumPy's <c>'NoneType' object has
        ///     no attribute 'capsule'</c>) — unlike <c>default_rng(null)</c>, a Generator never invents an
        ///     engine.</exception>
        public Generator Generator(BitGenerator bit_generator) => new Generator(bit_generator);

        #endregion
    }
}
