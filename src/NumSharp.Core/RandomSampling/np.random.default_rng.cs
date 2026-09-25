using System;
using System.Numerics;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Construct a new <see cref="Generator"/> with the default bit generator (PCG64),
        ///     seeded from fresh, unpredictable OS entropy.
        /// </summary>
        /// <returns>A new <see cref="Generator"/> over a fresh <see cref="PCG64"/>.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.default_rng.html
        ///     <br/>
        ///     This is the recommended constructor for the modern <see cref="Generator"/> API and is
        ///     independent of the legacy global <c>np.random</c> (<c>RandomState</c>) state. Log
        ///     <c>((SeedSequence)rng.bit_generator.seed_seq).entropy</c> to be able to reproduce the stream.
        /// </remarks>
        public Generator default_rng() => new Generator(new PCG64(new SeedSequence()));

        /// <summary>Construct a new <see cref="Generator"/> (PCG64) seeded from a single non-negative integer through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <returns>A new <see cref="Generator"/>; the same seed always yields NumPy's <c>default_rng(seed)</c> stream.</returns>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative (<c>expected non-negative integer</c>).</exception>
        public Generator default_rng(long seed) => new Generator(new PCG64(new SeedSequence(seed)));

        /// <summary>Construct a new <see cref="Generator"/> (PCG64) seeded from a single non-negative integer through <see cref="SeedSequence"/>.</summary>
        /// <param name="seed">The seed (the full uint64 range, e.g. <c>2**64 - 1</c>).</param>
        /// <returns>A new <see cref="Generator"/>.</returns>
        public Generator default_rng(ulong seed) => new Generator(new PCG64(new SeedSequence(seed)));

        /// <summary>Construct a new <see cref="Generator"/> (PCG64) seeded from an arbitrary-size non-negative integer (e.g. a logged 128-bit entropy).</summary>
        /// <param name="seed">The seed (must be non-negative).</param>
        /// <returns>A new <see cref="Generator"/>.</returns>
        /// <exception cref="ValueError"><paramref name="seed"/> is negative.</exception>
        public Generator default_rng(BigInteger seed) => new Generator(new PCG64(new SeedSequence(seed)));

        /// <summary>Construct a new <see cref="Generator"/> (PCG64) seeded from a sequence of non-negative integers.</summary>
        /// <param name="seed">The seed words (each must be non-negative).</param>
        /// <returns>A new <see cref="Generator"/>.</returns>
        /// <exception cref="ValueError">An element is negative.</exception>
        public Generator default_rng(int[] seed) => new Generator(new PCG64(new SeedSequence(seed)));

        /// <summary>Construct a new <see cref="Generator"/> (PCG64) seeded from a sequence of non-negative integers.</summary>
        /// <param name="seed">The seed words (each must be non-negative; values of 2**32 and above span several words).</param>
        /// <returns>A new <see cref="Generator"/>.</returns>
        /// <exception cref="ValueError">An element is negative.</exception>
        public Generator default_rng(long[] seed) => new Generator(new PCG64(new SeedSequence(seed)));

        /// <summary>Construct a new <see cref="Generator"/> (PCG64) seeded from uint32 words (NumPy's uint32-array pass-through).</summary>
        /// <param name="seed">The seed words.</param>
        /// <returns>A new <see cref="Generator"/>.</returns>
        public Generator default_rng(uint[] seed) => new Generator(new PCG64(new SeedSequence(seed)));

        /// <summary>
        ///     Construct a new <see cref="Generator"/> (PCG64) seeded from an integer array (NumPy's
        ///     <c>default_rng(np.array([...]))</c>): flattened in C order; a uint32 array passes through.
        /// </summary>
        /// <param name="seed">The integer seed array.</param>
        /// <returns>A new <see cref="Generator"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="seed"/> is null.</exception>
        /// <exception cref="TypeError">A 0-d, bool or non-integer array.</exception>
        /// <exception cref="ValueError">A negative element.</exception>
        public Generator default_rng(NDArray seed)
            => new Generator(new PCG64(new SeedSequence((object)(seed ?? throw new ArgumentNullException(nameof(seed))))));

        /// <summary>Construct a new <see cref="Generator"/> (PCG64) from a prepared seed sequence (NumPy's <c>PCG64(seed_seq)</c>).</summary>
        /// <param name="seed">The seed sequence — normally a <see cref="SeedSequence"/> (e.g. one of its <c>spawn</c>ed children).</param>
        /// <returns>A new <see cref="Generator"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="seed"/> is null.</exception>
        public Generator default_rng(ISeedSequence seed) => new Generator(new PCG64(seed));

        /// <summary>Wrap an existing bit generator in a <see cref="Generator"/> (NumPy passes it through).</summary>
        /// <param name="bitGenerator">The bit generator; the new Generator shares its state and lock.</param>
        /// <returns>A new <see cref="Generator"/> over <paramref name="bitGenerator"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="bitGenerator"/> is null.</exception>
        public Generator default_rng(BitGenerator bitGenerator) => new Generator(bitGenerator);

        /// <summary>Pass an existing <see cref="Generator"/> through unaltered (NumPy behavior).</summary>
        /// <param name="generator">The generator.</param>
        /// <returns>The same instance.</returns>
        public Generator default_rng(Generator generator) => generator;

        /// <summary>
        ///     Wrap a legacy RandomState's bit generator in a <see cref="Generator"/> (NumPy's
        ///     <c>default_rng(RandomState)</c> → <c>Generator(random_state._bit_generator)</c>).
        /// </summary>
        /// <param name="randomState">The legacy RandomState.</param>
        /// <returns>A new <see cref="Generator"/> over the SAME <see cref="MT19937"/> — draws from either advance both.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="randomState"/> is null.</exception>
        /// <remarks>
        ///     The Generator uses its own algorithms over the shared engine, so <c>default_rng(RandomState(42)).random(2)</c>
        ///     is <c>[0.3745401188473625, 0.9507143064099162]</c> — the same doubles as the legacy <c>random_sample</c>,
        ///     because MT19937's 53-bit double is common to both.
        /// </remarks>
        public Generator default_rng(NumPyRandom randomState)
            => new Generator((randomState ?? throw new ArgumentNullException(nameof(randomState))).randomizer);

        /// <summary>
        ///     NumPy's dynamically-typed <c>default_rng(seed)</c>: a <see cref="BitGenerator"/> is wrapped, a
        ///     <see cref="Generator"/> passes through, a legacy RandomState's engine is wrapped, and anything else seeds a
        ///     new <see cref="PCG64"/> (an <see cref="ISeedSequence"/> directly, other values as <see cref="SeedSequence"/> entropy).
        /// </summary>
        /// <param name="seed">null (OS entropy), an integer, a sequence of integers (C# array or list, nested allowed), an
        /// integer <see cref="NDArray"/>, an <see cref="ISeedSequence"/>, a <see cref="BitGenerator"/>, a <see cref="Generator"/>
        /// or a <see cref="NumPyRandom"/>.</param>
        /// <returns>The Generator.</returns>
        /// <exception cref="TypeError"><paramref name="seed"/> is not valid entropy (e.g. a float or a string).</exception>
        /// <exception cref="ValueError"><paramref name="seed"/> holds a negative value.</exception>
        public Generator default_rng(object seed) => seed switch
        {
            null => default_rng(),
            BitGenerator bg => new Generator(bg),
            Generator g => g,
            NumPyRandom rs => default_rng(rs),
            ISeedSequence ss => new Generator(new PCG64(ss)),
            _ => new Generator(new PCG64(new SeedSequence(seed))),
        };
    }
}
