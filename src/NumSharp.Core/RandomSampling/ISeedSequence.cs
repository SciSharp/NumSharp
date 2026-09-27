using System;

namespace NumSharp
{
    /// <summary>
    ///     The seed-sequence protocol a <see cref="BitGenerator"/> seeds itself from (NumPy's
    ///     <c>numpy.random.bit_generator.ISeedSequence</c>).
    /// </summary>
    /// <remarks>
    ///     A bit generator calls <see cref="generate_state"/> once in its constructor with the number of words its
    ///     state needs. <see cref="SeedSequence"/> is the standard implementation; <see cref="SeedlessSeedSequence"/>
    ///     is the placeholder for engines that need no seed. Implementations return what NumPy's protocol returns: a 1-D
    ///     uint32 (the default) or uint64 <see cref="NDArray"/> of at least <c>n_words</c> words, which the bit generators
    ///     read in C order.
    /// </remarks>
    public interface ISeedSequence
    {
        /// <summary>
        ///     Return the requested number of words for PRNG seeding (NumPy's <c>generate_state(n_words, dtype=np.uint32)</c>).
        /// </summary>
        /// <param name="n_words">The number of words (a Python int in NumPy, allocated as <c>np.zeros(n_words)</c>).</param>
        /// <param name="dtype"><c>uint32</c> (default) or <c>uint64</c>; a uint64 word costs two uint32 words.</param>
        /// <returns>A 1-D uint32 or uint64 NDArray of <paramref name="n_words"/> words.</returns>
        NDArray generate_state(long n_words, DType dtype = null);
    }

    /// <summary>
    ///     A seed sequence that can derive independent children (NumPy's
    ///     <c>numpy.random.bit_generator.ISpawnableSeedSequence</c>) — what <see cref="BitGenerator.spawn"/> and
    ///     <see cref="Generator.spawn"/> require.
    /// </summary>
    public interface ISpawnableSeedSequence : ISeedSequence
    {
        /// <summary>Spawn <paramref name="n_children"/> child seed sequences (NumPy's <c>spawn(n_children)</c>).</summary>
        /// <param name="n_children">The number of children.</param>
        /// <returns>The children, each able to seed an independent bit generator.</returns>
        ISpawnableSeedSequence[] spawn(int n_children);
    }

    /// <summary>
    ///     A seed sequence for bit generators that need no seed state (NumPy's
    ///     <c>numpy.random.bit_generator.SeedlessSeedSequence</c>).
    /// </summary>
    /// <remarks>
    ///     It cannot produce words — seeding a bit generator from it fails in the constructor, as in NumPy — and it
    ///     "spawns" itself: <c>spawn(n)</c> is <c>[self] * n</c>.
    /// </remarks>
    public sealed class SeedlessSeedSequence : ISpawnableSeedSequence
    {
        /// <summary>Always fails: a seedless sequence has no state to generate from.</summary>
        /// <param name="n_words">Ignored.</param>
        /// <param name="dtype">Ignored.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="NotImplementedException">Always (NumPy's <c>NotImplementedError('seedless SeedSequences cannot generate state')</c>).</exception>
        public NDArray generate_state(long n_words, DType dtype = null)
            => throw new NotImplementedException("seedless SeedSequences cannot generate state");

        /// <summary>Returns this same instance <paramref name="n_children"/> times (NumPy's <c>[self] * n_children</c>).</summary>
        /// <param name="n_children">The number of "children" (non-positive: none).</param>
        /// <returns>An array whose every element is this instance.</returns>
        public ISpawnableSeedSequence[] spawn(int n_children)
        {
            var r = new ISpawnableSeedSequence[Math.Max(0, n_children)];
            for (int i = 0; i < r.Length; i++)
                r[i] = this;
            return r;
        }
    }
}
