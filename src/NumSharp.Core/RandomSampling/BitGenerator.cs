namespace NumSharp
{
    /// <summary>
    ///     Base class for the pseudo-random bit generators that drive <see cref="Generator"/>.
    /// </summary>
    /// <remarks>
    ///     Mirrors NumPy 2.4.2's <c>numpy.random.BitGenerator</c>: a source of uniform 64-/32-bit
    ///     words plus the <c>[0,1)</c> float conversions the distribution kernels are built on.
    ///     <see cref="PCG64"/> is the concrete implementation used by <c>np.random.default_rng</c>.
    ///     The per-draw primitives (<c>NextUInt64</c> …) are the internal engine contract — NumPy
    ///     exposes only <c>random_raw</c> / <c>state</c> publicly — so they are <c>internal</c> and
    ///     the public surface stays NumPy-cased.
    /// </remarks>
    public abstract class BitGenerator
    {
        /// <summary>The next uniform 64-bit word.</summary>
        internal abstract ulong NextUInt64();

        /// <summary>The next uniform 32-bit word.</summary>
        internal abstract uint NextUInt32();

        /// <summary>A random double in <c>[0, 1)</c> with 53-bit precision (NumPy's <c>next_double</c>).</summary>
        internal virtual double NextDouble() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

        /// <summary>A random float in <c>[0, 1)</c> with 24-bit precision (NumPy's <c>next_float</c>).</summary>
        internal virtual float NextFloat() => (NextUInt32() >> 8) * (1.0f / 16777216.0f);

        /// <summary>The bit generator's name, e.g. <c>"PCG64"</c>. Drives <c>Generator</c>'s repr.</summary>
        internal abstract string Name { get; }
    }
}
