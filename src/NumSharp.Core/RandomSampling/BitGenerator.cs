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
        /// <summary>
        ///     The lock every draw from this bit generator holds (NumPy's <c>BitGenerator.lock</c>).
        /// </summary>
        /// <remarks>
        ///     A <see cref="Generator"/> holds this monitor for the whole of a fill, exactly as NumPy's
        ///     <c>with self.lock:</c> does. The engine state (PCG64's two 128-bit words plus its buffered
        ///     32-bit half) is updated non-atomically, so two threads drawing at once tear it and emit values
        ///     that belong to no seed's stream — measured at 72–99% foreign values for four threads before
        ///     the lock existed. Generators built over the SAME bit generator share this object, so their
        ///     draws serialize together, as NumPy's <c>Generator.lock = bit_generator.lock</c> makes them.
        ///     The monitor is re-entrant (like NumPy's <c>RLock</c>), so a draw that goes through another
        ///     public method (e.g. <c>choice</c> calling <c>integers</c>) re-acquires it without deadlock.
        ///     Code that calls the internal primitives directly must hold it too.
        /// </remarks>
        public object @lock { get; } = new object();

        /// <summary>The next uniform 64-bit word.</summary>
        /// <returns>A uniformly distributed 64-bit value; advances the engine state.</returns>
        internal abstract ulong NextUInt64();

        /// <summary>The next uniform 32-bit word.</summary>
        /// <returns>A uniformly distributed 32-bit value; advances (or consumes a buffered half of) the engine state.</returns>
        internal abstract uint NextUInt32();

        /// <summary>A random double in <c>[0, 1)</c> with 53-bit precision (NumPy's <c>next_double</c>).</summary>
        /// <returns>The top 53 bits of one 64-bit word scaled by <c>2**-53</c>.</returns>
        internal virtual double NextDouble() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

        /// <summary>A random float in <c>[0, 1)</c> with 24-bit precision (NumPy's <c>next_float</c>).</summary>
        /// <returns>The top 24 bits of one 32-bit word scaled by <c>2**-24</c>.</returns>
        internal virtual float NextFloat() => (NextUInt32() >> 8) * (1.0f / 16777216.0f);

        /// <summary>The bit generator's name, e.g. <c>"PCG64"</c>. Drives <c>Generator</c>'s repr.</summary>
        internal abstract string Name { get; }
    }
}
