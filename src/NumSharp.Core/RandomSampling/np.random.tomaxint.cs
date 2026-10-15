namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Return a sample of uniformly distributed random integers in the interval <c>[0, LONG_MAX]</c> (NumPy's legacy
        ///     <c>RandomState.tomaxint()</c> with <c>size=None</c>).
        /// </summary>
        /// <returns>A 0-d int64 array holding the draw.</returns>
        public NDArray tomaxint() => tomaxint(Shape.Scalar);

        /// <summary>
        ///     Return a sample of uniformly distributed random integers in the interval <c>[0, LONG_MAX]</c> (NumPy's legacy
        ///     <c>RandomState.tomaxint(size)</c>).
        /// </summary>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn integers (int64 — NumPy's dtype on every platform).</returns>
        /// <exception cref="ValueError"><paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     NumPy's <c>random_positive_int</c> in its legacy build, where C <c>long</c> sets the width: one 64-bit draw
        ///     shifted right by one (<c>next_uint64 &gt;&gt; 1</c>, values in <c>[0, 2**63 - 1]</c>) — NumPy's LP64 (Linux/macOS)
        ///     result, consistent with NumSharp's 64-bit-<c>long</c> legacy model (see <see cref="poisson(double, Shape)"/>).
        ///     NumPy's win-amd64 build, whose <c>long</c> is 32-bit, draws <c>next_uint32 &gt;&gt; 1</c> instead (values below
        ///     <c>2**31</c>, a different stream). For MT19937 a 64-bit draw is two 32-bit words, high word first. Holds the bit
        ///     generator's lock for the draws.
        /// </remarks>
        public NDArray tomaxint(Shape size)
        {
            if (IsScalarDraw(size))
                lock (randomizer.@lock)
                    return NDArray.Scalar((long)(randomizer.NextUInt64() >> 1));

            var ret = LegacyOutput(NPTypeCode.Int64, size);
            unsafe
            {
                // The raw 64-bit words are written in place, then halved: each value consumes exactly one next_uint64.
                var dst = (ulong*)ret.Address;
                long n = ret.size;
                lock (randomizer.@lock)
                    randomizer.FillUInt64(dst, n);
                for (long i = 0; i < n; i++)
                    dst[i] >>= 1;
            }

            return ret;
        }
    }
}
