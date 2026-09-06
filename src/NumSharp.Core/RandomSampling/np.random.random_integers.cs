using System;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Random integers of type <c>np.int_</c> between <paramref name="low"/> and
        ///     <paramref name="high"/>, inclusive.
        /// </summary>
        /// <param name="low">
        ///     Lowest (signed) integer to be drawn from the distribution (unless
        ///     <paramref name="high"/> is <c>null</c>, in which case this is the *highest* integer).
        /// </param>
        /// <param name="high">
        ///     If provided, the largest (signed) integer to be drawn. If <c>null</c> (the default),
        ///     results are from <c>[1, low]</c>.
        /// </param>
        /// <param name="size">Output shape. If default/scalar, a single value is returned.</param>
        /// <returns>
        ///     <paramref name="size"/>-shaped array of random integers from the closed interval
        ///     <c>[low, high]</c>, or a single such int if <paramref name="size"/> is not provided.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.random_integers.html
        ///     <br/>
        ///     This function is deprecated in NumPy in favour of <see cref="randint"/>. It is exactly
        ///     <c>randint(low, high + 1, size, dtype='l')</c> — i.e. the closed interval
        ///     <c>[low, high]</c> rather than <c>randint</c>'s half-open <c>[low, high)</c>. The result
        ///     dtype is the C <c>long</c> (<c>np.dtype('l')</c>), which is 32-bit on the win-amd64
        ///     reference platform, so the output is <c>int32</c> — matching NumPy 2.4.2 on Windows.
        /// </remarks>
        public NDArray random_integers(long low, long? high = null, Shape size = default)
        {
            long lo, hiInclusive;
            if (high == null)
            {
                // random_integers(low) -> [1, low]
                hiInclusive = low;
                lo = 1;
            }
            else
            {
                lo = low;
                hiInclusive = high.Value;
            }

            // random_integers is randint over the CLOSED interval, so the exclusive high is +1.
            long hiExclusive = hiInclusive + 1;

            // dtype='l' == C long == int32 on the win-amd64 reference build.
            const NPTypeCode typecode = NPTypeCode.Int32;

            // Reuse randint's validation (verbatim NumPy bounds errors) and fill helpers directly,
            // bypassing randint's high == -1 "not provided" sentinel (which random_integers, having
            // resolved both bounds, must not trigger).
            ValidateRandintBounds(lo, hiExclusive, typecode);

            bool needsLongRange = hiExclusive > int.MaxValue || lo < int.MinValue || (hiExclusive - lo) > int.MaxValue;

            if (size.IsEmpty || size.IsScalar)
            {
                long value = needsLongRange
                    ? randomizer.NextLong(lo, hiExclusive)
                    : randomizer.Next((int)lo, (int)hiExclusive);
                return NDArray.Scalar(value, typecode);
            }

            var nd = new NDArray(np.int32, size);
            if (needsLongRange)
                FillRandintLong(nd, lo, hiExclusive, typecode);
            else
                FillRandintInt(nd, (int)lo, (int)hiExclusive, typecode);

            return nd;
        }
    }
}
