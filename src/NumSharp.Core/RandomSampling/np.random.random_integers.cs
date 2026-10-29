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
        /// <exception cref="ValueError">The bounds fall outside int64 (<c>high is out of bounds for int64</c>) or the interval is empty.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.random_integers.html
        ///     <br/>
        ///     This function is deprecated in NumPy in favour of <see cref="randint(long, long?, Shape, DType)"/>. It is
        ///     exactly <c>randint(low, high + 1, size, dtype='l')</c> — i.e. the closed interval <c>[low, high]</c>
        ///     rather than <c>randint</c>'s half-open <c>[low, high)</c>. The result dtype is the C <c>long</c>
        ///     (<c>np.dtype('l')</c>) — int64 in NumSharp's LP64 model (<see cref="LegacyLong"/>), NumPy 2.4.2's
        ///     Linux/macOS result; NumPy's Windows build returns int32 and rejects <c>random_integers(0, 2**31)</c>, which
        ///     LP64 NumPy draws. The <c>high + 1</c> is exact (arbitrary precision, as NumPy's Python int), so
        ///     <c>random_integers(0, 2**63 - 1)</c> reaches the int64 maximum rather than wrapping.
        /// </remarks>
        public NDArray random_integers(long low, long? high = null, Shape size = default)
        {
            Int128 lo, hiInclusive;
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

            // randint(low, int(high) + 1, size=size, dtype='l'): the shared masked core, with the randint name in
            // any dtype error (dtype='l' is always supported, so that message never fires here).
            return BoundedIntegers.Draw(randomizer, lo, hiInclusive + 1, size, LegacyLong, endpoint: false,
                                        useMasked: true, "randint", legacyByteOrder: true);
        }
    }
}
