using System;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draws samples in [0, 1] from a power distribution with positive exponent a - 1.
        /// </summary>
        /// <param name="a">Shape parameter of the distribution. Must be positive (&gt; 0; NaN is accepted and samples NaN, as in NumPy).</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized power distribution, in range [0, 1] (float64).</returns>
        /// <exception cref="ValueError"><paramref name="a"/> is <c>&lt;= 0</c> (<c>a &lt;= 0</c>), or <paramref name="size"/> has a
        ///     negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.power.html
        ///     <br/>
        ///     Also known as the power function distribution. The probability density function is:
        ///     P(x; a) = a * x^(a-1), for 0 &lt;= x &lt;= 1, a &gt; 0.
        ///     <br/>
        ///     The power function distribution is the inverse of the Pareto distribution.
        ///     It may also be seen as a special case of the Beta distribution.
        ///     <br/>
        ///     NumPy's <c>legacy_power</c>: <c>(1 - exp(-E))^(1/a)</c> with <c>E = -log(1 - U)</c> — algebraically
        ///     <c>U^(1/a)</c>, but rounded through the exponential round trip, which the former <c>U^(1/a)</c> did not
        ///     reproduce. One draw per value (bulk-filled, transformed in place); byte-identical to
        ///     <c>np.random.RandomState(seed).power</c>. Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray power(double a, Shape size)
        {
            RandomConstraints.Check(a, "a", ConstraintType.CONS_POSITIVE);

            if (IsScalarDraw(size))
                lock (randomizer.@lock)
                    return NDArray.Scalar(LegacyPower(a));

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                lock (randomizer.@lock)
                    randomizer.FillDouble(dst, n);
                // The draws are all taken; legacy_power's transform of each, in place.
                for (long i = 0; i < n; i++)
                {
                    double e = -Math.Log(1.0 - dst[i]);
                    dst[i] = Math.Pow(1 - Math.Exp(-e), 1.0 / a);
                }
            }

            return ret;
        }

        /// <summary>
        ///     Draws samples in [0, 1] from a power distribution with positive exponent a - 1.
        /// </summary>
        /// <param name="a">Shape parameter of the distribution. Must be positive (&gt; 0).</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the parameterized power distribution, in range [0, 1].</returns>
        /// <exception cref="ValueError"><paramref name="a"/> is <c>&lt;= 0</c>, or a size dimension is negative.</exception>
        public NDArray power(double a, int[] size)
            => power(a, new Shape(size));

        /// <summary>
        ///     Draws samples in [0, 1] from a power distribution with positive exponent a - 1.
        /// </summary>
        /// <param name="a">Shape parameter of the distribution. Must be positive (&gt; 0).</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized power distribution, in range [0, 1].</returns>
        /// <exception cref="ValueError"><paramref name="a"/> is <c>&lt;= 0</c>, or a size dimension is negative.</exception>
        public NDArray power(double a, long[] size)
            => power(a, new Shape(size));

        /// <summary>
        ///     Draws samples in [0, 1] from a power distribution with positive exponent a - 1.
        /// </summary>
        /// <param name="a">Shape parameter of the distribution. Must be positive (&gt; 0).</param>
        /// <param name="size">Output shape as single int.</param>
        /// <returns>Drawn samples from the parameterized power distribution, in range [0, 1].</returns>
        /// <exception cref="ValueError"><paramref name="a"/> is <c>&lt;= 0</c>, or <paramref name="size"/> is negative.</exception>
        public NDArray power(double a, int size)
            => power(a, new int[] { size });

        /// <summary>
        ///     Draws a single sample in [0, 1] from a power distribution with positive exponent a - 1.
        /// </summary>
        /// <param name="a">Shape parameter of the distribution. Must be positive (&gt; 0).</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="a"/> is <c>&lt;= 0</c>.</exception>
        public NDArray power(double a) => power(a, Shape.Scalar);
    }
}
