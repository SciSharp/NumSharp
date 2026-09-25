using System;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a Weibull distribution.
        /// </summary>
        /// <param name="a">Shape parameter of the distribution. Must be non-negative.</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="a"/> is negative, including <c>-0.0</c> (<c>a &lt; 0</c>).</exception>
        public NDArray weibull(double a) => weibull(a, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a Weibull distribution.
        /// </summary>
        /// <param name="a">Shape parameter of the distribution. Must be non-negative (NaN is accepted and samples NaN, as in NumPy).</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the Weibull distribution (float64).</returns>
        /// <exception cref="ValueError">If a is negative, including <c>-0.0</c> (<c>a &lt; 0</c>), or <paramref name="size"/> has a
        ///     negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.weibull.html
        ///     <br/>
        ///     The Weibull distribution is a continuous probability distribution with
        ///     probability density function: f(x;a) = a * x^(a-1) * exp(-x^a) for x >= 0.
        ///     <br/>
        ///     This is the standard Weibull with scale=1. For Weibull with scale parameter,
        ///     use: scale * np.random.weibull(a, size).
        ///     <br/>
        ///     When a=1, the Weibull distribution reduces to the exponential distribution.
        ///     <br/>
        ///     NumPy's <c>legacy_weibull</c>: <c>(-log(1 - U))^(1/a)</c>, one uniform per value (bulk-filled, transformed in
        ///     place); <c>a == 0</c> returns zeros WITHOUT drawing, exactly as NumPy's early return does. Byte-identical to
        ///     <c>np.random.RandomState(seed).weibull</c>; holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray weibull(double a, Shape size)
        {
            RandomConstraints.Check(a, "a", ConstraintType.CONS_NON_NEGATIVE);

            if (IsScalarDraw(size))
                lock (randomizer.@lock)
                    return NDArray.Scalar(LegacyWeibull(a));

            if (a == 0.0)
                return np.zeros(new Shape(size.dimensions), NPTypeCode.Double); // legacy_weibull(0) draws nothing

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                lock (randomizer.@lock)
                    randomizer.FillDouble(dst, n);
                // The draws are all taken; legacy_weibull's transform of each, in place.
                double invA = 1.0 / a;
                for (long i = 0; i < n; i++)
                    dst[i] = Math.Pow(-Math.Log(1.0 - dst[i]), invA);
            }

            return ret;
        }
    }
}
