using System;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from an exponential distribution.
        /// </summary>
        /// <param name="scale">The scale parameter, β = 1/λ. Must be non-negative. Default is 1.0.</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="scale"/> is negative, including <c>-0.0</c> (<c>scale &lt; 0</c>).</exception>
        public NDArray exponential(double scale = 1.0) => exponential(scale, Shape.Scalar);

        /// <summary>
        ///     Draw samples from an exponential distribution.
        /// </summary>
        /// <param name="scale">The scale parameter, β = 1/λ. Must be non-negative (NaN is accepted and samples NaN, as in NumPy).</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized exponential distribution (float64).</returns>
        /// <exception cref="ValueError"><paramref name="scale"/> is negative, including <c>-0.0</c> (<c>scale &lt; 0</c>), or
        ///     <paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html
        ///     <br/>
        ///     The exponential distribution is a continuous analogue of the geometric distribution.
        ///     It describes many common situations, such as the size of raindrops measured over
        ///     many rainstorms, or the time between page requests to Wikipedia.
        ///     <br/>
        ///     NumPy's <c>legacy_exponential</c>: <c>scale * -log(1 - U)</c>, one uniform per value. Because every value
        ///     consumes exactly one draw, the uniforms are produced by the bit generator's bulk fill straight into the
        ///     output and transformed in place — the same stream as NumPy's per-value calls, with no intermediate arrays.
        ///     Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray exponential(double scale, Shape size)
        {
            RandomConstraints.Check(scale, "scale", ConstraintType.CONS_NON_NEGATIVE);

            if (IsScalarDraw(size))
                lock (randomizer.@lock)
                    return NDArray.Scalar(LegacyExponential(scale));

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                lock (randomizer.@lock)
                    randomizer.FillDouble(dst, n);
                // The draws are all taken; the transform no longer needs the lock.
                for (long i = 0; i < n; i++)
                    dst[i] = scale * -Math.Log(1.0 - dst[i]);
            }

            return ret;
        }
    }
}
