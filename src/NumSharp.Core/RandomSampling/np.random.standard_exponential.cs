using System;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from the standard exponential distribution.
        /// </summary>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        public NDArray standard_exponential() => standard_exponential(Shape.Scalar);

        /// <summary>
        ///     Draw samples from the standard exponential distribution.
        /// </summary>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the standard exponential distribution (scale=1, float64).</returns>
        /// <exception cref="ValueError"><paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_exponential.html
        ///     <br/>
        ///     The standard exponential distribution is the exponential distribution with scale=1.
        ///     It has mean=1 and variance=1.
        ///     <br/>
        ///     Equivalent to: exponential(scale=1.0, size=size)
        ///     <br/>
        ///     Uses inverse transform: X = -log(1 - U) where U ~ Uniform[0, 1) — NumPy's
        ///     <c>legacy_standard_exponential</c>, which does NOT redraw a zero uniform (U = 0 yields <c>-0.0</c>). One draw
        ///     per value, so the uniforms come from the bit generator's bulk fill and are transformed in place.
        ///     Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray standard_exponential(Shape size)
        {
            if (IsScalarDraw(size))
                lock (randomizer.@lock)
                    return NDArray.Scalar(LegacyStandardExponential());

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                lock (randomizer.@lock)
                    randomizer.FillDouble(dst, n);
                // The draws are all taken; the transform no longer needs the lock.
                for (long i = 0; i < n; i++)
                    dst[i] = -Math.Log(1.0 - dst[i]);
            }

            return ret;
        }
    }
}
