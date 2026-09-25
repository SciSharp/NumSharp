using System;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a Rayleigh distribution.
        /// </summary>
        /// <param name="scale">Scale parameter (also equals the mode). Must be non-negative. Default is 1.</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="scale"/> is negative, including <c>-0.0</c> (<c>scale &lt; 0</c>).</exception>
        public NDArray rayleigh(double scale = 1.0) => rayleigh(scale, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a Rayleigh distribution.
        /// </summary>
        /// <param name="scale">Scale parameter (also equals the mode). Must be non-negative. Default is 1.</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized Rayleigh distribution (float64).</returns>
        /// <exception cref="ValueError"><paramref name="scale"/> is negative, including <c>-0.0</c> (<c>scale &lt; 0</c>), or
        ///     <paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.rayleigh.html
        ///     <br/>
        ///     The probability density function for the Rayleigh distribution is:
        ///     P(x; scale) = (x / scale^2) * exp(-x^2 / (2 * scale^2))
        ///     <br/>
        ///     The Rayleigh distribution arises when the East and North components of wind
        ///     velocity have identical zero-mean Gaussian distributions. Then the wind speed
        ///     would have a Rayleigh distribution.
        ///     <br/>
        ///     For Rayleigh(scale), mean = scale * sqrt(pi/2) ≈ 1.253 * scale
        ///     <br/>
        ///     NumPy's <c>legacy_rayleigh</c>: <c>scale * sqrt(-2 * log1p(-U))</c>, one uniform per value — including
        ///     <c>scale == 0</c>, which the former shortcut answered without drawing. Bulk-filled and transformed in place;
        ///     byte-identical to <c>np.random.RandomState(seed).rayleigh</c>. Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray rayleigh(double scale, Shape size)
        {
            RandomConstraints.Check(scale, "scale", ConstraintType.CONS_NON_NEGATIVE);

            if (IsScalarDraw(size))
                lock (randomizer.@lock)
                    return NDArray.Scalar(LegacyRayleigh(scale));

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                lock (randomizer.@lock)
                    randomizer.FillDouble(dst, n);
                // The draws are all taken; legacy_rayleigh's transform of each, in place.
                for (long i = 0; i < n; i++)
                    dst[i] = scale * Math.Sqrt(-2.0 * Generator.Log1p(-dst[i]));
            }

            return ret;
        }
    }
}
