namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a Gumbel distribution.
        /// </summary>
        /// <param name="loc">The location of the mode of the distribution. Default is 0.</param>
        /// <param name="scale">The scale parameter of the distribution. Must be non-negative. Default is 1.</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="scale"/> is negative, including <c>-0.0</c> (<c>scale &lt; 0</c>).</exception>
        public NDArray gumbel(double loc = 0.0, double scale = 1.0) => gumbel(loc, scale, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a Gumbel distribution (extreme value type I).
        /// </summary>
        /// <param name="loc">The location of the mode of the distribution. Default is 0.</param>
        /// <param name="scale">The scale parameter of the distribution. Must be non-negative. Default is 1.</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized Gumbel distribution (float64).</returns>
        /// <exception cref="ValueError"><paramref name="scale"/> is negative, including <c>-0.0</c> (<c>scale &lt; 0</c>), or
        ///     <paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.gumbel.html
        ///     <br/>
        ///     The Gumbel (or Smallest Extreme Value (SEV) or the Smallest Extreme Value Type I)
        ///     distribution is used to model the distribution of the maximum (or minimum) of a
        ///     number of samples of various distributions.
        ///     <br/>
        ///     The probability density function is:
        ///     p(x) = (1/scale) * exp(-(x-loc)/scale) * exp(-exp(-(x-loc)/scale))
        ///     <br/>
        ///     For Gumbel(loc, scale):
        ///     - mean = loc + scale * γ (where γ ≈ 0.5772 is the Euler-Mascheroni constant)
        ///     - std = scale * π / sqrt(6) ≈ 1.283 * scale
        ///     <br/>
        ///     NumPy's <c>random_gumbel</c>: <c>loc - scale * log(-log(U))</c> with <c>U = 1 - next_double</c>, redrawing
        ///     only when <c>U == 1</c>. <c>scale == 0</c> STILL consumes a uniform per value (the former shortcut skipped
        ///     the draw and desynchronized every later value). Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray gumbel(double loc, double scale, Shape size)
        {
            RandomConstraints.Check(scale, "scale", ConstraintType.CONS_NON_NEGATIVE);

            if (IsScalarDraw(size))
            {
                unsafe
                {
                    // A one-double buffer IS NumPy's per-draw call sequence.
                    double word;
                    var one = new DrawBufferDouble(randomizer, &word, 1);
                    lock (randomizer.@lock)
                        return NDArray.Scalar(Distributions.RandomGumbel(ref one, loc, scale));
                }
            }

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                // Read-ahead draws (bulk-filled by the bit generator): every value draws at least one uniform.
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity);
                lock (randomizer.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = Distributions.RandomGumbel(ref src, loc, scale);
                    }
            }

            return ret;
        }
    }
}
