namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a Wald distribution.
        /// </summary>
        /// <param name="mean">Distribution mean, must be &gt; 0.</param>
        /// <param name="scale">Scale parameter, must be &gt; 0.</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="mean"/> or <paramref name="scale"/> is <c>&lt;= 0</c>.</exception>
        public NDArray wald(double mean, double scale) => wald(mean, scale, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a Wald, or inverse Gaussian, distribution.
        /// </summary>
        /// <param name="mean">Distribution mean, must be &gt; 0 (NaN is accepted and samples NaN, as in NumPy).</param>
        /// <param name="scale">Scale parameter, must be &gt; 0 (NaN is accepted and samples NaN).</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized Wald distribution (float64).</returns>
        /// <exception cref="ValueError"><paramref name="mean"/> (<c>mean &lt;= 0</c>, checked first) or <paramref name="scale"/>
        ///     (<c>scale &lt;= 0</c>) is not positive, or <paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.wald.html
        ///     <br/>
        ///     The inverse Gaussian distribution was first studied in relationship to
        ///     Brownian motion. As the scale approaches infinity, the distribution
        ///     becomes more like a Gaussian.
        ///     <br/>
        ///     The probability density function is:
        ///     P(x;mean,scale) = sqrt(scale/(2*pi*x^3)) * exp(-scale*(x-mean)^2 / (2*mean^2*x))
        ///     <br/>
        ///     NumPy's <c>legacy_wald</c>: <c>X = mean + mu_2l * (Y - sqrt(4*scale*Y + Y*Y))</c> with <c>Y = mean * N^2</c> —
        ///     the LEGACY spelling (the former code used the Generator's cancellation-free rewrite, a few ULP off NumPy's
        ///     RandomState stream). Byte-identical to <c>np.random.RandomState(seed).wald</c>; holds the bit generator's lock
        ///     for the draws.
        /// </remarks>
        public NDArray wald(double mean, double scale, Shape size)
        {
            RandomConstraints.Check(mean, "mean", ConstraintType.CONS_POSITIVE);
            RandomConstraints.Check(scale, "scale", ConstraintType.CONS_POSITIVE);

            // The per-call setup NumPy recomputes for every value, evaluated once (the same expressions — bit-neutral).
            var setup = new LegacyWaldSetup(mean, scale);

            if (IsScalarDraw(size))
            {
                unsafe
                {
                    // A one-double buffer IS NumPy's per-draw call sequence.
                    double word;
                    var one = new DrawBufferDouble(randomizer, &word, 1);
                    lock (randomizer.@lock)
                        return NDArray.Scalar(LegacyWald(ref one, in setup));
                }
            }

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                // Read-ahead draws (bulk-filled by the bit generator), consumed by the two-phase fill (LegacyWaldFill): every
                // value draws at least its acceptance uniform.
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity);
                lock (randomizer.@lock)
                    LegacyWaldFill(ref src, dst, n, in setup);
            }

            return ret;
        }
    }
}
