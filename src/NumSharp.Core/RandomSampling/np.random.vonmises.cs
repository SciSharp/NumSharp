namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a von Mises distribution.
        /// </summary>
        /// <param name="mu">Mode ("center") of the distribution in radians.</param>
        /// <param name="kappa">Concentration parameter of the distribution. Must be &gt;= 0.</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="kappa"/> is negative, including <c>-0.0</c> (<c>kappa &lt; 0</c>).</exception>
        public NDArray vonmises(double mu, double kappa) => vonmises(mu, kappa, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a von Mises distribution.
        /// </summary>
        /// <param name="mu">Mode ("center") of the distribution in radians.</param>
        /// <param name="kappa">
        ///     Concentration parameter of the distribution. Must be >= 0.
        ///     When kappa = 0, the distribution is uniform on the circle.
        ///     As kappa increases, the distribution becomes more concentrated around mu.
        ///     NaN is accepted and returns NaN without drawing, as in NumPy.
        /// </param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized von Mises distribution, in [-pi, pi] (float64).</returns>
        /// <exception cref="ValueError">If kappa is negative, including <c>-0.0</c> (<c>kappa &lt; 0</c>), or
        ///     <paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.vonmises.html
        ///     <br/>
        ///     The von Mises distribution (also known as the circular normal distribution)
        ///     is a continuous probability distribution on the unit circle. It may be thought
        ///     of as the circular analogue of the normal distribution.
        ///     <br/>
        ///     <br/>
        ///     The probability density function is:
        ///     p(x) = exp(kappa * cos(x - mu)) / (2 * pi * I_0(kappa))
        ///     where I_0(kappa) is the modified Bessel function of order 0.
        ///     <br/>
        ///     NumPy's <c>legacy_vonmises</c>: Best &amp; Fisher's rejection for every <c>kappa &gt;= 1e-8</c> (the legacy code
        ///     has NO wrapped-normal fallback above <c>1e6</c> — the former code applied the modern one and diverged there),
        ///     a uniform on <c>[-pi, pi)</c> below. Byte-identical to <c>np.random.RandomState(seed).vonmises</c>.
        ///     <br/>
        ///     One deliberate difference: once <c>4*kappa^2</c> overflows (<c>kappa &gt; ~6.7e153</c> or infinite) NumPy's loop
        ///     compares NaN forever and never returns; NumSharp returns <paramref name="mu"/> wrapped to <c>[-pi, pi]</c> —
        ///     the distribution's limit — without drawing. Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray vonmises(double mu, double kappa, Shape size)
        {
            RandomConstraints.Check(kappa, "kappa", ConstraintType.CONS_NON_NEGATIVE);

            // The per-call setup NumPy recomputes for every value, evaluated once (the same expressions — bit-neutral).
            var setup = new LegacyVonmisesSetup(mu, kappa);

            if (IsScalarDraw(size))
            {
                unsafe
                {
                    // A one-double buffer IS NumPy's per-draw call sequence.
                    double word;
                    var one = new DrawBufferDouble(randomizer, &word, 1);
                    lock (randomizer.@lock)
                        return NDArray.Scalar(LegacyVonmises(ref one, in setup));
                }
            }

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                // Read-ahead draws (bulk-filled by the bit generator): every value draws unless kappa is NaN or overflows the envelope (then per-draw).
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, setup.Draws ? DrawBufferDouble.Capacity : 1);
                lock (randomizer.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = LegacyVonmises(ref src, in setup);
                    }
            }

            return ret;
        }
    }
}
