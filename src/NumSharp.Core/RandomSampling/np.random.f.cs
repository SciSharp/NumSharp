namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from an F distribution.
        /// </summary>
        /// <param name="dfnum">Degrees of freedom in numerator, must be &gt; 0.</param>
        /// <param name="dfden">Degrees of freedom in denominator, must be &gt; 0.</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="dfnum"/> or <paramref name="dfden"/> is <c>&lt;= 0</c>.</exception>
        public NDArray f(double dfnum, double dfden) => f(dfnum, dfden, Shape.Scalar);

        /// <summary>
        ///     Draw samples from an F distribution.
        /// </summary>
        /// <param name="dfnum">Degrees of freedom in numerator, must be &gt; 0 (NaN is accepted and samples NaN, as in NumPy).</param>
        /// <param name="dfden">Degrees of freedom in denominator, must be &gt; 0 (NaN is accepted and samples NaN).</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized Fisher distribution (float64).</returns>
        /// <exception cref="ValueError"><paramref name="dfnum"/> (<c>dfnum &lt;= 0</c>, checked first) or <paramref name="dfden"/>
        ///     (<c>dfden &lt;= 0</c>) is not positive, or <paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.f.html
        ///     <br/>
        ///     Samples are drawn from an F distribution with specified parameters,
        ///     dfnum (degrees of freedom in numerator) and dfden (degrees of freedom in denominator).
        ///     <br/>
        ///     NumPy's <c>legacy_f</c>: <c>(chi2(dfnum) * dfden) / (chi2(dfden) * dfnum)</c> with the numerator chi-square
        ///     drawn first, per value — byte-identical to <c>np.random.RandomState(seed).f</c> (the former whole-array
        ///     composition drew every numerator before any denominator, a different stream). Holds the bit generator's lock
        ///     for the draws.
        /// </remarks>
        public NDArray f(double dfnum, double dfden, Shape size)
        {
            RandomConstraints.Check(dfnum, "dfnum", ConstraintType.CONS_POSITIVE);
            RandomConstraints.Check(dfden, "dfden", ConstraintType.CONS_POSITIVE);

            // The per-call setup NumPy recomputes for every value, evaluated once (the same expressions — bit-neutral).
            var setup = new LegacyFSetup(dfnum, dfden);

            if (IsScalarDraw(size))
            {
                unsafe
                {
                    // A one-double buffer IS NumPy's per-draw call sequence.
                    double word;
                    var one = new DrawBufferDouble(randomizer, &word, 1);
                    lock (randomizer.@lock)
                        return NDArray.Scalar(LegacyF(ref one, in setup));
                }
            }

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                // Read-ahead draws (bulk-filled by the bit generator): a value draws unless BOTH chi-square shapes underflow to 0 (then per-draw).
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, setup.Draws ? DrawBufferDouble.Capacity : 1);
                lock (randomizer.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = LegacyF(ref src, in setup);
                    }
            }

            return ret;
        }
    }
}
