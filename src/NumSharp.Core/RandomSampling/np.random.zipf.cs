namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a Zipf distribution.
        /// </summary>
        /// <param name="a">Distribution parameter. Must be greater than 1.</param>
        /// <returns>A 0-d int64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="a"/> is <c>&lt;= 1</c> or NaN (<c>a &lt;= 1 or a is NaN</c>).</exception>
        public NDArray zipf(double a) => zipf(a, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a Zipf distribution.
        /// </summary>
        /// <param name="a">Distribution parameter. Must be greater than 1.</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized Zipf distribution as int64.</returns>
        /// <exception cref="ValueError"><paramref name="a"/> is <c>&lt;= 1</c> or NaN (<c>a &lt;= 1 or a is NaN</c>), or
        ///     <paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.zipf.html
        ///     <br/>
        ///     The Zipf distribution (also known as the zeta distribution) is a discrete
        ///     distribution commonly used to model the frequency of words in texts, the
        ///     size of cities, and many other phenomena.
        ///     <br/>
        ///     The probability mass function is:
        ///     p(k) = k^(-a) / zeta(a)
        ///     <br/>
        ///     where k >= 1 and zeta(a) is the Riemann zeta function.
        ///     <br/>
        ///     NumPy's <c>legacy_random_zipf</c>: rejection over <c>U = 1 - next_double</c> WITHOUT the modern sampler's
        ///     <c>Umin</c> window (the former code used the modern one — the same values for most <c>a</c> but a different
        ///     stream near <c>a = 1</c>). Byte-identical to <c>np.random.RandomState(seed).zipf</c>; int64 output (NumPy returns
        ///     C <c>long</c>).
        ///     <br/>
        ///     One deliberate difference: for <c>a &gt;= 1025</c> (and <c>a = inf</c>) NumPy's legacy loop compares NaN forever
        ///     and never returns; NumSharp returns 1 — the value the distribution degenerates to, and the modern sampler's
        ///     answer — without drawing. Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray zipf(double a, Shape size)
        {
            RandomConstraints.Check(a, "a", ConstraintType.CONS_GT_1);

            // The per-call setup NumPy recomputes for every value, evaluated once (the same expressions — bit-neutral).
            var setup = new LegacyZipfSetup(a);

            if (IsScalarDraw(size))
            {
                unsafe
                {
                    // A one-double buffer IS NumPy's per-draw call sequence.
                    double word;
                    var one = new DrawBufferDouble(randomizer, &word, 1);
                    lock (randomizer.@lock)
                        return NDArray.Scalar(LegacyZipf(ref one, in setup, null));
                }
            }

            var ret = LegacyOutput(NPTypeCode.Int64, size);
            unsafe
            {
                var dst = (long*)ret.Address;
                long n = ret.size;
                // Read-ahead draws (bulk-filled by the bit generator): every value draws unless a >= 1025 (1 without drawing — per-draw).
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, setup.Draws ? DrawBufferDouble.Capacity : 1);
                // A memo of the acceptance test's pow(1 + 1/X, a - 1) for the small candidates X (bit-neutral — see LegacyZipf),
                // for fills long enough to repay it.
                double[] tMemo = null;
                if (n >= ZipfMemoMinFill && setup.Draws)
                {
                    tMemo = new double[256];
                    System.Array.Fill(tMemo, double.NaN);
                }
                lock (randomizer.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = LegacyZipf(ref src, in setup, tMemo);
                    }
            }

            return ret;
        }

        /// <summary>The smallest zipf fill that builds the acceptance-test memo (below it the memo costs more than it saves).</summary>
        private const long ZipfMemoMinFill = 16;
    }
}
