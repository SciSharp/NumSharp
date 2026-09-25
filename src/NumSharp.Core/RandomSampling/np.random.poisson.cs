namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a Poisson distribution.
        /// </summary>
        /// <param name="lam">Expected number of events occurring in a fixed-time interval, must be &gt;= 0. Default is 1.0.</param>
        /// <returns>A 0-d int64 array holding the count.</returns>
        /// <exception cref="ValueError"><paramref name="lam"/> is negative or NaN, or too large.</exception>
        public NDArray poisson(double lam = 1.0) => poisson(lam, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a Poisson distribution.
        /// </summary>
        /// <param name="lam">Expected number of events occurring in a fixed-time interval, must be &gt;= 0. Default is 1.0.</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized Poisson distribution (int64).</returns>
        /// <exception cref="ValueError"><paramref name="lam"/> is negative or NaN (<c>lam &lt; 0 or lam is NaN</c> — <c>-0.0</c>
        ///     passes), exceeds <see cref="_poisson_lam_max"/> (<c>lam value too large</c>), or <paramref name="size"/> has a
        ///     negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.poisson.html
        ///     <br/>
        ///     The Poisson distribution is the limit of the binomial distribution for large N.
        ///     <br/>
        ///     NumPy's <c>legacy_random_poisson</c>: the product-of-uniforms method for <c>lam &lt; 10</c>, Hörmann's PTRS
        ///     transformed rejection for <c>lam &gt;= 10</c> (the former Knuth-only loop drew ~lam uniforms per value and a
        ///     different stream), and 0 without a draw for <c>lam == 0</c>. Byte-identical to
        ///     <c>np.random.RandomState(seed).poisson</c>. int64 output with the int64 bound — NumPy's LP64 shape; its win-amd64
        ///     build returns int32 and stops at <c>lam = 2147020237.4999895</c>. Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray poisson(double lam, Shape size)
        {
            RandomConstraints.Check(lam, "lam", ConstraintType.LEGACY_CONS_POISSON);

            // The per-call setup NumPy recomputes for every value, evaluated once (the same expressions — bit-neutral).
            var setup = new PoissonSetup(lam, hoistRejectionLog: !IsScalarDraw(size));

            if (IsScalarDraw(size))
            {
                unsafe
                {
                    // A one-double buffer IS NumPy's per-draw call sequence.
                    double word;
                    var one = new DrawBufferDouble(randomizer, &word, 1);
                    lock (randomizer.@lock)
                        return NDArray.Scalar(Distributions.RandomPoisson(ref one, in setup));
                }
            }

            var ret = LegacyOutput(NPTypeCode.Int64, size);
            unsafe
            {
                var dst = (long*)ret.Address;
                long n = ret.size;
                // Read-ahead draws (bulk-filled by the bit generator): every value draws unless lam == 0 (0 without a draw — per-draw).
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, setup.Draws ? DrawBufferDouble.Capacity : 1);
                lock (randomizer.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = Distributions.RandomPoisson(ref src, in setup);
                    }
            }

            return ret;
        }
    }
}
