namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a Beta distribution.
        /// </summary>
        /// <param name="a">Alpha (α), positive (&gt;0).</param>
        /// <param name="b">Beta (β), positive (&gt;0).</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="a"/> or <paramref name="b"/> is <c>&lt;= 0</c> (<c>a &lt;= 0</c> / <c>b &lt;= 0</c>).</exception>
        public NDArray beta(double a, double b) => beta(a, b, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a Beta distribution.
        /// </summary>
        /// <param name="a">Alpha (α), positive (&gt;0). NaN is accepted and samples NaN, as in NumPy.</param>
        /// <param name="b">Beta (β), positive (&gt;0). NaN is accepted and samples NaN, as in NumPy.</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized Beta distribution (float64).</returns>
        /// <exception cref="ValueError"><paramref name="a"/> or <paramref name="b"/> is <c>&lt;= 0</c> (<c>a &lt;= 0</c> / <c>b &lt;= 0</c>),
        ///     or <paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html
        ///     <br/>
        ///     The Beta distribution is a special case of the Dirichlet distribution,
        ///     and is related to the Gamma distribution.
        ///     <br/>
        ///     NumPy's <c>legacy_beta</c>: Johnk's algorithm when both shapes are <c>&lt;= 1</c>, otherwise
        ///     <c>Ga / (Ga + Gb)</c> of two legacy gammas — byte-identical to <c>np.random.RandomState(seed).beta</c>.
        ///     Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray beta(double a, double b, Shape size)
        {
            RandomConstraints.Check(a, "a", ConstraintType.CONS_POSITIVE);
            RandomConstraints.Check(b, "b", ConstraintType.CONS_POSITIVE);

            // The per-call setup NumPy recomputes for every value, evaluated once (the same expressions — bit-neutral).
            var setup = new LegacyBetaSetup(a, b);

            if (IsScalarDraw(size))
            {
                unsafe
                {
                    // A one-double buffer IS NumPy's per-draw call sequence.
                    double word;
                    var one = new DrawBufferDouble(randomizer, &word, 1);
                    lock (randomizer.@lock)
                        return NDArray.Scalar(LegacyBeta(ref one, in setup));
                }
            }

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                // Read-ahead draws (bulk-filled by the bit generator): every value draws at least one uniform (Johnk or the gamma pair).
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity);
                lock (randomizer.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = LegacyBeta(ref src, in setup);
                    }
            }

            return ret;
        }
    }
}
