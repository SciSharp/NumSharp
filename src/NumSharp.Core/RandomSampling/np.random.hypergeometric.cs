namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a Hypergeometric distribution.
        /// </summary>
        /// <param name="ngood">Number of ways to make a good selection. Must be non-negative.</param>
        /// <param name="nbad">Number of ways to make a bad selection. Must be non-negative.</param>
        /// <param name="nsample">Number of items sampled. Must be &gt;= 1 and &lt;= ngood + nbad.</param>
        /// <param name="size">Output shape; null (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the hypergeometric distribution (number of good items in sample, int64).</returns>
        /// <exception cref="ValueError">
        ///     In NumPy's order: <c>ngood + nbad &lt; nsample</c> (checked BEFORE the individual constraints), then
        ///     <c>ngood &lt; 0</c>, <c>nbad &lt; 0</c>, <c>nsample &lt; 1 or nsample is NaN</c>; or <paramref name="size"/> has a
        ///     negative dimension.
        /// </exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.hypergeometric.html
        ///     <br/>
        ///     Consider an urn with ngood white marbles and nbad black marbles.
        ///     If you draw nsample balls without replacement, the hypergeometric distribution
        ///     describes the distribution of white balls in the drawn sample.
        ///     <br/>
        ///     Mean = nsample * ngood / (ngood + nbad)
        ///     <br/>
        ///     NumPy's <c>legacy_random_hypergeometric</c>: the urn simulation (<c>hyp</c>) for <c>nsample &lt;= 10</c> and
        ///     Stadlober's ratio-of-uniforms (<c>hrua</c>) above — byte-identical to
        ///     <c>np.random.RandomState(seed).hypergeometric</c>. When <c>ngood</c> or <c>nbad</c> is 0 and
        ///     <c>nsample &lt;= 10</c> the answer is known and nothing is drawn, as in NumPy. int64 output (NumPy returns C
        ///     <c>long</c>). Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray hypergeometric(long ngood, long nbad, long nsample, Shape? size = null)
        {
            // NumPy tests the (wrapping) int64 sum before the per-parameter constraints.
            if (ngood + nbad < nsample)
                throw new ValueError("ngood + nbad < nsample");
            RandomConstraints.Check(ngood, "ngood", ConstraintType.LEGACY_CONS_NON_NEGATIVE_INBOUNDS_LONG);
            RandomConstraints.Check(nbad, "nbad", ConstraintType.CONS_NON_NEGATIVE);
            RandomConstraints.Check(nsample, "nsample", ConstraintType.CONS_GTE_1);

            if (size is null || IsScalarDraw(size.Value))
            {
                unsafe
                {
                    // A one-double buffer IS NumPy's per-draw call sequence.
                    double word;
                    var one = new DrawBufferDouble(randomizer, &word, 1);
                    lock (randomizer.@lock)
                        return NDArray.Scalar(LegacyHypergeometric(ref one, ngood, nbad, nsample));
                }
            }

            var ret = LegacyOutput(NPTypeCode.Int64, size.Value);
            unsafe
            {
                var dst = (long*)ret.Address;
                long n = ret.size;
                // Read-ahead draws (bulk-filled by the bit generator): every value draws unless nsample <= 10 and a colour is absent (the urn answer is then known — per-draw).
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, nsample > 10 || (ngood > 0 && nbad > 0) ? DrawBufferDouble.Capacity : 1);
                if (nsample > 10)
                {
                    // HRUA's setup (a sqrt, four loggams) once for the fill, and — for fills long enough to repay its 16 KB —
                    // a memo of the per-candidate loggam sums; both are bit-neutral (LegacyHruaSetup, LoggamSumMemo).
                    var setup = new LegacyHruaSetup(ngood, nbad, nsample);
                    var memo = n >= HypergeometricMemoMinFill ? new LoggamSumMemo() : null;
                    lock (randomizer.@lock)
                        for (long i = 0; i < n; i++)
                        {
                            src.Owed = n - i;
                            dst[i] = LegacyHypergeometricHrua(ref src, ngood, nbad, nsample, in setup, memo);
                        }
                }
                else
                {
                    // The urn walk over a precomputed ratio table (no division per step) when its integer walk is exact
                    // and the fill repays the table's <= 110 divisions; otherwise NumPy's walk as written.
                    var table = n >= HypergeometricMemoMinFill && LegacyHypTable.Applicable(ngood, nbad)
                        ? new LegacyHypTable(ngood, nbad, nsample)
                        : null;
                    lock (randomizer.@lock)
                        for (long i = 0; i < n; i++)
                        {
                            src.Owed = n - i;
                            dst[i] = table != null
                                ? LegacyHypergeometricHyp(ref src, ngood, nbad, nsample, table)
                                : LegacyHypergeometricHyp(ref src, ngood, nbad, nsample);
                        }
                }
            }

            return ret;
        }

        /// <summary>
        ///     The smallest fill that builds a hypergeometric memo (HRUA's loggam-sum memo or the urn walk's ratio table):
        ///     below it, building the memo costs more than the values it would speed up.
        /// </summary>
        private const long HypergeometricMemoMinFill = 16;

        /// <summary>
        ///     Draw samples from a Hypergeometric distribution into a given (non-nullable) output shape.
        /// </summary>
        /// <param name="ngood">Number of ways to make a good selection. Must be non-negative.</param>
        /// <param name="nbad">Number of ways to make a bad selection. Must be non-negative.</param>
        /// <param name="nsample">Number of items sampled. Must be &gt;= 1 and &lt;= ngood + nbad.</param>
        /// <param name="size">Output shape; <c>default</c> / <see cref="Shape.Scalar"/> draw a single value (NumPy's
        ///     <c>None</c>, this class's convention).</param>
        /// <returns>Drawn samples from the hypergeometric distribution (int64).</returns>
        /// <exception cref="ValueError">A parameter violates its constraint (see <see cref="hypergeometric(long, long, long, Shape?)"/>)
        ///     or a size dimension is negative.</exception>
        /// <remarks>
        ///     Exists for overload resolution, not behaviour: with the array-parameter
        ///     <see cref="hypergeometric(NDArray, NDArray, NDArray, Shape)"/> beside the <see cref="Nullable{T}"/> form, a call
        ///     passing integers and a <see cref="Shape"/> was better than each candidate in one argument (the integers bind
        ///     the scalar form, the <see cref="Shape"/> binds the array form without the nullable wrap) — an ambiguity error.
        ///     This exact match wins every such call and forwards to the scalar sampler unchanged.
        /// </remarks>
        public NDArray hypergeometric(long ngood, long nbad, long nsample, Shape size)
            => hypergeometric(ngood, nbad, nsample, (Shape?)size);

        /// <summary>
        ///     Draw samples from a Hypergeometric distribution.
        /// </summary>
        /// <param name="ngood">Number of ways to make a good selection. Must be non-negative.</param>
        /// <param name="nbad">Number of ways to make a bad selection. Must be non-negative.</param>
        /// <param name="nsample">Number of items sampled. Must be &gt;= 1 and &lt;= ngood + nbad.</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the hypergeometric distribution.</returns>
        /// <exception cref="ValueError">A parameter violates its constraint (see <see cref="hypergeometric(long, long, long, Shape?)"/>)
        ///     or a size dimension is negative.</exception>
        public NDArray hypergeometric(long ngood, long nbad, long nsample, int[] size)
            => hypergeometric(ngood, nbad, nsample, new Shape(size));

        /// <summary>
        ///     Draw samples from a Hypergeometric distribution.
        /// </summary>
        /// <param name="ngood">Number of ways to make a good selection. Must be non-negative.</param>
        /// <param name="nbad">Number of ways to make a bad selection. Must be non-negative.</param>
        /// <param name="nsample">Number of items sampled. Must be &gt;= 1 and &lt;= ngood + nbad.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the hypergeometric distribution.</returns>
        /// <exception cref="ValueError">A parameter violates its constraint (see <see cref="hypergeometric(long, long, long, Shape?)"/>)
        ///     or a size dimension is negative.</exception>
        public NDArray hypergeometric(long ngood, long nbad, long nsample, long[] size)
            => hypergeometric(ngood, nbad, nsample, new Shape(size));

        /// <summary>
        ///     Draw samples from a Hypergeometric distribution.
        /// </summary>
        /// <param name="ngood">Number of ways to make a good selection. Must be non-negative.</param>
        /// <param name="nbad">Number of ways to make a bad selection. Must be non-negative.</param>
        /// <param name="nsample">Number of items sampled. Must be &gt;= 1 and &lt;= ngood + nbad.</param>
        /// <param name="size">Output shape as single int.</param>
        /// <returns>Drawn samples from the hypergeometric distribution.</returns>
        /// <exception cref="ValueError">A parameter violates its constraint (see <see cref="hypergeometric(long, long, long, Shape?)"/>)
        ///     or <paramref name="size"/> is negative.</exception>
        public NDArray hypergeometric(long ngood, long nbad, long nsample, int size)
            => hypergeometric(ngood, nbad, nsample, new int[] { size });

        /// <summary>
        ///     Draw a single sample from a Hypergeometric distribution.
        /// </summary>
        /// <param name="ngood">Number of ways to make a good selection. Must be non-negative.</param>
        /// <param name="nbad">Number of ways to make a bad selection. Must be non-negative.</param>
        /// <param name="nsample">Number of items sampled. Must be &gt;= 1 and &lt;= ngood + nbad.</param>
        /// <returns>A single sample from the hypergeometric distribution as 0-d array.</returns>
        /// <exception cref="ValueError">A parameter violates its constraint (see <see cref="hypergeometric(long, long, long, Shape?)"/>).</exception>
        public NDArray hypergeometric(long ngood, long nbad, long nsample)
            => hypergeometric(ngood, nbad, nsample, (Shape?)null);
    }
}
