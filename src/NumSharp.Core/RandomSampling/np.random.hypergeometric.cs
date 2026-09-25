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
                lock (randomizer.@lock)
                    return NDArray.Scalar(LegacyHypergeometric(ngood, nbad, nsample));

            var ret = LegacyOutput(NPTypeCode.Int64, size.Value);
            unsafe
            {
                var dst = (long*)ret.Address;
                long n = ret.size;
                lock (randomizer.@lock)
                    for (long i = 0; i < n; i++)
                        dst[i] = LegacyHypergeometric(ngood, nbad, nsample);
            }

            return ret;
        }

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
