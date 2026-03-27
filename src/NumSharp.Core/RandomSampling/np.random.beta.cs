namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a Beta distribution.
        /// </summary>
        /// <param name="a">Alpha (α), positive (>0).</param>
        /// <param name="b">Beta (β), positive (>0).</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized Beta distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html
        ///     <br/>
        ///     The Beta distribution is a special case of the Dirichlet distribution,
        ///     and is related to the Gamma distribution.
        /// </remarks>
        public NDArray beta(double a, double b, Shape size) => beta(a, b, size.dimensions);

        /// <summary>
        ///     Draw samples from a Beta distribution.
        /// </summary>
        /// <param name="a">Alpha (α), positive (>0).</param>
        /// <param name="b">Beta (β), positive (>0).</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized Beta distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html
        ///     <br/>
        ///     The Beta distribution is a special case of the Dirichlet distribution,
        ///     and is related to the Gamma distribution.
        /// </remarks>
        public NDArray beta(double a, double b, params int[] size) => beta(a, b, Shape.ComputeLongShape(size));

        /// <summary>
        ///     Draw samples from a Beta distribution.
        /// </summary>
        /// <param name="a">Alpha (α), positive (>0).</param>
        /// <param name="b">Beta (β), positive (>0).</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized Beta distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html
        ///     <br/>
        ///     The Beta distribution is a special case of the Dirichlet distribution,
        ///     and is related to the Gamma distribution.
        /// </remarks>
        public NDArray beta(double a, double b, params long[] size)
        {
            var x = gamma(a, 1.0, size);
            var y = gamma(b, 1.0, size);
            return x / (x + y);
        }
    }
}
