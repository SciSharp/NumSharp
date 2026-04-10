namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from an exponential distribution.
        /// </summary>
        /// <param name="scale">The scale parameter, β = 1/λ. Must be non-negative. Default is 1.0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized exponential distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html
        ///     <br/>
        ///     The exponential distribution is a continuous analogue of the geometric distribution.
        ///     It describes many common situations, such as the size of raindrops measured over
        ///     many rainstorms, or the time between page requests to Wikipedia.
        /// </remarks>
        public NDArray exponential(double scale, Shape size) => exponential(scale, size.dimensions);

        /// <summary>
        ///     Draw samples from an exponential distribution.
        /// </summary>
        /// <param name="scale">The scale parameter, β = 1/λ. Must be non-negative. Default is 1.0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized exponential distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html
        ///     <br/>
        ///     The exponential distribution is a continuous analogue of the geometric distribution.
        ///     It describes many common situations, such as the size of raindrops measured over
        ///     many rainstorms, or the time between page requests to Wikipedia.
        /// </remarks>
        public NDArray exponential(double scale, int[] size) => exponential(scale, Shape.ComputeLongShape(size));

        /// <summary>
        ///     Draw samples from an exponential distribution.
        /// </summary>
        /// <param name="scale">The scale parameter, β = 1/λ. Must be non-negative. Default is 1.0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized exponential distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html
        ///     <br/>
        ///     The exponential distribution is a continuous analogue of the geometric distribution.
        ///     It describes many common situations, such as the size of raindrops measured over
        ///     many rainstorms, or the time between page requests to Wikipedia.
        /// </remarks>
        public NDArray exponential(double scale, params long[] size)
        {
            var x = np.log(1 - uniform(0, 1, size));
            return np.negative(x) * scale;
        }
    }
}
