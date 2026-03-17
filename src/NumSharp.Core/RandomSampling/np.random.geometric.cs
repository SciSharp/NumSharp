using System;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from the geometric distribution.
        /// </summary>
        /// <param name="p">The probability of success of an individual trial.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized geometric distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.geometric.html
        ///     <br/>
        ///     Bernoulli trials are experiments with one of two outcomes: success or failure
        ///     (an example of such an experiment is flipping a coin). The geometric distribution
        ///     models the number of trials that must be run in order to achieve success.
        ///     It is therefore supported on the positive integers, k = 1, 2, ...
        /// </remarks>
        public NDArray geometric(double p, Shape size) => geometric(p, size.dimensions);

        /// <summary>
        ///     Draw samples from the geometric distribution.
        /// </summary>
        /// <param name="p">The probability of success of an individual trial.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized geometric distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.geometric.html
        ///     <br/>
        ///     Bernoulli trials are experiments with one of two outcomes: success or failure
        ///     (an example of such an experiment is flipping a coin). The geometric distribution
        ///     models the number of trials that must be run in order to achieve success.
        ///     It is therefore supported on the positive integers, k = 1, 2, ...
        /// </remarks>
        public NDArray geometric(double p, params int[] size)
        {
            var x = np.log(1 - uniform(0, 1, size));
            return x / (Math.Log(p) + 1);
        }
    }
}
