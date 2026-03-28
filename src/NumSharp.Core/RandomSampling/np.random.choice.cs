using System;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Generates a random sample from a given 1-D array.
        /// </summary>
        /// <param name="a">Array to sample from.</param>
        /// <param name="size">Output shape. Default is None, in which case a single value is returned.</param>
        /// <param name="replace">Whether the sample is with or without replacement. Default is True.</param>
        /// <param name="p">The probabilities associated with each entry in a. If not given, the sample assumes a uniform distribution over all entries.</param>
        /// <returns>The generated random samples.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        /// </remarks>
        public NDArray choice(NDArray a, Shape size = default, bool replace = true, double[] p = null)
        {
            long arrSize = a.size;
            NDArray mask = choice(arrSize, size, replace, p);
            return a[mask];
        }

        /// <summary>
        ///     Generates a random sample from np.arange(a).
        /// </summary>
        /// <param name="a">If a long, the random sample is generated from np.arange(a).</param>
        /// <param name="size">Output shape. Default is None, in which case a single value is returned.</param>
        /// <param name="replace">Whether the sample is with or without replacement. Default is True.</param>
        /// <param name="p">The probabilities associated with each entry. If not given, the sample assumes a uniform distribution.</param>
        /// <returns>The generated random samples.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        /// </remarks>
        public NDArray choice(long a, Shape size = default, bool replace = true, double[] p = null)
        {
            if (size.IsEmpty)
                size = Shape.Scalar;

            NDArray idx;

            if (p == null)
            {
                idx = randint(0, a, size);
            }
            else
            {
                NDArray cdf = np.cumsum(np.array(p));
                cdf /= cdf[cdf.size - 1];
                NDArray uniformSamples = uniform(0, 1, size);
                idx = np.searchsorted(cdf, uniformSamples);
            }

            return idx;
        }

        /// <summary>
        ///     Generates a random sample from np.arange(a).
        /// </summary>
        /// <param name="a">If an int, the random sample is generated from np.arange(a).</param>
        /// <param name="size">Output shape. Default is None, in which case a single value is returned.</param>
        /// <param name="replace">Whether the sample is with or without replacement. Default is True.</param>
        /// <param name="p">The probabilities associated with each entry. If not given, the sample assumes a uniform distribution.</param>
        /// <returns>The generated random samples.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        /// </remarks>
        public NDArray choice(int a, Shape size = default, bool replace = true, double[] p = null)
            => choice((long)a, size, replace, p);
    }
}
