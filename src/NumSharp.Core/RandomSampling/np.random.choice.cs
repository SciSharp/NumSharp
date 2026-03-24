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
            // choice operates on 1D arrays which are practically limited to int range
            if (a.size > int.MaxValue)
                throw new ArgumentException($"Array size {a.size} exceeds maximum supported size for choice", nameof(a));
            int arrSize = (int)a.size;
            NDArray mask = choice(arrSize, size, replace, p);
            return a[mask];
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
        {
            if (size.IsEmpty)
                size = Shape.Scalar;

            NDArray arr = np.arange(a);
            NDArray idx;

            if (p == null)
            {
                idx = randint(0, arr.size, size);
            }
            else
            {
                NDArray cdf = np.cumsum(p);
                cdf /= cdf[cdf.size - 1];
                NDArray uniformSamples = uniform(0, 1, (int[])size);
                idx = np.searchsorted(cdf, uniformSamples);
            }

            return idx;
        }
    }
}
