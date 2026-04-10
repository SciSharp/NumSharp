using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a standard Gamma distribution (scale=1).
        /// </summary>
        /// <param name="shape">The shape of the gamma distribution. Must be >= 0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the standard gamma distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_gamma.html
        ///     <br/>
        ///     Samples are drawn from a Gamma distribution with shape parameter and scale=1.
        ///     For a different scale, multiply the result: scale * standard_gamma(shape).
        ///     <br/>
        ///     The probability density function is:
        ///     p(x) = x^(shape-1) * e^(-x) / Gamma(shape)
        /// </remarks>
        public NDArray standard_gamma(double shape, Shape size)
            => standard_gamma(shape, size.dimensions);

        /// <summary>
        ///     Draw samples from a standard Gamma distribution (scale=1).
        /// </summary>
        /// <param name="shape">The shape of the gamma distribution. Must be >= 0.</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the standard gamma distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_gamma.html
        ///     <br/>
        ///     Samples are drawn from a Gamma distribution with shape parameter and scale=1.
        ///     For a different scale, multiply the result: scale * standard_gamma(shape).
        /// </remarks>
        public NDArray standard_gamma(double shape, int[] size)
        {
            // Parameter validation (matches NumPy error message)
            if (shape < 0)
                throw new ArgumentException("shape < 0", nameof(shape));

            if (size == null || size.Length == 0)
            {
                return NDArray.Scalar(shape == 0 ? 0.0 : SampleStandardGamma(shape));
            }

            var result = new NDArray<double>(size);
            ArraySlice<double> resultArray = result.Data<double>();

            if (shape == 0)
            {
                // Special case: shape=0 returns all zeros
                for (int i = 0; i < result.size; ++i)
                    resultArray[i] = 0.0;
            }
            else
            {
                for (int i = 0; i < result.size; ++i)
                    resultArray[i] = SampleStandardGamma(shape);
            }

            result.ReplaceData(resultArray);
            return result;
        }

        /// <summary>
        ///     Draw samples from a standard Gamma distribution (scale=1).
        /// </summary>
        /// <param name="shape">The shape of the gamma distribution. Must be >= 0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the standard gamma distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_gamma.html
        ///     <br/>
        ///     Samples are drawn from a Gamma distribution with shape parameter and scale=1.
        ///     For a different scale, multiply the result: scale * standard_gamma(shape).
        /// </remarks>
        public NDArray standard_gamma(double shape, params long[] size)
            => standard_gamma(shape, new Shape(size));

        // Note: SampleStandardGamma() and SampleMarsaglia() are already defined in np.random.standard_t.cs
    }
}
