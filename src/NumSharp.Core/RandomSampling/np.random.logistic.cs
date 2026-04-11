using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a logistic distribution.
        /// </summary>
        /// <param name="loc">Mean of the distribution. Default is 0.</param>
        /// <param name="scale">Scale parameter (must be >= 0). Default is 1.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized logistic distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.logistic.html
        ///     <br/>
        ///     The logistic distribution is used in extreme value problems, finance,
        ///     and for growth modeling. It is similar to the normal distribution but
        ///     has heavier tails.
        ///     <br/>
        ///     The probability density function is:
        ///     f(x; μ, s) = exp(-(x-μ)/s) / (s * (1 + exp(-(x-μ)/s))^2)
        ///     <br/>
        ///     Mean = loc, Variance = scale^2 * pi^2 / 3
        /// </remarks>
        public NDArray logistic(double loc = 0.0, double scale = 1.0, Shape size = default)
        {
            if (scale < 0)
                throw new ArgumentException("scale < 0", nameof(scale));

            if (size.IsEmpty)
            {
                return NDArray.Scalar(SampleLogistic(loc, scale));
            }

            var ret = new NDArray<double>(size);
            ArraySlice<double> data = ret.Data<double>();

            for (int i = 0; i < ret.size; i++)
            {
                data[i] = SampleLogistic(loc, scale);
            }

            return ret;
        }

        /// <summary>
        ///     Draw samples from a logistic distribution.
        /// </summary>
        /// <param name="loc">Mean of the distribution.</param>
        /// <param name="scale">Scale parameter (must be >= 0).</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the parameterized logistic distribution.</returns>
        public NDArray logistic(double loc, double scale, int[] size)
            => logistic(loc, scale, new Shape(size));

        /// <summary>
        ///     Draw samples from a logistic distribution.
        /// </summary>
        /// <param name="loc">Mean of the distribution.</param>
        /// <param name="scale">Scale parameter (must be >= 0).</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized logistic distribution.</returns>
        public NDArray logistic(double loc, double scale, params long[] size)
            => logistic(loc, scale, new Shape(size));

        /// <summary>
        ///     Draw samples from a logistic distribution.
        /// </summary>
        /// <param name="loc">Mean of the distribution.</param>
        /// <param name="scale">Scale parameter (must be >= 0).</param>
        /// <param name="size">Output shape as single int.</param>
        /// <returns>Drawn samples from the parameterized logistic distribution.</returns>
        public NDArray logistic(double loc, double scale, int size)
            => logistic(loc, scale, new int[] { size });

        /// <summary>
        ///     Sample from the logistic distribution using inverse transform method.
        /// </summary>
        /// <remarks>
        ///     Uses the formula: X = loc + scale * ln(U / (1 - U))
        ///     where U ~ Uniform(0, 1).
        ///
        ///     Special case: if scale == 0, returns loc directly.
        /// </remarks>
        private double SampleLogistic(double loc, double scale)
        {
            if (scale == 0)
                return loc;

            double U;
            do
            {
                U = randomizer.NextDouble();
            } while (U == 0.0 || U == 1.0);

            // X = loc + scale * ln(U / (1 - U))
            return loc + scale * Math.Log(U / (1.0 - U));
        }
    }
}
