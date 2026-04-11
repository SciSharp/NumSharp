using System;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a Weibull distribution.
        /// </summary>
        /// <param name="a">Shape parameter of the distribution. Must be non-negative.</param>
        /// <param name="size">Output shape. If null, a single value is returned.</param>
        /// <returns>Drawn samples from the Weibull distribution.</returns>
        /// <exception cref="ArgumentException">If a is negative.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.weibull.html
        ///     <br/>
        ///     The Weibull distribution is a continuous probability distribution with
        ///     probability density function: f(x;a) = a * x^(a-1) * exp(-x^a) for x >= 0.
        ///     <br/>
        ///     This is the standard Weibull with scale=1. For Weibull with scale parameter,
        ///     use: scale * np.random.weibull(a, size).
        ///     <br/>
        ///     When a=1, the Weibull distribution reduces to the exponential distribution.
        /// </remarks>
        public NDArray weibull(double a, Shape? size = null)
        {
            if (a < 0)
                throw new ArgumentException("a < 0", nameof(a));

            if (size == null)
            {
                // Return scalar
                if (a == 0)
                    return NDArray.Scalar(0.0);
                return NDArray.Scalar(WeibullSample(a));
            }

            return weibull(a, size.Value.dimensions);
        }

        /// <summary>
        ///     Draw samples from a Weibull distribution.
        /// </summary>
        /// <param name="a">Shape parameter of the distribution. Must be non-negative.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the Weibull distribution.</returns>
        /// <exception cref="ArgumentException">If a is negative.</exception>
        public NDArray weibull(double a, Shape size) => weibull(a, size.dimensions);

        /// <summary>
        ///     Draw samples from a Weibull distribution.
        /// </summary>
        /// <param name="a">Shape parameter of the distribution. Must be non-negative.</param>
        /// <param name="size">Output shape as a single dimension.</param>
        /// <returns>Drawn samples from the Weibull distribution.</returns>
        /// <exception cref="ArgumentException">If a is negative.</exception>
        public NDArray weibull(double a, int size) => weibull(a, new[] { size });

        /// <summary>
        ///     Draw samples from a Weibull distribution.
        /// </summary>
        /// <param name="a">Shape parameter of the distribution. Must be non-negative.</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the Weibull distribution.</returns>
        /// <exception cref="ArgumentException">If a is negative.</exception>
        public NDArray weibull(double a, int[] size)
            => weibull(a, Shape.ComputeLongShape(size));

        /// <summary>
        ///     Draw samples from a Weibull distribution.
        /// </summary>
        /// <param name="a">Shape parameter of the distribution. Must be non-negative.</param>
        /// <param name="size">Output shape as long array (for NumPy compatibility).</param>
        /// <returns>Drawn samples from the Weibull distribution.</returns>
        /// <exception cref="ArgumentException">If a is negative.</exception>
        public NDArray weibull(double a, long[] size)
        {
            if (a < 0)
                throw new ArgumentException("a < 0", nameof(a));

            unsafe
            {
                var array = new NDArray<double>(new Shape(size));
                var dst = array.Address;
                var count = array.size;

                if (a == 0)
                {
                    // When a=0, all values are 0 (NumPy behavior)
                    for (long i = 0; i < count; i++)
                        dst[i] = 0.0;
                }
                else
                {
                    // Inverse transform: X = (-ln(1-U))^(1/a) = (-ln(U))^(1/a)
                    // Using 1-U or U gives same distribution since U is uniform
                    double invA = 1.0 / a;
                    Func<double> nextDouble = randomizer.NextDouble;
                    for (long i = 0; i < count; i++)
                    {
                        double u = nextDouble();
                        // Use 1-u to avoid log(0) when u=0
                        dst[i] = Math.Pow(-Math.Log(1.0 - u), invA);
                    }
                }

                return array;
            }
        }

        /// <summary>
        ///     Generate a single sample from the Weibull distribution.
        /// </summary>
        private double WeibullSample(double a)
        {
            double u = randomizer.NextDouble();
            return Math.Pow(-Math.Log(1.0 - u), 1.0 / a);
        }
    }
}
