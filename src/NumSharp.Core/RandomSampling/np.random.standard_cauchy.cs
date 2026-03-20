using System;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a standard Cauchy distribution with mode = 0.
        /// </summary>
        /// <param name="size">Output shape. If null, a single value is returned.</param>
        /// <returns>Drawn samples from the standard Cauchy distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_cauchy.html
        ///     <br/>
        ///     Also known as the Lorentz distribution. The standard Cauchy distribution
        ///     has location parameter x0=0 and scale parameter gamma=1.
        ///     <br/>
        ///     The Cauchy distribution has no defined mean or variance (infinite tails).
        ///     The median is 0, and the interquartile range is 2 (from -1 to 1).
        ///     <br/>
        ///     Generated using inverse transform: X = tan(pi * (U - 0.5)) where U ~ Uniform(0, 1).
        /// </remarks>
        public NDArray standard_cauchy(Shape? size = null)
        {
            if (size == null)
            {
                // Return scalar
                return NDArray.Scalar(StandardCauchySample());
            }

            return standard_cauchy(size.Value.dimensions);
        }

        /// <summary>
        ///     Draw samples from a standard Cauchy distribution with mode = 0.
        /// </summary>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the standard Cauchy distribution.</returns>
        public NDArray standard_cauchy(Shape size) => standard_cauchy(size.dimensions);

        /// <summary>
        ///     Draw samples from a standard Cauchy distribution with mode = 0.
        /// </summary>
        /// <param name="size">Output shape as a single dimension.</param>
        /// <returns>Drawn samples from the standard Cauchy distribution.</returns>
        public NDArray standard_cauchy(int size) => standard_cauchy(new[] { size });

        /// <summary>
        ///     Draw samples from a standard Cauchy distribution with mode = 0.
        /// </summary>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the standard Cauchy distribution.</returns>
        public NDArray standard_cauchy(params int[] size)
        {
            unsafe
            {
                var array = new NDArray<double>(new Shape(size));
                var dst = array.Address;
                var count = array.size;

                // Inverse transform: X = tan(pi * (U - 0.5))
                Func<double> nextDouble = randomizer.NextDouble;
                for (int i = 0; i < count; i++)
                {
                    double u = nextDouble();
                    dst[i] = Math.Tan(Math.PI * (u - 0.5));
                }

                return array;
            }
        }

        /// <summary>
        ///     Generate a single sample from the standard Cauchy distribution.
        /// </summary>
        /// <returns>A random value from the standard Cauchy distribution.</returns>
        private double StandardCauchySample()
        {
            double u = randomizer.NextDouble();
            return Math.Tan(Math.PI * (u - 0.5));
        }
    }
}
