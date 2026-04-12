using System;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a standard Cauchy distribution.
        /// </summary>
        public NDArray standard_cauchy() => standard_cauchy(Shape.Scalar);

        /// <summary>
        ///     Draw samples from a standard Cauchy distribution with mode = 0.
        /// </summary>
        /// <param name="size">Output shape.</param>
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
        public NDArray standard_cauchy(Shape size)
        {
            if (size.IsScalar || size.IsEmpty)
                return NDArray.Scalar(StandardCauchySample());

            unsafe
            {
                var shape = size;
                var array = new NDArray<double>(shape);
                var dst = array.Address;
                var count = array.size;

                // Inverse transform: X = tan(pi * (U - 0.5))
                Func<double> nextDouble = randomizer.NextDouble;
                for (long i = 0; i < count; i++)
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
