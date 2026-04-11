using System;
using System.Runtime.CompilerServices;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a Gamma distribution.
        /// </summary>
        public NDArray gamma(double shape, double scale = 1.0) => gamma(shape, scale, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a Gamma distribution.
        /// </summary>
        /// <param name="shape">The shape of the gamma distribution. Must be non-negative.</param>
        /// <param name="scale">The scale of the gamma distribution. Must be non-negative. Default is 1.0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized gamma distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.gamma.html
        ///     <br/>
        ///     Samples are drawn from a Gamma distribution with specified parameters,
        ///     shape (sometimes designated "k") and scale (sometimes designated "theta"),
        ///     where both parameters are > 0.
        /// </remarks>
        public NDArray gamma(double shape, double scale, Shape size)
        {
            if (size.IsScalar || size.IsEmpty)
                return NDArray.Scalar(SampleStandardGamma(shape) * scale);

            if (shape < 1)
            {
                double d = shape + 1.0 - 1.0 / 3.0;
                double c = (1.0 / 3.0) / Math.Sqrt(d);

                NDArray u = uniform(0, 1, size);
                return scale * Marsaglia(d, c, size) * np.power(u, 1.0 / shape);
            }
            else
            {
                double d = shape - 1.0 / 3.0;
                double c = (1.0 / 3.0) / Math.Sqrt(d);

                return scale * Marsaglia(d, c, size);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private NDArray Marsaglia(double d, double c, Shape size)
        {
            var result = new NDArray<double>(size);
            unsafe
            {
                var dst = result.Address;
                var len = result.size;
                Func<double> nextDouble = randomizer.NextDouble;
                for (long i = 0; i < len; i++)
                {
                    while (true)
                    {
                        double x, t, v;

                        do
                        {
                            x = NextGaussian();
                            t = (1.0 + c * x);
                            v = t * t * t;
                        } while (v <= 0);

                        double U = nextDouble();
                        double x2 = x * x;

                        if (U < 1 - 0.0331 * x2 * x2)
                        {
                            dst[i] = d * v;
                            break;
                        }

                        if (Math.Log(U) < 0.5 * x2 + d * (1.0 - v + Math.Log(v)))
                        {
                            dst[i] = d * v;
                            break;
                        }
                    }
                }
            }

            return result;
        }
    }
}
