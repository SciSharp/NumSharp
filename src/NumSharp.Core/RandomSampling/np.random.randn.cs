using System;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Return a sample (or samples) from the "standard normal" distribution.
        /// </summary>
        /// <param name="shape">Output shape.</param>
        /// <returns>
        ///     Array of floating-point samples from the standard normal distribution.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html
        ///     <br/>
        ///     NumPy signature: randn(d0, d1, ..., dn) where d0..dn are dimension sizes.
        ///     <br/>
        ///     For random samples from N(μ, σ²), use: σ * np.random.randn(...) + μ
        /// </remarks>
        public NDArray randn(Shape shape) => randn(shape.dimensions);

        /// <summary>
        ///     Return a sample (or samples) from the "standard normal" distribution.
        /// </summary>
        /// <param name="shape">Dimensions of the returned array (d0, d1, ..., dn).</param>
        /// <returns>
        ///     Array of floating-point samples from the standard normal distribution.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html
        ///     <br/>
        ///     NumPy signature: randn(d0, d1, ..., dn) where d0..dn are dimension sizes.
        ///     <br/>
        ///     For random samples from N(μ, σ²), use: σ * np.random.randn(...) + μ
        /// </remarks>
        public NDArray randn(params int[] shape)
        {
            return standard_normal(shape);
        }

        /// <summary>
        ///     Return a scalar sample from the standard normal distribution.
        /// </summary>
        /// <typeparam name="T">The desired output type.</typeparam>
        /// <returns>A single random value.</returns>
        public T randn<T>()
        {
            return (T)Converts.ChangeType(randomizer.NextDouble(), InfoOf<T>.NPTypeCode);
        }

        /// <summary>
        ///     Draw random samples from a normal (Gaussian) distribution.
        /// </summary>
        /// <param name="loc">Mean ("centre") of the distribution. Default is 0.</param>
        /// <param name="scale">Standard deviation (spread or "width") of the distribution. Must be non-negative. Default is 1.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized normal distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
        ///     <br/>
        ///     The probability density function of the normal distribution, first derived
        ///     by De Moivre and 200 years later by both Gauss and Laplace independently,
        ///     is often called the bell curve because of its characteristic shape.
        /// </remarks>
        public NDArray normal(double loc, double scale, Shape size) => normal(loc, scale, size.dimensions);

        /// <summary>
        ///     Draw random samples from a normal (Gaussian) distribution.
        /// </summary>
        /// <param name="loc">Mean ("centre") of the distribution. Default is 0.</param>
        /// <param name="scale">Standard deviation (spread or "width") of the distribution. Must be non-negative. Default is 1.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized normal distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
        ///     <br/>
        ///     The probability density function of the normal distribution, first derived
        ///     by De Moivre and 200 years later by both Gauss and Laplace independently,
        ///     is often called the bell curve because of its characteristic shape.
        /// </remarks>
        public NDArray normal(double loc, double scale, params int[] size)
        {
            unsafe
            {
                var array = new NDArray<double>(new Shape(size));
                var dst = array.Address;

                Func<double> nextDouble = randomizer.NextDouble;
                for (long i = 0; i < array.size; i++)
                    dst[i] = loc + scale * Math.Sqrt(-2.0 * Math.Log(1.0 - nextDouble()))
                                        * Math.Sin(2.0 * Math.PI * (1.0 - nextDouble()));

                return array;
            }
        }

        /// <summary>
        ///     Draw samples from a standard Normal distribution (mean=0, stdev=1).
        /// </summary>
        /// <param name="size">Output shape.</param>
        /// <returns>A floating-point array of shape size of drawn samples.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_normal.html
        /// </remarks>
        public NDArray standard_normal(Shape size) => standard_normal(size.dimensions);

        /// <summary>
        ///     Draw samples from a standard Normal distribution (mean=0, stdev=1).
        /// </summary>
        /// <param name="size">Output shape.</param>
        /// <returns>A floating-point array of shape size of drawn samples.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_normal.html
        /// </remarks>
        public NDArray standard_normal(params int[] size)
        {
            return normal(0, 1.0, size);
        }
    }
}
