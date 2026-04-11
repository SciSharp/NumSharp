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
        public NDArray randn(params long[] shape)
        {
            if (shape.Length == 0)
                return standard_normal();
            return standard_normal(new Shape(shape));
        }

        /// <summary>
        ///     Return a scalar sample from the standard normal distribution.
        /// </summary>
        /// <typeparam name="T">The desired output type.</typeparam>
        /// <returns>A single random value.</returns>
        public T randn<T>()
        {
            return (T)Converts.ChangeType(NextGaussian(), InfoOf<T>.NPTypeCode);
        }

        /// <summary>
        ///     Draw a single sample from a normal (Gaussian) distribution.
        /// </summary>
        public NDArray normal(double loc = 0.0, double scale = 1.0) => normal(loc, scale, Shape.Scalar);

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
        public NDArray normal(double loc, double scale, Shape size)
        {
            if (size.IsScalar || size.IsEmpty)
                return NDArray.Scalar(loc + scale * NextGaussian());

            unsafe
            {
                var array = new NDArray<double>(size);
                var dst = array.Address;

                for (long i = 0; i < array.size; i++)
                    dst[i] = loc + scale * NextGaussian();

                return array;
            }
        }

        /// <summary>
        ///     Draw a single sample from a standard Normal distribution.
        /// </summary>
        public NDArray standard_normal() => standard_normal(Shape.Scalar);

        /// <summary>
        ///     Draw samples from a standard Normal distribution (mean=0, stdev=1).
        /// </summary>
        /// <param name="size">Output shape.</param>
        /// <returns>A floating-point array of shape size of drawn samples.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_normal.html
        /// </remarks>
        public NDArray standard_normal(Shape size)
        {
            return normal(0, 1.0, size);
        }
    }
}
