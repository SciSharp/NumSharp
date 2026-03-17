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
        /// <param name="d0">Dimension(s) of the returned array.</param>
        /// <returns>
        ///     A (d0, d1, ..., dn)-shaped array of floating-point samples from the standard
        ///     normal distribution, or a single such float if no parameters are supplied.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html
        ///     <br/>
        ///     For random samples from the normal distribution with mean mu and standard
        ///     deviation sigma, use: sigma * np.random.randn(...) + mu
        ///     <br/>
        ///     This is a convenience function for users porting code from Matlab.
        ///     For new code, use np.random.standard_normal instead.
        /// </remarks>
        public NDArray randn(params int[] d0)
        {
            return standard_normal(d0);
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
        public NDArray normal(double loc, double scale, params int[] size)
        {
            unsafe
            {
                var array = new NDArray<double>(new Shape(size));
                var dst = array.Address;

                Func<double> nextDouble = randomizer.NextDouble;
                for (int i = 0; i < array.size; i++)
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
        public NDArray standard_normal(params int[] size)
        {
            return normal(0, 1.0, size);
        }

        /// <summary>
        ///     Backwards compatibility alias for standard_normal (typo in older versions).
        /// </summary>
        [Obsolete("Use standard_normal instead (typo fixed)")]
        public NDArray stardard_normal(params int[] size) => standard_normal(size);
    }
}
