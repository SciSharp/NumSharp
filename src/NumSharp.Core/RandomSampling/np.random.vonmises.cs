using System;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a von Mises distribution.
        /// </summary>
        /// <param name="mu">Mode ("center") of the distribution in radians.</param>
        /// <param name="kappa">
        ///     Concentration parameter of the distribution. Must be >= 0.
        ///     When kappa = 0, the distribution is uniform on the circle.
        ///     As kappa increases, the distribution becomes more concentrated around mu.
        /// </param>
        /// <param name="size">Output shape. If null, a single value is returned.</param>
        /// <returns>Drawn samples from the parameterized von Mises distribution.</returns>
        /// <exception cref="ArgumentException">If kappa is negative.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.vonmises.html
        ///     <br/>
        ///     The von Mises distribution (also known as the circular normal distribution)
        ///     is a continuous probability distribution on the unit circle. It may be thought
        ///     of as the circular analogue of the normal distribution.
        ///     <br/>
        ///     Samples are drawn on the interval [-pi, pi].
        ///     <br/>
        ///     The probability density function is:
        ///     p(x) = exp(kappa * cos(x - mu)) / (2 * pi * I_0(kappa))
        ///     where I_0(kappa) is the modified Bessel function of order 0.
        /// </remarks>
        public NDArray vonmises(double mu, double kappa, Shape? size = null)
        {
            if (kappa < 0)
                throw new ArgumentException("kappa < 0", nameof(kappa));

            if (size == null)
            {
                // Return scalar
                return NDArray.Scalar(SampleVonMises(mu, kappa));
            }

            return vonmises(mu, kappa, size.Value.dimensions);
        }

        /// <summary>
        ///     Draw samples from a von Mises distribution.
        /// </summary>
        /// <param name="mu">Mode ("center") of the distribution in radians.</param>
        /// <param name="kappa">Concentration parameter of the distribution. Must be >= 0.</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the parameterized von Mises distribution.</returns>
        public NDArray vonmises(double mu, double kappa, int[] size)
            => vonmises(mu, kappa, new Shape(size));

        /// <summary>
        ///     Draw samples from a von Mises distribution.
        /// </summary>
        /// <param name="mu">Mode ("center") of the distribution in radians.</param>
        /// <param name="kappa">Concentration parameter of the distribution. Must be >= 0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized von Mises distribution.</returns>
        public NDArray vonmises(double mu, double kappa, params long[] size)
            => vonmises(mu, kappa, new Shape(size));

        /// <summary>
        ///     Draw samples from a von Mises distribution.
        /// </summary>
        /// <param name="mu">Mode ("center") of the distribution in radians.</param>
        /// <param name="kappa">Concentration parameter of the distribution. Must be >= 0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized von Mises distribution.</returns>
        public NDArray vonmises(double mu, double kappa, Shape size)
        {
            if (kappa < 0)
                throw new ArgumentException("kappa < 0", nameof(kappa));

            if (size.IsEmpty)
            {
                // Return scalar
                return NDArray.Scalar(SampleVonMises(mu, kappa));
            }

            unsafe
            {
                var ret = new NDArray<double>(size);
                var dst = ret.Address;

                for (int i = 0; i < ret.size; i++)
                {
                    dst[i] = SampleVonMises(mu, kappa);
                }

                return ret;
            }
        }

        /// <summary>
        ///     Draw samples from a von Mises distribution.
        /// </summary>
        /// <param name="mu">Mode ("center") of the distribution in radians.</param>
        /// <param name="kappa">Concentration parameter of the distribution. Must be >= 0.</param>
        /// <param name="size">Output shape as single int.</param>
        /// <returns>Drawn samples from the parameterized von Mises distribution.</returns>
        public NDArray vonmises(double mu, double kappa, int size)
            => vonmises(mu, kappa, new int[] { size });

        /// <summary>
        ///     Sample from the von Mises distribution using the same algorithm as NumPy.
        /// </summary>
        /// <remarks>
        ///     Based on NumPy's random_vonmises in distributions.c.
        ///     Uses rejection sampling for 1e-5 <= kappa <= 1e6,
        ///     uniform distribution for kappa < 1e-8,
        ///     and wrapped normal approximation for kappa > 1e6.
        /// </remarks>
        private double SampleVonMises(double mu, double kappa)
        {
            double s;
            double U, V, W, Y, Z;
            double result;

            // Handle NaN kappa
            if (double.IsNaN(kappa))
                return double.NaN;

            // For very small kappa, use uniform distribution on [-pi, pi]
            if (kappa < 1e-8)
            {
                return Math.PI * (2 * randomizer.NextDouble() - 1);
            }

            // Calculate s parameter based on kappa range
            if (kappa < 1e-5)
            {
                // Second order Taylor expansion around kappa = 0
                s = (1.0 / kappa + kappa);
            }
            else if (kappa <= 1e6)
            {
                // Standard path for 1e-5 <= kappa <= 1e6
                double r = 1 + Math.Sqrt(1 + 4 * kappa * kappa);
                double rho = (r - Math.Sqrt(2 * r)) / (2 * kappa);
                s = (1 + rho * rho) / (2 * rho);
            }
            else
            {
                // Fallback to wrapped normal distribution for kappa > 1e6
                result = mu + Math.Sqrt(1.0 / kappa) * NextGaussian();
                // Ensure result is within bounds [-pi, pi]
                if (result < -Math.PI)
                    result += 2 * Math.PI;
                if (result > Math.PI)
                    result -= 2 * Math.PI;
                return result;
            }

            // Rejection sampling loop
            while (true)
            {
                U = randomizer.NextDouble();
                Z = Math.Cos(Math.PI * U);
                W = (1 + s * Z) / (s + Z);
                Y = kappa * (s - W);
                V = randomizer.NextDouble();

                // Accept/reject test
                // V == 0.0 is ok here since Y >= 0 always leads to accept,
                // while Y < 0 always rejects
                if ((Y * (2 - Y) - V >= 0) || (Math.Log(Y / V) + 1 - Y >= 0))
                {
                    break;
                }
            }

            U = randomizer.NextDouble();
            result = Math.Acos(W);
            if (U < 0.5)
            {
                result = -result;
            }
            result += mu;

            // Wrap result to [-pi, pi]
            bool neg = result < 0;
            double mod = Math.Abs(result);
            mod = ((mod + Math.PI) % (2 * Math.PI)) - Math.PI;
            if (neg)
            {
                mod = -mod;
            }

            return mod;
        }
    }
}
