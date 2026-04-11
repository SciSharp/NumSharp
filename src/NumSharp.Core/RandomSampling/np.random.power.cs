using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draws samples in [0, 1] from a power distribution with positive exponent a - 1.
        /// </summary>
        /// <param name="a">Shape parameter of the distribution. Must be positive (&gt; 0).</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized power distribution, in range [0, 1].</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.power.html
        ///     <br/>
        ///     Also known as the power function distribution. The probability density function is:
        ///     P(x; a) = a * x^(a-1), for 0 &lt;= x &lt;= 1, a &gt; 0.
        ///     <br/>
        ///     The power function distribution is the inverse of the Pareto distribution.
        ///     It may also be seen as a special case of the Beta distribution.
        /// </remarks>
        public NDArray power(double a, Shape size)
        {
            if (a <= 0)
                throw new ArgumentException("a <= 0", nameof(a));

            var ret = new NDArray<double>(size);
            ArraySlice<double> data = ret.Data<double>();

            for (int i = 0; i < ret.size; i++)
            {
                data[i] = SamplePower(a);
            }

            return ret;
        }

        /// <summary>
        ///     Draws samples in [0, 1] from a power distribution with positive exponent a - 1.
        /// </summary>
        /// <param name="a">Shape parameter of the distribution. Must be positive (&gt; 0).</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the parameterized power distribution, in range [0, 1].</returns>
        public NDArray power(double a, int[] size)
            => power(a, new Shape(size));

        /// <summary>
        ///     Draws samples in [0, 1] from a power distribution with positive exponent a - 1.
        /// </summary>
        /// <param name="a">Shape parameter of the distribution. Must be positive (&gt; 0).</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized power distribution, in range [0, 1].</returns>
        public NDArray power(double a, long[] size)
            => power(a, new Shape(size));

        /// <summary>
        ///     Draws samples in [0, 1] from a power distribution with positive exponent a - 1.
        /// </summary>
        /// <param name="a">Shape parameter of the distribution. Must be positive (&gt; 0).</param>
        /// <param name="size">Output shape as single int.</param>
        /// <returns>Drawn samples from the parameterized power distribution, in range [0, 1].</returns>
        public NDArray power(double a, int size)
            => power(a, new int[] { size });

        /// <summary>
        ///     Draw a single sample from a power distribution.
        /// </summary>
        /// <param name="a">Shape parameter of the distribution. Must be positive (&gt; 0).</param>
        /// <returns>A single sample from the power distribution, in range [0, 1].</returns>
        public double power(double a)
        {
            if (a <= 0)
                throw new ArgumentException("a <= 0", nameof(a));

            return SamplePower(a);
        }

        /// <summary>
        ///     Sample from the power distribution using inverse transform method.
        /// </summary>
        /// <remarks>
        ///     NumPy uses: pow(-expm1(-standard_exponential()), 1/a)
        ///     which simplifies to: pow(1 - exp(-E), 1/a) where E ~ Exponential(1)
        ///     Since E = -ln(U), this becomes: pow(1 - U, 1/a) = pow(U, 1/a)
        ///     (because 1-U is also uniform when U is uniform)
        ///
        ///     We use the simpler equivalent: U^(1/a)
        /// </remarks>
        private double SamplePower(double a)
        {
            double U = randomizer.NextDouble();
            // U^(1/a) - inverse CDF of power distribution
            return Math.Pow(U, 1.0 / a);
        }
    }
}
