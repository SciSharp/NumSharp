using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a Zipf distribution.
        /// </summary>
        /// <param name="a">Distribution parameter. Must be greater than 1.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized Zipf distribution as int64.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.zipf.html
        ///     <br/>
        ///     The Zipf distribution (also known as the zeta distribution) is a discrete
        ///     distribution commonly used to model the frequency of words in texts, the
        ///     size of cities, and many other phenomena.
        ///     <br/>
        ///     The probability mass function is:
        ///     p(k) = k^(-a) / zeta(a)
        ///     <br/>
        ///     where k >= 1 and zeta(a) is the Riemann zeta function.
        ///     <br/>
        ///     Samples are positive integers.
        /// </remarks>
        public NDArray zipf(double a, Shape size = default)
        {
            if (a <= 1.0 || double.IsNaN(a))
                throw new ArgumentException("a <= 1 or a is NaN", nameof(a));

            if (size.IsEmpty)
            {
                return NDArray.Scalar(SampleZipf(a));
            }

            var ret = new NDArray<long>(size);
            ArraySlice<long> data = ret.Data<long>();

            for (int i = 0; i < ret.size; i++)
            {
                data[i] = SampleZipf(a);
            }

            return ret;
        }

        /// <summary>
        ///     Draw samples from a Zipf distribution.
        /// </summary>
        /// <param name="a">Distribution parameter. Must be greater than 1.</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the parameterized Zipf distribution.</returns>
        public NDArray zipf(double a, params int[] size)
            => zipf(a, new Shape(size));

        /// <summary>
        ///     Draw samples from a Zipf distribution.
        /// </summary>
        /// <param name="a">Distribution parameter. Must be greater than 1.</param>
        /// <param name="size">Output shape as single int.</param>
        /// <returns>Drawn samples from the parameterized Zipf distribution.</returns>
        public NDArray zipf(double a, int size)
            => zipf(a, new Shape(size));

        /// <summary>
        ///     Sample from the Zipf distribution using the same rejection algorithm as NumPy.
        /// </summary>
        /// <remarks>
        ///     Based on NumPy's random_zipf in distributions.c.
        ///     Uses rejection sampling.
        /// </remarks>
        private long SampleZipf(double a)
        {
            // For very large a, probability of getting > 1 is essentially 0
            // NumPy uses a >= 1025 threshold
            if (a >= 1025.0)
            {
                return 1L;
            }

            double am1 = a - 1.0;
            double b = Math.Pow(2.0, am1);

            // Umin is the minimum U value that could produce a valid X
            // Using long.MaxValue as RAND_INT_MAX equivalent
            double Umin = Math.Pow((double)long.MaxValue, -am1);

            while (true)
            {
                // U is sampled from (Umin, 1]. Note that Umin might be 0.
                double U01 = randomizer.NextDouble();
                double U = U01 * Umin + (1.0 - U01);
                double V = randomizer.NextDouble();
                double X = Math.Floor(Math.Pow(U, -1.0 / am1));

                // Reject if X is too large or less than 1
                if (X > (double)long.MaxValue || X < 1.0)
                {
                    continue;
                }

                double T = Math.Pow(1.0 + 1.0 / X, am1);
                if (V * X * (T - 1.0) / (b - 1.0) <= T / b)
                {
                    return (long)X;
                }
            }
        }
    }
}
