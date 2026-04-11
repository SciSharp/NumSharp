using System;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a logarithmic series distribution.
        /// </summary>
        /// <param name="p">Shape parameter for the distribution. Must be in the range [0, 1).</param>
        /// <param name="size">Output shape. If null, a single value is returned.</param>
        /// <returns>Drawn samples from the parameterized logarithmic series distribution.</returns>
        /// <exception cref="ArgumentException">If p is not in range [0, 1) or is NaN.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.logseries.html
        ///     <br/>
        ///     The probability density for the Log Series distribution is:
        ///     P(k) = -p^k / (k * ln(1-p))
        ///     <br/>
        ///     The log series distribution is frequently used to represent species
        ///     richness and occurrence, first proposed by Fisher, Corbet, and Williams in 1943.
        ///     <br/>
        ///     Returns positive integers (k >= 1).
        /// </remarks>
        public NDArray logseries(double p, Shape? size = null)
        {
            ValidateLogseriesP(p);

            if (size == null)
            {
                // Return scalar
                return NDArray.Scalar(SampleLogseries(p));
            }

            return logseries(p, Shape.ToIntArray(size.Value.dimensions));
        }

        /// <summary>
        ///     Draw samples from a logarithmic series distribution.
        /// </summary>
        /// <param name="p">Shape parameter for the distribution. Must be in the range [0, 1).</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the parameterized logarithmic series distribution.</returns>
        public NDArray logseries(double p, int[] size)
            => logseries(p, new Shape(size));

        /// <summary>
        ///     Draw samples from a logarithmic series distribution.
        /// </summary>
        /// <param name="p">Shape parameter for the distribution. Must be in the range [0, 1).</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized logarithmic series distribution.</returns>
        public NDArray logseries(double p, params long[] size)
            => logseries(p, new Shape(size));

        /// <summary>
        ///     Draw samples from a logarithmic series distribution.
        /// </summary>
        /// <param name="p">Shape parameter for the distribution. Must be in the range [0, 1).</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized logarithmic series distribution.</returns>
        public NDArray logseries(double p, Shape size)
        {
            ValidateLogseriesP(p);

            if (size.IsEmpty)
            {
                // Return scalar
                return NDArray.Scalar(SampleLogseries(p));
            }

            unsafe
            {
                var ret = new NDArray<long>(size);
                var dst = ret.Address;

                for (int i = 0; i < ret.size; i++)
                {
                    dst[i] = SampleLogseries(p);
                }

                return ret;
            }
        }

        /// <summary>
        ///     Draw samples from a logarithmic series distribution.
        /// </summary>
        /// <param name="p">Shape parameter for the distribution. Must be in the range [0, 1).</param>
        /// <param name="size">Output shape as single int.</param>
        /// <returns>Drawn samples from the parameterized logarithmic series distribution.</returns>
        public NDArray logseries(double p, int size)
            => logseries(p, new int[] { size });

        private static void ValidateLogseriesP(double p)
        {
            if (p < 0 || p >= 1 || double.IsNaN(p))
                throw new ArgumentException("p < 0, p >= 1 or p is NaN", nameof(p));
        }

        /// <summary>
        ///     Sample from the logarithmic series distribution using the same algorithm as NumPy.
        /// </summary>
        /// <remarks>
        ///     Based on NumPy's random_logseries in distributions.c.
        ///     Uses the algorithm from Kemp (1981).
        /// </remarks>
        private long SampleLogseries(double p)
        {
            double q, r, U, V;
            long result;

            r = Math.Log(1 - p); // log1p(-p)

            while (true)
            {
                V = randomizer.NextDouble();
                if (V >= p)
                {
                    return 1;
                }

                U = randomizer.NextDouble();
                q = 1 - Math.Exp(r * U); // -expm1(r * U) = -(exp(r*U) - 1) = 1 - exp(r*U)

                if (V <= q * q)
                {
                    result = (long)Math.Floor(1 + Math.Log(V) / Math.Log(q));
                    if (result < 1 || V == 0.0)
                    {
                        continue;
                    }
                    else
                    {
                        return result;
                    }
                }

                if (V >= q)
                {
                    return 1;
                }

                return 2;
            }
        }
    }
}
