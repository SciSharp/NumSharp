using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a Pareto II or Lomax distribution with specified shape.
        /// </summary>
        /// <param name="a">Shape of the distribution. Must be positive (&gt; 0).</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized Pareto distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.pareto.html
        ///     <br/>
        ///     NumPy's pareto returns samples from the Pareto II (Lomax) distribution,
        ///     not the classical Pareto distribution. The relationship is:
        ///     if Y ~ Pareto(a, m=1) then X = Y - 1 ~ Lomax(a).
        ///     <br/>
        ///     The probability density function is:
        ///     f(x; a) = a / (1 + x)^(a+1)  for x >= 0
        ///     <br/>
        ///     The mean is 1/(a-1) for a > 1, undefined otherwise.
        /// </remarks>
        public NDArray pareto(double a, Shape size)
        {
            if (a <= 0)
                throw new ArgumentException("a <= 0", nameof(a));

            if (size.IsScalar || size.IsEmpty)
                return NDArray.Scalar(SamplePareto(a));

            var ret = new NDArray<double>(size);
            ArraySlice<double> data = ret.Data<double>();

            for (int i = 0; i < ret.size; i++)
            {
                data[i] = SamplePareto(a);
            }

            return ret;
        }

        /// <summary>
        ///     Draw samples from a Pareto II or Lomax distribution with specified shape.
        /// </summary>
        /// <param name="a">Shape of the distribution. Must be positive (&gt; 0).</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the parameterized Pareto distribution.</returns>
        public NDArray pareto(double a, int[] size)
            => pareto(a, new Shape(size));

        /// <summary>
        ///     Draw samples from a Pareto II or Lomax distribution with specified shape.
        /// </summary>
        /// <param name="a">Shape of the distribution. Must be positive (&gt; 0).</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized Pareto distribution.</returns>
        public NDArray pareto(double a, long[] size)
            => pareto(a, new Shape(size));

        /// <summary>
        ///     Draw samples from a Pareto II or Lomax distribution with specified shape.
        /// </summary>
        /// <param name="a">Shape of the distribution. Must be positive (&gt; 0).</param>
        /// <param name="size">Output shape as single int.</param>
        /// <returns>Drawn samples from the parameterized Pareto distribution.</returns>
        public NDArray pareto(double a, int size)
            => pareto(a, new int[] { size });

        /// <summary>
        ///     Draw a single sample from a Pareto II or Lomax distribution.
        /// </summary>
        /// <param name="a">Shape of the distribution. Must be positive (&gt; 0).</param>
        /// <returns>A single sample from the Pareto distribution as 0-d array.</returns>
        public NDArray pareto(double a) => pareto(a, Shape.Scalar);

        /// <summary>
        ///     Sample from the Pareto II (Lomax) distribution using inverse transform.
        /// </summary>
        /// <remarks>
        ///     Uses the formula: X = (1 / U^(1/a)) - 1
        ///     where U ~ Uniform(0, 1).
        ///     Equivalently: X = exp(E/a) - 1 where E ~ Exponential(1).
        ///
        ///     NumPy uses: X = exp(standard_exponential() / a) - 1
        ///     which is equivalent to: X = exp(-log(U) / a) - 1 = (1 / U^(1/a)) - 1
        /// </remarks>
        private double SamplePareto(double a)
        {
            double U;
            do
            {
                U = randomizer.NextDouble();
            } while (U == 0.0);

            // X = (1 / U^(1/a)) - 1 = U^(-1/a) - 1
            return Math.Pow(U, -1.0 / a) - 1.0;
        }
    }
}
