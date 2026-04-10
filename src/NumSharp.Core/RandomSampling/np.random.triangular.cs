using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from the triangular distribution over the interval [left, right].
        /// </summary>
        /// <param name="left">Lower limit.</param>
        /// <param name="mode">The value where the peak of the distribution occurs (left &lt;= mode &lt;= right).</param>
        /// <param name="right">Upper limit, must be larger than left.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized triangular distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.triangular.html
        ///     <br/>
        ///     The triangular distribution is a continuous probability distribution with lower limit left,
        ///     peak at mode, and upper limit right.
        /// </remarks>
        public NDArray triangular(double left, double mode, double right, Shape size)
            => triangular(left, mode, right, size.dimensions);

        /// <summary>
        ///     Draw samples from the triangular distribution over the interval [left, right].
        /// </summary>
        /// <param name="left">Lower limit.</param>
        /// <param name="mode">The value where the peak of the distribution occurs (left &lt;= mode &lt;= right).</param>
        /// <param name="right">Upper limit, must be larger than left.</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the parameterized triangular distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.triangular.html
        ///     <br/>
        ///     The triangular distribution is a continuous probability distribution with lower limit left,
        ///     peak at mode, and upper limit right.
        /// </remarks>
        public NDArray triangular(double left, double mode, double right, int[] size)
            => triangular(left, mode, right, Shape.ComputeLongShape(size));

        /// <summary>
        ///     Draw samples from the triangular distribution over the interval [left, right].
        /// </summary>
        /// <param name="left">Lower limit.</param>
        /// <param name="mode">The value where the peak of the distribution occurs (left &lt;= mode &lt;= right).</param>
        /// <param name="right">Upper limit, must be larger than left.</param>
        /// <param name="size">Output shape as long array (for NumPy compatibility).</param>
        /// <returns>Drawn samples from the parameterized triangular distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.triangular.html
        ///     <br/>
        ///     The triangular distribution is a continuous probability distribution with lower limit left,
        ///     peak at mode, and upper limit right.
        /// </remarks>
        public NDArray triangular(double left, double mode, double right, params long[] size)
        {
            // Parameter validation (matches NumPy error messages exactly)
            if (left > mode)
                throw new ArgumentException("left > mode");
            if (mode > right)
                throw new ArgumentException("mode > right");
            if (left == right)
                throw new ArgumentException("left == right");

            if (size == null || size.Length == 0)
            {
                return NDArray.Scalar(SampleTriangular(left, mode, right));
            }

            var result = new NDArray<double>(size);
            ArraySlice<double> resultArray = result.Data<double>();

            for (int i = 0; i < result.size; ++i)
                resultArray[i] = SampleTriangular(left, mode, right);

            result.ReplaceData(resultArray);
            return result;
        }

        /// <summary>
        ///     Sample a single value from the triangular distribution using inverse transform sampling.
        /// </summary>
        /// <remarks>
        ///     Algorithm from NumPy's random_triangular in distributions.c
        /// </remarks>
        private double SampleTriangular(double left, double mode, double right)
        {
            // NumPy's exact implementation from distributions.c:
            // base = right - left
            // leftbase = mode - left
            // ratio = leftbase / base
            // leftprod = leftbase * base
            // rightprod = (right - mode) * base
            // U = random()
            // if U <= ratio: return left + sqrt(U * leftprod)
            // else: return right - sqrt((1 - U) * rightprod)

            double @base = right - left;
            double leftbase = mode - left;
            double ratio = leftbase / @base;
            double leftprod = leftbase * @base;
            double rightprod = (right - mode) * @base;

            double U = randomizer.NextDouble();
            if (U <= ratio)
            {
                return left + Math.Sqrt(U * leftprod);
            }
            else
            {
                return right - Math.Sqrt((1.0 - U) * rightprod);
            }
        }
    }
}
