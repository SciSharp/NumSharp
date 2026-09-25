using System;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from the triangular distribution.
        /// </summary>
        /// <param name="left">Lower limit.</param>
        /// <param name="mode">The value where the peak of the distribution occurs (left &lt;= mode &lt;= right).</param>
        /// <param name="right">Upper limit, must be larger than left.</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><c>left &gt; mode</c>, <c>mode &gt; right</c> or <c>left == right</c>.</exception>
        public NDArray triangular(double left, double mode, double right) => triangular(left, mode, right, Shape.Scalar);

        /// <summary>
        ///     Draw samples from the triangular distribution over the interval [left, right].
        /// </summary>
        /// <param name="left">Lower limit.</param>
        /// <param name="mode">The value where the peak of the distribution occurs (left &lt;= mode &lt;= right).</param>
        /// <param name="right">Upper limit, must be larger than left.</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized triangular distribution (float64).</returns>
        /// <exception cref="ValueError">In NumPy's order: <c>left &gt; mode</c>, <c>mode &gt; right</c>, <c>left == right</c>; or
        ///     <paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.triangular.html
        ///     <br/>
        ///     The triangular distribution is a continuous probability distribution with lower limit left,
        ///     peak at mode, and upper limit right.
        ///     <br/>
        ///     NumPy's <c>random_triangular</c> (shared by RandomState and Generator): one uniform per value, inverted through
        ///     the left or right leg of the CDF. The leg constants depend only on the parameters, so they are computed once
        ///     (bit-identical to NumPy recomputing them per value) and the uniforms come from the bit generator's bulk fill.
        ///     NaN parameters pass the checks (every comparison is false) and sample NaN, as in NumPy. Holds the bit
        ///     generator's lock for the draws.
        /// </remarks>
        public NDArray triangular(double left, double mode, double right, Shape size)
        {
            // Parameter validation (matches NumPy error messages and order exactly)
            if (left > mode)
                throw new ValueError("left > mode");
            if (mode > right)
                throw new ValueError("mode > right");
            if (left == right)
                throw new ValueError("left == right");

            if (IsScalarDraw(size))
                lock (randomizer.@lock)
                    return NDArray.Scalar(Distributions.RandomTriangular(randomizer, left, mode, right));

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                lock (randomizer.@lock)
                    randomizer.FillDouble(dst, n);

                // random_triangular's per-call setup, hoisted: pure functions of the parameters.
                double @base = right - left;
                double leftbase = mode - left;
                double ratio = leftbase / @base;
                double leftprod = leftbase * @base;
                double rightprod = (right - mode) * @base;
                for (long i = 0; i < n; i++)
                {
                    double U = dst[i];
                    dst[i] = U <= ratio
                        ? left + Math.Sqrt(U * leftprod)
                        : right - Math.Sqrt((1.0 - U) * rightprod);
                }
            }

            return ret;
        }
    }
}
