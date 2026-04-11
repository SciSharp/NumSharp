using System;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a chi-square distribution.
        /// </summary>
        public NDArray chisquare(double df) => chisquare(df, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a chi-square distribution.
        /// </summary>
        /// <param name="df">Number of degrees of freedom, must be > 0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized chi-square distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.chisquare.html
        ///     <br/>
        ///     When df independent random variables, each with standard normal distributions
        ///     (mean 0, variance 1), are squared and summed, the resulting distribution is
        ///     chi-square. This distribution is often used in hypothesis testing.
        /// </remarks>
        public NDArray chisquare(double df, Shape size)
        {
            if (df <= 0)
                throw new ArgumentException("df must be > 0", nameof(df));

            if (size.IsScalar || size.IsEmpty)
                return NDArray.Scalar(2.0 * SampleStandardGamma(df / 2.0));

            // NumPy: chisquare(df) = 2.0 * standard_gamma(df/2)
            // Must use per-element SampleStandardGamma to match RNG consumption order
            var shape = size;
            NDArray ret = new NDArray(NPTypeCode.Double, shape, false);

            // Handle empty arrays (any dimension is 0)
            if (shape.size == 0)
                return ret;

            double halfDf = df / 2.0;

            unsafe
            {
                var addr = (double*)ret.Address;
                var incr = new Utilities.ValueCoordinatesIncrementor(ref shape);

                do
                {
                    *(addr + shape.GetOffset(incr.Index)) = 2.0 * SampleStandardGamma(halfDf);
                } while (incr.Next() != null);
            }

            return ret;
        }
    }
}
