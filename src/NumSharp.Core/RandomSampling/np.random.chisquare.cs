using System;

namespace NumSharp
{
    public partial class NumPyRandom
    {
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
        public NDArray chisquare(double df, Shape size) => chisquare(df, size.dimensions);

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
        public NDArray chisquare(double df, params int[] size) => chisquare(df, Shape.ComputeLongShape(size));

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
        public NDArray chisquare(double df, params long[] size)
        {
            if (df <= 0)
                throw new ArgumentException("df must be > 0", nameof(df));

            return gamma(df / 2, 2.0, size);
        }
    }
}
