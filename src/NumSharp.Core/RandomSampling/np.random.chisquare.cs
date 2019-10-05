using System;
using System.Threading.Tasks;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a chi-square distribution.
        ///     When df independent random variables, each with standard normal distributions(mean 0, variance 1), are squared and summed, 
        ///     the resulting distribution is chi-square(see Notes). This distribution is often used in hypothesis testing.
        /// </summary>
        /// <param name="df">Number of degrees of freedom, should be > 0.</param>
        /// <param name="shape">Output Shape</param>
        /// <returns>Drawn samples from the parameterized chi-square distribution.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.chisquare.html</remarks>
        public NDArray chisquare(double df, Shape shape) => chisquare(df, shape.dimensions);

        /// <summary>
        ///     Draw samples from a chi-square distribution.
        ///     When df independent random variables, each with standard normal distributions(mean 0, variance 1), are squared and summed, 
        ///     the resulting distribution is chi-square(see Notes). This distribution is often used in hypothesis testing.
        /// </summary>
        /// <param name="df">Number of degrees of freedom, should be > 0.</param>
        /// <param name="dims">Output Shape</param>
        /// <returns>Drawn samples from the parameterized chi-square distribution.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.chisquare.html</remarks>
        public NDArray chisquare(double df, params int[] dims)
        {
            if (df <= 0)
                throw new ArgumentException("df should be > 0");

            return np.random.gamma(df / 2, 2, dims);
        }
    }
}
