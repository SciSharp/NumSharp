using System;
using System.Threading.Tasks;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        /// Draw samples from a Beta distribution.
        /// The Beta distribution is a special case of the Dirichlet distribution, and is related to the Gamma distribution.It has the probability distribution function
        /// </summary>
        /// <param name="alpha">Alpha value</param>
        /// <param name="betaValue">Beta value</param>
        /// <param name="shape">Output Shape</param>
        /// <returns></returns>
        public NDArray beta(double alpha, double betaValue, Shape shape) => beta(alpha, betaValue, shape.dimensions);

        /// <summary>
        /// Draw samples from a Beta distribution.
        /// The Beta distribution is a special case of the Dirichlet distribution, and is related to the Gamma distribution.It has the probability distribution function
        /// </summary>
        /// <param name="alpha">Alpha value</param>
        /// <param name="betaValue">Beta value</param>
        /// <param name="dims">Output Shape</param>
        /// <returns></returns>
        public NDArray beta(double alpha, double betaValue, params int[] dims)
        {
            var x = np.random.gamma(alpha, 1, dims);
            var y = np.random.gamma(betaValue, 1, dims);

            return x / (x + y);
        }
    }
}
