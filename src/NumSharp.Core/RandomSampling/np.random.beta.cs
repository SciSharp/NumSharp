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
        /// <param name="beta">Beta value</param>
        /// <param name="dims">Output Shape</param>
        /// <returns></returns>
        public NDArray beta(double alpha, double beta, params int[] dims)
        {
            var x = np.random.gamma(alpha, 1, dims);
            var y = np.random.gamma(beta, 1, dims);

            return x / (x + y);
        }
    }
}
