using System;
using System.Threading.Tasks;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        /// Draw samples from an exponential distribution.
        /// The exponential distribution is a continuous analogue of the geometric distribution. It describes many common situations, such as the size of raindrops measured over many rainstorms 
        /// </summary>
        /// <param name="scale">The scale parameter, \beta = 1/\lambda.</param>
        /// <param name="shape">Output Shape</param>
        /// <returns></returns>
        public NDArray exponential(double scale, Shape shape) => exponential(scale, shape.Dimensions);

        /// <summary>
        /// Draw samples from an exponential distribution.
        /// The exponential distribution is a continuous analogue of the geometric distribution. It describes many common situations, such as the size of raindrops measured over many rainstorms 
        /// </summary>
        /// <param name="scale">The scale parameter, \beta = 1/\lambda.</param>
        /// <param name="dims">Output Shape</param>
        /// <returns></returns>
        public NDArray exponential(double scale, params int[] dims)
        {
            var x = np.log(1 - np.random.uniform(0, 1, dims));
            return np.negative(x) / scale;
        }
    }
}
