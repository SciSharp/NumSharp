using System;
using System.Threading.Tasks;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a bernoulli distribution.
        /// </summary>
        /// <param name="p">Parameter of the distribution, >= 0 and &lt;=1.</param>
        /// <param name="shape">Output Shape</param>
        /// <returns>Drawn samples from the parameterized bernoulli distribution.</returns>
        public NDArray bernoulli(double p, Shape shape) => bernoulli(p, shape.dimensions);

        /// <summary>
        ///     Draw samples from a bernoulli distribution.
        /// </summary>
        /// <param name="p">Parameter of the distribution, >= 0 and &lt;=1.</param>
        /// <param name="dims">Output Shape</param>
        /// <returns>Drawn samples from the parameterized bernoulli distribution.</returns>
        public NDArray bernoulli(double p, params int[] dims)
        {
            if (dims == null || dims.Length == 0) //return scalar
                return NDArray.Scalar(randomizer.NextDouble());

            var result = new NDArray<double>(dims);
            unsafe
            {
                var addr = result.Address;
                var len = result.size;
                Func<double> nextDouble = randomizer.NextDouble;
                for (int i = 0; i < len; i++)
                    addr[i] = nextDouble() > p ? 1 : 0;
            }

            return result;
        }
    }
}
