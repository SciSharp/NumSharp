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
        /// Draw samples from a bernoulli distribution.
        /// </summary>
        /// <param name="p">Parameter of the distribution, >= 0 and <=1.</param>
        /// <param name="dims">Output Shape</param>
        /// <returns>Drawn samples from the parameterized bernoulli distribution.</returns>
        public NDArray bernoulli(double p, params int[] dims)
        {
            if (dims == null || dims.Length == 0) //return scalar
            {
                var ret = new NDArray<double>(new Shape(1));
                var data = new double[] { randomizer.NextDouble()};
                ret.ReplaceData(data);
                return ret;
            }

            var result = new NDArray<double>(dims);
            ArraySlice<double> resultArray = result.Data<double>();

            Parallel.For(0, result.size, (i) => {
                resultArray[i] = randomizer.NextDouble() > p ? 1 : 0;
            });

            result.ReplaceData(resultArray);
            return result;
        }
    }
}
