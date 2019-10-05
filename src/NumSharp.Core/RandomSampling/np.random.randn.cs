using System;
using System.Threading.Tasks;
using NumSharp.Backends;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        /// Return a sample (or samples) from the “standard normal” distribution.
        /// </summary>
        /// <param name="size"></param>
        /// <returns></returns>
        public NDArray randn(params int[] size)
        {
            return stardard_normal(size);
        }

        /// <summary>
        /// Scalar value
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public T randn<T>()
        {
            return (T)Converts.ChangeType(randomizer.NextDouble(), InfoOf<T>.NPTypeCode);            
        }

        /// <summary>
        /// Draw random samples from a normal (Gaussian) distribution.
        /// </summary>
        /// <param name="loc">Mean of the distribution</param>
        /// <param name="scale">Standard deviation of the distribution</param>
        /// <param name="dims"></param>
        /// <returns></returns>
        public NDArray normal(double loc, double scale, params int[] dims)
        {
            unsafe
            {
                var array = new NDArray<double>(new Shape(dims));
                var dst = array.Address;

                Func<double> nextDouble = randomizer.NextDouble;
                for (int i = 0; i < array.size; i++) 
                    dst[i] = loc + scale * Math.Sqrt(-2.0 * Math.Log(1.0 - nextDouble())) 
                                         * Math.Sin(2.0 * Math.PI * (1.0 - nextDouble())); //random normal(mean,stdDev^2)

                return array;
            }
        }

        /// <summary>
        /// Draw samples from a standard Normal distribution (mean=0, stdev=1).
        /// </summary>
        /// <param name="size"></param>
        /// <returns></returns>
        public NDArray stardard_normal(params int[] size)
        {
            return normal(0, 1.0, size);
        }
    }
}
