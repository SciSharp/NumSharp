using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

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
            switch (typeof(T).GetTypeCode())
            {
#if _REGEN
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
	            case NPTypeCode.#1:
	            {
                    return (T)Convert.ChangeType(randomizer.NextDouble(), typeof(T));
	            }
	            %
	            default:
		            throw new NotSupportedException();
#else
                case NPTypeCode.Byte:
                {
                    return (T)Convert.ChangeType(randomizer.NextDouble(), typeof(T));
                }

                case NPTypeCode.Int16:
                {
                    return (T)Convert.ChangeType(randomizer.NextDouble(), typeof(T));
                }

                case NPTypeCode.UInt16:
                {
                    return (T)Convert.ChangeType(randomizer.NextDouble(), typeof(T));
                }

                case NPTypeCode.Int32:
                {
                    return (T)Convert.ChangeType(randomizer.NextDouble(), typeof(T));
                }

                case NPTypeCode.UInt32:
                {
                    return (T)Convert.ChangeType(randomizer.NextDouble(), typeof(T));
                }

                case NPTypeCode.Int64:
                {
                    return (T)Convert.ChangeType(randomizer.NextDouble(), typeof(T));
                }

                case NPTypeCode.UInt64:
                {
                    return (T)Convert.ChangeType(randomizer.NextDouble(), typeof(T));
                }

                case NPTypeCode.Char:
                {
                    return (T)Convert.ChangeType(randomizer.NextDouble(), typeof(T));
                }

                case NPTypeCode.Double:
                {
                    return (T)Convert.ChangeType(randomizer.NextDouble(), typeof(T));
                }

                case NPTypeCode.Single:
                {
                    return (T)Convert.ChangeType(randomizer.NextDouble(), typeof(T));
                }

                case NPTypeCode.Decimal:
                {
                    return (T)Convert.ChangeType(randomizer.NextDouble(), typeof(T));
                }

                default:
                    throw new NotSupportedException();
#endif
            }
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
            var array = new NDArray(typeof(double), new Shape(dims));

            var arr = array.Data<double>();

            for (int i = 0; i < array.size; i++)
            {
                double u1 = 1.0 - randomizer.NextDouble(); //uniform(0,1] random doubles
                double u2 = 1.0 - randomizer.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                       Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
                double randNormal = loc + scale * randStdNormal; //random normal(mean,stdDev^2)
                arr[i] = randNormal;
            }

            array.ReplaceData(arr);

            return array;
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
