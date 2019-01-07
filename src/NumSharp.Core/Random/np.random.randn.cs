using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
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
            var nd = stardard_normal();
            switch (typeof(T).Name)
            {
                case "Double":
                    return (T)Convert.ChangeType(nd.Data<double>()[0], typeof(T));
                default:
                    throw new NotImplementedException("randn");
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
            Random rand = new Random(); //reuse this if you are generating many

            double[] arr = array.Storage.GetData<double>();

            for (int i = 0; i < array.size; i++)
            {
                double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
                double u2 = 1.0 - rand.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                             Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
                double randNormal = loc + scale * randStdNormal; //random normal(mean,stdDev^2)
                arr[i] = randNormal;
            }

            array.Storage.SetData(arr);

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
