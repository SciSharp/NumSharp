using NumSharp.Core.Extensions;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public class NDArrayRandom
    {
        public static int Seed { get; set; }

        /// <summary>
        /// Return a sample (or samples) from the “standard normal” distribution.
        /// </summary>
        /// <param name="size"></param>
        /// <returns></returns>
        public NDArrayGeneric<double> randn(params int[] size)
        {
            return this.stardard_normal(size);
        }

        /// <summary>
        /// Draw random samples from a normal (Gaussian) distribution.
        /// </summary>
        /// <param name="loc">Mean of the distribution</param>
        /// <param name="scale">Standard deviation of the distribution</param>
        /// <param name="size"></param>
        /// <returns></returns>
        public NDArrayGeneric<double> normal(double loc, double scale, params int[] size)
        {
            if (size.Length == 0)
                throw new Exception("d cannot be empty.");
            NDArrayGeneric<double> array = new NDArrayGeneric<double>();
            Random rand = new Random(); //reuse this if you are generating many
            array.Shape = new Shape(size);
            array.Data = new double[array.Shape.Size];

            for (int i = 0; i < array.Shape.Size; i++)
            {
                double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
                double u2 = 1.0 - rand.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                             Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
                double randNormal = loc + scale * randStdNormal; //random normal(mean,stdDev^2)
                array.Data[i] = randNormal;
            }
            return array;
        }

        /// <summary>
        /// Draw samples from a standard Normal distribution (mean=0, stdev=1).
        /// </summary>
        /// <param name="size"></param>
        /// <returns></returns>
        public NDArrayGeneric<double> stardard_normal(params int[] size)
        {
            return this.normal(0, 1.0, size);
        }
    }
}
