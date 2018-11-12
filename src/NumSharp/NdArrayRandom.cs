using NumSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public class NDArrayRandom
    {
        public static int Seed { get; set; }

        /// <summary>
        /// Return a sample (or samples) from the “standard normal” distribution.
        /// </summary>
        /// <param name="d"></param>
        /// <returns></returns>
        public NDArray<double> randn(params int[] d)
        {
            if (d.Length == 0)
                throw new Exception("d cannot be empty.");
            NDArray<double> array = new NDArray<double>();
            Random rand = new Random(); //reuse this if you are generating many
            array.Shape = new Shape(d);
            for (int i = 0; i < array.Shape.Size; i++)
            {
                double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
                double u2 = 1.0 - rand.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                             Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
                double randNormal = 0 + 1 * randStdNormal; //random normal(mean,stdDev^2)
                array.Data[i] = rand.Next();
            }
            return array;
        }
    }
}
