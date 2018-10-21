using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static double Dot(this NDArray<double> np, NDArray<double> np2)
        {
            double[] array1Double = np.Data.ToArray();
            double[] array2Double = np2.Data.ToArray();
            
            double sum = 0;

            for (int idx = 0; idx < array1Double.Length; idx++)
            {
                sum += array1Double[idx] * array2Double[idx];
            }

            return sum;
        }
        public static int Dot(this NDArray<int> np, NDArray<int> np2)
        {
            int[] array1Double = np.Data.ToArray();
            int[] array2Double = np2.Data.ToArray();
            
            int sum = 0;

            for (int idx = 0; idx < array1Double.Length; idx++)
            {
                sum += array1Double[idx] * array2Double[idx];
            }

            return sum;
        }
        public static NDArray<double> Dot(this NDArray<double> np, double scalar)
        {
            double[] array1Double = np.Data.ToArray();

            array1Double = array1Double.Select(x => scalar * x).ToArray();
            
            np.Data = array1Double;

            return np;
        }
        public static NDArray<float> Dot(this NDArray<float> np, float scalar)
        {
            np.Data = np.Data.Select(x => x * scalar).ToArray();
            return np;
        }
        public static NDArray<int> Dot(this NDArray<int> np, int scalar)
        {
            np.Data = np.Data.Select(x => x * scalar).ToArray();
            return np;
        }
    }
}
