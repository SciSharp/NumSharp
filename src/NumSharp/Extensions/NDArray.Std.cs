using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray_Legacy<double> Std(this NDArray_Legacy<NDArray_Legacy<double>> np, int axis = -1)
        {
            var std = new NDArray_Legacy<double>();

            var mean = np.Mean(axis);

            // axis == -1: DEFAULT; to compute the standard deviation of the flattened array.
            if (axis == -1)
            {
                var sum = np.Data.Select(d => d.Data.Select(p => Math.Pow(Math.Abs(p - mean.Data[0]), 2)).Sum()).Sum();

                std.Data.Add(Math.Sqrt(sum / np.Size));
            }
            // to compute mean by compressing row and row
            else if (axis == 0)
            {
                double[] sumVec = new double[np.Data[0].Length];
                for (int d = 0; d < np.Length; d++)
                {
                    for (int p = 0; p < np.Data[0].Length; p++)
                    {
                        sumVec[p] += np.Data[d][p];
                    }
                }
                for (int d = 0; d < np.Data[0].Length; d++)
                {
                    mean.Data.Add(sumVec[d] / np.Length);
                }
            }
            else if (axis == 1)
            {
                for (int d = 0; d < np.Length; d++)
                {
                    double rowSum = 0;
                    for (int p = 0; p < np.Data[0].Length; p++)
                    {
                        rowSum += np.Data[d][p];
                    }
                    mean.Data.Add(rowSum / np.Data[0].Length);
                }
            }

            return std;
        }
    }
}
