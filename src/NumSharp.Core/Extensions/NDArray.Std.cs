using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<double> Std(this NDArray<double> np, int axis = -1)
        {
            var std = new NDArray<double>();

            var mean = np.Mean(axis);

            // axis == -1: DEFAULT; to compute the standard deviation of the flattened array.
            if (axis == -1)
            {
                var sum = np.Data.Select(d => Math.Pow(Math.Abs(d - mean.Data[0]), 2)).Sum();

                std.Data = new double[]{ Math.Sqrt(sum / np.Size) };
            }
            // to compute mean by compressing row and row
            else if (axis == 0)
            {
                double[] sumVec = new double[np.Shape.Shapes[1]];
                for (int d = 0; d < np.Shape.Shapes[0]; d++)
                {
                    for (int p = 0; p < np.Shape.Shapes[1]; p++)
                    {
                        sumVec[p] += np[d,p];
                    }
                }
                var puffer = mean.Data.ToList();
                for (int d = 0; d < np.Shape.Shapes[1]; d++)
                {
                    puffer.Add(sumVec[d] / np.Shape.Shapes[0]);
                }
                mean.Data = puffer.ToArray();
            }
            else if (axis == 1)
            {
                var puffer = mean.Data.ToList();
                for (int d = 0; d < np.Shape.Shapes[0]; d++)
                {
                    double rowSum = 0;
                    for (int p = 0; p < np.Shape.Shapes[1]; p++)
                    {
                        rowSum += np[d,p];
                    }
                    puffer.Add(rowSum / np.Shape.Shapes[1]);
                }
                mean.Data = puffer.ToArray();
            }

            return std;
        }
    }
}
