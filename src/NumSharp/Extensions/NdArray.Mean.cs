using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<double> Mean(this NDArray<double> np, int axis = -1)
        {
            var mean = new NDArray<double>();

            // axis == -1: DEFAULT; to compute the mean of the flattened array.
            if (axis == -1)
            {
                var sum = np.Data.Sum();

                mean.Data = new double[] { sum / np.Size};
            }
            // to compute mean by compressing row and row
            else if (axis == 0)
            {
                double[] sumVec = new double[np.Shape.Shapes[0]];
                for (int d = 0; d < sumVec.Length; d++)
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
                mean.Shape = new Shape(mean.Data.Length);
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
                mean.Shape = new Shape(mean.Data.Length);
            }

            return mean;
        }
    }
}
