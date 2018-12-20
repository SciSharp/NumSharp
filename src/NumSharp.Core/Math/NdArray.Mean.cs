using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray mean(this NDArray np, int axis = -1)
        {
            var mean = new NDArray(typeof(double));

            mean.Storage.SetData(new double[0]);

            // axis == -1: DEFAULT; to compute the mean of the flattened array.
            if (axis == -1)
            {
                var sum = np.Storage.GetData<double>().Sum();

                mean.Storage.SetData(new double[] { sum / np.size});
            }
            // to compute mean by compressing row and row
            else if (axis == 0)
            {
                double[] sumVec = new double[np.shape.Dimensions[0]];

                for (int d = 0; d < sumVec.Length; d++)
                {
                    for (int p = 0; p < np.shape.Dimensions[1]; p++)
                    {
                        sumVec[p] += Convert.ToDouble(np[d,p]);
                    }
                }
                var puffer = mean.Storage.GetData<double>().ToList();

                for (int d = 0; d < np.shape.Dimensions[1]; d++)
                {
                    puffer.Add(sumVec[d] / np.shape.Dimensions[0]);
                }
                mean.Storage.SetData(puffer.ToArray());

                mean.Storage.Reshape(mean.Storage.GetData().Length);
            }
            else if (axis == 1)
            {
                var puffer = mean.Storage.GetData<double>().ToList();

                for (int d = 0; d < np.shape.Dimensions[0]; d++)
                {
                    double rowSum = 0;
                    
                    for (int p = 0; p < np.shape.Dimensions[1]; p++)
                    {
                        rowSum += Convert.ToDouble(np[d,p]);
                    }
                    puffer.Add(rowSum / np.shape.Dimensions[1]);
                }

                mean.Storage.SetData(puffer.ToArray());
                
                mean.Storage.Reshape(mean.Storage.GetData().Length);
            }

            return mean;
        }
    }
}
