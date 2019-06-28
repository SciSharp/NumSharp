using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Extensions;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public NDArray Mean(NDArray x, int axis = -1)
        {
            var mean = new NDArray(typeof(double));

            // axis == -1: DEFAULT; to compute the mean of the flattened array.
            if (axis == -1)
            {
                var data = x.Array;

                double sum = 0;

                switch (data)
                {
                    case double[] values:
                        for (int idx = 0; idx < values.Length; idx++)
                            sum += values[idx];
                        break;
                    case int[] values:
                        for (int idx = 0; idx < values.Length; idx++)
                            sum += values[idx];
                        break;
                    default:
                        throw new NotImplementedException($"mean {x.dtype.Name}");
                }

                mean.ReplaceData(new double[] { sum / x.size });
            }
            // to compute mean by compressing row and row
            else if (axis == 0)
            {
                double[] sumVec = new double[x.shape[0]];

                for (int d = 0; d < sumVec.Length; d++)
                {
                    for (int p = 0; p < x.shape[1]; p++)
                    {
                        sumVec[p] += x.Data<double>(d, p);
                    }
                }
                var puffer = new List<double>();

                for (int d = 0; d < x.shape[1]; d++)
                {
                    puffer.Add(sumVec[d] / x.shape[0]);
                }
                mean.ReplaceData(puffer.ToArray());

                mean.reshape(mean.Array.Length);
            }
            else if (axis == 1)
            {
                var puffer = new List<double>();

                for (int d = 0; d < x.shape[0]; d++)
                {
                    double rowSum = 0;

                    for (int p = 0; p < x.shape[1]; p++)
                    {
                        rowSum += x.Data<double>(d, p);
                    }
                    puffer.Add(rowSum / x.shape[1]);
                }

                mean.ReplaceData(puffer.ToArray());

                mean.reshape(mean.Array.Length);
            }

            return mean;
        }
    }

}
