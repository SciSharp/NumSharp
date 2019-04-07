using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class NDArray
    {

        /// <summary>
        /// Compute the arithmetic mean along the specified axis.
        /// Returns the average of the array elements.The average is taken over
        /// the flattened array by default, otherwise over the specified axis.
        /// double intermediate and return values are used for integer inputs.
        /// </summary>
        public NDArray mean(int axis = -1)
        {
            var mean = new NDArray(typeof(double));

            mean.Storage.SetData(new double[0]);

            // axis == -1: DEFAULT; to compute the mean of the flattened array.
            if (axis == -1)
            {
                var data = this.Storage.GetData();

                double sum = 0;

                for (int idx =0; idx < data.Length;idx++)
                    sum += Convert.ToDouble(data.GetValue(idx)); 

                mean.Storage.SetData(new double[] { sum / this.size});
            }
            // to compute mean by compressing row and row
            else if (axis == 0)
            {
                double[] sumVec = new double[this.shape[0]];

                for (int d = 0; d < sumVec.Length; d++)
                {
                    for (int p = 0; p < this.shape[1]; p++)
                    {
                        sumVec[p] += Convert.ToDouble(this[d,p]);
                    }
                }
                var puffer = mean.Storage.CloneData<double>().ToList();

                for (int d = 0; d < this.shape[1]; d++)
                {
                    puffer.Add(sumVec[d] / this.shape[0]);
                }
                mean.Storage.SetData(puffer.ToArray());

                mean.Storage.Reshape(mean.Storage.GetData().Length);
            }
            else if (axis == 1)
            {
                var puffer = mean.Storage.GetData<double>().ToList();

                for (int d = 0; d < this.shape[0]; d++)
                {
                    double rowSum = 0;
                    
                    for (int p = 0; p < this.shape[1]; p++)
                    {
                        rowSum += Convert.ToDouble(this[d,p]);
                    }
                    puffer.Add(rowSum / this.shape[1]);
                }

                mean.Storage.SetData(puffer.ToArray());
                
                mean.Storage.Reshape(mean.Storage.GetData().Length);
            }

            return mean;
        }
    }
}
