using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumSharp.Extensions;

namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray std(int axis = -1, Type dtype = null)
        {
            dtype = (dtype == null) ? typeof(double) : dtype;

            // in case have 1D array but user still using axis 0 ... can be used like -1
            axis = (axis == 0 && this.ndim == 1) ? -1 : axis;

            Array data = this.Storage.GetData();

            NDArray stdArr = new NDArray(dtype);

            if (axis == -1)
            {
                double mean = this.mean(axis).MakeGeneric<double>()[0];
                double sum = 0;
                for(int idx = 0; idx < data.Length;idx++)                
                    sum += Math.Pow(Convert.ToDouble(data.GetValue(idx)) - mean,2);
                
                double stdValue = Math.Sqrt(sum / this.size);
                stdArr.Storage.Allocate(new Shape(1));
                var puffer = Array.CreateInstance(dtype,1);
                puffer.SetValue(stdValue,0);
                stdArr.Storage.SetData(puffer);
            }
            else 
            {
                double[] stdValue = null;
                if (axis == 0)
                {
                    double[] sum = new double[this.shape[1]];
                    stdValue = new double[sum.Length];

                    double[] mean = this.mean(axis).Storage.GetData<double>();

                    for (int idx = 0; idx < sum.Length;idx++)
                    {
                        for(int jdx =0; jdx < this.shape[0];jdx++)
                        {
                            sum[idx] += Math.Pow(Data<double>(jdx,idx) - mean[idx],2); 
                        }
                        stdValue[idx] = Math.Sqrt(sum[idx] / this.shape[0]);
                    }
                            
                }
                else if (axis == 1)
                {
                    double[] sum = new double[this.shape[0]];
                    stdValue = new double[sum.Length];

                    double[] mean = this.mean(axis).Storage.GetData<double>();

                    for (int idx = 0; idx < sum.Length;idx++)
                    {
                        for(int jdx =0; jdx < this.shape[1];jdx++)
                        {
                            sum[idx] += Math.Pow(Data<double>(idx,jdx) - mean[idx],2); 
                        }
                        stdValue[idx] = Math.Sqrt(sum[idx] / this.shape[1]);
                    }
                }
                else
                {
                    throw new NotImplementedException();
                }
                stdArr.Storage.Allocate(new Shape(stdValue.Length));
                stdArr.Storage.SetData(stdValue);
            }
            return stdArr;
        }
    }
}
