using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class NDArray<TData>
    {
        public TData Dot(NDArray<TData> np)
        {
            dynamic array1 = this.Data;
            dynamic array2 = np.Data;

            Type elementType = typeof(TData);

            dynamic sumDyn = null;

            if (elementType == typeof(double))
            {
                double[] array1Double = (double[]) array1;
                double[] array2Double = (double[]) array2;

                double sum = 0;

                for (int idx = 0; idx < array1Double.Length; idx++)
                {
                    sum += array1Double[idx] * array2Double[idx];
                }

                sumDyn = sum;
            }
            else if( elementType == typeof(int))
            {
                int[] array1Double = (int[]) array1;
                int[] array2Double = (int[]) array2;

                int sum = 0;

                for (int idx = 0; idx < array1Double.Length; idx++)
                {
                    sum += array1Double[idx] * array2Double[idx];
                }

                sumDyn = sum;
            }
            
            return (TData) sumDyn;
        }
    }
}
