using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class NDArray<TData>
    {
        public NDArray<TData> Array(IEnumerable<TData> array, int ndim = 1)
        {
            var np = this;

            np.Data = array.Select(x => x).ToList();

            return np;
        }

        public NDArray<TData> Array(IList<List<int>> array, int ndim = 1)
        {
            var np = this;

            var npTmp = new NDArray<int>();

            for (int r = 0; r < array.Count(); r++)
            {
                for(int d =0; d < np.NDim; d++)
                {
                    npTmp.Data.Add(array[r][d]);
                }
            }
            dynamic puffer = npTmp;

            return puffer.ReShape(array.Count, np.NDim);
        }
        public NDArray<NDArray<double>> Array(double[,] array)
        {
            NDArray<NDArray<double>> returnArray = new NDArray<NDArray<double>>();

            returnArray.Data = new NDArray<double>[array.GetLength(0)];
            for (int idx = 0;idx < array.GetLength(0); idx++)
            {
                returnArray.Data[idx] = new NDArray<double>();
                returnArray.Data[idx].Data = new double[array.GetLength(1)];
                for (int jdx =0; jdx < array.GetLength(1); jdx++)
                {
                    returnArray.Data[idx].Data[jdx] = array[idx,jdx];
                }
            }
            return returnArray;
        }
    }
}
