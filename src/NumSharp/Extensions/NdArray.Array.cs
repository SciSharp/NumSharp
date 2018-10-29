using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<TData> Array<TData>(this NDArray<TData> np, IEnumerable<TData> array, int ndim = 1)
        {
            np.Data = array.Select(x => x).ToList();

            return np;
        }
        public static NDArray<TData[]> Array<TData>(this NDArray<TData[]> np, TData[][] array )
        {
            np.Data = array;

            return np;
        }
        public static NDArray<TData> Array<TData>(this NDArray<TData> np, TData[] array)
        {
            
            np.Data = array.Select(x => x).ToList();

            return np;
        }
        public static NDArray<TData> Array<TData>(this NDArray<TData> np, IList<List<int>> array, int ndim = 1)
        {
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
        public static NDArray<NDArray<double>> Array<TData>(this NDArray<TData> np, double[,] array)
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
