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
            np.Data = array.Select(x => x).ToArray();

            return np;
        }

        public static NDArray<TData[]> Array<TData>(this NDArray<TData[]> np, TData[][] array )
        {
            np.Data = array;

            return np;
        }

        public static NDArray<TData> Array<TData>(this NDArray<TData> np, TData[] array)
        {
            np.Data = array;
            np.Shape[0] = np.Data.Length;

            return np;
        }
    }
}
