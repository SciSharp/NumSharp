using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NdArrayExtensions
    {
        public static NdArray<T> Array<T>(this NdArray<T> np, IEnumerable<T> array, int ndim = 1)
        {
            np.Data = array.Select(x => x).ToList();
            np.NDim = ndim;

            return np;
        }

        public static NdArray<List<int>> Array(this NdArray<List<int>> np, IList<List<int>> array, int ndim = 1)
        {
            var npTmp = new NdArray<int>();

            for (int r = 0; r < array.Count(); r++)
            {
                if (np.NDim < 0)
                {
                    np.NDim = array[0].Count;
                }

                for(int d =0; d < np.NDim; d++)
                {
                    npTmp.Data.Add(array[r][d]);
                }
            }

            return npTmp.ReShape(array.Count, np.NDim);
        }
    }
}
