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

            np.Data = array.Select(x => x).ToArray();
            np.NDim = ndim;

            return np;
        }

        public NDArray<TData> Array(IList<List<int>> array, int ndim = 1)
        {
            var np = this;

            var npTmp = new NDArray<int>();

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
            dynamic puffer = npTmp;

            return puffer.ReShape(array.Count, np.NDim);
        }
    }
}
