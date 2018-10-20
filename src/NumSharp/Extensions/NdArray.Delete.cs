using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<TData> Delete<TData>(this NDArray<TData> np,  IEnumerable<TData> delete)
        {            
            return np.Array(np.Data.Where(x => !delete.Contains(x)));
        }
    }
}
