using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray_Legacy<TData> Delete<TData>(this NDArray_Legacy<TData> np,  IEnumerable<TData> delete)
        {            
            return np.Array(np.Data.Where(x => !delete.Contains(x)));
        }
    }
}
