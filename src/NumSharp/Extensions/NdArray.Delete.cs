using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NdArrayExtensions
    {
        public static NdArray<TData> Delete<TData>(this NdArray<TData> np, IEnumerable<TData> delete)
        {
            return np.Array(np.Data.Where(x => !delete.Contains(x)));
        }
    }
}
