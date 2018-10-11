using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NdArrayExtensions
    {
        public static NdArray<T> Unique<T>(this NdArray<T> np)
        {
            var list = np.Data.Distinct();

            return np.Array(list);
        }
    }
}
