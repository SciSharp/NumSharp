using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<double> HStack(this NDArray<double> np, NDArray<double> np2 )
        {
            var list = np.Data.ToList();
            list.AddRange(np2.Data);
            np.Data = list.ToArray();

            return np;
        }
    }
}
