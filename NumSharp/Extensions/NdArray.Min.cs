using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NdArrayExtensions
    {
        public static NdArray<double> Min(this NdArray<NdArray<double>> np)
        {
            var min = new NdArray<double>();

            for (int d = 0; d < np.NDim; d++)
            {
                var value = np.Data.Select(x => x.Data[d]).Min();
                min.Data.Add(value);
            }

            return min;
        }
    }
}
