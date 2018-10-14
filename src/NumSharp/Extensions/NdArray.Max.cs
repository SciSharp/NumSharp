using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NdArrayExtensions
    {
        public static NdArray<double> Max(this NdArray<NdArray<double>> np)
        {
            var max = new NdArray<double>();

            for (int d = 0; d < np.NDim; d++)
            {
                var value = np.Data.Select(x => x.Data[d]).Max();
                max.Data.Add(value);
            }

            return max;
        }
    }
}
