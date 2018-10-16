using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<double> Max(this NDArray<NDArray<double>> np)
        {
            var max = new NDArray<double>();

            for (int d = 0; d < np.NDim; d++)
            {
                var value = np.Data.Select(x => x.Data[d]).Max();
                max.Data.Add(value);
            }

            return max;
        }
    }
}
